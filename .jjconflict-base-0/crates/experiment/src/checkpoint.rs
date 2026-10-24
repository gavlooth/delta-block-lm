//! Checkpoint I/O boundary: synchronize target initialization, then verify loaded logical values.
use crate::{config::Config, data, model::LanguageModel, runtime, train};
use anyhow::{Result, ensure};
use burn::{
    module::{Module, ModuleVisitor, Param},
    prelude::*,
};
use burn_pack::Reader;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::Path};

#[derive(Debug, PartialEq, Serialize)]
struct TensorSignature {
    dtype: String,
    shape: Vec<usize>,
    param_id: Option<u64>,
    sha256: String,
}
fn record_tensors(reader: Reader) -> Result<BTreeMap<String, TensorSignature>> {
    let mut result = BTreeMap::new();
    for tensor in reader.into_tensors()? {
        let (name, dtype, shape, param_id, bytes) = tensor.into_parts()?;
        let signature = TensorSignature {
            dtype: format!("{dtype:?}"),
            shape: shape.iter().copied().collect(),
            param_id,
            sha256: data::hash(&bytes),
        };
        ensure!(
            result.insert(name, signature).is_none(),
            "duplicate checkpoint tensor name"
        );
    }
    Ok(result)
}
struct Validate {
    path: Vec<String>,
    expected: BTreeMap<String, TensorSignature>,
    errors: Vec<String>,
}
impl ModuleVisitor for Validate {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_owned());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let name = self.path.join(".");
        let expected = match self.expected.remove(&name) {
            Some(value) => value,
            None => {
                self.errors
                    .push(format!("{name}: parameter absent from saved record"));
                return;
            }
        };
        match param.transform_for_save().val().try_into_data() {
            Ok(value) => {
                let actual = TensorSignature {
                    dtype: format!("{:?}", value.dtype()),
                    shape: value.shape().iter().copied().collect(),
                    param_id: Some(param.id.val()),
                    sha256: data::hash(value.as_bytes()),
                };
                if actual != expected {
                    self.errors.push(format!("{name}: loaded logical values/shape/dtype/id differ: expected {expected:?}, actual {actual:?}"));
                }
            }
            Err(error) => self
                .errors
                .push(format!("{name}: loaded tensor read failed: {error:?}")),
        }
    }
}
/// Keep the constructed target alive across the barrier: queued default fills must finish before
/// their buffers can be replaced/reused by record uploads. This guard is measured on Intel Vulkan;
/// the precise upstream initializer/upload race mechanism remains an inference, not a backend patch.
pub fn load_model(
    target: LanguageModel,
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<LanguageModel> {
    runtime::sync(device)?;
    let path = path.as_ref();
    let model = target.try_load_file(path)?;
    runtime::sync(device)?;
    validate_model(&model, path)?;
    Ok(model)
}
pub fn validate_model(model: &LanguageModel, path: impl AsRef<Path>) -> Result<()> {
    let mut validation = Validate {
        path: Vec::new(),
        expected: record_tensors(Reader::from_file(path)?)?,
        errors: Vec::new(),
    };
    model.visit(&mut validation);
    ensure!(
        validation.expected.is_empty(),
        "saved record contains unvalidated model parameters: {:?}",
        validation.expected.keys().collect::<Vec<_>>()
    );
    ensure!(
        validation.errors.is_empty(),
        "checkpoint load validation failed: {}",
        validation.errors.join("; ")
    );
    Ok(())
}

/// Container byte order and tensor offsets are not optimizer semantics. Canonicalize every typed
/// tensor by name, plus scalar counters, path metadata, actual AdamWConfig and step learning rate.
pub fn optimizer_signature(c: &Config, bytes: burn::tensor::Bytes) -> Result<Value> {
    optimizer_reader_signature(c, Reader::from_bytes(bytes)?)
}
pub fn optimizer_file_signature(c: &Config, path: impl AsRef<Path>) -> Result<Value> {
    optimizer_reader_signature(c, Reader::from_file(path)?)
}
fn optimizer_reader_signature(c: &Config, reader: Reader) -> Result<Value> {
    let scalars = serde_json::to_value(reader.scalars())?;
    let metadata = reader.metadata().clone();
    let tensors = record_tensors(reader)?;
    Ok(
        json!({"tensors":tensors,"scalars":scalars,"metadata":metadata,"optimizer_config":serde_json::to_value(train::optimizer_config(c))?,"learning_rate":c.learning_rate}),
    )
}
pub fn load_optimizer(c: &Config, path: impl AsRef<Path>) -> Result<burn::optim::ModuleOptimizer> {
    let path = path.as_ref();
    let expected = optimizer_file_signature(c, path)?;
    let optimizer = train::optimizer(c).load(path)?;
    let actual = optimizer_signature(c, optimizer.into_bytes()?)?;
    ensure!(
        actual == expected,
        "optimizer load omitted or changed saved tensor/scalar/config state"
    );
    Ok(optimizer)
}
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(reverse: bool, counter: u64, changed: bool) -> burn::tensor::Bytes {
        let make = |name: &str, id, values: Vec<f32>| {
            let data = TensorData::new(values, [2]);
            let (bytes, shape, dtype) = data.into_parts();
            burn_pack::Tensor::new(name.into(), dtype, shape, Some(id), bytes)
        };
        let a = make("1.moment", 1, vec![if changed { 0.75 } else { 0.25 }, 0.5]);
        let b = make("2.moment", 2, vec![0.1, 0.2]);
        burn_pack::Writer::new(if reverse { vec![b, a] } else { vec![a, b] })
            .with_scalar("step", burn_pack::Scalar::UInt(counter))
            .with_metadata("path.1", "embedding")
            .into_bytes()
            .unwrap()
    }
    #[test]
    fn optimizer_canonicalization_ignores_container_order_not_state() {
        let c = Config::default();
        let a = record(false, 2, false);
        let b = record(true, 2, false);
        assert_ne!(data::hash(&a), data::hash(&b));
        let signature = optimizer_signature(&c, a).unwrap();
        assert_eq!(signature, optimizer_signature(&c, b).unwrap());
        assert_ne!(
            signature,
            optimizer_signature(&c, record(false, 3, false)).unwrap()
        );
        assert_ne!(
            signature,
            optimizer_signature(&c, record(false, 2, true)).unwrap()
        );
        let mut changed = c;
        changed.weight_decay *= 2.0;
        assert_ne!(
            signature,
            optimizer_signature(&changed, record(false, 2, false)).unwrap()
        );
    }
    #[test]
    fn optimizer_load_rejects_silently_skipped_partial_moments() {
        let c = Config::default();
        let device = Device::ndarray().autodiff();
        device.seed(443);
        let layer = burn::nn::LinearConfig::new(2, 2)
            .with_bias(false)
            .init(&device);
        let objective = layer
            .forward(Tensor::<2>::from_data([[1.0f32, 2.0]], &device))
            .square()
            .sum();
        let gradients = burn::optim::GradientsParams::from_grads(objective.backward(), &layer);
        let mut optimizer = train::optimizer(&c);
        let _updated = optimizer.step(c.learning_rate, layer, gradients);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "delta-optimizer-state-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let complete = directory.join("complete.burnpack");
        optimizer.save(&complete).unwrap();
        assert!(load_optimizer(&c, &complete).is_ok());
        let reader = Reader::from_file(&complete).unwrap();
        let scalars = reader.scalars().clone();
        let metadata = reader.metadata().clone();
        let mut tensors = reader.into_tensors().unwrap();
        assert!(tensors.len() >= 2);
        tensors.remove(0);
        let mut writer = burn_pack::Writer::new(tensors);
        for (name, value) in scalars {
            writer = writer.with_scalar(&name, value);
        }
        for (name, value) in metadata {
            writer = writer.with_metadata(&name, &value);
        }
        let partial = directory.join("partial.burnpack");
        writer.write_to_file(&partial).unwrap();
        assert!(
            train::optimizer(&c).load(&partial).is_ok(),
            "fixture must exercise the upstream silent-state-skip path"
        );
        assert!(
            load_optimizer(&c, &partial).is_err(),
            "partial moments must not be advertised as restored optimizer state"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
