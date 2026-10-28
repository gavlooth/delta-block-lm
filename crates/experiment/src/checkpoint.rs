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

const MODEL_CHECKPOINT_INCOMPATIBILITY: &str =
    "pre-correction checkpoints are incompatible: require corrected KronLSTM, WNet-Dyn J3/affine LayerNorm, four-tap Lngram, oHC and learned soft-feedback parameters";

pub(crate) fn state_metadata(state: &train::State) -> Result<Value> {
    let mut value = serde_json::to_value(state)?;
    value["model_math_revision"] = crate::config::MODEL_MATH_REVISION.into();
    Ok(value)
}

pub(crate) fn validate_state_metadata(value: &Value) -> Result<()> {
    ensure!(
        value["model_math_revision"].as_str() == Some(crate::config::MODEL_MATH_REVISION),
        "{MODEL_CHECKPOINT_INCOMPATIBILITY}; expected model revision {}, found {}",
        crate::config::MODEL_MATH_REVISION, value["model_math_revision"]
    );
    Ok(())
}

pub(crate) fn load_state(path: impl AsRef<Path>) -> Result<train::State> {
    let value: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_state_metadata(&value)?;
    Ok(serde_json::from_value(value)?)
}

#[derive(Default)]
struct ModelShapes {
    path: Vec<String>,
    shapes: BTreeMap<String, Vec<usize>>,
    gate_biases: usize,
}
impl ModuleVisitor for ModelShapes {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_owned());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let name = self.path.join(".");
        if [".mixer.erase.bias", ".mixer.write.bias", ".mixer.output_gate.bias"]
            .iter().any(|suffix| name.ends_with(suffix)) {
            self.gate_biases += 1;
        }
        self.shapes.insert(name, param.transform_for_save().val().dims().to_vec());
    }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        self.shapes.insert(self.path.join("."), param.transform_for_save().val().dims().to_vec());
    }
}

fn validate_model_shapes(target: &LanguageModel, reader: Reader) -> Result<()> {
    ensure!(reader.metadata().get("model_math_revision").map(String::as_str) == Some(crate::config::MODEL_MATH_REVISION),
        "{MODEL_CHECKPOINT_INCOMPATIBILITY}; missing or obsolete model_math_revision");
    let mut required = ModelShapes::default();
    target.visit(&mut required);
    ensure!(required.gate_biases == target.blocks.len()*3,
        "KronLSTM target is missing learned beta/target/output-gate biases");
    for tensor in reader.into_tensors()? {
        let (name, _, shape, _, _) = tensor.into_parts()?;
        let expected = required.shapes.remove(&name).ok_or_else(|| anyhow::anyhow!(
            "{MODEL_CHECKPOINT_INCOMPATIBILITY}; obsolete or unexpected tensor {name}"))?;
        ensure!(shape.iter().copied().eq(expected),
            "{MODEL_CHECKPOINT_INCOMPATIBILITY}; {name} has an incompatible shape");
    }
    ensure!(required.shapes.is_empty(),
        "{MODEL_CHECKPOINT_INCOMPATIBILITY}; missing tensors: {:?}",
        required.shapes.keys().collect::<Vec<_>>());
    Ok(())
}

/// Native on-save tensors and IDs, with the model equation revision in the record itself.
/// Module::save_file does not expose a metadata hook.
pub fn save_model(model: &LanguageModel, path: impl AsRef<Path>) -> Result<()> {
    #[derive(Default)]
    struct Save {
        path: Vec<String>,
        tensors: Vec<burn_pack::Tensor>,
    }
    impl Save {
        fn record(&mut self, id: burn::module::ParamId, data: TensorData) {
            let (bytes, shape, dtype) = data.into_parts();
            self.tensors.push(burn_pack::Tensor::new(self.path.join("."), dtype, shape, Some(id.val()), bytes));
        }
    }
    impl ModuleVisitor for Save {
        fn enter_module(&mut self, name: &str, _container_type: &str) { self.path.push(name.to_owned()); }
        fn exit_module(&mut self, _name: &str, _container_type: &str) { self.path.pop(); }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.record(param.id, param.transform_for_save().val().into_data());
        }
        fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
            self.record(param.id, param.transform_for_save().val().into_data());
        }
        fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
            self.record(param.id, param.transform_for_save().val().into_data());
        }
    }
    let mut save = Save::default();
    model.visit(&mut save);
    burn_pack::Writer::new(save.tensors)
        .with_metadata("model_math_revision", crate::config::MODEL_MATH_REVISION)
        .write_to_file(path)?;
    Ok(())
}

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
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        let name = self.path.join(".");
        let Some(expected) = self.expected.remove(&name) else {
            self.errors.push(format!("{name}: integer parameter absent from saved record"));
            return;
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
                    self.errors.push(format!("{name}: loaded integer values/shape/dtype/id differ: expected {expected:?}, actual {actual:?}"));
                }
            }
            Err(error) => self.errors.push(format!("{name}: loaded integer read failed: {error:?}")),
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
    validate_model_shapes(&target, Reader::from_file(path)?)?;
    let mut model = target.try_load_file(path)?;
    if let Some(circuit) = &mut model.codpula {
        circuit.freeze_state();
    }
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
    #[test]
    fn model_checkpoint_metadata_rejects_missing_or_obsolete_math_revision() {
        for value in [json!({}), json!({"kron_model_revision":"kronlstm-learned-gate-biases-v1"}), json!({"model_math_revision":"obsolete"})] {
            let error = validate_state_metadata(&value).unwrap_err().to_string();
            assert!(error.contains(MODEL_CHECKPOINT_INCOMPATIBILITY));
        }
        validate_state_metadata(&json!({
            "model_math_revision":crate::config::MODEL_MATH_REVISION
        })).unwrap();
    }

    #[test]
    fn raw_checkpoint_requires_exact_corrected_parameter_set_and_shapes() {
        let device = Device::ndarray();
        let c = Config { d_model:8, layers:1, heads:1, key_dim:2, value_dim:2,
            wavelet_layers:vec![0],lngram_layers:vec![0],lngram_routes:8,lngram_bits:1,lngram_memory_dim:2,
            ..Config::default() };
        let model = LanguageModel::new(&c, &c.arms[0], &device).unwrap();
        let mut biases = ModelShapes::default();
        model.visit(&mut biases);
        assert_eq!(biases.gate_biases, 3);
        assert!(biases.shapes.keys().any(|name|name.ends_with(".wavelet_norm.beta")));
        assert!(biases.shapes.iter().any(|(name,shape)|name.ends_with(".lngram.smoothing.weight") && shape == &[4,8]));
        let make = |omit: Option<usize>, wrong_shape: bool| {
            let tensors = biases.shapes.iter().enumerate().filter_map(|(index,(name,shape))| {
                if omit == Some(index) { return None; }
                let mut shape = shape.clone();
                if wrong_shape && index == 0 { shape[0] += 1; }
                let count = shape.iter().product::<usize>();
                let data = TensorData::new(vec![0.0f32;count], shape);
                let (bytes,shape,dtype) = data.into_parts();
                Some(burn_pack::Tensor::new(name.clone(),dtype,shape,None,bytes))
            }).collect();
            Reader::from_bytes(burn_pack::Writer::new(tensors)
                .with_metadata("model_math_revision",crate::config::MODEL_MATH_REVISION)
                .into_bytes().unwrap()).unwrap()
        };
        validate_model_shapes(&model, make(None,false)).unwrap();
        for missing in 0..biases.shapes.len() {
            let error = validate_model_shapes(&model,make(Some(missing),false)).unwrap_err().to_string();
            assert!(error.contains(MODEL_CHECKPOINT_INCOMPATIBILITY));
        }
        let error = validate_model_shapes(&model,make(None,true)).unwrap_err().to_string();
        assert!(error.contains("incompatible shape"));
        // Even an otherwise exact tensor set cannot impersonate revised equations.
        for revision in [None,Some("kronlstm-learned-gate-biases-v1")] {
            let mut writer=burn_pack::Writer::new(make(None,false).into_tensors().unwrap());
            if let Some(revision)=revision {writer=writer.with_metadata("model_math_revision",revision);}
            let reader=Reader::from_bytes(writer.into_bytes().unwrap()).unwrap();
            assert!(validate_model_shapes(&model,reader).unwrap_err().to_string().contains("missing or obsolete model_math_revision"));
        }
        // An obsolete outer Lngram norm is rejected rather than silently ignored.
        let mut tensors=make(None,false).into_tensors().unwrap();
        let (bytes,shape,dtype)=TensorData::new(vec![1.0f32;8],[8]).into_parts();
        tensors.push(burn_pack::Tensor::new("blocks.0.lngram_norm.gamma".into(),dtype,shape,None,bytes));
        let reader=Reader::from_bytes(burn_pack::Writer::new(tensors)
            .with_metadata("model_math_revision",crate::config::MODEL_MATH_REVISION).into_bytes().unwrap()).unwrap();
        assert!(validate_model_shapes(&model,reader).unwrap_err().to_string().contains("obsolete or unexpected tensor"));
    }

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
