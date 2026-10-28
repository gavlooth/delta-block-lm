use crate::config::Config;
use anyhow::{Context, Result, ensure};
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
#[derive(Clone)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn index(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.index(i + 1);
            values.swap(i, j);
        }
    }
}
#[derive(Clone, Debug)]
pub struct Example {
    pub input: Vec<i32>,
    pub target: Vec<i32>,
    pub weight: Vec<f32>,
    pub groups: Vec<Option<(usize, usize)>>,
}
impl Example {
    pub fn fingerprint(&self) -> String {
        let bytes: Vec<u8> = self
            .input
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .chain(self.target.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        hash(&bytes)
    }
}
pub struct Batch {
    pub input: Tensor<2, Int>,
    pub target: Tensor<2, Int>,
    pub weight: Tensor<2>,
    pub valid: Tensor<2>,
    pub reset: Tensor<2>,
}
impl Batch {
    pub fn new(examples: &[Example], device: &Device) -> Self {
        HostBatch::new(examples).upload(device)
    }
}

/// Owned upload payloads: exact capacities avoid geometric growth while flattening rows.
/// Upload consumes these buffers; retaining a second host copy would not be buffer reuse.
pub(crate) struct HostBatch {
    shape: [usize; 2],
    input: Vec<i32>,
    target: Vec<i32>,
    weight: Vec<f32>,
    reset: Vec<f32>,
}
impl HostBatch {
    pub(crate) fn new(examples: &[Example]) -> Self {
        let b = examples.len();
        let t = examples[0].input.len();
        let mut input = Vec::with_capacity(b * t);
        let mut target = Vec::with_capacity(b * t);
        let mut weight = Vec::with_capacity(b * t);
        let mut reset = vec![0.0; b * t];
        for (row, example) in examples.iter().enumerate() {
            input.extend_from_slice(&example.input);
            target.extend_from_slice(&example.target);
            weight.extend_from_slice(&example.weight);
            reset[row * t] = 1.0;
        }
        Self {
            shape: [b, t],
            input,
            target,
            weight,
            reset,
        }
    }
    pub(crate) fn upload(self, device: &Device) -> Batch {
        Batch {
            input: Tensor::from_data(TensorData::new(self.input, self.shape), device),
            target: Tensor::from_data(TensorData::new(self.target, self.shape), device),
            weight: Tensor::from_data(TensorData::new(self.weight, self.shape), device),
            valid: Tensor::ones(self.shape, device),
            reset: Tensor::from_data(TensorData::new(self.reset, self.shape), device),
        }
    }
}
fn recall_candidate(c: &Config, rng: &mut Rng, count: usize) -> Example {
    let mut keys: Vec<usize> = (0..c.key_symbols).collect();
    rng.shuffle(&mut keys);
    keys.truncate(count);
    let values: Vec<usize> = (0..count).map(|_| rng.index(c.value_symbols)).collect();
    let mut sequence = vec![1];
    let mut writes = Vec::with_capacity(count);
    for (&key, &value) in keys.iter().zip(&values) {
        writes.push(sequence.len());
        sequence.push((4 + key) as i32);
        sequence.push((4 + c.key_symbols + value) as i32);
    }
    sequence.resize(c.context + 1 - 3 * count, 3);
    let mut order: Vec<usize> = (0..count).collect();
    rng.shuffle(&mut order);
    let mut groups = vec![None; c.context];
    let mut weight = vec![0.0; c.context];
    for index in order {
        sequence.push(2);
        sequence.push((4 + keys[index]) as i32);
        let position = sequence.len() - 1;
        groups[position] = Some((count, position - writes[index]));
        weight[position] = 1.0;
        sequence.push((4 + c.key_symbols + values[index]) as i32);
    }
    assert_eq!(sequence.len(), c.context + 1);
    Example {
        input: sequence[..c.context].to_vec(),
        target: sequence[1..].to_vec(),
        weight,
        groups,
    }
}
pub fn recall(c: &Config, seed: u64, index: usize, evaluation: bool) -> Example {
    let count = c.association_counts[index % c.association_counts.len()];
    let mut rng = Rng(seed.wrapping_add((index as u64).wrapping_mul(0xd1342543de82ef95)));
    loop {
        let e = recall_candidate(c, &mut rng, count);
        // Partition by complete observable sequence AND labels: collisions can never cross splits.
        let split = Sha256::digest(e.fingerprint().as_bytes())[0] & 1;
        if (split == 1) == evaluation {
            return e;
        }
    }
}
#[derive(Deserialize, Serialize)]
struct Document {
    id: String,
    split: String,
    path: String,
    sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    tokenizer: String,
    documents: Vec<Document>,
}
#[derive(Clone)]
pub struct Corpus {
    pub train: Vec<Vec<u8>>,
    pub validation: Vec<Vec<u8>>,
    pub test: Vec<Vec<u8>>,
    pub manifest_sha256: String,
}
impl Corpus {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("prepare corpus first: {}", path.display()))?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        ensure!(
            manifest.tokenizer == "utf8-bytes-v1-bos256",
            "unsupported tokenizer"
        );
        let mut corpus = Self {
            train: Vec::new(),
            validation: Vec::new(),
            test: Vec::new(),
            manifest_sha256: hash(&bytes),
        };
        let root = path.parent().context("manifest parent")?;
        let mut ids = BTreeSet::new();
        let mut contents = BTreeSet::new();
        for doc in manifest.documents {
            ensure!(
                ids.insert(doc.id),
                "document belongs to more than one split"
            );
            let relative = Path::new(&doc.path);
            ensure!(
                !relative.is_absolute()
                    && relative
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_))),
                "unsafe corpus document path"
            );
            let text = std::fs::read(root.join(relative))?;
            let digest = hash(&text);
            ensure!(digest == doc.sha256, "corpus document hash mismatch");
            ensure!(
                contents.insert(digest),
                "duplicate content across corpus documents"
            );
            match doc.split.as_str() {
                "train" => corpus.train.push(text),
                "validation" => corpus.validation.push(text),
                "test" => corpus.test.push(text),
                _ => anyhow::bail!("unknown split"),
            }
        }
        ensure!(
            !corpus.train.is_empty() && !corpus.validation.is_empty() && !corpus.test.is_empty(),
            "all document-disjoint splits required"
        );
        Ok(corpus)
    }
    pub fn example(&self, c: &Config, seed: u64, index: usize, split: &str) -> Result<Example> {
        let docs = match split {
            "train" => &self.train,
            "validation" => &self.validation,
            "test" => &self.test,
            _ => anyhow::bail!("unknown split"),
        };
        ensure!(
            docs.iter().all(|d| d.len() > c.context),
            "document shorter than context"
        );
        let (doc, offset) = if split == "train" {
            let mut rng = Rng(seed.wrapping_add(index as u64 * 0x9e3779b9));
            let d = rng.index(docs.len());
            (d, rng.index(docs[d].len() - c.context))
        } else {
            let total: usize = docs.iter().map(|d| d.len() / c.context).sum();
            ensure!(
                total >= c.eval_examples,
                "heldout subset exceeds distinct complete document windows"
            );
            // One nonoverlapping midpoint window per stratum, spread across the full heldout text.
            let stratum = index % c.eval_examples;
            let mut window = ((2 * stratum + 1) * total) / (2 * c.eval_examples);
            let mut d = 0;
            while window >= docs[d].len() / c.context {
                window -= docs[d].len() / c.context;
                d += 1;
            }
            (d, window * c.context)
        };
        let bytes = &docs[doc][offset..offset + c.context];
        let mut input = Vec::with_capacity(c.context);
        input.push(256);
        input.extend(bytes[..c.context - 1].iter().map(|&v| v as i32));
        Ok(Example {
            input,
            target: bytes.iter().map(|&v| v as i32).collect(),
            weight: vec![1.0; c.context],
            groups: vec![None; c.context],
        })
    }
}
pub fn examples(
    c: &Config,
    corpus: Option<&Corpus>,
    seed: u64,
    start: usize,
    count: usize,
    split: &str,
) -> Result<Vec<Example>> {
    (start..start + count)
        .map(|index| {
            if let Some(corpus) = corpus {
                corpus.example(c, seed, index, split)
            } else {
                Ok(recall(
                    c,
                    if split == "train" { seed } else { c.eval_seed },
                    index,
                    split != "train",
                ))
            }
        })
        .collect()
}
pub fn fixture_hash(examples: &[Example]) -> String {
    hash(
        examples
            .iter()
            .flat_map(|e| e.fingerprint().into_bytes())
            .collect::<Vec<_>>()
            .as_slice(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_batch_preserves_row_order_sparse_weights_and_resets() {
        let rows = vec![
            Example {
                input: vec![256, 7, 8],
                target: vec![7, 8, 9],
                weight: vec![0.0, 0.5, 1.0],
                groups: vec![None; 3],
            },
            Example {
                input: vec![256, 3, 4],
                target: vec![3, 4, 5],
                weight: vec![1.0, 0.0, 0.25],
                groups: vec![None; 3],
            },
        ];
        let host = HostBatch::new(&rows);
        assert_eq!(host.shape, [2, 3]);
        assert_eq!(host.input, [256, 7, 8, 256, 3, 4]);
        assert_eq!(host.target, [7, 8, 9, 3, 4, 5]);
        assert_eq!(host.weight, [0.0, 0.5, 1.0, 1.0, 0.0, 0.25]);
        assert_eq!(host.reset, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let batch = host.upload(&Device::ndarray());
        assert_eq!(batch.input.dims(), [2, 3]);
        assert_eq!(
            batch.input.into_data().try_to_vec_as::<i32>().unwrap(),
            [256, 7, 8, 256, 3, 4]
        );
        assert_eq!(
            batch.target.into_data().try_to_vec_as::<i32>().unwrap(),
            [7, 8, 9, 3, 4, 5]
        );
        assert_eq!(
            batch.weight.into_data().try_to_vec::<f32>().unwrap(),
            [0.0, 0.5, 1.0, 1.0, 0.0, 0.25]
        );
        assert_eq!(
            batch.valid.into_data().try_to_vec::<f32>().unwrap(),
            [1.0; 6]
        );
        assert_eq!(
            batch.reset.into_data().try_to_vec::<f32>().unwrap(),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }
    #[test]
    fn recall_is_fixed_disjoint_and_causal() {
        let c = Config::default();
        let eval: Vec<_> = (0..40).map(|i| recall(&c, c.eval_seed, i, true)).collect();
        assert_eq!(
            fixture_hash(&eval),
            fixture_hash(
                &(0..40)
                    .map(|i| recall(&c, c.eval_seed, i, true))
                    .collect::<Vec<_>>()
            )
        );
        let fingerprints: BTreeSet<_> = eval.iter().map(Example::fingerprint).collect();
        for i in 0..100 {
            let train = recall(&c, 1, i, false);
            assert!(!fingerprints.contains(&train.fingerprint()));
            for (t, g) in train.groups.iter().enumerate() {
                if let Some((_, distance)) = g {
                    assert_eq!(train.input[t], train.input[t - distance]);
                    assert_eq!(train.target[t], train.target[t - distance]);
                }
            }
        }
    }
    #[test]
    fn heldout_windows_cover_document_strata_without_overlap() {
        let mut c = Config::default();
        c.context = 32;
        c.eval_examples = 4;
        let text: Vec<u8> = (0..1024).map(|i| (i / 32) as u8).collect();
        let corpus = Corpus {
            train: vec![],
            validation: vec![text],
            test: vec![],
            manifest_sha256: String::new(),
        };
        let starts: Vec<_> = (0..4)
            .map(|i| corpus.example(&c, 1, i, "validation").unwrap().target[0])
            .collect();
        assert_eq!(starts, vec![4, 12, 20, 28]);
        assert!(starts.windows(2).all(|p| p[1] > p[0]));
    }
}
