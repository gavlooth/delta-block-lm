//! Shared types every encoder operator uses: the GPU device and the layout of documents inside a padded batch.
//!
//! A batch is `[B, T]` token positions. Each row holds one or more documents packed back to back, followed by
//! padding. Every operator must keep documents separate: no information may cross from one document to another,
//! and padded positions must not influence any valid position.

pub mod profile;

use burn::prelude::*;
use burn::tensor::{DType, TensorData};
#[cfg(not(feature = "cuda"))]
use burn::tensor::DeviceKind;

/// Selects storage and projection arithmetic; sensitive reductions remain FP32.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    #[default]
    Fp32,
    Bf16Mixed,
}

impl Precision {
    pub fn dtype(self) -> DType {
        match self {
            Self::Fp32 => DType::F32,
            Self::Bf16Mixed => DType::BF16,
        }
    }
}

/// Explicit mixed projection with FP32 master parameters and bias/reduction boundary.
/// The casts remain in the autodiff graph, so gradients return to the FP32 parameters.
pub fn project<const D: usize>(layer: &burn::nn::Linear, input: Tensor<D>, mixed: bool) -> Tensor<D> {
    if !mixed {
        return layer.forward(input);
    }
    let output = burn::tensor::module::linear(
        input.cast(DType::BF16),
        layer.weight.val().cast(DType::BF16),
        None,
    ).cast(DType::F32);
    match &layer.bias {
        Some(bias) => output + bias.val().unsqueeze(),
        None => output,
    }
}

/// CUDA device 0 with `cuda`; otherwise the default Vulkan integrated adapter.
pub fn gpu_device() -> Device {
    static CUBECL_ENVIRONMENT: std::sync::Once = std::sync::Once::new();
    CUBECL_ENVIRONMENT.call_once(|| {
        use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

        // Kernel selection changed: never reuse Vulkan subgroup choices or CUDA
        // matmul tuning that silently lowered FP32 operands to TF32.
        let suffix = if cfg!(feature = "cuda") { "-dre-cuda-ieee-f32-v1" } else { "-dre-plane-width-1" };
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        #[cfg(feature = "cuda")]
        {
            // One backing allocator only: CUDA's async pool can reserve memory
            // outside the byte count enforced by our CubeCL allocation budget.
            config.memory.cuda.allocator = burn::cubecl::config::memory::CudaAllocator::Sync;
        }
        config.environment.name.push_str(suffix);
        if !CubeClRuntimeConfig::try_set(config) {
            #[cfg(feature = "cuda")]
            assert!(
                matches!(CubeClRuntimeConfig::get().memory.cuda.allocator, burn::cubecl::config::memory::CudaAllocator::Sync),
                "CUDA allocation safety requires the synchronous backing allocator before runtime initialization"
            );
            let mut name = CubeClRuntimeConfig::get().environment.name.clone();
            name.push_str(suffix);
            burn::cubecl::environment::activate(name);
        }
    });
    #[cfg(feature = "cuda")]
    { Device::cuda(0) }
    #[cfg(not(feature = "cuda"))]
    { Device::vulkan(DeviceKind::IntegratedGpu(0)) }
}

/// Document layout of a `[B, T]` batch, kept both on the host (for kernels and planners) and on the device.
///
/// For a valid position `t` in row `b`, `doc_start[b][t]` and `doc_end[b][t]` are the first and last positions of
/// the document containing `t` (inclusive). Padded positions have `doc_start = t` and `doc_end = t - 1`, so
/// `doc_end < doc_start` marks padding.
#[derive(Debug, Clone)]
pub struct DocumentLayout {
    batch: usize,
    tokens: usize,
    /// Lengths of the documents packed into each row, in order.
    documents_per_row: Vec<Vec<usize>>,
    document_count: usize,
    document_capacity: usize,
}

impl DocumentLayout {
    /// One document per row with the given lengths; `tokens` is the padded row length.
    pub fn one_document_per_row(lengths: &[usize], tokens: usize) -> Self {
        Self::packed(lengths.iter().map(|&length| vec![length]).collect(), tokens)
    }

    /// Several documents per row, packed from position 0; the rest of each row is padding.
    pub fn packed(documents_per_row: Vec<Vec<usize>>, tokens: usize) -> Self {
        for (row, documents) in documents_per_row.iter().enumerate() {
            let used: usize = documents.iter().sum();
            assert!(used <= tokens, "row {row} holds {used} tokens but rows have {tokens} positions");
            assert!(documents.iter().all(|&length| length > 0), "row {row} has an empty document");
        }
        let document_count = documents_per_row.iter().map(Vec::len).sum();
        Self { batch: documents_per_row.len(), tokens, documents_per_row, document_count, document_capacity: document_count }
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn documents_per_row(&self) -> &[Vec<usize>] {
        &self.documents_per_row
    }

    /// Number of real, nonempty documents. Physical document padding never appears in the row lists.
    pub fn document_count(&self) -> usize {
        self.document_count
    }

    /// Physical relay document axis; its first `document_count()` entries are the real documents.
    pub fn document_capacity(&self) -> usize {
        self.document_capacity
    }

    pub fn with_document_padding(mut self, capacity: usize) -> Self {
        assert!(capacity >= self.document_count, "document padding cannot remove real documents");
        assert!(capacity == 0 || (self.batch > 0 && self.tokens > 0), "document padding needs a physical source row");
        self.document_capacity = capacity;
        self
    }

    /// Shared source-position and relay-document metadata, including inactive execution entries.
    pub fn document_metadata_host(&self) -> DocumentMetadata {
        let mut metadata = DocumentMetadata {
            doc_start: (0..self.batch * self.tokens).map(|position| (position % self.tokens) as i32).collect(),
            doc_end: (0..self.batch * self.tokens).map(|position| (position % self.tokens) as i32 - 1).collect(),
            source_valid: vec![0.0; self.batch * self.tokens],
            position_ids: vec![0; self.batch * self.tokens],
            document_id: vec![-1; self.batch * self.tokens],
            document_rows: vec![0; self.document_capacity],
            document_start: vec![self.tokens as i32; self.document_capacity],
            // Keep this nonnegative for T>0: the relay backward kernel reads unsigned empty ranges.
            document_end: vec![self.tokens as i32 - 1; self.document_capacity],
            document_valid: vec![0.0; self.document_capacity],
            document_outside: vec![true; self.document_capacity * self.tokens],
        };
        let mut document = 0;
        for (row, documents) in self.documents_per_row.iter().enumerate() {
            let mut first = 0;
            for &length in documents {
                let last = first + length - 1;
                metadata.document_rows[document] = row as i32;
                metadata.document_start[document] = first as i32;
                metadata.document_end[document] = last as i32;
                metadata.document_valid[document] = 1.0;
                metadata.document_outside[document * self.tokens + first..=document * self.tokens + last].fill(false);
                for position in first..=last {
                    metadata.doc_start[row * self.tokens + position] = first as i32;
                    metadata.doc_end[row * self.tokens + position] = last as i32;
                    metadata.source_valid[row * self.tokens + position] = 1.0;
                    metadata.position_ids[row * self.tokens + position] = (position - first) as i32;
                    metadata.document_id[row * self.tokens + position] = document as i32;
                }
                document += 1;
                first += length;
            }
        }
        metadata
    }

    /// Host copy of `(doc_start, doc_end)` per position, row-major `[B * T]`, as `i32`.
    pub fn limits_host(&self) -> (Vec<i32>, Vec<i32>) {
        let mut starts = Vec::with_capacity(self.batch * self.tokens);
        let mut ends = Vec::with_capacity(self.batch * self.tokens);
        for documents in &self.documents_per_row {
            let mut position = 0usize;
            for &length in documents {
                for _ in 0..length {
                    starts.push(position as i32);
                    ends.push((position + length - 1) as i32);
                }
                position += length;
            }
            for t in position..self.tokens {
                starts.push(t as i32);
                ends.push(t as i32 - 1);
            }
        }
        (starts, ends)
    }

    /// Device copy of `(doc_start, doc_end)` as `Int` tensors `[B, T]`.
    pub fn limits(&self, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
        let (starts, ends) = self.limits_host();
        let shape = [self.batch, self.tokens];
        (
            Tensor::from_data(TensorData::new(starts, shape), device),
            Tensor::from_data(TensorData::new(ends, shape), device),
        )
    }

    /// `true` at valid (non-padding) positions, `[B, T]`.
    pub fn valid_mask(&self, device: &Device) -> Tensor<2, Bool> {
        let (starts, ends) = self.limits_host();
        let valid: Vec<bool> = starts.iter().zip(&ends).map(|(start, end)| end >= start).collect();
        Tensor::from_data(TensorData::new(valid, [self.batch, self.tokens]), device)
    }

    /// Valid positions as `1.0`, padding as `0.0`, shaped `[B, T, 1]` for multiplying features.
    pub fn valid_weights(&self, device: &Device) -> Tensor<3> {
        self.valid_mask(device).float().unsqueeze_dim(2)
    }
}

/// Host metadata uses exactly the same real-document prefix as the persistent relay bank.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentMetadata {
    pub doc_start: Vec<i32>,
    pub doc_end: Vec<i32>,
    pub source_valid: Vec<f32>,
    pub position_ids: Vec<i32>,
    pub document_id: Vec<i32>,
    pub document_rows: Vec<i32>,
    pub document_start: Vec<i32>,
    pub document_end: Vec<i32>,
    pub document_valid: Vec<f32>,
    pub document_outside: Vec<bool>,
}

/// A [`DocumentLayout`] together with its device tensors, built once per batch and shared by every operator.
#[derive(Debug, Clone)]
pub struct BatchLayout {
    pub host: DocumentLayout,
    /// First position of each position's document, `[B, T]`.
    pub doc_start: Tensor<2, Int>,
    /// Last position of each position's document, `[B, T]`; smaller than `doc_start` at padding.
    pub doc_end: Tensor<2, Int>,
    /// `1.0` at valid positions, `0.0` at padding, `[B, T, 1]`.
    pub valid: Tensor<3>,
    /// Document-local source positions, `[B,T]`; source padding uses position 0 and remains masked.
    pub position_ids: Tensor<2, Int>,
    /// Flattened real-document IDs per source position, `[B,T]`; source padding is -1.
    pub document_id: Tensor<2, Int>,
    pub document_rows: Tensor<1, Int>,
    pub document_start: Tensor<1, Int>,
    pub document_end: Tensor<1, Int>,
    /// Real relay documents are 1, inactive document capacity is 0, `[Dexec,1,1]`.
    pub document_valid: Tensor<3>,
    /// Relay-to-source key mask, `[Dexec,1,T]`; inactive documents mask every source key.
    pub document_outside: Tensor<3, Bool>,
}

impl BatchLayout {
    pub fn new(host: DocumentLayout, device: &Device) -> Self {
        let metadata = host.document_metadata_host();
        let (batch, tokens, documents) = (host.batch(), host.tokens(), host.document_capacity());
        Self {
            host,
            doc_start: Tensor::from_data(TensorData::new(metadata.doc_start, [batch, tokens]), device),
            doc_end: Tensor::from_data(TensorData::new(metadata.doc_end, [batch, tokens]), device),
            valid: Tensor::from_data(TensorData::new(metadata.source_valid, [batch, tokens, 1]), device),
            position_ids: Tensor::from_data(TensorData::new(metadata.position_ids, [batch, tokens]), device),
            document_id: Tensor::from_data(TensorData::new(metadata.document_id, [batch, tokens]), device),
            document_rows: Tensor::from_data(TensorData::new(metadata.document_rows, [documents]), device),
            document_start: Tensor::from_data(TensorData::new(metadata.document_start, [documents]), device),
            document_end: Tensor::from_data(TensorData::new(metadata.document_end, [documents]), device),
            document_valid: Tensor::from_data(TensorData::new(metadata.document_valid, [documents, 1, 1]), device),
            document_outside: Tensor::from_data(TensorData::new(metadata.document_outside, [documents, 1, tokens]), device),
        }
    }

    pub fn without_autodiff(&self) -> Self {
        Self {
            host: self.host.clone(), doc_start: self.doc_start.clone().inner(), doc_end: self.doc_end.clone().inner(),
            valid: self.valid.clone().inner(), position_ids: self.position_ids.clone().inner(),
            document_id: self.document_id.clone().inner(), document_rows: self.document_rows.clone().inner(),
            document_start: self.document_start.clone().inner(), document_end: self.document_end.clone().inner(),
            document_valid: self.document_valid.clone().inner(), document_outside: self.document_outside.clone().inner(),
        }
    }

    pub fn with_autodiff(&self) -> Self {
        Self {
            host: self.host.clone(), doc_start: self.doc_start.clone().autodiff(), doc_end: self.doc_end.clone().autodiff(),
            valid: self.valid.clone().autodiff(), position_ids: self.position_ids.clone().autodiff(),
            document_id: self.document_id.clone().autodiff(), document_rows: self.document_rows.clone().autodiff(),
            document_start: self.document_start.clone().autodiff(), document_end: self.document_end.clone().autodiff(),
            document_valid: self.document_valid.clone().autodiff(), document_outside: self.document_outside.clone().autodiff(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_limits_mark_documents_and_padding() {
        let layout = DocumentLayout::packed(vec![vec![2, 3], vec![1]], 6);
        let (starts, ends) = layout.limits_host();
        assert_eq!(starts, vec![0, 0, 2, 2, 2, 5, 0, 1, 2, 3, 4, 5]);
        assert_eq!(ends, vec![1, 1, 4, 4, 4, 4, 0, 0, 1, 2, 3, 4]);
    }

    #[test]
    fn exp004_document_capacity_keeps_real_rows_and_empty_ranges_cpu() {
        let rows = vec![vec![2, 1], vec![1], vec![], vec![]];
        let host = DocumentLayout::packed(rows.clone(), 4).with_document_padding(4);
        assert_eq!((host.batch(), host.document_count(), host.document_capacity()), (4, 3, 4));
        assert_eq!(host.documents_per_row(), rows.as_slice());
        let metadata = host.document_metadata_host();
        let (starts, ends) = host.limits_host();
        assert_eq!(metadata.doc_start, starts);
        assert_eq!(metadata.doc_end, ends);
        assert_eq!(metadata.position_ids, vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(metadata.document_id, vec![0, 0, 1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]);
        assert_eq!(metadata.document_rows, vec![0, 0, 1, 0]);
        assert_eq!(metadata.document_start, vec![0, 2, 0, 4]);
        assert_eq!(metadata.document_end, vec![1, 2, 0, 3]);
        assert_eq!(metadata.document_valid, vec![1.0, 1.0, 1.0, 0.0]);
        assert_eq!(metadata.document_outside, vec![
            false, false, true, true, true, true, false, true,
            false, true, true, true, true, true, true, true,
        ]);
        let device = Device::ndarray().autodiff();
        let layout = BatchLayout::new(host, &device);
        assert_eq!(layout.position_ids.dims(), [4, 4]);
        assert_eq!(layout.document_valid.dims(), [4, 1, 1]);
        assert_eq!(layout.document_outside.dims(), [4, 1, 4]);
        let detached = layout.without_autodiff();
        let attached = detached.with_autodiff();
        assert_eq!(attached.document_id.to_data(), layout.document_id.to_data());
        assert_eq!(attached.document_outside.to_data(), layout.document_outside.to_data());
        assert_eq!(attached.host.document_capacity(), 4);
        let source = Tensor::<3>::from_data(TensorData::new(vec![0.75f32; 4 * 4 * 2], [4, 4, 2]), &device).require_grad();
        let output = source.clone() * layout.valid.clone();
        let values = output.to_data().try_to_vec_as::<f32>().unwrap();
        assert!(values[2 * 4 * 2..].iter().all(|&value| value == 0.0));
        let gradients = output.sum().backward();
        let gradient = source.grad(&gradients).unwrap().into_data().try_to_vec_as::<f32>().unwrap();
        let valid = layout.valid.to_data().try_to_vec_as::<f32>().unwrap();
        for (position, channels) in gradient.chunks(2).enumerate() {
            assert!(channels.iter().all(|&value| value == valid[position]));
        }
    }

    #[test]
    fn exp004_inactive_document_axis_is_not_a_logical_document_cpu() {
        let rows = vec![vec![], vec![]];
        let host = DocumentLayout::packed(rows.clone(), 4).with_document_padding(2);
        assert_eq!(host.document_count(), 0);
        assert_eq!(host.documents_per_row(), rows.as_slice());
        let metadata = host.document_metadata_host();
        assert_eq!(metadata.document_start, vec![4, 4]);
        assert_eq!(metadata.document_end, vec![3, 3]);
        assert!(metadata.document_outside.iter().all(|&outside| outside));
        assert!(metadata.document_valid.iter().all(|&valid| valid == 0.0));
        assert!(metadata.document_id.iter().all(|&document| document == -1));
        let layout = BatchLayout::new(host, &Device::ndarray());
        assert_eq!(layout.document_valid.dims(), [2, 1, 1]);
        assert!(layout.valid.into_data().try_to_vec_as::<f32>().unwrap().iter().all(|&valid| valid == 0.0));
    }

    #[test]
    #[should_panic(expected = "document padding cannot remove real documents")]
    fn exp004_document_capacity_never_truncates_cpu() {
        let _ = DocumentLayout::packed(vec![vec![1, 1]], 2).with_document_padding(1);
    }
}
