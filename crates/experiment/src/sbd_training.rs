//! Training snapshots from the ordinary unsegmented clean forward.
use crate::{model::{DenoisingOutput, LanguageModel}, soft::SoftState};
use burn::prelude::*;
use delta_mixers::{profile::Sections, sbd::{BatchedMixerCache,BatchedWaveletCache,MixerCache,WaveletCache}};

#[derive(Default)]
pub struct CleanInputs(pub Vec<(usize,&'static str,Tensor<3>)>);
impl Sections for CleanInputs {
    const CAPTURE_CLEAN_INPUTS: bool = true;
    fn clean_input(&mut self, layer: usize, kind: &'static str, input: &Tensor<3>) {
        self.0.push((layer,kind,input.clone()));
    }
    fn section<T>(&mut self, _name: &'static str, run: impl FnOnce()->T) -> T {run()}
}
struct LayerCache {
    wavelet: Option<WaveletCache>,
    lngram: Option<Tensor<3>>,
    mixer: MixerCache,
}
pub struct TrainingCache { layers: Vec<LayerCache>, starts: Vec<usize> }
struct BatchedLngramCache {
    history: Option<(Tensor<3>,Tensor<2>)>,
}
impl BatchedLngramCache {
    fn new(lngram: &delta_mixers::lngram::Lngram, input: Tensor<3>, blocks: &[(usize,usize)]) -> Self {
        let history = lngram.orders.iter().copied().max().unwrap()-1
            + lngram.smoothing.as_ref().map_or(0,|s|2*s.dilation);
        if history == 0 {
            return Self {history:None};
        }
        let [rows,t,d] = input.dims();
        let device = input.device();
        let mut ids = Vec::with_capacity(blocks.len()*history);
        let mut valid = Vec::with_capacity(blocks.len()*history);
        for &(row,start) in blocks {
            assert!(row < rows && start <= t);
            for position in 0..history {
                let lag = history-position;
                ids.push((row*t+start.saturating_sub(lag)) as i32);
                valid.push(if start >= lag {1.0f32} else {0.0});
            }
        }
        let valid = Tensor::from_data(TensorData::new(valid,[blocks.len(),history]),&device);
        let selected = input.reshape([rows*t,d])
            .select(0,Tensor::<1,Int>::from_data(TensorData::new(ids,[blocks.len()*history]),&device))
            .reshape([blocks.len(),history,d])
            .mask_fill(valid.clone().equal_elem(0.0).unsqueeze_dim::<3>(2).expand([blocks.len(),history,d]),0.0);
        Self {history:Some((selected,valid))}
    }
    fn forward(&self, lngram: &delta_mixers::lngram::Lngram, input: Tensor<3>, valid: Tensor<2>) -> Tensor<3> {
        let [blocks,n,d] = input.dims();
        let device = input.device();
        let (combined,valid,history) = match &self.history {
            Some((clean,clean_valid)) => (
                Tensor::cat(vec![clean.clone(),input],1),
                Tensor::cat(vec![clean_valid.clone(),valid],1),
                clean.dims()[1],
            ),
            None => (input,valid,0),
        };
        let layout = delta_mixers::lngram::LngramLayout::new(valid,Tensor::zeros([blocks,history+n],&device));
        lngram.forward(combined,&layout).slice([0..blocks,history..history+n,0..d])
    }
}
struct BatchedLayerCache {
    wavelet: Option<BatchedWaveletCache>,
    lngram: Option<BatchedLngramCache>,
    mixer: BatchedMixerCache,
}
pub struct BatchedTrainingCache { layers: Vec<BatchedLayerCache>, blocks: usize }
impl LanguageModel {
    pub fn training_cache(&self, trace: &CleanInputs, row: usize, starts: &[usize]) -> TrainingCache {
        let layers = self.blocks.iter().enumerate().map(|(index,block)| {
            let get = |kind| {
                let x = &trace.0.iter().find(|(i,k,_)| *i == index && *k == kind).expect("clean training sublayer").2;
                let [_,t,d] = x.dims();
                x.clone().slice([row..row+1,0..t,0..d])
            };
            LayerCache {
                wavelet: block.wavelet.as_ref().map(|wavelet| wavelet.prepare_sbd(get("wavelet"))),
                lngram: block.lngram.as_ref().map(|_| get("lngram")),
                mixer: block.mixer.prepare_sbd(get("mixer"),starts),
            }
        }).collect();
        TrainingCache {layers,starts:starts.to_vec()}
    }
    pub fn cached_block(&self, cache: &TrainingCache, block_index: usize, soft: &SoftState, hard: bool) -> DenoisingOutput {
        self.cached_block_profiled(cache,block_index,soft,hard,&mut delta_mixers::profile::Unprofiled)
    }
    pub fn cached_block_profiled<P: Sections>(&self, cache: &TrainingCache, block_index: usize, soft: &SoftState, hard: bool, profile: &mut P) -> DenoisingOutput {
        let n = soft.observed.len();
        let start = cache.starts[block_index];
        let d = self.embedding.weight.val().dims()[1];
        let device = self.embedding.weight.val().device();
        let valid = Tensor::from_data(TensorData::new(soft.active.iter().map(|&x| if x {1.0f32} else {0.0}).collect::<Vec<_>>(),[1,n]),&device);
        let mut hidden = soft.embeddings(self.embedding.weight.val(),self.sbd_mask.as_ref().unwrap().val(),hard).reshape([1,n,d]);
        let lngram_forward = |lngram: &delta_mixers::lngram::Lngram, input: Tensor<3>, clean: &Tensor<3>| {
            let history = start.min(lngram.orders.iter().copied().max().unwrap()-1 + lngram.smoothing.as_ref().map_or(0,|s|2*s.dilation));
            let combined = Tensor::cat(vec![clean.clone().slice([0..1,start-history..start,0..d]),input],1);
            let layout = delta_mixers::lngram::LngramLayout::new(Tensor::ones([1,history+n],&device),Tensor::zeros([1,history+n],&device));
            lngram.forward(combined,&layout).slice([0..1,history..history+n,0..d])
        };
        if let Some(init) = &self.ohc_init {
            let mut streams = init.forward(hidden,valid.clone());
            for (index,block) in self.blocks.iter().enumerate() {
                let clean = &cache.layers[index];
                if let (Some(stage),Some(norm),Some(wavelet)) = (&block.ohc_wavelet,&block.wavelet_norm,&block.wavelet) {
                    let read = stage.read_selected(streams,self.ohc_backend);
                    let input = self.normalization_backend.rms(norm,read.work.clone());
                    let update = wavelet.forward_sbd_cached(input,clean.wavelet.as_ref().unwrap(),start);
                    streams = stage.write(read,update,valid.clone(),self.ohc_backend);
                }
                if let (Some(stage),Some(norm),Some(lngram)) = (&block.ohc_lngram,&block.lngram_norm,&block.lngram) {
                    let read = stage.read_selected(streams,self.ohc_backend);
                    let input = self.normalization_backend.rms(norm,read.work.clone());
                    let update = lngram_forward(lngram,input,clean.lngram.as_ref().unwrap());
                    streams = stage.write(read,update,valid.clone(),self.ohc_backend);
                }
                let stage = block.ohc_mixer.as_ref().unwrap();
                let read = stage.read_selected(streams,self.ohc_backend);
                let input = self.normalization_backend.rms(&block.norm,read.work.clone());
                let update = block.mixer.forward_sbd_cached_profiled(input,&clean.mixer,block_index,valid.clone(),profile);
                streams = stage.write(read,update,valid.clone(),self.ohc_backend);
                let stage = block.ohc_mlp.as_ref().unwrap();
                let read = stage.read_selected(streams,self.ohc_backend);
                let input = self.normalization_backend.rms(&block.ff_norm,read.work.clone());
                let update = block.down.forward(self.epilogue_backend.silu(block.up.forward(input)));
                streams = stage.write(read,update,valid.clone(),self.ohc_backend);
            }
            hidden = self.ohc_readout.as_ref().unwrap().readout(streams);
        } else {
            for (index,block) in self.blocks.iter().enumerate() {
                let clean = &cache.layers[index];
                if let (Some(norm),Some(wavelet)) = (&block.wavelet_norm,&block.wavelet) {
                    let input = self.normalization_backend.rms(norm,hidden.clone());
                    hidden = hidden + wavelet.forward_sbd_cached(input,clean.wavelet.as_ref().unwrap(),start);
                }
                if let (Some(norm),Some(lngram)) = (&block.lngram_norm,&block.lngram) {
                    let input = self.normalization_backend.rms(norm,hidden.clone());
                    hidden = hidden + lngram_forward(lngram,input,clean.lngram.as_ref().unwrap());
                }
                let input = self.normalization_backend.rms(&block.norm,hidden.clone());
                hidden = hidden + block.mixer.forward_sbd_cached_profiled(input,&clean.mixer,block_index,valid.clone(),profile);
                let input = self.normalization_backend.rms(&block.ff_norm,hidden.clone());
                hidden = hidden + block.down.forward(self.epilogue_backend.silu(block.up.forward(input)));
            }
        }
        let features = self.normalization_backend.rms(&self.norm,hidden);
        DenoisingOutput { logits: self.head_forward(features.clone()), features }
    }
    pub fn training_cache_batched(&self, trace: &CleanInputs, blocks: &[(usize,usize)]) -> BatchedTrainingCache {
        assert!(!blocks.is_empty());
        let layers = self.blocks.iter().enumerate().map(|(index,block)| {
            let get = |kind| trace.0.iter()
                .find(|(i,k,_)| *i == index && *k == kind)
                .expect("clean training sublayer").2.clone();
            BatchedLayerCache {
                wavelet: block.wavelet.as_ref().map(|wavelet| wavelet.prepare_sbd_batched(get("wavelet"),blocks)),
                lngram: block.lngram.as_ref().map(|lngram| BatchedLngramCache::new(lngram,get("lngram"),blocks)),
                mixer: block.mixer.prepare_sbd_batched(get("mixer"),blocks),
            }
        }).collect();
        BatchedTrainingCache {layers,blocks:blocks.len()}
    }
    /// The soft state is block-major, with inactive right padding in each row.
    pub fn cached_blocks(&self, cache: &BatchedTrainingCache, soft: &SoftState, hard: bool) -> DenoisingOutput {
        self.cached_blocks_profiled(cache,soft,hard,&mut delta_mixers::profile::Unprofiled)
    }
    pub fn cached_blocks_profiled<P: Sections>(&self, cache: &BatchedTrainingCache, soft: &SoftState, hard: bool, profile: &mut P) -> DenoisingOutput {
        let blocks = cache.blocks;
        assert!(soft.observed.len().is_multiple_of(blocks));
        let n = soft.observed.len()/blocks;
        assert!(n > 0);
        let d = self.embedding.weight.val().dims()[1];
        let device = self.embedding.weight.val().device();
        let valid = Tensor::from_data(TensorData::new(
            soft.active.iter().map(|&x| if x {1.0f32} else {0.0}).collect::<Vec<_>>(),[blocks,n]),&device);
        let mut hidden = soft.embeddings(self.embedding.weight.val(),self.sbd_mask.as_ref().unwrap().val(),hard)
            .reshape([blocks,n,d]);
        if let Some(init) = &self.ohc_init {
            let mut streams = init.forward(hidden,valid.clone());
            for (index,block) in self.blocks.iter().enumerate() {
                let clean = &cache.layers[index];
                if let (Some(residual),Some(norm),Some(wavelet)) = (&block.ohc_wavelet,&block.wavelet_norm,&block.wavelet) {
                    let read = residual.read_selected(streams,self.ohc_backend);
                    let input = self.normalization_backend.rms(norm,read.work.clone());
                    let update = wavelet.forward_sbd_batched(input,clean.wavelet.as_ref().unwrap());
                    streams = residual.write(read,update,valid.clone(),self.ohc_backend);
                }
                if let (Some(residual),Some(norm),Some(lngram)) = (&block.ohc_lngram,&block.lngram_norm,&block.lngram) {
                    let read = residual.read_selected(streams,self.ohc_backend);
                    let input = self.normalization_backend.rms(norm,read.work.clone());
                    let update = clean.lngram.as_ref().unwrap().forward(lngram,input,valid.clone());
                    streams = residual.write(read,update,valid.clone(),self.ohc_backend);
                }
                let residual = block.ohc_mixer.as_ref().unwrap();
                let read = residual.read_selected(streams,self.ohc_backend);
                let input = self.normalization_backend.rms(&block.norm,read.work.clone());
                let update = block.mixer.forward_sbd_batched_profiled(input,&clean.mixer,valid.clone(),profile);
                streams = residual.write(read,update,valid.clone(),self.ohc_backend);
                let residual = block.ohc_mlp.as_ref().unwrap();
                let read = residual.read_selected(streams,self.ohc_backend);
                let input = self.normalization_backend.rms(&block.ff_norm,read.work.clone());
                let update = block.down.forward(self.epilogue_backend.silu(block.up.forward(input)));
                streams = residual.write(read,update,valid.clone(),self.ohc_backend);
            }
            hidden = self.ohc_readout.as_ref().unwrap().readout(streams);
        } else {
            for (index,block) in self.blocks.iter().enumerate() {
                let clean = &cache.layers[index];
                if let (Some(norm),Some(wavelet)) = (&block.wavelet_norm,&block.wavelet) {
                    let input = self.normalization_backend.rms(norm,hidden.clone());
                    hidden = hidden + wavelet.forward_sbd_batched(input,clean.wavelet.as_ref().unwrap());
                }
                if let (Some(norm),Some(lngram)) = (&block.lngram_norm,&block.lngram) {
                    let input = self.normalization_backend.rms(norm,hidden.clone());
                    hidden = hidden + clean.lngram.as_ref().unwrap().forward(lngram,input,valid.clone());
                }
                let input = self.normalization_backend.rms(&block.norm,hidden.clone());
                hidden = hidden + block.mixer.forward_sbd_batched_profiled(input,&clean.mixer,valid.clone(),profile);
                let input = self.normalization_backend.rms(&block.ff_norm,hidden.clone());
                hidden = hidden + block.down.forward(self.epilogue_backend.silu(block.up.forward(input)));
            }
        }
        let features = self.normalization_backend.rms(&self.norm,hidden);
        DenoisingOutput { logits: self.head_forward(features.clone()), features }
    }
}
