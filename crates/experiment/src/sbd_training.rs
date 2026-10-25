use crate::{config::CocommitExecution, model::{DenoisingOutput, LanguageModel, NoisyRepresentation}, soft::SoftState};
pub(crate) use crate::model::NoisyPrefix;
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
#[derive(Clone)]
struct LayerCache {
    wavelet: Option<WaveletCache>,
    lngram: Option<Tensor<3>>,
    mixer: MixerCache,
}
pub struct TrainingCache { layers: Vec<LayerCache>, starts: Vec<usize> }
#[derive(Clone)]
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
#[derive(Clone)]
struct BatchedLayerCache {
    wavelet: Option<BatchedWaveletCache>,
    lngram: Option<BatchedLngramCache>,
    mixer: BatchedMixerCache,
}
pub struct BatchedTrainingCache { layers: Vec<BatchedLayerCache>, blocks: usize }
impl BatchedLngramCache {
    fn duplicate_rows(&self) -> Self {
        Self { history: self.history.as_ref().map(|(values,valid)| (
            Tensor::cat(vec![values.clone(),values.clone()],0),
            Tensor::cat(vec![valid.clone(),valid.clone()],0),
        )) }
    }
}
impl BatchedTrainingCache {
    fn duplicate_rows_from(&self, first: usize) -> Self {
        Self {
            layers: self.layers.iter().enumerate().map(|(index,layer)| if index < first {
                layer.clone()
            } else {
                BatchedLayerCache {
                    wavelet: layer.wavelet.as_ref().map(BatchedWaveletCache::duplicate_rows),
                    lngram: layer.lngram.as_ref().map(BatchedLngramCache::duplicate_rows),
                    mixer: layer.mixer.duplicate_rows(),
                }
            }).collect(),
            blocks: self.blocks*2,
        }
    }
}

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
        if self.cocommit_config.is_some() {
            let representation = if let Some(init) = &self.ohc_init {
                NoisyRepresentation::Streams(init.forward(hidden,valid.clone()))
            } else { NoisyRepresentation::Hidden(hidden) };
            let representation = self.cached_block_segment(cache,block_index,0,self.blocks.len(),representation,valid,profile);
            return self.output_from_representation(representation);
        }
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
        if self.cocommit_config.is_some() {
            let representation = if let Some(init) = &self.ohc_init {
                NoisyRepresentation::Streams(init.forward(hidden,valid.clone()))
            } else { NoisyRepresentation::Hidden(hidden) };
            let representation = self.cached_blocks_segment(cache,0,self.blocks.len(),representation,valid,profile);
            return self.output_from_representation(representation);
        }
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
    fn cached_block_segment<P: Sections>(
        &self, cache: &TrainingCache, cache_index: usize, first: usize, last: usize,
        mut representation: NoisyRepresentation, valid: Tensor<2>, profile: &mut P,
    ) -> NoisyRepresentation {
        let device = valid.device();
        let [rows, n] = valid.dims();
        let d = self.embedding.weight.val().dims()[1];
        let lngram_forward = |lngram: &delta_mixers::lngram::Lngram, input: Tensor<3>, clean: &Tensor<3>, start: usize| {
            let history = start.min(lngram.orders.iter().copied().max().unwrap()-1
                + lngram.smoothing.as_ref().map_or(0,|s|2*s.dilation));
            let combined = Tensor::cat(vec![clean.clone().slice([0..rows,start-history..start,0..d]),input],1);
            let layout = delta_mixers::lngram::LngramLayout::new(
                Tensor::ones([rows,history+n],&device),Tensor::zeros([rows,history+n],&device));
            lngram.forward(combined,&layout).slice([0..rows,history..history+n,0..d])
        };
        for index in first..last {
            let block = &self.blocks[index];
            let clean = &cache.layers[index];
            representation = match representation {
                NoisyRepresentation::Streams(mut streams) => {
                    if let (Some(stage),Some(norm),Some(wavelet)) = (&block.ohc_wavelet,&block.wavelet_norm,&block.wavelet) {
                        let read = stage.read_selected(streams,self.ohc_backend);
                        let input = self.normalization_backend.rms(norm,read.work.clone());
                        streams = stage.write(read,wavelet.forward_sbd_cached(input,clean.wavelet.as_ref().unwrap(),cache.starts[cache_index]),
                            valid.clone(),self.ohc_backend);
                    }
                    if let (Some(stage),Some(norm),Some(lngram)) = (&block.ohc_lngram,&block.lngram_norm,&block.lngram) {
                        let read = stage.read_selected(streams,self.ohc_backend);
                        let input = self.normalization_backend.rms(norm,read.work.clone());
                        let update = lngram_forward(lngram,input,clean.lngram.as_ref().unwrap(),cache.starts[cache_index]);
                        streams = stage.write(read,update,valid.clone(),self.ohc_backend);
                    }
                    let stage = block.ohc_mixer.as_ref().unwrap();
                    let read = stage.read_selected(streams,self.ohc_backend);
                    let input = self.normalization_backend.rms(&block.norm,read.work.clone());
                    let update = block.mixer.forward_sbd_cached_profiled(input,&clean.mixer,cache_index,valid.clone(),profile);
                    streams = stage.write(read,update,valid.clone(),self.ohc_backend);
                    let stage = block.ohc_mlp.as_ref().unwrap();
                    let read = stage.read_selected(streams,self.ohc_backend);
                    let input = self.normalization_backend.rms(&block.ff_norm,read.work.clone());
                    let update = block.down.forward(self.epilogue_backend.silu(block.up.forward(input)));
                    NoisyRepresentation::Streams(stage.write(read,update,valid.clone(),self.ohc_backend))
                }
                NoisyRepresentation::Hidden(mut hidden) => {
                    if let (Some(norm),Some(wavelet)) = (&block.wavelet_norm,&block.wavelet) {
                        let input = self.normalization_backend.rms(norm,hidden.clone());
                        hidden = hidden + wavelet.forward_sbd_cached(input,clean.wavelet.as_ref().unwrap(),cache.starts[cache_index]);
                    }
                    if let (Some(norm),Some(lngram)) = (&block.lngram_norm,&block.lngram) {
                        let input = self.normalization_backend.rms(norm,hidden.clone());
                        hidden = hidden + lngram_forward(lngram,input,clean.lngram.as_ref().unwrap(),cache.starts[cache_index]);
                    }
                    let input = self.normalization_backend.rms(&block.norm,hidden.clone());
                    hidden = hidden + block.mixer.forward_sbd_cached_profiled(input,&clean.mixer,cache_index,valid.clone(),profile);
                    let input = self.normalization_backend.rms(&block.ff_norm,hidden.clone());
                    hidden = hidden + block.down.forward(self.epilogue_backend.silu(block.up.forward(input)));
                    NoisyRepresentation::Hidden(hidden)
                }
            };
        }
        representation
    }

    pub fn cached_block_prefix_profiled<P: Sections>(
        &self, cache: &TrainingCache, block_index: usize, soft: &SoftState, hard: bool, profile: &mut P,
    ) -> NoisyPrefix {
        let n = soft.observed.len();
        let d = self.embedding.weight.val().dims()[1];
        let device = self.embedding.weight.val().device();
        let valid = Tensor::from_data(TensorData::new(
            soft.active.iter().map(|&x| if x {1.0f32} else {0.0}).collect::<Vec<_>>(),[1,n]),&device);
        let unresolved = Tensor::from_data(TensorData::new(
            soft.observed.iter().zip(&soft.active).map(|(observed,active)| if observed.is_none() && *active {1.0f32} else {0.0}).collect::<Vec<_>>(),[1,n]),&device);
        let hidden = soft.embeddings(self.embedding.weight.val(),self.sbd_mask.as_ref().unwrap().val(),hard).reshape([1,n,d]);
        let representation = if let Some(init) = &self.ohc_init {
            NoisyRepresentation::Streams(init.forward(hidden,valid.clone()))
        } else { NoisyRepresentation::Hidden(hidden) };
        let first_tail = self.blocks.len()-self.cocommit_config.as_ref().unwrap().tail_layers;
        let representation = self.cached_block_segment(cache,block_index,0,first_tail,representation,valid.clone(),profile);
        NoisyPrefix { representation, valid, unresolved }
    }

    pub fn cached_block_tail_profiled<P: Sections>(
        &self, cache: &TrainingCache, block_index: usize, prefix: NoisyPrefix,
        marks: Option<Tensor<2>>, profile: &mut P,
    ) -> DenoisingOutput {
        let first_tail = self.blocks.len()-self.cocommit_config.as_ref().unwrap().tail_layers;
        let marked = match (prefix.representation, marks) {
            (NoisyRepresentation::Hidden(hidden),Some(marks)) => {
                let marker = self.cocommit_marker.as_ref().unwrap().val().reshape([1,1,-1]);
                NoisyRepresentation::Hidden(hidden + marker * (marks*prefix.unresolved.clone()).unsqueeze_dim::<3>(2))
            }
            (NoisyRepresentation::Streams(streams),Some(marks)) => {
                let marker = self.cocommit_marker.as_ref().unwrap().val().reshape([1,1,4,-1]);
                NoisyRepresentation::Streams(streams + marker * (marks*prefix.unresolved.clone()).reshape([1,prefix.unresolved.dims()[1],1,1]))
            }
            (representation,None) => representation,
        };
        let representation = self.cached_block_segment(cache,block_index,first_tail,self.blocks.len(),marked,prefix.valid,profile);
        let hidden = match representation {
            NoisyRepresentation::Hidden(hidden) => hidden,
            NoisyRepresentation::Streams(streams) => self.ohc_readout.as_ref().unwrap().readout(streams),
        };
        let features = self.normalization_backend.rms(&self.norm,hidden);
        DenoisingOutput { logits: self.head_forward(features.clone()), features }
    }

    pub fn cached_blocks_prefix_profiled<P: Sections>(
        &self, cache: &BatchedTrainingCache, soft: &SoftState, hard: bool, profile: &mut P,
    ) -> NoisyPrefix {
        let blocks = cache.blocks;
        assert!(soft.observed.len().is_multiple_of(blocks));
        let n = soft.observed.len()/blocks;
        let d = self.embedding.weight.val().dims()[1];
        let device = self.embedding.weight.val().device();
        let valid = Tensor::from_data(TensorData::new(
            soft.active.iter().map(|&x| if x {1.0f32} else {0.0}).collect::<Vec<_>>(),[blocks,n]),&device);
        let unresolved = Tensor::from_data(TensorData::new(
            soft.observed.iter().zip(&soft.active).map(|(observed,active)| if observed.is_none() && *active {1.0f32} else {0.0}).collect::<Vec<_>>(),[blocks,n]),&device);
        let hidden = soft.embeddings(self.embedding.weight.val(),self.sbd_mask.as_ref().unwrap().val(),hard)
            .reshape([blocks,n,d]);
        let representation = if let Some(init) = &self.ohc_init {
            NoisyRepresentation::Streams(init.forward(hidden,valid.clone()))
        } else { NoisyRepresentation::Hidden(hidden) };
        let first_tail = self.blocks.len()-self.cocommit_config.as_ref().unwrap().tail_layers;
        let representation = self.cached_blocks_segment(cache,0,first_tail,representation,valid.clone(),profile);
        NoisyPrefix { representation, valid, unresolved }
    }

    fn cached_blocks_segment<P: Sections>(
        &self, cache: &BatchedTrainingCache, first: usize, last: usize,
        mut representation: NoisyRepresentation, valid: Tensor<2>, profile: &mut P,
    ) -> NoisyRepresentation {
        for index in first..last {
            let block=&self.blocks[index]; let clean=&cache.layers[index];
            representation = match representation {
                NoisyRepresentation::Streams(mut streams) => {
                    if let (Some(stage),Some(norm),Some(wavelet))=(&block.ohc_wavelet,&block.wavelet_norm,&block.wavelet) {
                        let read=stage.read_selected(streams,self.ohc_backend);
                        let update=wavelet.forward_sbd_batched(self.normalization_backend.rms(norm,read.work.clone()),clean.wavelet.as_ref().unwrap());
                        streams=stage.write(read,update,valid.clone(),self.ohc_backend);
                    }
                    if let (Some(stage),Some(norm),Some(lngram))=(&block.ohc_lngram,&block.lngram_norm,&block.lngram) {
                        let read=stage.read_selected(streams,self.ohc_backend);
                        let update=clean.lngram.as_ref().unwrap().forward(lngram,self.normalization_backend.rms(norm,read.work.clone()),valid.clone());
                        streams=stage.write(read,update,valid.clone(),self.ohc_backend);
                    }
                    let stage=block.ohc_mixer.as_ref().unwrap(); let read=stage.read_selected(streams,self.ohc_backend);
                    let update=block.mixer.forward_sbd_batched_profiled(self.normalization_backend.rms(&block.norm,read.work.clone()),&clean.mixer,valid.clone(),profile);
                    streams=stage.write(read,update,valid.clone(),self.ohc_backend);
                    let stage=block.ohc_mlp.as_ref().unwrap(); let read=stage.read_selected(streams,self.ohc_backend);
                    let update=block.down.forward(self.epilogue_backend.silu(block.up.forward(self.normalization_backend.rms(&block.ff_norm,read.work.clone()))));
                    NoisyRepresentation::Streams(stage.write(read,update,valid.clone(),self.ohc_backend))
                }
                NoisyRepresentation::Hidden(mut hidden) => {
                    if let (Some(norm),Some(wavelet))=(&block.wavelet_norm,&block.wavelet) {
                        let update=wavelet.forward_sbd_batched(self.normalization_backend.rms(norm,hidden.clone()),clean.wavelet.as_ref().unwrap());
                        hidden=hidden+update;
                    }
                    if let (Some(norm),Some(lngram))=(&block.lngram_norm,&block.lngram) {
                        let update=clean.lngram.as_ref().unwrap().forward(lngram,self.normalization_backend.rms(norm,hidden.clone()),valid.clone());
                        hidden=hidden+update;
                    }
                    let update=block.mixer.forward_sbd_batched_profiled(self.normalization_backend.rms(&block.norm,hidden.clone()),&clean.mixer,valid.clone(),profile);
                    hidden=hidden+update;
                    let update=block.down.forward(self.epilogue_backend.silu(block.up.forward(self.normalization_backend.rms(&block.ff_norm,hidden.clone()))));
                    hidden=hidden+update;
                    NoisyRepresentation::Hidden(hidden)
                }
            };
        }
        representation
    }

    fn output_from_representation(&self, representation: NoisyRepresentation) -> DenoisingOutput {
        let hidden=match representation { NoisyRepresentation::Hidden(x)=>x,
            NoisyRepresentation::Streams(x)=>self.ohc_readout.as_ref().unwrap().readout(x) };
        let features=self.normalization_backend.rms(&self.norm,hidden);
        DenoisingOutput {logits:self.head_forward(features.clone()),features}
    }

    pub fn cached_blocks_tail_profiled<P: Sections>(
        &self, cache: &BatchedTrainingCache, prefix: NoisyPrefix, marks: Option<Tensor<2>>, profile: &mut P,
    ) -> DenoisingOutput {
        let first_tail=self.blocks.len()-self.cocommit_config.as_ref().unwrap().tail_layers;
        let representation=match (prefix.representation,marks) {
            (NoisyRepresentation::Hidden(x),Some(marks)) => {
                let marker=self.cocommit_marker.as_ref().unwrap().val().reshape([1,1,-1]);
                NoisyRepresentation::Hidden(x+marker*(marks*prefix.unresolved.clone()).unsqueeze_dim::<3>(2))
            }
            (NoisyRepresentation::Streams(x),Some(marks)) => {
                let marker=self.cocommit_marker.as_ref().unwrap().val().reshape([1,1,4,-1]);
                NoisyRepresentation::Streams(x+marker*(marks*prefix.unresolved.clone()).reshape([prefix.unresolved.dims()[0],prefix.unresolved.dims()[1],1,1]))
            }
            (x,None)=>x,
        };
        self.output_from_representation(self.cached_blocks_segment(cache,first_tail,self.blocks.len(),representation,prefix.valid,profile))
    }

    pub fn cached_blocks_pair_profiled<P: Sections>(
        &self, cache: &BatchedTrainingCache, prefix: NoisyPrefix, marks: Tensor<2>, profile: &mut P,
    ) -> (DenoisingOutput,DenoisingOutput) {
        let first_tail=self.blocks.len()-self.cocommit_config.as_ref().unwrap().tail_layers;
        let retrieval_tail=self.blocks[first_tail..].iter().any(|block| block.mixer.has_prefix_retrieval());
        let doubled=self.cocommit_config.as_ref().unwrap().execution==CocommitExecution::Doubled && !retrieval_tail;
        if doubled {
            let b=prefix.valid.dims()[0];
            let marked=match (prefix.representation.clone(),marks) {
                (NoisyRepresentation::Hidden(x),marks) => {
                    let marker=self.cocommit_marker.as_ref().unwrap().val().reshape([1,1,-1]);
                    NoisyRepresentation::Hidden(x+marker*(marks*prefix.unresolved.clone()).unsqueeze_dim::<3>(2))
                }
                (NoisyRepresentation::Streams(x),marks) => {
                    let marker=self.cocommit_marker.as_ref().unwrap().val().reshape([1,1,4,-1]);
                    NoisyRepresentation::Streams(x+marker*(marks*prefix.unresolved.clone()).reshape([b,prefix.unresolved.dims()[1],1,1]))
                }
            };
            let representation=match (prefix.representation,marked) {
                (NoisyRepresentation::Hidden(a),NoisyRepresentation::Hidden(b)) =>
                    NoisyRepresentation::Hidden(Tensor::cat(vec![a,b],0)),
                (NoisyRepresentation::Streams(a),NoisyRepresentation::Streams(b)) =>
                    NoisyRepresentation::Streams(Tensor::cat(vec![a,b],0)),
                _=>unreachable!(),
            };
            let valid=Tensor::cat(vec![prefix.valid.clone(),prefix.valid],0);
            let paired=self.cached_blocks_segment(&cache.duplicate_rows_from(first_tail),first_tail,self.blocks.len(),representation,valid,profile);
            let paired=self.output_from_representation(paired);
            let [_,time,vocab]=paired.logits.dims();
            let width=paired.features.dims()[2];
            return (
                DenoisingOutput {logits:paired.logits.clone().slice([0..b,0..time,0..vocab]),
                    features:paired.features.clone().slice([0..b,0..time,0..width])},
                DenoisingOutput {logits:paired.logits.slice([b..2*b,0..time,0..vocab]),
                    features:paired.features.slice([b..2*b,0..time,0..width])},
            );
        }
        let a=self.cached_blocks_tail_profiled(cache,NoisyPrefix {
            representation:prefix.representation.clone(),valid:prefix.valid.clone(),unresolved:prefix.unresolved.clone()
        },None,profile);
        let b=self.cached_blocks_tail_profiled(cache,prefix,Some(marks),profile);
        (a,b)
    }
}
