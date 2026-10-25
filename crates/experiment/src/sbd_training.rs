//! Training snapshots from the ordinary unsegmented clean forward.
use crate::{model::LanguageModel, soft::SoftState};
use burn::prelude::*;
use delta_mixers::{profile::Sections, sbd::{MixerCache,WaveletCache}};

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
    pub fn cached_block(&self, cache: &TrainingCache, block_index: usize, soft: &SoftState, hard: bool) -> Tensor<3> {
        let n = soft.observed.len();
        let start = cache.starts[block_index];
        let d = self.embedding.weight.val().dims()[1];
        let device = self.embedding.weight.val().device();
        let valid = Tensor::ones([1,n],&device);
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
                let update = block.mixer.forward_sbd_cached(input,&clean.mixer,block_index);
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
                hidden = hidden + block.mixer.forward_sbd_cached(input,&clean.mixer,block_index);
                let input = self.normalization_backend.rms(&block.ff_norm,hidden.clone());
                hidden = hidden + block.down.forward(self.epilogue_backend.silu(block.up.forward(input)));
            }
        }
        self.head_forward(self.normalization_backend.rms(&self.norm,hidden))
    }
}
