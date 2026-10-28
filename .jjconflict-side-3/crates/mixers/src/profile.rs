//! Optional section observation shared by normal execution and the model profiler.
//! The no-op observer is monomorphized away; production forwards do not synchronize.

pub trait Sections {
    /// CPU diagnostic only; ordinary observers compile out cache-selection observation.
    const CAPTURE_CACHE_MEMBERSHIP: bool = false;
    const CAPTURE_RECURRENT_STATE: bool = false;
    fn recurrent_state(&mut self, _state: &burn::prelude::Tensor<4>) {}
    const CAPTURE_CLEAN_INPUTS: bool = false;
    fn clean_input(&mut self, _layer: usize, _kind: &'static str, _input: &burn::prelude::Tensor<3>) {}
    const CAPTURE_CLEAN_MIXERS: bool = false;
    fn clean_starts(&self) -> Option<&[usize]> { None }
    fn clean_mixer(&mut self, _clean: crate::sbd::CleanMixer) {}

    /// Called once per chunk with the slots kept (indices into cache ∪ chunk) and their
    /// token positions, for the current block.
    fn cache_membership(
        &mut self,
        _chunk_start: usize,
        _used_by_later_chunk: bool,
        _kept_slots: &burn::prelude::Tensor<3, burn::prelude::Int>,
        _positions: &burn::prelude::Tensor<3>,
    ) {
    }

    /// Kept slots to reuse for every chunk of the current block instead of ranking.
    fn replayed_cache_selection(
        &self,
    ) -> Option<Vec<burn::prelude::Tensor<3, burn::prelude::Int>>> {
        None
    }
    fn prefix_visibility(&mut self, _visible: Option<&burn::prelude::Tensor<4, burn::prelude::Bool>>) {}
    fn replayed_prefix_visibility(&self) -> Option<Vec<Option<burn::prelude::Tensor<4, burn::prelude::Bool>>>> {
        None
    }

    fn block(&mut self, _index: Option<usize>) {}

    fn section<T>(&mut self, name: &'static str, run: impl FnOnce() -> T) -> T;
}

pub struct Unprofiled;

impl Sections for Unprofiled {
    #[inline(always)]
    fn section<T>(&mut self, _name: &'static str, run: impl FnOnce() -> T) -> T {
        run()
    }
}
