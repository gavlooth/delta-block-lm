//! Optional section observation shared by normal execution and the model profiler.
//! The no-op observer is monomorphized away; production forwards do not synchronize.

pub trait Sections {
    /// CPU diagnostic only; ordinary observers compile out membership collection.
    const CAPTURE_CACHE_MEMBERSHIP: bool = false;

    fn cache_membership(
        &mut self,
        _chunk_start: usize,
        _used_by_later_chunk: bool,
        _positions: &burn::prelude::Tensor<3>,
    ) {
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
