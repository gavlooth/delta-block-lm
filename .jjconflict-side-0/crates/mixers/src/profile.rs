//! Optional section observation shared by normal execution and the model profiler.
//! The no-op observer is monomorphized away; production forwards do not synchronize.

pub trait Sections {
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
