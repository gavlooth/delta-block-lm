//! Opt-in synchronized component timings. Disabled in production; synchronization changes fusion.
use burn::prelude::*;
use std::sync::LazyLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct ComponentProfile {
    name: &'static str,
    device: Device,
    timing: bool,
    started: Instant,
    unix_start: f64,
}

/// Set `DRE_COMPONENT_PROFILE=1` (timings) or `DRE_MEMORY_PHASES=1` (memory phases at component boundaries)
/// before startup. No device cloning/synchronization when both are disabled.
pub fn component(name: &'static str, device: impl FnOnce() -> Device) -> Option<ComponentProfile> {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("DRE_COMPONENT_PROFILE").is_ok_and(|value| value == "1"));
    if !*ENABLED && !memory_phases_enabled() {
        return None;
    }
    let device = device();
    if memory_phases_enabled() {
        memory_phase("component_begin", &device, || format!("{{\"component\":\"{name}\"}}"));
    } else {
        device.sync().expect("component profile start synchronization");
    }
    Some(ComponentProfile {
        name,
        device,
        timing: *ENABLED,
        started: Instant::now(),
        unix_start: SystemTime::now().duration_since(UNIX_EPOCH).expect("wall clock").as_secs_f64(),
    })
}

impl Drop for ComponentProfile {
    fn drop(&mut self) {
        if memory_phases_enabled() {
            let name = self.name;
            memory_phase(name, &self.device, || format!("{{\"component\":\"{name}\"}}"));
        } else {
            self.device.sync().expect("component profile end synchronization");
        }
        if self.timing {
            eprintln!("{{\"kind\":\"component_profile\",\"name\":\"{}\",\"unix_start\":{},\"seconds\":{}}}",
                self.name, self.unix_start, self.started.elapsed().as_secs_f64());
        }
    }
}

/// `DRE_MEMORY_PHASES=1` enables per-phase CubeCL memory telemetry. Disabled in production; synchronization changes fusion.
pub fn memory_phases_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("DRE_MEMORY_PHASES").is_ok_and(|value| value == "1"));
    *ENABLED
}

/// Drains queued GPU work, then reports CUDA backing bytes (retained pool pages), live tensor bytes, host
/// `MemAvailable` and the backing high-water mark since the previous phase, and restarts that mark.
/// `details` must render a JSON value; it is evaluated only when enabled. With `DRE_MEMORY_PHASES_CLEANUP=1`
/// free pool pages are also released after the report, so the next phase's mark measures its own working set.
/// The device barrier does not surface failed lazy allocations; refusals appear on stderr between phase records
/// with `DRE_CUDA_BUDGET_DIAGNOSTICS=1`.
pub fn memory_phase(name: &str, device: &Device, details: impl FnOnce() -> String) {
    if !memory_phases_enabled() {
        return;
    }
    static INDEX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static PREVIOUS: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
    let index = INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sync_error = device.sync().err().map(|error| format!("{error:?}"));
    let now = Instant::now();
    let since_previous = PREVIOUS.lock().unwrap().replace(now).map(|previous| (now - previous).as_secs_f64());
    let unix_time = SystemTime::now().duration_since(UNIX_EPOCH).expect("wall clock").as_secs_f64();
    let available = std::fs::read_to_string("/proc/meminfo").ok().and_then(|text| text.lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .and_then(|line| line.split_ascii_whitespace().nth(1)?.parse::<u64>().ok())).map(|kib| kib * 1024);
    #[cfg(not(feature = "cuda"))]
    let memory = String::new();
    #[cfg(feature = "cuda")]
    let memory = {
        let usage = burn::cubecl::cuda::cuda_memory_usage();
        let client = burn::cubecl::Device::from(burn::cubecl::cuda::CudaDevice::default()).client();
        let live = client.memory_report(burn::cubecl::MemoryScope::Device).usage();
        let mut memory = format!(",\"committed_bytes\":{},\"phase_peak_committed_bytes\":{},\"reserved_bytes\":{},\"phase_peak_reserved_bytes\":{},\"live_bytes\":{},\"live_padding_bytes\":{},\"pool_reserved_bytes\":{}",
            usage.committed_bytes, usage.peak_committed_bytes, usage.reserved_bytes, usage.peak_reserved_bytes,
            live.bytes_in_use, live.bytes_padding, live.bytes_reserved);
        if std::env::var("DRE_MEMORY_PHASES_CLEANUP").is_ok_and(|value| value == "1") {
            let cleanup = client.memory_cleanup().err().map(|error| format!("{error:?}"));
            memory.push_str(&format!(",\"committed_after_cleanup_bytes\":{},\"cleanup_error\":{}",
                burn::cubecl::cuda::cuda_memory_usage().committed_bytes, json_string(cleanup.as_deref())));
        }
        burn::cubecl::cuda::reset_cuda_memory_peaks();
        memory
    };
    eprintln!("{{\"kind\":\"memory_phase\",\"index\":{index},\"name\":{},\"unix_time\":{unix_time},\"seconds_since_previous\":{},\"sync_error\":{},\"mem_available_bytes\":{}{memory},\"details\":{}}}",
        json_string(Some(name)), since_previous.map_or("null".into(), |seconds| seconds.to_string()),
        json_string(sync_error.as_deref()), available.map_or("null".into(), |bytes| bytes.to_string()), details());
}

fn json_string(value: Option<&str>) -> String {
    let Some(value) = value else { return "null".into() };
    let mut text = String::with_capacity(value.len() + 2);
    text.push('"');
    for character in value.chars() {
        match character {
            '"' => text.push_str("\\\""),
            '\\' => text.push_str("\\\\"),
            '\n' => text.push_str("\\n"),
            character if character.is_control() => text.push_str(&format!("\\u{:04x}", character as u32)),
            character => text.push(character),
        }
    }
    text.push('"');
    text
}

type ProfileCubeTensor = burn::backend::wgpu::CubeTensor;
type ProfileCubeBackend = burn::backend::wgpu::CubeBackend;
type ProfileFusedBackend = burn::backend::fusion::Fusion<ProfileCubeBackend>;

/// Opt-in EXP-004 timings materialize actual input/output buffers before waiting on their completion.
/// Only the profiling path resolves fusion work; a disabled call does not clone tensor/device handles.
pub struct Exp004Region {
    name: &'static str,
    id: u64,
    shape: Vec<usize>,
    unix_start: f64,
    overall_started: Instant,
    started: Instant,
    input_resolve_seconds: f64,
    input_gpu_wait_seconds: f64,
    tokens: usize,
    queries: usize,
    edges: usize,
}

fn resolve_buffer<const D: usize>(tensor: &Tensor<D>) -> ProfileCubeTensor {
    let tensor = tensor.clone().inner().try_into_primitive::<ProfileFusedBackend>()
        .expect("DRE_EXP004_PROFILE requires the fused Cube backend");
    tensor.client.clone().resolve_tensor_float::<ProfileCubeBackend>(tensor)
}

fn wait_buffers(buffers: &[ProfileCubeTensor]) {
    if let Some(first) = buffers.first() {
        burn::cubecl::future::block_on(first.client.sync_buffers(buffers.iter().map(|tensor| &tensor.handle)))
            .expect("EXP-004 profile buffer synchronization");
    }
}

pub fn exp004_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("DRE_EXP004_PROFILE").is_ok_and(|value| value == "1"));
    *ENABLED
}

pub fn exp004<const D: usize>(name: &'static str, input: &Tensor<D>) -> Option<Exp004Region> {
    if !exp004_enabled() { return None; }
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let overall_started = Instant::now();
    let unix_start = SystemTime::now().duration_since(UNIX_EPOCH).expect("wall clock").as_secs_f64();
    let shape = input.dims().to_vec();
    eprintln!("{{\"kind\":\"exp004_profile_begin\",\"scope\":\"core\",\"id\":{id},\"name\":\"{name}\",\"unix_start\":{unix_start},\"shape\":{shape:?}}}");
    let resolve_started = Instant::now();
    let buffer = resolve_buffer(input);
    let input_resolve_seconds = resolve_started.elapsed().as_secs_f64();
    let wait_started = Instant::now();
    wait_buffers(std::slice::from_ref(&buffer));
    let input_gpu_wait_seconds = wait_started.elapsed().as_secs_f64();
    Some(Exp004Region {name, id, shape, unix_start, overall_started, started: Instant::now(),
        input_resolve_seconds, input_gpu_wait_seconds, tokens: 0, queries: 0, edges: 0})
}

impl Exp004Region {
    pub fn dimensions(mut self, tokens: usize, queries: usize, edges: usize) -> Self {
        self.tokens = tokens;
        self.queries = queries;
        self.edges = edges;
        self
    }

    pub fn finish<const D: usize>(self, output: &Tensor<D>) {
        self.finish_many(std::iter::once(output));
    }

    pub fn finish_many<'a, const D: usize>(self, outputs: impl IntoIterator<Item = &'a Tensor<D>>) {
        let host_seconds = self.started.elapsed().as_secs_f64();
        let resolve_started = Instant::now();
        let buffers: Vec<_> = outputs.into_iter().map(resolve_buffer).collect();
        let resolve_seconds = resolve_started.elapsed().as_secs_f64();
        let wait_started = Instant::now();
        wait_buffers(&buffers);
        let gpu_wait_seconds = wait_started.elapsed().as_secs_f64();
        let total_seconds = self.overall_started.elapsed().as_secs_f64();
        eprintln!(
            "{{\"kind\":\"exp004_profile\",\"scope\":\"core\",\"id\":{},\"name\":\"{}\",\"unix_start\":{},\"seconds\":{},\"shape\":{:?},\"tokens\":{},\"queries\":{},\"edges\":{},\"host_seconds\":{},\"resolve_seconds\":{},\"gpu_wait_seconds\":{},\"total_seconds\":{},\"input_resolve_seconds\":{},\"input_gpu_wait_seconds\":{},\"resolved_buffers\":{}}}",
            self.id, self.name, self.unix_start, total_seconds, self.shape, self.tokens, self.queries, self.edges,
            host_seconds, resolve_seconds, gpu_wait_seconds, total_seconds, self.input_resolve_seconds,
            self.input_gpu_wait_seconds, buffers.len(),
        );
    }
}
