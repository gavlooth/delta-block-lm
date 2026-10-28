use crate::config::Config;
use anyhow::Result;
use burn::prelude::*;
use serde_json::{Value, json};

#[cfg(feature = "vulkan")]
static ADAPTER: std::sync::OnceLock<String> = std::sync::OnceLock::new();

pub fn device(c: &Config) -> Result<Device> {
    if matches!(c.backend.as_str(), "vulkan" | "cuda") {
        anyhow::ensure!(
            c.fusion == cfg!(feature = "fusion"),
            "config fusion={} does not match this binary's fusion feature={}; rebuild with --features vulkan{} (or the corresponding cuda features); historical configs must not silently enable fusion",
            c.fusion,
            cfg!(feature = "fusion"),
            if c.fusion { ",fusion" } else { "" },
        );
        anyhow::ensure!(
            c.autotune == cfg!(feature = "autotune"),
            "config autotune={} does not match this binary's autotune feature={}; historical configs must not silently enable autotuning",
            c.autotune,
            cfg!(feature = "autotune"),
        );
    }
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
        let mut config = CubeClRuntimeConfig::default();
        config.compilation.cache = true;
        config.autotune.disable_cache = false;
        if c.autotune {
            std::fs::create_dir_all(&c.artifact_dir)?;
            config.autotune.decisions.file = Some(c.artifact_dir.join("autotune-decisions.jsonl"));
        }
        config.environment.name = "delta-block-local".into();
        config.environment.path =
            cubecl::config::cache::CacheConfig::Directory(c.cache_dir.clone());
        std::fs::create_dir_all(&c.cache_dir)?;
        CubeClRuntimeConfig::set(config);
    }
    match c.backend.as_str() {
        "cpu" => {
            #[cfg(feature = "cpu")]
            {
                Ok(Device::ndarray())
            }
            #[cfg(not(feature = "cpu"))]
            {
                anyhow::bail!("rebuild with feature cpu")
            }
        }
        "vulkan" => {
            #[cfg(feature = "vulkan")]
            {
                let gpu =
                    cubecl::device::WgpuDevice::default().on(cubecl::device::WgpuBackend::Vulkan);
                let setup = cubecl::wgpu::try_init_setup::<cubecl::wgpu::AutoGraphicsApi>(
                    &gpu,
                    Default::default(),
                )
                .map_err(|error| anyhow::anyhow!("initializing Vulkan: {error}"))?;
                let _ = ADAPTER.set(format!("{:?}", setup.adapter.get_info()));
                let device = Device::vulkan(Default::default());
                let info = format!("{:?}", device.identity());
                anyhow::ensure!(
                    info.to_ascii_lowercase().contains("intel"),
                    "expected local Intel Vulkan adapter, got {info}"
                );
                Ok(device)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                anyhow::bail!("rebuild with --features vulkan")
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::cuda(0))
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA is opt-in and is not used for the local experiment")
            }
        }
        value => anyhow::bail!("unknown backend {value}"),
    }
}
pub fn identity(device: &Device) -> Value {
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    {
        #[cfg(feature = "vulkan")]
        let adapter = ADAPTER.get().map(String::as_str);
        #[cfg(not(feature = "vulkan"))]
        let adapter: Option<&str> = None;
        json!({"device":format!("{device:?}"),"identity":format!("{:?}",device.identity()),
            "autotune":cfg!(feature = "autotune"),"fusion":cfg!(feature = "fusion"),"adapter":adapter})
    }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        json!({"device":format!("{device:?}"),"identity":"CPU ndarray"})
    }
}
pub fn sync(device: &Device) -> Result<()> {
    device
        .sync()
        .map_err(|e| anyhow::anyhow!("device sync: {e:?}"))
}
pub fn sampled_reserved(device: &Device) -> Option<u64> {
    device.memory_pool_usage().map(|u| u.bytes_reserved)
}
pub fn write_json(path: &std::path::Path, value: &impl serde::Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let pending = path.with_extension("json.pending");
    std::fs::write(&pending, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(pending, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_fusion_mismatch_fails_before_device_creation() {
        let mut config = Config::default();
        config.backend = "vulkan".into();
        config.fusion = !cfg!(feature = "fusion");
        let error = device(&config).unwrap_err();
        assert!(error.to_string().contains("does not match this binary"));
    }

    #[test]
    fn gpu_autotune_mismatch_fails_before_device_creation() {
        let mut config = Config::default();
        config.backend = "vulkan".into();
        config.fusion = cfg!(feature = "fusion");
        config.autotune = !cfg!(feature = "autotune");
        let error = device(&config).unwrap_err();
        assert!(error.to_string().contains("autotune="));
    }
}
