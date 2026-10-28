# Precision and shape limits

## BF16 decision

Keep f32 for the local optimized model. The saved Intel Arc MTL Vulkan capability report lists F16/F32/F64 arithmetic and dot products, but no BF16 element type; all cooperative-matrix feature maps are empty. This is a report from the current backend/driver, not a statement that every Intel GPU or backend lacks BF16.

Evidence: `artifacts/gemm-efficiency/off-b8/gemm-micro-2330622-1792863485015265698/report.json`, `capabilities.cubecl_properties`; `artifacts/gemm-efficiency/vulkaninfo.txt` also records `shaderFloat16 = true`. The custom recurrence, attention, convolution, normalization, optimizer and loss operators currently require f32.

| Design | Expected gain | Cost and limitation | Decision |
|---|---|---|---|
| Retain f32 | Existing independent numerical gates and measured execution | No reduced-precision storage saving | Use on the local Vulkan target |
| Native BF16 with f32 recurrent state and reductions | Potentially half-sized eligible activation/weight buffers; throughput gain unmeasured | BF16 is absent from this backend's reported types; changing a tensor dtype does not establish an executable training path | Not available for this target; do not silently emulate or call FP16 BF16 |
| FP16 projections with f32 state, master weights and reductions | F16 arithmetic exists; eligible buffers could use two bytes/element | Requires explicit mixed-precision/autodiff/optimizer semantics and new numerical acceptance; conversion overhead and speedup unmeasured | Separate experiment, not an adopted optimization |

No BF16 or FP16 throughput, quality, full-model correctness or memory improvement is claimed. CUDA remains unmeasured without an agreed GPU window. Re-evaluate BF16 when the actual target's runtime capability report includes it.

## Wider state and longer context

The current optimized pilot is K32/V64 with checkpoint chunks1–16. The experiment configuration deliberately rejects other key/value widths. The tensor reference is dimension-generic; that does not make the custom kernels dimension-generic. Increasing context alone does not remove the checked address-span and allocation limits.

For a recurrent checkpoint/replay design, the state and each checkpoint scale with `B*H*K*V`; doubling both widths quadruples those terms. For fixed widths/chunk size, retained checkpoints and sparse HOLA pair storage grow linearly with context. These are source-derived work/storage estimates, not measured wider-state performance.

Do not relabel the completed B1/4/8/16/32 experiment as wider-state or longer-context validation. No wider-state or beyond-pilot-context adoption follows from these results. A future shape extension must generalize every forward/backward specialization, preserve the existing f32 reference gates, and measure the exact intended dimensions rather than only relaxing configuration checks.
