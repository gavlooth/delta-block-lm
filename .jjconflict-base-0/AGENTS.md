# Repository policy

Rust + Burn (CubeCL for custom GPU kernels). Local GPU: Intel Arc through Vulkan. Large runs: DGX Spark through CUDA.

## Build speed

These rules exist because the predecessor repository (`diffusion-relation-extractor`) measured:
- cold dev build 132 s and cold release build 232 s;
- 11–25 s rebuilds after one edit;
- a 1.5 GB debug binary;
- a cold `cargo test --no-run` that had not finished after about 20 minutes, because every file under `tests/` linked all of Burn into its own binary.

1. **One integration-test binary per crate.** Put integration tests in `tests/main.rs`, which declares the other test files as modules. Never add another top-level file under `tests/`. Unit tests stay in the library.
2. **No debug info in dev and test builds.** `[profile.dev]` and `[profile.test]` use `debug = false` (dependencies too). When a test fails and needs a debugger, rebuild only then with `CARGO_PROFILE_TEST_DEBUG=true` (or `CARGO_PROFILE_DEV_DEBUG=true`); never commit debug info back on.
3. **Optimized dependencies in dev.** `[profile.dev.package."*"] opt-level = 2`, so CPU tests and small runs do not need release builds. Workspace crates stay at the default dev opt-level.
4. **mold linker** on x86_64 (`clang` with `-fuse-ld=mold`, set in `.cargo/config.toml`; install with `pacman -S mold`). The Spark (aarch64) uses lld until mold is installed there; the predecessor repo needed lld on that machine because GNU ld left LLVM symbols unresolved.
5. **GPU-only Burn features stay off by default.** `fusion`, `autotune` and `cuda` sit behind crate features that only GPU run binaries enable. Tests and CPU checks build without them.
6. **No vendored Burn or CubeCL crates** unless a measured bug requires one. Record the bug, the measurement and the upstream issue next to the patch, and remove the patch when upstream fixes it.
7. **Crate split:** kernels (CubeCL), model, training, and binaries live in separate crates, so a model edit does not rebuild kernels.
8. **Experiments are configured by files, not code.** Hyperparameter and arm changes must not trigger a rebuild.
9. **Persistent CubeCL kernel cache.** Every run binary sets the cache directory, so restarts do not compile kernels again.
10. **Reuse the same backend feature combination.** Related build and test commands should keep their enabled backend features consistent. A GPU-only leaf-crate test can compile another copy of the backend dependency tree when the active experiment also enables CPU features. Prefer a consistent workspace feature set for recurring GPU checks; treat a new feature combination as a cold build with its own deadline, not a normal warm test rebuild.

### Build budgets

Measure after any change to dependencies, features, profiles or crate layout, and record the numbers in the session report:

| Measurement | Budget |
|---|---|
| Rebuild after editing one model file (dev) | ≤ 10 s |
| `cargo test --no-run` after the same edit | ≤ 60 s |

A change that exceeds a budget needs a recorded reason or a fix before it is merged.

### Running long commands

Give every build or benchmark longer than a minute a time limit and run it in the background. Report partial results instead of waiting. A cold measurement that runs past its limit is itself the result: record the limit it exceeded.

## Experiments

- **Start small and measurable, then scale.** A synthetic task (multi-query associative recall) and a tiny language model come before any large run.
- **Every mixer has a plain float64 sequential reference.** Chunked CPU and GPU implementations must match it, including gradients, before they are used in training.
- **One change per controlled comparison.** Arms differ only in the component under test. Report seeds, token counts and the exact config with every result.
- **Choose designs by expected return:** quality gained per unit of compute, memory, latency and maintenance. State which numbers are measured and which are assumed.

## Optimize components before scaling

Every new or materially changed model, training, data or decoding component must pass a small correctness and performance check before a large run depends on it. Include later wavelet, oHC, Lngram, circuit, coordination and remasking implementations when they enter the active model; this rule does not authorize implementing a deferred plan step.

1. **Measure before replacing.** Save a reproducible baseline at both pilot and intended larger shapes. Record actual device, dtype, dimensions, batch, masks, seeds and implementation selection. Compare at least two plausible designs, including reuse of existing primitives.
2. **Prove the component.** Compare outputs, state transitions and gradients against an independent reference. Include padding, resets, short/tail tiles, ties, zero values and noncontiguous inputs where supported. A known incorrect implementation is not the numerical reference for its replacement.
3. **Measure isolated and integrated cost.** Report synchronized warm forward and forward/backward samples, and full-model optimizer-step or decoding cost with the component enabled. Separate cold compilation, autotuning, transfers, checkpoint I/O and warm execution. Do not advertise an isolated-kernel speedup as an equal model speedup.
4. **Track memory and work.** Record temporary allocation sizes, retained state/cache sizes, dispatch counts where available, and sampled or measured peak memory with its limitations. Avoid copying or materializing large tensors merely to express a small sparse operation. Flag quadratic work explicitly.
5. **Optimize what measurements justify.** Profile projections/MLPs, normalization, optimizer, input pipeline and soft top-k feedback as well as custom mixers and attention. Keep existing library operations when they meet the workload's needs. A custom kernel must earn its implementation and maintenance cost.
6. **Set a component budget before scaling it.** Use measured integrated cost to decide acceptable latency, memory and startup/build overhead. Record the decision in the experiment results. No universal speedup target substitutes for the actual training or generation budget.
7. **Preserve honest comparisons.** Keep identical parameters, inputs, precision, masks and work budgets. Record reference defects and failed experiments. Do not loosen tolerances, hide failing cases or change the workload to manufacture a speedup.
8. **Keep execution choices explicit and repeatable.** Optimized configurations name their kernel paths. Do not silently change historical configuration/checkpoint semantics. Archive the measurements and rerun the relevant checks when shapes, precision, hardware, backend versions or implementation change.

## Notes and history

- Session reports live in `.agents/SESSION_REPORT.md`. The file is ignored by git and is never committed.
- Use `jj` for version control. Commit messages carry no AI attribution.
