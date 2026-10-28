import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";

const binary = resolve(Bun.argv[2] ?? "target/debug/delta-experiment");
const batch = Number(Bun.argv[3] ?? "1");
const tokens = Number(Bun.argv[4] ?? "128");
const warmups = Number(Bun.argv[5] ?? "2");
const trials = Number(Bun.argv[6] ?? "5");
if (![1, 8].includes(batch) || ![128, 512].includes(tokens)) {
  throw new Error("usage: bun scripts/benchmark-epilogues.ts BINARY BATCH(1|8) TOKENS(128|512) [WARMUPS=2] [TRIALS=5]");
}
if (!Number.isInteger(warmups) || warmups < 1 || warmups > 10 ||
    !Number.isInteger(trials) || trials < 3 || trials > 31) {
  throw new Error("warmups must be 1..10; trials must be 3..31");
}
if (!await Bun.file(binary).exists()) throw new Error("Parent must build the common-feature binary first");
const template = await Bun.file("configs/lm512-projections.json").json();
const directory = resolve(`artifacts/epilogues-b${batch}-t${tokens}-${Date.now()}-${process.pid}`);
await mkdir(directory, { recursive: true });
const config = {
  ...template,
  batch_size: batch,
  context: tokens,
  max_seconds: 150,
  fusion: false,
  autotune: false,
  epilogue_backend: "native",
  artifact_dir: directory,
  budget_file: join(directory, "budget.json"),
};
const configPath = join(directory, "config.json");
await Bun.write(configPath, JSON.stringify(config, null, 2) + "\n");
// Existing lock helper captures stdout/stderr and distinguishes queue/execution deadlines.
const child = Bun.spawn([
  process.execPath, "scripts/locked-gpu.ts", directory, "epilogue-benchmark",
  binary, "epilogue-benchmark", configPath,
  "--contexts", String(tokens), "--warmups", String(warmups), "--trials", String(trials),
], { stdout: "inherit", stderr: "inherit" });
const status = await child.exited;
console.log(JSON.stringify({ directory, config: configPath, batch, tokens, status }));
process.exitCode = status;
