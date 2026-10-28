import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";

type Config = { artifact_dir: string; arms: { name: string }[]; seeds?: number[] };
const binary = resolve(process.argv[2] ?? "target/debug/delta-experiment");
const configs = process.argv.slice(3);
if (configs.length === 0) configs.push("artifacts/recall128-vulkan/resolved-config.json", "artifacts/recall512-vulkan/resolved-config.json", "artifacts/lm512-vulkan/resolved-config.json");
if (!await Bun.file(binary).exists()) throw new Error("Build once before running certification; this script never builds or trains");
async function fileHash(path: string): Promise<string> {
  const hasher = new Bun.CryptoHasher("sha256");
  for await (const chunk of Bun.file(path).stream()) hasher.update(chunk);
  return hasher.digest("hex");
}
const batchDirectory = resolve(`artifacts/certification-batch-${Date.now()}-${process.pid}`);
await mkdir(batchDirectory, { recursive: true });
const outcomes = [];
for (const [configIndex, source] of configs.entries()) {
  const configPath = resolve(source);
  const config: Config = await Bun.file(configPath).json();
  const seeds = config.seeds ?? [1, 2, 3];
  if (seeds.length !== 3 || new Set(seeds).size !== 3) throw new Error("Certification requires three distinct paired seeds");
  const entries = [];
  for (const seed of seeds) for (const arm of config.arms) {
    const metricsPath = resolve(config.artifact_dir, `${arm.name}-seed${seed}`, "metrics.json");
    const metrics = await Bun.file(metricsPath).json();
    if (!metrics.completed || metrics.arm?.name !== arm.name || metrics.seed !== seed || typeof metrics.checkpoint !== "string") throw new Error(`Missing/mismatched completed original run: ${metricsPath}`);
    const checkpoint = resolve(metrics.checkpoint);
    entries.push({ arm: arm.name, seed, checkpoint, checkpoint_sha256: await fileHash(join(checkpoint, "model.burnpack")), state_sha256: await fileHash(join(checkpoint, "state.json")), training_metrics_path: metricsPath, training_metrics_sha256: await fileHash(metricsPath) });
  }
  const label = `${String(configIndex).padStart(3, "0")}-${source.replaceAll(/[^a-zA-Z0-9_-]/g, "_")}`;
  const manifestPath = join(batchDirectory, `${label}-checkpoints.json`);
  await Bun.write(manifestPath, JSON.stringify({ schema: 2, source_config_sha256: await fileHash(configPath), entries }, null, 2) + "\n");
  const stdout = join(batchDirectory, `${label}.stdout.jsonl`);
  const stderr = join(batchDirectory, `${label}.stderr.log`);
  const started = performance.now();
  const command = ["timeout", "--signal=TERM", "--kill-after=5s", "300s", binary, "certify", configPath, "--checkpoints", manifestPath];
  const child = Bun.spawn(command, { stdout: Bun.file(stdout), stderr: Bun.file(stderr) });
  const exit = await child.exited;
  outcomes.push({ config: configPath, manifest: manifestPath, deadline_seconds: 300, exit_code: exit, elapsed_seconds: (performance.now() - started) / 1000, stdout, stderr });
  await Bun.write(join(batchDirectory, "command-outcomes.json"), JSON.stringify({ new_training_tokens: 0, outcomes }, null, 2) + "\n");
  if (exit !== 0) throw new Error(`Certification failed (${exit}); original evidence preserved; see ${stderr}`);
}
console.log(JSON.stringify({ certification_batch: batchDirectory, new_training_tokens: 0, outcomes }));
