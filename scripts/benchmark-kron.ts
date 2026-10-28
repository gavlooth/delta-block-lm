import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";

const backend = process.argv[2] ?? "vulkan";
const binary = resolve(process.argv[3] ?? "target/debug/delta-experiment");
const warmups = Number(process.argv[4] ?? "2");
const trials = Number(process.argv[5] ?? "5");
const mode = process.argv[6] ?? "strict";
const legacyReport = process.argv[7] ? resolve(process.argv[7]) : undefined;
if (!["strict", "model"].includes(mode)) throw new Error("scope: strict | model");
if (legacyReport && !await Bun.file(legacyReport).exists()) throw new Error("retained legacy stress evidence does not exist");
if (!["cpu", "vulkan", "both"].includes(backend)) throw new Error("backend: cpu | vulkan | both (local only)");
if (!Number.isInteger(warmups) || warmups < 1 || warmups > 20 || !Number.isInteger(trials) || trials < 3 || trials > 31) throw new Error("warmups 1..20; trials 3..31");
if (!await Bun.file(binary).exists()) throw new Error("Build the CPU/Vulkan executable once centrally before benchmarking");
const template = await Bun.file("configs/kron-benchmark.json").json();
const directory = resolve(`artifacts/kron-benchmark-batch-${Date.now()}-${process.pid}`);
await mkdir(directory, { recursive: true });
const outcomes: object[] = [];
for (const selected of backend === "both" ? ["cpu", "vulkan"] : [backend]) {
  const config = { ...template, backend: selected, artifact_dir: join(directory, selected) };
  const configPath = join(directory, `${selected}-config.json`);
  const stdout = join(directory, `${selected}.stdout.jsonl`);
  const stderr = join(directory, `${selected}.stderr.log`);
  await Bun.write(configPath, JSON.stringify(config, null, 2) + "\n");
  const seconds = Math.ceil(config.max_seconds) + 30;
  const start = performance.now();
  const command = mode === "model" ? "kron-model-benchmark" : "kron-benchmark";
  const evidence = mode === "model" && legacyReport ? ["--legacy-stress-report", legacyReport] : [];
  const child = Bun.spawn(["timeout", "--signal=TERM", "--kill-after=5s", `${seconds}s`, binary, command, configPath, "--warmups", String(warmups), "--trials", String(trials), ...evidence], { stdout: Bun.file(stdout), stderr: Bun.file(stderr) });
  const exit = await child.exited;
  outcomes.push({ backend: selected, command, config: configPath, deadline_seconds: seconds, elapsed_seconds: (performance.now() - start) / 1000, exit_code: exit, stdout, stderr });
  await Bun.write(join(directory, "command-outcomes.json"), JSON.stringify({ directory, mode, legacy_report: legacyReport, warmups, trials, outcomes }, null, 2) + "\n");
  if (exit !== 0) throw new Error(`Kron benchmark ${selected} exited ${exit}; partial reports retained under ${config.artifact_dir}; see ${stderr}`);
}
console.log(JSON.stringify({ directory, mode, legacy_report: legacyReport, warmups, trials, outcomes }));
