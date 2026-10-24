import { mkdir } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

// Invoked only after both hardware locks have been acquired. Keep waiting
// outside the compute deadline and report it independently of GPU execution.
if (process.argv[2] === "--locked-worker") {
  const queueWait = (Date.now() - Number(process.argv[3])) / 1000;
  const start = performance.now();
  const child = Bun.spawn(["timeout", "--signal=TERM", "--kill-after=5s", "300s", ...process.argv.slice(4)], { stdout: "inherit", stderr: "inherit" });
  const exit = await child.exited;
  console.error(JSON.stringify({ soft_lock_measurement: true, queue_wait_seconds: queueWait, compute_seconds: (performance.now() - start) / 1000 }));
  process.exit(exit);
}

const [binaryArg, mode, warmupArg, trialArg, ...inputs] = process.argv.slice(2);
if (!binaryArg || !["selection", "step"].includes(mode) || inputs.length === 0) {
  throw new Error("usage: bun scripts/benchmark-soft.ts BINARY selection|step WARMUPS TRIALS CONFIG...");
}
const binary = resolve(binaryArg);
const warmups = Number(warmupArg);
const trials = Number(trialArg);
if (!Number.isInteger(warmups) || warmups < 1 || warmups > 3 || !Number.isInteger(trials) || trials < 1 || trials > 9) {
  throw new Error("warmups 1..3; trials 1..9");
}
if (!await Bun.file(binary).exists()) throw new Error("build once with ../build.sh softtopk first");
const directory = resolve(`artifacts/soft-topk/${mode}-${Date.now()}-${process.pid}`);
await mkdir(directory, { recursive: true });
const outcomes: object[] = [];
for (const input of inputs) {
  const config = await Bun.file(input).json();
  if (config.backend !== "vulkan" || config.fusion || config.autotune) throw new Error("requires heuristic non-fusion Vulkan");
  config.artifact_dir = directory;
  config.budget_file = null;
  config.max_seconds = 285;
  const name = basename(input, ".json");
  const path = join(directory, `${name}.json`);
  await Bun.write(path, JSON.stringify(config, null, 2) + "\n");
  const stdout = join(directory, `${name}.stdout.jsonl`);
  const stderr = join(directory, `${name}.stderr.log`);
  const command = ["flock", "/tmp/delta-block-lm-gpu.lock", "flock", "/tmp/delta-block-lm-build.lock", process.execPath, import.meta.path, "--locked-worker", String(Date.now()), binary, mode === "selection" ? "soft-benchmark" : "soft-step", path, "--warmups", String(warmups), "--trials", String(trials)];
  const started = performance.now();
  const child = Bun.spawn(command, { stdout: Bun.file(stdout), stderr: Bun.file(stderr) });
  const exit = await child.exited;
  const metadataLine = (await Bun.file(stderr).text()).split("\n").find(line => line.startsWith('{"soft_lock_measurement":true,'));
  const lockMeasurement = metadataLine ? JSON.parse(metadataLine) : null;
  outcomes.push({ input, config: path, command, exit_code: exit, lock_measurement: lockMeasurement, elapsed_seconds_including_lock_wait: (performance.now() - started) / 1000, stdout, stderr });
  if (exit === 0 && !lockMeasurement) throw new Error(`missing hardware lock measurement: ${stderr}`);
  await Bun.write(join(directory, "command-outcomes.json"), JSON.stringify({ directory, binary, mode, warmups, trials, outcomes }, null, 2) + "\n");
  if (exit !== 0) throw new Error(`soft ${mode} ${name} exited ${exit}; partial evidence: ${directory}`);
}
console.log(JSON.stringify({ directory, binary, mode, warmups, trials, outcomes }));
