import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";

const backend = process.argv[2] ?? "vulkan";
const binary = resolve(process.argv[3] ?? "target/debug/delta-experiment");
const warmups = Number(process.argv[4] ?? "2");
const trials = Number(process.argv[5] ?? "5");
if (!["cpu", "vulkan"].includes(backend)) throw new Error("backend: cpu | vulkan");
if (!Number.isInteger(warmups) || warmups < 1 || warmups > 20 || !Number.isInteger(trials) || trials < 3 || trials > 31) throw new Error("warmups 1..20; trials 3..31");
if (!await Bun.file(binary).exists()) throw new Error("Build the CPU/Vulkan executable once centrally before benchmarking");
const template = await Bun.file("configs/kron-benchmark.json").json();
const directory = resolve(`artifacts/gdn/benchmark-${Date.now()}-${process.pid}`);
await mkdir(directory, { recursive: true });
const config = {
  ...template, backend, batch_size: 8,
  artifact_dir: directory,
  arms: [
    { name: "gdn2", mixer: "gdn2", cache: "none" },
    { name: "gdn2-hola", mixer: "gdn2", cache: "surprise" },
  ],
  scan_backend: "tensor_solver",
};
const configPath = join(directory, "config.json");
const stdout = join(directory, "stdout.jsonl");
const stderr = join(directory, "stderr.log");
await Bun.write(configPath, JSON.stringify(config, null, 2) + "\n");
const seconds = Math.ceil(config.max_seconds) + 30;
const argv = ["timeout", "--signal=TERM", "--kill-after=5s", `${seconds}s`, binary, "gdn-benchmark", configPath, "--warmups", String(warmups), "--trials", String(trials)];
const started = performance.now();
const child = Bun.spawn(argv, { stdout: Bun.file(stdout), stderr: Bun.file(stderr) });
const exit = await child.exited;
const outcome = { directory, argv, deadline_seconds: seconds, elapsed_seconds: (performance.now() - started) / 1000, exit_code: exit, stdout, stderr };
await Bun.write(join(directory, "command-outcomes.json"), JSON.stringify(outcome, null, 2) + "\n");
console.log(JSON.stringify(outcome));
if (exit !== 0) throw new Error(`GDN benchmark exited ${exit}; partial evidence retained in ${directory}`);
