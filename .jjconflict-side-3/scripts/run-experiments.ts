import { mkdir } from "node:fs/promises";
import { resolve, join } from "node:path";

const suite = process.argv[2] ?? "smoke";
const backend = process.argv[3] ?? "cpu";
const binary = resolve(process.argv[4] ?? "target/debug/delta-experiment");
if (!["smoke", "recall128", "recall512", "lm-smoke", "lm512"].includes(suite)) throw new Error("suite: smoke | recall128 | recall512 | lm-smoke | lm512");
if (!["cpu", "vulkan"].includes(backend)) throw new Error("backend: cpu | vulkan (local only)");
if (!await Bun.file(binary).exists()) throw new Error("Build the requested backend once before running experiments");
const source = suite === "smoke" ? "recall-smoke" : suite;
const config = await Bun.file(`configs/${source}.json`).json();
config.backend = backend;
config.artifact_dir = `artifacts/${suite}-${backend}`;
if (config.budget_file) config.budget_file = `${config.artifact_dir}/budget.json`;
await mkdir(config.artifact_dir, { recursive: true });
const configPath = join(config.artifact_dir, "resolved-config.json");
await Bun.write(configPath, JSON.stringify(config, null, 2) + "\n");
const budgetReused = !!config.budget_file && await Bun.file(config.budget_file).exists();
const stages = suite === "smoke" ? ["reference", "soft", "compare"] : suite === "lm-smoke" ? ["compare"] : budgetReused ? ["compare", "decision"] : ["benchmark", "compare", "decision"];
const outcomes: { stage: string; deadline_seconds: number; elapsed_seconds: number; exit_code: number; stdout: string; stderr: string }[] = [];
for (const stage of stages) {
  const deadline = stage === "reference" ? 90 : stage === "soft" ? 30 : stage === "benchmark" ? 180 : stage === "decision" ? 900 : 300;
  const stdout = join(config.artifact_dir, `${stage}.stdout.jsonl`);
  const stderr = join(config.artifact_dir, `${stage}.stderr.log`);
  const start = performance.now();
  const child = Bun.spawn(["timeout", "--signal=TERM", "--kill-after=5s", `${deadline}s`, binary, stage, configPath], { stdout: Bun.file(stdout), stderr: Bun.file(stderr) });
  const exit = await child.exited;
  outcomes.push({ stage, deadline_seconds: deadline, elapsed_seconds: (performance.now() - start) / 1000, exit_code: exit, stdout, stderr });
  await Bun.write(join(config.artifact_dir, "command-outcomes.json"), JSON.stringify({ suite, backend, config: configPath, budget_reused: budgetReused, outcomes }, null, 2) + "\n");
  if (exit !== 0) throw new Error(`${stage} exited ${exit}; partial/checkpoint artifacts retained; see ${stderr}`);
  if (stage === "compare") {
    const smoke = await Bun.file(join(config.artifact_dir, "smoke.json")).json();
    if (!smoke.all_budgets_completed) throw new Error("One-seed smoke did not finish its fixed budget; resume the same command before a three-seed decision");
  }
}
console.log(JSON.stringify({ suite, backend, config: configPath, budget_reused: budgetReused, outcomes }));
