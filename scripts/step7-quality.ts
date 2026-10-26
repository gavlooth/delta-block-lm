import { mkdir, readdir, rm } from "node:fs/promises";
import { join, resolve } from "node:path";

// Each GPU child obtains and releases the shared GPU lock independently.
const root = "artifacts/step7-quality";
const script = resolve(import.meta.path);
const binary = resolve(process.env.DELTA_BINARY ?? "/tmp/delta-bins/s7q/delta-experiment");
const steps = 440;
const seeds = [1, 2, 3];
const names = ["off", "wavelet", "lngram", "ohc", "all"];
const armName = "kronlstm-hola";
const configPath = (name: string) => join(root, "configs", `${name}.json`);
const runPath = (name: string, seed: number) => join(root, name, `${armName}-seed${seed}`);
const json = async (path: string) => Bun.file(path).json();
const save = async (path: string, value: unknown) => Bun.write(path, JSON.stringify(value, null, 2) + "\n");
async function hash(path: string): Promise<string> {
  const digest = new Bun.CryptoHasher("sha256");
  for await (const chunk of Bun.file(path).stream()) digest.update(chunk);
  return digest.digest("hex");
}
async function prepare() {
  await mkdir(join(root, "configs"), { recursive: true });
  await mkdir(join(root, "commands"), { recursive: true });
  const base = await json("configs/lm512-optimized.json");
  const target = await json("configs/lm512-step7.json");
  const fields: Record<string, string[]> = {
    wavelet: ["wavelet_layers", "wavelet_backend"],
    lngram: ["lngram_layers", "lngram_orders", "lngram_routes", "lngram_bits", "lngram_memory_dim", "lngram_backend"],
    ohc: ["ohc_streams", "ohc_backend"],
  };
  for (const name of names) {
    const config = { ...base, artifact_dir: join(root, name), budget_file: null,
      batch_size: 8, context: 512, eval_examples: 4, seeds, train_tokens: steps * 8 * 512,
      checkpoint_every: 1_000_000_000, max_seconds: 820 };
    for (const [component, keys] of Object.entries(fields)) {
      if (name === component || name === "all") for (const key of keys) config[key] = target[key];
    }
    const path = configPath(name);
    const bytes = JSON.stringify(config, null, 2) + "\n";
    if (await Bun.file(path).exists() && await Bun.file(path).text() !== bytes) {
      throw new Error(`Existing study configuration differs: ${path}`);
    }
    await Bun.write(path, bytes);
  }
  const provenance = {
    schema: 1, names, seeds, steps, tokens_per_run: steps * 8 * 512,
    binary, binary_sha256: await hash(binary),
    base_config_sha256: await hash("configs/lm512-optimized.json"),
    step7_config_sha256: await hash("configs/lm512-step7.json"),
    cargo_lock_sha256: await hash("Cargo.lock"),
    corpus_manifest_sha256: await hash("data/books/manifest.json"),
    deadline_seconds_per_command: 900, training_soft_deadline_seconds: 820,
    target_total_gpu_seconds: 9000,
    evaluation: "Certified original four deterministic nonoverlapping stratified-midpoint 512-byte validation windows, plus separate 128-window validation of the same final checkpoints. Conclusions use the larger sample. Test is not used for selection.",
    budget_reason: "440 steps at the existing 1.775 s all-components median is about 781 s; reserve the rest of each 900 s invocation for startup, checkpoint and evaluations. Three seeds allow unchanged certification.",
  };
  await save(join(root, "study.json"), provenance);
}
async function locked(label: string, command: string[]) {
  const stdout = join(root, "commands", `${label}.stdout.jsonl`);
  const stderr = join(root, "commands", `${label}.stderr.log`);
  const outcome = join(root, "commands", `${label}.outcome.json`);
  if (await Bun.file(outcome).exists()) throw new Error(`Command evidence already exists: ${outcome}`);
  const started = performance.now();
  const startedAt = new Date().toISOString();
  const child = Bun.spawn(["timeout", "--signal=TERM", "--kill-after=5s", "900s", binary, ...command],
    { stdout: Bun.file(stdout), stderr: Bun.file(stderr) });
  const exit = await child.exited;
  const result = { label, command: [binary, ...command], deadline_seconds: 900,
    started_at: startedAt, elapsed_seconds: (performance.now() - started) / 1000,
    exit_code: exit, stdout, stderr, lock_wait_excluded: true };
  await save(outcome, result);
  console.log(JSON.stringify(result));
  if (exit !== 0) process.exit(exit);
}
async function execute(label: string, command: string[]) {
  const child = Bun.spawn(["flock", "/tmp/delta-block-lm-gpu.lock",
    process.execPath, script, "locked", label, ...command], { stdout: "inherit", stderr: "inherit" });
  const exit = await child.exited;
  if (exit !== 0) throw new Error(`${label} exited ${exit}; inspect retained command evidence`);
}
async function train(name: string, seed: number) {
  let attempt = 1;
  const directory = runPath(name, seed);
  for (;;) {
    const metrics = join(directory, "metrics.json");
    if (await Bun.file(metrics).exists() && (await json(metrics)).completed) break;
    while (await Bun.file(join(root, "commands", `${name}-seed${seed}-train${attempt}.outcome.json`)).exists()) attempt++;
    const latest = join(directory, "latest.json");
    const command = ["train", configPath(name), "--arm", armName, "--seed", String(seed)];
    if (await Bun.file(latest).exists()) command.push("--resume", (await json(latest)).checkpoint);
    await execute(`${name}-seed${seed}-train${attempt}`, command);
    attempt++;
  }
  const metrics = await json(join(directory, "metrics.json"));
  // Only this runner's obsolete partial checkpoints are removed after completion.
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && /^checkpoint-\d+$/.test(entry.name) && resolve(directory, entry.name) !== resolve(metrics.checkpoint)) {
      await rm(join(directory, entry.name), { recursive: true });
    }
  }
}
async function certify(name: string) {
  const label = `${name}-certify`;
  const outcome = join(root, "commands", `${label}.outcome.json`);
  if (await Bun.file(outcome).exists() && (await json(outcome)).exit_code === 0) return;
  const entries = [];
  for (const seed of seeds) {
    const metricsPath = join(runPath(name, seed), "metrics.json");
    const metrics = await json(metricsPath);
    if (!metrics.completed) throw new Error(`Incomplete run: ${metricsPath}`);
    const checkpoint = metrics.checkpoint;
    entries.push({ arm: armName, seed, checkpoint,
      checkpoint_sha256: await hash(join(checkpoint, "model.burnpack")),
      state_sha256: await hash(join(checkpoint, "state.json")),
      training_metrics_path: metricsPath, training_metrics_sha256: await hash(metricsPath) });
  }
  const manifest = join(root, `${name}-checkpoints.json`);
  await save(manifest, { schema: 2, source_config_sha256: await hash(configPath(name)), entries });
  await execute(label, ["certify", configPath(name), "--checkpoints", manifest]);
}
async function evaluateLarge(name: string, seed: number) {
  const label = `${name}-seed${seed}-evaluate128`;
  const outcome = join(root, "commands", `${label}.outcome.json`);
  if (await Bun.file(outcome).exists() && (await json(outcome)).exit_code === 0) return;
  const metrics = await json(join(runPath(name, seed), "metrics.json"));
  if (!metrics.completed) throw new Error(`Incomplete ${name} seed ${seed}`);
  await execute(label, ["evaluate", configPath(name), "--arm", armName, "--seed", String(seed),
    "--resume", metrics.checkpoint, "--eval-examples", "128"]);
}
async function summarize() {
  const outcomes = [];
  for (const entry of await readdir(join(root, "commands"))) {
    if (entry.endsWith(".outcome.json")) outcomes.push(await json(join(root, "commands", entry)));
  }
  outcomes.sort((a, b) => a.started_at.localeCompare(b.started_at));
  const rows = [];
  for (const name of names) {
    const result = await json(join(root, name, "certify.json"));
    const comparison = await json(result.comparison);
    for (const seed of seeds) {
      const metrics = await json(join(runPath(name, seed), "metrics.json"));
      const certified = comparison.runs.find((row: { seed: number }) => row.seed === seed);
      const large = await json(join(root, "commands", `${name}-seed${seed}-evaluate128.stdout.jsonl`));
      if (!large.passed) throw new Error(`Larger evaluation failed: ${name} seed ${seed}`);
      const curve = (await Bun.file(join(runPath(name, seed), "training.jsonl")).text()).trim().split("\n")
        .map(line => JSON.parse(line) as { loss: number });
      rows.push({ name, seed, nll: certified.validation.nll,
        initial_nll: certified.initial_validation.nll,
        large_nll: large.validation.nll, large_initial_nll: large.initial_validation.nll,
        large_evaluation: large.output, large_fixture_sha256: large.data.evaluation_sha256,
        paired_difference_vs_off: 0, large_paired_difference_vs_off: 0,
        live_initial_nll: metrics.initial_validation.nll,
        live_final_nll: metrics.validation.nll,
        initial_trusted: certified.initial_validation_trusted,
        parameters: metrics.parameters, tokens: metrics.tokens, steps: metrics.steps,
        tokens_per_second: metrics.training_tokens_per_second,
        training_seconds: metrics.total_training_seconds,
        train_loss_first: curve[0].loss, train_loss_last: curve.at(-1).loss,
        train_loss_first20_mean: curve.slice(0, 20).reduce((s, r) => s + r.loss, 0) / 20,
        train_loss_last20_mean: curve.slice(-20).reduce((s, r) => s + r.loss, 0) / 20,
        comparison: result.comparison });
    }
  }
  for (const row of rows) {
    const off = rows.find(r => r.name === "off" && r.seed === row.seed)!;
    row.paired_difference_vs_off = row.nll - off.nll;
    row.large_paired_difference_vs_off = row.large_nll - off.large_nll;
  }
  const stats = (values: number[]) => {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return { mean, population_sd: Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length),
      min: Math.min(...values), max: Math.max(...values) };
  };
  const summary = names.map(name => {
    const arm = rows.filter(row => row.name === name);
    return { name, nll: stats(arm.map(r => r.nll)), paired_difference_vs_off: stats(arm.map(r => r.paired_difference_vs_off)),
      large_nll: stats(arm.map(r => r.large_nll)),
      large_paired_difference_vs_off: stats(arm.map(r => r.large_paired_difference_vs_off)),
      tokens_per_second: stats(arm.map(r => r.tokens_per_second)), parameters: arm[0].parameters };
  });
  await save(join(root, "summary.json"), { rows, summary, outcomes,
    gpu_command_seconds: outcomes.reduce((total, row) => total + row.elapsed_seconds, 0),
    training_seconds: rows.reduce((total, row) => total + row.training_seconds, 0) });
  console.log(JSON.stringify({ summary, gpu_command_seconds: outcomes.reduce((total, row) => total + row.elapsed_seconds, 0) }));
}
const mode = process.argv[2] ?? "prepare";
if (mode === "locked") await locked(process.argv[3], process.argv.slice(4));
else if (mode === "prepare") await prepare();
else if (mode === "train") {
  if (!names.includes(process.argv[3]) || !seeds.includes(Number(process.argv[4]))) throw new Error("train ARM SEED");
  await train(process.argv[3], Number(process.argv[4]));
} else if (mode === "certify") {
  if (!names.includes(process.argv[3])) throw new Error("certify ARM");
  await certify(process.argv[3]);
} else if (mode === "evaluate") {
  if (!names.includes(process.argv[3]) || !seeds.includes(Number(process.argv[4]))) throw new Error("evaluate ARM SEED");
  await evaluateLarge(process.argv[3], Number(process.argv[4]));
} else if (mode === "summarize") await summarize();
else if (mode === "study") {
  await prepare();
  // Rotate arm order across paired seeds; the all-components run comes first to establish its actual cost.
  const order = ["all", "off", "wavelet", "lngram", "ohc"];
  for (const [index, seed] of seeds.entries()) {
    const rotated = [...order.slice(index), ...order.slice(0, index)];
    for (const name of rotated) await train(name, seed);
  }
  for (const name of names) await certify(name);
  for (const name of names) for (const seed of seeds) await evaluateLarge(name, seed);
  await summarize();
} else throw new Error("mode: prepare | study | train ARM SEED | certify ARM | evaluate ARM SEED | summarize");
