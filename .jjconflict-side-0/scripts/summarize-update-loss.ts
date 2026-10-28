// Usage: bun scripts/summarize-update-loss.ts OUTPUT.json REPORT.json [...]
// Reports contain interleaved, equal-start trajectories; positive differences mean saved time.
const [output, ...paths] = process.argv.slice(2);
if (!output || paths.length === 0) throw new Error("output and at least one report required");
function median(values: number[]): number {
  const sorted = values.toSorted((a, b) => a - b);
  const half = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[half] : (sorted[half - 1] + sorted[half]) / 2;
}
let state = 0x5eed;
function random(): number {
  state = (Math.imul(1664525, state) + 1013904223) >>> 0;
  return state / 4294967296;
}
const reports = [];
for (const path of paths) {
  const report = await Bun.file(path).json();
  if (!report.completed) throw new Error(`incomplete report: ${path}`);
  const samples: number[][] = report.samples_ms;
  if (samples.some(s => s.length !== samples[0].length || s.length < 3 || s.some(x => !Number.isFinite(x)))) throw new Error(`invalid paired samples: ${path}`);
  const modes = samples.map((samples, index) => {
    const saved = samples.map((ms, trial) => report.samples_ms[0][trial] - ms);
    const savedPercent = saved.map((ms, trial) => ms * 100 / report.samples_ms[0][trial]);
    const bootstrap = Array.from({ length: 10000 }, () => median(Array.from({ length: saved.length }, () => saved[Math.floor(random() * saved.length)]))).sort((a, b) => a - b);
    return {
      mode: report.modes[index], median_ms: median(samples),
      paired_median_saved_ms: median(saved), paired_median_saved_percent: median(savedPercent),
      paired_bootstrap95_saved_ms: [bootstrap[249], bootstrap[9749]],
      issued_kernels_per_step: report.dispatches[index].issued,
    };
  });
  reports.push({ path, batch: report.config.batch_size, context: report.config.context, trials: samples[0].length, modes,
    guarded_update_medians_ms: report.isolated_guarded_update_ms.map(median),
    final_norm_head_loss_f_fb_medians_ms: report.isolated_final_norm_head_loss_f_fb_ms?.map((mode: number[][]) => mode.map(median)),
    sampled_reserved_bytes: report.sampled_reserved_bytes });
}
const summary = { schema: 1, bootstrap: { statistic: "median of within-trial baseline-minus-candidate differences", samples: 10000, seed: "0x5eed", interval: "percentile 2.5%..97.5%; exploratory, not adjusted for multiple candidates" }, reports };
await Bun.write(output, JSON.stringify(summary, null, 2) + "\n");
console.log(JSON.stringify(summary));
