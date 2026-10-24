// Bounded, serial GPU evaluation across compile-time Burn autotune feature choices.
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
const [mode, offArg, onArg, configArg, rootArg, batchArg = "8", pairsArg = "11", headArg = "native"] = Bun.argv.slice(2);
if (!rootArg || !["prepare", "micro-off", "micro-on", "pair"].includes(mode)) {
  throw new Error("usage: bun scripts/benchmark-gemm.ts prepare|micro-off|micro-on|pair OFF_BINARY ON_BINARY CONFIG ROOT [batch=8 pairs=11 head=native|padded320|heuristic]");
}
if (!["native","padded320","heuristic"].includes(headArg)) throw new Error("unknown candidate head backend");
const root = resolve(rootArg), off = resolve(offArg), on = resolve(onArg);
await mkdir(root, { recursive: true });
const base = await Bun.file(configArg).json();
for (const batch of [1, 8, 16]) {
  for (const tuned of [false, true]) {
    const dir = `${root}/${tuned ? "on" : "off"}-b${batch}`;
    await mkdir(dir, { recursive: true });
    const config = { ...base, batch_size: batch, artifact_dir: dir, budget_file: null, max_seconds: 290 };
    if (tuned) config.autotune = true;
    else delete config.autotune;
    if (tuned && headArg !== "native") config.head_backend = headArg;
    else delete config.head_backend;
    await Bun.write(`${dir}/config.json`, JSON.stringify(config, null, 2) + "\n");
  }
}
async function run(argv: string[], label: string) {
  const start = performance.now();
  const proc = Bun.spawn(["timeout", "290s", ...argv], { stdout: "pipe", stderr: "pipe", timeout: 295_000 });
  const [stdout, stderr, status] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
  await Bun.write(`${root}/${label}.stdout`, stdout);
  await Bun.write(`${root}/${label}.stderr`, stderr);
  const wall = (performance.now() - start) / 1000;
  await Bun.write(`${root}/${label}.process.json`, JSON.stringify({ argv, status, wall_seconds: wall }, null, 2));
  if (status !== 0) throw new Error(`${label}: exit ${status}; raw output retained`);
  const lines = stdout.trim().split("\n");
  return JSON.parse(lines[lines.length - 1]);
}
const median = (xs: number[]) => {
  const a = xs.toSorted((a,b) => a-b), mid = a.length >> 1;
  return a.length % 2 ? a[mid] : (a[mid-1] + a[mid]) / 2;
};
function pairedSummary(rows: { off: number; on: number }[]) {
  const diffs = rows.map(x => x.off - x.on);
  let state = 3107;
  const draw = () => { state = (Math.imul(state,1664525)+1013904223) >>> 0; return state % rows.length; };
  const boots = Array.from({length:10000}, () => median(rows.map(() => diffs[draw()]))).sort((a,b) => a-b);
  const before = median(rows.map(x => x.off)), after = median(rows.map(x => x.on));
  return { off_median_ms: before, on_median_ms: after, speedup: before/after,
    paired_saved_median_ms: median(diffs), paired_saved_bootstrap_95_ms: [boots[249],boots[9749]],
    on_faster_pairs: diffs.filter(x => x>0).length, pairs: rows.length };
}
if (mode === "prepare") {
  const proc = Bun.spawn(["timeout", "20s", "vulkaninfo"], { stdout:"pipe", stderr:"pipe" });
  const [stdout,stderr,status] = await Promise.all([new Response(proc.stdout).text(),new Response(proc.stderr).text(),proc.exited]);
  await Bun.write(`${root}/vulkaninfo.txt`,stdout);
  await Bun.write(`${root}/vulkaninfo.stderr`,stderr);
  if(status!==0) throw new Error("vulkaninfo failed");
} else if (mode.startsWith("micro")) {
  const tuned = mode === "micro-on";
  const result = await run([tuned ? on : off,"gemm-micro",`${root}/${tuned ? "on" : "off"}-b8/config.json`,"--warmups","2","--trials","7"],mode);
  console.log(JSON.stringify({report:result.report_path, completed:result.completed, elapsed:result.elapsed_seconds}));
} else {
  const batch = Number(batchArg), pairs = Number(pairsArg);
  if (![1,8].includes(batch) || !Number.isInteger(pairs) || pairs<3 || pairs>31) throw new Error("pair needs B1/B8 and 3..31 pairs");
  const rows: { off:number; on:number; off_first_ms:number; on_first_ms:number }[] = [];
  for(let i=0;i<pairs;i++) {
    const row: Record<string,number> = {};
    for(const tuned of i%2 === 0 ? [false,true] : [true,false]) {
      const arm = tuned ? "on" : "off";
      const result = await run([tuned ? on : off,"gemm-step",`${root}/${arm}-b${batch}/config.json`,"--warmups","2","--trials","1"],`pair-b${batch}-${i}-${arm}`);
      row[arm] = result.median_ms;
      row[`${arm}_first_ms`] = result.first_use_step_ms;
    }
    rows.push(row as typeof rows[number]);
    await Bun.write(`${root}/paired-b${batch}.json`,JSON.stringify({batch,seed:base.seed,context:512,protocol:"fresh identical-seed trajectories; alternating AB/BA process order; first use + two warmups excluded; fourth complete guarded step measured; bootstrap of paired median saved ms, 10000 deterministic resamples",rows,...pairedSummary(rows)},null,2)+"\n");
  }
  console.log(JSON.stringify(pairedSummary(rows)));
}
