// Serialize validation/build transactions; distinguish bounded lock queueing from execution.
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
const argv = Bun.argv.slice(2);
const child = argv[0] === "--locked";
let requested = Date.now();
if (child) {
  argv.shift();
  requested = Number(argv.shift());
}
const [directoryArg, labelArg, ...command] = argv;
if (!directoryArg || !labelArg || command.length === 0 || !/^[a-zA-Z0-9_.-]+$/.test(labelArg)) {
  throw new Error("usage: bun scripts/locked-gpu.ts OUTPUT_DIR LABEL COMMAND [ARG...]");
}
const label = child ? labelArg : `${labelArg}-${requested}`;
const directory = resolve(directoryArg);
await mkdir(directory, { recursive: true });
if (!child) {
  const proc = Bun.spawn([
    "flock", "-E", "75", "-w", "80", "/tmp/delta-block-lm-gpu.lock",
    "flock", "-E", "75", "-w", "15", "/tmp/delta-block-lm-build.lock",
    process.execPath, resolve(Bun.argv[1]), "--locked", String(requested), directory, label, ...command,
  ], { stdout: "inherit", stderr: "inherit" });
  const status = await proc.exited;
  if (!(await Bun.file(`${directory}/${label}.process.json`).exists())) {
    const result = { command, status, phase: "lock_queue", queue_wait_seconds: (Date.now()-requested)/1000,
      execution_seconds: null, gpu_lock: "/tmp/delta-block-lm-gpu.lock", build_lock: "/tmp/delta-block-lm-build.lock" };
    await Bun.write(`${directory}/${label}.process.json`, JSON.stringify(result,null,2)+"\n");
    console.log(JSON.stringify(result));
  }
  process.exitCode = status;
} else {
  const start = Date.now();
  const proc = Bun.spawn(["timeout", "180s", ...command], { stdout:"pipe", stderr:"pipe", timeout:185_000 });
  const [stdout,stderr,status] = await Promise.all([new Response(proc.stdout).text(),new Response(proc.stderr).text(),proc.exited]);
  await Bun.write(`${directory}/${label}.stdout`,stdout);
  await Bun.write(`${directory}/${label}.stderr`,stderr);
  const result = { command, status, phase:"execution", queue_wait_seconds:(start-requested)/1000,
    execution_seconds:(Date.now()-start)/1000, execution_timeout_seconds:180,
    gpu_lock:"/tmp/delta-block-lm-gpu.lock",build_lock:"/tmp/delta-block-lm-build.lock",
    stdout:`${directory}/${label}.stdout`,stderr:`${directory}/${label}.stderr` };
  await Bun.write(`${directory}/${label}.process.json`,JSON.stringify(result,null,2)+"\n");
  console.log(JSON.stringify(result));
  process.exitCode=status;
}
