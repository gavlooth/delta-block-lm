import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";

type DocumentPin = { id: string; title: string; author: string; split: "train" | "validation" | "test"; url: string; sha256: string };
type Lock = { name: string; license: string; license_url: string; tokenizer: string; extraction: string; documents: DocumentPin[] };
const sha256 = (bytes: Uint8Array | string) => createHash("sha256").update(bytes).digest("hex");
const lockPath = process.argv[2] ?? "configs/corpus.lock.json";
const out = process.argv[3] ?? "data/books";
const lockBytes = new Uint8Array(await Bun.file(lockPath).arrayBuffer());
const lock: Lock = JSON.parse(new TextDecoder().decode(lockBytes));
await mkdir(join(out, "raw"), { recursive: true });
await mkdir(join(out, "documents"), { recursive: true });
const ids = new Set<string>();
const contentHashes = new Set<string>();
const documents = [];
for (const pin of lock.documents) {
  if (!/^pg[0-9]+$/.test(pin.id) || ids.has(pin.id)) throw new Error(`Invalid/duplicate document ID ${pin.id}`);
  ids.add(pin.id);
  const url = new URL(pin.url);
  if (url.origin !== "https://www.gutenberg.org" || !/^\/ebooks\/[0-9]+\.txt\.utf-8$/.test(url.pathname)) throw new Error("Unexpected corpus host/path");
  const rawPath = join(out, "raw", `${pin.id}.txt`);
  let bytes: Uint8Array;
  if (await Bun.file(rawPath).exists()) bytes = new Uint8Array(await Bun.file(rawPath).arrayBuffer());
  else {
    const response = await fetch(pin.url, { signal: AbortSignal.timeout(20_000) });
    if (!response.ok) throw new Error(`${pin.id}: HTTP ${response.status}`);
    bytes = new Uint8Array(await response.arrayBuffer());
  }
  if (bytes.length > 2_000_000 || sha256(bytes) !== pin.sha256) throw new Error(`${pin.id}: source content hash changed; do not silently repin`);
  const raw = new TextDecoder("utf-8", { fatal: true }).decode(bytes).replace(/\r\n?/g, "\n");
  if (!raw.includes("This eBook is for the use of anyone anywhere in the United States")) throw new Error(`${pin.id}: expected source license not found`);
  const start = /^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*\*\*\*\s*$/m.exec(raw);
  const end = /^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*\*\*\*\s*$/m.exec(raw);
  if (!start || !end || end.index <= start.index) throw new Error(`${pin.id}: missing document delimiters`);
  const text = raw.slice(start.index + start[0].length, end.index).trim() + "\n";
  const content = new TextEncoder().encode(text);
  const digest = sha256(content);
  if (contentHashes.has(digest)) throw new Error("Duplicate book content across splits");
  contentHashes.add(digest);
  const relative = `documents/${pin.id}.txt`;
  await Bun.write(rawPath, bytes); // Preserve the complete license-bearing source, separately from training text.
  await Bun.write(join(out, relative), content);
  documents.push({ id: pin.id, title: pin.title, author: pin.author, split: pin.split, path: relative, sha256: digest, bytes: content.length, source_url: pin.url, source_sha256: pin.sha256 });
}
const split_sha256 = Object.fromEntries(["train", "validation", "test"].map(split => [split, sha256(JSON.stringify(documents.filter(d => d.split === split).map(d => [d.id, d.sha256])))]));
const tokenizer = { name: lock.tokenizer, vocabulary: 257, byte_ids: [0, 255], bos_id: 256, normalization: "source wrapper removed; CRLF to LF; no learned vocabulary", document_boundary: "new BOS and reset for each independently sampled window; windows never cross books" };
await Bun.write(join(out, "tokenizer.json"), JSON.stringify(tokenizer, null, 2) + "\n");
const manifest = { schema: 1, name: lock.name, tokenizer: lock.tokenizer, tokenizer_sha256: sha256(JSON.stringify(tokenizer)), source_lock_sha256: sha256(lockBytes), license: lock.license, license_url: lock.license_url, extraction: lock.extraction, documents, split_sha256 };
const encoded = JSON.stringify(manifest, null, 2) + "\n";
await Bun.write(join(out, "manifest.json"), encoded);
console.log(JSON.stringify({ manifest: join(out, "manifest.json"), manifest_sha256: sha256(encoded), split_sha256, documents: documents.map(d => ({ id: d.id, split: d.split, bytes: d.bytes, sha256: d.sha256 })) }));
