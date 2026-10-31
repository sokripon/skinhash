// Proves the vendored pako deflate is byte-identical to Python's zlib (level 4, wbits 15, memLevel 8, default strategy)
// on real skins plus many randomized variants. Run: node web/test/pako_exact.mjs
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import pako from "../vendor/pako.esm.mjs";
import { decodePng, rawScanlines, RAW_LEN, W, H, DEFLATE_OPTS } from "../js/png.js";
import { prepareSkin, makeRng, pickVals, knobLayout } from "../js/job.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const skins = ["scorpion3013.png", "obus_test.png"].map((n) => path.join(here, "..", "..", "skins", n));
const rng = makeRng(12345);
const cases = []; // {name, raw}

for (const s of skins) {
  const rgba = decodePng(new Uint8Array(fs.readFileSync(s))).rgba;
  const name = path.basename(s);
  cases.push({ name: name + " original", raw: rawScanlines(rgba) });
  const prep = prepareSkin(rgba);
  cases.push({ name: name + " +watermark", raw: Uint8Array.from(prep.raw) });
  // random hidden RGB under every alpha-0 pixel
  for (let i = 0; i < 70; i++) {
    const v = Uint8Array.from(rgba);
    for (let p = 0; p < W * H; p++) if (v[p * 4 + 3] === 0) { v[p * 4] = rng.below(256); v[p * 4 + 1] = rng.below(256); v[p * 4 + 2] = rng.below(256); }
    cases.push({ name: `${name} hidden-RGB #${i}`, raw: rawScanlines(v) });
  }
  // random hidden RGB under a random subset, plus watermark
  for (let i = 0; i < 30; i++) {
    const v = Uint8Array.from(prep.rgba);
    for (let p = 0; p < W * H; p++) if (v[p * 4 + 3] === 0 && rng.below(3) === 0) { v[p * 4] = rng.below(256); v[p * 4 + 1] = rng.below(256); v[p * 4 + 2] = rng.below(256); }
    cases.push({ name: `${name} partial-hidden #${i}`, raw: rawScanlines(v) });
  }
  // search-like: random knob permutations of unused values in the last K pixels
  if (prep.eligible) for (const K of [5, 6]) for (let i = 0; i < 40; i++) {
    const raw = Uint8Array.from(prep.raw);
    const vals = Array.from(pickVals(prep, K, rng));
    knobLayout(K).knob.forEach((o, j) => { raw[o] = vals[j]; });
    cases.push({ name: `${name} knobs K=${K} #${i}`, raw });
  }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pako_exact_"));
const inp = path.join(tmp, "cases.bin"), outp = path.join(tmp, "zlib.bin");
const all = new Uint8Array(cases.length * RAW_LEN);
cases.forEach((c, i) => all.set(c.raw, i * RAW_LEN));
fs.writeFileSync(inp, all);
const r = spawnSync("py", [path.join(here, "zlib_ref.py"), inp, String(RAW_LEN), outp], { encoding: "utf8" });
if (r.status !== 0) { console.error(r.stdout, r.stderr); process.exit(2); }
console.log(r.stdout.trim());
const ref = new Uint8Array(fs.readFileSync(outp));
const dv = new DataView(ref.buffer, ref.byteOffset);
let pos = 0, bad = 0, distinct = new Set();
for (const c of cases) {
  const len = dv.getUint32(pos, true); pos += 4;
  const z = ref.subarray(pos, pos + len); pos += len;
  const p = pako.deflate(c.raw, DEFLATE_OPTS);
  distinct.add(z.length + ":" + z[z.length - 1] + z[z.length - 5]);
  if (p.length !== z.length || p.some((x, i) => x !== z[i])) {
    bad++;
    console.error(`MISMATCH: ${c.name}: pako ${p.length} bytes vs zlib ${z.length} bytes`);
  }
}
fs.rmSync(tmp, { recursive: true, force: true });
console.log(`pako vs zlib: ${cases.length - bad}/${cases.length} cases byte-identical`);
if (bad) { console.error("pako is NOT bit-exact; do not ship."); process.exit(1); }
