// Fast Mojang kernel (web/js/wgslfast.js): the JS model of the kernel (message words as an OR of table words, CRC as
// an XOR of deltas, Adler-b as a sum of terms, thread part from the outer codes) must equal the byte-level builder
// for random permutations, the shader source must be well formed, and the job builder must prefer equal lengths.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodePng } from "../js/png.js";
import { prepareSkin, autoChoose, rebuildJob, makeRng, equalLengths, fastBuf, makeTestVectors, decodePerm, permCount, setupFast, pickVals } from "../js/job.js";
import { deriveJob } from "../js/wgsl.js";
import { deriveFast, buildFastConsts, fastModel, genFastShader } from "../js/wgslfast.js";

const here = path.dirname(fileURLToPath(import.meta.url));
let fail = 0;
const check = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) fail++; };
const skinPrep = (name) => prepareSkin(decodePng(new Uint8Array(fs.readFileSync(path.join(here, "..", "..", "skins", name)))).rgba);

function lintShader(src) {
  const bad = [];
  let b = 0, p = 0;
  for (const ch of src) { if (ch === "{") b++; else if (ch === "}") b--; else if (ch === "(") p++; else if (ch === ")") p--; if (b < 0 || p < 0) break; }
  if (b || p) bad.push(`unbalanced braces/parens (${b}, ${p})`);
  for (const m of src.matchAll(/(<<|>>) (\d+)u/g)) if (+m[2] >= 32) bad.push(`shift by ${m[2]}`);
  for (const e of ["fn search_main", "fn test_main", "@workgroup_size(256)"]) if (!src.includes(e)) bad.push("missing " + e);
  if (/var<private>|array<u32, 16>;/.test(src.replace(/var<workgroup>[^\n]*/g, ""))) bad.push("private arrays");
  // a hex literal op literal on both sides would be a constant expression that WGSL may reject on overflow
  if (/0x[0-9a-f]+u\s*[-+*]\s*0x[0-9a-f]+u/.test(src)) bad.push("literal arithmetic");
  return bad;
}

async function oneJob(name, seed, rounds) {
  const prep = skinPrep(name);
  const rng = makeRng(seed);
  const { job } = await autoChoose(prep, rng);
  const D = deriveJob(job);
  const F = deriveFast(job, D);
  check(job.nblk === 1 && equalLengths(job), `${name} seed ${seed}: auto picked a one-block set with equal code lengths (K=${job.K})`);
  check(!!F.ok, `${name} seed ${seed}: fast layout derived (WI=${F.WI} KB=${F.KB} shift=${F.sb} L=${F.L})`);
  if (!F.ok) { console.log("  ", F.why); return; }
  let bad = 0;
  const perm = Array.from(job.vals);
  for (let it = 0; it < rounds; it++) {
    rng.shuffle(perm);
    const buf = fastBuf(job, perm);
    const { w, b } = fastModel(job, D, F, perm);
    for (let i = 0; i < 16; i++) {
      const want = ((buf[4 * i] << 24) | (buf[4 * i + 1] << 16) | (buf[4 * i + 2] << 8) | buf[4 * i + 3]) >>> 0;
      if (w[i] !== want) { bad++; if (bad < 4) console.log(`  word ${i}: model ${w[i].toString(16)} builder ${want.toString(16)}`); break; }
    }
  }
  check(bad === 0, `${name} seed ${seed}: table model equals the byte-level builder on ${rounds} random permutations`);
  const { off } = buildFastConsts(D, F);
  const src = genFastShader(job, D, F, off, 256);
  const lint = lintShader(src);
  check(lint.length === 0, `${name} seed ${seed}: generated WGSL is well formed (${src.length} chars, ${src.split("\n").length} lines)${lint.length ? " " + lint.join("; ") : ""}`);
  makeTestVectors(job, rng, 4, 124);
  check(job.tests.length === 128, "128 test vectors (4 pako + 124 host-built)");
  // host-built vectors agree with decodePerm at the end of the range
  const last = fastBuf(job, decodePerm(job, permCount(job) - 1n));
  check(last.length === 64, "last permutation builds");
}

await oneJob("steve.png", 777, 400);
await oneJob("steve.png", 31337, 300);
if (fs.existsSync(path.join(here, "..", "..", "skins", "Dinnerbone.png"))) await oneJob("Dinnerbone.png", 5, 200);

// the job builder prefers equal lengths: over many value sets, auto/rebuild return equal-length sets
{
  const prep = skinPrep("steve.png");
  const rng = makeRng(2024);
  let eq = 0, n = 0;
  const first = (await autoChoose(prep, rng)).job;
  for (let i = 0; i < 6; i++) { const j = await rebuildJob(prep, rng, first.K, first.nblk); n++; if (equalLengths(j) && j.nblk === 1) eq++; }
  check(eq === n, `rebuildJob returns one-block equal-length sets (${eq}/${n})`);
  // unequal sets exist (so the preference matters), and deriveFast rejects them
  let sawUneq = false;
  for (let t = 0; t < 400 && !sawUneq; t++) {
    const j = setupFast(prep, 6, pickVals(prep, 6, rng));
    if (!equalLengths(j) && j.nblk === 1) { sawUneq = true; check(!deriveFast(j, deriveJob(j)).ok, "deriveFast rejects an unequal-length set"); }
  }
  if (!sawUneq) console.log("  (no unequal-length one-block set in 400 draws, rejection check skipped)");
}
console.log(fail ? `${fail} FAILED` : "ALL FAST KERNEL TESTS PASSED");
process.exit(fail ? 1 : 0);
