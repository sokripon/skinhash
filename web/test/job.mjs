// Job builder tests: eligibility, auto K choice, fast path vs full pako (self-test), test vectors.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodePng } from "../js/png.js";
import { prepareSkin, autoChoose, rebuildJob, makeRng, selfTest, makeTestVectors, decodePerm, fastFile, permCount } from "../js/job.js";
import { findWatermark } from "../js/model.js";

const here = path.dirname(fileURLToPath(import.meta.url));
let fail = 0;
for (const name of ["steve.png"]) {
  const rgba = decodePng(new Uint8Array(fs.readFileSync(path.join(here, "..", "..", "skins", name)))).rgba;
  const prep = prepareSkin(rgba);
  console.log(`${name}: eligible=${prep.eligible} invisible tail=${prep.kvis} watermark=${JSON.stringify(prep.wm[0])}..${JSON.stringify(prep.wm.at(-1))} found=${JSON.stringify(findWatermark(prep.rgba))}`);
  if (!prep.eligible) { console.log("  " + prep.reason); continue; }
  const rng = makeRng(777);
  const t0 = performance.now();
  const { job, K, nblk } = await autoChoose(prep, rng);
  console.log(`  auto: K=${K} nblk=${nblk} flen=${job.flen} V0=${job.V0} nvar=${job.nvar} (${(performance.now() - t0).toFixed(0)} ms)`);
  const t1 = performance.now();
  const N = selfTest(job, rng, 250);
  console.log(`  self-test: ${N} random permutations identical to full pako (${(performance.now() - t1).toFixed(0)} ms)`);
  makeTestVectors(job, rng, 24);
  // index mapping is a bijection on a sample, and distinct indices give distinct files
  const seen = new Set();
  for (let i = 0n; i < 48n; i++) seen.add(Buffer.from(fastFile(job, decodePerm(job, i)).file).toString("hex"));
  if (seen.size !== 48) { fail++; console.error("  index mapping collision"); }
  console.log(`  ${job.tests.length} test vectors, ${permCount(job)} permutations per value set`);
}

// rollover: a fresh value set with the same K / block count passes the self-test too
{
  const rgba = decodePng(new Uint8Array(fs.readFileSync(path.join(here, "..", "..", "skins", "steve.png")))).rgba;
  const prep = prepareSkin(rgba);
  const rng = makeRng(99);
  const a = (await autoChoose(prep, rng)).job;
  const b = await rebuildJob(prep, rng, a.K, a.nblk);
  const same = Buffer.from(a.vals).equals(Buffer.from(b.vals));
  selfTest(b, rng, 100); makeTestVectors(b, rng, 8);
  console.log(`rollover: new value set (differs=${!same}) K=${b.K} nblk=${b.nblk}, self-test OK`);
  if (same || b.nblk !== a.nblk) { fail++; console.error("rollover problem"); }
  // ineligible skin: last pixel opaque
  const bad = Uint8Array.from(rgba); bad[(64 * 64 - 1) * 4 + 3] = 255;
  const p2 = prepareSkin(bad);
  console.log(`ineligible check: eligible=${p2.eligible} (${p2.reason})`);
  if (p2.eligible) { fail++; console.error("opaque tail should be ineligible"); }
}
process.exit(fail ? 1 : 0);
