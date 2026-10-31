// NameMC mode, CPU reference tests: known ids, midstate path, Python hashlib cross-check.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { decodePng, mojangPng } from "../js/png.js";
import { prepareNameMc, namemcIdHex, nmPixels, nmDigestFull, nmDigestMid, makeNmTests } from "../js/namemc.js";
import { makeRng } from "../js/job.js";
import { toHex } from "../js/hash.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..", "..");
let fail = 0;
const ok = (c, m) => { console.log((c ? "PASS " : "FAIL ") + m); if (!c) fail++; };
const load = (p) => decodePng(new Uint8Array(fs.readFileSync(p))).rgba;

// 1. NameMC id of classic Steve
const files = [path.join(repo, "skins", "steve.png")];
for (const f of files) {
  const id = namemcIdHex(load(f));
  ok(id === "12b92a9206470fe2", `NameMC id of ${path.relative(repo, f)} = ${id}`);
}
// the Mojang-side hidden RGB under alpha 0 must not matter
{
  const a = load(files[0]), b = Uint8Array.from(a);
  for (let p = 0; p < 4096; p++) if (b[p * 4 + 3] === 0) { b[p * 4] = 77; b[p * 4 + 1] = 5; b[p * 4 + 2] = 200; }
  ok(namemcIdHex(b) === namemcIdHex(a), "RGB under alpha 0 does not change the NameMC id");
}

// 2. random knob values: JS full / midstate / Python hashlib agree, for both layouts
const rng = makeRng(2024);
const cases = [], want = [];
for (const [name, unusedOnly] of [["steve.png", false], ["steve.png", true]]) {
  const prep = prepareNameMc(load(path.join(repo, "skins", name)), { unusedOnly });
  ok(prep.eligible, `${name} unusedOnly=${unusedOnly}: knobs ${JSON.stringify(prep.knobs)} nb=${prep.nb} fast=${prep.fast} preferred=${prep.isPreferred}`);
  for (let i = 0; i < 80; i++) {
    const c = rng.big(1n << 64n);
    const px = nmPixels(prep, c);
    const full = toHex(nmDigestFull(prep, c)), mid = toHex(nmDigestMid(prep, c));
    if (full !== mid || full.slice(0, 16) !== namemcIdHex(px)) { fail++; console.error("JS paths disagree", name, c); break; }
    cases.push(px); want.push(full.slice(0, 16));
  }
  makeNmTests(prep, rng, 96);
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "nm_"));
const all = new Uint8Array(cases.length * 16384);
cases.forEach((c, i) => all.set(c, i * 16384));
fs.writeFileSync(path.join(tmp, "c.bin"), all);
const r = spawnSync("py", [path.join(here, "namemc_ref.py"), path.join(tmp, "c.bin"), path.join(tmp, "o.txt")], { encoding: "utf8" });
const got = fs.readFileSync(path.join(tmp, "o.txt"), "utf8").trim().split(/\r?\n/);
const bad = want.filter((w, i) => w !== got[i]).length;
ok(r.status === 0 && bad === 0, `JS reference vs Python hashlib: ${want.length - bad}/${want.length} random knob values identical`);
// 3. the knob PNG round-trips and keeps the id
{
  const prep = prepareNameMc(load(files[0]));
  const px = nmPixels(prep, 0x123456789abcdefn);
  const back = decodePng(mojangPng(px)).rgba;
  ok(namemcIdHex(back) === namemcIdHex(px), "PNG round trip keeps the NameMC id");
}
fs.rmSync(tmp, { recursive: true, force: true });
process.exit(fail ? 1 : 0);
