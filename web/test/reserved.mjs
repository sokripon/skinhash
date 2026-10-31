// RESERVED pixels (read as data by Ears, Mine Little Pony, Tails, deadmau5): no search or watermark pixel may come from them.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodePng, W, H } from "../js/png.js";
import { RESERVED, MASKS, knobPixels, applyWatermark, findWatermark, WATERMARK_VERSION } from "../js/model.js";
import { prepareSkin } from "../js/job.js";
import { prepareNameMc } from "../js/namemc.js";

const here = path.dirname(fileURLToPath(import.meta.url));
let fail = 0;
const check = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) fail++; };
const idx = ([x, y]) => y * W + x;

const steve = decodePng(new Uint8Array(fs.readFileSync(path.join(here, "..", "..", "skins", "steve.png")))).rgba;
let seed = 12345;
const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const noise = () => { const a = new Uint8Array(W * H * 4); for (let i = 0; i < W * H; i++) { a[i * 4] = rnd() * 256; a[i * 4 + 1] = rnd() * 256; a[i * 4 + 2] = rnd() * 256; a[i * 4 + 3] = rnd() < 0.6 ? 0 : 255; } return a; };
const opaqueExcept = (free) => { const a = new Uint8Array(W * H * 4).fill(255); for (const i of free) a[i * 4 + 3] = 0; return a; };
const resIdx = [...RESERVED.keys()].filter((i) => RESERVED[i]);
const wmOpaque = new Uint8Array(W * H * 4); for (let x = 0; x < 6; x++) wmOpaque[(7 * W + x) * 4 + 3] = 255;
const skins = {
  steve, "all transparent": new Uint8Array(W * H * 4), "only reserved clear": opaqueExcept(resIdx),
  "(0-5,7) opaque": wmOpaque, "noise 1": noise(), "noise 2": noise(),
};

check(resIdx.length === 154, `RESERVED has ${resIdx.length} pixels (expected 154)`);
check(RESERVED[0] && RESERVED[2 * W + 2] && RESERVED[32 * W] && RESERVED[35 * W + 3] && RESERVED[32 * W + 52] && RESERVED[16 * W + 56] && RESERVED[18 * W + 61] && RESERVED[W + 24] && RESERVED[6 * W + 37]
  && !RESERVED[32 * W + 4] && !RESERVED[16 * W + 62] && !RESERVED[7 * W], "RESERVED edges");
for (let x = 0; x < 6; x++) if (RESERVED[7 * W + x]) check(false, "watermark spot reserved");

for (const [name, rgba] of Object.entries(skins)) {
  check(knobPixels(rgba).every((p) => !RESERVED[idx(p)]), `${name}: knobPixels avoids RESERVED`);
  for (const avoid of [[], Array.from({ length: 6 }, (_, i) => [W - 6 + i, H - 1])]) {
    const r = Uint8Array.from(rgba), wm = applyWatermark(r, avoid);
    check(wm.every((p) => !RESERVED[idx(p)] && !MASKS.base[idx(p)] && !MASKS.over[idx(p)]), `${name}: watermark ${JSON.stringify(wm[0])}..${JSON.stringify(wm.at(-1))} avoids RESERVED`);
    if (wm.length) check(findWatermark(r)?.[0] === WATERMARK_VERSION, `${name}: detected as v${WATERMARK_VERSION}`);
    if (name === "steve" || name === "all transparent") check(JSON.stringify(wm) === JSON.stringify([0, 1, 2, 3, 4, 5].map((x) => [x, 7])), `${name}: default watermark spot is (0-5,7)`);
  }
  const p = prepareSkin(rgba);
  check(p.knobs.every((q) => !RESERVED[idx(q)]) && p.wm.every((q) => !RESERVED[idx(q)]), `${name}: prepareSkin knobs and watermark avoid RESERVED`);
  for (const unusedOnly of [false, true]) {
    const n = prepareNameMc(rgba, { unusedOnly });
    if (n.eligible) check(n.knobIdx.every((i) => !RESERVED[i]) && n.wm.every((q) => !RESERVED[idx(q)]), `${name}: NameMC knobs ${JSON.stringify(n.knobs)} (unusedOnly=${unusedOnly}) avoid RESERVED`);
    else console.log(`  ${name}: NameMC unusedOnly=${unusedOnly} not eligible (${n.reason})`);
  }
}
// Mojang fast-mode tail and NameMC default knobs are outside RESERVED
check([58, 59, 60, 61, 62, 63, 55, 56, 57].every((x) => !RESERVED[63 * W + x]), "row-63 search pixels not reserved");
console.log(fail ? `${fail} FAILURE(S)` : "ALL PASSED");
process.exit(fail ? 1 : 0);
