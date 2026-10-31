// Skin model: UV masks, invisible ("knob") pixels, watermark v3. Ported from skinhash.py.
import { W, H } from "./png.js";

const BASE_PARTS = [
  [0, 0, 8, 8, 8], [16, 16, 8, 12, 4],
  [40, 16, 4, 12, 4], [40, 16, 3, 12, 4],
  [32, 48, 4, 12, 4], [32, 48, 3, 12, 4],
  [0, 16, 4, 12, 4], [16, 48, 4, 12, 4],
];
const OVERLAY_PARTS = [
  [32, 0, 8, 8, 8], [16, 32, 8, 12, 4],
  [40, 32, 4, 12, 4], [40, 32, 3, 12, 4],
  [48, 48, 4, 12, 4], [48, 48, 3, 12, 4],
  [0, 32, 4, 12, 4], [0, 48, 4, 12, 4],
];

export const WATERMARKS = {};
const wmBytes = (s, ver) => Uint8Array.from([...s].map((c) => c.charCodeAt(0)).concat([ver]));
WATERMARKS[3] = wmBytes("sokripon skinhash", 3);
WATERMARKS[2] = wmBytes("sokripon skinhash", 2);
WATERMARKS[1] = wmBytes("SKHAS", 1);
export const WATERMARK_VERSION = 3;
export const WATERMARK_SPOT = [0, 7];   // (0,7)-(5,7); v2 used (0,0)-(5,0), which Mine Little Pony reads

export function wmPixels(payload) {
  const p = Array.from(payload);
  while (p.length % 3) p.push(0);
  const px = [];
  for (let i = 0; i < p.length; i += 3) px.push([p[i], p[i + 1], p[i + 2]]);
  return px;
}

function boxPixels(set, [u, v, w, h, d]) {
  const rects = [
    [u + d, v, w, d], [u + d + w, v, w, d],
    [u, v + d, d, h], [u + d, v + d, w, h],
    [u + d + w, v + d, d, h], [u + 2 * d + w, v + d, w, h],
  ];
  for (const [x0, y0, rw, rh] of rects)
    for (let y = y0; y < y0 + rh; y++) for (let x = x0; x < x0 + rw; x++) set[y * W + x] = 1;
}

/** Pixels other software reads as data (index y*64+x). No search pixel and no watermark pixel may come from here.
 *  Same set as RESERVED in skinhash.py. */
export const RESERVED = (() => {
  const r = new Uint8Array(W * H);
  const rect = (x0, y0, x1, y1) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) r[y * W + x] = 1; };
  rect(0, 32, 3, 35);     // Ears: marker block x0-3, y32-35
  rect(52, 32, 55, 35);   // Ears: emissive palette x52-55, y32-35
  for (const [x, y] of [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [2, 2]]) r[y * W + x] = 1;   // Mine Little Pony triggers (CPM header = (0,0))
  rect(56, 16, 61, 18);   // Tails mod rows x56-61, y16-18
  boxPixels(r, [24, 0, 6, 6, 1]);   // vanilla deadmau5 ear box x24-37, y0-6
  return r;
})();

/** base[i]/over[i] = 1 if pixel i (y*64+x) is sampled by the base / overlay layer. */
export const MASKS = (() => {
  const base = new Uint8Array(W * H), over = new Uint8Array(W * H);
  for (const p of BASE_PARTS) boxPixels(base, p);
  for (const p of OVERLAY_PARTS) boxPixels(over, p);
  return { base, over };
})();

/** Pixels whose RGB never shows in game, latest in the file first. Each is [x, y]. */
export function knobPixels(rgba) {
  const { base, over } = MASKS;
  const knobs = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, a = rgba[i * 4 + 3];
    if (RESERVED[i]) continue;
    const unused = !base[i] && !over[i];
    if (a === 0 && (unused || over[i])) knobs.push([x, y]);
  }
  knobs.sort((p, q) => (q[1] - p[1]) || (q[0] - p[0]));
  return knobs;
}

function matchesAt(rgba, x, y, px) {
  for (let j = 0; j < px.length; j++) {
    const o = (y * W + x + j) * 4;
    if (rgba[o] !== px[j][0] || rgba[o + 1] !== px[j][1] || rgba[o + 2] !== px[j][2] || rgba[o + 3] !== 0) return false;
  }
  return true;
}

/** [version, x, y] of a known watermark, or null. */
export function findWatermark(rgba) {
  for (const ver of [3, 2, 1]) {
    const px = wmPixels(WATERMARKS[ver]);
    for (let y = 0; y < H; y++) for (let x = 0; x + px.length <= W; x++)
      if (matchesAt(rgba, x, y, px)) return [ver, x, y];
  }
  return null;
}

function watermarkSpot(rgba, n, avoid) {
  const { base, over } = MASKS;
  const av = new Set(avoid.map(([x, y]) => y * W + x));
  const free = (x, y) => { const i = y * W + x; return !base[i] && !over[i] && !RESERVED[i] && !av.has(i) && rgba[i * 4 + 3] === 0; };
  const run = (x, y) => { if (x + n > W) return false; for (let j = 0; j < n; j++) if (!free(x + j, y)) return false; return true; };
  if (run(...WATERMARK_SPOT)) return WATERMARK_SPOT;
  for (let y = 0; y < H; y++) for (let x = 0; x + n <= W; x++) if (run(x, y)) return [x, y];
  return null;
}

/** Write the watermark in place. Returns its pixels as [x, y] list ([] if there is no room). */
export function applyWatermark(rgba, avoid = []) {
  const px = wmPixels(WATERMARKS[WATERMARK_VERSION]);
  const found = findWatermark(rgba);
  if (found && found[0] === WATERMARK_VERSION) return px.map((_, j) => [found[1] + j, found[2]]);
  const spot = watermarkSpot(rgba, px.length, avoid);
  if (!spot) return [];
  const [x, y] = spot;
  px.forEach((rgb, j) => { const o = (y * W + x + j) * 4; rgba[o] = rgb[0]; rgba[o + 1] = rgb[1]; rgba[o + 2] = rgb[2]; rgba[o + 3] = 0; });
  return px.map((_, j) => [x + j, y]);
}

/** Is a change at pixel i visible in game, given the new alpha? */
export function isVisible(i, newAlpha) {
  return !!(MASKS.base[i] || (MASKS.over[i] && newAlpha !== 0));
}
