// NameMC mode, CPU side. Ported from namemcfinder.c / skinhash.py.
// NameMC id = first 8 bytes (16 hex) of SHA-256( be32(64) || be32(64) || RGBA row-major, RGB = 0 where alpha == 0 ).
// Knob: three pixels with alpha 1 whose RGB is a 64-bit counter:
//   kp[2] RGB = c & 0xFFFFFF, kp[1] RGB = (c >> 24) & 0xFFFFFF, kp[0] RGB = c >> 48   (kp in message order)
import { W, H } from "./png.js";
import { sha256, sha256Blocks, toHex, SHA_INIT } from "./hash.js";
import { MASKS, applyWatermark } from "./model.js";

export const NK = 3;
export const MSG_LEN = 8 + W * H * 4;   // 16392
const put32 = (a, o, v) => { a[o] = v >>> 24; a[o + 1] = (v >>> 16) & 255; a[o + 2] = (v >>> 8) & 255; a[o + 3] = v & 255; };

export function namemcMessage(rgba) {
  const m = new Uint8Array(MSG_LEN);
  put32(m, 0, W); put32(m, 4, H);
  for (let i = 0; i < W * H; i++) {
    const a = rgba[i * 4 + 3], o = 8 + i * 4;
    if (a) { m[o] = rgba[i * 4]; m[o + 1] = rgba[i * 4 + 1]; m[o + 2] = rgba[i * 4 + 2]; }
    m[o + 3] = a;
  }
  return m;
}
export const namemcIdHex = (rgba) => toHex(sha256(namemcMessage(rgba))).slice(0, 16);

const TAIL = Array.from({ length: 6 }, (_, i) => [W - 6 + i, H - 1]);   // pixels the Mojang fast mode permutes
const tailSet = new Set(TAIL.map(([x, y]) => y * W + x));

export function counterRgb(c) {
  return [Number(c >> 48n), Number((c >> 24n) & 0xFFFFFFn), Number(c & 0xFFFFFFn)];
}

/** orig: RGBA of the user's skin. Returns the NameMC job description. */
export function prepareNameMc(orig, { watermark = true, unusedOnly = false } = {}) {
  const rgba = Uint8Array.from(orig);
  const wm = watermark ? applyWatermark(rgba, TAIL) : [];
  const wmSet = new Set(wm.map(([x, y]) => y * W + x));
  const { base, over } = MASKS;
  const elig = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (rgba[i * 4 + 3] || base[i] || wmSet.has(i)) continue;
    if (over[i] && unusedOnly) continue;
    elig.push(i);
  }
  elig.sort((a, b) => b - a);
  const pref = elig.filter((i) => !tailSet.has(i));
  const pickFrom = pref.length ? pref : elig;
  const res = { orig, wm, wmSet, unusedOnly, elig: elig.length, eligible: elig.length >= NK && pickFrom.length >= NK };
  if (!res.eligible) { res.reason = `Only ${elig.length} usable pixel(s) for the NameMC knob (fully transparent overlay or unused pixels); need ${NK}.`; return res; }
  const pick = pickFrom.slice(0, NK).sort((a, b) => a - b);
  res.knobIdx = pick;
  res.knobs = pick.map((i) => [i % W, Math.floor(i / W)]);
  const preferred = [0, 1, 2].map((k) => (H - 1) * W + (W - 9 + k));
  res.isPreferred = pick.every((v, k) => v === preferred[k]);
  for (const i of pick) { rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = 0; rgba[i * 4 + 3] = 1; }
  res.rgba = rgba;
  const msg = namemcMessage(rgba);
  res.msg = msg;
  res.off = pick.map((i) => 8 + i * 4);
  const fb = Math.floor(res.off[0] / 64);
  const mid = Uint32Array.from(SHA_INIT);
  sha256Blocks(mid, msg, 0, fb);
  const rem = MSG_LEN - fb * 64, nb = Math.floor((rem + 9 + 63) / 64);
  const tail = new Uint8Array(nb * 64);
  tail.set(msg.subarray(fb * 64));
  tail[rem] = 0x80;
  const bits = MSG_LEN * 8;
  put32(tail, nb * 64 - 4, bits >>> 0);
  res.fb = fb; res.nb = nb; res.mid = mid; res.tail = tail;
  res.woff = res.off.map((o) => o - fb * 64);
  // fast layout: knob words 9,10,11 of the first of exactly 2 blocks
  res.fast = nb === 2 && res.woff.every((o, k) => o === 36 + 4 * k);
  return res;
}

/** RGBA of the skin with counter c written into the knob pixels. */
export function nmPixels(prep, c) {
  const rgba = Uint8Array.from(prep.rgba);
  counterRgb(c).forEach((v, k) => {
    const i = prep.knobIdx[k] * 4;
    rgba[i] = v >> 16; rgba[i + 1] = (v >> 8) & 255; rgba[i + 2] = v & 255; rgba[i + 3] = 1;
  });
  return rgba;
}

/** Full from-scratch digest of counter c (independent of the midstate path). */
export function nmDigestFull(prep, c) {
  const m = Uint8Array.from(prep.msg);
  counterRgb(c).forEach((v, k) => { const o = prep.off[k]; m[o] = v >> 16; m[o + 1] = (v >> 8) & 255; m[o + 2] = v & 255; });
  return sha256(m);
}

/** Digest via the midstate and the tail blocks only. */
export function nmDigestMid(prep, c) {
  const t = Uint8Array.from(prep.tail);
  counterRgb(c).forEach((v, k) => put32(t, prep.woff[k], ((v << 8) | 1) >>> 0));
  const s = Uint32Array.from(prep.mid);
  sha256Blocks(s, t, 0, prep.nb);
  const d = new Uint8Array(32);
  for (let i = 0; i < 8; i++) { d[4*i] = s[i] >>> 24; d[4*i+1] = (s[i] >>> 16) & 255; d[4*i+2] = (s[i] >>> 8) & 255; d[4*i+3] = s[i] & 255; }
  return d;
}

/** CPU test vectors for the GPU gate: counters and the first two digest words. */
export function makeNmTests(prep, rng, count = 96) {
  const edge = [0n, 1n, 1023n, 1024n, 0xFFFFFFn, 0x1000000n, 0xFFFFFFFFFFFFn, 0x1000000000000n, 0xFFFFFFFFFFFFFFFFn, 0x8000000000000000n, 0x00FF00FF00FF00FFn];
  const cs = edge.slice();
  while (cs.length < count) cs.push(rng.big(1n << 64n));
  return cs.map((c) => {
    const d = nmDigestFull(prep, c), d2 = nmDigestMid(prep, c);
    if (d.some((x, i) => x !== d2[i])) throw new Error("NameMC midstate digest differs from the full hash");
    const dv = new DataView(d.buffer);
    return { c, hi: dv.getUint32(0), lo: dv.getUint32(4) };
  });
}
