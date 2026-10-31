// Fast-mode job builder, ported from fastfinder.c (setup_fast, parse_deflate, auto_choose, self_test,
// dump_job test vectors). No DOM; runs in the browser and in Node.
import pako from "../vendor/pako.esm.mjs";
import { W, H, ROW, RAW_LEN, DEFLATE_OPTS, rawScanlines, pngFromRaw } from "./png.js";
import { sha256, sha256Blocks, stateToDigest, SHA_INIT, crc32 } from "./hash.js";
import { knobPixels, applyWatermark, RESERVED } from "./model.js";

export const MIN_K = 5;     // fewer than 5 pixels (15 values) gives too few orderings to be worth it
export const MAX_K = 6;
export const RIN = 4;       // inner (cached) positions, = GPU_R in the native tool
export const NPERM = 24;
export const AUTO_TRIES = 400;

export class Unsupported extends Error {}

// ---------- RNG ----------
export function makeRng(seed) {
  if (seed === undefined) {
    const a = new Uint32Array(1);
    (globalThis.crypto || (globalThis.crypto = {})).getRandomValues?.(a);
    seed = a[0] || (Date.now() >>> 0);
  }
  let s = seed >>> 0;
  const next = () => { // mulberry32
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0);
  };
  const below = (n) => Math.floor((next() / 4294967296) * n);
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = below(i + 1); const t = a[i]; a[i] = a[j]; a[j] = t; } return a; };
  const big = (limit) => { let v = 0n; for (let i = 0; i < 4; i++) v = (v << 32n) | BigInt(next()); return v % limit; };
  return { next, below, shuffle, big };
}

// ---------- skin preparation ----------
/** orig: RGBA Uint8Array(64*64*4). Applies the watermark and finds the invisible tail. */
export function prepareSkin(orig, { watermark = true, kmax = MAX_K } = {}) {
  const rgba = Uint8Array.from(orig);
  const tail = [];
  for (let i = 0; i < kmax; i++) tail.push([W - kmax + i, H - 1]);
  const wm = watermark ? applyWatermark(rgba, tail) : [];
  const wmSet = new Set(wm.map(([x, y]) => y * W + x));
  const knobs = knobPixels(rgba).filter(([x, y]) => !wmSet.has(y * W + x));
  const kset = new Set(knobs.map(([x, y]) => y * W + x));
  let kvis = 0;
  while (kvis < kmax && kset.has((H - 1) * W + (W - 1 - kvis))) kvis++;
  for (const [x, y] of tail) if (RESERVED[y * W + x]) kvis = 0;   // never permute a pixel other software reads
  const kmaxUsable = Math.min(kvis, kmax);
  const eligible = kmaxUsable >= MIN_K;
  let reason = null;
  if (!eligible) {
    const bad = [];
    for (let i = 0; i < MIN_K; i++) { const x = W - MIN_K + i; if (!kset.has((H - 1) * W + x)) bad.push(`(${x},${H - 1})`); }
    reason = `Fast mode needs the last ${MIN_K} pixels of the image (bottom row, right end) to be invisible: alpha 0 on the overlay layer or in an unused area. These are not: ${bad.join(" ")}.`;
  }
  return { orig, rgba, raw: rawScanlines(rgba), wm, knobs, kvis, kmaxUsable, eligible, reason };
}

export function knobLayout(K) {
  const knob = [], alphaOff = [];
  for (let i = 0; i < K; i++) {
    const x = W - K + i, base = (H - 1) * ROW + 1 + x * 4;
    for (let j = 0; j < 3; j++) knob.push(base + j);
    alphaOff.push(base + 3);
  }
  return { knob, alphaOff, split: knob[0] };
}

// ---------- reference encoder ----------
export function pngBytesFor(raw) { return pngFromRaw(raw); }

// ---------- deflate parser (puff style) ----------
const LBASE = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
const LEXT = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
const DBASE = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
const DEXT = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];

function construct(len, n) {
  const count = new Array(16).fill(0), symbol = new Array(n).fill(0), offs = new Array(16).fill(0);
  for (let i = 0; i < n; i++) count[len[i]]++;
  for (let l = 1; l < 15; l++) offs[l + 1] = offs[l] + count[l];
  for (let i = 0; i < n; i++) if (len[i]) symbol[offs[len[i]]++] = i;
  return { count, symbol };
}

function parseDeflate(z, start, len, raw, split) {
  const s = { pos: start * 8, end: (start + len) * 8 };
  const getbits = (n) => {
    let v = 0;
    for (let i = 0; i < n; i++) {
      if (s.pos >= s.end) return -1;
      v |= ((z[s.pos >> 3] >> (s.pos & 7)) & 1) << i;
      s.pos++;
    }
    return v;
  };
  const decode = (h) => {
    let code = 0, first = 0, index = 0;
    for (let l = 1; l <= 15; l++) {
      const b = getbits(1);
      if (b < 0) return -1;
      code |= b;
      const count = h.count[l];
      if (code - count < first) return h.symbol[index + (code - first)];
      index += count; first += count; first <<= 1; code <<= 1;
    }
    return -1;
  };
  const out = new Uint8Array(RAW_LEN + 400);
  let op = 0;
  const lens = new Array(320).fill(0);
  const final = getbits(1), type = getbits(2);
  if (final !== 1) throw new Unsupported("first deflate block is not final (multiple blocks)");
  let lc, dc;
  const litlen = new Array(288).fill(0);
  if (type === 1) {
    for (let i = 0; i < 288; i++) lens[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
    lc = construct(lens, 288);
    dc = construct(new Array(30).fill(5), 30);
    for (let i = 0; i < 288; i++) litlen[i] = lens[i];
  } else if (type === 2) {
    const order = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];
    const nlen = getbits(5) + 257, ndist = getbits(5) + 1, ncode = getbits(4) + 4;
    if (nlen > 286 || ndist > 30) throw new Unsupported("bad dynamic header counts");
    const cl = new Array(19).fill(0);
    for (let i = 0; i < ncode; i++) cl[order[i]] = getbits(3);
    const hc = construct(cl, 19);
    let idx = 0;
    while (idx < nlen + ndist) {
      const sym = decode(hc);
      if (sym < 0) throw new Unsupported("bad code-length code");
      if (sym < 16) lens[idx++] = sym;
      else {
        let l = 0, rep;
        if (sym === 16) { if (!idx) throw new Unsupported("repeat with no previous"); l = lens[idx - 1]; rep = 3 + getbits(2); }
        else if (sym === 17) rep = 3 + getbits(3);
        else rep = 11 + getbits(7);
        if (idx + rep > nlen + ndist) throw new Unsupported("too many code lengths");
        while (rep--) lens[idx++] = l;
      }
    }
    for (let i = 0; i < nlen; i++) litlen[i] = lens[i];
    lc = construct(lens.slice(0, nlen), nlen);
    dc = construct(lens.slice(nlen, nlen + ndist), ndist);
  } else throw new Unsupported(`stored/invalid block type ${type}`);

  let T_bits = -1, ntail = 0, E_bits = 0;
  for (;;) {
    const st = s.pos;
    const sym = decode(lc);
    if (sym < 0) throw new Unsupported("bad literal/length code");
    if (op === split && T_bits === -1) T_bits = st;
    if (op > split && T_bits === -1) throw new Unsupported("a match straddles the first knob byte");
    if (sym === 256) { E_bits = s.pos; break; }
    if (sym < 256) {
      if (op >= RAW_LEN) throw new Unsupported("output overflow");
      out[op++] = sym;
      if (T_bits !== -1) ntail++;
    } else {
      if (T_bits !== -1) throw new Unsupported("match symbol inside the tail");
      const ls = sym - 257;
      if (ls >= 29) throw new Unsupported("bad length symbol");
      const l = LBASE[ls] + getbits(LEXT[ls]);
      const ds = decode(dc);
      if (ds < 0 || ds >= 30) throw new Unsupported("bad distance symbol");
      const dist = DBASE[ds] + getbits(DEXT[ds]);
      if (dist > op || op + l > RAW_LEN) throw new Unsupported("bad match");
      for (let i = 0; i < l; i++, op++) out[op] = out[op - dist];
    }
  }
  if (op !== RAW_LEN) throw new Unsupported("parser did not reproduce the raw data");
  for (let i = 0; i < RAW_LEN; i++) if (out[i] !== raw[i]) throw new Unsupported("parser did not reproduce the raw data");
  if (T_bits === -1) throw new Unsupported("no symbol starts at the first knob byte");
  if (ntail !== RAW_LEN - split) throw new Unsupported("tail is not all literals");
  if (((E_bits - start * 8 + 7) >> 3) + 4 !== len) throw new Unsupported("trailing data after the block");
  return { litlen, T_bits: T_bits - start * 8, E_bits: E_bits - start * 8, ntail };
}

// ---------- fast path setup ----------
/** Choose 3K byte values absent from the raw data (knob bytes excluded). */
export function pickVals(prep, K, rng) {
  const { knob } = knobLayout(K);
  const isKnob = new Uint8Array(RAW_LEN);
  for (const k of knob) isKnob[k] = 1;
  const seen = new Uint8Array(256);
  for (let i = 0; i < RAW_LEN; i++) if (!isKnob[i]) seen[prep.raw[i]] = 1;
  const unused = [];
  for (let v = 0; v < 256; v++) if (!seen[v]) unused.push(v);
  if (unused.length < 3 * K) throw new Unsupported(`only ${unused.length} byte values are unused in this skin, need ${3 * K}`);
  rng.shuffle(unused);
  return Uint8Array.from(unused.slice(0, 3 * K));
}

export function setupFast(prep, K, vals) {
  const { knob, alphaOff, split } = knobLayout(K);
  const n = 3 * K, m = Math.min(n, 20);
  const raw = Uint8Array.from(prep.raw);
  knob.forEach((o, i) => { raw[o] = vals[i]; });
  const file = pngFromRaw(raw);
  const flen = file.length;
  const zlen = ((file[33] << 24) | (file[34] << 16) | (file[35] << 8) | file[36]) >>> 0;
  if (file[41] !== 0x78) throw new Unsupported("unexpected zlib header");
  const P = parseDeflate(file, 43, zlen - 2, raw, split);
  // canonical codes, bit-reversed for LSB-first output
  const blc = new Array(16).fill(0);
  for (let i = 0; i < 288; i++) blc[P.litlen[i]]++;
  blc[0] = 0;
  const next = new Array(16).fill(0);
  let code = 0;
  for (let b = 1; b < 16; b++) { code = (code + blc[b - 1]) << 1; next[b] = code; }
  const tab = new Uint32Array(257);
  for (let i = 0; i < 257; i++) {
    const l = P.litlen[i];
    if (!l) { tab[i] = 0; continue; }
    const c = next[l]++;
    let r = 0;
    for (let k = 0; k < l; k++) r |= ((c >> k) & 1) << (l - 1 - k);
    tab[i] = (r | (l << 16)) >>> 0;
  }
  const eob = tab[256];
  const alpha = new Uint8Array(K);
  for (let i = 0; i < K; i++) {
    alpha[i] = prep.raw[alphaOff[i]];
    if (!tab[alpha[i]]) throw new Unsupported("alpha literal has no code");
  }
  const Tfile = 43 * 8 + P.T_bits, Efile = 43 * 8 + P.E_bits;
  const vb = Tfile >> 3;
  const rem = Tfile & 7;
  const firstbits = file[vb] & ((1 << rem) - 1);
  const paddedEnd = (Efile + 7) >> 3;
  const nvar = (paddedEnd - vb) + 8;
  if (paddedEnd + 8 + 12 !== flen) throw new Unsupported("file layout mismatch");
  const crc0 = crc32(crc32(0, file, 37, 41), file, 41, vb);
  let a = 1, b = 0;
  for (let i = 0; i < split; i++) { a += raw[i]; b += a; }
  const adl_a = a % 65521, adl_b = b % 65521;
  let a2 = adl_a, b2 = adl_b;
  for (let i = split; i < RAW_LEN; i++) { a2 += raw[i]; b2 += a2; }
  a2 %= 65521; b2 %= 65521;
  if (file[paddedEnd] !== (b2 >> 8) || file[paddedEnd + 3] !== (a2 & 255)) throw new Unsupported("adler position mismatch");

  const B0 = Math.floor(vb / 64) * 64, V0 = vb - B0;
  const total = Math.floor((flen + 9 + 63) / 64) * 64;
  const nblk = (total - B0) / 64;
  const mid = Uint32Array.from(SHA_INIT);
  sha256Blocks(mid, file, 0, B0 / 64);
  const tmpl = new Uint8Array(nblk * 64);
  tmpl.set(file.subarray(B0, flen));
  tmpl[flen - B0] = 0x80;
  const bits = flen * 8;
  const dv = new DataView(tmpl.buffer);
  dv.setUint32(nblk * 64 - 8, Math.floor(bits / 4294967296));
  dv.setUint32(nblk * 64 - 4, bits >>> 0);
  return { K, n, m, knob, alphaOff, split, vals: Uint8Array.from(vals), raw: prep.raw, tab, eob, alpha, flen, B0, V0, nvar, nblk, rem,
    firstbits, crc0, adl_a, adl_b, mid, tmpl, ref: file, ntail: P.ntail, zeros: 0, pats: [], tests: [] };
}

/** Variable region (tail literal bits + Adler + CRC) for a permutation of the n values. */
export function fastBuf(job, perm) {
  const buf = Uint8Array.from(job.tmpl);
  let o = job.V0, acc = job.firstbits, nb = job.rem;
  let a = job.adl_a, b = job.adl_b;
  const put = (e) => { acc |= (e & 0xFFFF) << nb; nb += e >>> 16; while (nb >= 8) { buf[o++] = acc & 255; acc >>>= 8; nb -= 8; } };
  for (let i = 0; i < job.K; i++) {
    for (let j = 0; j < 3; j++) { const v = perm[3 * i + j]; put(job.tab[v]); a += v; b += a; }
    const al = job.alpha[i]; put(job.tab[al]); a += al; b += a;
  }
  put(job.eob);
  if (nb > 0) buf[o++] = acc & 255;
  a %= 65521; b %= 65521;
  buf[o++] = b >> 8; buf[o++] = b & 255; buf[o++] = a >> 8; buf[o++] = a & 255;
  const c = crc32(job.crc0, buf, job.V0, o);
  buf[o++] = c >>> 24; buf[o++] = (c >>> 16) & 255; buf[o++] = (c >>> 8) & 255; buf[o++] = c & 255;
  if (o - job.V0 !== job.nvar) throw new Error(`variable region size ${o - job.V0} != ${job.nvar}`);
  return buf;
}

export function fastFile(job, perm) {
  const buf = fastBuf(job, perm);
  const f = new Uint8Array(job.flen);
  f.set(job.ref.subarray(0, job.B0));
  f.set(buf.subarray(0, job.flen - job.B0), job.B0);
  return { file: f, buf };
}

/** SHA-256 digest via the midstate and the variable blocks only. */
export function fastDigest(job, buf) {
  const s = Uint32Array.from(job.mid);
  sha256Blocks(s, buf, 0, job.nblk);
  return stateToDigest(s);
}

export function permRaw(job, perm) {
  const raw = Uint8Array.from(job.raw);
  job.knob.forEach((o, i) => { raw[o] = perm[i]; });
  return raw;
}

// ---------- index <-> permutation ----------
export const factBig = (r) => { let f = 1n; for (let i = 2n; i <= BigInt(r); i++) f *= i; return f; };
export const permCount = (job) => factBig(job.m);

export function decodePerm(job, idx) {
  const m = job.m, R = RIN;
  let o = idx / 24n, t = Number(idx % 24n);
  const rem = Array.from(job.vals.slice(0, m));
  const perm = new Uint8Array(job.n);
  for (let i = 0; i < m - R; i++) {
    const r = BigInt(m - i);
    const d = Number(o % r); o /= r;
    perm[i] = rem[d]; rem.splice(d, 1);
  }
  const avail = [0, 1, 2, 3];
  for (let q = 0; q < R; q++) {
    const f = [1, 1, 2, 6][R - 1 - q];
    const d = Math.floor(t / f); t %= f;
    perm[m - R + q] = rem[avail[d]]; avail.splice(d, 1);
  }
  for (let i = m; i < job.n; i++) perm[i] = job.vals[i];
  return perm;
}

/** Lehmer digits of the first m-RIN positions of outer index o (BigInt). */
export function outerDigits(job, o) {
  const d = new Uint32Array(16);
  for (let i = 0; i < job.m - RIN; i++) { const r = BigInt(job.m - i); d[i] = Number(o % r); o /= r; }
  return d;
}

// ---------- checks ----------
/** N random permutations built via the fast path must equal full pako output byte for byte. */
export function selfTest(job, rng, N = 250) {
  const perm = Array.from(job.vals);
  for (let it = 0; it < N; it++) {
    rng.shuffle(perm);
    const { file, buf } = fastFile(job, perm);
    const full = pngFromRaw(permRaw(job, perm));
    if (full.length !== file.length || full.some((x, i) => x !== file[i])) {
      const at = full.findIndex((x, i) => x !== file[i]);
      throw new Error(`fast path differs from pako (self-test #${it}, first diff at byte ${at}, var region starts at ${job.B0 + job.V0})`);
    }
    const d1 = fastDigest(job, buf), d2 = sha256(full);
    if (d1.some((x, i) => x !== d2[i])) throw new Error("SHA midstate mismatch in self-test");
  }
  return N;
}

/** CPU test vectors for the GPU gate: idx (BigInt) and SHA-256 of the full PNG. */
export function makeTestVectors(job, rng, count = 24) {
  const mf = permCount(job);
  const tests = [];
  for (let t = 0; t < count; t++) {
    const idx = t === 0 ? 0n : t === 1 ? mf - 1n : rng.big(mf);
    const perm = decodePerm(job, idx);
    const { file, buf } = fastFile(job, perm);
    const full = pngFromRaw(permRaw(job, perm));
    if (full.some((x, i) => x !== file[i])) throw new Error("test vector: fast path differs from pako");
    const sha = sha256(full);
    const d2 = fastDigest(job, buf);
    if (sha.some((x, i) => x !== d2[i])) throw new Error("test vector: midstate digest differs");
    tests.push({ idx, sha });
  }
  job.tests = tests;
  return tests;
}

// ---------- auto choice of K and value set ----------
const tick = () => new Promise((r) => setTimeout(r, 0));

/** Largest K <= kmax whose variable region plus SHA padding fits one block; otherwise the best 2-block config. */
export async function autoChoose(prep, rng, { kmax = prep.kmaxUsable, onProgress } = {}) {
  let fb = null, lastErr = null;
  for (let K = kmax; K >= MIN_K; K--) {
    let ok = 0;
    for (let t = 0; t < AUTO_TRIES; t++) {
      let job;
      try { job = setupFast(prep, K, pickVals(prep, K, rng)); ok++; } catch (e) { if (!(e instanceof Unsupported)) throw e; lastErr = e.message; if (t === 0 && /byte values are unused/.test(e.message)) break; continue; }
      if (job.nblk === 1) return { job, K, nblk: 1 };
      if (!fb || (K === fb.K && job.nblk < fb.nblk)) fb = { job, K, nblk: job.nblk };
      if (t % 20 === 19) { onProgress?.(`searching a one-block layout (K=${K}, try ${t + 1}/${AUTO_TRIES})`); await tick(); }
    }
    if (fb && K <= fb.K - 2) break;
  }
  if (!fb) throw new Unsupported(`no usable pixel count / value set found${lastErr ? " (" + lastErr + ")" : ""}`);
  return fb;
}

/** New random value set with the same K, preferring the same block count (used at rollover). */
export async function rebuildJob(prep, rng, K, wantBlk) {
  let best = null;
  for (let t = 0; t < AUTO_TRIES * 4; t++) {
    let job;
    try { job = setupFast(prep, K, pickVals(prep, K, rng)); } catch (e) { if (e instanceof Unsupported) continue; throw e; }
    if (job.nblk <= wantBlk) return job;
    if (!best || job.nblk < best.nblk) best = job;
    if (t % 50 === 49) await tick();
  }
  if (!best) throw new Unsupported("could not build a new value set");
  return best;
}
