// Search targets: trimmed-id prefixes, suffixes, contains, leading zeros. Same semantics as the native tool:
// a prefix matches the id with leading zeros dropped (as in the URL), so a prefix may not start with 0.
// "Contains" is a hex string anywhere in the id: in the trimmed id for Mojang (digits that only match inside the
// dropped zeros do not count), in the 16 digits for NameMC.
// "Match" targets are hex patterns (patterns.js), matched on the full 64-digit Mojang hash or the 16-digit NameMC id.
import { parseLine } from "./patterns.js";
export const MAX_HEX = 16;
export const MAXCONT = 32;   // sliding entries the GPU kernels take: contains patterns and sliding pattern prefilters
export const MAXANCH = 32;   // anchored pattern prefilters baked into the Mojang shaders

export function parseHexList(text, kind, allowLeadingZero = false) {
  const out = [], errors = [];
  for (const raw of text.split(/[\s,;]+/).filter(Boolean)) {
    const s = raw.toLowerCase();
    if (!/^[0-9a-f]+$/.test(s)) errors.push(`"${raw}" is not hexadecimal (0-9, a-f).`);
    else if (s.length > MAX_HEX) errors.push(`"${raw}" is longer than ${MAX_HEX} digits.`);
    else if (kind === "prefix" && s[0] === "0" && !allowLeadingZero) errors.push(`Prefix "${raw}" starts with 0. The link drops leading zeros, so a prefix can't start with 0; use "minimum leading zeros" instead.`);
    else if (!out.includes(s)) out.push(s);
  }
  return { list: out, errors };
}

/** Targets: {prefixes:[], suffixes:[], contains:[], patterns:[Pattern], stats:[StatLine], zeros:Number} -> list of
 *  {kind, text}. A stat line is a normal target (kind "stat"); a record line ({max}/{min}, kind "record") is not: it
 *  has a best value (t.best, null until the first record) and is never "found". */
export function targetList({ prefixes, suffixes, contains = [], patterns = [], stats = [], zeros }) {
  const t = [];
  prefixes.forEach((p) => t.push({ kind: "prefix", text: p }));
  suffixes.forEach((s) => t.push({ kind: "suffix", text: s }));
  contains.forEach((c) => t.push({ kind: "contains", text: c }));
  patterns.forEach((p) => t.push({ kind: "match", text: p.text, pattern: p }));
  stats.forEach((l) => t.push({ kind: l.record ? "record" : "stat", text: l.text, pattern: l, ...(l.record ? { best: null } : {}) }));
  if (zeros > 0) t.push({ kind: "zeros", text: String(zeros) });
  return t;
}

/** The Pattern of a match target for an id of `width` digits (64 Mojang, 16 NameMC). */
const patCache = new Map();
export function patternOf(t, width) {
  if (t.pattern && t.pattern.width === width) return t.pattern;
  const k = `${width}:${t.text}`;
  if (!patCache.has(k)) patCache.set(k, parseLine(t.text, width));
  return patCache.get(k);
}
export const targetKey = (t) => `${t.kind}:${t.text}`;
export const isRecord = (t) => t.kind === "record";
/** The stat lines (threshold and record) of a target list for one width, in target order: what the GPU kernels bake in. */
export const statLinesOf = (targets, width) => targets.filter((t) => t.kind === "stat" || t.kind === "record").map((t) => patternOf(t, width));
/** The record lines of a target list (kernel order). */
export const recordTargets = (targets) => targets.filter(isRecord);

/** GPU pattern (64-bit value/mask split into u32 halves). */
export function gpuPattern(t) {
  const n = t.text.length;
  const v = BigInt("0x" + t.text);
  const field = (1n << BigInt(4 * n)) - 1n;
  const shift = t.kind === "prefix" ? BigInt(4 * (16 - n)) : 0n;
  const val = v << shift, mask = field << shift;
  const lo = (x) => Number(x & 0xFFFFFFFFn) >>> 0, hi = (x) => Number(x >> 32n) >>> 0;
  return { vh: hi(val), vl: lo(val), mh: hi(mask), ml: lo(mask), suffix: t.kind === "suffix" };
}

/** GPU contains pattern: digits 1-8 and 9-16 left-aligned in two u32 words with their masks, lim = last offset
 *  (in digits) at which it fits in `width` digits (64 for the Mojang digest, 16 for the NameMC id), lo = first
 *  offset (bits 0-15) + bit 16 "after the leading zeros" (Mojang). */
export function gpuContains(t, width) {
  const n = t.text.length, d = (k) => parseInt(t.text[k], 16);
  const p = [0, 0], m = [0, 0];
  for (let k = 0; k < n; k++) { const w = k < 8 ? 0 : 1, sh = 28 - 4 * (k & 7); p[w] = (p[w] | (d(k) << sh)) >>> 0; m[w] = (m[w] | (0xF << sh)) >>> 0; }
  return { p1: p[0], m1: m[0], p2: p[1], m2: m[1], lim: width - n, lo: width === 64 ? 1 << 16 : 0, n };
}

/** Sliding prefilter of a match target, in the contains table format. */
export function gpuSliding(f) {
  return { p1: f.p1, m1: f.m1, p2: f.p2, m2: f.m2, lim: f.lim, lo: (f.lo | (f.usezl ? 1 << 16 : 0)) >>> 0, n: 0 };
}

/** Anchored match targets of a Mojang target list: [{key, mw, vw}] (baked into the shader in this order). */
export function anchoredTargets(targets) {
  return targets.filter((t) => t.kind === "match" && patternOf(t, 64).prefilter().kind === "anchored")
    .map((t) => ({ key: targetKey(t), ...patternOf(t, 64).words() }));
}
/** Does the kernel need the sliding (contains) check? */
export const needsSliding = (targets, width) => targets.some((t) => t.kind === "contains" || (t.kind === "match" && patternOf(t, width).prefilter().kind === "sliding"));

export function gpuTargets(targets) {
  return {
    pats: targets.filter((t) => t.kind === "prefix" || t.kind === "suffix").map(gpuPattern),
    cont: [...targets.filter((t) => t.kind === "contains").map((t) => gpuContains(t, 64)),
      ...targets.filter((t) => t.kind === "match" && patternOf(t, 64).prefilter().kind === "sliding").map((t) => gpuSliding(patternOf(t, 64).prefilter()))],
    anchKeys: targets.filter((t) => t.kind === "match").map(targetKey),
    statKeys: targets.filter((t) => t.kind === "stat").map(targetKey),
    zeros: targets.find((t) => t.kind === "zeros") ? Number(targets.find((t) => t.kind === "zeros").text) : 0,
  };
}

/** Which of the given targets does this digest (64 hex chars) satisfy? */
export function matchTargets(hex, targets, fixed = false) {
  // fixed (NameMC): hex is the 16-digit id, matched as is. Mojang: prefixes match the id without leading zeros.
  const id = fixed ? hex : (hex.replace(/^0+/, "") || "0");
  const zeros = hex.length - (hex.replace(/^0+/, "") || "0").length;
  return targets.filter((t) =>
    t.kind === "prefix" ? id.startsWith(t.text) :
    t.kind === "suffix" ? hex.endsWith(t.text) :
    t.kind === "contains" ? id.includes(t.text) :
    t.kind === "match" || t.kind === "stat" || t.kind === "record" ? patternOf(t, fixed ? 16 : 64).match(hex) :
    zeros >= Number(t.text));
}

/** Rough expected number of candidates for one hit. */
export function expectedTries(t, fixed = false) {
  if (t.kind === "zeros") return Math.pow(16, Number(t.text));
  // contains: (L - n + 1) places of 16^-n each; L = 16 for NameMC, about 60 digits for a Mojang link
  if (t.kind === "contains") return Math.pow(16, t.text.length) / Math.max(1, (fixed ? 16 : 60) - t.text.length + 1);
  if (t.kind === "match" || t.kind === "stat") { const p = patternOf(t, fixed ? 16 : 64).probability(); return p > 0 ? 1 / p : Infinity; }
  if (t.kind === "record") { const p = patternOf(t, fixed ? 16 : 64).probability(t.best ?? null); return p > 0 ? 1 / p : Infinity; }
  return Math.pow(16, t.text.length) * (t.kind === "prefix" && !fixed ? 16 / 15 : 1);
}

/** NameMC kernel targets: patterns on the 64-bit id plus a zeros threshold on the id words. */
export function gpuNmTargets(targets) {
  const z = targets.find((t) => t.kind === "zeros");
  let zmode = 0, zthr = 0;
  if (z) { const n = Number(z.text); if (n <= 8) { zmode = 1; zthr = Math.pow(2, 32 - 4 * n); } else { zmode = 2; zthr = Math.pow(2, 64 - 4 * n); } }
  const nmMatch = targets.filter((t) => t.kind === "match").map((t) => patternOf(t, 16));
  const anch = nmMatch.filter((p) => p.prefilter().kind === "anchored").map((p) => {
    const { mw, vw } = p.words();
    return { vh: vw[0], vl: vw[1], mh: mw[0], ml: mw[1], suffix: false };
  });
  return { statKeys: targets.filter((t) => t.kind === "stat").map(targetKey),
    pats: [...targets.filter((t) => t.kind === "prefix" || t.kind === "suffix").map(gpuPattern), ...anch],
    cont: [...targets.filter((t) => t.kind === "contains").map((t) => gpuContains(t, 16)),
      ...nmMatch.filter((p) => p.prefilter().kind === "sliding").map((p) => gpuSliding(p.prefilter()))], zmode, zthr };
}
