// Hex patterns (the "Patterns" box, native -m): the same rules as patterns.py (the reference) and pattern.h.
//   hex digits, ^ start, $ end, ? any digit, ?d 0-9, ?l a-f, [0-3] [05af] classes, {n} repeat, * gap
// Mojang: the full 64-digit hash; ^ is its start (^0{12} = 12 leading zeros), a pattern without ^ must lie in the
// link id (after the leading zeros). NameMC: the 16-digit id.
// Each pattern compiles to one GPU prefilter (anchored nibble masks, or one sliding window of up to 8 + 8 tokens);
// every hit that passes it is re-checked with match() on the CPU.
// Lines in the reserved stat syntax (zeros >= 24, palindrome best, run(f) >= 10, ^cafe & zeros >= 10) are rejected
// with "not supported yet".

export const HEXDIGITS = "0123456789abcdef";
export const ALL = 0xFFFF, DIGITS = 0x03FF, LETTERS = 0xFC00;
export const MAX_PATTERNS = 32;   // per group in the browser (the GPU tables)
export const WARN_ODDS = 1e-7;

export class PatternError extends Error {}

const isHex = (c) => HEXDIGITS.includes(c);
const isWordChar = (c) => /[0-9A-Za-z_]/.test(c);

/** Tokens: {kind, value, pos}. kinds: lit any dig let class rep start end gap op word bad space */
export function tokenize(line) {
  const toks = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    const c = line[i];
    if (/\s/.test(c)) { toks.push({ kind: "space", pos: i }); i++; }
    else if (c === "^") { toks.push({ kind: "start", pos: i }); i++; }
    else if (c === "$") { toks.push({ kind: "end", pos: i }); i++; }
    else if (c === "*") { toks.push({ kind: "gap", pos: i }); i++; }
    else if (line.startsWith(">=", i) || line.startsWith("<=", i)) { toks.push({ kind: "op", value: line.slice(i, i + 2), pos: i }); i += 2; }
    else if ("()&".includes(c)) { toks.push({ kind: "op", value: c, pos: i }); i++; }
    else if (c === "?") {
      const nx = (line[i + 1] || "").toLowerCase();
      if (nx === "d") { toks.push({ kind: "dig", value: DIGITS, pos: i }); i += 2; }
      else if (nx === "l") { toks.push({ kind: "let", value: LETTERS, pos: i }); i += 2; }
      else { toks.push({ kind: "any", value: ALL, pos: i }); i++; }
    } else if (c === "[") {
      const j = line.indexOf("]", i + 1), k = line.indexOf("[", i + 1);
      if (j < 0 || (k >= 0 && k < j)) throw new PatternError(`unbalanced '[' at position ${i + 1}`);
      toks.push({ kind: "class", value: parseClass(line.slice(i + 1, j), i), pos: i }); i = j + 1;
    } else if (c === "]") throw new PatternError(`unbalanced ']' at position ${i + 1}`);
    else if (c === "{") {
      const j = line.indexOf("}", i + 1);
      if (j < 0) throw new PatternError(`unclosed '{' at position ${i + 1}`);
      const body = line.slice(i + 1, j);
      if (!/^\d+$/.test(body)) throw new PatternError(`'{${body}}' at position ${i + 1}: a repeat count must be a number`);
      if (Number(body) === 0) throw new PatternError(`'{0}' at position ${i + 1}: a repeat count must be at least 1`);
      toks.push({ kind: "rep", value: Number(body), pos: i }); i = j + 1;
    } else if (c === "}") throw new PatternError(`unbalanced '}' at position ${i + 1}`);
    else if (isWordChar(c)) {
      let j = i;
      while (j < n && isWordChar(line[j])) j++;
      const run = line.slice(i, j).toLowerCase();
      if ([...run].every(isHex)) [...run].forEach((ch, k) => toks.push({ kind: "lit", value: 1 << HEXDIGITS.indexOf(ch), pos: i + k }));
      else toks.push({ kind: "word", value: run, pos: i });
      i = j;
    } else { toks.push({ kind: "bad", value: c, pos: i }); i++; }
  }
  return toks;
}

function parseClass(body, at) {
  if (!body) throw new PatternError(`empty class '[]' at position ${at + 1}`);
  if (body[0] === "^") throw new PatternError(`'[^...]' at position ${at + 1}: negated classes are not supported`);
  const b = body.toLowerCase();
  let s = 0, i = 0;
  while (i < b.length) {
    const c = b[i];
    if (!isHex(c)) throw new PatternError(`'${body[i]}' in class '[${body}]' at position ${at + 1}: only hex digits and ranges like 0-7`);
    if (i + 2 < b.length && b[i + 1] === "-") {
      const e = b[i + 2];
      if (!isHex(e)) throw new PatternError(`bad range '${body.slice(i, i + 3)}' in class '[${body}]'`);
      const lo = HEXDIGITS.indexOf(c), hi = HEXDIGITS.indexOf(e);
      if (lo > hi) throw new PatternError(`range '${body.slice(i, i + 3)}' in class '[${body}]' goes backwards`);
      for (let v = lo; v <= hi; v++) s |= 1 << v;
      i += 3;
    } else if (i + 1 < b.length && b[i + 1] === "-") throw new PatternError(`range '${body.slice(i)}' in class '[${body}]' has no end`);
    else { s |= 1 << HEXDIGITS.indexOf(c); i++; }
  }
  return s;
}

/** The reserved stat syntax: an operator (>=, <=, (, &) or a word followed by "best". */
export function isReserved(toks) {
  const t = toks.filter((x) => x.kind !== "space");
  return t.some((x) => x.kind === "op") || t.slice(1).some((x) => x.kind === "word" && x.value === "best");
}

export function nibMask(s) {
  let a = 0xF, o = 0;
  for (let v = 0; v < 16; v++) if ((s >> v) & 1) { a &= v; o |= v; }
  const m = ~(a ^ o) & 0xF;
  return [m, a & m];
}
export const popcount = (x) => { let n = 0; x >>>= 0; while (x) { x &= x - 1; n++; } return n; };

function comb(n, k) { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; }

export class Pattern {
  constructor(text, width, start, end, pieces) { Object.assign(this, { text, width, start, end, pieces }); }
  get mojang() { return this.width === 64; }

  /** Exact match. hexid: the full 64-digit Mojang hash or the 16-digit NameMC id. */
  match(hexid) {
    const h = [...hexid.toLowerCase()].map((c) => HEXDIGITS.indexOf(c));
    const L = h.length, K = this.pieces.length;
    let zl = 0;
    if (L === 64 && !this.start) while (zl < 63 && h[zl] === 0) zl++;
    const at = (p, pos) => p.every((s, i) => (s >> h[pos + i]) & 1);
    let cur = this.start ? 0 : zl;
    const limit = this.end ? L - this.pieces[K - 1].length : L;
    for (let i = 0; i < K; i++) {
      const p = this.pieces[i], n = p.length;
      if (i === 0 && this.start) {
        if (K === 1 && this.end && n !== L) return false;
        if (n > L || !at(p, 0)) return false;
        cur = n;
      } else if (i === K - 1 && this.end) {
        const pos = L - n;
        if (pos < cur || !at(p, pos)) return false;
        cur = L;
      } else {
        const end = i < K - 1 ? limit : L;
        let pos = cur;
        while (pos + n <= end && !at(p, pos)) pos++;
        if (pos + n > end) return false;
        cur = pos + n;
      }
    }
    return true;
  }

  /** Chance that a random id matches. */
  probability() {
    let p = 1;
    for (const piece of this.pieces) for (const s of piece) p *= popcount(s) / 16;
    const K = this.pieces.length;
    const free = this.pieces.filter((_, i) => !(i === 0 && this.start) && !(i === K - 1 && this.end)).map((x) => x.length);
    const total = this.pieces.reduce((a, x) => a + x.length, 0), T = free.reduce((a, x) => a + x, 0);
    const span = this.width - (total - T), j = free.length;
    if (span < T) return 0;
    if (!j) return p;
    return -Math.expm1(-comb(span - T + j, j) * p);
  }

  anchored() {
    if (!this.start && !this.end) return null;
    const L = this.width, mask = new Array(L).fill(0), val = new Array(L).fill(0);
    if (this.start) this.pieces[0].forEach((s, i) => { [mask[i], val[i]] = nibMask(s); });
    if (this.end) {
      const p = this.pieces[this.pieces.length - 1];
      p.forEach((s, i) => { const [m, v] = nibMask(s), j = L - p.length + i; mask[j] |= m; val[j] |= v; });
    }
    return { mask, val };
  }

  sliding(j) {
    const p = this.pieces[j], tm = p.map(nibMask), bits = tm.map(([m]) => popcount(m));
    let bd = 0, bb = -1;
    for (let d = 0; d < p.length; d++) {
      const b = bits.slice(d, d + 8).reduce((a, x) => a + x, 0);
      if (b > bb) { bb = b; bd = d; }
    }
    const word = (w) => w.reduce(([pv, mv], [m, v], i) => [(pv | (v << (28 - 4 * i))) >>> 0, (mv | (m << (28 - 4 * i))) >>> 0], [0, 0]);
    const [p1, m1] = word(tm.slice(bd, bd + 8)), [p2, m2] = word(tm.slice(bd + 8, bd + 16));
    const before = this.pieces.slice(0, j).reduce((a, x) => a + x.length, 0);
    const after = this.pieces.slice(j).reduce((a, x) => a + x.length, 0);
    const lo = before + bd, lim = this.width - after + bd;
    const nb = popcount(m1) + popcount(m2);
    return { kind: "sliding", piece: j, d: bd, p1, m1, p2, m2, lo, lim, usezl: this.mojang && !this.start, bits: nb,
      pass: Math.min(1, Math.max(0, lim - lo + 1) * Math.pow(2, -nb)) };
  }

  /** The most selective single GPU check (memoised). */
  prefilter() {
    if (this._pf) return this._pf;
    let best = null;
    const a = this.anchored();
    if (a) {
      const nb = a.mask.reduce((x, m) => x + popcount(m), 0);
      best = { kind: "anchored", mask: a.mask, val: a.val, bits: nb, pass: Math.pow(2, -nb) };
    }
    const K = this.pieces.length;
    for (let j = 0; j < K; j++) {
      if ((j === 0 && this.start) || (j === K - 1 && this.end)) continue;
      const s = this.sliding(j);
      if (!best || s.pass < best.pass) best = s;
    }
    return (this._pf = best);
  }

  /** The GPU prefilter on an id (the Mojang kernels know at most 16 leading zeros). Tests only. */
  prefilterPass(hexid, zlCap = 16) {
    const f = this.prefilter(), h = [...hexid].map((c) => HEXDIGITS.indexOf(c));
    if (f.kind === "anchored") return h.every((x, i) => (x & f.mask[i]) === f.val[i]);
    let zl = 0;
    while (zl < Math.min(zlCap, h.length) && h[zl] === 0) zl++;
    const lo = f.lo + (f.usezl ? zl : 0);
    const word = (o) => { let w = 0; for (let i = 0; i < 8; i++) w = (w | ((o + i < h.length ? h[o + i] : 0) << (28 - 4 * i))) >>> 0; return w; };
    for (let o = 0; o < h.length; o++) if (((word(o) ^ f.p1) & f.m1) === 0 && ((word(o + 8) ^ f.p2) & f.m2) === 0 && o >= lo && o <= f.lim) return true;
    return false;
  }

  /** Anchored prefilter as big-endian 32-bit words: {mw, vw} (8 Mojang, 2 NameMC). */
  words() {
    const f = this.prefilter(), n = this.width / 8;
    const pack = (arr, k) => { let w = 0; for (let i = 0; i < 8; i++) w = (w | (arr[8 * k + i] << (28 - 4 * i))) >>> 0; return w; };
    return { mw: Array.from({ length: n }, (_, k) => pack(f.mask, k)), vw: Array.from({ length: n }, (_, k) => pack(f.val, k)) };
  }
}

/** One pattern line -> Pattern; throws PatternError naming the pattern. */
export function parseLine(line, width) {
  const text = line.trim();
  try { return parseTokens(text, width); } catch (e) {
    if (e instanceof PatternError) throw new PatternError(`pattern '${text}': ${e.message}`);
    throw e;
  }
}

function parseTokens(text, width) {
  if (!text) throw new PatternError("empty pattern");
  const toks = tokenize(text);
  if (isReserved(toks)) throw new PatternError("stat targets (zeros >= N, ... best, &) are not supported yet");
  for (const t of toks) {
    if (t.kind === "space") throw new PatternError(`space at position ${t.pos + 1} (one pattern per line or per comma)`);
    if (t.kind === "bad") throw new PatternError(`'${t.value}' at position ${t.pos + 1} is not allowed (hex digits, ? ?d ?l [..] {n} ^ $ *)`);
    if (t.kind === "word") { const k = [...t.value].findIndex((c) => !isHex(c)); throw new PatternError(`'${t.value[k]}' at position ${t.pos + 1 + k} is not a hex digit`); }
  }
  const start = toks.length > 0 && toks[0].kind === "start";
  const end = (toks.length > 1 && toks[toks.length - 1].kind === "end") || (toks.length === 1 && toks[0].kind === "end");
  const body = toks.slice(start ? 1 : 0, end ? toks.length - 1 : toks.length);
  for (const t of body) {
    if (t.kind === "start") throw new PatternError(`'^' at position ${t.pos + 1}: only allowed as the first character`);
    if (t.kind === "end") throw new PatternError(`'$' at position ${t.pos + 1}: only allowed as the last character`);
  }
  const pieces = [];
  let cur = [], prev = null;
  for (const t of body) {
    if (t.kind === "rep") {
      if (!["lit", "any", "dig", "let", "class"].includes(prev)) throw new PatternError(`'{${t.value}}' at position ${t.pos + 1} has nothing to repeat`);
      if (t.value > width) throw new PatternError(`'{${t.value}}' at position ${t.pos + 1} is longer than the id (${width} digits)`);
      for (let k = 1; k < t.value; k++) cur.push(cur[cur.length - 1]);
    } else if (t.kind === "gap") {
      if (!cur.length) throw new PatternError(`'*' at position ${t.pos + 1} ` + (pieces.length ? "follows another '*'" : "at the start does nothing: drop it"));
      pieces.push(cur); cur = [];
    } else cur.push(t.value);
    prev = t.kind;
  }
  if (!cur.length) throw new PatternError(pieces.length ? "'*' at the end does nothing: drop it" : "no digits");
  pieces.push(cur);
  const norm = text.toLowerCase().replace(/\s+/g, "");
  const total = pieces.reduce((a, x) => a + x.length, 0);
  if (total > width) throw new PatternError(`needs ${total} digits, the id has only ${width}`);
  if (start && end && pieces.length === 1 && total !== width) throw new PatternError(`with ^ and $ and no '*' the pattern must be exactly ${width} digits (it is ${total})`);
  return new Pattern(norm, width, start, end, pieces);
}

/** Split on commas outside [...]. */
export function splitCommas(s) {
  const out = [];
  let depth = 0, cur = "";
  for (const c of s) {
    if (c === "[") depth++;
    else if (c === "]") depth = Math.max(0, depth - 1);
    if (c === "," && depth === 0) { out.push(cur); cur = ""; } else cur += c;
  }
  out.push(cur);
  return out;
}

/**
 * The Patterns box: one pattern per line (comma lists allowed), # comments and blank lines ignored.
 * Returns {patterns: [Pattern], lines: [{line, text, pattern?, error?}], errors: [string]}; duplicates are dropped.
 */
export function parsePatternText(text, width) {
  const lines = [], patterns = [], errors = [], seen = new Set();
  text.split(/\r?\n/).forEach((raw, idx) => {
    const body = raw.split("#")[0].trim();
    if (!body) return;
    for (const part of splitCommas(body)) {
      const t = part.trim();
      if (!t) { const e = `line ${idx + 1}: empty pattern in a comma list`; lines.push({ line: idx + 1, text: part, error: e }); errors.push(e); continue; }
      try {
        const p = parseLine(t, width);
        lines.push({ line: idx + 1, text: t, pattern: p });
        if (!seen.has(p.text)) { seen.add(p.text); patterns.push(p); }
      } catch (e) {
        if (!(e instanceof PatternError)) throw e;
        const msg = `line ${idx + 1}: ${e.message}`;
        lines.push({ line: idx + 1, text: t, error: msg });
        errors.push(msg);
      }
    }
  });
  if (patterns.length > MAX_PATTERNS) errors.push(`${patterns.length} patterns; at most ${MAX_PATTERNS}.`);
  return { patterns, lines, errors };
}

export const fmtOdds = (p) => (p <= 0 ? "never" : 1 / p < 10 ? `~1 in ${Number((1 / p).toPrecision(2))}` : 1 / p < 1e4 ? `~1 in ${Math.round(1 / p)}` : `~1 in ${(1 / p).toExponential(1).replace("e+", "e")}`);
