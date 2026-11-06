// Hex patterns (the "Patterns" box, native -m): the same rules as patterns.py (the reference) and pattern.h.
//   hex digits, ^ start, $ end, ? any digit, ?d 0-9, ?l a-f, [0-3] [05af] classes, {n} repeat, * gap
// Mojang: the full 64-digit hash; ^ is its start (^0{12} = 12 leading zeros), a pattern without ^ must lie in the
// link id (after the leading zeros). NameMC: the 16-digit id.
// Each pattern compiles to one GPU prefilter (anchored nibble masks, or one sliding window of up to 8 + 8 tokens);
// every hit that passes it is re-checked with match() on the CPU.
// Stat lines (record categories): counts (#0{24,}), different digits (##{,9}), shapes (~{25} palindrome, +{12} / -{12}
// staircase, (?){11} periodic run), {max} / {min} record mode, A & B. parseLine returns a Pattern or a StatLine;
// the rules and the odds are those of patterns.py (parse_stat, StatTerm, StatLine).

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

const ENGLISH = { zeros: "#0{24,} (at least 24 zeros)", zero: "#0{24,} (at least 24 zeros)", letters: "#?l{50,} or #?l{,5}",
  digits: "#?d{50,} or ##{,9} (different digits)", palindrome: "~{25} or ~{max}", staircase: "+{12} (up) or -{12} (down)",
  stairs: "+{12} or -{12}", run: "(?)\\1{11} (a digit 12x in a row) or 0{max}", zigzag: "(??)\\1{6}", best: "{max} or {min}",
  words: "#(dead|beef){3,}", distinct: "##{,9}", different: "##{,9}" };

/** The old reserved forms (zeros >= 24, palindrome best): an error that shows the expression syntax, or null. */
export function englishHint(toks) {
  for (const t of toks) {
    if (t.kind === "op" && (t.value === ">=" || t.value === "<=")) return `'${t.value}' at position ${t.pos + 1} is not part of the syntax: counts are written #0{24,} (at least), #0{,3} (at most)`;
    if (t.kind === "word" && Object.hasOwn(ENGLISH, t.value)) return `'${t.value}' at position ${t.pos + 1}: words are not part of the syntax, write ${ENGLISH[t.value]}`;
  }
  return null;
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

/** One pattern line -> Pattern or StatLine; throws PatternError naming the pattern. */
export function parseLine(line, width) {
  const text = line.trim();
  try { return isStat(text) ? parseStat(text, width) : parseTokens(text, width); } catch (e) {
    if (e instanceof PatternError) throw new PatternError(`pattern '${text}': ${e.message}`);
    throw e;
  }
}

function parseTokens(text, width) {
  if (!text) throw new PatternError("empty pattern");
  const toks = tokenize(text);
  const hint = englishHint(toks);
  if (hint) throw new PatternError(hint);
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

/** Split on commas outside [...], {...} and (...). */
export function splitCommas(s) {
  const out = [];
  let depth = 0, cur = "";
  for (const c of s) {
    if ("[{(".includes(c)) depth++;
    else if ("]})".includes(c)) depth = Math.max(0, depth - 1);
    if (c === "," && depth === 0) { out.push(cur); cur = ""; } else cur += c;
  }
  out.push(cur);
  return out;
}

/** A line without its comment. '#' starts a comment, except where it starts a count term (#0{24,}, ##{,9}): at the
 *  start of the line or after ',' or '&', followed by a hex digit, ?, [, (, @ or #. */
export function stripComment(line) {
  for (let i = 0; i < line.length; i++) {
    if (line[i] !== "#") continue;
    const prev = line.slice(0, i).trimEnd(), nxt = (line[i + 1] || "").toLowerCase();
    if ((!prev || ",&".includes(prev[prev.length - 1])) && nxt && (HEXDIGITS + "?[(@#").includes(nxt)) { if (nxt === "#") i++; continue; }
    return line.slice(0, i);
  }
  return line;
}

/**
 * The Patterns box: one pattern per line (comma lists allowed), # comments and blank lines ignored.
 * Returns {patterns: [Pattern], stats: [StatLine], lines: [{line, text, pattern?, stat?, error?}], errors: [string]};
 * duplicates are dropped.
 */
export function parsePatternText(text, width) {
  const lines = [], patterns = [], stats = [], errors = [], seen = new Set();
  text.split(/\r?\n/).forEach((raw, idx) => {
    const body = stripComment(raw).trim();
    if (!body) return;
    for (const part of splitCommas(body)) {
      const t = part.trim();
      if (!t) { const e = `line ${idx + 1}: empty pattern in a comma list`; lines.push({ line: idx + 1, text: part, error: e }); errors.push(e); continue; }
      try {
        const p = parseLine(t, width);
        const isStatLine = p instanceof StatLine;
        lines.push({ line: idx + 1, text: t, ...(isStatLine ? { stat: p } : { pattern: p }) });
        if (!seen.has(p.text)) { seen.add(p.text); (isStatLine ? stats : patterns).push(p); }
      } catch (e) {
        if (!(e instanceof PatternError)) throw e;
        const msg = `line ${idx + 1}: ${e.message}`;
        lines.push({ line: idx + 1, text: t, error: msg });
        errors.push(msg);
      }
    }
  });
  if (patterns.length > MAX_PATTERNS) errors.push(`${patterns.length} patterns; at most ${MAX_PATTERNS}.`);
  if (stats.length > MAX_STAT_LINES) errors.push(`${stats.length} stat lines; at most ${MAX_STAT_LINES}.`);
  else if (stats.reduce((a, x) => a + x.terms.length, 0) > MAX_STAT_OPS) errors.push(`More than ${MAX_STAT_OPS} stat terms in all lines.`);
  else if (stats.reduce((a, x) => a + x.terms.reduce((b, t) => b + (t.kind === "words" ? t.words.length : 0), 0), 0) > MAX_STAT_WORDS) errors.push(`More than ${MAX_STAT_WORDS} words in all word counts.`);
  return { patterns, stats, lines, errors };
}


// ---- stat lines ----------------------------------------------------------------------------------------------
// A stat line is one or more terms joined by '&' (all must hold). Terms:
//   #S{range}        count of digits in S (a digit, ?, ?d, ?l or a class) in the id
//   #(w1|w2){range}  occurrences of hex words of 1-8 digits (overlaps count)
//   ##{range}        number of different digits
//   ~{n}  +{n}  -{n} a palindrome / staircase up / staircase down of at least n digits
//   (?)\1{n} (??)\1{n} (?{p})\1{n}   a run with period p (1..8) of p*(n+1) digits
//   hex pattern      as -m; X{max} on its last token: the longest such repeat (^f{max}, 0{max}, ^0{max})
// range: {n} exactly (counts) / at least n (shapes), {n,} {,n} {n,m}, {max} or {min} = record mode.
// Mojang: counts, ## and unanchored shapes look at the link id (leading zeros dropped), ^ at the full hash.
export const MAX_TERMS = 6, MAX_WORDS = 64, MAX_PERIOD = 8, MAX_STAT_OPS = 96, MAX_STAT_WORDS = 64, MAX_WORD_LEN = 8, MAX_STAT_LINES = 32;
const SHAPES = { "~": "pal", "+": "up", "-": "down" };
const RANGE_HELP = "{24,} (at least), {,3} (at most), {5} (exactly), {2,6} or {max} / {min}";

export function isStat(text) {
  const s = text.toLowerCase().replace(/\s+/g, "");
  return !!s && (s.includes("&") || "#~+-(".includes(s[0]) || s.includes("\\") || s.includes("{max}") || s.includes("{min}") || /\{\d*,\d*\}/.test(s));
}

export const nibbles = (hexid) => [...hexid.toLowerCase()].map((c) => HEXDIGITS.indexOf(c));
/** Leading zeros the link drops (Mojang, at most 63); 0 for a NameMC id. */
export function zerosOf(h) {
  if (h.length !== 64) return 0;
  let z = 0;
  while (z < 63 && h[z] === 0) z++;
  return z;
}

export function longestPeriod(v, p) {
  let run = 0, best = 0;
  for (let i = 0; i < v.length - p; i++) { run = v[i] === v[i + p] ? run + 1 : 0; best = Math.max(best, run); }
  const ln = best ? best + p : Math.min(p, v.length);
  return p * Math.floor(ln / p);
}
export function longestStair(v, up) {
  let run = 0, best = 0;
  for (let i = 0; i < v.length - 1; i++) {
    const ok = up ? v[i] !== 15 && v[i + 1] === v[i] + 1 : v[i] !== 0 && v[i + 1] === v[i] - 1;
    run = ok ? run + 1 : 0; best = Math.max(best, run);
  }
  return v.length ? best + 1 : 0;
}
export function longestPalindrome(v) {
  const n = v.length;
  let best = n ? 1 : 0;
  for (let c = 0; c < n; c++) for (const odd of [1, 0]) {
    let a = odd ? c - 1 : c, b = c + 1;
    while (a >= 0 && b < n && v[a] === v[b]) { a--; b++; }
    best = Math.max(best, b - a - 1);
  }
  return best;
}
export function countWords(v, words) {
  let total = 0;
  for (const w of words) for (let o = 0; o + w.length <= v.length; o++) if (w.every((d, i) => v[o + i] === d)) total++;
  return total;
}

// -- odds of the stats (Mojang: the link id, i.e. 64 digits minus the leading zeros, first digit 1-f) --
function mix(width, f) {
  if (width !== 64) return f(width, false);
  let t = 0;
  for (let z = 0; z < 16; z++) t += Math.pow(16, -z) * 15 / 16 * f(64 - z, true);
  return t;
}
function countP(cset, n, nz, lo, hi) {
  const k = popcount(cset), p = k / 16, q = nz ? popcount(cset & ~1) / 15 : p;
  hi = hi === null ? n : Math.min(hi, n);
  let tot = 0;
  for (let x = lo; x <= hi; x++) for (const [f, pf] of [[0, 1 - q], [1, q]]) {
    const r = x - f;
    if (r >= 0 && r <= n - 1 && pf) tot += pf * comb(n - 1, r) * Math.pow(p, r) * Math.pow(1 - p, n - 1 - r);
  }
  return tot;
}
function distinctP(n, nz, lo, hi) {
  let st = new Map([[nz ? "1,0" : "0,0", 1]]);
  for (let it = 0; it < (nz ? n - 1 : n); it++) {
    const nw = new Map();
    const add = (key, v) => nw.set(key, (nw.get(key) || 0) + v);
    for (const [key, p] of st) {
      const [j, z] = key.split(",").map(Number);
      add(`${j},1`, p / 16);
      if (j) add(`${j},${z}`, p * j / 16);
      if (15 - j) add(`${j + 1},${z}`, p * (15 - j) / 16);
    }
    st = nw;
  }
  let s = 0;
  for (const [key, p] of st) { const [j, z] = key.split(",").map(Number); if (lo <= j + z && (hi === null || j + z <= hi)) s += p; }
  return s;
}
function runP(m, r, q) {
  if (r <= 0) return 1;
  if (m < r) return 0;
  let st = [1, ...new Array(r - 1).fill(0)], hit = 0;
  for (let i = 0; i < m; i++) {
    const nw = new Array(r).fill(0);
    st.forEach((p, k) => {
      nw[0] += p * (1 - q);
      if (k + 1 >= r) hit += p * q; else nw[k + 1] += p * q;
    });
    st = nw;
  }
  return hit;
}
function periodP(n, p, L) {
  if (L > n) return 0;
  return L <= p ? 1 : runP(n - p, L - p, 1 / 16);
}
function stairP(n, nz, L, up) {
  if (L <= 1) return 1;
  if (n < L) return 0;
  let st = new Map();
  for (let d = nz ? 1 : 0; d < 16; d++) st.set(`${d},1`, nz ? 1 / 15 : 1 / 16);
  let hit = 0;
  for (let it = 0; it < n - 1; it++) {
    const nw = new Map();
    const add = (key, v) => nw.set(key, (nw.get(key) || 0) + v);
    let spread = 0;
    for (const [key, p] of st) {
      const [d, k] = key.split(",").map(Number);
      const q = p / 16;
      spread += q;
      const nx = up ? d + 1 : d - 1;
      if (nx >= 0 && nx <= 15) {
        add(`${nx},1`, -q);
        if (k + 1 >= L) hit += q; else add(`${nx},${k + 1}`, q);
      }
    }
    for (let e = 0; e < 16; e++) add(`${e},1`, spread);
    st = nw;
  }
  return hit;
}
function palP(n, L) {
  if (L <= 1) return 1;
  if (L > n) return 0;
  const he = Math.floor((L + 1) / 2), ho = Math.floor(L / 2);
  let lam = 0;
  for (let c = 0; c < n - 1; c++) if (Math.min(c + 1, n - 1 - c) >= he) lam += Math.pow(16, -he);
  for (let c = 0; c < n; c++) if (Math.min(c, n - 1 - c) >= ho) lam += Math.pow(16, -ho);
  return -Math.expm1(-lam);
}
function poissonRange(lam, lo, hi) {
  const cdf = (k) => {
    if (k < 0) return 0;
    let t = Math.exp(-lam), s = t;
    for (let i = 1; i <= k; i++) { t *= lam / i; s += t; }
    return Math.min(1, s);
  };
  return (hi === null ? 1 : cdf(hi)) - cdf(lo - 1);
}

const nibbleExact = (x) => { const [m, v] = nibMask(x); let s = 0; for (let d = 0; d < 16; d++) if ((d & m) === v) s |= 1 << d; return x === s; };

export class StatTerm {
  constructor(kind, text, width, lo = 0, hi = null, rec = null, kw = {}) {
    Object.assign(this, { kind, text, width, lo, hi, rec });
    this.cset = kw.cset || 0; this.words = kw.words || null; this.period = kw.period || 0;
    this.pattern = kw.pattern || null; this.prefix = kw.prefix || null; this.tok = kw.tok || 0; this.start = !!kw.start;
    this.end = !!kw.end;   // X{max}$: the run that ends the id
  }
  endRun(h) { let run = 0; while (run < h.length && (this.tok >> h[h.length - 1 - run]) & 1) run++; return run; }
  finderText() { return this.text; }
  /** {max} hex term with the token repeated k times: a Pattern, "always" or null (longer than the id). */
  hexK(k) {
    const toks = [...this.prefix, ...new Array(k).fill(this.tok)];
    if (!toks.length) return "always";
    if (toks.length > this.width) return null;
    return new Pattern(this.text, this.width, this.start, this.end, [toks]);
  }
  /** Reference value (null: a record term whose pattern does not match). h: nibbles of the whole id, vis: the visible ones. */
  value(h, vis, hexid) {
    switch (this.kind) {
      case "hex": {
        if (!this.rec) return this.pattern.match(hexid);
        if (this.end) return this.endRun(h);
        const s = this.start ? h : vis, pre = this.prefix;
        let best = null;
        const offs = this.start ? [0] : [...Array(Math.max(0, s.length - pre.length + 1)).keys()];
        for (const o of offs) {
          if (o + pre.length <= s.length && pre.every((x, i) => (x >> s[o + i]) & 1)) {
            let j = o + pre.length;
            while (j < s.length && (this.tok >> s[j]) & 1) j++;
            const run = j - o - pre.length;
            best = best === null ? run : Math.max(best, run);
          }
        }
        return best;
      }
      case "count": return vis.filter((c) => (this.cset >> c) & 1).length;
      case "words": return countWords(vis, this.words);
      case "distinct": return new Set(vis).size;
      case "period": return longestPeriod(vis, this.period);
      case "up": case "down": return longestStair(vis, this.kind === "up");
      default: return longestPalindrome(vis);
    }
  }
  gpuExact() {
    if (this.kind !== "hex") return true;
    if (!this.rec) return false;
    if (!this.prefix.length) return true;
    return this.start && this.prefix.every(nibbleExact);
  }
  /** The GPU value of an X{max} term (see StatTerm.gpu_hex_rec in patterns.py). */
  gpuHexRec(h, vis) {
    if (this.end) return this.endRun(h);
    if (this.start) {
      for (let i = 0; i < this.prefix.length; i++) { const [m, v] = nibMask(this.prefix[i]); if ((h[i] & m) !== v) return null; }
      let run = 0;
      const j = this.prefix.length;
      while (j + run < h.length && (this.tok >> h[j + run]) & 1) run++;
      return run;
    }
    let best = 0, run = 0;
    for (const d of vis) { run = (this.tok >> d) & 1 ? run + 1 : 0; best = Math.max(best, run); }
    return best;
  }
  holds(v) {
    if (this.kind === "hex") return !!v;
    return v !== null && v >= this.lo && (this.hi === null || v <= this.hi);
  }
  beats(v, best) {
    if (v === null) return false;
    if (best === null || best === undefined) return true;
    return this.rec === "max" ? v > best : v < best;
  }
  probRange(lo, hi) {
    const W = this.width;
    switch (this.kind) {
      case "count": return mix(W, (n, nz) => countP(this.cset, n, nz, lo, hi));
      case "distinct": return mix(W, (n, nz) => distinctP(n, nz, lo, hi));
      case "words": {
        const lam = (n) => this.words.reduce((a, w) => a + Math.max(0, n - w.length + 1) * Math.pow(16, -w.length), 0);
        return mix(W, (n) => poissonRange(lam(n), lo, hi));
      }
      case "period": { const p = this.period, L = p * Math.ceil(lo / p); return mix(W, (n) => periodP(n, p, L)); }
      case "up": case "down": return mix(W, (n, nz) => stairP(n, nz, lo, this.kind === "up"));
      case "pal": return mix(W, (n) => palP(n, lo));
      default: throw new Error(this.kind);
    }
  }
  /** Threshold term: P(it holds). Record term: P(it beats best) (1 without a best). */
  probability(best = null) {
    if (this.kind === "hex") {
      if (!this.rec) return this.pattern.probability();
      const p = this.hexK(best === null ? 0 : best + 1);
      return p === null ? 0 : p === "always" ? 1 : p.probability();
    }
    if (this.rec) {
      if (best === null) return 1;
      return this.rec === "max" ? this.probRange(best + 1, null) : (best > 0 ? this.probRange(0, best - 1) : 0);
    }
    return this.probRange(this.lo, this.hi);
  }
  tag() {
    const k = this.kind;
    if (k === "count") {
      const s = this.cset;
      const name = popcount(s) === 1 ? HEXDIGITS[31 - Math.clz32(s)] : ({ [LETTERS]: "l", [DIGITS]: "d", [ALL]: "all" })[s] || s.toString(16).padStart(4, "0");
      return `count${name}`;
    }
    if (k === "hex") return (this.end ? "hexend-" : this.start ? "hexstart-" : "hex-") + this.text.replace("{max}", "").replace(/[^0-9a-z]+/g, "").slice(0, 12);
    return ({ words: "words", distinct: "distinct", up: "up", down: "down", pal: "pal" })[k] || `run${this.period}`;
  }
}

export class StatLine {
  constructor(text, width, terms) {
    Object.assign(this, { text, width, terms });
    this.recTerm = terms.find((t) => t.rec) || null;
  }
  get record() { return this.recTerm ? this.recTerm.rec : null; }
  finderText() { return this.terms.map((t) => t.finderText()).join("&"); }
  gpuExact() { return this.terms.every((t) => t.gpuExact()); }
  /** {ok, value}: ok = every threshold term holds (and the record term has a value); value of the record term. */
  evaluate(hexid) {
    const h = nibbles(hexid), vis = h.slice(zerosOf(h));
    let val = null;
    for (const t of this.terms) {
      const v = t.value(h, vis, hexid);
      if (t.rec) { if (v === null) return { ok: false, value: null }; val = v; }
      else if (!t.holds(v)) return { ok: false, value: null };
    }
    return { ok: true, value: val };
  }
  match(hexid) { return this.evaluate(hexid).ok; }
  beats(value, best) { return this.recTerm.beats(value, best); }
  /** The GPU stat test (patterns.py gpu_pass). */
  gpuPass(hexid, best = null) {
    const h = nibbles(hexid), vis = h.slice(zerosOf(h));
    for (const t of this.terms) {
      if (t.kind === "hex" && !t.rec) {
        const a = t.pattern.anchored();
        if (a && !h.every((x, i) => (x & a.mask[i]) === a.val[i])) return false;
        continue;
      }
      const v = t.kind === "hex" ? t.gpuHexRec(h, vis) : t.value(h, vis, hexid);
      if (t.rec) { if (v === null || (best !== null && !t.beats(v, best))) return false; }
      else if (!t.holds(v)) return false;
    }
    return true;
  }
  probability(best = null) {
    let p = 1;
    for (const t of this.terms) p *= t.rec ? t.probability(best) : t.probability();
    return p;
  }
  tag() { return this.terms.length ? (this.recTerm || this.terms[0]).tag() : "stat"; }
}

function parseRange(s, i, P) {
  if (i >= s.length) throw new PatternError(`missing range after position ${P(i - 1)}: add ${RANGE_HELP}`);
  if (s[i] !== "{") throw new PatternError(`'${s[i]}' at position ${P(i)}: expected a range: ${RANGE_HELP}`);
  const j = s.indexOf("}", i);
  if (j < 0) throw new PatternError(`unclosed '{' at position ${P(i)}`);
  if (j !== s.length - 1) throw new PatternError(`'${s.slice(j + 1)}' at position ${P(j + 1)} after the range (join terms with '&')`);
  const body = s.slice(i + 1, j);
  if (body === "max" || body === "min") return { lo: null, hi: null, rec: body, exact: false };
  const m = /^(\d*)(,?)(\d*)$/.exec(body);
  if (!m || !(m[1] || m[3]) || (!m[2] && m[3])) throw new PatternError(`'{${body}}' at position ${P(i)}: a range is ${RANGE_HELP}`);
  const a = m[1] ? Number(m[1]) : 0;
  if (!m[2]) return { lo: a, hi: a, rec: null, exact: true };
  const b = m[3] ? Number(m[3]) : null;
  if (b !== null && a > b) throw new PatternError(`'{${body}}' at position ${P(i)} goes backwards`);
  return { lo: a, hi: b, rec: null, exact: false };
}

function parseTerm(raw, off, width) {
  const lead = raw.length - raw.trimStart().length;
  const sr = raw.trim();
  off += lead;
  const pos = [];
  [...sr].forEach((c, i) => { if (!/\s/.test(c)) pos.push(off + i); });
  const s = [...sr].filter((c) => !/\s/.test(c)).join("").toLowerCase();
  const P = (i) => pos[Math.max(0, Math.min(i, pos.length - 1))] + 1;
  const always = `'${s}' always holds`;
  const checkCount = (t, maxv) => {
    if (t.rec) return t;
    if (t.lo > maxv) throw new PatternError(`'${s}' can never hold: the id has ${width === 64 ? "at most " : ""}${maxv} ${t.kind === "distinct" ? "different digits" : "digits"}`);
    const p = t.probability();
    if (p >= 1 - 1e-12) throw new PatternError(always);
    if (p <= 0) throw new PatternError(`'${s}' can never hold`);
    return t;
  };
  if (s.startsWith("#@")) throw new PatternError(`'#@FILE' at position ${off + 1}: the browser cannot read files by name; write the words in the line: #(dead|beef|cafe){4,}`);
  if (s.startsWith("##")) {
    const { lo, hi, rec } = parseRange(s, 2, P);
    if (rec === "max") throw new PatternError(`'${s}': nearly every id uses all 16 digits; use ##{min} or a range like ##{,9}`);
    return checkCount(new StatTerm("distinct", s, width, lo, hi, rec), 16);
  }
  if (s.startsWith("#")) {
    let i = 1;
    if (i >= s.length) throw new PatternError(`'#' at position ${P(0)}: count what? e.g. #0{24,}, #?l{,5}, #[0-3]{20,}, #(dead|beef){3,}`);
    const c = s[i];
    let words = null, cset = 0;
    if (c === "(") {
      const j = s.indexOf(")", i);
      if (j < 0) throw new PatternError(`unclosed '(' at position ${P(i)}`);
      words = [];
      const seen = new Set();
      let k = i + 1;
      for (const w of s.slice(i + 1, j).split("|")) {
        if (!w || ![...w].every(isHex) || w.length > MAX_WORD_LEN) throw new PatternError(`'${w}' at position ${P(k)}: words in #(...) are hex words of 1-${MAX_WORD_LEN} digits joined by '|'`);
        if (!seen.has(w)) { seen.add(w); words.push([...w].map((ch) => HEXDIGITS.indexOf(ch))); }
        k += w.length + 1;
      }
      if (words.length > MAX_WORDS) throw new PatternError(`${words.length} words; at most ${MAX_WORDS}`);
      i = j + 1;
    } else if (c === "?") {
      const nx = s[i + 1];
      if (nx === "d") { cset = DIGITS; i += 2; } else if (nx === "l") { cset = LETTERS; i += 2; } else { cset = ALL; i += 1; }
    } else if (c === "[") {
      const j = s.indexOf("]", i);
      if (j < 0) throw new PatternError(`unbalanced '[' at position ${P(i)}`);
      cset = parseClass(s.slice(i + 1, j), P(i) - 1); i = j + 1;
    } else if (isHex(c)) { cset = 1 << HEXDIGITS.indexOf(c); i += 1; }
    else throw new PatternError(`'${c}' at position ${P(i)}: '#' counts a digit, ?, ?d, ?l, a class or (word|word)`);
    if (i < s.length && s[i] !== "{") {
      throw new PatternError(isHex(s[i])
        ? `'${s[i]}' at position ${P(i)}: '#' counts one digit, ?, ?d, ?l or class; for a word write #(${s.slice(1, i) + s.slice(i).split("{")[0]}){...}`
        : `'${s[i]}' at position ${P(i)}: expected a range: ${RANGE_HELP}`);
    }
    const { lo, hi, rec } = parseRange(s, i, P);
    if (words) {
      if (rec === "min") throw new PatternError(`'${s}': {min} of a word count is almost always 0; use {max} or a range`);
      if (!rec && lo === 0 && hi === null) throw new PatternError(always);
      return checkCount(new StatTerm("words", s, width, lo, hi, rec, { words }), width);
    }
    return checkCount(new StatTerm("count", s, width, lo, hi, rec, { cset }), width);
  }
  if (s[0] in SHAPES || s[0] === "(") {
    let kind, ri, p;
    if (s[0] === "(") {
      const m = /^\((\?+|\?\{(\d+)\})\)\\1(\{.*)$/.exec(s);
      if (!m) throw new PatternError(`'${s}' at position ${P(0)}: only the periodic-run form is supported: (?)\\1{n} (one digit n+1 times), (??)\\1{n} (a pair repeated), (?{p})\\1{n} (p = 1-8)`);
      p = m[2] ? Number(m[2]) : m[1].length;
      if (p < 1 || p > MAX_PERIOD) throw new PatternError(`period ${p} at position ${P(1)}: the period is 1 to ${MAX_PERIOD} digits`);
      kind = "period"; ri = s.length - m[3].length;
    } else { kind = SHAPES[s[0]]; ri = 1; p = 0; }
    let { lo, hi, rec, exact } = parseRange(s, ri, P);
    if (rec === "min") throw new PatternError(`'${s}': {min} is only for counts (#...) and ##`);
    if (!rec) {
      if (hi !== null && !exact) throw new PatternError(`'${s.slice(ri)}' at position ${P(ri)}: a shape takes {n} (at least n) or {max}`);
      let n = lo;
      if (kind === "period") {
        if (n < 1) throw new PatternError(`'${s.slice(ri)}' at position ${P(ri)}: the repeat count must be at least 1`);
        n = p * (n + 1);
      } else if (n < 2) throw new PatternError(always);
      if (n > width) throw new PatternError(`'${s}' needs ${n} digits, the id has only ${width}`);
      lo = n; hi = null;
    }
    return new StatTerm(kind, s, width, lo, hi, rec, { period: p });
  }
  // a hex pattern, maybe with X{max} at the end
  const rm = /\{\d*,\d*\}/.exec(s);
  if (rm) throw new PatternError(`'${rm[0]}' at position ${P(rm.index)}: ranges are for counts (#0{24,}) and shapes; in a hex pattern {n} repeats exactly`);
  if (s.includes("{min}")) throw new PatternError(`'{min}' at position ${P(s.indexOf("{min}"))}: {min} is only for counts (#...) and ##`);
  if (s.split("{max}").length > 2) throw new PatternError(`two {max} in '${s}'; one per line`);
  if (s.includes("{max}")) {
    const i = s.indexOf("{max}");
    const end = s.slice(i + 5) === "$";
    if ((s.slice(i + 5) && !end) || s.includes("*") || s.slice(0, i).includes("$")) throw new PatternError(`'{max}' at position ${P(i)}: in a hex pattern {max} repeats the last token of a pattern without '*': ^f{max}, 0{max}, ^dead0{max}, 0{max}$`);
    let pat;
    try { pat = parseTokens(s.slice(0, i) + "{1}", width); } catch (e) { if (e instanceof PatternError) throw new PatternError(`in '${s}': ${e.message}`); throw e; }
    const pc = pat.pieces[0];
    if (end && (pat.start || pc.length !== 1)) throw new PatternError(`'{max}$' at position ${P(i)}: a run at the end takes a single token: 0{max}$, ?l{max}$`);
    return new StatTerm("hex", s, width, 0, null, "max", { prefix: pc.slice(0, -1), tok: pc[pc.length - 1], start: pat.start, end });
  }
  let pat;
  try { pat = parseTokens(s, width); } catch (e) { if (e instanceof PatternError) throw new PatternError(`in '${s}': ${e.message}`); throw e; }
  return new StatTerm("hex", pat.text, width, 0, null, null, { pattern: pat });
}

/** A stat line -> StatLine. Terms are split on '&' outside [...] and (...). */
export function parseStat(text, width) {
  const parts = [];
  let depth = 0, start = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "[" || c === "(") depth++;
    else if (c === "]" || c === ")") depth = Math.max(0, depth - 1);
    else if (c === "&" && depth === 0) { parts.push([start, text.slice(start, i)]); start = i + 1; }
  }
  parts.push([start, text.slice(start)]);
  const terms = [];
  for (const [off, raw] of parts) {
    if (!raw.trim()) throw new PatternError(`empty term at position ${off + 1}: '&' needs an expression on both sides`);
    terms.push(parseTerm(raw, off, width));
  }
  if (terms.length > MAX_TERMS) throw new PatternError(`${terms.length} terms; at most ${MAX_TERMS} per line`);
  const recs = terms.filter((t) => t.rec).map((t) => t.text);
  if (recs.length > 1) throw new PatternError(`${recs.join(" and ")}: at most one {max}/{min} per line`);
  return new StatLine(terms.map((t) => t.text).join("&"), width, terms);
}

export const fmtOdds = (p) => (p <= 0 ? "never" : 1 / p < 10 ? `~1 in ${Number((1 / p).toPrecision(2))}` : 1 / p < 1e4 ? `~1 in ${Math.round(1 / p)}` : `~1 in ${(1 / p).toExponential(1).replace("e+", "e")}`);
