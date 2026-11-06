// Stat lines (patterns.js StatLine) in WGSL: counts, different digits, word counts, periodic runs, staircases, palindromes
// and X{max} runs, computed on the digest words of a candidate. Same values as the reference (StatTerm.value), see
// "Stat lines" in docs/how-it-works.md. Everything is baked into the generated shader text; a shader without stat lines
// contains none of it.
//
// The id is W digest words (8 Mojang, 2 NameMC), digit i = nibble i counted from the top of word 0. zl = leading zero
// digits of a Mojang hash (at most 63, 0 for NameMC); the visible id is digits zl..width-1.
//   - Flag words: one bit (bit 3) per nibble. A zero nibble is found with nzf(); a digit set S is an OR of those.
//   - Runs: the flags of all words are compacted into a 64-bit mask M (vec2: x = high word), bit 63-i = digit i. A run of
//     r flags is found by ANDing M with shifted copies of itself (doubling), after clearing the digits < zl.
//   - Palindromes: 8-digit windows of the digest are compared with windows of the nibble-reversed digest, one per
//     centre (odd and even); a centre that passes is expanded exactly (bounded by zl).
// Record lines keep their best value in the storage buffer RB (one i32 per record line, atomicMax / atomicMin). A line
// whose value the GPU computes exactly (gpuExact) raises the best itself and reports only improvements; for the others
// the host raises the best after its own check.
import { HEXDIGITS, nibMask, popcount, LETTERS, DIGITS, ALL } from "./patterns.js";

export const MAXSTAT = 32;
const hex = (v) => `0x${(v >>> 0).toString(16)}u`;
const COST = { hex: 0, count: 1, distinct: 2, words: 3, period: 4, up: 4, down: 4, pal: 5 };

/** Module-level helper functions (only emitted when there are stat lines). */
export function statLib(W) {
  const width = W * 8;
  const arr = `array<u32, ${W}>`;
  return `
@group(0) @binding(5) var<storage, read_write> RB: array<atomic<i32>>;
fn nzf(x: u32) -> u32 { let y = (x & 0x77777777u) + 0x77777777u; return ~(y | x | 0x77777777u); }
fn cmp8(f: u32) -> u32 {
  var a = (f >> 3u) & 0x11111111u;
  a = (a | (a >> 3u)) & 0x03030303u;
  a = (a | (a >> 6u)) & 0x000F000Fu;
  return (a | (a >> 12u)) & 0xFFu;
}
fn nibrev(x: u32) -> u32 {
  let b = (x << 24u) | ((x & 0xFF00u) << 8u) | ((x >> 8u) & 0xFF00u) | (x >> 24u);
  return ((b & 0x0F0F0F0Fu) << 4u) | ((b >> 4u) & 0x0F0F0F0Fu);
}
fn shl64(v: vec2<u32>, k: u32) -> vec2<u32> {
  if (k == 0u) { return v; }
  if (k >= 64u) { return vec2<u32>(0u, 0u); }
  if (k >= 32u) { return vec2<u32>(v.y << (k - 32u), 0u); }
  return vec2<u32>((v.x << k) | (v.y >> (32u - k)), v.y << k);
}
fn nz64(v: vec2<u32>) -> bool { return (v.x | v.y) != 0u; }
fn vis64(zl: u32) -> vec2<u32> {
  let n = 64u - zl;
  if (n >= 64u) { return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu); }
  if (n >= 32u) { return vec2<u32>((1u << (n - 32u)) - 1u, 0xFFFFFFFFu); }
  return vec2<u32>(0u, (1u << n) - 1u);
}
fn run_ge(m: vec2<u32>, r: u32) -> vec2<u32> {
  var a = m; var len = 1u;
  if (r <= 1u) { return a; }
  loop { if (len * 2u > r) { break; } a = a & shl64(a, len); len = len * 2u; }
  if (len < r) { a = a & shl64(a, r - len); }
  return a;
}
fn lng_run(m: vec2<u32>) -> u32 {
  var a = m; var len = 0u;
  loop { if (!nz64(a)) { break; } len = len + 1u; a = a & shl64(m, len); }
  return len;
}
fn clz64(v: vec2<u32>) -> u32 { return select(32u + countLeadingZeros(v.y), countLeadingZeros(v.x), v.x != 0u); }
fn end_run(v: vec2<u32>) -> u32 {   // set bits from bit ${64 - width} (the last digit) upwards, at most ${width}
  let sh = ${64 - width}u;
  var lo = v.y; var hi = v.x;
  if (sh >= 32u) { lo = v.x >> (sh - 32u); hi = 0u; } else if (sh > 0u) { lo = (v.y >> sh) | (v.x << (32u - sh)); hi = v.x >> sh; }
  var r = countTrailingZeros(~lo);
  if (r == 32u) { r = 32u + countTrailingZeros(~hi); }
  return min(r, ${width}u);
}
fn pmask(k: u32) -> u32 { return select(select(~(0xFFFFFFFFu >> (4u * k)), 0u, k == 0u), 0xFFFFFFFFu, k >= 8u); }
fn nib_at(x: ptr<function, ${arr}>, i: u32) -> u32 { return ((*x)[i >> 3u] >> (28u - 4u * (i & 7u))) & 15u; }
fn pal_len(x: ptr<function, ${arr}>, zl: u32, c: u32, odd: bool) -> u32 {
  var a = i32(select(c, c - 1u, odd)); var b = c + 1u;
  loop {
    if (a < i32(zl) || b >= ${width}u) { break; }
    if (nib_at(x, u32(a)) != nib_at(x, b)) { break; }
    a = a - 1; b = b + 1u;
  }
  return u32(i32(b) - a - 1);
}
`;
}

/** WGSL for the stat lines of one shader. words: names of the W digest words. zl: expression of the leading zeros (a u32
 *  expression, "0u" for NameMC). SE: index in PT of the enable word of line 0. Expects `var hit` in scope; sets it.
 *  rbIdx(ri): expression of the RB slot of record number ri (tests give every id its own). */
export function statCode(lines, words, { zl, SE, rbIdx = (ri) => `${ri}u` }) {
  const W = words.length, width = W * 8;
  let ri = 0;
  let s = `{\n  ${words.map((w, k) => `let x${k} = ${w};`).join(" ")}\n  let zl = ${zl};\n  let vis = vis64(zl);\n`;
  const X = (k) => (k >= 0 && k < W ? `x${k}` : "0u");
  // digest shifted left by d digits: word k of the shifted string
  const shifted = (name, d) => (k) => {
    const wq = Math.floor(d / 8), r = d % 8, a = name(k + wq), b = name(k + wq + 1);
    if (a === "0u") return "0u";
    return r === 0 ? a : (b === "0u" ? `(${a} << ${4 * r}u)` : `((${a} << ${4 * r}u) | (${b} >> ${32 - 4 * r}u))`);
  };
  const mask64 = (flags) => {   // flags(k) -> expression of the flag word k; returns vec2 expression
    const b = (k) => `(cmp8(${flags(k)}) << ${24 - 8 * (k % 4)}u)`;
    const half = (h) => Array.from({ length: 4 }, (_, j) => 4 * h + j).filter((k) => k < W).map(b).join(" | ") || "0u";
    return `vec2<u32>(${half(0)}, ${half(1)})`;
  };
  const lowClear = (c) => {   // bits >= c kept (literal vec2)
    const keep = (hi) => { let v = 0n; for (let b = 0; b < 32; b++) if (BigInt(hi * 32 + b) >= BigInt(c)) v |= 1n << BigInt(b); return hex(Number(v)); };
    return `vec2<u32>(${keep(1)}, ${keep(0)})`;
  };
  const setFlags = (S, x) => {
    if (S === ALL) return "0x88888888u";
    const lets = `(${x} & ((${x} << 1u) | (${x} << 2u)) & 0x88888888u)`;
    if (S === LETTERS) return lets;
    if (S === DIGITS) return `(~${lets} & 0x88888888u)`;
    const vals = [...Array(16).keys()].filter((v) => (S >> v) & 1);
    const comp = vals.length > 8;
    const use = comp ? [...Array(16).keys()].filter((v) => !((S >> v) & 1)) : vals;
    const orx = use.map((v) => `nzf(${x} ^ ${hex(v * 0x11111111)})`).join(" | ");
    return comp ? `(~(${orx}) & 0x88888888u)` : `(${orx})`;
  };
  // the fixed 'need' of a threshold term / the best-derived need of a record term
  const cmpCond = (v, t) => (t.hi === null ? `${v} >= ${t.lo}u` : t.lo === t.hi ? `${v} == ${t.lo}u` : `${v} >= ${t.lo}u && ${v} <= ${t.hi}u`);
  const claim = (line, recIdx, v) => {   // v: u32 expression of the value
    const slot = `RB[${rbIdx(recIdx)}]`, max = line.record === "max";
    const op = max ? ">" : "<";
    const exact = line.gpuExact();
    return exact
      ? `let rv = i32(${v}); if (rv ${op} atomicLoad(&${slot})) { let old = ${max ? "atomicMax" : "atomicMin"}(&${slot}, rv); if (rv ${op} old) { hit = true; } }`
      : `let rv = i32(${v}); if (rv ${op} atomicLoad(&${slot})) { hit = true; }`;
  };
  const needOf = (recIdx, step) => `let best = atomicLoad(&RB[${rbIdx(recIdx)}]); let need = select(${step}u, ((u32(best) + ${step}u) / ${step}u) * ${step}u, best >= 0);`;
  const termCode = (t, line, recIdx) => {
    // returns code that, inside `if (ok) {`, sets ok (threshold) or claims (record)
    const rec = !!t.rec;
    const fin = (v, ok) => (rec ? claim(line, recIdx, v) : `ok = ${ok};`);
    switch (t.kind) {
      case "hex": {
        if (!rec) {
          const a = t.pattern.anchored();
          if (!a) return "";
          const { mw, vw } = t.pattern.words();
          const terms = mw.map((m, k) => (m ? `((x${k} ^ ${hex(vw[k])}) & ${hex(m)})` : null)).filter(Boolean);
          return terms.length ? `ok = (${terms.join(" | ")}) == 0u;` : "";
        }
        // X{max}: prefix by nibble masks (with ^), then the run of the token set
        const P = t.prefix.length;
        let pre = "";
        if (t.start && P) {
          const mw = new Array(W).fill(0), vw = new Array(W).fill(0);
          t.prefix.forEach((x, i) => { const [m, v] = nibMask(x); mw[i >> 3] |= m << (28 - 4 * (i & 7)); vw[i >> 3] |= v << (28 - 4 * (i & 7)); });
          const terms = mw.map((m, k) => (m ? `((x${k} ^ ${hex(vw[k] >>> 0)}) & ${hex(m >>> 0)})` : null)).filter(Boolean);
          pre = `if ((${terms.join(" | ")}) == 0u) {`;
        } else pre = "{";
        const M = mask64((k) => setFlags(t.tok, X(k)));
        const val = t.end ? `end_run(${M})` : t.start ? `clz64(~shl64(${M}, ${P}u))` : "lng_run(" + M + " & vis)";
        return `${pre} let v = ${val}; ${claim(line, recIdx, "v")} }`;
      }
      case "count": {
        const S = t.cset;
        const tot = Array.from({ length: W }, (_, k) => `countOneBits(${setFlags(S, X(k))})`).join(" + ");
        const v = `(${tot} - ${S & 1 ? "zl" : "0u"})`;
        return `let v = ${v}; ${fin("v", cmpCond("v", t))}`;
      }
      case "distinct": {
        const ors = [];
        for (let k = 0; k < W; k++) for (let j = 0; j < 8; j++) ors.push(`(1u << ((${X(k)} >> ${28 - 4 * j}u) & 15u))`);
        const tz = Array.from({ length: W }, (_, k) => `countOneBits(nzf(${X(k)}))`).join(" + ");
        return `let bits = ${ors.join(" | ")}; let tz = ${tz}; let v = countOneBits((bits & ~1u) | select(0u, 1u, tz > zl)); ${fin("v", cmpCond("v", t))}`;
      }
      case "words": {
        let c = "";
        const win = (o) => {
          const w = o >> 3, sh = 4 * (o & 7);
          return sh === 0 ? X(w) : (w + 1 < W ? `((${X(w)} << ${sh}u) | (${X(w + 1)} >> ${32 - sh}u))` : `(${X(w)} << ${sh}u)`);
        };
        for (let o = 0; o < width; o++) c += `let w${o} = ${win(o)}; `;
        c += "var v = 0u; ";
        for (const wd of t.words) {
          const n = wd.length;
          let pat = 0;
          wd.forEach((d, i) => { pat |= d << (28 - 4 * i); });
          const mask = n === 8 ? 0xFFFFFFFF : (0xFFFFFFFF << (32 - 4 * n)) >>> 0;
          for (let o = 0; o + n <= width; o++) c += `v += select(0u, 1u, ((w${o} ^ ${hex(pat >>> 0)}) & ${hex(mask)}) == 0u && ${o}u >= zl); `;
        }
        return c + fin("v", cmpCond("v", t));
      }
      case "period": case "up": case "down": {
        const p = t.kind === "period" ? t.period : 1;
        let M;
        if (t.kind === "period") {
          const sh = shifted(X, p);
          M = `(${mask64((k) => `nzf(${X(k)} ^ ${sh(k)})`)} & ${lowClear(64 - width + p)})`;
        } else {
          const cx = (k) => (k >= 0 && k < W ? (t.kind === "down" ? `(~x${k})` : `x${k}`) : "0u");
          const sh = shifted(cx, 1);
          const inc = (x) => `(((${x} & 0x77777777u) + 0x11111111u) ^ (${x} & 0x88888888u))`;
          const f15 = (x) => `(${x} & (${x} << 1u) & (${x} << 2u) & (${x} << 3u) & 0x88888888u)`;
          M = `(${mask64((k) => `(nzf(${sh(k)} ^ ${inc(cx(k))}) & ~${f15(cx(k))})`)} & ${lowClear(64 - width + 1)})`;
        }
        if (!rec) return `ok = nz64(run_ge(${M} & vis, ${t.lo - p}u));`;
        const val = t.kind === "period" ? `select(${p}u, ${p}u * ((lr + ${p}u) / ${p}u), lr > 0u)` : "lr + 1u";
        return `${needOf(recIdx, p)} let mm = ${M} & vis; if (need <= ${p}u || nz64(run_ge(mm, need - ${p}u))) { let lr = lng_run(mm); let v = ${val}; ${claim(line, recIdx, "v")} }`;
      }
      case "pal": {
        let c = `var xa = array<u32, ${W}>(${Array.from({ length: W }, (_, k) => `x${k}`).join(", ")});\n`;
        c += Array.from({ length: W }, (_, k) => `let r${k} = nibrev(x${W - 1 - k});`).join(" ") + "\n";
        const win = (name, o) => {
          const w = o >> 3, sh = 4 * (o & 7);
          const a = w < W ? `${name}${w}` : "0u", b = w + 1 < W ? `${name}${w + 1}` : "0u";
          return sh === 0 ? a : (b === "0u" ? `(${a} << ${sh}u)` : `((${a} << ${sh}u) | (${b} >> ${32 - sh}u))`);
        };
        if (rec) c += needOf(recIdx, 1) + "\n"; else c += `let need = ${t.lo}u;\n`;
        c += "let ko = min(8u, need / 2u); let ke = min(8u, (need + 1u) / 2u); let mo = pmask(ko); let me = pmask(ke);\nvar pv = 1u;\n";
        for (let ctr = 0; ctr < width - 1; ctr++) {
          const t1 = ctr + 1;
          if (ctr >= 1) c += `if (((${win("x", t1)} ^ ${win("r", width - ctr)}) & mo) == 0u) { pv = max(pv, pal_len(&xa, zl, ${ctr}u, true)); }\n`;
          c += `if (((${win("x", t1)} ^ ${win("r", width - 1 - ctr)}) & me) == 0u) { pv = max(pv, pal_len(&xa, zl, ${ctr}u, false)); }\n`;
        }
        return c + (rec ? `{ let v = pv; ${claim(line, recIdx, "v")} }` : `ok = pv >= need;`);
      }
      default: throw new Error(`stat kind ${t.kind}`);
    }
  };
  lines.forEach((line, li) => {
    const recIdx = line.record ? ri++ : -1;
    s += `  {\n    var ok = ${line.record ? "true" : `!hit && PT[${SE + li}u] != 0u`};\n`;
    const plain = line.terms.filter((t) => !t.rec).sort((a, b) => COST[a.kind] - COST[b.kind]);
    for (const t of plain) {
      const code = termCode(t, line, recIdx);
      if (code) s += `    if (ok) {\n${code}\n    }\n`;
    }
    if (line.recTerm) s += `    if (ok) {\n${termCode(line.recTerm, line, recIdx)}\n    }\n`;
    else s += "    if (ok) { hit = true; }\n";
    s += "  }\n";
  });
  return s + "}\n";
}

/** Expression of the leading zero digits (at most 63) of a Mojang hash from its digest word names. */
export function zlExpr(words) {
  let e = "63u";
  for (let k = words.length - 1; k >= 0; k--) e = `select(${e}, ${8 * k}u + (countLeadingZeros(${words[k]}) >> 2u), ${words[k]} != 0u)`;
  return e;
}

/** Initial RB contents: the best of every record line (null: none yet). */
export const recordInit = (lines, bests) => Int32Array.from(lines.filter((l) => l.record).map((l, i) => (bests[i] ?? null) === null ? (l.record === "max" ? -1 : 0x7FFFFFFF) : bests[i]));
