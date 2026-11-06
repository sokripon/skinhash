// Node test of the pattern targets (js/patterns.js) and their plumbing into targets.js and the WGSL generators:
// parser, errors, the reserved stat syntax, the Patterns box (lines, comments, comma lists), matching, odds,
// prefilter masks, and that the prefilter never rejects a true match on random ids.
// (test_patterns.py checks that this file, patterns.py and pattern.h compile identically.)
// Run: node web/test/patterns.mjs
import { parseLine, parsePatternText, PatternError, fmtOdds, DIGITS, LETTERS, ALL } from "../js/patterns.js";
import { targetList, matchTargets, expectedTries, gpuTargets, gpuNmTargets, anchoredTargets, needsSliding, targetKey } from "../js/targets.js";
import { hitCode, anchoredCode, AE } from "../js/wgsl.js";

let fails = 0;
const check = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) fails++; };
const errOf = (t, w = 64) => { try { parseLine(t, w); return null; } catch (e) { if (!(e instanceof PatternError)) throw e; return e.message; } };
const m = (t, h) => parseLine(t, h.length).match(h);

// syntax
let p = parseLine("C0FFee", 64);
check(p.text === "c0ffee" && p.pieces[0].length === 6 && !p.start && !p.end, "hex literal, case-insensitive");
p = parseLine("^?d?l?[0-3]", 64);
check(p.start && p.pieces[0][0] === DIGITS && p.pieces[0][1] === LETTERS && p.pieces[0][2] === ALL && p.pieces[0][3] === 0xF, "^ ?d ?l ? [0-3]");
p = parseLine("[05af]{3}0{4}*beef$", 64);
check(p.end && p.pieces.length === 2 && p.pieces[0].length === 7 && p.pieces[0][0] === (1 | 1 << 5 | 1 << 10 | 1 << 15), "{n} on a class and a digit, * and $");
for (const w of ["cafe", "dead$", "^beef", "face*fade"]) check(errOf(w) === null, `hex word starting with a letter: ${w}`);
for (const [t, want] of [["", "empty"], ["x", "'x' at position 1 is not a hex digit"], ["{0}", "at least 1"], ["a{0}", "at least 1"],
  ["{2}", "nothing to repeat"], ["[", "unbalanced '['"], ["[]", "empty class"], ["[5-2]", "backwards"], ["a^", "only allowed as the first"],
  ["$a", "only allowed as the last"], ["*a", "at the start does nothing"], ["a*", "at the end does nothing"], ["a**b", "follows another"],
  ["0{65}", "longer than the id"], ["^0{12}$", "exactly 64 digits"], ["a b", "space at position 2"]]) {
  const e = errOf(t);
  check(e !== null && e.includes(want), `error for '${t}': ${e}`);
}
for (const s of ["zeros >= 24", "palindrome best", "run(f) >= 10", "^cafe & zeros >= 10"]) check(/not supported yet/.test(errOf(s) || ""), `reserved: ${s}`);

// matching
const Z = "000abc123000" + "d".repeat(52);
check(m("^000abc", Z) && !m("^abc", Z) && m("abc", Z) && !m("0a", Z) && m("^0{3}*d{4}$", Z), "Mojang: ^ on the full hash, unanchored in the link id");
check(m("^0{4}", "0000beef12345678") && !m("^beef", "0000beef12345678") && m("ef12*78$", "0000beef12345678"), "NameMC: the 16 digits");
check(m("0", "0".repeat(64)) && !m("00", "0".repeat(64)), "all-zero hash: id 0");

// odds and format
check(Math.abs(parseLine("^0{12}", 64).probability() - Math.pow(16, -12)) < 1e-25 && fmtOdds(Math.pow(16, -12)) === "~1 in 2.8e14", "odds ^0{12} ~1 in 2.8e14");
check(fmtOdds(1 / 256) === "~1 in 256" && Math.abs(parseLine("^?l", 64).probability() - 6 / 16) < 1e-12, "odds of a class, small odds format");

// prefilter
let f = parseLine("^[0-7]?l?d", 64).prefilter();
check(f.kind === "anchored" && f.mask.slice(0, 3).join() === "8,8,0" && f.val.slice(0, 3).join() === "0,8,0", "class masks [0-7] 8/0, ?l 8/8, ?d none");
const ww = parseLine("^cafe*babe$", 64).words();
check(ww.mw[0] === 0xFFFF0000 && ww.vw[0] === 0xCAFE0000 && ww.mw[7] === 0xFFFF && ww.vw[7] === 0xBABE, "^cafe*babe$ words");
f = parseLine("^?l*c0ffee", 64).prefilter();
check(f.kind === "sliding" && f.p1 === 0xC0FFEE00 && f.lo === 1 && !f.usezl, "^?l*c0ffee slides c0ffee after the first digit");

// the Patterns box
const box = parsePatternText("# comment\n^cafe  # note\n\nbeef$, ^0{8}\nxyz\n^CAFE\n", 64);
check(box.patterns.map((x) => x.text).join() === "^cafe,beef$,^0{8}" && box.errors.length === 1 && /line 5: pattern 'xyz'/.test(box.errors[0]),
  "Patterns box: comments, blank lines, comma lists, per-line errors, duplicates dropped");
check(box.lines.length === 5 && box.lines[3].error && box.lines[0].pattern, "Patterns box: one entry per pattern with its error or Pattern");

// targets and GPU plumbing
const tl = targetList({ prefixes: [], suffixes: [], patterns: parsePatternText("^0{6}\nc0ffee\n^cafe*babe$", 64).patterns, zeros: 0 });
check(tl.map(targetKey).join() === "match:^0{6},match:c0ffee,match:^cafe*babe$", "targetList adds match targets");
check(matchTargets("000000" + "1".repeat(58), tl).map(targetKey).join() === "match:^0{6}", "matchTargets on the full hash");
check(Math.abs(expectedTries(tl[0]) - Math.pow(16, 6)) < 1e-3, "expected tries of a pattern");
const an = anchoredTargets(tl);
check(an.length === 2 && an[0].key === "match:^0{6}" && an[0].mw[0] === 0xFFFFFF00 && needsSliding(tl, 64), "anchored list and sliding need");
const gt = gpuTargets(tl);
check(gt.cont.length === 1 && gt.cont[0].p1 === 0xC0FFEE00 && gt.cont[0].lo === (0 | 1 << 16) && gt.anchKeys.length === 3, "gpuTargets: sliding entry with the zero rule");
const nmt = targetList({ prefixes: ["ab"], suffixes: [], patterns: parsePatternText("^0{6}\nbeef", 16).patterns, zeros: 0 });
const ng = gpuNmTargets(nmt);
check(ng.pats.length === 2 && ng.pats[1].mh === 0xFFFFFF00 && ng.pats[1].vh === 0 && ng.cont.length === 1 && ng.cont[0].lo === 0 && ng.cont[0].lim === 12,
  "gpuNmTargets: anchored pattern joins the prefix table, sliding the contains table");
const code = anchoredCode(an, ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]);
check(code.includes(`PT[${AE}u] != 0u && (((s0 ^ 0x0u) & 0xffffff00u)) == 0u`) && !/s[1-6] \^/.test(code) && code.includes("s7 ^ 0xbabeu"),
  "anchoredCode: literals, only words with a mask, a run-time enable word per entry");
check(hitCode(false, an).includes("if (!hit) {") && !hitCode(false, []).includes(`PT[${AE}u]`), "hitCode bakes the anchored check only when there is one");

// prefilter soundness on random ids with planted matches
let seed = 7;
const rnd = (n) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed >>> 8) % n; };
const H = "0123456789abcdef";
const toks = () => { const k = rnd(10); if (k < 5) return H[rnd(16)]; if (k === 5) return "?"; if (k === 6) return "?d"; if (k === 7) return "?l"; const a = rnd(15), b = a + 1 + rnd(15 - a); return `[${H[a]}-${H[b]}]`; };
let tot = 0, pos = 0, miss = 0, first = "";
for (const width of [64, 16]) {
  for (let r = 0; r < 400; r++) {
    let text = "";
    const np = 1 + rnd(2);
    for (let i = 0; i < np; i++) { if (i) text += "*"; for (let k = 0, n = 1 + rnd(4); k < n; k++) text += toks() + (rnd(5) === 0 ? `{${2 + rnd(4)}}` : ""); }
    if (rnd(3) === 0) text = "^" + text;
    if (rnd(4) === 0) text += "$";
    let pat;
    try { pat = parseLine(text, width); } catch { continue; }
    for (let k = 0; k < 10; k++) {
      const h = [...Array(width)].map(() => rnd(16));
      if (rnd(3) === 0) for (let i = 0, z = 1 + rnd(12); i < z; i++) h[i] = 0;
      if (rnd(2)) {   // plant: pieces back to back at a random place, anchors honoured
        const total = pat.pieces.reduce((a, x) => a + x.length, 0);
        let at = pat.start ? 0 : rnd(width - total + 1);
        if (pat.end && !pat.start) at = width - total;
        pat.pieces.forEach((piece) => piece.forEach((s) => { const vals = [...Array(16).keys()].filter((v) => (s >> v) & 1); h[at++] = vals[rnd(vals.length)]; }));
        if (pat.end && pat.start && pat.pieces.length > 1) { const last = pat.pieces[pat.pieces.length - 1]; let a2 = width - last.length; last.forEach((s) => { const vals = [...Array(16).keys()].filter((v) => (s >> v) & 1); h[a2++] = vals[rnd(vals.length)]; }); }
      }
      const hex = h.map((v) => H[v]).join("");
      const want = pat.match(hex);
      tot++; pos += want;
      if (want && !pat.prefilterPass(hex)) { miss++; first = first || `${pat.text} on ${hex}`; }
    }
  }
}
check(miss === 0 && pos > tot / 6, `prefilter never rejects a true match: ${tot} ids, ${pos} matches ${first}`);

console.log(fails ? `\n${fails} FAILURE(S)` : "\nALL PATTERN TESTS PASSED");
process.exit(fails ? 1 : 0);
