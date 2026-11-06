// Node test of the pattern targets (js/patterns.js) and their plumbing into targets.js and the WGSL generators:
// parser, errors, the reserved stat syntax, the Patterns box (lines, comments, comma lists), matching, odds,
// prefilter masks, and that the prefilter never rejects a true match on random ids.
// (test_patterns.py checks that this file, patterns.py and pattern.h compile identically.)
// Run: node web/test/patterns.mjs
import { parseLine, parsePatternText, PatternError, fmtOdds, DIGITS, LETTERS, ALL, StatLine, Pattern, stripComment } from "../js/patterns.js";
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
for (const [s, want] of [["zeros >= 24", "#0{24,}"], ["palindrome best", "~{25} or ~{max}"], ["run(f) >= 10", "(a digit 12x in a row)"], ["^cafe & zeros >= 10", "#0{24,}"]]) check((errOf(s) || "").includes(want), `old English syntax gives a hint: ${s}`);

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

// ---- stat lines ----
const S = (t, w = 64) => parseLine(t, w);
const ok = (t, w = 64) => errOf(t, w) === null;
for (const t of ["#0{24,}", "#?l{,0}", "#?l{50,}", "#[0-3]{20,}", "#(dead|beef|cafe){3,}", "##{,9}", "##{min}", "#?l{min}", "#0{max}", "(?)\\1{11}", "(??)\\1{6}", "(?{3})\\1{2}",
  "+{12}", "-{12}", "~{25}", "~{max}", "+{max}", "(?)\\1{max}", "^f{max}", "0{max}", "^0{max}", "^dead & #0{10,}", " # 0 { 24 , } ", "#0{5}", "#0{2,6}", "#(DEAD|Beef){2,}", "^dead0{max} & #?d{20,}"])
  check(ok(t), `stat syntax accepted: ${t}`);
for (const [t, want] of [["#", "count what"], ["#0", "missing range"], ["#0{3", "unclosed"], ["#0{x}", "a range is"], ["#0{5,2}", "goes backwards"], ["#0{,}", "a range is"],
  ["##{max}", "nearly every id"], ["+{min}", "{min} is only for counts"], ["~{3,5}", "takes {n}"], ["~{1}", "always holds"], ["~{65}", "needs 65 digits"],
  ["(?)\\2{3}", "only the periodic-run form"], ["(?)", "only the periodic-run form"], ["(?{9})\\1{2}", "period 9"], ["(?{0})\\1{2}", "period 0"], ["(?)\\1{0}", "at least 1"],
  ["#0{3}{4}", "after the range"], ["#(xyz){3,}", "hex words"], ["#(deadbeefc){3,}", "1-8 digits"], ["#@words.txt{4,}", "write the words in the line"],
  ["#0{max} & #1{max}", "at most one"], ["^dead & ", "empty term"], ["0{min}", "{min} is only"], ["^f{max}$", "a run at the end takes a single token"], ["0{max}*1", "{max} repeats"], ["a{2,3}", "ranges are for counts"],
  ["#0{0,}", "always holds"], ["#0{70,}", "can never hold"], ["##{20,}", "can never hold"], ["#0{3,} & & #1{3,}", "empty term"], ["#dead{3,}", "write #("]]) {
  const e = errOf(t);
  check(e !== null && e.includes(want), `stat error for '${t}': ${e}`);
}
check(errOf("#0{24,}", 16).includes("can never hold"), "NameMC: #0{24,} can never hold (16 digits)");
check(/position \d/.test(errOf("#0{5,2}") ?? ""), "range error carries a position");
// the shape of a parsed line
let sl = S("#0{24,} & ^dead");
check(sl instanceof StatLine && sl.terms.length === 2 && sl.record === null && sl.finderText() === "#0{24,}&^dead" && sl.text === "#0{24,}&^dead", "StatLine: terms, normalised text, finder text");
sl = S("~{ max }");
check(sl.record === "max" && sl.terms[0].kind === "pal" && sl.text === "~{max}", "record line");
check(S("#?l{min}").record === "min" && S("(?)\\1{11}").terms[0].lo === 12 && S("(??)\\1{6}").terms[0].lo === 14 && S("(?{3})\\1{2}").terms[0].lo === 9, "periodic run length is p*(n+1)");
check(!(S("^cafe") instanceof StatLine) && S("^f{max}") instanceof StatLine && S("^f{max}").terms[0].prefix.length === 0, "plain hex stays a Pattern, X{max} is a stat line");
check(S("#(dead|beef){3,}").terms[0].words.length === 2 && S("#(dead|dead){3,}").terms[0].words.length === 1, "words: duplicates dropped");
check(stripComment("#0{24,} # note") === "#0{24,} " && stripComment("^dead & #0{3,}") === "^dead & #0{3,}" && stripComment("##{,9}") === "##{,9}" && stripComment("# comment") === "" && stripComment("beef # x") === "beef ",
  "stripComment keeps count terms, drops comments");
const sbox = parsePatternText("#0{24,}, ##{,9}\n~{max}  # best palindrome\n# only a comment\n^dead & #0{10,}\n#0{24,}\nbad{\n", 64);
check(sbox.stats.length === 4 && sbox.patterns.length === 0 && sbox.errors.length === 1 && sbox.lines[0].stat && sbox.lines[1].stat, "Patterns box: stat lines, comma lists, duplicates, comments");
check(parsePatternText("cafe\n#0{20,}", 64).patterns.length === 1 && parsePatternText("cafe\n#0{20,}", 64).stats.length === 1, "Patterns box: hex patterns and stat lines are kept apart");

// reference values on crafted ids
const ev = (t, h) => parseLine(t, h.length).evaluate(h);
const pad = (s, w) => (s + "1357".repeat(w)).slice(0, w);   // filler without runs, stairs or palindromes
check(!ev("#0{6,}", pad("000a000b00", 64)).ok && ev("#0{5,}", pad("000a000b00", 64)).ok, "count: leading zeros are not counted (Mojang)");
check(ev("#0{8,}", pad("0000abc0000d", 16)).ok && !ev("#0{9,}", pad("0000abc0000d", 16)).ok, "count: NameMC counts the 16 digits as shown");
check(ev("+{6}", pad("a123456", 64)).ok && !ev("+{8}", pad("a123456", 64)).ok, "staircase up of 6 digits");
check(!ev("+{4}", pad("1def", 64)).ok && ev("+{3}", pad("1def", 64)).ok && !ev("+{3}", pad("1ef0", 64)).ok, "no wrap f -> 0");
check(ev("-{5}", pad("1fedcb9", 64)).ok && ev("-{max}", pad("1fedcb9", 64)).value === 5, "staircase down, record value");
check(ev("(?)\\1{5}", pad("a77777777b", 64)).ok && !ev("(?)\\1{8}", pad("a77777777b", 64)).ok && ev("(?)\\1{max}", pad("a77777777b", 64)).value === 8, "same digit in a row");
check(ev("(??)\\1{max}", pad("17c7c7c7c7c1", 64)).value === 10 && ev("(??)\\1{4}", pad("17c7c7c7c7c1", 64)).ok, "zigzag 7c7c7c7c7c");
check(ev("~{max}", pad("0001234321", 64)).value >= 7, "palindrome value");
check(ev("##{,3}", "1212121212121212").ok && !ev("##{,3}", "1234121212121212").ok && ev("##{min}", "1234121212121212").value === 4, "different digits");
check(ev("#(dead|beef){3,}", pad("0000deadbeefdead", 64)).ok && ev("#(aa){max}", pad("1aaaaaa", 64)).value === 5, "word count, overlaps count");
check(ev("^f{max}", "fff" + "1".repeat(61)).value === 3 && ev("^f{max}", "1" + "f".repeat(63)).value === 0 && ev("0{max}", pad("10000", 64)).value === 4, "hex X{max}");
check(ev("^dead & #0{3,}", "dead" + "0".repeat(10) + "1".repeat(50)).ok && !ev("^dead & #0{3,}", "beef" + "0".repeat(10) + "1".repeat(50)).ok, "A & B");

// reference against naive models on random ids
let seed2 = 99;
const rnd2 = (n) => { seed2 = (Math.imul(seed2, 1664525) + 1013904223) >>> 0; return (seed2 >>> 8) % n; };
const H2 = "0123456789abcdef";
const naiveRuns = (v, f) => { let b = 0, r = 0; for (let i = 0; i < v.length - 1; i++) { r = f(v[i], v[i + 1]) ? r + 1 : 0; b = Math.max(b, r); } return b; };
let bad = "";
for (let it = 0; it < 3000 && !bad; it++) {
  const w = it % 2 ? 16 : 64;
  const h = [...Array(w)].map(() => (rnd2(3) === 0 ? rnd2(3) : rnd2(16)));
  if (w === 64 && rnd2(3) === 0) for (let i = 0, z = 1 + rnd2(8); i < z; i++) h[i] = 0;
  if (rnd2(3) === 0) { const a = rnd2(w - 9), d = rnd2(16); for (let i = 0; i < 8; i++) h[a + i] = (d + (it % 4 === 1 ? i : it % 4 === 2 ? -i : 0) + 16) % 16; }
  const hex = h.map((x) => H2[x]).join("");
  let z = 0; if (w === 64) while (z < 63 && h[z] === 0) z++;
  const v = h.slice(z);
  const same = naiveRuns(v, (a, b) => a === b) + 1;
  const up = naiveRuns(v, (a, b) => b === a + 1) + 1, dn = naiveRuns(v, (a, b) => b === a - 1) + 1;
  let pal = 1; for (let a = 0; a < v.length; a++) for (let b = a + 1; b <= v.length; b++) { const s = v.slice(a, b); if (b - a > pal && s.every((x, i) => x === s[s.length - 1 - i])) pal = b - a; }
  const got = { same: ev("(?)\\1{max}", hex).value, up: ev("+{max}", hex).value, dn: ev("-{max}", hex).value, pal: ev("~{max}", hex).value,
    zeros: ev("#0{max}", hex).value, letters: ev("#?l{max}", hex).value, dist: ev("##{min}", hex).value, w3: ev("#(a|bc|123){max}", hex).value };
  const want = { same, up, dn, pal, zeros: v.filter((x) => x === 0).length, letters: v.filter((x) => x >= 10).length, dist: new Set(v).size,
    w3: (() => { let n = 0; for (let o = 0; o < v.length; o++) { if (v[o] === 10) n++; if (v[o] === 11 && v[o + 1] === 12) n++; if (v[o] === 1 && v[o + 1] === 2 && v[o + 2] === 3) n++; } return n; })() };
  // up/down staircase: the f/0 wrap never counts (naive b === a + 1 already excludes it)
  for (const k of Object.keys(want)) if (got[k] !== want[k]) bad = bad || `${k} on ${hex}: ${got[k]} vs naive ${want[k]}`;
  for (const [t, pv] of [["#0{3,}", want.zeros >= 3], ["+{4}", up >= 4], ["~{9}", pal >= 9], ["(?)\\1{3}", same >= 4], ["#?l{,12}", want.letters <= 12]]) {
    if (ev(t, hex).ok !== pv) bad = bad || `${t} on ${hex}: ${ev(t, hex).ok} vs ${pv}`;
  }
}
check(!bad, `reference values equal naive models on 3000 random ids ${bad}`);

// odds: the numbers patterns.py gives
const near = (a, b) => Math.abs(a - b) <= 1e-9 * b;
for (const [w, t, pv] of [[64, "#0{24,}", 1.7397628041271516e-13], [64, "#?l{,5}", 5.9157009017784315e-08], [64, "##{,12}", 1.8095941990222714e-05], [64, "+{8}", 1.1269002673315155e-07],
  [64, "(?)\\1{7}", 1.9907018538239782e-07], [64, "(??)\\1{6}", 1.6986412276764072e-13], [64, "~{25}", 1.5051663619183656e-13], [64, "#(dead|beef|cafe){3,}", 3.60955612016206e-09],
  [16, "#0{6,}", 0.0002760726770940876], [16, "~{10}", 1.2397689261296547e-05], [16, "(?)\\1{4}", 0.00017260922596826125], [64, "#[0-3]{40,}", 2.239660037203474e-10],
  [16, "-{6}", 6.839627105237334e-06], [64, "^dead & #0{10,}", 8.322039522529295e-08]]) check(near(S(t, w).probability(), pv), `odds of ${t} (${w}) = Python: ${S(t, w).probability()}`);
check(near(S("~{max}").probability(18), 7.092846011710967e-10) && near(S("~{max}").probability(10), 5.458841728990572e-05) && near(S("#?l{min}").probability(18), 0.04422573655694634) && S("~{max}").probability() === 1, "record odds: chance of beating the best");

// targets: stat lines are targets; a record line matches any id it has a value for (the page compares with its best)
const st = targetList({ prefixes: [], suffixes: [], patterns: [], stats: parsePatternText("#0{3,}\n~{max}", 64).stats, zeros: 0 });
check(st.map(targetKey).join() === "stat:#0{3,},record:~{max}", "targetList: threshold stat lines and record lines");
check(matchTargets("1".repeat(64), st).map(targetKey).join() === "record:~{max}" && matchTargets("1".repeat(61) + "000", st).length === 2, "matchTargets: stat, record");
check(near(expectedTries(st[0]), 1 / S("#0{3,}").probability()), "expected tries of a stat line");

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
