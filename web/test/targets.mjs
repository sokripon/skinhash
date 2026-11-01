// Node test of the target logic, including the "contains" target: parsing, the trimmed-id rule, the GPU pattern
// words and the generated WGSL. Run: node web/test/targets.mjs
import { parseHexList, targetList, matchTargets, expectedTries, gpuContains, gpuTargets, gpuNmTargets } from "../js/targets.js";
import { hitCode } from "../js/wgsl.js";

let fails = 0;
const check = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) fails++; };

// parsing: any hex digits, leading 0 allowed, 1-16 digits
let r = parseHexList("beef, 0000 ABC", "contains");
check(r.list.join() === "beef,0000,abc" && !r.errors.length, "contains list parses hex, keeps a leading 0");
r = parseHexList("xyz", "contains");
check(r.errors.length === 1 && /not hexadecimal/.test(r.errors[0]), "non-hex is an error");
r = parseHexList("0123456789abcdef0", "contains");
check(r.errors.length === 1 && /longer than 16/.test(r.errors[0]), "17 digits is an error");
r = parseHexList("0123456789abcdef", "contains");
check(r.list.length === 1 && !r.errors.length, "16 digits is fine");

// matching: the trimmed id for Mojang, the 16 digits for NameMC
const contains = (text) => [{ kind: "contains", text }];
const full = "000abc123000" + "d".repeat(52);
check(matchTargets(full, contains("abc")).length === 1, "Mojang: abc is in the trimmed id");
check(matchTargets(full, contains("00a")).length === 0 && matchTargets(full, contains("0a")).length === 0, "Mojang: a match inside the dropped zeros does not count");
check(matchTargets(full, contains("c1230")).length === 1 && matchTargets(full, contains("000d")).length === 1, "Mojang: inner zeros are kept");
check(matchTargets("0".repeat(64), contains("0")).length === 1 && matchTargets("0".repeat(64), contains("00")).length === 0, "Mojang: an all-zero digest is the id 0");
check(matchTargets("0000beef12345678", contains("0000"), true).length === 1 && matchTargets("0000beef12345678", contains("beef0"), true).length === 0, "NameMC: the 16 digits, leading zeros count");

// target list and expected tries
const tl = targetList({ prefixes: ["c"], suffixes: ["d"], contains: ["beef"], zeros: 2 });
check(tl.map((t) => `${t.kind}:${t.text}`).join() === "prefix:c,suffix:d,contains:beef,zeros:2", "targetList order");
check(Math.abs(expectedTries({ kind: "contains", text: "beef" }, true) - 65536 / 13) < 1e-6, "expected tries NameMC: 16^4 / (16-4+1)");
check(Math.abs(expectedTries({ kind: "contains", text: "beef" }, false) - 65536 / 57) < 1e-6, "expected tries Mojang: 16^4 / (60-4+1)");

// GPU pattern words
const g = gpuContains({ kind: "contains", text: "0123456789abcdef" }, 64);
check(g.p1 === 0x01234567 && g.p2 === 0x89abcdef && g.m1 === 0xffffffff && g.m2 === 0xffffffff && g.lim === 48, "gpuContains 16 digits");
const g3 = gpuContains({ kind: "contains", text: "bee" }, 16);
check(g3.p1 === 0xbee00000 && g3.m1 === 0xfff00000 && g3.p2 === 0 && g3.m2 === 0 && g3.lim === 13, "gpuContains 3 digits, left-aligned");
const gt = gpuTargets([...tl]);
check(gt.pats.length === 2 && gt.cont.length === 1 && gt.zeros === 2, "gpuTargets splits prefix/suffix, contains and zeros");
check(gpuNmTargets(tl).cont[0].lim === 12, "gpuNmTargets uses 16 digits");

// the WGSL: absent without contains, present with it, windows reach every offset
check(!/contains/.test(hitCode(false)) && !hitCode(false).includes("mn = min"), "no contains code in the kernel when no contains target exists");
const wc = hitCode(true);
check((wc.match(/mn = min/g) || []).length === 64 && /63u <= lim && 63u >= zn/.test(wc), "contains code: 64 offsets and the leading-zero test against zn");

console.log(fails ? `\n${fails} FAILURE(S)` : "\nALL TARGET TESTS PASSED");
process.exit(fails ? 1 : 0);
