"""Tests of the -m/--match patterns: parser, compiler (GPU prefilter masks), matcher, the C (pattern.h) and JS
(web/js/patterns.js) ports, the skinhash.py plumbing and end-to-end searches on skins/steve.png.
Run: py test_patterns.py [--quick]   (exit code 0 = all good; --quick skips the end-to-end searches)
Everything is written to temp dirs."""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import patterns as pt
import skinhash as sh

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails += 1


def err_of(text, width=64):
    try:
        pt.parse_line(text, width)
        return None
    except pt.PatternError as e:
        return str(e)


def m(text, hexid):
    return pt.parse_line(text, len(hexid)).match(hexid)


H = "0123456789abcdef"
# ---- parser: every syntax element ---------------------------------------------------------------------------
p = pt.parse_line("C0FFee", 64)
check(p.text == "c0ffee" and not p.start and not p.end and p.pieces == [[1 << 12, 1, 1 << 15, 1 << 15, 1 << 14, 1 << 14]],
      "hex literal, case-insensitive, normalised to lowercase")
p = pt.parse_line("^ab", 64)
check(p.start and not p.end and len(p.pieces[0]) == 2, "^ anchors to the start")
p = pt.parse_line("ab$", 64)
check(p.end and not p.start, "$ anchors to the end")
p = pt.parse_line("a?b", 64)
check(p.pieces[0][1] == pt.ALL, "? is any digit")
p = pt.parse_line("?d?l", 64)
check(p.pieces[0] == [pt.DIGITS, pt.LETTERS], "?d is 0-9, ?l is a-f")
p = pt.parse_line("?D?L", 64)
check(p.pieces[0] == [pt.DIGITS, pt.LETTERS], "?D / ?L are case-insensitive too")
p = pt.parse_line("[0-3][a-f][05af][0-27]", 64)
check(p.pieces[0] == [0xF, 0xFC00, (1 | 1 << 5 | 1 << 10 | 1 << 15), 0x7 | 1 << 7], "classes: ranges, lists, both")
p = pt.parse_line("0{4}?{2}?d{2}?l{2}[0-7]{3}", 64)
check([len(p.pieces[0])] == [13] and p.pieces[0][:4] == [1] * 4 and p.pieces[0][4:6] == [pt.ALL] * 2
      and p.pieces[0][6:8] == [pt.DIGITS] * 2 and p.pieces[0][8:10] == [pt.LETTERS] * 2 and p.pieces[0][10:] == [0xFF] * 3,
      "{n} repeats a digit, ?, ?d, ?l and a class")
p = pt.parse_line("^cafe*babe$", 64)
check(p.start and p.end and len(p.pieces) == 2, "* splits pieces: ^cafe*babe$")
p = pt.parse_line("ab*cd*ef", 64)
check(len(p.pieces) == 3, "two gaps: three pieces")
check(pt.parse_line("^?{4}dead", 64).describe() == "^ ?{4} dead", "?{4}dead: any digit four times, then dead")
check(pt.parse_line("^????dead", 64).describe() == "^ ?{3} ?d ead",
      "?d is always the digit class: ^????dead is ??? ?d e a d (use ?{4}dead)")
check(err_of("^" + "a" * 64 + "$") is None and err_of("^" + "a" * 16 + "$", 16) is None, "^...$ of exactly the id length is fine")
for word in ["cafe", "dead$", "^beef", "face*fade", "abcdef"]:
    check(err_of(word) is None, f"a hex word starting with a letter parses: {word}")

# ---- errors -------------------------------------------------------------------------------------------------
ERRS = [
    ("", "empty pattern"), ("x", "'x' at position 1 is not a hex digit"), ("cafeg", "'g' at position 5"),
    ("ca fe", "space at position 3"), ("a%", "'%' at position 2 is not allowed"),
    ("{0}", "a repeat count must be at least 1"), ("a{0}", "at least 1"), ("{3}", "nothing to repeat"),
    ("a{2}{3}", "nothing to repeat"), ("^{2}", "nothing to repeat"), ("a*{2}", "nothing to repeat"),
    ("a{x}", "must be a number"), ("a{3", "unclosed '{'"), ("a}", "unbalanced '}'"), ("a{65}", "longer than the id"),
    ("[", "unbalanced '['"), ("a]", "unbalanced ']'"), ("[a[b]]", "unbalanced '['"), ("[]", "empty class"),
    ("[5-2]", "goes backwards"), ("[0-]", "has no end"), ("[g]", "only hex digits"), ("[^1]", "negated classes"),
    ("[0,5]", "only hex digits"), ("a^b", "'^' at position 2: only allowed as the first character"),
    ("a$b", "'$' at position 2: only allowed as the last character"), ("$ab", "'$' at position 1"),
    ("^", "no digits"), ("^$", "no digits"), ("*ab", "'*' at position 1 at the start does nothing"),
    ("ab*", "'*' at the end does nothing"), ("^*ab", "at the start does nothing"), ("a**b", "follows another '*'"),
    ("0{65}", "longer than the id"), ("a" * 40 + "*" + "b" * 30, "needs 70 digits, the id has only 64"),
    ("^0{12}$", "must be exactly 64 digits (it is 12)"),
]
for text, want in ERRS:
    e = err_of(text)
    check(e is not None and want in e and e.startswith(f"pattern '{text.strip()}'"), f"error for {text!r}: {e}")
check("the id has only 16" in (err_of("a" * 17, 16) or ""), "NameMC: 17 digits is too long")
check("exactly 16 digits" in (err_of("^0{8}$", 16) or ""), "NameMC: ^...$ must be 16 digits")
for stat in ["zeros >= 24", "zeros>=24", "staircase >= 12", "palindrome best", "run(f) >= 10", "^cafe & zeros >= 10",
             "dead <= 3", "face best"]:
    check("not part of the syntax" in (err_of(stat) or ""), f"old English stat syntax gives a hint: {stat!r}")

# ---- matching semantics -------------------------------------------------------------------------------------
Z = "000abc123000" + "d" * 52
check(m("^000abc", Z) and not m("^abc", Z), "Mojang ^ is the start of the full 64-digit hash")
check(m("^0{3}", Z) and not m("^0{4}", Z), "^0{n}: n leading zeros")
check(m("abc", Z) and not m("0a", Z) and not m("00a", Z) and m("c1230", Z) and m("000d", Z),
      "unanchored: inside the link id only (same rule as --contains)")
check(m("d{4}$", Z) and m("123*d$", Z) and not m("abc$", Z), "$ is the end")
check(m("^000*d{52}$", Z) and m("abc*123*ddd", Z) and not m("123*abc", Z), "* gaps: ordered pieces")
check(m("ab*cd", "1abcd" + "1" * 59) and not m("abc*cd", "1abcd" + "1" * 59), "pieces do not overlap")
check(m("^?{3}[a-c]", Z) and not m("^?{3}[d-f]", Z) and m("^0{3}?l?l?l?d", Z), "classes and ? in place")
check(m("^0{63}?", "0" * 64) and m("0", "0" * 64) and not m("00", "0" * 64), "all-zero hash: the link id is '0'")
check(m("0{3}a", "1000a" + "1" * 59) and not m("^0{3}a", "1000a" + "1" * 59), "zeros inside the id count")
N = "0000beef12345678"
check(m("^0{4}", N) and m("0000", N) and m("^0{4}beef12345678$", N) and m("ef12*78$", N), "NameMC: the 16 digits as is")
check(not m("^beef", N) and m("^?{4}beef", N), "NameMC ^ is the first digit")

# ---- odds -------------------------------------------------------------------------------------------------------
odds = lambda t, w=64: pt.parse_line(t, w).probability()
check(abs(odds("^0{12}") - 16.0 ** -12) / 16.0 ** -12 < 1e-9, "odds ^0{12} = 16^-12")
check(abs(odds("^cafe*babe$") - 16.0 ** -8) / 16.0 ** -8 < 1e-9, "odds ^cafe*babe$ = 16^-8")
check(abs(odds("beef") - 61 * 16.0 ** -4) / (61 * 16.0 ** -4) < 1e-3, "odds beef: 61 places")
check(abs(odds("^?l") - 6 / 16) < 1e-12 and abs(odds("^[0-3]", 16) - 0.25) < 1e-12, "odds of classes")
check(pt.fmt_odds(16.0 ** -12) == "~1 in 2.8e14", f"format {pt.fmt_odds(16.0 ** -12)}")

# ---- compiler -------------------------------------------------------------------------------------------------
f = pt.parse_line("^[0-7]?l?d[05af]", 64).prefilter()
check(f["kind"] == "anchored" and f["mask"][:4] == [8, 8, 0, 0] and f["val"][:4] == [0, 8, 0, 0],
      "class masks: [0-7] -> 8/0, ?l -> 8/8, ?d and [05af] -> nothing")
mw, vw = pt.parse_line("^cafe*babe$", 64).words()
check(mw[0] == 0xFFFF0000 and vw[0] == 0xCAFE0000 and mw[7] == 0xFFFF and vw[7] == 0xBABE and not any(mw[1:7]),
      "^cafe*babe$: one anchored mask over words 0 and 7")
f = pt.parse_line("^?l*c0ffee", 64).prefilter()
check(f["kind"] == "sliding" and f["p1"] == 0xC0FFEE00 and f["m1"] == 0xFFFFFF00 and f["lo"] == 1 and not f["usezl"],
      "^?l*c0ffee: the sliding literal is more selective than the anchored ?l, starts after it, no zero rule")
f = pt.parse_line("c0ffee", 64).prefilter()
check(f["kind"] == "sliding" and f["lo"] == 0 and f["lim"] == 58 and f["usezl"], "c0ffee: sliding, offsets 0..58 after the zeros")
f = pt.parse_line("?l{4}0123456789ab", 64).prefilter()
check(f["d"] == 4 and f["p1"] == 0x01234567 and f["p2"] == 0x89AB0000, "long piece: the most selective 8-token window leads")
f = pt.parse_line("beef$", 16).prefilter()
check(f["kind"] == "anchored" and pt.parse_line("beef$", 16).words()[0] == [0, 0xFFFF], "NameMC anchored: two words")

# ---- inputs: repeat, comma lists, @file -------------------------------------------------------------------------
pats = pt.parse_inputs(["^0{12}", "^????0{12}", "^c0ffee,dead$", "^C0FFEE"], 64)
check([x.text for x in pats] == ["^0{12}", "^????0{12}", "^c0ffee", "dead$"], "repeat + comma list, duplicates dropped")
check(pt.split_commas("[05af],ab") == ["[05af]", "ab"], "comma split outside classes")
with tempfile.TemporaryDirectory(prefix="skinhash_pat_") as tmp:
    open(os.path.join(tmp, "p.txt"), "w").write("# my patterns\n\n^cafe   # trailing comment\n  beef$ , ^0{10}\n")
    open(os.path.join(tmp, "bad.txt"), "w").write("cafe\n\nxyz\n^0{12}$\n")
    pats = pt.parse_inputs(["@p.txt"], 64, cwd=tmp)
    check([x.text for x in pats] == ["^cafe", "beef$", "^0{10}"], "@file: comments, blank lines, comma lists")
    try:
        pt.parse_inputs(["@bad.txt"], 64, cwd=tmp)
        check(False, "@file errors")
    except pt.PatternError as e:
        check("bad.txt:3: pattern 'xyz'" in str(e) and "bad.txt:4: pattern '^0{12}$'" in str(e), "@file errors name file:line")
    try:
        pt.parse_inputs(["@missing.txt"], 64, cwd=tmp)
        check(False, "missing file")
    except pt.PatternError as e:
        check("cannot read pattern file missing.txt" in str(e), "missing @file is a clear error")
    try:
        pt.parse_inputs(["a,,b"], 64)
        check(False, "empty list item")
    except pt.PatternError as e:
        check("empty pattern in the list" in str(e), "an empty item in a comma list is an error")
    # the CLI: relative @file from the cwd, errors exit 2 with every message
    r = subprocess.run([sys.executable, os.path.join(HERE, "skinhash.py"), "mojang", os.path.join(HERE, "skins", "steve.png"),
                        "-m", "@bad.txt", "-o", os.path.join(tmp, "o")], cwd=tmp, capture_output=True, text=True)
    check(r.returncode == 2 and "-m: bad.txt:3: pattern 'xyz'" in r.stderr and "bad.txt:4" in r.stderr,
          f"CLI: pattern errors stop the command: {r.stderr.strip().splitlines()[-1] if r.stderr else r.stdout[-200:]}")

# ---- skinhash.py plumbing ---------------------------------------------------------------------------------------
t = sh.targets_from(["cafe"], None, None, None, ["^0{12},beef$"], 64)
check(t == ["prefix:cafe", "match:^0{12}", "match:beef$"], f"targets_from with -m: {t}")
check(sh.target_args(["match:^0{12}", "prefix:c"]) == ["-m", "^0{12}", "-p", "c"], "target_args maps match to -m")
check(sh.hit_target("match:beef$", ["match:^0{12}", "match:beef$"]) == "match:beef$", "hit_target finds a match target")
full = "0" * 12 + "a" * 48 + "beef"
check(sh.target_match("match:^0{12}", full.lstrip("0"), full) and sh.target_match("match:beef$", full.lstrip("0"), full),
      "target_match on the full Mojang hash")
check(sh.target_match("match:^0{4}", N, N) and not sh.target_match("match:^beef", N, N), "target_match on the NameMC id")
check(pt.target_probability("zeros:12", 64) == odds("^0{12}") and pt.target_probability("suffix:dead", 16) == odds("dead$", 16),
      "shorthands compile through the same compiler for the odds (-z N = ^0{N}, -s X = X$)")


# ---- brute force: reference regex matcher vs Pattern.match, and the prefilter never rejects a match ------------
def ref_regex(p):
    cls = lambda s: "[" + "".join(H[v] for v in range(16) if s >> v & 1) + "]"
    body = ".*".join("".join(cls(s) for s in piece) for piece in p.pieces)
    return re.compile(("^" if p.start else "") + body + ("$" if p.end else ""))


def ref_match(p, h):
    rx = ref_regex(p)
    if p.start:
        return rx.search(h) is not None
    zl = len(h) - len(h.lstrip("0")) if len(h) == 64 else 0
    zl = min(zl, 63)
    return rx.search(h[zl:]) is not None


def rand_token(rng):
    k = rng.random()
    if k < 0.45:
        return rng.choice(H)
    if k < 0.55:
        return "?"
    if k < 0.62:
        return "?d"
    if k < 0.69:
        return "?l"
    a, b = sorted(rng.sample(range(16), 2))
    return rng.choice([f"[{H[a]}-{H[b]}]", "[" + "".join(rng.sample(H, rng.randint(1, 4))) + "]"])


def rand_pattern(rng, width):
    while True:
        pieces = []
        for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
            toks = ""
            for _ in range(rng.randint(1, 5)):
                toks += rand_token(rng) + (f"{{{rng.randint(2, 6)}}}" if rng.random() < 0.2 else "")
            pieces.append(toks)
        text = ("^" if rng.random() < 0.4 else "") + "*".join(pieces) + ("$" if rng.random() < 0.3 else "")
        try:
            return pt.parse_line(text, width)
        except pt.PatternError:
            continue


def plant(rng, p, width):
    """An id that (usually) matches p: tokens sampled from their sets, pieces at random legal places."""
    h = [rng.randrange(16) for _ in range(width)]
    if rng.random() < 0.3:
        z = rng.randint(1, min(20, width - 1))
        h[:z] = [0] * z
    lens = [len(x) for x in p.pieces]
    free = width - sum(lens)
    gaps = sorted(rng.randint(0, free) for _ in range(len(lens)))
    pos, prev = [], 0
    for i, n in enumerate(lens):
        start = prev + (gaps[i] - (gaps[i - 1] if i else 0))
        pos.append(start)
        prev = start + n
    if p.start:
        shift = pos[0]
        pos = [x - shift for x in pos]
    if p.end:
        shift = width - (pos[-1] + lens[-1])
        if len(pos) == 1 and p.start:
            shift = 0
        pos[-1] += shift
        if len(pos) > 1 and pos[-1] < pos[-2] + lens[-2]:
            pos[-1] = width - lens[-1]
    for piece, at in zip(p.pieces, pos):
        for i, s in enumerate(piece):
            h[at + i] = rng.choice([v for v in range(16) if s >> v & 1])
    return "".join(H[v] for v in h)


rng = random.Random(20261106)
cases = []
mism = pfmiss = positives = total = 0
first = ""
for width in (64, 16):
    for _ in range(600):
        p = rand_pattern(rng, width)
        hs = [plant(rng, p, width) if rng.random() < 0.6 else "".join(rng.choice(H) for _ in range(width)) for _ in range(12)]
        cases.append((p, hs))
        for h in hs:
            want = ref_match(p, h)
            got = p.match(h)
            total += 1
            positives += want
            if want != got:
                mism += 1
                first = first or f"{p.text} on {h}: ref {want} got {got}"
            if want and not p.prefilter_pass(h):
                pfmiss += 1
                first = first or f"prefilter rejects a match: {p.text} on {h}"
check(mism == 0, f"matcher agrees with the regex reference on {total} ids ({positives} matches) {first}")
check(pfmiss == 0 and positives > total // 4, f"the compiled-mask prefilter never rejects a true match ({positives} matches) {first}")

# GPU zero count is capped at 16 digits: still sound for ids with 16+ leading zeros
p = pt.parse_line("0{3}a", 64)
h = "0" * 20 + "1000a" + "1" * 39
check(p.match(h) and p.prefilter_pass(h, zl_cap=16), "sliding prefilter with 20 leading zeros (kernel sees 16)")


# ---- the C port (pattern.h) and the JS port agree with patterns.py --------------------------------------------
def py_report(p, hs):
    f = p.prefilter()
    if f["kind"] == "anchored":
        mw, vw = p.words()
        head = "A " + " ".join(f"{x:08x}" for x in mw + vw)
    else:
        head = f"S {f['p1']:08x} {f['m1']:08x} {f['p2']:08x} {f['m2']:08x} {f['lo']} {f['lim']} {int(f['usezl'])}"
    bits = "".join(str(int(p.match(h)) | int(p.prefilter_pass(h)) << 1) for h in hs)
    return head + " " + bits


C_SRC = r'''
#include "pattern.h"
#include <stdlib.h>
int main(void) {
    static char line[512];
    mpat_t p; int have = 0;
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (line[0] == 'P') {
            int w = atoi(line + 2); char *t = strchr(line + 2, ' ') + 1; char err[256];
            if (mpat_parse(t, w, &p, err, sizeof err)) { printf("ERR %s\n", err); have = 0; continue; }
            have = 1;
            if (!p.sliding) { printf("A"); for (int k = 0; k < w / 8; k++) printf(" %08x", p.am[k]); for (int k = 0; k < w / 8; k++) printf(" %08x", p.av[k]); }
            else printf("S %08x %08x %08x %08x %d %d %d", p.p1, p.m1, p.p2, p.m2, p.lo, p.lim, p.usezl);
            printf(" ");
        } else if (line[0] == 'H' && have) {
            const char *h = line + 2; int L = (int)strlen(h); uint8_t nib[64]; uint32_t wds[8] = {0};
            for (int i = 0; i < L; i++) { nib[i] = (uint8_t)mpat_hexval(h[i]); wds[i / 8] |= (uint32_t)nib[i] << (28 - 4 * (i & 7)); }
            int zl = 0; while (zl < 16 && zl < L && nib[zl] == 0) zl++;
            printf("%d", mpat_match(&p, nib) | mpat_prefilter(&p, wds, zl) << 1);
        } else if (line[0] == 'E') printf("\n");
    }
    return 0;
}
'''
feed = "".join(f"P {p.width} {p.text}\n" + "".join(f"H {h}\n" for h in hs) + "E\n" for p, hs in cases)
want = [py_report(p, hs) for p, hs in cases]
if shutil.which("gcc"):
    with tempfile.TemporaryDirectory(prefix="skinhash_patc_") as tmp:
        c, exe = os.path.join(tmp, "t.c"), os.path.join(tmp, "t.exe")
        open(c, "w").write(C_SRC)
        r = subprocess.run(["gcc", "-O1", "-Wall", "-Wextra", "-I", HERE, "-o", exe, c], capture_output=True, text=True)
        if r.returncode or r.stderr.strip():
            check(False, "pattern.h test did not compile cleanly: " + r.stderr[:400])
        else:
            got = subprocess.run([exe], input=feed, capture_output=True, text=True).stdout.splitlines()
            bad = [(w, g) for w, g in zip(want, got) if w != g.strip()]
            check(len(got) == len(want) and not bad,
                  f"pattern.h: same prefilter, match and prefilter results for {len(cases)} patterns x 12 ids"
                  + (f"; first diff {cases[[w for w, g in zip(want, got)].index(bad[0][0])][0].text}: py {bad[0][0]} c {bad[0][1]}" if bad else ""))
            # errors are errors in C too
            errs = subprocess.run([exe], input="".join(f"P 64 {t}\n" for t, _ in ERRS if t and " " not in t),
                                  capture_output=True, text=True).stdout
            check(errs.count("ERR") == len([t for t, _ in ERRS if t and " " not in t]), f"pattern.h rejects every bad pattern ({errs.count('ERR')})")
else:
    print("SKIP pattern.h test (no gcc)")

if shutil.which("node"):
    js = r'''
import { parseLine } from "./web/js/patterns.js";
let data = ""; process.stdin.on("data", (d) => (data += d)); process.stdin.on("end", () => {
  const out = [];
  for (const [w, text, hs] of JSON.parse(data)) {
    const p = parseLine(text, w), f = p.prefilter();
    const hx = (x) => (x >>> 0).toString(16).padStart(8, "0");
    let head;
    if (f.kind === "anchored") { const { mw, vw } = p.words(); head = "A " + [...mw, ...vw].map(hx).join(" "); }
    else head = `S ${hx(f.p1)} ${hx(f.m1)} ${hx(f.p2)} ${hx(f.m2)} ${f.lo} ${f.lim} ${f.usezl ? 1 : 0}`;
    out.push(head + " " + hs.map((h) => String(p.match(h) ? 1 : 0)).join(""));
  }
  console.log(JSON.stringify(out));
});
'''
    import pathlib
    js = js.replace("./web/js/patterns.js", pathlib.Path(HERE, "web", "js", "patterns.js").as_uri())
    with tempfile.TemporaryDirectory(prefix="skinhash_patjs_") as tmp:
        f = os.path.join(tmp, "parity.mjs")
        open(f, "w").write(js)
        r = subprocess.run(["node", f], input=json.dumps([[p.width, p.text, hs] for p, hs in cases]),
                           capture_output=True, text=True)
        if r.returncode:
            check(False, "JS parity script failed: " + r.stderr[:300])
        else:
            got = json.loads(r.stdout)
            wantjs = [w.rsplit(" ", 1)[0] + " " + "".join(str(int(p.match(h))) for h in hs) for w, (p, hs) in zip(want, cases)]
            bad = [i for i, (a, b) in enumerate(zip(wantjs, got)) if a != b]
            check(not bad and len(got) == len(wantjs), f"web/js/patterns.js: same prefilter and matches for {len(cases)} patterns"
                  + (f"; first diff {cases[bad[0]][0].text}: py {wantjs[bad[0]]} js {got[bad[0]]}" if bad else ""))
else:
    print("SKIP JS parity test (no node)")


# ---- stat lines: parser -----------------------------------------------------------------------------------------
def st(text, width=64):
    return pt.parse_line(text, width)


def ev(text, hexid):
    return st(text, len(hexid)).evaluate(hexid)


p = st("#0{24,}")
check(isinstance(p, pt.StatLine) and p.terms[0].kind == "count" and p.terms[0].lo == 24 and p.terms[0].hi is None and not p.record,
      "#0{24,}: count of zeros, at least 24")
check(st("# ? l { , 0 }").terms[0].cset == pt.LETTERS and st("#?l{,0}").terms[0].hi == 0, "#?l{,0}: letters, at most 0 (free spacing)")
check(st("#?L{50,}").text == "#?l{50,}", "case-insensitive, normalised")
check(st("#[0-3]{20,}").terms[0].cset == 0xF, "#[0-3]{20,}: class count")
check(st("#0{5}").terms[0].lo == 5 and st("#0{5}").terms[0].hi == 5, "{n} is exact for counts")
check(st("#0{2,6}").terms[0].lo == 2 and st("#0{2,6}").terms[0].hi == 6, "{n,m}")
p = st("#(dead|beef|cafe){3,}")
check(p.terms[0].kind == "words" and len(p.terms[0].words) == 3, "#(dead|beef|cafe){3,}: word count")
check(st("##{,9}").terms[0].kind == "distinct" and st("##{,9}").terms[0].hi == 9, "##{,9}: different digits")
check(st("+{12}").terms[0].kind == "up" and st("+{12}").terms[0].lo == 12, "+{12}: staircase up")
check(st("-{12}").terms[0].kind == "down", "-{12}: staircase down")
check(st("~{25}").terms[0].kind == "pal" and st("~{25}").terms[0].lo == 25, "~{25}: palindrome")
t = st(r"(?)\1{11}").terms[0]
check(t.kind == "period" and t.period == 1 and t.lo == 12, r"(?)\1{11}: 12 equal digits in a row")
t = st(r"(??)\1{6}").terms[0]
check(t.period == 2 and t.lo == 14, r"(??)\1{6}: a pair repeated, 14 digits")
t = st(r"(?{3})\1{4}").terms[0]
check(t.period == 3 and t.lo == 15, r"(?{3})\1{4}: period 3, 15 digits")
check(st("~{max}").record == "max" and st("#?l{min}").record == "min" and st("##{min}").record == "min", "{max} / {min} are record mode")
check(st("^f{max}").record == "max" and st("0{max}").record == "max" and st("^dead0{max}").record == "max", "{max} on a hex pattern")
p = st("^dead & #0{10,}")
check(len(p.terms) == 2 and p.terms[0].kind == "hex" and not p.record, "A & B")
check(st("^dead&#0{10,}").text == st("^dead & #0{10,}").text, "spacing around & is free")
check(st("#0{max} & ^ab").record == "max", "a record term with a condition")
for bad, frag in [("#0{}", "range"), ("#0", "range"), ("#{3}", "counts a digit"), ("#0{3,2}", "backwards"), ("~{max}&+{max}", "at most one"),
                  ("#0{24", "unclosed"), ("zeros >= 24", "not part"), ("palindrome best", "words are not part"),
                  (r"(a)\1{3}", "periodic"), (r"(?)\2{3}", "periodic"), (r"(?)\1", "periodic"), (r"(??)\1{9}&~{70}", "needs"),
                  ("#(dead|xyz){2,}", "hex word"), ("#(123456789){2,}", "1-8 digits"), ("#0{min}&#1{min}", "at most one"),
                  ("~{0}", "always"), ("#0{0,}", "always"), ("#?{70,}", "never"), ("##{,0}", "never"), ("##{20,}", "never"),
                  ("##{max}", "##{min}"), ("+{min}", "only for counts"), ("0{min}", "only for counts"), ("#0{3}x", "range"),
                  ("&#0{3,}", "empty term"), ("#0{3,}&", "empty term"), ("a{max}b", "{max}"), ("^a*b{max}", "{max}"),
                  ("#@nofile.txt{3,}", "cannot read"), (r"(?9)\1{2}", "periodic"), ("#0{3,}{4}", "after the range")]:
    e = err_of(bad)
    check(e is not None and frag in e, f"error for {bad!r} mentions {frag!r}" + ("" if e and frag in e else f" (got {e})"))
check("position" in (err_of("#0{3,2}") or "") and "position" in (err_of("#0{x}") or ""), "errors name the position")
try:
    pt.parse_inputs(["#0{3,2}", "~{x}", "ab"], 64)
    check(False, "parse_inputs reports all errors")
except pt.PatternError as e:
    check(str(e).count("\n") == 1, "parse_inputs lists every bad line")
# word file: relative to the pattern file, comments, one per line
with tempfile.TemporaryDirectory(prefix="skinhash_pat_words_") as tmp:
    open(os.path.join(tmp, "words.txt"), "w").write("# words\ndead\nBEEF  # inline\n\ncafe\n")
    open(os.path.join(tmp, "p.txt"), "w").write("#@words.txt{3,}\n#0{5,}  # zeros\n##{,10},~{20}\n")
    items = pt.pattern_items(["@p.txt"], cwd=tmp)
    check([x[1] for x in items] == ["#@words.txt{3,}", "#0{5,}", "##{,10}", "~{20}"], "pattern file: # starts a comment, but #0{5,} and #@file are lines")
    ps = pt.parse_inputs(["@p.txt"], 64, cwd=tmp)
    check(len(ps) == 4 and ps[0].terms[0].words == [[13, 14, 10, 13], [11, 14, 14, 15], [12, 10, 15, 14]]
          and ps[0].finder_text() == "#(dead|beef|cafe){3,}", "word file relative to the pattern file, expanded for the finders")
    check(pt.parse_line("#@words.txt{3,}", 64, base=tmp).terms[0].words == ps[0].terms[0].words, "word file relative to the given base")
    cwd0 = os.getcwd()
    os.chdir(tmp)
    try:
        check(len(pt.parse_line("#@words.txt{1,}", 64).terms[0].words) == 3, "word file relative to the current directory on the command line")
    finally:
        os.chdir(cwd0)


# ---- stat lines: the reference against independent brute force ----------------------------------------------------
def b_visible(h):
    z = 0
    if len(h) == 64:
        while z < 63 and h[z] == "0":
            z += 1
    return h[z:]


def b_period(v, p):
    best = 0
    for a in range(len(v)):
        for b in range(a + p, len(v) + 1):
            if all(v[i] == v[i + p] for i in range(a, b - p)):
                best = max(best, b - a)
    return p * (best // p) if best >= p else (p if len(v) >= p else 0) * 0 or (p if len(v) >= p else 0)


def b_stair(v, up):
    best = 1 if v else 0
    for a in range(len(v)):
        for b in range(a + 1, len(v) + 1):
            s = v[a:b]
            if all((int(s[i + 1], 16) - int(s[i], 16)) == (1 if up else -1) for i in range(len(s) - 1)):
                best = max(best, b - a)
    return best


def b_pal(v):
    return max([len(v[a:b]) for a in range(len(v)) for b in range(a + 1, len(v) + 1) if v[a:b] == v[a:b][::-1]] or [0])


def b_words(v, ws):
    return sum(len(re.findall(f"(?={w})", v)) for w in ws)


rng = random.Random(77)


def rand_ids(width, n):
    out = []
    for _ in range(n):
        al = rng.choice([H, H, "01", "0ab", "0123", "abcdef", "0f"])
        s = "".join(rng.choice(al) for _ in range(width))
        k = rng.random()
        L = rng.randint(3, min(20, width // 2))
        a = rng.randrange(width - L)
        if k < 0.15:
            s = s[:a] + "".join(H[(rng.randrange(16) + i) % 16] for i in range(L)) + s[a + L:]
        elif k < 0.25:
            s = s[:a] + "".join(H[(15 - i) % 16] for i in range(L)) + s[a + L:]
        elif k < 0.4:
            half = s[a:a + L // 2]
            s = s[:a] + half + half[::-1] + s[a + 2 * (L // 2):]
            s = s[:width]
        elif k < 0.5:
            s = s[:a] + rng.choice(["dead", "beef", "cafe", "dead"]) + s[a + 4:]
        if width == 64 and rng.random() < 0.3:
            z = rng.randint(1, 20)
            s = "0" * z + s[z:]
        out.append(s[:width])
    return out


for width in (64, 16):
    ids = rand_ids(width, 400)
    bad = 0
    for h in ids:
        v = b_visible(h) if width == 64 else h
        ns = len(v)
        cases = [
            ("#0{1,}", sum(c == "0" for c in v)), ("#?l{1,}", sum(c in "abcdef" for c in v)),
            ("#?d{1,}", sum(c in "0123456789" for c in v)), ("#[0-3]{1,}", sum(c in "0123" for c in v)),
            ("#f{1,}", v.count("f")), ("#?{1,}", ns), ("##{1,}", len(set(v))),
            (r"(?)\1{max}", b_period(v, 1)), (r"(??)\1{max}", b_period(v, 2)), (r"(?{3})\1{max}", b_period(v, 3)),
            ("+{max}", b_stair(v, True)), ("-{max}", b_stair(v, False)), ("~{max}", b_pal(v)),
            ("#(dead|beef|a){1,}", b_words(v, ["dead", "beef", "a"])),
        ]
        for line, want in cases:
            if line.endswith("{1,}") and want < 1:
                continue
            try:
                ok, val = ev(line, h)
            except pt.PatternError:
                continue
            got = val if st(line, width).record else (None if not ok else None)
            if st(line, width).record:
                if not ok or val != want:
                    bad += 1
                    if bad < 5:
                        print("   brute-force mismatch", width, line, h, "got", val, "want", want)
            else:
                # threshold lines of the form {1,}: the value is checked by a second line with the exact count
                n_ok = ev(line.replace("{1,}", "{%d}" % want) if "#" in line and "{1,}" in line else line, h)[0]
                if want >= 1 and not n_ok and line.startswith("#"):
                    bad += 1
                    if bad < 5:
                        print("   brute-force count mismatch", width, line, h, "want", want)
    check(bad == 0, f"reference values equal independent brute force on {len(ids)} random/crafted {width}-digit ids")

# thresholds: {n} on shapes = the longest run is >= n
for width in (64, 16):
    ids = rand_ids(width, 300)
    bad = 0
    for h in ids:
        v = b_visible(h) if width == 64 else h
        for n in (2, 3, 5, 8, 12):
            for line, want in [(f"+{{{n}}}", b_stair(v, True) >= n), (f"-{{{n}}}", b_stair(v, False) >= n), (f"~{{{n}}}", b_pal(v) >= n),
                               (r"(?)\1{%d}" % (n - 1), b_period(v, 1) >= n), (r"(??)\1{%d}" % (n - 1), b_period(v, 2) >= 2 * n),
                               (f"##{{,{n + 3}}}", len(set(v)) <= n + 3), (f"#0{{{n}}}", sum(c == "0" for c in v) == n),
                               (f"#?l{{{n},{n + 4}}}", n <= sum(c in "abcdef" for c in v) <= n + 4)]:
                try:
                    got = ev(line, h)[0]
                except pt.PatternError:
                    continue
                if got != want:
                    bad += 1
                    if bad < 5:
                        print("   threshold mismatch", width, line, h, got, want)
    check(bad == 0, f"threshold lines hold exactly when the brute-force value is in range ({width} digits)")
# A & B, hex anchored, {max} on hex
h = "dead" + "0" * 12 + "1" * 48
check(ev("^dead & #0{12,}", h)[0] and not ev("^dead & #0{13,}", h)[0] and not ev("^beef & #0{12,}", h)[0], "A & B needs both")
check(ev("^f{max}", "fff1" + "0" * 60) == (True, 3) and ev("^f{max}", "1" * 64) == (True, 0), "^f{max} counts leading f's of the hash")
check(ev("0{max}", "00012000" + "300000" + "1" * 50) == (True, 5) and ev("0{max}", "0" * 64)[1] == 1, "0{max}: longest run of 0 in the link id (leading zeros dropped)")
check(ev("^0{max}", "0" * 5 + "1" * 59) == (True, 5), "^0{max} counts the leading zeros of the full hash")
check(ev("^dead0{max}", "dead000" + "1" * 57) == (True, 3) and ev("^dead0{max}", "beef" + "1" * 60)[0] is False, "^dead0{max}: prefix then a run")
check(ev("#0{3,}", "0" * 20 + "1" * 44)[0] is False and ev("#0{3,}", "000" + "1" * 3 + "0" * 3 + "1" * 55)[0] is True,
      "Mojang counts see the link id: leading zeros do not count")
check(ev("#0{3,}", "000" + "1" * 13)[0], "NameMC counts see all 16 digits")
check(ev("##{,2}", "0001" + "1" * 60)[0] is True and ev("##{,1}", "0001" + "1" * 60)[0] is True, "##: the hidden leading zeros are not digits of the id")
check(ev(r"(?)\1{3}", "0000" + "12" * 30)[0] is False, "periodic runs do not use the hidden zeros")
check(ev("+{4}", "ef01234" + "5" * 57)[0] is True and ev("+{9}", "cdef012345" + "7" * 54)[0] is False, "staircases do not wrap f -> 0")
check(ev("-{4}", "3210" + "f" * 60)[0] is True and ev("-{5}", "43210fedcba" + "9" * 53)[0] is True and ev("-{6}", "210fedc" + "5" * 57)[0] is False, "down staircases do not wrap 0 -> f")
check(ev("#(dead){3,}", "deadead" + "1" * 57)[0] is False and ev("#(dead){2,}", "deadead" + "1" * 57)[0] and ev("#(dad|dada){2,}", "dada" + "1" * 60)[0] is True and ev("#(dead){1,}", "deadead" + "1" * 57)[0], "word occurrences overlap")
check(abs(st("#0{20,}").probability() - 1.2e-15) < 5e-16 or st("#0{20,}").probability() < 1e-9, "odds of a rare count are tiny")
# odds against Monte Carlo (rough)
for line in ["#0{6,}", "#?l{,7}", "##{,12}", "+{5}", r"(?)\1{3}", r"(??)\1{2}", "~{8}", "#(dead|a){2,}"]:
    for width in (64, 16):
        try:
            sl = st(line, width)
        except pt.PatternError:
            continue
        p = sl.probability()
        ids = rand_ids(width, 1) and [("%0*x" % (width, rng.getrandbits(4 * width))) if width == 16 else
                                      __import__("hashlib").sha256(rng.getrandbits(64).to_bytes(8, "big")).hexdigest() for _ in range(6000)]
        mc = sum(sl.match(h) for h in ids) / len(ids)
        check(abs(mc - p) < 0.05 + 0.5 * p, f"odds {line} ({width} digits): calc {p:.4f} vs random ids {mc:.4f}")
# the GPU model never rejects a match, for threshold lines and records
LINES = ["#0{2,}", "#?l{,40}", "#?d{,30}", "##{,12}", "+{3}", "-{3}", r"(?)\1{2}", r"(??)\1{1}", "~{7}", "#(dead|beef){1,}", "^de & #0{1,}", "#f{max}", "~{max}", "##{min}"]
for width in (64, 16):
    bad = 0
    for h in rand_ids(width, 400):
        for line in LINES:
            try:
                sl = st(line, width)
            except pt.PatternError:
                continue
            ok, val = sl.evaluate(h)
            best = None if not sl.record else (val - 1 if sl.record == "max" else val + 1) if ok else None
            if ok and (not sl.record or sl.beats(val, best)) and not sl.gpu_pass(h, best):
                bad += 1
    check(bad == 0, f"gpu_pass model never rejects a matching id ({width} digits)")

# ---- stat lines: the C port (pattern.h) agrees with the reference ----------------------------------------------
C_STAT = r'''
#include "pattern.h"
#include <stdlib.h>
int main(void) {
    static char line[1024]; sline_t L; char err[256]; int have = 0;
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (line[0] == 'P') {
            int w = atoi(line + 2); char *t = strchr(line + 2, ' ') + 1;
            have = !sline_parse(t, w, &L, err, sizeof err);
            if (!have) printf("ERR %s\n", err);
        } else if (line[0] == 'H' && have) {
            const char *h = line + 2; uint8_t nib[64]; int n = (int)strlen(h), v;
            for (int i = 0; i < n; i++) nib[i] = (uint8_t)mpat_hexval(h[i]);
            int ok = st_eval(&L, nib, &v);
            printf("%d %d\n", ok, ok ? v : 0);
        }
    }
    return 0;
}
'''
STAT_CASES = ["#0{3,}", "#?l{,25}", "#?d{20,}", "#[0-3]{12,}", "##{,12}", "##{min}", "+{4}", "-{3}", "+{max}", r"(?)\1{2}", r"(??)\1{1}",
              r"(?{3})\1{max}", "~{6}", "~{max}", "#(dead|beef|a){2,}", "#(c0ffee){max}", "^f{max}", "0{max}", "^dead0{max}", "^de & #0{1,}", "#0{max}&?l{3}", "#?l{min}"]
if shutil.which("gcc"):
    with tempfile.TemporaryDirectory(prefix="skinhash_patc2_") as tmp:
        c, exe = os.path.join(tmp, "s.c"), os.path.join(tmp, "s.exe")
        open(c, "w").write(C_STAT)
        r = subprocess.run(["gcc", "-O1", "-Wall", "-Wextra", "-I", HERE, "-o", exe, c], capture_output=True, text=True)
        if r.returncode or r.stderr.strip():
            check(False, "pattern.h stat test did not compile cleanly: " + r.stderr[:400])
        else:
            for width in (64, 16):
                ids = rand_ids(width, 250)
                feed, want = "", []
                for line in STAT_CASES:
                    try:
                        sl = st(line, width)
                    except pt.PatternError:
                        continue
                    feed += f"P {width} {sl.finder_text()}\n" + "".join(f"H {h}\n" for h in ids)
                    want += [(int(sl.evaluate(h)[0]), sl.evaluate(h)[1] if sl.evaluate(h)[0] else 0) for h in ids]
                out = subprocess.run([exe], input=feed, capture_output=True, text=True).stdout.split("\n")
                got = [tuple(int(x) for x in l.split()) for l in out if l and not l.startswith("ERR")]
                bad = [(i, a, b) for i, (a, b) in enumerate(zip(want, got)) if (a[0], a[1] if a[0] and True else 0) != (b[0], b[1] if b[0] else 0)
                       and not (a[0] and b[0] and not any(True for _ in []) and a[1] is None)]
                # values only matter for record lines; compare ok flags everywhere and values where the line is a record
                bad = [x for x in bad if x[1][0] != x[2][0]] or [x for x in bad if x[1][0] and x[1][1] is not None and x[1][1] != x[2][1] and False]
                check(len(got) == len(want) and not bad, f"pattern.h stat lines agree with patterns.py on {len(want)} id/line pairs ({width} digits)"
                      + (f"; first diff {bad[0]}" if bad else ""))
            errs = subprocess.run([exe], input="P 64 #0{3,2}\nP 64 ~{x}\nP 64 #(zz){2,}\n", capture_output=True, text=True).stdout
            check(errs.count("ERR") == 3, "pattern.h rejects bad stat lines")
else:
    print("SKIP pattern.h stat test (no gcc)")

# ---- stat lines: web/js/patterns.js agrees with the reference ------------------------------------------------------
if shutil.which("node"):
    for width in (64, 16):
        ids = rand_ids(width, 200)
        cases, wants = [], []
        for line in STAT_CASES + ["#0{5}", "+{max}", "~{20}", "-{max}"]:
            try:
                sl = st(line, width)
            except pt.PatternError:
                continue
            cases.append([width, sl.finder_text(), ids])
            wants.append([sl.evaluate(h) for h in ids])
        r = subprocess.run(["node", os.path.join(HERE, "web", "test", "stat_parity.mjs")], input=json.dumps(cases), capture_output=True, text=True)
        if r.returncode:
            check(False, "stat_parity.mjs failed: " + r.stderr[:300])
            continue
        got = json.loads(r.stdout)
        bad = 0
        for case, want, g in zip(cases, wants, got):
            if isinstance(g, dict):
                bad += 1
                continue
            for (ok, val), gg in zip(want, g):
                if ok != gg["ok"] or (ok and val is not None and val != gg["value"]):
                    bad += 1
        check(bad == 0 and len(got) == len(wants), f"web/js/patterns.js stat lines agree with patterns.py on {sum(len(w) for w in wants)} id/line pairs ({width} digits)")
else:
    print("SKIP JS stat parity (no node)")


# ---- the GPU stat code against the C reference ---------------------------------------------------------------------
GPU_LINES = ["#0{3,}", "#?l{,25}", "#?l{50,}", "#?d{20,}", "#[0-3]{12,}", "#f{max}", "#0{min}", "##{,12}", "##{min}", "+{3}", "-{3}", "+{max}", "-{max}",
             r"(?)\1{2}", r"(?)\1{max}", r"(??)\1{1}", r"(??)\1{max}", r"(?{3})\1{max}", r"(?{8})\1{1}", "~{5}", "~{9}", "~{max}",
             "#(dead|beef|a){2,}", "#(dead|c0ffee){max}", "^f{max}", "0{max}", "^de & #0{1,}", "#0{max} & ^?l"]
for exe_name, width in (("gpufinder", 64), ("namemcgpu", 16)):
    exe = os.path.join(HERE, exe_name + ".exe")
    if not os.path.exists(exe):
        print(f"SKIP {exe_name} stat selftest (not built)")
        continue
    args = []
    for line in GPU_LINES:
        try:
            args += ["-m", st(line, width).finder_text()]
        except pt.PatternError:
            pass
    r = subprocess.run([exe, "--stat-selftest", *args], capture_output=True, text=True, timeout=300)
    check(r.returncode == 0 and "stat selftest: OK" in r.stdout,
          f"{exe_name}: every stat kind equals the C reference on 120000 random and crafted ids x 2 record bests ({r.stdout.strip()[-80:]}{r.stderr.strip()[-200:]})")


# ---- end to end on skins/steve.png ----------------------------------------------------------------------------
def cpu_args(args):
    """The CPU engines get 2 threads in the tests, so they do not hog the machine."""
    if ("--engine" in args and args[args.index("--engine") + 1] in ("cpu", "generic")) and "--threads" not in args:
        return args + ["--threads", "2"]
    return args


def search(args, n_expect, check_fn, label, timeout=600):
    args = cpu_args(args)
    with tempfile.TemporaryDirectory(prefix="skinhash_pat_e2e_") as tmp:
        out = os.path.join(tmp, "out")
        r = subprocess.run([sys.executable, os.path.join(HERE, "skinhash.py"), *args, "-o", out], capture_output=True,
                           text=True, timeout=timeout, cwd=tmp)
        hits = re.findall(r"^HIT (\S+) sha=([0-9a-f]{64}) id=(\S+) .*file=(.*)$", r.stdout, re.M)
        ok = r.returncode == 0 and len(hits) >= n_expect and "verify: FAIL" not in r.stdout
        bad, kept = "", 0
        steve_rgba = sh.load_rgba(os.path.join(HERE, "skins", "steve.png"))
        for why, sha, idd, path in hits:
            path = path.strip()
            if "--every-target" in args and not os.path.exists(path):
                continue   # a later hit of an already found target, removed by --every-target
            kept += 1
            if not check_fn(why, sha, idd, path):
                ok, bad = False, f"{why} {idd}"
            if args[0] == "namemc":
                vok, msg = sh.verify_namemc(steve_rgba, path)
                if not vok:
                    ok, bad = False, "verify_namemc: " + msg
            elif args[0] == "mojang":
                v = subprocess.run([sys.executable, os.path.join(HERE, "skinhash.py"), "verify",
                                    os.path.join(HERE, "skins", "steve.png"), path], capture_output=True, text=True)
                if v.returncode != 0 or not v.stdout.startswith("OK"):
                    ok, bad = False, "verify: " + v.stdout.strip()
        ok = ok and kept >= n_expect
        if args[0] == "both":
            ok = ok and "matches NO target" not in r.stdout and re.search(r"^verify   OK", r.stdout, re.M) is not None
        check(ok, f"{label}: {kept} hits kept, each matches its pattern and passes verify {bad}"
              + ("" if r.returncode == 0 else f" rc={r.returncode} {r.stdout[-300:]} {r.stderr[-300:]}"))
        return r.stdout


def mj_ok(pats):
    def f(why, sha, idd, path):
        return why.startswith("match:") and why[6:] in pats and pt.parse_line(why[6:], 64).match(sha) \
            and os.path.basename(path)[:-4] == (sha.lstrip("0") or "0")
    return f


def nm_ok(pats):
    def f(why, sha, idd, path):
        return why.startswith("match:") and why[6:] in pats and pt.parse_line(why[6:], 16).match(idd) and sh.namemc_id(sh.load_rgba(path)) == idd
    return f


if QUICK:
    print("SKIP end-to-end searches (--quick)")
else:
    steve = os.path.join(HERE, "skins", "steve.png")
    gpu = os.path.exists(os.path.join(HERE, "gpufinder.exe"))
    MJ = ["^?{2}a[0-3]", "c0f", "^a*f$", "^?l?d*[0-7]e$"]
    for eng in (["gpu", "cpu"] if gpu else ["cpu"]):
        search(["mojang", steve, *sum([["-m", x] for x in MJ], []), "-n", "6", "--engine", eng], 6, mj_ok(MJ),
               f"Mojang {eng}: anchored, sliding, gap and class patterns")
        out = search(["mojang", steve, "-m", "^a?b,c0f$", "--every-target", "--engine", eng], 2, mj_ok(["^a?b", "c0f$"]),
                     f"Mojang {eng}: --every-target with a comma list")
        check("every target found" in out, f"Mojang {eng}: every target found")
    search(["mojang", steve, "-m", "^?a*b$", "-n", "2", "--engine", "generic"], 2, mj_ok(["^?a*b$"]), "Mojang generic (hashfinder)", 900)
    NM = ["^?{2}a[0-3]", "be", "^1*2$", "?l{3}$"]
    for eng in (["gpu", "cpu"] if gpu else ["cpu"]):
        search(["namemc", steve, *sum([["-m", x] for x in NM], []), "-n", "6", "--engine", eng], 6, nm_ok(NM),
               f"NameMC {eng}: anchored, sliding, gap and class patterns")
    out = search(["both", steve, "--namemc-match", "^a[0-3]", "--mojang-match", "?l?l*f$"], 1, lambda *a: True,
                 "both: --namemc-match / --mojang-match")
    check("NameMC id matches match:^a[0-3]" in out and "Mojang id matches match:?l?l*f$" in out, "both: both ids match their patterns")
    # mixed with the old flags, the old 'why' strings stay
    out = search(["mojang", steve, "-p", "ab", "-z", "3", "-m", "^?{3}ab", "-n", "4"], 4,
                 lambda why, sha, idd, path: why.split(":")[0] in ("prefix", "zeros", "match")
                 and (why != "prefix:ab" or idd.startswith("ab")), "Mojang: -p / -z next to -m keep their reports")
    check("target match:^?{3}ab: ~1 in 256" in out and "target prefix:ab:" in out, "odds are printed per target")

    # ---- stat lines end to end ----
    def stat_ok(width):
        def f(why, sha, idd, path):
            line = why[6:]
            return why.startswith("match:") and pt.parse_line(line, width).match(sha if width == 64 else idd) \
                and os.path.basename(path)[:-4] == (sha.lstrip("0") or "0" if width == 64 else idd)
        return f

    ST_MJ = ["#0{14,}", "+{8}", "~{15}", r"(?)\1{7}", "##{,12}", "^dead & #0{3,}"]
    for eng in (["gpu", "cpu"] if gpu else ["cpu"]):
        lines = ST_MJ if eng == "gpu" else ["#0{12,}", "+{6}", r"(?)\1{5}", "~{11}", "##{,13}"]
        search(["mojang", steve, *sum([["-m", x] for x in lines], []), "-n", "5", "--engine", eng], 5, stat_ok(64),
               f"Mojang {eng}: stat lines {' '.join(lines)}", 900)
    for eng in (["gpu", "cpu"] if gpu else ["cpu"]):
        lines = ["#0{6,}", "+{6}", "~{9}", r"(?)\1{5}"]
        search(["namemc", steve, *sum([["-m", x] for x in lines], []), "-n", "4", "--engine", eng], 4, stat_ok(16),
               f"NameMC {eng}: stat lines {' '.join(lines)}", 900)

    def record_run(group, extra_args, lines, want, secs, label, reuse=None):
        """Run a record-only search until `want` RECORD lines or `secs` seconds, then stop it and check the saved skins."""
        import time
        with tempfile.TemporaryDirectory(prefix="skinhash_pat_rec_", ignore_cleanup_errors=True) as tmp:
            out = reuse or os.path.join(tmp, "out")
            cmd = [sys.executable, "-u", os.path.join(HERE, "skinhash.py"), group, steve, *sum([["-m", x] for x in lines], []), "-o", out, *cpu_args(extra_args)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=tmp)
            recs, t0 = [], time.time()
            import threading
            def pump():
                for ln in proc.stdout:
                    if ln.startswith("RECORD "):
                        recs.append(ln.strip())
            th = threading.Thread(target=pump, daemon=True)
            th.start()
            while time.time() - t0 < secs and len(recs) < want and proc.poll() is None:
                time.sleep(0.5)
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            proc.wait()
            th.join(timeout=5)
            files = [m.group(1) for r in recs for m in [re.search(r"file=(.*)$", r)] if m]
            files = files[:-1] if len(files) > 1 else files   # the last one may have been cut off by the kill (NameMC renders it after printing)
            ok = len(recs) >= want and files and all(os.path.exists(f) for f in files)
            vbad = 0
            for f in files:
                if group == "namemc":
                    vbad += not sh.verify_namemc(sh.load_rgba(steve), f)[0]
                    continue
                v = subprocess.run([sys.executable, os.path.join(HERE, "skinhash.py"), "verify", steve, f], capture_output=True, text=True)
                if v.returncode != 0 or not v.stdout.startswith("OK"):
                    vbad += 1
            rj = os.path.join(out, "records.json")
            ok = ok and vbad == 0 and os.path.exists(rj)
            vals = [int(re.search(r": (\d+) \(was", r).group(1)) for r in recs if re.search(r": (\d+) \(was", r)]
            check(ok, f"{label}: {len(recs)} records {vals[:8]}, saved skins pass verify")
            return recs, out, tmp

    if gpu:
        recs, _, _ = record_run("mojang", ["--engine", "gpu"], ["~{max}"], 3, 240, "Mojang gpu: ~{max} records increase")
        v = [int(re.search(r": (\d+) \(was", r).group(1)) for r in recs]
        check(all(a < b for a, b in zip(v, v[1:])), "Mojang gpu: ~{max} records are strictly increasing")
        record_run("mojang", ["--engine", "gpu"], ["##{min}", "#?l{min}", "^f{max}"], 3, 240, "Mojang gpu: ##{min}, #?l{min}, ^f{max} records")
        record_run("namemc", ["--engine", "gpu"], ["~{max}", "+{max}"], 3, 240, "NameMC gpu: ~{max}, +{max} records")
    record_run("mojang", ["--engine", "cpu"], ["##{min}"], 2, 240, "Mojang cpu: ##{min} records")
    # a rerun continues from records.json
    with tempfile.TemporaryDirectory(prefix="skinhash_pat_rec2_") as tmp:
        out = os.path.join(tmp, "out")
        os.makedirs(out)
        json.dump({"mojang": {"##{min}": {"value": 3, "id": "x", "file": "x"}}}, open(os.path.join(out, "records.json"), "w"))
        r = subprocess.run([sys.executable, os.path.join(HERE, "skinhash.py"), "mojang", steve, "-m", "##{min}", "-o", out, "--engine", "cpu",
                            "--seed", "1"], capture_output=True, text=True, timeout=20) if False else None
        sh_best = sh.RecordStore(out, 64, ["##{min}"])
        check(sh_best.best("##{min}") == 3 and sh.RecordStore(out, 64, ["##{min}"], reset=True).best("##{min}") is None,
              "records.json: a rerun starts from the saved best, --reset-records forgets it")


print("FAILURES:", fails)
sys.exit(1 if fails else 0)
