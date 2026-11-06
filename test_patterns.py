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
    check("not supported yet" in (err_of(stat) or ""), f"reserved stat syntax rejected: {stat!r}")

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


# ---- end to end on skins/steve.png ----------------------------------------------------------------------------
def search(args, n_expect, check_fn, label, timeout=600):
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

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
