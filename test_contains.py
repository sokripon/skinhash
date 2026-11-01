"""Check the 'contains' target: a hex string anywhere in the id, in the trimmed Mojang id (leading zeros dropped,
as in the link) or in the 16-digit NameMC id. A match that lies partly or fully in the dropped zeros does not count.
Run: py test_contains.py   (exit code 0 = all good)"""
import os
import shutil
import subprocess
import sys
import tempfile

import skinhash as sh

HERE = os.path.dirname(os.path.abspath(__file__))
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails += 1


def mojang(full, pat):
    shown = full.lstrip("0") or "0"
    return sh.target_match("contains:" + pat, shown, full)


# a digest with 3 leading zeros: the link shows 'abc1230...'
FULL = "000abc123" + "0" * 3 + "d" * 52
assert len(FULL) == 64
check(mojang(FULL, "abc"), "inside the trimmed id: abc")
check(mojang(FULL, "c1230"), "inside the trimmed id, across inner zeros: c1230")
check(mojang(FULL, "3000d"), "inside the trimmed id: 3000d")
check(not mojang(FULL, "00a"), "only partly in the dropped zeros: 00a does not count")
check(not mojang(FULL, "0a"), "straddling the dropped zeros: 0a does not count")
check(not mojang(FULL, "000a"), "dropped zeros plus the first digit: 000a does not count")
check(mojang(FULL, "000d"), "inner zeros are kept: 000d counts")
check(mojang("0" * 64, "0"), "all-zero digest: the id is '0', contains 0")
check(not mojang("0" * 64, "00"), "all-zero digest: the id is '0', contains 00 does not")
check(mojang("beef" + "1" * 60, "beef") and mojang("1" * 60 + "beef", "beef"), "no leading zeros: first and last digits")
check(not mojang("1" * 64, "2") and mojang("1" * 63 + "2", "2"), "single digit")
# a pattern may start with 0 and 16 digits long
check(mojang("1" + "0123456789abcdef" + "2" * 47, "0123456789abcdef"), "16-digit pattern starting with 0")

# NameMC: the 16 digits as they are, leading zeros included
NM = "0000beef12345678"
check(sh.target_match("contains:000b", NM, NM) and sh.target_match("contains:0000", NM, NM), "NameMC: leading zeros count")
check(not sh.target_match("contains:beef0", NM, NM) and sh.target_match("contains:ef1234", NM, NM), "NameMC: inside the 16 digits")

# finder flag translation and target bookkeeping
t = sh.targets_from(["cafe"], ["dead"], [3], ["beef", "00ab"])
check(t == ["prefix:cafe", "suffix:dead", "zeros:3", "contains:beef", "contains:00ab"], f"targets_from: {t}")
check(sh.target_args(["contains:beef", "prefix:c"]) == ["-c", "beef", "-p", "c"], "target_args maps contains to -c")
check(sh.hit_target("contains:beef", ["prefix:c", "contains:beef", "contains:cafe"]) == "contains:beef", "hit_target finds the contains target")
check(sh.hit_target("contains:cafe", ["contains:beef"]) is None, "hit_target ignores other patterns")
check(sh.first_match(["contains:beef"], "1beef2", "01beef2") == "contains:beef", "first_match")

# argument validation
for arg, ok in [("beef", True), ("0000", True), ("0123456789abcdef", True), ("BEEF", True), ("", False),
                ("xyz", False), ("0123456789abcdef0", False)]:
    try:
        sh.hex_arg(arg)
        got = True
    except Exception:
        got = False
    check(got == ok, f"hex_arg({arg!r}) {'accepted' if ok else 'rejected'}")

# the same rule in the native finders (common.h contains_find), compiled with gcc if it is there
if shutil.which("gcc"):
    src = r'''
#include "common.h"
int main(void) {
    uint8_t dg[32] = {0};
    const char *full = "%s";
    for (int i = 0; i < 32; i++) { unsigned v; sscanf(full + 2 * i, "%%2x", &v); dg[i] = (uint8_t)v; }
    int bad = 0;
    struct { const char *pat; int want; } t[] = {
        {"abc", 1}, {"c1230", 1}, {"3000d", 1}, {"00a", 0}, {"0a", 0}, {"000a", 0}, {"000d", 1}, {"0123456789abcdef", 0}, {"dddd", 1}};
    for (unsigned i = 0; i < sizeof t / sizeof *t; i++) {
        ncpats = 0;
        if (parse_contains(t[i].pat)) { printf("parse %%s failed\n", t[i].pat); bad++; continue; }
        int got = contains_find(dg, 32, digest_zero_nibbles(dg)) != NULL;
        if (got != t[i].want) { printf("native %%s: got %%d want %%d\n", t[i].pat, got, t[i].want); bad++; }
    }
    ncpats = 0;
    parse_contains("0000");
    uint8_t nm[8] = {0x00, 0x00, 0xbe, 0xef, 0x12, 0x34, 0x56, 0x78};
    if (!contains_find(nm, 8, 0)) { printf("native NameMC 0000 not found\n"); bad++; }
    ncpats = 0; parse_contains("beef0");
    if (contains_find(nm, 8, 0)) { printf("native NameMC beef0 found\n"); bad++; }
    uint8_t zero[32] = {0};
    ncpats = 0; parse_contains("0");
    if (!contains_find(zero, 32, digest_zero_nibbles(zero))) { printf("native all-zero: 0 not found\n"); bad++; }
    ncpats = 0; parse_contains("00");
    if (contains_find(zero, 32, digest_zero_nibbles(zero))) { printf("native all-zero: 00 found\n"); bad++; }
    printf("%%d native mismatches\n", bad);
    return bad != 0;
}
''' % FULL
    with tempfile.TemporaryDirectory(prefix="skinhash_contains_") as tmp:
        c, exe = os.path.join(tmp, "t.c"), os.path.join(tmp, "t.exe")
        open(c, "w").write(src)
        r = subprocess.run(["gcc", "-O1", "-I", HERE, "-o", exe, c, "-lz", "-lpthread"], capture_output=True, text=True)
        if r.returncode:
            check(False, "native test did not compile: " + r.stderr[:300])
        else:
            r = subprocess.run([exe], capture_output=True, text=True)
            check(r.returncode == 0, "native contains_find agrees with the trimmed-id rule: " + r.stdout.strip().replace("\n", "; "))
else:
    print("SKIP native test (no gcc)")

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
