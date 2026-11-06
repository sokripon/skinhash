"""Independent check of a hit of a pattern search written by the web page, with the Python reference (patterns.py).
usage: py pattern_check.py mojang|namemc HIT.png PATTERN...
Mojang: SHA-256 of the file, leading zeros dropped, equals the file name and the full hash matches one of the patterns.
NameMC: the NameMC id of the file equals the file name and matches one of the patterns. Prints OK or FAIL."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import patterns  # noqa: E402
import skinhash  # noqa: E402

mode, hit, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
name = os.path.splitext(os.path.basename(hit))[0]
if mode == "mojang":
    full = hashlib.sha256(open(hit, "rb").read()).hexdigest()
    shown, width = full.lstrip("0") or "0", 64
else:
    full = shown = skinhash.namemc_id(skinhash.load_rgba(hit))
    width = 16
if shown != name:
    print(f"FAIL id {shown} != file name {name}")
    sys.exit(1)
ok = [p for p in pats if patterns.parse_line(p, width).match(full)]
if not ok:
    print(f"FAIL {full} matches none of {pats}")
    sys.exit(1)
print(f"OK {shown} matches {ok[0]}")
