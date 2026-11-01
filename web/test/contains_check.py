"""Independent check of a Mojang hit of a 'contains' search written by the web page.
usage: py contains_check.py HIT.png PATTERN
Checks: SHA-256 of the file, leading zeros dropped, equals the file name and contains PATTERN. Prints OK or FAIL."""
import hashlib, os, sys

hit, pat = sys.argv[1], sys.argv[2].lower()
trimmed = hashlib.sha256(open(hit, "rb").read()).hexdigest().lstrip("0") or "0"
name = os.path.splitext(os.path.basename(hit))[0]
if trimmed != name:
    print(f"FAIL id {trimmed} != file name {name}")
    sys.exit(1)
if pat not in trimmed:
    print(f"FAIL {pat} not in the trimmed id {trimmed}")
    sys.exit(1)
print(f"OK {pat} in {trimmed}")
