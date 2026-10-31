"""Independent check of a NameMC hit written by the web page (Pillow + hashlib).
usage: py namemc_check.py BASE.png HIT.png PREFIX [--unused-only]
Checks: NameMC id of HIT equals its filename and starts with PREFIX; only three knob pixels (alpha 0 -> 1, never
base layer) and the watermark (alpha 0 -> 0, unused area) differ from BASE. Prints OK or FAIL and the reason."""
import hashlib, os, struct, sys
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import skinhash  # only for the UV masks and the watermark detector

def nm_id(rgba):
    buf = bytearray(struct.pack(">II", 64, 64))
    for i in range(0, len(rgba), 4):
        buf += rgba[i:i + 4] if rgba[i + 3] else b"\0\0\0\0"
    return hashlib.sha256(buf).hexdigest()[:16]

def load(p):
    im = Image.open(p)
    assert im.size == (64, 64), "not 64x64"
    return im.convert("RGBA").tobytes()

def fail(msg):
    print("FAIL", msg); sys.exit(1)

base_p, hit_p, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
unused_only = "--unused-only" in sys.argv
base, hit = load(base_p), load(hit_p)
want = os.path.splitext(os.path.basename(hit_p))[0]
got = nm_id(hit)
if got != want: fail(f"NameMC id {got} != filename {want}")
if not got.startswith(prefix): fail(f"id {got} does not start with {prefix}")
masks_base, masks_over = skinhash.masks()
knobs = 0
for y in range(64):
    for x in range(64):
        i = (y * 64 + x) * 4
        if hit[i:i + 4] == base[i:i + 4]: continue
        unused = (x, y) not in masks_base and (x, y) not in masks_over
        if base[i + 3] == 0 and hit[i + 3] == 1 and (x, y) not in masks_base and (unused or not unused_only):
            knobs += 1
        elif base[i + 3] == 0 and hit[i + 3] == 0 and unused:
            pass
        else:
            fail(f"unexpected change at ({x},{y})")
if knobs != 3: fail(f"{knobs} knob pixels changed, expected 3")
if not skinhash.find_watermark(bytearray(hit)): fail("no watermark")
print(f"OK {got} knobs=3 watermark present")
