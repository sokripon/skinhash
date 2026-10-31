"""Check that no search pixel or watermark pixel is ever taken from RESERVED.
Run: py test_reserved.py   (exit code 0 = all good)"""
import os
import random
import sys

import skinhash as sh

HERE = os.path.dirname(os.path.abspath(__file__))
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails += 1


def skin_clear():
    return bytearray(sh.W * sh.H * 4)


def skin_opaque_except(free):
    s = bytearray(b"\x80\x80\x80\xff" * (sh.W * sh.H))
    for x, y in free:
        i = (y * sh.W + x) * 4
        s[i:i + 4] = b"\x00\x00\x00\x00"
    return s


def skin_noise(seed):
    r = random.Random(seed)
    s = bytearray()
    for _ in range(sh.W * sh.H):
        s += bytes([r.randrange(256) for _ in range(3)]) + bytes([0 if r.random() < 0.6 else 255])
    return s


skins = {
    "steve": sh.load_rgba(os.path.join(HERE, "skins", "steve.png")),
    "all transparent": skin_clear(),
    "only reserved pixels clear": skin_opaque_except(sh.RESERVED),
    "reserved + (0-5,7) opaque": None,
    "noise seed 1": skin_noise(1),
    "noise seed 2": skin_noise(2),
}
s = skin_clear()
for x in range(6):
    i = (7 * sh.W + x) * 4
    s[i + 3] = 255
skins["reserved + (0-5,7) opaque"] = s

# the set itself
check({(0, 0), (2, 2), (0, 32), (3, 35), (52, 32), (55, 35), (56, 16), (61, 18), (25, 0), (24, 1), (37, 6)} <= sh.RESERVED
      and (4, 32) not in sh.RESERVED and (62, 16) not in sh.RESERVED and (0, 7) not in sh.RESERVED,
      f"RESERVED has {len(sh.RESERVED)} pixels and the expected edges")
check(not (set((x, 7) for x in range(6)) & sh.RESERVED), "preferred watermark spot (0-5,7) is not reserved")

wm_payload = sh.wm_pixels(sh.WATERMARKS[sh.WATERMARK_VERSION])
for name, base in skins.items():
    for unused_only in (False, True):
        kp = sh.knob_pixels(base, unused_only)
        check(not (set(kp) & sh.RESERVED), f"{name}: knob_pixels(unused_only={unused_only}) avoids RESERVED ({len(kp)} pixels)")
        try:
            nk, elig = sh.namemc_knob_pixels(base, unused_only)
            check(not (set(nk) & sh.RESERVED) and not (set(elig) & sh.RESERVED),
                  f"{name}: namemc_knob_pixels(unused_only={unused_only}) avoids RESERVED")
        except SystemExit:
            check(True, f"{name}: namemc_knob_pixels(unused_only={unused_only}): too few pixels, refused")
    for avoid in (set(), sh.MOJANG_TAIL):
        r = bytearray(base)
        wm = sh.apply_watermark(r, avoid=avoid)
        check(not (set(wm) & sh.RESERVED), f"{name}: watermark {wm[:1]}..{wm[-1:]} avoids RESERVED")
        check(not wm or sh.find_watermark(r)[0] == sh.WATERMARK_VERSION, f"{name}: watermark detected as v{sh.WATERMARK_VERSION}")
        if name in ("steve", "all transparent"):
            check(wm == [(x, 7) for x in range(6)], f"{name}: default watermark spot is (0-5,7)")
    # full flow: watermark first, then the knobs the searches would use
    r = bytearray(base)
    wm = sh.apply_watermark(r, avoid=sh.MOJANG_TAIL)
    nk, elig = (None, None)
    try:
        nk, elig = sh.namemc_knob_pixels(r, False, avoid=set(wm))
        check(not (set(nk) & (sh.RESERVED | set(wm))), f"{name}: NameMC knobs {nk} avoid RESERVED and the watermark")
    except SystemExit:
        pass

# length tuning pixels (Mojang job builder): never reserved, watermark, tail, last row or visible
for name, base in skins.items():
    r = bytearray(base)
    wm = sh.apply_watermark(r, avoid=sh.MOJANG_TAIL)
    avoid = set(wm) | sh.MOJANG_TAIL
    tp = sh.tuning_pixels(r, avoid)
    bm, om = sh.masks()
    check(not (set(tp) & sh.RESERVED), f"{name}: tuning pixels ({len(tp)}) avoid RESERVED")
    check(not (set(tp) & avoid) and all(y < sh.H - 1 for _, y in tp), f"{name}: tuning pixels avoid the watermark, the tail and the last row")
    check(all(r[(y * sh.W + x) * 4 + 3] == 0 and (x, y) not in bm for x, y in tp), f"{name}: tuning pixels are alpha 0 and not base layer")
    # a tuned copy (random RGB on up to 6 of them) must pass verify; the same change on a reserved pixel must not
    rng = random.Random(3)
    t = bytearray(r)
    for x, y in rng.sample(tp, min(6, len(tp))):
        t[(y * sh.W + x) * 4:(y * sh.W + x) * 4 + 3] = bytes(rng.randrange(256) for _ in range(3))
    check(sh.reserved_change(r, t) is None, f"{name}: tuned image has no reserved pixel changed")
    bad = bytearray(r)
    px = next((p for p in sorted(sh.RESERVED) if r[(p[1] * sh.W + p[0]) * 4 + 3] == 0), None)
    if px:
        bad[(px[1] * sh.W + px[0]) * 4:(px[1] * sh.W + px[0]) * 4 + 3] = b"	"
        check(sh.reserved_change(r, bad) == px, f"{name}: a change at reserved pixel {px} is detected")

# the fixed tails used by the fast modes are not reserved
check(not ({(sh.W - 6 + i, sh.H - 1) for i in range(6)} & sh.RESERVED), "Mojang fast-mode tail (58-63,63) not reserved")
check(not ({(sh.W - 9 + i, sh.H - 1) for i in range(3)} & sh.RESERVED), "NameMC default knobs (55-57,63) not reserved")

# NameMC fallback: block the default knobs so the next ones are chosen
blocked = skin_clear()
for x in range(55, 58):
    blocked[((sh.H - 1) * sh.W + x) * 4 + 3] = 255
nk, _ = sh.namemc_knob_pixels(blocked, False)
check(not (set(nk) & sh.RESERVED) and not set(nk) & {(55, 63), (56, 63), (57, 63)}, f"NameMC fallback knobs {nk} avoid RESERVED")

print("ALL PASSED" if not fails else f"{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
