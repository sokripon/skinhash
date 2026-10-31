"""Helper around hashfinder.exe: knob-pixel selection, Mojang re-encoding, verification.

  py skinhash.py predict SKIN.png              -> texture id Mojang would assign
  py skinhash.py knobs SKIN.png [--unused-only] -> show which pixels the search may change
  py skinhash.py search SKIN.png [finder options...] [--unused-only] [--generic] [--k N] [--out DIR]
      fast mode (fastfinder.exe, permutes the RGB of the last K pixels) is used when those
      pixels are all invisible; otherwise (or with --generic) hashfinder.exe is used.
      -p HEX matches the trimmed id (leading zeros dropped, as in the URL); -z N = N leading zeros.
      --gpu: dump a job with fastfinder and search on the GPU with gpufinder.exe (CUDA); falls back
      to the CPU fastfinder if gpufinder is missing or fails its self-check.
      --all: keep going until every -p/-s/-z target has a hit (restarts without each found
      target, so no duplicates pile up; -n is ignored).
      Every search adds an invisible watermark (6 transparent unused pixels whose RGB spells
      "sokripon skinhash" + 02, normally at (0,0)-(5,0)); --no-watermark disables it.
  py skinhash.py verify BASE.png FOUND.png     -> check FOUND looks identical in game and hashes to its name
  py skinhash.py detect FILE|ID|URL...         -> was this skin made by skinhash? (IDs/URLs are downloaded)
"""
import hashlib
import os
import struct
import subprocess
import sys
import zlib

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 64
ROW = 1 + W * 4

# Box-UV parts of the 64x64 player model: (u, v, w, h, d). Arms listed both
# classic (w=4) and slim (w=3) so a pixel counts as used if either model samples it.
BASE_PARTS = [
    (0, 0, 8, 8, 8),     # head
    (16, 16, 8, 12, 4),  # body
    (40, 16, 4, 12, 4), (40, 16, 3, 12, 4),  # right arm
    (32, 48, 4, 12, 4), (32, 48, 3, 12, 4),  # left arm
    (0, 16, 4, 12, 4),   # right leg
    (16, 48, 4, 12, 4),  # left leg
]
OVERLAY_PARTS = [
    (32, 0, 8, 8, 8),    # hat
    (16, 32, 8, 12, 4),  # jacket
    (40, 32, 4, 12, 4), (40, 32, 3, 12, 4),  # right sleeve
    (48, 48, 4, 12, 4), (48, 48, 3, 12, 4),  # left sleeve
    (0, 32, 4, 12, 4),   # right pants
    (0, 48, 4, 12, 4),   # left pants
]


# Watermark: a run of horizontally adjacent, fully transparent pixels in an unused part
# of the layout whose RGB bytes, read left to right, spell the payload. Invisible in game
# and in image viewers; detect with `skinhash.py detect`. Preferred spot starts at (0,0),
# the unused top-left corner of the head; if that is not free (alpha > 0) the first free
# unused run in scan order is used, and detection scans the whole image.
#   v2 (current): "sokripon skinhash" + 0x02            -> 6 pixels, (0,0)-(5,0)
#   v1 (old):     "SKHAS" + 0x01                        -> 2 pixels, (0,0)-(1,0)
WATERMARKS = {2: b"sokripon skinhash\x02", 1: b"SKHAS\x01"}
WATERMARK_VERSION = 2
WATERMARK_SPOT = (0, 0)


def wm_pixels(payload):
    payload += b"\x00" * (-len(payload) % 3)
    return [payload[i:i + 3] for i in range(0, len(payload), 3)]


def box_pixels(u, v, w, h, d):
    rects = [
        (u + d, v, w, d), (u + d + w, v, w, d),                     # top, bottom
        (u, v + d, d, h), (u + d, v + d, w, h),                     # right, front
        (u + d + w, v + d, d, h), (u + 2 * d + w, v + d, w, h),     # left, back
    ]
    for x0, y0, rw, rh in rects:
        for y in range(y0, y0 + rh):
            for x in range(x0, x0 + rw):
                yield x, y


def masks():
    base, over = set(), set()
    for p in BASE_PARTS:
        base.update(box_pixels(*p))
    for p in OVERLAY_PARTS:
        over.update(box_pixels(*p))
    return base, over


def load_rgba(path):
    im = Image.open(path)
    if im.size != (W, H):
        sys.exit(f"{path}: need a 64x64 skin, got {im.size[0]}x{im.size[1]}")
    return bytearray(im.convert("RGBA").tobytes())


def raw_scanlines(rgba):
    out = bytearray()
    for y in range(H):
        out.append(0)
        out += rgba[y * W * 4:(y + 1) * W * 4]
    return bytes(out)


def mojang_png(rgba):
    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data))
    c = zlib.compressobj(4, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    idat = c.compress(raw_scanlines(rgba)) + c.flush()
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def texture_id(png_bytes):
    return hashlib.sha256(png_bytes).hexdigest().lstrip("0") or "0"


def knob_pixels(rgba, unused_only=False):
    """Pixels whose RGB never shows in game: unused layout pixels, plus overlay
    pixels that are fully transparent. Latest in the file first (faster search)."""
    base, over = masks()
    knobs = []
    for y in range(H):
        for x in range(W):
            a = rgba[(y * W + x) * 4 + 3]
            unused = (x, y) not in base and (x, y) not in over
            clear_overlay = (x, y) in over and a == 0
            if unused and a == 0 or (not unused_only and clear_overlay):
                knobs.append((x, y))
    knobs.sort(key=lambda p: (p[1], p[0]), reverse=True)
    return knobs


def knob_offsets(knobs, nbytes=8):
    offs = []
    for x, y in knobs:
        for c in range(3):  # RGB only; alpha stays 0
            offs.append(y * ROW + 1 + x * 4 + c)
            if len(offs) == nbytes:
                return offs, knobs[:(nbytes + 2) // 3]
    sys.exit(f"only {len(knobs)} usable transparent pixels; need {(nbytes + 2) // 3}")


def find_watermark(rgba):
    """Return (version, (x, y) of first pixel) of any known watermark, or None."""
    for ver, payload in WATERMARKS.items():
        px = wm_pixels(payload)
        n = len(px)
        for y in range(H):
            for x in range(W - n + 1):
                if all(rgba[(y * W + x + j) * 4:(y * W + x + j) * 4 + 3] == px[j]
                       and rgba[(y * W + x + j) * 4 + 3] == 0 for j in range(n)):
                    return ver, (x, y)
    return None


def watermark_spot(rgba, n, avoid=()):
    """Pick n adjacent unused, fully transparent pixels in one row for the watermark."""
    base, over = masks()
    def free(x, y):
        return ((x, y) not in base and (x, y) not in over and (x, y) not in avoid
                and rgba[(y * W + x) * 4 + 3] == 0)
    def run(x, y):
        return x + n <= W and all(free(x + j, y) for j in range(n))
    if run(*WATERMARK_SPOT):
        return WATERMARK_SPOT
    for y in range(H):
        for x in range(W - n + 1):
            if run(x, y):
                return x, y
    return None


def apply_watermark(rgba, avoid=()):
    """Write the current watermark in place; returns its pixels, or [] if no room."""
    px = wm_pixels(WATERMARKS[WATERMARK_VERSION])
    found = find_watermark(rgba)
    if found and found[0] == WATERMARK_VERSION:
        x, y = found[1]
        return [(x + j, y) for j in range(len(px))]
    spot = watermark_spot(rgba, len(px), avoid)
    if spot is None:
        return []
    x, y = spot
    for j, rgb in enumerate(px):
        i = (y * W + x + j) * 4
        rgba[i:i + 4] = rgb + b"\x00"
    return [(x + j, y) for j in range(len(px))]


def cmd_detect(targets):
    import urllib.request
    import io
    for t in targets:
        if os.path.exists(t):
            data, label = open(t, "rb").read(), t
        else:
            tid = t.rstrip("/").rsplit("/", 1)[-1].lower()
            url = f"https://textures.minecraft.net/texture/{tid}"
            data, label = urllib.request.urlopen(url, timeout=20).read(), url
        im = Image.open(io.BytesIO(data))
        if im.size != (W, H):
            print(f"no   {label} (not a 64x64 skin)")
            continue
        f = find_watermark(bytearray(im.convert("RGBA").tobytes()))
        print(f"{'YES' if f else 'no '}  {label}" + (f" (watermark v{f[0]} at {f[1]})" if f else ""))


def cmd_predict(path):
    png = mojang_png(load_rgba(path))
    print(texture_id(png))


def cmd_knobs(path, unused_only):
    rgba = load_rgba(path)
    k = knob_pixels(rgba, unused_only)
    offs, used = knob_offsets(k)
    print(f"{len(k)} invisible pixels available; search uses {used} (bytes {offs})")


# Exit codes meaning "stopped by Ctrl+C": Windows STATUS_CONTROL_C_EXIT, POSIX SIGINT.
INTERRUPT_CODES = {0xC000013A, -2, 130}


def stopped(msg="stopped (Ctrl+C)"):
    print(f"\n{msg}", flush=True)
    sys.exit(130)


STOPPED_BY_US = "stopped-by-wrapper"
ROLLOVER_RC = 7      # gpufinder: the permutation space of this value set is exhausted
LAST_ROLLOVER = None  # (cumulative tried, cumulative seconds) from the last "ROLLOVER" line


def run_finder(cmd, rgba, on_hit=None):
    """Run a finder, verifying hits. Ctrl+C ends the whole search (never a fallback).
    on_hit(why) -> True asks the finder to stop; remaining output is still drained and
    verified, and STOPPED_BY_US is returned."""
    global LAST_ROLLOVER
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    asked_stop = False
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("ROLLOVER "):
                try:
                    kv = dict(x.split("=", 1) for x in line.split()[1:])
                    LAST_ROLLOVER = (int(kv["tried"]), float(kv["secs"]))
                except (KeyError, ValueError):
                    LAST_ROLLOVER = None
            if line.startswith("HIT "):
                f = line.rsplit("file=", 1)[1].strip()
                ok, msg = verify(rgba, f)
                print(f"  verify: {'OK' if ok else 'FAIL'} {msg}", flush=True)
                if ok and on_hit and on_hit(line.split()[1], f) and not asked_stop:
                    asked_stop = True
                    proc.terminate()
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        stopped()
    if asked_stop:
        return STOPPED_BY_US
    if rc in INTERRUPT_CODES or rc & 0xFFFFFFFF in INTERRUPT_CODES:
        stopped()
    return rc


def split_targets(args):
    """Split finder args into (targets, other args). Targets: 'prefix:x', 'suffix:x', 'zeros:N'."""
    targets, rest, i = [], [], 0
    while i < len(args):
        a = args[i]
        if a in ("-p", "-s", "-z") and i + 1 < len(args):
            kind = {"-p": "prefix", "-s": "suffix", "-z": "zeros"}[a]
            targets.append(f"{kind}:{args[i + 1].lower()}")
            i += 2
        elif a == "-n" and i + 1 < len(args):
            i += 2  # --all decides when to stop
        else:
            rest.append(a)
            i += 1
    return targets, rest


def target_args(targets):
    out = []
    for t in targets:
        kind, val = t.split(":", 1)
        out += [{"prefix": "-p", "suffix": "-s", "zeros": "-z"}[kind], val]
    return out


def hit_target(why, targets):
    """Which remaining target does a finder 'why' (e.g. prefix:cafe, zeros:9) satisfy?"""
    kind, val = why.split(":", 1)
    for t in targets:
        tk, tv = t.split(":", 1)
        if tk == kind and (tv == val or (kind == "zeros" and int(val) >= int(tv))):
            return t
    return None


def drop_seed(args):
    """args without -S SEED (a rollover must not rebuild the same value set)."""
    out, i = [], 0
    while i < len(args):
        if args[i] == "-S" and i + 1 < len(args):
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def gpu_search(rawp, k, outdir, args, rgba, ext, on_hit=None):
    """Returns True if the GPU search ran to completion, False if the caller should fall back.
    The job builder (fastfinder --auto) picks the largest K <= k and a value set whose variable region plus SHA
    padding fit in one 64-byte block; a single value set only holds m! orderings (15! = 1.3e12 for K=5), so when
    gpufinder reports it exhausted (exit 7) a fresh job with a new random value set is dumped and the search
    continues with cumulative tried/time counters."""
    global LAST_ROLLOVER
    gpu = os.path.join(HERE, "gpufinder" + ext)
    fast = os.path.join(HERE, "fastfinder" + ext)
    if not os.path.exists(gpu):
        print("gpufinder not found (build it with build.ps1); falling back to CPU fastfinder", flush=True)
        return False
    job = os.path.join(outdir, "gpu.job")
    tried_off, time_off, rolls = 0, 0.0, 0
    while True:
        dump_args = args if rolls == 0 else drop_seed(args)
        r = subprocess.run([fast, rawp, str(k), outdir, *dump_args, "--auto", "--dump-job", job], stdout=subprocess.DEVNULL)
        if r.returncode in INTERRUPT_CODES or r.returncode & 0xFFFFFFFF in INTERRUPT_CODES:
            stopped()
        if r.returncode != 0 or not os.path.exists(job):
            print(f"fastfinder --dump-job failed (rc={r.returncode}); falling back to CPU fastfinder", flush=True)
            return False
        print("gpu mode: gpufinder" if rolls == 0 else f"gpu mode: new value set #{rolls + 1}", flush=True)
        extra = ["--tried-offset", str(tried_off), "--time-offset", f"{time_off:.3f}"] if rolls else []
        LAST_ROLLOVER = None
        rc = run_finder([gpu, job, outdir, *args, *extra], rgba, on_hit)
        try:
            os.remove(job)
        except OSError:
            pass
        if rc == STOPPED_BY_US:
            return rc
        if rc == ROLLOVER_RC and LAST_ROLLOVER:
            tried_off, time_off = LAST_ROLLOVER
            rolls += 1
            print(f"value set exhausted after {tried_off} candidates in total; switching to a new value set", flush=True)
            continue
        if rc != 0:
            print(f"gpufinder failed (rc={rc}, 5 = correctness gate failed); falling back to CPU fastfinder", flush=True)
            return False
        return True


def cmd_search(path, args, unused_only, generic=False, k=6, out=None, gpu=False, watermark=True,
               all_mode=False):
    orig = load_rgba(path)
    rgba = bytearray(orig)
    tail = {(W - k + i, H - 1) for i in range(k)}  # keep the fast-mode pixels free
    wm = apply_watermark(rgba, avoid=tail) if watermark else []
    if watermark:
        print(f"watermark at {wm}" if wm else "warning: no free unused pixels for the watermark", flush=True)
    kp = [p for p in knob_pixels(rgba, unused_only) if p not in wm]
    offs, used = knob_offsets(kp)
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = out or os.path.join(HERE, "found", name)
    os.makedirs(outdir, exist_ok=True)
    rawp = os.path.join(outdir, "base.raw")
    with open(rawp, "wb") as f:
        f.write(raw_scanlines(rgba))
    print(f"original id {texture_id(mojang_png(orig))}; knob pixels {used}", flush=True)
    rgba = orig  # hits are verified against the untouched original skin
    ext = ".exe" if os.name == "nt" else ""
    kset = set(kp)
    fast_ok = not generic and all((W - k + i, H - 1) in kset for i in range(k))

    def launch(args, on_hit=None):
        if fast_ok:
            if gpu:
                r = gpu_search(rawp, k, outdir, args, rgba, ext, on_hit)
                if r:
                    return r
            cmd = [os.path.join(HERE, "fastfinder" + ext), rawp, str(k), outdir, *args]
            print(f"fast mode: last {k} pixels", flush=True)
            rc = run_finder(cmd, rgba, on_hit)
            if rc != 4:
                return rc
            print("fastfinder: structure unsupported, falling back to hashfinder", flush=True)
        elif not generic:
            if gpu:
                print("--gpu needs the last pixels to be invisible; ignoring", flush=True)
            print("last pixels are not all invisible; using generic hashfinder", flush=True)
        return run_finder([os.path.join(HERE, "hashfinder" + ext), rawp, ",".join(map(str, offs)),
                           outdir, *args], rgba, on_hit)

    if not all_mode:
        launch(args)
        return
    remaining, rest = split_targets(args)
    if not remaining:
        sys.exit("--all needs at least one -p, -s or -z target")
    found = {}
    while remaining:
        print(f"--all: still looking for {', '.join(remaining)}", flush=True)
        newly, keep, firsts = [], set(), {}
        before = {f for f in os.listdir(outdir) if f.endswith(".png")}

        def on_hit(why, path):
            t = hit_target(why, remaining)
            if t and t not in newly:
                newly.append(t)
                keep.add(os.path.basename(path))
                firsts[t] = os.path.splitext(os.path.basename(path))[0]
            return bool(newly)
        r = launch(target_args(remaining) + rest, on_hit)
        for t in newly:
            remaining.remove(t)
            found[t] = firsts[t]
        # keep only the first verified hit per target; drop the rest of that batch
        extra = {f for f in os.listdir(outdir) if f.endswith(".png")} - before - keep
        for f in extra:
            os.remove(os.path.join(outdir, f))
        if extra:
            print(f"--all: removed {len(extra)} duplicate hit(s) from the same batch", flush=True)
        if r != STOPPED_BY_US and not newly:
            print(f"finder exited (rc={r}) before all targets were found", flush=True)
            break
    if not remaining:
        print("--all: every target found:", flush=True)
        for t, f in found.items():
            print(f"  {t:<20} https://textures.minecraft.net/texture/{f}", flush=True)


def verify(base_rgba, found_path):
    data = open(found_path, "rb").read()
    want = os.path.splitext(os.path.basename(found_path))[0]
    if texture_id(data) != want:
        return False, "file hash does not match its name"
    rgba = load_rgba(found_path)
    if mojang_png(rgba) != data:
        return False, "Python zlib re-encode differs from the searcher's bytes"
    base, over = masks()
    for y in range(H):
        for x in range(W):
            i = (y * W + x) * 4
            if rgba[i:i + 4] == base_rgba[i:i + 4]:
                continue
            visible = (x, y) in base or ((x, y) in over and rgba[i + 3] != 0)
            if visible or rgba[i + 3] != base_rgba[i + 3]:
                return False, f"visible change at ({x},{y})"
    wm = "" if find_watermark(rgba) else " (no watermark)"
    return True, f"https://textures.minecraft.net/texture/{want}{wm}"


def main():
    a = sys.argv[1:]
    unused_only = "--unused-only" in a
    a = [x for x in a if x != "--unused-only"]
    generic = "--generic" in a
    a = [x for x in a if x != "--generic"]
    gpu = "--gpu" in a
    a = [x for x in a if x != "--gpu"]
    watermark = "--no-watermark" not in a
    a = [x for x in a if x != "--no-watermark"]
    all_mode = "--all" in a
    a = [x for x in a if x != "--all"]
    k = 6
    out = None
    if "--out" in a:
        i = a.index("--out")
        out = os.path.abspath(a[i + 1])
        del a[i:i + 2]
    if "--k" in a:
        i = a.index("--k")
        k = int(a[i + 1])
        del a[i:i + 2]
    if len(a) >= 2 and a[0] == "predict":
        cmd_predict(a[1])
    elif len(a) >= 2 and a[0] == "knobs":
        cmd_knobs(a[1], unused_only)
    elif len(a) >= 2 and a[0] == "search":
        cmd_search(a[1], a[2:], unused_only, generic, k, out, gpu, watermark, all_mode)
    elif len(a) >= 2 and a[0] == "detect":
        cmd_detect(a[1:])
    elif len(a) == 3 and a[0] == "verify":
        ok, msg = verify(load_rgba(a[1]), a[2])
        print("OK" if ok else "FAIL", msg)
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stopped()
