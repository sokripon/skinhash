"""skinhash: find invisible variations of a Minecraft skin whose Mojang texture id and/or NameMC
id looks cool.

  py skinhash.py --help           the commands
  py skinhash.py COMMAND --help   options and examples of one command

Commands: mojang, namemc, both (searches), collide (NameMC id collisions), ids, verify, detect, pixels, bench.

Every argument that takes a skin accepts a PNG path, a texture id or textures.minecraft.net URL, a
UUID or a Minecraft username (looked up with the Mojang API, saved as skins/<name>.png), e.g.
  py skinhash.py mojang jeb_ -p c0ffee
  py skinhash.py ids 853c80ef-3c37-49fd-aa49-938b674adae6
Prefixes name:, uuid: and texture: force the interpretation.
The searches drive the native finders (hashfinder, fastfinder, gpufinder, namemcfinder, namemcgpu,
built with build.ps1) and verify every hit. See README.md and docs/how-it-works.md.
"""
import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 64
ROW = 1 + W * 4

# Box-UV parts of the 64x64 player model: (u, v, w, h, d). Arms are listed as classic
# (w=4) and slim (w=3), so a pixel counts as used if either model samples it.
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


# Watermark: horizontally adjacent, fully transparent, unused pixels whose RGB bytes spell
# the payload left to right. It goes at (0,7)-(5,7), unused and clear of the pixels other mods read
# (see RESERVED). If that spot has alpha > 0, the first free unused, non-reserved run in scan order
# is used instead. Detection scans the whole image and knows every version.
#   v3 (current): "sokripon skinhash" + 0x03            -> 6 pixels, (0,7)-(5,7)
#   v2 (old):     "sokripon skinhash" + 0x02            -> 6 pixels, (0,0)-(5,0)
#   v1 (old):     "SKHAS" + 0x01                        -> 2 pixels, (0,0)-(1,0)
WATERMARKS = {3: b"sokripon skinhash\x03", 2: b"sokripon skinhash\x02", 1: b"SKHAS\x01"}
WATERMARK_VERSION = 3
WATERMARK_SPOT = (0, 7)


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


# Pixels other software reads as data. No search pixel and no watermark pixel may ever come from here.
# Same set in web/js/model.js.
RESERVED = (
    {(x, y) for x in range(0, 4) for y in range(32, 36)}      # Ears: marker block x0-3, y32-35
    | {(x, y) for x in range(52, 56) for y in range(32, 36)}  # Ears: emissive palette x52-55, y32-35
    | {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (2, 2)}  # Mine Little Pony triggers (CPM header = (0,0))
    | {(x, y) for x in range(56, 62) for y in range(16, 19)}  # Tails mod rows x56-61, y16-18
    | set(box_pixels(24, 0, 6, 6, 1))                         # vanilla deadmau5 ear box x24-37, y0-6
)


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
    """Pixels whose RGB never shows in game: unused layout pixels and fully transparent
    overlay pixels. Latest in the file first, which makes the search faster."""
    base, over = masks()
    knobs = []
    for y in range(H):
        for x in range(W):
            a = rgba[(y * W + x) * 4 + 3]
            if (x, y) in RESERVED:
                continue
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
    """(version, (x, y) of the first pixel) of a known watermark, or None."""
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
    """n adjacent unused, fully transparent pixels in one row, or None."""
    base, over = masks()
    def free(x, y):
        return ((x, y) not in base and (x, y) not in over and (x, y) not in avoid
                and (x, y) not in RESERVED and rgba[(y * W + x) * 4 + 3] == 0)
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
    """Write the current watermark in place. Returns its pixels, or [] if there is no room."""
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


# Exit codes for Ctrl+C: Windows STATUS_CONTROL_C_EXIT, POSIX SIGINT.
INTERRUPT_CODES = {0xC000013A, -2, 130}


def stopped(msg="stopped (Ctrl+C)"):
    print(f"\n{msg}", flush=True)
    sys.exit(130)


STOPPED_BY_US = "stopped-by-wrapper"
ROLLOVER_RC = 7      # gpufinder: this value set is exhausted
LAST_ROLLOVER = None  # (cumulative tried, cumulative seconds) from the last ROLLOVER line
LAST_RATE = None      # candidates/s from the last RATE line (benchmark)
GPU_FAILED = False    # the GPU search fell back to the CPU


PROGRESS_RE = re.compile(r"^\[\s*(\d+)s\] tried=(\d+)\s+rate=(\d+)/s \(avg (\d+)/s\)\s+hits=(\d+)\s+best_zeros=(\d+)")


def pretty_progress(line):
    """Make the finders' progress/done lines readable: 141_180_272_640, elapsed as h:mm:ss."""
    m = PROGRESS_RE.match(line)
    if m:
        secs, tried, rate, avg, hits, best = (int(g) for g in m.groups())
        h, rest = divmod(secs, 3600)
        el = f"{h}:{rest // 60:02d}:{rest % 60:02d}"
        return (f"[{el:>9}] tried={tried:_}  rate={rate:_}/s (avg {avg:_}/s)  "
                f"hits={hits}  best_zeros={best}\n")
    if line.startswith("done:"):
        return re.sub(r"\d{5,}", lambda d: f"{int(d.group()):_}", line)
    return line


def run_finder(cmd, rgba, on_hit=None, verifier=None):
    """Run a finder and verify its hits. Ctrl+C ends the whole search, never a fallback.
    If on_hit(why, path) returns True the finder is stopped; the rest of its output is
    still verified and STOPPED_BY_US is returned."""
    global LAST_ROLLOVER, LAST_RATE
    # stderr (progress lines) is merged in so it can be reformatted
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    asked_stop = False
    try:
        for line in proc.stdout:
            print(pretty_progress(line), end="", flush=True)
            if line.startswith("RATE "):
                try:
                    LAST_RATE = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
            if line.startswith("ROLLOVER "):
                try:
                    kv = dict(x.split("=", 1) for x in line.split()[1:])
                    LAST_ROLLOVER = (int(kv["tried"]), float(kv["secs"]))
                except (KeyError, ValueError):
                    LAST_ROLLOVER = None
            if line.startswith("HIT "):
                f = line.rsplit("file=", 1)[1].strip()
                ok, msg = (verifier or verify)(rgba, f)
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


def target_args(targets):
    out = []
    for t in targets:
        kind, val = t.split(":", 1)
        out += [{"prefix": "-p", "suffix": "-s", "zeros": "-z"}[kind], val]
    return out


def hit_target(why, targets):
    """The remaining target satisfied by a finder 'why' such as prefix:cafe or zeros:9."""
    kind, val = why.split(":", 1)
    for t in targets:
        tk, tv = t.split(":", 1)
        if tk == kind and (tv == val or (kind == "zeros" and int(val) >= int(tv))):
            return t
    return None

def drop_seed(args):
    """args without -S SEED, so a rollover does not rebuild the same value set."""
    out, i = [], 0
    while i < len(args):
        if args[i] == "-S" and i + 1 < len(args):
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def gpu_search(rawp, k, outdir, args, rgba, ext, on_hit=None):
    """True if the GPU search ran to completion, False if the caller should fall back.
    fastfinder --auto picks the largest K <= k and a value set that fits one SHA block.
    One value set holds only m! orderings (15! = 1.3e12 for K=5). When gpufinder exits 7
    (exhausted), a new job with a fresh random value set is dumped and the search goes on
    with cumulative tried/time counters."""
    global LAST_ROLLOVER, GPU_FAILED
    gpu = os.path.join(HERE, "gpufinder" + ext)
    fast = os.path.join(HERE, "fastfinder" + ext)
    if not os.path.exists(gpu):
        print("gpufinder not found (build it with build.ps1); falling back to CPU fastfinder", flush=True)
        GPU_FAILED = True
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
            GPU_FAILED = True
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
            GPU_FAILED = True
            return False
        return True


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
            # NameMC knob: alpha 0 -> 1 on an overlay/unused pixel, any RGB. Not shown in game
            # (alpha 1 of 255 is invisible), and not a pixel other software reads.
            knob = (base_rgba[i + 3] == 0 and rgba[i + 3] == 1 and (x, y) not in base
                    and (x, y) not in RESERVED)
            if knob:
                continue
            if visible or rgba[i + 3] != base_rgba[i + 3]:
                return False, f"visible change at ({x},{y})"
    wm = "" if find_watermark(rgba) else " (no watermark)"
    return True, f"https://textures.minecraft.net/texture/{want}{wm}"


# ---- NameMC mode -------------------------------------------------------------------------------
# NameMC id = first 8 bytes of SHA-256( be32(w) || be32(h) || RGBA, RGB = 0 where alpha == 0 ).
NAMEMC_KNOBS = 3
MOJANG_TAIL = {(x, H - 1) for x in range(W - 6, W)}  # pixels the Mojang fast modes permute


def namemc_id(rgba):
    buf = bytearray(struct.pack(">II", W, H))
    for i in range(0, len(rgba), 4):
        if rgba[i + 3]:
            buf += rgba[i:i + 4]
        else:
            buf += b"\x00\x00\x00\x00"
    return hashlib.sha256(buf).hexdigest()[:16]


def namemc_knob_pixels(rgba, unused_only=False, avoid=()):
    """The three NameMC knob pixels, in message order, plus the list of all eligible ones.
    Eligible: alpha 0 and unused (never sampled) or, unless unused_only, overlay. Never base-layer.
    The latest ones in the file are taken, except the Mojang fast-mode tail so both modes combine."""
    base, over = masks()
    elig = []
    for y in range(H):
        for x in range(W):
            if rgba[(y * W + x) * 4 + 3] or (x, y) in base or (x, y) in avoid or (x, y) in RESERVED:
                continue
            if (x, y) in over and unused_only:
                continue
            elig.append((x, y))
    elig.sort(key=lambda p: (p[1], p[0]), reverse=True)
    pref = [p for p in elig if p not in MOJANG_TAIL] or elig
    pick = pref[:NAMEMC_KNOBS]
    if len(pick) < NAMEMC_KNOBS:
        sys.exit(f"only {len(elig)} usable pixels for the NameMC knob; need {NAMEMC_KNOBS}")
    return sorted(pick, key=lambda p: (p[1], p[0])), elig


def verify_namemc(base_rgba, found_path, nk=NAMEMC_KNOBS, want=None):
    """FOUND hashes to its name (or to WANT) under the NameMC formula, differs from BASE only in nk knob
    pixels (alpha 0 -> 1 on overlay/unused pixels) and the watermark (RGB under alpha 0 on unused pixels)."""
    data = open(found_path, "rb").read()
    want = want or os.path.splitext(os.path.basename(found_path))[0]
    rgba = load_rgba(found_path)
    if namemc_id(rgba) != want:
        return False, "NameMC id of the file does not match its name"
    if mojang_png(rgba) != data:
        return False, "file is not in Mojang's encoding"
    base, over = masks()
    knobs = 0
    for y in range(H):
        for x in range(W):
            i = (y * W + x) * 4
            if rgba[i:i + 4] == base_rgba[i:i + 4]:
                continue
            unused = (x, y) not in base and (x, y) not in over
            if (x, y) in RESERVED:
                return False, f"change at reserved pixel ({x},{y})"
            if base_rgba[i + 3] == 0 and rgba[i + 3] == 1 and ((x, y) in over or unused):
                knobs += 1
            elif base_rgba[i + 3] == 0 and rgba[i + 3] == 0 and unused:
                pass  # watermark: RGB under alpha 0 in the unused area
            else:
                return False, f"unexpected change at ({x},{y})"
    if knobs != nk:
        return False, f"{knobs} knob pixels changed, expected {nk}"
    moj = texture_id(data)
    return True, f"https://namemc.com/skin/{want}  mojang https://textures.minecraft.net/texture/{moj}"



# ---- shared helpers ----------------------------------------------------------------------------
EXT = ".exe" if os.name == "nt" else ""
SKIN_URL = "https://textures.minecraft.net/texture/"
NM_URL = "https://namemc.com/skin/"


def exe(name):
    return os.path.join(HERE, name + EXT)


def fmt_px(pixels):
    """[(58,63),...,(63,63)] -> '(58-63, 63)'. Consecutive pixels of one row are joined."""
    runs = []
    for x, y in sorted(pixels, key=lambda p: (p[1], p[0])):
        if runs and runs[-1][2] == y and runs[-1][1] + 1 == x:
            runs[-1][1] = x
        else:
            runs.append([x, x, y])
    return ", ".join(f"({a}, {y})" if a == b else f"({a}-{b}, {y})" for a, b, y in runs) or "none"


def px_state(rgba, p):
    """Why a pixel is or is not a free knob, in a few words."""
    base, over = masks()
    a = rgba[(p[1] * W + p[0]) * 4 + 3]
    if p in RESERVED:
        return "reserved (other mods read it)"
    if p in base:
        return "base layer (always drawn)"
    if p in over:
        return f"overlay with alpha {a}" if a else "overlay, transparent"
    return f"unused with alpha {a}" if a else "unused, transparent"


def target_match(t, shown, full):
    """Does an id satisfy target 'prefix:x' / 'suffix:x' / 'zeros:N'? shown = id as in the link,
    full = untrimmed hex digest."""
    kind, val = t.split(":", 1)
    if kind == "prefix":
        return shown.startswith(val)
    if kind == "suffix":
        return shown.endswith(val)
    return len(full) - len(full.lstrip("0")) >= int(val)


def first_match(targets, shown, full):
    return next((t for t in targets if target_match(t, shown, full)), None)


def finder_extra(threads, seed, bench=False):
    out = []
    if threads:
        out += ["-t", str(threads)]
    if seed is not None:
        out += ["-S", str(seed)]
    if bench:
        out.append("-b")
    return out


def search_args(targets, count, every, extra, bench=False):
    """The finder's own flags for the parsed options. --every-target stops by itself, so no -n."""
    a = target_args(targets)
    if not every and not bench:
        a += ["-n", str(count)]
    return a + extra


def run_targets(launch, targets, extra, outdir, url_prefix):
    """--every-target: run until every target has a verified hit. Restarts without the targets
    already found. Returns {target: path of its first hit}."""
    remaining = list(targets)
    found = {}
    while remaining:
        print(f"every-target: still looking for {', '.join(remaining)}", flush=True)
        newly, keep, firsts = [], set(), {}
        before = {f for f in os.listdir(outdir) if f.endswith(".png")}

        def on_hit(why, path):
            t = hit_target(why, remaining)
            if t and t not in newly:
                newly.append(t)
                keep.add(os.path.basename(path))
                firsts[t] = path
            return bool(newly)
        r = launch(target_args(remaining) + extra, on_hit)
        for t in newly:
            remaining.remove(t)
            found[t] = firsts[t]
        # keep the first verified hit per target, drop the rest of the batch
        dup = {f for f in os.listdir(outdir) if f.endswith(".png")} - before - keep
        for f in dup:
            os.remove(os.path.join(outdir, f))
        if dup:
            print(f"every-target: removed {len(dup)} duplicate hit(s) from the same batch", flush=True)
        if r != STOPPED_BY_US and not newly:
            print(f"finder exited (rc={r}) before all targets were found", flush=True)
            break
    if not remaining:
        print("every-target: every target found:", flush=True)
        for t, p in found.items():
            print(f"  {t:<20} {url_prefix}{os.path.splitext(os.path.basename(p))[0]}", flush=True)
    return found


# ---- Mojang search -----------------------------------------------------------------------------
def mojang_search(path, targets, count=1, every=False, engine="auto", no_generic=False,
                  unused_only=False, k=6, out=None, watermark=True, threads=None, seed=None, bench=False):
    """Search for Mojang texture ids. Returns the verified hit files (first hit per target with
    every=True)."""
    orig = load_rgba(path)
    rgba = bytearray(orig)
    tail_px = [(W - k + i, H - 1) for i in range(k)]
    wm = apply_watermark(rgba, avoid=set(tail_px)) if watermark else []  # the watermark must not take the tail
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

    kset = set(kp)  # reserved pixels are never in it
    bad = [p for p in tail_px if p not in kset]
    if engine == "generic":
        mode, why = "generic", "--engine generic"
    elif bad:
        mode = "generic"
        why = (f"the last {k} pixels are not all invisible: "
               + "; ".join(f"({x},{y}) {px_state(rgba, (x, y))}" for x, y in bad[:3]))
        if no_generic:
            sys.exit(f"fast mode is not possible for this skin and --no-generic forbids the generic search. {why}")
    elif engine == "cpu":
        mode, why = "cpu", "--engine cpu"
    elif engine == "gpu":
        mode, why = "gpu", "--engine gpu"
    elif os.path.exists(exe("gpufinder")):
        mode, why = "gpu", "gpufinder found, last pixels invisible"
    else:
        mode, why = "cpu", "gpufinder not built, last pixels invisible"
    label = {"gpu": "gpu (gpufinder)", "cpu": f"cpu fast mode (last {k} pixels)", "generic": "generic (hashfinder)"}[mode]
    print(f"engine: {label} - {why}", flush=True)

    def launch(fargs, on_hit=None):
        if mode in ("gpu", "cpu"):
            if mode == "gpu":
                r = gpu_search(rawp, k, outdir, fargs, orig, EXT, on_hit)
                if r:
                    return r
            print(f"fast mode: last {k} pixels", flush=True)
            rc = run_finder([exe("fastfinder"), rawp, str(k), outdir, *fargs], orig, on_hit)
            if rc != 4:
                return rc
            if no_generic:
                sys.exit("fastfinder reports this skin's compressed structure is unsupported; "
                         "drop --no-generic to use the generic search")
            print("fastfinder: structure unsupported, falling back to hashfinder", flush=True)
        return run_finder([exe("hashfinder"), rawp, ",".join(map(str, offs)), outdir, *fargs], orig, on_hit)

    extra = finder_extra(threads, seed, bench)
    if bench or not every:
        hits = []

        def collect(why_, p):
            hits.append(p)
            return False
        launch(search_args(targets, count, every, extra, bench), collect)
        return hits
    return list(run_targets(launch, targets, extra, outdir, SKIN_URL).values())


# ---- NameMC search -----------------------------------------------------------------------------
def namemc_search(path, targets, count=1, every=False, engine="auto", unused_only=False, out=None,
                  watermark=True, threads=None, seed=None, bench=False):
    """Search for NameMC ids. Returns the verified hit files (first hit per target with every=True)."""
    global GPU_FAILED
    orig = load_rgba(path)
    rgba = bytearray(orig)
    wm = apply_watermark(rgba, avoid=MOJANG_TAIL) if watermark else []
    if watermark:
        print(f"watermark at {wm}" if wm else "warning: no free unused pixels for the watermark", flush=True)
    knobs, elig = namemc_knob_pixels(rgba, unused_only, avoid=set(wm))
    pref = [(W - 9 + i, H - 1) for i in range(NAMEMC_KNOBS)]  # (55-57, 63)
    if knobs != pref and not unused_only:
        print(f"note: preferred pixels {pref} are not all eligible; using {knobs}", flush=True)
    for x, y in knobs:
        i = (y * W + x) * 4
        rgba[i:i + 4] = b"\x00\x00\x00\x01"
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = out or os.path.join(HERE, "found", name + "_namemc")
    os.makedirs(outdir, exist_ok=True)
    rgbap = os.path.join(outdir, "base.rgba")
    with open(rgbap, "wb") as f:
        f.write(rgba)
    kopt = ",".join(f"{x},{y}" for x, y in knobs)
    print(f"original NameMC id {namemc_id(orig)}; knob pixels {knobs} (alpha 1); "
          f"{len(elig)} eligible", flush=True)
    cpu, gpuexe = exe("namemcfinder"), exe("namemcgpu")
    if not os.path.exists(cpu):
        sys.exit("namemcfinder not found (build it with build.ps1)")
    if engine == "cpu":
        use_gpu, why = False, "--engine cpu"
    elif engine == "gpu":
        use_gpu, why = True, "--engine gpu"
    elif os.path.exists(gpuexe):
        use_gpu, why = True, "namemcgpu found"
    else:
        use_gpu, why = False, "namemcgpu not built"
    print(f"engine: {'gpu (namemcgpu)' if use_gpu else 'cpu (namemcfinder)'} - {why}", flush=True)

    def launch(fargs, on_hit=None):
        global GPU_FAILED
        if use_gpu:
            if not os.path.exists(gpuexe):
                print("namemcgpu not found (build it with build.ps1); falling back to CPU", flush=True)
                GPU_FAILED = True
            else:
                print("gpu mode: namemcgpu", flush=True)
                rc = run_finder([gpuexe, rgbap, outdir, "-k", kopt, "--render-exe", cpu, *fargs],
                                orig, on_hit, verify_namemc)
                if rc in (0, STOPPED_BY_US):
                    return rc
                print(f"namemcgpu failed (rc={rc}, 5 = correctness gate failed); falling back to CPU", flush=True)
                GPU_FAILED = True
        print("cpu mode: namemcfinder", flush=True)
        return run_finder([cpu, rgbap, outdir, "-k", kopt, *fargs], orig, on_hit, verify_namemc)

    extra = finder_extra(threads, seed, bench)
    if bench or not every:
        hits = []

        def collect(why_, p):
            hits.append(p)
            return False
        launch(search_args(targets, count, every, extra, bench), collect)
        return hits
    return list(run_targets(launch, targets, extra, outdir, NM_URL).values())


# ---- NameMC collisions -------------------------------------------------------------------------
# Mode A: a skin whose NameMC id equals one already in a list of known ids. Knobs: the last 2 pixels (62-63, 63),
# alpha 1 + 48-bit counter as RGB, so one try is a single SHA-256 block. Mode B (--self): two skins of our own
# with the same id; Pollard rho over 64 bits spread on the last 3 pixels (61-63, 63).
COLLIDE_RE = re.compile(r"^\[\s*(\d+)s\] tried=(\d+)\s+rate=(\d+)/s \(avg (\d+)/s\)\s+(.*)$")
DEFAULT_IDS = os.path.join(os.path.expanduser("~"), "Downloads", "namemc_skin_ids.u64le.bin")


def collide_pixels(mode, v):
    """(x, y, rgb24) of the knob pixels for counter v; must match namemccollide.cu."""
    if mode == "A":
        return [(62, 63, v & 0xFFFFFF), (63, 63, (v >> 24) & 0xFFFFFF)]
    return [(61, 63, v & 0xFFFFFF), (62, 63, (v >> 24) & 0xFFFFFF), (63, 63, ((v >> 48) & 0xFFFF) << 8)]


def collide_skin(base, mode, v):
    rgba = bytearray(base)
    for x, y, rgb in collide_pixels(mode, v):
        i = (y * W + x) * 4
        rgba[i:i + 4] = bytes((rgb >> 16 & 255, rgb >> 8 & 255, rgb & 255, 1))
    return rgba


def pretty_collide(line):
    m = COLLIDE_RE.match(line)
    if not m:
        return pretty_progress(line)
    secs, tried, rate, avg = (int(g) for g in m.groups()[:4])
    h, rest = divmod(secs, 3600)
    el = f"{h}:{rest // 60:02d}:{rest % 60:02d}"
    return f"[{el:>9}] tried={tried:_}  rate={rate:_}/s (avg {avg:_}/s)  {m.group(5)}\n"


def collide_search(path, ids_path, self_mode=False, out=None, watermark=True, seed=None, bench=False, count=1):
    """Run namemccollide. Returns the verified result files (mode A: the skin; mode B: both skins)."""
    global LAST_RATE
    mode = "B" if self_mode else "A"
    nk = 3 if self_mode else 2
    gpuexe = exe("namemccollide")
    if not os.path.exists(gpuexe):
        sys.exit("namemccollide not found (build it with build.ps1; needs an NVIDIA GPU and nvcc)")
    if mode == "A" and not os.path.exists(ids_path):
        sys.exit(f"id list not found: {ids_path} (pass --ids; sorted uint64 little-endian)")
    orig = load_rgba(path)
    rgba = bytearray(orig)
    wm = apply_watermark(rgba, avoid=MOJANG_TAIL) if watermark else []
    base_mask, _ = masks()
    knobs = [(W - nk + j, H - 1) for j in range(nk)]
    for x, y in knobs:
        if rgba[(y * W + x) * 4 + 3] or (x, y) in base_mask or (x, y) in RESERVED or (x, y) in wm:
            sys.exit(f"knob pixel ({x},{y}) is not free in this skin (it must be transparent and invisible); "
                     "the collision search needs the last pixels of the file")
        i = (y * W + x) * 4
        rgba[i:i + 4] = b"\x00\x00\x00\x01"
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = out or os.path.join(HERE, "found", name + "_collide")
    os.makedirs(outdir, exist_ok=True)
    rgbap = os.path.join(outdir, "base.rgba")
    with open(rgbap, "wb") as f:
        f.write(rgba)
    what = "self-collision (two own skins)" if self_mode else "match an id of " + ids_path
    print(f"original NameMC id {namemc_id(orig)}; mode {mode}: {what}; knob pixels {knobs} (alpha 1)", flush=True)
    cmd = [gpuexe, rgbap, "-m", mode] + ([ids_path] if mode == "A" else [])
    cmd += ["-n", str(count)] + (["-S", str(seed)] if seed is not None else []) + (["-b"] if bench else [])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    results = []
    try:
        for line in proc.stdout:
            print(pretty_collide(line), end="", flush=True)
            if line.startswith("RATE "):
                LAST_RATE = float(line.split()[1])
            elif line.startswith("HIT A "):
                kv = dict(x.split("=", 1) for x in line.split()[2:])
                results += collide_report_a(orig, rgba, outdir, kv["id"], int(kv["counter"]), ids_path)
            elif line.startswith("COLLISION B "):
                kv = dict(x.split("=", 1) for x in line.split()[2:])
                results += collide_report_b(orig, rgba, outdir, kv["id"], int(kv["x"]), int(kv["y"]))
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        stopped()
    if rc in INTERRUPT_CODES or rc & 0xFFFFFFFF in INTERRUPT_CODES:
        stopped()
    if rc:
        sys.exit(f"namemccollide failed (rc={rc}; 5 = correctness gate failed, 6 = CUDA error)")
    return results


def collide_save(orig, base, outdir, mode, v, fname, want):
    rgba = collide_skin(base, mode, v)
    out = os.path.join(outdir, fname)
    with open(out, "wb") as f:
        f.write(mojang_png(rgba))
    got = namemc_id(load_rgba(out))   # decoded back from the written PNG, hashed with the Python implementation
    ok, msg = verify_namemc(orig, out, 2 if mode == "A" else 3, want)
    return out, got, ok and got == want, msg


def collide_report_a(orig, base, outdir, id16, counter, ids_path):
    out, got, ok, msg = collide_save(orig, base, outdir, "A", counter, id16 + ".png", id16)
    print("\n== result: NameMC id collision with an existing skin")
    print(f"file          {out}")
    print(f"new skin id   {got}  {NM_URL}{got}   (what the upload will have)")
    print(f"existing id   {id16}  {NM_URL}{id16}   <- already on NameMC")
    try:
        import numpy as np
        a = np.memmap(ids_path, dtype="<u8", mode="r")
        v = np.uint64(int(id16, 16))
        k = int(np.searchsorted(a, v))
        inlist = k < len(a) and a[k] == v
        print(f"list check    {'id is in the list at index ' + str(k) if inlist else 'NOT FOUND IN LIST'} (independent numpy lookup)")
        ok = ok and inlist
    except ImportError:
        print("list check    skipped (numpy not installed); the finder already re-checked on the CPU")
    print(f"verify        {'OK' if ok else 'FAIL'} {msg}", flush=True)
    return [out] if ok else []


def collide_report_b(orig, base, outdir, id16, x, y):
    ra = collide_save(orig, base, outdir, "B", x, f"{id16}_a.png", id16)
    rb = collide_save(orig, base, outdir, "B", y, f"{id16}_b.png", id16)
    print("\n== result: two skins with the same NameMC id")
    for tag, r in (("a", ra), ("b", rb)):
        print(f"file {tag}        {r[0]}")
        print(f"id {tag}          {r[1]}  {NM_URL}{r[1]}")
    same = x != y and open(ra[0], "rb").read() != open(rb[0], "rb").read()
    ok = ra[2] and rb[2] and ra[1] == rb[1] == id16 and same
    print(f"verify        {'OK' if ok else 'FAIL'} both files decode to NameMC id {id16}; files differ: {same}", flush=True)
    return [ra[0], rb[0]] if ok else []


# ---- other commands ----------------------------------------------------------------------------
NAME_URL2 = "https://api.minecraftservices.com/minecraft/profile/lookup/name/"
SESSION_URL = "https://sessionserver.mojang.com/session/minecraft/profile/"
NAME_URL = "https://api.mojang.com/users/profiles/minecraft/"
SKINS_DIR = os.path.join(HERE, "skins")
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}")
NAME_RE = re.compile(r"[A-Za-z0-9_]{3,16}")


def classify(spec):
    """(kind, value) of a skin argument; kind is file, texture, uuid or name. An existing file always
    wins. Then: explicit prefix, textures.minecraft.net URL, UUID (32 hex or dashed), texture id
    (33-64 hex), username (3-16 of A-Za-z0-9_). A short hex word such as 'cafe' is a username; write
    texture:<id> for a texture id shorter than 33 digits."""
    if os.path.exists(spec):
        return "file", spec
    for pre in ("name", "uuid", "texture"):
        if spec.lower().startswith(pre + ":"):
            return pre, spec[len(pre) + 1:].strip()
    if "/" in spec:
        return "texture", spec
    if UUID_RE.fullmatch(spec):
        return "uuid", spec
    if re.fullmatch(r"[0-9a-fA-F]{33,64}", spec):
        return "texture", spec
    if NAME_RE.fullmatch(spec):
        return "name", spec
    raise ValueError(f"{spec}: not a file, texture id, textures.minecraft.net URL, UUID or username")


def http_get(url, what):
    """Body of a GET, or None for 204/404. ValueError (clean message) for 429 and network errors."""
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "skinhash"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return None if r.status == 204 else r.read()
    except urllib.error.HTTPError as e:
        if e.code in (404, 204):
            return None
        if e.code == 429:
            raise ValueError(f"{what}: rate-limited, try again in a minute")
        raise ValueError(f"{what}: HTTP {e.code}")
    except (OSError, ValueError) as e:
        raise ValueError(f"{what}: network error ({getattr(e, 'reason', e)})")


def download_texture(tid):
    """The skin bytes of a texture id; the sha256 must equal the id."""
    tid = tid.lower()
    data = http_get(SKIN_URL + tid, "textures.minecraft.net")
    if data is None:
        raise ValueError(f"texture {tid}: not found")
    if texture_id(data) != tid:
        raise ValueError(f"texture {tid}: downloaded bytes do not hash to the texture id")
    return data


CACHE_FILE = os.path.join(SKINS_DIR, ".lookups.json")
CACHE_TTL = 600  # seconds; Mojang allows roughly one profile request per UUID per minute


def cache_load():
    import json
    try:
        with open(CACHE_FILE, encoding="utf8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def cache_store(cache):
    import json
    try:
        os.makedirs(SKINS_DIR, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf8") as f:
            json.dump(cache, f)
    except OSError:
        pass


def lookup_player(kind, value):
    """(name, uuid, texture id, slim) from the Mojang API. ValueError if unknown or no custom skin.
    Results are cached in skins/.lookups.json for 10 minutes (and reused when Mojang rate-limits)."""
    import base64
    import json
    import time
    cache = cache_load()
    key = (value.lower() if kind == "name" else value.replace("-", "").lower())
    hit = cache.get(("n:" if kind == "name" else "u:") + key)
    if hit and time.time() - hit["t"] < CACHE_TTL:
        return hit["name"], hit["uuid"], hit["tid"], hit["slim"]
    try:
        if kind == "name":
            try:
                body = http_get(NAME_URL + value, "api.mojang.com")
            except ValueError:  # 5xx or rate limit: the same lookup on the other official host
                body = http_get(NAME_URL2 + value, "api.minecraftservices.com")
            if body is None:
                raise ValueError(f"unknown Minecraft username '{value}'")
            uuid = json.loads(body)["id"]
        else:
            uuid = key
        body = http_get(SESSION_URL + uuid, "sessionserver.mojang.com")
    except ValueError as e:
        if hit and not str(e).startswith("unknown"):
            return hit["name"], hit["uuid"], hit["tid"], hit["slim"]  # rate-limited or offline: stale answer
        raise
    if body is None:
        raise ValueError(f"unknown UUID {value}")
    prof = json.loads(body)
    name = prof.get("name", value)
    tex = next((p for p in prof.get("properties", []) if p.get("name") == "textures"), None)
    skin = None
    if tex:
        skin = json.loads(base64.b64decode(tex["value"])).get("textures", {}).get("SKIN")
    if not skin:
        raise ValueError(f"{name} ({uuid}) has no custom skin (default Steve/Alex)")
    slim = (skin.get("metadata") or {}).get("model") == "slim"
    tid = skin["url"].rstrip("/").rsplit("/", 1)[-1].lower()
    rec = {"t": time.time(), "name": name, "uuid": uuid, "tid": tid, "slim": slim}
    cache["u:" + uuid] = rec
    cache["n:" + name.lower()] = rec
    cache_store(cache)
    return name, uuid, tid, slim


def shown(path):
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def save_skin(base, data):
    os.makedirs(SKINS_DIR, exist_ok=True)
    path = os.path.join(SKINS_DIR, re.sub(r"[^A-Za-z0-9_.-]", "_", base) + ".png")
    if os.path.basename(path).lower() == "steve.png":
        path = os.path.join(SKINS_DIR, "steve_player.png")  # never overwrite the example skin
    with open(path, "wb") as f:
        f.write(data)
    return path


def check_size(data, label):
    import io
    try:
        size = Image.open(io.BytesIO(data)).size
    except Exception as e:
        raise ValueError(f"{label}: not an image ({e})")
    if size == (64, 32):
        raise ValueError(f"{label}: legacy 64x32 skin; this tool needs a 64x64 skin")
    if size != (W, H):
        raise ValueError(f"{label}: not a 64x64 skin ({size[0]}x{size[1]})")


def texture_arg_id(spec, value):
    tid = value.rstrip("/").rsplit("/", 1)[-1].lower()
    if not re.fullmatch(r"[0-9a-f]{1,64}", tid):
        raise ValueError(f"{spec}: not a texture id or a textures.minecraft.net URL")
    return tid


def import_skin(spec, quiet=False):
    """Resolve a file, name, UUID or texture id/URL to a saved PNG. Returns (path, bytes).
    Prints one line: name, UUID, classic/slim, texture id, saved path."""
    kind, value = classify(spec)
    if kind == "file":
        return value, open(value, "rb").read()
    if kind == "texture":
        tid = texture_arg_id(spec, value)
        data = download_texture(tid)
        check_size(data, tid)
        path = save_skin(tid[:16], data)
        if not quiet:
            print(f"texture {tid}  saved {shown(path)}")
        return path, data
    name, uuid, tid, slim = lookup_player(kind, value)
    data = download_texture(tid)
    check_size(data, f"{name}'s skin")
    path = save_skin(name, data)
    if not quiet:
        print(f"{name}  {uuid}  {'slim' if slim else 'classic'}  {tid}  saved {shown(path)}")
    return path, data


def skin_path(spec):
    """A path to a skin PNG for a skin argument (see classify). Exits cleanly on errors."""
    try:
        return import_skin(spec)[0]
    except ValueError as e:
        sys.exit(f"error: {e}")


def fetch_skin(t, quiet=False):
    """(bytes, label) of a file, texture id/URL, UUID or username. ValueError on failure.
    Players are saved to skins/<name>.png; plain texture ids are only downloaded."""
    kind, value = classify(t)
    if kind == "texture":
        tid = texture_arg_id(t, value)
        return download_texture(tid), SKIN_URL + tid
    path, data = import_skin(t, quiet)
    return data, path


def decode_skin(data, label):
    import io
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ValueError(f"{label}: not an image ({e})")
    if im.size == (64, 32):
        raise ValueError(f"{label}: legacy 64x32 skin; this tool needs a 64x64 skin")
    if im.size != (W, H):
        raise ValueError(f"{label}: not a 64x64 skin ({im.size[0]}x{im.size[1]})")
    return bytearray(im.convert("RGBA").tobytes())


def cmd_detect(args):
    bad = 0
    for t in args.inputs:
        try:
            data, label = fetch_skin(t)
            rgba = decode_skin(data, label)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            bad = 1
            continue
        f = find_watermark(rgba)
        print(f"{'YES' if f else 'no '}  {label}" + (f" (watermark v{f[0]} at {f[1]})" if f else ""))
    return bad


def cmd_ids(args):
    bad = 0
    for t in args.inputs:
        try:
            data, label = fetch_skin(t, quiet=args.legacy == "predict")
            rgba = decode_skin(data, label)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            bad = 1
            continue
        moj, nm = texture_id(mojang_png(rgba)), namemc_id(rgba)
        if args.legacy == "predict":
            print(moj)
        elif args.legacy == "namemc-id":
            print(f"{nm}  {label}  (mojang {moj})")
        else:
            print(label)
            print(f"  Mojang  {moj}  {SKIN_URL}{moj}")
            print(f"  NameMC  {nm}  {NM_URL}{nm}")
    return bad


def cmd_verify(args):
    ok, msg = verify(load_rgba(skin_path(args.original)), args.result)
    print("OK" if ok else "FAIL", msg)
    return 0 if ok else 1


def cmd_pixels(args):
    path, k = skin_path(args.skin), args.tail_pixels
    orig = load_rgba(path)
    print(f"{path}: Mojang id {texture_id(mojang_png(orig))}, NameMC id {namemc_id(orig)}")
    rgba = bytearray(orig)
    tail = [(W - k + i, H - 1) for i in range(k)]
    wm = apply_watermark(rgba, avoid=set(tail))
    kset = {p for p in knob_pixels(rgba, False) if p not in wm}
    bad = [p for p in tail if p not in kset]

    def generic(unused_only):
        kp = [p for p in knob_pixels(rgba, unused_only) if p not in wm]
        try:
            return fmt_px(knob_offsets(kp)[1]) + "  (3 pixels, 64-bit counter in RGB)"
        except SystemExit as e:
            return f"not possible: {e}"
    print("\nwatermark (every mode; RGB of 6 transparent unused pixels)")
    found = find_watermark(orig)
    print("  " + (fmt_px(wm) if wm else "no free unused pixels")
          + (f"  (the skin already carries watermark v{found[0]})" if found else ""))
    print("\nmojang (RGB under alpha 0 changes, alpha stays)")
    print(f"  fast, GPU       last K <= {k} pixels, usually {k - 1}: {fmt_px(tail[1:])}")
    print(f"  fast, CPU       last {k} pixels: {fmt_px(tail)}")
    print(f"  generic         {generic(False)}")
    print(f"  --unused-only   {generic(True)}   (never drawn)")
    if bad:
        print("  fast mode       not eligible: "
              + "; ".join(f"({x},{y}) {px_state(rgba, (x, y))}" for x, y in bad))
    else:
        print(f"  fast mode       eligible: all {k} tail pixels are invisible")
    print("\nnamemc (3 pixels get alpha 1 and a counter as RGB)")
    for label, uo in (("default", False), ("--unused-only", True)):
        try:
            nk, elig = namemc_knob_pixels(rgba, uo, avoid=set(wm))
            print(f"  {label:<15} {fmt_px(nk)}   ({len(elig)} eligible)")
        except SystemExit as e:
            print(f"  {label:<15} not possible: {e}")
    return 0


def cmd_bench(args):
    global LAST_RATE, GPU_FAILED
    skin = skin_path(args.skin)
    gpu_here = os.path.exists(exe("gpufinder")) and os.path.exists(exe("namemcgpu"))
    engines = [args.engine] if args.engine else (["gpu", "cpu"] if gpu_here else ["cpu"])
    if "gpu" in engines and not gpu_here:
        sys.exit("--engine gpu: gpufinder.exe / namemcgpu.exe not built (build.ps1 needs CUDA)")
    if not args.engine and not gpu_here:
        print("note: no CUDA build found, benchmarking the CPU only", flush=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="skinhash_bench_") as tmp:
        for eng in engines:
            for mode in ("mojang", "namemc"):
                print(f"\n== {mode} / {eng} (10 s, nothing is saved)", flush=True)
                LAST_RATE, GPU_FAILED = None, False
                d = os.path.join(tmp, f"{mode}_{eng}")
                if mode == "mojang":
                    mojang_search(skin, [], engine=eng, no_generic=True, out=d, threads=args.threads,
                                  seed=args.seed, bench=True)
                else:
                    namemc_search(skin, [], engine=eng, out=d, threads=args.threads, seed=args.seed, bench=True)
                results.append((mode, eng, LAST_RATE, GPU_FAILED))
    print(f"\nbenchmark on {skin}")
    for mode, eng, rate, failed in results:
        r = f"{rate:.3e} candidates/s" if rate else "no rate reported"
        print(f"  {mode:<7} {eng:<4} {r}" + ("  (GPU failed, this is the CPU)" if failed else ""))
    return 0


def targets_from(prefix, suffix, zeros):
    return ([f"prefix:{p}" for p in prefix or []] + [f"suffix:{s}" for s in suffix or []]
            + [f"zeros:{z}" for z in zeros or []])


def check_search_options(parser, args, has_target,
                         hint="use -p/--prefix, -s/--suffix or -z/--zeros"):
    if not has_target:
        parser.error("no target given: " + hint)
    if getattr(args, "no_generic", False) and args.engine == "generic":
        parser.error("--no-generic conflicts with --engine generic")
    if args.unused_only and (args.engine in ("gpu", "cpu") or getattr(args, "no_generic", False)):
        parser.error("--unused-only can't be combined with the fast modes (--engine gpu/cpu, --no-generic): "
                     "they use pixels that --unused-only excludes")


def default_out(args, suffix=""):
    if args.out:
        return os.path.abspath(args.out)
    return os.path.join(HERE, "found", os.path.splitext(os.path.basename(args.skin))[0] + suffix)


def cmd_mojang(args):
    targets = targets_from(args.prefix, args.suffix, args.zeros)
    check_search_options(args.parser, args, bool(targets))
    args.skin = skin_path(args.skin)
    mojang_search(args.skin, targets, args.count, args.every_target, args.engine, args.no_generic,
                  args.unused_only, args.tail_pixels, default_out(args), not args.no_watermark,
                  args.threads, args.seed)
    return 0


def cmd_namemc(args):
    targets = targets_from(args.prefix, args.suffix, args.zeros)
    check_search_options(args.parser, args, bool(targets))
    args.skin = skin_path(args.skin)
    namemc_search(args.skin, targets, args.count, args.every_target, args.engine, args.unused_only,
                  default_out(args, "_namemc"), not args.no_watermark, args.threads, args.seed)
    return 0


def cmd_collide(args):
    args.skin = skin_path(args.skin)
    res = collide_search(args.skin, args.ids, args.self_collision, args.out, not args.no_watermark, args.seed,
                         args.bench, args.count)
    return 0 if args.bench or res else 1


def cmd_both(args):
    nm_t = targets_from(args.namemc_prefix, args.namemc_suffix, args.namemc_zeros)
    mj_t = targets_from(args.mojang_prefix, args.mojang_suffix, args.mojang_zeros)
    check_search_options(args.parser, args, bool(nm_t or mj_t),
                         "use --namemc-prefix/-suffix/-zeros and/or --mojang-prefix/-suffix/-zeros")
    args.skin = skin_path(args.skin)
    outdir = default_out(args, "_both")
    orig = load_rgba(args.skin)
    wm = not args.no_watermark
    total = bool(nm_t) + bool(mj_t)
    step = 0
    nm_file = None
    if nm_t:
        step += 1
        print(f"== phase {step}/{total}: NameMC {', '.join(nm_t)}", flush=True)
        # generic only exists for Mojang; the NameMC phase then uses the CPU finder
        hits = namemc_search(args.skin, nm_t, 1, args.every_target,
                             "cpu" if args.engine == "generic" else args.engine,
                             args.unused_only, outdir, wm, args.threads, args.seed)
        if not hits:
            sys.exit("phase 1 (NameMC) ended without a verified hit")
        nm_file = hits[0]
        print(f"== phase {step}/{total} done: {NM_URL}{os.path.splitext(os.path.basename(nm_file))[0]}"
              + (f" (first of {len(hits)} hits, used as the base)" if len(hits) > 1 else ""), flush=True)
    finals = [nm_file]
    if mj_t:
        step += 1
        print(f"== phase {step}/{total}: Mojang {', '.join(mj_t)}" + (" on that hit" if nm_file else ""), flush=True)
        finals = mojang_search(nm_file or args.skin, mj_t, args.count, args.every_target, args.engine,
                               args.no_generic, args.unused_only, args.tail_pixels, outdir, wm,
                               args.threads, args.seed)
        if not finals:
            sys.exit(f"phase {step} (Mojang) ended without a verified hit")
        print(f"== phase {step}/{total} done", flush=True)

    print("\n== result", flush=True)
    bad = 0
    for f in finals:
        data = open(f, "rb").read()
        nm, full = namemc_id(load_rgba(f)), hashlib.sha256(data).hexdigest()
        moj = full.lstrip("0") or "0"
        ok, msg = (verify if mj_t else verify_namemc)(orig, f)
        print(f"file     {f}")
        print(f"NameMC   {NM_URL}{nm}")
        print(f"Mojang   {SKIN_URL}{moj}")
        checks = []
        if nm_t:
            m = first_match(nm_t, nm, nm)
            checks.append(f"NameMC id matches {m}" if m else "NameMC id matches NO target")
            bad += not m
        if mj_t:
            m = first_match(mj_t, moj, full)
            checks.append(f"Mojang id matches {m}" if m else "Mojang id matches NO target")
            bad += not m
        print(f"verify   {'OK' if ok else 'FAIL'} against the original skin ({args.skin}); "
              + "; ".join(checks) + ("" if ok else f" [{msg}]"), flush=True)
        bad += not ok
    return 1 if bad else 0


# ---- command line ------------------------------------------------------------------------------
def hex_arg(s):
    if not re.fullmatch(r"[0-9a-fA-F]+", s):
        raise argparse.ArgumentTypeError(f"'{s}' is not hex (use the digits 0-9 and a-f)")
    if len(s) > 16:
        raise argparse.ArgumentTypeError(f"'{s}' is too long (at most 16 hex digits)")
    return s.lower()


def mojang_hex(s):
    s = hex_arg(s)
    if s[0] == "0":
        rest = s.lstrip("0")
        hint = f"-z {len(s) - len(rest)} -p {rest}" if rest else f"-z {len(s)}"
        raise argparse.ArgumentTypeError(
            f"Mojang prefix '{s}' starts with 0, but the link drops leading zeros; use --zeros N ({hint})")
    return s


def int_range(lo, hi=None):
    def conv(s):
        try:
            n = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{s}' is not a number")
        if n < lo or (hi is not None and n > hi):
            raise argparse.ArgumentTypeError(f"{n} is out of range ({lo}-{hi if hi is not None else 'any'})")
        return n
    return conv


class Deprecated(argparse.Action):
    """A hidden old flag: prints a note on stderr and sets the new option."""
    def __init__(self, option_strings, dest, new=None, **kw):
        self.new = new
        super().__init__(option_strings, dest, default=argparse.SUPPRESS, help=argparse.SUPPRESS, **kw)

    def __call__(self, parser, ns, values, option_string=None):
        print(f"note: {option_string} is deprecated; use {self.new}", file=sys.stderr)
        setattr(ns, self.dest, self.const if self.nargs == 0 else values)


DEPRECATED_COMMANDS = {"search": "mojang", "predict": "ids", "namemc-id": "ids", "knobs": "pixels"}


def add_deprecated(g, kind):
    """The old flags that still work for a search command."""
    g.add_argument("--gpu", action=Deprecated, dest="engine", nargs=0, const="gpu", new="--engine gpu")
    g.add_argument("--all", action=Deprecated, dest="every_target", nargs=0, const=True, new="--every-target")
    if kind == "mojang":
        g.add_argument("--generic", action=Deprecated, dest="engine", nargs=0, const="generic", new="--engine generic")
        g.add_argument("--fast", action=Deprecated, dest="no_generic", nargs=0, const=True, new="--no-generic")
        g.add_argument("--k", action=Deprecated, dest="tail_pixels", type=int_range(1, 20), new="--tail-pixels")


SKIN_HELP = ("a 64x64 skin PNG (e.g. skins/steve.png), a texture id or URL, a UUID or a Minecraft "
             "username (saved as skins/<name>.png). Prefixes name:, uuid:, texture: force the meaning.")


def add_skin(p):
    p.add_argument("skin", metavar="SKIN", help=SKIN_HELP)


def add_targets(p, kind):
    g = p.add_argument_group("Targets (at least one; a hit matches any of them)")
    if kind == "mojang":
        g.add_argument("-p", "--prefix", metavar="HEX", type=mojang_hex, action="append",
                       help="link id starts with HEX (up to 16 digits, may not start with 0: use --zeros). Repeatable.")
        g.add_argument("-z", "--zeros", metavar="N", type=int_range(1, 64), action="append",
                       help="at least N leading zero digits (the shortest links). Repeatable.")
    else:
        g.add_argument("-p", "--prefix", metavar="HEX", type=hex_arg, action="append",
                       help="NameMC id starts with HEX (up to 16 digits; may start with 0, the id is 16 digits wide). Repeatable.")
        g.add_argument("-z", "--zeros", metavar="N", type=int_range(1, 16), action="append",
                       help="at least N leading zero digits of the 16-digit id. Repeatable.")
    g.add_argument("-s", "--suffix", metavar="HEX", type=hex_arg, action="append",
                   help="id ends with HEX (up to 16 digits). Repeatable.")


def add_stopping(p, both=False):
    g = p.add_argument_group("Stopping")
    g.add_argument("-n", "--count", metavar="N", type=int_range(1), default=1,
                   help="stop after N verified hits (default 1)"
                   + (" in the Mojang phase; the NameMC phase stops at its first hit" if both else ""))
    g.add_argument("--every-target", action="store_true",
                   help="keep searching until every target has a hit, keeping the first hit per target (-n is ignored)"
                   + ("; applies to both phases, the first NameMC hit is the base" if both else ""))


def add_engine(p, kind):
    g = p.add_argument_group("Engine")
    if kind == "namemc":
        g.add_argument("--engine", choices=["auto", "gpu", "cpu"], default="auto",
                       help="auto (default): namemcgpu if built and it passes its self-check, else namemcfinder on the CPU. "
                       "gpu, cpu: prefer that engine.")
        return
    if kind == "mojang":
        g.add_argument("--engine", choices=["auto", "gpu", "cpu", "generic"], default="auto",
                       help="auto (default): GPU if gpufinder is built and passes its self-check, else CPU fast mode, "
                       "else the ~300x slower generic search. gpu, cpu: prefer that fast engine. "
                       "generic: hashfinder only.")
    else:
        g.add_argument("--engine", choices=["auto", "gpu", "cpu", "generic"], default="auto",
                       help="as for the mojang and namemc commands. generic only affects the Mojang phase "
                       "(the NameMC phase then uses the CPU).")
    g.add_argument("--no-generic", action="store_true",
                   help="never fall back to the slow generic search; stop with the reason instead")


def add_pixels(p, kind):
    g = p.add_argument_group("Pixels")
    g.add_argument("--unused-only", action="store_true",
                   help="only use pixels no model face ever samples (never drawn, so certainly invisible). "
                   "Much slower; incompatible with the Mojang fast modes.")
    if kind != "namemc":
        g.add_argument("--tail-pixels", metavar="N", type=int_range(1, 20), default=6,
                       help="pixels at the end of the file that Mojang fast mode permutes (default 6). With the GPU "
                       "it is an upper bound: the largest K <= N that gives a one-block SHA is used.")
    g.add_argument("--no-watermark", action="store_true", help="do not add the invisible skinhash watermark")


def add_output(p, default):
    g = p.add_argument_group("Output")
    g.add_argument("-o", "--out", metavar="DIR", help=f"output directory (default {default})")


def add_advanced(p):
    g = p.add_argument_group("Advanced")
    g.add_argument("--threads", metavar="N", type=int_range(1), help="CPU threads (default: all; the GPU ignores it)")
    g.add_argument("--seed", metavar="N", type=int_range(0), help="start seed of the search (default: random)")


def build_parser():
    top = argparse.ArgumentParser(
        prog="py skinhash.py", allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Find invisible variations of a Minecraft skin whose Mojang texture id and/or NameMC id looks cool.",
        epilog="examples:\n"
               "  py skinhash.py mojang skins/steve.png -p c0ffee\n"
               "  py skinhash.py mojang jeb_ -p c0ffee            (a username or UUID works as the skin)\n"
               "  py skinhash.py both skins/steve.png --namemc-prefix beef --mojang-prefix cafe\n"
               "  py skinhash.py verify skins/steve.png found/steve/<id>.png\n\n"
               "Run 'py skinhash.py COMMAND --help' for the options of one command.\n"
               "Old names (search, predict, namemc-id, knobs, --gpu, --generic, --fast, --all, --k) still work\n"
               "and print a note.")
    sub = top.add_subparsers(dest="command", metavar="COMMAND", title="commands", required=True)

    def command(name, summary, epilog):
        p = sub.add_parser(name, help=summary, description=summary, allow_abbrev=False,
                           formatter_class=argparse.RawDescriptionHelpFormatter, epilog="examples:\n" + epilog)
        p.set_defaults(parser=p)
        return p

    p = command("mojang", "Search for a Mojang texture link (textures.minecraft.net/texture/...) that looks cool.",
                "  py skinhash.py mojang skins/steve.png -p c0ffee\n"
                "  py skinhash.py mojang skins/steve.png -z 9 -n 3 -o out/steve\n"
                "  py skinhash.py mojang skins/steve.png -s dead --engine cpu --no-generic\n"
                "  py skinhash.py mojang jeb_ -p c0ffee            (username; also a UUID or texture id)")
    add_skin(p)
    add_targets(p, "mojang")
    add_stopping(p)
    add_engine(p, "mojang")
    add_pixels(p, "mojang")
    add_output(p, "found/<skin>/")
    add_advanced(p)
    add_deprecated(p, "mojang")
    p.set_defaults(func=cmd_mojang)

    p = command("namemc", "Search for a NameMC skin page (namemc.com/skin/...) whose id looks cool.",
                "  py skinhash.py namemc skins/steve.png -p beef\n"
                "  py skinhash.py namemc skins/steve.png -z 6 -n 3\n"
                "  py skinhash.py namemc skins/steve.png -p dead -s cafe --every-target --engine cpu\n"
                "  py skinhash.py namemc jeb_ -p beef")
    add_skin(p)
    add_targets(p, "namemc")
    add_stopping(p)
    add_engine(p, "namemc")
    add_pixels(p, "namemc")
    add_output(p, "found/<skin>_namemc/")
    add_advanced(p)
    add_deprecated(p, "namemc")
    p.set_defaults(func=cmd_namemc)

    p = command("both", "Search for a NameMC id and a Mojang texture id on one skin: NameMC first, then Mojang on that hit.",
                "  py skinhash.py both skins/steve.png --namemc-prefix beef --mojang-prefix cafe\n"
                "  py skinhash.py both skins/steve.png --namemc-zeros 6 --mojang-zeros 9\n"
                "  py skinhash.py both skins/steve.png --mojang-prefix c0ffee     (only the Mojang phase)\n"
                "  py skinhash.py both jeb_ --namemc-prefix beef --mojang-prefix cafe")
    add_skin(p)
    g = p.add_argument_group("NameMC targets (phase 1; fixed 16 digits, prefixes may start with 0)")
    g.add_argument("--namemc-prefix", metavar="HEX", type=hex_arg, action="append", help="NameMC id starts with HEX. Repeatable.")
    g.add_argument("--namemc-suffix", metavar="HEX", type=hex_arg, action="append", help="NameMC id ends with HEX. Repeatable.")
    g.add_argument("--namemc-zeros", metavar="N", type=int_range(1, 16), action="append", help="at least N leading zero digits. Repeatable.")
    g = p.add_argument_group("Mojang targets (phase 2; run on the phase 1 hit)")
    g.add_argument("--mojang-prefix", metavar="HEX", type=mojang_hex, action="append", help="link id starts with HEX (may not start with 0: use --mojang-zeros). Repeatable.")
    g.add_argument("--mojang-suffix", metavar="HEX", type=hex_arg, action="append", help="link id ends with HEX. Repeatable.")
    g.add_argument("--mojang-zeros", metavar="N", type=int_range(1, 64), action="append", help="at least N leading zero digits. Repeatable.")
    add_stopping(p, both=True)
    add_engine(p, "both")
    add_pixels(p, "both")
    add_output(p, "found/<skin>_both/")
    add_advanced(p)
    p.set_defaults(func=cmd_both)

    p = command("ids", "Print the Mojang texture id and the NameMC id of skins, with both links.",
                "  py skinhash.py ids skins/steve.png\n"
                "  py skinhash.py ids found/steve/<id>.png\n"
                "  py skinhash.py ids https://textures.minecraft.net/texture/<id>\n"
                "  py skinhash.py ids jeb_ 853c80ef-3c37-49fd-aa49-938b674adae6")
    p.add_argument("inputs", metavar="SKIN", nargs="+", help=SKIN_HELP)
    p.set_defaults(func=cmd_ids)

    p = command("verify", "Check that RESULT looks identical to ORIGINAL in game and hashes to its file name.",
                "  py skinhash.py verify skins/steve.png found/steve/<id>.png\n"
                "  py skinhash.py verify skins/steve.png found/steve_both/<id>.png     (also NameMC and both results)")
    p.add_argument("original", metavar="ORIGINAL", help="the skin you started from (file, texture id/URL, UUID or username)")
    p.add_argument("result", metavar="RESULT", help="a hit file, named <id>.png")
    p.set_defaults(func=cmd_verify)

    p = command("detect", "Check whether skins were made by skinhash (the invisible watermark).",
                "  py skinhash.py detect found/steve/<id>.png\n"
                "  py skinhash.py detect https://textures.minecraft.net/texture/<id>\n"
                "  py skinhash.py detect jeb_")
    p.add_argument("inputs", metavar="SKIN", nargs="+", help=SKIN_HELP)
    p.set_defaults(func=cmd_detect)

    p = command("pixels", "Show which pixels each search mode would change on a skin, and whether fast mode is possible.",
                "  py skinhash.py pixels skins/steve.png\n"
                "  py skinhash.py pixels skins/me.png --tail-pixels 5\n"
                "  py skinhash.py pixels jeb_")
    add_skin(p)
    p.add_argument("--tail-pixels", metavar="N", type=int_range(1, 20), default=6,
                   help="size of the fast-mode tail to check (default 6)")
    p.add_argument("--unused-only", action="store_true", help=argparse.SUPPRESS)  # old 'knobs' flag; both variants are shown
    p.set_defaults(func=cmd_pixels)

    p = command("collide", "Find a skin whose NameMC id equals an id NameMC already has (or two own skins with one id).",
                "  py skinhash.py collide skins/me.png --ids namemc_skin_ids.u64le.bin\n"
                "  py skinhash.py collide skins/me.png --self")
    add_skin(p)
    g = p.add_argument_group("Mode")
    g.add_argument("--ids", metavar="FILE", default=DEFAULT_IDS,
                   help="sorted unique uint64 little-endian list of known NameMC ids (default ~/Downloads/namemc_skin_ids.u64le.bin). "
                   "Every candidate is checked against it on the GPU.")
    g.add_argument("--self", dest="self_collision", action="store_true",
                   help="instead find two different skins of your own with the same id (~2^32 steps, Pollard rho)")
    g.add_argument("-n", "--count", metavar="N", type=int_range(1), default=1, help="stop after N hits (list mode; default 1)")
    g.add_argument("--bench", action="store_true", help="10 s benchmark of the list mode, saves nothing")
    g.add_argument("--no-watermark", action="store_true", help="do not add the invisible skinhash watermark")
    add_output(p, "found/<skin>_collide/")
    g = p.add_argument_group("Advanced")
    g.add_argument("--seed", metavar="N", type=int_range(0), help="start seed (default: random)")
    p.set_defaults(func=cmd_collide)

    p = command("bench", "Run the 10 s benchmark of the search engines and print candidates per second. Saves nothing.",
                "  py skinhash.py bench\n"
                "  py skinhash.py bench --engine gpu\n"
                "  py skinhash.py bench skins/me.png --engine cpu --threads 8")
    p.add_argument("skin", metavar="SKIN", nargs="?", default=os.path.join(HERE, "skins", "steve.png"),
                   help="skin to benchmark on (default skins/steve.png); also a username, UUID or texture id")
    p.add_argument("--engine", choices=["gpu", "cpu"],
                   help="benchmark only this engine (default: GPU if built, then CPU). Runs the Mojang and the NameMC search.")
    add_advanced(p)
    p.set_defaults(func=cmd_bench)
    return top


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = None
    if argv and argv[0] in DEPRECATED_COMMANDS:
        legacy = argv[0]
        argv[0] = DEPRECATED_COMMANDS[legacy]
        print(f"note: '{legacy}' is deprecated; use '{argv[0]}'", file=sys.stderr)
    args = build_parser().parse_args(argv)
    args.legacy = legacy
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        stopped()
