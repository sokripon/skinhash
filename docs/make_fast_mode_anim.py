"""Generate docs/fast-mode-animated.svg (SMIL animation, renders on GitHub).

    py docs/make_fast_mode_anim.py

Values are illustrative: the shuffle, tail bytes and hashes are made up, but the
structure (what stays fixed, what changes) matches the real fast mode.
"""
import base64
import os
import random

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 940, 605
TOP = 185  # height of the skin section; everything else is shifted down by this
FRAMES = 4
DUR = 8.0  # seconds for the whole loop
MONO = "Consolas, Menlo, monospace"
RED, RED_BG, BLUE, BLUE_BG = "#e5484d", "#fdecec", "#8aa2c8", "#eef3fb"

rng = random.Random(3013)
VALUES = ["a3", "17", "c9", "5e", "f1", "2b", "d4", "08", "9a", "6c", "e2", "31", "b7", "4f", "85"]


def hexs(n):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


frames = []
for i in range(FRAMES):
    perm = VALUES[:] if i == 0 else rng.sample(VALUES, len(VALUES))
    sha = hexs(64)
    if i == FRAMES - 1:
        sha = "0000000" + "a6a0969f5b977a2f644c13bdac7cd83d2b5cd1d768a9b6b66602e45ff"
    frames.append({
        "perm": perm,
        "tail": [hexs(2) for _ in range(30)],
        "adler": hexs(8), "crc": hexs(8),
        "sha": sha,
    })


def visible(i, extra=""):
    """SMIL: show group i only during its slice of the loop."""
    a, b = i / FRAMES, (i + 1) / FRAMES
    if i == 0:
        kt, vals = f"0;{b:.4f};1", "1;0;0"
    elif i == FRAMES - 1:
        kt, vals = f"0;{a:.4f};1", "0;1;1"
    else:
        kt, vals = f"0;{a:.4f};{b:.4f};1", "0;1;0;0"
    return (f'<animate attributeName="opacity" calcMode="discrete" dur="{DUR}s" '
            f'repeatCount="indefinite" keyTimes="{kt}" values="{vals}"{extra}/>')


def pulse():
    """Flash once at the start of every frame."""
    kts, vals = [], []
    for i in range(FRAMES):
        t = i / FRAMES
        kts += [t, t + 0.02, t + 0.15]
        vals += ["0", "0.45", "0"]
    kts.append(1)
    vals.append("0")
    return (f'<animate attributeName="opacity" dur="{DUR}s" repeatCount="indefinite" '
            f'keyTimes="{";".join(f"{k:.4f}" for k in kts)}" values="{";".join(vals)}"/>')


o = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff"/>',
     '<text x="24" y="34" font-size="18" font-weight="600" fill="#1d1d1f">Fast mode, one candidate at a time</text>',
     '<text x="24" y="56" font-size="12.5" fill="#555">Red parts change every candidate. Blue parts are computed once '
     'and reused for all of them.</text>']

# --- skin ---------------------------------------------------------------
SKIN = os.path.join(os.path.dirname(HERE), "skins", "steve.png")
skin = Image.open(SKIN).convert("RGBA")
b64 = base64.b64encode(open(SKIN, "rb").read()).decode()
ZX, ZY, ZW, ZH = 48, 60, 16, 4  # zoomed region: x 48-63, rows 60-63
KNOBS = [(59 + j, 63) for j in range(5)]


def checker(x, y, w, h, cell):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#e8e8e8"/>']
    for cy in range(0, h, cell):
        for cx in range(0, w, cell):
            if (cx // cell + cy // cell) % 2:
                out.append(f'<rect x="{x + cx}" y="{y + cy}" width="{min(cell, w - cx)}" '
                           f'height="{min(cell, h - cy)}" fill="#cfcfcf"/>')
    return out


def zoom(x0, y0, cell, title, reveal):
    out = [f'<text x="{x0}" y="{y0 - 10}" font-size="12.5" font-weight="600" fill="#1d1d1f">{title}</text>']
    out += checker(x0, y0, ZW * cell, ZH * cell, cell // 2)
    for ry in range(ZH):
        for rx in range(ZW):
            r, g, b, a = skin.getpixel((ZX + rx, ZY + ry))
            if a and (ZX + rx, ZY + ry) not in KNOBS:
                out.append(f'<rect x="{x0 + rx * cell}" y="{y0 + ry * cell}" width="{cell}" height="{cell}" '
                           f'fill="#{r:02x}{g:02x}{b:02x}"/>')
    out.append(f'<rect x="{x0}" y="{y0}" width="{ZW * cell}" height="{ZH * cell}" fill="none" stroke="#999"/>')
    kx = x0 + (KNOBS[0][0] - ZX) * cell
    ky = y0 + (KNOBS[0][1] - ZY) * cell
    if reveal:
        for i, f in enumerate(frames):
            out += [f'<g opacity="{1 if i == 0 else 0}">', visible(i)]
            for j in range(5):
                col = "".join(f["perm"][3 * j:3 * j + 3])
                out.append(f'<rect x="{kx + j * cell}" y="{ky}" width="{cell}" height="{cell}" fill="#{col}"/>')
            out.append("</g>")
    out.append(f'<rect x="{kx}" y="{ky}" width="{5 * cell}" height="{cell}" fill="none" stroke="{RED}" '
               f'stroke-width="2" stroke-dasharray="4 2"/>')
    return out


o.append('<text x="24" y="88" font-size="13.5" font-weight="600" fill="#1d1d1f">The skin</text>')
o += checker(24, 98, 128, 128, 8)
o.append(f'<image x="24" y="98" width="128" height="128" style="image-rendering:pixelated" '
         f'href="data:image/png;base64,{b64}" xlink:href="data:image/png;base64,{b64}"/>')
o.append(f'<rect x="{24 + ZX * 2}" y="{98 + ZY * 2}" width="{ZW * 2}" height="{ZH * 2}" fill="none" '
         f'stroke="{RED}" stroke-width="1.5"/>')
o.append(f'<path d="M{24 + 64 * 2 + 2} {98 + 62 * 2} L196 {130 + 2 * 22}" stroke="{RED}" stroke-width="1.2" '
         'stroke-dasharray="4 3" fill="none"/>')
o += zoom(200, 130, 22, "Stored in the file (hidden RGB drawn)", reveal=True)
o += zoom(586, 130, 20, "Seen in game (alpha 0 = transparent)", reveal=False)
o.append(f'<text x="200" y="240" font-size="11.5" fill="#555">Rows 60–63, x 48–63. The 5 dashed pixels get new '
         'hidden colours every candidate; in game they stay invisible.</text>')
o.append(f'<g transform="translate(0,{TOP})">')

# --- pixels -------------------------------------------------------------
o.append('<text x="24" y="88" font-size="13.5" font-weight="600" fill="#1d1d1f">Last 5 pixels: '
         'the same 15 byte values in a new order (alpha stays 00)</text>')
px_x = [24 + j * 178 for j in range(5)]
for j, x in enumerate(px_x):
    o.append(f'<rect x="{x}" y="100" width="166" height="36" rx="6" fill="{RED_BG}" stroke="{RED}"/>')
    o.append(f'<text x="{x + 150}" y="123" font-size="11" text-anchor="end" fill="#999" font-family="{MONO}">α 00</text>')
    o.append(f'<rect x="{x}" y="100" width="166" height="36" rx="6" fill="{RED}">{pulse()}</rect>')
for i, f in enumerate(frames):
    g = [f'<g opacity="{1 if i == 0 else 0}">', visible(i)]
    for j, x in enumerate(px_x):
        rgb = " ".join(f["perm"][3 * j:3 * j + 3])
        g.append(f'<text x="{x + 14}" y="124" font-size="15" font-family="{MONO}" fill="#1d1d1f">{rgb}</text>')
    g.append("</g>")
    o += g

# --- file ---------------------------------------------------------------
o.append('<text x="24" y="176" font-size="13.5" font-weight="600" fill="#1d1d1f">PNG file (~3.5 KB)</text>')
o.append(f'<rect x="24" y="188" width="636" height="48" rx="6" fill="{BLUE_BG}" stroke="{BLUE}"/>')
o.append('<text x="342" y="210" font-size="12.5" text-anchor="middle" fill="#1d1d1f">header, Huffman tables, '
         'compressed pixels, watermark</text>')
o.append('<text x="342" y="227" font-size="11.5" text-anchor="middle" fill="#555">bit-identical for every candidate</text>')
o.append(f'<rect x="660" y="188" width="256" height="48" rx="6" fill="{RED_BG}" stroke="{RED}"/>')
o.append(f'<rect x="660" y="188" width="256" height="48" rx="6" fill="{RED}">{pulse()}</rect>')
for i, f in enumerate(frames):
    t = f["tail"]
    o += [f'<g opacity="{1 if i == 0 else 0}" font-family="{MONO}" font-size="11" fill="#1d1d1f">', visible(i),
          f'<text x="670" y="207">{" ".join(t[:12])}</text>',
          f'<text x="670" y="226">{" ".join(t[12:18])} <tspan fill="#a33">{f["adler"]} {f["crc"]}</tspan></text>',
          "</g>"]
o.append('<text x="916" y="252" font-size="11" text-anchor="end" fill="#555">tail literals · '
         '<tspan fill="#a33">Adler-32 · CRC-32</tspan></text>')
# connector from pixels to tail
o.append(f'<path d="M842 136 C842 160 800 165 790 186" fill="none" stroke="{RED}" stroke-width="1.4" '
         'stroke-dasharray="4 3"/>')

# --- SHA ----------------------------------------------------------------
o.append('<text x="24" y="290" font-size="13.5" font-weight="600" fill="#1d1d1f">SHA-256</text>')
o.append(f'<rect x="24" y="302" width="636" height="36" rx="6" fill="{BLUE_BG}" stroke="{BLUE}"/>')
o.append('<text x="342" y="325" font-size="12.5" text-anchor="middle" fill="#1d1d1f">~55 blocks hashed once '
         '→ saved midstate</text>')
o.append(f'<rect x="660" y="302" width="256" height="36" rx="6" fill="{RED_BG}" stroke="{RED}"/>')
o.append(f'<rect x="660" y="302" width="256" height="36" rx="6" fill="{RED}">{pulse()}</rect>')
o.append('<text x="788" y="325" font-size="12.5" text-anchor="middle" fill="#1d1d1f">last block, per candidate</text>')

# --- result -------------------------------------------------------------
for i, f in enumerate(frames):
    hit = i == FRAMES - 1
    sha = f["sha"]
    shown = (f'<tspan fill="{RED}" font-weight="700">{sha[:7]}</tspan>{sha[7:]}' if hit else sha)
    o += [f'<g opacity="{1 if i == 0 else 0}">', visible(i),
          f'<text x="24" y="372" font-size="12.5" fill="#555">candidate {i + 1}</text>',
          f'<text x="120" y="372" font-size="13" font-family="{MONO}" fill="#1d1d1f">{shown}</text>',
          (f'<text x="916" y="372" font-size="12.5" text-anchor="end" font-weight="600" fill="{RED}">'
           'hit: 7 leading zeros</text>' if hit else
           '<text x="916" y="372" font-size="12.5" text-anchor="end" fill="#999">no match</text>'),
          "</g>"]

o.append('<text x="24" y="404" font-size="11.5" fill="#777">Illustrative values. The real search does this '
         'about 3.85 billion times per second on an RTX 5060 Ti.</text>')
o.append("</g>")
o.append("</svg>")

path = os.path.join(HERE, "fast-mode-animated.svg")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(o))
print("wrote", path)
