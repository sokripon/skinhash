"""Generate docs/pixel-map.svg from the same layout tables skinhash.py uses.

    py docs/make_pixel_map.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import skinhash  # noqa: E402

C = 9            # cell size in px
OX, OY = 24, 56  # grid origin
GRID = 64 * C

COL = {
    "base": "#8aa2c8",
    "overlay": "#c9b6e8",
    "unused": "#f4f4f2",
    "watermark": "#f0a030",
    "fast": "#e5484d",
}

PARTS = [
    ("head", 0, 0, 32, 16, "base"), ("hat", 32, 0, 32, 16, "overlay"),
    ("right leg", 0, 16, 16, 16, "base"), ("body", 16, 16, 24, 16, "base"),
    ("right arm", 40, 16, 16, 16, "base"),
    ("right pants", 0, 32, 16, 16, "overlay"), ("jacket", 16, 32, 24, 16, "overlay"),
    ("right sleeve", 40, 32, 16, 16, "overlay"),
    ("left pants", 0, 48, 16, 16, "overlay"), ("left leg", 16, 48, 16, 16, "base"),
    ("left arm", 32, 48, 16, 16, "base"), ("left sleeve", 48, 48, 16, 16, "overlay"),
]


def main():
    base, over = skinhash.masks()
    wm = {(x, 0) for x in range(len(skinhash.wm_pixels(skinhash.WATERMARKS[skinhash.WATERMARK_VERSION])))}
    fast = {(x, 63) for x in range(58, 64)}  # last 5 (GPU, 1 SHA block) or 6 (CPU)
    W, H = OX * 2 + GRID + 250, OY + GRID + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff"/>',
           f'<text x="{OX}" y="30" font-size="18" font-weight="600" fill="#1d1d1f">'
           'The 64×64 skin texture: what the tool may change</text>']
    for y in range(64):
        for x in range(64):
            p = (x, y)
            kind = ("fast" if p in fast else "watermark" if p in wm else
                    "base" if p in base else "overlay" if p in over else "unused")
            out.append(f'<rect x="{OX + x * C}" y="{OY + y * C}" width="{C}" height="{C}" fill="{COL[kind]}"/>')
    for i in range(0, 65, 8):
        out.append(f'<line x1="{OX + i * C}" y1="{OY}" x2="{OX + i * C}" y2="{OY + GRID}" stroke="#ffffff" stroke-width="1"/>')
        out.append(f'<line x1="{OX}" y1="{OY + i * C}" x2="{OX + GRID}" y2="{OY + i * C}" stroke="#ffffff" stroke-width="1"/>')
    out.append(f'<rect x="{OX}" y="{OY}" width="{GRID}" height="{GRID}" fill="none" stroke="#444" stroke-width="1.5"/>')
    for name, x, y, w, h, _ in PARTS:
        cx, cy = OX + (x + w / 2) * C, OY + (y + h / 2) * C + 4
        out.append(f'<text x="{cx}" y="{cy}" font-size="11" text-anchor="middle" fill="#1d1d1f" '
                   f'stroke="#ffffff" stroke-width="3" paint-order="stroke">{name}</text>')
    out.append(f'<path d="M{OX + 3 * C} {OY + C} L{OX + 3 * C} {OY - 8}" stroke="#b36b00" stroke-width="1.5"/>')
    out.append(f'<text x="{OX + 3 * C + 6}" y="{OY - 10}" font-size="11" fill="#b36b00">watermark (0,0)–(5,0)</text>')
    fx = OX + 61 * C
    out.append(f'<path d="M{fx} {OY + GRID} L{fx} {OY + GRID + 14}" stroke="#c0282d" stroke-width="1.5"/>')
    out.append(f'<text x="{fx}" y="{OY + GRID + 28}" font-size="11" text-anchor="end" fill="#c0282d">'
               'fast-mode pixels: last 5 (GPU) or 6 (CPU) of row 63</text>')
    lx, ly = OX + GRID + 28, OY + 6
    legend = [
        ("base", "Base layer", ["drawn fully opaque:", "never touched"]),
        ("overlay", "Overlay layer", ["drawn with transparency: alpha-0", "pixels here show nothing"]),
        ("unused", "Unused", ["no model face samples it"]),
        ("watermark", "Watermark", ['RGB spells "sokripon skinhash"', "+ version byte, alpha 0"]),
        ("fast", "Fast-mode search pixels", ["transparent sleeve overlay:", "RGB permuted, alpha stays 0"]),
    ]
    for kind, title, lines in legend:
        out.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{COL[kind]}" stroke="#888" stroke-width="0.6"/>')
        out.append(f'<text x="{lx + 24}" y="{ly + 13}" font-size="13" font-weight="600" fill="#1d1d1f">{title}</text>')
        for i, t in enumerate(lines):
            out.append(f'<text x="{lx + 24}" y="{ly + 30 + i * 15}" font-size="11.5" fill="#555">{t}</text>')
        ly += 34 + 15 * len(lines) + 10
    out.append(f'<text x="{lx}" y="{ly + 10}" font-size="11.5" fill="#555">Only pixels invisible in game</text>')
    out.append(f'<text x="{lx}" y="{ly + 25}" font-size="11.5" fill="#555">ever change. verify checks</text>')
    out.append(f'<text x="{lx}" y="{ly + 40}" font-size="11.5" fill="#555">every hit.</text>')
    out.append("</svg>")
    path = os.path.join(HERE, "pixel-map.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
