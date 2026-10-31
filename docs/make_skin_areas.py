"""Generate docs/skin-areas.png: the 64x64 skin layout with the pixels that mods
read and the pixels skinhash changes.

    py docs/make_skin_areas.py

Mod areas as reported from the mods' source (Ears, Customizable Player Models,
Mine Little Pony). Layout masks and skinhash pixels come from skinhash.py.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import skinhash  # noqa: E402

S = 12                      # pixels per skin pixel
OX, OY = 30, 70             # grid origin
GW = 64 * S
W, H = OX + GW + 40 + 430, OY + GW + 120


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_T, F_H, F_B, F_S = font(22, True), font(15, True), font(14), font(12)

C = {
    "base": (138, 162, 200), "overlay": (201, 182, 232), "unused": (244, 244, 242),
    "grid": (255, 255, 255), "cpm": (90, 90, 90),
    "ears": (240, 130, 30), "emissive": (250, 205, 40), "alfalfa": (40, 120, 60),
    "mlp": (220, 30, 120), "wm": (0, 170, 190), "mojang": (229, 72, 77), "namemc": (60, 60, 220),
    "unusedknob": (120, 70, 20), "text": (29, 29, 31), "muted": (90, 90, 90),
}
ears_block = {(x, y) for x in range(0, 4) for y in range(32, 36)}
emissive = {(x, y) for x in range(52, 56) for y in range(32, 36)}
head_front = {(x, y) for x in range(8, 16) for y in range(8, 16)}
mlp = {(0, 0): "race", (1, 0): "tail", (2, 0): "gender", (3, 0): "size",
       (0, 1): "glow", (1, 1): "wear", (2, 1): "tail shape", (2, 2): "priority"}
wm = {(x, 0) for x in range(6)}
mojang_k = {(x, 63) for x in range(58, 64)}
namemc_k = {(x, 63) for x in range(55, 58)}
unused_k = {(x, 51) for x in range(61, 64)}


def cell(x, y):
    return OX + x * S, OY + y * S, OX + (x + 1) * S - 1, OY + (y + 1) * S - 1


def main():
    base, over = skinhash.masks()
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((OX, 22), "64×64 skin layout: what mods read, what skinhash changes", font=F_T, fill=C["text"])

    for y in range(64):
        for x in range(64):
            p = (x, y)
            col = C["base"] if p in base else C["overlay"] if p in over else C["unused"]
            d.rectangle(cell(x, y), fill=col)
            x0, y0, x1, y1 = cell(x, y)
            if p in base and p not in head_front:            # Alfalfa: alpha of base faces
                d.point([(x0 + 2, y0 + 2), (x1 - 2, y1 - 2)], fill=C["alfalfa"])
            if p not in base and p not in over:              # CPM free space: hatch
                d.line([(x0, y1), (x1, y0)], fill=(200, 200, 196))
    for x in range(0, 65, 8):
        d.line([(OX + x * S, OY), (OX + x * S, OY + GW)], fill=C["grid"], width=1)
    for y in range(0, 65, 8):
        d.line([(OX, OY + y * S), (OX + GW, OY + y * S)], fill=C["grid"], width=1)

    def fill(pixels, color):
        for x, y in pixels:
            x0, y0, x1, y1 = cell(x, y)
            d.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), fill=color)

    def outline(pixels, color, width=3):
        xs = [p[0] for p in pixels]; ys = [p[1] for p in pixels]
        d.rectangle((OX + min(xs) * S - 1, OY + min(ys) * S - 1,
                     OX + (max(xs) + 1) * S, OY + (max(ys) + 1) * S), outline=color, width=width)

    fill(ears_block, C["ears"]); outline(ears_block, (150, 70, 0))
    fill(emissive, C["emissive"]); outline(emissive, (160, 120, 0))
    outline(head_front, C["alfalfa"], 2)
    fill(wm, C["wm"])
    fill(mojang_k, C["mojang"])
    fill(namemc_k, C["namemc"])
    fill(unused_k, C["unusedknob"])
    for (x, y) in mlp:                                       # MLP: X over the cell
        x0, y0, x1, y1 = cell(x, y)
        d.line([(x0 + 1, y0 + 1), (x1 - 1, y1 - 1)], fill=C["mlp"], width=3)
        d.line([(x0 + 1, y1 - 1), (x1 - 1, y0 + 1)], fill=C["mlp"], width=3)
    d.rectangle((OX - 1, OY - 1, OX + GW, OY + GW), outline=(60, 60, 60), width=2)

    # axis ticks
    for v in range(0, 64, 8):
        d.text((OX + v * S + 2, OY - 16), str(v), font=F_S, fill=C["muted"])
        d.text((OX - 24, OY + v * S + 1), str(v), font=F_S, fill=C["muted"])

    # legend
    lx, ly = OX + GW + 40, OY

    def swatch(kind, color, title, lines):
        nonlocal ly
        box = (lx, ly + 2, lx + 18, ly + 20)
        if kind == "fill":
            d.rectangle(box, fill=color, outline=(120, 120, 120))
        elif kind == "x":
            d.rectangle(box, fill=C["unused"], outline=(120, 120, 120))
            d.line([(box[0] + 3, box[1] + 3), (box[2] - 3, box[3] - 3)], fill=color, width=3)
            d.line([(box[0] + 3, box[3] - 3), (box[2] - 3, box[1] + 3)], fill=color, width=3)
        elif kind == "hatch":
            d.rectangle(box, fill=C["unused"], outline=(120, 120, 120))
            for k in range(0, 18, 4):
                d.line([(box[0] + k, box[3]), (box[0] + k + 6, box[3] - 6)], fill=(150, 150, 146))
        elif kind == "dots":
            d.rectangle(box, fill=C["base"], outline=(120, 120, 120))
            d.point([(box[0] + 5, box[1] + 5), (box[2] - 5, box[3] - 5), (box[0] + 5, box[3] - 5)], fill=color)
        d.text((lx + 28, ly), title, font=F_H, fill=C["text"])
        ly += 20
        for t in lines:
            d.text((lx + 28, ly), t, font=F_S, fill=C["muted"])
            ly += 16
        ly += 8

    d.text((lx, ly), "Layout", font=F_B, fill=C["text"]); ly += 22
    swatch("fill", C["base"], "Base layer", ["drawn opaque (alpha forced)"])
    swatch("fill", C["overlay"], "Overlay layer", ["drawn with transparency"])
    swatch("hatch", None, "Unused = CPM data area", ["never drawn; Customizable Player Models",
                                                    "reads all of it (B,G,R(,A)) if header",
                                                    "0x53 at blue of (0,0) + checksum match"])
    d.text((lx, ly), "Mods", font=F_B, fill=C["text"]); ly += 22
    swatch("fill", C["ears"], "Ears marker block  x0–3, y32–35", ["(0,32) = #EA2501 (v1) or #3F23D8 (v0)",
                                                                  "turns Ears on; avoid entirely"])
    swatch("fill", C["emissive"], "Ears emissive palette  x52–55, y32–35", [])
    swatch("dots", C["alfalfa"], "Ears Alfalfa", ["data in ALPHA of base faces (alpha",
                                                  "128–254), except head front (outlined)"])
    swatch("x", C["mlp"], "Mine Little Pony triggers", ["RGB of (0–3,0) (0–2,1) (2,2), alpha",
                                                        "ignored; 000000 = no pony"])
    d.text((lx, ly), "skinhash", font=F_B, fill=C["text"]); ly += 22
    swatch("fill", C["wm"], "Watermark v2  (0–5,0)", ["overlaps MLP + CPM header pixel"])
    swatch("fill", C["mojang"], "Mojang fast mode  (58|59–63,63)", ["RGB under alpha 0"])
    swatch("fill", C["namemc"], "NameMC default  (55–57,63)", ["alpha 0 → 1"])
    swatch("fill", C["unusedknob"], "--unused-only  (61–63,51)", ["Steve / typical skin"])

    # zoomed insets
    def inset(title, x0, y0, w, h, z, top):
        d.text((lx, top), title, font=F_H, fill=C["text"])
        top += 22
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                p = (x, y)
                col = (C["mojang"] if p in mojang_k else C["namemc"] if p in namemc_k else
                       C["unusedknob"] if p in unused_k else C["wm"] if p in wm else
                       C["base"] if p in base else C["overlay"] if p in over else C["unused"])
                bx, by = lx + (x - x0) * z, top + (y - y0) * z
                d.rectangle((bx, by, bx + z - 1, by + z - 1), fill=col, outline=(255, 255, 255))
                if p in mlp:
                    d.line([(bx + 4, by + 4), (bx + z - 5, by + z - 5)], fill=C["mlp"], width=3)
                    d.line([(bx + 4, by + z - 5), (bx + z - 5, by + 4)], fill=C["mlp"], width=3)
        for x in range(x0, x0 + w, 2 if w > 8 else 1):
            d.text((lx + (x - x0) * z + 3, top + h * z + 2), str(x), font=F_S, fill=C["muted"])
        return top + h * z + 22

    ly = inset("Zoom: x 0–7, y 0–2 (watermark under MLP triggers)", 0, 0, 8, 3, 26, ly + 6)
    inset("Zoom: x 48–63, y 63 (search pixels)", 48, 63, 16, 1, 24, ly)

    out = os.path.join(HERE, "skin-areas.png")
    img.save(out, optimize=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
