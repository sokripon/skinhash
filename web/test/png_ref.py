"""Write PNG variants (via Pillow) and their expected RGBA to a directory.
usage: py png_ref.py SKIN.png OUTDIR"""
import os, sys
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
im = Image.open(src).convert("RGBA")
def save(name, img, **kw):
    p = os.path.join(out, name + ".png")
    img.save(p, **kw)
    open(p + ".rgba", "wb").write(Image.open(p).convert("RGBA").tobytes())
save("rgba", im)
rgb = im.convert("RGB"); save("rgb", rgb)
save("rgb_trns", rgb, transparency=(0, 0, 0))
l = im.convert("L"); save("gray", l)
la = im.convert("LA"); save("graya", la)
pal = im.convert("RGB").quantize(64); save("pal", pal)
pt = im.quantize(64, method=Image.FASTOCTREE); save("pal_trns", pt, transparency=3)
save("interlaced", im, interlace=1) if False else None
print("ok")
