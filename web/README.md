# skinhash in the browser (WebGPU)

A static web page version of the native tool: drop your own 64x64 Minecraft skin, pick targets (hex prefix,
suffix, leading zeros) and it searches invisible variations of the skin on your GPU. No build step, no server
code, no upload; it can be hosted as-is (for example on GitHub Pages).

## Run locally

    py web/serve.py            # http://127.0.0.1:8000/
    # or: py -m http.server -d web

`serve.py` is the stock server plus correct MIME types for `.js`. On some Windows machines the stock server
answers `text/plain` for `.js` and the browser refuses to load the ES modules. Needs a WebGPU browser
(desktop Chrome/Edge 113+, Chrome on Android 121+, Safari 26+, Firefox 141+ on Windows).

## How it maps to the native tool

| Web | Native |
|---|---|
| `js/model.js` | `skinhash.py`: UV masks, `knob_pixels`, watermark v2 |
| `js/png.js` | PNG decode without canvas (canvas would premultiply alpha and destroy hidden RGB); Mojang re-encode (`mojang_png`) |
| `js/job.js` | `fastfinder.c`: `setup_fast`, `parse_deflate`, `--auto`, self-test, test vectors, index to permutation |
| `js/hash.js` | SHA-256 (midstate), CRC-32 |
| `js/wgsl.js`, `js/gpu.js` | `gpufinder.cu`: same enumeration (outer x 24 orderings), word-wise CRC, one SHA block, u32 only |
| `js/verify.js` | `skinhash.py verify`: every hit rebuilt, hashed, pixel-compared, re-encoded |
| `vendor/pako.esm.mjs` | zlib; pako 2.1.0, unmodified, MIT and Zlib licences in `vendor/LICENSE-pako` |

### NameMC mode

The mode selector switches between the Mojang texture link and the NameMC skin page
(`namemc.com/skin/<16 hex>`). NameMC id = first 8 bytes of SHA-256(be32(64) || be32(64) || RGBA, RGB = 0 where
alpha = 0); no PNG or zlib is involved in the id. The knob is three fully transparent overlay (or, with
"unused pixels only", unused) pixels that get alpha 1 and a 64-bit counter as RGB: (55-57, 63) by default, which
leaves the Mojang fast-mode tail free. Prefixes are matched on the fixed 16 digits (a leading 0 is allowed).
Whether NameMC keeps leading zeros of the id in its URL is unverified. Alpha 1 of 255 should be invisible in
game; that is not verified either.

| Web | Native |
|---|---|
| `js/namemc.js` | `namemcfinder.c` / `skinhash.py`: knob choice, message, midstate, test vectors |
| `js/nmgpu.js` | `namemcgpu.cu`: fast kernel (2 blocks, constant second block baked in) and general kernel (any layout) |

Each hit is verified on the CPU (recomputed NameMC id, only the three knob pixels and the watermark differ),
and shown with its NameMC link, the Mojang texture link of the downloaded PNG, and a button "Use as skin for a
Mojang search": it loads the hit as the current skin and switches to Mojang mode. The alpha-1 pixels are fixed
there, so the NameMC id stays the same.

Differences: the zlib stream comes from pako instead of C zlib (proven byte-identical, see below); fast mode
only (no slow generic CPU mode); fast mode needs the last 5 pixels of the image invisible (the native tool asks
for 6 and uses K=5 or 6); the whole value-set rollover happens in the page.

Safety checks: before searching, and after every value-set rollover, a CPU self-test (250 random permutations
equal to full pako output) and a GPU gate (24 CPU test vectors reproduced exactly) must pass, otherwise the
search refuses to run. Every hit is rebuilt on the CPU, hashed with WebCrypto and JS, compared with the
original skin (only search pixels and watermark differ, alpha unchanged, nothing visible) and re-encoded with
pako before it is shown.

## Tests

    node web/test/pako_exact.mjs   # pako vs Python zlib, byte for byte (needs `py`)
    node web/test/png_decode.mjs   # PNG decoder vs Pillow (needs `py` + Pillow)
    node web/test/job.mjs          # job builder, self-test, rollover, eligibility
    node web/test/namemc.mjs       # NameMC ids, JS reference vs Python hashlib
    node web/test/browser_test.mjs # Chrome + WebGPU end to end (Mojang and NameMC modes, combined flow), checks the downloads with skinhash.py and namemc_check.py

`browser_test.mjs` needs `puppeteer-core`, which is deliberately not in this repo: install it in a scratch
folder and set `WEBTEST_DIR` to that folder.

On start the page loads classic Steve (texture `1a4af718…`) from textures.minecraft.net, which allows cross-origin requests, and checks that the bytes hash to that id. The texture isn't bundled. Dropping or choosing your own skin replaces it. Offline, the page simply starts empty.

All tests use `skins/steve.png` plus skins generated inside the tests.
