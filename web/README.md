# skinhash in the browser (WebGPU)

A static web page version of the native tool: drop your own 64x64 Minecraft skin, pick targets (hex prefix,
suffix, contains, leading zeros) and it searches invisible variations of the skin on your GPU. No build step, no server
code, no upload; it can be hosted as-is (for example on GitHub Pages).

Contains (hex) matches a hex string anywhere in the id: in the Mojang link id with leading zeros dropped, or in all 16 NameMC digits.

Patterns (one per line, in each target group) take the same syntax as the native `-m` (see the main README): `^` start,
`$` end, `?`, `?d`, `?l`, `[0-3]` classes, `{n}` repeats, `*` gaps, `#` comments and comma lists. Each line shows its odds
and the expected time at the last measured rate (or an assumed one), or its error. "Load file" reads a `.txt` file into
the box. The parser is `js/patterns.js` (same rules as `patterns.py` and `pattern.h`). Anchored patterns are baked into
the generated Mojang shader as literal masks, so words without a mask cost nothing; unanchored ones use the sliding
window of the contains check. Every hit is re-checked against the full pattern on the CPU.

The Patterns box also takes stat lines (counts, shapes, records), the same syntax as the native tool (main README,
"Stat lines"). Spaces and capitals are free, errors show the position:

| Line | Meaning |
|---|---|
| `#0{24,}` | at least 24 zeros (`{n}` exactly, `{n,}` at least, `{,n}` at most, `{n,m}`); `#?l{,0}` no letters, `#?l{50,}` 50+ letters, `#[0-3]{20,}` any class |
| `##{,9}` | at most 9 different digits |
| `#(dead\|beef\|cafe){3,}` | words of 1-8 digits found 3+ times together (`#@file` is native only) |
| `(?)\1{11}` | one digit 12 times in a row; `(??)\1{6}` a pair 7 times (zigzag), `(?{3})\1{4}` a triple, period 1-8 |
| `+{12}` `-{12}` | staircase up / down of 12 digits (no wrap f to 0) |
| `~{25}` | a palindrome of at least 25 digits |
| `^f{max}` `0{max}` | the longest run of that digit (from the start / anywhere) |
| `A & B` | all must hold, e.g. `^dead & #0{10,}` |
| `~{max}` `+{max}` `#?l{min}` `##{min}` | record lines: every new best is listed as `RECORD ~{max}: 19 (was 18)` with its PNG; no end, press Stop; the best is kept while the page is open ("Forget records" clears it); not in Both mode |

Mojang counts and shapes look at the link id (leading zeros dropped); NameMC at the 16 digits. A threshold line is a
normal target (it counts as a hit). The values are computed on the GPU: `js/wgslstat.js` generates the WGSL for each
line and bakes it into the Mojang (fast and generic) and NameMC kernels; a shader without stat lines contains none of
it. Counts and runs use nibble-wise flag arithmetic on the digest words (a 64-bit run mask for runs, palindromes
compare 8-digit windows with the nibble-reversed digest per centre); the best of a record line lives in a storage
buffer and only improvements are reported. Every reported candidate is re-checked with the JS reference first.

## Run locally

    py web/serve.py            # http://127.0.0.1:8000/
    # or: py -m http.server -d web

`serve.py` is the stock server plus correct MIME types for `.js`. On some Windows machines the stock server
answers `text/plain` for `.js` and the browser refuses to load the ES modules. Needs a WebGPU browser
(desktop Chrome/Edge 113+, Chrome on Android 121+, Safari 26+, Firefox 141+ on Windows).

## How it maps to the native tool

| Web | Native |
|---|---|
| `js/patterns.js` | `patterns.py` / `pattern.h`: pattern and stat line parser, mask compiler, matcher, odds |
| `js/wgslstat.js` | the GPU stat checks (`gpufinder.cu` / `namemcgpu.cu`): counts, runs, staircases, palindromes, records |
| `js/model.js` | `skinhash.py`: UV masks, `knob_pixels`, watermark v3, `RESERVED` pixels |
| `js/png.js` | PNG decode without canvas (canvas would premultiply alpha and destroy hidden RGB); Mojang re-encode (`mojang_png`) |
| `js/job.js` | `fastfinder.c`: `setup_fast`, `parse_deflate`, `--auto`, self-test, test vectors, index to permutation |
| `js/hash.js` | SHA-256 (midstate), CRC-32 |
| `js/wgsl.js`, `js/gpu.js` | `gpufinder.cu`, generic kernel: same enumeration (outer x 24 orderings), word-wise CRC, any layout, u32 only |
| `js/wgslfast.js` | `gpufinder.cu`, fast kernel (step 7 in `docs/optimizations.md`): generated per job, see below |
| `js/verify.js` | `skinhash.py verify`: every hit rebuilt, hashed, pixel-compared, re-encoded |
| `vendor/pako.esm.mjs` | zlib; pako 2.1.0, unmodified, MIT and Zlib licences in `vendor/LICENSE-pako` |

### NameMC mode

The mode selector switches between the Mojang texture link and the NameMC skin page
(`namemc.com/skin/<16 hex>`). NameMC id = first 8 bytes of SHA-256(be32(64) || be32(64) || RGBA, RGB = 0 where
alpha = 0); no PNG or zlib is involved in the id. The knob is three fully transparent overlay (or, with
"unused pixels only", unused) pixels that get alpha 1 and a 64-bit counter as RGB: (55-57, 63) by default, which
leaves the Mojang fast-mode tail free. Prefixes are matched on the fixed 16 digits (a leading 0 is allowed).
NameMC's URL shows all 16 digits, leading zeros included. Alpha 1 of 255 should be invisible in game; that
is not verified yet.

| Web | Native |
|---|---|
| `js/namemc.js` | `namemcfinder.c` / `skinhash.py`: knob choice, message, midstate, test vectors |
| `js/nmgpu.js` | `namemcgpu.cu`: fast kernel (2 blocks, constant second block baked in) and general kernel (any layout) |

Each hit is verified on the CPU (recomputed NameMC id, only the three knob pixels and the watermark differ),
and shown with its NameMC link, the Mojang texture link of the downloaded PNG, and a button "Use as skin for a
Mojang search": it loads the hit as the current skin and switches to Mojang mode. The alpha-1 pixels are fixed
there, so the NameMC id stays the same.

### Both mode

"Both (one click)" takes NameMC targets and Mojang targets in two groups and runs two phases: a NameMC search
(first verified hit, or the first of all targets with "until every target is found"), then, with that hit
loaded as the base skin, the Mojang search. The alpha-1 NameMC pixels are fixed, so fast mode stays eligible
(otherwise it stops with a message). The final PNG is verified against the original skin (NameMC id as in phase
1 and matching its targets, Mojang id matching its targets, only the NameMC knobs, the watermark and hidden RGB
under alpha 0 differ) and shown with both links. The phases add up instead of multiplying (6 hex digits each is
about 34 million tries instead of 16^12). With one group empty only the other phase runs. The native
`skinhash.py verify <original> <final>` accepts the result too (it treats alpha 0 to 1 on overlay or unused pixels
as a NameMC knob).

Mojang kernel: for one-block jobs whose permuted values all have the same Huffman code length (the job builder
prefers those, like `fastfinder --auto`) `js/wgslfast.js` generates a shader per job: message words are an OR of
per-(position, value) table entries, the CRC an XOR of precomputed deltas, Adler-b a sum of precomputed terms, and
SHA rounds 0..WI-1 and the thread-only schedule terms run once per thread. Other jobs use the generic kernel. The
status line says which one runs. In Chrome on an RTX 5060 Ti this is about 3.7 G/s (the generic kernel: 2.9 G/s;
native CUDA: 4.8 G/s). If the fast kernel does not reproduce every test vector the page falls back to the generic
kernel and gates that; it refuses to search only if both fail.

Differences: the zlib stream comes from pako instead of C zlib (proven byte-identical, see below); fast mode
only (no slow generic CPU mode); fast mode needs the last 5 pixels of the image invisible (the native tool asks
for 6 and uses K=5 or 6); the whole value-set rollover happens in the page.

Safety checks: before searching, and after every value-set rollover, a CPU self-test (250 random permutations
equal to full pako output) and a GPU gate (128 test vectors, 24 against full pako and 104 from the byte-level builder, reproduced exactly) must pass, otherwise the
search refuses to run. Every hit is rebuilt on the CPU, hashed with WebCrypto and JS, compared with the
original skin (only search pixels and watermark differ, alpha unchanged, nothing visible) and re-encoded with
pako before it is shown.

## Tests

    node web/test/pako_exact.mjs   # pako vs Python zlib, byte for byte (needs `py`)
    node web/test/png_decode.mjs   # PNG decoder vs Pillow (needs `py` + Pillow)
    node web/test/job.mjs          # job builder, self-test, rollover, eligibility
    node web/test/fastkernel.mjs   # fast Mojang kernel: table model vs byte-level builder, generated WGSL, equal-length preference
    node web/test/namemc.mjs       # NameMC ids, JS reference vs Python hashlib
    node web/test/patterns.mjs     # patterns: parser, errors, odds, masks, the Patterns box, prefilter soundness
    node web/test/patterns_gpu.mjs # generated pattern and stat WGSL on the GPU vs the JS model (needs WEBTEST_DIR)
    node web/test/stat_parity.mjs  # stdin JSON [[width, text, [ids]]], stdout the stat values (test_patterns.py compares with Python)
    node web/test/browser_test.mjs # Chrome + WebGPU end to end (Mojang and NameMC modes, combined flow), checks the downloads with skinhash.py and namemc_check.py; also imports Dinnerbone by name (a few playerdb/Mojang lookups)

`browser_test.mjs` needs `puppeteer-core`, which is deliberately not in this repo: install it in a scratch
folder and set `WEBTEST_DIR` to that folder.

On start the page loads classic Steve (texture `1a4af718…`) from textures.minecraft.net, which allows cross-origin requests, and checks that the bytes hash to that id. The texture isn't bundled. Dropping or choosing your own skin replaces it. Offline, the page simply starts empty.

Import by username: next to the drop zone, "Username or UUID" + Load (Enter works) also takes a texture id or URL. Mojang's APIs send no CORS headers, so the page looks the player up through playerdb.co (fallback api.ashcon.app); this is the only third-party dependency, and only for the lookup. The skin itself comes from textures.minecraft.net and its SHA-256 must equal the texture id, so a wrong answer cannot yield a different skin. Errors (not found, no custom skin, legacy 64x32 skin, rate-limited, network) show in the skin error box and keep the current skin. The native tool does the same with `py skinhash.py mojang <name> ...` (see the main README, "Skin input").

All tests use `skins/steve.png` plus skins generated inside the tests.
