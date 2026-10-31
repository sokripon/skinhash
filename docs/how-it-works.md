# How skinhash works

## 1. The texture link is a hash of the pixels

![Pipeline from skin pixels to texture link](pipeline.svg)

When you upload a skin, Mojang decodes it and **re-encodes** it in one fixed way:

- 64×64 RGBA, every row prefixed with filter byte 0;
- compressed with zlib level 4 into a single IDAT chunk;
- no other chunks.

The texture id is the SHA-256 of that file, written in hex **without leading zeros**. So the bytes of the file you upload don't matter, only the pixels. We confirmed this by predicting the id of a test upload exactly, byte for byte.

If you change any pixel, you get a completely different, unpredictable id. Finding a "cool" id therefore means trying many invisible variations until one hashes to something you like:

- a hex word at the start (`c0ffee…`, `cafebabe…`);
- many leading zeros, which make the link shorter.

## 2. Which pixels may change

![Map of the 64×64 skin layout](pixel-map.svg)

| Area | How Minecraft treats it | Used by the tool? |
|---|---|---|
| **Base layer** (blue) | Drawn fully opaque. The skin loader forces alpha to 255 here, so hidden colours would show. | Never touched |
| **Overlay layer** (purple) | Drawn with transparency. A pixel with alpha 0 shows nothing, whatever its RGB. | Only pixels that are already fully transparent |
| **Unused** (grey) | No face of the player model samples it, so it's never drawn. | Watermark, `--unused-only` |

### Which pixels each mode changes

| Mode | Pixels changed | Coordinates on the test skin |
|---|---|---|
| Watermark (every mode) | 6 unused transparent pixels; RGB spells `sokripon skinhash` + version byte `0x02` | (0, 0)–(5, 0), top-left corner of the head layout |
| GPU fast mode (`--gpu`) | the last 5 pixels of the file, as long as they are invisible | (59–63, 63), end of the left sleeve overlay |
| CPU fast mode (default) | the last 6 pixels of the file, as long as they are invisible | (58–63, 63), end of the left sleeve overlay |
| Generic mode (`--generic`, or when the last pixels are visible) | a 64-bit counter in the RGB of the last 3 invisible pixels found | (61–63, 63) |
| Generic + `--unused-only` | the same, but only pixels no model face ever samples (also disables fast mode) | (61–63, 51), unused corner next to the left sleeve |

**Rules that apply to all modes:**
- Alpha is never changed. Only the RGB under an alpha of 0 is.
- The tool **checks** each skin instead of assuming. Fast mode is only used if every one of its pixels is a transparent overlay pixel or an unused pixel. Otherwise it falls back to generic mode with other invisible pixels.
- Before you upload, `verify` re-checks every hit:
  - the file hashes to its name;
  - real zlib reproduces it byte for byte;
  - no visible pixel differs from the original skin.

### How sure are we that the changes are invisible?

| Claim | Status | Evidence |
|---|---|---|
| Mojang's servers keep the hidden RGB unchanged | **Proven** | A test upload and the 11-zero upload both got exactly the predicted id. That's only possible if every hidden byte survived. |
| The tool only touches transparent overlay or unused pixels | **Checked by code** | Selection rules above; `verify` checks every hit. |
| The Java client draws nothing for alpha-0 overlay pixels | **Very likely** | The skin loader's alpha-forcing only covers base-layer regions, not the sleeve overlay. The overlay is drawn with alpha cutout, and skin textures aren't filtered. Partly transparent overlays with leftover colour are common in real skins. |
| Every other renderer (Bedrock via Geyser, third-party viewers, mods) does the same | **Not tested** | `--unused-only` avoids the question entirely: those pixels are never drawn at all. |

To check yourself: wear a skin made by the tool, open **Options → Skin Customization** or press F5, and look at the end of the left sleeve. Nothing should be visible there.

## 3. The fast-mode trick

![Why fast mode only hashes the last block](fast-mode.svg)

Re-compressing the whole skin for every candidate is slow, about 170k tries/s. Fast mode avoids compression entirely:

1. It fills the search pixels with byte values that occur **nowhere else** in the image. zlib can't match them against earlier data, so they always come out as plain literals.
2. Each candidate is a **reordering** of the same values. The symbol frequencies don't change, so the Huffman tables, the compressed size and every bit before the tail are identical for all candidates.
3. Only the last ~50 bytes of the file change: the tail literal codes, the Adler-32 and the CRC-32.
4. SHA-256 processes 64-byte blocks, so the ~55 fixed blocks are hashed once into a saved midstate. Each candidate then only has to finish the last block or two.

The result is the same file real zlib would produce, built with a few dozen bit operations. A start-up self-test compares 1000 random candidates against real zlib, and every hit is rebuilt and checked again.

## 4. Inside GPU mode

The CPU prepares a small "job" once. It contains the fixed file prefix, the SHA-256 midstate, the Huffman codes of the search values, the Adler/CRC state and 16 test vectors. The GPU then hashes candidates from it.

### What made it fast (RTX 5060 Ti)

| Step | Rate |
|---|---|
| First working version (6 pixels, 2 SHA blocks per candidate) | 1.16 G/s |
| **5 pixels instead of 6.** The changing bytes plus SHA padding now fit into **one** 64-byte block, which halves the SHA work. | 1.93 G/s |
| **Smarter enumeration.** Each GPU thread decodes the first positions of its permutation once with 32-bit maths (no 64-bit divisions), then tries all 24 orderings of the last 4 positions. The bits after the shuffled region are constant, so only a few bytes are patched in. CRC-32 runs word by word, the Adler sum is updated linearly, and the global "best so far" counter is touched once per thread instead of once per candidate. | 3.58 G/s |
| Block-size and occupancy tuning | **~3.85 G/s** |

A kernel that does nothing but SHA-256 reaches about 4.46 G/s, so there's little headroom left.

**Choosing the pixel count:** `fastfinder --dump-job --auto` tries pixel counts from 6 downwards, with up to 400 random value sets each. It picks the largest count whose changing bytes fit one SHA block: 5 on the test skin, where 6 never fits. If nothing fits, it falls back to the best 2-block configuration, at about 2.1 G/s.

### Value-set rollover

5 pixels hold 15 byte values, so one value set has 15! ≈ 1.3 trillion orderings. That's about 5.6 minutes of GPU time. The search then **rolls over** to a new one:

1. gpufinder covers every ordering of the current set exactly once, prints `ROLLOVER` and exits with code 7.
2. `skinhash.py` builds a new job with a fresh random value set, and the GPU passes the correctness check again.
3. The search continues. The `tried=` counter, the rate and benchmarks keep counting across rollovers.

A rollover costs under a second per ~5.6 minutes of searching.

### Safety checks

- **Correctness check:** before searching, and again after every rollover, the GPU must reproduce all 16 CPU-computed test vectors exactly. Otherwise it exits with code 5 and `skinhash.py` falls back to the CPU.
- **Hit re-check:** every hit is rebuilt and re-hashed on the CPU before it's reported. `skinhash.py` then verifies it again with Python's real zlib.
- **Ctrl+C:** stops the whole search and never triggers the CPU fallback.

## 5. Speed

![Search speed comparison](speed.svg)

| Mode | Command | Rate |
|---|---|---|
| Generic (CPU) | `--generic` | ~170k/s |
| Fast mode (CPU) | default | ~43M/s |
| Fast mode (GPU) | `--gpu` | ~3.85G/s |

Each extra leading zero takes 16× longer on average:

| Target | GPU at 3.85G/s |
|---|---|
| 8-char hex word | ~1 s |
| 9 zeros | ~18 s |
| 10 zeros | ~5 min |
| 11 zeros | ~75 min |
| 12 zeros | ~20 h |

**Results so far:** an 11-zero id, `00000000000ce817…`. It's live as the link `textures.minecraft.net/texture/ce81796dbeb618dd6f893689e6f05a98faea679edba3bfda50315` (53 characters). As far as a ~64M-skin archive shows, the previous shortest real skin link had 8 zeros.

---

`pixel-map.svg` is generated from the same layout tables the tool uses: `py docs/make_pixel_map.py`.
