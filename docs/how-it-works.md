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

- **Base layer (blue):** Minecraft draws it fully opaque and ignores alpha there, so hidden colours would show up. The tool never touches it.
- **Overlay layer (purple):** drawn with transparency. A pixel with alpha 0 shows nothing, whatever its RGB, and Mojang keeps that RGB (verified by a real upload).
- **Unused (grey):** no part of the player model samples these pixels.
- **Watermark (orange):** 6 transparent unused pixels whose RGB bytes spell `sokripon skinhash` + a version byte. Check a skin with `py skinhash.py detect`.
- **Search pixels (red):** the fast modes vary the RGB of the last 6 pixels of the file. On this skin they are transparent sleeve-overlay pixels. Their alpha stays 0, so they stay invisible.

`verify` re-checks every hit: the file hashes to its name, real zlib reproduces it byte for byte, and no visible pixel differs from the original.

## 3. The fast-mode trick

![Why fast mode only hashes the last block](fast-mode.svg)

Re-compressing the whole skin for every candidate is slow, about 170k tries/s. Fast mode avoids compression entirely:

1. It fills the search pixels with byte values that occur **nowhere else** in the image. zlib can't match them against earlier data, so they always come out as plain literals.
2. Each candidate is a **reordering** of the same values. The symbol frequencies don't change, so the Huffman tables, the compressed size and every bit before the tail are identical for all candidates.
3. Only the last ~50 bytes of the file change: the tail literal codes, the Adler-32 and the CRC-32.
4. SHA-256 processes 64-byte blocks, so the ~55 fixed blocks are hashed once into a saved midstate. Each candidate then only finishes the last 1–2 blocks.

The result is the same file real zlib would produce, built with a few dozen bit operations. A start-up self-test compares 1000 random candidates against real zlib, and every hit is rebuilt and checked again.

## 4. Speed

![Search speed comparison](speed.svg)

| Mode | Command | Rate |
|---|---|---|
| Generic (CPU) | `--generic` | ~170k/s |
| Fast mode (CPU) | default | ~43M/s |
| Fast mode (GPU) | `--gpu` | ~1.25G/s (optimization in progress) |

Each extra leading zero takes 16× longer on average:

| Target | GPU at 1.25G/s |
|---|---|
| 8-char hex word | ~3 s |
| 9 zeros | ~1 min |
| 10 zeros | ~15 min |
| 11 zeros | ~4 h |

---

`pixel-map.svg` is generated from the same layout tables the tool uses: `py docs/make_pixel_map.py`.
