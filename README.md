# skinhashmaker

Brute-force a "cool looking" Minecraft skin texture id by varying pixels that are never rendered.

The texture id is the SHA-256 of the PNG Mojang serves, with leading zero digits dropped in the URL
(a zero-padded URL 404s). Mojang re-encodes uploads as signature + IHDR + exactly one IDAT
(zlib level 4, wbits 15, memLevel 8, filter 0 on every row) + IEND. RGB under alpha-0 pixels survives
that re-encode, so those bytes are free knobs.

## Usage

```
gcc -O3 -march=native -Wall -Wextra -o fastfinder.exe fastfinder.c -lz -lpthread
gcc -O3 -march=native -Wall -Wextra -o hashfinder.exe hashfinder.c -lz -lpthread

py skinhash.py predict skins/me.png                  # id Mojang would assign
py skinhash.py search  skins/me.png -p c0ffee        # id (trimmed) starts with c0ffee
py skinhash.py search  skins/me.png -z 8             # keep ids with >= 8 leading zeros (shortest links)
py skinhash.py search  skins/me.png -s dead -n 3     # id ends with dead, stop after 3 hits
py skinhash.py verify  skins/me.png found/me/<id>.png
```

Options of `search` (forwarded to the finder): `-p HEX` (prefix of the TRIMMED id, repeatable, may not
start with `0`), `-s HEX` (suffix), `-z N` (save anything with >= N leading zero nibbles), `-t N` threads,
`-S SEED`, `-n N` stop after N hits, `-b` benchmark (10 s, saves no files).
Python-only: `--generic` force hashfinder, `--k N` pixels permuted by fast mode (default 6),
`--unused-only` only use pixels outside the model UV (disables fast mode), `--out DIR` (default `found/<skin>/`).

`-p c0ffee` matches both `c0ffee...` and `000c0ffee...` hashes, because the link drops leading zeros.
Every hit is re-verified by `skinhash.py` (hash == filename, Python zlib re-encodes the file identically,
no visible pixel changed) and printed as `HIT <why> sha=<full hex> id=<trimmed id> counter=<n> file=<path>`.

## How it works

* `hashfinder.c` (generic, ~150k/s): writes a counter into knob bytes, deflates the changed tail from a cloned
  zlib state, hashes the whole PNG.
* `fastfinder.c` (fast, ~50M/s): takes the last K pixels (row 63) and gives their 3K RGB bytes 3K distinct byte
  values that occur nowhere else in the image. Every candidate is a permutation of those values (Heap's
  algorithm, reshuffled randomly after 12! steps). Then:
  1. every window of the tail contains a unique byte, so zlib emits only literals there and everything
     before the tail is unaffected;
  2. the literal multiset is the same for every permutation, so Huffman trees, block header, bit length,
     IDAT length and CRC position are constant;
  3. per candidate only the tail literal codes (permuted), Adler-32 and the IDAT CRC change; they are written
     straight into the file and SHA-256 resumes from a midstate over the fixed whole blocks (2 blocks/candidate).

  At start-up fastfinder builds a reference stream with real zlib, parses it with a small deflate parser to
  find the bit where the first knob byte starts, checks that the block is single/final and the tail is all
  literals, and self-tests 1000 random permutations byte-for-byte against real zlib. Every hit is rebuilt
  with real zlib again before being reported. If an assumption fails it exits with code 4 and
  `skinhash.py` falls back to hashfinder.

## Rough times (fastfinder at ~48M/s on 12 threads, expected time = 16^n / rate)

| goal                          | tries    | expected time |
|-------------------------------|----------|---------------|
| 6 hex chars prefix            | 1.7e7    | < 1 s         |
| 8 leading zeros               | 4.3e9    | ~1.5 min      |
| 9 leading zeros               | 6.9e10   | ~24 min       |
| 10 leading zeros              | 1.1e12   | ~6.4 h        |

(Generic hashfinder is ~300x slower: 8 zeros would take ~8 hours.) The permutation space for K=6 is
18! ~ 6.4e15 per value set, so it does not run out; more distinct value sets come from different `-S` seeds.

## GPU mode (CUDA)

`gpufinder.cu` runs the same fast search on an NVIDIA GPU (built for `sm_120`, RTX 5060 Ti; needs CUDA 12.8
and Visual Studio 2022 for cl.exe). zlib is not needed in the CUDA binary: `fastfinder --dump-job FILE`
(after its usual self-test against real zlib) writes a small job file with the constant file prefix, SHA-256
midstate, tail layout, Huffman code per byte value, the value set, Adler/CRC prefix state, the patterns and
16 CPU-computed test vectors, then exits. Each GPU thread turns a 64-bit index into a permutation (Lehmer
code over the first min(3K,20) values; the launch base is random), emits the literal codes while updating
Adler-32, finishes CRC-32 and resumes SHA-256 from the midstate (2 blocks).

```
powershell -File build.ps1                 # builds hashfinder, fastfinder, gpufinder (-NoGpu to skip CUDA)
py skinhash.py search skins/me.png --gpu -z 9
py skinhash.py search skins/me.png --gpu -p c0ffee -n 3
py skinhash.py search skins/me.png --gpu -b          # 10 s benchmark, prints RATE n, saves nothing

# manual:
fastfinder.exe found/me/base.raw 6 found/me --dump-job job.bin -p dead
gpufinder.exe job.bin found/me [-p HEX] [-s HEX] [-z N] [-S SEED] [-n N] [-b]
```

* `gpufinder` takes -p/-s/-z/-n/-S/-b like the other finders (patterns given to it replace those in the job)
  and prints the same `HIT ... file=` lines and progress lines. `skinhash.py` still verifies every hit.
* Safety: before searching, the GPU must reproduce all test vectors of the job bit for bit, otherwise it exits
  with code 5. Every hit is rebuilt on the CPU (independent code path) and re-hashed with a CPU SHA-256
  before it is reported or written. `--gpu` falls back to CPU fastfinder (with a message) if gpufinder.exe is
  missing or fails.
* Measured on an RTX 5060 Ti: about 1.24e9 candidates/s, versus about 60e6/s for fastfinder on 12 CPU threads
  (roughly 20x). Expected times for `-z N` (16^N / 1.24e9 per s):

| leading zeros | tries   | expected time |
|---------------|---------|---------------|
| 8             | 4.3e9   | ~3.5 s        |
| 9             | 6.9e10  | ~55 s         |
| 10            | 1.1e12  | ~15 min       |
| 11            | 1.8e13  | ~3.9 h        |
| 12            | 2.8e14  | ~63 h         |

* Limits: at most 4096 hits are kept per kernel launch (use a stricter pattern if you see the warning);
  `-t` is accepted and ignored; one value set holds 18! = 6.4e15 permutations for K=6, enough for 12 zeros
  on average, other value sets come from different `-S` seeds of the dump.

## Watermark

Every search stamps an invisible signature so skins made with this tool can be recognised:

- **Pixels:** a run of horizontally adjacent, fully transparent pixels in an unused part of the layout (no model face samples it). Their RGB bytes, read left to right, spell the payload.
- **v2 (current):** `sokripon skinhash` + byte `0x02` = 18 bytes = **6 pixels, normally (0,0)-(5,0)**, in the unused top-left corner of the head layout. RGBA: `736f6b00 72697000 6f6e2000 736b6900 6e686100 73680200`.
- **v1 (older runs):** `SKHAS` + `0x01` padded with a 0 byte = 2 pixels at (0,0)-(1,0). Still detected.
- **Visibility:** alpha is 0, so nothing shows in game or in image viewers. Mojang keeps RGB under alpha 0 (confirmed by a real upload).
- **Fallback:** if the preferred spot is not free (alpha > 0), the first free unused transparent run in scan order is used. Detection scans the whole image.
- **Speed:** the watermark sits in the fixed part of the image, so it costs no search speed. The fast modes automatically avoid its byte values.
- **Check a skin:** `py skinhash.py detect FILE|ID|URL ...` (ids and URLs are downloaded from textures.minecraft.net).
- **Turn it off:** `--no-watermark`.
