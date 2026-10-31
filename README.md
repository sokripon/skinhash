# skinhashmaker

Finds a "cool looking" Minecraft skin texture id by brute-forcing pixels that are invisible in game.

[How it works](docs/how-it-works.md) · [Optimization log](docs/optimizations.md)

The texture id is the SHA-256 of the PNG Mojang serves, with leading zero digits dropped in the URL
(a zero-padded URL 404s). Mojang re-encodes uploads as signature + IHDR + exactly one IDAT
(zlib level 4, wbits 15, memLevel 8, filter 0 on every row) + IEND. RGB under alpha-0 pixels survives
that re-encode, so those bytes are free knobs.

## Usage

```
gcc -O3 -march=native -Wall -Wextra -o fastfinder.exe fastfinder.c -lz -lpthread
gcc -O3 -march=native -Wall -Wextra -o hashfinder.exe hashfinder.c -lz -lpthread
gcc -O3 -march=native -Wall -Wextra -o namemcfinder.exe namemcfinder.c -lz -lpthread   # NameMC mode (below)

py skinhash.py predict skins/me.png                  # id Mojang would assign
py skinhash.py search  skins/me.png -p c0ffee        # id (trimmed) starts with c0ffee
py skinhash.py search  skins/me.png -z 8             # keep ids with >= 8 leading zeros (shortest links)
py skinhash.py search  skins/me.png -s dead -n 3     # id ends with dead, stop after 3 hits
py skinhash.py verify  skins/me.png found/me/<id>.png
py skinhash.py search  skins/me.png --gpu --fast -z 10 # never fall back to the slow generic search
```

`search` forwards these options to the finder:

- `-p HEX`: prefix of the trimmed id. Repeatable, may not start with `0`.
- `-s HEX`: suffix.
- `-z N`: save anything with >= N leading zero nibbles.
- `-t N`: threads.
- `-S SEED`: start seed.
- `-n N`: stop after N hits.
- `-b`: benchmark, 10 s, saves no files.

These are handled by `skinhash.py` itself:

- `--generic`: force hashfinder.
- `--k N`: pixels permuted by fast mode (default 6). With `--gpu` it is an upper bound: the job builder picks the largest K <= N that gives a one-block SHA.
- `--unused-only`: only use pixels outside the model UV. Disables fast mode.
- `--out DIR`: output directory (default `found/<skin>/`).

`-p c0ffee` matches both `c0ffee...` and `000c0ffee...` hashes, because the link drops leading zeros.
`skinhash.py` re-verifies every hit (hash equals filename, Python zlib re-encodes the file identically,
no visible pixel changed) and prints `HIT <why> sha=<full hex> id=<trimmed id> counter=<n> file=<path>`.

## Finders

`hashfinder.c` (generic, ~150–170k/s) writes a counter into knob bytes, deflates the changed tail from a cloned
zlib state and hashes the whole PNG.

`fastfinder.c` (~45–50M/s) takes the last K pixels (row 63) and gives their 3K RGB bytes 3K distinct values
that occur nowhere else in the image. Every candidate is a permutation of those values (Heap's algorithm,
reshuffled randomly after 12! steps). Then:

1. Every window of the tail contains a unique byte, so zlib emits only literals there and nothing before
   the tail is affected.
2. The literal multiset is the same for every permutation. Huffman trees, block header, bit length, IDAT
   length and CRC position are constant.
3. Per candidate only the permuted tail codes, Adler-32 and the IDAT CRC change. They are written straight
   into the file, and SHA-256 resumes from a midstate over the fixed whole blocks (2 blocks per candidate).

At start-up fastfinder builds a reference stream with real zlib and parses it to find the bit where the
first knob byte starts. It checks that the block is single and final and that the tail is all literals. Then
it self-tests 1000 random permutations byte for byte against real zlib. Every hit is rebuilt with real zlib
before it is reported. If an assumption fails it exits with code 4 and `skinhash.py` falls back to
hashfinder.

## Expected times

Fastfinder at ~48M/s on 12 threads, 16^n / rate:

| goal                          | tries    | expected time |
|-------------------------------|----------|---------------|
| 6 hex chars prefix            | 1.7e7    | < 1 s         |
| 8 leading zeros               | 4.3e9    | ~1.5 min      |
| 9 leading zeros               | 6.9e10   | ~24 min       |
| 10 leading zeros              | 1.1e12   | ~6.4 h        |

Hashfinder is ~300x slower: 8 zeros would take ~8 hours. The permutation space for K=6 is 18! ~ 6.4e15 per
value set, so it does not run out. Different `-S` seeds give more value sets.

## GPU mode (CUDA)

`gpufinder.cu` runs the same search on an NVIDIA GPU. It is built for `sm_120` (RTX 5060 Ti) and needs
CUDA 12.8 and Visual Studio 2022 for cl.exe. The CUDA binary does not need zlib. `fastfinder --dump-job FILE --auto`
runs its usual self-test against real zlib, then writes a job file and exits. The job holds the constant
file prefix, SHA-256 midstate, tail layout, Huffman code per byte value, value set, Adler/CRC prefix state,
patterns and 16 CPU-computed test vectors.

The rate is about 3.85e9 candidates/s on an RTX 5060 Ti, up from about 1.2e9. Four things get it there:

- **One SHA-256 block per candidate.** `--auto` looks for the largest K <= `--k` and a random value set
  where the variable region, the 9 padding bytes and the 12 constant IEND bytes fit in the last 64-byte
  block (`V0 + nvar + 12 + 9 <= 64`). For the test skin K=6 never fits: the variable region starts 8-11 bytes
  into its block and is 40 bytes long. K=5 fits (`V0` 7-9, 34-35 bytes), usually within a few value sets.
  If nothing fits, it falls back to the best 2-block configuration (about 2.1e9/s).
- **Cached decoding.** Candidate index = `outer * 24 + t`. One thread owns one `outer`. It decodes the first
  m-4 positions of the permutation once (32-bit math, its thread id carry-added to the launch's base digits),
  emits their Huffman codes into a word image and keeps the bit accumulator, CRC-32 state and Adler-b sum. It
  then loops over the 24 orderings `t` of the last 4 positions and re-emits only those four codes.
- **Constant tail.** Every permutation has the same bit length, so the end of the permuted region, the fixed
  tail codes, the EOB, Adler-a and the template bytes after it form one precomputed image. Only the pending
  bits, the two Adler-b bytes and the CRC bytes are patched in. CRC-32 advances a 32-bit word at a time. The
  CRC of the whole constant tail is two small linear table lookups (6 KB of shared memory).
- **No per-candidate global memory traffic** except the rare hit list. The best-zeros counter is updated
  once per thread.

```
powershell -File build.ps1                 # builds hashfinder, fastfinder, gpufinder (-NoGpu to skip CUDA)
py skinhash.py search skins/me.png --gpu -z 9
py skinhash.py search skins/me.png --gpu -p c0ffee -n 3
py skinhash.py search skins/me.png --gpu -b          # 10 s benchmark, prints RATE n, saves nothing

# manual:
fastfinder.exe found/me/base.raw 6 found/me --dump-job job.bin --auto -p dead
gpufinder.exe job.bin found/me [-p HEX] [-s HEX] [-z N] [-S SEED] [-n N] [-b]
```

gpufinder takes -p/-s/-z/-n/-S/-b like the other finders. Patterns given to it replace those in the job.
It prints the same `HIT ... file=` and progress lines, and `skinhash.py` still verifies every hit.

Safety:

- Before searching, the GPU must reproduce all test vectors of the job bit for bit. Otherwise it exits with code 5.
- Every hit is rebuilt on the CPU (independent code path) and re-hashed with a CPU SHA-256 before it is
  reported or written.
- `--gpu` falls back to CPU fastfinder, with a message, if gpufinder.exe is missing or fails.

Measured on an RTX 5060 Ti: about 3.85e9 candidates/s (K=5, 1 SHA block), against ~45–50M/s for fastfinder
on 12 CPU threads, roughly 80x. Expected times for `-z N` at 3.85e9/s:

| leading zeros | tries   | expected time |
|---------------|---------|---------------|
| 8             | 4.3e9   | ~1.1 s        |
| 9             | 6.9e10  | ~18 s         |
| 10            | 1.1e12  | ~4.8 min      |
| 11            | 1.8e13  | ~75 min       |
| 12            | 2.8e14  | ~20 h         |

**Value-set rollover.** With K=5 one value set holds 15! = 1.3e12 orderings, about 5.6 min of GPU time.
When gpufinder has covered all of them it prints `ROLLOVER tried=N secs=S` and exits with code 7.
`skinhash.py` then dumps a new job with a new random value set (`-S` is not reused) and restarts gpufinder
with `--tried-offset` / `--time-offset`, so the `tried=` counter, the rate and the `-b` benchmark stay
cumulative. Each new job goes through the full gate again. Value sets give disjoint candidates and are drawn
at random. A restart takes well under a second.

Limits: at most 4096 hits are kept per kernel launch (use a stricter pattern if you see the warning), and
`-t` is accepted and ignored. The extra gpufinder options for rollover and benchmarking are
`--tried-offset N`, `--time-offset S` and `--bench-secs S`.

## NameMC mode

NameMC's skin id (`namemc.com/skin/<16 hex>`) is a different hash from Mojang's texture id:

```
first 8 bytes (16 hex) of SHA-256( uint32_be(width) || uint32_be(height) || RGBA pixels, row-major,
                                   with R=G=B set to 0 wherever alpha == 0 )
```

Checked on about 1000 skins and on the skins in this repo. Because RGB under alpha 0 is zeroed, the Mojang
knob does nothing here. The NameMC knob is a pixel with alpha 1 and a free RGB.

```
py skinhash.py namemc-id skins/me.png                    # NameMC id (and Mojang id) of files
py skinhash.py namemc skins/me.png -p beef               # id starts with beef
py skinhash.py namemc skins/me.png --gpu -p 0000 -n 3    # prefixes may start with 0 (fixed 16 digits)
py skinhash.py namemc skins/me.png --gpu -s cafe --all   # also -z N, -t, -S, -b, --unused-only, --out DIR
```

Options as for `search` (`-p/-s/-z/-n/-t/-S/-b`, `--gpu`, `--all`, `--out`, default `found/<skin>_namemc/`).
`-z N` means N leading zero nibbles of the 16-digit id. `--gpu` falls back to the CPU if namemcgpu fails.
Every hit is checked by `skinhash.py`: the recomputed NameMC id equals the filename, only knob pixels changed
(alpha 0 to 1) plus the watermark, and all knob pixels are overlay or unused. The hit line also prints the
Mojang texture link of the file. Files are written in Mojang's PNG encoding.

Knob pixels (3 pixels = 9 RGB bytes, a 64-bit counter):

- Default: overlay or unused pixels that are fully transparent get alpha 1. The latest eligible pixels
  before the Mojang fast-mode tail are used: (55-57, 63) on the test skin. Base-layer pixels and pixels
  with alpha > 0 are never touched. At 1/255 opacity they should be invisible, but this needs an in-game
  check; it is not verified.
- `--unused-only`: only pixels no model face samples. Never drawn at all, so certainly invisible. On the
  test skin they are (61-63, 51), which is 50 SHA blocks per candidate instead of 2.

Finders: `namemcfinder.c` (CPU, zlib only for writing the hit PNG) and `namemcgpu.cu` (CUDA). Both resume
SHA-256 from a midstate over the fixed blocks. The GPU must reproduce 96 CPU test vectors before searching
(exit 5 otherwise), every hit is re-hashed on the CPU, and the PNG is written by `namemcfinder --render`.

Measured (12 threads, RTX 5060 Ti), candidates per second:

| mode           | CPU     | GPU     |
|----------------|---------|---------|
| default knob   | ~135M   | ~2.8G   |
| `--unused-only`| ~6.8M   | ~148M   |

A 16^n prefix takes 16^n / rate: 8 hex digits ~1.5 s, 10 digits ~6.5 min, 12 digits ~28 h on the GPU.

Combining with the Mojang id: NameMC ignores RGB under alpha 0 and the Mojang search only changes RGB under
alpha 0, so run the NameMC search first, then `search` on its output. The alpha-1 pixels are not knobs for
`search`, and the NameMC id of the result is the same. Limit: the extra bytes next to the tail usually stop
the GPU Mojang search from finding a one-block layout, so it runs at ~2.1G/s instead of ~3.9G/s.

```
py skinhash.py namemc skins/me.png --gpu -p beef -n 1
py skinhash.py search found/me_namemc/<id>.png --gpu -p cafe -n 1
```

Unverified: whether NameMC shows or accepts ids with leading zeros as 16 digits. The tool treats the id as
fixed width and matches prefixes on that.

## Watermark

Every search stamps an invisible signature so skins made with this tool can be recognised. It is a run of
horizontally adjacent, fully transparent pixels in an unused part of the layout that no model face samples.
Their RGB bytes, read left to right, spell the payload.

- **v2 (current):** `sokripon skinhash` + byte `0x02` = 18 bytes = 6 pixels, normally (0,0)-(5,0), in the
  unused top-left corner of the head layout. RGBA: `736f6b00 72697000 6f6e2000 736b6900 6e686100 73680200`.
- **v1 (older runs):** `SKHAS` + `0x01` padded with a 0 byte = 2 pixels at (0,0)-(1,0). Still detected.
- **Visibility:** alpha is 0, so nothing shows in game or in image viewers. Mojang keeps RGB under alpha 0
  (confirmed by a real upload).
- **Fallback:** if the preferred spot is not free (alpha > 0), the first free unused transparent run in scan
  order is used. Detection scans the whole image.
- **Speed:** the watermark sits in the fixed part of the image, so it costs no search speed. The fast modes
  avoid its byte values.

Check a skin with `py skinhash.py detect FILE|ID|URL ...` (ids and URLs are downloaded from
textures.minecraft.net). Turn the watermark off with `--no-watermark`.
