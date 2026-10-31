# skinhashmaker

Finds a "cool looking" Minecraft skin texture id by brute-forcing pixels that are invisible in game.

**Try it in your browser: https://sokripon.github.io/skinhash/** (WebGPU, runs locally, nothing is uploaded)

[How it works](docs/how-it-works.md) · [Optimization log](docs/optimizations.md) · [Browser version](web/README.md) · [Skin areas map](docs/skin-areas.png)

It finds Mojang texture links (`textures.minecraft.net/texture/…`) and NameMC skin pages (`namemc.com/skin/…`), or both on one skin. It runs natively (CPU, CUDA) or in the browser (WebGPU). The example skin is classic Steve, `skins/steve.png`.

The texture id is the SHA-256 of the PNG Mojang serves, with leading zero digits dropped in the URL
(a zero-padded URL 404s). Mojang re-encodes uploads as signature + IHDR + exactly one IDAT
(zlib level 4, wbits 15, memLevel 8, filter 0 on every row) + IEND. RGB under alpha-0 pixels survives
that re-encode, so those bytes are free knobs.

## Usage

Build the finders once (`powershell -File build.ps1`, `-NoGpu` to skip CUDA), then:

```
py skinhash.py mojang skins/me.png -p c0ffee        # Mojang link starting with c0ffee
py skinhash.py namemc skins/me.png -p beef          # NameMC page starting with beef
py skinhash.py both   skins/me.png --namemc-prefix beef --mojang-prefix cafe
py skinhash.py ids    skins/me.png                  # both ids and links of a skin
py skinhash.py mojang jeb_ -p c0ffee                # a username or UUID works as the skin too
py skinhash.py verify skins/me.png found/me/<id>.png
```

`py skinhash.py --help` lists the commands, `py skinhash.py COMMAND --help` shows the options and examples of one.

| Command | What it does |
|---|---|
| `mojang SKIN` | search for a Mojang texture link |
| `namemc SKIN` | search for a NameMC skin page |
| `both SKIN` | NameMC search, then Mojang search on its hit |
| `ids SKIN...` | Mojang texture id and NameMC id of skins |
| `verify ORIGINAL RESULT` | check a hit against the skin it came from |
| `detect SKIN...` | is the skin watermarked by skinhash |
| `pixels SKIN` | which pixels each mode would change |
| `bench` | 10 s benchmark of the engines |

### Skin input

Every argument that takes a skin (`SKIN`, `ORIGINAL`, the `ids` and `detect` inputs) accepts, in this order:

1. a PNG path (an existing file always wins),
2. an explicit `name:`, `uuid:` or `texture:` prefix, which forces that meaning,
3. a `textures.minecraft.net` URL (anything with a `/`),
4. a UUID (32 hex digits or dashed),
5. a texture id (33 to 64 hex digits),
6. a username (3-16 of `A-Za-z0-9_`).

A short hex word such as `cafe` is therefore a username; write `texture:<id>` for a texture id under 33 digits.
A name or UUID is resolved with the official Mojang APIs (`api.mojang.com`, falling back to
`api.minecraftservices.com`, then `sessionserver.mojang.com`). The skin is downloaded from
`textures.minecraft.net` and its SHA-256 must equal the texture id. It is saved as `skins/<name>.png` (a texture id
as `skins/<id16>.png`; `steve.png` is never overwritten) and one line is printed: name, UUID, classic/slim, texture
id, saved path. Searches then write to `found/<name>/` as usual. Lookups are cached for 10 minutes in
`skins/.lookups.json` because Mojang limits profile requests to about one per minute per UUID.

```
py skinhash.py mojang jeb_ -p c0ffee
py skinhash.py ids 853c80ef-3c37-49fd-aa49-938b674adae6
py skinhash.py ids uuid:853c80ef3c3749fdaa49938b674adae6
```

Errors are one line: unknown name or UUID, a player without a custom skin, a legacy 64x32 skin (the tool needs
64x64; `jeb_` has one), a network error, or "rate-limited, try again in a minute" (HTTP 429).

Options shared by the search commands (`mojang`, `namemc`, `both`):

| Option | Meaning |
|---|---|
| `-p, --prefix HEX` | id starts with HEX. Repeatable. |
| `-s, --suffix HEX` | id ends with HEX. Repeatable. |
| `-z, --zeros N` | at least N leading zero digits. Repeatable. |
| `-n, --count N` | stop after N verified hits (default 1) |
| `--every-target` | run until every target has a hit; the first hit per target is kept, `-n` is ignored |
| `--engine auto\|gpu\|cpu\|generic` | auto (default) uses the GPU if its exe is built and passes its self-check, else CPU fast mode, else the generic search. `generic` is Mojang only. |
| `--no-generic` | never fall back to the slow generic search (Mojang) |
| `--unused-only` | only pixels no model face samples; incompatible with the Mojang fast modes |
| `--tail-pixels N` | pixels permuted by Mojang fast mode (default 6) |
| `--no-watermark` | skip the invisible watermark |
| `-o, --out DIR` | output directory |
| `--threads N`, `--seed N` | CPU threads; start seed |

Prefixes are hex, at most 16 digits. Targets of one command are alternatives: a hit matches any of them.
Every hit is re-verified by `skinhash.py` (hash equals file name, Python zlib re-encodes the file identically,
no visible pixel changed) and printed as `HIT <why> sha=<full hex> id=<id> counter=<n> file=<path>`.
Ctrl+C stops everything and never starts a fallback. Unknown options are an error; nothing is passed on to the
finders unchecked.

### mojang

```
py skinhash.py mojang skins/me.png -p c0ffee            # id (trimmed) starts with c0ffee
py skinhash.py mojang skins/me.png -z 8                 # >= 8 leading zeros (shortest links)
py skinhash.py mojang skins/me.png -s dead -n 3         # ends with dead, stop after 3 hits
py skinhash.py mojang skins/me.png -z 10 --no-generic   # never use the slow search
```

Output goes to `found/<skin>/` by default. Mojang drops leading zeros from the link, so `-p` matches the
trimmed id (`-p c0ffee` matches `c0ffee...` and `000c0ffee...`) and may not start with `0`: use `-z`.

Engines: `auto` runs `gpufinder` if it is built and passes its self-check, else `fastfinder` on the CPU, else
`hashfinder` (generic, about 300x slower). One line printed at the start says which engine runs and why. Fast
mode needs the last K pixels (row 63) to be invisible; `pixels` tells you whether they are. `--engine gpu` and
`--engine cpu` prefer that engine and fall back as described. `--unused-only` rules out the fast modes, so it
combines with `auto` (generic search) but not with `--engine gpu|cpu` or `--no-generic`. On the GPU
`--tail-pixels` is an upper bound: the job builder takes the largest K <= N that gives a one-block SHA.

### namemc

```
py skinhash.py namemc skins/me.png -p beef                      # id starts with beef
py skinhash.py namemc skins/me.png -p 0000 -n 3                 # prefixes may start with 0 (fixed 16 digits)
py skinhash.py namemc skins/me.png -s cafe --every-target --engine gpu
```

Output goes to `found/<skin>_namemc/`. `--engine auto` uses `namemcgpu` if built, else `namemcfinder`.
`-z N` means N leading zero digits of the 16-digit id. `--engine generic`, `--no-generic` and `--tail-pixels`
do not exist here. See "NameMC mode" below.

### collide

Find a skin whose NameMC id equals an id NameMC already has, or two skins of yours that share one. NVIDIA GPU only
(`namemccollide`, built by `build.ps1`).

```
py skinhash.py collide skins/me.png --ids namemc_skin_ids.u64le.bin   # match any id of a sorted uint64-LE list
py skinhash.py collide skins/me.png --self                            # two own skins with the same id
```

The knobs are the last pixels of the file (alpha 1, RGB = counter; transparent overlay pixels, invisible). List mode
uses (62-63, 63), so one try is a single SHA-256 block from a midstate: about 3.2 G tries/s on an RTX 5060 Ti, with a
32 MB bitmap prefilter and an exact bucket lookup in the 51M-id list (expected ~3.6e11 tries, about 2 minutes).
`--self` uses (61-63, 63) and Pollard rho with distinguished points (~2^32 steps, seconds, only the trail ends are
stored). Every hit is re-hashed on the CPU, written as a PNG, decoded back and re-hashed in Python.
Output goes to `found/<skin>_collide/`.

### both

```
py skinhash.py both skins/me.png --namemc-prefix beef --mojang-prefix cafe
py skinhash.py both skins/me.png --namemc-zeros 6 --mojang-zeros 9 --every-target
py skinhash.py both skins/me.png --mojang-prefix c0ffee        # only one group: only that phase
```

| Option | Meaning |
|---|---|
| `--namemc-prefix/-suffix/-zeros` | NameMC targets (phase 1), same rules as `namemc` |
| `--mojang-prefix/-suffix/-zeros` | Mojang targets (phase 2), same rules as `mojang` |
| `-n`, `--every-target`, `--engine`, `--no-generic`, `--unused-only`, `--tail-pixels`, `--no-watermark`, `-o`, `--threads`, `--seed` | as above; `-n` applies to phase 2 |

Phase 1 runs a NameMC search up to its first verified hit (with `--every-target`, across the NameMC targets, and
the first hit is the base). Phase 2 runs a Mojang search on that hit. A summary prints both links, the final
file is verified against the original skin and both ids are checked against their targets. Output goes to
`found/<skin>_both/`. This mirrors the Both mode of the web page. `--engine generic` only affects phase 2.

### ids

```
py skinhash.py ids skins/me.png
py skinhash.py ids https://textures.minecraft.net/texture/<id>
```

Prints the Mojang texture id (trimmed, as in the URL), the NameMC id and both links. Texture ids and URLs are
downloaded from textures.minecraft.net.

### verify

```
py skinhash.py verify skins/me.png found/me/<id>.png
```

Prints `OK <link>` or `FAIL <reason>`; the exit code is 0 or 1. It also accepts NameMC and Both results (alpha 0
to 1 on overlay or unused pixels counts as a NameMC knob).

### detect

```
py skinhash.py detect found/me/<id>.png <texture id> https://textures.minecraft.net/texture/<id>
```

Reports whether each skin carries the watermark (see "Watermark").

### pixels

```
py skinhash.py pixels skins/me.png
```

Shows the pixels each mode would change on this skin: Mojang fast GPU and CPU tail, generic, generic with
`--unused-only`, NameMC default and with `--unused-only`, and the watermark spot. It says whether fast mode is
possible and, if not, which pixel stops it. `--tail-pixels N` changes the tail size that is checked.

### bench

```
py skinhash.py bench                  # GPU (if built) and CPU, Mojang and NameMC
py skinhash.py bench --engine gpu
py skinhash.py bench skins/me.png --engine cpu --threads 8
```

Runs the 10 s benchmark of each chosen engine on a skin (default `skins/steve.png`) and prints candidates per
second. Nothing is saved.

### Old names

These still work and print a one-line note on stderr: `search` (now `mojang`), `predict` and `namemc-id` (now
`ids`; they keep their old one-line output), `knobs` (now `pixels`), `--gpu` (`--engine gpu`), `--generic`
(`--engine generic`), `--fast` (`--no-generic`), `--all` (`--every-target`), `--k` (`--tail-pixels`).
`--out` is the long form of `-o`.

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

The rate is about 4.8e9 candidates/s on an RTX 5060 Ti, up from about 1.2e9. Five things get it there:

- **One SHA-256 block per candidate.** `--auto` looks for the largest K <= `--tail-pixels` and a random value set
  where the variable region, the 9 padding bytes and the 12 constant IEND bytes fit in the last 64-byte
  block (`V0 + nvar + 12 + 9 <= 64`). For the example skin, and the other skins we tried, K=6 did not fit: the variable region starts
  8-11 bytes into its block and is 40 bytes long. K=5 fits (`V0` 7-9, 34-35 bytes), usually within a few value sets.
  If nothing fits, it first tries length tuning: it gives 1-6 invisible alpha-0 pixels earlier in the image random
  RGB values until a one-block layout exists (typically 50-100 tries of ~1 ms), and only then falls back to the
  best 2-block configuration (about 2.1e9/s). `--no-length-tuning` turns this off.
- **Cached decoding.** Candidate index = `outer * 24 + t`. One thread owns one `outer`. It decodes the first
  m-4 positions of the permutation once (32-bit math, its thread id carry-added to the launch's base digits),
  emits their Huffman codes into a word image and keeps the bit accumulator, CRC-32 state and Adler-b sum. It
  then loops over the 24 orderings `t` of the last 4 positions and re-emits only those four codes.
- **Constant tail.** Every permutation has the same bit length, so the end of the permuted region, the fixed
  tail codes, the EOB, Adler-a and the template bytes after it form one precomputed image. Only the pending
  bits, the two Adler-b bytes and the CRC bytes are patched in. CRC-32 advances a 32-bit word at a time. The
  CRC of the whole constant tail is two small linear table lookups (6 KB of shared memory).
- **Fixed bit positions.** `--auto` prefers value sets whose permuted values all have the same Huffman code length
  (usual for these skins: the values occur once each). Then every code sits at a bit position that does not depend
  on the permutation, and a candidate's message words, CRC-32 and Adler-b are the OR / XOR / sum of per-(position, value)
  table entries kept in shared memory. The first SHA-256 rounds, which only see words that depend on the thread, run once
  per thread. This kernel is compiled per layout (`WI`, `KB`: the message word of the last four codes and of the Adler
  bytes; 24 combinations cover every one-block layout). Value sets with unequal lengths, 2-block jobs and a failed
  self-check of the fast kernel use the older generic kernel (~3.9e9/s).
- **No per-candidate global memory traffic** except the rare hit list. The best-zeros counter is updated
  once per thread.

```
powershell -File build.ps1                 # builds hashfinder, fastfinder, gpufinder (-NoGpu to skip CUDA)
py skinhash.py mojang skins/me.png --engine gpu -z 9
py skinhash.py mojang skins/me.png --engine gpu -p c0ffee -n 3
py skinhash.py bench --engine gpu                    # 10 s benchmark, saves nothing

# manual:
fastfinder.exe found/me/base.raw 6 found/me --dump-job job.bin --auto -p dead
gpufinder.exe job.bin found/me [-p HEX] [-s HEX] [-z N] [-S SEED] [-n N] [-b] [--generic]
```

gpufinder takes -p/-s/-z/-n/-S/-b like the other finders. Patterns given to it replace those in the job.
It prints the same `HIT ... file=` and progress lines, and `skinhash.py` still verifies every hit.

Safety:

- Before searching, the GPU must reproduce all test vectors of the job bit for bit (the 16 CPU vectors plus 112 more
  checked against gpufinder's own byte-oriented builder), with the kernel it is going to use. If only the fast kernel
  fails, gpufinder switches to the generic one and gates that; if that fails too it exits with code 5.
- Every hit is rebuilt on the CPU (independent code path) and re-hashed with a CPU SHA-256 before it is
  reported or written.
- `--engine gpu` (and `auto`, when gpufinder.exe is built) falls back to CPU fastfinder, with a message, if gpufinder fails its gate.

Measured on an RTX 5060 Ti: about 4.8e9 candidates/s (K=5, 1 SHA block), against ~45–50M/s for fastfinder
on 12 CPU threads, roughly 100x. Expected times for `-z N` at 4.8e9/s:

| leading zeros | tries   | expected time |
|---------------|---------|---------------|
| 8             | 4.3e9   | ~0.9 s        |
| 9             | 6.9e10  | ~14 s         |
| 10            | 1.1e12  | ~3.8 min      |
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
py skinhash.py ids skins/me.png                                  # NameMC id (and Mojang id) of files
py skinhash.py namemc skins/me.png -p beef                       # id starts with beef
py skinhash.py namemc skins/me.png --engine gpu -p 0000 -n 3     # prefixes may start with 0 (fixed 16 digits)
py skinhash.py namemc skins/me.png --engine gpu -s cafe --every-target
```

Options as for `mojang` (see Usage), default output `found/<skin>_namemc/`.
`-z N` means N leading zero nibbles of the 16-digit id. The GPU engine falls back to the CPU if namemcgpu fails.
Every hit is checked by `skinhash.py`: the recomputed NameMC id equals the filename, only knob pixels changed
(alpha 0 to 1) plus the watermark, and all knob pixels are overlay or unused. The hit line also prints the
Mojang texture link of the file. Files are written in Mojang's PNG encoding.

Knob pixels (3 pixels = 9 RGB bytes, a 64-bit counter):

- Default: overlay or unused pixels that are fully transparent get alpha 1. The latest eligible pixels
  before the Mojang fast-mode tail are used: (55-57, 63) on the example skin. Base-layer pixels and pixels
  with alpha > 0 are never touched. At 1/255 opacity they should be invisible, but this needs an in-game
  check; it is not verified.
- `--unused-only`: only pixels no model face samples. Never drawn at all, so certainly invisible. On the
  example skin they are (61-63, 51), which is 50 SHA blocks per candidate instead of 2.

Finders: `namemcfinder.c` (CPU, zlib only for writing the hit PNG) and `namemcgpu.cu` (CUDA). Both resume
SHA-256 from a midstate over the fixed blocks. The GPU must reproduce 96 CPU test vectors before searching
(exit 5 otherwise), every hit is re-hashed on the CPU, and the PNG is written by `namemcfinder --render`.

Measured (12 threads, RTX 5060 Ti), candidates per second:

| mode           | CPU     | GPU     |
|----------------|---------|---------|
| default knob   | ~135M   | ~3.1G   |
| `--unused-only`| ~6.8M   | ~148M   |

A 16^n prefix takes 16^n / rate: 8 hex digits ~1.5 s, 10 digits ~6.5 min, 12 digits ~28 h on the GPU.

Combining with the Mojang id: NameMC ignores RGB under alpha 0 and the Mojang search only changes RGB under
alpha 0, so run the NameMC search first, then `mojang` on its output (`both` does this in one command). The alpha-1 pixels are not knobs for
`mojang`, and the NameMC id of the result is the same. Limit: the extra bytes next to the tail usually stop
the GPU Mojang search from finding a one-block layout. Length tuning fixes that by changing the RGB of a few
other invisible (alpha 0) pixels, so the Mojang phase still runs at ~4.8G/s instead of ~2.1G/s.

```
py skinhash.py both skins/me.png --namemc-prefix beef --mojang-prefix cafe
# the same in two steps:
py skinhash.py namemc skins/me.png -p beef
py skinhash.py mojang found/me_namemc/<id>.png -p cafe
```

NameMC shows the id as a fixed 16 digits, leading zeros included (checked with `namemc.com/skin/00000000001d91b1`).
The tool matches prefixes on those 16 digits, so `-z N` gives a NameMC link starting with N zeros.

## Browser version (WebGPU)

`web/` is a static page with the same search: Mojang texture links, NameMC skin pages, or Both in one click
(NameMC phase, then the Mojang phase on that result). It runs on the visitor's GPU, and nothing is uploaded.

    py web/serve.py            # then open http://127.0.0.1:8000/

- Starts with classic Steve, fetched from textures.minecraft.net and checked against its texture id. Drop
  your own skin to replace it.
- Uses pako for the zlib-exact encoding (392 test cases byte-identical to Python's zlib) and WGSL kernels
  ported from the CUDA finders.
- Same safety checks: a GPU gate against CPU test vectors before every search, and a CPU check of every hit
  (ids, and only invisible pixels changed) before it's shown.
- About 2.9G/s for Mojang links and 2.4G/s for NameMC on an RTX 5060 Ti in Chrome (~75-80% of native).

Details and tests: [web/README.md](web/README.md).

## Watermark

Every search stamps an invisible signature so skins made with this tool can be recognised. It is a run of
horizontally adjacent, fully transparent pixels in an unused part of the layout that no model face samples.
Their RGB bytes, read left to right, spell the payload.

- **v3 (current):** `sokripon skinhash` + byte `0x03` = 18 bytes = 6 pixels, normally (0,7)-(5,7), in the
  unused corner of the head layout. RGBA: `736f6b00 72697000 6f6e2000 736b6900 6e686100 73680300`.
- **v2 (older runs):** the same with byte `0x02`, at (0,0)-(5,0). Moved because Mine Little Pony reads
  (0,0)-(3,0). Still detected.
- **v1 (older runs):** `SKHAS` + `0x01` padded with a 0 byte = 2 pixels at (0,0)-(1,0). Still detected.
- **Visibility:** alpha is 0, so nothing shows in game or in image viewers. Mojang keeps RGB under alpha 0
  (confirmed by a real upload).
- **Fallback:** if the preferred spot is not free (alpha > 0), the first free unused transparent run that is
  not reserved (below) in scan order is used. Detection scans the whole image.
- **Speed:** the watermark sits in the fixed part of the image, so it costs no search speed. The fast modes
  avoid its byte values.

### Reserved pixels

Some mods read pixels as data, so no search pixel (Mojang or NameMC knob, fast-mode tail) and no watermark
pixel is ever taken from this set, in any mode (`RESERVED` in `skinhash.py` and `web/js/model.js`):

- Ears: marker block x0-3, y32-35 and emissive palette x52-55, y32-35
- Mine Little Pony triggers: (0,0) (1,0) (2,0) (3,0) (0,1) (1,1) (2,1) (2,2); CPM's header is the blue of (0,0)
- Tails mod: x56-61, y16-18
- Vanilla deadmau5 ear box: x24-37, y0-6

`docs/skin-areas.png` shows them. Fast mode is refused if one of its last K pixels were reserved (today's
(58-63, 63) are not). Check with `py test_reserved.py`.

Check a skin with `py skinhash.py detect FILE|ID|URL ...` (ids and URLs are downloaded from
textures.minecraft.net). Turn the watermark off with `--no-watermark`.
