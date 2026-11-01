# Optimization log

The search went from ~26 thousand to ~4.8 billion candidates per second, about 185,000× faster, on one PC
(Ryzen 5 5600X, RTX 5060 Ti). Every step keeps the output byte-identical to what Mojang produces, and each
was checked against real zlib before its numbers were trusted. The ideas behind the steps are in
[how-it-works.md](how-it-works.md).

## Steps

| # | Step | Rate | Gain vs previous |
|---|---|---|---|
| 1 | Generic C search, one thread | 26 k/s | (baseline) |
| 1 | Generic C search, 12 threads | ~150–170 k/s | ×6 |
| 2 | CPU fast mode (permutations + SHA midstate) | ~43–51 M/s | ×300 |
| 3 | First CUDA version | 1.16 G/s | ×25 |
| 4a | GPU: 5 pixels so every candidate fits one SHA block | 1.93 G/s | ×1.7 |
| 4b | GPU: new enumeration, cached work, constant tail | 3.58 G/s | ×1.85 |
| 4c | GPU: block size / occupancy tuning | ~3.85 G/s | ×1.07 |
| 5 | Mojang search on skins that had no one-block layout (Dinnerbone, Grumm, every NameMC hit): length tuning | 2.1 → ~3.7–4.1 G/s on those skins | ×1.8 on them |
| 6 | NameMC GPU kernel: constant rounds and schedule hoisted out of the 1024-candidate loop | 2.8 → ~3.1 G/s | ×1.12 |
| 7 | Mojang GPU kernel: fixed bit positions (equal code lengths), table-driven tail, first SHA rounds and schedule terms once per thread | ~3.9 → ~4.8 G/s | ×1.23 |

For 10 leading zeros (≈1.1 trillion tries on average) that is ~75 days with step 1, ~7 hours with step 2 and
~4 minutes now.

All rates come from each finder's `-b` 10-second benchmark on a typical 64×64 player skin.

## Step 0: what gets hashed

- Mojang re-encodes uploads. The served file is signature + IHDR (64×64, 8-bit RGBA) + exactly one IDAT +
  IEND. The IDAT is zlib level 4 (wbits 15, memLevel 8, default strategy) over the raw rows, each prefixed
  with filter byte 0.
- The texture id is the SHA-256 of that file, with leading zeros dropped in the URL.
- RGB under alpha-0 pixels survives the re-encode. That is the knob the search varies.

This was reproduced byte for byte on real skins and confirmed by predicting the id of a test upload exactly.
Every later step must produce the same bytes as stock zlib. That rules out libdeflate and zlib-ng, which
compress correctly but produce different bytes.

## Step 1: generic C search, ~150–170 k/s

`hashfinder.c`. For every candidate: write a counter into invisible pixels, compress the image with real zlib,
build the PNG, hash it.

| Technique | Why it helps |
|---|---|
| Put the changing pixels as late in the image as possible | deflate processes data in order, so everything before the first changed byte is the same for every candidate. |
| Compress the fixed prefix once, then clone the zlib state (`deflateCopy`) | A candidate only feeds the tail to zlib. The output equals one-shot compression, checked against Python's zlib. |
| Arena allocator for the cloned state | `deflateCopy` allocates ~256 KB per call. A bump allocator reset per candidate avoids malloc/free. |
| SHA-256 with SHA-NI | Hardware SHA, picked at run time with a generic fallback. |
| One worker per hardware thread, interleaved counters | 12 threads on the 5600X. |

| Variant | Rate |
|---|---|
| 1 thread | 26 k/s |
| 12 threads, changing pixels in row 51 (only unused pixels) | 105 k/s |
| 12 threads, changing pixels in row 63 (the last row) | 170 k/s |

Moving the changing pixels from row 51 to row 63 gave ×1.6, because zlib re-processes 3.3 KB less per candidate.

Why it stops here: every candidate still runs zlib's matching and Huffman coding over the tail, rebuilds the
whole block, and hashes the full ~3.5 KB file. A SHA-256 midstate doesn't help, because the IDAT length near
the start of the file changes with every candidate.

## Step 2: CPU fast mode, ~43–51 M/s (×300)

`fastfinder.c`. Take compression out of the loop instead of speeding it up.

1. The changing pixels are the last K pixels of the image (K=6 on the CPU). Their RGB bytes are filled with
   values that appear nowhere else in the image. zlib can't match against a value it has never seen, so every
   byte from the first changed pixel to the end is a literal.
2. Each candidate is a permutation of that value set. Same values mean the same symbol frequencies, so the
   Huffman tables, the compressed size, the IDAT length and every bit before the tail are identical.
3. Only the last ~50 bytes of the file change: the tail literal codes, the Adler-32 and the CRC-32. The tail
   is built by concatenating known Huffman codes, and both checksums are updated incrementally from saved
   state.
4. SHA-256 resumes from a midstate covering the ~55 fixed 64-byte blocks. A candidate hashes only the last 2
   blocks.

Supporting pieces:

- A small deflate parser reads the reference zlib stream once. It extracts the Huffman codes and the exact
  bit offset where the tail starts, and confirms one final block with an all-literal tail.
- Heap's algorithm steps through permutations with one swap per candidate, starting from random shuffles.
- A start-up self-test compares 1000 random permutations with real zlib output. Every hit is rebuilt with
  real zlib before it's reported. If an assumption fails the program exits with code 4 and `skinhash.py`
  falls back to step 1.

Result: 45–51 M/s on 12 threads, about 300× step 1.

## Step 3: first CUDA version, 1.16 G/s (×25)

`gpufinder.cu`. The fast-mode loop is plain 32-bit integer work (bit packing, Adler-32, a table-based CRC-32,
2 SHA-256 blocks), which suits a GPU.

- The CPU writes a job file (`fastfinder --dump-job`): fixed prefix bytes, SHA midstate, Huffman codes,
  checksum state, patterns and 16 test vectors. zlib never runs on the GPU and isn't needed in the CUDA build.
- Each GPU thread turns a 64-bit index into a permutation with a Lehmer-code decode, builds the tail, the
  checksums and 2 SHA blocks, and checks the patterns. Hits go into a small buffer through an atomic counter.
- The GPU must reproduce all 16 CPU test vectors before searching, or it exits with code 5. The host re-hashes
  every hit on the CPU.

The first measurement was 1.24 G/s. A later one gave 1.16 G/s.

## Step 4: GPU tuning, 3.85 G/s (×3.3)

### 4a. One SHA block per candidate: 1.16 → 1.93 G/s

With K=6 the test skin's file is 3580 bytes, and 3580 mod 64 = 60. The changing bytes plus the 9 bytes of SHA
padding spill into a second block. That block is constant but is still processed for every candidate.

With K=5 the variable region (~34 bytes, starting 7–9 bytes into its block) and the padding fit in one block,
which halves the SHA work. Over 400 random value sets K=6 never fit for this skin.

The job builder picks K automatically (`fastfinder --dump-job --auto`). It tries K from the requested value
downwards, with up to 400 random value sets each, and takes the first configuration with one block. If none
fits, it uses the best 2-block configuration.

### 4b. New enumeration and per-thread caching: 1.93 → 3.58 G/s

| Change | Why |
|---|---|
| No 64-bit divisions. Candidate index = `outer × 24 + t`; each thread owns one `outer`, decodes it once with 32-bit maths, then loops over the 24 orderings of the last 4 positions | The old decode did up to 18 64-bit divisions per candidate, and 64-bit division is very slow on GPUs. |
| Constant tail image. The total bit length is the same for every permutation, so everything after the shuffled region is precomputed; only the pending bits, the two Adler-b bytes and the CRC bytes are patched in | Removes most per-candidate byte assembly. |
| Word-wise CRC-32, linear Adler-b update, CRC table built on the host | Less work per candidate, no per-block table rebuild. |
| The global best-zeros counter is touched once per thread instead of read per candidate | Removes a global memory read from the hot loop. |

### 4c. Launch tuning: 3.58 → ~3.85 G/s

256 threads per block, at least 3 blocks per SM, CRC lookup tables in shared memory, and small guards that
skip unused words in the finishing code.

### Value-set rollover

K=5 means 15 values, so one value set has 15! ≈ 1.3 × 10¹² orderings, about 5.6 minutes of GPU time.
gpufinder covers each set once, prints `ROLLOVER` and exits with code 7. `skinhash.py` builds a fresh job with
a new random value set, which passes the correctness check again, and continues with cumulative counters.
A rollover costs under a second.

### Tried, no gain

| Experiment | Result |
|---|---|
| CRC tables precomputed on the host, on their own | No measurable change |
| Inner loop over the last 5 positions (120 orderings) instead of 4 | 3.7 G/s, slower than 4 |
| New kernel with the 2-block configuration (K=6) | 2.1 G/s: one block is worth ×1.8 |

### Ceiling and what's left

- (At the time of step 4) A kernel variant that does only the SHA-256 block reached about 4.46 G/s, so the full
  search was within about 15% of the hashing ceiling. Step 7 goes past that figure by removing SHA work as well.
- (Done in step 7: the rounds that depend only on the fixed first bytes of the block run once per thread, with
  one kernel per layout, and the tail is table-driven.)
- Nsight Compute counters are blocked on this machine (`ERR_NVGPUCTRPERM`), so all tuning was by experiment.

## Step 5: length tuning for 2-block skins, 2.1 → ~3.7–4.1 G/s on them

Step 4a needs the variable region (tail literals, Adler-32, CRC-32) plus the 9 SHA padding bytes and the 12
IEND bytes to fit in the last 64-byte block: `V0 + nvar + 12 + 9 <= 64`, where V0 is the offset of the tail
inside its block. V0 is `(compressed length before the tail) mod 64`, so for some skins it is simply a bad
number: Dinnerbone and Grumm (V0 around 27-36), and every skin that already carries NameMC alpha-1 pixels,
which is the Mojang phase of `both`. No K and no value set helps, the search drops to two blocks per candidate
and 2.1 G/s.

The fix is to move V0. Changing the RGB of an alpha-0 pixel earlier in the image changes how zlib codes that
region and shifts the compressed length by a few bytes (about 6 per changed pixel). `fastfinder --auto`
(given the allowed pixels with `--tune-pixels`) now does this when no one-block layout with K >= 5 exists:

1. It draws random RGB for 1, then 2, ... up to 6 tuning pixels (24 tries per pixel count, at most 4000 tries).
2. A cheap screen builds one K=5 job and drops the try when V0 is above 24. Otherwise 12 random value sets
   are probed at K=6, then K=5, for a one-block layout. One try costs one zlib compress and one deflate parse,
   about 1 ms.
3. The tuned image is written back (`--tune-out`). `skinhash.py` replaces its raw image with it, so value-set
   rollovers and the CPU fallback keep the same tuning. The value set is always chosen after tuning, from the
   byte values absent in the tuned image, so tuning bytes can never collide with it.

Tuning pixels are alpha 0, unused (preferred) or transparent overlay, and never RESERVED, watermark, in the
last row (Mojang tail, NameMC knobs) or base layer. They are RGB-only changes under alpha 0, so `verify`
accepts them, the NameMC id (which ignores RGB under alpha 0) stays the same, and `verify` additionally
rejects any change at a RESERVED pixel. `py skinhash.py pixels` lists the candidates and the CLI prints the
pixels it changed. `--no-length-tuning` turns it off. The browser version (`web/js/job.js`, `tuneSearch`) does
the same.

Measured over 8 seeds per skin: Dinnerbone 8/8 within 51-79 tries (3-4 pixels, K=5 or 6), Grumm 8/8 within 1-25
tries (1-2 pixels), the obus test skin 8/8 within 49-58 tries (3 pixels), six random NameMC-hit variants of
Steve 48/48 within 54-91 tries (3-4 pixels, nearly always K=6). A whole tuning pass takes 0.3-0.6 s, most of it
the failed 400-set searches that run first. K=6 fits in most tuned cases, which also means 18! value sets and
no rollover.

| Case (10 s `bench`) | Before | After |
|---|---|---|
| steve.png (already one block) | 3.97 G/s | 3.85-3.95 G/s (no tuning needed, noise only) |
| NameMC hit made from steve.png | 2.15 G/s | 3.94-4.07 G/s |
| Dinnerbone | 2.13 G/s | 3.67-4.06 G/s (K=5 tuned layouts with V0=0 ran a little slower) |

## Step 6: NameMC kernel, hoisting constant work, 2.8 → ~3.1 G/s

The NameMC default layout hashes two blocks per candidate. In the first block only message word 11 changes per
candidate (the 1024 counters of one thread differ only in the last knob pixel); words 9 and 10 change per
thread, and everything else is a job constant. The old kernel recomputed all 128 rounds and the whole message
schedule for every candidate.

| Hoisted | Where |
|---|---|
| State after rounds 0-8 (constant words only) | Host, uploaded as `c_s9` |
| Rounds 9-10 | Once per thread |
| Schedule words that do not depend on word 11 (16, 17, 19, 21, 23) | Once per thread |
| For every other schedule word, the sum of its terms that do not depend on word 11 (`pw`, 19 values) | Once per thread |
| Rounds 11-63, the dependent schedule words, the constant second block | Per candidate (117 of 128 rounds, ~12% fewer operations) |

Result with the same correctness gate (96 CPU test vectors, exit 5 on mismatch): 2.78 → 3.2 G/s in direct A/B
runs of `namemcgpu -b` (the `skinhash.py bench` run on steve.png gave 3.0-3.15 G/s). Block size and occupancy
variants (128/256/512 threads, 2-4 blocks per SM) were all within noise of each other.

The WGSL kernel (`web/js/nmgpu.js`, `fastParts`) generates the same split, with the constant parts folded in
JavaScript (WGSL rejects overflowing constant expressions) and the same text used by the search loop and the
gate. Chrome A/B on the same page: 2.24 → 2.41 G/s (+7%).

## Step 7: Mojang kernel, fixed bit positions, 3.9 → ~4.8 G/s

A SASS review of the step 4-6 kernel showed 1948 instructions per candidate, of which about 300 were the tail
(permutation codes → bit image, CRC-32, Adler-b): a branchy `EMIT` chain with a local-memory image (800 bytes of
stack per thread) and a per-thread code table that was indexed dynamically.

**Observation.** The permuted values occur once each in the image, so zlib usually gives them all the same Huffman
code length (10-11 bits on the skins we looked at). With equal lengths every code sits at a bit position that does
not depend on the permutation; only which value sits there does. Everything per candidate is then linear:

| Quantity | Per candidate |
|---|---|
| SHA message words WI..WI+3 (the last four codes) | OR of four 16-byte shared-memory table entries `[position][value]` (entries are pre-shifted to their bit position) |
| CRC-32 of the variable region | XOR of four CRC deltas (CRC is linear over GF(2)) and two lookups for the Adler-b bytes. The thread's own part (running state plus pending bits pushed through the rest of the region) is computed once per thread |
| Adler-b | sum of four `weight × value` table terms, one `% 65521` |
| Adler-b and CRC bytes in the message | a few shifts, three words ORed in at the fixed word `KB` |
| Permutation of the 4 inner values | one `PRMT` with a per-ordering selector, then four byte extracts |

The tail went from ~300 to ~40 instructions and the local arrays are gone (96 bytes of stack, no spills).

**Constant message words.** Message words below `WI` depend on the thread only (5-6 words for V0 around 9), words
above the Adler/CRC bytes are job constants, and only `WI..WI+3` and `KB..KB+2` vary per candidate. The kernel is a
template on `(WI, KB)` and classifies all 64 schedule words at compile time (constant / thread / candidate):

- rounds 0..WI-1 run once per thread;
- thread-only schedule words are computed once per thread;
- for the other words the sum of the terms that do not depend on the candidate is computed once per thread
  (`pwv`, up to 16 values), so a candidate only adds its own terms;
- rounds that use constant words take `K + w` from a precomputed constant (`c_kw`).

**Delivery: template instantiations, generic kernel as fallback, not NVRTC.** All one-block layouts
(`V0 + nvar + 21 <= 64`) give only 24 distinct `(WI, KB)` pairs, so 24 × 2 kernels (search, gate) are compiled in:
the build takes about one extra minute and the exe grows by about 1 MB. NVRTC would specialise even more (job
constants as literals), but needs `nvrtc64_*.dll` and `nvrtc-builtins64_*.dll` next to the exe (they ship with the
toolkit, not the driver), the driver API to load the module, and a second copy of the kernel as source text. The
templates need none of that and degrade gracefully: the generic kernel is used for 2-block jobs, for value sets
with unequal code lengths, for a layout that is not in the table, and with `--generic` / `GPUFINDER_GENERIC=1`
(for A/B runs).

**Job builder.** `fastfinder --auto` now prefers a one-block value set with equal code lengths. If it finds none it
takes the best one-block set with unequal lengths (generic kernel) and never goes below K=5 for this purpose
(a smaller K would roll over many times per second). The job format and the index → permutation mapping did not
change.

How often equal lengths occur (100 random value sets per case, K=6 / K=5):

| Skin | Equal-length sets |
|---|---|
| steve | 100 / 85 |
| Dinnerbone (tuned) | 87 / 100 |
| obus test skin | 100 / 70 |
| scorpion3013 | 67 / 100 |
| Grumm, untuned image | 0 / 0 (it is a 2-block image anyway; its tuned images are fine) |

Whether the singletons share one depth is a property of the image's Huffman shape. With `--auto` (400 value sets per
K, length tuning included) a one-block equal-length set was found in all ~380 job builds we ran (5 skins × 25-40
tuning seeds, 150 perturbed Steve variants at K=4-6). A rollover builds its next value set the same way.

**Measured** (10 s `skinhash.py bench --engine gpu`, Mojang phase, 3 alternating runs each; "before" is the same
binary with `GPUFINDER_GENERIC=1`, i.e. the old kernel on the same kind of job):

| Case | Before (G/s) | After (G/s) |
|---|---|---|
| steve.png | 3.91, 4.03, 3.90 | 4.88, 4.94, 4.87 |
| NameMC hit made from steve.png (length tuned) | 4.11, 4.03, 3.85 | 4.87, 4.83, 4.89 |
| Dinnerbone (local skin, length tuned) | 3.92, 3.89, 3.93 | 4.70, 4.76, 4.71 |
| `gpufinder -b` on one job, old exe vs new exe | 3.84, 3.43, 4.00 | 4.72, 4.65, 4.84 |

The spread (3.4-4.0 for the old kernel on the same job) is GPU clock and temperature, so compare the alternating
pairs. The NameMC rate (~3.1 G/s) is unchanged.

| SASS of the search kernel | Generic (before) | Fast (after) |
|---|---|---|
| Instructions in the candidate loop | 1948 | 1430 (SHF 611, LOP3 347, IADD3 338, ~40 other) |
| Registers | 55 | 61 |
| Stack frame / spills | 800 B / 0 | 96 B / 0 |
| Shared memory per block | 7.7 KB | 9.6 KB |

At ~1430 instructions per candidate, 36 SMs × 64 integer lanes × ~3.0 GHz allow about 4.8-5.0 G/s, so the kernel
is now at the cost of the plain SHA-256 rounds. What is left is micro-work: treating the unused fourth table word and
third Adler/CRC word as constants (≈10 instructions) and folding round WI into the thread-level state (≈10).

Block size and occupancy (`MINB` 2/3/4: 4.65-4.79 G/s) are within noise, so 3 stays.

**WGSL port.** The browser page has the same kernel (`web/js/wgslfast.js`). WGSL has no templates, so the shader
text is generated per job in JS with `(WI, KB)`, the shift of the Adler/CRC bytes, the code length and all job
constants (mid-state, constant message words, `K + w` sums, CRC constants) written in as literals, with the constant
parts of the schedule folded in JS (WGSL rejects overflowing constant expressions). The outer positions are unrolled,
so the bit position, the word flushes and the CRC steps are static and the image needs no dynamically indexed
array. The per-(position, value) tables live in workgroup memory, the k-th free value is found with a branch-free
popcount search instead of a loop, and the 24 orderings stay a runtime loop with a per-ordering shift word. It is
used for one-block jobs with equal code lengths (the job builder prefers them as `--auto` does); every other job,
and any job whose fast kernel fails the gate, runs the generic kernel. In Chrome on the RTX 5060 Ti the Mojang rate
went from 2.9 to about 3.7 G/s (1.31x on steve.png, 1.29x on a NameMC hit made from steve, 1.30x on Dinnerbone,
three alternating runs each), about 78% of the native 4.8. The browser gate for the fast kernel uses 128 vectors
(24 against full pako, 104 from the byte-level builder).

### Gate for the fast kernel

The gate of every job and rollover now covers the fast kernel itself and uses more vectors: the 16 CPU vectors
plus 112 more (random indices and both ends of the range) checked against the independent byte-oriented host builder
in gpufinder, which is itself compared with the CPU vectors. If the fast kernel disagrees on any vector, gpufinder
says so, switches to the generic kernel, repeats the gate, and exits with code 5 only if that fails too. Hits are
re-hashed on the CPU as before.

### Cost of the contains target

`--contains HEX` (a hex string anywhere in the id) is checked on every candidate that no prefix, suffix or zeros
target already took. Per nibble offset the kernels build a 32-bit window of the digest with one funnel shift, mask and
compare it with the first 8 pattern digits (one LOP3), and fold the 64 results with `min`; the other 8 digits and the
range test (offset at or after the leading zeros, so a match inside the dropped zeros does not count) only run when
a window matched. That is about 3 instructions per offset, ~190 per pattern on top of ~1500 for the candidate, and
the kernels are compiled without it (a template flag; WGSL: no generated code) when no contains pattern is given, so
the other targets run exactly as fast as before (0.99-1.01 against the build before the change, alternating runs).
Measured with alternating runs while another search shared the GPU, as the ratio to the same build without
`--contains` (the absolute rates were about half of the idle 4.8 / 3.1 G/s):

| Kernel | 1 pattern | 4 patterns |
|---|---|---|
| Mojang fast, CUDA | 0.87 | 0.65 |
| NameMC fast, CUDA (16 offsets) | 0.975 | 0.90 |
| Mojang fast, WGSL in Chrome | 0.89 | 0.69 |

The cost does not depend on the pattern length. Tried and dropped: a byte-aligned prefilter (a zero-byte test of the
digest xor the two byte values that a pattern of 3+ digits must contain at an even or an odd offset, 3 instructions per
word and value). About a quarter of the digests pass it, so almost every warp still runs the full check and the net
result was slower (0.84-0.85 against 0.87). Replacing the per-offset branch by the `min` fold made no measurable
difference (0.86-0.87 either way).

## Rejected ideas

| Idea | Why not |
|---|---|
| Faster zlib replacements (libdeflate, zlib-ng) | They produce different bytes, so the hash wouldn't match what Mojang serves. |
| zlib on the GPU | It would have to match stock zlib level 4 bit for bit. Fast mode makes it unnecessary. |
| SHA-256 midstate in the generic search | The IDAT length near the start of the file changes with the compressed size. |
| Changing any alpha-0 pixel | Minecraft forces the base layer opaque, so hidden colours there would show. Only transparent overlay and unused pixels are safe. |
| Pattern matching on the full 64-digit hash | The link drops leading zeros. `-p` matches the trimmed id, so `-p c0ffee` also catches `000c0ffee…`. (`-c` does the same: the kernels skip matches inside the dropped zeros, and the host re-checks the trimmed id.) |

## Correctness checks

No speed-up counted until it reproduced real zlib's bytes.

| Step | Check |
|---|---|
| 1 | Hits compared against Python's one-shot zlib |
| 2 | 1000-permutation self-test at start-up; every hit rebuilt with real zlib |
| 3–4 | GPU must match 16 CPU test vectors exactly, at start-up and after every rollover; every hit re-hashed on the CPU |
| 5 | The tuned image goes through the same 1000-permutation self-test and test vectors; `verify` rejects any change at a RESERVED pixel |
| 6 | The NameMC kernel must still reproduce 96 CPU test vectors; the WGSL gate runs the hoisted code itself |
| 7 | The fast kernel passes the gate (16 CPU vectors + 112 host-reference vectors) for every job; on a mismatch gpufinder falls back to the generic kernel and gates that |
| All | `skinhash.py` re-verifies every hit: the hash matches the filename, Python's zlib reproduces the identical bytes, and no visible pixel changed |

Real uploads confirmed the whole pipeline: Mojang assigned exactly the predicted ids.
