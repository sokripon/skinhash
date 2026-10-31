# Optimization log

The search went from ~26 thousand to ~3.85 billion candidates per second, about 150,000× faster, on one PC
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

For 10 leading zeros (≈1.1 trillion tries on average) that is ~75 days with step 1, ~7 hours with step 2 and
~5 minutes now.

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

- A kernel variant that does only the SHA-256 block reaches about 4.46 G/s, so the full search is within about
  15% of the hashing ceiling.
- Together about 5–10% is left: precomputing the SHA rounds that depend only on the fixed first bytes of the
  block (needs one kernel per byte offset), and compiling the kernel at run time for each job (NVRTC) so job
  constants become literals.
- Nsight Compute counters are blocked on this machine (`ERR_NVGPUCTRPERM`), so all tuning was by experiment.

## Rejected ideas

| Idea | Why not |
|---|---|
| Faster zlib replacements (libdeflate, zlib-ng) | They produce different bytes, so the hash wouldn't match what Mojang serves. |
| zlib on the GPU | It would have to match stock zlib level 4 bit for bit. Fast mode makes it unnecessary. |
| SHA-256 midstate in the generic search | The IDAT length near the start of the file changes with the compressed size. |
| Changing any alpha-0 pixel | Minecraft forces the base layer opaque, so hidden colours there would show. Only transparent overlay and unused pixels are safe. |
| Pattern matching on the full 64-digit hash | The link drops leading zeros. `-p` matches the trimmed id, so `-p c0ffee` also catches `000c0ffee…`. |

## Correctness checks

No speed-up counted until it reproduced real zlib's bytes.

| Step | Check |
|---|---|
| 1 | Hits compared against Python's one-shot zlib |
| 2 | 1000-permutation self-test at start-up; every hit rebuilt with real zlib |
| 3–4 | GPU must match 16 CPU test vectors exactly, at start-up and after every rollover; every hit re-hashed on the CPU |
| All | `skinhash.py` re-verifies every hit: the hash matches the filename, Python's zlib reproduces the identical bytes, and no visible pixel changed |

Real uploads confirmed the whole pipeline: Mojang assigned exactly the predicted ids, including the 11-zero one.
