/*
 * gpufinder: CUDA backend of fastfinder.
 *
 * Reads a job file written by `fastfinder RAW K OUTDIR ... --dump-job job.bin [--auto]`, which holds the
 * constant PNG prefix, the SHA-256 midstate, the Huffman code of each byte value, Adler/CRC prefix state,
 * the patterns and CPU-computed test vectors.
 *
 * Search structure (see fastfinder.c "GPU job dump" for the index <-> permutation mapping):
 *   candidate index = outer * R! + t     (R = 4, t < 24)
 *   - one GPU thread owns one `outer` value: it decodes the first m-R permutation positions once (32-bit math,
 *     carry-adding its thread id onto the launch base digits), emits their Huffman codes into a word image,
 *     and keeps bit-accumulator / CRC-32 / Adler state;
 *   - then it loops over the 24 orderings of the last R positions, re-emitting only those R codes;
 *     the tail (fixed codes, EOB, Adler-a, template) is a precomputed constant image because the total bit
 *     length of every permutation is the same;
 *   - CRC-32 is advanced word by word as 32-bit words complete, the Adler-b sum is linear in the values;
 *   - SHA-256 resumes from the midstate (1 block when the job says nblk == 1).
 *
 * usage: gpufinder job.bin OUTDIR [-p HEX]... [-s HEX]... [-z N] [-S SEED] [-n N] [-b]
 *                  [--tried-offset N] [--time-offset S] [--bench-secs S]
 *   -p/-s/-z given here replace those stored in the job.
 * exit codes: 0 ok, 1 usage/job error, 5 correctness gate failed, 6 CUDA error / no device,
 *             7 permutation space of this value set exhausted (prints "ROLLOVER tried=.. secs=.." on stdout;
 *               the caller should dump a new job and continue with --tried-offset/--time-offset)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define MAXK 20
#define MAXN 60
#define MAXBLK 8
#define MAXPAT 64
#define MAXHITS 4096
#ifndef RIN
#define RIN 4                 /* inner (cached) permutation positions; must equal fastfinder GPU_R */
#endif
#define NPERM (RIN == 3 ? 6 : RIN == 4 ? 24 : RIN == 5 ? 120 : 720)
#define MAXKW 6               /* words of post-permutation image */
#ifndef BLOCK
#define BLOCK 256
#endif
#ifndef MINB
#define MINB 3
#endif

typedef unsigned long long u64;
typedef uint32_t u32;
typedef uint8_t u8;

/* ------------------------------------------------------------------ job */
struct Pat { u64 val, mask; u32 suffix; char text[24]; };
struct TestVec { u64 idx; u8 sha[32]; };
struct Job {
    u32 K, n, m, R, flen, B0, V0, nvar, nblk, rem, eob, crc0, adl_a, adl_b;
    u64 firstbits;
    u32 mid[8], tab[257];
    u8 alpha[MAXK], vals[MAXN];
    std::vector<u8> tmpl, ref;
    u32 zeros;
    std::vector<Pat> pats;
    std::vector<TestVec> tests;
};

static bool load_job(const char *path, Job &J) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    bool ok = true;
    auto r = [&](void *p, size_t n) { if (fread(p, 1, n, f) != n) ok = false; };
    u32 magic = 0; r(&magic, 4);
    if (!ok || magic != 0x324A4B53u) { fclose(f); return false; }
    r(&J.K, 4); r(&J.n, 4); r(&J.m, 4); r(&J.R, 4); r(&J.flen, 4); r(&J.B0, 4); r(&J.V0, 4); r(&J.nvar, 4);
    r(&J.nblk, 4); r(&J.rem, 4); r(&J.eob, 4); r(&J.crc0, 4); r(&J.adl_a, 4); r(&J.adl_b, 4);
    r(&J.firstbits, 8); r(J.mid, 32); r(J.tab, 257 * 4); r(J.alpha, MAXK); r(J.vals, MAXN);
    if (!ok || J.K < 1 || J.K > MAXK || J.n != 3 * J.K || J.nblk < 1 || J.nblk > MAXBLK) { fclose(f); return false; }
    u32 l = 0; r(&l, 4);
    if (!ok || l != J.nblk * 64) { fclose(f); return false; }
    J.tmpl.resize(l); r(J.tmpl.data(), l);
    r(&l, 4);
    if (!ok || l != J.flen || l > (1u << 20)) { fclose(f); return false; }
    J.ref.resize(l); r(J.ref.data(), l);
    u32 np = 0; r(&J.zeros, 4); r(&np, 4);
    if (!ok || np > MAXPAT) { fclose(f); return false; }
    for (u32 i = 0; i < np; i++) {
        Pat p; memset(&p, 0, sizeof p);
        r(&p.val, 8); r(&p.mask, 8); r(&p.suffix, 4); r(p.text, 20);
        J.pats.push_back(p);
    }
    u32 nt = 0; r(&nt, 4);
    for (u32 i = 0; ok && i < nt && i < 1000; i++) { TestVec t; r(&t.idx, 8); r(t.sha, 32); J.tests.push_back(t); }
    fclose(f);
    return ok;
}

static bool parse_pat(const char *hex, int suffix, Pat &p) {
    size_t n = strlen(hex);
    if (n == 0 || n > 16) return false;
    if (!suffix && hex[0] == '0') return false;   /* prefixes apply to the TRIMMED id */
    memset(&p, 0, sizeof p);
    p.suffix = suffix;
    for (size_t i = 0; i < n; i++) {
        char c = hex[i];
        int v = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
                (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (v < 0) return false;
        int shift = suffix ? (int)(4 * (n - 1 - i)) : (int)(60 - 4 * i);
        p.val |= (u64)v << shift;
        p.mask |= (u64)0xF << shift;
        p.text[i] = (char)(hex[i] | 0x20);
    }
    return true;
}

/* ------------------------------------------------------------------ device */
struct DevParams {
    u32 m, nblk, W0w, nb0, acc0, crc_w0, nwL, nbL, KW, boff, Lc, b_const;
    u32 zeros, npats, has_suffix, c0;
    u32 mid[8];
};
struct BaseDigits { u8 d[24]; };

__constant__ DevParams c_p;
__constant__ u32 c_tmpl[MAXBLK * 16];   /* template words, big-endian (as SHA-256 reads them) */
__constant__ u32 c_post[MAXKW];         /* constant image after the permuted region, little-endian words from word nwL */
__constant__ u32 c_tab[64];             /* code (reversed | len<<16) of value index j */
__constant__ u32 c_val[64];             /* value bytes */
__constant__ u32 c_wt[64];              /* Adler-b weight of position p */
__constant__ u32 c_ac[64];              /* alpha code emitted after position p (0 if none) */
__constant__ u32 c_al[64];              /* its length */
__constant__ u32 c_crc[256];
__constant__ u32 c_gt[6 * 256];          /* leaf CRC tables: 4 for the running state, 2 for the Adler-b bytes */
__constant__ u8 c_perm[NPERM * RIN];
__constant__ u64 c_pval[MAXPAT], c_pmask[MAXPAT];
__constant__ u32 c_psuf[MAXPAT];
__constant__ u32 c_K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ u32 d_best_zeros;
__device__ u32 d_nhits;
__device__ u64 d_hits[MAXHITS];

__device__ __forceinline__ u32 ror(u32 x, int n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ u32 bswap(u32 x) { return __byte_perm(x, 0, 0x0123); }

/* one SHA-256 block over big-endian words w[0..15] */
__device__ __forceinline__ void sha_block(u32 st[8], const u32 *img) {
    u32 w[64];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = img[i];
#pragma unroll
    for (int i = 16; i < 64; i++) {
        u32 s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3);
        u32 s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + c_K256[i] + w[i];
        u32 t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

/* CRC-32 register update with one little-endian 32-bit word (4 bytes) */
#define CRCW(c, w)                                                              \
    do {                                                                        \
        (c) ^= (w);                                                             \
        (c) = s_crc[(c) & 255] ^ ((c) >> 8);                                    \
        (c) = s_crc[(c) & 255] ^ ((c) >> 8);                                    \
        (c) = s_crc[(c) & 255] ^ ((c) >> 8);                                    \
        (c) = s_crc[(c) & 255] ^ ((c) >> 8);                                    \
    } while (0)

/* append a code of clen bits; flush a 32-bit word into the SHA image when complete */
#define EMIT(acc, nb, nw, crc, comb, clen)                                      \
    do {                                                                        \
        (acc) |= (u64)(comb) << (nb); (nb) += (clen);                           \
        if ((nb) >= 32) {                                                       \
            u32 w_ = (u32)(acc);                                                \
            img[nw] = bswap(w_);                                                \
            CRCW(crc, w_);                                                      \
            (nw)++; (acc) >>= 32; (nb) -= 32;                                   \
        }                                                                       \
    } while (0)

/*
 * One thread = one `outer` value (all 24 orderings of the last RIN positions).
 * TEST: evaluate only ordering t = tlo and write the 8 digest words to out.
 * otherwise: test every candidate against the patterns and record hits.
 */
template <bool TEST>
__device__ __forceinline__ void run_outer(const BaseDigits &bd, u32 gid, u32 tlo, u32 thi, u64 obase,
                                          const u32 *s_crc, const u32 *s_gt, const u32 *s_tab, const u32 *s_val, u32 *out) {
    u32 img[MAXBLK * 16];
    const u32 m = c_p.m, nblk = c_p.nblk;
#pragma unroll 1
    for (u32 i = 0; i < nblk * 16; i++) img[i] = c_tmpl[i];

    u64 acc = c_p.acc0;
    u32 nb = c_p.nb0, nw = c_p.W0w, crc = c_p.crc_w0, bacc = 0, used = 0, v = gid;
    const u32 mo = m - RIN;
#pragma unroll 1
    for (u32 p = 0; p < mo; p++) {
        u32 r = m - p, t = (u32)bd.d[p] + v, q = t / r, dg = t - q * r;
        v = q;
        u32 vi = __fns(~used, 0, dg + 1);
        used |= 1u << vi;
        u32 e = s_tab[vi], len = e >> 16;
        u32 comb = (e & 0xFFFFu) | (c_ac[p] << len), clen = len + c_al[p];
        bacc += c_wt[p] * s_val[vi];
        EMIT(acc, nb, nw, crc, comb, clen);
    }

    /* the RIN remaining values (ascending index) and their per-position code / weight table */
    uint4 E[RIN * RIN];
    {
        const u32 freem = ~used & ((1u << m) - 1u);
#pragma unroll
        for (int pi = 0; pi < RIN; pi++) {
            u32 vi = __fns(freem, 0, pi + 1);
            u32 e = s_tab[vi], len = e >> 16, bv = s_val[vi];
#pragma unroll
            for (int q = 0; q < RIN; q++) {
                u32 p = mo + q;
                E[q * RIN + pi] = make_uint4((e & 0xFFFFu) | (c_ac[p] << len), len + c_al[p], c_wt[p] * bv, 0);
            }
        }
    }

    u32 zbest = 0;
#pragma unroll 1
    for (u32 t = tlo; t < thi; t++) {
        u64 a2 = acc; u32 nb2 = nb, nw2 = nw, crc2 = crc, b2 = bacc;
#pragma unroll
        for (int q = 0; q < RIN; q++) {
            u32 pi = c_perm[t * RIN + q];
            uint4 e = E[q * RIN + pi];
            b2 += e.z;
            EMIT(a2, nb2, nw2, crc2, e.x, e.y);
        }
        /* nb2 == nbL and nw2 == nwL by construction (same multiset of codes for every ordering) */
        const u32 nwL = c_p.nwL, KW = c_p.KW;
        u32 vv[MAXKW];
#pragma unroll
        for (int k = 0; k < MAXKW; k++) vv[k] = 0;
#pragma unroll
        for (int k = 0; k < MAXKW; k++) { if ((u32)k >= KW) break; vv[k] = c_post[k]; }
        vv[0] |= (u32)a2;
        const u32 b = (c_p.b_const + b2) % 65521u;
        {
            u32 bbe = ((b >> 8) | (b << 8)) & 0xFFFFu;
            u64 x = (u64)bbe << ((c_p.boff & 3) * 8);
            u32 kb = c_p.boff >> 2;
#pragma unroll
            for (int k = 0; k < MAXKW; k++) {
                if ((u32)k >= KW) break;
                vv[k] |= ((u32)k == kb ? (u32)x : 0u) | ((u32)k == kb + 1 ? (u32)(x >> 32) : 0u);
            }
        }
        /* CRC-32 of the post-permutation bytes is affine in (running state ^ pending bits) and in the two b bytes */
        u32 x = crc2 ^ (u32)a2;
        u32 c = s_gt[x & 255] ^ s_gt[256 + ((x >> 8) & 255)] ^ s_gt[512 + ((x >> 16) & 255)] ^ s_gt[768 + (x >> 24)] ^ c_p.c0 ^
                s_gt[1024 + (b >> 8)] ^ s_gt[1280 + (b & 255)];
        c = ~c;
        {
            u32 off = c_p.boff + 4;
            u64 x = (u64)bswap(c) << ((off & 3) * 8);
            u32 kc = off >> 2;
#pragma unroll
            for (int k = 0; k < MAXKW; k++) {
                if ((u32)k >= KW) break;
                vv[k] |= ((u32)k == kc ? (u32)x : 0u) | ((u32)k == kc + 1 ? (u32)(x >> 32) : 0u);
            }
        }
#pragma unroll
        for (int k = 0; k < MAXKW; k++) {
            if ((u32)k >= KW) break;
            img[nwL + k] = bswap(vv[k]);
        }

        u32 s[8];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] = c_p.mid[i];
#pragma unroll 1
        for (u32 blk = 0; blk < nblk; blk++) sha_block(s, img + blk * 16);

        if (TEST) {
            for (int i = 0; i < 8; i++) out[i] = s[i];
            return;
        }
        u64 hi = (u64)s[0] << 32 | s[1];
        u32 zn = hi ? (u32)(__clzll(hi) >> 2) : 16u;
        zbest = max(zbest, zn);
        bool hit = c_p.zeros && zn >= c_p.zeros;
        if (c_p.npats) {
            u64 mi = (u64)s[2] << 32 | s[3];
            u64 lo = (u64)s[6] << 32 | s[7];
            u64 tp;
            if (hi) { int sh = __clzll(hi) & ~3; tp = sh ? (hi << sh) | (mi >> (64 - sh)) : hi; }
            else tp = mi ? mi << (__clzll(mi) & ~3) : 0;
            for (u32 i = 0; i < c_p.npats; i++) {
                u64 vv2 = c_psuf[i] ? lo : tp;
                if ((vv2 & c_pmask[i]) == c_pval[i]) { hit = true; break; }
            }
        }
        if (hit) {
            u32 pos = atomicAdd(&d_nhits, 1u);
            if (pos < MAXHITS) d_hits[pos] = (obase + gid) * NPERM + t;
        }
    }
    if (!TEST && zbest > 0 && zbest > *(volatile u32 *)&d_best_zeros) atomicMax(&d_best_zeros, zbest);
}

__device__ __forceinline__ void load_shared(u32 *s_crc, u32 *s_gt, u32 *s_tab, u32 *s_val) {
    for (u32 i = threadIdx.x; i < 256; i += blockDim.x) s_crc[i] = c_crc[i];
    for (u32 i = threadIdx.x; i < 6 * 256; i += blockDim.x) s_gt[i] = c_gt[i];
    for (u32 i = threadIdx.x; i < 64; i += blockDim.x) { s_tab[i] = c_tab[i]; s_val[i] = c_val[i]; }
    __syncthreads();
}

__global__ void __launch_bounds__(BLOCK, MINB) search_kernel(BaseDigits bd, u64 obase, u32 count) {
    __shared__ u32 s_crc[256], s_gt[6 * 256], s_tab[64], s_val[64];
    load_shared(s_crc, s_gt, s_tab, s_val);
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= count) return;
    run_outer<false>(bd, gid, 0, NPERM, obase, s_crc, s_gt, s_tab, s_val, nullptr);
}

/* gate: thread i evaluates ordering t[i] of outer = (digits bd[i]) + gid[i] */
__global__ void test_kernel(const BaseDigits *bds, const u32 *gids, const u32 *ts, u32 *out, int nt) {
    __shared__ u32 s_crc[256], s_gt[6 * 256], s_tab[64], s_val[64];
    load_shared(s_crc, s_gt, s_tab, s_val);
    if ((int)threadIdx.x < nt) {
        u32 s[8];
        BaseDigits bd = bds[threadIdx.x];
        run_outer<true>(bd, gids[threadIdx.x], ts[threadIdx.x], ts[threadIdx.x] + 1, 0, s_crc, s_gt, s_tab, s_val, s);
        for (int i = 0; i < 8; i++) out[threadIdx.x * 8 + i] = s[i];
    }
}

/* ------------------------------------------------------------------ host CPU reference (independent of the kernel) */
static const u32 K256h[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define HROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha256_blocks_h(u32 s[8], const u8 *p, size_t nb) {
    while (nb--) {
        u32 w[64], a, b, c, d, e, f, g, h;
        for (int i = 0; i < 16; i++) w[i] = (u32)p[4*i] << 24 | (u32)p[4*i+1] << 16 | (u32)p[4*i+2] << 8 | p[4*i+3];
        for (int i = 16; i < 64; i++) {
            u32 s0 = HROR(w[i-15], 7) ^ HROR(w[i-15], 18) ^ (w[i-15] >> 3);
            u32 s1 = HROR(w[i-2], 17) ^ HROR(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        a = s[0]; b = s[1]; c = s[2]; d = s[3]; e = s[4]; f = s[5]; g = s[6]; h = s[7];
        for (int i = 0; i < 64; i++) {
            u32 t1 = h + (HROR(e, 6) ^ HROR(e, 11) ^ HROR(e, 25)) + ((e & f) ^ (~e & g)) + K256h[i] + w[i];
            u32 t2 = (HROR(a, 2) ^ HROR(a, 13) ^ HROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
        p += 64;
    }
}
static void sha256_h(const u8 *msg, size_t len, u8 out[32]) {
    u32 s[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t full = len / 64;
    sha256_blocks_h(s, msg, full);
    u8 tail[128] = {0};
    size_t rem = len - full * 64;
    memcpy(tail, msg + full * 64, rem);
    tail[rem] = 0x80;
    size_t tl = rem < 56 ? 64 : 128;
    u64 bits = (u64)len * 8;
    for (int i = 0; i < 8; i++) tail[tl - 1 - i] = (u8)(bits >> (8 * i));
    sha256_blocks_h(s, tail, tl / 64);
    for (int i = 0; i < 8; i++) { out[4*i] = s[i] >> 24; out[4*i+1] = s[i] >> 16; out[4*i+2] = s[i] >> 8; out[4*i+3] = s[i]; }
}

static u32 crc_tab_h[256];
static void init_crc_h() {
    for (u32 i = 0; i < 256; i++) { u32 c = i; for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; crc_tab_h[i] = c; }
}

static u64 fact_h(int r) { u64 f = 1; for (int i = 2; i <= r; i++) f *= (u64)i; return f; }

/* index -> permutation of the n values (same mapping as fastfinder decode_perm; written independently) */
static std::vector<u8> host_perm(const Job &J, u64 idx) {
    u32 m = J.m, R = J.R;
    u64 RF = fact_h((int)R), o = idx / RF, t = idx % RF;
    std::vector<u8> perm(J.n), rem(J.vals, J.vals + m);
    for (u32 i = 0; i < m - R; i++) {
        u32 r = m - i, d = (u32)(o % r);
        o /= r;
        perm[i] = rem[d];
        rem.erase(rem.begin() + d);
    }
    std::vector<int> order;
    for (u32 i = 0; i < R; i++) order.push_back((int)i);
    for (u32 q = 0; q < R; q++) {
        u64 f = fact_h((int)(R - 1 - q));
        u32 d = (u32)(t / f);
        t %= f;
        perm[m - R + q] = rem[order[d]];
        order.erase(order.begin() + d);
    }
    for (u32 i = m; i < J.n; i++) perm[i] = J.vals[i];
    return perm;
}

/* build the file from the permutation index: ref prefix + tail region (byte-oriented, as fastfinder does) */
static std::vector<u8> host_file(const Job &J, u64 idx) {
    std::vector<u8> perm = host_perm(J, idx);
    std::vector<u8> buf(J.tmpl);
    buf.resize(J.nblk * 64 + 64);
    u64 acc = J.firstbits;
    int nb = (int)J.rem;
    u8 *o = &buf[J.V0];
    u32 a = J.adl_a, b = J.adl_b;
    auto put = [&](u32 e) { acc |= (u64)(e & 0xFFFF) << nb; nb += (int)(e >> 16); while (nb >= 8) { *o++ = (u8)acc; acc >>= 8; nb -= 8; } };
    for (u32 i = 0; i < J.K; i++) {
        for (int j = 0; j < 3; j++) { u8 v = perm[3 * i + j]; put(J.tab[v]); a += v; b += a; }
        put(J.tab[J.alpha[i]]); a += J.alpha[i]; b += a;
    }
    put(J.eob);
    if (nb > 0) *o++ = (u8)acc;
    a %= 65521; b %= 65521;
    *o++ = (u8)(b >> 8); *o++ = (u8)b; *o++ = (u8)(a >> 8); *o++ = (u8)a;
    u32 c = ~J.crc0;
    for (u8 *q = &buf[J.V0]; q < o; q++) c = crc_tab_h[(c ^ *q) & 255] ^ (c >> 8);
    c = ~c;
    *o++ = (u8)(c >> 24); *o++ = (u8)(c >> 16); *o++ = (u8)(c >> 8); *o++ = (u8)c;
    if ((u32)(o - &buf[J.V0]) != J.nvar) { fprintf(stderr, "host: variable region size %d != job nvar %u\n", (int)(o - &buf[J.V0]), J.nvar); exit(5); }
    std::vector<u8> file(J.ref.begin(), J.ref.begin() + J.B0);
    file.insert(file.end(), buf.begin(), buf.begin() + (J.flen - J.B0));
    return file;
}

static int clz64h(u64 x) { int n = 0; while (!(x >> 63)) { x <<= 1; n++; } return n; }

static const char *match_why(const Job &J, const u8 d[32], std::string &why) {
    u64 hi = 0, mi = 0, lo = 0;
    for (int i = 0; i < 8; i++) { hi = hi << 8 | d[i]; mi = mi << 8 | d[8 + i]; lo = lo << 8 | d[24 + i]; }
    u64 tp;
    if (hi) { int sh = clz64h(hi) & ~3; tp = sh ? (hi << sh) | (mi >> (64 - sh)) : hi; }
    else tp = mi ? mi << (clz64h(mi) & ~3) : 0;
    for (const Pat &p : J.pats) {
        u64 v = p.suffix ? lo : tp;
        if ((v & p.mask) == p.val) { why = std::string(p.suffix ? "suffix:" : "prefix:") + p.text; return why.c_str(); }
    }
    int zn = hi ? clz64h(hi) / 4 : 16;
    if (J.zeros && (u32)zn >= J.zeros) { why = "zeros:" + std::to_string(zn); return why.c_str(); }
    return nullptr;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static void hexify(const u8 d[32], char out[65]) {
    static const char *hx = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[2*i] = hx[d[i] >> 4]; out[2*i+1] = hx[d[i] & 15]; }
    out[64] = 0;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "gpufinder: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(6); } } while (0)

static u64 splitmix(u64 &s) {
    u64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ------------------------------------------------------------------ job -> device constants */
struct DevData {
    DevParams P;
    u32 tmpl[MAXBLK * 16], post[MAXKW], tab[64], val[64], wt[64], ac[64], al[64];
    u32 gt[6 * 256];
    u8 perm[NPERM * RIN];
};

static bool derive(const Job &J, DevData &D) {
    memset(&D, 0, sizeof D);
    const u32 n = J.n, K = J.K, m = J.m, N = n + K;
    if (J.R != RIN || m < RIN + 1 || m > 31) { fprintf(stderr, "gpufinder: unsupported job (R=%u m=%u)\n", J.R, m); return false; }
    DevParams &P = D.P;
    P.m = m; P.nblk = J.nblk;
    /* stream start: word aligned, the bytes before V0 inside the word are part of the initial accumulator */
    u32 W0 = J.V0 & ~3u, s0 = J.V0 - W0;
    P.W0w = W0 / 4;
    u32 acc0 = 0;
    for (u32 i = 0; i < s0; i++) acc0 |= (u32)J.tmpl[W0 + i] << (8 * i);
    acc0 |= (u32)J.firstbits << (8 * s0);
    P.acc0 = acc0; P.nb0 = 8 * s0 + J.rem;
    /* CRC register at byte B0+W0 of the file (IDAT type + data from file offset 41) */
    {
        u32 c = 0xFFFFFFFFu;
        const u8 idat[4] = {'I', 'D', 'A', 'T'};
        for (int i = 0; i < 4; i++) c = crc_tab_h[(c ^ idat[i]) & 255] ^ (c >> 8);
        for (u32 i = 41; i < J.B0 + W0; i++) c = crc_tab_h[(c ^ J.ref[i]) & 255] ^ (c >> 8);
        P.crc_w0 = c;
        for (u32 i = J.B0 + W0; i < J.B0 + J.V0; i++) c = crc_tab_h[(c ^ J.ref[i]) & 255] ^ (c >> 8);
        if (~c != J.crc0) { fprintf(stderr, "gpufinder: CRC prefix state mismatch\n"); return false; }
    }
    /* per-position tables */
    for (u32 j = 0; j < n; j++) { D.val[j] = J.vals[j]; D.tab[j] = J.tab[J.vals[j]]; D.wt[j] = N - (j + j / 3); }
    u64 bconst = (u64)J.adl_b + (u64)N * J.adl_a, asum = J.adl_a;
    for (u32 j = 0; j < K; j++) {
        bconst += (u64)(N - (4 * j + 3)) * J.alpha[j];
        asum += J.alpha[j];
    }
    for (u32 j = 0; j < n; j++) asum += J.vals[j];
    for (u32 p = 0; p < m; p++) if (p % 3 == 2) { D.ac[p] = J.tab[J.alpha[p / 3]] & 0xFFFF; D.al[p] = J.tab[J.alpha[p / 3]] >> 16; }
    for (u32 p = m; p < n; p++) bconst += (u64)D.wt[p] * J.vals[p];
    P.b_const = (u32)bconst;
    u32 a_fin = (u32)(asum % 65521);
    /* end of the permuted region (constant for every permutation) */
    u64 pos = 8ull * J.V0 + J.rem;
    for (u32 p = 0; p < m; p++) { pos += D.tab[p] >> 16; if (p % 3 == 2) pos += D.al[p]; }
    P.nwL = (u32)(pos / 32); P.nbL = (u32)(pos % 32);
    /* constant image of everything after the permuted region */
    std::vector<u8> Pb(J.nblk * 64 + 16, 0);
    auto putbits = [&](u32 code, u32 len) {
        for (u32 i = 0; i < len; i++, pos++) if ((code >> i) & 1) Pb[pos / 8] |= (u8)(1u << (pos % 8));
    };
    for (u32 p = m; p < n; p++) {
        putbits(D.tab[p] & 0xFFFF, D.tab[p] >> 16);
        if (p % 3 == 2) { u32 e = J.tab[J.alpha[p / 3]]; putbits(e & 0xFFFF, e >> 16); }
    }
    putbits(J.eob & 0xFFFF, J.eob >> 16);
    u32 Epad = (u32)((pos + 7) / 8);
    if (Epad + 8 != J.V0 + J.nvar) { fprintf(stderr, "gpufinder: variable region length mismatch (%u vs %u)\n", Epad + 8, J.V0 + J.nvar); return false; }
    Pb[Epad + 2] = (u8)(a_fin >> 8); Pb[Epad + 3] = (u8)a_fin;
    for (u32 i = Epad + 8; i < J.nblk * 64; i++) Pb[i] = J.tmpl[i];
    P.boff = Epad - 4 * P.nwL;
    P.Lc = P.boff + 4;
    P.KW = (Epad + 8 - 4 * P.nwL + 3) / 4;
    if (Epad < 4 * P.nwL || P.KW > MAXKW || P.nwL + P.KW > J.nblk * 16) { fprintf(stderr, "gpufinder: post image does not fit (KW=%u)\n", P.KW); return false; }
    for (u32 k = 0; k < P.KW; k++)
        D.post[k] = (u32)Pb[4 * (P.nwL + k)] | (u32)Pb[4 * (P.nwL + k) + 1] << 8 | (u32)Pb[4 * (P.nwL + k) + 2] << 16 | (u32)Pb[4 * (P.nwL + k) + 3] << 24;
    /* leaf CRC tables (CRC is linear): bytes [4*nwL, Epad+4) = pending bits | constant tail | b | a */
    {
        auto run = [&](u32 c, const std::vector<u8> &bytes) {
            for (u8 by : bytes) c = crc_tab_h[(c ^ by) & 255] ^ (c >> 8);
            return c;
        };
        std::vector<u8> zeros(P.Lc, 0), pm(Pb.begin() + 4 * P.nwL, Pb.begin() + 4 * P.nwL + P.Lc);
        for (u32 i = 0; i < 4; i++) for (u32 v = 0; v < 256; v++) D.gt[i * 256 + v] = run(v << (8 * i), zeros);
        P.c0 = run(0, pm);
        for (u32 j = 0; j < 2; j++) for (u32 v = 0; v < 256; v++) {
            std::vector<u8> e(P.Lc, 0); e[P.boff + j] = (u8)v;
            D.gt[(4 + j) * 256 + v] = run(0, e);
        }
    }
    for (u32 w = 0; w < J.nblk * 16; w++)
        D.tmpl[w] = (u32)J.tmpl[4 * w] << 24 | (u32)J.tmpl[4 * w + 1] << 16 | (u32)J.tmpl[4 * w + 2] << 8 | J.tmpl[4 * w + 3];
    memcpy(P.mid, J.mid, 32);
    P.zeros = J.zeros; P.npats = (u32)J.pats.size();
    P.has_suffix = 0;
    for (const Pat &p : J.pats) if (p.suffix) P.has_suffix = 1;
    /* orderings of the inner positions: lexicographic permutations of 0..RIN-1 */
    int pr[RIN];
    for (int i = 0; i < RIN; i++) pr[i] = i;
    int t = 0;
    do { for (int i = 0; i < RIN; i++) D.perm[t * RIN + i] = (u8)pr[i]; t++; } while (std::next_permutation(pr, pr + RIN));
    return t == NPERM;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: gpufinder job.bin OUTDIR [-p HEX]... [-s HEX]... [-z N] [-S SEED] [-n N] [-b]\n"); return 1; }
    Job J;
    if (!load_job(argv[1], J)) { fprintf(stderr, "gpufinder: cannot read job file %s\n", argv[1]); return 1; }
    std::string outdir = argv[2];
    long long max_hits = -1;
    int bench = 0;
    double bench_secs = 10, time_off = 0;
    u64 tried_off = 0;
    u64 seed = (u64)std::chrono::system_clock::now().time_since_epoch().count() * 0x9E3779B97F4A7C15ULL;
    std::vector<Pat> cli;
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : nullptr;
        Pat p;
        if ((!strcmp(a, "-p") || !strcmp(a, "-s")) && v) {
            if (!parse_pat(v, a[1] == 's', p) || cli.size() >= MAXPAT) { fprintf(stderr, "bad %s %s\n", a, v); return 1; }
            cli.push_back(p); i++;
        }
        else if (!strcmp(a, "-z") && v) { J.zeros = (u32)atoi(v); i++; }
        else if (!strcmp(a, "-S") && v) { seed = strtoull(v, nullptr, 0); i++; }
        else if (!strcmp(a, "-n") && v) { max_hits = atoll(v); i++; }
        else if (!strcmp(a, "-t") && v) { i++; }   /* accepted for CLI compatibility, ignored */
        else if (!strcmp(a, "--tried-offset") && v) { tried_off = strtoull(v, nullptr, 0); i++; }
        else if (!strcmp(a, "--time-offset") && v) { time_off = atof(v); i++; }
        else if (!strcmp(a, "--bench-secs") && v) { bench_secs = atof(v); i++; }
        else if (!strcmp(a, "-b")) bench = 1;
        else { fprintf(stderr, "unknown option %s\n", a); return 1; }
    }
    if (!cli.empty()) J.pats = cli;
    init_crc_h();

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { fprintf(stderr, "gpufinder: no CUDA device\n"); return 6; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    /* upload job */
    static DevData D;
    if (!derive(J, D)) return 1;
    u32 crc_tab[256];
    memcpy(crc_tab, crc_tab_h, sizeof crc_tab);
    u64 pv[MAXPAT] = {0}, pm[MAXPAT] = {0}; u32 ps[MAXPAT] = {0};
    for (size_t i = 0; i < J.pats.size(); i++) { pv[i] = J.pats[i].val; pm[i] = J.pats[i].mask; ps[i] = J.pats[i].suffix; }
    CK(cudaMemcpyToSymbol(c_p, &D.P, sizeof D.P));
    CK(cudaMemcpyToSymbol(c_tmpl, D.tmpl, sizeof D.tmpl));
    CK(cudaMemcpyToSymbol(c_post, D.post, sizeof D.post));
    CK(cudaMemcpyToSymbol(c_tab, D.tab, sizeof D.tab));
    CK(cudaMemcpyToSymbol(c_val, D.val, sizeof D.val));
    CK(cudaMemcpyToSymbol(c_wt, D.wt, sizeof D.wt));
    CK(cudaMemcpyToSymbol(c_ac, D.ac, sizeof D.ac));
    CK(cudaMemcpyToSymbol(c_al, D.al, sizeof D.al));
    CK(cudaMemcpyToSymbol(c_crc, crc_tab, sizeof crc_tab));
    CK(cudaMemcpyToSymbol(c_gt, D.gt, sizeof D.gt));
    CK(cudaMemcpyToSymbol(c_perm, D.perm, sizeof D.perm));
    CK(cudaMemcpyToSymbol(c_pval, pv, sizeof pv));
    CK(cudaMemcpyToSymbol(c_pmask, pm, sizeof pm));
    CK(cudaMemcpyToSymbol(c_psuf, ps, sizeof ps));

    const u32 m = J.m;
    const u64 RF = fact_h(RIN), mfact = fact_h((int)m), NOUT = mfact / RF;
    auto digits_of = [&](u64 o, BaseDigits &bd) {
        memset(&bd, 0, sizeof bd);
        for (u32 i = 0; i < m - RIN; i++) { u32 r = m - i; bd.d[i] = (u8)(o % r); o /= r; }
    };

    /* correctness gate: GPU must reproduce the CPU test vectors exactly */
    {
        int nt = (int)J.tests.size();
        if (nt < 1) { fprintf(stderr, "gpufinder: GATE FAILED: job has no test vectors\n"); return 5; }
        if (nt > 128) nt = 128;
        std::vector<BaseDigits> bds(nt);
        std::vector<u32> gids(nt), ts(nt);
        for (int i = 0; i < nt; i++) {
            u64 idx = J.tests[i].idx;
            if (idx >= mfact) { fprintf(stderr, "gpufinder: GATE FAILED: test index out of range\n"); return 5; }
            u64 o = idx / RF;
            /* exercise the carry-add of the thread id: start from a base below `o` */
            u64 g = std::min<u64>(o, 100000ull + 7 * (u64)i);
            gids[i] = (u32)g; ts[i] = (u32)(idx % RF);
            digits_of(o - g, bds[i]);
        }
        BaseDigits *d_bd; u32 *d_g, *d_t, *d_out;
        CK(cudaMalloc(&d_bd, nt * sizeof(BaseDigits))); CK(cudaMalloc(&d_g, nt * 4)); CK(cudaMalloc(&d_t, nt * 4));
        CK(cudaMalloc(&d_out, nt * 32));
        CK(cudaMemcpy(d_bd, bds.data(), nt * sizeof(BaseDigits), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_g, gids.data(), nt * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_t, ts.data(), nt * 4, cudaMemcpyHostToDevice));
        test_kernel<<<1, 128>>>(d_bd, d_g, d_t, d_out, nt);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<u32> out(nt * 8);
        CK(cudaMemcpy(out.data(), d_out, nt * 32, cudaMemcpyDeviceToHost));
        cudaFree(d_bd); cudaFree(d_g); cudaFree(d_t); cudaFree(d_out);
        int bad = 0;
        for (int i = 0; i < nt; i++) {
            u8 d[32];
            for (int k = 0; k < 8; k++) { d[4*k] = out[i*8+k] >> 24; d[4*k+1] = out[i*8+k] >> 16; d[4*k+2] = out[i*8+k] >> 8; d[4*k+3] = out[i*8+k]; }
            if (memcmp(d, J.tests[i].sha, 32)) bad++;
            /* also cross-check the host reference builder against the CPU expectation */
            std::vector<u8> file = host_file(J, J.tests[i].idx);
            u8 hd[32]; sha256_h(file.data(), file.size(), hd);
            if (memcmp(hd, J.tests[i].sha, 32)) { fprintf(stderr, "gpufinder: GATE FAILED: host reference disagrees with job test vector %d\n", i); return 5; }
        }
        if (bad) { fprintf(stderr, "gpufinder: GATE FAILED: %d of %d test vectors differ between GPU and CPU\n", bad, nt); return 5; }
        fprintf(stderr, "gate: GPU reproduced %d/%d CPU test vectors exactly\n", nt, nt);
    }

    u64 rng = seed;
    u64 obase = splitmix(rng) % NOUT, covered = 0;
    u64 batch = 1u << 20;   /* threads (outer values) per launch; 24 candidates each */
    fprintf(stderr, "device=%s (sm_%d%d, %d SMs) K=%u nblk=%u m=%u (V0=%u nvar=%u KW=%u) perms/set=%llu patterns=%zu zeros>=%u seed=%llu\n", prop.name,
            prop.major, prop.minor, prop.multiProcessorCount, J.K, J.nblk, m, J.V0, J.nvar, D.P.KW, mfact, J.pats.size(), J.zeros, seed);
    fflush(stderr);

    u32 zero = 0;
    CK(cudaMemcpyToSymbol(d_nhits, &zero, 4));
    CK(cudaMemcpyToSymbol(d_best_zeros, &zero, 4));
    double t0 = now_s(), last = t0;
    u64 tried = 0, last_n = 0, nhit_total = 0;
    bool stop = false, exhausted = false;
    while (!stop) {
        double ts = now_s();
        u64 cnt = std::min<u64>(batch, std::min(NOUT - covered, NOUT - obase));
        BaseDigits bd; digits_of(obase, bd);
        search_kernel<<<(unsigned)((cnt + BLOCK - 1) / BLOCK), BLOCK>>>(bd, obase, (u32)cnt);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        tried += cnt * RF;
        covered += cnt;
        obase += cnt; if (obase >= NOUT) obase = 0;
        double te = now_s();
        /* aim for ~60 ms per launch */
        if (te - ts < 0.03 && batch < (1ull << 27)) batch *= 2;
        else if (te - ts > 0.12 && batch > 4096) batch /= 2;

        u32 nh = 0;
        CK(cudaMemcpyFromSymbol(&nh, d_nhits, 4));
        if (nh) {
            std::vector<u64> hits(nh < MAXHITS ? nh : MAXHITS);
            CK(cudaMemcpyFromSymbol(hits.data(), d_hits, hits.size() * 8));
            CK(cudaMemcpyToSymbol(d_nhits, &zero, 4));
            if (bench) hits.clear();   /* benchmark: count only, no output */
            if (nh > MAXHITS) fprintf(stderr, "warning: %u hits in one launch, only %d kept\n", nh, MAXHITS);
            for (u64 idx : hits) {
                std::vector<u8> file = host_file(J, idx);
                u8 d[32]; sha256_h(file.data(), file.size(), d);
                std::string why;
                if (!match_why(J, d, why)) { fprintf(stderr, "gpufinder: GPU hit idx %llu fails CPU re-check (ignored)\n", (unsigned long long)idx); continue; }
                char hex[65]; hexify(d, hex);
                const char *url = hex;
                while (*url == '0' && url[1]) url++;
                std::string path = outdir + "/" + url + ".png";
                if (!bench) {
                    FILE *fp = fopen(path.c_str(), "wb");
                    if (!fp) { fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
                    fwrite(file.data(), 1, file.size(), fp); fclose(fp);
                }
                printf("HIT %s sha=%s id=%s counter=%llu file=%s\n", why.c_str(), hex, url, (unsigned long long)idx, path.c_str());
                fflush(stdout);
                nhit_total++;
                if (max_hits > 0 && (long long)nhit_total >= max_hits) { stop = true; break; }
            }
        }
        if (covered >= NOUT) { exhausted = true; stop = true; }
        double t = now_s();
        if (bench && t - t0 + time_off >= bench_secs) stop = true;
        if (t - last >= 5 || stop) {
            u32 bz = 0; CK(cudaMemcpyFromSymbol(&bz, d_best_zeros, 4));
            fprintf(stderr, "[%7.0fs] tried=%llu  rate=%.0f/s (avg %.0f/s)  hits=%llu  best_zeros=%u\n", t - t0 + time_off,
                    (unsigned long long)(tried + tried_off), (tried - last_n) / (t - last), tried / (t - t0),
                    (unsigned long long)nhit_total, bz);
            fflush(stderr);
            last = t; last_n = tried;
        }
    }
    double el = now_s() - t0;
    fprintf(stderr, "done: tried=%llu in %.1fs = %.0f/s\n", (unsigned long long)tried, el, tried / el);
    bool done_by_limit = (max_hits > 0 && (long long)nhit_total >= max_hits) || (bench && el + time_off >= bench_secs);
    if (exhausted && !done_by_limit) {
        printf("ROLLOVER tried=%llu secs=%.3f\n", (unsigned long long)(tried + tried_off), el + time_off);
        fflush(stdout);
        return 7;
    }
    if (bench) printf("RATE %.0f\n", (tried + tried_off) / (el + time_off));
    return 0;
}
