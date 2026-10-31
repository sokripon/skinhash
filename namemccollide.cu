/*
 * namemccollide: CUDA search for NameMC skin-id COLLISIONS.
 *
 * NameMC id = first 8 bytes of SHA-256( be32(64) || be32(64) || RGBA, RGB = 0 where alpha == 0 )   (16392 bytes)
 * The knobs are the LAST pixels of the image, alpha 1 with a counter as RGB (invisible: they sit on transparent
 * overlay pixels). The last two pixels (62,63) and (63,63) are bytes 16384..16391 of the message: they share the
 * final SHA-256 block (block 256) with the padding, so with the midstate after block 255 one try is ONE block.
 *
 *   mode A (list):  match any id of a sorted uint64-LE list (e.g. every NameMC id seen, 51M of them).
 *                   knobs (62,63),(63,63): 48 bits. counter c: p62 RGB = c & 0xFFFFFF, p63 RGB = (c >> 24) & 0xFFFFFF.
 *                   Each candidate id is tested against a 2^28-bit bitmap (32 MB, fits L2); a bitmap hit is confirmed
 *                   with a 2^24-bucket offset table over the sorted array.
 *   mode B (self):  two of OUR skins with the same id. Pollard rho with distinguished points on
 *                   f(x) = id(skin(x)), x a 64-bit value spread over knobs (61,63),(62,63),(63,63):
 *                   p61 RGB = x & 0xFFFFFF, p62 RGB = (x >> 24) & 0xFFFFFF, p63 R = x >> 56, G = (x >> 48) & 0xFF, B = 0.
 *                   Two blocks per step (255 and 256). ~2^32 steps, a few thousand distinguished points stored.
 *
 * usage: namemccollide rgba.bin -m A IDS.u64le.bin [-S SEED] [-n HITS] [-b]
 *        namemccollide rgba.bin -m B [-S SEED] [-d BITS]
 * rgba.bin: 64x64 RGBA, the knob pixels already alpha 1. Output (stdout):
 *   HIT A id=<16 hex> counter=<dec> index=<position in the list>
 *   COLLISION B id=<16 hex> x=<dec> y=<dec>
 * Every hit is re-hashed from scratch on the CPU. exit codes: 0 ok, 1 usage/error, 5 correctness gate failed, 6 CUDA.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#define W_ 64
#define H_ 64
#define NPIX (W_ * H_)
#define MSG_LEN (8 + NPIX * 4)          /* 16392 */
#define BM_BITS 28                      /* bitmap: 2^28 bits = 32 MB */
#define BK_BITS 24                      /* offset table: 2^24 buckets */
#define INNER 1024
#define MAXHITS 4096
#define MAXDP (1 << 16)
#ifndef BLOCK
#define BLOCK 256
#endif
#ifndef MINB
#define MINB 2
#endif

typedef unsigned long long u64;
typedef uint32_t u32;
typedef uint8_t u8;

__constant__ u32 c_mid[8];              /* A: state after block 255.  B: state after block 254 */
__constant__ u32 c_b255[16];            /* B: words of block 255 (word 15 = knob) */
__constant__ u32 c_K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ u32 d_nhits;
__device__ u64 d_hits[MAXHITS];
__device__ u64 d_nbm;                   /* candidates that passed the bitmap (stats) */

__device__ __forceinline__ u32 ror(u32 x, int n) { return __funnelshift_r(x, x, n); }

#define ROUND(kwv)                                                                   \
    do {                                                                             \
        u32 t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + (kwv); \
        u32 t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));    \
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;           \
    } while (0)

/* one SHA-256 block; st is updated (feed-forward included) */
__device__ __forceinline__ void sha_block(u32 st[8], const u32 *w16) {
    u32 w[64];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = w16[i];
#pragma unroll
    for (int i = 16; i < 64; i++) {
        u32 s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3);
        u32 s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
    for (int i = 0; i < 64; i++) ROUND(c_K256[i] + w[i]);
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

/* mode A: id of counter c (one block from the midstate after block 255) */
__device__ __forceinline__ u64 id_A(u32 w0, u32 w1) {
    u32 st[8];
#pragma unroll
    for (int i = 0; i < 8; i++) st[i] = c_mid[i];
    u32 w[16];
    w[0] = w0; w[1] = w1; w[2] = 0x80000000u;
#pragma unroll
    for (int i = 3; i < 15; i++) w[i] = 0;
    w[15] = (u32)(MSG_LEN * 8);
    sha_block(st, w);
    return ((u64)st[0] << 32) | st[1];
}

/* mode B: one rho step, x -> id of the skin whose knob pixels encode x (two blocks) */
__device__ __forceinline__ u64 step_B(u64 x) {
    u32 st[8];
#pragma unroll
    for (int i = 0; i < 8; i++) st[i] = c_mid[i];
    u32 w[16];
#pragma unroll
    for (int i = 0; i < 15; i++) w[i] = c_b255[i];
    w[15] = (((u32)x & 0xFFFFFFu) << 8) | 1u;
    sha_block(st, w);
    w[0] = ((((u32)(x >> 24)) & 0xFFFFFFu) << 8) | 1u;
    w[1] = (((u32)(x >> 48)) << 16) | 1u;
    w[2] = 0x80000000u;
#pragma unroll
    for (int i = 3; i < 15; i++) w[i] = 0;
    w[15] = (u32)(MSG_LEN * 8);
    sha_block(st, w);
    return ((u64)st[0] << 32) | st[1];
}

__device__ __forceinline__ bool in_list(u64 id, const u64 *__restrict__ ids, const u32 *__restrict__ tab,
                                        const u32 *__restrict__ bm, bool count) {
    u32 bi = (u32)(id >> (64 - BM_BITS));
    if (!((bm[bi >> 5] >> (bi & 31)) & 1)) return false;
    if (count) atomicAdd(&d_nbm, 1ull);
    u32 b = (u32)(id >> (64 - BK_BITS));
    u32 lo = tab[b], hi = tab[b + 1];
    for (u32 i = lo; i < hi; i++)
        if (ids[i] == id) return true;
    return false;
}

/* mode A search kernel: thread owns INNER consecutive counters (they differ only in p62) */
__global__ void __launch_bounds__(BLOCK, MINB) kern_A(u64 base, const u64 *__restrict__ ids,
                                                      const u32 *__restrict__ tab, const u32 *__restrict__ bm, u32 n) {
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= n) return;
    u64 cb = (base + (u64)gid * INNER) & 0xFFFFFFFFFFFFull;
    u32 w1 = ((((u32)(cb >> 24)) & 0xFFFFFFu) << 8) | 1u;
    u32 lo = (u32)cb & 0xFFFFFFu;
    for (u32 t = 0; t < INNER; t++) {
        u64 id = id_A(((lo + t) << 8) | 1u, w1);
        if (in_list(id, ids, tab, bm, false)) {
            u32 pos = atomicAdd(&d_nhits, 1);
            if (pos < MAXHITS) d_hits[pos] = cb + t;
        }
    }
}

/* gate kernels */
__global__ void kern_A_test(const u64 *tc, u32 n, u64 *out) {
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= n) return;
    u64 c = tc[gid];
    out[gid] = id_A((((u32)c & 0xFFFFFFu) << 8) | 1u, ((((u32)(c >> 24)) & 0xFFFFFFu) << 8) | 1u);
}
__global__ void kern_B_test(const u64 *tc, u32 n, u64 *out) {
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= n) return;
    out[gid] = step_B(tc[gid]);
}
__global__ void kern_list_test(const u64 *q, u32 n, const u64 *ids, const u32 *tab, const u32 *bm, u8 *out) {
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= n) return;
    out[gid] = in_list(q[gid], ids, tab, bm, false);
}

__device__ __forceinline__ u64 splitmix(u64 z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct DP { u64 x0, id; u32 len, pad; };
__device__ u32 d_ndp;
__device__ DP d_dp[MAXDP];

/* mode B: each thread walks a trail x -> f(x); a distinguished point (top DPB bits of the id zero) is reported with
 * the trail's start and length, then a fresh random start is taken. Trails longer than maxlen (cycles) are dropped. */
__global__ void __launch_bounds__(BLOCK, MINB) kern_B(u32 steps, u64 salt, int dpb, u32 maxlen, u32 n,
                                                      u64 *cx, u64 *cx0, u32 *clen, u32 *crs, int init) {
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= n) return;
    u64 x, x0; u32 len, rs;
    if (init) { rs = 0; x0 = splitmix(salt + (u64)gid * 0xD1B54A32D192ED03ull); x = x0; len = 0; }
    else { x = cx[gid]; x0 = cx0[gid]; len = clen[gid]; rs = crs[gid]; }
    for (u32 s = 0; s < steps; s++) {
        u64 id = step_B(x);
        len++;
        bool dp = (id >> (64 - dpb)) == 0;
        if (dp) {
            u32 pos = atomicAdd(&d_ndp, 1);
            if (pos < MAXDP) { d_dp[pos].x0 = x0; d_dp[pos].id = id; d_dp[pos].len = len; }
        }
        if (dp || len >= maxlen) {
            rs++;
            x0 = splitmix(salt + (u64)gid * 0xD1B54A32D192ED03ull + (u64)rs * 0x9E3779B97F4A7C15ull);
            x = x0; len = 0;
        } else x = id;
    }
    cx[gid] = x; cx0[gid] = x0; clen[gid] = len; crs[gid] = rs;
}

/* ---------------- host ---------------- */
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
static void expand_h(const u8 *p, u32 w[64]) {
    for (int i = 0; i < 16; i++) w[i] = (u32)p[4*i] << 24 | (u32)p[4*i+1] << 16 | (u32)p[4*i+2] << 8 | p[4*i+3];
    for (int i = 16; i < 64; i++) {
        u32 s0 = HROR(w[i-15], 7) ^ HROR(w[i-15], 18) ^ (w[i-15] >> 3);
        u32 s1 = HROR(w[i-2], 17) ^ HROR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
}
static void sha256_blocks_h(u32 s[8], const u8 *p, size_t nb) {
    while (nb--) {
        u32 w[64], a, b, c, d, e, f, g, h;
        expand_h(p, w);
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
static const u32 IV[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
static void sha256_h(const u8 *msg, size_t len, u8 out[32]) {
    u32 s[8];
    memcpy(s, IV, sizeof s);
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

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "namemccollide: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(6); } } while (0)
static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static void put32(u8 *p, u32 v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }

static u8 g_rgba[NPIX * 4];
static u8 g_msg[MSG_LEN];
static char g_mode = 'A';

/* knob pixel p (0 = (62,63) / 1 = (63,63) in mode A; (61,63),(62,63),(63,63) in mode B) -> message offset */
static size_t pix_off(int x, int y) { return 8 + ((size_t)y * W_ + x) * 4; }
static void set_rgb(u8 *m, int x, int y, u32 rgb24) { size_t o = pix_off(x, y); m[o] = rgb24 >> 16; m[o+1] = rgb24 >> 8; m[o+2] = rgb24; m[o+3] = 1; }

static void fill_A(u8 *m, u64 c) { set_rgb(m, 62, 63, (u32)c & 0xFFFFFF); set_rgb(m, 63, 63, (u32)(c >> 24) & 0xFFFFFF); }
static void fill_B(u8 *m, u64 x) {
    set_rgb(m, 61, 63, (u32)x & 0xFFFFFF);
    set_rgb(m, 62, 63, (u32)(x >> 24) & 0xFFFFFF);
    set_rgb(m, 63, 63, ((u32)(x >> 48) & 0xFFFF) << 8);
}
static u64 ref_id(u64 c) {
    static u8 m[MSG_LEN];
    memcpy(m, g_msg, MSG_LEN);
    if (g_mode == 'A') fill_A(m, c); else fill_B(m, c);
    u8 d[32]; sha256_h(m, MSG_LEN, d);
    u64 id = 0; for (int i = 0; i < 8; i++) id = id << 8 | d[i];
    return id;
}
static u64 mix64(u64 z) { z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: namemccollide rgba.bin -m A IDS.u64le.bin [-S SEED] [-n HITS] [-b]\n       namemccollide rgba.bin -m B [-S SEED] [-d BITS]\n"); return 1; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp || fread(g_rgba, 1, sizeof g_rgba, fp) != sizeof g_rgba) { fprintf(stderr, "bad rgba file\n"); return 1; }
    fclose(fp);
    std::string idspath;
    long long max_hits = 1;
    int bench = 0, dpb = 13;
    u64 seed = (u64)std::chrono::system_clock::now().time_since_epoch().count() * 0x9E3779B97F4A7C15ULL;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!strcmp(a, "-m") && v) { g_mode = v[0]; i++; if (g_mode == 'A' && i + 1 < argc && argv[i + 1][0] != '-') idspath = argv[++i]; }
        else if (!strcmp(a, "-S") && v) { seed = strtoull(v, nullptr, 0); i++; }
        else if (!strcmp(a, "-n") && v) { max_hits = atoll(v); i++; }
        else if (!strcmp(a, "-d") && v) { dpb = atoi(v); i++; }
        else if (!strcmp(a, "-b")) bench = 1;
        else { fprintf(stderr, "unknown option %s\n", a); return 1; }
    }
    if (g_mode != 'A' && g_mode != 'B') { fprintf(stderr, "-m must be A or B\n"); return 1; }
    if (g_mode == 'A' && idspath.empty()) { fprintf(stderr, "mode A needs the id list\n"); return 1; }
    if (dpb < 4 || dpb > 32) { fprintf(stderr, "-d out of range\n"); return 1; }

    /* message; knob pixels must be alpha 1 in the input, their RGB is irrelevant (overwritten per candidate) */
    int nk = g_mode == 'A' ? 2 : 3;
    for (int j = 0; j < nk; j++) {
        const u8 *p = g_rgba + ((size_t)63 * W_ + (W_ - nk + j)) * 4;
        if (p[3] != 1) { fprintf(stderr, "knob pixel (%d,63) must have alpha 1 in the input (has %d)\n", W_ - nk + j, p[3]); return 1; }
    }
    put32(g_msg, W_); put32(g_msg + 4, H_);
    for (int i = 0; i < NPIX; i++) {
        const u8 *p = g_rgba + i * 4; u8 *q = g_msg + 8 + i * 4;
        q[0] = p[3] ? p[0] : 0; q[1] = p[3] ? p[1] : 0; q[2] = p[3] ? p[2] : 0; q[3] = p[3];
    }
    u8 abc[32]; sha256_h((const u8 *)"abc", 3, abc);
    if (abc[0] != 0xba || abc[1] != 0x78 || abc[31] != 0xad) { fprintf(stderr, "host SHA-256 self-test failed\n"); return 5; }

    /* midstates */
    u32 mid[8]; memcpy(mid, IV, sizeof mid);
    u32 b255[16] = {0};
    if (g_mode == 'A') sha256_blocks_h(mid, g_msg, 256);
    else {
        sha256_blocks_h(mid, g_msg, 255);
        for (int i = 0; i < 16; i++) b255[i] = (u32)g_msg[255*64+4*i] << 24 | (u32)g_msg[255*64+4*i+1] << 16 | (u32)g_msg[255*64+4*i+2] << 8 | g_msg[255*64+4*i+3];
        b255[15] = 1;
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { fprintf(stderr, "namemccollide: no CUDA device\n"); return 6; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    CK(cudaMemcpyToSymbol(c_mid, mid, sizeof mid));
    CK(cudaMemcpyToSymbol(c_b255, b255, sizeof b255));

    /* ---- list (mode A) ---- */
    std::vector<u64> ids;
    u64 *d_ids = nullptr; u32 *d_tab = nullptr, *d_bm = nullptr;
    if (g_mode == 'A') {
        FILE *lf = fopen(idspath.c_str(), "rb");
        if (!lf) { fprintf(stderr, "cannot open %s\n", idspath.c_str()); return 1; }
        fseek(lf, 0, SEEK_END); long long sz = ftell(lf); fseek(lf, 0, SEEK_SET);
        if (sz <= 0 || sz % 8) { fprintf(stderr, "%s: size %lld is not a multiple of 8\n", idspath.c_str(), sz); return 1; }
        ids.resize(sz / 8);
        if (fread(ids.data(), 8, ids.size(), lf) != ids.size()) { fprintf(stderr, "short read\n"); return 1; }
        fclose(lf);
        for (size_t i = 1; i < ids.size(); i++)
            if (ids[i] <= ids[i-1]) { fprintf(stderr, "list is not strictly ascending at %zu\n", i); return 1; }
        std::vector<u32> tab((size_t(1) << BK_BITS) + 1, 0), bm(size_t(1) << (BM_BITS - 5), 0);
        for (u64 v : ids) { tab[(v >> (64 - BK_BITS)) + 1]++; u32 bi = (u32)(v >> (64 - BM_BITS)); bm[bi >> 5] |= 1u << (bi & 31); }
        for (size_t i = 1; i < tab.size(); i++) tab[i] += tab[i-1];
        CK(cudaMalloc(&d_ids, ids.size() * 8)); CK(cudaMalloc(&d_tab, tab.size() * 4)); CK(cudaMalloc(&d_bm, bm.size() * 4));
        CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tab, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_bm, bm.data(), bm.size() * 4, cudaMemcpyHostToDevice));
        size_t setbits = 0; for (u32 x : bm) setbits += __popcnt(x);
        fprintf(stderr, "list: %zu ids, bitmap %.1f%% set (false-positive rate), expected tries ~ %.3g\n", ids.size(),
                100.0 * setbits / (double(1) * (size_t(1) << BM_BITS)), 18446744073709551616.0 / ids.size());
    }

    auto grid_for = [](u32 n) { return (n + BLOCK - 1) / BLOCK; };

    /* ---- correctness gate ---- */
    {
        std::vector<u64> tcs;
        if (g_mode == 'A') tcs = {0, 1, 1023, 1024, 0xFFFFFFull, 0x1000000ull, 0xFFFFFFFFFFFFull, 0x800000000000ull, 0x00FF00FF00FFull};
        else tcs = {0, 1, 0xFFFFFFull, 0x1000000ull, 0xFFFFFFFFFFFFull, 0x1000000000000ull, ~0ull, 0x8000000000000000ull, 0x00FF00FF00FF00FFull};
        u64 r = seed ^ 0xA5A5A5A55A5A5A5Aull;
        while (tcs.size() < 128) { r = mix64(r); tcs.push_back(g_mode == 'A' ? (r & 0xFFFFFFFFFFFFull) : r); }
        u32 nt = (u32)tcs.size();
        u64 *d_tc, *d_out;
        CK(cudaMalloc(&d_tc, nt * 8)); CK(cudaMalloc(&d_out, nt * 8));
        CK(cudaMemcpy(d_tc, tcs.data(), nt * 8, cudaMemcpyHostToDevice));
        if (g_mode == 'A') kern_A_test<<<grid_for(nt), BLOCK>>>(d_tc, nt, d_out); else kern_B_test<<<grid_for(nt), BLOCK>>>(d_tc, nt, d_out);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<u64> got(nt);
        CK(cudaMemcpy(got.data(), d_out, nt * 8, cudaMemcpyDeviceToHost));
        cudaFree(d_tc); cudaFree(d_out);
        int bad = 0;
        for (u32 i = 0; i < nt; i++) if (got[i] != ref_id(tcs[i])) bad++;
        if (bad) { fprintf(stderr, "namemccollide: GATE FAILED: %d of %u hash test vectors differ between GPU and CPU\n", bad, nt); return 5; }
        fprintf(stderr, "gate: GPU reproduced %u/%u CPU hash vectors exactly\n", nt, nt);
        if (g_mode == 'A') {   /* list lookup gate: members must hit, near-misses must not */
            std::vector<u64> q; std::vector<u8> want;
            for (int i = 0; i < 4096; i++) { u64 v = ids[mix64(seed + i) % ids.size()]; q.push_back(v); want.push_back(1); }
            for (int i = 0; i < 4096; i++) {
                u64 v = ids[mix64(seed + 7777 + i) % ids.size()] ^ (1ull << (mix64(i) % 64));
                q.push_back(v); want.push_back(std::binary_search(ids.begin(), ids.end(), v));
            }
            for (int i = 0; i < 4096; i++) { u64 v = mix64(seed ^ (0xABCDull + i)); q.push_back(v); want.push_back(std::binary_search(ids.begin(), ids.end(), v)); }
            u32 nq = (u32)q.size(); u64 *d_q; u8 *d_o;
            CK(cudaMalloc(&d_q, nq * 8)); CK(cudaMalloc(&d_o, nq));
            CK(cudaMemcpy(d_q, q.data(), nq * 8, cudaMemcpyHostToDevice));
            kern_list_test<<<grid_for(nq), BLOCK>>>(d_q, nq, d_ids, d_tab, d_bm, d_o);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<u8> o(nq); CK(cudaMemcpy(o.data(), d_o, nq, cudaMemcpyDeviceToHost));
            cudaFree(d_q); cudaFree(d_o);
            int badq = 0; for (u32 i = 0; i < nq; i++) badq += (o[i] != want[i]);
            if (badq) { fprintf(stderr, "namemccollide: GATE FAILED: %d of %u list lookups wrong\n", badq, nq); return 5; }
            fprintf(stderr, "gate: GPU list lookup correct for %u/%u queries (members, 1-bit misses, random)\n", nq, nq);
        }
    }
    fflush(stderr);

    double t0 = now_s(), last = t0;
    u64 tried = 0, last_n = 0;

    if (g_mode == 'A') {
        u64 base = seed & 0xFFFFFFFFFFFFull & ~(u64)(INNER - 1);
        u32 threads = 1u << 14;
        u32 zero = 0; u64 zero64 = 0;
        CK(cudaMemcpyToSymbol(d_nhits, &zero, 4));
        CK(cudaMemcpyToSymbol(d_nbm, &zero64, 8));
        fprintf(stderr, "device=%s (sm_%d%d, %d SMs) mode=A knobs=(62,63)(63,63) seed=%llu\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, (unsigned long long)seed);
        double expect = 18446744073709551616.0 / ids.size();
        long long nhit_total = 0; bool stop = false;
        while (!stop) {
            double ts = now_s();
            kern_A<<<grid_for(threads), BLOCK>>>(base, d_ids, d_tab, d_bm, threads);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            tried += (u64)threads * INNER; base = (base + (u64)threads * INNER) & 0xFFFFFFFFFFFFull;
            double te = now_s();
            if (te - ts < 0.03 && threads < (1u << 24)) threads *= 2;
            else if (te - ts > 0.12 && threads > 4096) threads /= 2;
            u32 nh = 0; CK(cudaMemcpyFromSymbol(&nh, d_nhits, 4));
            if (nh) {
                std::vector<u64> hits(nh < MAXHITS ? nh : MAXHITS);
                CK(cudaMemcpyFromSymbol(hits.data(), d_hits, hits.size() * 8));
                CK(cudaMemcpyToSymbol(d_nhits, &zero, 4));
                if (bench) hits.clear();
                for (u64 c : hits) {
                    u64 id = ref_id(c);
                    auto it = std::lower_bound(ids.begin(), ids.end(), id);
                    if (it == ids.end() || *it != id) { fprintf(stderr, "namemccollide: GPU hit counter %llu fails CPU re-check (ignored)\n", (unsigned long long)c); continue; }
                    printf("HIT A id=%016llx counter=%llu index=%zu\n", (unsigned long long)id, (unsigned long long)c, (size_t)(it - ids.begin()));
                    fflush(stdout);
                    if (++nhit_total >= max_hits) { stop = true; break; }
                }
            }
            double t = now_s();
            if (bench && t - t0 >= 10) stop = true;
            if (t - last >= 5 || stop) {
                fprintf(stderr, "[%7.0fs] tried=%llu  rate=%.0f/s (avg %.0f/s)  hits=%lld  expected_tries=%.3g (%.0f%%)\n", t - t0,
                        (unsigned long long)tried, (tried - last_n) / (t - last), tried / (t - t0), nhit_total, expect, 100.0 * tried / expect);
                fflush(stderr); last = t; last_n = tried;
            }
        }
        double el = now_s() - t0;
        fprintf(stderr, "done: tried=%llu in %.1fs = %.0f/s\n", (unsigned long long)tried, el, tried / el);
        if (bench) printf("RATE %.0f\n", tried / el);
        return 0;
    }

    /* ---- mode B: rho with distinguished points ---- */
    const u32 T = 1u << 16, MAXLEN = 40u << dpb;
    u64 *cx, *cx0; u32 *clen, *crs;
    CK(cudaMalloc(&cx, T * 8)); CK(cudaMalloc(&cx0, T * 8)); CK(cudaMalloc(&clen, T * 4)); CK(cudaMalloc(&crs, T * 4));
    u32 zero = 0; CK(cudaMemcpyToSymbol(d_ndp, &zero, 4));
    fprintf(stderr, "device=%s (sm_%d%d, %d SMs) mode=B knobs=(61,63)(62,63)(63,63) threads=%u dp_bits=%d seed=%llu expected ~%.3g steps\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount, T, dpb, (unsigned long long)seed, 1.25 * 4294967296.0);
    fflush(stderr);
    struct Tr { u64 x0; u32 len; };
    std::unordered_map<u64, Tr> seen;
    std::vector<DP> buf(MAXDP);
    u32 steps = 256; int init = 1; u64 ndp_total = 0;
    while (true) {
        double ts = now_s();
        kern_B<<<grid_for(T), BLOCK>>>(steps, seed, dpb, MAXLEN, T, cx, cx0, clen, crs, init);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        init = 0; tried += (u64)T * steps;
        double te = now_s();
        if (te - ts < 0.05 && steps < (1u << 20)) steps *= 2; else if (te - ts > 0.2 && steps > 16) steps /= 2;
        u32 nd = 0; CK(cudaMemcpyFromSymbol(&nd, d_ndp, 4));
        if (nd) {
            if (nd > MAXDP) { fprintf(stderr, "warning: %u distinguished points in one launch, %d kept\n", nd, MAXDP); nd = MAXDP; }
            CK(cudaMemcpyFromSymbol(buf.data(), d_dp, nd * sizeof(DP)));
            CK(cudaMemcpyToSymbol(d_ndp, &zero, 4));
            for (u32 i = 0; i < nd; i++) {
                ndp_total++;
                auto ins = seen.emplace(buf[i].id, Tr{buf[i].x0, buf[i].len});
                if (ins.second || ins.first->second.x0 == buf[i].x0) continue;
                /* two different trails end at the same distinguished point: walk them to the merge */
                u64 a = ins.first->second.x0, b = buf[i].x0; u32 la = ins.first->second.len, lb = buf[i].len;
                while (la > lb) { a = ref_id(a); la--; }
                while (lb > la) { b = ref_id(b); lb--; }
                if (a == b) continue;                       /* one start lies on the other trail: no collision */
                for (u32 k = 0; k <= la; k++) {
                    u64 fa = ref_id(a), fb = ref_id(b);
                    if (fa == fb) {
                        printf("COLLISION B id=%016llx x=%llu y=%llu\n", (unsigned long long)fa, (unsigned long long)a, (unsigned long long)b);
                        fflush(stdout);
                        fprintf(stderr, "done: %llu steps in %.1fs, %llu distinguished points\n", (unsigned long long)tried, now_s() - t0, (unsigned long long)ndp_total);
                        return 0;
                    }
                    a = fa; b = fb;
                }
            }
        }
        double t = now_s();
        if (t - last >= 5) {
            fprintf(stderr, "[%7.0fs] tried=%llu  rate=%.0f/s (avg %.0f/s)  dps=%llu  expected_tries=%.3g (%.0f%%)\n", t - t0, (unsigned long long)tried,
                    (tried - last_n) / (t - last), tried / (t - t0), (unsigned long long)ndp_total, 1.25 * 4294967296.0, 100.0 * tried / (1.25 * 4294967296.0));
            fflush(stderr); last = t; last_n = tried;
        }
    }
}
