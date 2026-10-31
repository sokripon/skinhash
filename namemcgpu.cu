/*
 * namemcgpu: CUDA search for NameMC skin ids (GPU version of namemcfinder).
 *
 * NameMC id = first 8 bytes (16 hex) of
 *   SHA-256( be32(64) || be32(64) || RGBA pixels, RGB set to 0 where alpha == 0 )      (16392 bytes)
 * Knob: three pixels (alpha 1) whose RGB is a counter. Each pixel is one 32-bit message word
 * (R<<24 | G<<16 | B<<8 | 1). The CPU hashes the fixed blocks once (midstate) and prepares the remaining
 * blocks; a thread finishes them. Blocks without a knob word have a constant message schedule, so
 * K+W is precomputed for them.
 *
 * Counter c (64 bit) -> pixels, knob pixels in message order kp[0] < kp[1] < kp[2]:
 *   kp[2] RGB = c & 0xFFFFFF,  kp[1] RGB = (c >> 24) & 0xFFFFFF,  kp[0] RGB = c >> 48
 *
 * Kernels:
 *   fast:    knob words 9,10,11 of the first of exactly 2 blocks (the default layout, pixels (55-57,63)).
 *            One thread owns INNER consecutive counters that differ only in kp[2]; everything before
 *            kp[2]'s word is computed once per thread.
 *   general: any layout (e.g. --unused-only), one candidate per thread.
 *
 * usage: namemcgpu rgba.bin OUTDIR -k x,y,x,y,x,y [--render-exe namemcfinder.exe] [-p HEX]... [-s HEX]...
 *                  [-z N] [-S SEED] [-n N] [-b]
 * Every hit is re-hashed on the CPU; the PNG is written by `namemcfinder --render`.
 * exit codes: 0 ok, 1 usage/error, 5 correctness gate failed, 6 CUDA error / no device.
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

#define W_ 64
#define H_ 64
#define NPIX (W_ * H_)
#define MSG_LEN (8 + NPIX * 4)
#define NK 3
#define MAXNB 64
#define MAXPAT 64
#define MAXHITS 4096
#define INNER 1024
#ifndef BLOCK
#define BLOCK 256
#endif
#ifndef MINB
#define MINB 2
#endif

typedef unsigned long long u64;
typedef uint32_t u32;
typedef uint8_t u8;

__constant__ u32 c_mid[8];
__constant__ u32 c_img[MAXNB * 16];     /* base words, big-endian; knob words hold 0x00000001 */
__constant__ u32 c_kw[MAXNB * 64];      /* K[i] + W[i] for blocks without a knob word */
__constant__ u32 c_isc[MAXNB];          /* 1 = block has no knob word */
__constant__ u32 c_nb, c_np;
__constant__ u32 c_pblk[NK], c_pwd[NK];
__constant__ u64 c_pval[MAXPAT], c_pmask[MAXPAT];
__constant__ u64 c_zthr;                /* id < zthr is a hit (0 = off) */
__constant__ u32 c_K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ u64 d_best;                  /* smallest id seen */
__device__ u32 d_nhits;
__device__ u64 d_hits[MAXHITS];

__device__ __forceinline__ u32 ror(u32 x, int n) { return __funnelshift_r(x, x, n); }

#define ROUND(kwv)                                                                   \
    do {                                                                             \
        u32 t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + (kwv); \
        u32 t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));    \
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;           \
    } while (0)

/* one block with a message that depends on the candidate */
__device__ __forceinline__ void sha_full(u32 st[8], const u32 *w16) {
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

/* one block whose message is the same for every candidate: K+W comes from a table */
__device__ __forceinline__ void sha_const(u32 st[8], int off) {
    u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
    for (int i = 0; i < 64; i++) ROUND(c_kw[off + i]);
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

__device__ __forceinline__ bool is_hit(u64 id) {
    if (id < c_zthr) return true;
    for (u32 i = 0; i < c_np; i++)
        if ((id & c_pmask[i]) == c_pval[i]) return true;
    return false;
}

__device__ __forceinline__ void record(u64 c) {
    u32 pos = atomicAdd(&d_nhits, 1);
    if (pos < MAXHITS) d_hits[pos] = c;
}

/* fast kernel. MODE 1 (gate): thread i evaluates counter tc[i] and writes its id to out[i]. */
template <int MODE>
__global__ void __launch_bounds__(BLOCK, MINB) fast_kernel(u64 base, const u64 *tc, u32 n, u64 *out) {
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= n) return;
    u64 cb = MODE ? (tc[gid] & ~(u64)(INNER - 1)) : base + (u64)gid * INNER;
    u32 tsel = MODE ? (u32)(tc[gid] & (INNER - 1)) : 0;
    u32 w[16];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = c_img[i];
    w[9] = ((u32)(cb >> 48) << 8) | 1u;
    w[10] = (((u32)(cb >> 24) & 0xFFFFFFu) << 8) | 1u;
    u32 lo = (u32)cb & 0xFFFFFFu;
    u64 best = ~0ull;
    for (u32 t = 0; t < INNER; t++) {
        w[11] = ((lo + t) << 8) | 1u;
        u32 st[8];
#pragma unroll
        for (int i = 0; i < 8; i++) st[i] = c_mid[i];
        sha_full(st, w);
        sha_const(st, 64);
        u64 id = ((u64)st[0] << 32) | st[1];
        if (MODE) { if (t == tsel) out[gid] = id; }
        else {
            best = id < best ? id : best;
            if (is_hit(id)) record(cb + t);
        }
    }
    if (!MODE) atomicMin(&d_best, best);
}

/* general kernel: any block layout, one candidate per thread */
template <int MODE>
__global__ void __launch_bounds__(BLOCK, MINB) general_kernel(u64 base, const u64 *tc, u32 n, u64 *out) {
    u32 gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= n) return;
    u64 c = MODE ? tc[gid] : base + gid;
    u32 vw[NK] = {((u32)(c >> 48) << 8) | 1u, (((u32)(c >> 24) & 0xFFFFFFu) << 8) | 1u, (((u32)c & 0xFFFFFFu) << 8) | 1u};
    u32 st[8];
#pragma unroll
    for (int i = 0; i < 8; i++) st[i] = c_mid[i];
    u32 nb = c_nb;
#pragma unroll 1
    for (u32 b = 0; b < nb; b++) {
        if (c_isc[b]) sha_const(st, b * 64);
        else {
            u32 w[16];
#pragma unroll
            for (int i = 0; i < 16; i++) {
                w[i] = c_img[b * 16 + i];
#pragma unroll
                for (int j = 0; j < NK; j++)
                    if (c_pblk[j] == b && c_pwd[j] == (u32)i) w[i] = vw[j];
            }
            sha_full(st, w);
        }
    }
    u64 id = ((u64)st[0] << 32) | st[1];
    if (MODE) out[gid] = id;
    else {
        atomicMin(&d_best, id);
        if (is_hit(id)) record(c);
    }
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

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "namemcgpu: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(6); } } while (0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Pat { u64 val, mask; bool suffix; char text[24]; };
static bool parse_pat(const char *hex, bool suffix, Pat &p) {
    size_t n = strlen(hex);
    if (n == 0 || n > 16) return false;
    p.val = p.mask = 0; p.suffix = suffix;
    for (size_t i = 0; i < n; i++) {
        char c = hex[i];
        int v = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (v < 0) return false;
        int shift = suffix ? (int)(4 * (n - 1 - i)) : (int)(60 - 4 * i);
        p.val |= (u64)v << shift;
        p.mask |= (u64)0xF << shift;
        p.text[i] = (char)(c | 0x20);
    }
    p.text[n] = 0;
    return true;
}

static u8 g_rgba[NPIX * 4];
static u8 g_msg[MSG_LEN];
static int g_kx[NK], g_ky[NK];
static size_t g_off[NK];
static std::vector<Pat> g_pats;
static u32 g_zeros = 0;

static void put32(u8 *p, u32 v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }

static void fill_counter(u8 *m, u64 c) {
    u32 v[NK] = {(u32)(c >> 48), (u32)(c >> 24) & 0xFFFFFF, (u32)c & 0xFFFFFF};
    for (int j = 0; j < NK; j++) { m[g_off[j]] = v[j] >> 16; m[g_off[j] + 1] = v[j] >> 8; m[g_off[j] + 2] = v[j]; }
}

/* reference id of counter c: from-scratch SHA-256 of the whole message (independent of the midstate path) */
static u64 ref_id(u64 c, u8 sha[32] = nullptr) {
    static u8 m[MSG_LEN];
    memcpy(m, g_msg, MSG_LEN);
    fill_counter(m, c);
    u8 d[32];
    sha256_h(m, MSG_LEN, d);
    if (sha) memcpy(sha, d, 32);
    u64 id = 0;
    for (int i = 0; i < 8; i++) id = id << 8 | d[i];
    return id;
}

#ifdef _WIN32
#include <windows.h>
/* No handle inheritance, so a killed parent can't leave a child holding the caller's stdout pipe. */
static int run_cmd(const std::string &cmd) {
    STARTUPINFOA si; PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si); si.cb = sizeof si;
    std::vector<char> buf(cmd.begin(), cmd.end()); buf.push_back(0);
    if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD rc = 1; GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int)rc;
}
#else
static int run_cmd(const std::string &cmd) { return system(cmd.c_str()); }
#endif

static int clz64(u64 x) { int n = 0; while (n < 64 && !(x >> (63 - n) & 1)) n++; return n; }

static bool match_why(u64 id, std::string &why) {
    for (auto &p : g_pats)
        if ((id & p.mask) == p.val) { why = std::string(p.suffix ? "suffix:" : "prefix:") + p.text; return true; }
    int zn = id ? clz64(id) / 4 : 16;
    if (g_zeros && (u32)zn >= g_zeros) { why = "zeros:" + std::to_string(zn); return true; }
    return false;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: namemcgpu rgba.bin OUTDIR -k x,y,x,y,x,y [--render-exe EXE] [-p HEX]... [-s HEX]... [-z N] [-S SEED] [-n N] [-b]\n"); return 1; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp || fread(g_rgba, 1, sizeof g_rgba, fp) != sizeof g_rgba) { fprintf(stderr, "bad rgba file\n"); return 1; }
    fclose(fp);
    std::string outdir = argv[2], render_exe = "namemcfinder.exe", rgba_path = argv[1];
    long long max_hits = -1;
    int bench = 0, have_k = 0;
    u64 seed = (u64)std::chrono::system_clock::now().time_since_epoch().count() * 0x9E3779B97F4A7C15ULL;
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : nullptr;
        Pat p;
        if (!strcmp(a, "-k") && v) {
            if (sscanf(v, "%d,%d,%d,%d,%d,%d", &g_kx[0], &g_ky[0], &g_kx[1], &g_ky[1], &g_kx[2], &g_ky[2]) != 6) { fprintf(stderr, "bad -k %s\n", v); return 1; }
            have_k = 1; i++;
        }
        else if ((!strcmp(a, "-p") || !strcmp(a, "-s")) && v) {
            if (!parse_pat(v, a[1] == 's', p) || g_pats.size() >= MAXPAT) { fprintf(stderr, "bad %s %s\n", a, v); return 1; }
            g_pats.push_back(p); i++;
        }
        else if (!strcmp(a, "-z") && v) { g_zeros = (u32)atoi(v); i++; }
        else if (!strcmp(a, "-S") && v) { seed = strtoull(v, nullptr, 0); i++; }
        else if (!strcmp(a, "-n") && v) { max_hits = atoll(v); i++; }
        else if (!strcmp(a, "-t") && v) { i++; }   /* accepted for CLI compatibility, ignored */
        else if (!strcmp(a, "--render-exe") && v) { render_exe = v; i++; }
        else if (!strcmp(a, "-b")) bench = 1;
        else { fprintf(stderr, "unknown option %s\n", a); return 1; }
    }
    if (!have_k) { fprintf(stderr, "-k is required\n"); return 1; }
    for (int a = 0; a < NK; a++)
        for (int b = a + 1; b < NK; b++)
            if (g_ky[b] * W_ + g_kx[b] < g_ky[a] * W_ + g_kx[a]) {
                std::swap(g_kx[a], g_kx[b]); std::swap(g_ky[a], g_ky[b]);
            }
    for (int j = 0; j < NK; j++)
        if (g_kx[j] < 0 || g_kx[j] >= W_ || g_ky[j] < 0 || g_ky[j] >= H_ || (j && g_ky[j] * W_ + g_kx[j] == g_ky[j-1] * W_ + g_kx[j-1])) { fprintf(stderr, "bad knob pixels\n"); return 1; }
    char kopt[128];
    snprintf(kopt, sizeof kopt, "%d,%d,%d,%d,%d,%d", g_kx[0], g_ky[0], g_kx[1], g_ky[1], g_kx[2], g_ky[2]);

    /* message */
    put32(g_msg, W_); put32(g_msg + 4, H_);
    for (int j = 0; j < NK; j++) g_rgba[(g_ky[j] * W_ + g_kx[j]) * 4 + 3] = 1;
    for (int i = 0; i < NPIX; i++) {
        const u8 *p = g_rgba + i * 4; u8 *q = g_msg + 8 + i * 4;
        q[0] = p[3] ? p[0] : 0; q[1] = p[3] ? p[1] : 0; q[2] = p[3] ? p[2] : 0; q[3] = p[3];
    }
    for (int j = 0; j < NK; j++) {
        g_off[j] = 8 + ((size_t)g_ky[j] * W_ + g_kx[j]) * 4;
        g_msg[g_off[j]] = g_msg[g_off[j] + 1] = g_msg[g_off[j] + 2] = 0;
    }
    u8 abc[32]; sha256_h((const u8 *)"abc", 3, abc);
    if (abc[0] != 0xba || abc[1] != 0x78 || abc[31] != 0xad) { fprintf(stderr, "host SHA-256 self-test failed\n"); return 5; }

    /* midstate and per-block tables */
    size_t fb = g_off[0] / 64;
    u32 mid[8]; memcpy(mid, IV, sizeof mid);
    sha256_blocks_h(mid, g_msg, fb);
    size_t rem = MSG_LEN - fb * 64, nb = (rem + 9 + 63) / 64;
    if (nb > MAXNB) { fprintf(stderr, "namemcgpu: %zu blocks after the first knob, max %d\n", nb, MAXNB); return 1; }
    std::vector<u8> tail(nb * 64, 0);
    memcpy(tail.data(), g_msg + fb * 64, rem);
    tail[rem] = 0x80;
    u64 bits = (u64)MSG_LEN * 8;
    for (int i = 0; i < 8; i++) tail[nb * 64 - 1 - i] = (u8)(bits >> (8 * i));
    u32 pblk[NK], pwd[NK];
    std::vector<u32> isc(MAXNB, 0), img(MAXNB * 16, 0), kw(MAXNB * 64, 0);
    for (size_t b = 0; b < nb; b++) isc[b] = 1;
    for (int j = 0; j < NK; j++) {
        size_t o = g_off[j] - fb * 64;
        pblk[j] = (u32)(o / 64); pwd[j] = (u32)((o % 64) / 4);
        isc[pblk[j]] = 0;
        put32(&tail[o], 1);
    }
    for (size_t b = 0; b < nb; b++) {
        u32 w[64]; expand_h(&tail[b * 64], w);
        for (int i = 0; i < 16; i++) img[b * 16 + i] = w[i];
        for (int i = 0; i < 64; i++) kw[b * 64 + i] = K256h[i] + w[i];
    }
    bool fast = nb == 2 && pblk[0] == 0 && pblk[1] == 0 && pblk[2] == 0 && pwd[0] == 9 && pwd[1] == 10 && pwd[2] == 11 && isc[1];

    u64 pv[MAXPAT] = {0}, pm[MAXPAT] = {0};
    for (size_t i = 0; i < g_pats.size(); i++) { pv[i] = g_pats[i].val; pm[i] = g_pats[i].mask; }
    u32 np = (u32)g_pats.size(), nb32 = (u32)nb;
    u64 zthr = g_zeros == 0 ? 0 : g_zeros >= 16 ? 1 : (1ull << (64 - 4 * g_zeros));

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { fprintf(stderr, "namemcgpu: no CUDA device\n"); return 6; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    CK(cudaMemcpyToSymbol(c_mid, mid, sizeof mid));
    CK(cudaMemcpyToSymbol(c_img, img.data(), MAXNB * 16 * 4));
    CK(cudaMemcpyToSymbol(c_kw, kw.data(), MAXNB * 64 * 4));
    CK(cudaMemcpyToSymbol(c_isc, isc.data(), MAXNB * 4));
    CK(cudaMemcpyToSymbol(c_nb, &nb32, 4));
    CK(cudaMemcpyToSymbol(c_np, &np, 4));
    CK(cudaMemcpyToSymbol(c_pblk, pblk, sizeof pblk));
    CK(cudaMemcpyToSymbol(c_pwd, pwd, sizeof pwd));
    CK(cudaMemcpyToSymbol(c_pval, pv, sizeof pv));
    CK(cudaMemcpyToSymbol(c_pmask, pm, sizeof pm));
    CK(cudaMemcpyToSymbol(c_zthr, &zthr, 8));

    auto launch = [&](bool test, u64 base, const u64 *tc, u32 n, u64 *out) {
        u32 grid = (n + BLOCK - 1) / BLOCK;
        if (fast) { if (test) fast_kernel<1><<<grid, BLOCK>>>(base, tc, n, out); else fast_kernel<0><<<grid, BLOCK>>>(base, tc, n, out); }
        else { if (test) general_kernel<1><<<grid, BLOCK>>>(base, tc, n, out); else general_kernel<0><<<grid, BLOCK>>>(base, tc, n, out); }
        CK(cudaGetLastError());
    };

    /* correctness gate: the search kernel must reproduce CPU-computed ids */
    {
        std::vector<u64> tcs = {0, 1, 1023, 1024, 0xFFFFFFull, 0x1000000ull, 0xFFFFFFFFFFFFull, 0x1000000000000ull,
                                0xFFFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0x00FF00FF00FF00FFull};
        u64 r = seed ^ 0xA5A5A5A55A5A5A5Aull;
        while (tcs.size() < 96) {
            r += 0x9E3779B97F4A7C15ull;
            u64 z = r; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
            tcs.push_back(z);
        }
        u32 nt = (u32)tcs.size();
        u64 *d_tc, *d_out;
        CK(cudaMalloc(&d_tc, nt * 8)); CK(cudaMalloc(&d_out, nt * 8));
        CK(cudaMemcpy(d_tc, tcs.data(), nt * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_out, 0xAB, nt * 8));
        launch(true, 0, d_tc, nt, d_out);
        CK(cudaDeviceSynchronize());
        std::vector<u64> got(nt);
        CK(cudaMemcpy(got.data(), d_out, nt * 8, cudaMemcpyDeviceToHost));
        cudaFree(d_tc); cudaFree(d_out);
        int bad = 0;
        for (u32 i = 0; i < nt; i++) if (got[i] != ref_id(tcs[i])) bad++;
        if (bad) { fprintf(stderr, "namemcgpu: GATE FAILED: %d of %u test vectors differ between GPU and CPU\n", bad, nt); return 5; }
        fprintf(stderr, "gate: GPU reproduced %u/%u CPU test vectors exactly\n", nt, nt);
    }

    u64 base = seed & ~(u64)(INNER - 1);
    u32 per = fast ? INNER : 1;
    u32 threads = fast ? (1u << 14) : (1u << 20);
    fprintf(stderr, "device=%s (sm_%d%d, %d SMs) kernel=%s knobs=(%d,%d)(%d,%d)(%d,%d) blocks=%zu patterns=%zu zeros>=%u seed=%llu\n", prop.name,
            prop.major, prop.minor, prop.multiProcessorCount, fast ? "fast" : "general", g_kx[0], g_ky[0], g_kx[1], g_ky[1], g_kx[2], g_ky[2],
            nb, g_pats.size(), g_zeros, (unsigned long long)seed);
    fflush(stderr);

    u32 zero = 0; u64 allones = ~0ull;
    CK(cudaMemcpyToSymbol(d_nhits, &zero, 4));
    CK(cudaMemcpyToSymbol(d_best, &allones, 8));
    double t0 = now_s(), last = t0;
    u64 tried = 0, last_n = 0, nhit_total = 0;
    bool stop = false;
    while (!stop) {
        double ts = now_s();
        launch(false, base, nullptr, threads, nullptr);
        CK(cudaDeviceSynchronize());
        tried += (u64)threads * per;
        base += (u64)threads * per;
        double te = now_s();
        if (te - ts < 0.03 && threads < (1u << 26)) threads *= 2;
        else if (te - ts > 0.12 && threads > 4096) threads /= 2;

        u32 nh = 0;
        CK(cudaMemcpyFromSymbol(&nh, d_nhits, 4));
        if (nh) {
            std::vector<u64> hits(nh < MAXHITS ? nh : MAXHITS);
            CK(cudaMemcpyFromSymbol(hits.data(), d_hits, hits.size() * 8));
            CK(cudaMemcpyToSymbol(d_nhits, &zero, 4));
            if (bench) hits.clear();
            if (nh > MAXHITS) fprintf(stderr, "warning: %u hits in one launch, only %d kept\n", nh, MAXHITS);
            for (u64 c : hits) {
                u8 sha[32]; u64 id = ref_id(c, sha);
                std::string why;
                if (!match_why(id, why)) { fprintf(stderr, "namemcgpu: GPU hit counter %llu fails CPU re-check (ignored)\n", (unsigned long long)c); continue; }
                char hex[65], id16[17];
                for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", sha[i]);
                snprintf(id16, sizeof id16, "%016llx", (unsigned long long)id);
                std::string path = outdir + "/" + id16 + ".png";
                std::string cmd = "\"" + render_exe + "\" \"" + rgba_path + "\" \"" + outdir + "\" -k " + kopt + " --render " +
                                  std::to_string(c) + " --expect " + id16;
                if (run_cmd(cmd) != 0) { fprintf(stderr, "namemcgpu: rendering the PNG failed (%s)\n", cmd.c_str()); return 1; }
                printf("HIT %s sha=%s id=%s counter=%llu file=%s\n", why.c_str(), hex, id16, (unsigned long long)c, path.c_str());
                fflush(stdout);
                nhit_total++;
                if (max_hits > 0 && (long long)nhit_total >= max_hits) { stop = true; break; }
            }
        }
        double t = now_s();
        if (bench && t - t0 >= 10) stop = true;
        if (t - last >= 5 || stop) {
            u64 bst = 0; CK(cudaMemcpyFromSymbol(&bst, d_best, 8));
            int bz = bst ? clz64(bst) / 4 : 16;
            fprintf(stderr, "[%7.0fs] tried=%llu  rate=%.0f/s (avg %.0f/s)  hits=%llu  best_zeros=%d\n", t - t0,
                    (unsigned long long)tried, (tried - last_n) / (t - last), tried / (t - t0), (unsigned long long)nhit_total, bz);
            fflush(stderr);
            last = t; last_n = tried;
        }
    }
    double el = now_s() - t0;
    fprintf(stderr, "done: tried=%llu in %.1fs = %.0f/s\n", (unsigned long long)tried, el, tried / el);
    if (bench) printf("RATE %.0f\n", tried / el);
    return 0;
}
