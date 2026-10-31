/* Shared by hashfinder.c and fastfinder.c: SHA-256 (generic and SHA-NI), arena, patterns. */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <zlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#include <cpuid.h>
#define HAVE_X86 1
#endif

#define W 64
#define H 64
#define ROW (1 + W * 4)
#define RAW_LEN (ROW * H)
#define MAX_OFFS 64
#define MAX_PAT 64

/* SHA-256 */
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha256_block_generic(uint32_t s[8], const uint8_t *p, size_t nblocks) {
    while (nblocks--) {
        uint32_t w[64], a, b, c, d, e, f, g, h;
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = ROR(w[i-15], 7) ^ ROR(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = ROR(w[i-2], 17) ^ ROR(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        a = s[0]; b = s[1]; c = s[2]; d = s[3]; e = s[4]; f = s[5]; g = s[6]; h = s[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
            uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
        p += 64;
    }
}

#ifdef HAVE_X86
/* SHA-NI block function, after the public-domain Intel / J. Walton pattern. */
__attribute__((target("sha,sse4.1")))
static void sha256_block_shani(uint32_t s[8], const uint8_t *p, size_t nblocks) {
    const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i TMP = _mm_loadu_si128((const __m128i *)&s[0]);
    __m128i STATE1 = _mm_loadu_si128((const __m128i *)&s[4]);
    TMP = _mm_shuffle_epi32(TMP, 0xB1);
    STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);
    __m128i STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);
    STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0);
    while (nblocks--) {
        __m128i ABEF = STATE0, CDGH = STATE1, MSG, M0, M1, M2, M3;
#define RND2(Mi, k0, k1)                                                        \
        MSG = _mm_add_epi32(Mi, _mm_set_epi64x(k1, k0));                         \
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);                     \
        MSG = _mm_shuffle_epi32(MSG, 0x0E);                                      \
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        M0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 0)), MASK);
        M1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 16)), MASK);
        M2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 32)), MASK);
        M3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 48)), MASK);
        __m128i Ms[4] = {M0, M1, M2, M3};
        for (int r = 0; r < 16; r++) {
            __m128i Mi = Ms[r & 3];
            long long k0 = (long long)((uint64_t)K256[4*r+1] << 32 | K256[4*r]);
            long long k1 = (long long)((uint64_t)K256[4*r+3] << 32 | K256[4*r+2]);
            RND2(Mi, k0, k1);
            if (r < 12) {
                __m128i a = Ms[r & 3], b = Ms[(r+1) & 3], c = Ms[(r+2) & 3], d = Ms[(r+3) & 3];
                __m128i t = _mm_sha256msg1_epu32(a, b);
                t = _mm_add_epi32(t, _mm_alignr_epi8(d, c, 4));
                Ms[r & 3] = _mm_sha256msg2_epu32(t, d);
            }
        }
#undef RND2
        STATE0 = _mm_add_epi32(STATE0, ABEF);
        STATE1 = _mm_add_epi32(STATE1, CDGH);
        p += 64;
    }
    TMP = _mm_shuffle_epi32(STATE0, 0x1B);
    STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);
    STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);
    STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);
    _mm_storeu_si128((__m128i *)&s[0], STATE0);
    _mm_storeu_si128((__m128i *)&s[4], STATE1);
}
#endif

static void (*sha256_block)(uint32_t *, const uint8_t *, size_t) = sha256_block_generic;

static void sha256(const uint8_t *msg, size_t len, uint8_t out[32]) {
    uint32_t s[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t full = len / 64;
    sha256_block(s, msg, full);
    uint8_t tail[128] = {0};
    size_t rem = len - full * 64;
    memcpy(tail, msg + full * 64, rem);
    tail[rem] = 0x80;
    size_t tl = rem < 56 ? 64 : 128;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) tail[tl - 1 - i] = (uint8_t)(bits >> (8 * i));
    sha256_block(s, tail, tl / 64);
    for (int i = 0; i < 8; i++) {
        out[4*i] = s[i] >> 24; out[4*i+1] = s[i] >> 16; out[4*i+2] = s[i] >> 8; out[4*i+3] = s[i];
    }
}

/* Bump allocator for deflateCopy; the caller resets `used` to free everything. */
typedef struct { uint8_t *base; size_t cap, used; } arena_t;
__attribute__((unused)) static voidpf arena_alloc(voidpf op, uInt items, uInt size) {
    arena_t *a = op;
    size_t n = ((size_t)items * size + 63) & ~(size_t)63;
    if (a->used + n > a->cap) return Z_NULL;
    void *p = a->base + a->used;
    a->used += n;
    return p;
}
__attribute__((unused)) static void arena_free(voidpf op, voidpf p) { (void)op; (void)p; }

/* Patterns */
typedef struct { uint64_t val, mask; int suffix; char text[17]; } pat_t;
static pat_t pats[MAX_PAT];
static int npats = 0;

static int parse_pat(const char *hex, int suffix) {
    size_t n = strlen(hex);
    if (n == 0 || n > 16 || npats >= MAX_PAT) return -1;
    if (!suffix && hex[0] == '0') return -1;   /* prefixes match the trimmed id, which never starts with 0 */
    pat_t *p = &pats[npats];
    p->val = 0; p->mask = 0; p->suffix = suffix;
    for (size_t i = 0; i < n; i++) {
        char c = hex[i];
        int v = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
                (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (v < 0) return -1;
        int shift = suffix ? (int)(4 * (n - 1 - i)) : (int)(60 - 4 * i);
        p->val |= (uint64_t)v << shift;
        p->mask |= (uint64_t)0xF << shift;
    }
    for (size_t i = 0; i < n; i++) p->text[i] = (char)(hex[i] | 0x20);
    p->text[n] = 0;
    npats++;
    return 0;
}


/* Top 64 bits of the id with leading zero nibbles dropped, as in the texture URL.
 * hi/mid are the first 16 digest bytes as two big-endian 64-bit words. */
static inline uint64_t trimmed_top64(uint64_t hi, uint64_t mid) {
    if (hi) {
        int sh = __builtin_clzll(hi) & ~3;
        return sh ? (hi << sh) | (mid >> (64 - sh)) : hi;
    }
    if (!mid) return 0;
    return mid << (__builtin_clzll(mid) & ~3);
}
