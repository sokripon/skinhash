/*
 * fastfinder: permutation-based search for Minecraft skin texture ids.
 *
 * The last K pixels (row 63, alpha 0) get 3K distinct byte values that occur nowhere
 * else in the raw scanlines. Every candidate is a permutation of that value set, so:
 *   - the tail is all literals (every 3-byte window holds a unique byte),
 *   - the literal frequencies are the same, so zlib's Huffman trees, block header and
 *     total bit length do not change,
 *   - only the tail literal bits, Adler-32 and the IDAT CRC differ.
 * A candidate emits those, then finishes SHA-256 from a midstate over the fixed whole
 * 64-byte blocks of the file.
 *
 * usage: fastfinder <raw.bin> <K> <outdir> [-p HEX]... [-s HEX]... [-z N] [-t N] [-S SEED] [-n N] [-b]
 * exit code 4: a structural assumption failed; the caller falls back to hashfinder
 */
#define _POSIX_C_SOURCE 200809L
#include "common.h"
#include <setjmp.h>

#define MAXK 20
#define MAXN (3 * MAXK)

static uint8_t g_raw[RAW_LEN];
static int g_K, g_n, g_split;
static int g_knob[MAXN], g_alpha_off[MAXK];
static uint8_t g_vals[MAXN];
static int g_zeros = 0;
static long long g_max_hits = -1;
static const char *g_outdir;
static uint64_t g_seed;
static int g_threads;
static atomic_ullong g_tried;
static atomic_llong g_hits;
static atomic_int g_best_zeros;
static atomic_int g_stop;
static int g_nosave;           /* benchmark mode: never write files */
static const char *g_dump;     /* --dump-job FILE: write the GPU job and exit */
static int g_auto;             /* --auto: with --dump-job, look for K <= given K and a value set with nblk == 1 */
static pthread_mutex_t g_io = PTHREAD_MUTEX_INITIALIZER;

static const uint8_t PNG_SIG[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
static const uint8_t IEND[12] = {0,0,0,0,'I','E','N','D',0xAE,0x42,0x60,0x82};

static int g_probe;            /* auto-job search: UNSUPPORTED jumps back instead of exiting */
static jmp_buf g_jb;
#define UNSUPPORTED(...) do { if (g_probe) longjmp(g_jb, 1); fprintf(stderr, "fastfinder: unsupported: " __VA_ARGS__); fputc('\n', stderr); exit(4); } while (0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void hexify(const uint8_t d[32], char out[65]) {
    static const char *hx = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[2*i] = hx[d[i] >> 4]; out[2*i+1] = hx[d[i] & 15]; }
    out[64] = 0;
}

static inline uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint32_t rng_below(uint64_t *s, uint32_t n) {
    return (uint32_t)(((rng_next(s) >> 32) * (uint64_t)n) >> 32);
}
static void shuffle(uint8_t *a, int n, uint64_t *s) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)rng_below(s, (uint32_t)(i + 1));
        uint8_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

/* reference encoder: same settings as Mojang */
static size_t build_header(uint8_t *f) {
    memcpy(f, PNG_SIG, 8);
    uint8_t ihdr[17] = {'I','H','D','R', 0,0,0,W, 0,0,0,H, 8, 6, 0, 0, 0};
    f[8] = 0; f[9] = 0; f[10] = 0; f[11] = 13;
    memcpy(f + 12, ihdr, 17);
    uint32_t crc = crc32(0, ihdr, 17);
    f[29] = crc >> 24; f[30] = crc >> 16; f[31] = crc >> 8; f[32] = crc;
    return 33;
}
#define FILE_CAP (33 + 8 + (RAW_LEN + RAW_LEN / 1000 + 128) + 4 + 12)

static size_t real_png(const uint8_t *raw, uint8_t *file) {
    size_t hdr = build_header(file);
    uint8_t *zout = file + hdr + 8;
    size_t zcap = compressBound(RAW_LEN) + 64;
    z_stream z;
    memset(&z, 0, sizeof z);
    if (deflateInit2(&z, 4, Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY) != Z_OK) { fprintf(stderr, "deflateInit2 failed\n"); exit(2); }
    z.next_in = (Bytef *)raw; z.avail_in = RAW_LEN;
    z.next_out = zout; z.avail_out = (uInt)zcap;
    if (deflate(&z, Z_NO_FLUSH) != Z_OK) { fprintf(stderr, "deflate failed\n"); exit(2); }
    if (deflate(&z, Z_FINISH) != Z_STREAM_END) { fprintf(stderr, "deflate finish failed\n"); exit(2); }
    size_t zlen = zcap - z.avail_out;
    deflateEnd(&z);
    uint8_t *p = file + hdr;
    p[0] = zlen >> 24; p[1] = zlen >> 16; p[2] = zlen >> 8; p[3] = zlen;
    memcpy(p + 4, "IDAT", 4);
    uint32_t crc = crc32(crc32(0, (const uint8_t *)"IDAT", 4), zout, (uInt)zlen);
    p = zout + zlen;
    p[0] = crc >> 24; p[1] = crc >> 16; p[2] = crc >> 8; p[3] = crc;
    memcpy(p + 4, IEND, 12);
    return (size_t)(p + 16 - file);
}

/* small deflate parser, puff style */
typedef struct { const uint8_t *in; size_t len; size_t pos; } bs_t;   /* pos in bits */
typedef struct { short count[16]; short symbol[320]; } huff_t;

static int getbits(bs_t *s, int n) {
    int v = 0;
    for (int i = 0; i < n; i++) {
        if ((s->pos >> 3) >= s->len) return -1;
        v |= ((s->in[s->pos >> 3] >> (s->pos & 7)) & 1) << i;
        s->pos++;
    }
    return v;
}
static void construct(huff_t *h, const uint8_t *len, int n) {
    short offs[16];
    for (int l = 0; l < 16; l++) h->count[l] = 0;
    for (int i = 0; i < n; i++) h->count[len[i]]++;
    offs[1] = 0;
    for (int l = 1; l < 15; l++) offs[l + 1] = offs[l] + h->count[l];
    for (int i = 0; i < n; i++) if (len[i]) h->symbol[offs[len[i]]++] = (short)i;
}
static int decode(bs_t *s, const huff_t *h) {
    int code = 0, first = 0, index = 0;
    for (int l = 1; l <= 15; l++) {
        int b = getbits(s, 1);
        if (b < 0) return -1;
        code |= b;
        int count = h->count[l];
        if (code - count < first) return h->symbol[index + (code - first)];
        index += count; first += count; first <<= 1; code <<= 1;
    }
    return -1;
}

static const uint16_t LBASE[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
static const uint16_t LEXT[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
static const uint16_t DBASE[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
static const uint16_t DEXT[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

typedef struct {
    uint8_t litlen[288];
    size_t T_bits;     /* bit offset (in deflate data) of the symbol producing raw[split] */
    size_t E_bits;     /* bit offset right after the EOB code */
    int ntail;         /* literal symbols from split to EOB */
} parsed_t;

/* z: deflate data (after the 2-byte zlib header), len includes the trailing adler32 */
static void parse_deflate(const uint8_t *z, size_t len, const uint8_t *raw, int split, parsed_t *P) {
    bs_t s = {z, len, 0};
    static uint8_t out[RAW_LEN + 400];
    size_t op = 0;
    huff_t lc, dc;
    uint8_t lens[320];
    int final = getbits(&s, 1), type = getbits(&s, 2);
    if (final != 1) UNSUPPORTED("first deflate block is not final (multiple blocks)");
    if (type == 1) {
        int i = 0;
        for (; i < 144; i++) lens[i] = 8;
        for (; i < 256; i++) lens[i] = 9;
        for (; i < 280; i++) lens[i] = 7;
        for (; i < 288; i++) lens[i] = 8;
        construct(&lc, lens, 288);
        uint8_t dl[30]; memset(dl, 5, sizeof dl);
        construct(&dc, dl, 30);
        memcpy(P->litlen, lens, 288);
    } else if (type == 2) {
        static const short order[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
        int nlen = getbits(&s, 5) + 257, ndist = getbits(&s, 5) + 1, ncode = getbits(&s, 4) + 4;
        if (nlen > 286 || ndist > 30) UNSUPPORTED("bad dynamic header counts");
        uint8_t cl[19] = {0};
        for (int i = 0; i < ncode; i++) cl[order[i]] = (uint8_t)getbits(&s, 3);
        huff_t hc;
        construct(&hc, cl, 19);
        int idx = 0;
        memset(lens, 0, sizeof lens);
        while (idx < nlen + ndist) {
            int sym = decode(&s, &hc);
            if (sym < 0) UNSUPPORTED("bad code-length code");
            if (sym < 16) lens[idx++] = (uint8_t)sym;
            else {
                int l = 0, rep;
                if (sym == 16) { if (!idx) UNSUPPORTED("repeat with no previous"); l = lens[idx - 1]; rep = 3 + getbits(&s, 2); }
                else if (sym == 17) rep = 3 + getbits(&s, 3);
                else rep = 11 + getbits(&s, 7);
                if (idx + rep > nlen + ndist) UNSUPPORTED("too many code lengths");
                while (rep--) lens[idx++] = (uint8_t)l;
            }
        }
        memset(P->litlen, 0, 288);
        memcpy(P->litlen, lens, nlen);
        construct(&lc, lens, nlen);
        construct(&dc, lens + nlen, ndist);
    } else UNSUPPORTED("stored/invalid block type %d", type);

    P->T_bits = (size_t)-1;
    P->ntail = 0;
    for (;;) {
        size_t start = s.pos;
        int sym = decode(&s, &lc);
        if (sym < 0) UNSUPPORTED("bad literal/length code");
        if (op == (size_t)split && P->T_bits == (size_t)-1) P->T_bits = start;
        if (op > (size_t)split && P->T_bits == (size_t)-1) UNSUPPORTED("a match straddles the first knob byte");
        if (sym == 256) { P->E_bits = s.pos; break; }
        if (sym < 256) {
            if (op >= RAW_LEN) UNSUPPORTED("output overflow");
            out[op++] = (uint8_t)sym;
            if (P->T_bits != (size_t)-1) P->ntail++;
        } else {
            if (P->T_bits != (size_t)-1) UNSUPPORTED("match symbol inside the tail");
            sym -= 257;
            if (sym >= 29) UNSUPPORTED("bad length symbol");
            int l = LBASE[sym] + getbits(&s, LEXT[sym]);
            int ds = decode(&s, &dc);
            if (ds < 0 || ds >= 30) UNSUPPORTED("bad distance symbol");
            size_t dist = DBASE[ds] + (size_t)getbits(&s, DEXT[ds]);
            if (dist > op || op + l > RAW_LEN) UNSUPPORTED("bad match");
            for (int i = 0; i < l; i++, op++) out[op] = out[op - dist];
        }
    }
    if (op != RAW_LEN || memcmp(out, raw, RAW_LEN)) UNSUPPORTED("parser did not reproduce the raw data");
    if (P->T_bits == (size_t)-1) UNSUPPORTED("no symbol starts at the first knob byte");
    if (P->ntail != RAW_LEN - split) UNSUPPORTED("tail is not all literals");
    if (((P->E_bits + 7) >> 3) + 4 != len) UNSUPPORTED("trailing data after the block");
}

/* fast path */
typedef struct {
    uint32_t tab[257];          /* reversed code | len << 16 */
    uint8_t alpha[MAXK];
    uint32_t eob;
    uint64_t firstbits;
    int rem;
    size_t V0, nvar, flen, nblk, B0;
    uint32_t mid[8];
    uint32_t crc0, adl_a, adl_b;
    uint8_t *ref_file;
    uint8_t *tmpl;
} fast_t;
static fast_t F;

#define PUT(e) do { uint32_t t_ = (e); acc |= (uint64_t)(t_ & 0xFFFF) << nb; nb += (int)(t_ >> 16); \
    memcpy(o, &acc, 8); o += nb >> 3; acc >>= (nb & ~7); nb &= 7; } while (0)

static inline void build_buf(const uint8_t *p, uint8_t *buf) {
    uint8_t tmp[320];
    uint64_t acc = F.firstbits;
    int nb = F.rem;
    uint8_t *o = tmp;
    uint32_t a = F.adl_a, b = F.adl_b;
    for (int i = 0; i < g_K; i++) {
        uint32_t v0 = p[3*i], v1 = p[3*i+1], v2 = p[3*i+2], v3 = F.alpha[i];
        PUT(F.tab[v0]); PUT(F.tab[v1]); PUT(F.tab[v2]); PUT(F.tab[v3]);
        a += v0; b += a; a += v1; b += a; a += v2; b += a; a += v3; b += a;
    }
    PUT(F.eob);
    memcpy(o, &acc, 8);
    o += (nb + 7) >> 3;
    a %= 65521; b %= 65521;
    o[0] = (uint8_t)(b >> 8); o[1] = (uint8_t)b; o[2] = (uint8_t)(a >> 8); o[3] = (uint8_t)a;
    uint32_t c = (uint32_t)crc32(F.crc0, tmp, (uInt)(o + 4 - tmp));
    o[4] = (uint8_t)(c >> 24); o[5] = (uint8_t)(c >> 16); o[6] = (uint8_t)(c >> 8); o[7] = (uint8_t)c;
    memcpy(buf + (F.V0), tmp, F.nvar);
}

static void set_raw(uint8_t *raw, const uint8_t *perm) {
    for (int i = 0; i < g_n; i++) raw[g_knob[i]] = perm[i];
}

/* Build the file from prefix + buf and compare it with real zlib. */
static void check_candidate(const uint8_t *perm, const uint8_t *buf, const char *what) {
    static __thread uint8_t raw[RAW_LEN], f1[FILE_CAP], f2[FILE_CAP];
    memcpy(raw, g_raw, RAW_LEN);
    set_raw(raw, perm);
    size_t l2 = real_png(raw, f2);
    memcpy(f1, F.ref_file, F.B0);
    memcpy(f1 + F.B0, buf, F.flen - F.B0);
    if (l2 != F.flen || memcmp(f1, f2, F.flen)) {
        fprintf(stderr, "fastfinder: MISMATCH vs real zlib (%s): real len %zu fast len %zu\n", what, l2, F.flen);
        for (size_t i = 0; i < l2 && i < F.flen; i++) if (f1[i] != f2[i]) { fprintf(stderr, "first diff at byte %zu (var region starts at %zu)\n", i, F.V0 + F.B0); break; }
        exit(3);
    }
}

static void setup_fast(uint64_t *rng) {
    free(F.tmpl); free(F.ref_file); F.tmpl = NULL; F.ref_file = NULL;
    /* choose the value set: byte values absent from the raw data (knob bytes excluded) */
    int cnt[256] = {0};
    uint8_t is_knob[RAW_LEN] = {0};
    for (int i = 0; i < g_n; i++) is_knob[g_knob[i]] = 1;
    for (int i = 0; i < RAW_LEN; i++) if (!is_knob[i]) cnt[g_raw[i]]++;
    uint8_t unused[256];
    int nun = 0;
    for (int v = 0; v < 256; v++) if (!cnt[v]) unused[nun++] = (uint8_t)v;
    if (nun < g_n) UNSUPPORTED("only %d unused byte values, need %d (lower K)", nun, g_n);
    shuffle(unused, nun, rng);
    memcpy(g_vals, unused, g_n);

    static uint8_t raw[RAW_LEN], file[FILE_CAP];
    memcpy(raw, g_raw, RAW_LEN);
    set_raw(raw, g_vals);
    size_t flen = real_png(raw, file);
    size_t zlen = ((size_t)file[33] << 24 | file[34] << 16 | file[35] << 8 | file[36]);
    const uint8_t *zs = file + 41;
    if (zs[0] != 0x78) UNSUPPORTED("unexpected zlib header");
    parsed_t P;
    parse_deflate(zs + 2, zlen - 2, raw, g_split, &P);

    /* canonical codes from code lengths, bit-reversed for LSB-first output */
    int blc[16] = {0};
    for (int i = 0; i < 288; i++) blc[P.litlen[i]]++;
    blc[0] = 0;
    uint32_t next[16] = {0}, code = 0;
    for (int b = 1; b < 16; b++) { code = (code + blc[b - 1]) << 1; next[b] = code; }
    for (int i = 0; i < 257; i++) {
        int l = P.litlen[i];
        if (!l) { F.tab[i] = 0; continue; }
        uint32_t c = next[l]++, r = 0;
        for (int k = 0; k < l; k++) r |= ((c >> k) & 1) << (l - 1 - k);
        F.tab[i] = r | ((uint32_t)l << 16);
    }
    F.eob = F.tab[256];
    for (int i = 0; i < g_K; i++) {
        F.alpha[i] = g_raw[g_alpha_off[i]];
        if (!F.tab[F.alpha[i]]) UNSUPPORTED("alpha literal has no code");
    }

    size_t Tfile = (size_t)43 * 8 + P.T_bits, Efile = (size_t)43 * 8 + P.E_bits;
    size_t vb = Tfile >> 3;
    F.rem = (int)(Tfile & 7);
    F.firstbits = file[vb] & ((1u << F.rem) - 1);
    size_t padded_end = (Efile + 7) >> 3;
    F.nvar = (padded_end - vb) + 8;
    F.flen = flen;
    if (padded_end + 8 + 12 != flen) UNSUPPORTED("file layout mismatch");
    F.crc0 = (uint32_t)crc32(crc32(0, (const uint8_t *)"IDAT", 4), file + 41, (uInt)(vb - 41));
    uint64_t a = 1, b = 0;
    for (int i = 0; i < g_split; i++) { a += raw[i]; b += a; }
    F.adl_a = a % 65521; F.adl_b = b % 65521;
    /* Adler-32 is the last 4 zdata bytes; check it against the reference. */
    uint64_t a2 = F.adl_a, b2 = F.adl_b;
    for (int i = g_split; i < RAW_LEN; i++) { a2 += raw[i]; b2 += a2; }
    a2 %= 65521; b2 %= 65521;
    if (file[padded_end] != (b2 >> 8) || file[padded_end + 3] != (a2 & 255)) UNSUPPORTED("adler position mismatch");

    F.B0 = (vb / 64) * 64;
    F.V0 = vb - F.B0;
    size_t total = ((flen + 9 + 63) / 64) * 64;
    F.nblk = (total - F.B0) / 64;
    F.mid[0] = 0x6a09e667; F.mid[1] = 0xbb67ae85; F.mid[2] = 0x3c6ef372; F.mid[3] = 0xa54ff53a;
    F.mid[4] = 0x510e527f; F.mid[5] = 0x9b05688c; F.mid[6] = 0x1f83d9ab; F.mid[7] = 0x5be0cd19;
    sha256_block(F.mid, file, F.B0 / 64);
    F.tmpl = calloc(1, total - F.B0 + 64);
    memcpy(F.tmpl, file + F.B0, flen - F.B0);
    F.tmpl[flen - F.B0] = 0x80;
    uint64_t bits = (uint64_t)flen * 8;
    for (int i = 0; i < 8; i++) F.tmpl[total - F.B0 - 1 - i] = (uint8_t)(bits >> (8 * i));
    F.ref_file = malloc(flen);
    memcpy(F.ref_file, file, flen);
    if (!g_probe) fprintf(stderr, "fast path: file=%zu bytes, var region from byte %zu (%d bit offset), sha blocks/candidate=%zu, "
            "tail symbols=%d, litlens ok\n", flen, vb, F.rem, F.nblk, P.ntail);
}


/* Set the knob layout to the last K pixels of row 63. */
static void configure_k(int K) {
    g_K = K; g_n = 3 * K;
    for (int i = 0; i < g_K; i++) {
        int x = W - g_K + i, base = (H - 1) * ROW + 1 + x * 4;
        for (int j = 0; j < 3; j++) g_knob[3 * i + j] = base + j;
        g_alpha_off[i] = base + 3;
    }
    g_split = g_knob[0];
}

/* --auto: find a (K, value set) whose variable region plus SHA padding fit in one 64-byte
 * block, which halves the GPU work. Tries the largest K <= kmax first, several random value
 * sets per K. Otherwise falls back to the best multi-block configuration of the largest K
 * that works at all. */
#define AUTO_TRIES 400
static int try_setup(uint64_t *rng) {
    int ok = 0;
    g_probe = 1;
    if (setjmp(g_jb) == 0) { setup_fast(rng); ok = 1; }
    g_probe = 0;
    return ok;
}
static void auto_choose(uint64_t *rng) {
    int kmax = g_K, fb_k = -1, fb_n = 1 << 30;
    uint64_t fb_rng = 0;
    for (int K = kmax; K >= 2; K--) {
        int ok_cnt = 0;
        for (int t = 0; t < AUTO_TRIES; t++) {
            uint64_t saved = *rng;
            configure_k(K);
            if (!try_setup(rng)) continue;
            ok_cnt++;
            if (F.nblk == 1) {
                fprintf(stderr, "auto: K=%d value set #%d gives nblk=1 (var region at block offset %zu, %zu bytes)\n", K, t + 1, F.V0, F.nvar);
                return;
            }
            if (fb_k < 0 || (K == fb_k && (int)F.nblk < fb_n)) { fb_k = K; fb_n = (int)F.nblk; fb_rng = saved; }
        }
        fprintf(stderr, "auto: K=%d: no single-block value set in %d tries (%d usable)\n", K, AUTO_TRIES, ok_cnt);
        if (fb_k >= 0 && K <= fb_k - 2) break;   /* stay near the first usable K */
    }
    if (fb_k < 0) UNSUPPORTED("no usable K / value set found in auto mode");
    configure_k(fb_k);
    *rng = fb_rng;
    setup_fast(rng);
    fprintf(stderr, "auto: fallback to K=%d, nblk=%zu\n", fb_k, F.nblk);
}

static void self_test(uint64_t *rng) {
    uint8_t perm[MAXN];
    uint8_t *buf = malloc(F.nblk * 64 + 64);
    memcpy(buf, F.tmpl, F.nblk * 64);
    memcpy(perm, g_vals, g_n);
    for (int it = 0; it < 1000; it++) {
        shuffle(perm, g_n, rng);
        build_buf(perm, buf);
        check_candidate(perm, buf, "self-test");
        /* The midstate path must equal a full SHA-256 of the real file. */
        uint32_t s[8];
        memcpy(s, F.mid, sizeof s);
        sha256_block(s, buf, F.nblk);
        uint8_t d1[32], d2[32];
        for (int i = 0; i < 8; i++) { d1[4*i] = s[i] >> 24; d1[4*i+1] = s[i] >> 16; d1[4*i+2] = s[i] >> 8; d1[4*i+3] = s[i]; }
        static __thread uint8_t f[FILE_CAP];
        memcpy(f, F.ref_file, F.B0);
        memcpy(f + F.B0, buf, F.flen - F.B0);
        sha256(f, F.flen, d2);
        if (memcmp(d1, d2, 32)) { fprintf(stderr, "fastfinder: SHA midstate mismatch\n"); exit(3); }
    }
    free(buf);
    fprintf(stderr, "self-test: 1000 random permutations identical to real zlib output\n");
}

/* GPU job dump */
/* Index -> permutation mapping shared with gpufinder. m = min(n,20) values are permuted (values m..n-1 stay).
 * idx = outer * R! + t, with R = inner_r() (=GPU_R) and t < R!.
 *  - outer: Lehmer code over the first m-R positions: digit i (least significant first) has radix m-i and
 *    picks the d-th smallest still-free value index;
 *  - the R values left over (ascending value index) fill the last R positions in the t-th lexicographic
 *    permutation of 0..R-1.
 * The GPU enumerates the t's in the inner loop and caches everything that depends only on `outer`. */
#ifndef GPU_R
#define GPU_R 4
#endif
static int perm_m(void) { return g_n < 20 ? g_n : 20; }
static int inner_r(void) { return perm_m() < GPU_R ? perm_m() : GPU_R; }
static uint64_t fact_u(int r) { uint64_t f = 1; for (int i = 2; i <= r; i++) f *= (uint64_t)i; return f; }
static uint64_t perm_count(void) { return fact_u(perm_m()); }
static void decode_perm(uint64_t idx, uint8_t *perm) {
    int m = perm_m(), R = inner_r();
    uint64_t RF = fact_u(R), o = idx / RF, t = idx % RF;
    uint8_t rem[MAXN];
    memcpy(rem, g_vals, m);
    for (int i = 0; i < m - R; i++) {
        int r = m - i, d = (int)(o % (uint64_t)r);
        o /= (uint64_t)r;
        perm[i] = rem[d];
        memmove(rem + d, rem + d + 1, (size_t)(r - 1 - d));
    }
    int avail[8];
    for (int i = 0; i < R; i++) avail[i] = i;
    for (int q = 0; q < R; q++) {
        uint64_t f = fact_u(R - 1 - q);
        int d = (int)(t / f);
        t %= f;
        perm[m - R + q] = rem[avail[d]];
        for (int j = d; j < R - q - 1; j++) avail[j] = avail[j + 1];
    }
    for (int i = m; i < g_n; i++) perm[i] = g_vals[i];
}
static void w32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void dump_job(const char *path, uint64_t *rng) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    w32(f, 0x324A4B53u);   /* "SKJ2" */
    w32(f, (uint32_t)g_K); w32(f, (uint32_t)g_n); w32(f, (uint32_t)perm_m()); w32(f, (uint32_t)inner_r());
    w32(f, (uint32_t)F.flen); w32(f, (uint32_t)F.B0); w32(f, (uint32_t)F.V0); w32(f, (uint32_t)F.nvar);
    w32(f, (uint32_t)F.nblk); w32(f, (uint32_t)F.rem); w32(f, F.eob);
    w32(f, F.crc0); w32(f, F.adl_a); w32(f, F.adl_b); w64(f, F.firstbits);
    for (int i = 0; i < 8; i++) w32(f, F.mid[i]);
    for (int i = 0; i < 257; i++) w32(f, F.tab[i]);
    uint8_t pad[MAXN] = {0};
    memcpy(pad, F.alpha, g_K); fwrite(pad, 1, MAXK, f);
    memset(pad, 0, sizeof pad); memcpy(pad, g_vals, g_n); fwrite(pad, 1, MAXN, f);
    w32(f, (uint32_t)(F.nblk * 64)); fwrite(F.tmpl, 1, F.nblk * 64, f);
    w32(f, (uint32_t)F.flen); fwrite(F.ref_file, 1, F.flen, f);
    w32(f, (uint32_t)g_zeros); w32(f, (uint32_t)npats);
    for (int i = 0; i < npats; i++) {
        w64(f, pats[i].val); w64(f, pats[i].mask); w32(f, (uint32_t)pats[i].suffix);
        char t[20] = {0}; memcpy(t, pats[i].text, strlen(pats[i].text)); fwrite(t, 1, 20, f);
    }
    /* test vectors: index and the CPU-computed SHA-256 of the whole PNG */
    uint64_t mf = perm_count();
    enum { NT = 16 };
    w32(f, NT);
    uint8_t *buf = malloc(F.nblk * 64 + 64);
    for (int t = 0; t < NT; t++) {
        uint64_t idx = t == 0 ? 0 : t == 1 ? mf - 1 : rng_next(rng) % mf;
        uint8_t perm[MAXN];
        decode_perm(idx, perm);
        memcpy(buf, F.tmpl, F.nblk * 64);
        build_buf(perm, buf);
        check_candidate(perm, buf, "job test vector");
        uint32_t s[8]; memcpy(s, F.mid, sizeof s);
        sha256_block(s, buf, F.nblk);
        uint8_t d[32];
        for (int i = 0; i < 8; i++) { d[4*i] = s[i] >> 24; d[4*i+1] = s[i] >> 16; d[4*i+2] = s[i] >> 8; d[4*i+3] = s[i]; }
        w64(f, idx); fwrite(d, 1, 32, f);
    }
    free(buf);
    fclose(f);
    fprintf(stderr, "job written to %s (K=%d, nblk=%zu, m=%d, %llu permutations per value set)\n", path, g_K, F.nblk, perm_m(), (unsigned long long)mf);
}

/* search */
static void report(const uint8_t *file, size_t flen, const uint8_t d[32], uint64_t ctr, const char *why) {
    char hex[65];
    hexify(d, hex);
    const char *url = hex;
    while (*url == '0' && url[1]) url++;
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.png", g_outdir, url);
    pthread_mutex_lock(&g_io);
    FILE *fp = g_nosave ? NULL : fopen(path, "wb");
    if (fp) { fwrite(file, 1, flen, fp); fclose(fp); }
    printf("HIT %s sha=%s id=%s counter=%llu file=%s\n", why, hex, url, (unsigned long long)ctr, path);
    fflush(stdout);
    pthread_mutex_unlock(&g_io);
}

static void *worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    uint64_t rng = g_seed + 0x1234567ULL * (uint64_t)(tid + 1);
    size_t bl = F.nblk * 64;
    uint8_t *buf = malloc(bl + 64);
    memcpy(buf, F.tmpl, bl);
    uint8_t perm[MAXN];
    memcpy(perm, g_vals, g_n);
    int m = g_n < 12 ? g_n : 12;
    uint64_t local = 0, total = 0;
    int best = atomic_load(&g_best_zeros);

    while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) {
        shuffle(perm, g_n, &rng);
        int c[12] = {0};
        int more = 1;
        while (more) {
            build_buf(perm, buf);
            uint32_t s[8];
            memcpy(s, F.mid, sizeof s);
            sha256_block(s, buf, F.nblk);
            uint64_t hi = (uint64_t)s[0] << 32 | s[1];
            uint64_t lo = (uint64_t)s[6] << 32 | s[7];
            uint64_t tp = trimmed_top64(hi, (uint64_t)s[2] << 32 | s[3]);

            const char *why = NULL;
            char whybuf[48];
            for (int i = 0; i < npats; i++) {
                uint64_t v = pats[i].suffix ? lo : tp;
                if ((v & pats[i].mask) == pats[i].val) {
                    snprintf(whybuf, sizeof whybuf, "%s:%s", pats[i].suffix ? "suffix" : "prefix", pats[i].text);
                    why = whybuf;
                    break;
                }
            }
            int zn = hi ? __builtin_clzll(hi) / 4 : 16;
            if (zn > best) {
                best = atomic_load(&g_best_zeros);
                while (zn > best && !atomic_compare_exchange_weak(&g_best_zeros, &best, zn)) {}
                if (zn > best) best = zn;
            }
            if (!why && g_zeros && zn >= g_zeros) {
                snprintf(whybuf, sizeof whybuf, "zeros:%d", zn);
                why = whybuf;
            }
            if (why) {
                check_candidate(perm, buf, "hit");
                static __thread uint8_t raw[RAW_LEN], file[FILE_CAP];
                memcpy(raw, g_raw, RAW_LEN);
                set_raw(raw, perm);
                size_t flen = real_png(raw, file);
                uint8_t d[32];
                sha256(file, flen, d);
                for (int i = 0; i < 8; i++) {
                    if (d[4*i] != (s[i] >> 24) || d[4*i+1] != (uint8_t)(s[i] >> 16) ||
                        d[4*i+2] != (uint8_t)(s[i] >> 8) || d[4*i+3] != (uint8_t)s[i]) {
                        fprintf(stderr, "fastfinder: hit digest mismatch\n"); exit(3);
                    }
                }
                report(file, flen, d, total + local, why);
                long long h = atomic_fetch_add(&g_hits, 1) + 1;
                if (g_max_hits > 0 && h >= g_max_hits) atomic_store(&g_stop, 1);
            }

            /* next permutation: one Heap swap */
            more = 0;
            for (int i = 1; i < m; i++) {
                if (c[i] < i) {
                    int k = (i & 1) ? c[i] : 0;
                    uint8_t t = perm[k]; perm[k] = perm[i]; perm[i] = t;
                    c[i]++;
                    more = 1;
                    break;
                }
                c[i] = 0;
            }
            if (++local == 4096) {
                atomic_fetch_add_explicit(&g_tried, local, memory_order_relaxed);
                total += local; local = 0;
                if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
            }
        }
    }
    atomic_fetch_add(&g_tried, local);
    return NULL;
}

static int ncpus(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1;
#endif
}

static uint64_t rand_seed(void) {
    uint64_t s = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL ^ (uint64_t)(now_s() * 1e9);
    s ^= (uint64_t)(uintptr_t)&s;
    s ^= s >> 31; s *= 0xBF58476D1CE4E5B9ULL; s ^= s >> 29;
    return s;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: fastfinder raw.bin K outdir [-p HEX]... [-s HEX]... [-z N] [-t N] [-S SEED] [-n N] [-b]\n");
        return 1;
    }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp || fread(g_raw, 1, RAW_LEN, fp) != RAW_LEN) { fprintf(stderr, "bad raw file\n"); return 1; }
    fclose(fp);
    g_K = atoi(argv[2]);
    if (g_K < 1 || g_K > MAXK) { fprintf(stderr, "K must be 1..%d\n", MAXK); return 1; }
    g_n = 3 * g_K;
    g_outdir = argv[3];
    g_threads = ncpus();
    g_seed = rand_seed();
    int bench = 0;
    for (int i = 4; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "-p") && v) { if (parse_pat(v, 0)) { fprintf(stderr, "bad -p %s\n", v); return 1; } i++; }
        else if (!strcmp(a, "-s") && v) { if (parse_pat(v, 1)) { fprintf(stderr, "bad -s %s\n", v); return 1; } i++; }
        else if (!strcmp(a, "-z") && v) { g_zeros = atoi(v); i++; }
        else if (!strcmp(a, "-t") && v) { g_threads = atoi(v); i++; }
        else if (!strcmp(a, "-S") && v) { g_seed = strtoull(v, NULL, 0); i++; }
        else if (!strcmp(a, "-n") && v) { g_max_hits = atoll(v); i++; }
        else if (!strcmp(a, "--dump-job") && v) { g_dump = v; i++; }
        else if (!strcmp(a, "--auto")) g_auto = 1;
        else if (!strcmp(a, "-b")) bench = g_nosave = 1;
        else { fprintf(stderr, "unknown option %s\n", a); return 1; }
    }
#ifdef HAVE_X86
    unsigned a, b, c, d;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & (1u << 29))) sha256_block = sha256_block_shani;
#endif
    uint8_t dg[32]; char hx[65];
    sha256((const uint8_t *)"abc", 3, dg); hexify(dg, hx);
    if (strcmp(hx, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")) {
        fprintf(stderr, "SHA-256 self-test failed\n"); return 2;
    }
    if (g_threads < 1) g_threads = 1;

    /* knob bytes: RGB of the last K pixels of row 63 */
    configure_k(g_K);

    uint64_t rng = g_seed ^ 0xA5A5A5A55A5A5A5AULL;
    if (g_dump && g_auto) auto_choose(&rng);
    else setup_fast(&rng);
    self_test(&rng);
    if (g_dump) { dump_job(g_dump, &rng); return 0; }
    fprintf(stderr, "threads=%d seed=%llu K=%d patterns=%d zeros>=%d sha=%s\n", g_threads,
            (unsigned long long)g_seed, g_K, npats, g_zeros,
            sha256_block == sha256_block_generic ? "generic" : "sha-ni");
    fflush(stderr);

    pthread_t *th = calloc(g_threads, sizeof *th);
    for (int i = 0; i < g_threads; i++) pthread_create(&th[i], NULL, worker, (void *)(intptr_t)i);
    double t0 = now_s(), last = t0;
    unsigned long long last_n = 0;
    while (!atomic_load(&g_stop)) {
#ifdef _WIN32
        Sleep(250);
#else
        struct timespec ts = {0, 250000000}; nanosleep(&ts, NULL);
#endif
        double t = now_s();
        if (bench && t - t0 >= 10) atomic_store(&g_stop, 1);
        if (t - last >= 5 || atomic_load(&g_stop)) {
            unsigned long long n = atomic_load(&g_tried);
            fprintf(stderr, "[%7.0fs] tried=%llu  rate=%.0f/s (avg %.0f/s)  hits=%lld  best_zeros=%d\n",
                    t - t0, n, (n - last_n) / (t - last), n / (t - t0), (long long)atomic_load(&g_hits),
                    atomic_load(&g_best_zeros));
            fflush(stderr);
            last = t; last_n = n;
        }
    }
    for (int i = 0; i < g_threads; i++) pthread_join(th[i], NULL);
    unsigned long long n = atomic_load(&g_tried);
    double el = now_s() - t0;
    fprintf(stderr, "done: tried=%llu in %.1fs = %.0f/s\n", n, el, n / el);
    if (bench) printf("RATE %.0f\n", n / el);
    return 0;
}
