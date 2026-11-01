/*
 * namemcfinder: CPU search for NameMC skin ids.
 *
 * NameMC id = first 8 bytes (16 hex) of
 *   SHA-256( be32(64) || be32(64) || RGBA pixels row-major, RGB set to 0 where alpha == 0 ).
 * Knob: three pixels whose alpha is 1 and whose RGB is a counter. The message is 16392 bytes, so the
 * counter bytes sit in the last blocks; SHA-256 resumes from a midstate over the fixed blocks before them.
 *
 * Counter c (64 bit) -> pixels, knob pixels in message order kp[0] < kp[1] < kp[2]:
 *   kp[2] RGB = c & 0xFFFFFF,  kp[1] RGB = (c >> 24) & 0xFFFFFF,  kp[0] RGB = c >> 48
 *
 * usage: namemcfinder rgba.bin OUTDIR -k x,y,x,y,x,y [-p HEX]... [-s HEX]... [-c HEX]... [-z N] [-t N] [-S SEED]
 *                     [-n N] [-b]
 *        namemcfinder rgba.bin OUTDIR -k ... --render COUNTER --expect ID16   (write one PNG, used by namemcgpu)
 * rgba.bin: 64*64*4 bytes, the skin with the three knob pixels at alpha 1.
 * Writes OUTDIR/<id16>.png in Mojang's encoding for every hit.
 */
#define _POSIX_C_SOURCE 200809L
#include "common.h"

#define NPIX (W * H)
#define MSG_LEN (8 + NPIX * 4)
#define NK 3
#define CHUNK (1ull << 16)
#define FILE_CAP_ (33 + 8 + (RAW_LEN + RAW_LEN / 1000 + 128) + 4 + 12)

static uint8_t g_rgba[NPIX * 4];
static int g_kx[NK], g_ky[NK];
static size_t g_off[NK];             /* byte offset of each knob pixel in the message */
static uint8_t g_msg[MSG_LEN];        /* message with knob RGB = 0 */
static size_t g_fb, g_nblk;           /* first block containing a knob, blocks hashed per candidate */
static uint32_t g_mid[8];
static uint8_t g_tail[(MSG_LEN / 64 + 3) * 64];
static size_t g_woff[NK];             /* knob word offset inside g_tail */
static int g_zeros = 0;
static long long g_max_hits = -1;
static const char *g_outdir;
static int g_threads, g_nosave;
static uint64_t g_seed;
static atomic_ullong g_tried, g_next;
static atomic_llong g_hits;
static atomic_int g_stop, g_best_zeros;
static pthread_mutex_t g_io = PTHREAD_MUTEX_INITIALIZER;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void hex16(uint64_t id, char out[17]) { snprintf(out, 17, "%016llx", (unsigned long long)id); }

static uint64_t rand_seed(void) {
    uint64_t s = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL ^ (uint64_t)(now_s() * 1e9);
    s ^= (uint64_t)(uintptr_t)&s;
    s ^= s >> 31; s *= 0xBF58476D1CE4E5B9ULL; s ^= s >> 29;
    return s;
}

static void put32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }

/* message for counter c, written to m (MSG_LEN bytes) from the base message */
static void fill_counter(uint8_t *m, uint64_t c) {
    uint32_t v[NK] = {(uint32_t)(c >> 48), (uint32_t)(c >> 24) & 0xFFFFFF, (uint32_t)c & 0xFFFFFF};
    for (int j = 0; j < NK; j++) { m[g_off[j]] = v[j] >> 16; m[g_off[j] + 1] = v[j] >> 8; m[g_off[j] + 2] = v[j]; }
}

static uint64_t id_of_message(const uint8_t *m) {
    uint8_t d[32];
    sha256(m, MSG_LEN, d);
    uint64_t id = 0;
    for (int i = 0; i < 8; i++) id = id << 8 | d[i];
    return id;
}

static void build_base(void) {
    put32(g_msg, W); put32(g_msg + 4, H);
    for (int j = 0; j < NK; j++) g_rgba[(g_ky[j] * W + g_kx[j]) * 4 + 3] = 1;
    for (int i = 0; i < NPIX; i++) {
        const uint8_t *p = g_rgba + i * 4;
        uint8_t *q = g_msg + 8 + i * 4;
        q[0] = p[3] ? p[0] : 0; q[1] = p[3] ? p[1] : 0; q[2] = p[3] ? p[2] : 0; q[3] = p[3];
    }
    for (int j = 0; j < NK; j++) {
        g_off[j] = 8 + ((size_t)g_ky[j] * W + g_kx[j]) * 4;
        g_msg[g_off[j]] = g_msg[g_off[j] + 1] = g_msg[g_off[j] + 2] = 0;
    }
}

/* midstate over the whole blocks before the first knob, padded remainder in g_tail */
static void build_midstate(void) {
    g_fb = g_off[0] / 64;
    uint32_t s[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    sha256_block(s, g_msg, g_fb);
    memcpy(g_mid, s, sizeof s);
    size_t rem = MSG_LEN - g_fb * 64;
    g_nblk = (rem + 9 + 63) / 64;
    memset(g_tail, 0, g_nblk * 64);
    memcpy(g_tail, g_msg + g_fb * 64, rem);
    g_tail[rem] = 0x80;
    uint64_t bits = (uint64_t)MSG_LEN * 8;
    for (int i = 0; i < 8; i++) g_tail[g_nblk * 64 - 1 - i] = (uint8_t)(bits >> (8 * i));
    for (int j = 0; j < NK; j++) g_woff[j] = g_off[j] - g_fb * 64;
}

/* ---- PNG in Mojang's encoding: signature, IHDR, one IDAT (zlib level 4, memLevel 8, filter 0), IEND ---- */
static size_t mojang_png(const uint8_t *rgba, uint8_t *file) {
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    static const uint8_t iend[12] = {0,0,0,0,'I','E','N','D',0xAE,0x42,0x60,0x82};
    uint8_t *raw = malloc(RAW_LEN);
    for (int y = 0; y < H; y++) { raw[y * ROW] = 0; memcpy(raw + y * ROW + 1, rgba + y * W * 4, W * 4); }
    memcpy(file, sig, 8);
    uint8_t ihdr[17] = {'I','H','D','R', 0,0,0,W, 0,0,0,H, 8, 6, 0, 0, 0};
    put32(file + 8, 13);
    memcpy(file + 12, ihdr, 17);
    put32(file + 29, (uint32_t)crc32(0, ihdr, 17));
    uint8_t *zout = file + 33 + 8;
    size_t zcap = compressBound(RAW_LEN) + 64;
    z_stream z;
    memset(&z, 0, sizeof z);
    if (deflateInit2(&z, 4, Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY) != Z_OK) { fprintf(stderr, "deflateInit2 failed\n"); exit(2); }
    z.next_in = raw; z.avail_in = RAW_LEN; z.next_out = zout; z.avail_out = (uInt)zcap;
    if (deflate(&z, Z_FINISH) != Z_STREAM_END) { fprintf(stderr, "deflate failed\n"); exit(2); }
    size_t zlen = zcap - z.avail_out;
    deflateEnd(&z);
    free(raw);
    uint8_t *p = file + 33;
    put32(p, (uint32_t)zlen);
    memcpy(p + 4, "IDAT", 4);
    put32(zout + zlen, (uint32_t)crc32(crc32(0, (const uint8_t *)"IDAT", 4), zout, (uInt)zlen));
    memcpy(zout + zlen + 4, iend, 12);
    return 33 + 8 + zlen + 4 + 12;
}

/* write the PNG for counter c; returns 0 on success */
static int write_hit(uint64_t c, uint64_t want_id, char *path, size_t pathcap) {
    static __thread uint8_t m[MSG_LEN];
    static __thread uint8_t rgba[NPIX * 4];
    static __thread uint8_t file[FILE_CAP_];
    memcpy(m, g_msg, MSG_LEN);
    fill_counter(m, c);
    if (id_of_message(m) != want_id) return 1;
    memcpy(rgba, g_rgba, sizeof rgba);
    for (int j = 0; j < NK; j++) memcpy(rgba + (g_ky[j] * W + g_kx[j]) * 4, m + g_off[j], 3);
    size_t flen = mojang_png(rgba, file);
    char hx[17];
    hex16(want_id, hx);
    snprintf(path, pathcap, "%s/%s.png", g_outdir, hx);
    if (!g_nosave) {
        FILE *fp = fopen(path, "wb");
        if (!fp) return 2;
        fwrite(file, 1, flen, fp);
        fclose(fp);
    }
    return 0;
}

static void report(uint64_t c, uint64_t id, const uint8_t *sha_hex_src, const char *why) {
    (void)sha_hex_src;
    char path[1024], hx[17];
    if (write_hit(c, id, path, sizeof path)) { fprintf(stderr, "namemcfinder: hit re-check failed (counter %llu)\n", (unsigned long long)c); exit(3); }
    hex16(id, hx);
    pthread_mutex_lock(&g_io);
    uint8_t d[32];
    static __thread uint8_t m[MSG_LEN];
    memcpy(m, g_msg, MSG_LEN);
    fill_counter(m, c);
    sha256(m, MSG_LEN, d);
    char full[65];
    for (int i = 0; i < 32; i++) snprintf(full + 2 * i, 3, "%02x", d[i]);
    printf("HIT %s sha=%s id=%s counter=%llu file=%s\n", why, full, hx, (unsigned long long)c, path);
    fflush(stdout);
    pthread_mutex_unlock(&g_io);
}

static const char *match_why(uint64_t id, char *buf, size_t cap) {
    for (int i = 0; i < npats; i++)
        if ((id & pats[i].mask) == pats[i].val) {
            snprintf(buf, cap, "%s:%s", pats[i].suffix ? "suffix" : "prefix", pats[i].text);
            return buf;
        }
    if (ncpats) {
        uint8_t b[8];
        for (int i = 0; i < 8; i++) b[i] = (uint8_t)(id >> (56 - 8 * i));
        const char *c = contains_find(b, 8, 0);
        if (c) { snprintf(buf, cap, "contains:%s", c); return buf; }
    }
    int zn = id ? __builtin_clzll(id) / 4 : 16;
    if (g_zeros && zn >= g_zeros) { snprintf(buf, cap, "zeros:%d", zn); return buf; }
    return NULL;
}

static void *worker(void *arg) {
    (void)arg;
    uint8_t *buf = malloc(g_nblk * 64 + 64);
    memcpy(buf, g_tail, g_nblk * 64);
    int best = atomic_load(&g_best_zeros);
    while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) {
        uint64_t c0 = atomic_fetch_add(&g_next, CHUNK);
        uint32_t v0 = (uint32_t)(c0 >> 48), v1 = (uint32_t)(c0 >> 24) & 0xFFFFFF;
        put32(buf + g_woff[0], v0 << 8 | 1);
        put32(buf + g_woff[1], v1 << 8 | 1);
        for (uint64_t i = 0; i < CHUNK; i++) {
            uint64_t c = c0 + i;
            put32(buf + g_woff[2], ((uint32_t)c & 0xFFFFFF) << 8 | 1);
            uint32_t s[8];
            memcpy(s, g_mid, sizeof s);
            sha256_block(s, buf, g_nblk);
            uint64_t id = (uint64_t)s[0] << 32 | s[1];
            int zn = id ? __builtin_clzll(id) / 4 : 16;
            if (zn > best) {
                best = atomic_load(&g_best_zeros);
                while (zn > best && !atomic_compare_exchange_weak(&g_best_zeros, &best, zn)) {}
                if (zn > best) best = zn;
            }
            char wb[48];
            const char *why = match_why(id, wb, sizeof wb);
            if (why) {
                long long h = atomic_fetch_add(&g_hits, 1) + 1;
                if (g_max_hits > 0 && h > g_max_hits) { atomic_store(&g_stop, 1); break; }
                report(c, id, NULL, why);
                if (g_max_hits > 0 && h >= g_max_hits) { atomic_store(&g_stop, 1); break; }
            }
        }
        atomic_fetch_add_explicit(&g_tried, CHUNK, memory_order_relaxed);
    }
    free(buf);
    return NULL;
}

static int ncpus(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1;
#endif
}

/* start-up self-test: midstate path against a from-scratch hash */
static void self_test(void) {
    uint64_t rng = g_seed;
    uint8_t *buf = malloc(g_nblk * 64);
    static uint8_t m[MSG_LEN];
    for (int t = 0; t < 64; t++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        uint64_t c = t == 0 ? 0 : t == 1 ? ~0ULL : rng ^ (rng << 29);
        memcpy(buf, g_tail, g_nblk * 64);
        put32(buf + g_woff[0], (uint32_t)(c >> 48) << 8 | 1);
        put32(buf + g_woff[1], ((uint32_t)(c >> 24) & 0xFFFFFF) << 8 | 1);
        put32(buf + g_woff[2], ((uint32_t)c & 0xFFFFFF) << 8 | 1);
        uint32_t s[8];
        memcpy(s, g_mid, sizeof s);
        sha256_block(s, buf, g_nblk);
        memcpy(m, g_msg, MSG_LEN);
        fill_counter(m, c);
        if (((uint64_t)s[0] << 32 | s[1]) != id_of_message(m)) { fprintf(stderr, "namemcfinder: midstate self-test failed\n"); exit(2); }
    }
    free(buf);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: namemcfinder rgba.bin OUTDIR -k x,y,x,y,x,y [-p HEX]... [-s HEX]... [-c HEX]... [-z N] [-t N] [-S SEED] [-n N] [-b]\n");
        return 1;
    }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp || fread(g_rgba, 1, sizeof g_rgba, fp) != sizeof g_rgba) { fprintf(stderr, "bad rgba file\n"); return 1; }
    fclose(fp);
    g_outdir = argv[2];
    g_threads = ncpus();
    g_seed = rand_seed();
    pat_fixed_width = 1;
    int bench = 0, have_k = 0, render = 0;
    uint64_t render_ctr = 0, expect = 0;
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "-k") && v) {
            int n = sscanf(v, "%d,%d,%d,%d,%d,%d", &g_kx[0], &g_ky[0], &g_kx[1], &g_ky[1], &g_kx[2], &g_ky[2]);
            if (n != 6) { fprintf(stderr, "bad -k %s\n", v); return 1; }
            have_k = 1; i++;
        }
        else if (!strcmp(a, "-p") && v) { if (parse_pat(v, 0)) { fprintf(stderr, "bad -p %s\n", v); return 1; } i++; }
        else if (!strcmp(a, "-s") && v) { if (parse_pat(v, 1)) { fprintf(stderr, "bad -s %s\n", v); return 1; } i++; }
        else if (!strcmp(a, "-c") && v) { if (parse_contains(v)) { fprintf(stderr, "bad -c %s\n", v); return 1; } i++; }
        else if (!strcmp(a, "-z") && v) { g_zeros = atoi(v); i++; }
        else if (!strcmp(a, "-t") && v) { g_threads = atoi(v); i++; }
        else if (!strcmp(a, "-S") && v) { g_seed = strtoull(v, NULL, 0); i++; }
        else if (!strcmp(a, "-n") && v) { g_max_hits = atoll(v); i++; }
        else if (!strcmp(a, "-b")) bench = g_nosave = 1;
        else if (!strcmp(a, "--render") && v) { render = 1; render_ctr = strtoull(v, NULL, 0); i++; }
        else if (!strcmp(a, "--expect") && v) { expect = strtoull(v, NULL, 16); i++; }
        else { fprintf(stderr, "unknown option %s\n", a); return 1; }
    }
    if (!have_k) { fprintf(stderr, "-k is required\n"); return 1; }
    /* order the knob pixels by message offset */
    for (int a = 0; a < NK; a++)
        for (int b = a + 1; b < NK; b++)
            if (g_ky[b] * W + g_kx[b] < g_ky[a] * W + g_kx[a]) {
                int t = g_kx[a]; g_kx[a] = g_kx[b]; g_kx[b] = t;
                t = g_ky[a]; g_ky[a] = g_ky[b]; g_ky[b] = t;
            }
    for (int j = 0; j < NK; j++)
        if (g_kx[j] < 0 || g_kx[j] >= W || g_ky[j] < 0 || g_ky[j] >= H || (j && g_ky[j] * W + g_kx[j] == g_ky[j-1] * W + g_kx[j-1])) {
            fprintf(stderr, "bad knob pixels\n"); return 1;
        }
#ifdef HAVE_X86
    unsigned ea, eb, ec, ed;
    if (__get_cpuid_count(7, 0, &ea, &eb, &ec, &ed) && (eb & (1u << 29))) sha256_block = sha256_block_shani;
#endif
    uint8_t dg[32];
    sha256((const uint8_t *)"abc", 3, dg);
    if (dg[0] != 0xba || dg[1] != 0x78 || dg[31] != 0xad) { fprintf(stderr, "SHA-256 self-test failed\n"); return 2; }
    if (g_threads < 1) g_threads = 1;

    build_base();
    build_midstate();
    self_test();

    if (render) {
        char path[1024];
        int r = write_hit(render_ctr, expect, path, sizeof path);
        if (r) fprintf(stderr, "namemcfinder: render failed (%d): id mismatch or write error\n", r);
        return r;
    }

    fprintf(stderr, "threads=%d seed=%llu knobs=(%d,%d)(%d,%d)(%d,%d) blocks=%zu patterns=%d contains=%d zeros>=%d sha=%s\n", g_threads,
            (unsigned long long)g_seed, g_kx[0], g_ky[0], g_kx[1], g_ky[1], g_kx[2], g_ky[2], g_nblk, npats, ncpats, g_zeros,
            sha256_block == sha256_block_generic ? "generic" : "sha-ni");
    fflush(stderr);
    atomic_store(&g_next, (g_seed & ~(CHUNK - 1)));
    pthread_t *th = calloc(g_threads, sizeof *th);
    for (int i = 0; i < g_threads; i++) pthread_create(&th[i], NULL, worker, NULL);
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
