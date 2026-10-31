/*
 * hashfinder: brute-force Minecraft skin texture ids.
 *
 * Mojang re-encodes an uploaded skin as:
 *   PNG signature + IHDR(64x64, 8-bit RGBA) + one IDAT (zlib level 4, wbits 15,
 *   memLevel 8, default strategy, every scanline filter 0) + IEND
 * and the texture id is SHA-256 of that file (leading zeros dropped in the URL).
 *
 * We vary the RGB of a few "knob" pixels (chosen by skinhash.py so they are never
 * rendered) and hash each candidate. The knob pixels sit as late in the image as
 * possible: the deflate state after the unchanged prefix is computed once
 * and cloned with deflateCopy(), so each candidate only re-runs LZ77 on the tail.
 *
 * usage: hashfinder <raw.bin> <offsets> <outdir> [options]
 *   raw.bin   64*(1+256) bytes of filtered scanlines (filter byte 0 per row)
 *   offsets   comma-separated byte offsets into raw.bin that may be overwritten
 *             (at least 8; the 64-bit counter goes into the first 8)
 * options:
 *   -p HEX    match if the trimmed id (leading zeros dropped) starts with HEX (not 0...)
 *   -s HEX    match if sha256 hex ends with HEX     (repeatable, <=16 chars)
 *   -z N      also save any hash with >= N leading zero nibbles
 *   -t N      threads (default: all logical CPUs)
 *   -S SEED   64-bit start counter (default: random)
 *   -n N      stop after N hits (default: run forever)
 *   -b        benchmark: run 10 s, print rate, exit
 */
#define _POSIX_C_SOURCE 200809L
#include "common.h"

/* ---------------- shared state ---------------- */
static uint8_t g_raw[RAW_LEN];
static int g_offs[MAX_OFFS], g_noffs;
static int g_split;            /* first modified byte; prefix [0, split) is fixed */
static int g_zeros = 0;        /* save if >= this many leading zero nibbles (0 = off) */
static long long g_max_hits = -1;
static const char *g_outdir;
static uint64_t g_seed;
static int g_threads;
static atomic_ullong g_tried;
static atomic_llong g_hits;
static atomic_int g_best_zeros;
static atomic_int g_stop;
static int g_nosave;           /* benchmark mode: never write files */
static pthread_mutex_t g_io = PTHREAD_MUTEX_INITIALIZER;

static const uint8_t PNG_SIG[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};

static size_t build_header(uint8_t *f) {
    memcpy(f, PNG_SIG, 8);
    uint8_t ihdr[17] = {'I','H','D','R', 0,0,0,W, 0,0,0,H, 8, 6, 0, 0, 0};
    f[8] = 0; f[9] = 0; f[10] = 0; f[11] = 13;
    memcpy(f + 12, ihdr, 17);
    uint32_t crc = crc32(0, ihdr, 17);
    f[29] = crc >> 24; f[30] = crc >> 16; f[31] = crc >> 8; f[32] = crc;
    return 33;
}

static void hexify(const uint8_t d[32], char out[65]) {
    static const char *hx = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[2*i] = hx[d[i] >> 4]; out[2*i+1] = hx[d[i] & 15]; }
    out[64] = 0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

/* deflate the fixed prefix once; returns compressed bytes produced so far */
static int prepare_base(z_stream *base, arena_t *copy_arena, uint8_t *zout, size_t zcap) {
    memset(base, 0, sizeof *base);
    if (deflateInit2(base, 4, Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY) != Z_OK) return -1;
    base->next_in = g_raw;
    base->avail_in = (uInt)g_split;
    base->next_out = zout;
    base->avail_out = (uInt)zcap;
    if (deflate(base, Z_NO_FLUSH) != Z_OK || base->avail_in != 0) return -1;
    /* clones are carved from the per-thread arena, freed by resetting it */
    base->zalloc = arena_alloc;
    base->zfree = arena_free;
    base->opaque = copy_arena;
    return (int)(zcap - base->avail_out);
}

static void *worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    uint8_t *raw = malloc(RAW_LEN);
    memcpy(raw, g_raw, RAW_LEN);
    size_t zcap = compressBound(RAW_LEN) + 64;
    uint8_t *file = malloc(33 + 8 + zcap + 4 + 12);
    size_t hdr = build_header(file);
    uint8_t *zout = file + hdr + 8;
    arena_t arena = {malloc(1 << 20), 1 << 20, 0};
    z_stream base;
    int pre = prepare_base(&base, &arena, zout, zcap);
    if (pre < 0) { fprintf(stderr, "deflate init failed\n"); exit(2); }
    const uint8_t *tail = raw + g_split;
    uInt tail_len = RAW_LEN - g_split;
    uint32_t crc_idat = crc32(0, (const uint8_t *)"IDAT", 4);
    uint64_t ctr = g_seed + (uint64_t)tid;
    uint64_t local = 0;
    int best = atomic_load(&g_best_zeros);

    while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) {
        for (int i = 0; i < 8; i++) raw[g_offs[i]] = (uint8_t)(ctr >> (8 * i));

        z_stream z;
        arena.used = 0;
        if (deflateCopy(&z, &base) != Z_OK) { fprintf(stderr, "deflateCopy failed\n"); exit(2); }
        z.next_in = (Bytef *)tail;
        z.avail_in = tail_len;
        z.next_out = zout + pre;
        z.avail_out = (uInt)(zcap - pre);
        if (deflate(&z, Z_FINISH) != Z_STREAM_END) { fprintf(stderr, "deflate failed\n"); exit(2); }
        size_t zlen = pre + (zcap - pre - z.avail_out);
        /* no deflateEnd needed: arena is reset next round */

        uint8_t *p = file + hdr;
        p[0] = zlen >> 24; p[1] = zlen >> 16; p[2] = zlen >> 8; p[3] = zlen;
        memcpy(p + 4, "IDAT", 4);
        uint32_t crc = crc32(crc_idat, zout, (uInt)zlen);
        p = zout + zlen;
        p[0] = crc >> 24; p[1] = crc >> 16; p[2] = crc >> 8; p[3] = crc;
        static const uint8_t IEND[12] = {0,0,0,0,'I','E','N','D',0xAE,0x42,0x60,0x82};
        memcpy(p + 4, IEND, 12);
        size_t flen = (size_t)(p + 16 - file);

        uint8_t d[32];
        sha256(file, flen, d);
        uint64_t hi = 0, lo = 0, mid = 0;
        for (int i = 0; i < 8; i++) { hi = hi << 8 | d[i]; mid = mid << 8 | d[8 + i]; lo = lo << 8 | d[24 + i]; }
        uint64_t tp = trimmed_top64(hi, mid);

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
            report(file, flen, d, ctr, why);
            long long h = atomic_fetch_add(&g_hits, 1) + 1;
            if (g_max_hits > 0 && h >= g_max_hits) atomic_store(&g_stop, 1);
        }

        ctr += (uint64_t)g_threads;
        if (++local == 256) { atomic_fetch_add_explicit(&g_tried, local, memory_order_relaxed); local = 0; }
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
        fprintf(stderr, "usage: hashfinder raw.bin offsets outdir [-p HEX]... [-s HEX]... [-z N] [-t N] [-S SEED] [-n N] [-b]\n");
        return 1;
    }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp || fread(g_raw, 1, RAW_LEN, fp) != RAW_LEN) { fprintf(stderr, "bad raw file\n"); return 1; }
    fclose(fp);
    char *offs = strdup(argv[2]);
    for (char *t = strtok(offs, ","); t && g_noffs < MAX_OFFS; t = strtok(NULL, ",")) {
        int o = atoi(t);
        if (o <= 0 || o >= RAW_LEN || o % ROW == 0) { fprintf(stderr, "bad offset %d\n", o); return 1; }
        g_offs[g_noffs++] = o;
    }
    if (g_noffs < 8) { fprintf(stderr, "need >= 8 offsets\n"); return 1; }
    g_split = RAW_LEN;
    for (int i = 0; i < g_noffs; i++) if (g_offs[i] < g_split) g_split = g_offs[i];
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
        else if (!strcmp(a, "-b")) bench = g_nosave = 1;
        else { fprintf(stderr, "unknown option %s\n", a); return 1; }
    }
#ifdef HAVE_X86
    unsigned a, b, c, d;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & (1u << 29))) sha256_block = sha256_block_shani;
#endif
    /* self-test SHA-256 ("abc") */
    uint8_t dg[32]; char hx[65];
    sha256((const uint8_t *)"abc", 3, dg); hexify(dg, hx);
    if (strcmp(hx, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")) {
        fprintf(stderr, "SHA-256 self-test failed\n"); return 2;
    }
    if (g_threads < 1) g_threads = 1;
    fprintf(stderr, "threads=%d seed=%llu tail=%d bytes (of %d) sha=%s patterns=%d zeros>=%d\n",
            g_threads, (unsigned long long)g_seed, RAW_LEN - g_split, RAW_LEN,
            sha256_block == sha256_block_generic ? "generic" : "sha-ni", npats, g_zeros);
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
