/* Hex patterns (-m): parser, mask compiler and full matcher. Plain C99, also compiled as C++ by nvcc (host code).
 * Same rules as patterns.py (the reference; skinhash.py validates every pattern with friendly messages before a
 * finder sees it) and web/js/patterns.js:
 *   hex digits, ^ (start) and $ (end) at the edges, ? any digit, ?d 0-9, ?l a-f, [0-3a] classes, {n} repeat,
 *   * a gap of any length between pieces.
 * Mojang ids are the full 64-digit hash; a pattern without ^ must lie after the leading zeros (the link drops them).
 * NameMC ids are the 16 digits.
 *
 * Every pattern compiles to one GPU/CPU prefilter, the most selective of
 *   anchored: nibble mask/value over the id from the ^ and $ pieces (am/av, big-endian 32-bit words), or
 *   sliding:  up to 8 + 8 tokens of one unanchored piece (p1/m1, p2/m2) compared at every offset o with
 *             lo + (usezl ? leading zeros : 0) <= o <= lim.
 * Unaligned classes only constrain the bits all their digits share, so the prefilter can pass non-matches;
 * mpat_match() is the exact check. */
#pragma once
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define MPAT_MAXTOK 64
#define MPAT_MAXPIECE 32
#define MPAT_TEXT 160

typedef struct {
    char text[MPAT_TEXT];
    int width, start, end, npiece, ntok;
    int pstart[MPAT_MAXPIECE], plen[MPAT_MAXPIECE];
    uint16_t tok[MPAT_MAXTOK];      /* allowed nibbles: bit v = digit v */
    /* prefilter */
    int sliding;                    /* 0 anchored, 1 sliding */
    uint32_t am[8], av[8];          /* anchored: masks / values, width/8 words */
    uint32_t p1, m1, p2, m2;        /* sliding windows */
    int lo, lim, usezl;
    double pass;                    /* prefilter pass probability per candidate (for the choice) */
} mpat_t;

static inline int mpat_hexval(int c) {
    return (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
}
static inline int mpat_popc(uint32_t x) { int n = 0; while (x) { x &= x - 1; n++; } return n; }

/* tightest 4-bit mask/value containing every digit of set s */
static inline void mpat_nib(uint16_t s, uint32_t *m, uint32_t *v) {
    uint32_t a = 15, o = 0;
    for (int d = 0; d < 16; d++) if (s >> d & 1) { a &= (uint32_t)d; o |= (uint32_t)d; }
    *m = ~(a ^ o) & 15u; *v = a & *m;
}

static inline int mpat_err(char *err, int errlen, const char *msg, const char *text) {
    if (err) snprintf(err, (size_t)errlen, "pattern '%s': %s", text, msg);
    return -1;
}

static inline void mpat_compile(mpat_t *p);

/* Parse pattern text for an id of `width` digits (64 or 16). Returns 0, or -1 with a message in err. */
static inline int mpat_parse(const char *s, int width, mpat_t *p, char *err, int errlen) {
    memset(p, 0, sizeof *p);
    size_t n = strlen(s);
    if (n == 0) return mpat_err(err, errlen, "empty pattern", s);
    if (n >= MPAT_TEXT) return mpat_err(err, errlen, "too long", s);
    for (size_t i = 0; i < n; i++) p->text[i] = (char)((s[i] >= 'A' && s[i] <= 'Z') ? s[i] | 0x20 : s[i]);
    p->width = width;
    size_t i = 0, e = n;
    if (p->text[0] == '^') { p->start = 1; i = 1; }
    if (e > i && p->text[e - 1] == '$') { p->end = 1; e--; }
    int last_tok = 0;   /* the previous item can be repeated */
    p->pstart[0] = 0;
    while (i < e) {
        char c = p->text[i];
        uint16_t set = 0;
        if (mpat_hexval(c) >= 0) { set = (uint16_t)(1u << mpat_hexval(c)); i++; }
        else if (c == '?') {
            if (i + 1 < e && p->text[i + 1] == 'd') { set = 0x03FF; i += 2; }
            else if (i + 1 < e && p->text[i + 1] == 'l') { set = 0xFC00; i += 2; }
            else { set = 0xFFFF; i++; }
        } else if (c == '[') {
            size_t j = i + 1;
            while (j < e && p->text[j] != ']') {
                int a = mpat_hexval(p->text[j]);
                if (a < 0) return mpat_err(err, errlen, "bad class", s);
                if (j + 2 < e && p->text[j + 1] == '-' && p->text[j + 2] != ']') {
                    int b = mpat_hexval(p->text[j + 2]);
                    if (b < a) return mpat_err(err, errlen, "bad class range", s);
                    for (int d = a; d <= b; d++) set |= (uint16_t)(1u << d);
                    j += 3;
                } else { set |= (uint16_t)(1u << a); j++; }
            }
            if (j >= e || !set) return mpat_err(err, errlen, "bad class", s);
            i = j + 1;
        } else if (c == '{') {
            size_t j = i + 1; int r = 0;
            while (j < e && p->text[j] >= '0' && p->text[j] <= '9' && r <= 1000) r = r * 10 + (p->text[j++] - '0');
            if (j >= e || p->text[j] != '}' || j == i + 1) return mpat_err(err, errlen, "bad repeat", s);
            if (r < 1 || !last_tok || r > width) return mpat_err(err, errlen, "bad repeat count", s);
            if (p->ntok + r - 1 > MPAT_MAXTOK) return mpat_err(err, errlen, "too many digits", s);
            for (int k = 1; k < r; k++) { p->tok[p->ntok] = p->tok[p->ntok - 1]; p->ntok++; p->plen[p->npiece]++; }
            last_tok = 0; i = j + 1;
            continue;
        } else if (c == '*') {
            if (p->plen[p->npiece] == 0 || i + 1 >= e) return mpat_err(err, errlen, "'*' needs digits on both sides", s);
            if (++p->npiece >= MPAT_MAXPIECE) return mpat_err(err, errlen, "too many pieces", s);
            p->pstart[p->npiece] = p->ntok;
            last_tok = 0; i++;
            continue;
        } else return mpat_err(err, errlen, "bad character", s);
        if (p->ntok >= MPAT_MAXTOK) return mpat_err(err, errlen, "too many digits", s);
        p->tok[p->ntok++] = set; p->plen[p->npiece]++;
        last_tok = 1;
    }
    if (p->plen[p->npiece] == 0) return mpat_err(err, errlen, "no digits", s);
    p->npiece++;
    if (p->ntok > width) return mpat_err(err, errlen, "longer than the id", s);
    if (p->start && p->end && p->npiece == 1 && p->ntok != width) return mpat_err(err, errlen, "with ^ and $ and no '*' it must be as long as the id", s);
    mpat_compile(p);
    return 0;
}

static inline void mpat_compile(mpat_t *p) {
    const int L = p->width, nw = L / 8;
    double best = 2.0;
    if (p->start || p->end) {
        uint32_t m, v;
        memset(p->am, 0, sizeof p->am); memset(p->av, 0, sizeof p->av);
        int nb = 0;
        for (int side = 0; side < 2; side++) {
            if (side == 0 && !p->start) continue;
            if (side == 1 && !p->end) continue;
            int k = side == 0 ? 0 : p->npiece - 1, base = side == 0 ? 0 : L - p->plen[k];
            for (int i = 0; i < p->plen[k]; i++) {
                mpat_nib(p->tok[p->pstart[k] + i], &m, &v);
                int pos = base + i, sh = 28 - 4 * (pos & 7);
                p->am[pos >> 3] |= m << sh; p->av[pos >> 3] |= v << sh;
            }
        }
        for (int k = 0; k < nw; k++) nb += mpat_popc(p->am[k]);
        best = ldexp(1.0, -nb);
        p->sliding = 0;
    }
    int before = 0;
    for (int j = 0; j < p->npiece; j++) {
        const int pl = p->plen[j];
        if (!((j == 0 && p->start) || (j == p->npiece - 1 && p->end))) {
            uint32_t tm[MPAT_MAXTOK], tv[MPAT_MAXTOK];
            for (int i = 0; i < pl; i++) mpat_nib(p->tok[p->pstart[j] + i], &tm[i], &tv[i]);
            int bd = 0, bb = -1;
            for (int d = 0; d < pl; d++) {
                int b = 0;
                for (int i = d; i < d + 8 && i < pl; i++) b += mpat_popc(tm[i]);
                if (b > bb) { bb = b; bd = d; }
            }
            uint32_t q1 = 0, n1 = 0, q2 = 0, n2 = 0;
            for (int i = 0; i < 16 && bd + i < pl; i++) {
                int sh = 28 - 4 * (i & 7);
                if (i < 8) { q1 |= tv[bd + i] << sh; n1 |= tm[bd + i] << sh; }
                else { q2 |= tv[bd + i] << sh; n2 |= tm[bd + i] << sh; }
            }
            int after = 0;
            for (int k = j; k < p->npiece; k++) after += p->plen[k];
            int lo = before + bd, lim = L - after + bd, pos = lim - lo + 1;
            double pass = (pos > 0 ? pos : 0) * ldexp(1.0, -(mpat_popc(n1) + mpat_popc(n2)));
            if (pass > 1.0) pass = 1.0;
            if (pass < best) {
                best = pass;
                p->sliding = 1; p->p1 = q1; p->m1 = n1; p->p2 = q2; p->m2 = n2; p->lo = lo; p->lim = lim;
                p->usezl = L == 64 && !p->start;
            }
        }
        before += pl;
    }
    p->pass = best;
}

static inline int mpat_at(const mpat_t *p, int k, const uint8_t *nib, int pos) {
    for (int i = 0; i < p->plen[k]; i++) if (!(p->tok[p->pstart[k] + i] >> nib[pos + i] & 1)) return 0;
    return 1;
}

/* Exact match on the id nibbles nib[0..width-1]. */
static inline int mpat_match(const mpat_t *p, const uint8_t *nib) {
    const int L = p->width, K = p->npiece;
    int zl = 0;
    if (L == 64 && !p->start) while (zl < 63 && nib[zl] == 0) zl++;
    int cur = p->start ? 0 : zl, limit = p->end ? L - p->plen[K - 1] : L;
    for (int k = 0; k < K; k++) {
        const int n = p->plen[k];
        if (k == 0 && p->start) {
            if (K == 1 && p->end && n != L) return 0;
            if (!mpat_at(p, 0, nib, 0)) return 0;
            cur = n;
        } else if (k == K - 1 && p->end) {
            int pos = L - n;
            if (pos < cur || !mpat_at(p, k, nib, pos)) return 0;
            cur = L;
        } else {
            int end = k < K - 1 ? limit : L, pos = cur;
            while (pos + n <= end && !mpat_at(p, k, nib, pos)) pos++;
            if (pos + n > end) return 0;
            cur = pos + n;
        }
    }
    return 1;
}

/* The prefilter on the id as big-endian 32-bit words w[0..width/8-1]; zl = leading zero digits (any lower bound). */
static inline int mpat_prefilter(const mpat_t *p, const uint32_t *w, int zl) {
    const int nw = p->width / 8;
    if (!p->sliding) {
        uint32_t acc = 0;
        for (int k = 0; k < nw; k++) acc |= (w[k] ^ p->av[k]) & p->am[k];
        return acc == 0;
    }
    const int lo = p->lo + (p->usezl ? zl : 0);
    for (int o = lo; o <= p->lim; o++) {
        int k = o >> 3, sh = 4 * (o & 7);
        uint32_t a = w[k], b = k + 1 < nw ? w[k + 1] : 0, c = k + 2 < nw ? w[k + 2] : 0;
        uint32_t x1 = sh ? (a << sh) | (b >> (32 - sh)) : a;
        if (((x1 ^ p->p1) & p->m1) == 0) {
            uint32_t x2 = sh ? (b << sh) | (c >> (32 - sh)) : b;
            if (((x2 ^ p->p2) & p->m2) == 0) return 1;
        }
    }
    return 0;
}

/* First pattern of the list that matches the digest bytes dg (32 bytes Mojang / 8 bytes NameMC), or -1.
 * zl: leading zero digits of the Mojang hash (0 for NameMC). */
static inline int mpat_find(const mpat_t *pats, int n, const uint8_t *dg) {
    if (!n) return -1;
    const int L = pats[0].width, nw = L / 8;
    uint32_t w[8];
    uint8_t nib[64];
    for (int k = 0; k < nw; k++) w[k] = (uint32_t)dg[4*k] << 24 | (uint32_t)dg[4*k+1] << 16 | (uint32_t)dg[4*k+2] << 8 | dg[4*k+3];
    int zl = 0;
    while (zl < L - 1 && !((zl & 1) ? dg[zl >> 1] & 15 : dg[zl >> 1] >> 4)) zl++;
    int have_nib = 0;
    for (int i = 0; i < n; i++) {
        if (!mpat_prefilter(&pats[i], w, zl)) continue;
        if (!have_nib) { for (int b = 0; b < L / 2; b++) { nib[2*b] = dg[b] >> 4; nib[2*b+1] = dg[b] & 15; } have_nib = 1; }
        if (mpat_match(&pats[i], nib)) return i;
    }
    return -1;
}
