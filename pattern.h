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

/* ======================= stat lines (record categories) =======================
 * Same rules as patterns.py (parse_stat / StatLine, the reference) and web/js/patterns.js. The finders receive the
 * normalised text (lower case, no spaces, word files expanded): terms joined by '&', each one of
 *   #S{range}  count of the digits in S (a digit, ?, ?d, ?l, [class]) in the visible id
 *   #(w1|w2){range}   occurrences of the words (1-8 digits; overlaps count)
 *   ##{range}  number of different digits
 *   ~{n} +{n} -{n}  palindrome / staircase up / staircase down of >= n digits
 *   (?)\1{n} (??)\1{n} (?{p})\1{n}  periodic run of p*(n+1) digits
 *   hex pattern, or X{max}: the longest repeat of the last token (^f{max})
 * range: {n} {n,} {,n} {n,m}, {max} or {min} (record mode). The visible id is the 64-digit hash without its leading
 * zeros (the link) for Mojang, the 16 digits for NameMC. Records: a line with {max}/{min} reports every id that
 * beats the best value so far. The GPUs compute the same values (SWAR over the digest words) and the host re-checks
 * every candidate with st_eval(). */
#define ST_MAXTERM 6
#define ST_MAXWORDS 64
#define ST_MAXLINES 32
#define ST_TEXT 480
#define ST_NONE_MAX (-1)
#define ST_NONE_MIN 0x7fffffff
enum { ST_HEX, ST_COUNT, ST_DISTINCT, ST_PERIOD, ST_UP, ST_DOWN, ST_PAL, ST_WORDS };

typedef struct {
    int kind, rec, lo, hi;          /* rec: 0 threshold, 1 {max}, 2 {min}; hi < 0: no upper bound */
    int period;                     /* ST_PERIOD */
    uint16_t cset;                  /* ST_COUNT */
    int nwords; uint8_t wlen[ST_MAXWORDS]; uint8_t word[ST_MAXWORDS][8];
    int start, npre; uint16_t pre[MPAT_MAXTOK], tok;   /* ST_HEX with {max}: prefix tokens and the repeated token */
    int end;                                           /* X{max}$: the run that ends the id (no prefix) */
    mpat_t pat;                     /* ST_HEX threshold */
} sterm_t;

typedef struct {
    char text[ST_TEXT];
    int width, nterm, rec_term;     /* rec_term: index of the {max}/{min} term or -1 */
    sterm_t t[ST_MAXTERM];
} sline_t;

static inline int st_err(char *err, int errlen, const char *msg, const char *text) {
    if (err) snprintf(err, (size_t)errlen, "stat line '%s': %s", text, msg);
    return -1;
}

/* Does the (normalised) text use the stat syntax? */
static inline int st_is_stat(const char *s) {
    if (!*s) return 0;
    if (strchr("#~+-(", s[0])) return 1;
    if (strchr(s, '&') || strchr(s, '\\') || strstr(s, "{max}") || strstr(s, "{min}")) return 1;
    for (const char *p = strchr(s, '{'); p; p = strchr(p + 1, '{')) {
        const char *q = p + 1;
        while (*q >= '0' && *q <= '9') q++;
        if (*q == ',') return 1;
    }
    return 0;
}

/* range at s (must run to e): {n} {n,} {,n} {n,m} {max} {min}. exact: plain {n}. Returns 0 or -1. */
static inline int st_range(const char *s, const char *e, int *lo, int *hi, int *rec, int *exact) {
    *rec = 0; *exact = 0; *lo = 0; *hi = -1;
    if (s >= e || *s != '{' || e[-1] != '}') return -1;
    if (e - s == 5 && !memcmp(s, "{max}", 5)) { *rec = 1; return 0; }
    if (e - s == 5 && !memcmp(s, "{min}", 5)) { *rec = 2; return 0; }
    const char *p = s + 1, *end = e - 1;
    int a = 0, b = 0, ha = 0, hb = 0, comma = 0;
    while (p < end && *p >= '0' && *p <= '9') { a = a * 10 + (*p++ - '0'); ha = 1; if (a > 100000) return -1; }
    if (p < end && *p == ',') { comma = 1; p++; }
    while (p < end && *p >= '0' && *p <= '9') { b = b * 10 + (*p++ - '0'); hb = 1; if (b > 100000) return -1; }
    if (p != end || (!ha && !hb) || (!comma && hb)) return -1;
    if (comma && ha && hb && a > b) return -1;
    *lo = a;
    if (!comma) { *hi = a; *exact = 1; }
    else *hi = hb ? b : -1;
    return 0;
}

static inline int st_class(const char *s, const char *e, uint16_t *set) {   /* s = after '[', e = at ']' */
    uint16_t r = 0;
    while (s < e) {
        int a = mpat_hexval(*s);
        if (a < 0) return -1;
        if (s + 2 < e && s[1] == '-') {
            int b = mpat_hexval(s[2]);
            if (b < a) return -1;
            for (int d = a; d <= b; d++) r |= (uint16_t)(1u << d);
            s += 3;
        } else { r |= (uint16_t)(1u << a); s++; }
    }
    if (!r) return -1;
    *set = r;
    return 0;
}

static inline int st_parse_term(const char *s, const char *e, int width, sterm_t *t, char *err, int errlen, const char *whole) {
    memset(t, 0, sizeof *t);
    int lo, hi, rec, exact;
    t->hi = -1;
    if (s[0] == '#' && s[1] == '#') {
        if (st_range(s + 2, e, &lo, &hi, &rec, &exact)) return st_err(err, errlen, "bad ## term", whole);
        t->kind = ST_DISTINCT; t->lo = lo; t->hi = hi; t->rec = rec;
        return 0;
    }
    if (s[0] == '#') {
        const char *p = s + 1;
        if (*p == '(') {
            const char *q = strchr(p, ')');
            if (!q || q >= e) return st_err(err, errlen, "unclosed (", whole);
            t->kind = ST_WORDS;
            for (const char *w = p + 1; w <= q; ) {
                const char *x = w;
                while (x < q && *x != '|') x++;
                int n = (int)(x - w);
                if (n < 1 || n > 8 || t->nwords >= ST_MAXWORDS) return st_err(err, errlen, "bad word list", whole);
                for (int i = 0; i < n; i++) { int v = mpat_hexval(w[i]); if (v < 0) return st_err(err, errlen, "bad word", whole); t->word[t->nwords][i] = (uint8_t)v; }
                t->wlen[t->nwords++] = (uint8_t)n;
                w = x + 1;
                if (x == q) break;
            }
            p = q + 1;
        } else {
            t->kind = ST_COUNT;
            if (*p == '?') {
                if (p[1] == 'd') { t->cset = 0x03FF; p += 2; }
                else if (p[1] == 'l') { t->cset = 0xFC00; p += 2; }
                else { t->cset = 0xFFFF; p++; }
            } else if (*p == '[') {
                const char *q = strchr(p, ']');
                if (!q || q >= e || st_class(p + 1, q, &t->cset)) return st_err(err, errlen, "bad class", whole);
                p = q + 1;
            } else if (mpat_hexval(*p) >= 0) { t->cset = (uint16_t)(1u << mpat_hexval(*p)); p++; }
            else return st_err(err, errlen, "bad count", whole);
        }
        if (st_range(p, e, &lo, &hi, &rec, &exact)) return st_err(err, errlen, "bad range", whole);
        t->lo = lo; t->hi = hi; t->rec = rec;
        return 0;
    }
    if (s[0] == '~' || s[0] == '+' || s[0] == '-') {
        t->kind = s[0] == '~' ? ST_PAL : s[0] == '+' ? ST_UP : ST_DOWN;
        if (st_range(s + 1, e, &lo, &hi, &rec, &exact)) return st_err(err, errlen, "bad range", whole);
        t->rec = rec; t->lo = lo; t->hi = -1;
        return 0;
    }
    if (s[0] == '(') {
        const char *p = s + 1;
        int per = 0;
        if (*p == '?' && p[1] == '{') {
            p += 2;
            while (*p >= '0' && *p <= '9') per = per * 10 + (*p++ - '0');
            if (*p++ != '}') return st_err(err, errlen, "bad period", whole);
        } else {
            while (*p == '?') { per++; p++; }
        }
        if (per < 1 || per > 8 || p[0] != ')' || p[1] != '\\' || p[2] != '1') return st_err(err, errlen, "bad periodic run", whole);
        if (st_range(p + 3, e, &lo, &hi, &rec, &exact)) return st_err(err, errlen, "bad range", whole);
        t->kind = ST_PERIOD; t->period = per; t->rec = rec; t->lo = per * (lo + 1); t->hi = -1;
        return 0;
    }
    /* hex pattern, maybe X{max} */
    char buf[MPAT_TEXT];
    size_t n = (size_t)(e - s);
    if (n >= sizeof buf - 3) return st_err(err, errlen, "term too long", whole);
    memcpy(buf, s, n); buf[n] = 0;
    t->kind = ST_HEX;
    if (n > 6 && !memcmp(buf + n - 6, "{max}$", 6)) {
        memcpy(buf + n - 6, "{1}", 4);
        mpat_t mp;
        if (mpat_parse(buf, width, &mp, err, errlen)) return -1;
        if (mp.start || mp.ntok != 1) return st_err(err, errlen, "a run at the end takes a single token", whole);
        t->rec = 1; t->end = 1; t->npre = 0; t->tok = mp.tok[0];
        return 0;
    }
    if (n > 5 && !memcmp(buf + n - 5, "{max}", 5)) {
        memcpy(buf + n - 5, "{1}", 4);
        mpat_t mp;
        if (mpat_parse(buf, width, &mp, err, errlen)) return -1;
        t->rec = 1; t->start = mp.start; t->npre = mp.ntok - 1; t->tok = mp.tok[mp.ntok - 1];
        for (int i = 0; i < t->npre; i++) t->pre[i] = mp.tok[i];
        return 0;
    }
    return mpat_parse(buf, width, &t->pat, err, errlen);
}

/* Parse a normalised stat line for an id of `width` digits. Returns 0 or -1 with a message in err. */
static inline int sline_parse(const char *text, int width, sline_t *L, char *err, int errlen) {
    memset(L, 0, sizeof *L);
    size_t n = strlen(text);
    if (n == 0 || n >= ST_TEXT) return st_err(err, errlen, "empty or too long", text);
    for (size_t i = 0; i < n; i++) L->text[i] = (char)((text[i] >= 'A' && text[i] <= 'Z') ? text[i] | 0x20 : text[i]);
    L->width = width; L->rec_term = -1;
    const char *s = L->text, *end = s + n;
    int depth = 0;
    const char *start = s;
    for (const char *p = s; p <= end; p++) {
        if (p < end && (*p == '[' || *p == '(')) depth++;
        else if (p < end && (*p == ']' || *p == ')')) depth--;
        if (p == end || (*p == '&' && depth == 0)) {
            if (p == start || L->nterm >= ST_MAXTERM) return st_err(err, errlen, "empty or too many terms", text);
            if (st_parse_term(start, p, width, &L->t[L->nterm], err, errlen, text)) return -1;
            if (L->t[L->nterm].rec) { if (L->rec_term >= 0) return st_err(err, errlen, "two records", text); L->rec_term = L->nterm; }
            L->nterm++;
            start = p + 1;
        }
    }
    return 0;
}

/* ---- reference values (nibble arrays) ---- */
static inline int st_zeros(const uint8_t *h, int L) {
    int z = 0;
    if (L == 64) while (z < 63 && h[z] == 0) z++;
    return z;
}

static inline int st_longest_period(const uint8_t *v, int n, int p) {
    int run = 0, best = 0;
    for (int i = 0; i + p < n; i++) { run = v[i] == v[i + p] ? run + 1 : 0; if (run > best) best = run; }
    int ln = best ? best + p : (p < n ? p : n);
    return p * (ln / p);
}
static inline int st_longest_stair(const uint8_t *v, int n, int up) {
    int run = 0, best = 0;
    for (int i = 0; i + 1 < n; i++) {
        int ok = up ? (v[i] != 15 && v[i + 1] == v[i] + 1) : (v[i] != 0 && v[i + 1] == v[i] - 1);
        run = ok ? run + 1 : 0;
        if (run > best) best = run;
    }
    return n ? best + 1 : 0;
}
static inline int st_longest_pal(const uint8_t *v, int n) {
    int best = n ? 1 : 0;
    for (int c = 0; c < n; c++)
        for (int odd = 1; odd >= 0; odd--) {
            int a = odd ? c - 1 : c, b = c + 1;
            while (a >= 0 && b < n && v[a] == v[b]) { a--; b++; }
            if (b - a - 1 > best) best = b - a - 1;
        }
    return best;
}
static inline int st_count_words(const uint8_t *v, int n, const sterm_t *t) {
    int total = 0;
    for (int w = 0; w < t->nwords; w++) {
        int len = t->wlen[w];
        for (int o = 0; o + len <= n; o++) if (!memcmp(v + o, t->word[w], (size_t)len)) total++;
    }
    return total;
}
static inline int st_hexrec(const sterm_t *t, const uint8_t *h, int L, int zl) {
    if (t->end) {
        int run = 0;
        while (run < L && (t->tok >> h[L - 1 - run] & 1)) run++;
        return run;
    }
    const uint8_t *s = t->start ? h : h + zl;
    int n = t->start ? L : L - zl, best = -1;
    for (int o = 0; o < (t->start ? 1 : n - t->npre + 1); o++) {
        int ok = o + t->npre <= n;
        for (int i = 0; ok && i < t->npre; i++) if (!(t->pre[i] >> s[o + i] & 1)) ok = 0;
        if (!ok) continue;
        int j = o + t->npre;
        while (j < n && (t->tok >> s[j] & 1)) j++;
        if (j - o - t->npre > best) best = j - o - t->npre;
    }
    return best;
}

/* Value of a term, or -1 for a hex record term whose prefix does not match. nib: the L id digits. */
static inline int st_value(const sterm_t *t, const uint8_t *nib, int L, int zl) {
    const uint8_t *v = nib + zl; const int n = L - zl;
    switch (t->kind) {
    case ST_COUNT: { int c = 0; for (int i = 0; i < n; i++) c += t->cset >> v[i] & 1; return c; }
    case ST_DISTINCT: { unsigned m = 0; for (int i = 0; i < n; i++) m |= 1u << v[i]; return mpat_popc(m); }
    case ST_WORDS: return st_count_words(v, n, t);
    case ST_PERIOD: return st_longest_period(v, n, t->period);
    case ST_UP: return st_longest_stair(v, n, 1);
    case ST_DOWN: return st_longest_stair(v, n, 0);
    case ST_PAL: return st_longest_pal(v, n);
    default: return st_hexrec(t, nib, L, zl);
    }
}

/* Does the line hold on the id? val: the value of the record term. */
static inline int st_eval(const sline_t *l, const uint8_t *nib, int *val) {
    const int zl = st_zeros(nib, l->width);
    *val = 0;
    for (int k = 0; k < l->nterm; k++) {
        const sterm_t *t = &l->t[k];
        if (t->kind == ST_HEX && !t->rec) { if (!mpat_match(&t->pat, nib)) return 0; continue; }
        int v = st_value(t, nib, l->width, zl);
        if (t->rec) { if (v < 0) return 0; *val = v; continue; }
        if (v < t->lo || (t->hi >= 0 && v > t->hi)) return 0;
    }
    return 1;
}

static inline int st_beats(int rec, int v, int best) { return rec == 1 ? v > best : v < best; }

typedef struct {
    int thr;                          /* first threshold line that holds, or -1 */
    int nrec, rline[ST_MAXLINES], rval[ST_MAXLINES];   /* record lines that beat best[] */
} stat_res_t;

/* Check every stat line on the digest dg (32 bytes Mojang / 8 bytes NameMC). best[k]: current record of line k. */
static inline void stat_find(const sline_t *ls, int n, const uint8_t *dg, const int *best, stat_res_t *r) {
    r->thr = -1; r->nrec = 0;
    if (!n) return;
    const int L = ls[0].width;
    uint8_t nib[64];
    for (int b = 0; b < L / 2; b++) { nib[2*b] = dg[b] >> 4; nib[2*b+1] = dg[b] & 15; }
    for (int i = 0; i < n; i++) {
        int v;
        if (!st_eval(&ls[i], nib, &v)) continue;
        if (ls[i].rec_term < 0) { if (r->thr < 0) r->thr = i; }
        else if (st_beats(ls[i].t[ls[i].rec_term].rec, v, best[i])) { r->rline[r->nrec] = i; r->rval[r->nrec++] = v; }
    }
}
