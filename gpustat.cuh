/*
 * Stat lines on the GPU (record categories, see pattern.h / patterns.py): counts, different digits, periodic runs,
 * staircases, palindromes, word counts and hex repeats, all computed EXACTLY (the same value as st_eval) with SWAR
 * code over the digest words (8 words = 64 Mojang digits, 2 words = 16 NameMC digits). Shared by gpufinder.cu and
 * namemcgpu.cu. The lines are an op table in constant memory; the code is only compiled into the "stat" kernel
 * variant, the kernels without stat lines are unchanged.
 *
 * Layout: nibble i of the id is digit i (MSB first). zl = leading zero digits of a Mojang hash (the link drops
 * them; 0 for NameMC). Flags: bit 3 of a nibble. Run tests compact the flags into a 64-bit mask M (bit 63-i = digit
 * i) and AND it with shifted copies of itself (doubling), so "a run of r" costs about log2(r) shifts.
 *   count     flags of digit v: zflag(w ^ v*0x11111111) -> popcount; zeros hidden in zl are subtracted
 *   distinct  OR of 1 << nibble; bit 0 only if a zero is visible (total zeros > zl)
 *   periodic  digit i == digit i+p : zflag(w ^ (w shifted p nibbles)), run of p*n flags
 *   stair     digit i+1 == digit i + 1 (nibble-wise increment, digit i != 15), down = the same on ~w
 *   palindrome  8-digit windows of the digest against the nibble-reversed digest at every centre; only centres that
 *             pass are expanded exactly (out of line)
 *   words     one 32-bit window compare per word and offset
 * Record lines keep their best in d_rbest[] (atomicMax / atomicMin on the exact value) when every term is exact.
 */
#pragma once
#include <vector>
#include <cstdio>
#include <climits>
#include "pattern.h"

#define MAXSOP 96
#define MAXSL ST_MAXLINES
#define MAXSW 64
#define MAXSM 16
#define FAILV (-2147483647 - 1)

enum { SK_COUNT = 1, SK_DIST, SK_PER, SK_UP, SK_DOWN, SK_PAL, SK_WORDS, SK_HEXMASK, SK_HEXREC, SK_HEXRUN };
struct StatOp { u32 kind, rec, a, b, c, d, e; };         /* c = upper bound (INT_MAX = none) */
struct StatLineC { u32 first, n, rec, exact; };

__constant__ StatOp c_sop[MAXSOP];
__constant__ StatLineC c_sl[MAXSL];
__constant__ u32 c_snl;
__constant__ u32 c_sw[MAXSW * 3];     /* word: value, mask (left aligned in 32 bits), length */
__constant__ u32 c_sm[MAXSM * 16];    /* hex masks: 8 mask words, 8 value words */
__device__ int d_rbest[MAXSL];

template <int NW> struct Dg { u32 w[NW]; };

__device__ __forceinline__ u32 zflag(u32 x) { return ~(((x & 0x77777777u) + 0x77777777u) | x | 0x77777777u); }
__device__ __forceinline__ u32 compact8(u32 f) {
    u32 a = (f >> 3) & 0x11111111u;
    a = (a | (a >> 3)) & 0x03030303u;
    a = (a | (a >> 6)) & 0x000F000Fu;
    return (a | (a >> 12)) & 0xFFu;
}
template <int NW> __device__ __forceinline__ u64 compact_flags(const u32 (&f)[NW]) {
    u64 M = 0;
#pragma unroll
    for (int k = 0; k < NW; k++) M |= (u64)compact8(f[k]) << (56 - 8 * k);
    return M;
}
template <int NW> __device__ __forceinline__ u64 vis_mask(int zl) { return (~0ull >> zl) & (~0ull << (64 - 8 * NW)); }

/* the digest shifted left by d nibbles (d = 0..8), zero filled */
template <int NW> __device__ __forceinline__ void shl_nib(const u32 (&w)[NW], int d, u32 (&o)[NW]) {
#pragma unroll
    for (int i = 0; i < NW; i++) {
        const u32 n1 = i + 1 < NW ? w[i + 1] : 0u;
        o[i] = d == 0 ? w[i] : d >= 8 ? n1 : __funnelshift_l(n1, w[i], 4 * d);
    }
}
/* 32-bit window of digits o..o+7 (constant o) */
template <int NW> __device__ __forceinline__ u32 win(const u32 (&a)[NW], int o) {
    const int k = o >> 3, sh = 4 * (o & 7);
    const u32 w0 = k < NW ? a[k < NW ? k : 0] : 0u, w1 = k + 1 < NW ? a[k + 1 < NW ? k + 1 : 0] : 0u;
    return sh ? __funnelshift_l(w1, w0, sh) : w0;
}

/* longest run of ones (digits) if it is >= need, else FAILV. need <= 0 means any. M: bit 63-i = digit i. */
__device__ __forceinline__ int run_len(u64 M, int need) {
    if (M == 0) return need <= 0 ? 0 : FAILV;
    const int r = need < 1 ? 1 : need;
    if (r > 64) return FAILV;
    u64 A = M;
    int len = 1;
    while (len * 2 <= r) { A &= A << len; len *= 2; }
    if (len < r) A &= A << (r - len);
    if (!A) return FAILV;
    int R = r;
    while (R < 64) {
        const u64 B = A & (M << R);
        if (!B) break;
        A = B; R++;
    }
    return R;
}

template <int NW> __device__ __forceinline__ int popc_flags(const u32 (&f)[NW]) {
    int c = 0;
#pragma unroll
    for (int k = 0; k < NW; k++) c += __popc(f[k] & 0x88888888u);
    return c;
}
/* membership flags of the digit set S (16 bits) */
template <int NW> __device__ __forceinline__ void set_flags(const u32 (&w)[NW], u32 S, u32 (&f)[NW]) {
    const bool inv = __popc(S) > 8;
    const u32 T = inv ? (~S & 0xFFFFu) : S;
#pragma unroll
    for (int k = 0; k < NW; k++) f[k] = 0;
    for (u32 t = T; t; t &= t - 1) {
        const u32 v = __ffs(t) - 1;
#pragma unroll
        for (int k = 0; k < NW; k++) f[k] |= zflag(w[k] ^ (v * 0x11111111u));
    }
#pragma unroll
    for (int k = 0; k < NW; k++) f[k] = inv ? (~f[k] & 0x88888888u) : f[k];
}

template <int NW> __device__ __noinline__ int pal_longest(Dg<NW> dg, int zl) {
    const int W = 8 * NW;
    auto nib = [&](int i) { return (dg.w[i >> 3] >> (28 - 4 * (i & 7))) & 15u; };
    int best = W > zl ? 1 : 0;
    for (int c = zl; c < W; c++)
        for (int odd = 1; odd >= 0; odd--) {
            int a = odd ? c - 1 : c, b = c + 1;
            while (a >= zl && b < W && nib(a) == nib(b)) { a--; b++; }
            if (b - a - 1 > best) best = b - a - 1;
        }
    return best;
}

template <int NW> __device__ __noinline__ int eval_pal(Dg<NW> dg, int zl, int need) {
    const int W = 8 * NW;
    const u32 (&h)[NW] = dg.w;
    if (need > 1) {
        u32 r[NW];
#pragma unroll
        for (int k = 0; k < NW; k++) {
            u32 x = __byte_perm(h[NW - 1 - k], 0, 0x0123);
            r[k] = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
        }
        const int kE = min((need + 1) / 2, 8), kO = min(need / 2, 8);
        const u32 mE = kE >= 8 ? ~0u : ~0u << (32 - 4 * kE), mO = kO >= 8 ? ~0u : ~0u << (32 - 4 * kO);
        u32 mn = 0xFFFFFFFFu;
#pragma unroll
        for (int c = 0; c < W - 1; c++) {
            const u32 hw = win(h, c + 1);
            mn = min(mn, (hw ^ win(r, W - 1 - c)) & mE);
            if (c >= 1) mn = min(mn, (hw ^ win(r, W - c)) & mO);
        }
        if (mn != 0u) return FAILV;
    }
    return 0;
}

template <int NW> __device__ __forceinline__ bool hexmask_ok(const u32 (&w)[NW], u32 idx) {
    u32 acc = 0;
#pragma unroll
    for (int k = 0; k < NW; k++) acc |= (w[k] ^ c_sm[idx * 16 + 8 + k]) & c_sm[idx * 16 + k];
    return acc == 0u;
}

template <int NW> __device__ __noinline__ int count_words(Dg<NW> dg, int zl, u32 first, u32 nw) {
    constexpr int W = 8 * NW;
    int cnt = 0;
#pragma unroll
    for (int o = 0; o < W; o++) {
        const u32 x = win(dg.w, o);
        for (u32 j = 0; j < nw; j++) {
            const u32 wi = first + j;
            cnt += ((x ^ c_sw[wi * 3]) & c_sw[wi * 3 + 1]) == 0u && o >= zl && o + (int)c_sw[wi * 3 + 2] <= W;
        }
    }
    return cnt;
}

/* The value of one op, or FAILV when a run-type op cannot reach `need` (digits). Scalar ops (count, distinct,
 * words) always return their value; the caller applies the range. */
template <int NW> __device__ __forceinline__ int eval_op(const StatOp &op, const Dg<NW> &dg, int zl, int need) {
    constexpr int W = 8 * NW;
    const u32 (&w)[NW] = dg.w;
    const int n = W - zl;
    switch (op.kind) {
    case SK_COUNT: {
        const u32 S = op.a;
        int c;
        if (S == 0xFFFFu) c = W;
        else if (S == 0xFC00u) {                      /* letters a-f: b3 & (b2 | b1) */
            c = 0;
#pragma unroll
            for (int k = 0; k < NW; k++) c += __popc(w[k] & ((w[k] << 1) | (w[k] << 2)) & 0x88888888u);
        } else if (S == 0x03FFu) {
            c = 0;
#pragma unroll
            for (int k = 0; k < NW; k++) c += __popc(w[k] & ((w[k] << 1) | (w[k] << 2)) & 0x88888888u);
            c = W - c;
        } else {
            u32 f[NW];
            set_flags<NW>(w, S, f);
            c = popc_flags<NW>(f);
        }
        return c - ((S & 1u) ? zl : 0);
    }
    case SK_DIST: {
        u32 m = 0;
#pragma unroll
        for (int i = 0; i < W; i++) m |= 1u << ((w[i >> 3] >> (28 - 4 * (i & 7))) & 15u);
        int z = 0;
#pragma unroll
        for (int k = 0; k < NW; k++) z += __popc(zflag(w[k]) & 0x88888888u);
        m = (m & ~1u) | (z > zl ? 1u : 0u);
        return __popc(m);
    }
    case SK_PER: {
        const int p = (int)op.a;
        u32 sh[NW], f[NW];
        shl_nib<NW>(w, p, sh);
#pragma unroll
        for (int k = 0; k < NW; k++) f[k] = zflag(w[k] ^ sh[k]);
        u64 M = compact_flags<NW>(f) & vis_mask<NW>(zl) & (~0ull << (64 - W + p));
        const int R = run_len(M, need - p);
        if (R == FAILV) return FAILV;
        const int ln = R > 0 ? R + p : (p < n ? p : n);
        return p * (ln / p);
    }
    case SK_UP: case SK_DOWN: {
        u32 f[NW], cur[NW], nxt[NW];
#pragma unroll
        for (int k = 0; k < NW; k++) { cur[k] = op.kind == SK_UP ? w[k] : ~w[k]; }
        shl_nib<NW>(cur, 1, nxt);
        /* nxt is shifted from ~w too, but its zero fill must stay a non-digit: positions past the end are masked */
#pragma unroll
        for (int k = 0; k < NW; k++) {
            const u32 c = cur[k];
            const u32 inc = ((c & 0x77777777u) + 0x11111111u) ^ (c & 0x88888888u);
            const u32 is15 = c & (c << 1) & (c << 2) & (c << 3) & 0x88888888u;
            f[k] = zflag(nxt[k] ^ inc) & ~is15;
        }
        u64 M = compact_flags<NW>(f) & vis_mask<NW>(zl) & (~0ull << (64 - W + 1));
        const int R = run_len(M, need - 1);
        return R == FAILV ? FAILV : R + 1;
    }
    case SK_PAL: {
        if (eval_pal<NW>(dg, zl, need) == FAILV) return FAILV;
        const int v = pal_longest<NW>(dg, zl);
        return v >= need ? v : FAILV;
    }
    case SK_WORDS:
        return count_words<NW>(dg, zl, op.a, op.d);
    case SK_HEXMASK:
        return hexmask_ok<NW>(w, op.a) ? 1 : FAILV;
    case SK_HEXREC: {
        if (op.e != 0xFFFFFFFFu && !hexmask_ok<NW>(w, op.e)) return FAILV;
        u32 f[NW];
        set_flags<NW>(w, op.d, f);
        const u64 M = compact_flags<NW>(f);
        const int npre = (int)op.a;
        const u64 x = ~(M << npre);
        const int run = x ? __clzll(x) : 64;
        return run > W - npre ? W - npre : run;
    }
    default: {   /* SK_HEXRUN: longest run of the token set anywhere in the visible id */
        u32 f[NW];
        set_flags<NW>(w, op.d, f);
        const u64 M = compact_flags<NW>(f) & vis_mask<NW>(zl);
        return run_len(M, need);
    }
    }
}

/* One line: every term holds; for a record line the record term must beat `best`. val: its value. */
template <int NW> __device__ __forceinline__ bool stat_line(u32 li, const Dg<NW> &dg, int zl, int best, int *val) {
    const StatLineC L = c_sl[li];
    for (u32 j = 0; j < L.n; j++) {
        const StatOp op = c_sop[L.first + j];
        const bool scalar = op.kind == SK_COUNT || op.kind == SK_DIST || op.kind == SK_WORDS;
        if (!op.rec) {
            const int v = eval_op<NW>(op, dg, zl, (int)op.b);
            if (v == FAILV) return false;
            if (scalar && (v < (int)op.b || v > (int)op.c)) return false;
        } else {
            const int step = op.kind == SK_PER ? (int)op.a : 1;
            const int v = eval_op<NW>(op, dg, zl, best + step);
            if (v == FAILV) return false;
            if (op.rec == 1 ? v <= best : v >= best) return false;
            *val = v;
        }
    }
    return true;
}

template <int NW> __device__ __forceinline__ int stat_zl(const Dg<NW> &dg) {
    if (NW < 8) return 0;
    int zl = 0;
    bool z = true;
#pragma unroll
    for (int k = 0; k < NW; k++) {
        const int add = dg.w[k] ? (__clz(dg.w[k]) >> 2) : 8;
        zl += z ? add : 0;
        z = z && dg.w[k] == 0u;
    }
    return min(zl, 63);
}

/* All lines on a digest. True if the candidate must be reported: a threshold line holds, or a record line has a
 * new best (exact lines claim the device best with an atomic, the others compare with the host-updated best). */
#ifndef STAT_CALL
#define STAT_CALL __noinline__
#endif
template <int NW> __device__ STAT_CALL bool stat_check(Dg<NW> dg) {
    const int zl = stat_zl<NW>(dg);
    bool report = false;
    for (u32 li = 0; li < c_snl; li++) {
        const StatLineC L = c_sl[li];
        int v = 0;
        if (!L.rec) {
            if (!report && stat_line<NW>(li, dg, zl, 0, &v)) report = true;
        } else {
            const int best = *(volatile int *)&d_rbest[li];
            if (stat_line<NW>(li, dg, zl, best, &v)) {
                if (L.exact) {
                    const int old = L.rec == 1 ? atomicMax(&d_rbest[li], v) : atomicMin(&d_rbest[li], v);
                    if (L.rec == 1 ? v > old : v < old) report = true;
                } else report = true;
            }
        }
    }
    return report;
}

/* self-test kernel: per digest and line, the pass flag (bit 0) and value, with the best of the line given. */
template <int NW> __global__ void stat_selftest(const u32 *dgs, const int *bests, int n, int *out) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    Dg<NW> dg;
    for (int k = 0; k < NW; k++) dg.w[k] = dgs[g * NW + k];
    const int zl = stat_zl<NW>(dg);
    for (u32 li = 0; li < c_snl; li++) {
        int v = 0;
        const bool ok = stat_line<NW>(li, dg, zl, bests[li], &v);
        out[g * MAXSL * 2 + li * 2] = ok;
        out[g * MAXSL * 2 + li * 2 + 1] = v;
    }
}

/* ---- host: compile the parsed lines into the tables ---- */
struct StatTables {
    std::vector<StatOp> ops; std::vector<StatLineC> lines; std::vector<u32> words, masks;
    int nl = 0;
};

static inline bool st_nibble_exact(uint16_t s) {
    uint32_t m, v;
    mpat_nib(s, &m, &v);
    for (int d = 0; d < 16; d++) if (((d & m) == v) != ((s >> d & 1) != 0)) return false;
    return true;
}

static inline bool stat_build(const std::vector<sline_t> &ls, int NWd, StatTables &T, char *err, int errlen) {
    T = StatTables();
    for (const sline_t &L : ls) {
        StatLineC lc; lc.first = (u32)T.ops.size(); lc.n = 0; lc.exact = 1;
        lc.rec = L.rec_term >= 0 ? (u32)L.t[L.rec_term].rec : 0;
        for (int k = 0; k < L.nterm; k++) {
            const sterm_t &t = L.t[k];
            StatOp op; memset(&op, 0, sizeof op);
            op.rec = (u32)t.rec; op.b = (u32)t.lo; op.c = t.hi < 0 ? (u32)INT_MAX : (u32)t.hi; op.e = 0xFFFFFFFFu;
            switch (t.kind) {
            case ST_COUNT: op.kind = SK_COUNT; op.a = t.cset; break;
            case ST_DISTINCT: op.kind = SK_DIST; break;
            case ST_PERIOD: op.kind = SK_PER; op.a = (u32)t.period; break;
            case ST_UP: op.kind = SK_UP; break;
            case ST_DOWN: op.kind = SK_DOWN; break;
            case ST_PAL: op.kind = SK_PAL; break;
            case ST_WORDS:
                op.kind = SK_WORDS; op.a = (u32)(T.words.size() / 3); op.d = (u32)t.nwords;
                for (int i = 0; i < t.nwords; i++) {
                    u32 v = 0, m = 0;
                    for (int j = 0; j < t.wlen[i]; j++) { v |= (u32)t.word[i][j] << (28 - 4 * j); m |= 0xFu << (28 - 4 * j); }
                    T.words.push_back(v); T.words.push_back(m); T.words.push_back((u32)t.wlen[i]);
                }
                if (T.words.size() / 3 > MAXSW) { snprintf(err, (size_t)errlen, "too many words"); return false; }
                break;
            case ST_HEX:
                if (!t.rec) {
                    lc.exact = 0;
                    if (!(t.pat.start || t.pat.end)) continue;   /* nothing for the GPU to test; the host checks it */
                    op.kind = SK_HEXMASK; op.a = (u32)(T.masks.size() / 16);
                    for (int i = 0; i < 8; i++) T.masks.push_back(i < NWd ? t.pat.am[i] : 0);
                    for (int i = 0; i < 8; i++) T.masks.push_back(i < NWd ? t.pat.av[i] : 0);
                } else {
                    op.d = t.tok;
                    if (t.start) {
                        op.kind = SK_HEXREC; op.a = (u32)t.npre;
                        if (t.npre) {
                            u32 am[8] = {0}, av[8] = {0};
                            for (int i = 0; i < t.npre; i++) {
                                u32 m, v; mpat_nib(t.pre[i], &m, &v);
                                am[i >> 3] |= m << (28 - 4 * (i & 7)); av[i >> 3] |= v << (28 - 4 * (i & 7));
                                if (!st_nibble_exact(t.pre[i])) lc.exact = 0;
                            }
                            op.e = (u32)(T.masks.size() / 16);
                            for (int i = 0; i < 8; i++) T.masks.push_back(am[i]);
                            for (int i = 0; i < 8; i++) T.masks.push_back(av[i]);
                        }
                    } else {
                        op.kind = SK_HEXRUN;
                        if (t.npre) lc.exact = 0;
                    }
                }
                break;
            }
            if (T.ops.size() >= MAXSOP) { snprintf(err, (size_t)errlen, "too many stat terms (max %d)", MAXSOP); return false; }
            if (T.masks.size() / 16 > MAXSM) { snprintf(err, (size_t)errlen, "too many anchored hex terms in stat lines (max %d)", MAXSM); return false; }
            /* cheap terms first: the order of patterns.py KIND_ORDER */
            T.ops.push_back(op); lc.n++;
        }
        if (lc.n == 0) { snprintf(err, (size_t)errlen, "stat line '%s' has nothing the GPU can test", L.text); return false; }
        /* sort this line's ops by cost (stable); the record op keeps working wherever it is */
        auto cost = [](const StatOp &o) { return o.kind == SK_HEXMASK || o.kind == SK_HEXREC ? 0 : o.kind == SK_COUNT ? 1 : o.kind == SK_DIST ? 2 : o.kind == SK_WORDS ? 5 : o.kind == SK_PAL ? 6 : 3; };
        std::stable_sort(T.ops.begin() + lc.first, T.ops.end(), [&](const StatOp &a, const StatOp &b) { return cost(a) < cost(b); });
        T.lines.push_back(lc);
    }
    T.nl = (int)T.lines.size();
    return true;
}

template <int NWd> static inline void stat_upload(const StatTables &T, const int *best_init) {
    StatOp ops[MAXSOP]; memset(ops, 0, sizeof ops);
    StatLineC ls[MAXSL]; memset(ls, 0, sizeof ls);
    u32 sw[MAXSW * 3] = {0}, sm[MAXSM * 16] = {0};
    int rb[MAXSL];
    for (size_t i = 0; i < T.ops.size(); i++) ops[i] = T.ops[i];
    for (size_t i = 0; i < T.lines.size(); i++) { ls[i] = T.lines[i]; rb[i] = best_init[i]; }
    for (size_t i = T.lines.size(); i < MAXSL; i++) rb[i] = 0;
    for (size_t i = 0; i < T.words.size(); i++) sw[i] = T.words[i];
    for (size_t i = 0; i < T.masks.size(); i++) sm[i] = T.masks[i];
    u32 nl = (u32)T.nl;
    cudaMemcpyToSymbol(c_sop, ops, sizeof ops);
    cudaMemcpyToSymbol(c_sl, ls, sizeof ls);
    cudaMemcpyToSymbol(c_snl, &nl, 4);
    cudaMemcpyToSymbol(c_sw, sw, sizeof sw);
    cudaMemcpyToSymbol(c_sm, sm, sizeof sm);
    cudaMemcpyToSymbol(d_rbest, rb, sizeof rb);
}

/* ---- self-test: the device value against st_eval on random and crafted digests ---- */
static inline u64 st_rng(u64 &s) { s += 0x9E3779B97F4A7C15ull; u64 z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }

/* digits of a test id: random, small alphabets, planted runs / staircases / palindromes / words, leading zeros */
static inline void st_make_digest(u64 &rng, int W, uint8_t *h) {
    static const char *WORDS[] = {"dead", "beef", "cafe", "a", "0", "ff", "c0ffee", "deadbeef", "00"};
    int alpha = (int)(st_rng(rng) % 5);
    int na = alpha == 0 ? 16 : alpha == 1 ? 2 : alpha == 2 ? 3 : alpha == 3 ? 4 : 6;
    uint8_t al[16];
    for (int i = 0; i < 16; i++) al[i] = (uint8_t)i;
    for (int i = 15; i > 0; i--) { int j = (int)(st_rng(rng) % (u64)(i + 1)); uint8_t t = al[i]; al[i] = al[j]; al[j] = t; }
    for (int i = 0; i < W; i++) h[i] = al[st_rng(rng) % (u64)na];
    int plant = (int)(st_rng(rng) % 7);
    int len = 2 + (int)(st_rng(rng) % 28), at = (int)(st_rng(rng) % (u64)W);
    if (at + len > W) len = W - at;
    if (plant == 1) { int p = 1 + (int)(st_rng(rng) % 8); for (int i = p; i < len; i++) h[at + i] = h[at + i - p]; }
    else if (plant == 2) { for (int i = 1; i < len; i++) h[at + i] = (uint8_t)((h[at + i - 1] + 1) & 15); }
    else if (plant == 3) { for (int i = 1; i < len; i++) h[at + i] = (uint8_t)((h[at + i - 1] + 15) & 15); }
    else if (plant == 4) { for (int i = 0; i < len / 2; i++) h[at + len - 1 - i] = h[at + i]; }
    else if (plant == 5) { const char *w = WORDS[st_rng(rng) % 9]; for (int i = 0; w[i] && at + i < W; i++) h[at + i] = (uint8_t)mpat_hexval(w[i]); }
    if (W == 64 && st_rng(rng) % 3 == 0) { int z = (int)(st_rng(rng) % 22); if (st_rng(rng) % 40 == 0) z = 64; for (int i = 0; i < z && i < W; i++) h[i] = 0; }
    if (W == 16 && st_rng(rng) % 5 == 0) { int z = (int)(st_rng(rng) % 8); for (int i = 0; i < z; i++) h[i] = 0; }
}

template <int NW> static inline int stat_selftest_main(const std::vector<sline_t> &ls, const StatTables &T) {
    const int W = 8 * NW, N = 120000, nl = (int)ls.size();
    std::vector<u32> dg((size_t)N * NW);
    std::vector<uint8_t> nib((size_t)N * 64);
    u64 rng = 12345;
    for (int g = 0; g < N; g++) {
        st_make_digest(rng, W, &nib[(size_t)g * 64]);
        for (int k = 0; k < NW; k++) {
            u32 w = 0;
            for (int j = 0; j < 8; j++) w |= (u32)nib[(size_t)g * 64 + 8 * k + j] << (28 - 4 * j);
            dg[(size_t)g * NW + k] = w;
        }
    }
    u32 *d_dg; int *d_best, *d_out;
    cudaMalloc(&d_dg, dg.size() * 4); cudaMalloc(&d_best, MAXSL * 4); cudaMalloc(&d_out, (size_t)N * MAXSL * 2 * 4);
    cudaMemcpy(d_dg, dg.data(), dg.size() * 4, cudaMemcpyHostToDevice);
    long long bad = 0, passes = 0;
    for (int round = 0; round < 2; round++) {
        int best[MAXSL] = {0};
        for (int li = 0; li < nl; li++) {
            const int r = ls[li].rec_term >= 0 ? ls[li].t[ls[li].rec_term].rec : 0;
            best[li] = r == 2 ? ST_NONE_MIN : ST_NONE_MAX;
            if (r && round == 1) {   /* a best that about half of the ids beat: the value of a sample id */
                int v;
                for (int g = 0; g < 400; g++) if (st_eval(&ls[li], &nib[(size_t)g * 64], &v)) { best[li] = v; if (g > 20 + li) break; }
            }
        }
        cudaMemcpy(d_best, best, sizeof best, cudaMemcpyHostToDevice);
        stat_selftest<NW><<<(N + 127) / 128, 128>>>(d_dg, d_best, N, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "stat selftest: CUDA error\n"); return 6; }
        std::vector<int> out((size_t)N * MAXSL * 2);
        cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost);
        for (int g = 0; g < N; g++)
            for (int li = 0; li < nl; li++) {
                int v = 0;
                bool want = st_eval(&ls[li], &nib[(size_t)g * 64], &v) != 0;
                const int r = ls[li].rec_term >= 0 ? ls[li].t[ls[li].rec_term].rec : 0;
                if (want && r) want = st_beats(r, v, best[li]);
                if (!T.lines[li].exact && !r) {}   /* threshold lines: the GPU test must be implied */
                bool got = out[(size_t)g * MAXSL * 2 + li * 2] != 0;
                bool hexonly_loose = false;
                for (int k = 0; k < ls[li].nterm; k++) if (ls[li].t[k].kind == ST_HEX && !ls[li].t[k].rec) hexonly_loose = true;
                bool fine = got == want || (hexonly_loose && got && !want) || (!T.lines[li].exact && r && got && !want);
                if (got && want) passes++;
                if (!fine || (got && want && r && out[(size_t)g * MAXSL * 2 + li * 2 + 1] != v && T.lines[li].exact)) {
                    if (bad < 8) fprintf(stderr, "stat selftest MISMATCH line %d '%s' digest %d: gpu %d (value %d) reference %d (value %d)\n", li, ls[li].text, g, got, out[(size_t)g * MAXSL * 2 + li * 2 + 1], want, v);
                    bad++;
                }
            }
    }
    cudaFree(d_dg); cudaFree(d_best); cudaFree(d_out);
    printf("stat selftest: %s (%d digests x %d lines x 2 rounds, %lld passing)\n", bad ? "FAILED" : "OK", N, nl, passes);
    return bad ? 5 : 0;
}

/* `exe --stat-selftest -m LINE...`: no job needed. */
template <int NW> static inline int stat_selftest_cli(int argc, char **argv) {
    std::vector<sline_t> ls;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-m") || i + 1 >= argc) { fprintf(stderr, "usage: %s --stat-selftest -m STATLINE...\n", argv[0]); return 1; }
        sline_t sl; char err[256];
        if (sline_parse(argv[++i], 8 * NW, &sl, err, sizeof err) || ls.size() >= ST_MAXLINES) { fprintf(stderr, "%s\n", err); return 1; }
        ls.push_back(sl);
    }
    StatTables T; char err[256];
    if (!stat_build(ls, NW, T, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    int best[ST_MAXLINES] = {0};
    stat_upload<NW>(T, best);
    return stat_selftest_main<NW>(ls, T);
}
