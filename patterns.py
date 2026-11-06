"""Hex patterns for skinhash targets (-m/--match). Reference implementation; web/js/patterns.js and pattern.h
implement the same rules.

Syntax (case-insensitive, normalised to lowercase):
  c0ffee      hex literal, anywhere in the id
  ^ / $       anchor to the start / end (only as the first / last character)
  ?           any hex digit        ?d  0-9        ?l  a-f
  [0-3] [a-f] [05af]               a class: ranges and lists
  {n}         repeat the previous digit, ?, ?d, ?l or class n times (n >= 1)
  *           a gap of any length (also 0) between two pieces: ^cafe*babe$

Ids: Mojang = the full 64-digit SHA-256. ^ is the start of those 64 digits, so ^0{12} means 12 leading zeros.
A pattern without ^ must lie in the link id (leading zeros dropped), the same rule as --contains.
NameMC = the 16-digit id as shown.

Compilation ("masks"): every token becomes the tightest nibble mask/value that contains its set ([0-7] -> mask 8
value 0, [a-f] -> mask 8 value 8, ?d -> no constraint). Each pattern gets one GPU prefilter:
  anchored: one mask/value over the id (the ^ piece and/or the $ piece), or
  sliding:  a window of up to 16 tokens of one unanchored piece, compared at every offset in a range.
The finder re-checks the full pattern on the host for every candidate that passes the prefilter.

Stat lines (record categories) are parsed by parse_stat() below: counts (#0{24,}), different digits (##{,9}),
shapes (~{25} palindrome, +{12} / -{12} staircase, (?){11} periodic run), {max} / {min} record mode and A & B.
"""
import math
import os
import re

WIDTHS = {"mojang": 64, "namemc": 16}
HEXDIGITS = "0123456789abcdef"
ALL = 0xFFFF
DIGITS = 0x03FF      # 0-9
LETTERS = 0xFC00     # a-f
MAX_PATTERNS = 64    # per group (the GPU tables)
MAX_SLIDING = 32     # sliding prefilters per group, --contains patterns included
WARN_ODDS = 1e-7     # warn when a pattern (or its GPU prefilter) passes more often than this


class PatternError(ValueError):
    pass


# ---- tokenizer ----------------------------------------------------------------------------------------------
# token: (kind, value, pos). kinds: lit (nibble set), any/dig/let (nibble set), class (nibble set), rep (n),
# start (^), end ($), gap (*), op (>= <= ( ) &), word (a run of letters/digits that is not all hex), bad (char)
def tokenize(line):
    toks, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            toks.append(("space", None, i)); i += 1
        elif c == "^":
            toks.append(("start", None, i)); i += 1
        elif c == "$":
            toks.append(("end", None, i)); i += 1
        elif c == "*":
            toks.append(("gap", None, i)); i += 1
        elif line.startswith((">=", "<="), i):
            toks.append(("op", line[i:i + 2], i)); i += 2
        elif c in "()&":
            toks.append(("op", c, i)); i += 1
        elif c == "?":
            nx = line[i + 1:i + 2].lower()
            if nx == "d":
                toks.append(("dig", DIGITS, i)); i += 2
            elif nx == "l":
                toks.append(("let", LETTERS, i)); i += 2
            else:
                toks.append(("any", ALL, i)); i += 1
        elif c == "[":
            j = line.find("]", i + 1)
            k = line.find("[", i + 1)
            if j < 0 or (0 <= k < j):
                raise PatternError(f"unbalanced '[' at position {i + 1}")
            toks.append(("class", parse_class(line[i + 1:j], i), i)); i = j + 1
        elif c == "]":
            raise PatternError(f"unbalanced ']' at position {i + 1}")
        elif c == "{":
            j = line.find("}", i + 1)
            if j < 0:
                raise PatternError(f"unclosed '{{' at position {i + 1}")
            body = line[i + 1:j]
            if not re.fullmatch(r"\d+", body):
                raise PatternError(f"'{{{body}}}' at position {i + 1}: a repeat count must be a number")
            if int(body) == 0:
                raise PatternError(f"'{{0}}' at position {i + 1}: a repeat count must be at least 1")
            toks.append(("rep", int(body), i)); i = j + 1
        elif c == "}":
            raise PatternError(f"unbalanced '}}' at position {i + 1}")
        elif c.isalnum() or c == "_":
            j = i
            while j < n and (line[j].isalnum() or line[j] == "_"):
                j += 1
            run = line[i:j].lower()
            if all(ch in HEXDIGITS for ch in run):
                for k, ch in enumerate(run):
                    toks.append(("lit", 1 << HEXDIGITS.index(ch), i + k))
            else:
                toks.append(("word", run, i))
            i = j
        else:
            toks.append(("bad", c, i)); i += 1
    return toks


def parse_class(body, at):
    if not body:
        raise PatternError(f"empty class '[]' at position {at + 1}")
    if body.startswith("^"):
        raise PatternError(f"'[^...]' at position {at + 1}: negated classes are not supported")
    s, i, b = 0, 0, body.lower()
    while i < len(b):
        c = b[i]
        if c not in HEXDIGITS:
            raise PatternError(f"'{body[i]}' in class '[{body}]' at position {at + 1}: only hex digits and ranges like 0-7")
        if i + 2 < len(b) and b[i + 1] == "-":
            e = b[i + 2]
            if e not in HEXDIGITS:
                raise PatternError(f"bad range '{body[i:i + 3]}' in class '[{body}]'")
            lo, hi = HEXDIGITS.index(c), HEXDIGITS.index(e)
            if lo > hi:
                raise PatternError(f"range '{body[i:i + 3]}' in class '[{body}]' goes backwards")
            for v in range(lo, hi + 1):
                s |= 1 << v
            i += 3
        elif i + 1 < len(b) and b[i + 1] == "-":
            raise PatternError(f"range '{body[i:]}' in class '[{body}]' has no end")
        else:
            s |= 1 << HEXDIGITS.index(c)
            i += 1
    return s


ENGLISH = {"zeros": "#0{24,} (at least 24 zeros)", "zero": "#0{24,} (at least 24 zeros)",
           "letters": "#?l{50,} or #?l{,5}", "digits": "#?d{50,} or ##{,9} (different digits)",
           "palindrome": "~{25} or ~{max}", "staircase": "+{12} (up) or -{12} (down)", "stairs": "+{12} or -{12}",
           "run": r"(?)\1{11} (a digit 12x in a row) or 0{max}", "zigzag": r"(??)\1{6}", "best": "{max} or {min}",
           "words": "#(dead|beef){3,} or #@words.txt{3,}", "distinct": "##{,9}", "different": "##{,9}"}


def english_hint(toks):
    """The old reserved forms (zeros >= 24, palindrome best): an error that shows the expression syntax."""
    for kind, v, pos in toks:
        if kind == "op" and v in (">=", "<="):
            return f"'{v}' at position {pos + 1} is not part of the syntax: counts are written #0{{24,}} (at least), #0{{,3}} (at most)"
        if kind == "word" and v in ENGLISH:
            return f"'{v}' at position {pos + 1}: words are not part of the syntax, write {ENGLISH[v]}"
    return None


# ---- one line -> target -------------------------------------------------------------------------------------
def nib_mask(s):
    """Tightest (mask, value) over 4 bits containing every nibble of set s."""
    a, o = 0xF, 0
    for v in range(16):
        if s >> v & 1:
            a &= v
            o |= v
    m = ~(a ^ o) & 0xF
    return m, a & m


def popcount(x):
    return bin(x).count("1")


class Pattern:
    """A parsed pattern: pieces of nibble sets, anchors, the id width (64 Mojang, 16 NameMC)."""

    def __init__(self, text, width, start, end, pieces):
        self.text, self.width, self.start, self.end, self.pieces = text, width, start, end, pieces

    def __repr__(self):
        return f"Pattern({self.text!r}, {self.width})"

    @property
    def mojang(self):
        return self.width == 64

    def describe(self):
        def tok(s):
            if s == ALL:
                return "?"
            if s == DIGITS:
                return "?d"
            if s == LETTERS:
                return "?l"
            if popcount(s) == 1:
                return HEXDIGITS[s.bit_length() - 1]
            return "[" + "".join(HEXDIGITS[v] for v in range(16) if s >> v & 1) + "]"
        out = []
        for p in self.pieces:
            parts, i = [], 0
            while i < len(p):
                j = i
                while j < len(p) and p[j] == p[i]:
                    j += 1
                t, r = tok(p[i]), j - i
                if len(t) == 1 and t != "?" and r < 4:   # short literal runs stay literal text: c0ffee
                    if parts and parts[-1][0] in HEXDIGITS:
                        parts[-1] += t * r
                    else:
                        parts.append(t * r)
                else:
                    parts.append(t if r == 1 else f"{t}{{{r}}}")
                i = j
            out.append(" ".join(parts))
        return ("^ " if self.start else "") + " * ".join(out) + (" $" if self.end else "")

    # -- matching (host re-check, verification) --
    def match(self, hexid):
        """Does the id match? hexid: the full 64-digit Mojang hash or the 16-digit NameMC id."""
        h = [HEXDIGITS.index(c) for c in hexid.lower()]
        L = len(h)
        zl = 0
        if L == 64 and not self.start:
            while zl < 63 and h[zl] == 0:
                zl += 1
        k = len(self.pieces)

        def at(p, pos):
            return all(p[i] >> h[pos + i] & 1 for i in range(len(p)))
        cur = 0 if self.start else zl
        limit = L - len(self.pieces[-1]) if self.end else L
        for i, p in enumerate(self.pieces):
            n = len(p)
            if i == 0 and self.start:
                if k == 1 and self.end and n != L:
                    return False
                if n > L or not at(p, 0):
                    return False
                cur = n
            elif i == k - 1 and self.end:
                pos = L - n
                if pos < cur or not at(p, pos):
                    return False
                cur = L
            else:
                end = limit if i < k - 1 else L
                pos = cur
                while pos + n <= end and not at(p, pos):
                    pos += 1
                if pos + n > end:
                    return False
                cur = pos + n
        return True

    # -- odds --
    def probability(self):
        """Chance that a random id matches (union bound with a Poisson correction)."""
        p = 1.0
        for piece in self.pieces:
            for s in piece:
                p *= popcount(s) / 16
        free = [len(x) for i, x in enumerate(self.pieces)
                if not (i == 0 and self.start) and not (i == len(self.pieces) - 1 and self.end)]
        anch = sum(len(x) for x in self.pieces) - sum(free)
        span, T, j = self.width - anch, sum(free), len(free)
        if span < T:
            return 0.0
        if not j:
            return p
        placements = math.comb(span - T + j, j)
        return -math.expm1(-placements * p)

    # -- GPU prefilter --
    def anchored(self):
        """(mask, value) nibble lists over the whole id from the ^ / $ pieces, or None."""
        if not (self.start or self.end):
            return None
        L = self.width
        mask, val = [0] * L, [0] * L
        if self.start:
            for i, s in enumerate(self.pieces[0]):
                mask[i], val[i] = nib_mask(s)
        if self.end:
            p = self.pieces[-1]
            for i, s in enumerate(p):
                m, v = nib_mask(s)
                j = L - len(p) + i
                mask[j] |= m      # ^...$ without a gap: the same piece, same values
                val[j] |= v
        return mask, val

    def sliding(self, j):
        """Sliding prefilter for unanchored piece j: best window of up to 8 + 8 tokens."""
        p = self.pieces[j]
        tm = [nib_mask(s) for s in p]
        bits = [popcount(m) for m, _ in tm]
        best_d, best_b = 0, -1
        for d in range(len(p)):
            b = sum(bits[d:d + 8])
            if b > best_b:
                best_d, best_b = d, b
        d = best_d
        w1, w2 = tm[d:d + 8], tm[d + 8:d + 16]
        word = lambda w: (sum(v << (28 - 4 * i) for i, (m, v) in enumerate(w)),
                          sum(m << (28 - 4 * i) for i, (m, v) in enumerate(w)))
        p1, m1 = word(w1)
        p2, m2 = word(w2)
        before = sum(len(x) for x in self.pieces[:j])
        after = sum(len(x) for x in self.pieces[j:])
        lo = before + d
        lim = self.width - after + d
        usezl = self.mojang and not self.start
        nb = popcount(m1) + popcount(m2)
        positions = max(0, lim - lo + 1)
        return {"kind": "sliding", "piece": j, "d": d, "p1": p1, "m1": m1, "p2": p2, "m2": m2, "lo": lo, "lim": lim,
                "usezl": usezl, "bits": nb, "pass": min(1.0, positions * 2.0 ** -nb)}

    def prefilter(self):
        """The most selective single GPU check: anchored mask or one sliding window."""
        best = None
        a = self.anchored()
        if a:
            nb = sum(popcount(m) for m in a[0])
            best = {"kind": "anchored", "mask": a[0], "val": a[1], "bits": nb, "pass": 2.0 ** -nb}
        for j in range(len(self.pieces)):
            if (j == 0 and self.start) or (j == len(self.pieces) - 1 and self.end):
                continue
            s = self.sliding(j)
            if best is None or s["pass"] < best["pass"]:
                best = s
        return best

    def words(self):
        """Anchored prefilter as big-endian 32-bit (mask, value) words (8 Mojang, 2 NameMC)."""
        f = self.prefilter()
        assert f["kind"] == "anchored"
        n = self.width // 8
        mw = [sum(f["mask"][8 * k + i] << (28 - 4 * i) for i in range(8)) for k in range(n)]
        vw = [sum(f["val"][8 * k + i] << (28 - 4 * i) for i in range(8)) for k in range(n)]
        return mw, vw

    def prefilter_pass(self, hexid, zl_cap=16):
        """Emulates the GPU prefilter on an id (zl capped as the Mojang kernel computes it)."""
        f = self.prefilter()
        h = [HEXDIGITS.index(c) for c in hexid]
        if f["kind"] == "anchored":
            return all((h[i] & f["mask"][i]) == f["val"][i] for i in range(len(h)))
        zl = 0
        while zl < min(zl_cap, len(h)) and h[zl] == 0:
            zl += 1
        lo = f["lo"] + (zl if f["usezl"] else 0)
        word = lambda o: sum((h[o + i] if o + i < len(h) else 0) << (28 - 4 * i) for i in range(8))
        for o in range(len(h)):
            if (word(o) ^ f["p1"]) & f["m1"] == 0 and (word(o + 8) ^ f["p2"]) & f["m2"] == 0 and lo <= o <= f["lim"]:
                return True
        return False


def parse_line(line, width, base=None):
    """One pattern line -> Pattern (a hex pattern) or StatLine (counts, shapes, records, A & B). Raises PatternError
    with a message that names the line. base: directory for #@words.txt (default: the current directory)."""
    text = line.strip()
    try:
        if is_stat(text):
            return parse_stat(text, width, base)
        return _parse(text, width)
    except PatternError as e:
        raise PatternError(f"pattern '{text}': {e}") from None


def _parse(text, width):
    if not text:
        raise PatternError("empty pattern")
    toks = tokenize(text)
    hint = english_hint(toks)
    if hint:
        raise PatternError(hint)
    for kind, v, pos in toks:
        if kind == "space":
            raise PatternError(f"space at position {pos + 1} (one pattern per line or per comma)")
        if kind == "bad":
            raise PatternError(f"'{v}' at position {pos + 1} is not allowed (hex digits, ? ?d ?l [..] {{n}} ^ $ *)")
        if kind == "word":
            bad = next(c for c in v if c not in HEXDIGITS)
            raise PatternError(f"'{bad}' at position {pos + 1 + v.index(bad)} is not a hex digit")
    start = bool(toks) and toks[0][0] == "start"
    end = len(toks) > 1 and toks[-1][0] == "end" or (len(toks) == 1 and toks[0][0] == "end")
    body = toks[(1 if start else 0):(len(toks) - 1 if end else len(toks))]
    for kind, v, pos in body:
        if kind == "start":
            raise PatternError(f"'^' at position {pos + 1}: only allowed as the first character")
        if kind == "end":
            raise PatternError(f"'$' at position {pos + 1}: only allowed as the last character")
    pieces, cur, prev = [], [], None
    for kind, v, pos in body:
        if kind == "rep":
            if prev not in ("lit", "any", "dig", "let", "class"):
                raise PatternError(f"'{{{v}}}' at position {pos + 1} has nothing to repeat")
            if v > width:
                raise PatternError(f"'{{{v}}}' at position {pos + 1} is longer than the id ({width} digits)")
            cur.extend([cur[-1]] * (v - 1))
        elif kind == "gap":
            if not cur:
                raise PatternError(f"'*' at position {pos + 1} " + ("follows another '*'" if pieces else
                                   "at the start does nothing: drop it"))
            pieces.append(cur)
            cur = []
        else:
            cur.append(v)
        prev = kind
    if not cur:
        if pieces:
            raise PatternError("'*' at the end does nothing: drop it")
        raise PatternError("no digits")
    pieces.append(cur)
    norm = "".join(text.lower().split())
    total = sum(len(p) for p in pieces)
    if total > width:
        raise PatternError(f"needs {total} digits, the id has only {width}")
    if start and end and len(pieces) == 1 and total != width:
        raise PatternError(f"with ^ and $ and no '*' the pattern must be exactly {width} digits (it is {total})")
    return Pattern(norm, width, start, end, pieces)


# ---- stat lines ----------------------------------------------------------------------------------------------
# A stat line is one or more terms joined by '&' (all must hold). Terms:
#   #S{range}        count of digits in S (a digit, ?, ?d, ?l or a class) in the id
#   #(w1|w2){range}  occurrences of hex words (overlaps count); #@FILE{range}: the words of a file
#   ##{range}        number of different digits
#   ~{n}  +{n}  -{n} a palindrome / staircase up / staircase down of at least n digits
#   (?)\1{n} (??)\1{n} (?{p})\1{n}   a run with period p (1..8) of p*(n+1) digits
#   hex pattern      as -m; X{max} on its last token: the longest such repeat (^f{max}, 0{max}, ^0{max})
# range: {n} exactly (counts) / at least n (shapes), {n,} {,n} {n,m}, {max} or {min} = record mode.
# Mojang: counts, ## and unanchored shapes look at the link id (leading zeros dropped), ^ at the full hash.
MAX_TERMS = 6         # terms per line
MAX_WORDS = 64        # words per count term
MAX_PERIOD = 8
MAX_STAT_OPS = 96     # GPU op table per group: the terms of all stat lines
MAX_STAT_WORDS = 64   # words of all word-count terms of a group (the GPU compares each at every offset)
MAX_WORD_LEN = 8      # a word is 1-8 hex digits: one 32-bit window compare on the GPU
MAX_STAT_LINES = 32
KIND_ORDER = {"hex": 0, "count": 2, "distinct": 3, "period": 4, "up": 4, "down": 4, "words": 5, "pal": 6}
SHAPES = {"~": "pal", "+": "up", "-": "down"}
RANGE_HELP = "{24,} (at least), {,3} (at most), {5} (exactly), {2,6} or {max} / {min}"


def is_stat(text):
    """Does the line use the stat syntax (rather than a plain hex pattern)?"""
    s = "".join(text.lower().split())
    return bool(s) and ("&" in s or s[0] in "#~+-(" or "\\" in s or "{max}" in s or "{min}" in s
                        or re.search(r"\{\d*,\d*\}", s) is not None)


def nibbles(hexid):
    return [HEXDIGITS.index(c) for c in hexid.lower()]


def zeros_of(h):
    """Leading zeros the link drops (Mojang, at most 63); 0 for a NameMC id."""
    if len(h) != 64:
        return 0
    z = 0
    while z < 63 and h[z] == 0:
        z += 1
    return z


def longest_period(v, p):
    """Longest substring of v with period p (v[i] == v[i+p] throughout), in whole periods: p * floor(len / p)."""
    run = best = 0
    for i in range(len(v) - p):
        run = run + 1 if v[i] == v[i + p] else 0
        best = max(best, run)
    ln = best + p if best else min(p, len(v))
    return p * (ln // p)


def longest_stair(v, up):
    run = best = 0
    for i in range(len(v) - 1):
        ok = (v[i] != 15 and v[i + 1] == v[i] + 1) if up else (v[i] != 0 and v[i + 1] == v[i] - 1)
        run = run + 1 if ok else 0
        best = max(best, run)
    return best + 1 if v else 0


def longest_palindrome(v):
    n, best = len(v), (1 if v else 0)
    for c in range(n):
        for odd in (1, 0):
            a, b = c - 1 if odd else c, c + 1
            while a >= 0 and b < n and v[a] == v[b]:
                a -= 1
                b += 1
            best = max(best, b - a - 1)
    return best


def count_words(v, words):
    total = 0
    for w in words:
        n = len(w)
        total += sum(1 for o in range(len(v) - n + 1) if v[o:o + n] == w)
    return total


# -- odds of the stats (Mojang: the link id, i.e. 64 digits minus the leading zeros, first digit 1-f) --
def _mix(width, f):
    """E[f(n, first_nonzero)] over the visible id: NameMC 16 uniform digits; Mojang 64 - z digits with z leading
    zeros (P = 16^-z * 15/16) and a first digit 1-f."""
    if width != 64:
        return f(width, False)
    return sum(16.0 ** -z * 15 / 16 * f(64 - z, True) for z in range(16))


def _count_p(cset, n, nz, lo, hi):
    k = popcount(cset)
    p = k / 16
    q = popcount(cset & ~1) / 15 if nz else p
    hi = n if hi is None else min(hi, n)
    tot = 0.0
    for x in range(lo, hi + 1):   # x = first digit in S (0/1) + Binomial(n - 1, p)
        for f, pf in ((0, 1 - q), (1, q)):
            r = x - f
            if 0 <= r <= n - 1 and pf:
                tot += pf * math.comb(n - 1, r) * p ** r * (1 - p) ** (n - 1 - r)
    return tot


def _distinct_p(n, nz, lo, hi):
    st = {(1, 0): 1.0} if nz else {(0, 0): 1.0}
    for _ in range(n - 1 if nz else n):
        new = {}
        for (j, z), p in st.items():
            for key, q in (((j, 1), 1 / 16), ((j, z), j / 16), ((j + 1, z), (15 - j) / 16)):
                if q:
                    new[key] = new.get(key, 0.0) + p * q
        st = new
    return sum(p for (j, z), p in st.items() if lo <= j + z and (hi is None or j + z <= hi))


def _run_p(m, r, q):
    """P(a run of >= r successes in m Bernoulli(q) trials)."""
    if r <= 0:
        return 1.0
    if m < r:
        return 0.0
    st, hit = [1.0] + [0.0] * (r - 1), 0.0
    for _ in range(m):
        new = [0.0] * r
        for k, p in enumerate(st):
            new[0] += p * (1 - q)
            if k + 1 >= r:
                hit += p * q
            else:
                new[k + 1] += p * q
        st = new
    return hit


def _period_p(n, p, L):
    """P(a substring of >= L digits with period p): the equalities v[i] == v[i+p] are independent, 1/16 each."""
    if L > n:
        return 0.0
    return 1.0 if L <= p else _run_p(n - p, L - p, 1 / 16)


def _stair_p(n, nz, L, up):
    if L <= 1:
        return 1.0
    if n < L:
        return 0.0
    st = {(d, 1): (1 / 15 if nz else 1 / 16) for d in range(1 if nz else 0, 16)}
    hit = 0.0
    for _ in range(n - 1):
        new, spread = {}, 0.0
        for (d, k), p in st.items():
            q = p / 16
            spread += q            # next digit uniform: run restarts at 1 ...
            nx = d + 1 if up else d - 1
            if 0 <= nx <= 15:      # ... except the next step of the stair
                new[(nx, 1)] = new.get((nx, 1), 0.0) - q
                if k + 1 >= L:
                    hit += q
                else:
                    new[(nx, k + 1)] = new.get((nx, k + 1), 0.0) + q
        for e in range(16):
            new[(e, 1)] = new.get((e, 1), 0.0) + spread
        st = new
    return hit


def _pal_p(n, L):
    """Poisson approximation: lambda = sum over centers of 16^-(pairs needed for >= L digits)."""
    if L <= 1:
        return 1.0
    if L > n:
        return 0.0
    he, ho = (L + 1) // 2, L // 2
    lam = sum(16.0 ** -he for c in range(n - 1) if min(c + 1, n - 1 - c) >= he)
    lam += sum(16.0 ** -ho for c in range(n) if min(c, n - 1 - c) >= ho)
    return -math.expm1(-lam)


def _poisson_range(lam, lo, hi):
    def cdf(k):   # P(X <= k)
        if k < 0:
            return 0.0
        t = s = math.exp(-lam)
        for i in range(1, k + 1):
            t *= lam / i
            s += t
        return min(1.0, s)
    return (1.0 if hi is None else cdf(hi)) - cdf(lo - 1)


class StatTerm:
    """One term of a stat line. value() is the reference (exact) value; the GPUs compute the same value."""

    def __init__(self, kind, text, width, lo=0, hi=None, rec=None, **kw):
        self.kind, self.text, self.width, self.lo, self.hi, self.rec = kind, text, width, lo, hi, rec
        self.cset = kw.get("cset", 0)
        self.words = kw.get("words")          # lists of nibbles
        self.wsrc = kw.get("wsrc")            # '#@file' source text, or None
        self.period = kw.get("period", 0)
        self.pattern = kw.get("pattern")      # hex term
        self.prefix, self.tok, self.start = kw.get("prefix"), kw.get("tok", 0), kw.get("start", False)
        self.end = kw.get("end", False)       # X{max}$: the run that ends the id

    def __repr__(self):
        return f"StatTerm({self.text!r})"

    def finder_text(self):
        """The text the finders parse: word files expanded to #(w1|w2|...)."""
        if self.kind == "words" and self.wsrc:
            rng = self.text[self.text.rfind("{"):]
            return "#(" + "|".join("".join(HEXDIGITS[c] for c in w) for w in self.words) + ")" + rng
        return self.text

    def hex_k(self, k):
        """{max} hex term with the token repeated k times: a Pattern, "always" (nothing left to check) or None (longer
        than the id)."""
        toks = self.prefix + [self.tok] * k
        if not toks:
            return "always"
        if len(toks) > self.width:
            return None
        return Pattern(self.text, self.width, self.start, self.end, [toks])

    def end_run(self, h):
        """X{max}$: length of the run of the token that ends the full id."""
        run = 0
        while run < len(h) and self.tok >> h[len(h) - 1 - run] & 1:
            run += 1
        return run

    def value(self, h, vis):
        k = self.kind
        if k == "hex":
            if not self.rec:
                return self.pattern.match("".join(HEXDIGITS[c] for c in h))
            if self.end:
                return self.end_run(h)
            s = h if self.start else vis
            pre, best = self.prefix, None
            for o in ([0] if self.start else range(len(s) - len(pre) + 1)):
                if o + len(pre) <= len(s) and all(pre[i] >> s[o + i] & 1 for i in range(len(pre))):
                    j = o + len(pre)
                    while j < len(s) and self.tok >> s[j] & 1:
                        j += 1
                    run = j - o - len(pre)
                    best = run if best is None else max(best, run)
            return best
        if k == "count":
            return sum(1 for c in vis if self.cset >> c & 1)
        if k == "words":
            return count_words(vis, self.words)
        if k == "distinct":
            return len(set(vis))
        if k == "period":
            return longest_period(vis, self.period)
        if k in ("up", "down"):
            return longest_stair(vis, k == "up")
        return longest_palindrome(vis)

    def gpu_exact(self):
        if self.kind != "hex":
            return True
        if not self.rec:
            return False
        if not self.prefix:
            return True
        return self.start and all(self._nibble_exact(x) for x in self.prefix)

    @staticmethod
    def _nibble_exact(x):
        m, v = nib_mask(x)
        return x == sum(1 << d for d in range(16) if d & m == v)

    def gpu_hex_rec(self, h, vis):
        """The GPU value of an X{max} term: with ^ the prefix is tested by nibble masks and the run counted after it;
        without ^ the longest run of the token anywhere in the id (the prefix is left to the host); X{max}$ the run that
        ends the id."""
        if self.end:
            return self.end_run(h)
        if self.start:
            for i, x in enumerate(self.prefix):
                m, v = nib_mask(x)
                if h[i] & m != v:
                    return None
            j = len(self.prefix)
            run = 0
            while j + run < len(h) and self.tok >> h[j + run] & 1:
                run += 1
            return run
        best = run = 0
        for d in vis:
            run = run + 1 if self.tok >> d & 1 else 0
            best = max(best, run)
        return best

    def holds(self, v):
        if self.kind == "hex":
            return bool(v)
        return v is not None and v >= self.lo and (self.hi is None or v <= self.hi)

    def beats(self, v, best):
        if v is None:
            return False
        if best is None:
            return True
        return v > best if self.rec == "max" else v < best

    # -- odds --
    def prob_range(self, lo, hi):
        """P(lo <= value <= hi) for a random id (hi None = no upper bound)."""
        k, W = self.kind, self.width
        if k == "count":
            return _mix(W, lambda n, nz: _count_p(self.cset, n, nz, lo, hi))
        if k == "distinct":
            return _mix(W, lambda n, nz: _distinct_p(n, nz, lo, hi))
        if k == "words":
            lam = lambda n: sum(max(0, n - len(w) + 1) * 16.0 ** -len(w) for w in self.words)
            return _mix(W, lambda n, nz: _poisson_range(lam(n), lo, hi))
        if k == "period":
            p = self.period
            L = p * -(-lo // p)
            return _mix(W, lambda n, nz: _period_p(n, p, L))
        if k in ("up", "down"):
            return _mix(W, lambda n, nz: _stair_p(n, nz, lo, k == "up"))
        if k == "pal":
            return _mix(W, lambda n, nz: _pal_p(n, lo))
        raise ValueError(k)

    def probability(self, best=None):
        """Threshold term: P(it holds). Record term: P(it beats best) (1 without a best)."""
        if self.kind == "hex":
            if not self.rec:
                return self.pattern.probability()
            p = self.hex_k(0 if best is None else best + 1)
            return 0.0 if p is None else 1.0 if p == "always" else p.probability()
        if self.rec:
            if best is None:
                return 1.0
            return self.prob_range(best + 1, None) if self.rec == "max" else (self.prob_range(0, best - 1) if best > 0 else 0.0)
        return self.prob_range(self.lo, self.hi)

    def tag(self):
        k = self.kind
        if k == "count":
            s = self.cset
            name = HEXDIGITS[s.bit_length() - 1] if popcount(s) == 1 else {LETTERS: "l", DIGITS: "d", ALL: "all"}.get(s, f"{s:04x}")
            return f"count{name}"
        if k == "hex":
            return ("hexend-" if self.end else "hexstart-" if self.start else "hex-") + re.sub(r"[^0-9a-z]+", "", self.text.replace("{max}", "")) [:12]
        return {"words": "words", "distinct": "distinct", "up": "up", "down": "down", "pal": "pal"}.get(k, f"run{self.period}")


class StatLine:
    """A parsed stat line: terms joined by '&'; at most one record term ({max}/{min})."""

    def __init__(self, text, width, terms):
        self.text, self.width, self.terms = text, width, terms
        self.rec_term = next((t for t in terms if t.rec), None)

    def __repr__(self):
        return f"StatLine({self.text!r}, {self.width})"

    @property
    def record(self):
        return self.rec_term.rec if self.rec_term else None

    def finder_text(self):
        return "&".join(t.finder_text() for t in self.terms)

    def evaluate(self, hexid):
        """(ok, value): ok = every threshold term holds (and the record term has a value); value of the record term."""
        h = nibbles(hexid)
        vis = h[zeros_of(h):]
        val = None
        for t in self.terms:
            v = t.value(h, vis)
            if t.rec:
                if v is None:
                    return False, None
                val = v
            elif not t.holds(v):
                return False, None
        return True, val

    def match(self, hexid):
        return self.evaluate(hexid)[0]

    def beats(self, value, best):
        return self.rec_term.beats(value, best)

    def gpu_exact(self):
        """Do the GPUs compute the record value of this line exactly? (Then the best value can live on the device.)"""
        return all(t.gpu_exact() for t in self.terms)

    def gpu_pass(self, hexid, best=None):
        """The GPU stat test: every threshold term holds, the record term beats best (strictly). Hex terms of a line are
        only tested by their anchored nibble mask (none without ^ / $), so this is never False where evaluate() holds
        and the record is beaten."""
        h = nibbles(hexid)
        vis = h[zeros_of(h):]
        for t in self.terms:
            if t.kind == "hex" and not t.rec:
                a = t.pattern.anchored()
                if a and not all((h[i] & a[0][i]) == a[1][i] for i in range(len(h))):
                    return False
                continue
            v = t.gpu_hex_rec(h, vis) if t.kind == "hex" else t.value(h, vis)
            if t.rec:
                if v is None or (best is not None and not t.beats(v, best)):
                    return False
            elif not t.holds(v):
                return False
        return True

    def probability(self, best=None):
        """Chance that a random id satisfies the line (record line: beats best). Terms are taken as independent."""
        p = 1.0
        for t in self.terms:
            p *= t.probability(best) if t.rec else t.probability()
        return p

    def tag(self):
        base = "-".join(t.tag() for t in ([self.rec_term] if self.rec_term else self.terms[:1]))
        if len(self.terms) > 1:
            import hashlib
            base += "-" + hashlib.sha1(self.text.encode()).hexdigest()[:6]
        return base


def parse_range(s, i, P):
    """The range at s[i:] (must end the term) -> (lo, hi, rec, exact)."""
    if i >= len(s):
        raise PatternError(f"missing range after position {P(i - 1)}: add {RANGE_HELP}")
    if s[i] != "{":
        raise PatternError(f"'{s[i]}' at position {P(i)}: expected a range: {RANGE_HELP}")
    j = s.find("}", i)
    if j < 0:
        raise PatternError(f"unclosed '{{' at position {P(i)}")
    if j != len(s) - 1:
        raise PatternError(f"'{s[j + 1:]}' at position {P(j + 1)} after the range (join terms with '&')")
    body = s[i + 1:j]
    if body in ("max", "min"):
        return None, None, body, False
    m = re.fullmatch(r"(\d*)(,?)(\d*)", body)
    if not m or not (m.group(1) or m.group(3)) or (not m.group(2) and m.group(3)):
        raise PatternError(f"'{{{body}}}' at position {P(i)}: a range is {RANGE_HELP}")
    a = int(m.group(1)) if m.group(1) else 0
    if not m.group(2):
        return a, a, None, True
    b = int(m.group(3)) if m.group(3) else None
    if b is not None and a > b:
        raise PatternError(f"'{{{body}}}' at position {P(i)} goes backwards")
    return a, b, None, False


def read_words(path, base):
    full = os.path.join(base or os.getcwd(), path)
    try:
        lines = open(full, encoding="utf-8-sig").read().splitlines()
    except OSError as e:
        raise PatternError(f"cannot read word file {path}: {e.strerror}") from None
    words, seen = [], set()
    for n, line in enumerate(lines, 1):
        for w in line.split("#", 1)[0].replace(",", " ").split():
            w = w.lower()
            if not all(c in HEXDIGITS for c in w) or len(w) > MAX_WORD_LEN:
                raise PatternError(f"word file {path}:{n}: '{w}' is not a hex word of 1-{MAX_WORD_LEN} digits")
            if w not in seen:
                seen.add(w)
                words.append([HEXDIGITS.index(c) for c in w])
    if not words:
        raise PatternError(f"word file {path} has no words")
    if len(words) > MAX_WORDS:
        raise PatternError(f"word file {path} has {len(words)} words; at most {MAX_WORDS}")
    return words


def parse_term(raw, off, width, base):
    lead = len(raw) - len(raw.lstrip())
    sr, off = raw.strip(), off + lead
    pos = [off + i for i, c in enumerate(sr) if not c.isspace()]
    s = "".join(c for c in sr if not c.isspace()).lower()
    P = lambda i: pos[max(0, min(i, len(pos) - 1))] + 1
    always = f"'{s}' always holds"

    def check_count(t, maxv):
        if t.rec:
            return t
        if t.lo > maxv:
            raise PatternError(f"'{s}' can never hold: the id has {'at most ' if width == 64 else ''}{maxv} "
                               + ("different digits" if t.kind == "distinct" else "digits"))
        p = t.probability()
        if p >= 1 - 1e-12:
            raise PatternError(always)
        if p <= 0:
            raise PatternError(f"'{s}' can never hold")
        return t

    if s.startswith("#@"):
        at, j = sr.index("@"), sr.rfind("{")
        if j < at:
            raise PatternError(f"'#@FILE' at position {off + 1} needs a range: #@words.txt{{4,}}")
        path = sr[at + 1:j].strip()
        rs = "".join(sr[j:].split()).lower()
        lo, hi, rec, _ = parse_range(rs, 0, lambda i: off + j + 1)
        if rec == "min":
            raise PatternError(f"'{s}': {{min}} of a word count is almost always 0; use {{max}} or a range")
        if not rec and lo == 0 and hi is None:
            raise PatternError(always)
        words = read_words(path, base)
        return check_count(StatTerm("words", f"#@{path}{rs}", width, lo, hi, rec, words=words, wsrc=path), width)
    if s.startswith("##"):
        lo, hi, rec, _ = parse_range(s, 2, P)
        if rec == "max":
            raise PatternError(f"'{s}': nearly every id uses all 16 digits; use ##{{min}} or a range like ##{{,9}}")
        return check_count(StatTerm("distinct", s, width, lo, hi, rec), 16)
    if s.startswith("#"):
        i = 1
        if i >= len(s):
            raise PatternError(f"'#' at position {P(0)}: count what? e.g. #0{{24,}}, #?l{{,5}}, #[0-3]{{20,}}, #(dead|beef){{3,}}")
        c, words, cset = s[i], None, 0
        if c == "(":
            j = s.find(")", i)
            if j < 0:
                raise PatternError(f"unclosed '(' at position {P(i)}")
            words, seen = [], set()
            k = i + 1
            for w in s[i + 1:j].split("|"):
                if not w or not all(ch in HEXDIGITS for ch in w) or len(w) > MAX_WORD_LEN:
                    raise PatternError(f"'{w}' at position {P(k)}: words in #(...) are hex words of 1-{MAX_WORD_LEN} digits joined by '|'")
                if w not in seen:
                    seen.add(w)
                    words.append([HEXDIGITS.index(ch) for ch in w])
                k += len(w) + 1
            if len(words) > MAX_WORDS:
                raise PatternError(f"{len(words)} words; at most {MAX_WORDS}")
            i = j + 1
        elif c == "?":
            nx = s[i + 1:i + 2]
            cset, i = (DIGITS, i + 2) if nx == "d" else (LETTERS, i + 2) if nx == "l" else (ALL, i + 1)
        elif c == "[":
            j = s.find("]", i)
            if j < 0:
                raise PatternError(f"unbalanced '[' at position {P(i)}")
            cset, i = parse_class(s[i + 1:j], P(i) - 1), j + 1
        elif c in HEXDIGITS:
            cset, i = 1 << HEXDIGITS.index(c), i + 1
        else:
            raise PatternError(f"'{c}' at position {P(i)}: '#' counts a digit, ?, ?d, ?l, a class or (word|word)")
        if i < len(s) and s[i] != "{":
            raise PatternError(f"'{s[i]}' at position {P(i)}: '#' counts one digit, ?, ?d, ?l or class; for a word "
                               f"write #({s[1:i] + s[i:].split('{')[0]}){{...}}" if s[i] in HEXDIGITS else
                               f"'{s[i]}' at position {P(i)}: expected a range: {RANGE_HELP}")
        lo, hi, rec, _ = parse_range(s, i, P)
        if words is not None:
            if rec == "min":
                raise PatternError(f"'{s}': {{min}} of a word count is almost always 0; use {{max}} or a range")
            if not rec and lo == 0 and hi is None:
                raise PatternError(always)
            return check_count(StatTerm("words", s, width, lo, hi, rec, words=words), width)
        return check_count(StatTerm("count", s, width, lo, hi, rec, cset=cset), width)
    if s[0] in SHAPES or s[0] == "(":
        if s[0] == "(":
            m = re.fullmatch(r"\((\?+|\?\{(\d+)\})\)\\1(\{.*)", s)
            if not m:
                raise PatternError(f"'{s}' at position {P(0)}: only the periodic-run form is supported: (?)\\1{{n}} "
                                   f"(one digit n+1 times), (??)\\1{{n}} (a pair repeated), (?{{p}})\\1{{n}} (p = 1-8)")
            p = int(m.group(2)) if m.group(2) else len(m.group(1))
            if not 1 <= p <= MAX_PERIOD:
                raise PatternError(f"period {p} at position {P(1)}: the period is 1 to {MAX_PERIOD} digits")
            kind, ri, what = "period", m.start(3), "repeat count"
        else:
            kind, ri, p, what = SHAPES[s[0]], 1, 0, "length"
        lo, hi, rec, exact = parse_range(s, ri, P)
        if rec == "min":
            raise PatternError(f"'{s}': {{min}} is only for counts (#...) and ##")
        if not rec:
            if hi is not None and not exact:
                raise PatternError(f"'{s[ri:]}' at position {P(ri)}: a shape takes {{n}} (at least n) or {{max}}")
            n = lo
            if kind == "period":
                if n < 1:
                    raise PatternError(f"'{s[ri:]}' at position {P(ri)}: the repeat count must be at least 1")
                n = p * (n + 1)
            elif n < 2:
                raise PatternError(always)
            if n > width:
                raise PatternError(f"'{s}' needs {n} digits, the id has only {width}")
            lo, hi = n, None
        return StatTerm(kind, s, width, lo, hi, rec, period=p)
    # a hex pattern, maybe with X{max} at the end
    m = re.search(r"\{\d*,\d*\}", s)
    if m:
        raise PatternError(f"'{m.group()}' at position {P(m.start())}: ranges are for counts (#0{{24,}}) and shapes; "
                           f"in a hex pattern {{n}} repeats exactly")
    if "{min}" in s:
        raise PatternError(f"'{{min}}' at position {P(s.index('{min}'))}: {{min}} is only for counts (#...) and ##")
    if s.count("{max}") > 1:
        raise PatternError(f"two {{max}} in '{s}'; one per line")
    if "{max}" in s:
        i = s.index("{max}")
        end = s[i + 5:] == "$"
        if (s[i + 5:] and not end) or "*" in s or "$" in s[:i]:
            raise PatternError(f"'{{max}}' at position {P(i)}: in a hex pattern {{max}} repeats the last token of a "
                               f"pattern without '*': ^f{{max}}, 0{{max}}, ^dead0{{max}}, 0{{max}}$")
        try:
            pat = _parse(s[:i] + "{1}", width)
        except PatternError as e:
            raise PatternError(f"in '{s}': {e}") from None
        if end and (pat.start or len(pat.pieces[0]) != 1):
            raise PatternError(f"'{{max}}$' at position {P(i)}: a run at the end takes a single token: 0{{max}}$, ?l{{max}}$")
        return StatTerm("hex", s, width, rec="max", prefix=pat.pieces[0][:-1], tok=pat.pieces[0][-1], start=pat.start,
                        end=end)
    try:
        pat = _parse(s, width)
    except PatternError as e:
        raise PatternError(f"in '{s}': {e}") from None
    return StatTerm("hex", pat.text, width, pattern=pat)


def parse_stat(text, width, base=None):
    """A stat line -> StatLine. Terms are split on '&' outside [...] and (...)."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "[(":
            depth += 1
        elif c in "])":
            depth = max(0, depth - 1)
        elif c == "&" and depth == 0:
            parts.append((start, text[start:i]))
            start = i + 1
    parts.append((start, text[start:]))
    terms = []
    for off, raw in parts:
        if not raw.strip():
            raise PatternError(f"empty term at position {off + 1}: '&' needs an expression on both sides")
        terms.append(parse_term(raw, off, width, base))
    if len(terms) > MAX_TERMS:
        raise PatternError(f"{len(terms)} terms; at most {MAX_TERMS} per line")
    recs = [t.text for t in terms if t.rec]
    if len(recs) > 1:
        raise PatternError(f"{' and '.join(recs)}: at most one {{max}}/{{min}} per line")
    return StatLine("&".join(t.text for t in terms), width, terms)


# ---- inputs: repeat, comma lists, @file ---------------------------------------------------------------------
def split_commas(s):
    """Split on commas outside [...], {...} and (...)."""
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "[{(":
            depth += 1
        elif c in "]})":
            depth = max(0, depth - 1)
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    out.append(cur)
    return out


def split_comment(line):
    """(pattern part, comment text) of a pattern file line. '#' starts a comment, except where it starts a count term
    (#0{24,}, ##{,9}, #@words.txt): at the start of the line or after ',' or '&', followed by a digit, ?, [, (, @ or #."""
    i = 0
    while i < len(line):
        if line[i] == "#":
            prev, nxt = line[:i].rstrip(), line[i + 1:i + 2].lower()
            if (not prev or prev[-1] in ",&") and nxt and nxt in HEXDIGITS + "?[(@#":
                i += 2 if nxt == "#" else 1
                continue
            return line[:i], line[i + 1:].strip()
        i += 1
    return line, ""


def strip_comment(line):
    """A pattern file line without its comment (see split_comment)."""
    return split_comment(line)[0]


def pattern_items(values, cwd=None):
    """-m values -> [(label, text, base, note)]. A value is a pattern, a comma list, or @file (one pattern per line,
    comments after ' #', blank lines ignored, relative paths from cwd). base: the directory #@words.txt is relative
    to (the pattern file's directory, else cwd). note: the comment after the pattern on its file line ('' if none)."""
    out = []
    for v in values or []:
        if v.startswith("@"):
            path = os.path.join(cwd or os.getcwd(), v[1:])
            try:
                lines = open(path, encoding="utf-8-sig").read().splitlines()
            except OSError as e:
                raise PatternError(f"cannot read pattern file {v[1:]}: {e.strerror}") from None
            base = os.path.dirname(os.path.abspath(path))
            for n, line in enumerate(lines, 1):
                line, note = split_comment(line)
                line = line.strip()
                if line:
                    out += [(f"{v[1:]}:{n}", x.strip(), base, note) for x in split_commas(line) if x.strip()]
        else:
            parts = split_commas(v)
            if len(parts) > 1 and any(not x.strip() for x in parts):
                raise PatternError(f"empty pattern in the list '{v}'")
            out += [(None, x.strip(), cwd, "") for x in parts]
    return out


def pattern_lines(values, cwd=None):
    """-m values -> [(label, text)] (see pattern_items)."""
    return [(label, text) for label, text, _, _ in pattern_items(values, cwd)]


def parse_inputs(values, width, cwd=None):
    """-m values -> [Pattern] (duplicates dropped). All errors are collected into one PatternError."""
    pats, errors, seen = [], [], set()
    try:
        lines = pattern_items(values, cwd)
    except PatternError as e:
        raise PatternError(str(e)) from None
    for label, text, base, note in lines:
        try:
            p = parse_line(text, width, base)
        except PatternError as e:
            errors.append(f"{label}: {e}" if label else str(e))
            continue
        p.note = note
        if p.text not in seen:
            seen.add(p.text)
            pats.append(p)
    hexp = [p for p in pats if isinstance(p, Pattern)]
    stat = [p for p in pats if isinstance(p, StatLine)]
    if not errors and len(hexp) > MAX_PATTERNS:
        errors.append(f"{len(hexp)} patterns; at most {MAX_PATTERNS}")
    if not errors and len(stat) > MAX_STAT_LINES:
        errors.append(f"{len(stat)} stat lines; at most {MAX_STAT_LINES}")
    if not errors and sum(len(s.terms) for s in stat) > MAX_STAT_OPS:
        errors.append(f"{sum(len(s.terms) for s in stat)} stat terms in all lines; the GPU takes at most {MAX_STAT_OPS}")
    if not errors and sum(len(t.words) for s in stat for t in s.terms if t.kind == "words") > MAX_STAT_WORDS:
        errors.append(f"more than {MAX_STAT_WORDS} words in all word counts (#(...) / #@file); the GPU takes at most {MAX_STAT_WORDS}")
    if errors:
        raise PatternError("\n".join(errors))
    return pats


# ---- odds of any target, for the start-up summary -----------------------------------------------------------
def target_pattern(t, width):
    """The shorthand targets as patterns (for odds): -s X = X$, -c X = X, -z N = ^0{N}, NameMC -p X = ^X. A Mojang
    prefix matches the trimmed id, so it is ^X with 16/15 more chances (it may follow leading zeros)."""
    kind, val = t.split(":", 1)
    if kind in ("match", "record"):
        return parse_line(val, width)
    if kind == "zeros":
        return parse_line(f"^0{{{val}}}", width)
    if kind == "suffix":
        return parse_line(f"{val}$", width)
    if kind == "contains":
        return parse_line(val, width)
    return parse_line(f"^{val}", width)


def target_probability(t, width, best=None):
    """Chance that a random id hits the target (a record target: that it beats `best`)."""
    pat = target_pattern(t, width)
    p = pat.probability(best) if isinstance(pat, StatLine) and pat.record else pat.probability()
    if t.startswith("prefix:") and width == 64:
        p *= 16 / 15
    return p


def _exp(x):
    return re.sub(r"e\+?(-?)0*(\d)", r"e\1\2", f"{x:.1e}")


def fmt_odds(p):
    x = 1 / p if p > 0 else 0
    return "never" if p <= 0 else f"~1 in {x:.2g}" if x < 10 else f"~1 in {round(x)}" if x < 1e4 else f"~1 in {_exp(x)}"


def fmt_secs(s):
    if s < 0.1:
        return "< 0.1 s"
    if s < 90:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    if s < 3.2e7 * 3:
        return f"{s / 86400:.1f} days"
    return f"{_exp(s / 3.15e7)} years"


def fmt_eta(p, rate):
    """Expected time to the first hit of a target with probability p at rate candidates/s."""
    if p <= 0:
        return "never"
    s = fmt_secs(1 / p / rate)
    return s if s.startswith("<") else "~" + s


def fmt_rate(r):
    return f"{r / 1e9:.2f} G/s" if r >= 1e9 else f"{r / 1e6:.3g} M/s" if r >= 1e6 else f"{r / 1e3:.3g} k/s"
