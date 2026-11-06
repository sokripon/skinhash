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

Lines that look like the reserved stat syntax (zeros >= 24, palindrome best, run(f) >= 10, ^cafe & zeros >= 10)
raise PatternError("... not supported yet").
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


def is_reserved(toks):
    """The future stat syntax: an operator (>=, <=, (, &) or a word followed by 'best'."""
    kinds = [t for t in toks if t[0] != "space"]
    if any(t[0] == "op" for t in kinds):
        return True
    return any(t[0] == "word" and t[1] == "best" for t in kinds[1:])


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


def parse_line(line, width):
    """One pattern line -> Pattern. Raises PatternError with a message that names the pattern."""
    text = line.strip()
    try:
        return _parse(text, width)
    except PatternError as e:
        raise PatternError(f"pattern '{text}': {e}") from None


def _parse(text, width):
    if not text:
        raise PatternError("empty pattern")
    toks = tokenize(text)
    if is_reserved(toks):
        raise PatternError("stat targets (zeros >= N, ... best, &) are not supported yet")
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


# ---- inputs: repeat, comma lists, @file ---------------------------------------------------------------------
def split_commas(s):
    """Split on commas outside [...]."""
    out, depth, cur = [], 0, ""
    for c in s:
        if c == "[":
            depth += 1
        elif c == "]":
            depth = max(0, depth - 1)
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    out.append(cur)
    return out


def pattern_lines(values, cwd=None):
    """-m values -> [(label, text)]. A value is a pattern, a comma list, or @file (one pattern per line, # comments,
    blank lines ignored, relative paths from cwd)."""
    out = []
    for v in values or []:
        if v.startswith("@"):
            path = os.path.join(cwd or os.getcwd(), v[1:])
            try:
                lines = open(path, encoding="utf-8-sig").read().splitlines()
            except OSError as e:
                raise PatternError(f"cannot read pattern file {v[1:]}: {e.strerror}") from None
            for n, line in enumerate(lines, 1):
                line = line.split("#", 1)[0].strip()
                if line:
                    out += [(f"{v[1:]}:{n}", x.strip()) for x in split_commas(line) if x.strip()]
        else:
            parts = split_commas(v)
            if len(parts) > 1 and any(not x.strip() for x in parts):
                raise PatternError(f"empty pattern in the list '{v}'")
            out += [(None, x.strip()) for x in parts]
    return out


def parse_inputs(values, width, cwd=None):
    """-m values -> [Pattern] (duplicates dropped). All errors are collected into one PatternError."""
    pats, errors, seen = [], [], set()
    try:
        lines = pattern_lines(values, cwd)
    except PatternError as e:
        raise PatternError(str(e)) from None
    for label, text in lines:
        try:
            p = parse_line(text, width)
        except PatternError as e:
            errors.append(f"{label}: {e}" if label else str(e))
            continue
        if p.text not in seen:
            seen.add(p.text)
            pats.append(p)
    if not errors and len(pats) > MAX_PATTERNS:
        errors.append(f"{len(pats)} patterns; at most {MAX_PATTERNS}")
    if errors:
        raise PatternError("\n".join(errors))
    return pats


# ---- odds of any target, for the start-up summary -----------------------------------------------------------
def target_pattern(t, width):
    """The shorthand targets as patterns (for odds): -s X = X$, -c X = X, -z N = ^0{N}, NameMC -p X = ^X. A Mojang
    prefix matches the trimmed id, so it is ^X with 16/15 more chances (it may follow leading zeros)."""
    kind, val = t.split(":", 1)
    if kind == "match":
        return parse_line(val, width)
    if kind == "zeros":
        return parse_line(f"^0{{{val}}}", width)
    if kind == "suffix":
        return parse_line(f"{val}$", width)
    if kind == "contains":
        return parse_line(val, width)
    return parse_line(f"^{val}", width)


def target_probability(t, width):
    p = target_pattern(t, width).probability()
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
