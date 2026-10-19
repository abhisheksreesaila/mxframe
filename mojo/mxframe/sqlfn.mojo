"""The scalar functions of SQL as plain functions on values (F-156): what a text function answers for one word, what a date function answers for one day.

No query machinery in here: the binder folds a function of constants with these, and the emitter (`sqlprog.mojo`) evaluates a text function once per word of a
dictionary and hands the answers to the kernel as a table. Everything follows DuckDB 1.5 (the oracle: `tests/test_mojo_exprs.py`), including its
quirks: `length` counts code points, `lower` and `upper` are utf8proc's simple mappings (`scripts/gen_unicase.py` asked DuckDB for them), `trim` strips
spaces only, `substr` clips its window to the string, a negative length reads backwards, `LIKE` has no escape character unless ESCAPE names one.
"""
from .csv import days_from_civil, _days_in_month, parse_float_text, F_OK, F_RANGE
from .table import TEXT, INT, DECIMAL, DATE, BOOL, FLOAT, F32_COLUMN
from .unicase import LOWER_RUNS, UPPER_RUNS

# ── UTF-8 ────────────────────────────────────────────────────────────────────────────────────────────

def ulen(b: UInt8) -> Int:
    """The length of the UTF-8 sequence that starts with byte b (1 for a stray continuation byte)."""
    if b < UInt8(0x80):
        return 1
    if b < UInt8(0xE0):
        return 2 if b >= UInt8(0xC0) else 1
    if b < UInt8(0xF0):
        return 3
    return 4


def decode(s: String) -> List[Int]:
    """The code points of s."""
    var out = List[Int]()
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        var c = Int(b[i])
        var k = ulen(b[i])
        if k == 1 or i + k > n:
            out.append(c)
            i += 1
            continue
        var cp = c & (0x1F if k == 2 else 0x0F if k == 3 else 0x07)
        for j in range(1, k):
            cp = (cp << 6) | (Int(b[i + j]) & 0x3F)
        out.append(cp)
        i += k
    return out^


def encode_into(cp: Int, mut out: List[UInt8]):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def encode(cps: List[Int], a: Int, b: Int) -> String:
    """The characters cps[a:b] as a string."""
    var out = List[UInt8]()
    for i in range(a, b):
        encode_into(cps[i], out)
    return String(unsafe_from_utf8=out)


def char_count(s: String) -> Int:
    """The number of code points (DuckDB's length())."""
    var n = 0
    for b in s.as_bytes():
        if (b & UInt8(0xC0)) != UInt8(0x80):
            n += 1
    return n


# ── case mapping ─────────────────────────────────────────────────────────────────────────────────────

struct CaseTables(Movable):
    """lower / upper: dense delta arrays below U+20000, built from the generated runs (`unicase.mojo`)."""
    var lower: List[Int32]
    var upper: List[Int32]

    def __init__(out self) raises:
        self.lower = _expand(LOWER_RUNS)
        self.upper = _expand(UPPER_RUNS)

    def map(self, s: String, up: Bool) -> String:
        var b = s.as_bytes()
        var ascii = True
        for x in b:
            if x >= UInt8(0x80):
                ascii = False
                break
        var out = List[UInt8]()
        if ascii:
            for x in b:
                if not up and UInt8(65) <= x <= UInt8(90):
                    out.append(x + 32)
                elif up and UInt8(97) <= x <= UInt8(122):
                    out.append(x - 32)
                else:
                    out.append(x)
            return String(unsafe_from_utf8=out)
        ref tab = self.upper if up else self.lower
        for cp in decode(s):
            var m = cp
            if cp < len(tab):
                m = cp + Int(tab[cp])
            encode_into(m, out)
        return String(unsafe_from_utf8=out)


def ascii_case(s: String, up: Bool) -> String:
    """lower / upper of a string that holds only ASCII."""
    var out = List[UInt8]()
    for x in s.as_bytes():
        if not up and UInt8(65) <= x <= UInt8(90):
            out.append(x + 32)
        elif up and UInt8(97) <= x <= UInt8(122):
            out.append(x - 32)
        else:
            out.append(x)
    return String(unsafe_from_utf8=out)


def _expand(spec: StaticString) raises -> List[Int32]:
    var tab = List[Int32](length=0x20000, fill=0)
    for run in spec.split(","):
        var f = run.split(":")
        var a = Int(f[0])
        var b = Int(f[1])
        var d = Int(f[2])
        var step = Int(f[3])
        var c = a
        while c <= b:
            tab[c] = Int32(d)
            c += step
    return tab^


# ── text functions ───────────────────────────────────────────────────────────────────────────────────

def substr(s: String, start: Int, length: Int, has_len: Bool) -> String:
    """DuckDB's substr: 1-based characters; a negative start counts from the end; the window [p, p + length - 1] (backwards for a negative length) is
    clipped to the string; no length reads to the end."""
    var cps = decode(s)
    var n = len(cps)
    var p = start if start > 0 else (n + start + 1 if start < 0 else 0)
    var lo = p
    var hi = n
    if has_len:
        if length > 0:
            hi = p + length - 1
        elif length < 0:
            lo = p + length
            hi = p - 1
        else:
            return String()
    if lo < 1:
        lo = 1
    if hi > n:
        hi = n
    if lo > hi:
        return String()
    return encode(cps, lo - 1, hi)


def left_of(s: String, k: Int) -> String:
    """left(s, k): the first k characters; a negative k drops the last -k."""
    var cps = decode(s)
    var n = len(cps)
    var m = k if k >= 0 else n + k
    if m <= 0:
        return String()
    return encode(cps, 0, min(m, n))


def right_of(s: String, k: Int) -> String:
    var cps = decode(s)
    var n = len(cps)
    var m = k if k >= 0 else n + k
    if m <= 0:
        return String()
    return encode(cps, n - min(m, n), n)


def reverse_of(s: String) -> String:
    var cps = decode(s)
    var out = List[UInt8]()
    for i in range(len(cps) - 1, -1, -1):
        encode_into(cps[i], out)
    return String(unsafe_from_utf8=out)


def trim_chars(s: String, chars: String, left: Bool, right: Bool) -> String:
    """trim / ltrim / rtrim with a set of characters (DuckDB's default is the space only)."""
    var cps = decode(s)
    var set = decode(chars)
    var a = 0
    var b = len(cps)
    if left:
        while a < b and _in(set, cps[a]):
            a += 1
    if right:
        while b > a and _in(set, cps[b - 1]):
            b -= 1
    return encode(cps, a, b)


def _in(set: List[Int], c: Int) -> Bool:
    for x in set:
        if x == c:
            return True
    return False


def strpos_of(s: String, sub: String) -> Int:
    """The 1-based character position of the first `sub` in s, 0 when it is not there (1 for the empty string)."""
    if sub.byte_length() == 0:
        return 1
    var i = s.find(sub)
    if i < 0:
        return 0
    return char_count(String(s[byte=0:i])) + 1


def replace_all(s: String, find: String, repl: String) -> String:
    if find.byte_length() == 0:
        return s
    return s.replace(find, repl)


struct LikePattern(Copyable, Movable):
    """A LIKE pattern as tokens: 0 any run (`%`), 1 any one character (`_`), 2 a literal character (its bytes in `lit`)."""
    var kind: List[Int]
    var lo: List[Int]
    var hi: List[Int]
    var lit: List[UInt8]

    def __init__(out self, pattern: String, escape: String, has_escape: Bool):
        self.kind = List[Int]()
        self.lo = List[Int]()
        self.hi = List[Int]()
        self.lit = List[UInt8]()
        var b = pattern.as_bytes()
        var e = -1
        if has_escape and escape.byte_length() > 0:
            e = decode(escape)[0]
        var n = len(b)
        var i = 0
        while i < n:
            var k = ulen(b[i])
            if i + k > n:
                k = n - i
            var cp = Int(b[i])
            if k > 1:
                cp = decode(String(unsafe_from_utf8=b[i:i + k]))[0]
            var literal = False
            if e >= 0 and cp == e and i + k < n:                  # the escape makes the next character literal
                i += k
                k = ulen(b[i])
                if i + k > n:
                    k = n - i
                literal = True
                cp = 0
            if not literal and b[i] == UInt8(37):
                self.kind.append(0)
                self.lo.append(0)
                self.hi.append(0)
            elif not literal and b[i] == UInt8(95):
                self.kind.append(1)
                self.lo.append(0)
                self.hi.append(0)
            else:
                self.kind.append(2)
                self.lo.append(len(self.lit))
                for j in range(k):
                    self.lit.append(b[i + j])
                self.hi.append(len(self.lit))
            i += k

    def matches(self, s: String) -> Bool:
        var b = s.as_bytes()
        var n = len(b)
        var m = len(self.kind)
        var si = 0
        var pi = 0
        var star_pi = -1
        var star_si = 0
        while si < n:
            if pi < m and self.kind[pi] == 1:
                si += min(ulen(b[si]), n - si)
                pi += 1
            elif pi < m and self.kind[pi] == 2 and self._lit_at(b, si, pi):
                si += self.hi[pi] - self.lo[pi]
                pi += 1
            elif pi < m and self.kind[pi] == 0:
                star_pi = pi
                star_si = si
                pi += 1
            elif star_pi >= 0:
                star_si += min(ulen(b[star_si]), n - star_si)
                si = star_si
                pi = star_pi + 1
            else:
                return False
        while pi < m and self.kind[pi] == 0:
            pi += 1
        return pi == m

    def _lit_at(self, b: Span[UInt8, _], si: Int, pi: Int) -> Bool:
        var k = self.hi[pi] - self.lo[pi]
        if si + k > len(b):
            return False
        for j in range(k):
            if b[si + j] != self.lit[self.lo[pi] + j]:
                return False
        return True


# ── dates ────────────────────────────────────────────────────────────────────────────────────────────

def civil_from_days(days: Int) -> Tuple[Int, Int, Int]:
    """(year, month, day) of a day number counted from 1970-01-01 (proleptic Gregorian; floor divisions, so dates before 1970 are right)."""
    var z = days + 719468
    var era = z // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    return (y + (1 if m <= 2 else 0), m, d)


def is_leap(y: Int) -> Bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def add_months(days: Int, months: Int) -> Int:
    """The date `months` after `days`; a day past the end of the target month clamps to its last day (Jan 31 + 1 month = Feb 28 / 29)."""
    var c = civil_from_days(days)
    var total = c[0] * 12 + (c[1] - 1) + months
    var y = total // 12
    var m = total - y * 12 + 1
    return days_from_civil(y, m, min(c[2], _days_in_month(y, m)))


comptime DP_YEAR = 1
comptime DP_MONTH = 2
comptime DP_DAY = 3
comptime DP_DOW = 4          # 0 = Sunday
comptime DP_ISODOW = 5       # 1 = Monday .. 7 = Sunday
comptime DP_DOY = 6
comptime DP_QUARTER = 7
comptime DP_EPOCH = 8        # seconds since 1970-01-01
comptime DP_DECADE = 9
comptime DP_WEEK = 10        # trunc / diff only: weeks start on Monday
comptime DP_CENTURY = 11
comptime DP_MILLENNIUM = 12


def date_part(days: Int, part: Int) -> Int:
    var c = civil_from_days(days)
    if part == DP_YEAR:
        return c[0]
    if part == DP_MONTH:
        return c[1]
    if part == DP_DAY:
        return c[2]
    if part == DP_DOW:
        return (days + 4) % 7
    if part == DP_ISODOW:
        return (days + 3) % 7 + 1
    if part == DP_DOY:
        return days - days_from_civil(c[0], 1, 1) + 1
    if part == DP_QUARTER:
        return (c[1] - 1) // 3 + 1
    if part == DP_EPOCH:
        return days * 86400
    if part == DP_DECADE:
        return c[0] // 10
    if part == DP_CENTURY:
        return (c[0] + 99) // 100 if c[0] > 0 else c[0] // 100
    return (c[0] + 999) // 1000 if c[0] > 0 else c[0] // 1000


def date_trunc_days(days: Int, part: Int) -> Int:
    var c = civil_from_days(days)
    if part == DP_YEAR:
        return days_from_civil(c[0], 1, 1)
    if part == DP_QUARTER:
        return days_from_civil(c[0], ((c[1] - 1) // 3) * 3 + 1, 1)
    if part == DP_MONTH:
        return days_from_civil(c[0], c[1], 1)
    if part == DP_WEEK:
        return days - (days + 3) % 7                              # back to Monday
    if part == DP_DECADE:
        return days_from_civil((c[0] // 10) * 10, 1, 1)
    if part == DP_CENTURY:                                        # DuckDB: the year divided by 100, floored (1948 is 1900, not 1901)
        return days_from_civil((c[0] // 100) * 100, 1, 1)
    if part == DP_MILLENNIUM:
        return days_from_civil((c[0] // 1000) * 1000, 1, 1)
    return days


def date_diff_days(a: Int, b: Int, part: Int) -> Int:
    """DuckDB's date_diff(part, a, b): the number of `part` boundaries crossed going from a to b (months, quarters, years, decades), whole weeks (the days
    over 7, truncated toward zero), or days."""
    if part == DP_DAY:
        return b - a
    if part == DP_WEEK:
        var d = b - a
        return d // 7 if d >= 0 else -((-d) // 7)
    var ca = civil_from_days(a)
    var cb = civil_from_days(b)
    if part == DP_YEAR:
        return cb[0] - ca[0]
    if part == DP_DECADE:
        return cb[0] // 10 - ca[0] // 10
    if part == DP_CENTURY:
        return cb[0] // 100 - ca[0] // 100
    if part == DP_MILLENNIUM:
        return cb[0] // 1000 - ca[0] // 1000
    var ma = ca[0] * 12 + ca[1] - 1
    var mb = cb[0] * 12 + cb[1] - 1
    if part == DP_QUARTER:
        return mb // 3 - ma // 3
    return mb - ma


def part_code(name: String) -> Int:
    """A date part's code from its name (lower case), -1 when there is no such part."""
    if name == "year" or name == "years" or name == "y":
        return DP_YEAR
    if name == "month" or name == "months" or name == "mon":
        return DP_MONTH
    if name == "day" or name == "days" or name == "d":
        return DP_DAY
    if name == "dow" or name == "dayofweek":
        return DP_DOW
    if name == "isodow":
        return DP_ISODOW
    if name == "doy" or name == "dayofyear":
        return DP_DOY
    if name == "quarter":
        return DP_QUARTER
    if name == "epoch":
        return DP_EPOCH
    if name == "decade":
        return DP_DECADE
    if name == "week" or name == "weeks":
        return DP_WEEK
    if name == "century":
        return DP_CENTURY
    if name == "millennium":
        return DP_MILLENNIUM
    return -1


# ── text of numbers and dates ────────────────────────────────────────────────────────────────────────

def pad_left(v: Int, w: Int) -> String:
    var s = String(v)
    while s.byte_length() < w:
        s = "0" + s
    return s


def decimal_text(v: Int64, scale: Int) -> String:
    """A stored decimal as DuckDB prints it: 12.50, -0.05, 5 (scale 0)."""
    var neg = v < 0
    var digits = String(UInt64(v) if not neg else UInt64(0) - UInt64(v))
    if scale == 0:
        return ("-" if neg else "") + digits
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var cut = digits.byte_length() - scale
    return ("-" if neg else "") + String(digits[byte=:cut]) + "." + String(digits[byte=cut:])


def date_text(days: Int) -> String:
    """YYYY-MM-DD, the year padded to four digits (years 1 to 9999)."""
    var c = civil_from_days(days)
    return pad_left(c[0], 4) + "-" + pad_left(c[1], 2) + "-" + pad_left(c[2], 2)


# ── a call of a text function, as data ───────────────────────────────────────────────────────────────

comptime TF_LOWER = 1
comptime TF_UPPER = 2
comptime TF_TRIM = 3          # flags: 1 the left end, 2 the right end; the character set is the first string (space when none)
comptime TF_SUBSTR = 4        # ints: start, [length]
comptime TF_LENGTH = 5
comptime TF_STARTS = 6        # the first string
comptime TF_ENDS = 7
comptime TF_CONTAINS = 8
comptime TF_LIKE = 9          # strs: pattern, escape; flags: 1 ILIKE (the pattern and the word are lower-cased first), 2 an escape is given
comptime TF_STRPOS = 10
comptime TF_LEFT = 11         # ints: count
comptime TF_RIGHT = 12
comptime TF_REVERSE = 13
comptime TF_REPLACE = 14      # strs: what to find, what to put


def tf_kind(func: Int) -> Int:
    """What the function answers: TEXT, INT or BOOL (a ColumnKind value)."""
    if func == TF_LENGTH or func == TF_STRPOS:
        return INT.value
    if func == TF_STARTS or func == TF_ENDS or func == TF_CONTAINS or func == TF_LIKE:
        return BOOL.value
    return TEXT.value


struct TFn(Copyable, Movable):
    """One text function with its constant arguments; `to_text`, `to_int` and `to_bool` answer for one word (LOWER and UPPER need `CaseTables`:
    the caller maps the word, and for ILIKE lower-cases it before `to_bool`)."""
    var func: Int
    var flags: Int
    var strs: List[String]
    var ints: List[Int]
    var like: LikePattern

    def __init__(out self, func: Int, flags: Int, strs: List[String], ints: List[Int]):
        self.func = func
        self.flags = flags
        self.strs = strs.copy()
        self.ints = ints.copy()
        self.like = LikePattern(strs[0] if func == TF_LIKE and len(strs) > 0 else String(), strs[1] if func == TF_LIKE and len(strs) > 1 else String(),
                                (flags & 2) != 0)

    def to_text(self, w: String) -> String:
        if self.func == TF_TRIM:
            return trim_chars(w, self.strs[0] if len(self.strs) > 0 else String(" "), (self.flags & 1) != 0, (self.flags & 2) != 0)
        if self.func == TF_SUBSTR:
            return substr(w, self.ints[0], self.ints[1] if len(self.ints) > 1 else 0, len(self.ints) > 1)
        if self.func == TF_LEFT:
            return left_of(w, self.ints[0])
        if self.func == TF_RIGHT:
            return right_of(w, self.ints[0])
        if self.func == TF_REVERSE:
            return reverse_of(w)
        if self.func == TF_REPLACE:
            return replace_all(w, self.strs[0], self.strs[1])
        return w

    def to_int(self, w: String) -> Int:
        if self.func == TF_STRPOS:
            return strpos_of(w, self.strs[0])
        return char_count(w)

    def to_bool(self, w: String) -> Bool:
        if self.func == TF_STARTS:
            return w.startswith(self.strs[0])
        if self.func == TF_ENDS:
            return w.endswith(self.strs[0])
        if self.func == TF_CONTAINS:
            return w.find(self.strs[0]) >= 0
        return self.like.matches(w)


# ── reading a word as a number, a date or a boolean (CAST from TEXT), once per word ────────────────────

struct Parsed(Copyable, Movable):
    """`ok`: the word reads as the type; `value`: its stored value (an integer, a scaled decimal, a day number, 0 / 1, a float's bits)."""
    var ok: Bool
    var value: Int64

    def __init__(out self, ok: Bool, value: Int64 = 0):
        self.ok = ok
        self.value = value


def _is_space(b: UInt8) -> Bool:
    return b == UInt8(32) or (b >= UInt8(9) and b <= UInt8(13))


struct _Num(Movable):
    var ok: Bool
    var neg: Bool
    var mant: UInt128          # the significant digits (leading zeros dropped; at most 38 are kept)
    var e10: Int               # the value is mant * 10^e10
    var huge: Bool             # an integer digit was dropped: the value is at least 10^38

    def __init__(out self):
        self.ok = False
        self.neg = False
        self.mant = UInt128(0)
        self.e10 = 0
        self.huge = False


def _scan_number(w: String) -> _Num:
    """`[ws] [+-] digits [. digits] [e [+-] digits] [ws]` (at least one digit before the exponent)."""
    var r = _Num()
    var b = w.as_bytes()
    var n = len(b)
    var i = 0
    while i < n and _is_space(b[i]):
        i += 1
    var end = n
    while end > i and _is_space(b[end - 1]):
        end -= 1
    if i >= end:
        return r^
    if b[i] == UInt8(45) or b[i] == UInt8(43):
        r.neg = b[i] == UInt8(45)
        i += 1
    var digits = 0
    var frac = 0
    var in_frac = False
    var kept = 0
    var limit = UInt128(10) ** UInt128(37)
    while i < end:
        var c = b[i]
        if c == UInt8(46):
            if in_frac:
                return r^
            in_frac = True
            i += 1
            continue
        if c < UInt8(48) or c > UInt8(57):
            break
        digits += 1
        var d = UInt128(Int(c) - 48)
        if kept == 0 and d == UInt128(0):                      # a leading zero
            if in_frac:
                frac += 1
        elif r.mant < limit:
            r.mant = r.mant * UInt128(10) + d
            kept += 1
            if in_frac:
                frac += 1
        else:
            if not in_frac:
                r.huge = True
                frac -= 1                                      # a dropped integer digit is a factor of ten
        i += 1
    if digits == 0:
        return r^
    var ex = 0
    if i < end:
        if (b[i] | UInt8(32)) != UInt8(101):
            return r^
        i += 1
        var eneg = False
        if i < end and (b[i] == UInt8(45) or b[i] == UInt8(43)):
            eneg = b[i] == UInt8(45)
            i += 1
        if i >= end:
            return r^
        var ed = 0
        while i < end:
            var d = Int(b[i]) - 48
            if d < 0 or d > 9:
                return r^
            if ed < 100000:
                ed = ed * 10 + d
            i += 1
        ex = -ed if eneg else ed
    r.e10 = ex - frac
    r.ok = True
    return r^


def _scaled(r: _Num, scale: Int, mut out: Int64) -> Bool:
    """The number times 10^scale as an int64, halves rounded away from zero; False when it does not fit."""
    if r.huge:
        return False
    var k = r.e10 + scale
    var m = r.mant
    if m == UInt128(0):
        out = 0
        return True
    if k >= 0:
        if k > 19:
            return False
        for _ in range(k):
            if m > UInt128(9223372036854775808):
                return False
            m = m * UInt128(10)
    else:
        if -k > 38:
            out = 0
            return True
        var p = UInt128(10) ** UInt128(-k)
        var q = m // p
        var rem = m - q * p
        if rem * UInt128(2) >= p:
            q += UInt128(1)
        m = q
    if m > (UInt128(9223372036854775808) if r.neg else UInt128(9223372036854775807)):
        return False
    out = Int64(0) - Int64(m) if r.neg and m != UInt128(9223372036854775808) else (Int64.MIN if r.neg else Int64(m))
    return True


def parse_text(w: String, kind: Int, scale: Int, bits: Int, prec: Int) -> Parsed:
    """The word as a value of `kind` (a ColumnKind value): INT (its width in `bits`), DECIMAL (`scale` places, at most `prec` digits), FLOAT (`scale` is
    F32_COLUMN for a 32-bit float), DATE or BOOL; not ok when it is not one."""
    if kind == INT.value or kind == DECIMAL.value:
        var r = _scan_number(w)
        if not r.ok:
            return Parsed(False)
        var v = Int64(0)
        if not _scaled(r, scale, v):
            return Parsed(False)
        if kind == INT.value and bits < 64:
            var lim = Int64(1) << Int64(bits - 1)
            if v < -lim or v >= lim:
                return Parsed(False)
        if kind == DECIMAL.value and prec < 18:
            var p = Int64(1)
            for _ in range(prec):
                p *= 10
            if v <= -p or v >= p:
                return Parsed(False)
        return Parsed(True, v)
    if kind == FLOAT.value:
        var b = w.as_bytes()
        var i = 0
        var end = len(b)
        while i < end and _is_space(b[i]):
            i += 1
        while end > i and _is_space(b[end - 1]):
            end -= 1
        if i >= end:
            return Parsed(False)
        var t = String(unsafe_from_utf8=b[i:end])
        var f = Float64(0.0)
        var st = parse_float_text(t, f)
        if st == F_RANGE:
            f = Float64(from_bits=UInt64(0xFFF0000000000000 if t.startswith("-") else 0x7FF0000000000000))
        elif st != F_OK:
            return Parsed(False)
        if scale == F32_COLUMN:
            f = Float64(Float32(f))
        return Parsed(True, Int64(f.to_bits()))
    if kind == DATE.value:
        return _parse_date_word(w)
    if kind == BOOL.value:
        var low = ascii_case(w, False)
        if low == "true" or low == "t" or low == "yes" or low == "y" or low == "1":
            return Parsed(True, 1)
        if low == "false" or low == "f" or low == "no" or low == "n" or low == "0":
            return Parsed(True, 0)
        return Parsed(False)
    return Parsed(False)


def _parse_date_word(w: String) -> Parsed:
    """`[ws] YYYY-M-D [ws]`: a year of 4 digits, a month and a day of 1 or 2."""
    var b = w.as_bytes()
    var i = 0
    var end = len(b)
    while i < end and _is_space(b[i]):
        i += 1
    while end > i and _is_space(b[end - 1]):
        end -= 1
    var parts = List[Int]()
    var digits = List[Int]()
    var cur = 0
    var nd = 0
    while i <= end:
        if i == end or b[i] == UInt8(45):
            if nd == 0:
                return Parsed(False)
            parts.append(cur)
            digits.append(nd)
            cur = 0
            nd = 0
        elif b[i] >= UInt8(48) and b[i] <= UInt8(57):
            cur = cur * 10 + Int(b[i]) - 48
            nd += 1
            if nd > 4:
                return Parsed(False)
        else:
            return Parsed(False)
        i += 1
    if len(parts) != 3 or digits[0] != 4 or digits[1] > 2 or digits[2] > 2:
        return Parsed(False)
    var y = parts[0]
    var m = parts[1]
    var d = parts[2]
    if y < 1 or m < 1 or m > 12 or d < 1 or d > _days_in_month(y, m):
        return Parsed(False)
    return Parsed(True, Int64(days_from_civil(y, m, d)))
