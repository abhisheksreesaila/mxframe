"""Expressions over the groups of an aggregate result (F-150): HAVING, ORDER BY keys and select items.

After aggregation a query has one value per group for each group key and each aggregate (`ValCol`s). The expressions
that combine them (`sum(x) / count(*)`, `max(a) - min(a)`, `count(*) > 5 AND region = 'West'`) are evaluated here one
node at a time over the whole column of groups: each operator is a tight loop over a vector, the way the kernel's
registers work, with SQL's rules: three-valued logic, NULL in, NULL out, exact int64 (an overflow raises).

A value is an Int64 per group (a FLOAT holds a float64's bits) and a flag per group for NULL; a boolean is 0 / 1 with
the same flag for unknown.
"""
from std.math import fma
from .table import ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .frame import _text_less
from .bind import (BoundQuery, BExpr, B_COL, B_LIT, B_STR, B_NULL, B_ADD, B_SUB, B_MUL, B_IDIV, B_IMOD, B_NEG, B_DIV, B_RESCALE,
                   B_CMP, B_AND, B_OR, B_NOT, B_ISNULL, B_IN, B_AGG, B_KEY, C_LT, C_LE, C_GT, C_GE, C_EQ, C_NE, K_NULL, pow10_checked,
                   kind_name)
from .sqlprog import DictAccess

comptime OVERFLOW_TEXT = "a sum or an expression passed the int64 range (about 9.2e18)"


struct ValCol(Copyable, Movable):
    var kind: Int
    var scale: Int
    var src: Int                     # a TEXT value: the source column whose dictionary its codes belong to
    var v: List[Int64]
    var nul: List[UInt8]             # 1 where NULL; empty when none is

    def __init__(out self, kind: Int, scale: Int, n: Int):
        self.kind = kind
        self.scale = scale
        self.src = -1
        self.v = List[Int64](length=n, fill=0)
        self.nul = List[UInt8]()

    def n(self) -> Int:
        return len(self.v)

    def is_null(self, i: Int) -> Bool:
        return len(self.nul) > 0 and self.nul[i] != 0

    def set_null(mut self, i: Int):
        if len(self.nul) == 0:
            self.nul = List[UInt8](length=len(self.v), fill=0)
        self.nul[i] = 1

    def has_nulls(self) -> Bool:
        for x in self.nul:
            if x != 0:
                return True
        return False

    def f(self, i: Int) -> Float64:
        """The value as a float64 (a DECIMAL divided by its scale, as DuckDB casts it)."""
        if self.kind == FLOAT.value:
            return Float64(from_bits=UInt64(self.v[i]))
        if self.scale > 0:
            return Float64(self.v[i]) / Float64(Int(_p10(self.scale)))
        return Float64(self.v[i])


def _p10(k: Int) -> Int64:
    var v: Int64 = 1
    for _ in range(k):
        v *= 10
    return v


def float_bits(x: Float64) -> Int64:
    return Int64(x.to_bits())


def _combine_nulls(mut out: ValCol, a: ValCol, b: ValCol):
    if len(a.nul) == 0 and len(b.nul) == 0:
        return
    out.nul = List[UInt8](length=out.n(), fill=0)
    for i in range(out.n()):
        out.nul[i] = 1 if (a.is_null(i) or b.is_null(i)) else 0


def _const_col(kind: Int, scale: Int, value: Int64, n: Int) -> ValCol:
    var c = ValCol(kind, scale, n)
    for i in range(n):
        c.v[i] = value
    return c^


def eval_post(q: BoundQuery, e: Int, keys: List[ValCol], aggs: List[ValCol], dicts: DictAccess, n: Int) raises -> ValCol:
    """The post-aggregate expression `e` of `q` over `n` groups whose key values are `keys` and aggregate values `aggs`."""
    var x = q.pexprs[e].copy()
    var op = x.op
    if op == B_KEY:
        return keys[Int(x.value)].copy()
    if op == B_AGG:
        return aggs[Int(x.value)].copy()
    if op == B_LIT:
        return _const_col(x.kind, x.scale, x.value, n)
    if op == B_NULL:
        var c = ValCol(x.kind if x.kind != K_NULL else INT.value, x.scale, n)
        c.nul = List[UInt8](length=n, fill=1)
        return c^
    if op == B_RESCALE:
        var a = eval_post(q, x.a, keys, aggs, dicts, n)
        var m = _p10(Int(x.value))
        var out = ValCol(DECIMAL.value, x.scale, n)
        for i in range(n):
            var r = a.v[i] * m
            if a.v[i] != 0 and r // a.v[i] != m:
                if not a.is_null(i):
                    raise Error(OVERFLOW_TEXT)
            out.v[i] = r
        out.nul = a.nul.copy()
        return out^
    if op == B_NEG:
        var a = eval_post(q, x.a, keys, aggs, dicts, n)
        var out = ValCol(a.kind, a.scale, n)
        for i in range(n):
            if a.kind == FLOAT.value:
                out.v[i] = float_bits(-a.f(i))
            else:
                if a.v[i] == Int64.MIN and not a.is_null(i):
                    raise Error(OVERFLOW_TEXT)
                out.v[i] = -a.v[i]
        out.nul = a.nul.copy()
        return out^
    if op == B_ADD or op == B_SUB or op == B_MUL or op == B_DIV or op == B_IDIV or op == B_IMOD:
        var a = eval_post(q, x.a, keys, aggs, dicts, n)
        var b = eval_post(q, x.b, keys, aggs, dicts, n)
        var out = ValCol(x.kind, x.scale, n)
        _combine_nulls(out, a, b)
        if x.kind == FLOAT.value:
            for i in range(n):
                if out.is_null(i):
                    continue
                var p = a.f(i)
                var r: Float64
                if op == B_DIV:
                    if b.f(i) == 0.0:
                        out.set_null(i)
                        continue
                    r = p / b.f(i)
                elif op == B_ADD:
                    r = p + b.f(i)
                elif op == B_SUB:
                    r = p - b.f(i)
                else:
                    r = p * b.f(i)
                out.v[i] = float_bits(r)
            return out^
        for i in range(n):
            if out.is_null(i):
                continue
            var p = a.v[i]
            var r = b.v[i]
            var y: Int64
            if op == B_ADD:
                y = p + r
                if ((p ^ y) & (r ^ y)) < 0:
                    raise Error(OVERFLOW_TEXT)
            elif op == B_SUB:
                y = p - r
                if ((p ^ r) & (p ^ y)) < 0:
                    raise Error(OVERFLOW_TEXT)
            elif op == B_MUL:
                y = p * r
                if (p == -1 and r == Int64.MIN) or (r == -1 and p == Int64.MIN) or (p != 0 and y // p != r):
                    raise Error(OVERFLOW_TEXT)
            else:                                                 # // and %: truncating, x / 0 is NULL
                if r == 0:
                    out.set_null(i)
                    continue
                if r == -1 and p == Int64.MIN:
                    raise Error(OVERFLOW_TEXT)
                var qt = p // r
                if p % r != 0 and ((p < 0) != (r < 0)):
                    qt += 1
                y = qt if op == B_IDIV else p - qt * r
            out.v[i] = y
        return out^
    if op == B_CMP:
        return _compare(q, x, keys, aggs, dicts, n)
    if op == B_AND or op == B_OR:
        var a = eval_post(q, x.a, keys, aggs, dicts, n)
        var b = eval_post(q, x.b, keys, aggs, dicts, n)
        var out = ValCol(BOOL.value, 0, n)
        for i in range(n):
            var an = a.is_null(i)
            var bn = b.is_null(i)
            var at = (not an) and a.v[i] != 0
            var bt = (not bn) and b.v[i] != 0
            var af = (not an) and a.v[i] == 0
            var bf = (not bn) and b.v[i] == 0
            if op == B_AND:
                if af or bf:
                    out.v[i] = 0
                elif an or bn:
                    out.set_null(i)
                else:
                    out.v[i] = 1
            else:
                if at or bt:
                    out.v[i] = 1
                elif an or bn:
                    out.set_null(i)
                else:
                    out.v[i] = 0
        return out^
    if op == B_NOT:
        var a = eval_post(q, x.a, keys, aggs, dicts, n)
        var out = ValCol(BOOL.value, 0, n)
        for i in range(n):
            out.v[i] = 1 if a.v[i] == 0 else 0
        out.nul = a.nul.copy()
        return out^
    if op == B_ISNULL:
        var a = eval_post(q, x.a, keys, aggs, dicts, n)
        var out = ValCol(BOOL.value, 0, n)
        for i in range(n):
            out.v[i] = 1 if a.is_null(i) else 0
        return out^
    if op == B_IN:
        var a = eval_post(q, x.a, keys, aggs, dicts, n)
        var out = ValCol(BOOL.value, 0, n)
        var vals = List[Int64]()
        for it in x.items:
            var l = q.pexprs[it].copy()
            if l.op == B_STR:
                var code = dicts.code_of(q.pexprs[x.a].src if a.src < 0 else a.src, l.text)
                if code >= 0:
                    vals.append(Int64(code))
            else:
                vals.append(l.value)
        for i in range(n):
            if a.is_null(i):
                out.set_null(i)
                continue
            var hit = False
            if a.kind == FLOAT.value:
                raise Error("IN on a FLOAT value is not supported")
            for v in vals:
                if v == a.v[i]:
                    hit = True
                    break
            if hit:
                out.v[i] = 1
            elif x.value != 0:
                out.set_null(i)
        return out^
    raise Error("internal: expression kind " + String(op) + " cannot be evaluated after aggregation")


def _compare(q: BoundQuery, x: BExpr, keys: List[ValCol], aggs: List[ValCol], dicts: DictAccess, n: Int) raises -> ValCol:
    var code = Int(x.value)
    var a = eval_post(q, x.a, keys, aggs, dicts, n)
    var out = ValCol(BOOL.value, 0, n)
    if a.kind == TEXT.value:
        var lit = q.pexprs[x.b].text
        var src = a.src
        for i in range(n):
            if a.is_null(i):
                out.set_null(i)
                continue
            var c = Int(a.v[i])
            var w = dicts.word(src, c)
            var hit: Bool
            if code == C_EQ:
                hit = w == lit
            elif code == C_NE:
                hit = w != lit
            elif code == C_LT:
                hit = _text_less(w, lit)
            elif code == C_GT:
                hit = _text_less(lit, w)
            elif code == C_LE:
                hit = not _text_less(lit, w)
            else:
                hit = not _text_less(w, lit)
            out.v[i] = 1 if hit else 0
        return out^
    var b = eval_post(q, x.b, keys, aggs, dicts, n)
    _combine_nulls(out, a, b)
    var as_float = a.kind == FLOAT.value or b.kind == FLOAT.value
    for i in range(n):
        if out.is_null(i):
            continue
        var r: Bool
        if as_float:
            var p = a.f(i)
            var s = b.f(i)
            r = (p == s) if code == C_EQ else (p != s) if code == C_NE else (p < s) if code == C_LT else \
                (p <= s) if code == C_LE else (p > s) if code == C_GT else (p >= s)
        else:
            var p = a.v[i]
            var s = b.v[i]
            r = (p == s) if code == C_EQ else (p != s) if code == C_NE else (p < s) if code == C_LT else \
                (p <= s) if code == C_LE else (p > s) if code == C_GT else (p >= s)
        out.v[i] = 1 if r else 0
    return out^



def ld_div(num: Int64, den: UInt128) -> Float64:
    """num / den as an x87 `long double` division rounded to a double, which is how DuckDB computes AVG: the quotient is
    rounded to a 64-bit mantissa, then to 53 bits. That is not always the correctly rounded quotient (about 1 average
    in 20,000 differs by one ulp), and the oracle is DuckDB's, so this reproduces its rounding exactly.

    The two roundings differ only when the 64-bit quotient lands exactly on a midpoint of the 53-bit grid, so unless the
    exact quotient lies within 1/512 ulp of one, the plain double division (correctly rounded) is the answer: found
    from the exact residual (an FMA). Otherwise, and for operands past 2^53, the quotient is computed bit by bit."""
    if num == 0:
        return 0.0
    var neg = num < 0
    var n = UInt64(0) - UInt64(num) if neg else UInt64(num)
    if n < (UInt64(1) << 53) and den < (UInt128(1) << 53):
        var nf = Float64(n)
        var df = Float64(UInt64(den))
        var q0 = nf / df
        var r = fma(-q0, df, nf)
        if r == 0.0:
            return -q0 if neg else q0
        var q0_bits = UInt64(q0.to_bits())
        var neighbor = Float64(from_bits=q0_bits + 1 if r > 0.0 else q0_bits - 1)
        var spacing = neighbor - q0 if r > 0.0 else q0 - neighbor
        var frac = (r if r > 0.0 else -r) / df
        var gap = frac - spacing * 0.5
        if (gap if gap > 0.0 else -gap) > spacing * (1.0 / 512.0):
            return -q0 if neg else q0
    var bn = _bitlen64(n)
    var bd = _bitlen128(den)
    var s = 67 - bn + bd                                        # (n << s) / den has 67 or 68 bits
    var total = bn + s
    var rem = UInt128(0)
    var q = UInt128(0)
    for i in range(total):
        var idx = total - 1 - i
        var bit = UInt128(0)
        if idx >= s:
            bit = UInt128((n >> UInt64(idx - s)) & 1)
        rem = (rem << UInt128(1)) | bit
        q = q << UInt128(1)
        if rem >= den:
            rem -= den
            q |= UInt128(1)
    var bq = _bitlen128(q)
    var drop = bq - 64
    var low = q & ((UInt128(1) << UInt128(drop)) - UInt128(1))
    var m64 = q >> UInt128(drop)
    var half = UInt128(1) << UInt128(drop - 1)
    if low > half or (low == half and (rem != UInt128(0) or (m64 & UInt128(1)) != UInt128(0))):
        m64 += UInt128(1)
    var exp = drop - s                                          # the value is m64 * 2^exp
    var low11 = m64 & UInt128(0x7FF)
    var m53 = m64 >> UInt128(11)
    if low11 > UInt128(0x400) or (low11 == UInt128(0x400) and (m53 & UInt128(1)) != UInt128(0)):
        m53 += UInt128(1)
    exp += 11
    var out = Float64(UInt64(m53)) * Float64(from_bits=UInt64((1023 + exp) << 52))
    return -out if neg else out


def _bitlen64(x: UInt64) -> Int:
    var n = 0
    var v = x
    while v != 0:
        n += 1
        v >>= 1
    return n


def _bitlen128(x: UInt128) -> Int:
    var n = 0
    var v = x
    while v != UInt128(0):
        n += 1
        v >>= UInt128(1)
    return n
