"""A float64 as the shortest decimal text that reads back to the same bits (F-157): 0.1 is `0.1`, 0.30000000000000004 is itself.

Mojo's own `String(Float64)` is shortest for almost every value but not all (about 1 in 1,000 random doubles print one digit too few and read
back as the neighbour: 8.69563793934264e+17 for 8.695637939342639e+17), so the text is checked by reading it back; a text that does not
read back is rebuilt from the exact decimal expansion of the double, rounded to 16 and then 17 significant digits. The layout is Python's
`repr`: fixed notation for decimal exponents -4 .. 15 (`100.0`, `0.0001`), else `d.ddde+XX` (`1e+16`, `5e-324`). Finite values only
(the callers write inf and NaN their own way).
"""
from .csv import parse_float_text, F_OK


def _exact_digits(x: Float64) -> Tuple[List[UInt8], Int]:
    """The exact decimal digits (most significant first, no leading zeros) of |x| and the decimal exponent of the first digit: |x| =
    0.d1d2d3... * 10^(exp10 + 1)... precisely, x = d1.d2d3... * 10^exp10. |x| must be finite and nonzero."""
    var bits = UInt64(x.to_bits())
    var biased = Int((bits >> 52) & 0x7FF)
    var frac = bits & 0xFFFFFFFFFFFFF
    var m = frac if biased == 0 else frac | (UInt64(1) << 52)
    var e2 = (-1074 if biased == 0 else biased - 1075)
    # m * 2^e2 as decimal: for e2 >= 0 multiply by 2 e2 times; for e2 < 0 it is m * 5^-e2 / 10^-e2
    var dig = List[UInt8]()                     # little endian decimal digits of the integer part being built
    while m > 0:
        dig.append(UInt8(m % 10))
        m //= 10
    var times = e2 if e2 >= 0 else -e2
    var base = UInt64(2) if e2 >= 0 else UInt64(5)
    while times > 0:                            # in chunks: 2^16 or 5^8 at a time
        var k = min(times, 16 if e2 >= 0 else 8)
        var factor = UInt64(1)
        for _ in range(k):
            factor *= base
        var carry = UInt64(0)
        for i in range(len(dig)):
            var v = UInt64(dig[i]) * factor + carry
            dig[i] = UInt8(v % 10)
            carry = v // 10
        while carry > 0:
            dig.append(UInt8(carry % 10))
            carry //= 10
        times -= k
    var out = List[UInt8]()
    var i = len(dig) - 1
    while i >= 0:
        out.append(dig[i])
        i -= 1
    var point = len(out) - (0 if e2 >= 0 else -e2)        # digits before the decimal point
    return (out^, point - 1)


def _round_digits(digits: List[UInt8], n: Int) -> Tuple[List[UInt8], Int]:
    """The first n digits rounded half to even from all of them; the second value is 1 when rounding carried into a new leading digit."""
    if len(digits) <= n:
        var same = digits.copy()
        return (same^, 0)
    var out = List[UInt8]()
    for i in range(n):
        out.append(digits[i])
    var up = False
    var d = digits[n]
    if d > 5:
        up = True
    elif d == 5:
        var rest = False
        for i in range(n + 1, len(digits)):
            if digits[i] != 0:
                rest = True
                break
        up = rest or (out[n - 1] % 2 == 1)
    var carried = 0
    if up:
        var i = n - 1
        while i >= 0:
            if out[i] == 9:
                out[i] = 0
                i -= 1
            else:
                out[i] += 1
                break
        if i < 0:
            out.insert(0, 1)
            out.pop()
            carried = 1
    return (out^, carried)


def _layout(neg: Bool, digits: List[UInt8], exp10: Int) -> String:
    """Digits d1 d2 ... (trailing zeros dropped) and x = d1.d2... * 10^exp10 as text in Python's repr layout."""
    var n = len(digits)
    while n > 1 and digits[n - 1] == 0:
        n -= 1
    var s = String("-") if neg else String()
    if -4 <= exp10 < 16:
        if exp10 >= 0:
            for i in range(exp10 + 1):
                s += String(Int(digits[i])) if i < n else "0"
            s += "."
            if n > exp10 + 1:
                for i in range(exp10 + 1, n):
                    s += String(Int(digits[i]))
            else:
                s += "0"
        else:
            s += "0."
            for _ in range(-exp10 - 1):
                s += "0"
            for i in range(n):
                s += String(Int(digits[i]))
        return s^
    s += String(Int(digits[0]))
    if n > 1:
        s += "."
        for i in range(1, n):
            s += String(Int(digits[i]))
    s += "e" + ("-" if exp10 < 0 else "+")
    var a = -exp10 if exp10 < 0 else exp10
    s += ("0" if a < 10 else "") + String(a)
    return s^


def float_text(x: Float64) -> String:
    """The shortest text that reads back to x's bits (x finite)."""
    if x == 0.0:
        return "-0.0" if UInt64(x.to_bits()) >> 63 != 0 else "0.0"
    var text = String(x)
    var back = Float64(0.0)
    var sig = 0                                  # significant digits of Mojo's text
    var seen_nonzero = False
    var count = 0
    for ch in text.as_bytes():
        if ch == 101 or ch == 69:                # the exponent
            break
        if ch >= 48 and ch <= 57:
            if ch != 48:
                seen_nonzero = True
            if seen_nonzero:
                count += 1
                if ch != 48:
                    sig = count                  # (trailing zeros are not significant)
    # up to 15 digits the shortest text is unique and Mojo's is it; with 16 or 17 Mojo may pick a digit that is not the nearest on an exact tie
    # (or one too few that does not read back), so those are rebuilt from the exact expansion, rounding half to even as Python's repr does
    if sig <= 15 and parse_float_text(text, back) == F_OK and UInt64(back.to_bits()) == UInt64(x.to_bits()):
        return text^
    var neg = x < 0.0
    var ex = _exact_digits(-x if neg else x)
    for n in [16, 17]:
        var r = _round_digits(ex[0], n)
        var cand = _layout(neg, r[0], ex[1] + r[1])
        if parse_float_text(cand, back) == F_OK and UInt64(back.to_bits()) == UInt64(x.to_bits()):
            return cand^
    return _layout(neg, _round_digits(ex[0], 17)[0], ex[1])        # (unreachable: 17 digits always read back)
