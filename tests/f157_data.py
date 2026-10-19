"""F-157: float data with every awkward value, shared by the FLOAT tests (tests/test_mojo_float*.py)."""
import math
import random
import struct

NEG_NAN = struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000123))[0]       # a negative NaN with a payload (inf - inf is one)
POS_NAN = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
PAYLOAD_NAN = struct.unpack("<d", struct.pack("<Q", 0x7FF800000000BEEF))[0]
SPECIALS = [0.0, -0.0, 1.0, -1.0, 0.5, 1e-310, -5e-324, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
            -1.7976931348623157e308, math.inf, -math.inf, POS_NAN, NEG_NAN, PAYLOAD_NAN, 0.1, 0.2, 0.3, 1e16, 1e-16, 123456789.123456789]


def bits(x):
    return struct.unpack("<q", struct.pack("<d", x))[0]


def from_bits(b):
    return struct.unpack("<d", struct.pack("<q", b))[0]


def random_doubles(n, seed, null_rate=0.1, special_rate=0.15, spread=1e6):
    """n doubles (or None): mostly smooth random values, some from SPECIALS, some NULL."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        u = r.random()
        if u < null_rate:
            out.append(None)
        elif u < null_rate + special_rate:
            out.append(r.choice(SPECIALS))
        else:
            out.append(r.uniform(-spread, spread) * 10 ** r.randint(-3, 3))
    return out
