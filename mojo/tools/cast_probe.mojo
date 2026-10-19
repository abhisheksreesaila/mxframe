# F-156: reads words (one per line: an `x`, then the bytes as hex, so an empty word is a line too) and prints how CAST(word AS type) reads each one, for every text cast the engine has:
#
#   cast_probe words.txt
#
# One output line per word: the status and value for INTEGER, BIGINT, TINYINT, DECIMAL(12,3), DECIMAL(18,0), DECIMAL(5,2), DOUBLE, REAL, DATE, BOOLEAN, separated
# by `|`: `ok:<value>`, `bad` (the word is not one) or `unsure` (DuckDB's quirk that the engine does not read). tests/test_mojo_cast_fuzz.py compares it with DuckDB.
from std.sys import argv
from mxframe.sqlfn import parse_text
from mxframe import INT, DECIMAL, DATE, BOOL, FLOAT
from mxframe.table import F32_COLUMN


def hex_value(c: UInt8) -> Int:
    if c >= UInt8(97):
        return Int(c) - 97 + 10
    if c >= UInt8(65):
        return Int(c) - 65 + 10
    return Int(c) - 48


def decode_hex(h: String) -> String:
    var b = h.as_bytes()
    var out = List[UInt8]()
    var i = 0
    while i + 1 < len(b):
        out.append(UInt8(hex_value(b[i]) * 16 + hex_value(b[i + 1])))
        i += 2
    return String(unsafe_from_utf8=out)


def show(kind: Int, scale: Int, bits: Int, prec: Int, w: String) -> String:
    var r = parse_text(w, kind, scale, bits, prec)
    if r.unsure:
        return "unsure"
    if not r.ok:
        return "bad"
    return "ok:" + String(r.value)


def main() raises:
    var args = argv()
    var text = String()
    with open(String(args[1]), "r") as fh:
        text = fh.read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var w = decode_hex(String(line[byte=1:]))
        var parts = List[String]()
        parts.append(show(INT.value, 0, 32, 18, w))
        parts.append(show(INT.value, 0, 64, 18, w))
        parts.append(show(INT.value, 0, 8, 18, w))
        parts.append(show(DECIMAL.value, 3, 64, 12, w))
        parts.append(show(DECIMAL.value, 0, 64, 18, w))
        parts.append(show(DECIMAL.value, 2, 64, 5, w))
        parts.append(show(FLOAT.value, 0, 64, 18, w))
        parts.append(show(FLOAT.value, F32_COLUMN, 64, 18, w))
        parts.append(show(DATE.value, 0, 64, 18, w))
        parts.append(show(BOOL.value, 0, 64, 18, w))
        print("|".join(parts))
