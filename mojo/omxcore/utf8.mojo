"""UTF-8 well-formedness, one rule for the CSV reader (refuses what is not UTF-8) and the JSON writer (escapes it as a backstop)."""


def utf8_seq_len(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8, avail: Int) -> Int:
    """The length (1 to 4) of the well-formed UTF-8 sequence that starts with `b0`, given the next bytes and how many bytes are
    left (`avail`, counting `b0`), or 0 when it is not well-formed: a stray continuation byte, an overlong form, a surrogate, a code
    point past U+10FFFF, or a sequence cut off by the end."""
    if b0 < 0x80:
        return 1
    var need = 0
    var lo = UInt8(0x80)
    var hi = UInt8(0xBF)
    if b0 >= 0xC2 and b0 <= 0xDF:
        need = 1
    elif b0 == 0xE0:
        need = 2
        lo = 0xA0
    elif b0 == 0xED:
        need = 2
        hi = 0x9F
    elif b0 >= 0xE1 and b0 <= 0xEF:
        need = 2
    elif b0 == 0xF0:
        need = 3
        lo = 0x90
    elif b0 == 0xF4:
        need = 3
        hi = 0x8F
    elif b0 >= 0xF1 and b0 <= 0xF3:
        need = 3
    else:
        return 0
    if avail < need + 1 or b1 < lo or b1 > hi:
        return 0
    if need >= 2 and (b2 & 0xC0) != 0x80:
        return 0
    if need == 3 and (b3 & 0xC0) != 0x80:
        return 0
    return need + 1
