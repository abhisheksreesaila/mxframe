# F-151: tables as text. json_string and csv_quote always produce valid UTF-8 (a byte that is not UTF-8 becomes U+FFFD, a backstop:
# the CSV and Parquet readers refuse such bytes), and a CSV cell for the empty string is two quotes (a NULL is an empty cell).
from omxcore.output import json_string, csv_quote, decimal_text, civil_date
from omxcore.csv import first_invalid_utf8, BP


def check_eq(got: String, want: String, what: String) raises:
    if got != want:
        raise Error(what + "\n   got:    " + got + "\n   wanted: " + want)


def bytes_text(b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=b)


def main() raises:
    check_eq(json_string("plain"), '"plain"', "plain")
    check_eq(json_string('a"b\\c\n\t\r' + chr(1)), '"a\\"b\\\\c\\n\\t\\r\\u0001"', "escapes")
    check_eq(json_string("café €"), '"café €"', "valid UTF-8 is kept")
    # "caf" + 0xE9 (Latin-1 e acute), a lone 0xFF, a cut-off 3-byte sequence, an overlong 0xC0 0x80 and a surrogate 0xED 0xA0 0x80
    var bad = List[UInt8]()
    for c in String("caf").as_bytes():
        bad.append(c)
    bad.append(UInt8(0xE9))
    bad.append(UInt8(0xFF))
    bad.append(UInt8(0xE2))
    bad.append(UInt8(0x82))
    var got = json_string(bytes_text(bad))
    check_eq(got, '"caf����"', "each bad byte becomes U+FFFD")
    var over = List[UInt8]()
    over.append(UInt8(0xC0))
    over.append(UInt8(0x80))
    over.append(UInt8(0xED))
    over.append(UInt8(0xA0))
    over.append(UInt8(0x80))
    check_eq(json_string(bytes_text(over)), '"�����"', "overlong and surrogate forms")
    check_eq(csv_quote(""), '""', "the empty string")
    check_eq(csv_quote("a b"), "a b", "a space needs no quotes")
    check_eq(csv_quote("a,b"), '"a,b"', "comma")
    check_eq(csv_quote('say "hi"'), '"say ""hi"""', "quote")
    check_eq(csv_quote("two\nlines"), '"two\nlines"', "newline")
    check_eq(csv_quote(bytes_text(bad)), "caf����", "bad bytes in CSV")
    check_eq(decimal_text(-5, 2), "-0.05", "decimal")
    check_eq(civil_date(0), "1970-01-01", "date")
    print("ok test_output")
