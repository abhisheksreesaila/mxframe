# Test tool (tests/test_mojo_csv.py): reads a CSV with mxframe's reader and prints the table as JSON lines, so an
# independent reader (pyarrow) can be compared with it. First line: [[name, kind, scale], ...]; then one array per
# row: INT / DECIMAL / DATE / BOOL as their stored integers, TEXT as strings, null as null.
# Usage: csv_dump <path> [threads] [chunk_bytes] [infer_rows] [float_columns] [decimal_columns], comma separated
# (a FLOAT cell is printed as its float64 bits, a signed integer)
from std.sys import argv
from omxcore import read_csv, CsvOptions, TEXT


def json_string(s: String) -> String:
    var out = List[UInt8]()
    out.append(34)
    for b in s.as_bytes():
        if b == 34 or b == 92:
            out.append(92)
            out.append(b)
        elif b == 10:
            out.append(92)
            out.append(110)
        elif b == 13:
            out.append(92)
            out.append(114)
        elif b == 9:
            out.append(92)
            out.append(116)
        elif b < 32:
            out.extend(String("\\u00").as_bytes())
            out.append(48 + (b >> 4))
            out.append(48 + (b & 15) if (b & 15) < 10 else 87 + (b & 15))
        else:
            out.append(b)
    out.append(34)
    return String(unsafe_from_utf8=Span(out))


def main() raises:
    var args = argv()
    var options = CsvOptions()
    if len(args) > 2:
        options.threads = Int(String(args[2]))
    if len(args) > 3:
        options.chunk_bytes = Int(String(args[3]))
    if len(args) > 4:
        options.infer_rows = Int(String(args[4]))
    if len(args) > 5:
        for name in String(args[5]).split(","):
            options.float_columns.append(String(name))
    if len(args) > 6:
        for name in String(args[6]).split(","):
            options.decimal_columns.append(String(name))
    var r = read_csv(String(args[1]), options)
    var head = String("[")
    for c in range(r.table.columns()):
        head += ("," if c > 0 else "") + "[" + json_string(r.table.name(c)) + ",\"" + String(r.table.kind(c)) + "\"," + String(r.table.scale(c)) + "]"
    print(head + "]")
    for row in range(r.rows):
        var line = String("[")
        for c in range(r.table.columns()):
            if c > 0:
                line += ","
            if r.table.is_null(c, row):
                line += "null"
            elif r.table.kind(c) == TEXT:
                line += json_string(r.table.text_of(c, Int(r.table.get(c, row))))
            else:
                line += String(r.table.get(c, row))
        print(line + "]")
