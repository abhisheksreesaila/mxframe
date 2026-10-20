# F-141: dumps every cell of a Parquet file, read through read_parquet (mode read) or through scan_parquet's read_group (mode scan),
# TEXT as the word (or its code with a fourth argument `codes`), null as N, one tab-separated line per row (tests/test_mojo_parquet.py compares it with pyarrow).
from std.sys import argv
from omxcore import read_parquet, scan_parquet, TEXT


def main() raises:
    var args = argv()
    var path = String(args[1])
    var mode = String(args[2])
    var codes = len(args) > 3 and String(args[3]) == "codes"          # TEXT as its code, not its word
    var out = String("")
    if mode == "read":
        var r = read_parquet(path)
        ref t = r.table
        for row in range(t.rows()):
            for c in range(t.columns()):
                if c > 0:
                    out += "\t"
                if t.is_null(c, row):
                    out += "N"
                elif t.kind(c) == TEXT and not codes:
                    out += t.text_of(c, Int(t.get(c, row)))
                else:
                    out += String(t.get(c, row))
            out += "\n"
    else:
        var s = scan_parquet(path)
        var cols = List[Int]()
        for c in range(s.columns()):
            cols.append(c)
        for g in range(s.row_groups()):
            var t = s.read_group(g, cols)
            for row in range(t.rows()):
                for c in range(t.columns()):
                    if c > 0:
                        out += "\t"
                    if t.is_null(c, row):
                        out += "N"
                    elif t.kind(c) == TEXT and not codes:
                        out += s.text_of(c, Int(t.get(c, row)))
                    else:
                        out += String(t.get(c, row))
                out += "\n"
    print(out, end="")
