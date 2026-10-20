# F-176: range aggregates and a block read over positions a..b of an order of a Parquet file, printed as typed JSON
# (tests/test_mojo_range.py compares them with DuckDB: rows a..b of an ORDER BY via row_number).
#
#   range_probe [--mode table|scan|frame|framescan] [--edits FILE] [--order COL[:desc]|none] [--out agg|block|both] [--repeat N]
#               --a A --b B --cols c1,c2,... path.parquet
#
# --mode   table: read the file whole and use the SharedTable calls; scan: a Frame over the streamed file; frame: a Frame over the table
#          after the edit script (the format of mojo/tools/sql_probe.mojo: set / append / remove / merge); framescan: the same over a file.
# --order  a column to sort by (`Frame.permutation`, stable, nulls last), or `none`: the rows themselves.
# Output: {"rows": b - a, "columns": [{"name","kind","scale","count","sum","mean","min","max"}], "block": [[cell, ...], ...], "ms": [...]}
# Cells: INT a number, DECIMAL / DATE / TEXT a string, BOOL true / false, FLOAT the float64's bits as "0x....", NULL null; `sum` of a FLOAT is its
# bits, `mean` is always the bits of a float64.
from std.sys import argv
from std.time import perf_counter_ns
from mxframe import (Frame, Table, SharedTable, read_parquet, scan_parquet, range_aggregate, read_block, RangeResult, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT)


def hex_digit(d: Int) -> String:
    var digits = "0123456789abcdef"
    return String(digits[byte=d:d + 1])


def esc_utf8(s: String) -> String:
    var out = String('"')
    var i = 0
    var b = s.as_bytes()
    var start = 0
    while i < len(b):
        var c = b[i]
        if c == UInt8(34) or c == UInt8(92) or c < UInt8(32):
            out += String(unsafe_from_utf8=b[start:i])
            if c == UInt8(34):
                out += '\\"'
            elif c == UInt8(92):
                out += "\\\\"
            else:
                out += "\\u00" + hex_digit(Int(c) // 16) + hex_digit(Int(c) % 16)
            start = i + 1
        i += 1
    out += String(unsafe_from_utf8=b[start:len(b)])
    return out + '"'


def pad(v: Int, w: Int) -> String:
    var s = String(v)
    while s.byte_length() < w:
        s = "0" + s
    return s


def civil(days: Int) -> String:
    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    var yy = y + (1 if m <= 2 else 0)
    return pad(yy, 4) + "-" + pad(m, 2) + "-" + pad(d, 2)


def decimal_text(v: Int64, scale: Int) -> String:
    var neg = v < 0
    var digits = String(UInt64(v) if not neg else UInt64(0) - UInt64(v))
    if scale == 0:
        return ("-" if neg else "") + digits
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var cut = digits.byte_length() - scale
    return ("-" if neg else "") + String(digits[byte=:cut]) + "." + String(digits[byte=cut:])


def hex64(v: Int64) -> String:
    var u = UInt64(v)
    var out = String()
    for k in range(16):
        out += hex_digit(Int((u >> UInt64(60 - 4 * k)) & 0xF))
    return "0x" + out


def split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    for part in s.split(sep):
        out.append(String(part))
    return out^


def apply_edits(mut f: Frame, path: String) raises:
    var text = String()
    with open(path, "r") as fh:
        text = fh.read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var p = split(String(line), "\t")
        if p[0] == "set":
            f.set_text(f.column(p[1]), Int(p[2]), p[3] if len(p) > 3 else "")
        elif p[0] == "append":
            var vals = List[String]()
            for i in range(1, len(p)):
                vals.append(p[i])
            _ = f.append(vals)
        elif p[0] == "remove":
            f.remove(Int(p[1]))
        elif p[0] == "merge":
            var c = f.column(p[1])
            var into = f.code_of(c, p[2])
            var codes = List[Int64]()
            for w in split(p[3], ","):
                var k = f.code_of(c, w)
                if k >= 0:
                    codes.append(Int64(k))
            _ = f.merge(c, codes, Int64(into))
        else:
            raise Error("edit script: unknown edit `" + p[0] + "`")


def cell(kind: Int, scale: Int, v: Int64, word: String) -> String:
    if kind == INT.value:
        return String(v)
    if kind == DECIMAL.value:
        return '"' + decimal_text(v, scale) + '"'
    if kind == TEXT.value:
        return esc_utf8(word)
    if kind == DATE.value:
        return '"' + civil(Int(v)) + '"'
    if kind == BOOL.value:
        return "true" if v != 0 else "false"
    return '"' + hex64(v) + '"'


def main() raises:
    var args = argv()
    var mode = String("table")
    var edits = String()
    var order_by = String("none")
    var out_what = String("both")
    var cols_arg = String()
    var a = 0
    var b = 0
    var repeat = 1
    var path = String()
    var i = 1
    while i < len(args):
        var x = String(args[i])
        if x == "--mode":
            mode = String(args[i + 1])
            i += 2
        elif x == "--edits":
            edits = String(args[i + 1])
            i += 2
        elif x == "--order":
            order_by = String(args[i + 1])
            i += 2
        elif x == "--out":
            out_what = String(args[i + 1])
            i += 2
        elif x == "--cols":
            cols_arg = String(args[i + 1])
            i += 2
        elif x == "--a":
            a = Int(String(args[i + 1]))
            i += 2
        elif x == "--b":
            b = Int(String(args[i + 1]))
            i += 2
        elif x == "--repeat":
            repeat = Int(String(args[i + 1]))
            i += 2
        else:
            path = x
            i += 1
    var in_table = mode == "table"
    var shared = Table.empty().share()
    var f = Frame(shared)
    if mode == "scan" or mode == "framescan":
        f = Frame(scan_parquet(path))
    else:
        var rp = read_parquet(path)
        shared = rp.take_table().share()
        f = Frame(shared)
    if edits.byte_length() > 0:
        apply_edits(f, edits)
    var cols = List[Int]()
    for name in split(cols_arg, ","):
        cols.append(f.column(name))
    var order = List[Int]()
    var identity = order_by == "none"
    if not identity:
        var parts = split(order_by, ":")
        order = f.permutation(f.column(parts[0]), len(parts) > 1 and parts[1] == "desc")
    try:
        var times = List[Float64]()
        var res = RangeResult(0)
        for _ in range(repeat):
            var t0 = perf_counter_ns()
            if in_table:
                res = range_aggregate(shared, order, cols, a, b) if not identity else range_aggregate(shared, cols, a, b)
            else:
                res = range_aggregate(f, order, cols, a, b) if not identity else range_aggregate(f, cols, a, b)
            times.append(Float64(perf_counter_ns() - t0) / 1e6)
        var out = String('{"rows": ') + String(res.rows) + ', "columns": ['
        if out_what != "block":
            for k in range(len(cols)):
                var c = cols[k]
                var g = res.columns[k].copy()
                var kind = g.kind.value
                var word = (word_of(in_table, shared, f, c, g.min), word_of(in_table, shared, f, c, g.max)) if kind == TEXT.value and g.count > 0 else (String(), String())
                out += ("" if k == 0 else ", ") + '{"name": ' + esc_utf8(f.name(c)) + ', "kind": "' + kind_name(kind) + '", "scale": ' + String(g.scale) + ', "count": ' + String(g.count)
                if g.has_sum:
                    out += ', "sum": ' + (('"' + hex64(g.sum) + '"') if kind == FLOAT.value else cell(kind, g.scale, g.sum, "")) + ', "mean": "' + hex64(Int64(g.mean.to_bits())) + '"'
                else:
                    out += ', "sum": null, "mean": null'
                if g.count > 0:
                    out += ', "min": ' + cell(kind, g.scale, g.min, word[0]) + ', "max": ' + cell(kind, g.scale, g.max, word[1]) + "}"
                else:
                    out += ', "min": null, "max": null}'
        out += '], "block": ['
        if out_what != "agg":
            var n = b - a
            var values = List[Int64](length=max(len(cols) * n, 1), fill=0)
            var nulls = List[UInt8](length=max(len(cols) * n, 1), fill=0)
            if in_table:
                read_block(shared, order, cols, a, b, values, nulls) if not identity else read_block(shared, cols, a, b, values, nulls)
            else:
                read_block(f, order, cols, a, b, values, nulls) if not identity else read_block(f, cols, a, b, values, nulls)
            for r in range(n):
                out += ("" if r == 0 else ", ") + "["
                for k in range(len(cols)):
                    out += "" if k == 0 else ", "
                    if nulls[k * n + r] != 0:
                        out += "null"
                        continue
                    var kind = f.kind(cols[k]).value
                    out += cell(kind, f.scale(cols[k]), values[k * n + r], word_of(in_table, shared, f, cols[k], values[k * n + r]) if kind == TEXT.value else "")
                out += "]"
        out += '], "ms": ['
        for k in range(len(times)):
            out += ("" if k == 0 else ", ") + String(times[k])
        print(out + "]}")
    except e:
        print('{"error": ' + esc_utf8(String(e)) + "}")
        raise Error("range failed")


def kind_name(k: Int) -> String:
    var names: List[String] = ["INT", "DECIMAL", "TEXT", "DATE", "BOOL", "FLOAT"]
    return names[k]


def word_of(in_table: Bool, shared: SharedTable, f: Frame, c: Int, code: Int64) raises -> String:
    return shared.text_of(c, Int(code)) if in_table else f.text_of(c, code)
