# F-157: runs float64 query plans of the Query builder over a Parquet file with columns id, g (int, the group 0 .. 7), k, x, y, z (float64),
# d (decimal(12, 2)), i (int), f (a 32-bit float), s (text) and prints one JSON object (tests/test_mojo_float_builder.py compares it with DuckDB).
#
#   float_probe <path> <table|scan|frame|framescan> <plan> [--threads N] [--morsels on|off] [--block N] [--baked PATH --seed N]
#
# table: the file read whole; scan: streamed a row group at a time; frame / framescan: a Frame over either, after 300 deterministic edits
# (floats set from text and as numbers, nulls, appended and removed rows, two formula columns); --baked writes the edited table for the oracle.
# Output: {"counts": [...], "aggs": [{"float": 0|1, "present": [...], "vals": [...]}]}: a float aggregate's values are the float64 bits as
# "0x....", the others are numbers; top_k's lists are in "top".
from std.sys import argv
from mxframe import (Query, QueryResult, Frame, Table, SharedTable, col, lit, lit_float, lit_f32, sum, count, mean, min_of, max_of,
                     count_distinct, percentile, median, top_k, read_parquet, scan_parquet, write_parquet, ZSTD, AUTO, CPU,
                     MORSELS_ON, MORSELS_OFF)


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def int(mut self, lo: Int, hi: Int) -> Int:       # [lo, hi]
        return lo + Int(self.next() % UInt64(hi - lo + 1))


def hex_digit(d: Int) -> String:
    var digits = "0123456789abcdef"
    return String(digits[byte=d:d + 1])


def hex64(v: Int64) -> String:
    var u = UInt64(v)
    var out = String()
    for k in range(16):
        out += hex_digit(Int((u >> UInt64(60 - 4 * k)) & 0xF))
    return "0x" + out


def plan(var q: Query, which: String) raises -> Query:
    if which == "global":
        return q^.agg(count(), count(col("x")), sum(col("x")), mean(col("x")), min_of(col("x")), max_of(col("x")), sum(col("y")), mean(col("y")),
                      min_of(col("z")), max_of(col("z")), count(col("z")), sum(col("z")))
    if which == "grouped":
        return q^.group_by(col("g"), groups=8).agg(count(), count(col("x")), sum(col("x")), mean(col("x")), min_of(col("x")), max_of(col("x")),
                                                    sum(col("y")), mean(col("y")), min_of(col("z")), max_of(col("z")), count(col("z")), sum(col("z")))
    if which == "filtered":
        return q^.where((col("z") > lit_float(1e300)) | (col("x") < lit_float(-990.5))).group_by(col("g"), groups=8).agg(
            count(), sum(col("y")), max_of(col("z")), min_of(col("x")))
    if which == "arith":
        return q^.group_by(col("g"), groups=8).agg(
            sum(col("x") * 2.0 + col("y")), sum(col("d") * col("y")), sum(col("i") + col("y")), sum(col("x") / lit_float(4.0)),
            sum(-col("x")), max_of(col("x") - col("y")), sum(col("i") / col("y")), mean(col("i") * lit_float(0.5)), min_of(col("d") + col("x")),
            sum(col("i") * col("d")), max_of(col("y") / col("i")))
    if which == "holistic":
        return q^.group_by(col("g"), groups=8).agg(count(), count_distinct(col("z")), median(col("z")), percentile(col("z"), 0.25),
                                                    percentile(col("z"), 1.0), median(col("x")), percentile(col("x"), 0.9), top_k(col("x"), 3),
                                                    count_distinct(col("k")), percentile(col("k"), 0.5))
    if which == "holistic_global":
        return q^.agg(count_distinct(col("z")), median(col("z")), percentile(col("z"), 0.5), median(col("y")), top_k(col("y"), 2))
    if which == "f32":
        return q^.where(col("f") <= lit_f32(7.07)).agg(count(), sum(col("f")), max_of(col("f")))
    if which == "f64":
        return q^.where(col("f") <= lit_float(7.07)).agg(count())
    if which == "formula":
        return q^.group_by(col("g"), groups=8).agg(count(), sum(col("fx")), max_of(col("fx")), sum(col("fd")), min_of(col("fx")))
    if which == "key":
        return q^.group_by(col("z"), groups=4).agg(count())
    if which.startswith("cmp"):
        var e = col("z") == lit_float(0.0)
        if which == "cmp1":
            e = col("z") != lit_float(0.0)
        elif which == "cmp2":
            e = col("z") == col("z")
        elif which == "cmp3":
            e = col("z") > lit_float(1e300)
        elif which == "cmp4":
            e = col("z") >= lit_float(1e300)
        elif which == "cmp5":
            e = col("z") < lit_float(1e300)
        elif which == "cmp6":
            e = col("z") <= lit_float(-1e300)
        elif which == "cmp7":
            e = ~(col("z") > lit_float(1.0))
        elif which == "cmp8":
            e = col("z") > col("k")
        elif which == "cmp9":
            e = col("z") == col("k")
        elif which == "cmp10":
            e = (col("z") <= col("k")) | (col("x") > lit_float(500.0))
        elif which == "cmp11":
            e = col("z") > col("i")
        elif which == "cmp12":
            e = col("d") < col("y")
        return q^.where(e).agg(count())
    raise Error("unknown plan " + which)


def edit(mut f: Frame, seed: Int) raises:
    """300 deterministic edits: floats from text (exponents, nan, inf, -0.0, subnormals) and from numbers, nulls, appended and removed rows."""
    var rng = Rng(UInt64(seed))
    var g = f.column("g")
    var k = f.column("k")
    var x = f.column("x")
    var y = f.column("y")
    var z = f.column("z")
    var i = f.column("i")
    var texts: List[String] = ["1e3", "-2.5", "0", "-0.0", "nan", "inf", "-inf", "123456.789", "5e-324", "1.7976931348623157e308", "-7.25e-10", "42"]
    for step in range(300):
        var row = rng.int(0, f.rows() - 1)
        while not f.alive(row):
            row = rng.int(0, f.rows() - 1)
        var what = rng.int(0, 11)
        if what <= 2:
            f.set_text(z, row, texts[rng.int(0, len(texts) - 1)])
        elif what == 3:
            f.set_text(k, row, texts[rng.int(0, len(texts) - 1)])
        elif what == 4:
            f.set_float(x, row, Float64(rng.int(-90000, 90000)) / 100.0)
        elif what == 5:
            f.set_null(z, row)
        elif what == 6:
            f.set_null(y, row)
        elif what == 7:
            _ = f.append([String(rng.int(0, 7)), String(rng.int(0, 9)), "-0.0", String(rng.int(-500, 500)), String(rng.int(1, 99)), "nan", "12.50", String(rng.int(-9, 9)), "1.5", "west"])
        elif what == 8:
            f.remove(row)
        elif what == 9:
            f.set(i, row, Int64(rng.int(-300, 300)))
        elif what == 10:
            f.set_float(y, row, Float64(rng.int(1, 9999)) / 100.0)
        else:
            f.set_null(g, row)
    _ = f.formula("fx = x * 2.5 + y / 3")
    _ = f.formula("fd = d * y - i")


def json_of(r: QueryResult) -> String:
    var out = String('{"counts": [')
    for g in range(len(r.counts)):
        out += ("" if g == 0 else ", ") + String(r.counts[g])
    out += '], "aggs": ['
    for i in range(len(r.kinds)):
        var is_float = len(r.agg_float) > i and r.agg_float[i]
        if r.hol_of[i] >= 0:
            is_float = r.hfloat[r.hol_of[i]]
        out += ("" if i == 0 else ", ") + '{"float": ' + ("1" if is_float else "0") + ', "present": ['
        for g in range(len(r.counts)):
            out += ("" if g == 0 else ", ") + ("1" if r.present(i, g) else "0")
        if r.hol_of[i] >= 0 and len(r.htop_off[r.hol_of[i]]) > 0:
            out += '], "top": ['
            for g in range(len(r.counts)):
                var t = r.top(i, g)
                out += ("" if g == 0 else ", ") + "["
                for z in range(len(t)):
                    out += ("" if z == 0 else ", ") + ('"' + hex64(Int64(t[z])) + '"' if is_float else String(t[z]))
                out += "]"
            out += "]}"
        else:
            out += '], "vals": ['
            if is_float:
                var f = r.floats(i)
                for g in range(len(f)):
                    out += ("" if g == 0 else ", ") + '"' + hex64(Int64(f[g].to_bits())) + '"'
            else:
                var v = r.values(i)
                for g in range(len(v)):
                    out += ("" if g == 0 else ", ") + String(v[g])
            out += "]}"
    return out + "]}"


def main() raises:
    var args = argv()
    var path = String(args[1])
    var mode = String(args[2])
    var which = String(args[3])
    var threads = 0
    var morsels = -1
    var block = 0
    var baked = String()
    var seed = 1
    var a = 4
    while a < len(args):
        var o = String(args[a])
        if o == "--threads":
            threads = Int(String(args[a + 1]))
        elif o == "--morsels":
            morsels = MORSELS_ON if String(args[a + 1]) == "on" else MORSELS_OFF
        elif o == "--block":
            block = Int(String(args[a + 1]))
        elif o == "--baked":
            baked = String(args[a + 1])
        elif o == "--seed":
            seed = Int(String(args[a + 1]))
        a += 2
    try:
        var q: Query
        if mode == "table":
            var rp = read_parquet(path)
            q = Query(rp.take_table().share())
        elif mode == "scan":
            q = Query(scan_parquet(path))
        elif mode == "frame":
            var rp = read_parquet(path)
            var f = Frame(rp.take_table().share())
            edit(f, seed)
            if baked.byte_length() > 0:
                write_parquet(baked, f.bake(), "", ZSTD, 4096)
            q = Query(f)
        else:
            var f = Frame(scan_parquet(path))
            edit(f, seed)
            if baked.byte_length() > 0:
                f.save(baked)
            q = Query(f)
        if threads > 0:
            q = q^.threads(threads)
        if morsels >= 0:
            q = q^.morsels(morsels)
        if block > 0:
            q = q^.block_rows(block)
        var r = plan(q^, which).compute(AUTO)
        print(json_of(r))
    except e:
        print('{"error": "' + String(e).replace('"', "'") + '"}')
        raise Error("float_probe failed")
