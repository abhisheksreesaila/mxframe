# F-148: runs count_distinct / median / percentile / top_k queries over a Parquet file with columns g (int, the group),
# x (int), d (decimal(.., 2)) and s (string), either streamed (`scan`), after reading it whole (`read`), or over a Frame
# of it after a deterministic series of edits (`frame`; the baked table is written to a second path for the oracle),
# and prints one JSON line: the words of s (code order), the time, and per aggregate its values, presents and top lists.
# Usage: holistic_probe <path> <scan|read|frame> <all|global|filtered> <groups> [<baked path> <edit seed>]
# (tests/test_mojo_holistic.py compares it with DuckDB.)
from std.sys import argv
from omxcore import (Query, QueryResult, Frame, Table, SharedTable, col, lit, sum, count, count_distinct, percentile,
                     median, top_k, read_parquet, scan_parquet, write_parquet, ZSTD, AUTO)


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


def aggregates(var q: Query, grouped: Bool, groups: Int) raises -> Query:
    if not grouped:
        return q^.agg(count(), count_distinct(col("x")), median(col("x")), percentile(col("x"), 0.25), percentile(col("x"), 0.9),
                      top_k(col("x"), 3), count_distinct(col("s")), median(col("s")), top_k(col("s"), 2),
                      median(col("d")), percentile(col("d"), 0.75), count_distinct(col("d")), percentile(col("x"), 0.0),
                      percentile(col("x"), 1.0))
    return q^.group_by(col("g"), groups=groups).agg(count(), count_distinct(col("x")), median(col("x")),
                                                     percentile(col("x"), 0.25), percentile(col("x"), 0.9), top_k(col("x"), 3),
                                                     count_distinct(col("s")), median(col("s")), top_k(col("s"), 2),
                                                     median(col("d")), percentile(col("d"), 0.75), count_distinct(col("d")),
                                                     percentile(col("x"), 0.0), percentile(col("x"), 1.0))


def plan(var q: Query, which: String, groups: Int) raises -> Query:
    if which == "topk":                    # top_k alone on its values: the kernel hook feeds per-worker heaps, no pairs written
        return q^.group_by(col("g"), groups=groups).agg(count(), top_k(col("x"), 3), top_k(col("s"), 2))
    if which == "filtered":
        return aggregates(q^.where(col("x").between(-5000, 5000)), True, groups)
    return aggregates(q^, which != "global", groups)


def edit(mut f: Frame, seed: Int, groups: Int) raises:
    """A few hundred deterministic edits: values, nulls, new and old words, appended and removed rows, one merge."""
    var rng = Rng(UInt64(seed))
    var g = f.column("g")
    var x = f.column("x")
    var d = f.column("d")
    var s = f.column("s")
    var n = f.rows()
    var new_words: List[String] = ["zzz-new", "aaa-new", "mmm", "q"]
    for step in range(300):
        var row = rng.int(0, f.rows() - 1)
        while not f.alive(row):
            row = rng.int(0, f.rows() - 1)
        var what = rng.int(0, 11)
        if what <= 2:
            f.set(x, row, Int64(rng.int(-3000, 3000)))
        elif what == 3:
            f.set(g, row, Int64(rng.int(-1, groups)))                       # a group outside 0 .. groups-1 leaves the query
        elif what == 4:
            f.set_null(x, row)
        elif what == 5:
            f.set_null(s, row)
        elif what == 6:
            f.set_text(s, row, new_words[rng.int(0, 3)])
        elif what == 7:
            _ = f.append([String(rng.int(0, groups - 1)), String(rng.int(-4000, 4000)), "12.50", new_words[rng.int(0, 3)]])
        elif what == 8:
            f.remove(row)
        elif what == 9:
            f.set(d, row, Int64(rng.int(-9999, 9999)))
        elif what == 10:
            f.set_null(d, row)
        else:
            f.set_null(g, row)
    if f.word_count(s) > 3:
        _ = f.merge(s, [Int64(1), Int64(2)], Int64(0))                       # three words become one


def json_of(r: QueryResult, words: String, ms: Float64) -> String:
    var out = String('{"words": "') + words + '", "ms": ' + String(ms) + ', "counts": ['
    for g in range(len(r.counts)):
        out += ("" if g == 0 else ", ") + String(r.counts[g])
    out += '], "aggs": ['
    for i in range(1, len(r.kinds)):
        out += ("" if i == 1 else ", ") + '{"present": ['
        for g in range(len(r.counts)):
            out += ("" if g == 0 else ", ") + ("1" if r.present(i, g) else "0")
        if r.hol_of[i] >= 0 and len(r.htop_off[r.hol_of[i]]) > 0:             # the top_k aggregates
            out += '], "top": ['
            for g in range(len(r.counts)):
                var t = r.top(i, g)
                out += ("" if g == 0 else ", ") + "["
                for z in range(len(t)):
                    out += ("" if z == 0 else ", ") + String(t[z])
                out += "]"
            out += "]}"
        else:
            out += '], "vals": ['
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
    var groups = Int(String(args[4]))
    var words = String("")
    if mode == "scan":
        var scan = scan_parquet(path)
        var c = scan.index("s")
        for k in range(scan.word_count(c)):
            words += scan.text_of(c, k) + "|"
        var r = plan(Query(scan), which, groups).compute(AUTO)
        print(json_of(r, words, r.milliseconds))
        return
    var rp = read_parquet(path)
    var t = rp.take_table().share()
    var c = t.index("s")
    if mode == "read":
        for k in range(len(t.dictionary(c))):
            words += t.text_of(c, k) + "|"
        var r = plan(Query(t), which, groups).compute(AUTO)
        print(json_of(r, words, r.milliseconds))
        return
    var f = Frame(t)
    edit(f, Int(String(args[6])), groups)
    for k in range(f.word_count(c)):
        words += f.text_of(c, Int64(k)) + "|"
    var r = plan(Query(f), which, groups).compute(AUTO)
    write_parquet(String(args[5]), f.bake(), "", ZSTD, 4096)
    print(json_of(r, words, r.milliseconds))
