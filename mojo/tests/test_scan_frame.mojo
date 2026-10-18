# F-147: a Frame over a streamed Parquet file (`Frame(scan_parquet(path))`) behaves as `Frame(read_parquet(path))`
# with the same edits: queries (plain and holistic), get / is_null / edited, permutation and the saved file, after
# every batch of random edits (sets, nulls, typed text with new words, appends, removes, merges, formulas) and after
# undoing them. The oracle is the in-memory Frame, whose own correctness F-136 checks against `bake()`.
from std.os import getenv, setenv, makedirs
from mxframe import (Table, SharedTable, Frame, Query, QueryResult, col, sum, count, mean, min_of, max_of,
                     count_distinct, median, percentile, top_k, INT, DECIMAL, TEXT, DATE, BOOL, write_parquet,
                     read_parquet, scan_parquet, ZSTD, SNAPPY, UNCOMPRESSED)


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


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


comptime N = 20_000
comptime GROUP = 1_600           # 12.5 row groups: the last holds 800 rows


def scratch(name: String) raises -> String:
    """Under ~/.cache, never /tmp (a shared tmpfs with a quota)."""
    var dir = getenv("HOME") + "/.cache/mxframe-test-tmp"
    makedirs(dir, exist_ok=True)
    return dir + "/f147_" + name


def make_table(n: Int) raises -> SharedTable:
    """region TEXT (later row groups bring new words), note TEXT with nulls, carrier INT with nulls, revenue
    DECIMAL(2) with nulls (all null for row group 3), cost DECIMAL(2), day DATE, paid BOOL with nulls."""
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var nt = t.add_column("note", TEXT)
    var c = t.add_column("carrier", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", DECIMAL, scale=2)
    var d = t.add_column("day", DATE)
    var p = t.add_column("paid", BOOL)
    var words: List[String] = ["West", "east", "North", "South", "Central", "Nörth-East", "Far", "west"]
    var rng = Rng(7)
    for i in range(n):
        var rg = i // GROUP
        t.append_text(g, words[(rg + rng.int(0, 2)) % 8])
        if i % 5 == 0:
            t.append_null(nt)
        else:
            t.append_text(nt, "n" + String(rng.int(0, 3 + rg)))
        if i % 29 == 3:
            t.append_null(c)
        else:
            t.append(c, Int64(rng.int(-1, 7)))
        if rg == 3 or i % 13 == 1:
            t.append_null(r)
        else:
            t.append(r, Int64(rng.int(-1_000_00, 2_000_00)))
        t.append(k, Int64(rng.int(0, 900_00)))
        t.append(d, Int64(18_000 + rng.int(0, 400)))
        if i % 11 == 6:
            t.append_null(p)
        else:
            t.append(p, Int64(rng.int(0, 1)))
    return t^.share()


def same(a: QueryResult, b: QueryResult, label: String) raises:
    expect(len(a.counts) == len(b.counts), label + ": groups")
    expect(len(a.kinds) == len(b.kinds), label + ": aggregates")
    for g in range(len(a.counts)):
        expect(a.counts[g] == b.counts[g], label + ": count group " + String(g) + " " + String(a.counts[g]) + " vs " + String(b.counts[g]))
        for i in range(len(a.kinds)):
            expect(a.present(i, g) == b.present(i, g), label + ": present " + String(i) + " group " + String(g))
            if not a.present(i, g):
                continue
            if a.kinds[i] == 5:
                expect(abs(a.means(i)[g] - b.means(i)[g]) < 1e-9, label + ": mean " + String(i))
            else:
                expect(a.values(i)[g] == b.values(i)[g], label + ": agg " + String(i) + " group " + String(g) + " " +
                       String(a.values(i)[g]) + " vs " + String(b.values(i)[g]))
            var ta = a.top(i, g)
            var tb = b.top(i, g)
            expect(len(ta) == len(tb), label + ": top " + String(i) + " group " + String(g) + " length")
            for x in range(len(ta)):
                expect(ta[x] == tb[x], label + ": top " + String(i) + " group " + String(g))


def run_queries(f: Frame, formulas: Bool) raises -> List[QueryResult]:
    var c = f.column("region")
    var n = f.word_count(c)
    var out = List[QueryResult]()
    out.append(Query(f).group_by(col("region"), groups=n).agg(sum(col("revenue")), count(), count(col("revenue")),
                                                              min_of(col("revenue")), max_of(col("cost")), mean(col("cost"))).compute())
    out.append(Query(f).where(col("carrier").between(1, 4)).where(col("day") > 18_100)
               .group_by(col("region") * 8 + col("carrier"), groups=n * 8).agg(sum(col("cost")), mean(col("revenue"))).compute())
    out.append(Query(f).where(col("region") == 0).agg(sum(col("revenue") - col("cost")), count()).compute())
    out.append(Query(f).group_by(col("note"), groups=f.word_count(f.column("note"))).agg(count(), sum(col("cost"))).compute())
    out.append(Query(f).group_by(col("region"), groups=n).agg(count_distinct(col("carrier")), median(col("revenue")),
                                                              top_k(col("cost"), 3), percentile(col("cost"), 0.9),
                                                              median(col("note")), sum(col("cost"))).compute())
    if formulas:
        out.append(Query(f).where(col("half") > 0).group_by(col("region"), groups=n).agg(sum(col("margin")), count(col("margin")),
                                                                                         max_of(col("half"))).compute())
    return out^


def same_cell(a: Frame, b: Frame, c: Int, row: Int, label: String) raises:
    var at = label + " col " + String(c) + " row " + String(row)
    expect(a.alive(row) == b.alive(row), at + ": alive")
    expect(a.is_null(c, row) == b.is_null(c, row), at + ": null")
    expect(a.get(c, row) == b.get(c, row), at + ": " + String(a.get(c, row)) + " vs " + String(b.get(c, row)))
    if c < a.stored():
        expect(a.edited(c, row) == b.edited(c, row), at + ": edited")


def same_order(a: List[Int], b: List[Int], label: String) raises:
    expect(len(a) == len(b), label + ": length " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        expect(a[i] == b[i], label + ": position " + String(i))


def same_file(path: String, want: Table, label: String) raises:
    """The saved file read back equals the baked table, value by value (text by word: codes are first-seen)."""
    var rp = read_parquet(path)
    var got = rp.take_table().share()
    expect(got.rows() == want.rows(), label + ": saved " + String(got.rows()) + " rows, wanted " + String(want.rows()))
    expect(got.columns() == want.columns(), label + ": saved columns")
    for c in range(want.columns()):
        expect(got.name(c) == want.name(c) and got.kind(c) == want.kind(c) and got.scale(c) == want.scale(c), label + ": schema of " + want.name(c))
        for row in range(want.rows()):
            expect(got.is_null(c, row) == want.is_null(c, row), label + ": null at " + want.name(c) + " row " + String(row))
            if not want.is_null(c, row):
                if want.kind(c) == TEXT:
                    expect(got.text_of(c, Int(got.get(c, row))) == want.text_of(c, Int(want.get(c, row))),
                           label + ": word at " + want.name(c) + " row " + String(row))
                else:
                    expect(got.get(c, row) == want.get(c, row), label + ": value at " + want.name(c) + " row " + String(row))


def check(a: Frame, b: Frame, rng: Rng, label: String, formulas: Bool, saved: String) raises:
    """`a` is Frame(read_parquet), `b` Frame(scan_parquet): both ways of seeing the same file and the same edits."""
    var rng2 = rng.s
    expect(a.rows() == b.rows(), label + ": rows")
    expect(a.edits == b.edits, label + ": edits")
    expect(a.columns() == b.columns(), label + ": columns")
    for c in range(a.stored()):
        if a.kind(c) == TEXT:
            expect(a.word_count(c) == b.word_count(c), label + ": word count of " + a.name(c))
    var qa = run_queries(a, formulas)
    var qb = run_queries(b, formulas)
    for i in range(len(qa)):
        same(qb[i], qa[i], label + ": query " + String(i))
    var r = Rng(rng2)
    for _ in range(150):                                     # random cells, every column, formulas too
        var row = r.int(0, a.rows() - 1)
        for c in range(a.columns()):
            same_cell(a, b, c, row, label)
    for c in range(a.columns()):                             # every column's sort order, both ways
        same_order(a.permutation(c, False), b.permutation(c, False), label + ": order of " + a.name(c))
        same_order(a.permutation(c, True), b.permutation(c, True), label + ": descending order of " + a.name(c))
    b.save(saved, "{\"k\": 1}")
    same_file(saved, a.bake(), label + ": save")


def test_random_edits_undo_and_save_equal_the_in_memory_frame() raises:
    var table = make_table(N)
    var path = scratch("edit.parquet")
    write_parquet(path, table, "", ZSTD, GROUP)
    var rp = read_parquet(path)
    var a = Frame(rp.take_table().share())
    var scan = scan_parquet(path)
    var b = Frame(scan)
    b.set_cache_limit(3 * 8 * GROUP)                         # a handful of column chunks: the cache evicts all the time
    var saved = scratch("edit.saved.parquet")
    var rng = Rng(11)
    check(a, b, rng, "untouched", False, saved)
    var rc = a.column("region")
    var nc = a.column("note")
    var rv = a.column("revenue")
    var cs = a.column("cost")
    var ca = a.column("carrier")
    var dy = a.column("day")
    var pd = a.column("paid")
    var steps = 0
    for round in range(4):
        for _ in range(120):
            steps += 1
            var row = rng.int(0, a.rows() - 1)
            while not a.alive(row):
                row = rng.int(0, a.rows() - 1)
            var what = rng.int(0, 15)
            if what <= 2:
                var v = Int64(rng.int(-1_000_00, 3_000_00))
                a.set(rv, row, v)
                b.set(rv, row, v)
            elif what == 3:
                a.set_null(rv, row)
                b.set_null(rv, row)
            elif what == 4:
                var w = String("West")
                if rng.int(0, 1) == 0:
                    w = "Fresh" + String(rng.int(0, 3))
                a.set_text(rc, row, w)
                b.set_text(rc, row, w)
            elif what == 5:
                var w = "memo " + String(rng.int(0, 5))
                a.set_text(nc, row, w)
                b.set_text(nc, row, w)
            elif what == 6:
                var vals: List[String] = ["East", "n2", "3", "12.50", "7.25", "2019-05-03", "true"]
                _ = a.append(vals)
                _ = b.append(vals)
            elif what == 7:
                var vals: List[String] = ["Zed", "", "", "", "", "", ""]
                _ = a.append(vals)
                _ = b.append(vals)
            elif what == 8 or what == 9:
                a.remove(row)
                b.remove(row)
            elif what == 10:
                a.set_null(ca, row)
                b.set_null(ca, row)
            elif what == 11:
                a.set_text(cs, row, "7.25")
                b.set_text(cs, row, "7.25")
            elif what == 12:
                a.set_text(dy, row, "2020-02-29")
                b.set_text(dy, row, "2020-02-29")
            elif what == 13:
                a.set_null(pd, row)
                b.set_null(pd, row)
            elif what == 14:
                a.set_text(pd, row, "false")
                b.set_text(pd, row, "false")
            else:
                var x = Int64(rng.int(0, a.word_count(rc) - 1))
                var y = Int64(rng.int(0, a.word_count(rc) - 1))
                var codes: List[Int64] = [x]
                expect(a.merge(rc, codes, y) == b.merge(rc, codes, y), "a merge folds as many values in both")
        if round == 1:
            expect(a.formula("margin = revenue - cost") == b.formula("margin = revenue - cost"), "formula column")
            expect(a.formula("half = margin / 2") == b.formula("half = margin / 2"), "formula over a formula")
        check(a, b, rng, "after " + String(steps) + " edits", round >= 1, saved)
    var undone = 0
    while a.undo():
        expect(b.undo(), "undo in the scan frame as often as in the memory frame")
        undone += 1
        if undone % 150 == 0:
            check(a, b, rng, "after " + String(undone) + " undos", a.columns() > a.stored(), saved)
    expect(not b.undo(), "nothing left to undo")
    check(a, b, rng, "all undone", a.columns() > a.stored(), saved)
    # fully undone, the saved file holds the original rows
    same_file(saved, a.bake(), "undone")


def test_the_cache_is_bounded_and_serves_a_scrolling_grid() raises:
    var table = make_table(N)
    var path = scratch("cache.parquet")
    write_parquet(path, table, "", ZSTD, GROUP)
    var f = Frame(scan_parquet(path))
    var chunk = 8 * GROUP + GROUP // 8
    f.set_cache_limit(2 * chunk + 100)                       # room for two column chunks
    var cs = f.column("cost")
    var rv = f.column("revenue")
    expect(f.cache_loads() == 0, "opening reads no column")
    for row in range(0, GROUP):                              # a screenful after screenful in one row group: one decode
        _ = f.get(cs, row)
    expect(f.cache_loads() == 1, "one column chunk decoded for a whole row group of reads, not " + String(f.cache_loads()))
    for row in range(0, GROUP, 7):
        _ = f.get(cs, row)
        _ = f.is_null(rv, row)
    expect(f.cache_loads() == 2, "two columns of the group")
    for g in range(N // GROUP):                              # walking the file: the cache never holds more than two chunks
        _ = f.get(cs, g * GROUP + 5)
        _ = f.get(rv, g * GROUP + 5)
        expect(f.cache_held() <= 2 * chunk + 100, "the cache stays within its limit: " + String(f.cache_held()))
    var table_value = table.get(table.index("cost"), 3 * GROUP + 5)
    expect(f.get(cs, 3 * GROUP + 5) == table_value, "a value read after eviction is the file's")
    f.set_cache_limit(0)
    expect(f.cache_held() == 0, "a limit of zero empties the cache")
    expect(f.get(cs, 17) == table.get(table.index("cost"), 17), "and reads still work (one chunk at a time)")


def test_baking_a_streamed_frame_is_refused_and_save_is_the_way() raises:
    var table = make_table(2_000)
    var path = scratch("bake.parquet")
    write_parquet(path, table, "", ZSTD, 800)
    var f = Frame(scan_parquet(path))
    var msg = String("")
    try:
        _ = f.bake()
    except e:
        msg = String(e)
    expect(msg == "a Frame over a streamed file cannot be baked into memory: save it to a new file with `save(path)`", "bake says what to do: " + msg)
    f.remove(0)
    f.save(scratch("bake.out.parquet"))
    var back = read_parquet(scratch("bake.out.parquet"))
    expect(back.table.rows() == 1_999, "the saved file has the remaining rows")


def test_save_over_the_file_that_is_open() raises:
    var table = make_table(3_000)
    var path = scratch("inplace.parquet")
    write_parquet(path, table, "", SNAPPY, 800)
    var f = Frame(scan_parquet(path))
    f.set(f.column("cost"), 5, 12345)
    f.remove(7)
    f.save(path)                                             # replaces by rename: the scan keeps reading what it opened
    expect(f.get(f.column("cost"), 5) == 12345, "the open frame still reads")
    var back = read_parquet(path)
    var t = back.take_table().share()
    expect(t.rows() == 2_999 and t.get(t.index("cost"), 5) == 12345, "the file on disk has the edit")


def test_an_empty_result_and_codecs() raises:
    var table = make_table(48)
    var path = scratch("empty.parquet")
    write_parquet(path, table, "", UNCOMPRESSED, 8)
    var f = Frame(scan_parquet(path))
    for row in range(48):
        f.remove(row)
    f.save(scratch("empty.out.parquet"), "", SNAPPY)
    var back = read_parquet(scratch("empty.out.parquet"))
    expect(back.table.rows() == 0 and back.table.columns() == 7, "everything removed: an empty file with the schema")
    _ = f.append(["a", "b", "1", "2", "3", "2020-01-01", "true"])
    f.save(scratch("empty.out.parquet"), "", UNCOMPRESSED)
    var again = read_parquet(scratch("empty.out.parquet"))
    expect(again.table.rows() == 1 and again.table.text_of(0, Int(again.table.get(0, 0))) == "a", "only an appended row")


def main() raises:
    test_baking_a_streamed_frame_is_refused_and_save_is_the_way()
    test_save_over_the_file_that_is_open()
    test_an_empty_result_and_codecs()
    test_the_cache_is_bounded_and_serves_a_scrolling_grid()
    # F-149 scans row groups on the workers by blocks, or one group at a time: the overlay must hold on both
    _ = setenv("MXFRAME_SCAN_MORSELS", "0")
    test_random_edits_undo_and_save_equal_the_in_memory_frame()
    _ = setenv("MXFRAME_SCAN_MORSELS", "1")
    _ = setenv("MXFRAME_MORSEL_ROWS", "200")                 # blocks of 200 rows: 8 per group of 1,600, not all aligned to a page
    test_random_edits_undo_and_save_equal_the_in_memory_frame()
    _ = setenv("MXFRAME_SCAN_MORSELS", "")
    _ = setenv("MXFRAME_MORSEL_ROWS", "")
