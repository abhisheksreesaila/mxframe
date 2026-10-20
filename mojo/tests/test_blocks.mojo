# F-176: range aggregates and block reads over a row order. The oracle is the per-cell path (`get` / `is_null`) the new calls replace:
# every aggregate is recomputed from the cells of the range read one at a time (with the same float order the call documents), every
# block cell by cell. Sources: a table, a Frame with edits / appended rows / removed rows / a merge, and a Frame over a Parquet file with
# row groups of an odd size (a range crosses them). DuckDB checks the aggregates in tests/test_mojo_range.py.
from std.time import perf_counter_ns
from mxframe import (Table, SharedTable, Frame, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, write_parquet, scan_parquet, ZSTD,
                     range_aggregate, read_block, RangeResult, MORSEL)
from mxframe.frame import text_ranks


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


def expect_msg(what: String, got: String, want: String) raises:
    if got != want:
        raise Error("FAILED: " + what + ": the error was `" + got + "`, wanted `" + want + "`")


comptime I = 0       # column numbers of make_table
comptime D2 = 1
comptime D4 = 2
comptime FL = 3
comptime TX = 4
comptime DT = 5
comptime BL = 6
comptime F32 = 7


def float_pool(mut rng: Rng) -> Float64:
    """Special values often, else a random double of any size."""
    var k = rng.int(0, 15)
    if k == 0:
        return Float64(0.0)
    if k == 1:
        return -Float64(0.0)
    if k == 2:
        return Float64(from_bits=UInt64(0x7FF8000000000000))          # NaN
    if k == 3:
        return Float64(from_bits=UInt64(0x7FF0000000000000))          # +inf
    if k == 4:
        return Float64(from_bits=UInt64(0xFFF0000000000000))          # -inf
    if k == 5:
        return Float64(from_bits=UInt64(1))                           # the smallest subnormal
    if k == 6:
        return Float64(1e300) * Float64(rng.int(1, 9))
    if k == 7:
        return -Float64(1e300) * Float64(rng.int(1, 9))
    return Float64(rng.int(-1_000_000, 1_000_000)) / 64.0 + Float64(rng.int(0, 99)) * 1e-7


def make_table(n: Int, seed: UInt64, specials: Bool = True) raises -> SharedTable:
    """int (nulls), two decimals of different scales (nulls), a float with NaN / inf / -0.0 / subnormals (nulls), text with case variants
    (nulls), a date (nulls), a bool (nulls), a 32-bit float column."""
    var t = Table.empty()
    var ci = t.add_column("i", INT)
    var c2 = t.add_column("d2", DECIMAL, scale=2)
    var c4 = t.add_column("d4", DECIMAL, scale=4)
    var cf = t.add_column("f", FLOAT)
    var cs = t.add_column("s", TEXT)
    var cd = t.add_column("dt", DATE)
    var cb = t.add_column("b", BOOL)
    var c32 = t.add_column("f32", FLOAT, scale=32)
    var words: List[String] = ["West", "west", "WEST", "east", "East", "North", "apple", "Zebra", "zebra", "mango", "Mango", "ändern", "Zoë"]
    for w in words:
        _ = t.add_word(cs, w)
    var rng = Rng(seed)
    for _ in range(n):
        if rng.int(0, 9) == 0:
            t.append_null(ci)
        else:
            t.append(ci, Int64(rng.int(-1000, 1000)))
        if rng.int(0, 9) == 0:
            t.append_null(c2)
        else:
            t.append(c2, Int64(rng.int(-100_000, 100_000)))
        if rng.int(0, 7) == 0:
            t.append_null(c4)
        else:
            t.append(c4, Int64(rng.int(-10_000_000, 10_000_000)))
        if rng.int(0, 9) == 0:
            t.append_null(cf)
        else:
            t.append_float(cf, float_pool(rng) if specials else Float64(rng.int(-5000, 5000)) / 8.0)
        if rng.int(0, 9) == 0:
            t.append_null(cs)
        else:
            t.append(cs, Int64(rng.int(0, len(words) - 1)))
        if rng.int(0, 19) == 0:
            t.append_null(cd)
        else:
            t.append(cd, Int64(rng.int(-3000, 25000)))
        if rng.int(0, 9) == 0:
            t.append_null(cb)
        else:
            t.append(cb, Int64(rng.int(0, 1)))
        if rng.int(0, 9) == 0:
            t.append_null(c32)
        else:
            t.append_float(c32, Float64(Float32(Float64(rng.int(-5000, 5000)) / 7.0)))
    return t^.share()


def all_columns(n: Int) -> List[Int]:
    var out = List[Int]()
    for c in range(n):
        out.append(c)
    return out^


def shuffled(n: Int, mut rng: Rng) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append(i)
    for i in range(n - 1, 0, -1):
        var j = rng.int(0, i)
        var tmp = out[i]
        out[i] = out[j]
        out[j] = tmp
    return out^


# ── the per-cell oracle ──────────────────────────────────────────────────────────────────────────────

struct Cells(Movable):
    var vals: List[List[Int64]]       # per requested column, the range's cells in order (null: 0)
    var nul: List[List[Bool]]

    def __init__(out self):
        self.vals = List[List[Int64]]()
        self.nul = List[List[Bool]]()


def cells_table(t: SharedTable, order: List[Int], cols: List[Int], a: Int, b: Int, identity: Bool) -> Cells:
    var out = Cells()
    for c in cols:
        var v = List[Int64]()
        var n = List[Bool]()
        for p in range(a, b):
            var row = p if identity else order[p]
            n.append(t.is_null(c, row))
            v.append(0 if t.is_null(c, row) else t.get(c, row))
        out.vals.append(v^)
        out.nul.append(n^)
    return out^


def cells_frame(f: Frame, order: List[Int], cols: List[Int], a: Int, b: Int, identity: Bool) raises -> Cells:
    var out = Cells()
    for c in cols:
        var v = List[Int64]()
        var n = List[Bool]()
        for p in range(a, b):
            var row = p if identity else order[p]
            var nl = f.is_null(c, row)
            n.append(nl)
            v.append(0 if nl else f.get(c, row))
        out.vals.append(v^)
        out.nul.append(n^)
    return out^


def same_float(a: Float64, b: Float64) -> Bool:
    return (not (a == a) and not (b == b)) or a == b


def check_aggregates(label: String, r: RangeResult, cells: Cells, kinds: List[Int], scales: List[Int], words: List[List[String]]) raises:
    var n = len(cells.vals[0]) if len(cells.vals) > 0 else 0
    expect(r.rows == n, label + ": rows")
    expect(len(r.columns) == len(kinds), label + ": columns")
    for k in range(len(kinds)):
        var kind = kinds[k]
        var tag = label + " col " + String(k) + " (" + String(kind) + ")"
        ref agg = r.columns[k]
        var cnt = 0
        for i in range(n):
            if not cells.nul[k][i]:
                cnt += 1
        expect(agg.count == cnt, tag + ": count " + String(agg.count) + " vs " + String(cnt))
        expect(agg.has_value() == (cnt > 0), tag + ": has_value")
        if cnt == 0:
            expect(not agg.has_sum, tag + ": no sum of nothing")
            continue
        if kind == FLOAT.value:
            var total = Float64(0.0)
            var i = 0
            while i < n:                                  # the documented order: morsels from the first position, each added in order
                var s = Float64(0.0)
                var seen = False
                for j in range(i, min(i + MORSEL, n)):
                    if not cells.nul[k][j]:
                        s += Float64(from_bits=UInt64(cells.vals[k][j]))
                        seen = True
                if seen:
                    total += s
                i += MORSEL
            expect(agg.has_sum and same_float(agg.sum_float(), total), tag + ": float sum " + String(agg.sum_float()) + " vs " + String(total))
            expect(same_float(agg.mean, total / Float64(cnt)), tag + ": float mean")
            var mn = Float64(0.0)
            var mx = Float64(0.0)
            var have = False
            var nans = 0
            for j in range(n):
                if cells.nul[k][j]:
                    continue
                var x = Float64(from_bits=UInt64(cells.vals[k][j]))
                if not (x == x):
                    nans += 1
                    continue
                if not have:
                    mn = x
                    mx = x
                    have = True
                else:
                    if x < mn:
                        mn = x
                    if x > mx:
                        mx = x
            var want_min = mn if have else Float64(from_bits=UInt64(0x7FF8000000000000))
            var want_max = Float64(from_bits=UInt64(0x7FF8000000000000)) if nans > 0 else mx
            expect(same_float(agg.min_float(), want_min), tag + ": float min " + String(agg.min_float()) + " vs " + String(want_min))
            expect(same_float(agg.max_float(), want_max), tag + ": float max " + String(agg.max_float()) + " vs " + String(want_max))
        elif kind == TEXT.value:
            expect(not agg.has_sum, tag + ": text has no sum")
            var rank = text_ranks(words[k])
            var lo = -1
            var hi = -1
            for j in range(n):
                if cells.nul[k][j]:
                    continue
                var code = Int(cells.vals[k][j])
                if lo < 0 or rank[code] < rank[lo]:
                    lo = code
                if hi < 0 or rank[code] > rank[hi]:
                    hi = code
            expect(Int(agg.min) == lo and Int(agg.max) == hi, tag + ": text min/max codes " + String(agg.min) + "," + String(agg.max) + " vs " + String(lo) + "," + String(hi))
        else:
            var s: Int64 = 0
            var mn = Int64.MAX
            var mx = Int64.MIN
            for j in range(n):
                if cells.nul[k][j]:
                    continue
                var v = cells.vals[k][j]
                s += v
                mn = min(mn, v)
                mx = max(mx, v)
            expect(agg.min == mn and agg.max == mx, tag + ": min/max " + String(agg.min) + "," + String(agg.max) + " vs " + String(mn) + "," + String(mx))
            if kind == INT.value or kind == DECIMAL.value:
                expect(agg.has_sum and agg.sum == s, tag + ": sum " + String(agg.sum) + " vs " + String(s))
                var p10 = Float64(1.0)
                for _ in range(scales[k]):
                    p10 *= 10.0
                var want = Float64(s) / (Float64(cnt) * p10)
                expect(abs(agg.mean - want) <= abs(want) * 1e-15, tag + ": mean " + String(agg.mean) + " vs " + String(want))
            else:
                expect(not agg.has_sum, tag + ": no sum of a date or a bool")


def check_block(label: String, cells: Cells, values: List[Int64], nulls: List[UInt8], n: Int) raises:
    for k in range(len(cells.vals)):
        for i in range(n):
            var want_null = cells.nul[k][i]
            expect((nulls[k * n + i] == 1) == want_null, label + ": null flag col " + String(k) + " row " + String(i))
            expect(values[k * n + i] == cells.vals[k][i], label + ": value col " + String(k) + " row " + String(i) + " " +
                   String(values[k * n + i]) + " vs " + String(cells.vals[k][i]))


def kinds_of(t: SharedTable, cols: List[Int]) -> List[Int]:
    var out = List[Int]()
    for c in cols:
        out.append(t.kind(c).value)
    return out^


def scales_of(t: SharedTable, cols: List[Int]) -> List[Int]:
    var out = List[Int]()
    for c in cols:
        out.append(t.scale(c))
    return out^


def words_of_table(t: SharedTable, cols: List[Int]) -> List[List[String]]:
    var out = List[List[String]]()
    for c in cols:
        var w = List[String]()
        if t.kind(c) == TEXT:
            for x in t.dictionary(c):
                w.append(x)
        out.append(w^)
    return out^


def words_of_frame(f: Frame, cols: List[Int]) raises -> List[List[String]]:
    var out = List[List[String]]()
    for c in cols:
        var w = List[String]()
        if f.kind(c) == TEXT:
            for k in range(f.word_count(c)):
                w.append(f.text_of(c, Int64(k)))
        out.append(w^)
    return out^


def ranges_for(n: Int, mut rng: Rng) -> List[Tuple[Int, Int]]:
    var out = List[Tuple[Int, Int]]()
    out.append((0, 0))
    out.append((n // 2, n // 2))
    out.append((0, 1))
    out.append((n - 1, n))
    out.append((3, 4))
    out.append((0, n))
    out.append((0, min(MORSEL, n)))
    out.append((min(1, n), min(MORSEL, n)))
    out.append((min(MORSEL - 1, n), min(MORSEL + 1, n)))
    out.append((min(5, n), min(5 + 3 * MORSEL + 7, n)))
    for _ in range(8):
        var a = rng.int(0, n)
        out.append((a, rng.int(a, n)))
    return out^


# ── tests ────────────────────────────────────────────────────────────────────────────────────────────

def test_table() raises:
    var n = 3 * MORSEL + 4_321
    var t = make_table(n, 11)
    var rng = Rng(5)
    var cols = all_columns(t.columns())
    var kinds = kinds_of(t, cols)
    var scales = scales_of(t, cols)
    var words = words_of_table(t, cols)
    var f = Frame(t)
    var by_int = f.permutation(I)
    var by_text_desc = f.permutation(TX, True)
    var orders = List[List[Int]]()
    orders.append(by_int.copy())
    orders.append(by_text_desc.copy())
    orders.append(shuffled(n, rng))
    var repeats = List[Int]()                                   # a row list may repeat rows and need not cover the table
    for _ in range(MORSEL + 100):
        repeats.append(rng.int(0, n - 1))
    orders.append(repeats^)
    for o in range(len(orders)):
        for rg in ranges_for(len(orders[o]), rng):
            var cells = cells_table(t, orders[o], cols, rg[0], rg[1], False)
            var label = "table order " + String(o) + " " + String(rg[0]) + ".." + String(rg[1])
            check_aggregates(label, range_aggregate(t, orders[o], cols, rg[0], rg[1]), cells, kinds, scales, words)
            var values = List[Int64](length=len(cols) * (rg[1] - rg[0]), fill=-7)
            var nulls = List[UInt8](length=len(cols) * (rg[1] - rg[0]), fill=9)
            read_block(t, orders[o], cols, rg[0], rg[1], values, nulls)
            check_block(label + " block", cells, values, nulls, rg[1] - rg[0])
    for rg in ranges_for(n, rng):                               # no order: the rows themselves
        var cells = cells_table(t, List[Int](), cols, rg[0], rg[1], True)
        check_aggregates("table rows " + String(rg[0]) + ".." + String(rg[1]), range_aggregate(t, cols, rg[0], rg[1]), cells, kinds, scales, words)
        var values = List[Int64](length=len(cols) * (rg[1] - rg[0]), fill=-7)
        var nulls = List[UInt8](length=len(cols) * (rg[1] - rg[0]), fill=9)
        read_block(t, cols, rg[0], rg[1], values, nulls)
        check_block("table rows block", cells, values, nulls, rg[1] - rg[0])
    # a subset and a reordering of columns
    var some: List[Int] = [TX, FL, I]
    var cells = cells_table(t, orders[0], some, 10, 9000, False)
    check_aggregates("subset", range_aggregate(t, orders[0], some, 10, 9000), cells, kinds_of(t, some), scales_of(t, some), words_of_table(t, some))
    print("ok   table: aggregates and blocks equal the per-cell loop (4 orders, 18 ranges each, 8 columns)")


def test_sums_exact() raises:
    # int64 totals: only a total outside the range raises; partial sums past int64 inside a morsel do not
    var t = Table.empty()
    var c = t.add_column("big", INT)
    var d = t.add_column("m", DECIMAL, scale=18)
    var vals: List[Int64] = [6_000_000_000_000_000_000, 6_000_000_000_000_000_000, -6_000_000_000_000_000_000, -6_000_000_000_000_000_000, 7, -3]
    for v in vals:
        t.append(c, v)
        t.append(d, v)
    var s = t^.share()
    var cols: List[Int] = [0, 1]
    var r = range_aggregate(s, cols, 0, 6)
    expect(r.columns[0].sum == 4 and r.columns[1].sum == 4, "a partial sum past int64 is fine when the total fits: " + String(r.columns[0].sum))
    expect(r.columns[0].min == -6_000_000_000_000_000_000 and r.columns[0].max == 6_000_000_000_000_000_000, "big min / max")
    var msg = String()
    try:
        _ = range_aggregate(s, cols, 0, 2)
    except e:
        msg = String(e)
    expect_msg("sum past int64", msg, "the sum of column `big` passed the int64 range (about 9.2e18)")
    var one: List[Int] = [0]
    msg = String()
    try:
        _ = range_aggregate(s, one, 2, 4)
    except e:
        msg = String(e)
    expect_msg("sum below int64", msg, "the sum of column `big` passed the int64 range (about 9.2e18)")
    print("ok   sums: exact int64 totals, partial overflow inside a morsel is not an error")


def test_bounds() raises:
    var t = make_table(10, 3)
    var cols: List[Int] = [I]
    var order: List[Int] = [3, 1, 2, 0, 9, 8, 7, 6, 5, 4]
    var values = List[Int64](length=10, fill=0)
    var nulls = List[UInt8](length=10, fill=0)
    var r = range_aggregate(t, order, cols, 4, 4)
    expect(r.rows == 0 and r.columns[0].count == 0 and not r.columns[0].has_value(), "an empty range has nothing")
    var msg = String()
    try:
        _ = range_aggregate(t, order, cols, -1, 5)
    except e:
        msg = String(e)
    expect_msg("start before 0", msg, "the range -1..5 starts before the first position")
    msg = String()
    try:
        _ = range_aggregate(t, order, cols, 5, 3)
    except e:
        msg = String(e)
    expect_msg("end before start", msg, "the range 5..3 ends before it starts")
    msg = String()
    try:
        _ = range_aggregate(t, order, cols, 0, 11)
    except e:
        msg = String(e)
    expect_msg("past the order", msg, "the range 0..11 goes past the end: the order has 10 rows")
    msg = String()
    try:
        _ = range_aggregate(t, cols, 2, 11)
    except e:
        msg = String(e)
    expect_msg("past the rows", msg, "the range 2..11 goes past the end: there are 10 rows")
    var one_row = List[Int]()
    one_row.append(9)
    msg = String()
    try:
        _ = range_aggregate(t, one_row, cols, 0, 2)
    except e:
        msg = String(e)
    expect_msg("past one row", msg, "the range 0..2 goes past the end: the order has 1 row")
    var bad_col: List[Int] = [99]
    msg = String()
    try:
        _ = range_aggregate(t, bad_col, 0, 1)
    except e:
        msg = String(e)
    expect_msg("no such column", msg, "column 99 does not exist (the table has 8)")
    var wild: List[Int] = [0, 1, 12, 3]
    msg = String()
    try:
        _ = range_aggregate(t, wild, cols, 0, 4)
    except e:
        msg = String(e)
    expect_msg("a row past the table", msg, "position 2 of the order is row 12, but there are only 10 rows")
    var neg: List[Int] = [0, -1]
    msg = String()
    try:
        read_block(t, neg, cols, 0, 2, values, nulls)
    except e:
        msg = String(e)
    expect_msg("a negative row", msg, "position 1 of the order is row -1, but there are only 10 rows")
    var tiny = List[Int64](length=3, fill=0)
    msg = String()
    try:
        read_block(t, cols, 0, 10, tiny, nulls)
    except e:
        msg = String(e)
    expect_msg("small buffer", msg, "the buffers hold 3 cells but the block has 10 (columns x rows)")
    var f = Frame(t)
    f.remove(4)
    msg = String()
    try:
        _ = range_aggregate(f, cols, 0, 10)
    except e:
        msg = String(e)
    expect_msg("removed row", msg, "row 4 (position 4) was removed")
    var around: List[Int] = [0, 1, 2, 3, 5, 6]
    var ok = range_aggregate(f, around, cols, 0, 6)
    expect(ok.columns[0].count > 0, "a range that does not touch the removed row is fine")
    var formula = f.formula("twice = i + i")
    var fcols: List[Int] = [formula]
    msg = String()
    try:
        _ = range_aggregate(f, fcols, 0, 2)
    except e:
        msg = String(e)
    expect_msg("formula column", msg, "column `twice` is a formula: ranges read the stored columns")
    print("ok   bounds: empty, single, whole, out of range and bad rows raise readably")


def edit_frame(mut f: Frame, n: Int, seed: UInt64) raises:
    """Sets, nulls, new words, appended rows, removed rows and a merge."""
    var rng = Rng(seed)
    var words: List[String] = ["West", "zzz-new", "Mid", "Q", "apple"]
    for _ in range(300):
        var what = rng.int(0, 11)
        var row = rng.int(0, n - 1)
        if what <= 2 and f.alive(row):
            f.set(I, row, Int64(rng.int(-1500, 1500)))
        elif what == 3 and f.alive(row):
            f.set_null(rng.int(0, 6), row)
        elif what == 4 and f.alive(row):
            f.set_text(D2, row, String(rng.int(-500, 500)) + "." + String(rng.int(10, 99)))
        elif what == 5 and f.alive(row):
            f.set_text(TX, row, words[rng.int(0, 4)])
        elif what == 6 and f.alive(row):
            f.set_float(FL, row, float_pool(rng))
        elif what == 7:
            _ = f.append([String(rng.int(-9, 9)), "1.50", "0.0001", "2.5", words[rng.int(0, 4)], "2020-02-03", "true", ""])
        elif what == 8 and f.alive(row):
            f.set_text(DT, row, "2021-0" + String(rng.int(1, 9)) + "-1" + String(rng.int(0, 9)))
        elif what == 9 and row % 3 == 0:
            f.remove(row)
        elif what == 10 and f.alive(row):
            f.set_float(F32, row, Float64(rng.int(-100, 100)) / 4.0)
    var codes = List[Int64]()
    codes.append(Int64(f.code_of(TX, "zzz-new")))
    codes.append(Int64(f.code_of(TX, "Mid")))
    codes.append(Int64(f.code_of(TX, "east")))
    _ = f.merge(TX, codes, Int64(f.code_of(TX, "West")))


def run_frame(label: String, mut f: Frame, n: Int) raises:
    var rng = Rng(77)
    var cols = all_columns(8)
    var kinds = List[Int]()
    var scales = List[Int]()
    for c in cols:
        kinds.append(f.kind(c).value)
        scales.append(f.scale(c))
    var words = words_of_frame(f, cols)
    var orders = List[List[Int]]()
    orders.append(f.permutation(I))
    orders.append(f.permutation(TX, True))
    orders.append(f.permutation(FL))
    orders.append(f.permutation(DT, True))
    for o in range(len(orders)):
        for rg in ranges_for(len(orders[o]), rng):
            var cells = cells_frame(f, orders[o], cols, rg[0], rg[1], False)
            var tag = label + " order " + String(o) + " " + String(rg[0]) + ".." + String(rg[1])
            check_aggregates(tag, range_aggregate(f, orders[o], cols, rg[0], rg[1]), cells, kinds, scales, words)
            var values = List[Int64](length=len(cols) * (rg[1] - rg[0]), fill=-7)
            var nulls = List[UInt8](length=len(cols) * (rg[1] - rg[0]), fill=9)
            read_block(f, orders[o], cols, rg[0], rg[1], values, nulls)
            check_block(tag + " block", cells, values, nulls, rg[1] - rg[0])
    # the words of a block are reachable: a TEXT code reads as the frame's word, merges and session words included
    var v = List[Int64](length=3 * 50, fill=0)
    var nl = List[UInt8](length=3 * 50, fill=0)
    var some: List[Int] = [TX, I, D2]
    read_block(f, orders[0], some, 100, 150, v, nl)
    for i in range(50):
        if nl[i] == 0:
            expect(f.text_of(TX, v[i]) == f.text_of(TX, f.get(TX, orders[0][100 + i])), label + ": word of a block code")
    # the rows themselves are fine as long as none was removed; a removed row raises
    var removed_any = False
    for row in range(f.rows()):
        if not f.alive(row):
            removed_any = True
            break
    if removed_any:
        var all_rows = List[Int]()
        for _ in range(0):
            all_rows.append(0)
        var raised = False
        try:
            _ = range_aggregate(f, cols, 0, f.rows())
        except e:
            raised = String(e).find("was removed") >= 0
        expect(raised, label + ": a removed row in the identity range raises")


def test_frame() raises:
    var n = 2 * MORSEL + 5_000
    var f = Frame(make_table(n, 21))
    edit_frame(f, n, 99)
    run_frame("frame", f, n)
    print("ok   frame: edits, appended rows, removed rows, new words and a merge are in the aggregates and the blocks")
    # undo takes them out again
    while f.undo():
        pass
    run_frame("undone frame", f, n)
    print("ok   frame: after undoing every edit the results are the table's")


def test_scan() raises:
    var n = 2 * MORSEL + 5_000
    var t = make_table(n, 21)
    var path = "f176_scan.parquet"
    write_parquet(path, t, "", ZSTD, 3_000)                     # odd row groups: a range crosses them, none holds a multiple of a morsel
    var f = Frame(scan_parquet(path))
    run_frame("scan", f, n)
    print("ok   scan: a Frame over a 13-row-group file equals the per-cell loop, " + String(f.cache_loads()) + " chunks decoded")
    edit_frame(f, n, 99)
    run_frame("edited scan", f, n)
    print("ok   scan: edits, appended and removed rows, a merge over a file")
    var m = Frame(t)
    edit_frame(m, n, 99)
    var cols = all_columns(8)
    var order = m.permutation(I)
    var order2 = f.permutation(I)
    expect(len(order) == len(order2), "same order length")
    var ra = range_aggregate(m, order, cols, 17, n - 5)
    var rb = range_aggregate(f, order2, cols, 17, n - 5)
    for k in range(8):
        expect(ra.columns[k].count == rb.columns[k].count and ra.columns[k].sum == rb.columns[k].sum and ra.columns[k].min == rb.columns[k].min and
               ra.columns[k].max == rb.columns[k].max, "scan and table frames agree on column " + String(k))
    # a small window of a long file reuses the cache: the second read decodes nothing
    f.set_cache_limit(64 * 1024 * 1024)
    var some: List[Int] = [I, D2, FL]
    var v = List[Int64](length=3 * 50, fill=0)
    var nl = List[UInt8](length=3 * 50, fill=0)
    read_block(f, order2, some, 1000, 1050, v, nl)
    var loads = f.cache_loads()
    read_block(f, order2, some, 1000, 1050, v, nl)
    expect(f.cache_loads() == loads, "the second read of a window decodes nothing: " + String(f.cache_loads()) + " vs " + String(loads))
    print("ok   scan: the cache serves a repeated window (" + String(loads) + " chunks held)")


def test_timings() raises:
    var n = 1_000_000
    var t = make_table(n, 5, False)
    var f = Frame(t)
    var rng = Rng(9)
    var order = shuffled(n, rng)
    var cols: List[Int] = [I, D2, FL]
    var best = 1e9
    for _ in range(5):
        var t0 = perf_counter_ns()
        var r = range_aggregate(t, order, cols, 0, n)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
        expect(r.columns[0].count > 0, "timing run")
    print("time range sum+count+min+max+mean of 3 columns over 1M rows in a random order (table): " + String(best) + " ms")
    var one: List[Int] = [D2]
    best = 1e9
    for _ in range(5):
        var t0 = perf_counter_ns()
        var r = range_aggregate(t, order, one, 0, n)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
        expect(r.columns[0].count > 0, "timing run")
    print("time one DECIMAL column over 1M rows, random order (table): " + String(best) + " ms")
    best = 1e9
    for _ in range(5):
        var t0 = perf_counter_ns()
        var r = range_aggregate(t, one, 0, n)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
        expect(r.columns[0].count > 0, "timing run")
    print("time one DECIMAL column over 1M rows in table order (table): " + String(best) + " ms")
    var wide = Table.empty()                                     # a grid window: 40 columns
    for c in range(40):
        _ = wide.add_column("c" + String(c), INT if c % 3 == 0 else DECIMAL, scale=0 if c % 3 == 0 else 2)
    for r in range(200_000):
        for c in range(40):
            if (r + c) % 11 == 0:
                wide.append_null(c)
            else:
                wide.append(c, Int64(r * 7 + c))
    var ws = wide^.share()
    var wo = shuffled(200_000, rng)
    var wcols = all_columns(40)
    var values = List[Int64](length=40 * 50, fill=0)
    var nulls = List[UInt8](length=40 * 50, fill=0)
    best = 1e9
    for i in range(200):
        var t0 = perf_counter_ns()
        read_block(ws, wo, wcols, 1000 + i, 1050 + i, values, nulls)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
    print("time block read 50 rows x 40 columns (table, random order): " + String(best) + " ms")
    var wf = Frame(ws)
    wf.set(0, 5, Int64(1))
    wf.set_null(1, 9)
    best = 1e9
    for i in range(200):
        var t0 = perf_counter_ns()
        read_block(wf, wo, wcols, 1000 + i, 1050 + i, values, nulls)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
    print("time block read 50 rows x 40 columns (frame with two edits, random order): " + String(best) + " ms")
    _ = f.rows()


def main() raises:
    test_table()
    test_sums_exact()
    test_bounds()
    test_frame()
    test_scan()
    test_timings()
