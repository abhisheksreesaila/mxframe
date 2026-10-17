# F-133 (M2): Query on the plan IR, lowered to the native CPU kernel. Every result is checked against a plain loop
# over the same random table (an independent oracle: no kernel, no program).
from mxframe import Table, SharedTable, Expr, INT, DECIMAL, TEXT, Query, col, lit, least, greatest, sum, count, mean, min_of, max_of


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


comptime N = 300_000


def make_table(seed: UInt64, null_every: Int) raises -> SharedTable:
    """region TEXT (5 words), carrier INT 0..6 with some out-of-range keys, day INT 0..400, revenue DECIMAL(2)
    (cents; every 50th value past 2^53), cost INT; every `null_every`-th cell of revenue / day / carrier is null."""
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var c = t.add_column("carrier", INT)
    var d = t.add_column("day", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", INT)
    var words: List[String] = ["West", "East", "North", "South", "Central"]
    for w in words:
        _ = t.add_word(g, w)
    var rng = Rng(seed)
    for i in range(N):
        t.append(g, Int64(rng.int(0, 4)))
        if null_every > 0 and i % null_every == 3:
            t.append_null(c)
        else:
            t.append(c, Int64(rng.int(-1, 7)))             # -1 and 7 are outside groups=7: dropped
        if null_every > 0 and i % null_every == 5:
            t.append_null(d)
        else:
            t.append(d, Int64(rng.int(0, 400)))
        if null_every > 0 and i % null_every == 1:
            t.append_null(r)
        else:
            if i % 50 == 0:                                  # past 2^53 (a float sum would drift), sparse enough
                var big = Int64(1 << 53) + Int64(rng.int(0, 1000))   # that no running total passes int64
                t.append(r, big if (i // 50) % 2 == 0 else -big + 7)
            else:
                t.append(r, Int64(rng.int(-100_000_00, 100_000_00)))
        t.append(k, Int64(rng.int(-5000, 5000)))
    return t^.share()


# ── the oracle ───────────────────────────────────────────────────────────────────────────────────────

def ref_group(t: SharedTable, key_kind: Int, groups: Int, flt: Int) -> Tuple[List[Int64], List[Int64], List[Int64], List[Int64], List[Int64], List[Int64], List[Int64]]:
    """(sum revenue, count(*), count(revenue), min cost, max cost, min revenue, max revenue) per group.
    key_kind 0: carrier; 1: region * 7 + carrier (a pivot); 2: least(day // 7, 51) (a bin); 3: constant 0.
    flt 0: none; 1: region == 1 and 10 <= day <= 300."""
    var reg = t.index("region")
    var car = t.index("carrier")
    var day = t.index("day")
    var rev = t.index("revenue")
    var cost = t.index("cost")
    var s = List[Int64](length=groups, fill=0)
    var n = List[Int64](length=groups, fill=0)
    var nv = List[Int64](length=groups, fill=0)
    var lo = List[Int64](length=groups, fill=Int64.MAX)
    var hi = List[Int64](length=groups, fill=Int64.MIN)
    var rlo = List[Int64](length=groups, fill=Int64.MAX)
    var rhi = List[Int64](length=groups, fill=Int64.MIN)
    for i in range(t.rows()):
        if flt == 1 and (t.is_null(reg, i) or t.is_null(day, i) or t.get(reg, i) != 1 or
                         t.get(day, i) < 10 or t.get(day, i) > 300):
            continue
        var key: Int64
        if key_kind == 0:
            if t.is_null(car, i):
                continue
            key = t.get(car, i)
        elif key_kind == 1:
            if t.is_null(car, i) or t.is_null(reg, i):
                continue
            key = t.get(reg, i) * 7 + t.get(car, i)
        elif key_kind == 2:
            if t.is_null(day, i):
                continue
            key = min(t.get(day, i) // 7, 51)
        else:
            key = 0
        if key < 0 or key >= Int64(groups):
            continue
        var g = Int(key)
        n[g] += 1
        if not t.is_null(rev, i):
            s[g] += t.get(rev, i)
            nv[g] += 1
            rlo[g] = min(rlo[g], t.get(rev, i))
            rhi[g] = max(rhi[g], t.get(rev, i))
        lo[g] = min(lo[g], t.get(cost, i))
        hi[g] = max(hi[g], t.get(cost, i))
    return (s^, n^, nv^, lo^, hi^, rlo^, rhi^)


def check_against_oracle(t: SharedTable, key_kind: Int, groups: Int, flt: Int, label: String) raises:
    var q = Query(t)
    if flt == 1:
        q = q^.where(col("region") == 1).where(col("day").between(10, 300))
    if key_kind == 0:
        q = q^.group_by(col("carrier"), groups=groups)
    elif key_kind == 1:
        q = q^.group_by(col("region") * 7 + col("carrier"), groups=groups)
    elif key_kind == 2:
        q = q^.group_by(least(col("day") // 7, lit(51)), groups=groups)
    var r = q^.agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost")),
                   min_of(col("revenue")), max_of(col("revenue"))).compute()
    var want = ref_group(t, key_kind, groups, flt)
    expect(len(r.sums) == groups and len(r.counts) == groups, label + ": one entry per group")
    for g in range(groups):
        expect(r.sums[g] == Int(want[0][g]), label + " sum group " + String(g) + ": " + String(r.sums[g]) + " vs " + String(want[0][g]))
        expect(r.counts[g] == Int(want[1][g]), label + " count(*) group " + String(g))
        expect(r.values(2)[g] == Int(want[2][g]), label + " count(revenue) group " + String(g))
        if want[1][g] > 0:
            expect(r.values(3)[g] == Int(want[3][g]) and r.values(4)[g] == Int(want[4][g]), label + " min/max group " + String(g))
        else:
            expect(r.is_empty(g), label + " empty group " + String(g))
        if want[2][g] > 0:                           # min / max over revenue, which has nulls
            expect(r.values(5)[g] == Int(want[5][g]) and r.values(6)[g] == Int(want[6][g]), label + " nullable min/max " + String(g))
        expect(r.present(5, g) == (want[2][g] > 0), label + " present " + String(g))


def test_every_app_shape_equals_the_oracle() raises:
    for null_every in [0, 11]:
        var t = make_table(UInt64(7 + null_every), null_every)
        var tag = " (nulls every " + String(null_every) + ")"
        check_against_oracle(t, 0, 7, 0, "chart: text-free key" + tag)
        check_against_oracle(t, 1, 35, 0, "pivot" + tag)
        check_against_oracle(t, 2, 52, 0, "bin" + tag)
        check_against_oracle(t, 3, 1, 1, "report tile, filtered" + tag)
        check_against_oracle(t, 0, 7, 1, "report chart, filtered" + tag)
        check_against_oracle(t, 1, 35, 1, "pivot, filtered" + tag)
        check_against_oracle(t, 2, 52, 1, "bin, filtered" + tag)


def test_mean_and_a_formula_value() raises:
    var t = make_table(3, 0)
    var r = Query(t).group_by(col("region"), groups=5).agg(sum(col("cost") * 3 - col("day")), mean(col("cost"))).compute()
    var reg = t.index("region")
    for g in range(5):
        var s: Int64 = 0
        var c: Int64 = 0
        var n = 0
        for i in range(t.rows()):
            if t.get(reg, i) == Int64(g):
                s += t.get(t.index("cost"), i) * 3 - t.get(t.index("day"), i)
                c += t.get(t.index("cost"), i)
                n += 1
        expect(r.values(0)[g] == Int(s), "formula sum group " + String(g))
        expect(abs(r.means(1)[g] - Float64(c) / Float64(n)) < 1e-9, "mean group " + String(g))


def test_overflow_and_unknown_column_raise() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    for _ in range(10):
        t.append(a, Int64(1 << 62))
    var s = t^.share()
    var raised = False
    try:
        _ = Query(s).agg(sum(col("a"))).compute()
    except e:
        raised = "int64" in String(e)
        expect(raised, "overflow message: " + String(e))
    expect(raised, "an int64 overflow raised")
    raised = False
    try:
        _ = Query(s).agg(sum(col("nope"))).compute()
    except e:
        raised = "column `nope` not found" in String(e)
    expect(raised, "an unknown column raised")


def small_table() raises -> SharedTable:
    """a, b with negatives, zeros and int64 extremes, g with nulls (rows 2 and 5)."""
    var t = Table.empty()
    var a = t.add_column("a", INT)
    var b = t.add_column("b", INT)
    var g = t.add_column("g", INT)
    var av: List[Int] = [-7, 7, -7, 7, 0, 5, 3, -9, 9, 1]
    var bv: List[Int] = [2, -2, -2, 2, 3, 0, 0, 1, 2, 1]
    for i in range(10):
        t.append(a, Int64(av[i]))
        t.append(b, Int64(bv[i]))
        if i == 2 or i == 5:
            t.append_null(g)
        else:
            t.append(g, Int64(i % 3))
    return t^.share()


def per_row(t: SharedTable, e: Expr) raises -> List[Int]:
    """e's value on each row: group by the row (a key of 0 .. rows-1 needs a row id; use a sum per row instead)."""
    var out = List[Int]()
    for i in range(t.rows()):
        var r = Query(t).where(col("a") == Int(t.get(0, i))).where(col("b") == Int(t.get(1, i))).agg(sum(e)).compute()
        out.append(r.sums[0])
    return out^


def test_integer_arithmetic_matches_mojo() raises:
    """// and % follow Mojo's Int64 (floor; the remainder takes the divisor's sign), x // 0 = x % 0 = 0, unary minus,
    least / greatest; checked row by row against Mojo itself (rows are distinct (a, b) pairs)."""
    var t = small_table()
    var fd = per_row(t, col("a") // col("b"))
    var md = per_row(t, col("a") % col("b"))
    var mn = per_row(t, -col("a") + least(col("a"), col("b")) * 2 - greatest(col("a"), col("b")))
    for i in range(t.rows()):
        var a = t.get(0, i)
        var b = t.get(1, i)
        var want_fd = a // b if b != 0 else 0
        var want_md = a % b if b != 0 else 0
        expect(fd[i] == Int(want_fd), "a // b row " + String(i) + ": " + String(fd[i]) + " vs " + String(want_fd))
        expect(md[i] == Int(want_md), "a % b row " + String(i) + ": " + String(md[i]) + " vs " + String(want_md))
        expect(mn[i] == Int(-a + min(a, b) * 2 - max(a, b)), "neg / least / greatest row " + String(i))


def test_int64_min_floordiv_minus_one_raises() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    t.append(a, Int64.MIN)
    var s = t^.share()
    var raised = False
    try:
        _ = Query(s).agg(sum(col("a") // -1)).compute()
    except e:
        raised = "int64" in String(e)
    expect(raised, "INT64_MIN // -1 raised")
    expect(Query(s).agg(sum(col("a") % -1)).compute().sums[0] == 0, "INT64_MIN % -1 is 0")


def test_or_not_and_ne_filters_and_the_null_rule() raises:
    var t = small_table()
    var n_or = Query(t).where((col("a") > 0) | (col("g") == 1)).agg(count()).compute().counts[0]
    var n_not = Query(t).where(~(col("a") > 0)).agg(count()).compute().counts[0]
    var n_ne = Query(t).where(col("a") != 7).agg(count()).compute().counts[0]
    var want_or = 0
    var want_not = 0
    var want_ne = 0
    for i in range(t.rows()):
        var a = t.get(0, i)
        if not t.is_null(2, i) and (a > 0 or t.get(2, i) == 1):   # the documented rule: a null makes the filter false
            want_or += 1
        if a <= 0:
            want_not += 1
        if a != 7:
            want_ne += 1
    expect(n_or == want_or, "OR with the null rule: " + String(n_or) + " vs " + String(want_or))
    expect(n_not == want_not and n_ne == want_ne, "NOT, !=")


def test_mean_and_minmax_with_nulls_and_an_all_null_group() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    var ks: List[Int] = [0, 0, 0, 1, 1, 2]
    for i in range(6):
        t.append(k, Int64(ks[i]))
        if i == 1 or ks[i] == 1:
            t.append_null(v)
        else:
            t.append(v, Int64(10 * i - 25))
    var s = t^.share()
    var r = Query(s).group_by(col("k"), groups=3).agg(mean(col("v")), min_of(col("v")), max_of(col("v"))).compute()
    expect(abs(r.means(0)[0] - Float64(-25 + -5) / 2.0) < 1e-12, "mean skips nulls")
    expect(r.values(1)[0] == -25 and r.values(2)[0] == -5 and r.values(1)[2] == 25, "min / max skip nulls")
    expect(not r.present(1, 1) and not r.present(0, 1) and r.counts[1] == 2, "an all-null group has rows, no value")
    expect(r.present(1, 0) and r.present(2, 2), "groups with values")


def test_where_after_group_by_and_one_group_by() raises:
    var t = small_table()
    var r = Query(t).group_by(col("g"), groups=3).where(col("a") > 0).agg(count(), sum(col("a"))).compute()
    var want_n = List[Int](length=3, fill=0)
    for i in range(t.rows()):
        if not t.is_null(2, i) and t.get(0, i) > 0:
            want_n[Int(t.get(2, i))] += 1
    for g in range(3):
        expect(r.counts[g] == want_n[g], "where after group_by: group " + String(g))
    var msg = String("")
    try:
        _ = Query(t).agg(count()).group_by(col("g"), groups=3)
    except e:
        msg = String(e)
    expect("group_by must come before agg" in msg, "agg then group_by: " + msg)
    var raised = False
    try:
        _ = Query(t).group_by(col("g"), groups=3).group_by(col("a"), groups=2)
    except e:
        raised = "one group_by" in String(e)
    expect(raised, "a second group_by raised")


def test_kinds_and_scales_are_checked() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    var d = t.add_column("d", DECIMAL, scale=2)
    t.append(a, 1)
    t.append(d, 150)
    var s = t^.share()
    var msgs = List[String]()
    try:
        _ = Query(s).agg(sum((col("a") > 0) + 0)).compute()
    except e:
        msgs.append(String(e))
    try:
        _ = Query(s).where(col("a") & col("a")).agg(count()).compute()
    except e:
        msgs.append(String(e))
    try:
        _ = Query(s).agg(sum(col("a") + col("d"))).compute()
    except e:
        msgs.append(String(e))
    expect(len(msgs) == 3, "three errors, got " + String(len(msgs)))
    expect("`+` needs numbers on both sides" in msgs[0], msgs[0])
    expect("`&` needs comparisons on both sides" in msgs[1], msgs[1])
    expect("0 and 2 decimal places" in msgs[2], msgs[2])
    expect(Query(s).agg(sum(col("d") * col("a"))).compute().sums[0] == 150, "a product may mix scales")


def test_keys_past_int32_and_negative_are_left_out() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var keys: List[Int] = [(1 << 32) + 1, -1, 1, Int(Int64.MIN), 1 << 31, 3]
    for key in keys:
        t.append(k, Int64(key))
    var s = t^.share()
    var r = Query(s).group_by(col("k"), groups=4).agg(count()).compute()
    expect(r.counts[0] == 0 and r.counts[1] == 1 and r.counts[2] == 0 and r.counts[3] == 1,
           "2^32+1 must not land in group 1; -1, INT64_MIN and 2^31 are left out")


def test_a_sum_over_only_nulls_is_not_present() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    t.append(k, 0)
    t.append_null(v)
    t.append(k, 1)
    t.append(v, 5)
    var s = t^.share()
    var r = Query(s).group_by(col("k"), groups=2).agg(sum(col("v"))).compute()
    expect(not r.present(0, 0) and r.present(0, 1) and r.sums[1] == 5, "sum presence follows non-null values")


def test_a_key_formula_is_not_evaluated_on_filtered_rows() raises:
    var t = Table.empty()
    var x = t.add_column("x", INT)
    t.append(x, 1)
    t.append(x, 2)
    t.append(x, Int64(1 << 62))                    # x * 4 would overflow, but the filter leaves the row out
    var s = t^.share()
    var r = Query(s).where(col("x") < 10).group_by(col("x") * 4, groups=9).agg(count()).compute()
    expect(r.counts[4] == 1 and r.counts[8] == 1, "keys 4 and 8")


def main() raises:
    test_integer_arithmetic_matches_mojo()
    test_int64_min_floordiv_minus_one_raises()
    test_or_not_and_ne_filters_and_the_null_rule()
    test_mean_and_minmax_with_nulls_and_an_all_null_group()
    test_where_after_group_by_and_one_group_by()
    test_kinds_and_scales_are_checked()
    test_a_key_formula_is_not_evaluated_on_filtered_rows()
    test_keys_past_int32_and_negative_are_left_out()
    test_a_sum_over_only_nulls_is_not_present()
    test_every_app_shape_equals_the_oracle()
    test_mean_and_a_formula_value()
    test_overflow_and_unknown_column_raise()
