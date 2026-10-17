# F-133 (M2): Query on the plan IR, lowered to the native CPU kernel. Every result is checked against a plain loop
# over the same random table (an independent oracle: no kernel, no program).
from mxframe import Table, SharedTable, INT, DECIMAL, TEXT, Query, col, lit, least, greatest, sum, count, mean, min_of, max_of


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

def ref_group(t: SharedTable, key_kind: Int, groups: Int, flt: Int) -> Tuple[List[Int64], List[Int64], List[Int64], List[Int64], List[Int64]]:
    """(sum revenue, count(*), count(revenue), min cost, max cost) per group.
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
        lo[g] = min(lo[g], t.get(cost, i))
        hi[g] = max(hi[g], t.get(cost, i))
    return (s^, n^, nv^, lo^, hi^)


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
    var r = q^.agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost"))).compute()
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


def test_every_app_shape_equals_the_oracle() raises:
    for null_every in [0, 11]:
        var t = make_table(UInt64(7 + null_every), null_every)
        var tag = " (nulls every " + String(null_every) + ")"
        check_against_oracle(t, 0, 7, 0, "chart: text-free key" + tag)
        check_against_oracle(t, 1, 35, 0, "pivot" + tag)
        check_against_oracle(t, 2, 52, 0, "bin" + tag)
        check_against_oracle(t, 3, 1, 1, "report tile, filtered" + tag)
        check_against_oracle(t, 0, 7, 1, "report chart, filtered" + tag)


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


def main() raises:
    test_every_app_shape_equals_the_oracle()
    test_mean_and_a_formula_value()
    test_overflow_and_unknown_column_raise()
