# F-148: count_distinct, median / percentile and top_k through Query, over a table, a Frame with edits (queried
# live, checked against a plain loop over its baked table) and a Parquet scan of a multi-row-group file. Every answer
# is checked against an oracle written here: plain loops over the cells, sorting with a merge sort, no kernel.
from mxframe import (Table, SharedTable, Query, QueryResult, Frame, col, lit, least, sum, count, mean, min_of, max_of,
                     count_distinct, percentile, median, top_k, INT, DECIMAL, TEXT, write_parquet, read_parquet,
                     scan_parquet, ZSTD, CPU, GPU, AUTO, DeviceTable, gpu_available)


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


comptime N = 150_000
comptime GROUP = 20_000


def make_table(n: Int, seed: UInt64) raises -> SharedTable:
    """region TEXT (mixed case, nullable), carrier INT -1..7 (nullable), cust INT 0..3000 (nullable), revenue
    DECIMAL(2) (ties, negatives, nullable), cost INT (narrow, many ties, a few extremes near +-2^60), day INT 0..400."""
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var c = t.add_column("carrier", INT)
    var u = t.add_column("cust", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", INT)
    var d = t.add_column("day", INT)
    var words: List[String] = ["west", "East", "north", "South", "central"]
    for w in words:
        _ = t.add_word(g, w)
    var rng = Rng(seed)
    for i in range(n):
        if i % 23 == 5:
            t.append_null(g)
        else:
            t.append(g, Int64(rng.int(0, 4)))
        if i % 31 == 7:
            t.append_null(c)
        else:
            t.append(c, Int64(rng.int(-1, 7)))
        if i % 17 == 2:
            t.append_null(u)
        else:
            t.append(u, Int64(rng.int(0, 3000)))
        if i % 13 == 1:
            t.append_null(r)
        else:
            t.append(r, Int64(rng.int(-500, 500) * (1 if i % 7 != 0 else 1000)))
        if i % 997 == 0:
            t.append(k, Int64(1 << 60) + Int64(rng.int(0, 3)) if i % 2 == 0 else -Int64(1 << 60) - Int64(rng.int(0, 3)))
        else:
            t.append(k, Int64(rng.int(-50, 50)))
        t.append(d, Int64(rng.int(0, 400)))
    return t^.share()


# ── the oracle ───────────────────────────────────────────────────────────────────────────────────────

comptime K_NONE = 0          # one group
comptime K_CARRIER = 1       # carrier, 7 groups
comptime K_REGION = 2        # region, 5 groups (more once a frame adds words)
comptime K_PIVOT = 3         # region * 7 + carrier

comptime V_CUST = 0
comptime V_REVENUE = 1
comptime V_COST = 2
comptime V_REGION = 3
comptime V_COST10 = 4        # cost // 10
comptime V_COST2 = 5         # cost * 2 - 1
comptime V_DAY = 6


def key_of(t: SharedTable, row: Int, kk: Int) -> Tuple[Int, Bool]:
    """(group, usable)."""
    if kk == K_NONE:
        return (0, True)
    var car = t.index("carrier")
    var reg = t.index("region")
    if kk == K_CARRIER:
        if t.is_null(car, row):
            return (0, False)
        return (Int(t.get(car, row)), True)
    if t.is_null(reg, row):
        return (0, False)
    if kk == K_REGION:
        return (Int(t.get(reg, row)), True)
    if t.is_null(car, row):
        return (0, False)
    return (Int(t.get(reg, row)) * 7 + Int(t.get(car, row)), True)


def val_of(t: SharedTable, row: Int, vk: Int) -> Tuple[Int64, Bool]:
    """(value, usable): a null value is left out."""
    var name = "cust" if vk == V_CUST else "revenue" if vk == V_REVENUE else "region" if vk == V_REGION else "day" if vk == V_DAY else "cost"
    var c = t.index(name)
    if t.is_null(c, row):
        return (0, False)
    var v = t.get(c, row)
    if vk == V_COST10:
        v = v // 10
    elif vk == V_COST2:
        v = v * 2 - 1
    return (v, True)


def gather(t: SharedTable, kk: Int, groups: Int, filtered: Bool, vk: Int) -> List[List[Int64]]:
    """The values of each group, as a plain loop sees them (the key in range and non-null; the filter true; value non-null)."""
    var out = List[List[Int64]]()
    for _ in range(groups):
        out.append(List[Int64]())
    var day = t.index("day")
    for row in range(t.rows()):
        if filtered and not (t.get(day, row) >= 10 and t.get(day, row) <= 300):
            continue
        var k = key_of(t, row, kk)
        if not k[1] or k[0] < 0 or k[0] >= groups:
            continue
        var v = val_of(t, row, vk)
        if v[1]:
            out[k[0]].append(v[0])
    return out^


def msort(mut v: List[Int64]):
    var n = len(v)
    var tmp = List[Int64](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if v[j] < v[i]:
                    tmp[k] = v[j]
                    j += 1
                else:
                    tmp[k] = v[i]
                    i += 1
                k += 1
            while i < mid:
                tmp[k] = v[i]
                i += 1
                k += 1
            while j < hi:
                tmp[k] = v[j]
                j += 1
                k += 1
            lo += 2 * width
        swap(v, tmp)
        width *= 2


def word_less(a: String, b: String) -> Bool:
    """Alphabetical ignoring ASCII case, bytes as the tie-break (the order of Frame.permutation)."""
    var la = a.lower()
    var lb = b.lower()
    if la != lb:
        return la < lb
    return a < b


def text_ranks_of(t: SharedTable, c: Int) -> List[Int64]:
    var n = len(t.dictionary(c))
    var rank = List[Int64](length=n, fill=0)
    for a in range(n):
        for b in range(n):
            if word_less(t.text_of(c, b), t.text_of(c, a)):
                rank[a] += 1
    return rank^


def ceil_index(p: Float64, n: Int) -> Int:
    var x = p * Float64(n)
    var c = Int(x)
    if Float64(c) < x:
        c += 1
    return max(c - 1, 0)


def no_rank() -> List[Int64]:
    return List[Int64]()


def check_agg(r: QueryResult, i: Int, kind: String, param: Float64, k: Int, lists: List[List[Int64]], rank: List[Int64],
              label: String) raises:
    """Aggregate i of the result against the per-group value lists. kind: "distinct", "pct" (param = p) or "top" (k).
    `rank` (text): the values are codes, ordered by their rank."""
    var G = len(lists)
    expect(len(r.counts) == G, label + ": groups " + String(len(r.counts)) + " vs " + String(G))
    var back = List[Int64](length=len(rank), fill=0)                  # rank -> code
    for code in range(len(rank)):
        back[Int(rank[code])] = Int64(code)
    for g in range(G):
        var key = List[Int64]()
        for v in lists[g]:
            key.append(rank[Int(v)] if len(rank) > 0 else v)
        msort(key)
        var m = len(key)
        var tag = label + " group " + String(g) + " (" + String(m) + " values)"
        expect(r.present(i, g) == (m > 0), tag + ": present")
        if kind == "distinct":
            var d = 0
            for x in range(m):
                if x == 0 or key[x] != key[x - 1]:
                    d += 1
            expect(r.values(i)[g] == d, tag + ": count distinct is " + String(d) + ", got " + String(r.values(i)[g]))
        elif kind == "median":                       # the mean of the two middle values at scale + 1: 5 * (a + b)
            var want = 0
            if m > 0:
                want = Int(5 * (key[ceil_index(0.5, m)] + key[m // 2]))
            expect(r.values(i)[g] == want, tag + ": median is " + String(want) + ", got " + String(r.values(i)[g]))
        elif kind == "pct":
            var want = 0
            if m > 0:
                var v = key[ceil_index(param, m)]
                want = Int(back[Int(v)] if len(rank) > 0 else v)
            expect(r.values(i)[g] == want, tag + ": percentile " + String(param) + " is " + String(want) + ", got " + String(r.values(i)[g]))
        else:
            var got = r.top(i, g)
            expect(len(got) == min(k, m), tag + ": top_k has " + String(min(k, m)) + " values, got " + String(len(got)))
            for x in range(len(got)):
                var v = key[m - 1 - x]
                var want = Int(back[Int(v)] if len(rank) > 0 else v)
                expect(got[x] == want, tag + ": top value " + String(x) + " is " + String(want) + ", got " + String(got[x]))


# ── the scenarios, one plan each, built on any source ────────────────────────────────────────────────

def plan(var q: Query, s: Int, nreg: Int) raises -> Query:
    if s == 0:
        return q^.group_by(col("carrier"), groups=7).agg(count(), sum(col("day")), count_distinct(col("cust")),
                                                          median(col("revenue")), percentile(col("revenue"), 0.9),
                                                          top_k(col("cost"), 3))
    if s == 1:
        return q^.where(col("day").between(10, 300)).agg(count_distinct(col("cust")), median(col("cost")),
                                                          percentile(col("cost"), 0.0), percentile(col("cost"), 1.0),
                                                          top_k(col("revenue"), 5), count_distinct(col("region")))
    if s == 2:
        return q^.group_by(col("carrier"), groups=7).agg(median(col("region")), top_k(col("region"), 2),
                                                          count_distinct(col("region")), percentile(col("region"), 0.0))
    if s == 3:
        return q^.group_by(col("region") * 7 + col("carrier"), groups=nreg * 7).agg(count_distinct(col("cust")))
    if s == 5:                  # top_k alone on a source: the kernel hook feeds per-worker heaps, no pairs are written
        return q^.group_by(col("carrier"), groups=7).agg(top_k(col("region"), 3), top_k(col("cust"), 2), count_distinct(col("day")))
    return q^.group_by(col("region"), groups=nreg).agg(count_distinct(col("cost") // 10), percentile(col("cost") * 2 - 1, 0.25),
                                                       top_k(col("cost") * 2 - 1, 4), count(), sum(col("day")))


def table_results(t: SharedTable, nreg: Int, window_rows: Int) raises -> List[QueryResult]:
    var out = List[QueryResult]()
    for s in range(6):
        var q = plan(Query(t), s, nreg)
        q._window_rows = window_rows
        out.append(q.compute())
    return out^


def check_all(t: SharedTable, rs: List[QueryResult], label: String) raises:
    """The five scenarios' results `rs` against the oracle over `t` (the table the source holds or bakes to)."""
    var rrank = text_ranks_of(t, t.index("region"))
    var nreg = len(t.dictionary(t.index("region")))
    var by_car_cust = gather(t, K_CARRIER, 7, False, V_CUST)
    var by_car_rev = gather(t, K_CARRIER, 7, False, V_REVENUE)
    var by_car_cost = gather(t, K_CARRIER, 7, False, V_COST)
    check_agg(rs[0], 2, "distinct", 0.0, 0, by_car_cust, no_rank(), label + " 0 distinct(cust)")
    check_agg(rs[0], 3, "median", 0.5, 0, by_car_rev, no_rank(), label + " 0 median(revenue)")
    check_agg(rs[0], 4, "pct", 0.9, 0, by_car_rev, no_rank(), label + " 0 p90(revenue)")
    check_agg(rs[0], 5, "top", 0.0, 3, by_car_cost, no_rank(), label + " 0 top3(cost)")
    var g_cust = gather(t, K_NONE, 1, True, V_CUST)
    var g_cost = gather(t, K_NONE, 1, True, V_COST)
    var g_rev = gather(t, K_NONE, 1, True, V_REVENUE)
    var g_reg = gather(t, K_NONE, 1, True, V_REGION)
    check_agg(rs[1], 0, "distinct", 0.0, 0, g_cust, no_rank(), label + " 1 distinct(cust)")
    check_agg(rs[1], 1, "median", 0.5, 0, g_cost, no_rank(), label + " 1 median(cost)")
    check_agg(rs[1], 2, "pct", 0.0, 0, g_cost, no_rank(), label + " 1 p0(cost)")
    check_agg(rs[1], 3, "pct", 1.0, 0, g_cost, no_rank(), label + " 1 p100(cost)")
    check_agg(rs[1], 4, "top", 0.0, 5, g_rev, no_rank(), label + " 1 top5(revenue)")
    check_agg(rs[1], 5, "distinct", 0.0, 0, g_reg, no_rank(), label + " 1 distinct(region)")
    var car_reg = gather(t, K_CARRIER, 7, False, V_REGION)
    check_agg(rs[2], 0, "pct", 0.5, 0, car_reg, rrank, label + " 2 median(region)")
    check_agg(rs[2], 1, "top", 0.0, 2, car_reg, rrank, label + " 2 top2(region)")
    check_agg(rs[2], 2, "distinct", 0.0, 0, car_reg, no_rank(), label + " 2 distinct(region)")
    check_agg(rs[2], 3, "pct", 0.0, 0, car_reg, rrank, label + " 2 p0(region)")
    check_agg(rs[5], 0, "top", 0.0, 3, car_reg, rrank, label + " 5 top3(region), alone")
    check_agg(rs[5], 1, "top", 0.0, 2, by_car_cust, no_rank(), label + " 5 top2(cust), alone")
    check_agg(rs[5], 2, "distinct", 0.0, 0, gather(t, K_CARRIER, 7, False, V_DAY), no_rank(), label + " 5 distinct(day)")
    var piv = gather(t, K_PIVOT, nreg * 7, False, V_CUST)
    check_agg(rs[3], 0, "distinct", 0.0, 0, piv, no_rank(), label + " 3 pivot distinct(cust)")
    var by_reg_10 = gather(t, K_REGION, nreg, False, V_COST10)
    var by_reg_2 = gather(t, K_REGION, nreg, False, V_COST2)
    check_agg(rs[4], 0, "distinct", 0.0, 0, by_reg_10, no_rank(), label + " 4 distinct(cost // 10)")
    check_agg(rs[4], 1, "pct", 0.25, 0, by_reg_2, no_rank(), label + " 4 p25(cost * 2 - 1)")
    check_agg(rs[4], 2, "top", 0.0, 4, by_reg_2, no_rank(), label + " 4 top4(cost * 2 - 1)")


def check_plain(t: SharedTable, rs: List[QueryResult], nreg: Int, label: String) raises:
    """The plain aggregates beside the holistic ones: count() and sum(day) per carrier and per region."""
    var day = t.index("day")
    var cnt = List[Int](length=7, fill=0)
    var sd = List[Int](length=7, fill=0)
    for row in range(t.rows()):
        var k = key_of(t, row, K_CARRIER)
        if k[1] and k[0] >= 0 and k[0] < 7:
            cnt[k[0]] += 1
            sd[k[0]] += Int(t.get(day, row))
    expect(len(rs[0].sums) == 7, label + ": r.sums has a value per carrier")
    for g in range(7):
        expect(rs[0].sums[g] == sd[g], label + ": r.sums (sum(day)) of carrier " + String(g) + " is " + String(sd[g]) + ", got " + String(rs[0].sums[g]))
        expect(rs[0].counts[g] == cnt[g], label + ": count of carrier " + String(g) + " is " + String(cnt[g]) + ", got " + String(rs[0].counts[g]))
        expect(rs[0].values(1)[g] == sd[g], label + ": sum(day) of carrier " + String(g))
    var cr = List[Int](length=nreg, fill=0)
    var sr = List[Int](length=nreg, fill=0)
    for row in range(t.rows()):
        var k = key_of(t, row, K_REGION)
        if k[1] and k[0] < nreg:
            cr[k[0]] += 1
            sr[k[0]] += Int(t.get(day, row))
    for g in range(nreg):
        expect(rs[4].sums[g] == sr[g], label + ": r.sums of region " + String(g))
        expect(rs[4].values(3)[g] == cr[g] and rs[4].values(4)[g] == sr[g], label + ": count and sum(day) of region " + String(g))


# ── tables ───────────────────────────────────────────────────────────────────────────────────────────

def test_over_a_table() raises:
    var t = make_table(N, 1)
    var nreg = len(t.dictionary(t.index("region")))
    var one = table_results(t, nreg, 0)
    check_all(t, one, "table")
    check_plain(t, one, nreg, "table")
    var cut = table_results(t, nreg, 65536)                 # 3 windows of 65,536 rows: the states merge across windows
    check_all(t, cut, "table in windows")
    check_plain(t, cut, nreg, "table in windows")


def test_one_row_no_rows_all_null() raises:
    var t = Table.empty()
    var u = t.add_column("cust", INT)
    var k = t.add_column("k", INT)
    t.append(u, 5)
    t.append(k, 0)
    t.append_null(u)
    t.append(k, 0)
    t.append(u, 5)
    t.append(k, 1)
    var s = t^.share()
    var r = Query(s).group_by(col("k"), groups=3).agg(count_distinct(col("cust")), median(col("cust")), top_k(col("cust"), 4)).compute()
    expect(r.values(0)[0] == 1 and r.values(1)[0] == 50 and len(r.top(2, 0)) == 1 and r.top(2, 0)[0] == 5, "group 0: one value among a null")
    expect(r.values(0)[1] == 1 and r.values(1)[1] == 50, "group 1")
    expect(r.values(0)[2] == 0 and not r.present(0, 2) and not r.present(1, 2) and len(r.top(2, 2)) == 0, "an empty group has no value")
    var e = Table.empty()
    _ = e.add_column("cust", INT)
    var empty = e^.share()
    var r0 = Query(empty).agg(count_distinct(col("cust")), median(col("cust")), top_k(col("cust"), 2)).compute()
    expect(r0.values(0)[0] == 0 and not r0.present(1, 0) and len(r0.top(2, 0)) == 0, "no rows")


def test_median_is_the_mean_of_the_middle_values_percentile_the_lower() raises:
    var t = Table.empty()
    var v = t.add_column("v", INT)
    var w = t.add_column("w", DECIMAL, scale=2)
    for x in [4, 1, 3, 2]:
        t.append(v, Int64(x))
        t.append(w, Int64(x * 100 + 1))
    var s = t^.share()
    var r = Query(s).agg(median(col("v")), percentile(col("v"), 0.5), percentile(col("v"), 0.25), percentile(col("v"), 0.26),
                         percentile(col("v"), 1.0), median(col("w"))).compute()
    expect(r.values(0)[0] == 25, "median of 1 2 3 4 is 2.5, returned at one more decimal place: 25, got " + String(r.values(0)[0]))
    expect(r.values(1)[0] == 2, "percentile(v, 0.5) is DuckDB's quantile_disc: the lower middle value, 2")
    expect(r.values(2)[0] == 1 and r.values(3)[0] == 2 and r.values(4)[0] == 4, "ceil(p * n) - 1: p25 of 4 values is the 1st, p26 the 2nd")
    expect(r.values(5)[0] == 2510, "median of 1.01 2.01 3.01 4.01 (scale 2) is 2.51 at scale 3: 2510, got " + String(r.values(5)[0]))
    var odd = Table.empty()
    var o = odd.add_column("v", INT)
    for x in [9, 1, 5]:
        odd.append(o, Int64(x))
    var ro = Query(odd^.share()).agg(median(col("v"))).compute()
    expect(ro.values(0)[0] == 50, "median of 1 5 9 is 5, returned as 50")


def test_a_median_too_large_for_one_more_decimal_place_raises() raises:
    var t = Table.empty()
    var v = t.add_column("v", INT)
    t.append(v, Int64.MAX - 5)
    t.append(v, Int64.MAX - 7)
    var msg = String("")
    try:
        _ = Query(t^.share()).agg(median(col("v"))).compute()
    except e:
        msg = String(e)
    expect("median" in msg and "too large" in msg, "an overflowing median says so: " + msg)
    for v in [Int64(-922337203685477581), Int64(-922337203685477582)]:   # one step past the negative limit
        var n = Table.empty()
        var c = n.add_column("v", INT)
        n.append(c, v)
        var nmsg = String("")
        try:
            var r = Query(n^.share()).agg(median(col("v"))).compute()
            nmsg = "returned " + String(r.values(0)[0])
        except e:
            nmsg = String(e)
        expect("too large" in nmsg, "a negative median past the limit raises, never wraps: " + nmsg)
    var edge = Table.empty()
    var ec = edge.add_column("v", INT)
    edge.append(ec, Int64(-922337203685477580))
    var at = Query(edge^.share()).agg(median(col("v"))).compute()
    expect(Int64(at.values(0)[0]) == Int64(-9223372036854775800), "the negative limit itself is exact")


def test_a_huge_k_is_clamped_or_refused_never_wrapped() raises:
    """k * groups used to overflow Int: top_k(x, 2^61) over 6 groups passed the size guard and read another group's heap."""
    var t = make_table(5_000, 8)
    var huge = 1 << 61
    var r = Query(t).group_by(col("carrier"), groups=6).agg(top_k(col("cost"), huge), count()).compute()
    var lists = gather(t, K_CARRIER, 6, False, V_COST)
    check_agg(r, 0, "top", 0.0, huge, lists, no_rank(), "huge k over a table")
    var f = Frame(t)
    var msg = String("")
    try:
        _ = Query(f).group_by(col("carrier"), groups=6).agg(top_k(col("cost"), huge)).compute()
    except e:
        msg = String(e)
    expect("top_k" in msg and "would hold more than" in msg, "a frame cannot clamp k to its rows: " + msg)
    msg = ""
    try:
        _ = Query(t).group_by(col("carrier"), groups=6).agg(top_k(col("cost"), 0)).compute()
    except e:
        msg = String(e)
    expect("k must be at least 1" in msg, "k = 0: " + msg)


def test_top_is_empty_for_other_aggregates() raises:
    var t = make_table(2_000, 9)
    var r = Query(t).group_by(col("carrier"), groups=7).agg(count_distinct(col("cust")), sum(col("day")), top_k(col("cost"), 2)).compute()
    expect(len(r.top(0, 0)) == 0 and len(r.top(1, 0)) == 0 and len(r.top(2, 0)) == 2, "top() is for top_k aggregates only")


def test_errors_are_readable() raises:
    var t = make_table(1000, 2)
    var msg = String("")
    try:
        _ = Query(t).agg(percentile(col("cost"), 1.5)).compute()
    except e:
        msg = String(e)
    expect("between 0 and 1" in msg, "p out of range: " + msg)
    msg = ""
    try:
        _ = Query(t).agg(top_k(col("cost"), 0)).compute()
    except e:
        msg = String(e)
    expect("k must be at least 1" in msg, "k = 0: " + msg)
    msg = ""
    try:
        _ = Query(t).agg(count_distinct(col("nope"))).compute()
    except e:
        msg = String(e)
    expect(msg == "column `nope` not found", "unknown column: " + msg)
    msg = ""
    try:
        _ = Query(t).agg(median(col("cost") > 0)).compute()
    except e:
        msg = String(e)
    expect("not a comparison" in msg, "a comparison as a value: " + msg)
    msg = ""
    try:
        _ = Query(t).agg(median(col("region") + 1)).compute()
    except e:
        msg = String(e)
    expect("needs the column itself" in msg, "text in an expression: " + msg)
    msg = ""
    try:
        _ = Query(t).agg(median(col("cost"))).compute(GPU)
    except e:
        msg = String(e)
    expect("CPU" in msg, "GPU: " + msg)
    var r = Query(t).agg(median(col("cost"))).compute(AUTO)
    expect(r.device == CPU, "AUTO runs it on the CPU")


def test_a_group_key_out_of_range_and_the_same_value_twice() raises:
    """Rows whose group is outside 0 .. groups-1 are left out; aggregates of one expression share its pairs."""
    var t = make_table(20_000, 3)
    var r = Query(t).group_by(col("carrier"), groups=4).agg(count_distinct(col("cust")), median(col("cust")), top_k(col("cust"), 2),
                                                            percentile(col("cust"), 0.9)).compute()
    var lists = gather(t, K_CARRIER, 4, False, V_CUST)
    check_agg(r, 0, "distinct", 0.0, 0, lists, no_rank(), "groups=4 distinct")
    check_agg(r, 1, "median", 0.5, 0, lists, no_rank(), "groups=4 median")
    check_agg(r, 2, "top", 0.0, 2, lists, no_rank(), "groups=4 top2")
    check_agg(r, 3, "pct", 0.9, 0, lists, no_rank(), "groups=4 p90")


# ── a frame with edits ───────────────────────────────────────────────────────────────────────────────

def test_over_a_frame_with_edits() raises:
    var t = make_table(60_000, 4)
    var f = Frame(t)
    var rng = Rng(9)
    for step in range(500):
        var row = rng.int(0, f.rows() - 1)
        while not f.alive(row):
            row = rng.int(0, f.rows() - 1)
        var what = rng.int(0, 11)
        if what <= 2:
            f.set(f.column("cust"), row, Int64(rng.int(0, 4000)))                   # past the shared range: new values
        elif what == 3:
            f.set(f.column("revenue"), row, Int64(rng.int(-700, 700)))
        elif what == 4:
            f.set_null(f.column("cust"), row)
        elif what == 5:
            f.set_text(f.column("region"), row, "Zulu" if step % 3 == 0 else "apple" if step % 3 == 1 else "East")   # new words, an old one
        elif what == 6:
            _ = f.append(["East", "3", "77", "1.50", "9", "100"])
        elif what == 7:
            _ = f.append(["Zulu", "", "3001", "", "-1", "50"])
        elif what == 8:
            f.remove(row)
        elif what == 9:
            f.set_null(f.column("carrier"), row)
        elif what == 10:
            f.set(f.column("cost"), row, Int64(rng.int(-90, 90)))
        else:
            f.set_null(f.column("revenue"), row)
    _ = f.merge(f.column("region"), [Int64(2)], Int64(1))                          # "north" folds into "East"
    var nreg = f.word_count(f.column("region"))
    var baked = f.bake().share()
    var got = List[QueryResult]()
    for s in range(6):
        got.append(plan(Query(f), s, nreg).compute())
    check_all(baked, got, "frame")
    check_plain(baked, got, nreg, "frame")
    var shared = table_results(t, 5, 0)                                            # the edits matter
    var differs = False
    for g in range(7):
        differs = differs or shared[0].values(2)[g] != got[0].values(2)[g] or shared[0].values(3)[g] != got[0].values(3)[g]
    expect(differs, "the frame's answers differ from the shared table's")
    while f.undo():                                                                # undoing every edit: the shared answers
        pass
    var back = List[QueryResult]()
    for s in range(6):
        back.append(plan(Query(f), s, 5).compute())
    check_all(t, back, "frame after undo")


# ── a streamed scan ──────────────────────────────────────────────────────────────────────────────────

def test_over_a_scan_of_several_row_groups() raises:
    var mem = make_table(N, 5)
    var path = String("build/mojo-tests/f148_scan.parquet")
    write_parquet(path, mem, "{}", ZSTD, GROUP)
    var scan = scan_parquet(path)
    expect(scan.row_groups() == 8, "the file has " + String(scan.row_groups()) + " row groups")
    var nreg = scan.word_count(scan.index("region"))
    var rp = read_parquet(path)
    var whole = rp.take_table().share()                          # the oracle table: the file read whole (the scan shares its codes)
    expect(len(whole.dictionary(whole.index("region"))) == nreg, "same words")
    var got = List[QueryResult]()
    for s in range(6):
        got.append(plan(Query(scan), s, nreg).compute())
    check_all(whole, got, "scan")
    check_plain(whole, got, nreg, "scan")
    var mm = table_results(whole, nreg, 0)
    for s in range(6):
        for i in range(len(got[s].kinds)):
            if got[s].hol_of[i] >= 0:
                for g in range(len(got[s].counts)):
                    expect(got[s].values(i)[g] == mm[s].values(i)[g] and got[s].present(i, g) == mm[s].present(i, g),
                           "scan equals the in-memory run: scenario " + String(s) + " aggregate " + String(i) + " group " + String(g))


def words_by_k(t: SharedTable) -> List[List[Int64]]:
    """The `w` codes of each k group (k in 0..2)."""
    var out = List[List[Int64]]()
    for _ in range(3):
        out.append(List[Int64]())
    var k = t.index("k")
    var w = t.index("w")
    for row in range(t.rows()):
        if not t.is_null(w, row):
            out[Int(t.get(k, row))].append(t.get(w, row))
    return out^


def test_a_scan_with_an_all_null_row_group_and_words_that_come_late() raises:
    var t = Table.empty()
    var w = t.add_column("w", TEXT)
    var v = t.add_column("v", INT)
    var k = t.add_column("k", INT)
    var rng = Rng(21)
    var words: List[String] = ["pear", "Apple", "fig", "Zebra", "kiwi", "banana"]
    for i in range(6000):
        var rg = i // 1000
        t.append_text(w, words[min(rg, 5) if rng.int(0, 3) == 0 else rng.int(0, min(rg, 5))])        # later groups bring new words
        if rg == 2:
            t.append_null(v)
        else:
            t.append(v, Int64(rng.int(-20, 20)))
        t.append(k, Int64(rng.int(0, 2)))
    var mem = t^.share()
    var path = String("build/mojo-tests/f148_scan2.parquet")
    write_parquet(path, mem, "", ZSTD, 1000)
    var scan = scan_parquet(path)
    var rp = read_parquet(path)
    var whole = rp.take_table().share()
    var a = Query(scan).group_by(col("k"), groups=3).agg(median(col("w")), top_k(col("w"), 3), count_distinct(col("w")),
                                                         count_distinct(col("v")), percentile(col("v"), 0.75), top_k(col("v"), 2)).compute()
    var b = Query(whole).group_by(col("k"), groups=3).agg(median(col("w")), top_k(col("w"), 3), count_distinct(col("w")),
                                                          count_distinct(col("v")), percentile(col("v"), 0.75), top_k(col("v"), 2)).compute()
    for g in range(3):
        for i in range(6):
            expect(a.present(i, g) == b.present(i, g), "present " + String(i) + " " + String(g))
            if i == 1 or i == 5:
                var x = a.top(i, g)
                var y = b.top(i, g)
                expect(len(x) == len(y), "top size")
                for z in range(len(x)):
                    expect(x[z] == y[z], "top value")
            else:
                expect(a.values(i)[g] == b.values(i)[g], "value " + String(i) + " group " + String(g))
    var ranks = text_ranks_of(whole, whole.index("w"))                               # the words' order, from the oracle
    var lists = words_by_k(whole)
    check_agg(a, 0, "pct", 0.5, 0, lists, ranks, "late words median")
    check_agg(a, 1, "top", 0.0, 3, lists, ranks, "late words top3")


def test_a_device_table_does_not_move_them_to_the_gpu() raises:
    """AUTO with a DeviceTable attached and enough rows would use the GPU for plain aggregates; these run on the CPU
    and say why. compute(GPU) refuses. (Skipped without a GPU.)"""
    if not gpu_available():
        return
    var t = make_table(1_000_000, 6)
    var dt = DeviceTable(t)
    var r = Query(t).on_device(dt).group_by(col("carrier"), groups=7).agg(median(col("cost")), count_distinct(col("cust")), sum(col("day"))).compute(AUTO)
    expect(r.device == CPU and "CPU" in r.fallback_reason, "AUTO ran on the CPU and said why: " + r.fallback_reason)
    var plain = Query(t).group_by(col("carrier"), groups=7).agg(median(col("cost")), count_distinct(col("cust")), sum(col("day"))).compute(CPU)
    for g in range(7):
        expect(r.values(0)[g] == plain.values(0)[g] and r.values(1)[g] == plain.values(1)[g] and r.values(2)[g] == plain.values(2)[g],
               "the same answers as the CPU run, group " + String(g))
    var msg = String("")
    try:
        _ = Query(t).on_device(dt).agg(median(col("cost"))).compute(GPU)
    except e:
        msg = String(e)
    expect("run on the CPU" in msg, "compute(GPU): " + msg)
    dt.release()


def main() raises:
    test_over_a_table()
    test_one_row_no_rows_all_null()
    test_median_is_the_mean_of_the_middle_values_percentile_the_lower()
    test_a_median_too_large_for_one_more_decimal_place_raises()
    test_a_huge_k_is_clamped_or_refused_never_wrapped()
    test_top_is_empty_for_other_aggregates()
    test_errors_are_readable()
    test_a_group_key_out_of_range_and_the_same_value_twice()
    test_over_a_frame_with_edits()
    test_over_a_scan_of_several_row_groups()
    test_a_scan_with_an_all_null_row_group_and_words_that_come_late()
    test_a_device_table_does_not_move_them_to_the_gpu()
    print("ok")
