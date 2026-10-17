# F-136 (M5): a session's edits, merges, formulas and sort; queries over a Frame include them. The main oracle: a
# query over the frame equals the same query over `f.bake()` (the edits applied into a plain table), which runs
# the kernel alone; undo returns to the shared results exactly.
from mxframe import (Table, SharedTable, Frame, INT, DECIMAL, TEXT, DATE, BOOL, Query, QueryResult, Expr, col, least,
                     greatest, sum, count, mean, min_of, max_of)


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def int(mut self, lo: Int, hi: Int) -> Int:
        return lo + Int(self.next() % UInt64(hi - lo + 1))


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def base_table(n: Int) raises -> SharedTable:
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var c = t.add_column("carrier", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", DECIMAL, scale=2)
    var q = t.add_column("qty", INT)
    var words: List[String] = ["West", "East", "west", "North", "South"]
    for w in words:
        _ = t.add_word(g, w)
    var rng = Rng(5)
    for i in range(n):
        t.append(g, Int64(rng.int(0, 4)))
        t.append(c, Int64(rng.int(0, 6)))
        if i % 17 == 3:
            t.append_null(r)
        else:
            t.append(r, Int64(rng.int(-500_00, 2_000_00)))
        t.append(k, Int64(rng.int(0, 900_00)))
        t.append(q, Int64(rng.int(1, 9)))
    return t^.share()


def run(q: Query) raises -> QueryResult:
    return q.compute()


def queries_on_frame(f: Frame) raises -> List[QueryResult]:
    var out = List[QueryResult]()
    out.append(Query(f).group_by(col("region"), groups=8).agg(sum(col("revenue")), count(), count(col("revenue")),
                                                                min_of(col("revenue")), max_of(col("cost"))).compute())
    out.append(Query(f).where(col("carrier").between(1, 4)).group_by(col("region") * 7 + col("carrier"), groups=56)
               .agg(sum(col("qty")), mean(col("revenue"))).compute())
    out.append(Query(f).where(col("region") == 0).agg(sum(col("revenue") - col("cost")), count()).compute())
    return out^


def queries_on_table(t: SharedTable) raises -> List[QueryResult]:
    var out = List[QueryResult]()
    out.append(Query(t).group_by(col("region"), groups=8).agg(sum(col("revenue")), count(), count(col("revenue")),
                                                                min_of(col("revenue")), max_of(col("cost"))).compute())
    out.append(Query(t).where(col("carrier").between(1, 4)).group_by(col("region") * 7 + col("carrier"), groups=56)
               .agg(sum(col("qty")), mean(col("revenue"))).compute())
    out.append(Query(t).where(col("region") == 0).agg(sum(col("revenue") - col("cost")), count()).compute())
    return out^


def same(a: QueryResult, b: QueryResult, label: String) raises:
    expect(len(a.counts) == len(b.counts), label + ": groups")
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


def test_queries_over_edits_equal_queries_over_the_baked_table() raises:
    var t = base_table(50_000)
    var f = Frame(t)
    var rng = Rng(11)
    for step in range(400):
        var row = rng.int(0, f.rows() - 1)
        while not f.alive(row):
            row = rng.int(0, f.rows() - 1)
        var what = rng.int(0, 9)
        if what <= 3:
            f.set(f.column("revenue"), row, Int64(rng.int(-1_000_00, 3_000_00)))
        elif what == 4:
            f.set_null(f.column("revenue"), row)
        elif what == 5:
            f.set_text(f.column("region"), row, "Central" if step % 2 == 0 else "West")   # a new word, an old one
        elif what == 6:
            _ = f.append(["East", "3", "12.50", "", "4"])
        elif what == 7:
            f.remove(row)
        elif what == 8:
            f.set_null(f.column("carrier"), row)
        else:
            f.set_text(f.column("cost"), row, "7.25")
    _ = f.merge(f.column("region"), [Int64(2)], Int64(0))     # "west" folds into "West"
    var got = queries_on_frame(f)
    var baked = f.bake().share()
    var want = queries_on_table(baked)
    for i in range(len(got)):
        same(got[i], want[i], "query " + String(i))
    expect(got[0].counts[2] == 0 and got[0].counts[0] > 0, "the merged spelling has no rows of its own")
    var untouched = queries_on_table(t)                      # the edits must matter, or this test proves nothing
    var differs = False
    for g in range(8):
        differs = differs or untouched[0].counts[g] != got[0].counts[g] or untouched[0].sums[g] != got[0].sums[g]
    expect(differs, "the session's results differ from the shared table's")


def test_undo_returns_to_the_shared_results() raises:
    var t = base_table(20_000)
    var before = queries_on_table(t)
    var f = Frame(t)
    f.set(f.column("revenue"), 5, 99_999_00)
    f.set_text(f.column("region"), 7, "Central")
    _ = f.append(["North", "2", "1.00", "2.00", "3"])
    f.remove(11)
    _ = f.merge(f.column("region"), [Int64(1), Int64(2)], Int64(0))
    f.set_null(f.column("cost"), 13)
    var edits = f.edits
    while f.undo():
        pass
    expect(f.edits > edits and len(f.changed_rows()) == 0 and f.rows() == t.rows(), "everything undone")
    var after = queries_on_frame(f)
    for i in range(len(after)):
        same(after[i], before[i], "undo query " + String(i))


def test_formulas_follow_the_apps_decimal_rules() raises:
    var t = Table.empty()
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var c = t.add_column("cost", DECIMAL, scale=2)
    var q = t.add_column("qty", INT)
    var rate = t.add_column("rate", DECIMAL, scale=4)
    t.append(r, 1250)        # 12.50
    t.append(c, 375)         # 3.75
    t.append(q, 3)
    t.append(rate, 1250)     # 0.1250
    var f = Frame(t^.share())
    var margin = f.formula("margin = revenue - cost")
    var total = f.formula("total = revenue * qty")
    var half = f.formula("half = revenue / 2")
    var taxed = f.formula("taxed = revenue * rate")           # scale 6 brought back to 2
    var bins = f.formula("bins = revenue // 5")
    var mixed = f.formula("mixed = margin + 1.5")             # a formula over a formula, a decimal literal
    var big = f.formula("big = max(cost, 4) - min(qty, 2)")
    expect(f.get(margin, 0) == 875 and f.scale(margin) == 2, "12.50 - 3.75 = 8.75")
    expect(f.get(total, 0) == 3750 and f.scale(total) == 2, "12.50 * 3 = 37.50")
    expect(f.get(half, 0) == 625 and f.scale(half) == 2, "12.50 / 2 = 6.25")
    expect(f.get(taxed, 0) == 156 and f.scale(taxed) == 2, "12.50 * 0.1250 = 1.5625 -> 1.56 (floor at cents)")
    expect(f.get(bins, 0) == 2 and f.scale(bins) == 0, "12.50 // 5 = 2")
    expect(f.get(mixed, 0) == 1025, "8.75 + 1.5 = 10.25")
    expect(f.get(big, 0) == 400 - 200, "max(3.75, 4) - min(3, 2) = 4.00 - 2.00")
    expect(f.depends(mixed, 0) and not f.depends(bins, 1), "dependencies through formulas")
    var deps = f.dependants(1)
    expect(len(deps) == 3, "cost is read by margin, mixed, big: got " + String(len(deps)))
    var sums = Query(f).agg(sum(col("mixed")), sum(col("taxed"))).compute()
    expect(sums.values(0)[0] == 1025 and sums.values(1)[0] == 156, "aggregations over formulas run in the kernel")


def test_formula_errors_are_readable() raises:
    var t = Table.empty()
    _ = t.add_column("a", INT)
    var f = Frame(t^.share())
    var msgs = List[String]()
    var bad: List[String] = ["no equals sign", "a = 1", "x = nope + 1", "x = (a + 1", "x = a +", "x = min(a)"]
    for b in bad:
        try:
            _ = f.formula(b)
        except e:
            msgs.append(String(e))
    expect(len(msgs) == 6, "six errors, got " + String(len(msgs)))
    expect("margin = revenue - cost" in msgs[0], msgs[0])
    expect("is a stored column" in msgs[1], msgs[1])
    expect("column `nope` not found" in msgs[2], msgs[2])
    expect("missing )" in msgs[3], msgs[3])
    expect("ends early" in msgs[4], msgs[4])
    expect("two values" in msgs[5], msgs[5])
    _ = f.formula("p = a + 1")
    _ = f.formula("q = p * 2")
    var cycle = False
    try:
        _ = f.formula("p = q + 1")                             # p -> q -> p
    except e:
        cycle = "cycle" in String(e)
    expect(cycle, "a formula cycle raises")


def test_permutation_is_stable_alphabetical_and_sees_edits() raises:
    var t = Table.empty()
    var g = t.add_column("name", TEXT)
    var v = t.add_column("v", INT)
    var names: List[String] = ["pear", "Apple", "fig", "pear", "apple", "fig"]
    var vals: List[Int] = [3, 1, 2, 3, 0, 9]
    for i in range(6):
        t.append_text(g, names[i])
        t.append(v, Int64(vals[i]))
    var f = Frame(t^.share())
    f.set_text(0, 5, "banana")                               # a new word sorts among the old ones
    f.set_null(1, 2)
    var by_name = f.permutation(0)
    var want: List[Int] = [1, 4, 5, 2, 0, 3]                 # Apple apple banana fig pear pear (bytes: upper first)
    for i in range(6):
        expect(by_name[i] == want[i], "by name at " + String(i) + ": " + String(by_name[i]))
    var by_v = f.permutation(1, descending=True)
    var want_v: List[Int] = [5, 0, 3, 1, 4, 2]               # 9 3 3 1 0 then the null; equal 3s keep row order
    for i in range(6):
        expect(by_v[i] == want_v[i], "by v desc at " + String(i) + ": " + String(by_v[i]))
    f.remove(0)
    expect(len(f.permutation(1)) == 5, "removed rows are not in the order")


def test_edits_check_their_values() raises:
    var t = Table.empty()
    var g = t.add_column("g", TEXT)
    var b = t.add_column("b", BOOL)
    t.append_text(g, "x")
    t.append(b, 1)
    var f = Frame(t^.share())
    var msgs = List[String]()
    try:
        f.set(0, 0, 5)
    except e:
        msgs.append(String(e))
    try:
        f.set(1, 0, 2)
    except e:
        msgs.append(String(e))
    try:
        f.set_text(1, 0, "maybe")
    except e:
        msgs.append(String(e))
    try:
        f.set(0, 9, 0)
    except e:
        msgs.append(String(e))
    expect(len(msgs) == 4, "four errors")
    expect("not one of its words" in msgs[0] and "not 0 or 1" in msgs[1] and "isn't true or false" in msgs[2] and
           "row 9 does not exist" in msgs[3], "messages")


def main() raises:
    test_formulas_follow_the_apps_decimal_rules()
    test_formula_errors_are_readable()
    test_permutation_is_stable_alphabetical_and_sees_edits()
    test_edits_check_their_values()
    test_undo_returns_to_the_shared_results()
    test_queries_over_edits_equal_queries_over_the_baked_table()
