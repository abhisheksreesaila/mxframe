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
    # a side brought to the other's scale (qty * 100) must still pass the query compiler's scale check
    _ = f.formula("plain = revenue + qty")
    _ = f.formula("less = cost - qty")
    _ = f.formula("top = max(revenue, qty)")
    var aligned = Query(f).agg(sum(col("plain")), sum(col("big")), sum(col("less")), sum(col("top"))).compute()
    expect(aligned.values(0)[0] == 1550, "12.50 + 3 = 15.50, got " + String(aligned.values(0)[0]))
    expect(aligned.values(1)[0] == 200, "big in a query, got " + String(aligned.values(1)[0]))
    expect(aligned.values(2)[0] == 75, "3.75 - 3 = 0.75, got " + String(aligned.values(2)[0]))
    expect(aligned.values(3)[0] == 1250, "max(12.50, 3), got " + String(aligned.values(3)[0]))
    var scaled = Query(f).where(col("plain") > 1500).agg(count()).compute()
    expect(scaled.counts[0] == 1, "a filter over an aligned formula")


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


def test_a_refused_redefinition_leaves_the_frame_as_it_was() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    t.append(a, 4)
    var f = Frame(t^.share())
    var p = f.formula("p = a + 1")
    var q = f.formula("q = p * 2")
    var edits = f.edits
    var refused = False
    try:
        _ = f.formula("p = q + 1")
    except e:
        refused = "cycle" in String(e)
    expect(refused and f.edits == edits and f.columns() == 3, "the cycle is refused and nothing moves")
    expect(f.formula_text(p) == "p = a + 1" and f.get(p, 0) == 5 and f.get(q, 0) == 10, "p and q read as before")
    var sums = Query(f).agg(sum(col("p")), sum(col("q"))).compute()
    expect(sums.values(0)[0] == 5 and sums.values(1)[0] == 10, "queries over p and q still run")
    var other = False
    try:
        _ = f.formula("p = nope + 1")
    except e:
        other = "not found" in String(e)
    expect(other and f.get(p, 0) == 5, "an unknown column leaves p alone too")


def test_redefining_a_formula_refreshes_what_reads_it() raises:
    var t = Table.empty()
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var q = t.add_column("qty", INT)
    t.append(r, 1250)
    t.append(q, 3)
    var f = Frame(t^.share())
    var a = f.formula("a = revenue")
    var b = f.formula("b = a * qty")
    var c = f.formula("c = b + a")
    expect(f.scale(b) == 2 and f.get(b, 0) == 3750 and f.get(c, 0) == 5000, "before: 37.50 and 50.00")
    _ = f.formula("a = qty")
    expect(f.scale(a) == 0 and f.scale(b) == 0 and f.scale(c) == 0, "scales follow the new definition")
    expect(f.get(b, 0) == 9 and f.get(c, 0) == 12, "3 * 3 and 9 + 3, got " + String(f.get(b, 0)) + " " + String(f.get(c, 0)))
    expect(f.formula_text(b) == "b = a * qty", "the text is kept")
    var s = Query(f).agg(sum(col("b")), sum(col("c"))).compute()
    expect(s.values(0)[0] == 9 and s.values(1)[0] == 12, "queries use the new definition")


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


def test_text_sorts_ignoring_case_with_bytes_as_the_tie_break() raises:
    var t = Table.empty()
    var g = t.add_column("name", TEXT)
    var names: List[String] = ["Zebra", "apple", "banana", "Apple", "zebra", "Banana", "ab", "AB", "a"]
    for n in names:
        t.append_text(g, n)
    var f = Frame(t^.share())
    var order = f.permutation(0)
    var want: List[String] = ["a", "AB", "ab", "Apple", "apple", "Banana", "banana", "Zebra", "zebra"]
    for i in range(len(want)):
        expect(names[order[i]] == want[i], "at " + String(i) + ": " + names[order[i]] + " instead of " + want[i])


def test_edited_marks_session_edits_that_survive_undo() raises:
    var t = Table.empty()
    var g = t.add_column("name", TEXT)
    var v = t.add_column("v", INT)
    for w in ["a", "b", "c"]:
        t.append_text(g, w)
        t.append(v, 1)
    var f = Frame(t^.share())
    expect(not f.edited(1, 0) and not f.edited(0, 0), "a fresh frame has no edited cells")
    f.set(1, 0, 5)
    f.set_null(1, 1)
    f.set_text(0, 2, "zed")
    expect(f.edited(1, 0) and f.edited(1, 1) and f.edited(0, 2), "set, set_null and set_text mark the cell")
    expect(not f.edited(0, 0) and not f.edited(1, 2), "other cells are not marked")
    f.set(1, 0, 6)                                           # edited twice
    _ = f.undo()
    expect(f.edited(1, 0), "undoing the second edit leaves the first in place")
    _ = f.undo()
    _ = f.undo()
    _ = f.undo()
    _ = f.undo()
    expect(not f.edited(1, 0) and not f.edited(1, 1) and not f.edited(0, 2), "undone edits are not marked")
    expect(f.word_count(0) == 4, "3 shared + \"zed\" (a word stays after the undo): " + String(f.word_count(0)))
    f.set_text(0, 0, "x")
    f.set_text(0, 1, "y")
    f.set_text(0, 2, "x")
    expect(f.word_count(0) == 6, "3 shared + zed + x + y: " + String(f.word_count(0)))
    expect(f.word_count(1) == 0, "an INT column has no words")


# The old sort, kept as the oracle: a stable merge sort of the live rows over what `get` and `is_null` report.
def ref_less(f: Frame, c: Int, a: Int, b: Int, descending: Bool) raises -> Bool:
    var na = f.is_null(c, a)
    var nb = f.is_null(c, b)
    if na or nb:
        return not na and nb
    var x = f.get(c, a)
    var y = f.get(c, b)
    if c < f.stored() and f.kind(c) == TEXT:
        var sx = f.text_of(c, x).as_bytes()
        var sy = f.text_of(c, y).as_bytes()
        var d = 0
        for i in range(min(len(sx), len(sy))):
            var lx = sx[i] + 32 if UInt8(65) <= sx[i] <= UInt8(90) else sx[i]
            var ly = sy[i] + 32 if UInt8(65) <= sy[i] <= UInt8(90) else sy[i]
            if lx != ly:
                d = -1 if lx < ly else 1
                break
        if d == 0 and len(sx) != len(sy):
            d = -1 if len(sx) < len(sy) else 1
        if d == 0:
            for i in range(len(sx)):
                if sx[i] != sy[i]:
                    d = -1 if sx[i] < sy[i] else 1
                    break
        return d > 0 if descending else d < 0
    return x > y if descending else x < y


def ref_permutation(f: Frame, c: Int, descending: Bool) raises -> List[Int]:
    var cur = List[Int]()
    for row in range(f.rows()):
        if f.alive(row):
            cur.append(row)
    var n = len(cur)
    var width = 1
    while width < n:
        var nxt = List[Int]()
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            while i < mid and j < hi:
                if ref_less(f, c, cur[j], cur[i], descending):
                    nxt.append(cur[j])
                    j += 1
                else:
                    nxt.append(cur[i])
                    i += 1
            while i < mid:
                nxt.append(cur[i])
                i += 1
            while j < hi:
                nxt.append(cur[j])
                j += 1
            lo += 2 * width
        cur = nxt^
        width *= 2
    return cur^


def check_sorts(f: Frame, label: String) raises:
    for c in range(f.columns()):
        for d in range(2):
            var got = f.permutation(c, descending=(d == 1))
            var want = ref_permutation(f, c, descending=(d == 1))
            expect(len(got) == len(want), label + ": length, column " + String(c))
            for i in range(len(want)):
                expect(got[i] == want[i], label + ": column " + String(c) + (" desc" if d == 1 else " asc") +
                       " at " + String(i) + ": " + String(got[i]) + " instead of " + String(want[i]))


def extremes_table(n: Int, seed: UInt64) raises -> SharedTable:
    var t = Table.empty()
    var g = t.add_column("name", TEXT)
    var v = t.add_column("v", INT)
    var d = t.add_column("d", DECIMAL, scale=2)
    var words: List[String] = ["pear", "Apple", "apple", "fig", "", "Zed", "zed", "figs", "FIG", "\xc3\xa9a"]
    for w in words:
        _ = t.add_word(g, w)
    var big: List[Int64] = [Int64.MIN, Int64.MIN + 2, -1, 0, 1, Int64.MAX, Int64.MAX - 1, -7, 7, 3]
    var rng = Rng(seed)
    for i in range(n):
        if rng.int(0, 6) == 0:
            t.append_null(g)
        else:
            t.append(g, Int64(rng.int(0, 9)))
        if rng.int(0, 6) == 0:
            t.append_null(v)
        else:
            t.append(v, big[rng.int(0, 9)] if rng.int(0, 2) > 0 else Int64(rng.int(-5, 5)))
        t.append(d, Int64(rng.int(-300, 300)))
    return t^.share()


def test_permutation_equals_the_merge_sort_on_random_frames() raises:
    for seed in range(1, 9):
        var f = Frame(extremes_table(40 + seed * 37, UInt64(seed)))
        check_sorts(f, "fresh " + String(seed))                    # the no-edit fast path
        var rng = Rng(UInt64(seed) * 31)
        for step in range(60):
            var row = rng.int(0, f.rows() - 1)
            while not f.alive(row):
                row = rng.int(0, f.rows() - 1)
            var what = rng.int(0, 7)
            if what == 0:
                f.set(1, row, Int64.MIN if step % 2 == 0 else Int64.MAX)
            elif what == 1:
                f.set_null(1, row)
            elif what == 2:
                f.set_text(0, row, "new" + String(step % 3) if step % 4 != 0 else "APPLE")
            elif what == 3:
                _ = f.append(["fig", "-4", "1.50"])
            elif what == 4:
                _ = f.append(["", "", ""])
            elif what == 5:
                f.remove(row)
            elif what == 6:
                f.set_null(0, row)
            else:
                f.set(2, row, Int64(rng.int(-300, 300)))
        _ = f.merge(0, [Int64(2), Int64(8)], Int64(1))           # apple and FIG fold into Apple
        _ = f.formula("m = d * 3 - 100")
        _ = f.formula("q = m // 4")
        check_sorts(f, "edited " + String(seed))
        _ = f.undo()
        check_sorts(f, "after an undo " + String(seed))
        var empty = Frame(extremes_table(0, 1))
        check_sorts(empty, "empty")


def test_the_null_marker_cannot_be_typed() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    var d = t.add_column("d", DECIMAL, scale=2)
    t.append(a, 1)
    t.append(d, 100)
    var f = Frame(t^.share())
    var msgs = List[String]()
    try:
        f.set_text(a, 0, "9221120237041090561")
    except e:
        msgs.append(String(e))
    try:
        f.set_text(d, 0, "92211202370410905.61")
    except e:
        msgs.append(String(e))
    try:
        _ = f.append(["9221120237041090561", ""])
    except e:
        msgs.append(String(e))
    try:
        _ = f.append(["1", "92211202370410905.61"])
    except e:
        msgs.append(String(e))
    expect(len(msgs) == 4 and "reserved" in msgs[0] and "reserved" in msgs[1] and "reserved" in msgs[2] and "reserved" in msgs[3],
           "four refusals")
    expect(f.rows() == 1 and f.get(a, 0) == 1 and not f.is_null(a, 0) and f.get(d, 0) == 100 and len(f.log) == 0,
           "nothing was entered")
    f.set_text(a, 0, "9221120237041090560")
    expect(f.get(a, 0) == 9221120237041090560, "the next value up is fine")


def test_a_dead_row_does_not_leave_a_word_behind() raises:
    var t = Table.empty()
    var g = t.add_column("g", TEXT)
    t.append_text(g, "x")
    var f = Frame(t^.share())
    f.remove(0)
    var refused = False
    try:
        f.set_text(g, 0, "new")
    except e:
        refused = "does not exist" in String(e)
    expect(refused and f.code_of(g, "new") == -1, "the word was not added")


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


def words_table(n: Int, words: Int) raises -> SharedTable:
    var t = Table.empty()
    var g = t.add_column("name", TEXT)
    var v = t.add_column("v", INT)
    var d = t.add_column("cost", DECIMAL, scale=2)
    for w in range(words):
        _ = t.add_word(g, "w" + String(w))
    var rng = Rng(3)
    for _ in range(n):
        t.append(g, Int64(rng.int(0, words - 1)))
        t.append(v, Int64(rng.int(1, 50)))
        t.append(d, Int64(rng.int(0, 100_00)))
    return t^.share()


def test_a_merge_of_many_values_queries_like_its_baked_table() raises:
    var t = words_table(3000, 14)
    var f = Frame(t)
    var codes: List[Int64] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expect(f.merge(0, codes, 0) == 9, "nine values merged")
    var again: List[Int64] = [0]
    expect(f.merge(0, again, 12) == 1, "a merge that repoints the earlier ones")
    var more: List[Int64] = [13, 10]
    _ = f.merge(0, more, 11)
    var baked = f.bake().share()
    var got = Query(f).group_by(col("name"), groups=14).agg(sum(col("v")), count(), max_of(col("cost"))).compute()
    var want = Query(baked).group_by(col("name"), groups=14).agg(sum(col("v")), count(), max_of(col("cost"))).compute()
    same(got, want, "group by a many-times merged column")
    expect(got.counts[12] > 0 and got.counts[0] == 0 and got.counts[5] == 0 and got.counts[13] == 0, "everything folded")
    var g2 = Query(f).where(col("name") == 12).agg(sum(col("v")), count()).compute()
    var w2 = Query(baked).where(col("name") == 12).agg(sum(col("v")), count()).compute()
    same(g2, w2, "where on a merged column")
    expect(g2.counts[0] > 0, "the filter finds the folded rows")


def test_where_on_a_formula_equals_the_baked_table() raises:
    var t = base_table(4000)
    var f = Frame(t)
    f.set(f.column("revenue"), 7, 9_000_00)
    _ = f.append(["East", "3", "12.50", "1.00", "4"])
    f.remove(9)
    _ = f.formula("margin = revenue - cost")
    var got = Query(f).where(col("margin") > 100000).group_by(col("region"), groups=8).agg(sum(col("qty")), count()).compute()
    var baked = f.bake().share()
    var want = Query(baked).where(col("revenue") - col("cost") > 100000).group_by(col("region"), groups=8).agg(
        sum(col("qty")), count()).compute()
    same(got, want, "where on a formula")
    var total = 0
    for g in range(8):
        total += got.counts[g]
    expect(total > 0, "some rows pass")


def test_a_group_whose_shared_rows_are_all_edited() raises:
    var t = Table.empty()
    var g = t.add_column("g", INT)
    var v = t.add_column("v", INT)
    for i in range(6):
        t.append(g, Int64(i % 2))
        t.append(v, Int64(10 + i))
    var f = Frame(t^.share())
    for row in range(6):
        if row % 2 == 1:
            f.set(v, row, Int64(100 * (row + 1)))      # group 1: 200, 400, 600, all edited
    var r = Query(f).group_by(col("g"), groups=2).agg(min_of(col("v")), max_of(col("v")), sum(col("v")), count()).compute()
    expect(r.values(0)[0] == 10 and r.values(1)[0] == 14 and r.counts[0] == 3, "group 0 is untouched")
    expect(r.values(0)[1] == 200 and r.values(1)[1] == 600 and r.values(2)[1] == 1200 and r.counts[1] == 3,
           "group 1: min 200 max 600, got " + String(r.values(0)[1]) + " " + String(r.values(1)[1]))
    var want = Query(f.bake().share()).group_by(col("g"), groups=2).agg(
        min_of(col("v")), max_of(col("v")), sum(col("v")), count()).compute()
    same(r, want, "baked")


def test_a_frame_with_only_appended_rows() raises:
    var t = base_table(5000)
    var f = Frame(t)
    _ = f.append(["East", "3", "12.50", "1.00", "4"])
    _ = f.append(["West", "1", "", "2.00", "5"])
    expect(len(f.overlay().mask) == 0, "no mask when no shared row changed")
    var got = queries_on_frame(f)
    var want = queries_on_table(f.bake().share())
    for i in range(len(got)):
        same(got[i], want[i], "appended-only query " + String(i))
    f.remove(1)                                            # a shared row now: the mask appears
    expect(len(f.overlay().mask) > 0, "a mask once a shared row changes")


def test_a_merge_never_touches_a_number_column() raises:
    var t = Table.empty()
    _ = t.add_column("big", INT)
    _ = t.add_column("word", TEXT)
    var collides = (Int64(1) << 32) | 2                    # the merge key of (column 1, code 2), held as a number
    t.append(0, collides)
    t.append_text(1, "a")
    t.append(0, Int64(7))
    t.append_text(1, "b")
    t.append(0, Int64(1))
    t.append_text(1, "c")
    var f = Frame(t^.share())
    _ = f.merge(1, [Int64(2)], Int64(0))                   # "c" folds into "a"
    expect(f.get(0, 0) == collides, "a number is a value, never a merged code: " + String(f.get(0, 0)))
    expect(f.bake().get(0, 0) == collides, "and bakes as itself")
    var p = f.permutation(0, False)
    expect(p[0] == 2 and p[1] == 1 and p[2] == 0, "sort and get agree")
    var r = Query(f).agg(max_of(col("big"))).compute()
    expect(Int64(r.values(0)[0]) == collides, "and a query reads it too")


def test_a_group_nulled_only_by_edits_is_not_present() raises:
    var t = Table.empty()
    var g = t.add_column("g", INT)
    var v = t.add_column("v", INT)
    t.append(g, 0)
    t.append(v, 5)
    t.append(g, 1)
    t.append(v, 6)
    t.append(g, 1)
    t.append(v, 7)
    var f = Frame(t^.share())
    f.set_null(v, 1)
    f.set_null(v, 2)
    var r = Query(f).group_by(col("g"), groups=2).agg(sum(col("v")), min_of(col("v")), count(), count(col("v"))).compute()
    expect(r.present(0, 0) and r.values(0)[0] == 5 and r.counts[0] == 1, "group 0")
    expect(not r.present(0, 1) and not r.present(1, 1), "group 1 has rows but every value is null")
    expect(r.counts[1] == 2, "its rows still count")


def main() raises:
    test_a_merge_never_touches_a_number_column()
    test_a_refused_redefinition_leaves_the_frame_as_it_was()
    test_redefining_a_formula_refreshes_what_reads_it()
    test_text_sorts_ignoring_case_with_bytes_as_the_tie_break()
    test_the_null_marker_cannot_be_typed()
    test_a_dead_row_does_not_leave_a_word_behind()
    test_a_merge_of_many_values_queries_like_its_baked_table()
    test_where_on_a_formula_equals_the_baked_table()
    test_a_group_whose_shared_rows_are_all_edited()
    test_a_frame_with_only_appended_rows()
    test_a_group_nulled_only_by_edits_is_not_present()
    test_formulas_follow_the_apps_decimal_rules()
    test_formula_errors_are_readable()
    test_permutation_is_stable_alphabetical_and_sees_edits()
    test_edits_check_their_values()
    test_edited_marks_session_edits_that_survive_undo()
    test_permutation_equals_the_merge_sort_on_random_frames()
    test_undo_returns_to_the_shared_results()
    test_queries_over_edits_equal_queries_over_the_baked_table()
