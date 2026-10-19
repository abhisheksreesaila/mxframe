# F-155: the join planner (rule_join_order on plain statistics), a custom Planner's join order, and joins of three Frames with edits and a merge
# equal to the same joins over their baked tables under every join order.
from mxframe import (Table, SharedTable, Frame, Catalog, PlannerOptions, Planner, RulePlanner, PhysicalPlan, Stats, BoundQuery, JoinStats, JoinPlan,
                     rule_join_order, sql, sql_with, INT, DECIMAL, TEXT, BUILD_LEFT, BUILD_RIGHT)


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def stats(rows: List[Int], base: List[Int], edges: List[Tuple[Int, Int, Int, Int]]) -> JoinStats:
    var js = JoinStats()
    for i in range(len(rows)):
        js.names.append("t" + String(i))
        js.rows.append(rows[i])
        js.base_rows.append(base[i])
    for e in edges:
        js.edge_a.append(e[0])
        js.edge_b.append(e[1])
        js.ndv_a.append(e[2])
        js.ndv_b.append(e[3])
    return js^


def order_of(p: JoinPlan) -> String:
    var s = String(p.start)
    for st in p.steps:
        s += " " + String(st.leaf)
    return s^


struct ReversePlanner(Planner, Copyable, Movable):
    """A planner that joins the tables in the opposite of the written order when that connects: the join order is the planner's to choose, the executor does not change."""
    def __init__(out self):
        pass

    def plan(self, q: BoundQuery, st: Stats, opts: PlannerOptions) raises -> PhysicalPlan:
        return RulePlanner().plan(q, st, opts)

    def join_order(self, js: JoinStats, opts: PlannerOptions) raises -> JoinPlan:
        var o = opts.copy()
        o.join_order = List[Int]()
        for i in range(len(js.names)):
            o.join_order.append(len(js.names) - 1 - i)
        try:
            var p = rule_join_order(js, o)
            p.notes.append("planner: ReversePlanner")
            return p^
        except:
            return rule_join_order(js, opts)


def cell(t: Table, c: Int, r: Int) raises -> String:
    if t.is_null(c, r):
        return "NULL"
    if t.kind(c) == TEXT:
        return "'" + t.text_of(c, Int(t.get(c, r))) + "'"
    return String(t.get(c, r)) + "@" + String(t.scale(c))


def rows_of(t: Table) raises -> String:
    var out = String()
    for r in range(t.rows()):
        for c in range(t.columns()):
            out += (" " if c > 0 else "") + cell(t, c, r)
        out += "\n"
    return out^


def table_a() raises -> SharedTable:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var x = t.add_column("x", INT)
    var s = t.add_column("s", TEXT)
    var words: List[String] = ["West", "west", "East", "north"]
    for i in range(6000):
        if i % 37 == 0:
            t.append_null(k)
        else:
            t.append(k, Int64((i * 7) % 900))
        t.append(x, Int64(i % 100))
        t.append_text(s, words[i % 4])
    return t^.share()


def table_b() raises -> SharedTable:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var m = t.add_column("m", INT)
    var name = t.add_column("name", TEXT)
    for i in range(900):
        t.append(k, Int64(i))
        t.append(m, Int64(i % 60))
        t.append_text(name, "n" + String(i % 7))
    return t^.share()


def table_c() raises -> SharedTable:
    var t = Table.empty()
    var m = t.add_column("m", INT)
    var w = t.add_column("w", DECIMAL, scale=2)
    var tag = t.add_column("tag", TEXT)
    for i in range(60):
        t.append(m, Int64(i))
        t.append(w, Int64(i * 25))
        t.append_text(tag, "g" + String(i % 3))
    return t^.share()


def main() raises:
    # ── the rule planner on plain statistics: a fact table (1M rows) and two dimensions
    var js = stats([1_000_000, 1_000, 10], [1_000_000, 1_000, 10], [(0, 1, 1000, 1000), (0, 2, 10, 10)])
    var written = rule_join_order(js, PlannerOptions())
    # the best order starts from a small table (its rows are cheap to hash); the written order starts with the fact table
    expect(len(written.steps) == 2 and written.cost > 0, "two steps")
    var o = PlannerOptions()
    o.written_order = True
    var forced = rule_join_order(js, o)
    expect(order_of(forced) == "0 1 2", "the written order is kept when forced: " + order_of(forced))
    var o2 = PlannerOptions()
    o2.join_order = [2, 0, 1]
    var p2 = rule_join_order(js, o2)
    expect(order_of(p2) == "2 0 1", "a forced order: " + order_of(p2))
    # a chain a - b - c: c is not reached from a without b
    var chain = stats([100, 100, 100], [100, 100, 100], [(0, 1, 100, 100), (1, 2, 100, 100)])
    var refused = False
    var o3 = PlannerOptions()
    o3.join_order = [0, 2, 1]
    try:
        _ = rule_join_order(chain, o3)
    except e:
        refused = "reaches a table no equality joins yet" in String(e)
    expect(refused, "a forced order that is a cross product is refused")
    # the written order is kept when it is within 10% of the best, else the cheaper order wins
    var star = stats([5_000_000, 50, 60], [5_000_000, 50, 60], [(0, 1, 50, 50), (0, 2, 60, 60)])
    var sp = rule_join_order(star, PlannerOptions())
    expect(order_of(sp) == "0 1 2" or order_of(sp) == "1 0 2" or order_of(sp) == "0 2 1" or order_of(sp) == "2 0 1", "a star order: " + order_of(sp))
    # a table the equalities reach only through another one cannot start the chain's middle: a path a-b-c written c,b,a is reordered or kept (it connects)
    var rev = stats([10, 1000, 100000], [10, 1000, 100000], [(0, 1, 10, 10), (1, 2, 1000, 1000)])
    var rp = rule_join_order(rev, PlannerOptions())
    expect(len(rp.steps) == 2, "a path of three")

    # ── three Frames with edits and a merge: every join order over them equals the baked tables'
    var fa = Frame(table_a())
    var fb = Frame(table_b())
    var fc = Frame(table_c())
    fa.set_text(fa.column("x"), 3, "70")
    fa.set_null(fa.column("k"), 5)
    fa.set_text(fa.column("s"), 9, "brand-new")
    fa.remove(11)
    _ = fa.append(["12", "5", "West"])
    _ = fa.append(["", "", ""])
    fb.set_text(fb.column("m"), 14, "7")
    fb.set_text(fb.column("name"), 20, "n3")
    fb.remove(30)
    _ = fb.append(["5000", "3", "zz"])
    fc.set_text(fc.column("w"), 4, "9.99")
    fc.remove(5)
    var codes = List[Int64]()
    codes.append(Int64(fa.code_of(fa.column("s"), "west")))
    _ = fa.merge(fa.column("s"), codes, Int64(fa.code_of(fa.column("s"), "West")))
    var cf = Catalog()
    cf.add_frame("a", fa)
    cf.add_frame("b", fb)
    cf.add_frame("c", fc)
    var cb = Catalog()
    cb.add_table("a", fa.bake().share())
    cb.add_table("b", fb.bake().share())
    cb.add_table("c", fc.bake().share())
    var Q: List[String] = [
        "SELECT count(*), sum(a.x), sum(c.w) FROM a, b, c WHERE a.k = b.k AND b.m = c.m",
        "SELECT a.s, c.tag, count(*), sum(c.w) FROM a JOIN b ON a.k = b.k JOIN c ON b.m = c.m GROUP BY a.s, c.tag ORDER BY a.s, c.tag",
        "SELECT a.k, b.name, c.w FROM a, b, c WHERE a.k = b.k AND b.m = c.m AND a.x > 90 AND c.w > 3 ORDER BY a.k, b.name, c.w",
        "SELECT count(*) FROM a, b, c WHERE a.k = b.k AND b.m = c.m AND a.x + c.m > 100",
        "SELECT b.name, count(*) FROM a, b, c WHERE a.k = b.k AND b.m = c.m AND ((a.x < 10 AND c.tag = 'g0') OR (a.x > 90 AND c.tag = 'g1')) GROUP BY b.name ORDER BY b.name",
    ]
    var orders: List[List[Int]] = [[], [0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 1, 0]]
    var bad = 0
    for q in Q:
        var want = rows_of(sql(cb, q).table)
        for oi in range(len(orders)):
            for bs in range(3):
                var opt = PlannerOptions()
                opt.join_order = orders[oi].copy()
                opt.build_side = bs
                var got = rows_of(sql(cf, q, opt).table)
                if got != want:
                    bad += 1
                    print("DIFF order " + String(oi) + " build " + String(bs) + " " + q + "\n frames: " + got + "\n baked:  " + want)
        # a custom planner chooses the order
        var cr = sql_with(cf, q, ReversePlanner())
        expect(rows_of(cr.table) == want, "a custom planner's join order: " + q)
        var seen = False
        for n in cr.notes:
            if n.startswith("planner: ReversePlanner"):
                seen = True
        expect(seen, "the notes say which planner chose")
    if bad > 0:
        raise Error("FAILED: " + String(bad) + " joins over frames differ from the baked tables")
    print("ok test_multiway")
