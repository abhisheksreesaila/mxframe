# F-153: a query over edited Frames (edits on both sides: session words, NULL keys, appends, removes, and a merge of TEXT values on each side)
# must equal the same query over their baked tables: row queries, aggregates and joins, fused and materialized.
from mxframe import Table, SharedTable, Frame, Catalog, PlannerOptions, sql, sql_with, INT, DECIMAL, TEXT

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

def left_table() raises -> SharedTable:
    var t = Table.empty()
    var k = t.add_column("k", INT); var s = t.add_column("s", TEXT); var d = t.add_column("d", DECIMAL, scale=2); var x = t.add_column("x", INT)
    var words: List[String] = ["West", "west", "East", "WEST", "", "north"]
    for i in range(50000):
        if i % 41 == 0: t.append_null(k)
        else: t.append(k, Int64((i * 37) % 3500))
        if i % 23 == 0: t.append_null(s)
        else: t.append_text(s, words[i % 6])
        t.append(d, Int64((i % 400) * 25))
        t.append(x, Int64(i % 1000 - 500))
    return t^.share()

def right_table() raises -> SharedTable:
    var t = Table.empty()
    var k = t.add_column("k", INT); var s = t.add_column("s", TEXT); var d = t.add_column("d", DECIMAL, scale=4); var y = t.add_column("y", INT)
    var words: List[String] = ["north", "East", "west", "West", "Zed"]
    for i in range(3000):
        t.append(k, Int64(i))                       # unique
        t.append_text(s, words[i % 5])
        t.append(d, Int64((i % 400) * 2500))
        t.append(y, Int64(i))
    return t^.share()

def main() raises:
    var lf = Frame(left_table())
    var rf = Frame(right_table())
    lf.set_text(lf.column("s"), 3, "brand-new")      # a session word on the probe side
    lf.set_text(rf.column("s"), 4, "Zed")            # a word only the other side had
    lf.set_null(lf.column("k"), 5)
    lf.set_text(lf.column("k"), 6, "77")
    lf.set_text(lf.column("d"), 7, "1.23")
    lf.remove(8); lf.remove(9)
    _ = lf.append(["12", "West", "0.50", "3"])
    _ = lf.append(["", "", "", ""])
    rf.set_text(rf.column("s"), 10, "brand-new")      # the same session word on the build side
    rf.set_text(rf.column("s"), 11, "WEST")
    rf.set_null(rf.column("k"), 12)
    rf.set_text(rf.column("k"), 13, "5000")           # still unique
    rf.set_text(rf.column("d"), 14, "0.0100")
    rf.remove(15)
    _ = rf.append(["4000", "east", "1.5", "77"])
    var codes = List[Int64](); codes.append(Int64(rf.code_of(rf.column("s"), "Zed")))
    _ = rf.merge(rf.column("s"), codes, Int64(rf.code_of(rf.column("s"), "north")))     # Zed reads as north on the build side
    var lcodes = List[Int64](); lcodes.append(Int64(lf.code_of(lf.column("s"), "west"))); lcodes.append(Int64(lf.code_of(lf.column("s"), "WEST")))
    _ = lf.merge(lf.column("s"), lcodes, Int64(lf.code_of(lf.column("s"), "West")))     # west and WEST read as West on the probe side
    var cf = Catalog(); cf.add_frame("l", lf); cf.add_frame("r", rf)
    var cb = Catalog(); cb.add_table("l", lf.bake().share()); cb.add_table("r", rf.bake().share())
    var Q: List[String] = [
        "SELECT s FROM l WHERE x > 480 ORDER BY k, x, d, s LIMIT 200",
        "SELECT s, k FROM l WHERE k < 40 ORDER BY k, s",
        "SELECT s, count(*) FROM l GROUP BY s ORDER BY s",
        "SELECT s FROM r ORDER BY y LIMIT 50",
        "SELECT s, count(*) FROM r GROUP BY s ORDER BY s",
        "SELECT s, y FROM r WHERE y < 30 ORDER BY y",
        "SELECT count(*) FROM r WHERE s = 'Zed'",
        "SELECT count(*), sum(l.x), sum(r.y), sum(l.d), sum(r.d) FROM l JOIN r ON l.k = r.k",
        "SELECT r.s, count(*), sum(l.x) FROM l JOIN r ON l.k = r.k GROUP BY r.s ORDER BY r.s",
        "SELECT l.s, r.s, count(*) FROM l JOIN r ON l.s = r.s GROUP BY l.s, r.s ORDER BY l.s, r.s",
        "SELECT count(*) FROM l JOIN r ON l.s = r.s AND l.k = r.k",
        "SELECT count(*), sum(l.x) FROM l JOIN r ON l.d = r.d",
        "SELECT l.k, l.s, r.s, r.y FROM l JOIN r ON l.k = r.k WHERE l.x > 400 AND r.y < 2000 ORDER BY l.k, l.s, r.y, r.s",
        "SELECT l.k, r.y FROM l JOIN r ON l.k = r.k WHERE r.s = 'brand-new' OR l.s = 'brand-new' ORDER BY 1, 2",
        "SELECT count(*) FROM l JOIN r ON l.s = r.s WHERE r.s = 'north'",
        "SELECT count(*), count(r.y), sum(l.x), sum(r.y) FROM l LEFT JOIN r ON l.k = r.k",
        "SELECT l.s, r.s, count(*) FROM l LEFT JOIN r ON l.s = r.s GROUP BY l.s, r.s ORDER BY l.s, r.s",
        "SELECT l.k, l.s, r.s, r.y FROM l LEFT JOIN r ON l.k = r.k WHERE l.x > 400 ORDER BY l.k, l.s, r.y, r.s",
        "SELECT count(*), count(r.y) FROM l LEFT JOIN r ON l.k = r.k AND r.y < 2000",
        "SELECT count(*), count(r.y) FROM l LEFT JOIN r ON l.k = r.k WHERE l.x = r.y",
        "SELECT count(*), count(r.y) FROM l LEFT JOIN r ON l.k = r.k WHERE l.s = r.s",
        "SELECT count(*), count(r.y) FROM l LEFT JOIN r ON l.s = r.s WHERE l.k = r.k",
        "SELECT count(*), sum(x) FROM l WHERE k IN (SELECT k FROM r)",
        "SELECT count(*) FROM l WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.k = l.k)",
        "SELECT s, count(*) FROM l WHERE s IN (SELECT s FROM r) GROUP BY s ORDER BY s",
        "SELECT s, count(*) FROM l WHERE s NOT IN (SELECT s FROM r) GROUP BY s ORDER BY s",
        "SELECT count(*) FROM l WHERE k NOT IN (SELECT k FROM r)",
        "SELECT count(*) FROM l WHERE k NOT IN (SELECT k FROM r WHERE k IS NOT NULL)",
        "SELECT k, s FROM l WHERE EXISTS (SELECT 1 FROM r WHERE r.s = l.s AND r.y > 2900) ORDER BY k, s LIMIT 100",
        "SELECT count(*) FROM r WHERE s IN (SELECT s FROM l WHERE x > 100)",
    ]
    var bad = 0
    for q in Q:
        for fuse in range(2):
            var o = PlannerOptions(); o.fuse = fuse == 0
            var a = sql(cf, q, o)
            var b = sql(cb, q, o)
            var ga = rows_of(a.table); var gb = rows_of(b.table)
            if ga != gb:
                bad += 1
                for n in a.notes:
                    print("   note: " + n)
                print("DIFF fuse=" + String(fuse == 0) + " " + q + "\n frames: " + ga[byte=:min(400, ga.byte_length())] + "\n baked:  " + gb[byte=:min(400, gb.byte_length())])
    if bad > 0:
        raise Error("FAILED: " + String(bad) + " queries over frames differ from the same queries over the baked tables")
    print("ok test_frame_merge")
