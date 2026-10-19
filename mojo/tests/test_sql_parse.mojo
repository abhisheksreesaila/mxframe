# F-150: the SQL lexer and parser. A parse that works gives back the text of the statement as the tree renders it;
# one that does not raises the whole message (position included).
from mxframe.sql import parse_sql, SqlStatement



def fail(msg: String) raises:
    raise Error(msg)


def check_eq(got: String, want: String, what: String) raises:
    if got != want:
        raise Error(what + "\n   got:    " + got + "\n   wanted: " + want)


def renders(sql: String, want: String) raises:
    var st = parse_sql(sql)
    var out = String()
    for k in range(len(st.items)):
        out += (", " if k > 0 else "") + st.render(st.items[k].expr)
        if st.items[k].has_alias:
            out += " AS " + st.items[k].alias
    out += " FROM " + st.table
    if st.where >= 0:
        out += " WHERE " + st.render(st.where)
    if len(st.group_by) > 0:
        out += " GROUP BY "
        for k in range(len(st.group_by)):
            out += (", " if k > 0 else "") + st.render(st.group_by[k])
    if st.having >= 0:
        out += " HAVING " + st.render(st.having)
    if len(st.order_by) > 0:
        out += " ORDER BY "
        for k in range(len(st.order_by)):
            out += (", " if k > 0 else "") + st.render(st.order_by[k].expr) + (" DESC" if st.order_by[k].desc else "")
            if st.order_by[k].nulls == 1:
                out += " NULLS FIRST"
    if st.limit >= 0:
        out += " LIMIT " + String(st.limit)
    if st.offset > 0:
        out += " OFFSET " + String(st.offset)
    check_eq(out, want, sql)


def fails(sql: String, want: String) raises:
    try:
        _ = parse_sql(sql)
    except e:
        check_eq(String(e), want, sql)
        return
    fail("`" + sql + "` parsed, but should fail with " + want)


def main() raises:
    renders("select a, sum(b) as s from t", "a, sum(b) AS s FROM t")
    renders("SELECT region r, count(*) FROM orders o WHERE qty > 2 AND (region = 'West' OR region = 'it''s') GROUP BY region HAVING count(*) >= 5 ORDER BY 2 DESC NULLS FIRST, r LIMIT 10 OFFSET 3",
            "region AS r, count(*) FROM orders WHERE (qty > 2) AND ((region = 'West') OR (region = 'it''s')) GROUP BY region HAVING count(*) >= 5 ORDER BY 2 DESC NULLS FIRST, r LIMIT 10 OFFSET 3")
    renders("select x from t where a + b * 2 - -3 <> c // 2 % 5 and not d is null", "x FROM t WHERE (((a + (b * 2)) - -3) <> ((c // 2) % 5)) AND (NOT (d IS NULL))")
    renders("select 1 from t where a between 1 and 5 and b not in (1, 2, 3) and c is not null and d not between 0.5 and 7",
            "1 FROM t WHERE (((a BETWEEN 1 AND 5) AND (b NOT IN (1, 2, 3))) AND (c IS NOT NULL)) AND (d NOT BETWEEN 0.5 AND 7)")
    renders("select count(distinct x), quantile_disc(x, 0.5), median(x) from t -- trailing comment\n;", "count(DISTINCT x), quantile_disc(x, 0.5), median(x) FROM t")
    renders("select \"My Col\" from \"T 1\" where d >= DATE '2024-01-31' /* c */ limit 1 offset 2", "My Col FROM T 1 WHERE d >= DATE '2024-01-31' LIMIT 1 OFFSET 2")
    renders("select t.a from t where t.a = -5 and b = - 2.50", "t.a FROM t WHERE (t.a = -5) AND (b = -2.50)")

    fails("", "SQL error at line 1, column 1: the statement is empty")
    fails("selct a from t", "SQL error at line 1, column 1: expected SELECT or EXPLAIN, found `selct` (did you mean SELECT?)")
    fails("select a form t", "SQL error at line 1, column 10: expected FROM, found `form` (did you mean FROM?)")
    fails("select a from t\nwhere", "SQL error at line 2, column 6: expected an expression, found the end of the statement")
    fails("select a from t join u on t.a = u.a", "SQL error at line 1, column 17: `JOIN` is not supported yet (F-153)")
    fails("select a from t left join u on 1", "SQL error at line 1, column 17: `LEFT JOIN` is not supported yet (F-153)")
    fails("select a from t, u", "SQL error at line 1, column 16: more than one table in FROM (a comma join) is not supported yet (F-153)")
    fails("select distinct a from t", "SQL error at line 1, column 8: `SELECT DISTINCT` is not supported yet (F-152)")
    fails("select * from t", "SQL error at line 1, column 8: `*` in the select list is not supported yet (F-152)")
    fails("select a from (select 1) x", "SQL error at line 1, column 15: a subquery in FROM is not supported yet (F-159)")
    fails("with x as (select 1) select * from x", "SQL error at line 1, column 1: `WITH` (common table expressions) is not supported yet (F-159)")
    fails("select case when a then 1 end from t", "SQL error at line 1, column 8: `CASE` is not supported yet (F-156)")
    fails("select cast(a as int) from t", "SQL error at line 1, column 8: `CAST(...)` is not supported yet (F-156)")
    fails("select a from t where b like 'x%'", "SQL error at line 1, column 25: `LIKE` is not supported yet (F-156)")
    fails("select a::int from t", "SQL error at line 1, column 9: `::` (a cast) is not supported yet (F-156)")
    fails("select a from t where b in (select 1)", "SQL error at line 1, column 29: a subquery in IN (...) is not supported yet (F-154)")
    fails("select 1e3 from t", "SQL error at line 1, column 8: the number `1e3` has an exponent: floating-point literals are not supported yet (FLOAT arrives with F-157)")
    fails("select a from t where b = 'open", "SQL error at line 1, column 27: this string is never closed (expected a closing ')")
    fails("select a from t where b @ 1", "SQL error at line 1, column 25: unexpected character `@`")
    fails("insert into t values (1)", "SQL error at line 1, column 1: `INSERT` statements are not supported: only SELECT (and EXPLAIN SELECT) is")
    fails("select a from t limit x", "SQL error at line 1, column 23: LIMIT needs a whole number, found `x` (expressions are not supported there)")
    fails("select a from t group by all", "SQL error at line 1, column 26: `GROUP BY ALL` is not supported yet")
    fails("explain analyze select a from t", "SQL error at line 1, column 9: `EXPLAIN ANALYZE` is not supported yet (F-155)")
    fails("select a from t; select b from t", "SQL error at line 1, column 18: expected the end of the statement, found `select`")
    fails("select f(a over (partition by b)) from t", "SQL error at line 1, column 12: expected `)`, found `over`")
    fails("select sum(a) over (order by b) from t", "SQL error at line 1, column 15: `OVER` (window functions) is not supported yet")
    print("ok")
