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

    renders("select case when a > 1 then 'x' when a < 0 then 'y' end, case a when 1 then 2 else 3 end from t", "CASE WHEN a > 1 THEN 'x' WHEN a < 0 THEN 'y' END, CASE a WHEN 1 THEN 2 ELSE 3 END FROM t")
    renders("select cast(a as int), try_cast(b as decimal(10, 2)), c::double, d::varchar(20) from t", "CAST(a AS INTEGER), TRY_CAST(b AS DECIMAL(10,2)), CAST(c AS DOUBLE), CAST(d AS VARCHAR(20)) FROM t")
    renders("select a from t where b like 'x%' and c not ilike 'y_' escape '!' and d is distinct from e and f is not distinct from 1 and g is true and h is not false",
            "a FROM t WHERE (((((b LIKE 'x%') AND (c NOT ILIKE 'y_' ESCAPE '!')) AND (d IS DISTINCT FROM e)) AND (f IS NOT DISTINCT FROM 1)) AND (g IS TRUE)) AND (h IS NOT FALSE)")
    renders("select 'a' || b || 'c', extract(year from d), date_part('dow', d), d + interval '3' month, d - interval 2 day, d + interval (n) year from t",
            "('a' || b) || 'c', extract('year', d), date_part('dow', d), d + INTERVAL 3 MONTH, d - INTERVAL 2 DAY, d + INTERVAL n YEAR FROM t")
    renders("select substring(s from 2 for 3), substr(s, 2), trim(s), coalesce(a, b, 0), nullif(a, 0) from t", "substring(s, 2, 3), substr(s, 2), trim(s), coalesce(a, b, 0), nullif(a, 0) FROM t")

    fails("", "SQL error at line 1, column 1: the statement is empty")
    fails("selct a from t", "SQL error at line 1, column 1: expected SELECT or EXPLAIN, found `selct` (did you mean SELECT?)")
    fails("select a form t", "SQL error at line 1, column 10: expected FROM, found `form` (did you mean FROM?)")
    fails("select a from t\nwhere", "SQL error at line 2, column 6: expected an expression, found the end of the statement")
    fails("select a from t right join u on 1", "SQL error at line 1, column 17: `RIGHT JOIN` is not supported yet")
    fails("select a from t join u on t.a = u.a join v on 1", "SQL error at line 1, column 37: more than two tables in FROM is not supported yet (F-155)")
    fails("select a from t, u, v", "SQL error at line 1, column 19: more than two tables in FROM is not supported yet (F-155)")
    fails("select a from t join u", "SQL error at line 1, column 23: expected ON or USING after JOIN, found the end of the statement")
    fails("select distinct on (a) a from t", "SQL error at line 1, column 8: `SELECT DISTINCT ON` is not supported yet")
    fails("select a, t.* + 1 from t", "SQL error at line 1, column 15: expected FROM, found `+`")
    fails("select a from (select 1) x", "SQL error at line 1, column 15: a subquery in FROM is not supported yet (F-159)")
    fails("with x as (select 1) select * from x", "SQL error at line 1, column 1: `WITH` (common table expressions) is not supported yet (F-159)")
    fails("select a from t where b glob 'x*'", "SQL error at line 1, column 25: `GLOB` is not supported yet")
    fails("select timestamp '2024-01-01 10:00:00' from t", "SQL error at line 1, column 8: `TIMESTAMP` literals is not supported yet (MX frame has no timestamp kind)")
    fails("select a from t where b like 'x%' escape", "SQL error at line 1, column 41: expected an expression, found the end of the statement")
    fails("select cast(a as hugeint) from t", "SQL error at line 1, column 18: the type `hugeint` in a cast (the casts are INTEGER, BIGINT, SMALLINT, TINYINT, DECIMAL(p, s), DOUBLE, REAL, VARCHAR, DATE and BOOLEAN) is not supported yet")
    fails("select a + interval '3' hour from t", "SQL error at line 1, column 12: an interval in hours is not supported yet (MX frame has no timestamp kind: days, weeks, months and years only)")
    fails("select case when a then 1 from t", "SQL error at line 1, column 27: expected END, found `from` (did you mean FROM?)")
    fails("select a from t where b in (select 1)", "SQL error at line 1, column 37: expected FROM, found `)`")
    renders("select 1e3, 2.5E-3, 7e+2 from t", "1e3, 2.5E-3, 7e+2 FROM t")                       # (F-157: a number with an exponent is a DOUBLE literal)
    fails("select 1e3x from t", "SQL error at line 1, column 8: the number `1e3` runs into the letters after it")
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
