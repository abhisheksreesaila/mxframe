"""F-156: SQL expressions through the real `mx` command (tests/mx_oracle.py compares typed values with DuckDB's, kinds and scales too).

`test_mojo_exprs.py` runs every expression over a table, an edited frame and a streamed file with `sql_probe`; this file is the same engine behind the agent
command line: the JSON an agent reads (dates as ISO text, decimals as exact numbers, floats, text with accents), `EXPLAIN` showing the per-word tables, and an error
that names its position.
"""
import pytest

from mx_oracle import check, mx_binary, run_mx
from test_mojo_exprs import data  # noqa: F401  (a session fixture: the generated Parquet file and DuckDB tables)


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


QUERIES = [
    ("SELECT id, CASE WHEN x > 10 THEN 'hi' WHEN x < -10 THEN 'lo' ELSE 'mid' END AS bucket, COALESCE(d, 0) AS d0, NULLIF(y, 0) AS y0 FROM t ORDER BY id", None, True),
    ("SELECT id, lower(s) AS l, upper(s) AS u, length(s) AS n, substr(s, 2, 3) AS part, s || '!' AS bang FROM t ORDER BY id", None, True),
    ("SELECT id, s LIKE 'W%' AS a, s ILIKE '%ÉA%' AS b, s NOT LIKE '_a' AS c, starts_with(s, 'W') AS d, contains(s, 'a') AS e FROM t ORDER BY id", None, True),
    ("SELECT id, CAST(x AS DECIMAL(10,2)) AS dx, CAST(d AS INTEGER) AS di, CAST(d AS DOUBLE) AS dd, CAST(nums AS DOUBLE) AS nd, CAST(dt AS VARCHAR) AS ds FROM t ORDER BY id", None, True),
    ("SELECT id, EXTRACT(year FROM dt) AS y, EXTRACT(dow FROM dt) AS dow, date_trunc('month', dt) AS m, dt + INTERVAL 1 MONTH AS next, datediff('day', dt, dt2) AS gap FROM t ORDER BY id",
     "SELECT id, CAST(EXTRACT(year FROM dt) AS BIGINT) AS y, CAST(EXTRACT(dow FROM dt) AS BIGINT) AS dow, CAST(date_trunc('month', dt) AS DATE) AS m, CAST(dt + INTERVAL 1 MONTH AS DATE) AS next, "
     "datediff('day', dt, dt2) AS gap FROM t ORDER BY id", True),
    ("SELECT g, sum(CASE WHEN x > 0 THEN d ELSE 0 END) AS pos, count(CASE WHEN y = 0 THEN 1 END) AS zero, max(lower(s)) AS top FROM t GROUP BY g",
     "SELECT g, sum(CASE WHEN x > 0 THEN d ELSE 0 END) AS pos, count(CASE WHEN y = 0 THEN 1 END) AS zero, arg_max(lower(s), (alower(lower(s)), lower(s))) AS top FROM t GROUP BY g", False),
    ("SELECT lower(s) AS k, count(*), sum(x), count(DISTINCT upper(s2)) FROM t GROUP BY lower(s)", None, False),
    ("SELECT count(*) FROM t WHERE s = s2 OR lower(s) IN ('west', 'east') OR x IN (" + ",".join(str(i) for i in range(60)) + ")", None, False),
    ("SELECT x / y AS q, x // y AS r, x % y AS m, d / y AS dq, 7 / x AS inv FROM t ORDER BY id", None, True),
    ("SELECT x IS DISTINCT FROM y AS a, s IS NOT DISTINCT FROM s2 AS b, b IS TRUE AS c, b IS NOT FALSE AS e FROM t ORDER BY id", None, True),
]


@pytest.mark.parametrize("sql,duck,ordered", QUERIES, ids=[q[0][:40] for q in QUERIES])
def test_mx_answers_equal_duckdb(mx, data, sql, duck, ordered):
    check(mx, sql, data.table, files={"t": data.path}, duck_sql=duck, ordered=ordered)


def test_explain_shows_the_per_word_tables_and_the_logical_plan(mx, data):
    res = run_mx(mx, ["explain", "SELECT lower(s) AS l, count(*) FROM t WHERE s LIKE 'W%' GROUP BY lower(s)", "--format", "json"], {"t": data.path})
    text = str(res)
    assert "per-word table" in text and "never per row" in text and "lower(s)" in text, text


def test_an_expression_error_names_its_position(mx, data):
    out = run_mx(mx, ["sql", "SELECT id, CAST(s AS BLOB) FROM t"], {"t": data.path}, expect_exit=1)
    assert (out["error"]["line"], out["error"]["column"]) == (1, 22) and out["error"]["message"].startswith("the type `BLOB` in a cast"), out
    out = run_mx(mx, ["sql", "SELECT lower(x) FROM t"], {"t": data.path}, expect_exit=1)
    assert (out["error"]["line"], out["error"]["column"]) == (1, 14) and out["error"]["message"] == "lower() needs TEXT, found INT (CAST it AS VARCHAR first)", out
