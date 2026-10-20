"""F-065: SQL subqueries and CTEs. Scalar subqueries (WHERE / HAVING / SELECT; correlated ones that
decorrelate to join + aggregate), IN / EXISTS as semi joins, NOT IN / NOT EXISTS as anti joins (NOT IN
with DuckDB's NULL rules), WITH, derived tables. Every variant is checked against DuckDB, with NULLs in
the outer column, in the subquery's column and with empty subqueries; forms we cannot translate raise
NotImplementedError naming the form instead of changing meaning."""
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest

import omxcore as mx


def _t():
    return pa.table({
        "k": pa.array([1, 1, 2, 2, 3, 4, 5, 6, 7, None], pa.int64()),
        "v": pa.array([10, 20, 30, 12, 50, 60, 15, 25, 35, 45], pa.int64()),
        "x": pa.array([1, None, 3, 4, 7, 6, 7, 8, 9, 10], pa.int64()),
        "s": pa.array(["a", "b", "a", "c", "b", "a", "c", "b", "a", "c"], pa.string()),
    })


def _u():
    return pa.table({
        "k": pa.array([1, 2, 2, 3, 9, None], pa.int64()),
        "y": pa.array([4, 6, None, 7, 8, 9], pa.int64()),
        "w": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
    })


def _un():                       # no NULL anywhere
    return pa.table({"k": pa.array([1, 2, 3, 9], pa.int64()), "y": pa.array([4, 6, 7, 8], pa.int64())})


def _e():                        # no rows
    return pa.table({"k": pa.array([], pa.int64()), "y": pa.array([], pa.int64())})


TABLES = dict(t=_t(), u=_u(), un=_un(), e=_e())


def _norm(x):
    if isinstance(x, (int, float, Decimal)) and not isinstance(x, bool):
        return round(float(x), 6)
    return x


def _rows(tbl, cols):
    return list(zip(*[[_norm(x) for x in tbl.column(c).to_pylist()] for c in cols]))


def _check(q, nonempty=True):
    con = duckdb.connect()
    for n, t in TABLES.items():
        con.register(n, t)
    exp = con.execute(q).to_arrow_table()
    got = mx.sql(q, **TABLES).compute(device="cpu")
    assert got.column_names == exp.column_names, q
    cols = exp.column_names
    g, e = _rows(got, cols), _rows(exp, cols)
    if "ORDER BY" not in q:
        g, e = sorted(g, key=repr), sorted(e, key=repr)
    assert g == e, q
    if nonempty:
        assert len(e) > 0, q


SCALAR = [
    "SELECT k, v FROM t WHERE v > (SELECT AVG(v) FROM t)",
    "SELECT k, v FROM t WHERE v = (SELECT MAX(v) FROM t)",
    "SELECT k, v FROM t WHERE (SELECT MIN(v) FROM t) < v AND k < 4",
    "SELECT k, v FROM t WHERE v < (SELECT SUM(v) FROM t) / 10",
    "SELECT k, v FROM t WHERE v > (SELECT AVG(y) * 2 FROM u WHERE y > 3)",
    "SELECT k, v FROM t WHERE v > (SELECT COUNT(*) FROM u) * 5",
    "SELECT k, v FROM t WHERE v > (SELECT AVG(x) FROM t)",
    "SELECT k, v FROM t WHERE v > (SELECT AVG(v) FROM t WHERE v > (SELECT MIN(v) FROM t))",
    "SELECT s, SUM(v) AS tot FROM t GROUP BY s HAVING SUM(v) > (SELECT SUM(v) / 4 FROM t)",
    "SELECT s, SUM(v) AS tot FROM t GROUP BY s HAVING SUM(v) > (SELECT SUM(v) FROM t) * 0.3 ORDER BY tot",
    "SELECT s, COUNT(*) AS n FROM t GROUP BY s HAVING COUNT(*) >= (SELECT COUNT(*) FROM u) - 3",
    "SELECT k, v - (SELECT MIN(v) FROM t) AS d FROM t",
    "SELECT k, (SELECT MAX(y) FROM u) AS m FROM t",
    "SELECT s, SUM(v) / (SELECT SUM(v) FROM t) AS share FROM t GROUP BY s",
    "SELECT k, v FROM t WHERE v > (SELECT AVG(y) FROM u) AND s = 'a'",
    "SELECT k, v FROM t WHERE v > (SELECT AVG(y) FROM u) OR k = 5",
]

SCALAR_EMPTY = [      # the subquery has no rows: its value is NULL, a comparison keeps nothing
    "SELECT k, v FROM t WHERE v > (SELECT MAX(y) FROM u WHERE y > 1000)",
    "SELECT k, v FROM t WHERE v <= (SELECT MIN(y) FROM e)",
    "SELECT s, SUM(v) AS tot FROM t GROUP BY s HAVING SUM(v) > (SELECT SUM(y) FROM e)",
]


@pytest.mark.parametrize("q", SCALAR)
def test_scalar_subquery(q):
    _check(q)


@pytest.mark.parametrize("q", SCALAR_EMPTY)
def test_scalar_subquery_over_no_rows_is_null(q):
    _check(q, nonempty=False)


CORRELATED = [
    "SELECT k, v FROM t WHERE v > (SELECT AVG(v) FROM t t2 WHERE t2.k = t.k)",
    "SELECT k, v FROM t WHERE v = (SELECT MAX(v) FROM t t2 WHERE t2.k = t.k)",
    "SELECT k, v FROM t WHERE v >= (SELECT MIN(w) * 5 FROM u WHERE u.k = t.k)",
    "SELECT k, v FROM t WHERE x = (SELECT MIN(y) - 3 FROM u WHERE u.k = t.k)",
    "SELECT k, v FROM t WHERE v * 0.1 < (SELECT 0.5 * SUM(y) FROM u WHERE u.k = t.k AND u.w > 1)",
    "SELECT k, v FROM t WHERE (SELECT SUM(y) FROM u WHERE u.k = t.k) > 5 AND v > 10",
    "SELECT k, v FROM t WHERE v > (SELECT AVG(y) FROM u WHERE k = t.k)",
    "SELECT t.k, v FROM t WHERE v > (SELECT SUM(y) FROM u WHERE u.k = t.k) ORDER BY v",
    "SELECT * FROM t WHERE v > (SELECT AVG(y) FROM u WHERE u.k = t.k)",
    "SELECT s, SUM(v) AS tot FROM t WHERE v > (SELECT AVG(y) FROM u WHERE u.k = t.k) GROUP BY s",
]


@pytest.mark.parametrize("q", CORRELATED)
def test_correlated_scalar_subquery(q):
    _check(q)


IN = [
    "SELECT k, v FROM t WHERE k IN (SELECT k FROM u)",
    "SELECT k, v FROM t WHERE k IN (SELECT k FROM u WHERE w > 2)",
    "SELECT k, v FROM t WHERE k IN (SELECT k FROM un) AND v > 10",
    "SELECT k, v FROM t WHERE x IN (SELECT y FROM u)",                     # NULLs on both sides
    "SELECT k, v FROM t WHERE x IN (SELECT y + w FROM u)",
    "SELECT k, v FROM t WHERE k IN (SELECT k FROM u GROUP BY k HAVING SUM(w) > 3)",
    "SELECT k, v FROM t WHERE k IN (SELECT k FROM u GROUP BY k HAVING COUNT(*) > 1)",
    "SELECT k, v FROM t WHERE k IN (SELECT t2.k FROM t t2 WHERE t2.v > 40)",
    "SELECT k, v FROM t WHERE k IN (SELECT k FROM u) AND x IN (SELECT y FROM u)",
    "SELECT k, v FROM t WHERE v IN (SELECT y * w FROM u WHERE u.k = t.k)",   # correlated IN
    "SELECT s, SUM(v) AS tot FROM t WHERE k IN (SELECT k FROM u) GROUP BY s",
    "SELECT * FROM t WHERE k IN (SELECT k FROM u)",
    "SELECT k, v FROM t WHERE (k) IN (SELECT k FROM u)",
    "SELECT k, v FROM t WHERE k IN (SELECT k + 0 FROM un)",
    "SELECT k, v FROM t WHERE k IN (SELECT y - 3 FROM u WHERE u.k > 0)",
]


@pytest.mark.parametrize("q", IN)
def test_in_subquery_is_a_semi_join(q):
    _check(q)


def test_in_over_an_empty_subquery_keeps_nothing():
    _check("SELECT k, v FROM t WHERE k IN (SELECT k FROM u WHERE u.w > 100) AND v > 0", nonempty=False)


NOT_IN = [
    "SELECT k, v FROM t WHERE k NOT IN (SELECT k FROM un)",                 # NULL in the outer k: dropped
    "SELECT k, v FROM t WHERE k NOT IN (SELECT k FROM un WHERE k > 100)",   # subquery empty: all rows, NULL k too
    "SELECT k, v FROM t WHERE k NOT IN (SELECT k FROM e)",                  # empty table
    "SELECT k, v FROM t WHERE x NOT IN (SELECT y FROM un)",
    "SELECT k, v FROM t WHERE x NOT IN (SELECT y FROM u WHERE w <> 3)",     # the NULL row is filtered out
    "SELECT k, v FROM t WHERE k NOT IN (SELECT k FROM un WHERE k > 100) AND v > 20",
    "SELECT k, v FROM t WHERE k NOT IN (SELECT k FROM un) AND x NOT IN (SELECT y FROM un)",
    "SELECT k, v FROM t WHERE NOT k IN (SELECT k FROM un)",
    "SELECT s, SUM(v) AS tot FROM t WHERE k NOT IN (SELECT k FROM un) GROUP BY s",
]

NOT_IN_NULL = [       # a NULL in the subquery's column: NOT IN is never true
    "SELECT k, v FROM t WHERE k NOT IN (SELECT k FROM u)",
    "SELECT k, v FROM t WHERE x NOT IN (SELECT y FROM u)",
    "SELECT k, v FROM t WHERE k NOT IN (SELECT y FROM u) AND v > 0",
]


@pytest.mark.parametrize("q", NOT_IN)
def test_not_in_subquery_follows_duckdbs_null_rules(q):
    _check(q)


@pytest.mark.parametrize("q", NOT_IN_NULL)
def test_not_in_with_a_null_in_the_subquery_returns_no_rows(q):
    _check(q, nonempty=False)
    assert mx.sql(q, **TABLES).compute(device="cpu").num_rows == 0


def test_not_in_over_an_empty_subquery_keeps_the_null_key_too():
    got = mx.sql("SELECT k, v FROM t WHERE k NOT IN (SELECT k FROM e)", **TABLES).compute(device="cpu")
    assert sorted(got.column("v").to_pylist()) == sorted(_t().column("v").to_pylist())
    assert None in got.column("k").to_pylist()


EXISTS = [
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
    "SELECT k, v FROM t WHERE EXISTS (SELECT * FROM u WHERE u.k = t.k AND u.w > 1)",
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM u WHERE t.k = u.k)",
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.y = t.x)",
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) AND v > 10",
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.k = t.k AND t2.v > 25)",
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) AND EXISTS (SELECT 1 FROM un WHERE un.k = t.k)",
    "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM u WHERE y > 6)",         # uncorrelated: all rows
    "SELECT s, SUM(v) AS tot FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) GROUP BY s",
    "SELECT k, v FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",   # the NULL k has no partner: kept
    "SELECT k, v FROM t WHERE NOT EXISTS (SELECT 1 FROM un WHERE un.k = t.k)",
    "SELECT k, v FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w > 2)",
    "SELECT k, v FROM t WHERE NOT EXISTS (SELECT 1 FROM e WHERE e.k = t.k)",
    "SELECT k, v FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k) AND v > 20",
    "SELECT k, v FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE y > 600)",     # uncorrelated: all rows
    "SELECT * FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
]
EXISTS_EMPTY = [
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM u WHERE y > 600)",         # uncorrelated: none
    "SELECT k, v FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE y > 6)",
    "SELECT k, v FROM t WHERE EXISTS (SELECT 1 FROM e WHERE e.k = t.k)",
]


@pytest.mark.parametrize("q", EXISTS)
def test_exists_is_a_semi_join_and_not_exists_an_anti_join(q):
    _check(q)


@pytest.mark.parametrize("q", EXISTS_EMPTY)
def test_exists_with_no_partner_rows(q):
    _check(q, nonempty=False)


CTE = [
    "WITH a AS (SELECT s, SUM(v) AS tot FROM t GROUP BY s) SELECT s, tot FROM a WHERE tot > 100",
    "WITH a AS (SELECT s, SUM(v) AS tot FROM t GROUP BY s) SELECT * FROM a",
    "WITH a AS (SELECT s, SUM(v) AS tot FROM t GROUP BY s), b AS (SELECT s FROM a WHERE tot > 100) "
    "SELECT s FROM b",
    "WITH a AS (SELECT s, SUM(v) AS tot FROM t GROUP BY s) SELECT s, tot FROM a "
    "WHERE tot = (SELECT MAX(tot) FROM a)",                                    # TPC-H Q15's shape
    "WITH a AS (SELECT k, v FROM t WHERE v > 10) SELECT k, v FROM a WHERE v > (SELECT AVG(v) FROM a)",
    "WITH a AS (SELECT k, v FROM t WHERE v > 10) SELECT a.k, u.y FROM a JOIN u ON a.k = u.k",
    "WITH big AS (SELECT k, v FROM t WHERE v > 20) SELECT k, v FROM big WHERE k IN (SELECT k FROM u)",
    "SELECT d.s, d.tot FROM (SELECT s, SUM(v) AS tot FROM t GROUP BY s) AS d WHERE d.tot > 100",
    "SELECT s, tot FROM (SELECT s, SUM(v) AS tot FROM t GROUP BY s) d "
    "WHERE tot > (SELECT AVG(v) * 3 FROM t)",
]


@pytest.mark.parametrize("q", CTE)
def test_cte_and_derived_table(q):
    _check(q)


def test_a_cte_shadows_a_table_of_the_same_name():
    q = "WITH t AS (SELECT k, v FROM t WHERE v > 40) SELECT k, v FROM t"
    _check(q)
    assert mx.sql(q, **TABLES).compute(device="cpu").num_rows == 3


UNSUPPORTED = [
    ("SELECT k FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) OR v > 50", "EXISTS"),
    ("SELECT k FROM t WHERE k IN (SELECT k FROM u) OR v > 50", "IN (SELECT"),
    ("SELECT k FROM t WHERE NOT (k IN (SELECT k FROM u) AND v > 50)", "IN (SELECT"),
    ("SELECT k, CASE WHEN k IN (SELECT k FROM u) THEN 1 ELSE 0 END AS f FROM t", "IN (SELECT"),
    ("SELECT k, EXISTS (SELECT 1 FROM u WHERE u.k = t.k) AS f FROM t", "EXISTS"),
    ("SELECT k FROM t WHERE v > (SELECT COUNT(*) FROM u WHERE u.k = t.k)", "COUNT"),
    ("SELECT k FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.y <> t.x)", "correlated predicate"),
    ("SELECT k FROM t WHERE v > (SELECT AVG(y) FROM u WHERE u.k < t.k)", "correlated predicate"),
    ("SELECT k FROM t WHERE k NOT IN (SELECT k FROM u WHERE u.w = t.v)", "NOT IN"),
    ("SELECT k, count(*), (SELECT AVG(y) FROM u WHERE u.k = t.k) AS a FROM t GROUP BY k", "correlated scalar subquery"),
    ("SELECT k FROM t WHERE v > (SELECT k, v FROM t)", "one column"),
    ("SELECT k FROM t WHERE v > (SELECT v FROM t)", "single aggregate"),
    ("SELECT k FROM t WHERE v > (SELECT AVG(v) FROM t GROUP BY s)", "single aggregate"),
    ("SELECT k FROM t WHERE k IN (SELECT k, y FROM u)", "one column"),
    ("SELECT k FROM t WHERE k IN (SELECT * FROM un)", "SELECT *"),
    ("SELECT k FROM t WHERE k + 1 IN (SELECT k FROM u)", "plain column"),
    ("SELECT k FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k LIMIT 1)", "LIMIT"),
    ("SELECT k FROM t WHERE k IN (SELECT k FROM u UNION SELECT k FROM un)", "UNION"),
    ("SELECT k, v FROM t WHERE v > (SELECT CASE WHEN MAX(y) > 5 THEN MAX(y) ELSE 1 END FROM u WHERE u.k = t.k)", "CASE"),
    ("SELECT k, v FROM t WHERE v > (SELECT COALESCE(MAX(y), 1) FROM u WHERE u.k = t.k)", "COALESCE"),
    ("SELECT k, v FROM t WHERE v > (SELECT MAX(y) IS NULL FROM u WHERE u.k = t.k)", "IS NULL"),
    ("SELECT k, v FROM t WHERE k IN (SELECT y / 2 FROM u)", "float"),
    ("SELECT t.k FROM t, u JOIN un ON u.k = un.k WHERE t.k = u.k", "mixed with JOIN"),
    ("WITH a(x, y) AS (SELECT k, v FROM t) SELECT x FROM a", "column list"),
    ("WITH RECURSIVE a AS (SELECT k FROM t) SELECT k FROM a", "RECURSIVE"),
]


@pytest.mark.parametrize("q,word", UNSUPPORTED)
def test_unsupported_forms_raise_naming_the_form(q, word):
    with pytest.raises((NotImplementedError, ValueError)) as e:
        mx.sql(q, **TABLES).compute(device="cpu")
    assert word in str(e.value), str(e.value)


def test_an_aggregate_in_the_outer_where_still_says_so():
    with pytest.raises(ValueError, match="aggregate functions are not allowed in WHERE"):
        mx.sql("SELECT k FROM t WHERE SUM(v) > 3", **TABLES)


def test_an_aggregate_inside_a_subquery_in_where_is_not_an_outer_aggregate():
    got = mx.sql("SELECT k FROM t WHERE v = (SELECT MAX(v) FROM t)", **TABLES).compute(device="cpu")
    assert got.column("k").to_pylist() == [4]


def test_integer_literals_apply_to_the_in_item_only_not_to_the_subquerys_where():
    """`w / 2 > 1` inside the subquery is SQL division (1.5 > 1), not an integer one that floors."""
    _check("SELECT k, v FROM t WHERE k IN (SELECT k * 1 FROM u WHERE w / 2 > 1)")
    _check("SELECT k, v FROM t WHERE k IN (SELECT k FROM u GROUP BY k HAVING SUM(w) / 2 > 1)")


def test_a_correlated_aggregate_combined_by_arithmetic_is_still_decorrelated():
    _check("SELECT k, v FROM t WHERE v > (SELECT 2 * MAX(y) - MIN(y) / 2 FROM u WHERE u.k = t.k)")
