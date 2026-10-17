"""F-124: a subquery nested in a correlated EXISTS that refers to a column two levels out
(`EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND EXISTS (SELECT 1 FROM w WHERE w.k = t.k))`).

* the nested EXISTS / NOT EXISTS that does not touch the middle query's own columns is a condition on the outer
  row alone, so it is lifted out and decorrelated like any EXISTS of the outer query (any depth);
* everything else two levels out (the nested query also reads the middle one, a NOT EXISTS above it, IN, a scalar
  subquery) raises "correlation two levels out is not supported";
* the F-112 review's should-fixes: the refusals (ORDER BY on a subquery, non-equality or expression correlation,
  LIMIT) say what they refuse, repeated outer keys equal DuckDB, ORDER BY an output name is case-insensitive.
Expected answers come from DuckDB native tables; NULLs in the outer key, the inner key, mixed-case names."""
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import pipeline_kernels as pk

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]


def _i(*v):
    return pa.array(list(v), pa.int64())


def _tables():
    return {
        "t": pa.table({"K": _i(1, 2, 3, 4, None), "a": _i(10, 20, 30, 40, 50), "j": _i(1, 1, 2, 3, 4)}),
        "u": pa.table({"k": _i(1, 1, 2, 3, None, 9), "b": _i(5, 7, 8, None, 1, 2), "Name": _i(1, 2, 3, 4, 5, 6),
                       "j": _i(1, 2, 2, 3, 4, 4)}),
        "w": pa.table({"k": _i(1, 2, 2, 5, None), "m": _i(1, 2, 3, 4, 5), "a": _i(1, 2, 3, 4, 5)}),
    }


def _con():
    con = duckdb.connect()
    for name, tbl in _tables().items():
        con.register(name + "_arrow", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
    return con


def _norm(tbl):
    def v(x):
        return int(x) if isinstance(x, Decimal) and x == int(x) else float(x) if isinstance(x, Decimal) else x
    cols = tbl.column_names
    return cols, sorted((tuple(v(x) for x in r) for r in zip(*[tbl.column(c).to_pylist() for c in cols])), key=repr)


LIFTED = [                                               # the nested subquery needs only the outer row
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND NOT EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K AND w.m > 1))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.m = t.j))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.b > 5 AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",         # no outer key
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.b > 100 AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",       # none
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K) "
    "AND NOT EXISTS (SELECT 1 FROM w WHERE w.m = t.a))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.K) AND u.k = t.K AND u.b > 5)",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K "
    "AND EXISTS (SELECT 1 FROM u u2 WHERE u2.j = w.a)))",                                                                  # three deep
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE W.K = T.k))",      # case-insensitive
    "SELECT t.K FROM t WHERE t.a > 10 AND EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K)) "
    "AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K)",
    "SELECT x.K FROM t x WHERE EXISTS (SELECT 1 FROM u WHERE u.k = x.K AND EXISTS (SELECT 1 FROM w WHERE w.k = x.K))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = u.k))",     # one level: as before
    "SELECT t.K FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = u.k))",
]

# F-112 review: repeated outer keys, in every form that takes correlation
REPEATED_KEYS = [
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND u.k = t.K)",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND u.j = t.K)",
    "SELECT t.K FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND u.j = t.K)",
    "SELECT t.K FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND u.j = t.j)",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND u.k = t.j)",
    "SELECT t.K FROM t WHERE t.j IN (SELECT u.j FROM u WHERE u.k = t.K AND u.j = t.K)",
    "SELECT t.K FROM t WHERE t.a IN (SELECT u.b * 5 FROM u WHERE u.k = t.K AND u.j = t.K)",
    "SELECT t.K FROM t WHERE t.a > (SELECT sum(b) FROM u WHERE u.k = t.K AND u.j = t.K)",
    "SELECT t.K FROM t WHERE t.a > (SELECT sum(b) FROM u WHERE u.k = t.K AND u.k = t.j)",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K AND u.k = t.K) AS m FROM t",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K AND u.j = t.K) AS m FROM t",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K AND u.j = t.j) AS m, (SELECT count(*) FROM u WHERE u.k = t.K AND u.j = t.j) AS c FROM t",
]

# ORDER BY an output name matches without regard to case
ORDER_BY_NAME = [
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY M, K",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS Mm FROM t ORDER BY mM DESC, K",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY k",
    "SELECT K, (SELECT count(*) FROM u WHERE u.k = t.K) AS C FROM t ORDER BY c DESC, K LIMIT 3",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", LIFTED + REPEATED_KEYS + ORDER_BY_NAME)
def test_matches_duckdb(q, device):
    exp = _con().execute(q).to_arrow_table()
    got = mx.sql(q, **_tables()).compute(device=device)
    assert _norm(got) == _norm(exp)


TWO_OUT = "correlation two levels out is not supported"
REFUSED_TWO_OUT = [                                      # DuckDB answers all of these; we say what is not supported
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = u.k AND w.m = t.a))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND NOT EXISTS (SELECT 1 FROM w WHERE w.k = u.k AND w.m = t.a))",
    "SELECT t.K FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",
    "SELECT t.K FROM t WHERE t.a IN (SELECT u.b * 5 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND u.k IN (SELECT w.k FROM w WHERE w.m = t.a))",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K)) AS m FROM t",
    "SELECT t.K FROM t WHERE t.a > (SELECT sum(b) FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = t.K))",
    "SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.k = u.k AND EXISTS "
    "(SELECT 1 FROM u u2 WHERE u2.b = t.a)))",
]


@pytest.mark.parametrize("q", REFUSED_TWO_OUT)
def test_two_levels_out_says_so(q):
    _con().execute(q)                                    # DuckDB accepts it
    with pytest.raises(NotImplementedError, match=TWO_OUT):
        mx.sql(q, **_tables())


REFUSED = [                                              # (query, what it says); the F-112 should-fixes
    ("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY (SELECT max(b) FROM u WHERE u.k = t.K)",
     "supported only as a comparison that is a top-level AND condition of WHERE"),
    ("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY m + 1",
     "supported only as a comparison that is a top-level AND condition of WHERE"),
    ("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY M + 1",
     "supported only as a comparison that is a top-level AND condition of WHERE"),
    ("SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND u.b > t.a)", "u.b > t.a cannot be decorrelated"),
    ("SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K / 1)", "u.k = t.K / 1 cannot be decorrelated"),
    ("SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K AND EXISTS (SELECT 1 FROM w WHERE w.m = t.a / 10))",
     "w.m = t.a / 10 cannot be decorrelated"),                          # lifted, then refused like any outer expression
    ("SELECT t.K FROM t WHERE t.a IN (SELECT u.b FROM u WHERE u.b > t.a)", "u.b > t.a cannot be decorrelated"),
    ("SELECT K, (SELECT max(b) FROM u WHERE u.b > t.a) AS m FROM t", "u.b > t.a cannot be decorrelated"),
    ("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K / 1) AS m FROM t", "u.k = t.K / 1 cannot be decorrelated"),
    ("SELECT t.K FROM t WHERE t.a > (SELECT max(b) FROM u WHERE u.b > t.a)", "u.b > t.a cannot be decorrelated"),
    ("SELECT t.K FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.K LIMIT 1)", "LIMIT in the subquery is not supported"),
    ("SELECT t.K FROM t WHERE t.a IN (SELECT u.b FROM u WHERE u.k = t.K LIMIT 1)", "LIMIT in the subquery is not supported"),
    ("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K LIMIT 1) AS m FROM t", "not a grouped, DISTINCT or LIMIT subquery"),
    ("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K GROUP BY u.k) AS m FROM t", "not a grouped, DISTINCT or LIMIT subquery"),
]


@pytest.mark.parametrize("q,message", REFUSED)
def test_refusals_say_what_they_refuse(q, message):
    with pytest.raises(NotImplementedError, match=message):
        mx.sql(q, **_tables())


def test_order_by_an_expression_over_a_subquery_alias_raises_in_duckdb_too():
    for q in ("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY m + 1",
              "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY M + 1"):
        with pytest.raises(duckdb.BinderException, match="has a subquery"):
            _con().execute(q)
