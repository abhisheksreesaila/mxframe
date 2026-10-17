"""F-112: (1) a correlated EXISTS / NOT EXISTS / IN whose inner WHERE uses an inner SELECT alias (DuckDB accepts
the alias there); (2) a correlated scalar subquery in the SELECT list, decorrelated to a LEFT JOIN with the
per-key aggregate: an outer row without partners (or with a NULL key) gets NULL, and COUNT gives 0 (F-065's
empty-input rules), also inside ``count(*) + 1`` and ``coalesce(sum(x), 0)``. Expected answers come from DuckDB
native tables; mixed-case names, NULLs in the outer key, the inner key and the aggregated column."""
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
        "t": pa.table({"K": _i(1, 2, 3, 4, None), "a": _i(10, 20, 30, 40, 50)}),
        "u": pa.table({"k": _i(1, 1, 2, 3, None, 9), "b": _i(5, 7, 8, None, 1, 2), "Name": _i(1, 2, 3, 4, 5, 6)}),
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


ALIAS_IN_WHERE = [
    "SELECT t.K FROM t WHERE EXISTS (SELECT u.b AS bb FROM u WHERE bb > 5 AND u.k = t.K)",
    "SELECT t.K FROM t WHERE NOT EXISTS (SELECT u.b AS bb FROM u WHERE bb > 5 AND u.k = t.K)",
    "SELECT t.K FROM t WHERE EXISTS (SELECT u.b + 1 AS bb FROM u WHERE bb > 6 AND u.k = t.K)",
    "SELECT t.K FROM t WHERE EXISTS (SELECT u.NAME AS NN FROM u WHERE nn > 1 AND U.K = T.k)",     # case-insensitive
    "SELECT t.K FROM t WHERE EXISTS (SELECT u.b AS bb FROM u WHERE bb > 5 AND bb < 8 AND u.k = t.K)",
    "SELECT t.K FROM t WHERE t.a IN (SELECT u.b * 2 AS bb FROM u WHERE bb > 6 AND u.k = t.K)",
    "SELECT t.K FROM t WHERE EXISTS (SELECT u.b AS bb FROM u WHERE (bb > 7 OR bb = 5) AND u.k = t.K)",
    "SELECT t.K FROM t WHERE EXISTS (SELECT u.b AS k FROM u WHERE k = 1 AND u.k = t.K)",          # the column k wins
    "SELECT t.K FROM t WHERE EXISTS (SELECT u.b AS bb FROM u WHERE bb > 5 AND u.k = t.K) AND t.a > 10",
]

SCALAR_IN_SELECT = [
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t",
    "SELECT K, (SELECT min(b) FROM u WHERE u.k = t.K) AS m, a FROM t",
    "SELECT K, (SELECT count(*) FROM u WHERE u.k = t.K) AS c FROM t",
    "SELECT K, (SELECT count(b) FROM u WHERE u.k = t.K) AS c FROM t",
    "SELECT K, (SELECT sum(b) FROM u WHERE u.k = t.K) AS s FROM t",
    "SELECT K, (SELECT avg(b) FROM u WHERE u.k = t.K) AS s FROM t",
    "SELECT K, (SELECT max(b) + 1 FROM u WHERE u.k = t.K) AS m FROM t",
    "SELECT K, (SELECT max(b) - min(b) FROM u WHERE u.k = t.K) AS m FROM t",
    "SELECT K, (SELECT count(*) FROM u WHERE u.k = t.K) + 1 AS c FROM t",
    "SELECT K, a + coalesce((SELECT sum(b) FROM u WHERE u.k = t.K), 0) AS m FROM t",
    "SELECT K, (SELECT CASE WHEN count(*) > 1 THEN 10 ELSE 20 END FROM u WHERE u.k = t.K) AS c FROM t",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m, (SELECT count(*) FROM u WHERE u.k = t.K) AS c FROM t",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K AND u.b > 5) AS m FROM t",
    "SELECT K, (SELECT count(*) FROM u WHERE u.k = t.K AND u.b > 100) AS c FROM t",             # nothing matches anywhere
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K AND u.b > 100) AS m FROM t",
    "SELECT K, (SELECT max(b) FROM u WHERE U.K = T.k) AS m FROM t",                              # case-insensitive
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t WHERE a > 10",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t ORDER BY m DESC, K",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t WHERE K > 1 ORDER BY K LIMIT 2",
    "SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) - 1 AS m FROM t WHERE (SELECT max(b) FROM u WHERE u.k = t.K) > 5",
    "SELECT *, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t",
    "SELECT t.*, (SELECT count(*) FROM u WHERE u.k = t.K) AS m FROM t",
    "SELECT DISTINCT (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t",
    "SELECT x.K, (SELECT max(b) FROM u WHERE u.k = x.K) AS m FROM t x",
    "SELECT K, (SELECT max(bb) FROM (SELECT b AS bb, k FROM u) w WHERE w.k = t.K) AS m FROM t",
    "SELECT t.K, (SELECT max(u.b) FROM u JOIN t t2 ON t2.K = u.k WHERE t2.a = t.a) AS m FROM t",
    "SELECT K, (SELECT max(b) FROM u) AS m FROM t",                                                # uncorrelated, as before
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", ALIAS_IN_WHERE + SCALAR_IN_SELECT)
def test_matches_duckdb(q, device):
    exp = _con().execute(q).to_arrow_table()
    got = mx.sql(q, **_tables()).compute(device=device)
    assert _norm(got) == _norm(exp)


def test_unaliased_scalar_keeps_a_name():
    got = mx.sql("SELECT K, (SELECT max(b) FROM u WHERE u.k = t.K) FROM t", **_tables()).compute(device="cpu")
    assert got.column_names[0] == "K" and len(got.column_names) == 2
    assert sorted(got.column(1).to_pylist(), key=repr) == [7, 8, None, None, None]


def test_scalar_in_select_with_outer_group_by_says_so():
    with pytest.raises(NotImplementedError, match="GROUP BY or aggregates in the outer query"):
        mx.sql("SELECT K, count(*) AS c, (SELECT max(b) FROM u WHERE u.k = t.K) AS m FROM t GROUP BY K", **_tables())
