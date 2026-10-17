"""F-111: a bare column name that two joined tables have is an error (DuckDB: "Ambiguous reference to column
name"), not a silent pick of the left one. USING / NATURAL columns have one merged value, a name only one table
has is fine, qualified names are fine. Case-insensitive (`Region` vs `region`). Expected behaviour comes from
DuckDB native tables: each error case first checks that DuckDB raises, each good case compares the answer."""
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
        "t": pa.table({"k": _i(1, 2, 3, None), "v": _i(10, 20, 30, 40), "Region": _i(1, 2, 2, 3), "w": _i(5, 6, 7, 8)}),
        "u": pa.table({"k": _i(1, 3, 3, 5), "v": _i(11, 31, 32, 51), "region": _i(1, 2, 3, 3), "z": _i(7, 8, 9, 10)}),
        "x": pa.table({"k": _i(1, 3, 9), "q": _i(1, 1, 2)}),
    }


def _con():
    con = duckdb.connect()
    for name, tbl in _tables().items():
        con.register(name + "_arrow", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
    return con


def _norm(tbl):
    cols = tbl.column_names
    return cols, sorted(zip(*[tbl.column(c).to_pylist() for c in cols]), key=repr)


# every clause, two and three tables, a comma join, same spelling and case-only clashes, subqueries
AMBIGUOUS = [
    ("select", "SELECT v FROM t JOIN u ON t.k = u.k"),
    ("select-case", "SELECT region FROM t JOIN u ON t.k = u.k"),
    ("select-case-up", "SELECT REGION FROM t JOIN u ON t.k = u.k"),
    ("select-expr", "SELECT t.w + v FROM t JOIN u ON t.k = u.k"),
    ("select-agg", "SELECT sum(v) FROM t JOIN u ON t.k = u.k"),
    ("where", "SELECT t.w FROM t JOIN u ON t.k = u.k WHERE v > 1"),
    ("where-case", "SELECT t.w FROM t JOIN u ON t.k = u.k WHERE region > 1"),
    ("group-by", "SELECT t.v FROM t JOIN u ON t.k = u.k GROUP BY v"),
    ("having", "SELECT sum(t.w) FROM t JOIN u ON t.k = u.k GROUP BY t.v HAVING v > 1"),
    ("order-by", "SELECT v AS vv FROM t JOIN u ON t.k = u.k ORDER BY v"),
    ("order-by-agg", "SELECT t.k FROM t JOIN u ON t.k = u.k GROUP BY t.k ORDER BY sum(v)"),
    ("on", "SELECT t.w FROM t JOIN u ON k = u.k"),
    ("on-second", "SELECT t.w FROM t JOIN u ON t.k = u.k JOIN x ON x.q = 1 AND v = 1"),
    ("left", "SELECT t.w FROM t LEFT JOIN u ON t.k = u.k WHERE v = 1"),
    ("three-tables", "SELECT t.w FROM t JOIN u ON t.k = u.k JOIN x ON x.k = t.k WHERE k = 1"),
    ("comma", "SELECT t.w FROM t, u WHERE t.k = u.k AND v = 1"),
    ("alias-not-where", "SELECT t.v AS k FROM t JOIN u ON t.k = u.k WHERE k > 0"),
    ("alias-not-group-by", "SELECT t.v AS k FROM t JOIN u ON t.k = u.k GROUP BY k"),
    ("self-join", "SELECT a.v FROM t a JOIN t b ON a.k = b.k WHERE v > 1"),
    ("using-other-column", "SELECT v FROM t JOIN u USING (k)"),
    ("using-key-twice-on-left", "SELECT t.w FROM t JOIN u ON t.k = u.k JOIN x USING (k)"),
    ("subquery-outer", "SELECT t.w FROM t JOIN u ON t.k = u.k WHERE t.v IN (SELECT q FROM x WHERE v = 1)"),
    ("subquery-own-join", "SELECT t.w FROM t WHERE EXISTS (SELECT 1 FROM u JOIN x ON u.k = x.k WHERE k = 1)"),
]

GOOD = [
    "SELECT t.v, u.v AS uv FROM t JOIN u ON t.k = u.k",
    "SELECT w, z FROM t JOIN u ON t.k = u.k",                                  # names only one table has
    "SELECT w FROM t JOIN u ON t.k = u.k WHERE z > 7",
    "SELECT k FROM t JOIN u USING (k)",
    "SELECT k, w, z FROM t JOIN u USING (k) WHERE k > 0 ORDER BY k, z",
    "SELECT k, count(*) AS n FROM t JOIN u USING (k) GROUP BY k HAVING k > 0 ORDER BY k",
    "SELECT k FROM t LEFT JOIN u USING (k) ORDER BY k",
    "SELECT k FROM t NATURAL JOIN x",
    "SELECT k, w FROM t SEMI JOIN u ON t.k = u.k",                             # a SEMI table's columns are not in scope
    "SELECT k, w FROM t JOIN u USING (k) JOIN x USING (k)",
    "SELECT k, w FROM t JOIN u USING (k) JOIN x ON x.q = 1",                   # the merged k wins over x.k
    "SELECT t.v FROM t JOIN u ON t.k = u.k ORDER BY v",                        # ORDER BY: the SELECT output wins
    "SELECT t.v AS vv FROM t JOIN u ON t.k = u.k ORDER BY vv",
    "SELECT t.v AS k FROM t JOIN u ON t.k = u.k ORDER BY k",
    "SELECT t.w FROM t WHERE EXISTS (SELECT 1 FROM u WHERE k = 1)",            # inner scope first: only u in it
    "SELECT t.w FROM t WHERE EXISTS (SELECT 1 FROM u WHERE v = t.v)",
    "SELECT t.w FROM t JOIN u ON t.k = u.k JOIN x ON x.q = 1 AND z > 0",       # an ON sees the tables so far
    "SELECT a.v, b.v AS bv FROM t a JOIN t b ON a.k = b.k",
    "SELECT v FROM (SELECT t.v FROM t JOIN u ON t.k = u.k) s",
    "SELECT sum(w) FROM (SELECT * FROM t JOIN u ON t.k = u.k) s",
]


@pytest.mark.parametrize("name,q", AMBIGUOUS, ids=[n for n, _ in AMBIGUOUS])
def test_ambiguous_name_raises(name, q):
    with pytest.raises(duckdb.BinderException, match="Ambiguous"):
        _con().execute(q)
    with pytest.raises(ValueError, match=r"ambiguous column reference: \w+ \(tables \w+, \w+"):
        mx.sql(q, **_tables())


def test_error_names_both_tables():
    with pytest.raises(ValueError) as e:
        mx.sql("SELECT region FROM t JOIN u ON t.k = u.k", **_tables())
    assert str(e.value) == "ambiguous column reference: region (tables t, u): qualify it with its table"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", GOOD)
def test_unambiguous_still_equals_duckdb(q, device):
    exp = _con().execute(q).to_arrow_table()
    got = mx.sql(q, **_tables()).compute(device=device)
    assert _norm(got) == _norm(exp)
