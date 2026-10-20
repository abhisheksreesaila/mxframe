"""F-041: SQL GROUP BY with SELECT aliases. A SELECT alias on a group key names the output column,
GROUP BY may name the alias or repeat the aliased expression, and the result columns follow the
SELECT list. Every case is checked against DuckDB."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore.np_arrow import array as A


def _t(n=20_000, seed=5):
    r = np.random.default_rng(seed)
    return pa.table({
        "s": pa.array(np.array(["a", "b", "c", "d"])[r.integers(0, 4, n)].tolist(), pa.string()),
        "k": A(r.integers(0, 7, n).astype(np.int64)),
        "v": A(r.integers(0, 100, n).astype(np.int64)),
    })


QUERIES = [
    "SELECT s AS b, SUM(v) AS x FROM t GROUP BY s",
    "SELECT s AS b, SUM(v) AS x FROM t GROUP BY b",
    "SELECT SUM(v) AS x, s AS b FROM t GROUP BY s",
    "SELECT SUM(v) AS x FROM t GROUP BY s",
    "SELECT k AS kk, s, COUNT(v) AS n FROM t GROUP BY k, s",
    "SELECT s, k AS kk, MAX(v) AS m FROM t GROUP BY kk, s",
    "SELECT s AS b, SUM(v) AS x FROM t WHERE v > 10 GROUP BY b ORDER BY b",
    "SELECT s AS b, SUM(v) AS x FROM t GROUP BY s ORDER BY x DESC LIMIT 2",
    "SELECT k + 1 AS k1, SUM(v) AS x FROM t GROUP BY k + 1",
    "SELECT k + 1 AS k1, SUM(v) AS x FROM t GROUP BY k1",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s",
    # review round 1: ORDER BY the original name or expression of an aliased key
    "SELECT s AS b, SUM(v) AS x FROM t GROUP BY s ORDER BY s",
    "SELECT k + 1 AS k1, SUM(v) AS x FROM t GROUP BY k + 1 ORDER BY k + 1",
    "SELECT SUM(v) AS x FROM t GROUP BY s ORDER BY s",
    # qualified and unqualified names of one column
    "SELECT t.s AS b, SUM(v) AS x FROM t GROUP BY s",
    "SELECT s AS b, SUM(v) AS x FROM t GROUP BY t.s",
    # one key selected twice
    "SELECT s, s AS b, SUM(v) AS x FROM t GROUP BY s",
    "SELECT s AS a, s AS b, SUM(v) AS x FROM t GROUP BY s",
    # a join, with the alias on the joined key
    "SELECT u.name AS nm, SUM(v) AS x FROM t JOIN u ON t.k = u.kid GROUP BY u.name ORDER BY nm",
]


def _rows(tbl, cols):
    d = tbl.to_pydict() if isinstance(tbl, pa.Table) else tbl
    return sorted(zip(*[[float(x) if isinstance(x, (int, float)) else x for x in d[c]] for c in cols]))


@pytest.mark.parametrize("q", QUERIES)
def test_matches_duckdb(q):
    t = _t()
    u = pa.table({"kid": A(np.arange(7, dtype=np.int64)), "name": pa.array(list("pqrstuv"), pa.string())})
    exp = duckdb.sql(q).arrow()
    if isinstance(exp, pa.RecordBatchReader):
        exp = exp.read_all()
    got = mx.sql(q, t=t, u=u).compute(device="cpu")
    assert got.column_names == exp.column_names
    if "ORDER BY" in q:
        for c in exp.column_names:
            g, e = got.column(c).to_pylist(), exp.column(c).to_pylist()
            assert [float(x) if isinstance(x, (int, float)) else x for x in g] == \
                   [float(x) if isinstance(x, (int, float)) else x for x in e]
    else:
        assert _rows(got, got.column_names) == _rows(exp, exp.column_names)
