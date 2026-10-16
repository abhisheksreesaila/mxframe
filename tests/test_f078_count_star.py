"""F-078: COUNT(*) counts the group's rows, NULL keys and NULL values included. Every case is
checked against DuckDB, on cpu and gpu, at a size small enough for the Arrow fallback and a size
large enough for the native / fused / GPU paths."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import pipeline_kernels as pk

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]


def _t(n):
    r = np.random.default_rng(78)
    k1 = r.integers(0, 6, n)
    k2 = r.integers(0, 3, n)
    c = r.random(n)
    return pa.table({
        "k": pa.array(k1, pa.int64(), mask=r.random(n) < 0.3),
        "j": pa.array(k2, pa.int64(), mask=r.random(n) < 0.3),
        "c": pa.array(c, pa.float64(), mask=r.random(n) < 0.4),
        "v": pa.array(r.integers(0, 100, n).astype(np.float64)),
    })


QUERIES = [
    "SELECT k, COUNT(*) AS n FROM t GROUP BY k",
    "SELECT k, COUNT(*) AS n, COUNT(c) AS nc FROM t GROUP BY k",
    "SELECT k, COUNT(*) AS n, SUM(v) AS s FROM t GROUP BY k",
    "SELECT k, j, COUNT(*) AS n, SUM(v) AS s FROM t GROUP BY k, j",
    "SELECT COUNT(*) AS n, COUNT(c) AS nc FROM t",
    "SELECT COUNT(*) AS n FROM t WHERE v > 50",
    "SELECT t.k, COUNT(*) AS n, SUM(t.v) AS s FROM t JOIN u ON t.j = u.j GROUP BY t.k",
    "SELECT u.name, COUNT(*) AS n FROM t JOIN u ON t.k = u.k GROUP BY u.name",
]


def _norm(tbl):
    d = tbl.to_pydict()
    cols = tbl.column_names
    return sorted((tuple(x if x is None or isinstance(x, str) else round(float(x), 6) for x in row) for row in zip(*[d[c] for c in cols])),
                  key=lambda r: tuple((x is None, str(x)) for x in r))


@pytest.mark.parametrize("n", [50, 20_000])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", QUERIES)
def test_count_star_matches_duckdb(q, device, n):
    t = _t(n)
    u = pa.table({"j": pa.array([0, 1, 2, 2], pa.int64()), "k": pa.array([0, 1, 2, 3], pa.int64()),
                  "name": pa.array(list("abcd"))})
    exp = duckdb.sql(q).arrow()
    if isinstance(exp, pa.RecordBatchReader):
        exp = exp.read_all()
    got = mx.sql(q, t=t, u=u).compute(device=device)
    assert got.column_names == exp.column_names
    assert _norm(got) == _norm(exp)


def test_null_key_group_counts_its_rows():
    t = pa.table({"k": pa.array([1, None, 1, None, 2], pa.int64())})
    got = mx.sql("SELECT k, COUNT(*) AS n FROM t GROUP BY k", t=t).compute(device="cpu").to_pydict()
    assert sorted(zip(got["k"], got["n"]), key=lambda r: (r[0] is None, r[0])) == [(1, 2), (2, 1), (None, 2)]
