"""F-125: an expression that yields NULL (modulo by zero, NULLIF, CASE without ELSE) is NULL for every aggregate
(count skips it, sum / avg / min / max ignore it) and forms one NULL group as a GROUP BY key, even when none of
the columns it reads holds a NULL. Expected answers come from DuckDB. cpu and gpu, native on and off, at a size
for the Arrow fallback and a size for the native / fused paths, with and without NULLs in the inputs."""
import math

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import pipeline_kernels as pk

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]


def _t(n, nulls):
    r = np.random.default_rng(125)
    mask = (lambda p: r.random(n) < p) if nulls else (lambda p: None)
    return pa.table({
        "g": pa.array(r.integers(0, 5, n), pa.int64(), mask=mask(0.1)),
        "a": pa.array(r.integers(0, 100, n), pa.int64(), mask=mask(0.1)),
        "z": pa.array(r.integers(0, 3, n), pa.int64(), mask=mask(0.1)),     # a third of the divisors are 0
        "f": pa.array(r.random(n) * 10, pa.float64(), mask=mask(0.1)),
    })


AGGS = "count({e}) AS c, sum({e}) AS s, avg({e}) AS v, min({e}) AS mn, max({e}) AS mx"
EXPRS = ["a % z", "a % 0", "nullif(a, 5)", "nullif(a, z)", "case when a > 8 then a end", "f % z",
         "case when a > 8 then f end", "(a % z) + 1", "(a % z) * 2", "coalesce(a % z, -1)", "nullif(a, 5) % z"]
QUERIES = (
    [f"SELECT {AGGS.format(e=e)} FROM t" for e in EXPRS]
    + [f"SELECT g, {AGGS.format(e=e)} FROM t GROUP BY g ORDER BY g" for e in EXPRS]
    + [f"SELECT g, {AGGS.format(e=e)} FROM t WHERE f > 2 GROUP BY g ORDER BY g" for e in ("a % z", "nullif(a, 5)")]
    + [
        "SELECT a % 0 AS k, count(*) AS n FROM t GROUP BY 1",
        "SELECT a % z AS k, count(*) AS n FROM t GROUP BY 1 ORDER BY 1",
        "SELECT a % z AS k, sum(f) AS s FROM t GROUP BY 1 ORDER BY 1",
        "SELECT a % z AS k, g, count(*) AS n, sum(f) AS s FROM t GROUP BY 1, 2 ORDER BY 1, 2",
        "SELECT g, a % z AS k, count(*) AS n, count(a % z) AS c FROM t GROUP BY 1, 2 ORDER BY 1, 2",
        "SELECT nullif(a, 5) AS k, count(*) AS n FROM t WHERE a < 8 GROUP BY 1 ORDER BY 1",
        "SELECT case when a > 90 then g end AS k, count(*) AS n, count(a) AS c FROM t GROUP BY 1 ORDER BY 1",
        "SELECT f % z AS k, count(*) AS n FROM t WHERE a < 3 GROUP BY 1 ORDER BY 1",
    ])


def _rows(tbl):
    d = tbl.to_pydict()
    rows = zip(*[d[c] for c in tbl.column_names])
    return [tuple("nan" if isinstance(x, float) and math.isnan(x) else x if x is None or isinstance(x, str)
                  else round(float(x), 6) for x in row) for row in rows]


@pytest.mark.parametrize("native", ["1", "0"])
@pytest.mark.parametrize("nulls", [False, True])
@pytest.mark.parametrize("n", [60, 20_000])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", QUERIES)
def test_null_yielding_expressions_match_duckdb(q, device, n, nulls, native, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    t = _t(n, nulls)                                         # noqa: F841 (DuckDB finds it by name)
    exp = duckdb.sql(q).arrow()
    exp = exp.read_all() if isinstance(exp, pa.RecordBatchReader) else exp
    got = mx.sql(q, t=t).compute(device=device)
    assert got.column_names == exp.column_names
    g, e = _rows(got), _rows(exp)
    if "ORDER BY" not in q:
        g, e = sorted(g, key=repr), sorted(e, key=repr)
    assert g == e


def test_count_skips_the_remainder_of_a_zero_divisor():
    t = pa.table({"a": pa.array([5, 6, 7, 8, 9, 10], pa.int64()), "z": pa.array([0, 0, 0, 0, 0, 3], pa.int64())})
    got = mx.sql("SELECT count(a % z) AS c, sum(a % z) AS s FROM t", t=t).compute(device="cpu").to_pydict()
    assert got == {"c": [1], "s": [1]}


def test_group_by_modulo_by_zero_is_one_null_group():
    t = pa.table({"a": pa.array([5, 6, 7], pa.int64())})
    got = mx.sql("SELECT a % 0 AS k, count(*) AS n FROM t GROUP BY 1", t=t).compute(device="cpu").to_pydict()
    assert got == {"k": [None], "n": [3]}
