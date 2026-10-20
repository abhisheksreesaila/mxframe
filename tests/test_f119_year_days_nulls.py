"""F-119: `extract(year from <int32 day-number column with NULLs>)` (TPC-H Parquet layout) and the same on a date32
column, in WHERE, in the SELECT list and in an aggregate. A NULL date is NULL, so its row is not kept. Expected
answers come from DuckDB over the same days as DATEs. cpu and gpu, native on and off."""
import datetime

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import pipeline_kernels as pk

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]
EPOCH = datetime.date(1970, 1, 1)


def _days(n, nulls):
    r = np.random.default_rng(119)
    days = r.integers(8000, 21000, n)                       # 1991 .. 2027
    days[: n // 3] = r.integers(19723, 20089, n // 3)       # a third in 2024
    return days, (r.random(n) < 0.2) if nulls else None


def _t(n, nulls):
    days, mask = _days(n, nulls)
    return pa.table({
        "i": pa.array(np.arange(n), pa.int64()),
        "v": pa.array(np.arange(n) % 7, pa.int64()),
        "dd": pa.array(days, pa.int32(), mask=mask),
        "dt": pa.array(days, pa.int32(), mask=mask).cast(pa.date32()),
    })


def _duck(q):
    """The same query over DuckDB DATEs: DuckDB has no EXTRACT on an integer, so the day column is added to 1970-01-01."""
    return q.replace("extract(year from dd)", "extract(year from date '1970-01-01' + dd)")


QUERIES = [
    "SELECT i FROM t WHERE extract(year from {c}) = 2024 ORDER BY i",
    "SELECT i FROM t WHERE extract(year from {c}) <> 2024 ORDER BY i",
    "SELECT i FROM t WHERE NOT (extract(year from {c}) = 2024) ORDER BY i",
    "SELECT i FROM t WHERE extract(year from {c}) >= 2000 AND extract(year from {c}) < 2010 ORDER BY i",
    "SELECT i FROM t WHERE extract(year from {c}) = 2024 OR v = 3 ORDER BY i",
    "SELECT i FROM t WHERE extract(year from {c}) IN (1995, 2024) ORDER BY i",
    "SELECT count(*) AS n, sum(v) AS s FROM t WHERE extract(year from {c}) = 2024",
    "SELECT v, count(*) AS n FROM t WHERE extract(year from {c}) = 2024 GROUP BY v ORDER BY v",
    "SELECT extract(year from {c}) AS y, count(*) AS n FROM t GROUP BY 1 ORDER BY 1",
]


@pytest.mark.parametrize("native", ["1", "0"])
@pytest.mark.parametrize("nulls", [False, True])
@pytest.mark.parametrize("n", [50, 20_000])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("col", ["dd", "dt"])
@pytest.mark.parametrize("q", QUERIES)
def test_extract_year_matches_duckdb(q, col, device, n, nulls, native, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    q = q.format(c=col)
    t = _t(n, nulls)                                         # noqa: F841 (DuckDB finds it by name)
    exp = duckdb.sql(_duck(q)).arrow()
    exp = exp.read_all() if isinstance(exp, pa.RecordBatchReader) else exp
    got = mx.sql(q, t=t).compute(device=device)
    assert got.column_names == exp.column_names
    assert got.to_pydict() == exp.to_pydict()


def test_null_day_is_not_kept():
    t = pa.table({"i": pa.array([0, 1, 2], pa.int64()), "dd": pa.array([19723, None, 19000], pa.int32())})
    got = mx.sql("SELECT i FROM t WHERE extract(year from dd) = 2024", t=t).compute(device="cpu").to_pydict()
    assert got == {"i": [0]}
