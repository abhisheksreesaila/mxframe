"""F-096 review: a window expression (`.over(...)`, spec `(list, list, bool)`) must have a hashable signature,
because auto mode keys the GPU-failure memory by `plan.signature()`. A LEFT JOIN with a condition on both tables
puts windows into a SQL plan, and any big DataFrame window hit the same TypeError on main."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import auto_mode
from mxframe.lazy_expr import col, row_number


def test_window_signature_is_hashable():
    e = col("v").max().over("g", order_by="o")
    hash(e.signature())
    hash(row_number().over("g").signature())
    assert e.signature() != col("v").max().over("h", order_by="o").signature()      # the spec is part of it


@pytest.fixture
def auto_decides_even_for_small_inputs(monkeypatch):
    monkeypatch.setattr(auto_mode, "SMALL_QUERY_ROWS", 0)           # the device chooser sees every query


def test_dataframe_window_under_auto(auto_decides_even_for_small_inputs):
    t = pa.table({"g": pa.array([1, 1, 2, 2, 2], pa.int64()), "v": pa.array([5, 7, 1, 9, 3], pa.int64())})
    got = mx.from_arrow(t).with_columns(col("v").max().over("g").alias("m")).compute(device="auto")
    assert got.column("m").to_pylist() == [7, 7, 9, 9, 9]


def test_sql_left_join_residual_under_auto(auto_decides_even_for_small_inputs):
    t = pa.table({"id": pa.array([1, 2, 3, 4], pa.int64()), "k": pa.array([1, 2, 2, 9], pa.int64()),
                  "x": pa.array([10, 20, 5, 7], pa.int64())})
    u = pa.table({"k": pa.array([1, 2, 2], pa.int64()), "y": pa.array([5, 25, 10], pa.int64())})
    q = "SELECT t.id, u.y FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y ORDER BY t.id"
    got = mx.sql(q, t=t, u=u).compute(device="auto")
    assert [tuple(r.values()) for r in got.to_pylist()] == [(1, 5), (2, 10), (3, None), (4, None)]


def test_big_sql_left_join_residual_under_auto_matches_duckdb(tmp_path):
    """Past the real small-query threshold (200,000 rows), from a Parquet file, default thresholds."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(3)
    n = 300_000
    t = pa.table({"k": pa.array(rng.integers(0, 50_000, n), pa.int64()), "x": pa.array(rng.integers(0, 100, n), pa.int64())})
    u = pa.table({"k": pa.array(np.arange(50_000), pa.int64()), "y": pa.array(rng.integers(0, 100, 50_000), pa.int64())})
    pq.write_table(t, tmp_path / "t.parquet")
    q = "SELECT COUNT(*) AS n, COUNT(u.y) AS m, SUM(u.y) AS s FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y"
    con = duckdb.connect()
    con.register("t", t)
    con.register("u", u)
    exp = con.execute(q).to_arrow_table().to_pylist()
    got = mx.sql(q, t=mx.scan_parquet(str(tmp_path / "t.parquet")), u=u).compute(device="auto").to_pylist()
    assert got == exp


def test_big_dataframe_window_under_auto(tmp_path):
    import pyarrow.parquet as pq
    n = 300_000
    t = pa.table({"g": pa.array(np.arange(n) % 1000, pa.int64()), "v": pa.array(np.arange(n), pa.int64())})
    pq.write_table(t, tmp_path / "w.parquet")
    got = mx.scan_parquet(str(tmp_path / "w.parquet")).with_columns(col("v").max().over("g").alias("m")).compute(device="auto")
    assert got.column("m").to_pylist()[:3] == [n - 1000, n - 999, n - 998]
