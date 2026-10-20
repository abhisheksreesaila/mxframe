"""F-018: a composite key whose widest column is skewed must not be joined on that column alone."""
import numpy as np
import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from omxcore import LazyFrame, col
from omxcore import join_engine


def _i32(a):
    return pa.array(np.asarray(a, dtype=np.int32))


def test_skewed_widest_column_falls_back_to_hashing_the_pair(monkeypatch):
    rng = np.random.default_rng(21)
    n = 20_000
    a1 = np.where(rng.random(n) < 0.95, 0, rng.integers(0, 100_000, n))          # 95% on one value
    a2 = np.where(rng.random(n) < 0.95, 0, rng.integers(0, 100_000, n))
    left = pa.table({"a": _i32(a1), "b": _i32(rng.integers(0, 50, n)), "v": pa.array(rng.random(n))})
    right = pa.table({"a": _i32(a2), "b": _i32(rng.integers(0, 50, n)), "w": pa.array(rng.random(n))})

    seen = []
    real = join_engine._residual_split

    def spy(*args):
        out = real(*args)
        seen.append(out)
        return out

    monkeypatch.setattr(join_engine, "_residual_split", spy)
    monkeypatch.setenv("MXFRAME_STAR", "0")        # repeated keys on both sides: the star scan would expand them natively
    got = (LazyFrame(left).join(LazyFrame(right), on=["a", "b"])
           .groupby().agg(col("v").count().alias("n"), col("w").sum().alias("s")).compute(device="cpu"))
    con = duckdb.connect()
    con.register("l", left)
    con.register("r", right)
    ref = con.execute("select count(v) n, sum(w) s from l join r on l.a = r.a and l.b = r.b").to_arrow_table()
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), ref.column("s").to_pylist(), rtol=1e-9)
    assert seen and seen[-1] is None          # the one-column join would have produced ~1.6e8 candidate pairs
