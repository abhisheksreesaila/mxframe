"""F-002 review fixes: literal-only aggregates, empty-right left join, length validation, stale .so."""
import numpy as np
import pyarrow as pa
import pytest

from omxcore import LazyFrame, col, lit
from omxcore.aot_kernels import AOT_AVAILABLE, AOTKernels
from omxcore.lazy_frame import Scan

pytestmark = pytest.mark.skipif(not AOT_AVAILABLE, reason="libmxkernels_aot.so not built")


def test_aggregate_of_literals_reads_no_columns_grouped():
    t = pa.table({"k": pa.array([1, 1, 2, 2, 2], type=pa.int32()), "v": pa.array([1, 2, 3, 4, 5], type=pa.float32())})
    out = LazyFrame(Scan(t)).groupby("k").agg((lit(1.0) * lit(2.0)).sum().alias("s")).sort("k").compute(device="cpu")
    assert out.column("s").to_pylist() == [4.0, 6.0]


def test_aggregate_of_literals_global_and_filtered():
    t = pa.table({"v": pa.array([1, 2, 3, 4, 5], type=pa.float32())})
    out = LazyFrame(Scan(t)).groupby().agg((lit(1.0) * lit(2.0)).sum().alias("s")).compute(device="cpu")
    assert out.column("s")[0].as_py() == 10.0
    out = (LazyFrame(Scan(t)).filter(col("v") > lit(2.5)).groupby()
           .agg((lit(1.0) * lit(2.0)).sum().alias("s")).compute(device="cpu"))
    assert out.column("s")[0].as_py() == 6.0


def test_left_join_with_empty_right_does_not_match_key_zero():
    left = pa.table({"k": pa.array([0, 1, 2], type=pa.int64()), "a": pa.array([10, 11, 12], type=pa.int64())})
    right = pa.table({"k": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.int64())})
    out = LazyFrame(Scan(left)).join(LazyFrame(Scan(right)), on="k", how="left").compute(device="cpu")
    assert out.num_rows == 3
    assert out.column("b").to_pylist() == [None, None, None]


def test_length_mismatch_raises_instead_of_reading_past_buffers():
    aot = AOTKernels()
    v = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError):
        aot.group_sum(v, np.zeros(50, dtype=np.int32), 2)
    with pytest.raises(ValueError):
        aot.masked_global_sum(v, np.ones(10, dtype=np.int32))
    with pytest.raises(ValueError):
        aot.masked_global_sum_product(v, np.ones(10, dtype=np.float32), np.ones(100, dtype=np.int32))
    with pytest.raises(ValueError):
        aot.masked_global_min(v, np.ones(99, dtype=np.int32))


def test_stale_library_warns_once_and_falls_back(monkeypatch):
    import warnings
    from omxcore import custom_ops

    def boom(*a, **k):
        raise RuntimeError("stale kernel library")

    monkeypatch.setattr(custom_ops, "AOTKernels", boom)
    monkeypatch.setattr(custom_ops, "_AOT_WARNED", False, raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c1 = custom_ops.CustomOpsCompiler(device="cpu")
        c2 = custom_ops.CustomOpsCompiler(device="cpu")
    assert c1._aot is None and c2._aot is None
    assert sum("stale" in str(x.message) for x in w) == 1
    t = pa.table({"k": pa.array([1, 1, 2], type=pa.int32()), "v": pa.array([1, 2, 3], type=pa.float32())})
    out = LazyFrame(Scan(t)).groupby("k").agg(col("v").sum().alias("s")).sort("k").compute(device="cpu")
    assert out.column("s").to_pylist() == [3.0, 3.0]
