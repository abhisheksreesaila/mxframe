"""Co-processing for fused aggregates under device="auto": resident split + background upload."""
import gc
import time

import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit
from mxframe import coproc_fused as cf
from mxframe import pipeline_kernels as pk

needs_gpu = pytest.mark.skipif(not pk.gpu_available() or not pk.cpu_available(), reason="needs both pipeline libraries and a GPU")


def table(n=2_000_000, groups=5, seed=0):
    rng = np.random.default_rng(seed)
    return pa.table({
        "k": pa.array(rng.integers(0, groups, n).astype(np.int32)),
        "v": pa.array(rng.random(n).astype(np.float32)),
        "w": pa.array(rng.random(n)),
        "d": pa.array(rng.integers(0, 100, n).astype(np.int32)),
    })


def query(t):
    return (LazyFrame(t).filter(col("d") > lit(20)).groupby("k")
            .agg(col("v").sum().alias("s"), (col("v") * col("w")).sum().alias("p"),
                 col("w").max().alias("m"), col("v").count().alias("n"))
            .sort("k"))


def assert_same(a, b):
    assert a.column_names == b.column_names
    assert a.column("k").to_pylist() == b.column("k").to_pylist()
    for c in ("s", "p", "m", "n"):
        np.testing.assert_allclose(a.column(c).to_numpy(), b.column(c).to_numpy(), rtol=1e-9)


def test_policy_off_by_default_outside_auto():
    assert cf.current_policy() is None


def test_balanced_fraction():
    # cpu 60 ms, gpu 40 ms (4 of which fixed): (1-f)*60 == 4 + f*36  ->  f = 56/96
    assert cf.balanced_fraction(60.0, 40.0, gpu_alpha=4.0) == pytest.approx(56 / 96)
    assert cf.balanced_fraction(1.0, 500.0, gpu_alpha=4.0) == 0.0   # gpu hopeless
    assert cf.balanced_fraction(500.0, 5.0, gpu_alpha=0.0) == 1.0   # cpu hopeless


@needs_gpu
def test_background_upload_then_split_matches_cpu():
    cf.clear_residency()
    t = table()
    want = query(t).compute(device="cpu")
    lf = query(t)
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):   # 2M rows: balanced split would keep it all on CPU
        first = lf.compute(device="cpu")          # host data: CPU runs, upload starts in the background
        assert cf.wait_for_uploads(30)
        assert cf.resident_count() == 1
        second = lf.compute(device="cpu")         # resident now: split
    assert_same(first, want)
    assert_same(second, want)
    prov = lf.last_compile_provenance
    assert prov["path"] == "fused_pipeline_split" and prov["device"] == "cpu+gpu"
    assert prov["coproc"]["resident"] is True and prov["coproc"]["gpu_fraction"] == 0.5


def test_small_resident_query_stays_on_cpu_when_gpu_fixed_cost_dominates():
    assert cf.balanced_fraction(1.2, 2.3, gpu_alpha=cf.DEFAULT_GPU_ALPHA_MS) == 0.0


@needs_gpu
@pytest.mark.parametrize("f", [0.0, 0.3, 0.7, 1.0])
def test_fixed_fraction_results_identical(f):
    cf.clear_residency()
    t = table(600_000, groups=37)
    want = query(t).compute(device="cpu")
    with cf.policy(min_upload_bytes=0, gpu_fraction=f):
        query(t).compute(device="cpu")
        cf.wait_for_uploads(30)
        got = query(t).compute(device="cpu")
    assert_same(got, want)


@needs_gpu
def test_residency_freed_when_table_dies():
    cf.clear_residency()
    t = table(300_000)
    with cf.policy(min_upload_bytes=0):
        query(t).compute(device="cpu")
        cf.wait_for_uploads(30)
    assert cf.resident_count() == 1
    del t
    gc.collect()
    assert cf.resident_count() == 0


@needs_gpu
def test_small_inputs_are_not_uploaded():
    cf.clear_residency()
    t = table(10_000)
    with cf.policy():                              # default threshold
        query(t).compute(device="cpu")
        cf.wait_for_uploads(10)
    assert cf.resident_count() == 0


def test_auto_enables_policy_and_results_match():
    t = table(400_000)
    want = query(t).compute(device="cpu")
    got = query(t).compute(device="auto")
    assert_same(got, want)
