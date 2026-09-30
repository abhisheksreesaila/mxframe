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


# ── review fixes ─────────────────────────────────────────────────────────────

def _two_tables_sharing_x(n=1_000_000):
    rng = np.random.default_rng(1)
    x = pa.array(np.ones(n, dtype=np.float32))
    a = pa.table({"x": x, "s": pa.array(np.where(rng.random(n) < 0.5, "a", "b"))})
    b = pa.table({"x": x, "s": pa.array(np.where(rng.random(n) < 0.8, "a", "b"))})
    return a, b


def _s_query(t):
    return LazyFrame(t).filter(col("s") == lit("a")).groupby().agg(col("x").sum().alias("t"))


@needs_gpu
def test_converted_columns_never_serve_a_stale_device_copy():
    cf.clear_residency()
    a, b = _two_tables_sharing_x()
    want_a = _s_query(a).compute(device="cpu").column("t")[0].as_py()
    want_b = _s_query(b).compute(device="cpu").column("t")[0].as_py()
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        for _ in range(4):
            for t, want in ((a, want_a), (b, want_b)):
                got = _s_query(t).compute(device="cpu").column("t")[0].as_py()
                cf.wait_for_uploads(30)
                assert got == pytest.approx(want, rel=1e-9)


@needs_gpu
def test_eviction_during_a_split_never_frees_memory_in_use(monkeypatch):
    cf.clear_residency()
    ts = [table(1_500_000, seed=i) for i in range(4)]
    wants = [query(t).compute(device="cpu") for t in ts]
    per_table = cf._input_bytes(cf_bind(ts[0]), None, ts[0].num_rows) + 4 * ts[0].num_rows
    monkeypatch.setattr(cf, "_budget_bytes", lambda: int(per_table * 1.5))   # room for one entry
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        for _ in range(3):
            for t, want in zip(ts, wants):
                assert_same(query(t).compute(device="cpu"), want)   # uploads evict older entries meanwhile
    cf.wait_for_uploads(30)


def cf_bind(t):
    from mxframe.pipeline_program import compile_pipeline
    prog = compile_pipeline([col("d") > lit(20)], [col("v").sum(), (col("v") * col("w")).sum(), col("w").max()], t)
    return prog.bind_chunks(t)


@needs_gpu
def test_gpu_error_inside_split_is_redone_on_cpu(monkeypatch):
    cf.clear_residency()
    t = table(800_000)
    want = query(t).compute(device="cpu")
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        query(t).compute(device="cpu")
        cf.wait_for_uploads(30)

        def boom(*a, **k):
            raise pk.GpuUnsupported("simulated")

        monkeypatch.setattr(pk, "run_pipeline_gpu", boom)
        assert_same(query(t).compute(device="cpu"), want)


@needs_gpu
def test_sticky_cuda_error_in_split_disables_gpu(monkeypatch):
    from mxframe import auto_mode as am
    cf.clear_residency()
    t = table(800_000)
    want = query(t).compute(device="cpu")
    try:
        with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
            query(t).compute(device="cpu")
            cf.wait_for_uploads(30)

            def boom(*a, **k):
                raise RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS")

            monkeypatch.setattr(pk, "run_pipeline_gpu", boom)
            assert_same(query(t).compute(device="cpu"), want)
        assert am._gpu_disabled is not None
    finally:
        am.forget_gpu_failures()


@needs_gpu
def test_derived_group_keys_do_not_upload():
    cf.clear_residency()
    t = table(600_000)
    q = LazyFrame(t).groupby(col("d").year().alias("y")).agg(col("v").sum().alias("s"))
    with cf.policy(min_upload_bytes=0):
        for _ in range(3):
            q.compute(device="cpu")
            cf.wait_for_uploads(30)
    assert cf.resident_count() == 0


@needs_gpu
@pytest.mark.parametrize("threads", ["1", "2", None])
def test_split_shapes_match_cpu_for_thread_counts(monkeypatch, threads):
    from mxframe import aot_kernels as ak
    from mxframe import pipeline_kernels as pkm
    if threads:
        monkeypatch.setattr(pkm, "default_threads", lambda: int(threads))
    cf.clear_residency()
    t = table(700_000, groups=50)
    shapes = [
        lambda: LazyFrame(t).filter(col("d") > lit(20)).groupby().agg(col("v").sum().alias("s"), col("w").max().alias("m")),
        lambda: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).filter(col("s") > lit(6000.0)).sort("k"),
        lambda: LazyFrame(t).filter(col("d") > lit(98)).groupby("k").agg(col("v").sum().alias("s")).sort("k"),
    ]
    for make in shapes:
        want = make().compute(device="cpu")
        with cf.policy(min_upload_bytes=0, gpu_fraction=0.4):
            make().compute(device="cpu")
            cf.wait_for_uploads(30)
            got = make().compute(device="cpu")
        assert got.column_names == want.column_names and got.num_rows == want.num_rows
        for c in got.column_names:
            np.testing.assert_allclose(np.asarray(got.column(c).to_pylist(), dtype=float),
                                       np.asarray(want.column(c).to_pylist(), dtype=float), rtol=1e-9)


@needs_gpu
def test_concurrent_callers_on_one_table():
    import threading
    cf.clear_residency()
    t = table(1_000_000)
    want = query(t).compute(device="cpu")
    errors = []

    def worker():
        try:
            with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
                for _ in range(5):
                    assert_same(query(t).compute(device="cpu"), want)
        except BaseException as e:
            errors.append(e)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    cf.wait_for_uploads(30)
    assert not errors, errors[0]


@needs_gpu
def test_profile_reports_both_devices_busy():
    cf.clear_residency()
    t = table()
    lf = query(t)
    lf.compute(device="auto", gpu_fraction=0.5)      # host data: cpu + background upload (above 64 MB? no: small)
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        lf.compute(device="cpu")
        cf.wait_for_uploads(30)
    lf.compute(device="auto", gpu_fraction=0.5, profile=True)
    d = lf.last_auto_decision
    assert d.coproc is not None and d.coproc["cpu_ms"] > 0 and d.coproc["gpu_ms"] > 0
    assert "co-processing" in d.explain()


def test_first_query_makes_no_cuda_call_on_the_query_thread(monkeypatch):
    import threading
    from mxframe import gpu_stream
    main = threading.get_ident()
    calls = []
    real = gpu_stream.mem_info

    def spy():
        calls.append(threading.get_ident())
        return real()

    monkeypatch.setattr(gpu_stream, "mem_info", spy)
    cf.clear_residency()
    t = table(600_000)
    with cf.policy(min_upload_bytes=0):
        query(t).compute(device="cpu")
        cf.wait_for_uploads(30)
    assert main not in calls


@needs_gpu
def test_failed_gpu_program_is_not_reuploaded_every_query(monkeypatch):
    cf.clear_residency()
    t = table(600_000)
    uploads = []
    real = pk.upload_columns
    monkeypatch.setattr(pk, "upload_columns", lambda *a, **k: uploads.append(1) or real(*a, **k))
    monkeypatch.setattr(pk, "run_pipeline_gpu", lambda *a, **k: (_ for _ in ()).throw(pk.GpuUnsupported("x")))
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        for _ in range(4):
            query(t).compute(device="cpu")
            cf.wait_for_uploads(30)
    assert len(uploads) == 1
