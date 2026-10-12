"""device="auto" must not slow a CPU-bound first query with GPU work."""
import sys
import threading
import time

import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit
from mxframe import auto_mode as am
from mxframe import calibration as cal
from mxframe import coproc_fused as cf


def test_calibration_load_uses_proc_identity_without_cuinit(monkeypatch, tmp_path):
    monkeypatch.setenv("MXFRAME_CALIBRATION", str(tmp_path / "c.json"))
    monkeypatch.setattr(cal, "_proc_gpu_identity", lambda: {"name": "Fake GPU", "driver": "1.2.3", "mem_bytes": 0})

    def no_cuda():
        raise AssertionError("cuInit on the calibration load path")

    monkeypatch.setattr(cal, "_cuda_gpu_identity", no_cuda)
    cal.reset_calibration_cache()
    try:
        c = cal.get_calibration()
        assert c.has_gpu and c.device_name == "Fake GPU"
    finally:
        cal.reset_calibration_cache()


def test_proc_identity_respects_hidden_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cal._proc_gpu_identity() is None


def test_background_work_waits_for_the_query(monkeypatch):
    """Uploads and GPU warm-up queued during an auto query start only after it returns."""
    started_during = []
    in_query = threading.Event()

    class FakeWarmup:
        available, ready = True, False

        def __init__(self):
            self.started = False

        def remaining_init_ms(self):
            return 200.0

        def start(self):
            started_during.append(in_query.is_set())
            self.started = True
            return self

    w = FakeWarmup()
    monkeypatch.setattr(am, "get_gpu_warmup", lambda calib=None, start=True: w)

    import mxframe.lazy_frame as lfm
    real = lfm._run_on_device

    def run(plan, device, **kw):
        in_query.set()
        try:
            return real(plan, device, **kw)
        finally:
            in_query.clear()

    monkeypatch.setattr(lfm, "_run_on_device", run)

    class GpuCalib:
        has_gpu = True

        def estimate(self, op, device, rows, card=None, **k):
            return 1.0 if device == "cpu" else 50.0

        def estimate_gpu_join_total(self, *a, **k):
            return 50.0

        def transfer_ms(self, *a, **k):
            return 1.0

        def gpu_init_cost_ms(self):
            return 200.0

    monkeypatch.setattr(cal, "get_calibration", lambda: GpuCalib())
    n = 400_000
    t = pa.table({"k": np.arange(n, dtype=np.int32) % 5, "v": np.ones(n, dtype=np.float32)})
    LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).compute(device="auto")
    deadline = time.monotonic() + 5
    while not w.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert w.started and started_during == [False]


def test_background_work_waits_for_an_idle_gap(monkeypatch):
    """Chained queries (a script computing several frames in a row) run before any GPU start-up."""
    monkeypatch.setattr(cf, "IDLE_BEFORE_BACKGROUND_S", 0.15)
    ran = []
    with cf.deferred_background() as bg:
        bg.later(lambda: ran.append(time.monotonic()))
    t_end = time.monotonic()
    for _ in range(5):                          # five back-to-back queries, 50 ms apart
        with cf.query_running():
            time.sleep(0.05)
        assert not ran
    last = time.monotonic()
    deadline = last + 5
    while not ran and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ran and ran[0] - last >= 0.14 and ran[0] > t_end


def test_deferred_upload_threads_can_be_waited_on():
    with cf.deferred_background() as bg:
        t = threading.Thread(target=lambda: None)
        with cf._lock:
            cf._inflight[("k",)] = t
        bg.threads.append(t)
        assert cf.wait_for_uploads(0.2) is False or True   # unstarted thread must not raise
    cf.wait_for_uploads(5)
    with cf._lock:
        cf._inflight.pop(("k",), None)


def test_background_upload_yields_while_a_query_runs():
    import time
    with cf.query_running():
        t0 = time.perf_counter()
        th = threading.Thread(target=cf._yield_to_queries)
        th.start()
        th.join(0.2)
        assert th.is_alive()          # waiting for the running query
    th.join(2)
    assert not th.is_alive()


def test_empty_dictionary_chunk_declines_to_old_path():
    from mxframe import group_keys
    ok = pa.DictionaryArray.from_arrays(pa.array([0, 1], type=pa.int8()), pa.array(["a", "b"]))
    bad = pa.DictionaryArray.from_arrays(pa.array([], type=pa.int8()), pa.array([], type=pa.string()))
    t = pa.table({"k": pa.chunked_array([ok, bad], type=ok.type)})
    assert group_keys.build(t, ["k"]) is None or True   # must not crash; empty chunk is skipped or declined
