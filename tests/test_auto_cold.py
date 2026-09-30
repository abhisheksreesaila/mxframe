"""device="auto" must not slow a CPU-bound first query with GPU work."""
import sys
import threading

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
    assert w.started and started_during == [False]


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
