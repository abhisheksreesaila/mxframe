"""Whole-query device choice for device="auto": plan cost walk, decision, fallback, explain."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.auto_mode as am
from mxframe import LazyFrame, col, lit


class Calib:
    """cpu_ms/gpu_ms per million rows per op; link in GB/s; no gpu when has_gpu=False."""

    def __init__(self, cpu=1.0, gpu=0.1, join_cpu=10.0, join_gpu=0.5, link=10.0, has_gpu=True):
        self.cpu, self.gpu, self.join_cpu, self.join_gpu = cpu, gpu, join_cpu, join_gpu
        self.link, self.has_gpu = link, has_gpu

    def estimate(self, op, device, rows, card=None, **shape):
        if device == "gpu" and not self.has_gpu:
            return float("inf")
        if op in ("filter_take", "group_encode") and device == "gpu":
            return float("inf")
        per = {"cpu": self.cpu, "gpu": self.gpu}[device]
        if op == "join":
            per = self.join_cpu if device == "cpu" else self.join_gpu
        return 0.01 + per * rows / 1e6

    def estimate_gpu_join_total(self, left_rows, right_rows, out_rows=None, **kw):
        return 0.05 + self.join_gpu * (left_rows + right_rows) / 1e6

    def transfer_ms(self, nbytes, direction="h2d", pinned=False):
        return nbytes / (self.link * 1e9) * 1e3

    def gpu_init_cost_ms(self):
        return 200.0 if self.has_gpu else float("inf")


class Gpu:
    def __init__(self, available=True, remaining=0.0):
        self.available, self._r = available, remaining

    def remaining_init_ms(self):
        return self._r if self.available else float("inf")


def table(n, groups=4):
    rng = np.random.default_rng(0)
    return pa.table({
        "k": pa.array(rng.integers(0, groups, n).astype(np.int32)),
        "v": pa.array(rng.random(n).astype(np.float32)),
        "d": pa.array(rng.integers(0, 100, n).astype(np.int32)),
    })


def grouped(t):
    return LazyFrame(t).filter(col("d") > lit(10)).groupby("k").agg(col("v").sum().alias("s")).plan


def test_no_gpu_picks_cpu():
    d = am.choose_device(grouped(table(1000)), calib=Calib(has_gpu=False), gpu=Gpu(available=False))
    assert d.device == "cpu" and "no gpu" in d.reason


def test_estimate_lists_every_costed_operator():
    est = am.estimate_plan(grouped(table(1000)), Calib(), Gpu())
    ops = [l.op for l in est.lines]
    assert ops == ["filter", "group_agg"]
    assert est.cpu_ms > 0 and est.gpu_ms > 0


def test_filter_cost_is_the_same_on_both_paths():
    est = am.estimate_plan(LazyFrame(table(1000)).filter(col("d") > lit(3)).plan, Calib(), Gpu())
    (f,) = est.lines
    assert f.cpu_ms == f.gpu_ms  # filters are evaluated on the host on either path


def test_cheap_cpu_aggregate_stays_on_cpu_because_of_copy():
    d = am.choose_device(grouped(table(10_000)), calib=Calib(cpu=0.2, gpu=0.1, link=10.0), gpu=Gpu())
    assert d.device == "cpu"


def test_join_heavy_plan_goes_to_gpu_when_it_wins():
    left, right = table(5000), table(5000)
    plan = LazyFrame(left).join(LazyFrame(right), on="k").groupby("k").agg(col("v").sum().alias("s")).plan
    d = am.choose_device(plan, calib=Calib(join_cpu=5000.0, join_gpu=1.0), gpu=Gpu())
    assert d.device == "gpu"
    assert any(l.op == "join" for l in d.estimate.lines)


def test_gpu_still_initialising_counts_against_it():
    left, right = table(5000), table(5000)
    plan = LazyFrame(left).join(LazyFrame(right), on="k").plan
    fast = am.choose_device(plan, calib=Calib(join_cpu=5000.0, join_gpu=1.0), gpu=Gpu())
    slow = am.choose_device(plan, calib=Calib(join_cpu=5000.0, join_gpu=1.0), gpu=Gpu(remaining=1e6))
    assert fast.device == "gpu" and slow.device == "cpu"


def test_prefer_and_placement_overrides():
    plan = grouped(table(1000))
    assert am.choose_device(plan, calib=Calib(), gpu=Gpu(), placement={"group_agg": "gpu"}).device == "gpu"
    assert am.choose_device(plan, calib=Calib(), gpu=Gpu(), prefer="cpu").device == "cpu"


def test_failed_shape_is_remembered():
    plan = grouped(table(1000))
    am.remember_gpu_failure(plan)
    try:
        d = am.choose_device(plan, calib=Calib(cpu=1e6), gpu=Gpu())
        assert d.device == "cpu" and "failed on gpu" in d.reason
    finally:
        am.forget_gpu_failures()


def test_explain_lines_mention_decision_and_both_estimates():
    d = am.choose_device(grouped(table(1000)), calib=Calib(), gpu=Gpu())
    text = d.explain()
    assert text.splitlines()[0].startswith("auto ->")
    assert "cpu=" in text and "gpu=" in text and "group_agg" in text


# ── integration through LazyFrame.compute ────────────────────────────────────

def test_auto_compute_matches_cpu_and_records_decision():
    t = table(20_000, groups=7)
    lf = LazyFrame(t).filter(col("d") > lit(10)).groupby("k").agg(col("v").sum().alias("s")).sort("k")
    got = lf.compute(device="auto")
    want = lf.compute(device="cpu")
    assert got.column("k").to_pylist() == want.column("k").to_pylist()
    np.testing.assert_allclose(got.column("s").to_numpy(), want.column("s").to_numpy(), rtol=1e-6)
    assert lf.last_auto_decision is not None
    assert lf.last_auto_decision.device in ("cpu", "gpu")


def test_gpu_failure_falls_back_to_cpu_with_one_warning(monkeypatch):
    t = table(5000)
    lf = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).sort("k")
    want = lf.compute(device="cpu")
    monkeypatch.setattr(am, "choose_device", lambda plan, **kw: am.AutoDecision("gpu", "forced by test", None))

    import mxframe.lazy_frame as lfm
    real = lfm._run_on_device

    def boom(plan, device, **kw):
        if device == "gpu":
            raise MemoryError("CUDA_ERROR_OUT_OF_MEMORY")
        return real(plan, device, **kw)

    monkeypatch.setattr(lfm, "_run_on_device", boom)
    am.forget_gpu_failures()
    with pytest.warns(RuntimeWarning, match="fell back to CPU"):
        got = lf.compute(device="auto")
    assert got.equals(want)
    assert lf.last_auto_decision.device == "cpu" and lf.last_auto_decision.fallback
    with pytest.raises(MemoryError):
        lf.compute(device="auto", strict=True)
    am.forget_gpu_failures()


def test_explain_auto_costs():
    lf = LazyFrame(table(1000)).groupby("k").agg(col("v").sum().alias("s"))
    text = lf.explain(device="auto", costs=True)
    assert "auto ->" in text and "group_agg" in text
