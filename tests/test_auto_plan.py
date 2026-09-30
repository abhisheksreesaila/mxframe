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


# ── review fixes ─────────────────────────────────────────────────────────────

class MinMaxCalib(Calib):
    """GPU min/max unusable (as calibration reports above 8192 groups)."""

    def estimate(self, op, device, rows, card=None, **shape):
        if device == "gpu" and op in ("group_min", "group_max"):
            return float("inf")
        return super().estimate(op, device, rows, card=card, **shape)


def test_any_unusable_gpu_operator_makes_the_gpu_plan_impossible():
    left, right = table(5000), table(5000)
    plan = (LazyFrame(left).join(LazyFrame(right), on="k")
            .groupby("k").agg(col("v").max().alias("m")).plan)
    d = am.choose_device(plan, calib=MinMaxCalib(join_cpu=5000.0, join_gpu=1.0), gpu=Gpu())
    assert d.device == "cpu"
    assert d.estimate.gpu_ms == float("inf")


def test_known_broken_gpu_op_keeps_query_on_cpu(monkeypatch):
    monkeypatch.setitem(am.KNOWN_BROKEN_GPU, "sort", "pretend the GPU sort is broken")
    left, right = table(5000), table(5000)
    plan = LazyFrame(left).join(LazyFrame(right), on="k").sort("k").plan
    d = am.choose_device(plan, calib=Calib(join_cpu=5000.0, join_gpu=1.0), gpu=Gpu())
    assert d.device == "cpu"
    assert any(l.op == "sort" and l.gpu_ms == float("inf") for l in d.estimate.lines)


def test_gpu_sort_is_priced_now_that_it_is_fixed():
    assert "sort" not in am.KNOWN_BROKEN_GPU
    left, right = table(5000), table(5000)
    plan = LazyFrame(left).join(LazyFrame(right), on="k").sort("k").plan
    d = am.choose_device(plan, calib=Calib(join_cpu=5000.0, join_gpu=1.0), gpu=Gpu())
    assert all(l.gpu_ms != float("inf") for l in d.estimate.lines if l.op == "sort")


def test_string_key_group_count_is_sampled_not_guessed():
    n = 20_000
    t = pa.table({"a": pa.array(["A", "N", "R", "A"] * (n // 4)),
                  "b": pa.array(["F", "O"] * (n // 2)),
                  "v": pa.array(np.ones(n, dtype=np.float32))})
    est = am.estimate_plan(LazyFrame(t).groupby("a", "b").agg(col("v").sum().alias("s")).plan, Calib(), Gpu())
    (g,) = est.lines
    assert "~6 groups" in g.note  # 3 x 2 sampled distinct values


def test_join_output_rows_do_not_depend_on_side_order():
    small, big = table(1000), table(9000)
    a = am.estimate_plan(LazyFrame(small).join(LazyFrame(big), on="k").groupby("k").agg(col("v").sum()).plan, Calib(), Gpu())
    b = am.estimate_plan(LazyFrame(big).join(LazyFrame(small), on="k").groupby("k").agg(col("v").sum()).plan, Calib(), Gpu())
    assert a.lines[-1].rows == b.lines[-1].rows == 9000


def test_global_agg_copies_the_unfiltered_input(monkeypatch):
    monkeypatch.setattr(am, "KNOWN_BROKEN_GPU", {})
    t = table(10_000)
    est = am.estimate_plan(LazyFrame(t).filter(col("d") > lit(3)).groupby().agg(col("v").sum().alias("s")).plan,
                           Calib(), Gpu())
    g = est.lines[-1]
    assert g.op == "global_agg"
    assert g.h2d_bytes >= 10_000 * 4  # engine runs masked kernels over all rows


def test_sticky_gpu_error_disables_gpu_for_the_process(monkeypatch):
    import mxframe.custom_ops as co
    real = co.CustomOpsCompiler.compile_and_run

    def fail_on_gpu(self, plan, **kw):
        if self._session_device == "gpu":
            raise RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS")
        return real(self, plan, **kw)

    monkeypatch.setattr(co.CustomOpsCompiler, "compile_and_run", fail_on_gpu)
    am.forget_gpu_failures()
    from mxframe import gpu_stream
    if not gpu_stream.cuda_available():
        pytest.skip("needs a GPU to reach the GPU compiler")
    lf = LazyFrame(table(3000)).groupby("k").agg(col("v").sum().alias("s"))
    want = lf.compute(device="cpu")
    with pytest.warns(RuntimeWarning):
        got = lf.compute(device="auto", placement={"group_agg": "gpu"})
    assert got.sort_by("k").equals(want.sort_by("k"))
    other = LazyFrame(table(3000)).groupby("d").agg(col("v").sum().alias("s"))
    d = am.choose_device(other.plan, calib=Calib(), gpu=Gpu(), placement={"group_agg": "gpu"})
    assert d.device == "cpu" and "disabled" in d.reason
    am.forget_gpu_failures()


def test_no_gpu_explain_prints_na():
    d = am.choose_device(grouped(table(1000)), calib=Calib(has_gpu=False), gpu=Gpu(available=False))
    assert "gpu=n/a" in d.explain()


def test_group_count_of_sorted_keys_is_not_undercounted():
    n = 2_000_000
    keys = np.repeat(np.arange(n // 4, dtype=np.int32), 4)       # sorted, 500K groups
    t = pa.table({"k": keys, "v": np.ones(n, dtype=np.float32)})
    est = am.estimate_plan(LazyFrame(t).groupby("k").agg(col("v").min().alias("m")).plan, Calib(), Gpu())
    card = int(est.lines[0].note.split("~")[1].split(" ")[0].replace(",", ""))
    assert card >= 250_000


def test_placement_override_cannot_select_a_broken_gpu_path(monkeypatch):
    monkeypatch.setitem(am.KNOWN_BROKEN_GPU, "sort", "pretend the GPU sort is broken")
    left, right = table(5000), table(5000)
    plan = LazyFrame(left).join(LazyFrame(right), on="k").sort("k").plan
    d = am.choose_device(plan, calib=Calib(), gpu=Gpu(), placement={"join": "gpu"})
    assert d.device == "cpu" and "unusable" in d.reason


def test_only_device_errors_disable_the_gpu():
    assert am.is_sticky_gpu_error(RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS"))
    assert not am.is_sticky_gpu_error(MemoryError("x"))
    assert not am.is_sticky_gpu_error(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert not am.is_sticky_gpu_error(TypeError("unsupported dtype"))


def test_gpu_global_agg_is_blocked_until_verified():
    t = table(10_000)
    plan = LazyFrame(t).filter(col("d") > lit(3)).groupby().agg(col("v").sum().alias("s")).plan
    d = am.choose_device(plan, calib=Calib(cpu=1e4, gpu=0.001, link=1e6), gpu=Gpu())
    assert d.device == "cpu" and "unusable" in d.reason


def test_small_inputs_skip_calibration_entirely(monkeypatch):
    import mxframe.calibration as cal

    def boom():
        raise AssertionError("calibration loaded for a tiny query")

    monkeypatch.setattr(cal, "get_calibration", boom)
    d = am.choose_device(grouped(table(1000)))
    assert d.device == "cpu" and "small" in d.reason
