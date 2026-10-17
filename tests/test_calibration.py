"""F-003 calibration store.  Deterministic: fake probes advance a fake clock, no timings."""
import json
import math
import subprocess
import sys

import pytest

import mxframe
import mxframe.calibration as cal_mod
from mxframe.calibration import Calibration, Fit, Probe, calibrate, default_calibration

GPU = {"name": "Fake RTX", "driver": "999.1", "mem_bytes": 24 * 2**30}
OTHER_GPU = {"name": "Other GPU", "driver": "1.0", "mem_bytes": 8 * 2**30}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def fake_probe(clock, op, dev, alpha_ms, c_ms_per_row, g_ms_per_group=0.0, **kw):
    def make(ctx, n, card):
        cost = (alpha_ms + c_ms_per_row * n + g_ms_per_group * card) / 1e3
        def run():
            clock.t += cost
        return run
    return Probe(op, dev, make, **kw)


def fake_registry(clock):
    return {
        ("group_sum", "cpu"): fake_probe(clock, "group_sum", "cpu", 0.01, 2e-6),
        ("group_sum", "gpu"): fake_probe(clock, "group_sum", "gpu", 0.5, 4e-7),
        ("masked_sum", "cpu"): fake_probe(clock, "masked_sum", "cpu", 0.002, 1e-6),
        ("masked_sum", "gpu"): fake_probe(clock, "masked_sum", "gpu", 0.3, 1e-7),
    }


def fake_gpu_runner(clock, registry):
    """Stands in for the child process: yields the same items the worker would emit."""
    def run(sizes, budget_s, link_mb):
        items = [{"kind": "link",
                  "gbps": {"h2d_pageable": 10.0, "h2d_pinned": 22.0,
                           "d2h_pageable": 9.0, "d2h_pinned": 21.0},
                  "latency_ms": {"h2d_pageable": 0.05, "h2d_pinned": 0.02,
                                 "d2h_pageable": 0.05, "d2h_pinned": 0.02},
                  "pinned_available": True}]
        for (op, dev), p in registry.items():
            if dev == "gpu":
                pts = cal_mod._collect_points(p, object(), sizes, budget_s, clock, 0.0)
                items.append({"kind": "points", "op": op, "points": pts})
        return items
    return run


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MXFRAME_CALIBRATION", str(tmp_path / "calibration.json"))
    monkeypatch.setattr(cal_mod, "_gpu_identity", lambda: GPU)
    cal_mod.reset_calibration_cache()
    yield
    cal_mod.reset_calibration_cache()


def run_fake_calibration(identity=GPU, **kw):
    clock = Clock()
    reg = fake_registry(clock)
    return calibrate(registry=reg, gpu_runner=fake_gpu_runner(clock, reg),
                     gpu_init=lambda: {"total_ms": 250.0, "context_ms": 200.0, "first_kernel_ms": 50.0},
                     identity=identity, timer=clock, ctx=object(), **kw)


# ---------------------------------------------------------------- fitting ---
def test_calibrate_recovers_linear_fits():
    cal = run_fake_calibration(save=False)
    assert cal.source == "measured" and cal.has_gpu
    f = cal.fits["group_sum@cpu"]
    assert f.alpha == pytest.approx(0.01, rel=1e-3)
    assert f.c == pytest.approx(2e-6, rel=1e-3)
    assert cal.estimate("group_sum", "gpu", 1_000_000) == pytest.approx(0.5 + 0.4, rel=1e-3)
    assert cal.estimate("masked_sum", "cpu", 2_000_000) == pytest.approx(0.002 + 2.0, rel=1e-3)


def test_cardinality_term_is_fitted():
    clock = Clock()
    reg = {("group_sum", "cpu"): fake_probe(clock, "group_sum", "cpu", 0.01, 1e-6, 5e-5, uses_card=True)}
    cal = calibrate(registry=reg, identity=None, timer=clock, ctx=object(), save=False)
    f = cal.fits["group_sum@cpu"]
    assert f.g == pytest.approx(5e-5, rel=1e-2)
    assert cal.estimate("group_sum", "cpu", 1_000_000, card=100_000) > \
        cal.estimate("group_sum", "cpu", 1_000_000, card=10) + 4.0


def test_link_and_init_recorded_and_pinned_faster():
    cal = run_fake_calibration(save=False)
    assert cal.gpu_init_ms == 250.0 and cal.pinned_available
    nbytes = 256 * 2**20
    assert cal.transfer_ms(nbytes, "h2d", pinned=True) < cal.transfer_ms(nbytes, "h2d")
    assert cal.transfer_ms(nbytes, "h2d") == pytest.approx(0.05 + nbytes / 10e9 * 1e3)
    assert cal.transfer_ms(0, "d2h") == 0.0
    with pytest.raises(ValueError):
        cal.transfer_ms(1, "sideways")


def test_pinned_unavailable_falls_back_to_pageable():
    c = default_calibration(GPU)
    c.pinned_available = False
    assert c.transfer_ms(10**8, "h2d", pinned=True) == c.transfer_ms(10**8, "h2d", pinned=False)


def test_cpu_thread_count_recorded(monkeypatch):
    from mxframe import aot_kernels as _ak
    monkeypatch.setattr(_ak, "_cpu_threads", 7)
    assert run_fake_calibration(save=False).cpu_threads == 7


def test_failing_probe_keeps_defaults_and_records_error():
    clock = Clock()
    def boom(ctx, n, card):
        raise TypeError("kernel signature changed")
    reg = {("group_sum", "cpu"): Probe("group_sum", "cpu", boom),
           ("masked_sum", "cpu"): fake_probe(clock, "masked_sum", "cpu", 0.002, 1e-6)}
    cal = calibrate(registry=reg, identity=None, timer=clock, ctx=object(), save=False)
    assert "group_sum@cpu" in cal.errors and "TypeError" in cal.errors["group_sum@cpu"]
    assert cal.fits["group_sum@cpu"].n_points == 0            # still the built-in default
    assert cal.fits["masked_sum@cpu"].n_points >= 3


def test_gpu_worker_crash_keeps_partial_results_and_never_raises():
    clock = Clock()
    reg = fake_registry(clock)
    good = fake_gpu_runner(clock, reg)
    def crashing(sizes, budget, link_mb):
        items = good(sizes, budget, link_mb)[:2]                # link + first op, then it "dies"
        return items + [{"kind": "crash", "what": "gpu", "msg": "GPU probe process stopped: hang"}]
    cal = calibrate(registry=reg, gpu_runner=crashing, gpu_init=lambda: {"total_ms": 1.0},
                    identity=GPU, timer=clock, ctx=object(), save=False)
    assert "gpu" in cal.errors and cal.fits["group_sum@gpu"].n_points >= 3


def test_gpu_init_failure_recorded_not_raised():
    clock = Clock()
    reg = fake_registry(clock)
    def bad():
        raise RuntimeError("no context")
    cal = calibrate(registry=reg, gpu_runner=fake_gpu_runner(clock, reg), gpu_init=bad,
                    identity=GPU, timer=clock, ctx=object(), save=False)
    assert "gpu_init" in cal.errors and cal.gpu_init_ms == cal_mod._DEFAULT_GPU_INIT_MS


# ------------------------------------------------------------- no GPU path ---
def test_no_gpu_is_cpu_only_and_never_touches_gpu():
    clock = Clock()
    reg = fake_registry(clock)
    def must_not_run(*a):
        raise AssertionError("GPU runner called without a GPU")
    cal = calibrate(registry=reg, gpu_runner=must_not_run, identity=None, timer=clock,
                    ctx=object(), save=False)
    assert not cal.has_gpu
    assert not any(k.endswith("@gpu") for k in cal.fits)
    assert cal.estimate("group_sum", "gpu", 10**6) == math.inf
    assert cal.gpu_init_cost_ms() == math.inf
    assert math.isfinite(cal.estimate("group_sum", "cpu", 10**6))


def test_no_gpu_defaults_and_get_calibration(monkeypatch):
    monkeypatch.setattr(cal_mod, "_gpu_identity", lambda: None)
    cal_mod.reset_calibration_cache()
    c = cal_mod.get_calibration()
    assert not c.has_gpu and c.source == "defaults"
    assert cal_mod.estimate("join", "gpu", 1000) == math.inf


# ------------------------------------------------------------- persistence ---
def test_save_and_reload_roundtrip(tmp_path):
    cal = run_fake_calibration()
    p = tmp_path / "calibration.json"
    doc = json.loads(p.read_text())
    assert doc["schema"] == cal_mod.SCHEMA_VERSION and cal.key in doc["entries"]
    cal_mod.reset_calibration_cache()
    got = cal_mod.get_calibration()
    assert got.source == "measured"
    assert got.estimate("group_sum", "cpu", 5_000_000) == pytest.approx(cal.estimate("group_sum", "cpu", 5_000_000))
    assert got.gpu_init_ms == 250.0 and got.pinned_available


def test_get_calibration_is_cached_in_process():
    assert cal_mod.get_calibration() is cal_mod.get_calibration()


def test_env_override_path(tmp_path, monkeypatch):
    other = tmp_path / "sub" / "dir" / "c.json"
    monkeypatch.setenv("MXFRAME_CALIBRATION", str(other))
    assert cal_mod.calibration_path() == other
    run_fake_calibration()
    assert other.exists()


def test_missing_file_gives_defaults_without_crash():
    c = cal_mod.get_calibration()
    assert c.source == "defaults" and c.has_gpu
    assert c.estimate("group_sum", "cpu", 1000) < c.estimate("group_sum", "gpu", 1000)   # CPU wins small


@pytest.mark.parametrize("content", [
    "", "{not json", "[]", "null", '{"schema": 1}', '{"schema": 1, "entries": []}',
    '{"schema": 999, "entries": {}}', "\x00\x01\x02",
])
def test_corrupt_file_gives_defaults(tmp_path, content):
    (tmp_path / "calibration.json").write_text(content)
    c = cal_mod.get_calibration()
    assert c.source == "defaults"


def test_entry_with_bad_fields_gives_defaults(tmp_path):
    cal = run_fake_calibration()
    p = tmp_path / "calibration.json"
    doc = json.loads(p.read_text())
    doc["entries"][cal.key]["fits"]["group_sum@cpu"]["alpha"] = "oops"
    p.write_text(json.dumps(doc))
    cal_mod.reset_calibration_cache()
    assert cal_mod.get_calibration().source == "defaults"


def test_stale_mxframe_version_gives_defaults(monkeypatch):
    run_fake_calibration()
    monkeypatch.setattr(mxframe, "__version__", "99.0.0")
    cal_mod.reset_calibration_cache()
    assert cal_mod.get_calibration().source == "defaults"


def test_other_machine_gives_defaults_and_entries_coexist(tmp_path, monkeypatch):
    first = run_fake_calibration()
    monkeypatch.setattr(cal_mod, "_gpu_identity", lambda: OTHER_GPU)
    cal_mod.reset_calibration_cache()
    assert cal_mod.get_calibration().source == "defaults"
    second = run_fake_calibration(identity=OTHER_GPU)
    keys = set(json.loads((tmp_path / "calibration.json").read_text())["entries"])
    assert keys == {first.key, second.key}


def test_changed_cpu_thread_count_invalidates_cpu_fits_only(monkeypatch):
    run_fake_calibration()
    from mxframe import aot_kernels as _ak
    monkeypatch.setattr(_ak, "_cpu_threads", 16)
    cal_mod.reset_calibration_cache()
    c = cal_mod.get_calibration()
    assert c.fits["group_sum@cpu"].n_points == 0          # back to default
    assert c.fits["group_sum@gpu"].n_points >= 3          # GPU fits kept
    assert c.cpu_threads == 16


def test_save_to_unwritable_path_does_not_raise(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("MXFRAME_CALIBRATION", str(blocker / "calibration.json"))
    assert default_calibration(GPU).save() is False


# ------------------------------------------------------------------ observe ---
def test_observe_below_min_samples_changes_nothing():
    f = Fit(1.0, 1e-3)
    before = f.predict(1000)
    for _ in range(cal_mod.MIN_OBS - 1):
        f.observe(1000, 50.0)
    assert f.predict(1000) == before


def test_observe_refines_toward_truth_and_outweighs_prior():
    # prior is 10x too pessimistic; truth: 0.2 + 1e-4 * n
    f = Fit(2.0, 1e-3)
    truth = lambda n: 0.2 + 1e-4 * n
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    errs = [abs(f.predict(n) - truth(n)) / truth(n) for n in sizes]
    for i in range(24):
        n = sizes[i % 4]
        f.observe(n, truth(n))
    new = [abs(f.predict(n) - truth(n)) / truth(n) for n in sizes]
    assert max(new) < 0.2 and max(new) < max(errs) / 5


def test_observe_single_size_rescales_prior():
    f = Fit(1.0, 1e-3)
    for _ in range(10):
        f.observe(10_000, 22.0)               # prior predicts 11.0 at this size; truth is 2x
    assert f.predict(10_000) == pytest.approx(22.0, rel=0.15)
    assert f.predict(100_000) > 1.5 * (1.0 + 1e-3 * 100_000) * 0.9


def test_observe_sample_list_is_bounded_and_ignores_garbage():
    f = Fit(1.0, 1e-3)
    for i in range(cal_mod.MAX_OBS_SAMPLES * 3):
        f.observe(1000 + i, 5.0)
    f.observe(10, float("nan")); f.observe(-1, 1.0); f.observe(10, -3.0)
    assert len(f.obs) == cal_mod.MAX_OBS_SAMPLES


def test_calibration_observe_creates_unknown_op_and_persists_periodically(tmp_path):
    c = default_calibration(GPU)
    p = tmp_path / "calibration.json"
    for i in range(cal_mod.PERSIST_EVERY_N - 1):
        c.observe("new_op", "gpu", 1000 + i, 3.0)
    assert "new_op@gpu" in c.fits and not p.exists()      # not yet: persisted only occasionally
    c.observe("new_op", "gpu", 5000, 3.0)
    saved = json.loads(p.read_text())["entries"][c.key]
    assert len(saved["fits"]["new_op@gpu"]["obs"]) == cal_mod.PERSIST_EVERY_N


def test_observations_survive_reload(tmp_path):
    c = default_calibration(GPU)
    for i in range(6):
        c.observe("group_sum", "cpu", 10_000 * (i + 1), 1.0 + i, persist=False)
    assert c.save()
    cal_mod.reset_calibration_cache()
    got = cal_mod.get_calibration()
    assert len(got.fits["group_sum@cpu"].obs) == 6
    assert got.estimate("group_sum", "cpu", 30_000) == pytest.approx(c.estimate("group_sum", "cpu", 30_000))


def test_module_level_helpers_use_process_calibration():
    c = cal_mod.get_calibration()
    assert cal_mod.estimate("join", "cpu", 1000) == c.estimate("join", "cpu", 1000)
    assert cal_mod.transfer_ms(10**6, "h2d") == c.transfer_ms(10**6, "h2d")
    cal_mod.observe("join", "cpu", 1000, 2.0)
    assert len(c.fits["join@cpu"].obs) == 1


def test_unknown_op_uses_generic_fit():
    c = default_calibration(GPU)
    assert c.estimate("totally_new_op", "cpu", 10**6) > 0
    assert math.isfinite(c.estimate("totally_new_op", "gpu", 10**6))


def test_probe_registry_covers_engine_kernels():
    ops = {op for op, _ in cal_mod.PROBES}
    assert {"group_sum", "group_count", "group_min", "group_max", "masked_sum", "join",
            "gather", "sort_topk", "group_encode", "filter_take"} <= ops
    assert ("filter_take", "cpu") in cal_mod.PROBES


# ---------------------------------------------------------------- packaging ---
def test_import_mxframe_does_not_load_calibration_and_exports_calibrate():
    code = ("import sys, mxframe; "
            "assert 'mxframe.calibration' not in sys.modules; "
            "assert callable(mxframe.calibrate) and 'calibrate' in mxframe.__all__")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__('os').environ, "PYTHONPATH": ":".join(sys.path)})
    assert out.returncode == 0, out.stderr


def test_observed_outweighs_prior_after_a_handful_of_samples():
    f = Fit(2.0, 1e-3)                                    # prior says ~12 ms at 10k rows
    for n in (1_000, 10_000, 100_000, 10_000, 100_000, 1_000):
        f.observe(n, 0.2 + 1e-4 * n)                      # truth says ~1.2 ms
    assert abs(f.predict(10_000) - 1.2) < abs(f.predict(10_000) - 12.0)


# ============================ review fixes (contract: compute-only estimates) ===
# Measured bare-kernel GPU group_sum cost per million rows by group count.
CARD_COST = {8: 1.48, 1000: 10.5, 200_000: 2.85}


def _card_cost(card):
    if card <= 64:
        return 1.48
    if card <= 1024:
        return 10.5
    if card <= 8192:
        return 6.0
    return 2.85


def test_gpu_group_cost_tracks_cardinality_within_30_percent():
    clock = Clock()

    def make(ctx, n, card):
        cost = (0.1 + _card_cost(card) * n / 1e6) / 1e3

        def run():
            clock.t += cost
        return run
    reg = {("group_sum", "gpu"): Probe("group_sum", "gpu", make, uses_card=True, bucketed=True)}

    def runner(sizes, budget, link_mb):
        pts = cal_mod._collect_points(reg[("group_sum", "gpu")], object(), sizes, budget, clock, 0.0)
        return [{"kind": "points", "op": "group_sum", "points": pts}]
    cal = calibrate(registry=reg, gpu_runner=runner, gpu_init=lambda: {"total_ms": 1.0},
                    identity=GPU, timer=clock, ctx=object(), save=False)
    for card, per_m in CARD_COST.items():
        est = cal.estimate("group_sum", "gpu", 6_000_000, card=card)
        truth = 0.1 + per_m * 6
        assert abs(est - truth) / truth < 0.3, (card, est, truth)


def test_bucketed_fits_survive_reload_and_route_observations():
    c = default_calibration(GPU)
    c.fits["group_sum@gpu:g8"] = Fit(0.1, 1.5e-6, n_points=4)
    c.fits["group_sum@gpu:g200000"] = Fit(0.1, 3e-6, n_points=4)
    assert c.estimate("group_sum", "gpu", 1_000_000, card=8) == pytest.approx(0.1 + 1.5)
    assert c.estimate("group_sum", "gpu", 1_000_000, card=10**6) == pytest.approx(0.1 + 3.0)
    c.observe("group_sum", "gpu", 1000, 1.0, card=8, persist=False)
    assert len(c.fits["group_sum@gpu:g8"].obs) == 1
    assert c.save()
    cal_mod.reset_calibration_cache()
    assert "group_sum@gpu:g8" in cal_mod.get_calibration().fits


def test_op_aliases_resolve_to_measured_fits():
    c = default_calibration(GPU)
    for alias, real in [("group_agg", "group_sum"), ("join_probe", "join"),
                        ("sort", "sort_topk"), ("filter", "filter_take")]:
        assert c.estimate(alias, "cpu", 10**6) == c.estimate(real, "cpu", 10**6)
    assert c.estimate("group_agg", "gpu", 10**6, card=8) == c.estimate("group_sum", "gpu", 10**6, card=8)


def test_pipeline_op_is_reserved_with_default_fit():
    c = default_calibration(GPU)
    assert c.fits["pipeline@cpu"].n_points == 0 and c.estimate("pipeline", "cpu", 10**7) > 0
    assert math.isfinite(c.estimate("pipeline", "gpu", 10**7))


def test_unknown_op_logs_once_at_debug(caplog):
    import logging
    c = default_calibration(GPU)
    with caplog.at_level(logging.DEBUG, logger="mxframe.calibration"):
        c.estimate("mystery_op_xyz", "cpu", 10)
        c.estimate("mystery_op_xyz", "cpu", 10)
    assert sum("mystery_op_xyz" in r.getMessage() for r in caplog.records) == 1


def test_gpu_group_min_max_have_finite_cost_at_any_group_count():
    """F-012 replaced the shared-memory-only min/max kernel (it faulted above 8192 groups)
    with one that has a global-atomic path, so no group count is off limits any more."""
    c = default_calibration(GPU)
    for op in ("group_min", "group_max", "group_sum", "group_count"):
        for card in (8192, 8193, 16_000, 10**6, 1_500_000):
            assert math.isfinite(c.estimate(op, "gpu", 10**6, card=card))
        assert math.isfinite(c.estimate(op, "cpu", 10**6, card=10**6))


def test_blend_drops_group_term_when_refit_drops_it():
    f = Fit(0.1, 1e-6, g=1e-4)                       # prior: strong per-group cost
    for i in range(12):                              # constant card, varying rows
        n = 1000 * 2 ** (i % 6)
        f.observe(n, 0.1 + 1e-6 * n, card=50)
    assert f.predict(10_000, card=500_000) == pytest.approx(f.predict(10_000, card=10), rel=1e-6)


def test_module_observe_accepts_cardinality():
    cal_mod.observe("group_sum", "cpu", 1000, 2.0, cardinality=77)
    assert cal_mod.get_calibration().fits["group_sum@cpu"].obs[-1][1] == 77


def test_concurrent_saves_do_not_lose_entries(tmp_path):
    import threading
    p = tmp_path / "calibration.json"
    cals = [default_calibration({"name": f"GPU{i}", "driver": "1", "mem_bytes": 1}) for i in range(8)]
    ts = [threading.Thread(target=c.save) for c in cals]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert set(json.loads(p.read_text())["entries"]) == {c.key for c in cals}


# -- GPU child-process runner: hang / crash / progress -----------------------
def _child(script):
    return [sys.executable, "-c", script]


def test_runner_kills_hung_child_keeps_partial_output():
    import time
    script = ("import sys,time,json\n"
              "print(json.dumps({'kind':'link','gbps':{},'latency_ms':{},'pinned_available':False}),flush=True)\n"
              "time.sleep(60)\n")
    t0 = time.time()
    items = cal_mod._run_gpu_subprocess((1,), 1.0, (1,), item_timeout_s=1.0, child_cmd=_child(script))
    assert time.time() - t0 < 20
    assert items[0]["kind"] == "link" and items[-1]["kind"] == "crash" and "no progress" in items[-1]["msg"]


def test_runner_reports_crash_with_stderr_tail():
    script = ("import sys,json\n"
              "print(json.dumps({'kind':'launch','ms':1.0}),flush=True)\n"
              "sys.stderr.write('CUDA_ERROR_ILLEGAL_ADDRESS\\n'); sys.exit(3)\n")
    items = cal_mod._run_gpu_subprocess((1,), 1.0, (1,), child_cmd=_child(script))
    assert items[0]["kind"] == "launch"
    assert items[-1]["kind"] == "crash" and "ILLEGAL_ADDRESS" in items[-1]["msg"]


def test_runner_progress_lines_reset_watchdog():
    script = ("import time,json\n"
              "for i in range(4):\n"
              "    time.sleep(0.7); print(json.dumps({'kind':'progress','n':i}),flush=True)\n"
              "print(json.dumps({'kind':'launch','ms':2.0}),flush=True)\n")
    items = cal_mod._run_gpu_subprocess((1,), 1.0, (1,), item_timeout_s=1.2, child_cmd=_child(script))
    assert items[-1] == {"kind": "launch", "ms": 2.0}          # 2.8 s total, never 1.2 s silent


def test_collect_points_reports_progress_per_size():
    clock = Clock()
    seen = []
    p = fake_probe(clock, "x", "cpu", 0.01, 1e-6)
    cal_mod._collect_points(p, object(), (100, 1000, 10000), 1.0, clock, 0.0,
                            progress=lambda n: seen.append(n))
    assert seen == [100, 1000, 10000]


# -- GPU probes time bare kernels on device-resident buffers ------------------
class FakeCU:
    def __init__(self):
        self.h2d_calls = 0
        self.d2h_calls = 0
        self.n = 0

    def malloc(self, nbytes):
        self.n += 1
        return self.n

    def free(self, ptr):
        pass

    def sync(self):
        pass

    def h2d(self, *a):
        self.h2d_calls += 1

    def d2h(self, *a):
        self.d2h_calls += 1


class FakeGPU:
    def __init__(self):
        self._cu = FakeCU()
        self.bare = []

    def _upload(self, arr):
        return self._cu.malloc(arr.nbytes)

    def _ptr(self, p):
        return p

    def clear_buf_cache(self):
        pass

    def _gather_f32(self, *a):
        self.bare.append("gather")

    def _join_count_gpu(self, *a):
        self.bare.append("join_count")

    def _join_scatter_gpu(self, *a):
        self.bare.append("join_scatter")

    def gather_f32(self, *a):
        raise AssertionError("wrapper (host copies) must not be timed")

    def hash_join(self, *a):
        raise AssertionError("wrapper (host copies) must not be timed")


@pytest.mark.parametrize("op,expect", [("gather", ["gather"]), ("join", ["join_count", "join_scatter"])])
def test_gpu_gather_and_join_probes_time_bare_kernels_only(op, expect):
    ctx = cal_mod.ProbeCtx(20_000)
    ctx._gpu = FakeGPU()
    run = cal_mod.PROBES[(op, "gpu")].make(ctx, 10_000, 8)
    ctx._gpu.bare.clear()
    h2d, d2h = ctx._gpu._cu.h2d_calls, ctx._gpu._cu.d2h_calls
    run()
    assert ctx._gpu.bare == expect
    assert (ctx._gpu._cu.h2d_calls, ctx._gpu._cu.d2h_calls) == (h2d, d2h)     # no copies inside timed call


def test_join_host_step_is_modelled_separately_on_cpu():
    assert ("join_host", "cpu") in cal_mod.PROBES
    assert cal_mod.default_calibration(GPU).estimate("join_host", "cpu", 10**6) > 0


def test_module_docstring_states_compute_only_contract():
    assert "NO H2D/D2H" in cal_mod.__doc__


# ============================ fit accuracy (second review) ======================
# Recorded bare-kernel GPU timings on the RTX 3090 (rows -> ms).  Super-linear ops.
RECORDED = {
    "gather": [(10_000, .023), (100_000, .023), (1_000_000, .073), (3_000_000, .209),
               (6_000_000, .671), (12_000_000, 1.509), (30_000_000, 4.279)],
    "join": [(10_000, .033), (100_000, .043), (1_000_000, .227), (3_000_000, .49),
             (6_000_000, 1.389), (12_000_000, 4.381), (30_000_000, 15.014)],
    "sort_topk": [(10_000, .222), (100_000, .442), (1_000_000, 3.579), (3_000_000, 14.909),
                  (6_000_000, 43.208), (12_000_000, 122.738), (30_000_000, 431.456)],
    "masked_sum": [(10_000, .074), (100_000, .109), (1_000_000, .772), (3_000_000, 2.358),
                   (6_000_000, 4.462), (12_000_000, 8.941), (30_000_000, 22.131)],
}
QUICK_GRID = (10_000, 100_000, 1_000_000, 6_000_000, 30_000_000)


@pytest.mark.parametrize("op", sorted(RECORDED))
def test_fit_matches_recorded_points_at_large_sizes(op):
    """Fit on the quick-grid subset; must hit the biggest sizes (where the planner decides)."""
    pts = [[float(n), 8.0, ms] for n, ms in RECORDED[op] if n in QUICK_GRID]
    fits, err = cal_mod._fit_points(pts, False, False)
    f = fits[""]
    for n, ms in RECORDED[op]:
        if n >= 6_000_000:
            assert abs(f.predict(n) - ms) / ms < 0.3, (op, n, f.predict(n), ms)
    # interpolated size that was NOT in the fit
    ms12 = dict(RECORDED[op])[12_000_000]
    assert abs(f.predict(12_000_000) - ms12) / ms12 < 0.4


def test_fit_exponent_persists_and_linear_ops_stay_linear():
    pts = [[float(n), 8.0, ms] for n, ms in RECORDED["sort_topk"] if n in QUICK_GRID]
    f = cal_mod._fit_points(pts, False, False)[0][""]
    assert f.p > 1.2
    assert Fit.from_dict(f.to_dict()).predict(6_000_000) == pytest.approx(f.predict(6_000_000))
    lin = [[float(n), 8.0, 0.05 + 2e-6 * n] for n in QUICK_GRID]
    g = cal_mod._fit_points(lin, False, False)[0][""]
    assert g.p == 1.0 and g.predict(20_000_000) == pytest.approx(0.05 + 40.0, rel=0.02)


def test_quick_grid_reaches_30m_rows_for_gpu_gather_join_sort():
    assert max(cal_mod.QUICK_SIZES) == 30_000_000
    for op in ("gather", "join", "sort_topk"):
        assert (cal_mod.PROBES[(op, "gpu")].max_rows or 10**9) >= 30_000_000


def test_group_cost_interpolates_within_bucket_on_log_groups():
    clock = Clock()

    def per_m(card):                       # >8192 bucket: 7.5 ms/M at 10k falling to 2.8 at 200k
        if card <= 8192:
            return 1.5
        import math as m
        return 7.5 + (2.8 - 7.5) * m.log(card / 1e4) / m.log(20)

    def make(ctx, n, card):
        cost = (0.05 + per_m(card) * n / 1e6) / 1e3

        def run():
            clock.t += cost
        return run
    reg = {("group_sum", "gpu"): Probe("group_sum", "gpu", make, uses_card=True, bucketed=True)}

    def runner(sizes, budget, link_mb):
        pts = cal_mod._collect_points(reg[("group_sum", "gpu")], object(), sizes, budget, clock, 0.0)
        return [{"kind": "points", "op": "group_sum", "points": pts}]
    cal = calibrate(registry=reg, gpu_runner=runner, gpu_init=lambda: {"total_ms": 1.0},
                    identity=GPU, timer=clock, ctx=object(), save=False)
    for card in (10_000, 30_000, 100_000, 200_000):
        truth = 0.05 + per_m(card) * 30
        assert abs(cal.estimate("group_sum", "gpu", 30_000_000, card=card) - truth) / truth < 0.3, card


def test_join_total_prices_kernels_host_steps_and_round_trips():
    c = default_calibration(GPU)
    left, right = 6_000_000, 750_000
    expect = (c.estimate("join", "gpu", left) + c.estimate("join_host", "cpu", left)
              + c.transfer_ms(4 * left, "d2h") + c.transfer_ms(8 * left, "d2h")
              + c.transfer_ms(4 * left + 4 * right + 3 * 4 * right, "h2d"))
    assert c.estimate_gpu_join_total(left, right) == pytest.approx(expect)
    assert c.estimate("join_total", "gpu", left, right_rows=right) == pytest.approx(expect)
    assert c.estimate("join_total", "cpu", left) == c.estimate("join", "cpu", left)
    assert c.estimate_gpu_join_total(left, right) > c.estimate("join", "gpu", left)
    assert default_calibration(None).estimate_gpu_join_total(left, right) == math.inf


def test_join_host_is_cpu_only_and_join_build_not_aliased():
    c = default_calibration(GPU)
    assert c.estimate("join_host", "gpu", 10**6) == math.inf
    assert math.isfinite(c.estimate("join_host", "cpu", 10**6))
    assert "join_build" not in cal_mod.OP_ALIASES
    assert c.estimate("join_build", "cpu", 10**6) == c.fit("generic", "cpu").predict(10**6)


# ============================ round 3: end-to-end join, schema 2, tail extrapolation ===
# Recorded end-to-end AOTKernelsGPU.hash_join timings (left_rows, right_rows, ms), min of reps.
JOIN_E2E = [(100_000, 12_500, .71), (100_000, 100_000, 1.35), (1_000_000, 125_000, 8.4),
            (1_000_000, 1_000_000, 15.2), (6_000_000, 750_000, 33.9), (6_000_000, 6_000_000, 91.6),
            (30_000_000, 3_750_000, 165.6)]


def test_join_total_fit_matches_recorded_end_to_end_points_at_both_ratios():
    pts = [[float(a), float(b), ms] for a, b, ms in JOIN_E2E]
    fits, err = cal_mod._fit_points(pts, False, False, True)
    f = fits[""]
    for a, b, ms in JOIN_E2E:
        tol = 0.4 if a <= 1_000_000 else 0.3
        assert abs(f.predict(a, b) - ms) / ms < tol, (a, b, f.predict(a, b), ms)
    # a size/ratio pair that was not measured: 12M rows, 1:4
    assert 40 < f.predict(12_000_000, 3_000_000) < 130


def test_estimate_gpu_join_total_uses_measured_fit_and_out_rows_swaps_result_copy():
    c = default_calibration(GPU)
    pts = [[float(a), float(b), ms] for a, b, ms in JOIN_E2E]
    c.fits["join_total@gpu"] = cal_mod._fit_points(pts, False, False, True)[0][""]
    base = c.estimate_gpu_join_total(6_000_000, 750_000)
    assert base == pytest.approx(c.fits["join_total@gpu"].predict(6_000_000, 750_000))
    assert c.estimate("join_total", "gpu", 6_000_000, right_rows=750_000) == pytest.approx(base)
    more = c.estimate_gpu_join_total(6_000_000, 750_000, out_rows=24_000_000)
    assert more - base == pytest.approx(c.transfer_ms(8 * 24_000_000, "d2h") - c.transfer_ms(8 * 6_000_000, "d2h"))


def test_join_total_probe_is_end_to_end_hash_join_with_two_ratios():
    p = cal_mod.PROBES[("join_total", "gpu")]
    assert p.two_d and p.stat == "min"
    grid = p.grid(cal_mod.QUICK_SIZES)
    ratios = {round(r / n, 3) for n, r in grid}
    assert {0.125, 1.0} <= ratios and max(n for n, _ in grid) == 30_000_000


def test_join_total_probe_calls_the_wrapper():
    calls = []

    class G:
        def hash_join(self, left, right):
            calls.append((len(left), len(right)))
    ctx = cal_mod.ProbeCtx(10_000)
    ctx._gpu = G()
    cal_mod.PROBES[("join_total", "gpu")].make(ctx, 10_000, 1_250)()
    assert calls == [(10_000, 1_250)]


def test_schema_version_is_3_and_older_files_load_as_defaults(tmp_path):
    assert cal_mod.SCHEMA_VERSION == 3
    cal = run_fake_calibration()
    p = tmp_path / "calibration.json"
    doc = json.loads(p.read_text())
    doc["schema"] = 2      # v2 holds PyArrow sort_topk@cpu and default pipeline fits
    p.write_text(json.dumps(doc))
    cal_mod.reset_calibration_cache()
    assert cal_mod.get_calibration().source == "defaults"


def test_beyond_largest_measured_size_extrapolates_linearly_from_end_slope():
    f = Fit(0.0, 1e-9, p=1.4, n_max=30_000_000)
    slope = 1e-9 * 1.4 * 30_000_000 ** 0.4
    assert f.predict(60_000_000) == pytest.approx(f.predict(30_000_000) + slope * 30_000_000)
    assert f.predict(60_000_000) < 1e-9 * 60_000_000 ** 1.4              # power law would overshoot
    assert f.predict(10_000_000) == pytest.approx(1e-9 * 10_000_000 ** 1.4)   # inside range unchanged


def test_scaling_fit_records_n_max_and_roundtrips():
    pts = [[float(n), 8.0, ms] for n, ms in RECORDED["sort_topk"] if n in QUICK_GRID]
    f = cal_mod._fit_points(pts, False, False)[0][""]
    assert f.n_max == 30_000_000 and Fit.from_dict(f.to_dict()).n_max == 30_000_000
    assert f.predict(60_000_000) < f.alpha + f.c * 60_000_000 ** f.p


def test_time_call_min_stat_takes_fastest_sample():
    clock = Clock()
    costs = iter([0.5, 0.5, 0.3, 0.1, 0.2, 0.4, 0.4])

    def fn():
        clock.t += next(costs)
    assert cal_mod._time_call(fn, clock, 1.0, stat="min") == pytest.approx(100.0)


# ============================ F-013: probe refresh ==============================
def test_cpu_sort_probe_times_the_radix_sort_kernel():
    calls = []

    class C:
        def sort_indices(self, keys):
            calls.append(len(keys))
    ctx = cal_mod.ProbeCtx(10_000)
    ctx._cpu = C()
    cal_mod.PROBES[("sort_topk", "cpu")].make(ctx, 5_000, 8)()
    assert calls == [5_000]
    assert (cal_mod.PROBES[("sort_topk", "cpu")].max_rows or 10**9) >= 30_000_000   # no 10M cap any more


def test_no_separate_join_total_cpu_fit():
    """The planner prices CPU joins with `join`; `join_total` on the CPU is an alias, never a probe or a fit."""
    assert ("join_total", "cpu") not in cal_mod.PROBES
    c = default_calibration(GPU)
    assert "join_total@cpu" not in c.fits
    assert c.estimate("join_total", "cpu", 3_000_000) == c.estimate("join", "cpu", 3_000_000)
    c.observe("join", "cpu", 1_000_000, 5.0)
    assert "join_total@cpu" not in c.fits


def test_pipeline_probes_exist_on_both_devices_and_are_compute_only():
    assert ("pipeline", "cpu") in cal_mod.PROBES and ("pipeline", "gpu") in cal_mod.PROBES
    assert cal_mod.PROBES[("pipeline", "gpu")].h2d_bytes_per_row == 0.0    # inputs resident: no upload in the fit
    assert (cal_mod.PROBES[("pipeline", "gpu")].max_rows or 10**9) >= 30_000_000


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_pipeline_probe_runs_the_q1_shaped_program_correctly(dev):
    import numpy as np
    from mxframe import pipeline_kernels as pk
    if dev == "cpu" and not pk.cpu_available():
        pytest.skip("libmxpipeline_cpu.so not built")
    if dev == "gpu":
        try:
            ok = pk.gpu_available() and pk._get_cuda() is not None
        except Exception:
            ok = False
        if not ok:
            pytest.skip("no GPU or libmxpipeline_gpu.so not built")
    n = 50_000
    ctx = cal_mod.ProbeCtx(100_000)
    box = {}
    name = "run_pipeline_" + dev
    orig = getattr(pk, name)

    def spy(*a, **kw):
        box["p"] = orig(*a, **kw)
        return box["p"]
    setattr(pk, name, spy)
    try:
        cal_mod.PROBES[("pipeline", dev)].make(ctx, n, 4)()
    finally:
        setattr(pk, name, orig)
    prog, cols, gids = cal_mod._pipeline_case(ctx, n)
    ctx.release_gpu_buffers()
    by_name = dict(zip(prog.columns, cols))
    keep = by_name["l_shipdate"] <= 2436
    part = box["p"]
    assert part.vals.shape == (prog.n_slots, cal_mod.PIPELINE_GROUPS) and 0.9 < keep.mean() < 1.0
    np.testing.assert_array_equal(part.cnt, np.bincount(gids[keep], minlength=4))
    want_qty = np.bincount(gids[keep], weights=by_name["l_quantity"][keep].astype(np.float64), minlength=4)
    assert any(np.allclose(row, want_qty, rtol=1e-6) for row in part.vals)


def test_a_measured_native_join_replaces_every_default_of_its_name():
    """F-128 review: per-build-size defaults (`native_join@cpu:g...`) won over a measured plain fit in `_predict`,
    so a calibrated box still priced joins from the reference box. Measured fits replace defaults of the same op."""
    base = default_calibration(None)
    d = base.to_dict()
    d["fits"] = {"native_join@cpu": Fit(0.5, 10 * 5.2e-6, 0.0).to_dict()}      # a box 10x slower than the defaults
    loaded = Calibration.from_dict(d)
    assert not [k for k in loaded.fits if k.startswith("native_join@cpu:")]
    assert loaded.estimate("native_join", "cpu", 9e6, card=1e6) == pytest.approx(0.5 + 10 * 5.2 * 9, rel=1e-6)
    assert base.estimate("native_join", "cpu", 9e6, card=1e6) < 20                # the defaults still apply unmeasured
    cal = default_calibration(None)
    cal_mod._store_fits(cal, "native_group", "cpu", {"": Fit(0.1, 1e-6, 0.0), ":g1000": Fit(0.1, 2e-6, 0.0)})
    assert sorted(k for k in cal.fits if k.startswith("native_group@cpu")) == ["native_group@cpu", "native_group@cpu:g1000"]


def test_the_native_join_probe_measures_each_build_size():
    p = cal_mod.PROBES[("native_join", "cpu")]
    grid = p.grid(cal_mod.QUICK_SIZES)
    for m in cal_mod.NATIVE_JOIN_BUILDS:
        assert len({n for n, b in grid if b == m}) >= 3            # three sizes: a line per build size
    assert p.bucketed and max(n for n, _ in grid) <= p.max_rows
