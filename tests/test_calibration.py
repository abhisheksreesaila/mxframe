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
    monkeypatch.setenv("MXFRAME_NUM_THREADS", "7")
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
    monkeypatch.setenv("MXFRAME_NUM_THREADS", "16")
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
