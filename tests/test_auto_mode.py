"""Placement decisions for device="auto" (cost model over a calibration)."""
import pytest

from mxframe.auto_mode import GpuWarmup, Placer


class FakeCalib:
    """Linear costs: cpu 1 ms per 1M rows, gpu 0.1 ms per 1M rows, link 10 GB/s."""

    def __init__(self, cpu_per_m=1.0, gpu_per_m=0.1, gpu_alpha=0.05, link_gbps=10.0, gpu_init_ms=300.0):
        self.cpu_per_m, self.gpu_per_m, self.gpu_alpha = cpu_per_m, gpu_per_m, gpu_alpha
        self.link_gbps, self.gpu_init_ms = link_gbps, gpu_init_ms

    def estimate(self, op, device, rows, **shape):
        if device == "cpu":
            return 0.01 + self.cpu_per_m * rows / 1e6
        return self.gpu_alpha + self.gpu_per_m * rows / 1e6

    def transfer_ms(self, nbytes, direction="h2d", pinned=False):
        return nbytes / (self.link_gbps * 1e9) * 1e3


class ReadyGpu:
    available = True

    def remaining_init_ms(self):
        return 0.0


class NoGpu:
    available = False

    def remaining_init_ms(self):
        return float("inf")


def test_no_gpu_always_cpu():
    p = Placer(FakeCalib(), gpu=NoGpu())
    d = p.place("group_agg", rows=100_000_000, bytes_in=800_000_000)
    assert d.device == "cpu"
    assert "no gpu" in d.reason


def test_tiny_input_stays_on_cpu():
    p = Placer(FakeCalib(), gpu=ReadyGpu())
    d = p.place("group_agg", rows=1_000, bytes_in=8_000)
    assert d.device == "cpu"


def test_transfer_cost_can_keep_work_on_cpu():
    # 10M rows * 80 B = 800 MB -> 80 ms over the link; cpu = 10 ms.
    p = Placer(FakeCalib(), gpu=ReadyGpu())
    d = p.place("group_agg", rows=10_000_000, bytes_in=800_000_000)
    assert d.device == "cpu"
    assert d.est["gpu"] > d.est["cpu"]
    assert d.bytes_h2d == 0


def test_resident_input_goes_to_gpu_when_faster():
    p = Placer(FakeCalib(), gpu=ReadyGpu())
    d = p.place("join_probe", rows=50_000_000, bytes_in=400_000_000, resident="gpu")
    assert d.device == "gpu"
    assert d.bytes_h2d == 0


def test_gpu_init_cost_counts_when_not_ready():
    class ColdGpu(ReadyGpu):
        def remaining_init_ms(self):
            return 5_000.0

    calib = FakeCalib(cpu_per_m=10.0)  # cpu 500 ms for 50M rows
    warm = Placer(calib, gpu=ReadyGpu()).place("join_probe", rows=50_000_000, bytes_in=4_000_000)
    cold = Placer(calib, gpu=ColdGpu()).place("join_probe", rows=50_000_000, bytes_in=4_000_000)
    assert warm.device in ("gpu", "split")
    assert cold.device == "cpu"


def test_split_chosen_when_both_devices_are_comparable():
    # cpu 100 ms, gpu ~ 5 ms compute + 80 ms copy -> both similar -> split beats either.
    calib = FakeCalib(cpu_per_m=10.0, gpu_per_m=0.5)
    p = Placer(calib, gpu=ReadyGpu())
    d = p.place("group_agg", rows=10_000_000, bytes_in=800_000_000, splittable=True)
    assert d.device == "split"
    assert 0.0 < d.gpu_fraction < 1.0
    assert d.est["split"] < min(d.est["cpu"], d.est["gpu"])
    assert d.bytes_h2d == pytest.approx(800_000_000 * d.gpu_fraction, rel=1e-6)


def test_not_splittable_never_splits():
    calib = FakeCalib(cpu_per_m=10.0, gpu_per_m=0.5)
    d = Placer(calib, gpu=ReadyGpu()).place("sort", rows=10_000_000, bytes_in=800_000_000)
    assert d.device in ("cpu", "gpu")


def test_overrides_and_prefer():
    calib = FakeCalib()
    forced = Placer(calib, gpu=ReadyGpu(), placement={"group_agg": "gpu"})
    assert forced.place("group_agg", rows=10, bytes_in=80).device == "gpu"
    fixed = Placer(calib, gpu=ReadyGpu(), gpu_fraction=0.25)
    d = fixed.place("group_agg", rows=10_000_000, bytes_in=800_000_000, splittable=True)
    assert d.device == "split" and d.gpu_fraction == 0.25


def test_override_to_gpu_without_gpu_falls_back_to_cpu():
    d = Placer(FakeCalib(), gpu=NoGpu(), placement={"group_agg": "gpu"}).place("group_agg", rows=10, bytes_in=80)
    assert d.device == "cpu"


def test_decisions_are_recorded_for_explain():
    p = Placer(FakeCalib(), gpu=ReadyGpu())
    p.place("filter", rows=1000, bytes_in=4000)
    p.place("group_agg", rows=10, bytes_in=80)
    lines = p.explain_lines()
    assert len(lines) == 2
    assert lines[0].startswith("filter") and "cpu" in lines[0] and "est" in lines[0]


def test_gpu_warmup_reports_ready_and_remaining():
    import threading

    gate = threading.Event()
    w = GpuWarmup(init_fn=lambda: gate.wait(5) or True, expected_ms=1000.0)
    w.start()
    assert w.available is True  # optimistic while initialising
    assert 0.0 < w.remaining_init_ms() <= 1000.0
    gate.set()
    assert w.wait(5) is True
    assert w.remaining_init_ms() == 0.0


def test_gpu_warmup_failure_means_unavailable():
    def boom():
        raise RuntimeError("no cuda")

    w = GpuWarmup(init_fn=boom, expected_ms=10.0)
    w.start()
    assert w.wait(5) is False
    assert w.available is False
    assert w.remaining_init_ms() == float("inf")
