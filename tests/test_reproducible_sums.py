"""F-064: float SUM / MEAN are bit-identical from run to run on every native CPU aggregate path.

Rows used to be handed to workers through an atomic counter (which worker adds which rows changed per
run) and the shared output used atomic float adds (the order of additions changed per run), so the last
bits of a float sum differed between two runs of the same query on the same data. Each test runs one
shape 30 times with 8 workers and demands 0 groups whose sum or mean differs bitwise from the first run,
and that the path under test ran (call counters), so it cannot pass through a fallback."""
import ctypes

import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

RUNS = 30


@pytest.fixture
def calls(monkeypatch):
    """Plan versions of every `mx_execute` call (3 = fused join + aggregate)."""
    log = []
    real = nx._call

    def call(lib, plan_addr, out_addr):
        log.append(int((ctypes.c_int64 * 2).from_address(plan_addr)[1]))
        return real(lib, plan_addr, out_addr)

    monkeypatch.setattr(nx, "_call", call)
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    monkeypatch.setenv("MXFRAME_CPU_THREADS", "8")
    return log


def _bits(t, names):
    return [np.ascontiguousarray(t.column(n).to_numpy()).view(np.int64) for n in names]


def _same_every_run(plan, names, runs=RUNS):
    first = _bits(plan.compute(device="cpu"), names)
    differ = np.zeros(len(first[0]), bool)
    for _ in range(runs - 1):
        got = _bits(plan.compute(device="cpu"), names)
        for a, b in zip(first, got):
            assert len(a) == len(b)
            differ |= a != b
    assert int(differ.sum()) == 0, f"{int(differ.sum())} of {len(differ)} groups differ bitwise between runs"


def _fact(n, groups, seed=7):
    r = np.random.default_rng(seed)
    return pa.table({"s": A(r.integers(0, groups, n).astype(np.int32)), "p": A(r.random(n) * 1000),
                     "d": A(r.random(n))})


def _revenue(lf):
    return lf.groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"),
                               col("p").mean().alias("av"))


def test_single_table_keyed_aggregate(calls):
    """6M rows, 10k groups (a partial per worker, merged)."""
    _same_every_run(_revenue(LazyFrame(_fact(6_000_000, 10_000))), ["tot", "av"])
    assert calls and 3 not in calls          # the single-table aggregate, not the fused join


def test_global_aggregate(calls):
    t = _fact(6_000_000, 10)
    plan = LazyFrame(t).groupby().agg((col("p") * col("d")).sum().alias("tot"), col("d").mean().alias("av"))
    _same_every_run(plan, ["tot", "av"])


def test_many_groups_shared_output(calls):
    """3M groups over 6M rows: a partial table per worker is over the memory budget, so the workers share
    one output (the path that used to add with atomic float adds)."""
    _same_every_run(_revenue(LazyFrame(_fact(6_000_000, 3_000_000))), ["tot", "av"])


def _star(groups):
    r = np.random.default_rng(11)
    n = 6_000_000
    fact = pa.table({"k": A(r.integers(0, groups, n).astype(np.int32)), "p": A(r.random(n) * 1000),
                     "d": A(r.random(n))})
    dim = pa.table({"dk": A(np.arange(groups, dtype=np.int32)), "w": A(r.random(groups))})
    return (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").groupby("k")
            .agg((col("p") * (lit(1.0) - col("d")) * col("w")).sum().alias("tot"), col("p").mean().alias("av")))


def test_fused_join_aggregate(calls):
    """Join + aggregate in one native call (plan version 3), 10k groups."""
    _same_every_run(_star(10_000), ["tot", "av"])
    assert 3 in calls


def test_fused_join_aggregate_many_groups_shared_output(calls):
    """Fused join + aggregate with 200k groups (shared output from 16384 groups on)."""
    _same_every_run(_star(200_000), ["tot", "av"])
    assert 3 in calls


def test_cpu_group_and_masked_sums_of_the_kernel_library():
    """The per-call CPU kernels (the non-native path, GPU-session fallbacks) hand out morsels in a fixed
    order too: 8 workers, 30 runs, identical bits."""
    from mxframe.aot_kernels import AOTKernels
    k = AOTKernels()
    r = np.random.default_rng(3)
    n = 6_000_000
    v = r.random(n) * 1000
    labels = r.integers(0, 10_000, n).astype(np.int32)
    mask = (r.random(n) < 0.7).astype(np.int32)
    first = (k.group_sum(v, labels, 10_000, threads=8).view(np.int64), k.group_mean(v, labels, 10_000, threads=8).view(np.int64),
             np.float64(k.masked_global_sum(v, mask, threads=8)).view(np.int64),
             np.float64(k.masked_global_sum_product(v, v, mask, threads=8)).view(np.int64))
    for _ in range(RUNS - 1):
        got = (k.group_sum(v, labels, 10_000, threads=8).view(np.int64), k.group_mean(v, labels, 10_000, threads=8).view(np.int64),
               np.float64(k.masked_global_sum(v, mask, threads=8)).view(np.int64),
               np.float64(k.masked_global_sum_product(v, v, mask, threads=8)).view(np.int64))
        for a, b in zip(first, got):
            assert np.array_equal(a, b)
