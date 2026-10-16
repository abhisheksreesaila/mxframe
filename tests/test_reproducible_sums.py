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
    assert calls and 3 not in calls


def test_many_groups_shared_output(calls):
    """3M groups over 6M rows: a partial table per worker is over the memory budget, so the workers share
    one output (the path that used to add with atomic float adds)."""
    _same_every_run(_revenue(LazyFrame(_fact(6_000_000, 3_000_000))), ["tot", "av"])
    assert calls and 3 not in calls


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


def _tot(t, extra, threads, monkeypatch):
    """The `tot` column of a grouped revenue sum next to `extra` more aggregates, with `threads` workers."""
    monkeypatch.setenv("MXFRAME_CPU_THREADS", str(threads))
    aggs = [(col("p") * (lit(1.0) - col("d"))).sum().alias("tot")]
    aggs += [(col("p") * col("d") * lit(float(i + 2))).sum().alias(f"x{i}") for i in range(extra)]
    return _bits(LazyFrame(t).groupby("s").agg(*aggs).compute(device="cpu"), ["tot"])[0]


@pytest.mark.parametrize("groups", [300, 800, 10_000])
def test_a_sum_does_not_depend_on_the_other_aggregates(calls, monkeypatch, groups):
    """`agg(tot)` and `agg(tot, x0, x1)` give the same bits for `tot` (the number of accumulator lanes
    used to depend on the aggregate list)."""
    t = _fact(6_000_000, groups)
    base = _tot(t, 0, 8, monkeypatch)
    for extra in (2, 5):
        assert int((base != _tot(t, extra, 8, monkeypatch)).sum()) == 0


@pytest.mark.parametrize("groups", [300, 10_000, 200_000])
def test_a_sum_does_not_depend_on_the_thread_count(calls, monkeypatch, groups):
    """4, 8 and 16 workers give the same bits (200k groups: the staged many-groups path)."""
    t = _fact(6_000_000, groups)
    base = _tot(t, 0, 4, monkeypatch)
    for threads in (8, 16):
        assert int((base != _tot(t, 0, threads, monkeypatch)).sum()) == 0


def _canonical(t, groups, morsel=65536):
    """The defined order for more than 1024 groups: per morsel, each group's rows added in row order;
    then the morsels' sub-totals added in morsel order."""
    s = t.column("s").to_numpy()
    v = t.column("p").to_numpy() * (1.0 - t.column("d").to_numpy())
    out = np.zeros(groups)
    for lo in range(0, len(s), morsel):
        sub = np.zeros(groups)
        np.add.at(sub, s[lo:lo + morsel], v[lo:lo + morsel])
        out += sub
    return out.view(np.int64)


@pytest.mark.parametrize("groups", [10_000, 60_000, 200_000])
def test_the_per_morsel_table_path_and_the_staged_path_give_the_defined_order(calls, monkeypatch, groups):
    """10k and 60k groups use a table per morsel, 200k the staged path (tables over the budget): all equal
    the same NumPy reference of the defined order, bit for bit."""
    t = _fact(3_000_000, groups)
    assert int((_tot(t, 0, 8, monkeypatch) != _canonical(t, groups)).sum()) == 0


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
