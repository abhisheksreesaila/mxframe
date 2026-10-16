"""F-067: float SUM / MEAN are bit-identical from run to run on the GPU and across a CPU+GPU split, with
`reproducible=True` (F-064 did the native CPU paths).

The fused GPU kernel accumulated with float64 atomics when a query has more groups than fit per-thread
accumulators (shared-memory atomics folded into the output with global atomics, or per-row global atomics):
which thread adds first changed per run, so the last bits of a sum differed. Each test runs one shape
repeatedly and demands 0 groups whose sum or mean differs bitwise from the first run. Inputs are small
(2M rows) because the GPU is shared; the path under test is asserted through the provenance."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit
from mxframe import coproc_fused as cf
from mxframe import native_exec as nx
from mxframe import pipeline_kernels as pk
from mxframe.np_arrow import array as A

needs_gpu = pytest.mark.skipif(not pk.gpu_available() or not pk.cpu_available(),
                               reason="needs both pipeline libraries and a GPU")
RUNS = 12


def _fact(n, groups, seed=7):
    r = np.random.default_rng(seed)
    return pa.table({"s": A(r.integers(0, groups, n).astype(np.int32)), "p": A(r.random(n) * 1000),
                     "d": A(r.random(n))})


def _revenue(t):
    return LazyFrame(t).groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"),
                                         col("p").mean().alias("av"))


def _bits(t, names=("tot", "av")):
    order = np.argsort(t.column("s").to_numpy())
    return [np.ascontiguousarray(t.column(n).to_numpy()[order]).view(np.int64) for n in names]


def _differing_groups(run, runs=RUNS):
    first = _bits(run())
    differ = np.zeros(len(first[0]), bool)
    for _ in range(runs - 1):
        for a, b in zip(first, _bits(run())):
            assert len(a) == len(b)
            differ |= a != b
    return int(differ.sum()), len(differ)


# 200 groups: a shared-memory table per block (mode 1); 100k groups: per-row global atomics (mode 2)
@needs_gpu
@pytest.mark.parametrize("groups", [200, 100_000])
def test_gpu_group_by_gives_the_same_bits_every_run(groups):
    plan = _revenue(_fact(2_000_000, groups))
    bad, total = _differing_groups(lambda: plan.compute(device="gpu", reproducible=True))
    assert plan.last_compile_provenance["path"] == "fused_pipeline_gpu"
    assert bad == 0, f"{bad} of {total} groups differ bitwise between runs"


@needs_gpu
@pytest.mark.parametrize("groups", [3, 200, 100_000])
def test_gpu_reproducible_sums_are_close_to_the_exact_sum(groups):
    """The ordered combine still adds the right numbers (a lost block or a double-counted row would show)."""
    t = _fact(2_000_000, groups)
    got = _revenue(t).compute(device="gpu", reproducible=True)
    order = np.argsort(got.column("s").to_numpy())
    s = t.column("s").to_numpy()
    want = np.bincount(s, weights=t.column("p").to_numpy() * (1.0 - t.column("d").to_numpy()), minlength=groups)
    np.testing.assert_allclose(got.column("tot").to_numpy()[order], want[np.sort(got.column("s").to_numpy())], rtol=1e-9)
    cnt = np.bincount(s, minlength=groups)
    np.testing.assert_allclose(got.column("av").to_numpy()[order],
                               np.bincount(s, weights=t.column("p").to_numpy(), minlength=groups)[np.sort(got.column("s").to_numpy())] / cnt[np.sort(got.column("s").to_numpy())],
                               rtol=1e-9)


@needs_gpu
@pytest.mark.parametrize("groups", [1, 5])
def test_gpu_few_groups_and_global_aggregate_are_already_ordered(groups):
    """Modes 0 and 3 (a partial per thread, block rows merged in block order) were deterministic before
    F-067: pinned so the new switch cannot break them."""
    t = _fact(2_000_000, groups)
    if groups == 1:
        plan = LazyFrame(t).groupby().agg((col("p") * col("d")).sum().alias("tot"), col("d").mean().alias("av"))
        runs = [plan.compute(device="gpu", reproducible=True) for _ in range(RUNS)]
        for r in runs[1:]:
            for n in ("tot", "av"):
                assert r.column(n).to_numpy().view(np.int64).tolist() == runs[0].column(n).to_numpy().view(np.int64).tolist()
    else:
        bad, _ = _differing_groups(lambda: _revenue(t).compute(device="gpu", reproducible=True))
        assert bad == 0


@needs_gpu
def test_the_default_mode_keeps_the_atomics():
    """No flag: the many-groups kernel is the fast one, and still right (its last bits may differ per run)."""
    t = _fact(2_000_000, 100_000)
    a = _revenue(t).compute(device="gpu")
    b = _revenue(t).compute(device="gpu", reproducible=True)
    oa, ob = np.argsort(a.column("s").to_numpy()), np.argsort(b.column("s").to_numpy())
    np.testing.assert_allclose(a.column("tot").to_numpy()[oa], b.column("tot").to_numpy()[ob], rtol=1e-9)


def test_the_environment_variable_reaches_the_gpu_kernel(monkeypatch):
    monkeypatch.setenv("MXFRAME_REPRODUCIBLE", "1")
    assert nx.reproducible() is True


# ── a forced CPU+GPU split ───────────────────────────────────────────────────

def _resident_split(plan, fraction):
    """Compute `plan` under the co-processing policy with a fixed GPU share, after its columns are resident."""
    with cf.policy(min_upload_bytes=0, gpu_fraction=fraction):
        plan.compute(device="cpu", reproducible=True)       # host data: CPU runs, the upload starts
        assert cf.wait_for_uploads(60)
    return lambda: _compute_split(plan, fraction)


def _compute_split(plan, fraction):
    with cf.policy(min_upload_bytes=0, gpu_fraction=fraction):
        return plan.compute(device="cpu", reproducible=True)


@needs_gpu
@pytest.mark.parametrize("groups", [200, 20_000])
def test_forced_split_gives_the_same_bits_every_run(groups):
    cf.clear_residency()
    plan = _revenue(_fact(2_000_000, groups))
    run = _resident_split(plan, 0.5)
    bad, total = _differing_groups(run, runs=8)
    prov = plan.last_compile_provenance
    assert prov["path"] == "fused_pipeline_split" and prov["coproc"]["gpu_fraction"] == 0.5
    assert bad == 0, f"{bad} of {total} groups differ bitwise between runs"


@needs_gpu
def test_a_split_merges_the_gpu_share_then_the_cpu_share_in_row_order():
    """The split's bits equal the two shares computed alone and added GPU rows first, CPU rows second."""
    from mxframe.pipeline_program import merge_partials
    cf.clear_residency()
    t = _fact(2_000_000, 200)
    plan = _revenue(t)
    got = _resident_split(plan, 0.25)()
    prog, cc = _program_and_columns(plan, t)
    gids = t.column("s").to_numpy().astype(np.int32)
    k = int(round(t.num_rows * 0.25))
    tok = nx._reproducible.set(True)
    try:
        want = merge_partials(prog, [pk.run_pipeline_gpu(prog, cc, gids, 200, 0, k),
                                     pk.run_pipeline_cpu(prog, cc, gids, 200, k, t.num_rows)])
    finally:
        nx._reproducible.reset(tok)
    tot = got.column("tot").to_numpy()[np.argsort(got.column("s").to_numpy())]
    assert tot.view(np.int64).tolist() == want.vals[0].view(np.int64).tolist()


def _program_and_columns(plan, t):
    from mxframe import pipeline_exec
    from mxframe.plan_validation import validate_plan_or_raise  # noqa: F401  (plan is already valid)
    predicates, agg_node, table = pipeline_exec._table_of(plan.plan)
    return pipeline_exec._bind(predicates, agg_node, table)


# ── a view compared with its own maximum (the Q15 shape) ─────────────────────

@needs_gpu
@pytest.mark.parametrize("device", ["gpu", "auto"])
def test_a_view_compared_with_its_own_maximum_runs_its_aggregate_once(device, monkeypatch):
    """F-064's plan-level sharing already covers the Q15 shape on any device: the subquery and the outer plan
    read ONE materialized aggregate, so they cannot land on different devices (or different bits). The
    aggregate runs once per compute and the filter keeps exactly the maximum's group."""
    from mxframe import pipeline_exec
    t = _fact(2_000_000, 20_000)
    view = LazyFrame(t).groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"))
    calls = []
    real = pipeline_exec.try_fused_aggregate

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(pipeline_exec, "try_fused_aggregate", spy)
    for _ in range(5):
        calls.clear()
        got = view.filter(col("tot") == view.scalar(col("tot").max())).compute(device=device)
        assert got.num_rows == 1
        assert len(calls) <= 1
