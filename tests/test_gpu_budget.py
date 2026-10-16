"""F-056 step 1: a query whose GPU working set is bigger than the GPU's memory finishes, correctly.

A fake small GPU (`MXFRAME_GPU_MEM_BUDGET`) stands in for a big data set. Pieces that do not fit run on
the CPU (pre-check); a piece that runs out of memory anyway is halved and retried (aggregates) or rerun on
the CPU, and the shape is remembered. Results are checked against DuckDB. The OOM is injected at the
driver (`cuMemAlloc` of the fused pipeline) and at the library boundary (F-055's `_fail`), so the tests are
deterministic on a shared GPU; `test_real_*` use the real GPU (opt-in).
"""
import os
import subprocess
import sys
import warnings

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, custom_ops, gpu_budget, gpu_stream
from mxframe import auto_mode as am
from mxframe.optimizer import optimize_plan
from mxframe.aot_kernels import get_aot_gpu

from test_gpu_kernel_errors import HOG, _fail

pytestmark = pytest.mark.skipif(not gpu_stream.cuda_available() or get_aot_gpu() is None, reason="no CUDA GPU")

N = 400_000


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MXFRAME_GPU_MEM_BUDGET", raising=False)
    gpu_budget.forget()
    am.forget_gpu_failures()
    custom_ops.clear_cache()               # a cached join result would skip the join piece
    yield
    gpu_budget.forget()
    am.forget_gpu_failures()


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(7)
    t = pa.table({"k": rng.integers(0, 50, N).astype(np.int32), "j": rng.integers(0, 1000, N).astype(np.int32),
                  "v": rng.integers(0, 100, N).astype(np.int32), "i": rng.permutation(N).astype(np.int32)})
    u = pa.table({"id": np.arange(1000, dtype=np.int32), "w": (np.arange(1000) % 7).astype(np.int32)})
    return t, u


def aggregate(t, u):
    return (LazyFrame(t).filter(col("v") > 10).groupby("k")
            .agg(col("v").sum().alias("s"), col("v").count().alias("c")).sort("k"),
            "SELECT k, sum(v)::DOUBLE AS s, count(v) AS c FROM t WHERE v > 10 GROUP BY k ORDER BY k")


def join(t, u):
    return (LazyFrame(t).join(LazyFrame(u), left_on="j", right_on="id").groupby("w")
            .agg(col("v").sum().alias("s")).sort("w"),
            "SELECT w, sum(v)::DOUBLE AS s FROM t JOIN u ON t.j = u.id GROUP BY w ORDER BY w")


def topk(t, u):
    return LazyFrame(t).sort("i", descending=True).limit(20), "SELECT * FROM t ORDER BY i DESC LIMIT 20"


def full_sort(t, u):
    return LazyFrame(t).sort("i"), "SELECT * FROM t ORDER BY i"


QUERIES = {"aggregate": aggregate, "join": join, "top-k": topk, "sort": full_sort}
PIECE = {"aggregate": "aggregate", "join": "join", "top-k": "sort", "sort": "sort"}


def reference(sql, t, u):
    con = duckdb.connect()
    con.register("t", t)
    con.register("u", u)
    return con.sql(sql).arrow().read_all()


def same(got, want):
    assert got.column_names == want.column_names
    for name in want.column_names:
        g, w = got.column(name).to_numpy(), want.column(name).to_numpy()
        assert np.array_equal(g.astype(np.float64), w.astype(np.float64)), name


def run(lf, device, monkeypatch, **kw):
    """compute() on `device`; under auto the decision is forced to the GPU, as a big warm GPU would choose."""
    if device == "auto":
        monkeypatch.setattr(am, "choose_device", lambda plan, **k: am.AutoDecision("gpu", "forced by test", None))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = lf.compute(device=device, **kw)
    return out, [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]


def pieces(lf):
    return lf.last_compile_provenance.get("gpu_pieces", "").split("; ") if lf.last_compile_provenance.get("gpu_pieces") else []


def test_the_queries_run_on_the_gpu_when_they_fit(tables):
    """Control: with no budget set no piece leaves the GPU (so the tests below do exercise a GPU path)."""
    from mxframe import aot_kernels
    for name, q in QUERIES.items():
        lf, sql = q(*tables)
        aot_kernels.gpu_call_log = log = []
        try:
            got = lf.compute(device="gpu")
        finally:
            aot_kernels.gpu_call_log = None
        same(got, reference(sql, *tables))
        assert pieces(lf) == [], name
        assert lf.last_compile_provenance["device"] in ("gpu",) or log, name


@pytest.mark.parametrize("device", ["gpu", "auto"])
@pytest.mark.parametrize("name", sorted(QUERIES))
def test_a_piece_over_the_budget_runs_on_the_cpu_with_the_right_result(tables, monkeypatch, name, device):
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "1M")
    lf, sql = QUERIES[name](*tables)
    got, warned = run(lf, device, monkeypatch)
    same(got, reference(sql, *tables))
    first = pieces(lf)[0]
    assert first.startswith(f"{PIECE[name]}: ran on CPU, needs about ")
    assert first.endswith("of GPU memory, over 90% of the 1.0 MB free")
    assert (len(warned) > 0) == (device == "gpu")          # an explicit gpu says so; auto just explains
    if device == "auto":
        assert first in lf.last_auto_decision.explain()


def test_explain_shows_why_a_piece_left_the_gpu(tables, monkeypatch):
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "1M")
    lf, _ = aggregate(*tables)
    lf.compute(device="gpu")
    text = lf.explain(include_runtime=True)
    assert "gpu_pieces=aggregate: ran on CPU, needs about" in text
    assert "of the 1.0 MB free" in text


def test_strict_raises_a_memory_error_with_the_estimate(tables, monkeypatch):
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "1M")
    lf, _ = aggregate(*tables)
    with pytest.raises(MemoryError, match=r"aggregate: needs about .* of GPU memory, over 90% of the 1.0 MB free"):
        lf.compute(device="gpu", strict=True)


def test_the_budget_is_per_gpu_pool(monkeypatch):
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "0:512M,1:2G")
    assert (gpu_budget.budget_override(0), gpu_budget.budget_override(1), gpu_budget.budget_override(2)) == \
        (512 << 20, 2 << 30, None)
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "64K")
    assert (gpu_budget.budget_override(0), gpu_budget.budget_override(1)) == (64 << 10, 64 << 10)
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "1M")
    assert gpu_budget.free_bytes(0) == 1 << 20


# --- out of memory anyway -------------------------------------------------------------------

def limit_cuda_alloc(monkeypatch, biggest):
    """The fused pipeline's driver allocations fail above `biggest` bytes, like a GPU with that much left."""
    from mxframe import pipeline_kernels as pk
    cu = pk._get_cuda()
    real = cu.malloc

    def malloc(nbytes):
        if nbytes > biggest:
            raise MemoryError(f"CUDA_ERROR_OUT_OF_MEMORY: cuMemAlloc({nbytes}) failed")
        return real(nbytes)
    monkeypatch.setattr(cu, "malloc", malloc)


@pytest.mark.parametrize("device", ["gpu", "auto"])
@pytest.mark.parametrize("fn", ["sum", "min", "max", "mean"])
def test_an_aggregate_that_runs_out_of_memory_is_halved_and_retried_on_the_gpu(tables, monkeypatch, device, fn):
    lf = (LazyFrame(tables[0]).filter(col("v") > 10).groupby("k")
          .agg(getattr(col("v"), fn)().alias("s"), col("v").count().alias("c")).sort("k"))
    sql = f"SELECT k, {fn if fn != 'mean' else 'avg'}(v)::DOUBLE AS s, count(v) AS c FROM t WHERE v > 10 GROUP BY k ORDER BY k"
    limit_cuda_alloc(monkeypatch, N * 3)           # a column of all rows is 1.6 MB; half of them fit
    got, warned = run(lf, device, monkeypatch)
    same(got, reference(sql, *tables))
    assert pieces(lf) == ["aggregate: ran out of GPU memory, retried on the GPU with half the rows per chunk"]
    assert warned == []


def test_an_aggregate_that_still_runs_out_of_memory_reruns_on_the_cpu_and_is_remembered(tables, monkeypatch):
    lf, sql = aggregate(*tables)
    with monkeypatch.context() as m:
        limit_cuda_alloc(m, N)                     # not even half the rows fit
        got, warned = run(lf, "gpu", monkeypatch)
    same(got, reference(sql, *tables))
    first = pieces(lf)[0]
    assert first.startswith("aggregate: ran on CPU, ran out of GPU memory (needed about ")
    assert lf.last_compile_provenance["path"] == "fused_pipeline_cpu"
    assert len(warned) == 1
    # the GPU is healthy again, but this shape is remembered: straight to the CPU
    got, _ = run(lf, "gpu", monkeypatch)
    same(got, reference(sql, *tables))
    assert pieces(lf) == ["aggregate: ran on CPU, this shape ran out of GPU memory before"]


def test_auto_remembers_the_query_shape_after_a_piece_ran_out_of_memory(tables, monkeypatch):
    lf, sql = aggregate(*tables)
    with monkeypatch.context() as m:
        limit_cuda_alloc(m, N)
        got, _ = run(lf, "auto", monkeypatch)
    same(got, reference(sql, *tables))
    monkeypatch.undo()
    decision = am.choose_device(optimize_plan(lf.plan).plan)
    assert (decision.device, decision.reason) == ("cpu", "this query shape failed on gpu before")


def test_a_remembered_shape_gets_another_gpu_try_once_much_more_memory_is_free(tables, monkeypatch):
    lf, sql = aggregate(*tables)
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "10M")
    with monkeypatch.context() as m:
        limit_cuda_alloc(m, N)
        run(lf, "gpu", monkeypatch)
    assert pieces(lf)[0].startswith("aggregate: ran on CPU, ran out of GPU memory")
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "14M")           # not 1.5x more yet
    run(lf, "gpu", monkeypatch)
    assert pieces(lf) == ["aggregate: ran on CPU, this shape ran out of GPU memory before"]
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "15M")           # 1.5x: the GPU again
    got, _ = run(lf, "gpu", monkeypatch)
    same(got, reference(sql, *tables))
    assert pieces(lf) == [] and lf.last_compile_provenance["path"] == "fused_pipeline_gpu"


def test_auto_tries_the_gpu_again_when_much_more_memory_is_free(tables, monkeypatch):
    lf, _ = aggregate(*tables)
    real_choose = am.choose_device
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "10M")
    with monkeypatch.context() as m:
        limit_cuda_alloc(m, N)
        run(lf, "auto", monkeypatch)
    plan = optimize_plan(lf.plan).plan
    monkeypatch.setattr(am, "choose_device", real_choose)
    assert am.choose_device(plan).reason == "this query shape failed on gpu before"
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "15M")
    assert am.choose_device(plan).reason != "this query shape failed on gpu before"


def test_another_query_is_not_pinned_to_the_cpu_by_a_remembered_shape(tables, monkeypatch):
    a, _ = aggregate(*tables)
    with monkeypatch.context() as m:
        limit_cuda_alloc(m, N)
        run(a, "gpu", monkeypatch)
    b = (LazyFrame(tables[0]).filter(col("v") > 10).groupby("k")        # same rows, groups and slots, other columns
         .agg(col("j").sum().alias("s"), col("j").count().alias("c")).sort("k"))
    run(b, "gpu", monkeypatch)
    assert pieces(b) == []


def test_forgetting_gpu_failures_forgets_remembered_pieces(tables, monkeypatch):
    lf, _ = aggregate(*tables)
    with monkeypatch.context() as m:
        limit_cuda_alloc(m, N)
        run(lf, "gpu", monkeypatch)
    am.forget_gpu_failures()
    run(lf, "gpu", monkeypatch)
    assert pieces(lf) == []


def test_a_join_output_past_the_kernel_limit_is_a_size_error_not_an_out_of_memory():
    def too_big():
        raise gpu_budget.JoinOutputTooLarge("join output would have 3,000,000,000 rows")
    with pytest.raises(gpu_budget.JoinOutputTooLarge, match="3,000,000,000"):
        gpu_budget.run_piece("join", 1, too_big, lambda: "cpu", shape=("x",))
    assert gpu_budget._remembered == {}


def test_a_string_filter_over_the_budget_runs_on_the_cpu(monkeypatch):
    """The GPU string kernels upload the string bytes (the likeliest out-of-memory): at the expression level."""
    names = np.array(["alpha", "beta", "gamma", "delta"])
    t = pa.table({"s": pa.array(names[np.arange(N) % 4]), "v": np.arange(N, dtype=np.int32)})
    comp = custom_ops.CustomOpsCompiler(device="gpu")
    exprs = [col("s") == "beta", col("s") != "beta", col("s").isin(["beta", "gamma"]), col("s").startswith("g"),
             col("s").contains("mm")]
    for budget, pieces_expected in [(None, 0), ("1M", 1)]:
        if budget:
            monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", budget)
        for e in exprs:
            with gpu_budget.track(device="auto") as r:
                got = comp._eval_expr_arrow(e, t).to_numpy(zero_copy_only=False)
            want = {"eq": names[np.arange(N) % 4] == "beta", "ne": names[np.arange(N) % 4] != "beta",
                    "isin": np.isin(names[np.arange(N) % 4], ["beta", "gamma"]),
                    "startswith": names[np.arange(N) % 4] == "gamma", "contains": names[np.arange(N) % 4] == "gamma"}[e.op]
            assert np.array_equal(got, want), e.op
            assert len(r.events) == pieces_expected, e.op
            if budget:
                assert r.events[0]["line"].startswith("string filter: ran on CPU, needs about "), e.op


def test_masked_global_aggregates_over_the_budget_run_on_the_cpu_kernel(monkeypatch):
    comp = custom_ops.CustomOpsCompiler(device="gpu")
    v = (np.arange(N) % 100).astype(np.float32)
    mask = (np.arange(N) % 3 == 0).astype(np.int32)
    want = {"sum": float(v[mask == 1].sum()), "min": float(v[mask == 1].min()), "max": float(v[mask == 1].max())}
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "1M")
    for fn in want:
        with gpu_budget.track(device="auto") as r:
            got = getattr(comp, f"_run_masked_global_{fn}")(v, mask, N)
        assert got == want[fn], fn
        assert r.lines == [f"masked_global_{fn}: ran on CPU, needs about 3.1 MB of GPU memory, over 90% of the 1.0 MB free"]
    with gpu_budget.track(device="auto") as r:
        assert comp._run_masked_global_sum_product(v, v, mask, N) == float((v[mask == 1].astype(np.float64) ** 2).sum())
    assert r.lines[0].startswith("masked_global_sum_product: ran on CPU")


def test_group_encoding_over_the_budget_hands_back_to_the_cpu_encoder(monkeypatch):
    comp = custom_ops.CustomOpsCompiler(device="gpu")
    keys = pa.array((np.arange(N) % 50).astype(np.int32))
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "1M")
    with gpu_budget.track(device="auto") as r:
        assert comp._build_group_ids_gpu_single_int(keys, "k", N) is None
    assert r.lines[0].startswith("group encode: ran on CPU, needs about ")


@pytest.mark.parametrize("kernel, name", [("join_count_i32_gpu", "join"), ("sort_topk_i32_gpu", "top-k")])
def test_a_join_or_sort_that_runs_out_of_memory_reruns_on_the_cpu_and_is_remembered(tables, monkeypatch, kernel, name):
    lf, sql = QUERIES[name](*tables)
    g = get_aot_gpu()
    with monkeypatch.context() as m:
        _fail(m, g, kernel, -2)
        got, warned = run(lf, "gpu", monkeypatch)
    same(got, reference(sql, *tables))
    assert pieces(lf)[0].startswith(f"{PIECE[name]}: ran on CPU, ran out of GPU memory (needed about ")
    assert "CUDA_ERROR_OUT_OF_MEMORY" in pieces(lf)[0] and len(warned) == 1
    custom_ops.clear_cache()                         # (a cached join result would skip the piece)
    got, _ = run(lf, "gpu", monkeypatch)             # the kernel works again; the shape is remembered
    same(got, reference(sql, *tables))
    assert pieces(lf) == [f"{PIECE[name]}: ran on CPU, this shape ran out of GPU memory before"]


def test_strict_does_not_hide_an_out_of_memory(tables, monkeypatch):
    lf, _ = aggregate(*tables)
    limit_cuda_alloc(monkeypatch, N)
    with pytest.raises(MemoryError, match="aggregate: ran out of GPU memory"):
        lf.compute(device="gpu", strict=True)


# --- the real GPU (opt-in) --------------------------------------------------------------------

REAL = pytest.mark.skipif(not os.environ.get("MXFRAME_TEST_GPU_BUDGET"),
                          reason="runs 30M-row queries on the real GPU: set MXFRAME_TEST_GPU_BUDGET=1")

REAL_SCRIPT = r"""
import warnings, numpy as np, pyarrow as pa, duckdb
from mxframe import LazyFrame, col
from mxframe import gpu_budget
n = 30_000_000
rng = np.random.default_rng(3)
t = pa.table({"k": rng.integers(0, 100, n).astype(np.int32), "v": rng.integers(0, 50, n).astype(np.int32)})
lf = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).sort("k")
want = duckdb.sql("SELECT k, sum(v)::DOUBLE AS s FROM t GROUP BY k ORDER BY k").arrow().read_all()
#PATCH#
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    got = lf.compute(device="gpu")
assert got.column("s").to_numpy().tolist() == want.column("s").to_numpy().tolist()
print("PIECES", lf.last_compile_provenance.get("gpu_pieces", ""), "|", lf.last_compile_provenance["path"])
"""


def _real(patch, env, hog_mb=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               PYTHONPATH=os.pathsep.join([sys.path[0], os.environ.get("PYTHONPATH", "")]), **env)
    hog = None
    if hog_mb:
        hog = subprocess.Popen([sys.executable, "-c", HOG, str(hog_mb), "120"], env=env, stdout=subprocess.PIPE, text=True)
        hog.stdout.readline()
    try:
        r = subprocess.run([sys.executable, "-c", REAL_SCRIPT.replace("#PATCH#", patch)], env=env, capture_output=True,
                           text=True, timeout=600)
    finally:
        if hog:
            hog.kill()
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


@REAL
def test_real_gpu_with_a_budget_smaller_than_the_data_finishes_on_the_cpu():
    out = _real("", {"MXFRAME_GPU_MEM_BUDGET": "64M"})
    assert out.startswith("PIECES aggregate: ran on CPU, needs about 229.9 MB of GPU memory, over 90% of the 64.0 MB free")
    assert out.endswith("| fused_pipeline_cpu")


@REAL
def test_real_gpu_without_a_budget_runs_the_same_query_on_the_gpu():
    assert _real("", {}) == "PIECES  | fused_pipeline_gpu"


@REAL
def test_real_gpu_that_runs_out_of_memory_finishes_correctly():
    """Another process holds the GPU's memory; the pre-check is told the GPU is empty, so the OOM is real."""
    out = _real("gpu_budget.free_bytes = lambda pool=None: 1 << 40", {}, hog_mb=300)
    assert out.startswith("PIECES aggregate: ran ") and "out of GPU memory" in out, out
