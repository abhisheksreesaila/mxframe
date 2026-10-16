"""F-008 fused pipeline kernel: bytecode compiler and CPU/GPU kernels vs a NumPy float64 reference."""
import os

import numpy as np
import pyarrow as pa
import pytest

from mxframe.lazy_expr import col, lit, when
from mxframe import pipeline_kernels as pk
from mxframe.pipeline_program import (
    compile_pipeline, finalize, merge_partials, empty_partials, Partials,
)

RNG = np.random.default_rng(7)


def _gpu_ok():
    if not pk.gpu_available():
        return False
    try:
        pk._get_cuda()
        return True
    except Exception:
        return False


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    if request.param == "cpu" and not pk.cpu_available():
        pytest.skip("libmxpipeline_cpu.so not built")
    if request.param == "gpu" and not _gpu_ok():
        pytest.skip("no GPU or libmxpipeline_gpu.so not built")
    return request.param


def make_table(n, n_keys=4):
    return pa.table({
        "i32": pa.array(RNG.integers(-50, 50, n).astype(np.int32)),
        "i64": pa.array(RNG.integers(0, 1000, n).astype(np.int64)),
        "f32": pa.array((RNG.random(n) * 100).astype(np.float32)),
        "f64": pa.array(RNG.random(n) * 10 - 2),
        "date": pa.array(RNG.integers(19920101, 19981231, n).astype(np.int32)),
        "flag": pa.array(RNG.integers(0, 2, n).astype(bool)),
        "s": pa.array(RNG.choice(["AIR", "RAIL", "SHIP", "TRUCK", "MAIL"], n)),
    })


# ── NumPy float64 reference ────────────────────────────────────────────────

def ref_eval(e, t):
    op, a = e.op, e.args
    n = t.num_rows
    if op == "col":
        c = t.column(a[0]).combine_chunks()
        if pa.types.is_string(c.type):
            return np.asarray(c.to_pylist(), dtype=object)
        return np.asarray(c.to_numpy(zero_copy_only=False)).astype(np.float64)
    if op == "lit":
        return a[0]
    if op == "add": return ref_eval(a[0], t) + ref_eval(a[1], t)
    if op == "sub": return ref_eval(a[0], t) - ref_eval(a[1], t)
    if op == "mul": return ref_eval(a[0], t) * ref_eval(a[1], t)
    if op == "div": return ref_eval(a[0], t) / ref_eval(a[1], t)
    if op in ("lt", "le", "gt", "ge", "eq", "ne"):
        fn = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
              "eq": np.equal, "ne": np.not_equal}[op]
        for x_e, c_e in ((a[0], a[1]), (a[1], a[0])):
            # float32 column vs literal: the literal takes the column's type (DuckDB / Polars / NumPy 2)
            if (x_e.op == "col" and c_e.op == "lit" and isinstance(c_e.args[0], (int, float))
                    and t.schema.field(x_e.args[0]).type == pa.float32()):
                x = t.column(x_e.args[0]).to_numpy().astype(np.float32)
                c = np.float32(c_e.args[0])
                return fn(x, c) if x_e is a[0] else fn(c, x)
        return fn(ref_eval(a[0], t), ref_eval(a[1], t))
    if op == "and": return np.logical_and(ref_eval(a[0], t), ref_eval(a[1], t))
    if op == "or": return np.logical_or(ref_eval(a[0], t), ref_eval(a[1], t))
    if op == "not": return np.logical_not(ref_eval(a[0], t))
    if op == "isin": return np.isin(ref_eval(a[0], t), np.asarray(a[1], dtype=object if isinstance(a[1][0], str) else np.float64))
    if op == "startswith": return np.array([str(x).startswith(a[1]) for x in ref_eval(a[0], t)])
    if op == "contains": return np.array([a[1] in str(x) for x in ref_eval(a[0], t)])
    if op == "case_when":
        return np.where(ref_eval(a[0], t), ref_eval(a[1], t), ref_eval(a[2], t))
    if op == "year": return np.floor(ref_eval(a[0], t) / 10000)
    raise NotImplementedError(op)


def ref_aggregate(t, preds, aggs, gids, n_groups, start, end):
    """Returns {name: array over groups} plus counts, over rows [start, end)."""
    mask = np.ones(t.num_rows, dtype=bool)
    for p in preds:
        mask &= np.asarray(ref_eval(p, t), dtype=bool)
    sel = np.zeros(t.num_rows, dtype=bool)
    sel[start:end] = True
    mask &= sel
    g = gids if gids is not None else np.zeros(t.num_rows, dtype=np.int32)
    out = {}
    cnt = np.bincount(g[mask], minlength=n_groups)
    for i, e in enumerate(aggs):
        name = e._alias or f"agg_{i}"
        if e.op == "count":
            out[name] = cnt.astype(np.float64)
            continue
        v = np.broadcast_to(np.asarray(ref_eval(e.args[0], t), dtype=np.float64), (t.num_rows,))
        res = np.zeros(n_groups)
        if e.op in ("sum", "mean"):
            res = np.bincount(g[mask], weights=v[mask], minlength=n_groups)
            if e.op == "mean":
                res = np.where(cnt > 0, res / np.maximum(cnt, 1), 0.0)
        else:
            fn = np.minimum if e.op == "min" else np.maximum
            init = np.inf if e.op == "min" else -np.inf
            res = np.full(n_groups, init)
            fn.at(res, g[mask], v[mask])
            res[cnt == 0] = 0.0
        out[name] = res
    return out, cnt


def to_named(program, partials, grouped):
    present, cols = finalize(program, partials, grouped)
    return present, cols


# ── bytecode compiler ──────────────────────────────────────────────────────

def test_compiles_q1_shaped_program_with_shared_subexpressions():
    t = make_table(10)
    disc = col("f32") * (lit(1.0) - col("f64"))
    aggs = [col("f32").sum().alias("a"), disc.sum().alias("b"),
            (disc * (lit(1.0) + col("f64"))).sum().alias("c"),
            col("f32").mean().alias("d"), col("f32").count().alias("e")]
    prog = compile_pipeline([col("date") <= lit(19980902)], aggs, t)
    assert prog is not None
    ops = prog.op_names()
    # `1 - f64` and `f32 * (1 - f64)` are computed once; sum(f32) and mean(f32) share a slot
    assert ops.count("SUB") == 1
    assert ops.count("MUL") == 2
    assert prog.n_slots == 3
    assert prog.columns == ["date", "f32", "f64"]


def test_between_pattern_uses_fused_op():
    t = make_table(10)
    prog = compile_pipeline([(col("i32") * lit(2)).between(5, 30)], [col("f32").sum()], t)
    assert "BETWEEN" in prog.op_names()
    assert "AND" not in prog.op_names()


def test_float_between_keeps_the_fused_op_for_constant_bounds_only():
    t = make_table(10)                         # NaN fails `x <= hi` for a constant hi, so IEEE's BETWEEN is DuckDB's (F-070)
    prog = compile_pipeline([(col("f64") * lit(2.0)).between(0.5, 3.0)], [col("f32").sum()], t)
    assert "BETWEEN" in prog.op_names() and "AND" not in prog.op_names()
    for lo, hi in ((col("f32"), 3.0), (0.5, col("f32")), (float("nan"), 3.0), (0.5, float("nan"))):
        prog = compile_pipeline([(col("f64") * lit(2.0)).between(lo, hi)], [col("f32").sum()], t)
        assert "BETWEEN" not in prog.op_names() and "AND" in prog.op_names(), (lo, hi)


def test_constant_subexpressions_fold():
    t = make_table(10)
    prog = compile_pipeline([col("f64") >= lit(0.06) - lit(0.01)], [col("f32").sum()], t)
    assert list(prog.consts).count(0.06 - 0.01) == 1
    assert "SUB" not in prog.op_names()


@pytest.mark.parametrize("pred, aggs", [
    (None, [col("s").sum()]),                                  # string as number
    (None, [(col("i32") / col("i64")).sum()]),                 # integer division has floor semantics
    (col("s") > lit("AIR"), [col("f32").sum()]),               # ordered string compare
    (col("f32") == lit("x"), [col("f32").sum()]),              # string literal on numeric column
    (None, [col("f32").rank().sum()]),                         # window function
    (None, [(col("f32").sum() / col("f64").sum())]),           # aggregate arithmetic
    (col("f64") + lit(1.0), [col("f32").sum()]),               # non-boolean predicate
    (None, [col("nope").sum()]),                               # unknown column
])
def test_unsupported_returns_none(pred, aggs):
    t = make_table(10)
    assert compile_pipeline([pred] if pred is not None else [], aggs, t) is None


@pytest.mark.parametrize("pred", [
    col("i64") == lit(2 ** 53 + 1),
    col("i64") < lit(2 ** 60),
    lit(2 ** 55) > col("i64"),
    col("i64").isin([1, 2 ** 54]),
])
def test_integer_literals_beyond_2_53_decline(pred):
    t = make_table(10)
    assert compile_pipeline([pred], [col("f32").sum()], t) is None
    assert compile_pipeline([col("i64") == lit(2 ** 53)], [col("f32").sum()], t) is not None


def test_nulls_in_string_column_are_unsupported():
    t = pa.table({"s": pa.array(["a", None, "b"]), "x": pa.array([1.0, 2.0, 3.0])})
    assert compile_pipeline([col("s") == lit("a")], [col("x").sum()], t) is None


def test_string_predicates_translate_to_dictionary_membership():
    t = make_table(2000)
    prog = compile_pipeline([col("s").isin(["AIR", "SHIP"]) | (col("s") == lit("MAIL")) | col("s").startswith("TR")],
                            [col("f32").sum()], t)
    assert prog is not None
    assert "s" in prog.dict_arrays and prog.op_names().count("IN_SET") == 3
    # absent literal -> empty membership, not an error
    assert compile_pipeline([col("s") == lit("NOPE")], [col("f32").sum()], t) is not None


# ── CPU kernel vs reference ────────────────────────────────────────────────

PREDS = {
    "none": [],
    "cmp": [col("date") <= lit(19980902)],
    "q6": [(col("date") >= lit(19940101)) & (col("date") < lit(19950101)) &
           col("f64").between(0.05, 0.07) & (col("f32") < lit(24.0))],
    "or_not": [(col("i32") > lit(10)) | ~(col("f64") < lit(1.0))],
    "isin_num": [col("i64").isin([1, 5, 9, 500, 999])],
    "isin_str": [col("s").isin(["AIR", "RAIL"]) & (col("f32") > lit(10))],
    "startswith": [col("s").startswith("TR") | (col("s") == lit("MAIL"))],
    "ne_str": [col("s") != lit("AIR")],
    "bool_col": [col("flag") & (col("i32") >= lit(0))],
    "all_filtered": [col("f32") < lit(-1.0)],
    "const_false": [lit(1) < lit(0)],
}
AGGS = {
    "sum": [col("f32").sum().alias("s")],
    "all_ops": [col("f32").sum().alias("a"), col("f64").min().alias("b"), col("i32").max().alias("c"),
                col("f64").mean().alias("d"), col("i64").count().alias("e")],
    "expr": [(col("f32") * (lit(1.0) - col("f64"))).sum().alias("a"),
             (col("f32") * (lit(1.0) - col("f64")) * (lit(1.0) + col("f64"))).mean().alias("b"),
             (col("i32") + col("i64") - lit(3)).min().alias("c"),
             ((col("f64") + lit(2.0)) / lit(3.0)).max().alias("d")],
    "case_year": [when(col("s") == lit("AIR"), col("f32") * col("f64"), lit(0.0)).sum().alias("a"),
                  col("date").year().max().alias("b")],
}


def run_case(t, preds, aggs, n_groups, start, end, threads=4, gids=None, device="cpu"):  # noqa
    prog = compile_pipeline(preds, aggs, t)
    assert prog is not None, (preds, aggs)
    arrays = prog.bind(t)
    if n_groups > 1 and gids is None:
        gids = RNG.integers(0, n_groups, t.num_rows).astype(np.int32)
    fn = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    parts = fn(prog, arrays, gids if n_groups > 1 else None, n_groups, start, end, **({"threads": threads} if device == "cpu" else {}))
    ref, ref_cnt = ref_aggregate(t, preds, aggs, gids if n_groups > 1 else None, n_groups, start, end)
    return prog, parts, ref, ref_cnt


def check(prog, parts, ref, ref_cnt, grouped):
    assert np.array_equal(parts.cnt, ref_cnt)
    present, cols = finalize(prog, parts, grouped)
    keep = present if grouped else np.ones(1, dtype=bool)
    for name, want in ref.items():
        got = cols[name]
        w = want[keep] if grouped else want
        np.testing.assert_allclose(got.astype(np.float64), w, rtol=1e-9, atol=1e-9, err_msg=name)


@pytest.fixture(scope="module")
def cpu_table():
    return make_table(50_000)


@pytest.mark.parametrize("pname", list(PREDS))
@pytest.mark.parametrize("aname", list(AGGS))
@pytest.mark.parametrize("n_groups", [1, 4])
def test_matches_reference(device, cpu_table, pname, aname, n_groups):
    n = cpu_table.num_rows
    prog, parts, ref, cnt = run_case(cpu_table, PREDS[pname], AGGS[aname], n_groups, 0, n, device=device)
    check(prog, parts, ref, cnt, grouped=n_groups > 1)


@pytest.mark.parametrize("dtype_col", ["i32", "i64", "f32", "f64", "flag"])
def test_every_column_dtype_loads(device, cpu_table, dtype_col):
    aggs = [col(dtype_col).sum().alias("s")] if dtype_col != "flag" else [col("f32").sum().alias("s")]
    preds = [] if dtype_col != "flag" else [col("flag")]
    n = cpu_table.num_rows
    prog, parts, ref, cnt = run_case(cpu_table, preds, aggs, 1, 0, n, device=device)
    check(prog, parts, ref, cnt, grouped=False)


@pytest.mark.parametrize("n, start, end", [
    (0, 0, 0), (1, 0, 1), (1, 1, 1), (7, 0, 7), (1023, 0, 1023), (1024, 0, 1024), (1025, 0, 1025),
    (5000, 3, 4999), (200_003, 12_345, 199_999), (70_000, 65_536, 70_000),
])
@pytest.mark.parametrize("n_groups", [1, 4])
def test_ranges_not_aligned_to_vectors(device, n, start, end, n_groups):
    t = make_table(max(n, 1)).slice(0, n) if n else make_table(1).slice(0, 0)
    prog, parts, ref, cnt = run_case(t, PREDS["q6"][:0] + [col("f64") > lit(0.0)], AGGS["all_ops"], n_groups, start, end, device=device)
    check(prog, parts, ref, cnt, grouped=n_groups > 1)


def test_many_groups(device):
    n, g = 400_000, 1_500_000
    t = make_table(n)
    gids = RNG.integers(0, g, n).astype(np.int32)
    aggs = [col("f32").sum().alias("s"), col("f64").min().alias("lo"), col("f64").max().alias("hi"),
            col("i32").count().alias("c")]
    prog, parts, ref, cnt = run_case(t, [col("f32") > lit(5.0)], aggs, g, 0, n, gids=gids, device=device)
    check(prog, parts, ref, cnt, grouped=True)


@pytest.mark.parametrize("n_groups", [1, 4])
def test_split_ranges_merge_to_full_result(device, cpu_table, n_groups):
    t = cpu_table
    n = t.num_rows
    preds, aggs = PREDS["cmp"], AGGS["all_ops"] + AGGS["expr"]
    prog = compile_pipeline(preds, aggs, t)
    arrays = prog.bind(t)
    gids = RNG.integers(0, n_groups, n).astype(np.int32) if n_groups > 1 else None
    run = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    full = run(prog, arrays, gids, n_groups, 0, n)
    cuts = [0, 1, 4097, 20_000, 33_333, n]
    parts = [run(prog, arrays, gids, n_groups, a, b) for a, b in zip(cuts, cuts[1:])]
    merged = merge_partials(prog, parts[::-1])   # merge order must not matter
    assert np.array_equal(merged.cnt, full.cnt)
    np.testing.assert_allclose(merged.vals, full.vals, rtol=1e-12)


@pytest.mark.parametrize("threads", [1, 2, 7, 24])
def test_thread_count_does_not_change_result(cpu_table, threads):
    if not pk.cpu_available():
        pytest.skip("libmxpipeline_cpu.so not built")
    n = cpu_table.num_rows
    prog, parts, ref, cnt = run_case(cpu_table, PREDS["cmp"], AGGS["expr"], 4, 0, n, threads=threads)
    check(prog, parts, ref, cnt, grouped=True)


def test_global_agg_over_no_rows_finalizes_to_null_and_count_zero(device, cpu_table):
    prog, parts, ref, cnt = run_case(cpu_table, PREDS["all_filtered"], AGGS["all_ops"], 1, 0, cpu_table.num_rows, device=device)
    present, cols = finalize(prog, parts, grouped=False)
    for name, op, _ in prog.outputs:                  # SQL: sum/min/max/mean over no rows is NULL, count is 0
        if op == "count":
            assert cols[name].tolist() == [0] and cols[name].dtype == np.int64
        else:
            assert np.ma.getmaskarray(cols[name]).all(), name


def test_grouped_finalize_drops_empty_groups(device, cpu_table):
    n = cpu_table.num_rows
    gids = np.where(np.arange(n) % 2 == 0, 0, 2).astype(np.int32)   # groups 1 and 3 never occur
    prog, parts, ref, cnt = run_case(cpu_table, [], AGGS["sum"], 4, 0, n, gids=gids, device=device)
    present, cols = finalize(prog, parts, grouped=True)
    assert present.tolist() == [True, False, True, False]
    assert len(cols["s"]) == 2


def test_gpu_accepts_device_resident_columns(cpu_table):
    if not _gpu_ok():
        pytest.skip("no GPU or libmxpipeline_gpu.so not built")
    t = cpu_table
    n = t.num_rows
    prog = compile_pipeline(PREDS["cmp"], AGGS["all_ops"], t)
    arrays = prog.bind(t)
    gids = RNG.integers(0, 4, n).astype(np.int32)
    want = pk.run_pipeline_cpu(prog, arrays, gids, 4, 10_000, 40_000)
    with pk.upload_columns(prog, arrays, gids, 10_000, 40_000) as dc:   # a morsel uploaded by the Router
        got = pk.run_pipeline_gpu(prog, dc, None, 4, 10_000, 40_000)
        half = pk.run_pipeline_gpu(prog, dc, None, 4, 10_000, 25_000)
    assert np.array_equal(got.cnt, want.cnt)
    np.testing.assert_allclose(got.vals, want.vals, rtol=1e-12)
    assert half.cnt.sum() < got.cnt.sum()


def _chunked(t, sizes):
    """Same rows, cut into chunks of the given sizes; one column cut differently."""
    pieces, off = [], 0
    for sz in sizes:
        pieces.append(t.slice(off, sz))
        off += sz
    out = pa.concat_tables(pieces)
    # misalign one column's chunk boundaries against the others
    f32 = pa.chunked_array([t.column("f32").slice(0, 777), t.column("f32").slice(777)])
    return out.set_column(out.schema.get_field_index("f32"), "f32", f32)


@pytest.mark.parametrize("n_groups", [1, 4])
def test_multi_chunk_tables_need_no_concatenation(device, n_groups):
    t = make_table(9000)
    ct = _chunked(t, [1, 1023, 2000, 3, 4096, 1877])
    assert ct.column("i32").num_chunks == 6 and ct.column("f32").num_chunks == 2
    preds = PREDS["isin_str"]
    aggs = AGGS["all_ops"] + AGGS["expr"]
    prog = compile_pipeline(preds, aggs, ct)
    cc = prog.bind_chunks(ct)
    assert len(cc.bounds) > 6                       # union of both layouts
    assert cc.bounds[0][0] == 0 and cc.bounds[-1][1] == 9000
    assert all(a[1] == b[0] for a, b in zip(cc.bounds, cc.bounds[1:]))
    gids = RNG.integers(0, n_groups, 9000).astype(np.int32) if n_groups > 1 else None
    run = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    flat = compile_pipeline(preds, aggs, t)
    want = run(flat, flat.bind(t), gids, n_groups, 0, 9000)
    for start, end in [(0, 9000), (5, 8999), (1024, 4000)]:
        got = run(prog, cc, gids, n_groups, start, end)
        ref = run(flat, flat.bind(t), gids, n_groups, start, end)
        assert np.array_equal(got.cnt, ref.cnt)
        np.testing.assert_allclose(got.vals, ref.vals, rtol=1e-12)
    assert want.cnt.sum() > 0


@pytest.mark.parametrize("n_groups", [2, 8, 12, 100, 1000, 3000, 200_000])
def test_group_count_regimes(device, n_groups):
    """Per-thread partials (few groups), a shared table per block (some hundreds to thousands) and
    per-row global atomics (many) must all agree with the reference, min / max / count included."""
    n = 120_000
    t = make_table(n)
    gids = RNG.integers(0, n_groups, n).astype(np.int32)
    aggs = [col("f32").sum().alias("s"), (col("f32") * col("f64")).sum().alias("p"),
            col("f64").min().alias("lo"), col("i32").max().alias("hi"), col("i64").count().alias("c")]
    prog, parts, ref, cnt = run_case(t, [col("f32") > lit(5.0)], aggs, n_groups, 0, n, gids=gids, device=device)
    check(prog, parts, ref, cnt, grouped=True)
