"""F-070: comparisons on float columns order values like DuckDB: NaN is above every number, +inf included, and
NaN = NaN. So `x > 1`, `x >= 1`, `x <> 1` are true for NaN, `x = 'nan'` matches it, `x <> x` is false, BETWEEN /
IN / NOT follow, MIN skips NaN (NaN only if the group is all NaN), MAX is NaN if the group holds one, and sorts put
NaN last ascending and first descending. NULL stays NULL: a comparison with NULL never keeps the row.

Every expected answer comes from a NATIVE DuckDB table (`create table t as select * from <registered arrow table>`:
a query over the registered table itself treats NaN differently). Each case runs for float32 and float64 columns,
with and without NULLs, on cpu and gpu, with the native executor on and off, and asserts the path that answered
(`last_compile_provenance["path"]`, the native filter) so a right answer from a different engine cannot pass.

Before the fix every group of these cases failed on main (NaN rows were dropped by `>`, `>=`, `<>`, `NOT`, ...).
"""
import math

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe import gpu_stream
from mxframe.custom_ops import CustomOpsCompiler

NAN, INF = float("nan"), math.inf
GPU = gpu_stream.cuda_available()

X = [NAN, INF, -INF, 0.0, -0.0, 1.5, 2.0, 1.0, -3.0, NAN, 1.5, 0.5, -1.0, 1e30]
Y = [NAN, 1.0, INF, -0.0, 0.0, NAN, 2.0, -INF, -3.0, 1.0, 1.5, NAN, 0.5, INF]
N_ = [1, -1, 0, 2, 5, 3, -2, 0, 1, 4, 2, -3, 0, 1]
G = [3, 0, 0, 1, 2, 2, 2, 2, 0, 3, 1, 2, 0, 1]        # group 0 sums +inf and -inf (a negative NaN), group 3 holds two NaNs
NULL_X, NULL_Y, NULL_N = (6, 12), (3, 8), (4, 0, 5)     # n is NULL on rows where x and / or y is NaN too


def make_table(dtype, nulls):
    def f(vals, holes):
        v = [None if nulls and i in holes else a for i, a in enumerate(vals)]
        return pa.array(v, dtype)

    return pa.table({
        "i": pa.array(range(len(X)), pa.int32()), "g": pa.array(G, pa.int32()),
        "x": f(X, NULL_X), "y": f(Y, NULL_Y), "z": pa.array([0.0] * len(X), dtype),
        "n": pa.array([None if nulls and i in NULL_N else a for i, a in enumerate(N_)], pa.int32()),
    })


@pytest.fixture(scope="module")
def world():
    """{(dtype name, nulls): (arrow table, native DuckDB connection)}"""
    out = {}
    for name, dt in (("float32", pa.float32()), ("float64", pa.float64())):
        for nulls in (False, True):
            t = make_table(dt, nulls)
            con = duckdb.connect()
            con.register("_t", t)
            con.execute("create table t as select * from _t")
            out[(name, nulls)] = (t, con)
    return out


@pytest.fixture(params=[(d, nl, dev, nat) for d in ("float32", "float64") for nl in (False, True)
                        for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-{'nulls' if p[1] else 'nonull'}-{p[2]}-native{p[3]}")
def cfg(request, world, monkeypatch):
    dtype, nulls, device, native = request.param
    if device == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    table, con = world[(dtype, nulls)]
    paths = []                      # the path of every plan run for a query (a projection above an aggregate runs it first)
    run = CustomOpsCompiler._compile_and_run

    def recording(self, *a, **k):
        out = run(self, *a, **k)
        paths.append((self.last_compile_provenance or {}).get("path"))
        return out

    monkeypatch.setattr(CustomOpsCompiler, "_compile_and_run", recording)
    return dict(dtype=dtype, nulls=nulls, device=device, native=native == "1", table=table, con=con, paths=paths,
                T="double" if dtype == "float64" else "float")


# ── comparing results ───────────────────────────────────────────────────────────────────────────

def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return a == pytest.approx(b, rel=1e-6, abs=1e-9)
    return a == b


def _rows(t):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    return [tuple(c[r] for c in cols) for r in range(t.num_rows)]


def _same(got, exp):
    assert len(got) == len(exp), (got, exp)
    for g, e in zip(got, exp):
        assert len(g) == len(e) and all(_eq(a, b) for a, b in zip(g, e)), (g, e)


def _run(cfg, sql):
    """(rows, paths) of `sql` over table t on the config's device: the path of every plan that ran."""
    nx._filter_last.update(path=None, decline=None)
    cfg["paths"].clear()
    got = mx.sql(sql, t=cfg["table"]).compute(device=cfg["device"])
    return _rows(got), list(cfg["paths"])


_MASKED = ("masked_global_agg_mojo", "cpu_aot", "native", "fused_pipeline_cpu", "fused_pipeline_gpu", "arrow_null_semantics")


def _agg_path(cfg):
    if cfg["nulls"]:
        return "arrow_null_semantics"           # a null in a column the plan reads: Arrow's null semantics
    if cfg["device"] == "gpu":
        return "fused_pipeline_gpu"
    return "native" if cfg["native"] else "fused_pipeline_cpu"


def check(cfg, sql, path, any_of=()):
    """The answer equals DuckDB's (native table), `path` (a name, or a tuple of names) is among the paths that ran
    (all of them), and so is one of `any_of` when given."""
    sql = sql.replace("{T}", cfg["T"])
    exp = [tuple(r) for r in cfg["con"].execute(sql).fetchall()]
    got, ran = _run(cfg, sql)
    _same(got, exp)
    assert {path} <= set(ran) if isinstance(path, str) else set(path) <= set(ran), (ran, path)
    assert not any_of or set(ran) & set(any_of), (ran, any_of)
    return got


# ── WHERE: a row filter folded into an aggregate (native executor, fused pipeline cpu / gpu) ───────

NAN_ = "cast('nan' as {T})"
INF_ = "cast('inf' as {T})"
NINF_ = "cast('-inf' as {T})"

# more than 16 floats: the compiled programs decline it, so Arrow's `is_in` (with -0.0 = 0.0 and NaN = NaN) builds the mask
WIDE_IN = f"x in ({', '.join(f'{v}.0' for v in range(18))}, {NAN_})"

PREDICATES = [
    # a number
    "x > 1.0", "x >= 1.0", "x < 1.0", "x <= 1.0", "x = 1.5", "x <> 1.5", "x = 0.0", "x > 0.0", "x <= 0.0",
    # infinities
    f"x > {INF_}", f"x >= {INF_}", f"x < {INF_}", f"x <= {INF_}", f"x = {INF_}", f"x <> {INF_}",
    f"x > {NINF_}", f"x <= {NINF_}", f"x = {NINF_}",
    # NaN
    f"x > {NAN_}", f"x >= {NAN_}", f"x < {NAN_}", f"x <= {NAN_}", f"x = {NAN_}", f"x <> {NAN_}",
    # two columns, and a column with itself
    "x > y", "x >= y", "x < y", "x <= y", "x = y", "x <> y", "x <> x", "x >= x", "x = x", "x < x",
    # an integer column and a float one; arithmetic that makes NaN
    "n < x", "n >= x", "n = x", "n < y", "n > y", "n <= y", f"n < {NAN_}", f"n > {NAN_}", f"n >= {NAN_}", f"n <= {NAN_}",
    f"{NAN_} > n", f"{NAN_} < n", f"n = {NAN_}", f"n <> {NAN_}", f"n not in (1, 2)", f"y > n", "x + 1.0 > y", "x - x >= 0.0", "x - x = x - x", "x * 0.0 <> 0.0",
    # BETWEEN, IN, NOT, AND / OR
    "x between 0.0 and 2.0", f"x between 1.0 and {INF_}", "x between y and 3.0", "x not between 0.0 and 2.0",
    f"x in (1.5, {NAN_}, 0.0)", f"x not in (1.5, {NAN_})", "x in (2.0, 1.0)", "x not in (2.0, 1.0)",
    WIDE_IN,
    "not (x > 1.0)", f"not (x = {NAN_})", "not (x >= y)", "x > 1.0 and not (y < 1.0)", "x > 1.0 or y > 1.0",
    "not (x > 1.0 and y > 1.0)",
]


@pytest.mark.parametrize("pred", PREDICATES)
def test_where_in_an_aggregate(cfg, pred):
    sql = f"select count(*) c, sum(i) s from t where {pred}"
    if pred == WIDE_IN:
        check(cfg, sql, (), any_of=_MASKED)
    else:
        check(cfg, sql, _agg_path(cfg))


@pytest.mark.parametrize("pred", PREDICATES)
def test_where_in_a_grouped_aggregate(cfg, pred):
    sql = f"select g, count(*) c, sum(i) s from t where {pred} group by g order by g"
    if pred == WIDE_IN:
        check(cfg, sql, (), any_of=_MASKED)
    else:
        check(cfg, sql, _agg_path(cfg))


@pytest.mark.parametrize("case", ["case when x > y then 1 else 0 end", "case when x >= 1.0 then 1 else 0 end",
                                  "case when x = y then 2 when x < y then 1 else 0 end"])
def test_case_when_conditions(cfg, case):
    check(cfg, f"select g, sum({case}) c from t group by g order by g", _agg_path(cfg))


# ── WHERE: a plain row filter (the native leaf filter, or Arrow's compute when it declines) ────────────

@pytest.mark.parametrize("pred", PREDICATES)
def test_where_in_a_row_filter(cfg, pred):
    check(cfg, f"select i from t where {pred} order by i", "pyarrow_shortcut")
    if cfg["device"] == "cpu" and cfg["native"] and not cfg["nulls"] and pred != WIDE_IN:
        assert nx.last_filter() == ("native_filter", None), nx.last_filter()


# ── HAVING ────────────────────────────────────────────────────────────────────────────────────────────

HAVINGS = [
    "max(x) > 1.0", "max(x) >= 1.0", "max(x) < 1.0", "max(x) <= 1.0", "max(x) = 2.0", "max(x) <> 2.0",
    f"max(x) = {NAN_}", f"max(x) >= {NAN_}", f"max(x) > {INF_}", f"max(x) >= {INF_}", f"max(x) <> {NAN_}",
    "min(x) > 0.5", "min(x) >= 0.0", "min(x) < 1.0", f"min(x) = {NAN_}", f"min(x) <= {NINF_}", f"min(x) <> {NAN_}",
    "sum(x) > 1.0", "sum(x) >= 1.0", "sum(x) < 1.0", f"sum(x) = {NAN_}", f"sum(x) <> {NAN_}", f"sum(x) >= {INF_}",
    f"sum(x) < {INF_}", "sum(z) / sum(z) > 1.0", "sum(z) / sum(z) = sum(z) / sum(z)", "sum(z) / sum(z) <> 1.0",
    "sum(z) / sum(z) <= 1.0", "sum(x) > sum(y)", "max(x) = max(y)", "max(x) <> max(y)", "max(x) >= max(y)",
    "count(*) > 2 and max(x) > 1.0", "count(*) > 2 or max(x) < 0.0", f"max(x) between 1.0 and {INF_}",
    f"max(x) in (2.0, {NAN_})", f"max(x) not in (2.0, {NAN_})", "not (max(x) > 1.0)", "not (sum(x) <> sum(x))",
]


@pytest.mark.parametrize("having", HAVINGS)
def test_having(cfg, having):
    # every aggregate the HAVING reads is in the select list, so the aggregate's own path answers (a hidden one
    # adds a projection on top, whose path the provenance would report)
    check(cfg, f"select g, count(*) c, max(x) hx, min(x) lx, sum(x) sx, max(y) hy, sum(z) / sum(z) q from t "
               f"group by g having {having} order by g", _agg_path(cfg))


@pytest.mark.parametrize("having", ["max(x) > 1.0", f"max(x) = {NAN_}", f"sum(x) >= {INF_}", "sum(z) / sum(z) > 1.0",
                                    "min(x) >= 0.0"])
def test_having_on_a_global_aggregate(cfg, having):
    check(cfg, f"select count(*) c, max(x) hx, min(x) lx, sum(x) sx, sum(z) / sum(z) q from t having {having}",
          _agg_path(cfg))


# ── WHERE over an aggregate result ───────────────────────────────────────────────────────────────────

AGG_RESULT = [
    ("sum(x)", "a > 1.0"), ("sum(x)", f"a = {NAN_}"), ("sum(x)", f"a <> {NAN_}"), ("sum(x)", f"a >= {INF_}"),
    ("sum(x)", f"a < {INF_}"), ("sum(x)", "a <> a"), ("sum(x)", "a >= a"), ("sum(x)", f"a between 1.0 and {INF_}"),
    ("sum(x)", f"a in (5.0, {NAN_})"), ("sum(x)", f"a not in (5.0, {NAN_})"), ("sum(x)", "not (a > 1.0)"),
    ("max(x)", "a > 1.0"), ("max(x)", f"a >= {NAN_}"), ("max(x)", "a < 1.0"), ("max(x)", f"a = {NAN_}"),
    ("min(x)", "a < 1.0"), ("min(x)", "a >= 0.0"), ("min(x)", f"a <= {NINF_}"), ("min(x)", f"a <> {NAN_}"),
    ("sum(z) / sum(z)", "a > 1.0"), ("sum(z) / sum(z)", "a = a"), ("sum(z) / sum(z)", f"a >= {NAN_}"),
]


@pytest.mark.parametrize("agg,pred", AGG_RESULT)
def test_where_over_an_aggregate_result(cfg, agg, pred):
    check(cfg, f"select g, a from (select g, {agg} a from t group by g) where {pred} order by g",
          "pyarrow_shortcut")           # the filter above the aggregate is Arrow's compute (the native HAVING takes `agg <op> number` only)


# ── MIN / MAX ─────────────────────────────────────────────────────────────────────────────────────────

def test_min_and_max_by_group(cfg):
    check(cfg, "select g, min(x) lo, max(x) hi, min(y) ylo, max(y) yhi from t group by g order by g", _agg_path(cfg))


def test_global_min_and_max(cfg):
    check(cfg, "select min(x) lo, max(x) hi, min(y) ylo, max(y) yhi, min(x + y) slo, max(x + y) shi from t",
          _agg_path(cfg))


def test_min_and_max_of_a_group_of_only_nan(cfg):
    check(cfg, f"select g, min(x) lo, max(x) hi from t where g = 3 group by g", _agg_path(cfg))
    got, _ = _run(cfg, "select min(x) lo, max(x) hi from t where g = 3")
    assert math.isnan(got[0][0]) and math.isnan(got[0][1])


# ── sorts: NaN last ascending, first descending, equal to other NaNs ──────────────────────────────────────

SORTS = [
    "select i, x from t order by x, i", "select i, x from t order by x desc, i", "select i, x from t order by x, i limit 5",
    "select i, x from t order by x desc, i limit 5", "select i, x, y from t order by y desc, x, i limit 6",
    "select i, x from t where y > 0.5 order by x desc, i", "select i, x from t where y > 0.5 order by x, i limit 4",
    "select g, max(x) a from t group by g order by a, g", "select g, max(x) a from t group by g order by a desc, g",
    "select g, sum(x) a from t group by g order by a, g",             # group 0 sums to -NaN, group 3 to NaN: one value
    "select g, sum(x) a from t group by g order by a desc, g", "select g, sum(x) a from t group by g order by a, g limit 3",
    "select g, sum(x) a from t group by g order by a desc, g limit 3",
]


@pytest.mark.parametrize("sql", SORTS)
def test_sort_puts_nan_where_duckdb_does(cfg, sql):
    # whichever engine answers, the rows are DuckDB's; the aggregate-fed sorts also assert their engine
    sql = sql.replace("{T}", cfg["T"])
    exp = [tuple(r) for r in cfg["con"].execute(sql).fetchall()]
    got, ran = _run(cfg, sql)
    _same(got, exp)
    if "group by" in sql:
        # the native executor hands a sort over NaN values back (E_NAN_SORT): the fused pipeline answers then
        assert {_agg_path(cfg), "fused_pipeline_cpu" if _agg_path(cfg) == "native" else _agg_path(cfg)} & set(ran), ran
    else:
        assert "pyarrow_shortcut" in ran, ran


# ── pinned answers (DuckDB 1.4, checked by hand), so a harness that agreed with a wrong engine would show ─────

def _pinned(device, native, monkeypatch, sql, dtype=pa.float64()):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    t = pa.table({"i": pa.array(range(5), pa.int32()), "x": pa.array([1.0, NAN, INF, -0.0, 5.0], dtype)})
    return _rows(mx.sql(sql, t=t).compute(device=device))


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not GPU, reason="no CUDA GPU"))])
@pytest.mark.parametrize("native", ["1", "0"])
def test_pinned_answers(device, native, monkeypatch):
    run = lambda sql, **k: _pinned(device, native, monkeypatch, sql, **k)           # noqa: E731
    assert run("select count(*) c from t where x > 1.0") == [(3,)]                   # 5.0, inf, NaN
    assert run("select count(*) c from t where x > cast('nan' as double)") == [(0,)]
    assert run("select i from t where x >= cast('nan' as double) order by i") == [(1,)]
    assert run("select i from t where x <> x order by i") == []
    assert run("select i from t where x = x order by i") == [(0,), (1,), (2,), (3,), (4,)]
    assert math.isnan(run("select max(x) from t")[0][0])
    assert run("select min(x) from t") == [(-0.0,)]
    assert [r[0] for r in run("select i, x from t order by x desc, i")] == [1, 2, 4, 0, 3]
    assert [r[0] for r in run("select i, x from t order by x, i")] == [3, 0, 4, 2, 1]


# ── the same rules through the LazyFrame API (Python NaN / inf literals) ────────────────────────────────────

LAZY = [
    (lambda: col("x") > float("nan"), "x > {nan}"), (lambda: col("x") >= float("nan"), "x >= {nan}"),
    (lambda: col("x") < float("nan"), "x < {nan}"), (lambda: col("x") <= float("nan"), "x <= {nan}"),
    (lambda: col("x") == float("nan"), "x = {nan}"), (lambda: col("x") != float("nan"), "x <> {nan}"),
    (lambda: col("x") > math.inf, "x > {inf}"), (lambda: col("x") <= math.inf, "x <= {inf}"),
    (lambda: col("x") < -math.inf, "x < {ninf}"), (lambda: col("x") >= -math.inf, "x >= {ninf}"),
    (lambda: col("x").isin([1.5, float("nan")]), "x in (1.5, {nan})"),
    (lambda: ~col("x").isin([1.5, float("nan")]), "x not in (1.5, {nan})"),
    (lambda: col("x").between(0.0, math.inf), "x between 0.0 and {inf}"),
    (lambda: col("x").between(float("nan"), float("nan")), "x between {nan} and {nan}"),
    (lambda: ~(col("x") > 1.0), "not (x > 1.0)"),
    # three-valued IN: x IN (1.0, NULL) is true or NULL, so NOT IN is never true
    (lambda: col("x").isin([1.0, None]), "x in (1.0, NULL)"), (lambda: ~col("x").isin([1.0, None]), "x not in (1.0, NULL)"),
    (lambda: ~col("x").isin([float("nan"), None]), "x not in ({nan}, NULL)"),
    (lambda: ~col("n").isin([1, None]), "n not in (1, NULL)"),
]


@pytest.mark.parametrize("make,sql", LAZY, ids=[s for _, s in LAZY])
def test_lazyframe_literals(cfg, make, sql):
    sql = sql.format(nan=NAN_, inf=INF_, ninf=NINF_).replace("{T}", cfg["T"])
    exp = [tuple(r) for r in cfg["con"].execute(f"select i from t where {sql} order by i").fetchall()]
    got = _rows(LazyFrame(cfg["table"]).filter(make()).select(col("i")).sort(col("i")).compute(device=cfg["device"]))
    _same(got, exp)


# ── the SQL front end writes NaN and infinity as CAST('nan' AS DOUBLE) ─────────────────────────────────────

def test_cast_of_a_special_float_literal(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    t = pa.table({"i": pa.array(range(4), pa.int32()), "x": pa.array([NAN, INF, -INF, 1.0])})
    for lit_, ids in (("cast('nan' as double)", [0]), ("cast('inf' as double)", [1]), ("cast('-inf' as float)", [2]),
                      ("cast('+Infinity' as double)", [1])):
        assert _rows(mx.sql(f"select i from t where x = {lit_} order by i", t=t).compute(device="cpu")) == [(i,) for i in ids]
    with pytest.raises(ValueError, match="not a number"):
        mx.sql("select i from t where x = cast('abc' as double)", t=t)


# ── a Parquet file whose row-group statistics ignore NaN ─────────────────────────────────────────────────

@pytest.mark.parametrize("native", ["1", "0"])
@pytest.mark.parametrize("pred,expr", [("x > 1.0", lambda: col("x") > 1.0), ("x >= 100.0", lambda: col("x") >= 100.0),
                                       ("x = 0.2", lambda: col("x") == 0.2), ("x < 0.15", lambda: col("x") < 0.15),
                                       ("x <> 0.2", lambda: col("x") != 0.2), ("x <= 100.0", lambda: col("x") <= 100.0)])
def test_parquet_row_groups_with_nan(tmp_path, monkeypatch, native, pred, expr):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    x = [0.1, 0.2, NAN, 0.3] * 3 + [5.0, 6.0, 7.0, 8.0] * 3 + [NAN] * 12          # stats: 0.1..0.3 | 5..8 | none
    t = pa.table({"i": pa.array(range(len(x)), pa.int32()), "x": pa.array(x)})
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=12)
    con = duckdb.connect()
    con.register("_t", t)
    con.execute("create table t as select * from _t")
    exp = [tuple(r) for r in con.execute(f"select i from t where {pred} order by i").fetchall()]
    got = _rows(mx.scan_parquet(path).filter(expr()).select(col("i")).sort(col("i")).compute(device="cpu"))
    _same(got, exp)


# ── the NaN-aware instructions are only emitted where NaN can tell the difference ───────────────────────────

def _flags(expr, **cols):
    """(op, flag) of every compare instruction in the compiled predicate."""
    import mxframe.pipeline_program as pp
    t = pa.table(cols)
    prog = pp.compile_pipeline([expr], [], t, concat_dicts=True)
    code = prog.code[16:].reshape(-1, 5).tolist()
    return [(ins[0], ins[4] >> 8 if ins[0] == pp.OP_CMPC else ins[4]) for ins in code if ins[0] in (pp.OP_CMPC, *range(pp.OP_LT, pp.OP_NE + 1))]


def test_integer_compares_and_compares_nan_cannot_change_stay_ieee():
    import mxframe.pipeline_program as pp
    f, i = pa.array([1.0, NAN]), pa.array([1, 2], pa.int64())
    assert _flags(col("n") > 1, n=i) == [(pp.OP_CMPC, 0)]                                   # integer column: unchanged
    assert _flags(col("n") >= col("m"), n=i, m=i) == [(pp.OP_GE, 0)]
    assert _flags(col("x") > 1.0, x=f) == [(pp.OP_CMPC, 1)]                                 # NaN is above 1.0
    assert _flags(col("x") >= 1.0, x=f) == [(pp.OP_CMPC, 1)]
    for e in (col("x") < 1.0, col("x") <= 1.0, col("x") == 1.0, col("x") != 1.0):          # NaN is not below / equal 1.0
        assert _flags(e, x=f)[0][1] == 0, e
    assert _flags(col("x") > col("y"), x=f, y=f) == [(pp.OP_GT, 1)]                          # two columns: NaN-aware
    assert _flags((col("x") + 1.0) < 3.0, x=f) == [(pp.OP_LT, 0)]
    assert _flags(lit(3.0) < (col("x") + 1.0), x=f) == [(pp.OP_LT, 1)]                       # c < x: NaN is above c


@pytest.mark.parametrize("native", ["1", "0"])
@pytest.mark.parametrize("pred", [f"n < {NAN_}", f"n > {NAN_}", "n < y", "y > n", f"{NAN_} > n", "n >= y"])
def test_null_integers_against_nan_at_50k_rows(monkeypatch, native, pred):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    rng = np.random.default_rng(3)
    n = 50_000
    y = np.where(rng.random(n) < 0.4, np.nan, rng.normal(0, 3, n))
    t = pa.table({"i": pa.array(np.arange(n, dtype=np.int32)), "y": pa.array(y),
                  "n": pa.array(rng.integers(-3, 4, n).astype(np.int64), mask=rng.random(n) < 0.2)})
    con = duckdb.connect()
    con.register("_t", t)
    con.execute("create table t as select * from _t")
    sql = f"select count(*) c, sum(i) s from t where {pred}".replace("{T}", "double")
    got = _rows(mx.sql(sql, t=t).compute(device="cpu"))
    _same(got, [tuple(r) for r in con.execute(sql).fetchall()])
