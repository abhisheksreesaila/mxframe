"""F-060b: integer `+ - *` run in int64 in the bytecode (OP_IADD / OP_ISUB / OP_IMUL on registers that hold int64
bits), with a runtime flag instead of F-037's pre-pass: an int64 result outside int64 sets the flag (the kernel
returns E_OVERFLOW and the checked Arrow path raises like DuckDB), and an int64 value that meets a float operation
(OP_I2F) outside +-2^53 sets it too (Arrow answers). No query reads a column's min / max first: `col_bound` and
`_minmax` are never called, and a column only the int64 ops read is not range-checked either.

Every answer is compared with DuckDB's exact value, and the path (or GPU mode) that produced it is asserted."""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
import mxframe.native_exec as nx
import mxframe.pipeline_exec as pe
import mxframe.pipeline_kernels as pk
from mxframe import LazyFrame, col, gpu_stream, lit
from mxframe.custom_ops import CustomOpsCompiler
from mxframe.io_parquet import scan_parquet
from mxframe.np_arrow import array as A
from mxframe.pipeline_program import compile_pipeline

HAS_GPU = gpu_stream.cuda_available()

# (rows, groups): a global aggregate, a few groups, a thousand, many (the shared atomic output: ~1 row per group), and
# more than the table path holds (the staged path, in reproducible mode)
SHAPES = {
    "global": (300_000, 1),
    "few": (300_000, 7),
    "thousands": (300_000, 1000),
    "many": (1_200_000, 1_000_000),
    "staged": (400_000, 200_000),
}


def _table(rows, groups, hashed=False, seed=60):
    """a, b are 31-bit integers, so a * b is up to 2^62: past 2^53 (a float64 product rounds) and inside int64. c is
    a * b - d for a small d, so a * b - c = d exactly and a group's sum is small (and fits int64 whatever the rows)."""
    r = np.random.default_rng(seed)
    sign = lambda: np.where(r.random(rows) < 0.5, -1, 1)
    a = r.integers(1 << 30, 1 << 31, rows).astype(np.int64) * sign()
    b = r.integers(1 << 30, 1 << 31, rows).astype(np.int64) * sign()
    d = r.integers(-1000, 1000, rows).astype(np.int64)
    g = (r.permutation(rows) % groups).astype(np.int64)
    cols = {"a": A(a), "b": A(b), "c": A(a * b - d), "d": A(d), "f": A(r.integers(-1000, 1000, rows) * 0.5),
            "w": A(r.integers(-1000, 1000, rows).astype(np.int32)), "g": A(g)}
    if hashed:
        cols["g"] = A(g.astype(np.float64) * 0.5 + 0.25)         # float keys: hashed groups
    return pa.table(cols)


MODES = [
    pytest.param("1", "cpu", False, id="native"),
    pytest.param("1", "cpu", True, id="native-repro"),
    pytest.param("0", "cpu", False, id="pipeline"),
    pytest.param("0", "cpu", True, id="pipeline-repro"),
    pytest.param("0", "gpu", False, id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
    pytest.param("0", "gpu", True, id="gpu-repro", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
]


@pytest.fixture
def prepass(monkeypatch):
    """Counts every pass over the data that decides whether an integer expression is exact: F-037's `col_bound` and
    `_minmax`, and F-032's range check of an int64 column the float64 registers load."""
    from mxframe import join_engine
    from mxframe.pipeline_program import _Builder
    calls = []
    real_mm = join_engine._minmax
    monkeypatch.setattr(join_engine, "_minmax", lambda *a, **k: (calls.append("_minmax"), real_mm(*a, **k))[1])
    real_cb = getattr(_Builder, "col_bound", None)
    if real_cb is not None:                                         # (F-060b deletes it)
        monkeypatch.setattr(_Builder, "col_bound", lambda *a, **k: (calls.append("col_bound"), real_cb(*a, **k))[1])
    real_rng = pe._column_past_2_53
    monkeypatch.setattr(pe, "_column_past_2_53", lambda c: (calls.append("range") if c.type == pa.int64() else None, real_rng(c))[1])
    return calls


def _check_path(prov, native, device, repro, shape):
    if native == "1" and device == "cpu":
        assert prov["path"] == "native", prov
    elif device == "gpu":
        if not repro or shape in ("global", "few", "thousands"):
            assert prov["path"] == "fused_pipeline_gpu", prov
    else:
        assert prov["path"] == "fused_pipeline_cpu", prov


def _gpu_mode(t, aggs, groups, repro):
    """The GPU mode (0 to 4, negative: declined) the kernel library picks for this query's program."""
    prog = compile_pipeline([], aggs, t)
    return pk._load_gpu().mx_pipeline_gpu_mode(groups, prog.n_slots, int(prog.code[1]), int(len(prog.code)), int(repro))


# The mode the GPU runs for each shape of the two-slot queries below: 0 per-thread partials (one group), 3 per-thread in
# shared memory (a few groups), 1 shared table + atomics (a thousand), 2 global atomics (many), 4 the ordered table
# (reproducible), -6 when the ordered table does not hold the groups (the CPU's defined order answers)
GPU_MODES = {"global": (0, 0), "few": (3, 3), "thousands": (1, 4), "many": (2, -6), "staged": (2, -6)}


@pytest.mark.parametrize("hashed", [False, True], ids=["direct", "hashed"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("native,device,repro", MODES)
def test_int64_products_are_exact_past_2_53_with_no_pre_pass(monkeypatch, prepass, native, device, repro, shape, hashed):
    rows, groups = SHAPES[shape]
    if hashed and (groups == 1 or device == "gpu"):
        pytest.skip("hashed groups: the executor's float keys")
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(rows, groups, hashed)
    keyed = groups > 1
    aggs = [(col("a") * col("b") - col("c")).sum().alias("s"), (col("a") + col("b")).max().alias("m")]
    lf = LazyFrame(t).groupby("g").agg(*aggs).sort("g") if keyed else LazyFrame(t).groupby().agg(*aggs)
    got = lf.compute(device=device, reproducible=repro)
    con = duckdb.connect()
    con.register("t", t)
    q = "select {} sum(a * b - c) as s, max(a + b) as m from t {}".format(
        "g," if keyed else "", "group by g order by g" if keyed else "")
    want = con.execute(q).fetchall()
    first = 1 if keyed else 0                                       # (the key is whatever DuckDB says: a float when hashed)
    assert [tuple(r.values()) for r in got.to_pylist()] == [r[:first] + tuple(int(x) for x in r[first:]) for r in want]
    assert got.schema.field("s").type == pa.int64() and got.schema.field("m").type == pa.int64()
    assert prepass == []                                            # no min / max, no range check of a, b, c
    _check_path(lf.last_compile_provenance, native, device, repro, shape)
    if device == "gpu":
        assert _gpu_mode(t, [aggs[0], aggs[1]], groups, repro) == GPU_MODES[shape][1 if repro else 0]


@pytest.mark.parametrize("native,device,repro", MODES)
def test_a_product_past_2_53_is_the_exact_integer_not_a_rounded_double(monkeypatch, prepass, native, device, repro):
    """The sum of products of odd numbers near 2^27: a double rounds each product (53 bits), int64 does not."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    n = 40
    a = np.arange(n, dtype=np.int64) * 2 + (1 << 27) + 1
    b = np.arange(n, dtype=np.int64) * 2 + (1 << 27) + 3
    t = pa.table({"a": A(a), "b": A(b)})
    assert int((a * b).max()) > 2 ** 54 and any(int(x) != int(float(x)) for x in a * b)
    lf = LazyFrame(t).groupby().agg((col("a") * col("b")).sum().alias("s"), (col("a") * col("b")).min().alias("lo"),
                                    (col("a") * col("b")).max().alias("hi"))
    got = lf.compute(device=device, reproducible=repro)
    assert got.to_pylist() == [{"s": int((a * b).sum()), "lo": int((a * b).min()), "hi": int((a * b).max())}]
    assert prepass == []
    assert lf.last_compile_provenance["path"] == {("1", "cpu"): "native", ("0", "cpu"): "fused_pipeline_cpu",
                                                  ("0", "gpu"): "fused_pipeline_gpu"}[(native, device)]


@pytest.mark.parametrize("native,device,repro", MODES)
def test_filters_keys_and_sums_over_int64_expressions(monkeypatch, prepass, native, device, repro):
    """An int64 expression under a filter, as a group key and as a sum, where every intermediate is past 2^53 but the
    value that meets the compare / key (a * b - c = d) is small."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(100_000, 1)
    con = duckdb.connect()
    con.register("t", t)
    lf = (LazyFrame(t).filter((col("a") * col("b") - col("c")) > lit(0)).groupby()
          .agg((col("a") * col("b") - col("c")).sum().alias("s"), col("w").count().alias("n")))
    got = lf.compute(device=device, reproducible=repro)
    want = con.execute("select sum(a * b - c), count(w) from t where a * b - c > 0").fetchall()
    assert [tuple(r.values()) for r in got.to_pylist()] == [tuple(int(x) for x in want[0])]
    assert prepass == []
    lf = LazyFrame(t).groupby((col("a") * col("b") - col("c")) % lit(4)).agg(col("w").count().alias("n"))
    lf = lf.sort("n")
    got = lf.compute(device=device, reproducible=repro)
    want = con.execute("select count(w) n from t group by (a * b - c) % 4 order by n").fetchall()
    assert sorted(got.column("n").to_pylist()) == sorted(r[0] for r in want)


# -- overflow raises like DuckDB, on every kernel ----------------------------------------------------------

@pytest.mark.parametrize("native,device,repro", MODES)
@pytest.mark.parametrize("op,words,a,b", [
    ("mul", "Overflow in multiplication of INT64", 1 << 32, 1 << 32),
    ("mul", "Overflow in multiplication of INT64", -(1 << 32), (1 << 31) + 1),
    ("add", "Overflow in addition of INT64", (1 << 62) + 5, (1 << 62) + 5),
    ("add", "Overflow in addition of INT64", -(1 << 62) - 5, -(1 << 62) - 7),
    ("sub", "Overflow in subtraction of INT64", -(1 << 62) - 5, (1 << 62) + 5),
    ("sub", "Overflow in subtraction of INT64", (1 << 62) + 5, -(1 << 62) - 5),
])
def test_an_int64_result_outside_int64_raises_like_duckdb(monkeypatch, native, device, repro, op, words, a, b):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    sym = {"mul": "*", "add": "+", "sub": "-"}[op]
    rows = 5000
    av = np.full(rows, 3, dtype=np.int64)
    bv = np.full(rows, 4, dtype=np.int64)
    av[rows // 2 + 1], bv[rows // 2 + 1] = a, b                     # one row in the middle overflows
    t = pa.table({"a": A(av), "b": A(bv), "g": A((np.arange(rows) % 3).astype(np.int64))})
    con = duckdb.connect()
    con.register("t", t)
    e = {"mul": col("a") * col("b"), "add": col("a") + col("b"), "sub": col("a") - col("b")}[op]
    with pytest.raises(duckdb.OutOfRangeException, match=words):
        con.execute(f"select sum(a {sym} b) from t").fetchall()
    for lf in (LazyFrame(t).groupby().agg(e.sum().alias("s")), LazyFrame(t).groupby("g").agg(e.sum().alias("s")),
               LazyFrame(t).groupby().agg(e.max().alias("s")), LazyFrame(t).filter(e > lit(0)).groupby().agg(col("a").sum().alias("s"))):
        with pytest.raises(OverflowError, match=words):
            lf.compute(device=device, reproducible=repro)


@pytest.mark.parametrize("native,device", [("1", "cpu"), ("0", "cpu"), pytest.param("0", "gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
def test_the_largest_products_that_fit_are_exact_and_the_next_one_raises(monkeypatch, native, device):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    imax, imin = (1 << 63) - 1, -(1 << 63)
    # 3037000499 ** 2 = 9223372030926249001 <= 2^63 - 1 < 3037000500 ** 2; -2^31 * 2^32 = -2^63 = INT64_MIN fits
    rows = [(3037000499, 3037000499), (-(1 << 31), 1 << 32), (1, imax), (imax, 1), (imin, 1), (-1, imax)]
    for a, b in rows:
        t = pa.table({"a": A(np.array([a, 2], dtype=np.int64)), "b": A(np.array([b, 3], dtype=np.int64))})
        got = LazyFrame(t).groupby().agg((col("a") * col("b")).max().alias("hi")).compute(device=device)
        assert got.to_pylist() == [{"hi": max(a * b, 6)}], (a, b)
    for a, b in [(3037000500, 3037000500), (imin, -1), (-1, imin), (imax, 2), (1 << 32, -(1 << 31) - 1)]:
        t = pa.table({"a": A(np.array([a, 2], dtype=np.int64)), "b": A(np.array([b, 3], dtype=np.int64))})
        with pytest.raises(OverflowError, match="Overflow in multiplication of INT64"):
            LazyFrame(t).groupby().agg((col("a") * col("b")).max().alias("hi")).compute(device=device)


@pytest.mark.parametrize("native,device", [("1", "cpu"), ("0", "cpu"), pytest.param("0", "gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
def test_rows_a_filter_drops_do_not_overflow(monkeypatch, native, device):
    """DuckDB filters first: an overflowing product in a row the WHERE removed is never computed."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    rows = 3000
    a = np.arange(rows, dtype=np.int64)
    b = np.full(rows, 5, dtype=np.int64)
    a[1500], b[1500] = 1 << 40, 1 << 40                               # the product is 2^80, in a row w < 0 removes
    w = np.ones(rows, dtype=np.int32)
    w[1500] = -1
    t = pa.table({"a": A(a), "b": A(b), "w": A(w)})
    con = duckdb.connect()
    con.register("t", t)
    lf = LazyFrame(t).filter(col("w") > lit(0)).groupby().agg((col("a") * col("b")).sum().alias("s"))
    got = lf.compute(device=device)
    assert got.to_pylist()[0]["s"] == int(con.execute("select sum(a * b) from t where w > 0").fetchall()[0][0])
    assert lf.last_compile_provenance["path"] == {"1": "native", "0": "fused_pipeline_" + device}[native]


# -- an int64 expression next to floats -------------------------------------------------------------------------

@pytest.mark.parametrize("native,device", [("1", "cpu"), ("0", "cpu"), pytest.param("0", "gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
def test_an_int64_expression_within_2_53_meets_floats_on_the_bytecode(monkeypatch, prepass, native, device):
    """a * b - c = d is small: converting it for a float add or compare is exact, so no flag, and the kernel answers."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(100_000, 7)
    con = duckdb.connect()
    con.register("t", t)
    queries = [
        ("sum(f * 2.0 + (a * b - c)) as s", "s", lambda: (col("f") * lit(2.0) + (col("a") * col("b") - col("c"))).sum()),
        ("sum(f + (a * b - c)) as s", "s", lambda: (col("f") + (col("a") * col("b") - col("c"))).sum()),
        ("max((a * b - c) / 4.0) as s", "s", lambda: ((col("a") * col("b") - col("c")) / lit(4.0)).max()),
    ]
    for sql, name, build in queries:
        lf = LazyFrame(t).groupby("g").agg(build().alias(name)).sort("g")
        got = lf.compute(device=device)
        want = con.execute(f"select g, {sql} from t group by g order by g").fetchall()
        assert [tuple(r.values()) for r in got.to_pylist()] == [tuple(r) for r in want], sql
        assert prepass == []
        assert lf.last_compile_provenance["path"] == {("1", "cpu"): "native", ("0", "cpu"): "fused_pipeline_cpu",
                                                      ("0", "gpu"): "fused_pipeline_gpu"}[(native, device)], sql
    lf = LazyFrame(t).filter((col("a") * col("b") - col("c")) > col("f")).groupby("g").agg(col("w").count().alias("n")).sort("g")
    got = lf.compute(device=device)
    want = con.execute("select g, count(w) from t where a * b - c > f group by g order by g").fetchall()
    assert [tuple(r.values()) for r in got.to_pylist()] == [tuple(r) for r in want]
    assert prepass == [] and lf.last_compile_provenance["path"] != "arrow_null_semantics"


@pytest.mark.parametrize("native,device", [("1", "cpu"), ("0", "cpu"), pytest.param("0", "gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
def test_an_int64_past_2_53_that_meets_a_float_op_declines_to_the_exact_path(monkeypatch, prepass, native, device):
    """a * b is ~2^60 and is added to a float / compared with a float / divided: the register cannot hold it exactly
    as a double, the kernel flags it, and Arrow answers with DuckDB's value (an int64 cast to double)."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(20_000, 5)
    con = duckdb.connect()
    con.register("t", t)
    for sql, build in [
        ("sum(a * b + f)", lambda: (col("a") * col("b") + col("f")).sum()),
        ("sum(a * b / 3.0)", lambda: (col("a") * col("b") / lit(3.0)).sum()),
        ("sum(f * (a * b))", lambda: (col("f") * (col("a") * col("b"))).sum()),
    ]:
        lf = LazyFrame(t).groupby("g").agg(build().alias("s")).sort("g")
        got = lf.compute(device=device)
        want = con.execute(f"select g, {sql} from t group by g order by g").fetchall()
        for r, w in zip(got.to_pylist(), want):
            assert r["g"] == w[0] and r["s"] == pytest.approx(w[1], rel=1e-9), sql
        assert lf.last_compile_provenance["path"] not in ("native", "fused_pipeline_cpu", "fused_pipeline_gpu"), sql
    # (a min / max of a float expression is computed by a projection under the aggregate: Arrow's checked kernels)
    lf = LazyFrame(t).groupby("g").agg((col("f") * (col("a") * col("b"))).min().alias("s")).sort("g")
    got = lf.compute(device=device)
    want = con.execute("select g, min(f * (a * b)) from t group by g order by g").fetchall()
    assert [(r["g"], r["s"]) for r in got.to_pylist()] == [tuple(r) for r in want]
    lf = LazyFrame(t).filter(col("a") * col("b") > col("f")).groupby().agg(col("w").count().alias("n"))
    got = lf.compute(device=device)
    assert got.to_pylist() == [{"n": con.execute("select count(w) from t where a * b > f").fetchall()[0][0]}]


# -- SQL, Parquet, streamed, split, join -----------------------------------------------------------------------

SQL = [
    "select sum(a * b - c) as s, sum(a + b - a) as t from t",
    "select g, sum(a * b - c) as s, min(a * b) as lo, max(a * b) as hi from t group by g order by g",
    "select g, sum(a * b - c) as s from t where a * b - c > 0 group by g order by g",
    "select g, sum(a * b - c) as s from t group by g having sum(a * b - c) > 0 order by g",
    "select (a * b - c) % 5 as k, sum(a - b + b) as s from t group by k order by k",
    "select sum(a * 3 - b * 5 + 7) as s, max(a * 2 + b) as m from t",
]


@pytest.mark.parametrize("q", SQL)
@pytest.mark.parametrize("native,device", [("1", "cpu"), ("0", "cpu"), pytest.param("0", "gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
def test_sql_equals_duckdb_with_no_pre_pass(monkeypatch, prepass, native, device, q):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(60_000, 7)
    con = duckdb.connect()
    con.register("t", t)
    out = con.execute(q).arrow()
    want = out.read_all() if hasattr(out, "read_all") else out
    got = mx.sql(q, t=t).compute(device=device)
    assert got.to_pylist() == want.to_pylist()
    assert prepass == []


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
def test_a_parquet_scan_with_many_row_groups_and_a_streamed_run(tmp_path, monkeypatch, prepass, grouped):
    t = _table(120_000, 7)
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=500)
    want = duckdb.connect().execute(f"select {'g,' if grouped else ''} sum(a * b - c), max(a + b) from read_parquet('{path}')"
                                    f"{' group by g order by g' if grouped else ''}").fetchall()
    out = {}
    for budget in ("64K", "2M", "64G"):
        monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", budget)
        lf = scan_parquet(path)
        lf = (lf.groupby("g") if grouped else lf.groupby()).agg((col("a") * col("b") - col("c")).sum().alias("s"),
                                                               (col("a") + col("b")).max().alias("m"))
        got = lf.compute(device="cpu")
        out[budget] = (sorted(tuple(r.values()) for r in got.to_pylist()), lf.last_compile_provenance.get("streamed", {}).get("batches"))
        assert out[budget][0] == sorted(tuple(int(x) for x in r) for r in want), budget
    assert out["64G"][1] is None and out["64K"][1] > out["2M"][1] > 1
    assert prepass == []


def test_a_streamed_overflow_raises(tmp_path, monkeypatch):
    a = np.full(40_000, 3, dtype=np.int64)
    a[31_337] = 1 << 40
    t = pa.table({"a": A(a), "b": A(a.copy())})
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=500)
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64K")
    with pytest.raises(OverflowError, match="Overflow in multiplication of INT64"):
        scan_parquet(path).groupby().agg((col("a") * col("b") * col("a")).sum().alias("s")).compute(device="cpu")


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")
@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
def test_a_cpu_gpu_split_runs_the_int64_ops_on_both_devices(fraction, grouped):
    import mxframe.coproc_fused as cf
    cf.clear_residency()
    t = _table(600_000, 37)
    con = duckdb.connect()
    con.register("t", t)
    want = con.execute(f"select {'g,' if grouped else ''} sum(a * b - c), sum(a * 3 - b * 5), max(w) from t"
                       f"{' group by g order by g' if grouped else ''}").fetchall()

    def query():
        lf = LazyFrame(t).groupby("g") if grouped else LazyFrame(t).groupby()
        return lf.agg((col("a") * col("b") - col("c")).sum().alias("s"), (col("a") * lit(3) - col("b") * lit(5)).sum().alias("q"),
                      col("w").max().alias("m"))
    with cf.policy(min_upload_bytes=0, gpu_fraction=fraction):
        query().compute(device="cpu")
        cf.wait_for_uploads(30)
        lf = query().sort("g") if grouped else query()
        got = lf.compute(device="cpu")
    assert [tuple(r.values()) for r in got.to_pylist()] == [tuple(int(x) for x in r) for r in want]
    if 0.0 < fraction < 1.0:
        assert lf.last_compile_provenance["path"] == "fused_pipeline_split", lf.last_compile_provenance["path"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
@pytest.mark.parametrize("q", [
    "select grp, sum(a * b - c) as s from f join d on f.k = d.k group by grp order by grp",
    "select sum(a * b - c) as s from f join d on f.k = d.k where grp = 1",
    "select grp, sum(a * b - c + bonus) as s from f join d on f.k = d.k group by grp order by grp",
])
def test_integer_arithmetic_over_a_join_is_exact(device, q):
    n = 50_000
    r = np.random.default_rng(3)
    a = r.integers(1 << 30, 1 << 31, n).astype(np.int64)
    b = r.integers(1 << 30, 1 << 31, n).astype(np.int64)
    dd = r.integers(-1000, 1000, n).astype(np.int64)
    f = pa.table({"k": A(r.integers(0, 100, n)), "a": A(a), "b": A(b), "c": A(a * b - dd)})
    d = pa.table({"k": A(np.arange(100)), "grp": A(np.arange(100) % 5), "bonus": A(np.arange(100) << 40)})
    con = duckdb.connect()
    con.register("f", f)
    con.register("d", d)
    want = con.execute(q).fetchall()
    got = mx.sql(q, f=f, d=d).compute(device=device)
    assert [tuple(r.values()) for r in got.to_pylist()] == [tuple(int(x) for x in r) for r in want]


# -- the older pipeline paths and the executor's bytecode sites ------------------------------------------------

def test_no_int64_column_is_range_checked_when_only_the_int_ops_read_it(monkeypatch, prepass):
    """`a`, `b` and `c` are read by OP_LOADI only (their values may pass 2^53): the executor's native pass over int64 value
    columns (E_INEXACT) is skipped for them, so an int64 column holding 2^62 does not decline."""
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(100_000, 3)
    lf = LazyFrame(t).groupby("g").agg((col("a") * col("b") - col("c")).sum().alias("s")).sort("g")
    lf.compute(device="cpu")
    assert lf.last_compile_provenance["path"] == "native" and prepass == []


@pytest.mark.parametrize("native", ["1", "0"], ids=["native", "pipeline"])
def test_a_column_also_read_as_a_float_is_still_range_checked(monkeypatch, native):
    """`a` is also compared as a double: past 2^53 that read declines (as before), and the answer is still exact."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(20_000, 3)
    con = duckdb.connect()
    con.register("t", t)
    t2 = t.set_column(t.schema.get_field_index("a"), "a", A(t.column("a").to_numpy() * (1 << 31)))
    con.register("t2", t2)
    lf = LazyFrame(t2).filter(col("a") > col("f")).groupby().agg((col("a") + col("b")).max().alias("m"))
    got = lf.compute(device="cpu")
    want = con.execute("select max(a + b) from t2 where a > f").fetchall()[0][0]
    assert got.to_pylist() == [{"m": int(want)}]


# -- the executor's other entry points: row filters, the top-k filter, the fused join's fact filter ---------------

@pytest.mark.parametrize("native", ["1", "0"], ids=["native", "pipeline"])
def test_row_filters_run_on_int64_ops_and_decline_when_a_double_cannot_hold_the_value(monkeypatch, prepass, native):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(20_000, 3).append_column("row", A(np.arange(20_000, dtype=np.int64)))
    con = duckdb.connect()
    con.register("t", t)
    # a * b - c = d is small: the compare converts it exactly, no flag, the native filter answers
    got = LazyFrame(t).filter((col("a") * col("b") - col("c")) > lit(0)).compute(device="cpu")
    want = [r[0] for r in con.execute("select row from t where a * b - c > 0 order by row").fetchall()]
    assert sorted(got.column("row").to_pylist()) == want and 0 < len(want) < 20_000
    if native == "1":
        assert nx.last_filter()[0] == "native_filter", nx.last_filter()
    assert prepass == []
    # a * b is past 2^53 and meets a double compare: the kernel flags it and Arrow answers
    got = LazyFrame(t).filter(col("a") * col("b") > col("f")).compute(device="cpu")
    want = [r[0] for r in con.execute("select row from t where a * b > f order by row").fetchall()]
    assert sorted(got.column("row").to_pylist()) == want and 0 < len(want) < 20_000
    if native == "1":
        assert "code -21" in (nx.last_filter()[1] or ""), nx.last_filter()


def test_a_filter_whose_product_overflows_raises_from_the_native_filter(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    a = np.full(3000, 5, dtype=np.int64)
    a[1234] = 1 << 40
    t = pa.table({"a": A(a), "row": A(np.arange(3000, dtype=np.int64))})
    with pytest.raises(OverflowError, match="Overflow in multiplication of INT64"):
        LazyFrame(t).filter(col("a") * col("a") * col("a") > lit(0)).compute(device="cpu")


@pytest.mark.parametrize("native", ["1", "0"], ids=["native", "pipeline"])
def test_a_top_k_filter_over_int64_arithmetic_equals_duckdb(monkeypatch, prepass, native):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(20_000, 3).append_column("row", A(np.arange(20_000, dtype=np.int64)))
    con = duckdb.connect()
    con.register("t", t)
    want = [r[0] for r in con.execute("select row from t where a * b - c > 0 order by d desc, row limit 10").fetchall()]
    got = LazyFrame(t).filter((col("a") * col("b") - col("c")) > lit(0)).sort("d", "row", descending=[True, False]).limit(10)
    assert got.compute(device="cpu").column("row").to_pylist() == want
    assert prepass == []


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
@pytest.mark.parametrize("flagged", [False, True], ids=["exact", "flagged"])
def test_the_fused_joins_fact_filter_over_int64_arithmetic(device, flagged):
    n = 50_000
    r = np.random.default_rng(5)
    a = r.integers(1 << 30, 1 << 31, n).astype(np.int64)
    b = r.integers(1 << 30, 1 << 31, n).astype(np.int64)
    c = a * b - r.integers(-1000, 1000, n).astype(np.int64)
    f = pa.table({"k": A(r.integers(0, 100, n)), "a": A(a), "b": A(b), "c": A(c), "w": A(np.ones(n)), "fl": A(r.integers(-10**9, 10**9, n) * 1e9)})
    d = pa.table({"k": A(np.arange(100)), "grp": A(np.arange(100) % 5)})
    cond = "a * b > fl" if flagged else "a * b - c > 0"                 # (flagged: an int64 past 2^53 meets a double compare)
    q = f"select grp, sum(w) as s from f join d on f.k = d.k where {cond} group by grp order by grp"
    con = duckdb.connect()
    con.register("f", f)
    con.register("d", d)
    want = con.execute(q).fetchall()
    got = mx.sql(q, f=f, d=d).compute(device=device)
    assert [(r_["grp"], r_["s"]) for r_ in got.to_pylist()] == [tuple(r_) for r_ in want]
