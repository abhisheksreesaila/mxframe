"""F-101: integer `%` by a nonzero constant runs in the native bytecode (OP_MOD = fmod on the integer-valued
float64 registers: the sign follows the dividend, exact because the dividend is bound within 2^53) on the native
executor, the older fused pipeline and every GPU kernel. A divisor that is a column or zero could give NULL
(DuckDB: an integer `% 0` is NULL) and float / decimal / past-2^53 operands have Arrow's semantics, so those
decline to Arrow; they must still equal DuckDB.

Every answer is compared with DuckDB's exactly, and the path that produced it is asserted: before this ticket a `%`
anywhere in a filter, key or aggregate moved the whole query to Arrow / `cpu_aot` (`pyarrow_shortcut`, `cpu_aot`)."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import LazyFrame, col, gpu_stream, lit
from mxframe.custom_ops import CustomOpsCompiler
from mxframe.np_arrow import array as A

HAS_GPU = gpu_stream.cuda_available()
N = 300_000
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


def _table():
    r = np.random.default_rng(101)
    a = r.integers(-1000, 1000, N).astype(np.int64)
    big = r.integers(1 << 59, 1 << 61, N).astype(np.int64) * np.where(r.random(N) < 0.5, -1, 1)
    big[:4] = [I64_MIN, I64_MAX, I64_MIN + 1, (1 << 53) + 1]
    edge = r.integers(-(1 << 53) + 1, (1 << 53), N).astype(np.int64)          # the widest the float64 registers hold
    z = r.integers(-3, 4, N).astype(np.int64)                                 # divisors: negatives and zeros
    nul = [None if i % 10 == 0 else int(x) for i, x in enumerate(a)]
    return pa.table({
        "a": A(a), "i32": A(r.integers(-2 ** 31, 2 ** 31, N).astype(np.int32)),
        "i16": A(r.integers(-2 ** 15, 2 ** 15, N).astype(np.int16)), "u8": A(r.integers(0, 256, N).astype(np.uint8)),
        "u32": A(r.integers(0, 2 ** 32, N).astype(np.uint32)), "big": A(big), "edge": A(edge), "z": A(z),
        "nul": pa.array(nul, pa.int64()), "g": A((np.arange(N) % 7).astype(np.int64)),
        "v": A(r.random(N)),
    })


T = _table()
_DUCK = duckdb.connect()
_DUCK.register("t", T)


def _want(q):
    return _DUCK.execute(q).fetchall()


def _norm(rows):
    rows = [tuple(None if v is None else int(v) if isinstance(v, (int, np.integer)) else v for v in row) for row in rows]
    return sorted(rows, key=lambda r: tuple((v is None, str(v)) for v in r))


MODES = [
    pytest.param("1", "cpu", id="native"),
    pytest.param("0", "cpu", id="pipeline"),
    pytest.param("0", "gpu", id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
    pytest.param("1", "auto", id="auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
]
CPU_GPU = [m for m in MODES if m.id != "auto"]


@pytest.fixture
def run(monkeypatch):
    """run(sql, native, device) -> (rows, paths of every plan run, native filter's (path, why))."""
    paths = []
    real = CustomOpsCompiler._compile_and_run

    def recording(self, *a, **k):
        out = real(self, *a, **k)
        paths.append((self.last_compile_provenance or {}).get("path"))
        return out

    monkeypatch.setattr(CustomOpsCompiler, "_compile_and_run", recording)

    def go(q, native, device):
        monkeypatch.setenv("MXFRAME_NATIVE", native)
        if native == "1" and not nx.available():
            pytest.skip("libmxexec_cpu.so not built")
        paths.clear()
        nx._filter_last.update(path=None, decline=None)
        got = mx.sql(q, t=T).compute(device=device)
        return _norm(zip(*[got.column(n).to_pylist() for n in got.column_names])), list(paths), nx.last_filter(), got
    return go


def _on_the_bytecode(paths, native, device):
    """The aggregate ran in a bytecode engine: the executor, the older pipeline, or the GPU's."""
    want = {("1", "cpu"): "native", ("0", "cpu"): "fused_pipeline_cpu", ("0", "gpu"): "fused_pipeline_gpu"}
    return bool(paths) and set(paths) == {want[(native, device)]}


# -- aggregates and keys with `%` stay on the bytecode ----------------------------------------------------

AGGREGATES = [
    "select g, sum(a) as s, count(*) as c from t where a % 7 = 0 and v > 0.3 group by g",
    "select g, sum(a) as s, count(*) as c from t where a % 7 = -3 group by g",
    "select g, count(*) as c from t where a % -7 = 3 or i32 % 5 = 4 group by g",
    "select g, count(*) as c from t where i32 % 1000 = -5 and a % 3 = 1 group by g",
    "select g, sum(a) as s from t where (a % 7) * 2 + (i32 % 5) > 3 group by g",
    "select count(*) as c, sum(a) as s from t where a % 7 = 0",
    "select a % 10 as m, sum(i32) as s, count(*) as c from t group by a % 10",
    "select a % 1000 as m, count(*) as c from t group by a % 1000",
    "select i32 % 100 as m, count(*) as c from t group by i32 % 100",
    "select a % 7 as m, g % 2 as h, count(*) as c from t group by a % 7, g % 2",
    "select g, sum(a % 7) as s, sum(i32 % 100) as r from t group by g",
    "select sum(a % 7) as s from t",
    "select edge % 7 as m, count(*) as c from t group by edge % 7",
    "select count(*) as c from t where edge % 1000003 = 5",
    "select g, min(a % 7) as lo, max(a % 7) as hi from t group by g",
    "select g, sum(edge) as s from t where a % 7 = 0 group by g",
]


@pytest.mark.parametrize("q", AGGREGATES)
@pytest.mark.parametrize("native,device", CPU_GPU)
def test_aggregates_with_a_modulo_are_native_and_equal_duckdb(run, q, native, device):
    rows, paths, _, got = run(q, native, device)
    assert rows == _norm(_want(q))
    assert _on_the_bytecode(paths, native, device), paths


@pytest.mark.parametrize("q", AGGREGATES)
def test_aggregates_with_a_modulo_equal_duckdb_under_auto(run, q):
    if not HAS_GPU:
        pytest.skip("no CUDA GPU")
    assert run(q, "1", "auto")[0] == _norm(_want(q))


FILTERS = [
    "select a, i32, g from t where a % 7 = 0 and v > 0.5",
    "select a, g from t where a % 7 = -3",
    "select a, g from t where a % -7 = 3",
    "select a, g from t where -a % 7 = 3",
    "select i32, g from t where i32 % 100 = -17",
    "select edge from t where edge % 1000003 = 5",
]


@pytest.mark.parametrize("q", FILTERS)
def test_a_row_filter_with_a_modulo_is_the_native_filter(run, q):
    rows, _, (path, why), _ = run(q, "1", "cpu")
    assert rows == _norm(_want(q))
    assert (path, why) == ("native_filter", None)


# -- negatives, zero divisors, NULLs: the sign follows the dividend; the rest declines and stays exact -----

@pytest.mark.parametrize("k", [3, -3, 1, -1, 7, 2 ** 40])
@pytest.mark.parametrize("native,device", CPU_GPU)
def test_the_sign_follows_the_dividend_in_the_kernels(run, native, device, k):
    """-7 % 3 = -1, 7 % -3 = 1: through the bytecode, not Arrow (asserted), for both signs and values up to 2^53.
    The one divisor that is not compiled is -1 (F-126: INT_MIN % -1 overflows, which only Arrow's checked modulo can
    see): same answer, from Arrow."""
    q = (f"select g, sum(edge % {k}) as s, sum(i32 % {k}) as r, count(*) as c from t "
         f"where edge % {k} < 0 or i32 % {k} > 0 group by g")
    rows, paths, _, _ = run(q, native, device)
    assert rows == _norm(_want(q))
    if k == -1:
        assert not set(paths) & {"native", "fused_pipeline_cpu", "fused_pipeline_gpu"}, paths
    else:
        assert _on_the_bytecode(paths, native, device), paths


@pytest.mark.parametrize("native,device", CPU_GPU)
@pytest.mark.parametrize("dt,low", [(np.int8, -128), (np.int16, -2 ** 15), (np.int32, -2 ** 31), (np.int64, I64_MIN)])
@pytest.mark.parametrize("shape", ["global", "grouped", "filter", "key"])
def test_the_minimum_of_a_type_modulo_minus_one_raises_like_duckdb(monkeypatch, native, device, dt, low, shape):
    """INT_MIN % -1 is an overflow in DuckDB ("Overflow in division of -2147483648 / -1"), never 0, on every path."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    n = 40_000
    x = ((np.arange(n) % 100) - 50).astype(dt)
    x[17] = low
    t = pa.table({"x": A(x), "g": A((np.arange(n) % 3).astype(np.int64))})
    q = {"global": "select sum(x % -1) as s from t", "grouped": "select g, sum(x % -1) as s from t group by g",
         "filter": "select count(*) as n from t where x % -1 = 0", "key": "select x % -1 as m, count(*) as n from t group by x % -1"}[shape]
    con = duckdb.connect()
    con.register("t", t)
    with pytest.raises(duckdb.OutOfRangeException, match="Overflow in division of"):
        con.execute(q).fetchall()
    with pytest.raises(OverflowError, match=f"Overflow in division of {low} / -1"):
        mx.sql(q, t=t).compute(device=device)


@pytest.mark.parametrize("native,device", CPU_GPU)
def test_modulo_minus_one_without_the_minimum_is_zero_and_a_wider_divisor_does_not_overflow(monkeypatch, native, device):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    n = 40_000
    x = ((np.arange(n) % 100) - 50).astype(np.int32)
    lowest = x.copy()
    lowest[17] = -2 ** 31
    t = pa.table({"x": A(x), "lowest": A(lowest), "m": A(np.full(n, -1, dtype=np.int64)), "g": A((np.arange(n) % 3).astype(np.int64))})
    con = duckdb.connect()
    con.register("t", t)
    for q in ("select g, sum(x % -1) as s, count(*) as n from t group by g order by g",
              "select g, sum(lowest % m) as s, count(*) as n from t group by g order by g"):    # INT32_MIN % BIGINT -1 = 0
        got = mx.sql(q, t=t).compute(device=device)
        assert [tuple(r.values()) for r in got.to_pylist()] == con.execute(q).fetchall(), q


DECLINED = [
    "select g, sum(a % 0) as s, count(*) as n from t group by g",               # integer zero divisor: NULL
    "select count(*) as c from t where a % 0 = 0",
    "select count(*) as c from t where a % z = 0",                                # a column divisor holding zeros
    "select g, sum(a % z) as s, count(a % z) as c from t group by g",
    "select count(*) as c from t where nul % 7 = 0",                              # NULL in, NULL out
    "select g, count(nul % 7) as c from t group by g",
    "select count(*) as c, sum(big % 1000) as s from t where big % 7 = 3",        # past 2^53, INT64 MIN and MAX
    "select big % 1000 as m, count(*) as c from t group by big % 1000",
    "select count(*) as c from t where v % 0.5 < 0.25",                           # float modulo
    "select count(*) as c from t where a % 2.5 = 0.5",                            # decimal divisor
]


@pytest.mark.parametrize("q", DECLINED)
@pytest.mark.parametrize("native,device", CPU_GPU)
def test_a_zero_or_column_divisor_nulls_and_wide_values_stay_exact_on_arrow(run, q, native, device):
    rows, paths, _, _ = run(q, native, device)
    assert rows == _norm(_want(q))
    if "v %" not in q and "2.5" not in q and "group by big" not in q:       # (those only mask / key on Arrow, then count)
        assert not set(paths) & {"native", "fused_pipeline_cpu", "fused_pipeline_gpu"}, paths


NARROW = [
    "select g, count(*) as c from t where i16 % 100 = -17 group by g",
    "select g, sum(u8) as s from t where u8 % 3 = 1 and a % 2 = 0 group by g",
    "select u32 % 10 as m, count(*) as c from t group by u32 % 10",
    "select g, sum(i16 % 100) as s, sum(u8 % 7) as r from t group by g",
]


@pytest.mark.parametrize("q", NARROW)
@pytest.mark.parametrize("native,device", CPU_GPU)
def test_narrow_integer_columns_run_on_a_bytecode_engine(run, q, native, device):
    """int16 / uint8 / uint32 columns are not the native executor's types; the older pipeline takes them."""
    rows, paths, _, _ = run(q, native, device)
    assert rows == _norm(_want(q))
    assert set(paths) == {"fused_pipeline_gpu" if device == "gpu" else "fused_pipeline_cpu"}, paths


def test_the_extremes_of_int64_are_exact():
    t = pa.table({"a": A(np.array([I64_MIN, I64_MAX, I64_MIN + 1, I64_MAX - 1, -1, 0, 1], dtype=np.int64)),
                  "g": A(np.zeros(7, dtype=np.int64))})
    con = duckdb.connect()
    con.register("t", t)
    for k in (2, 3, 7, 10, -10, 1000000007, I64_MAX, I64_MIN):
        q = f"select a % {k} as m from t"
        got = mx.sql(q, t=t).compute(device="cpu")
        assert got.column("m").to_pylist() == [r[0] for r in con.execute(q).fetchall()], k


# -- every kernel: direct, few, thousands, many groups; reproducible; filter with a `%` --------------------

SHAPES = {"global": (1_200_000, 1), "few": (1_200_000, 7), "thousands": (1_200_000, 1000),
          "many": (1_000_000, 500_000), "staged": (400_000, 200_000)}
KMODES = [
    pytest.param("1", "cpu", False, id="native"),
    pytest.param("1", "cpu", True, id="native-repro"),
    pytest.param("0", "cpu", False, id="pipeline"),
    pytest.param("0", "cpu", True, id="pipeline-repro"),
    pytest.param("0", "gpu", False, id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
    pytest.param("0", "gpu", True, id="gpu-repro", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
]


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("native,device,repro", KMODES)
def test_a_modulo_filter_and_an_integer_sum_past_2_53_in_every_kernel(monkeypatch, native, device, repro, shape):
    """Sums of ~2^60 (exact in int64 only) over the rows whose `a % 7 = 0`: every accumulator site next to the new op."""
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    rows, groups = SHAPES[shape]
    r = np.random.default_rng(5)
    per = max(rows // groups, 1)
    lo = min(1 << (60 - int(np.ceil(np.log2(per)))), 1 << 52)
    v = r.integers(lo, 2 * lo, rows).astype(np.int64) * np.where(r.random(rows) < 0.2, -1, 1)
    t = pa.table({"v": A(v), "a": A(r.integers(-100_000, 100_000, rows).astype(np.int64)),
                  "g": A((r.permutation(rows) % groups).astype(np.int64))})
    con = duckdb.connect()
    con.register("t", t)
    keyed = groups > 1
    lf = LazyFrame(t).filter((col("a") % lit(7)) == lit(0))
    lf = lf.groupby("g") if keyed else lf.groupby()
    lf = lf.agg(col("v").sum().alias("s"), col("v").count().alias("c"))
    if keyed:
        lf = lf.sort("g")
    got = lf.compute(device=device, reproducible=repro)
    q = "select {} sum(v), count(v) from t where a % 7 = 0 {}".format("g," if keyed else "", "group by g order by g" if keyed else "")
    want = [tuple(None if x is None else int(x) for x in row) for row in con.execute(q).fetchall()]
    names = (["g"] if keyed else []) + ["s", "c"]
    assert [tuple(row[n] for n in names) for row in got.to_pylist()] == want
    if rows // groups > 100:
        assert max(abs(w[-2]) for w in want) > 2 ** 53
    prov = lf.last_compile_provenance["path"]
    if native == "1":
        assert prov == "native", prov
    elif device == "gpu" and (not repro or shape in ("global", "few", "thousands")):
        assert prov == "fused_pipeline_gpu", prov
    elif device == "cpu":
        assert prov == "fused_pipeline_cpu", prov
