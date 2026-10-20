"""F-060c: the MIN / MAX of an integer column is taken in int64 by every kernel (OP_LOADI feeds OP_AGG_IMIN /
OP_AGG_IMAX; the slot's identity is INT64_MAX / INT64_MIN as bits), so it is exact past 2^53, at INT64 MIN / MAX, and
comes back as the column's own integer type with no float64 on the way. Before this ticket a column with a value past
2^53 was answered by Arrow (`arrow_null_semantics`, after a `min_max` pass over the column: F-090) and the kernels'
float64 slots could not hold it.

Every answer is compared with DuckDB's, type included, and the path that produced it is asserted."""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx
import omxcore.native_exec as nx
from omxcore import LazyFrame, col, gpu_stream, lit
from omxcore.custom_ops import CustomOpsCompiler
from omxcore.io_parquet import scan_parquet
from omxcore.np_arrow import array as A

HAS_GPU = gpu_stream.cuda_available()
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
N = 30_000

# integer columns the min / max is asked of (name -> DuckDB's type name is read back from DuckDB itself)
INTS = ["big", "small", "i32", "i16", "u8", "u32", "neg"]


def _table():
    r = np.random.default_rng(60)
    k = np.arange(N)
    big = ((1 << 60) + r.integers(-(1 << 40), 1 << 40, N)) * np.where(r.random(N) < 0.5, -1, 1)
    big[:6] = [I64_MIN, I64_MAX, I64_MIN + 1, I64_MAX - 1, (1 << 53) + 1, -(1 << 53) - 1]
    g = (k % 7).astype(np.int64)
    big[g == 5] = I64_MAX                              # group 5 holds only INT64_MAX: its min IS the identity of a min slot
    big[g == 6] = I64_MIN                              # group 6 only INT64_MIN: the identity of a max slot
    return pa.table({
        "big": A(big.astype(np.int64)), "neg": A((-(1 << 55) - r.integers(0, 1000, N)).astype(np.int64)),
        "small": A(r.integers(-1000, 1000, N).astype(np.int64)),
        "i32": A(r.integers(-2 ** 31, 2 ** 31, N).astype(np.int32)),
        "i16": A(r.integers(-2 ** 15, 2 ** 15, N).astype(np.int16)),
        "u8": A(r.integers(0, 256, N).astype(np.uint8)), "u32": A(r.integers(0, 2 ** 32, N).astype(np.uint32)),
        "nul": pa.array([None if x % 9 == 0 else int(b) for x, b in zip(k, big)], pa.int64()),
        "g": A(g), "v": A(r.random(N)),
    })


T = _table()
_DUCK = duckdb.connect()
_DUCK.register("t", T)


def _want(q):
    cur = _DUCK.execute(q)
    return cur.fetchall(), [d[1] for d in cur.description]


def _duck_arrow(q):
    out = _DUCK.execute(q).arrow()
    return out.read_all() if hasattr(out, "read_all") else out


MODES = [
    pytest.param("1", "cpu", id="native"),
    pytest.param("0", "cpu", id="pipeline"),
    pytest.param("0", "gpu", id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
]
EXPECT = {("1", "cpu"): "native", ("0", "cpu"): "fused_pipeline_cpu", ("0", "gpu"): "fused_pipeline_gpu"}
NARROW = ("i16", "u8", "u32")      # not the native executor's column types (int32 / int64 / float): the older pipeline takes them


def _expect(native, device, c=None):
    return "fused_pipeline_cpu" if c in NARROW and (native, device) == ("1", "cpu") else EXPECT[(native, device)]


@pytest.fixture
def run(monkeypatch):
    """run(lazyframe builder, native, device) -> (table, paths of every plan run)."""
    paths = []
    real = CustomOpsCompiler._compile_and_run

    def recording(self, *a, **k):
        out = real(self, *a, **k)
        paths.append((self.last_compile_provenance or {}).get("path"))
        return out

    monkeypatch.setattr(CustomOpsCompiler, "_compile_and_run", recording)

    def go(build, native, device, table=T, **kw):
        monkeypatch.setenv("MXFRAME_NATIVE", native)
        if native == "1" and not nx.available():
            pytest.skip("libmxexec_cpu.so not built")
        paths.clear()
        return build(LazyFrame(table)).compute(device=device, **kw), list(paths)
    return go


def _rows(t):
    return sorted(zip(*[t.column(n).to_pylist() for n in t.column_names]), key=lambda r: tuple((v is None, v) for v in r))


# -- global and grouped, every width, against DuckDB's values and types ------------------------------------

@pytest.mark.parametrize("c", INTS)
@pytest.mark.parametrize("native,device", MODES)
def test_a_global_integer_min_max_is_exact_typed_and_on_the_bytecode(run, native, device, c):
    got, paths = run(lambda f: f.groupby().agg(col(c).min().alias("lo"), col(c).max().alias("hi")), native, device)
    want = _duck_arrow(f"select min({c}) as lo, max({c}) as hi from t")
    assert got.schema == want.schema
    assert got.to_pylist() == want.to_pylist()
    assert set(paths) == {_expect(native, device, c)}, paths


@pytest.mark.parametrize("c", INTS)
@pytest.mark.parametrize("native,device", MODES)
def test_a_grouped_integer_min_max_is_exact_typed_and_on_the_bytecode(run, native, device, c):
    got, paths = run(lambda f: f.groupby("g").agg(col(c).min().alias("lo"), col(c).max().alias("hi")).sort("g"),
                     native, device)
    want = _duck_arrow(f"select g, min({c}) as lo, max({c}) as hi from t group by g order by g")
    assert got.schema == want.schema
    assert got.to_pylist() == want.to_pylist()
    assert set(paths) == {_expect(native, device, c)}, paths


def test_the_values_the_float_slots_could_not_hold():
    """2^53 + 1, INT64 MIN and MAX, and a group whose only value is the identity of its slot."""
    want = {r[0]: r[1:] for r in _DUCK.execute("select g, min(big), max(big) from t group by g").fetchall()}
    got = LazyFrame(T).groupby("g").agg(col("big").min().alias("lo"), col("big").max().alias("hi")).compute(device="cpu")
    mine = {r["g"]: (r["lo"], r["hi"]) for r in got.to_pylist()}
    assert mine == want
    assert mine[5] == (I64_MAX, I64_MAX) and mine[6] == (I64_MIN, I64_MIN)
    assert any(v != int(float(v)) for lo_hi in mine.values() for v in lo_hi)       # float64 would have rounded one of them
    assert LazyFrame(T).groupby().agg(col("big").min().alias("lo"), col("big").max().alias("hi")) \
        .compute(device="cpu").to_pylist() == [{"lo": I64_MIN, "hi": I64_MAX}]


# -- with a filter, other aggregates of the same query, NULLs, empty input -----------------------------------

@pytest.mark.parametrize("native,device", MODES)
def test_min_max_next_to_sums_counts_and_a_filter(run, native, device):
    def build(f):
        return (f.filter(col("small") > lit(-500)).groupby("g")
                .agg(col("big").min().alias("lo"), col("big").max().alias("hi"), col("small").sum().alias("s"),
                     col("i32").max().alias("m32"), col("small").count().alias("n")).sort("g"))
    got, paths = run(build, native, device)
    q = ("select g, min(big) as lo, max(big) as hi, sum(small)::bigint as s, max(i32) as m32, count(small) as n "
         "from t where small > -500 group by g order by g")
    want = _duck_arrow(q)
    assert got.schema == want.schema and got.to_pylist() == want.to_pylist()
    assert set(paths) == {EXPECT[(native, device)]}, paths


@pytest.mark.parametrize("native,device", MODES)
def test_a_column_read_for_its_values_too_stays_exact(run, native, device):
    """`big` is also summed (a float64 read of the same column): the extreme must not take the float slot."""
    small_big = T.set_column(T.schema.get_field_index("big"), "big", A(np.arange(N, dtype=np.int64) * 3 - 1000))
    big_t = pa.table({"big": T.column("big"), "g": T.column("g")})
    for table in (small_big, big_t):
        con = duckdb.connect()
        con.register("t", table)
        got, _ = run(lambda f: f.groupby("g").agg(col("big").min().alias("lo"), col("big").max().alias("hi"),
                                                 (col("big") + lit(0)).count().alias("n")).sort("g"),
                     native, device, table=table)
        want = con.execute("select g, min(big), max(big), count(big) from t group by g order by g").fetchall()
        assert [tuple(r.values()) for r in got.to_pylist()] == want


def test_nulls_are_skipped_and_stay_exact():
    q = "select g, min(nul) as lo, max(nul) as hi from t group by g order by g"
    got = LazyFrame(T).groupby("g").agg(col("nul").min().alias("lo"), col("nul").max().alias("hi")).sort("g") \
        .compute(device="cpu")
    want = _duck_arrow(q)
    assert got.schema == want.schema and got.to_pylist() == want.to_pylist()


@pytest.mark.parametrize("native,device", MODES)
def test_no_surviving_rows_give_null_of_the_columns_type_or_no_groups(run, native, device):
    got, _ = run(lambda f: f.filter(col("small") > lit(5000)).groupby().agg(col("big").min().alias("lo"),
                                                                         col("i32").max().alias("hi")), native, device)
    want = _duck_arrow("select min(big) as lo, max(i32) as hi from t where small > 5000")
    assert got.schema == want.schema and got.to_pylist() == [{"lo": None, "hi": None}]
    got, _ = run(lambda f: f.filter(col("small") > lit(5000)).groupby("g").agg(col("big").min().alias("lo")),
                 native, device)
    assert got.num_rows == 0 and got.schema.field("lo").type == pa.int64()


# -- every kernel: direct, few, thousands, many groups, reproducible, hashed -----------------------------------

SHAPES = {"global": (1_500_000, 1), "few": (1_500_000, 7), "thousands": (1_500_000, 1000),
          "many": (1_200_000, 1_000_000), "staged": (400_000, 200_000)}
KMODES = [
    pytest.param("1", "cpu", False, id="native"),
    pytest.param("1", "cpu", True, id="native-repro"),
    pytest.param("0", "cpu", False, id="pipeline"),
    pytest.param("0", "cpu", True, id="pipeline-repro"),
    pytest.param("0", "gpu", False, id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
    pytest.param("0", "gpu", True, id="gpu-repro", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
]


def _shape_table(rows, groups, hashed=False, seed=61):
    r = np.random.default_rng(seed)
    v = r.integers(1 << 54, 1 << 62, rows).astype(np.int64) * np.where(r.random(rows) < 0.5, -1, 1)
    v[:2] = [I64_MIN, I64_MAX]
    g = (r.permutation(rows) % groups).astype(np.int64)
    cols = {"v": A(v), "w": A(r.integers(-2 ** 31, 2 ** 31, rows).astype(np.int32)), "g": A(g)}
    if hashed:
        cols["g"] = A(g.astype(np.float64) * 0.5 + 0.25)         # float keys: hashed groups
    return pa.table(cols)


@pytest.mark.parametrize("hashed", [False, True], ids=["direct", "hashed"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("native,device,repro", KMODES)
def test_every_kernel_takes_the_extreme_in_int64(monkeypatch, native, device, repro, shape, hashed):
    rows, groups = SHAPES[shape]
    if hashed and (groups == 1 or device == "gpu"):
        pytest.skip("hashed groups: the executor's float keys")
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _shape_table(rows, groups, hashed)
    con = duckdb.connect()
    con.register("t", t)
    keyed = groups > 1
    lf = LazyFrame(t).groupby("g") if keyed else LazyFrame(t).groupby()
    lf = lf.agg(col("v").min().alias("lo"), col("v").max().alias("hi"), col("w").min().alias("wlo"))
    if keyed:
        lf = lf.sort("g")
    got = lf.compute(device=device, reproducible=repro)
    q = "select {} min(v) as lo, max(v) as hi, min(w) as wlo from t {}".format(
        "g," if keyed else "", "group by g order by g" if keyed else "")
    want = con.execute(q).arrow()
    want = want.read_all() if hasattr(want, "read_all") else want
    assert got.schema == want.schema
    assert got.to_pylist() == want.to_pylist()
    assert max(abs(x) for x in got.column("hi").to_pylist() + got.column("lo").to_pylist()) > 2 ** 53
    prov = lf.last_compile_provenance["path"]
    if native == "1" and device == "cpu":
        assert prov == "native", prov
    elif device == "gpu" and (not repro or shape in ("global", "few", "thousands")):
        assert prov == "fused_pipeline_gpu", prov
    elif device == "cpu":
        assert prov == "fused_pipeline_cpu", prov


# -- the SQL front end, ORDER BY / HAVING over an extreme, Parquet, streamed -------------------------------------

@pytest.mark.parametrize("q", [
    "select min(big) as lo, max(big) as hi from t",
    "select g, max(big) as hi, min(i16) as lo from t group by g order by g",
    "select g, max(big) as hi from t group by g having max(big) > 0 order by g",
    "select g, max(big) as hi from t group by g order by hi desc, g limit 3",
    "select g, min(big) as lo from t group by g order by lo, g",
    "select min(u32) as lo, max(u8) as hi, min(i16) as a, max(i32) as b from t",
])
@pytest.mark.parametrize("native,device", MODES)
def test_sql_extremes_equal_duckdb_values_and_types(monkeypatch, native, device, q):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    got = mx.sql(q, t=T).compute(device=device)
    want = _duck_arrow(q)
    assert got.schema == want.schema
    assert got.to_pylist() == want.to_pylist()


def test_a_parquet_scan_with_many_row_groups(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(T, path, row_group_size=1000)
    for grouped in (False, True):
        lf = scan_parquet(path)
        lf = lf.groupby("g") if grouped else lf.groupby()
        lf = lf.agg(col("big").min().alias("lo"), col("big").max().alias("hi"), col("i16").min().alias("a"))
        got = lf.compute(device="cpu")
        con = duckdb.connect()
        want = con.execute(f"select {'g,' if grouped else ''} min(big), max(big), min(i16) from read_parquet('{path}')"
                           f"{' group by g' if grouped else ''}").fetchall()
        key = (lambda r: r[0]) if grouped else (lambda r: 0)
        assert sorted((tuple(r.values()) for r in got.to_pylist()), key=key) == sorted(want, key=key)
        assert [str(f.type) for f in got.schema][-3:] == ["int64", "int64", "int16"]


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
def test_a_streamed_extreme_equals_the_in_memory_one(tmp_path, monkeypatch, grouped):
    path = str(tmp_path / "t.parquet")
    pq.write_table(T, path, row_group_size=100)
    out = {}
    for budget in ("5K", "300K", "64G"):
        monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", budget)
        lf = scan_parquet(path)
        lf = (lf.groupby("g") if grouped else lf.groupby()).agg(col("big").min().alias("lo"),
                                                                col("big").max().alias("hi"),
                                                                col("i32").max().alias("m"))
        got = lf.compute(device="cpu")
        out[budget] = (got.schema, sorted(got.to_pylist(), key=lambda r: r.get("g", 0)),
                       lf.last_compile_provenance.get("streamed", {}).get("batches"))
    assert out["64G"][2] is None and out["5K"][2] > out["300K"][2] > 1
    assert out["5K"][:2] == out["300K"][:2] == out["64G"][:2]
    want = _DUCK.execute(f"select {'g,' if grouped else ''} min(big), max(big), max(i32) from t"
                         f"{' group by g' if grouped else ''}").fetchall()
    key = (lambda r: r[0]) if grouped else (lambda r: 0)
    assert [tuple(r.values()) for r in out["64G"][1]] == sorted(want, key=key)
    assert [str(f.type) for f in out["5K"][0]][-3:] == ["int64", "int64", "int32"]


# -- the F-090 route is gone from the early path: no pass over the column --------------------------------------

def test_the_early_path_no_longer_reads_the_column_to_find_values_past_2_53(monkeypatch):
    """F-090 ran `pc.min_max` over an integer column for every min / max query before any kernel ran."""
    import pyarrow.compute as real_pc
    calls = []
    real = real_pc.min_max
    monkeypatch.setattr(real_pc, "min_max", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    LazyFrame(T).groupby("g").agg(col("big").max().alias("hi")).compute(device="cpu")
    assert calls == []


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")
@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
def test_a_cpu_gpu_split_merges_the_int64_extremes(fraction, grouped):
    import omxcore.coproc_fused as cf
    cf.clear_residency()
    t = _shape_table(600_000, 37)

    def query():
        lf = LazyFrame(t).groupby("g") if grouped else LazyFrame(t).groupby()
        return lf.agg(col("v").min().alias("lo"), col("v").max().alias("hi"), col("w").max().alias("m"))
    con = duckdb.connect()
    con.register("t", t)
    out = con.execute(f"select {'g,' if grouped else ''} min(v) as lo, max(v) as hi, max(w) as m from t"
                      f"{' group by g' if grouped else ''}").arrow()
    want = out.read_all() if hasattr(out, "read_all") else out
    with cf.policy(min_upload_bytes=0, gpu_fraction=fraction):
        query().compute(device="cpu")
        cf.wait_for_uploads(30)
        lf = query()
        got = lf.compute(device="cpu")
    key = (lambda r: r["g"]) if grouped else (lambda r: 0)
    assert got.schema == want.schema
    assert sorted(got.to_pylist(), key=key) == sorted(want.to_pylist(), key=key)
    if 0.0 < fraction < 1.0:
        assert lf.last_compile_provenance["path"] == "fused_pipeline_split", lf.last_compile_provenance["path"]


# -- over a join: the fused join keeps float slots and declines past 2^53; the joined table is aggregated exactly ---

@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))])
@pytest.mark.parametrize("q", [
    "select grp, max(big) as hi, min(s) as lo from f join d on f.k = d.k group by grp order by grp",
    "select max(big) as hi, min(s) as lo from f join d on f.k = d.k where grp = 1",
    "select g, min(big) as lo from f join d on f.k = d.k group by g order by g",
])
def test_an_extreme_over_a_join_is_exact_and_typed(device, q):
    n = 50_000
    r = np.random.default_rng(3)
    f = pa.table({"k": A(r.integers(0, 100, n)), "big": A((1 << 60) + r.integers(-1000, 1000, n)),
                  "s": A(r.integers(-1000, 1000, n).astype(np.int32)), "g": A(r.integers(0, 4, n))})
    d = pa.table({"k": A(np.arange(100)), "grp": A(np.arange(100) % 5)})
    con = duckdb.connect()
    con.register("f", f)
    con.register("d", d)
    out = con.execute(q).arrow()
    want = out.read_all() if hasattr(out, "read_all") else out
    got = mx.sql(q, f=f, d=d).compute(device=device)
    assert got.schema == want.schema and got.to_pylist() == want.to_pylist()
