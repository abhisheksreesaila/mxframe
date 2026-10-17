"""F-060: an integer SUM is int64 on every path, whatever the data.

DuckDB sums every integer type into a HUGEINT; ours is int64 and a total past int64 raises
(`OverflowError: Overflow in sum of INT64`, F-061). So the result type depends on the input type
only: never on the values (the old rule answered a double when an F-037 bound proved the sum fit in
2^53), never on nulls, the device, the native executor, or how a streamed query was batched.
Every case runs on cpu / gpu / auto, native on / off, grouped and global, and equals DuckDB's value
exactly (past 2^53 as well). `avg` of an integer stays a double, as in DuckDB."""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import LazyFrame, col, gpu_stream, lit
from mxframe.io_parquet import scan_parquet
from mxframe.np_arrow import array as A

HAS_GPU = gpu_stream.cuda_available()
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
           pytest.param("auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))]

B = (1 << 53) + 1                      # the first integer float64 cannot hold
N = 400                                # 400 rows of ~2^53 sum to 3.6e18: inside int64


@pytest.fixture(params=["1", "0"], ids=["native", "legacy"])
def mode(request, monkeypatch):
    if request.param == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", request.param)
    return request.param


def _table():
    k = np.arange(N)
    r = np.random.default_rng(60)
    return pa.table({
        "big": A((B + k).astype(np.int64)),
        "neg": A((-(B + 3 * k)).astype(np.int64)),
        "small": A(r.integers(-1000, 1000, N).astype(np.int64)),
        "i32": A(r.integers(-2 ** 31, 2 ** 31, N).astype(np.int32)),
        "i16": A(r.integers(-2 ** 15, 2 ** 15, N).astype(np.int16)),
        "i8": A(r.integers(-128, 128, N).astype(np.int8)),
        "u8": A(r.integers(0, 256, N).astype(np.uint8)),
        "u32": A(r.integers(0, 2 ** 32, N).astype(np.uint32)),
        "nul": pa.array([None if x % 7 == 0 else B + x for x in k], pa.int64()),
        "nul32": pa.array([None if x % 5 == 0 else int(x) for x in k], pa.int32()),
        "g": A((k % 4).astype(np.int64)),
        "f": A(r.random(N)),
    })


T = _table()
SUMMED = ["big", "neg", "small", "i32", "i16", "i8", "u8", "u32", "nul", "nul32"]


def _duck(q, t=T):
    con = duckdb.connect()
    con.register("t", t)
    return con.execute(q).fetchall()


def _ints(rows):
    return [None if v is None else int(v) for v in rows]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("c", SUMMED)
def test_a_global_integer_sum_is_int64_and_equals_duckdb(mode, device, c):
    got = LazyFrame(T).groupby().agg(col(c).sum().alias("s")).compute(device=device)
    assert got.schema.field("s").type == pa.int64()
    assert got.column("s").to_pylist() == _ints(r[0] for r in _duck(f"select sum({c}) from t"))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("c", SUMMED)
def test_a_grouped_integer_sum_is_int64_and_equals_duckdb(mode, device, c):
    got = LazyFrame(T).groupby("g").agg(col(c).sum().alias("s")).sort("g").compute(device=device)
    assert got.schema.field("s").type == pa.int64()
    want = _duck(f"select g, sum({c}) from t group by g order by g")
    assert got.column("g").to_pylist() == [r[0] for r in want]
    assert got.column("s").to_pylist() == _ints(r[1] for r in want)


def test_the_type_does_not_depend_on_the_values(mode):
    """Same column type, tiny values or past 2^53: the same result type (it used to be a double for the
    first and int64 for the second)."""
    tiny = pa.table({"x": A(np.arange(10, dtype=np.int64)), "g": A(np.zeros(10, dtype=np.int64))})
    huge = pa.table({"x": A(np.full(10, B, dtype=np.int64)), "g": A(np.zeros(10, dtype=np.int64))})
    for t, want in ((tiny, 45), (huge, 10 * B)):
        for lf in (LazyFrame(t).groupby().agg(col("x").sum().alias("s")),
                   LazyFrame(t).groupby("g").agg(col("x").sum().alias("s"))):
            got = lf.compute(device="cpu")
            assert got.schema.field("s").type == pa.int64()
            assert got.column("s").to_pylist() == [want]


@pytest.mark.parametrize("device", DEVICES)
def test_sums_of_expressions_and_filters_are_int64_and_exact(mode, device):
    lf = (LazyFrame(T).filter(col("small") > lit(-500)).groupby("g")
          .agg((col("big") - lit(1 << 53)).sum().alias("a"), (col("i32") * col("i16")).sum().alias("b"),
               (col("big") + col("neg")).sum().alias("c"), (col("small") * lit(3)).sum().alias("d"))
          .sort("g").compute(device=device))
    for n in "abcd":
        assert lf.schema.field(n).type == pa.int64()
    want = _duck("select g, sum(big - 9007199254740992), sum(i32::bigint * i16), sum(big + neg), sum(small * 3) "
                 "from t where small > -500 group by g order by g")
    for i, n in enumerate("abcd"):
        assert lf.column(n).to_pylist() == _ints(r[i + 1] for r in want)


@pytest.mark.parametrize("device", DEVICES)
def test_a_mix_of_sums_means_and_counts_in_one_query(mode, device):
    got = (LazyFrame(T).groupby("g")
           .agg(col("big").sum().alias("s"), col("big").mean().alias("m"), col("i8").sum().alias("t"),
                col("f").sum().alias("fs"), col("big").count().alias("n"), col("big").max().alias("mx"))
           .sort("g").compute(device=device))
    assert [str(got.schema.field(n).type) for n in ("s", "m", "t", "fs", "n", "mx")][:2] == ["int64", "double"]
    assert got.schema.field("t").type == pa.int64() and got.schema.field("fs").type == pa.float64()
    want = _duck("select g, sum(big), avg(big), sum(i8), count(big) from t group by g order by g")
    assert got.column("s").to_pylist() == _ints(r[1] for r in want)
    assert got.column("m").to_pylist() == pytest.approx([float(r[2]) for r in want], rel=1e-15)
    assert got.column("t").to_pylist() == _ints(r[3] for r in want)
    assert got.column("n").to_pylist() == [r[4] for r in want]


def test_an_integer_mean_past_2_53_is_accumulated_in_integers(mode):
    """avg of 2^53 + 1, 2^53 + 3 ... is exact before the division: float64 accumulation drifts."""
    x = np.array([B, B + 2, B + 4, B + 6] * 100, dtype=np.int64)
    t = pa.table({"x": A(x), "g": A(np.zeros(len(x), dtype=np.int64))})
    for lf in (LazyFrame(t).groupby().agg(col("x").mean().alias("m")),
               LazyFrame(t).groupby("g").agg(col("x").mean().alias("m"))):
        got = lf.compute(device="cpu")
        assert got.schema.field("m").type == pa.float64()
        assert got.column("m").to_pylist() == [float(_duck("select avg(x) from t", t)[0][0])]


def test_a_group_with_only_nulls_sums_to_null_and_the_type_stays_int64(mode):
    t = pa.table({"x": pa.array([None, None, 5, 7], pa.int64()), "g": A(np.array([0, 0, 1, 1], dtype=np.int64))})
    got = LazyFrame(t).groupby("g").agg(col("x").sum().alias("s")).sort("g").compute(device="cpu")
    assert got.schema.field("s").type == pa.int64()
    assert got.column("s").to_pylist() == [None, 12]
    got = LazyFrame(t.slice(0, 2)).groupby().agg(col("x").sum().alias("s")).compute(device="cpu")
    assert got.schema.field("s").type == pa.int64() and got.column("s").to_pylist() == [None]


def test_an_empty_input_sums_to_null_with_the_int64_type(mode):
    t = pa.table({"x": A(np.arange(5, dtype=np.int64)), "g": A(np.zeros(5, dtype=np.int64))})
    got = LazyFrame(t).filter(col("x") > lit(99)).groupby().agg(col("x").sum().alias("s")).compute(device="cpu")
    assert got.schema.field("s").type == pa.int64() and got.column("s").to_pylist() == [None]


# -- overflow --------------------------------------------------------------------------------------------

def _no_overflow_until_the_merge(rows, v):
    return pa.table({"a": A(np.full(rows, v, dtype=np.int64)), "g": A((np.arange(rows) % 3).astype(np.int64))})


@pytest.mark.parametrize("device", DEVICES)
def test_a_total_past_int64_raises_even_when_no_worker_partial_does(mode, device):
    """300,000 rows of 2^45 (no value near 2^53): the total is 2^63.2. Each of the workers' partials stays
    under 2^63, so only the merge sees it. DuckDB's HUGEINT holds it; ours is int64 and says so."""
    t = _no_overflow_until_the_merge(300_000, 1 << 45)
    assert _duck("select sum(a) from t", t)[0][0] > (1 << 63) - 1
    with pytest.raises(OverflowError, match="sum of INT64"):
        LazyFrame(t).groupby().agg(col("a").sum().alias("s")).compute(device=device)
    one = _no_overflow_until_the_merge(300_000 * 3, 1 << 45)             # every group still past it
    with pytest.raises(OverflowError, match="sum of INT64"):
        LazyFrame(one).groupby("g").agg(col("a").sum().alias("s")).compute(device=device)


@pytest.mark.parametrize("device", DEVICES)
def test_the_largest_total_that_fits_is_exact(mode, device):
    rows = 262_143                                               # 262,143 * 2^45 = 2^63 - 2^45
    t = _no_overflow_until_the_merge(rows, 1 << 45)
    want = _duck("select sum(a) from t", t)[0][0]
    assert want == rows * (1 << 45) < (1 << 63)
    got = LazyFrame(t).groupby().agg(col("a").sum().alias("s")).compute(device=device)
    assert got.schema.field("s").type == pa.int64() and got.column("s").to_pylist() == [want]
    got = LazyFrame(t).groupby("g").agg(col("a").sum().alias("s")).sort("g").compute(device=device)
    assert got.column("s").to_pylist() == _ints(r[0] for r in _duck("select sum(a) from t group by g order by g", t))


@pytest.mark.parametrize("device", DEVICES)
def test_partial_sums_that_pass_int64_and_cancel_are_not_an_overflow(mode, device):
    """+2^52 x 2500 then -2^52 x 2500: a running total passes 2^63 in the middle, the total is 0."""
    a = np.concatenate([np.full(2500, 1 << 52), np.full(2500, -(1 << 52)), [7]]).astype(np.int64)
    t = pa.table({"a": A(a), "g": A(np.zeros(len(a), dtype=np.int64))})
    for lf in (LazyFrame(t).groupby().agg(col("a").sum().alias("s")),
               LazyFrame(t).groupby("g").agg(col("a").sum().alias("s"))):
        got = lf.compute(device=device)
        assert got.schema.field("s").type == pa.int64() and got.column("s").to_pylist() == [7]


@pytest.mark.parametrize("device", DEVICES)
def test_the_overflow_of_one_group_raises(mode, device):
    rows = 5000
    v = np.where(np.arange(rows) % 2 == 0, 1 << 52, 1).astype(np.int64)        # group 0: 2500 x 2^52 = 2^63.3
    t = pa.table({"a": A(v), "g": A((np.arange(rows) % 2).astype(np.int64))})
    with pytest.raises(OverflowError, match="sum of INT64"):
        LazyFrame(t).groupby("g").agg(col("a").sum().alias("s")).compute(device=device)


# -- the SQL front end -------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "select sum(i32) as s from t", "select sum(big) as s from t", "select g, sum(i8) as s from t group by g order by g",
    "select g, sum(big - 9007199254740992) as s from t where small > 0 group by g order by g",
    "select sum(nul) as s from t", "select sum(i16) + 1 as s from t",
    "select sum(i32) / 2 as s from t",
])
def test_sql_sums_equal_duckdb_with_the_same_value(mode, q):
    got = mx.sql(q, t=T).compute()
    want = _duck(q)
    assert [tuple(r.values()) for r in got.to_pylist()] == [tuple(None if v is None else int(v) if not isinstance(v, float) else v for v in r) for r in want]
    if "/ 2" not in q:
        assert got.schema.field("s").type == pa.int64()


# -- streamed --------------------------------------------------------------------------------------------

def _file(tmp_path, t, rows_per_group):
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=rows_per_group)
    return path


@pytest.fixture
def _streaming(monkeypatch):
    def at(budget):
        monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", budget)
    return at


@pytest.mark.parametrize("c", ["big", "small", "i32", "i8", "nul"])
@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
def test_a_streamed_integer_sum_is_int64_and_equals_the_in_memory_one(tmp_path, _streaming, c, grouped):
    n = 60_000
    k = np.arange(n)
    r = np.random.default_rng(6)
    base = {"big": (1 << 40) + k, "small": r.integers(0, 1000, n), "i32": r.integers(0, 1 << 30, n).astype(np.int32),
            "i8": r.integers(-128, 128, n).astype(np.int8), "g": (k % 5).astype(np.int32)}
    base["nul"] = pa.array([None if x % 11 == 0 else (1 << 40) + int(x) for x in k], pa.int64())
    path = _file(tmp_path, pa.table(base), 500)

    def q(s):
        s = s.groupby("g") if grouped else s.groupby()
        return s.agg(col(c).sum().alias("s"))
    out = {}
    for budget in ("20K", "200K", "64G"):
        _streaming(budget)
        lf = q(scan_parquet(path))
        got = lf.compute(device="cpu")
        out[budget] = (got.schema.field("s").type, sorted(got.to_pylist(), key=lambda r: r.get("g", 0)),
                       lf.last_compile_provenance.get("streamed", {}).get("batches"))
    assert out["64G"][2] is None and out["20K"][2] > out["200K"][2] > 1
    assert {v[0] for v in out.values()} == {pa.int64()}
    assert out["20K"][1] == out["200K"][1] == out["64G"][1]
    con = duckdb.connect()
    want = con.execute(f"select {'g, ' if grouped else ''}sum({c}) from read_parquet('{path}')"
                       f"{' group by g' if grouped else ''}").fetchall()
    key = (lambda r: r[0]) if grouped else (lambda r: 0)
    assert [tuple(r.values()) for r in out["64G"][1]] == [tuple(int(v) for v in r) for r in sorted(want, key=key)]
