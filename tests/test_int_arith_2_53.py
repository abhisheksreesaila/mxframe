"""F-037: integer arithmetic whose intermediate result passes 2^53 is never computed in the
float64 interpreter. The compiler bounds every int `*` `+` `-` from the columns' real min / max
and declines (the caller's exact path answers); results equal DuckDB for every entry point that
evaluates expressions: mx_execute (group-by and global aggregates), mx_filter (plain filters and
star joins), mx_filter_topk, and the legacy fused pipeline."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A
from mxframe.pipeline_program import compile_pipeline

P52 = 1 << 52
N = 3000

# one predicate per operator; each is true for the "big" rows only when computed exactly
PREDS = {
    "mul": (lambda: (col("a") * col("b")) - lit(2 ** 53) == lit(261134297),
            "(a * b) - 9007199254740992 = 261134297"),
    "add": (lambda: (col("x") + col("y")) - lit(2 ** 53) == lit(7),
            "(x + y) - 9007199254740992 = 7"),
    "sub": (lambda: (col("u") - col("z")) - lit(2 ** 53) == lit(P52 + 3),
            "(u - z) - 9007199254740992 = %d" % (P52 + 3)),
}


def _table():
    r = np.random.default_rng(3)
    big = np.arange(N) % 3 != 0                       # two thirds of the rows are the big ones
    a = np.where(big, 94906267, r.integers(1, 1000, N)).astype(np.int64)
    # add: x + y = 2^53 + 7 (odd, so float64 rounds it); sub: u - z = 2^53 + 2^52 + 3
    x = np.where(big, 1 << 52, r.integers(1, 1000, N)).astype(np.int64)
    y = np.where(big, P52 + 7, r.integers(1, 1000, N)).astype(np.int64)
    u = np.where(big, 1 << 53, r.integers(1, 1000, N)).astype(np.int64)
    z = np.where(big, -(P52 + 3), r.integers(1, 1000, N)).astype(np.int64)
    return pa.table({
        "row": A(np.arange(N, dtype=np.int64)), "g": A((np.arange(N) % 4).astype(np.int64)),
        "a": A(a), "b": A(a.copy()), "x": A(x), "y": A(y), "u": A(u), "z": A(z), "v": A(np.ones(N)),
    })


def _con(**tables):
    con = duckdb.connect()
    for k, v in tables.items():
        con.register(k, v)
    return con


@pytest.fixture(params=["1", "0"], ids=["native", "legacy"])
def mode(request, monkeypatch):
    if request.param == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", request.param)
    return request.param


def _want_rows(t, sql):
    return [r[0] for r in _con(t=t).execute(f"select row from t where {sql} order by row").fetchall()]


@pytest.mark.parametrize("name", PREDS)
def test_the_compiler_declines_an_intermediate_past_2_53(name):
    t = _table()
    assert compile_pipeline([PREDS[name][0]()], [col("v").sum()], t) is None
    assert compile_pipeline([], [(col("a") * col("b")).max()], t) is None


def test_small_integers_still_compile_even_when_the_type_could_overflow():
    t = pa.table({"p": A(np.arange(1, 100, dtype=np.int64)), "q": A(np.arange(1, 100, dtype=np.int64)),
                  "i": A(np.arange(1, 100, dtype=np.int32)), "v": A(np.ones(99))})
    assert compile_pipeline([(col("p") * col("q")) > lit(50)], [col("v").sum()], t) is not None
    assert compile_pipeline([(col("i") * col("i") + lit(1)) > lit(50)], [col("v").sum()], t) is not None
    assert compile_pipeline([], [(col("p") * lit(2) - col("q")).max()], t) is not None
    # a product that stays exactly 2^53 is fine; a bigger one is not
    big = pa.table({"p": A(np.array([1 << 26], dtype=np.int64)), "v": A(np.ones(1))})
    assert compile_pipeline([(col("p") * col("p") * lit(2)) == lit(1 << 53)], [col("v").sum()], big) is not None
    assert compile_pipeline([(col("p") * col("p") * lit(3)) > lit(1)], [col("v").sum()], big) is None


def test_constant_arithmetic_past_2_53_declines():
    t = _table()
    assert compile_pipeline([col("a") == lit(94906267) * lit(94906267) - lit(2 ** 53)], [col("v").sum()], t) is None
    assert compile_pipeline([col("a") + lit(94906267) * lit(94906267) > lit(1)], [col("v").sum()], t) is None


@pytest.mark.parametrize("name", PREDS)
def test_filter_matches_duckdb(mode, name):
    t = _table()
    pred, sql = PREDS[name]
    want = _want_rows(t, sql)
    assert len(want) == 2000                                    # the big rows, every one
    got = LazyFrame(t).filter(pred()).compute(device="cpu")
    assert sorted(got.column("row").to_pylist()) == want


@pytest.mark.parametrize("name", PREDS)
def test_global_and_grouped_aggregate_match_duckdb(mode, name):
    t = _table()
    pred, sql = PREDS[name]
    con = _con(t=t)
    want_n = con.execute(f"select sum(v) from t where {sql}").fetchone()[0]
    assert want_n == 2000.0
    got = LazyFrame(t).filter(pred()).groupby().agg(col("v").sum().alias("s")).compute(device="cpu")
    assert got.column("s").to_pylist() == [want_n]
    want_g = dict(con.execute(f"select g, sum(v) from t where {sql} group by g").fetchall())
    got = LazyFrame(t).filter(pred()).groupby("g").agg(col("v").sum().alias("s")).sort("g").compute(device="cpu")
    assert dict(zip(got.column("g").to_pylist(), got.column("s").to_pylist())) == want_g


def test_arithmetic_inside_an_aggregate_matches_duckdb(mode):
    t = _table()
    want = _con(t=t).execute("select min(a * b - 9007199254740992), max(a * b - 9007199254740992) from t").fetchone()
    got = LazyFrame(t).groupby().agg(((col("a") * col("b")) - lit(2 ** 53)).min().alias("lo"),
                                     ((col("a") * col("b")) - lit(2 ** 53)).max().alias("hi")).compute(device="cpu")
    assert (got.column("lo").to_pylist()[0], got.column("hi").to_pylist()[0]) == tuple(float(w) for w in want)


@pytest.mark.parametrize("name", PREDS)
def test_filter_topk_matches_duckdb(mode, name):
    t = _table()
    pred, sql = PREDS[name]
    want = [r[0] for r in _con(t=t).execute(
        f"select row from t where {sql} order by row desc limit 7").fetchall()]
    got = LazyFrame(t).filter(pred()).sort("row", descending=True).limit(7).compute(device="cpu")
    assert got.column("row").to_pylist() == want


@pytest.mark.parametrize("name", PREDS)
def test_star_join_filter_matches_duckdb(mode, name):
    t = _table()
    dim = pa.table({"g": A(np.arange(4, dtype=np.int64)), "w": A(np.arange(4, dtype=np.int64) * 10)})
    pred, sql = PREDS[name]
    want = [r[0] for r in _con(t=t, d=dim).execute(
        f"select row from t join d on t.g = d.g where {sql} order by row").fetchall()]
    got = LazyFrame(t).filter(pred()).join(LazyFrame(dim), on="g").compute(device="cpu")
    assert sorted(got.column("row").to_pylist()) == want
