"""F-037, F-060b: integer arithmetic whose intermediate result passes 2^53 is never computed in the float64
interpreter. F-037 bounded every int `*` `+` `-` from the columns' real min / max and declined; F-060b computes it in
int64 registers (OP_IADD / ISUB / IMUL) when the column TYPES cannot keep it within 2^53, with a runtime flag instead of a
pass over the data. Results equal DuckDB for every entry point that evaluates expressions: mx_execute (group-by and global
aggregates), mx_filter (plain filters and star joins), mx_filter_topk, and the legacy fused pipeline. The compiler without
int64 registers (`int_regs=False`: a fused join's gathered columns, the late-stage question) still declines."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit, when
from mxframe.np_arrow import array as A
from mxframe.pipeline_program import compile_pipeline

P52 = 1 << 52
N = 3000

# one predicate per operator; each is true for the "big" rows only when computed exactly
PREDS = {
    # bounds carried through case_when, year(), a negative factor and nested arithmetic
    "case_when": (lambda: (when(col("g") >= lit(0), col("a"), col("b")) * col("a")) - lit(2 ** 53) == lit(261134297),
                  "(case when t.g >= 0 then a else b end * a) - 9007199254740992 = 261134297"),
    "mixed_sign": (lambda: (col("neg") * col("a")) + lit(2 ** 53) == lit(2 ** 53 - 94906267 * 94906267),
                   "(neg * a) + 9007199254740992 = 9007199254740992 - 94906267::bigint * 94906267"),
    "nested": (lambda: (col("c") * col("c") + col("c") * col("c")) - lit(2 ** 53) == lit(2 * 70_000_000 ** 2 - 2 ** 53),
               "(c * c + c * c) - 9007199254740992 = %d" % (2 * 70_000_000 ** 2 - 2 ** 53)),
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
    c = np.where(big, 70_000_000, r.integers(1, 1000, N)).astype(np.int64)
    neg = np.where(big, -94906267, r.integers(1, 1000, N)).astype(np.int64)
    d = np.where(big, 949_062_673_333, r.integers(10_000, 90_000, N)).astype(np.int64)
    return pa.table({
        "c": A(c), "neg": A(neg), "d": A(d),
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
def test_the_compiler_computes_an_intermediate_past_2_53_in_int64(name):
    t = _table()
    prog = compile_pipeline([PREDS[name][0]()], [col("v").sum()], t)
    assert any(op in ("IADD", "ISUB", "IMUL") for op in prog.op_names()), prog.op_names()
    assert compile_pipeline([PREDS[name][0]()], [col("v").sum()], t, int_regs=False) is None      # types alone cannot prove it
    prog = compile_pipeline([], [(col("a") * col("b")).max()], t)
    assert "IMUL" in prog.op_names() and prog.slot_kinds == ["imax"]
    assert compile_pipeline([], [(col("a") * col("b")).max()], t, int_regs=False) is None


def test_small_integers_compile_to_int64_ops_when_the_type_could_overflow(monkeypatch):
    t = pa.table({"p": A(np.arange(1, 100, dtype=np.int64)), "q": A(np.arange(1, 100, dtype=np.int64)),
                  "i": A(np.arange(1, 100, dtype=np.int32)), "v": A(np.ones(99))})
    assert "IMUL" in compile_pipeline([(col("p") * col("q")) > lit(50)], [col("v").sum()], t).op_names()
    assert "IMUL" in compile_pipeline([(col("i") * col("i") + lit(1)) > lit(50)], [col("v").sum()], t).op_names()
    assert "ISUB" in compile_pipeline([], [(col("p") * lit(2) - col("q")).max()], t).op_names()
    # the data does not decide any more: a product of 2^26 squared is computed in int64 whatever the values
    big = pa.table({"p": A(np.array([1 << 26], dtype=np.int64)), "v": A(np.ones(1))})
    assert compile_pipeline([(col("p") * col("p") * lit(2)) == lit(1 << 53)], [col("v").sum()], big) is not None
    assert compile_pipeline([(col("p") * col("p") * lit(3)) > lit(1)], [col("v").sum()], big) is not None
    # and the query runs in the native executor, equal to DuckDB
    if nx.available():
        monkeypatch.setenv("MXFRAME_NATIVE", "1")
        got = LazyFrame(t).filter((col("p") * col("q")) > lit(50)).groupby().agg(
            (col("p") * lit(2) - col("q")).max().alias("m"), col("v").sum().alias("s")).compute(device="cpu")
        assert nx.last_path() == "native"
        want = _con(t=t).execute("select max(p * 2 - q), sum(v) from t where p * q > 50").fetchone()
        assert (got.column("m").to_pylist()[0], float(got.column("s").to_pylist()[0])) == tuple(float(x) for x in want)


def test_the_types_decide_and_no_query_reads_column_data(monkeypatch):
    """int32 + int32, year(), int32 * a small literal and case_when stay float64 ops; int32 * int32 and int64 * int64
    are int64 ops. None of them reads the data (F-037 read it for the last two)."""
    import mxframe.join_engine as je
    reads = []
    real = je._minmax
    monkeypatch.setattr(je, "_minmax", lambda a: reads.append(len(a)) or real(a))
    n = 5000
    t = pa.table({"i": A(np.arange(n, dtype=np.int32)), "j": A(np.arange(n, dtype=np.int32)),
                  "d": A(np.arange(n, dtype=np.int32)), "v": A(np.ones(n))})
    for pred in (col("i") + col("j") > lit(5), col("d").year() + col("i") > lit(5), col("i") * lit(1000) > lit(5),
                 when(col("i") > lit(3), col("i"), col("j")) + col("j") > lit(5)):
        names = compile_pipeline([pred], [col("v").sum()], t).op_names()
        assert not {"IADD", "ISUB", "IMUL", "I2F"} & set(names), names
        assert compile_pipeline([pred], [col("v").sum()], t, int_regs=False) is not None
    assert "IMUL" in compile_pipeline([col("i") * col("j") > lit(5)], [col("v").sum()], t).op_names()
    t64 = pa.table({"p": A(np.arange(n, dtype=np.int64)), "v": A(np.ones(n))})
    assert "IMUL" in compile_pipeline([col("p") * col("p") > lit(5)], [col("v").sum()], t64).op_names()
    assert reads == []


def test_year_carries_its_bound_into_a_later_product():
    big = pa.table({"d": A(np.array([949_062_673_333], dtype=np.int64)), "v": A(np.ones(1))})     # year 94906267
    small = pa.table({"d": A(np.array([20_240_101], dtype=np.int64)), "v": A(np.ones(1))})
    pred = (col("d").year() * col("d").year()) > lit(5)
    # an int64 column's year is up to 9.2e14: its square is an int64 op, whatever the values
    assert "IMUL" in compile_pipeline([pred], [col("v").sum()], big).op_names()
    assert "IMUL" in compile_pipeline([pred], [col("v").sum()], small).op_names()
    int32_year = pa.table({"d": A(np.array([20_240_101], dtype=np.int32)), "v": A(np.ones(1))})     # 2.1e9 / 1e4: squares within 2^53
    assert "IMUL" not in compile_pipeline([pred], [col("v").sum()], int32_year).op_names()


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


# -- integer SUM: the float64 accumulator keeps whole numbers only within 2^53 ----------------------

def _sum_table():
    r = np.random.default_rng(5)
    n = 200_000
    a = r.integers(1 << 39, 1 << 40, n).astype(np.int64)          # total ~ 2^57
    b = r.integers(1, 10, n).astype(np.int64)
    m = r.integers(-(1 << 40), 1 << 40, n).astype(np.int64)        # mixed sign
    nul = pa.array(r.integers(1 << 39, 1 << 40, n).astype(np.int64), mask=(np.arange(n) % 7 == 0))   # nullable
    return pa.table({"a": A(a), "m": A(m), "nul": nul, "b": A(b), "i": A(r.integers(1, 100, n).astype(np.int32)),
                     "g": A((np.arange(n) % 3).astype(np.int64)), "v": A(r.random(n))})


def test_integer_sum_is_exact_past_2_53(mode):
    t = _sum_table()
    con = _con(t=t)
    want = con.execute("select sum(a) from t").fetchone()[0]
    assert want == int(t.column("a").to_numpy().sum()) and want > 1 << 53
    got = LazyFrame(t).groupby().agg(col("a").sum().alias("s")).compute(device="cpu")
    assert got.schema.field("s").type == pa.int64()
    assert got.column("s").to_pylist() == [want]
    # an integer expression, a filter, and groups
    want_e = con.execute("select sum(a * b) from t where g = 1").fetchone()[0]
    got = LazyFrame(t).filter(col("g") == lit(1)).groupby().agg((col("a") * col("b")).sum().alias("s")).compute(device="cpu")
    assert got.column("s").to_pylist() == [want_e]
    want_g = dict(con.execute("select g, sum(a) from t group by g").fetchall())
    got = LazyFrame(t).groupby("g").agg(col("a").sum().alias("s")).sort("g").compute(device="cpu")
    assert dict(zip(got.column("g").to_pylist(), got.column("s").to_pylist())) == want_g


def test_integer_sums_that_stay_within_2_53_run_natively_and_equal_duckdb(monkeypatch):
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    t = _sum_table()
    want = _con(t=t).execute("select g, sum(i), sum(b) from t group by g order by g").fetchall()
    got = LazyFrame(t).groupby("g").agg(col("i").sum().alias("si"), col("b").sum().alias("sb")).sort("g").compute(device="cpu")
    assert nx.last_path() == "native"
    assert list(zip(got.column("g").to_pylist(), got.column("si").to_pylist(), got.column("sb").to_pylist())) == \
        [(g, float(si), float(sb)) for g, si, sb in want]


def test_integer_sums_compile_whatever_rows_times_the_largest_value_is():
    """F-060: the kernels add integers in int64, so a sum no longer declines when rows x max|value| passes 2^53
    (it did, because the float64 accumulator lost digits); its slot is an int64 one in both cases."""
    t = _sum_table()
    for c in ("i", "a"):                                                   # 200k * 100, and 200k * 2^40
        prog = compile_pipeline([], [col(c).sum()], t)
        assert prog is not None and prog.slot_kinds == ["isum"] and prog.outputs[0][1] == "isum"
    assert compile_pipeline([], [col("v").sum()], t).slot_kinds == ["sum"]   # a float sum stays a float one


@pytest.mark.parametrize("column", ["m", "nul"])
def test_mixed_sign_and_nullable_integer_sums_equal_duckdb_in_value_and_type(mode, column):
    t = _sum_table()
    con = _con(t=t)
    want = con.execute(f"select sum({column}) from t").fetchone()[0]
    got = LazyFrame(t).groupby().agg(col(column).sum().alias("s")).compute(device="cpu")
    assert got.schema.field("s").type == pa.int64()           # an exact integer, as DuckDB's SUM of a BIGINT
    assert got.column("s").to_pylist() == [int(want)]
    want_g = {g: int(x) for g, x in con.execute(f"select g, sum({column}) from t group by g").fetchall()}
    got = LazyFrame(t).groupby("g").agg(col(column).sum().alias("s")).sort("g").compute(device="cpu")
    assert got.schema.field("s").type == pa.int64()
    assert dict(zip(got.column("g").to_pylist(), got.column("s").to_pylist())) == want_g
