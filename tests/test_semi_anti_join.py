"""F-032 (M2j): semi and anti joins (`how="semi"` / `how="anti"`, as in Polars) and `str_slice`.

A semi join keeps the left rows that have a partner (each once, however many partners), an anti join
the left rows that have none; both return the left columns only, in the left table's row order.
Null keys never match: a semi join drops those rows, an anti join keeps them (SQL EXISTS / NOT
EXISTS). Checked against DuckDB. TPC-H Q22 ("customers who never ordered") needs both features."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A


def _tables(n=50_000, seed=5):
    r = np.random.default_rng(seed)
    left = pa.table({
        "k": pa.array(np.where(r.random(n) < 0.02, None, r.integers(0, 8000, n)).tolist(), pa.int64()),
        "s": pa.array([f"c{v % 700}" for v in r.integers(0, 10_000, n)]),
        "j": A(r.integers(0, 5, n).astype(np.int32)),
        "v": A(r.random(n)),
    })
    m = 30_000
    right = pa.table({                                           # repeated keys: a semi join must not repeat rows
        "rk": A(r.integers(0, 5000, m).astype(np.int64)),
        "rs": pa.array([f"c{v % 900}" for v in r.integers(0, 10_000, m)]),
        "rj": A(r.integers(0, 3, m).astype(np.int32)),
        "w": A(r.random(m)),
    })
    return left, right


def _duck(sql, left, right):
    """``sql`` over l and r; ``ORDER BY rowid`` means the left table's row order (a hidden column)."""
    con = duckdb.connect()
    con.register("l", left.append_column("rowid_", A(np.arange(left.num_rows, dtype=np.int64))))
    con.register("r", right)
    out = con.execute(sql.replace("SELECT * FROM l", "SELECT * EXCLUDE (rowid_) FROM l")
                      .replace("ORDER BY rowid", "ORDER BY rowid_")).fetch_arrow_table()
    return out


def _same_rows_in_order(got: pa.Table, want: pa.Table):
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows
    for c in got.column_names:
        assert got.column(c).to_pylist() == want.column(c).to_pylist(), c


CASES = [
    ("int key", ["k"], ["rk"], "l.k = r.rk"),
    ("string key", ["s"], ["rs"], "l.s = r.rs"),
    ("composite key", ["j", "s"], ["rj", "rs"], "l.j = r.rj AND l.s = r.rs"),
]


@pytest.mark.parametrize("how", ["semi", "anti"])
@pytest.mark.parametrize("name,lk,rk,cond", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_semi_and_anti_joins_match_duckdb(how, name, lk, rk, cond, device):
    left, right = _tables()
    got = LazyFrame(left).join(LazyFrame(right), left_on=lk, right_on=rk, how=how).compute(device=device)
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    want = _duck(f"SELECT * FROM l WHERE {exists} (SELECT 1 FROM r WHERE {cond}) ORDER BY rowid", left, right)
    _same_rows_in_order(got, want)


def test_filters_above_push_to_the_left_side_and_the_answer_holds():
    left, right = _tables()
    q = (LazyFrame(left).join(LazyFrame(right).filter(col("w") < lit(0.5)), left_on="k", right_on="rk", how="anti")
         .filter((col("j") > lit(1)) & (col("v") < lit(0.9))).select("k", "v"))
    plan = str(q.explain())
    assert plan.index("Filter") > plan.index("Join")          # the left filter moved below the join
    want = _duck("SELECT k, v FROM l WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.rk = l.k AND r.w < 0.5) "
                 "AND j > 1 AND v < 0.9 ORDER BY rowid", left, right)
    _same_rows_in_order(q.compute(device="cpu"), want)


def test_semi_join_feeding_an_aggregate():
    left, right = _tables()
    got = (LazyFrame(left).join(LazyFrame(right), left_on="k", right_on="rk", how="semi")
           .groupby("j").agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("j").compute(device="cpu"))
    want = _duck("SELECT j, sum(v) AS s, count(v) AS n FROM l WHERE EXISTS (SELECT 1 FROM r WHERE r.rk = l.k) "
                 "GROUP BY j ORDER BY j", left, right)
    assert got.column("j").to_pylist() == want.column("j").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), want.column("s").to_pylist(), rtol=1e-12)
    assert got.column("n").to_pylist() == want.column("n").to_pylist()


def test_semi_and_anti_split_the_left_table():
    left, right = _tables()
    lf, rf = LazyFrame(left), LazyFrame(right)
    semi = lf.join(rf, left_on="k", right_on="rk", how="semi").compute(device="cpu")
    anti = lf.join(rf, left_on="k", right_on="rk", how="anti").compute(device="cpu")
    assert semi.num_rows + anti.num_rows == left.num_rows
    assert semi.column_names == anti.column_names == left.column_names


def test_empty_right_side():
    left, right = _tables()
    empty = right.slice(0, 0)
    assert LazyFrame(left).join(LazyFrame(empty), left_on="k", right_on="rk", how="semi").compute(device="cpu").num_rows == 0
    anti = LazyFrame(left).join(LazyFrame(empty), left_on="k", right_on="rk", how="anti").compute(device="cpu")
    _same_rows_in_order(anti, left)


def test_unknown_join_type_is_rejected():
    left, right = _tables(100)
    with pytest.raises(ValueError, match="join type"):
        LazyFrame(left).join(LazyFrame(right), left_on="k", right_on="rk", how="outer")


@pytest.mark.parametrize("dictionary", [False, True])
def test_str_slice(dictionary):
    vals = ["13-555-1234", "31-777", "2", "", "héllo-wörld", None]
    arr = pa.array(vals)
    if dictionary:
        arr = arr.dictionary_encode()
    t = pa.table({"p": arr, "i": A(np.arange(len(vals), dtype=np.int64))})
    got = LazyFrame(t).with_columns(col("p").str_slice(0, 2).alias("a"), col("p").str_slice(3).alias("b")).compute(device="cpu")
    assert got.column("a").to_pylist() == [None if v is None else v[0:2] for v in vals]
    assert got.column("b").to_pylist() == [None if v is None else v[3:] for v in vals]
    filt = LazyFrame(t).filter(col("p").str_slice(0, 2).isin(["13", "31"])).compute(device="cpu")
    assert filt.column("i").to_pylist() == [0, 1]
