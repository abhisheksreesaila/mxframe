"""F-018: a condition across two leaves sitting between joins (TPC-H Q5 shape)."""
import numpy as np
import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mxframe import LazyFrame, col, lit


def _i32(a):
    return pa.array(np.asarray(a, dtype=np.int32))


def _sql(sql, **tables):
    con = duckdb.connect()
    for name, t in tables.items():
        con.register(name, t)
    return con.execute(sql).to_arrow_table()


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_condition_between_joins_is_applied_when_its_leaves_meet(device):
    rng = np.random.default_rng(13)
    n = 40_000
    li = pa.table({"o": _i32(rng.integers(0, 2000, n)), "s": _i32(rng.integers(0, 100, n)),
                   "v": pa.array(rng.random(n))})
    orders = pa.table({"o": _i32(np.arange(2000)), "c": _i32(rng.integers(0, 300, 2000))})
    cust = pa.table({"c": _i32(np.arange(300)), "cn": _i32(rng.integers(0, 25, 300))})
    supp = pa.table({"s": _i32(np.arange(100)), "sn": _i32(rng.integers(0, 25, 100))})
    nation = pa.table({"cn": _i32(np.arange(25)), "region": _i32(np.arange(25) % 5),
                       "nm": pa.array([f"n{i}" for i in range(25)])})
    lf = (LazyFrame(li).join(LazyFrame(orders), on="o").join(LazyFrame(cust), on="c")
          .join(LazyFrame(supp), on="s")
          .filter(col("cn") == col("sn"))
          .join(LazyFrame(nation).filter(col("region") == lit(2)), on="cn"))
    got = (lf.groupby("nm").agg(col("v").sum().alias("t"), col("v").count().alias("k"))
           .sort("nm").compute(device=device))
    ref = _sql("""select nm, sum(v) t, count(v) k
                  from li join orders using (o) join cust using (c) join supp using (s)
                  join nation on cust.cn = nation.cn
                  where cust.cn = supp.sn and region = 2 group by nm order by nm""",
               li=li, orders=orders, cust=cust, supp=supp, nation=nation)
    assert got.num_rows > 0
    assert got.column("nm").to_pylist() == ref.column("nm").to_pylist()
    assert got.column("k").to_pylist() == ref.column("k").to_pylist()
    np.testing.assert_allclose(got.column("t").to_pylist(), ref.column("t").to_pylist(), rtol=1e-9)


def test_condition_with_nulls_drops_those_rows():
    a = pa.table({"k": _i32([1, 2, 3, 4]), "x": pa.array([1, None, 3, 4], pa.int32())})
    b = pa.table({"k": _i32([1, 2, 3, 4]), "y": pa.array([1, 2, None, 5], pa.int32())})
    c = pa.table({"k": _i32([1, 2, 3, 4]), "z": _i32([10, 20, 30, 40])})
    got = (LazyFrame(a).join(LazyFrame(b), on="k").filter(col("x") == col("y"))
           .join(LazyFrame(c), on="k").compute(device="cpu"))
    assert sorted(got.column("k").to_pylist()) == [1]
