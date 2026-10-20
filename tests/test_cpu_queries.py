"""F-002: the CPU query paths (grouped, masked-global, join) use the float64 multi-core kernels.

End-to-end through LazyFrame(device="cpu"), against float64 NumPy / DuckDB references computed from
the *same float32 input values*, so any drift comes from the engine, not from the data.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import TPCH_DATA
from omxcore import LazyFrame, col, lit
from omxcore.aot_kernels import AOT_AVAILABLE, set_cpu_threads, get_cpu_threads
from omxcore.lazy_frame import Scan

pytestmark = pytest.mark.skipif(not AOT_AVAILABLE, reason="libmxkernels_aot.so not built")


@pytest.fixture(autouse=True)
def _restore_threads():
    before = get_cpu_threads()
    yield
    set_cpu_threads(before)


def _lineitem_like(n, n_groups, seed=0):
    r = np.random.default_rng(seed)
    return pa.table({
        "g": pa.array(r.integers(0, n_groups, n).astype(np.int32)),
        "price": pa.array((r.random(n) * 100000 + 900).astype(np.float32)),
        "disc": pa.array((r.integers(0, 11, n) / 100).astype(np.float32)),
        "tax": pa.array((r.integers(0, 9, n) / 100).astype(np.float32)),
        "ship": pa.array(r.integers(8000, 10600, n).astype(np.int32)),
    })


@pytest.mark.parametrize("threads", [1, 3, 24])
def test_grouped_sums_are_float64_exact_for_float32_inputs(threads):
    set_cpu_threads(threads)
    t = _lineitem_like(3_000_017, 4)
    out = (LazyFrame(Scan(t)).filter(col("ship") <= lit(10471)).groupby("g")
           .agg(col("price").sum().alias("s"),
                (col("price") * (lit(1.0) - col("disc"))).sum().alias("sd"),
                (col("price") * (lit(1.0) - col("disc")) * (lit(1.0) + col("tax"))).sum().alias("sc"),
                col("price").mean().alias("m"),
                col("price").count().alias("c"))
           .sort("g").compute(device="cpu"))
    g = t.column("g").to_numpy()
    price = t.column("price").to_numpy().astype(np.float64)
    disc = t.column("disc").to_numpy().astype(np.float64)
    tax = t.column("tax").to_numpy().astype(np.float64)
    keep = t.column("ship").to_numpy() <= 10471
    for i, gid in enumerate(out.column("g").to_pylist()):
        sel = keep & (g == gid)
        ref = {"s": price[sel].sum(), "sd": (price * (1 - disc))[sel].sum(),
               "sc": (price * (1 - disc) * (1 + tax))[sel].sum(), "m": price[sel].mean(),
               "c": float(sel.sum())}
        for name, want in ref.items():
            got = out.column(name)[i].as_py()
            assert got == pytest.approx(want, rel=1e-9), (name, gid, got, want)
    assert out.schema.field("s").type == pa.float64()
    assert out.schema.field("c").type == pa.int64()


def test_masked_global_sums_are_float64_exact():
    t = _lineitem_like(2_000_003, 1, seed=1)
    price = t.column("price").to_numpy().astype(np.float64)
    disc = t.column("disc").to_numpy().astype(np.float64)
    ship = t.column("ship").to_numpy()
    out = (LazyFrame(Scan(t))
           .filter((col("ship") >= lit(9000)) & (col("ship") < lit(9800))
                   & (col("disc") >= lit(0.045)) & (col("disc") <= lit(0.075)))
           .groupby()
           .agg((col("price") * col("disc")).sum().alias("rev"),
                (col("price") * (lit(1.0) - col("disc"))).sum().alias("rev2"),
                col("price").sum().alias("s"),
                col("price").min().alias("lo"), col("price").max().alias("hi"))
           .compute(device="cpu"))
    # discount bounds sit between the k/100 grid points, so float32-vs-literal rounding cannot flip a row
    keep = (ship >= 9000) & (ship < 9800) & (disc >= 0.045) & (disc <= 0.075)
    assert out.column("rev")[0].as_py() == pytest.approx((price * disc)[keep].sum(), rel=1e-9)
    assert out.column("rev2")[0].as_py() == pytest.approx((price * (1 - disc))[keep].sum(), rel=1e-9)
    assert out.column("s")[0].as_py() == pytest.approx(price[keep].sum(), rel=1e-9)
    assert out.column("lo")[0].as_py() == price[keep].min()
    assert out.column("hi")[0].as_py() == price[keep].max()


@pytest.mark.parametrize("how", ["inner", "left"])
def test_join_matches_pyarrow(how):
    r = np.random.default_rng(7)
    left = pa.table({"k": pa.array(r.integers(0, 30_000, 400_000).astype(np.int64)),
                     "a": pa.array(np.arange(400_000, dtype=np.int64))})
    right = pa.table({"k": pa.array(np.arange(0, 30_000, 2, dtype=np.int64)),
                      "b": pa.array(np.arange(15_000, dtype=np.int64) * 3)})
    got = (LazyFrame(Scan(left)).join(LazyFrame(Scan(right)), on="k", how=how).compute(device="cpu"))
    want = left.join(right, keys="k", join_type="inner" if how == "inner" else "left outer")
    assert got.num_rows == want.num_rows
    g = got.select(["a", "b"]).to_pandas().sort_values("a").reset_index(drop=True)
    w = want.select(["a", "b"]).to_pandas().sort_values("a").reset_index(drop=True)
    assert g["a"].tolist() == w["a"].tolist()
    assert g["b"].fillna(-1).tolist() == w["b"].fillna(-1).tolist()


@pytest.mark.skipif(not (TPCH_DATA / "tpch_sf1" / "lineitem.parquet").exists(), reason="TPC-H SF1 data not present")
def test_tpch_sf1_q1_matches_duckdb_to_1e9():
    duckdb = pytest.importorskip("duckdb")
    path = str(TPCH_DATA / "tpch_sf1" / "lineitem.parquet")
    tbl = pq.read_table(path, columns=["l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice",
                                       "l_discount", "l_tax", "l_shipdate"])
    out = (LazyFrame(Scan(tbl)).filter(col("l_shipdate") <= lit(10471))
           .groupby("l_returnflag", "l_linestatus")
           .agg(col("l_quantity").sum().alias("sum_qty"),
                col("l_extendedprice").sum().alias("sum_base_price"),
                (col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("sum_disc_price"),
                (col("l_extendedprice") * (lit(1.0) - col("l_discount")) * (lit(1.0) + col("l_tax"))).sum().alias("sum_charge"),
                col("l_quantity").mean().alias("avg_qty"),
                col("l_extendedprice").mean().alias("avg_price"),
                col("l_discount").mean().alias("avg_disc"),
                col("l_quantity").count().alias("count_order"))
           .sort("l_returnflag", "l_linestatus").compute(device="cpu"))
    ref = duckdb.connect().execute(f"""
        SELECT sum(CAST(l_quantity AS DOUBLE)), sum(CAST(l_extendedprice AS DOUBLE)),
               sum(CAST(l_extendedprice AS DOUBLE) * (1 - CAST(l_discount AS DOUBLE))),
               sum(CAST(l_extendedprice AS DOUBLE) * (1 - CAST(l_discount AS DOUBLE)) * (1 + CAST(l_tax AS DOUBLE))),
               avg(CAST(l_quantity AS DOUBLE)), avg(CAST(l_extendedprice AS DOUBLE)),
               avg(CAST(l_discount AS DOUBLE)), count(*)
        FROM read_parquet('{path}') WHERE l_shipdate <= 10471
        GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus""").fetchall()
    names = ["sum_qty", "sum_base_price", "sum_disc_price", "sum_charge",
             "avg_qty", "avg_price", "avg_disc", "count_order"]
    assert out.num_rows == len(ref)
    for i, row in enumerate(ref):
        for name, want in zip(names, row):
            got = out.column(name)[i].as_py()
            assert got == pytest.approx(float(want), rel=1e-9), (name, i, got, want)
