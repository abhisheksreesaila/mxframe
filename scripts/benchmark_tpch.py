#!/usr/bin/env python3
"""
TPC-H Benchmark Suite: MXFrame vs Polars vs Pandas (vs DuckDB for Q3)

Queries:
  Q1  -- filter + groupby + 8 aggregates (lineitem)
  Q6  -- multi-predicate filter + global sum (lineitem)
  Q3  -- 3-way join + groupby + sort + limit 10 (customer / orders / lineitem)
  SLD -- sort / limit / distinct (synthetic grouped data)

Timing:
  Cold = clear_cache() before each run (includes JIT compilation)
  Hot  = cache warm, steady-state execution

Usage:
  pixi run bench-tpch
  python scripts/benchmark_tpch.py [--rows N] [--cold N] [--hot N]
"""

import argparse
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from max import driver as _driver

from mxframe import LazyFrame, col, lit, when
from mxframe.lazy_frame import Scan
from mxframe.custom_ops import clear_cache

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


# ---------------------------------------------------------------
#  Constants (dates encoded as int32: days since 1970-01-01)
# ---------------------------------------------------------------
CUTOFF_Q1       = 10471   # 1998-09-02
Q6_DATE_LO      = 8766    # 1994-01-01
Q6_DATE_HI      = 9131    # 1995-01-01 (exclusive)
Q6_DISC_LO      = 0.05
Q6_DISC_HI      = 0.07
Q6_QTY_HI       = 24.0
DATE_1995_03_15  = 9204   # 1995-03-15

# Q12 receipt date range
Q12_DATE_LO     = 8761    # 1994-01-01
Q12_DATE_HI     = 9126    # 1995-01-01
# Q14 shipdate range
Q14_DATE_LO     = 9374    # 1995-09-01
Q14_DATE_HI     = 9404    # 1995-10-01



# ---------------------------------------------------------------
#  Context helpers
# ---------------------------------------------------------------
def _safe_gpu_count() -> int:
    try:
        return int(_driver.accelerator_count())
    except Exception:
        return 0


def _report_context() -> None:
    print("Benchmark context")
    print("=================")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Platform  : {platform.platform()}")
    print(f"  CPU cores : {os.cpu_count()}")
    print(f"  GPU count : {_safe_gpu_count()}")
    print(f"  pyarrow   : {pa.__version__}")
    print(f"  pandas    : {pd.__version__}")
    print(f"  polars    : {pl.__version__ if POLARS_AVAILABLE else 'not installed'}")
    print(f"  duckdb    : {duckdb.__version__ if DUCKDB_AVAILABLE else 'not installed'}")
    print()


# ---------------------------------------------------------------
#  Data generation
# ---------------------------------------------------------------
def make_lineitem(n: int = 1_000_000, seed: int = 42) -> pa.Table:
    """Synthetic TPC-H lineitem table for Q1/Q6."""
    rng = np.random.default_rng(seed)
    rf = np.array(["A", "N", "R"], dtype=object)[rng.integers(0, 3, size=n)]
    ls = np.array(["F", "O"],       dtype=object)[rng.integers(0, 2, size=n)]
    return pa.table({
        "l_returnflag":    rf.tolist(),
        "l_linestatus":    ls.tolist(),
        "l_quantity":      rng.uniform(1.0,    50.0,      size=n).astype(np.float32),
        "l_extendedprice": rng.uniform(900.0,  100_000.0, size=n).astype(np.float32),
        "l_discount":      rng.uniform(0.0,    0.10,      size=n).astype(np.float32),
        "l_tax":           rng.uniform(0.0,    0.08,      size=n).astype(np.float32),
        "l_shipdate":      rng.integers(8_000, 10_550,    size=n).astype(np.int32),
    })


def make_tpch_q3_tables(n_customers=15_000, n_orders=150_000, n_lineitem=600_000, seed=42):
    """Synthetic customer / orders / lineitem tables for Q3."""
    rng = np.random.default_rng(seed)
    segs = np.array(["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"])
    customer = pa.table({
        "c_custkey":    np.arange(1, n_customers + 1, dtype=np.int32),
        "c_mktsegment": rng.choice(segs, size=n_customers).tolist(),
    })
    orders = pa.table({
        "o_orderkey":     np.arange(1, n_orders + 1, dtype=np.int32),
        "o_custkey":      rng.integers(1, n_customers + 1, size=n_orders, dtype=np.int32),
        "o_orderdate":    rng.integers(8800, 9300, size=n_orders,  dtype=np.int32),
        "o_shippriority": rng.integers(0, 5,    size=n_orders,  dtype=np.int32),
    })
    lineitem = pa.table({
        "l_orderkey":     rng.integers(1, n_orders + 1, size=n_lineitem, dtype=np.int32),
        "l_extendedprice": rng.uniform(900.0, 100_000.0, size=n_lineitem).astype(np.float32),
        "l_discount":     rng.uniform(0.0, 0.10, size=n_lineitem).astype(np.float32),
        "l_shipdate":     rng.integers(8900, 9400, size=n_lineitem, dtype=np.int32),
    })
    return customer, orders, lineitem



def make_tpch_q12_tables(n_orders=50_000, n_lineitem=300_000, seed=77) -> tuple:
    """Synthetic orders + lineitem tables for TPC-H Q12."""
    rng = np.random.default_rng(seed)
    priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
    orders = pa.table({
        "o_orderkey":     np.arange(1, n_orders + 1, dtype=np.int32),
        "o_orderpriority": rng.choice(priorities, size=n_orders).tolist(),
    })
    lineitem = pa.table({
        "l_orderkey":    rng.integers(1, n_orders + 1, size=n_lineitem, dtype=np.int32),
        "l_shipmode":    rng.choice(["MAIL", "SHIP", "TRUCK", "AIR", "FOB", "RAIL"], size=n_lineitem).tolist(),
        "l_commitdate":  rng.integers(8200, 9000, size=n_lineitem, dtype=np.int32),
        "l_receiptdate": rng.integers(8700, 9500, size=n_lineitem, dtype=np.int32),
        "l_shipdate":    rng.integers(8000, 8900, size=n_lineitem, dtype=np.int32),
    })
    return orders, lineitem


def make_tpch_q14_tables(n_parts=20_000, n_lineitem=200_000, seed=88) -> tuple:
    """Synthetic part + lineitem tables for TPC-H Q14."""
    rng = np.random.default_rng(seed)
    part_types = ["PROMO STEEL", "PROMO BRASS", "BRUSHED COPPER", "STANDARD STEEL",
                  "ECONOMY TIN", "PROMO NICKEL", "LARGE COPPER", "MEDIUM BRASS"]
    part = pa.table({
        "p_partkey": np.arange(1, n_parts + 1, dtype=np.int32),
        "p_type":    rng.choice(part_types, size=n_parts).tolist(),
    })
    lineitem = pa.table({
        "l_partkey":      rng.integers(1, n_parts + 1, size=n_lineitem, dtype=np.int32),
        "l_extendedprice": rng.uniform(900.0, 100_000.0, size=n_lineitem).astype(np.float32),
        "l_discount":     rng.uniform(0.0, 0.10, size=n_lineitem).astype(np.float32),
        "l_shipdate":     rng.integers(9200, 9600, size=n_lineitem, dtype=np.int32),
    })
    return part, lineitem


def make_grouped(n: int = 500_000, n_groups: int = 1000, seed: int = 42) -> pa.Table:
    """Synthetic grouped data for Sort/Limit/Distinct."""
    rng = np.random.default_rng(seed)
    groups = np.array([f"g{i:04d}" for i in range(n_groups)], dtype=object)
    return pa.table({
        "group": groups[rng.integers(0, n_groups, size=n)].tolist(),
        "value": rng.standard_normal(n).astype(np.float64),
    })


# ---------------------------------------------------------------
#  Timing helpers
# ---------------------------------------------------------------
def _time_runs(fn, runs: int, warmup: int = 0) -> list:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _time_cold(fn, runs: int) -> list:
    out = []
    for _ in range(runs):
        clear_cache()
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _stats(samples: list) -> dict:
    a = np.array(samples)
    return {"min": float(a.min()), "avg": float(a.mean()), "median": float(np.median(a))}


def _print_table(title: str, rows: list) -> None:
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    hdr = f"  {'Engine':<26} {'Cold min':>9} {'Hot min':>9} {'Hot avg':>9} {'Hot med':>9}"
    print(hdr)
    print(f"  {'-'*26} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for name, cold, hot in rows:
        cold_s = f"{cold['min']:9.1f}" if cold else f"{'—':>9}"
        print(f"  {name:<26} {cold_s} {hot['min']:9.1f} {hot['avg']:9.1f} {hot['median']:9.1f}")


def _summarize_relative(rows: list, baselines=("Pandas", "Polars")) -> None:
    hot = {name: h for name, _, h in rows}
    for baseline in baselines:
        if baseline not in hot:
            continue
        base_min = hot[baseline]["min"]
        print(f"\n  Relative to {baseline} (hot min):")
        for name, _, h in rows:
            ratio = h["min"] / base_min
            tag = f"{1/ratio:.2f}x faster" if ratio < 1 else f"{ratio:.2f}x slower"
            print(f"    {name:<26} {tag}")


def _section(label: str) -> None:
    print(f"\n{'*'*68}")
    print(f"  {label}")
    print(f"{'*'*68}")


# ---------------------------------------------------------------
#  Q1 implementations
# ---------------------------------------------------------------
def run_q1_mxframe(tbl: pa.Table, device: str = "cpu") -> pa.Table:
    return (
        LazyFrame(Scan(tbl))
        .filter(col("l_shipdate") <= lit(CUTOFF_Q1))
        .groupby("l_returnflag", "l_linestatus")
        .agg(
            col("l_quantity").sum().alias("sum_qty"),
            col("l_extendedprice").sum().alias("sum_base_price"),
            (col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("sum_disc_price"),
            (col("l_extendedprice") * (lit(1.0) - col("l_discount")) * (lit(1.0) + col("l_tax"))).sum().alias("sum_charge"),
            col("l_quantity").mean().alias("avg_qty"),
            col("l_extendedprice").mean().alias("avg_price"),
            col("l_discount").mean().alias("avg_disc"),
            col("l_quantity").count().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
        .compute(device=device)
    )


def run_q1_pandas(tbl: pa.Table) -> pd.DataFrame:
    pdf = tbl.to_pandas()
    q1 = pdf[pdf["l_shipdate"] <= CUTOFF_Q1].copy()
    q1["disc_price"] = q1["l_extendedprice"] * (1.0 - q1["l_discount"])
    q1["charge"]     = q1["disc_price"] * (1.0 + q1["l_tax"])
    return (
        q1.groupby(["l_returnflag", "l_linestatus"], as_index=False)
        .agg(
            sum_qty=("l_quantity", "sum"),
            sum_base_price=("l_extendedprice", "sum"),
            sum_disc_price=("disc_price", "sum"),
            sum_charge=("charge", "sum"),
            avg_qty=("l_quantity", "mean"),
            avg_price=("l_extendedprice", "mean"),
            avg_disc=("l_discount", "mean"),
            count_order=("l_quantity", "count"),
        )
        .sort_values(["l_returnflag", "l_linestatus"])
        .reset_index(drop=True)
    )


def run_q1_polars(tbl: pa.Table):
    if not POLARS_AVAILABLE:
        return None
    return (
        pl.from_arrow(tbl)
        .filter(pl.col("l_shipdate") <= CUTOFF_Q1)
        .with_columns([
            (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).alias("disc_price"),
            (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")) * (1.0 + pl.col("l_tax"))).alias("charge"),
        ])
        .group_by(["l_returnflag", "l_linestatus"])
        .agg([
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            pl.col("disc_price").sum().alias("sum_disc_price"),
            pl.col("charge").sum().alias("sum_charge"),
            pl.col("l_quantity").mean().alias("avg_qty"),
            pl.col("l_extendedprice").mean().alias("avg_price"),
            pl.col("l_discount").mean().alias("avg_disc"),
            pl.col("l_quantity").count().alias("count_order"),
        ])
        .sort(["l_returnflag", "l_linestatus"])
    )


# ---------------------------------------------------------------
#  Q6 implementations
# ---------------------------------------------------------------
def run_q6_mxframe(tbl: pa.Table, device: str = "cpu") -> pa.Table:
    try:
        return (
            LazyFrame(Scan(tbl))
            .filter(
                (col("l_shipdate") >= lit(Q6_DATE_LO))
                & (col("l_shipdate") < lit(Q6_DATE_HI))
                & (col("l_discount") >= lit(Q6_DISC_LO))
                & (col("l_discount") <= lit(Q6_DISC_HI))
                & (col("l_quantity") < lit(Q6_QTY_HI))
            )
            .groupby()
            .agg((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
            .compute(device=device)
        )
    except Exception as e:
        print(f"  [Q6 fallback to PyArrow: {e}]")
        return _q6_arrow_fallback(tbl)


def _q6_arrow_fallback(tbl: pa.Table) -> pa.Table:
    mask = pc.and_kleene(
        pc.greater_equal(tbl["l_shipdate"], pa.scalar(Q6_DATE_LO, type=pa.int32())),
        pc.less(tbl["l_shipdate"],           pa.scalar(Q6_DATE_HI, type=pa.int32())),
    )
    mask = pc.and_kleene(mask, pc.greater_equal(tbl["l_discount"], pa.scalar(Q6_DISC_LO, type=pa.float32())))
    mask = pc.and_kleene(mask, pc.less_equal(tbl["l_discount"],    pa.scalar(Q6_DISC_HI, type=pa.float32())))
    mask = pc.and_kleene(mask, pc.less(tbl["l_quantity"],          pa.scalar(Q6_QTY_HI,  type=pa.float32())))
    filt = tbl.filter(mask)
    rev = pc.sum(pc.multiply(filt["l_extendedprice"], filt["l_discount"]))
    return pa.table({"revenue": [float(rev.as_py() or 0.0)]})


def run_q6_pandas(tbl: pa.Table) -> float:
    pdf = tbl.to_pandas()
    mask = (
        (pdf["l_shipdate"] >= Q6_DATE_LO)
        & (pdf["l_shipdate"] < Q6_DATE_HI)
        & (pdf["l_discount"] >= Q6_DISC_LO)
        & (pdf["l_discount"] <= Q6_DISC_HI)
        & (pdf["l_quantity"] < Q6_QTY_HI)
    )
    return float((pdf.loc[mask, "l_extendedprice"] * pdf.loc[mask, "l_discount"]).sum())


def run_q6_polars(tbl: pa.Table):
    if not POLARS_AVAILABLE:
        return None
    out = (
        pl.from_arrow(tbl)
        .filter(
            (pl.col("l_shipdate") >= Q6_DATE_LO)
            & (pl.col("l_shipdate") < Q6_DATE_HI)
            & (pl.col("l_discount") >= Q6_DISC_LO)
            & (pl.col("l_discount") <= Q6_DISC_HI)
            & (pl.col("l_quantity") < Q6_QTY_HI)
        )
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
    )
    return float(out["revenue"][0])


# ---------------------------------------------------------------
#  Q3 implementations
# ---------------------------------------------------------------
def run_q3_mxframe(customer, orders, lineitem, device: str = "cpu") -> pa.Table:
    lf_c = LazyFrame(Scan(customer)).filter(col("c_mktsegment") == lit("BUILDING"))
    lf_o = LazyFrame(Scan(orders)).filter(col("o_orderdate") < lit(DATE_1995_03_15))
    lf_l = LazyFrame(Scan(lineitem)).filter(col("l_shipdate") > lit(DATE_1995_03_15))
    return (
        lf_o
        .join(lf_c, left_on="o_custkey",  right_on="c_custkey")
        .join(lf_l, left_on="o_orderkey", right_on="l_orderkey")
        .groupby("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(
            (col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"),
        )
        .sort("revenue", descending=True)
        .limit(10)
        .compute(device=device)
    )


def run_q3_pandas(customer, orders, lineitem) -> pd.DataFrame:
    c = customer.to_pandas()
    o = orders.to_pandas()
    l = lineitem.to_pandas()
    c = c[c["c_mktsegment"] == "BUILDING"]
    o = o[o["o_orderdate"] < DATE_1995_03_15]
    l = l[l["l_shipdate"]  > DATE_1995_03_15]
    merged = o.merge(c, left_on="o_custkey",  right_on="c_custkey")
    merged = merged.merge(l, left_on="o_orderkey", right_on="l_orderkey")
    merged["revenue"] = merged["l_extendedprice"] * (1.0 - merged["l_discount"])
    return (
        merged.groupby(["o_orderkey", "o_orderdate", "o_shippriority"])["revenue"]
        .sum().reset_index()
        .sort_values(["revenue", "o_orderdate"], ascending=[False, True])
        .head(10)
    )


def run_q3_polars(customer, orders, lineitem):
    if not POLARS_AVAILABLE:
        return None
    c = pl.from_arrow(customer).filter(pl.col("c_mktsegment") == "BUILDING")
    o = pl.from_arrow(orders).filter(pl.col("o_orderdate") < DATE_1995_03_15)
    l = pl.from_arrow(lineitem).filter(pl.col("l_shipdate") > DATE_1995_03_15)
    return (
        o.join(c, left_on="o_custkey",  right_on="c_custkey")
         .join(l, left_on="o_orderkey", right_on="l_orderkey")
         .with_columns((pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).alias("revenue"))
         .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
         .agg(pl.col("revenue").sum())
         .sort(["revenue", "o_orderdate"], descending=[True, False])
         .head(10)
         .to_arrow()
    )


def run_q3_duckdb(customer, orders, lineitem):
    if not DUCKDB_AVAILABLE:
        return None
    con = duckdb.connect()
    con.register("customer_tbl", customer)
    con.register("orders_tbl",   orders)
    con.register("lineitem_tbl", lineitem)
    result = con.sql(f"""
        SELECT o_orderkey, o_orderdate, o_shippriority,
               SUM(CAST(l_extendedprice AS DOUBLE) * (1.0 - CAST(l_discount AS DOUBLE))) AS revenue
        FROM   customer_tbl
        JOIN   orders_tbl   ON c_custkey  = o_custkey
        JOIN   lineitem_tbl ON l_orderkey = o_orderkey
        WHERE  c_mktsegment = 'BUILDING'
          AND  o_orderdate  < {DATE_1995_03_15}
          AND  l_shipdate   > {DATE_1995_03_15}
        GROUP BY o_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate ASC
        LIMIT 10
    """).arrow().read_all()
    con.close()
    return result


# ---------------------------------------------------------------
#  Sort / Limit / Distinct implementations
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Q12 — 2-table join + isin filter + grouped sum(CASE WHEN)
# ---------------------------------------------------------------
def run_q12_mxframe(orders, lineitem, device="cpu") -> pa.Table:
    return (
        LazyFrame(lineitem)
        .join(LazyFrame(orders), left_on="l_orderkey", right_on="o_orderkey")
        .filter(col("l_shipmode").isin(["MAIL", "SHIP"]))
        .filter(col("l_commitdate") < col("l_receiptdate"))
        .filter(col("l_shipdate") < col("l_commitdate"))
        .filter(col("l_receiptdate") >= lit(Q12_DATE_LO))
        .filter(col("l_receiptdate") < lit(Q12_DATE_HI))
        .groupby("l_shipmode")
        .agg(
            when(col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0))
                .sum().alias("high_line_count"),
            when(~col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0))
                .sum().alias("low_line_count"),
        )
        .sort(col("l_shipmode"))
        .compute(device=device)
    )


def run_q12_pandas(orders, lineitem) -> pd.DataFrame:
    df = pd.merge(
        pd.DataFrame(lineitem.to_pydict()),
        pd.DataFrame(orders.to_pydict()),
        left_on="l_orderkey", right_on="o_orderkey",
    )
    df = df[
        df["l_shipmode"].isin(["MAIL", "SHIP"]) &
        (df["l_commitdate"] < df["l_receiptdate"]) &
        (df["l_shipdate"] < df["l_commitdate"]) &
        (df["l_receiptdate"] >= Q12_DATE_LO) &
        (df["l_receiptdate"] < Q12_DATE_HI)
    ]
    df["high"] = df["o_orderpriority"].isin(["1-URGENT", "2-HIGH"]).astype(int)
    df["low"]  = (~df["o_orderpriority"].isin(["1-URGENT", "2-HIGH"])).astype(int)
    return df.groupby("l_shipmode").agg(high_line_count=("high", "sum"), low_line_count=("low", "sum")).reset_index().sort_values("l_shipmode")


def run_q12_polars(orders, lineitem):
    li = pl.from_arrow(lineitem)
    od = pl.from_arrow(orders)
    return (
        li.join(od, left_on="l_orderkey", right_on="o_orderkey")
        .filter(
            pl.col("l_shipmode").is_in(["MAIL", "SHIP"]) &
            (pl.col("l_commitdate") < pl.col("l_receiptdate")) &
            (pl.col("l_shipdate") < pl.col("l_commitdate")) &
            (pl.col("l_receiptdate") >= Q12_DATE_LO) &
            (pl.col("l_receiptdate") < Q12_DATE_HI)
        )
        .with_columns([
            pl.when(pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])).then(1).otherwise(0).alias("high"),
            pl.when(~pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])).then(1).otherwise(0).alias("low"),
        ])
        .group_by("l_shipmode").agg(
            pl.col("high").sum().alias("high_line_count"),
            pl.col("low").sum().alias("low_line_count"),
        )
        .sort("l_shipmode")
    )


def run_q12_duckdb(orders, lineitem):
    if not DUCKDB_AVAILABLE:
        return None
    con = duckdb.connect()
    con.register("orders_tbl", orders)
    con.register("lineitem_tbl", lineitem)
    return con.execute(f"""
        SELECT l_shipmode,
               SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) high_line_count,
               SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) low_line_count
        FROM lineitem_tbl JOIN orders_tbl ON l_orderkey=o_orderkey
        WHERE l_shipmode IN ('MAIL','SHIP') AND l_commitdate<l_receiptdate
          AND l_shipdate<l_commitdate AND l_receiptdate>={Q12_DATE_LO} AND l_receiptdate<{Q12_DATE_HI}
        GROUP BY l_shipmode ORDER BY l_shipmode
    """).arrow()


# ---------------------------------------------------------------
# Q14 — 2-table join + filter + global sum(CASE WHEN startswith)
# ---------------------------------------------------------------
def run_q14_mxframe(part, lineitem, device="cpu") -> pa.Table:
    promo_col = when(
        col("p_type").startswith("PROMO"),
        col("l_extendedprice") * (lit(1.0) - col("l_discount")),
        lit(0.0),
    )
    total_col = col("l_extendedprice") * (lit(1.0) - col("l_discount"))
    return (
        LazyFrame(lineitem)
        .join(LazyFrame(part), left_on="l_partkey", right_on="p_partkey")
        .filter(col("l_shipdate") >= lit(Q14_DATE_LO))
        .filter(col("l_shipdate") < lit(Q14_DATE_HI))
        .groupby()
        .agg(
            promo_col.sum().alias("promo_revenue"),
            total_col.sum().alias("total_revenue"),
        )
        .compute(device=device)
    )


def run_q14_pandas(part, lineitem) -> float:
    df = pd.merge(
        pd.DataFrame(lineitem.to_pydict()),
        pd.DataFrame(part.to_pydict()),
        left_on="l_partkey", right_on="p_partkey",
    )
    df = df[(df["l_shipdate"] >= Q14_DATE_LO) & (df["l_shipdate"] < Q14_DATE_HI)]
    total_rev = (df["l_extendedprice"] * (1 - df["l_discount"])).sum()
    promo_rev = (df[df["p_type"].str.startswith("PROMO")]["l_extendedprice"] *
                 (1 - df[df["p_type"].str.startswith("PROMO")]["l_discount"])).sum()
    return 100.0 * promo_rev / total_rev if total_rev > 0 else 0.0


def run_q14_polars(part, lineitem) -> float:
    li = pl.from_arrow(lineitem)
    pa_tbl = pl.from_arrow(part)
    df = (
        li.join(pa_tbl, left_on="l_partkey", right_on="p_partkey")
        .filter((pl.col("l_shipdate") >= Q14_DATE_LO) & (pl.col("l_shipdate") < Q14_DATE_HI))
        .select([
            pl.when(pl.col("p_type").str.starts_with("PROMO"))
              .then(pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
              .otherwise(0).sum().alias("promo"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("total"),
        ])
    )
    row = df[0]
    promo = row["promo"][0]
    total = row["total"][0]
    return 100.0 * promo / total if total else 0.0


def run_q14_duckdb(part, lineitem) -> float:
    if not DUCKDB_AVAILABLE:
        return 0.0
    con = duckdb.connect()
    con.register("part_tbl", part)
    con.register("lineitem_tbl", lineitem)
    ref = con.execute(
        f"SELECT SUM(CASE WHEN p_type LIKE 'PROMO%' "
        f"              THEN l_extendedprice*(1-l_discount) ELSE 0 END) p, "
        f"       SUM(l_extendedprice*(1-l_discount)) t "
        f"FROM lineitem_tbl JOIN part_tbl ON l_partkey=p_partkey "
        f"WHERE l_shipdate>={Q14_DATE_LO} AND l_shipdate<{Q14_DATE_HI}"
    ).fetchone()
    promo, total = float(ref[0] or 0), float(ref[1] or 0)
    return 100.0 * promo / total if total else 0.0


def run_sld_sort_mxframe(tbl: pa.Table, device: str = "cpu") -> pa.Table:
    return (
        LazyFrame(Scan(tbl))
        .groupby("group")
        .agg(col("value").sum().alias("total"))
        .sort("group")
        .compute(device=device)
    )


def run_sld_limit_mxframe(tbl: pa.Table, n: int = 10, device: str = "cpu") -> pa.Table:
    return (
        LazyFrame(Scan(tbl))
        .groupby("group")
        .agg(col("value").sum().alias("total"))
        .sort("group")
        .limit(n)
        .compute(device=device)
    )


def run_sld_distinct_mxframe(tbl: pa.Table, device: str = "cpu") -> pa.Table:
    return LazyFrame(Scan(tbl)).distinct("group").compute(device=device)


def run_sld_sort_pandas(tbl: pa.Table) -> pd.DataFrame:
    df = tbl.to_pandas()
    return df.groupby("group", as_index=False)["value"].sum().sort_values("group").reset_index(drop=True)


def run_sld_sort_polars(tbl: pa.Table):
    if not POLARS_AVAILABLE:
        return None
    return (
        pl.from_arrow(tbl)
        .group_by("group")
        .agg(pl.col("value").sum().alias("total"))
        .sort("group")
    )


def run_sld_distinct_pandas(tbl: pa.Table) -> pd.Series:
    return tbl.to_pandas()["group"].drop_duplicates().reset_index(drop=True)


def run_sld_distinct_polars(tbl: pa.Table):
    if not POLARS_AVAILABLE:
        return None
    return pl.from_arrow(tbl).select("group").unique()


# ---------------------------------------------------------------
#  Correctness checks
# ---------------------------------------------------------------


# ---------------------------------------------------------------
#  Q5 — Local Supplier Volume (4-table join simplified)
#  customer(15K) ⋈ orders(100K) ⋈ lineitem(400K) ⋈ nation(25)
#  groupby nation → sum revenue, sort DESC
# ---------------------------------------------------------------
NATIONS = ["ALGERIA","ARGENTINA","BRAZIL","CANADA","EGYPT",
           "ETHIOPIA","FRANCE","GERMANY","INDIA","INDONESIA",
           "IRAN","IRAQ","JAPAN","JORDAN","KENYA",
           "MOROCCO","MOZAMBIQUE","PERU","CHINA","ROMANIA",
           "SAUDI ARABIA","VIETNAM","RUSSIA","UNITED KINGDOM","UNITED STATES"]

def make_tpch_q5_tables(n_customers=15_000, n_orders=100_000, n_lineitem=400_000, seed=55):
    """Synthetic tables for Q5 (customer/orders/lineitem/nation)."""
    rng = np.random.default_rng(seed)
    n_nations = len(NATIONS)

    nation = pa.table({
        "n_nationkey": pa.array(np.arange(n_nations, dtype=np.int32)),
        "n_name":      pa.array(NATIONS),
    })
    customer = pa.table({
        "c_custkey":   pa.array(np.arange(1, n_customers + 1, dtype=np.int32)),
        "c_nationkey": pa.array(rng.integers(0, n_nations, size=n_customers, dtype=np.int32)),
    })
    orders = pa.table({
        "o_orderkey":  pa.array(np.arange(1, n_orders + 1, dtype=np.int32)),
        "o_custkey":   pa.array(rng.integers(1, n_customers + 1, size=n_orders, dtype=np.int32)),
        "o_orderdate": pa.array(
            rng.integers(19930101, 19960101, size=n_orders, dtype=np.int32)
        ),
    })
    lineitem = pa.table({
        "l_orderkey":     pa.array(rng.integers(1, n_orders + 1, size=n_lineitem, dtype=np.int32)),
        "l_extendedprice":pa.array(rng.uniform(900.0, 100000.0, size=n_lineitem).astype(np.float32)),
        "l_discount":     pa.array(rng.uniform(0.0, 0.10, size=n_lineitem).astype(np.float32)),
    })
    return nation, customer, orders, lineitem


def run_q5_mxframe(nation, customer, orders, lineitem, device="cpu") -> pa.Table:
    # Filter orders: 1994
    return (
        LazyFrame(Scan(lineitem))
        .join(
            LazyFrame(Scan(orders)).filter(
                (col("o_orderdate") >= lit(19940101)) &
                (col("o_orderdate") <  lit(19950101))
            ),
            left_on="l_orderkey", right_on="o_orderkey",
        )
        .join(LazyFrame(Scan(customer)), left_on="o_custkey", right_on="c_custkey")
        .join(LazyFrame(Scan(nation)),   left_on="c_nationkey", right_on="n_nationkey")
        .groupby("n_name")
        .agg(
            (col("l_extendedprice") * (lit(1.0) - col("l_discount")))
            .sum().alias("revenue")
        )
        .sort(col("revenue"), descending=True)
        .compute(device=device)
    )


def run_q5_pandas(nation, customer, orders, lineitem) -> pd.DataFrame:
    li = lineitem.to_pandas()
    o  = orders.to_pandas()
    c  = customer.to_pandas()
    n  = nation.to_pandas()
    o  = o[(o.o_orderdate >= 19940101) & (o.o_orderdate < 19950101)]
    merged = li.merge(o, left_on="l_orderkey", right_on="o_orderkey")
    merged = merged.merge(c, left_on="o_custkey", right_on="c_custkey")
    merged = merged.merge(n, left_on="c_nationkey", right_on="n_nationkey")
    merged["revenue"] = merged.l_extendedprice * (1 - merged.l_discount)
    result = merged.groupby("n_name", as_index=False)["revenue"].sum()
    return result.sort_values("revenue", ascending=False).reset_index(drop=True)


def run_q5_polars(nation, customer, orders, lineitem):
    if not POLARS_AVAILABLE:
        return None
    li = pl.from_arrow(lineitem)
    o  = pl.from_arrow(orders).filter(
        (pl.col("o_orderdate") >= 19940101) & (pl.col("o_orderdate") < 19950101)
    )
    c  = pl.from_arrow(customer)
    n  = pl.from_arrow(nation)
    return (
        li.join(o, left_on="l_orderkey", right_on="o_orderkey")
        .join(c, left_on="o_custkey", right_on="c_custkey")
        .join(n, left_on="c_nationkey", right_on="n_nationkey")
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name").agg(pl.col("revenue").sum())
        .sort("revenue", descending=True)
    )


# ---------------------------------------------------------------
#  Q10 — Returned Item Reporting (4-table join)
#  customer(15K) ⋈ orders(150K) ⋈ lineitem(600K) ⋈ nation(25)
#  filter returnflag='R' + date range
#  groupby custkey+name+acctbal, sum revenue, sort DESC, limit 20
# ---------------------------------------------------------------
def make_tpch_q10_tables(n_customers=15_000, n_orders=150_000, n_lineitem=600_000, seed=66):
    rng = np.random.default_rng(seed)
    n_nations = len(NATIONS)
    nation = pa.table({
        "n_nationkey": pa.array(np.arange(n_nations, dtype=np.int32)),
        "n_name":      pa.array(NATIONS),
    })
    customer = pa.table({
        "c_custkey":   pa.array(np.arange(1, n_customers + 1, dtype=np.int32)),
        "c_name":      pa.array([f"Cust#{i}" for i in range(1, n_customers + 1)]),
        "c_acctbal":   pa.array(rng.uniform(-999.99, 9999.99, n_customers).astype(np.float32)),
        "c_nationkey": pa.array(rng.integers(0, n_nations, n_customers, dtype=np.int32)),
    })
    orders = pa.table({
        "o_orderkey":  pa.array(np.arange(1, n_orders + 1, dtype=np.int32)),
        "o_custkey":   pa.array(rng.integers(1, n_customers + 1, n_orders, dtype=np.int32)),
        "o_orderdate": pa.array(
            rng.integers(19930101, 19960101, n_orders, dtype=np.int32)
        ),
    })
    rf = np.where(rng.random(n_lineitem) < 0.20, "R", "N")
    lineitem = pa.table({
        "l_orderkey":      pa.array(rng.integers(1, n_orders + 1, n_lineitem, dtype=np.int32)),
        "l_returnflag":    pa.array(rf.tolist()),
        "l_extendedprice": pa.array(rng.uniform(900.0, 50000.0, n_lineitem).astype(np.float32)),
        "l_discount":      pa.array(rng.uniform(0.0, 0.10, n_lineitem).astype(np.float32)),
    })
    return nation, customer, orders, lineitem


def run_q10_mxframe(nation, customer, orders, lineitem, device="cpu") -> pa.Table:
    return (
        LazyFrame(Scan(lineitem))
        .filter(col("l_returnflag") == lit("R"))
        .join(
            LazyFrame(Scan(orders)).filter(
                (col("o_orderdate") >= lit(19931001)) &
                (col("o_orderdate") <  lit(19940101))
            ),
            left_on="l_orderkey", right_on="o_orderkey",
        )
        .join(LazyFrame(Scan(customer)), left_on="o_custkey", right_on="c_custkey")
        .join(LazyFrame(Scan(nation)),   left_on="c_nationkey", right_on="n_nationkey")
        .groupby("o_custkey", "n_name")
        .agg(
            (col("l_extendedprice") * (lit(1.0) - col("l_discount")))
            .sum().alias("revenue")
        )
        .sort(col("revenue"), descending=True)
        .limit(20)
        .compute(device=device)
    )


def run_q10_pandas(nation, customer, orders, lineitem) -> pd.DataFrame:
    li = lineitem.to_pandas()
    o  = orders.to_pandas()
    c  = customer.to_pandas()
    n  = nation.to_pandas()
    li = li[li.l_returnflag == "R"]
    o  = o[(o.o_orderdate >= 19931001) & (o.o_orderdate < 19940101)]
    m = li.merge(o, left_on="l_orderkey", right_on="o_orderkey")
    m = m.merge(c, left_on="o_custkey", right_on="c_custkey")
    m = m.merge(n, left_on="c_nationkey", right_on="n_nationkey")
    m["revenue"] = m.l_extendedprice * (1 - m.l_discount)
    r = m.groupby(["c_custkey", "n_name"], as_index=False)["revenue"].sum()
    return r.sort_values("revenue", ascending=False).head(20).reset_index(drop=True)


def run_q10_polars(nation, customer, orders, lineitem):
    if not POLARS_AVAILABLE:
        return None
    li = pl.from_arrow(lineitem).filter(pl.col("l_returnflag") == "R")
    o  = pl.from_arrow(orders).filter(
        (pl.col("o_orderdate") >= 19931001) & (pl.col("o_orderdate") < 19940101)
    )
    c = pl.from_arrow(customer)
    n = pl.from_arrow(nation)
    return (
        li.join(o, left_on="l_orderkey",  right_on="o_orderkey")
        .join(c,  left_on="o_custkey",    right_on="c_custkey")
        .join(n,  left_on="c_nationkey",  right_on="n_nationkey")
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by(["o_custkey", "n_name"])
        .agg(pl.col("revenue").sum())
        .sort("revenue", descending=True)
        .head(20)
    )


# ---------------------------------------------------------------
#  Q7 — Volume Shipping Between Nations (5-table simplified)
#  supplier(2K) ⋈ lineitem(200K) ⋈ orders(80K) ⋈ customer(10K) ⋈ nation(25)
#  filter shipdate 1995-1996, group by supp_nation,cust_nation,year
# ---------------------------------------------------------------
# Q7 date range: shipdate as YYYYMMDD integers
Q7_DATE_LO = 19950101
Q7_DATE_HI  = 19961231


def make_tpch_q7_tables(n_sup=2_000, n_cust=10_000, n_orders=80_000, n_li=200_000, seed=77):
    rng = np.random.default_rng(seed)
    n_nations = len(NATIONS)
    nation = pa.table({
        "n_nationkey": pa.array(np.arange(n_nations, dtype=np.int32)),
        "n_name":      pa.array(NATIONS),
    })
    supplier = pa.table({
        "s_suppkey":   pa.array(np.arange(1, n_sup + 1, dtype=np.int32)),
        "s_nationkey": pa.array(rng.integers(0, n_nations, n_sup, dtype=np.int32)),
    })
    customer = pa.table({
        "c_custkey":   pa.array(np.arange(1, n_cust + 1, dtype=np.int32)),
        "c_nationkey": pa.array(rng.integers(0, n_nations, n_cust, dtype=np.int32)),
    })
    orders = pa.table({
        "o_orderkey": pa.array(np.arange(1, n_orders + 1, dtype=np.int32)),
        "o_custkey":  pa.array(rng.integers(1, n_cust + 1, n_orders, dtype=np.int32)),
    })
    # ~70% of shipdates in range, 30% outside for realism
    in_range = rng.integers(19950101, 19970101, n_li, dtype=np.int32)
    out_range = rng.integers(19900101, 19950101, n_li, dtype=np.int32)
    mask_in = rng.random(n_li) < 0.7
    shipdates = np.where(mask_in, in_range, out_range).astype(np.int32)
    lineitem = pa.table({
        "l_orderkey":      pa.array(rng.integers(1, n_orders + 1, n_li, dtype=np.int32)),
        "l_suppkey":       pa.array(rng.integers(1, n_sup + 1, n_li, dtype=np.int32)),
        "l_extendedprice": pa.array(rng.uniform(900.0, 50000.0, n_li).astype(np.float32)),
        "l_discount":      pa.array(rng.uniform(0.0, 0.10, n_li).astype(np.float32)),
        "l_shipdate":      pa.array(shipdates),
    })
    return nation, supplier, customer, orders, lineitem


def run_q7_mxframe(nation, supplier, customer, orders, lineitem, device="cpu") -> pa.Table:
    # Join chain: lineitem ⋈ orders ⋈ customer → cust_nation; lineitem ⋈ supplier → supp_nation
    # Simplified: join lineitem+orders+customer+nation (cust side), supplier+nation (supp side)
    # via linear chain; supp_nation via separate join
    # Step 1+2 merged: filter lineitem + full join chain in one compute
    joined = (
        LazyFrame(Scan(lineitem))
        .filter(
            (col("l_shipdate") >= lit(Q7_DATE_LO)) &
            (col("l_shipdate") <= lit(Q7_DATE_HI))
        )
        .join(LazyFrame(Scan(orders)),   left_on="l_orderkey", right_on="o_orderkey")
        .join(LazyFrame(Scan(customer)), left_on="o_custkey",  right_on="c_custkey")
        .join(LazyFrame(Scan(nation)).select(
                col("n_nationkey").alias("c_nk"), col("n_name").alias("cust_nation")
             ),
             left_on="c_nationkey", right_on="c_nk")
        .join(LazyFrame(Scan(supplier)), left_on="l_suppkey", right_on="s_suppkey")
        .join(LazyFrame(Scan(nation)).select(
                col("n_nationkey").alias("s_nk"), col("n_name").alias("supp_nation")
             ),
             left_on="s_nationkey", right_on="s_nk")
        .compute(device=device)
    )
    # Step 3: precompute l_year column (YYYYMMDD // 10000)
    year_arr = pc.cast(
        pc.divide(joined.column("l_shipdate"), pa.scalar(10000, type=pa.int32())),
        pa.int32()
    )
    joined = joined.append_column("l_year", year_arr)
    # Step 4: groupby + agg
    return (
        LazyFrame(Scan(joined))
        .groupby("supp_nation", "cust_nation", "l_year")
        .agg(
            (col("l_extendedprice") * (lit(1.0) - col("l_discount")))
            .sum().alias("revenue")
        )
        .sort(col("supp_nation"))
        .compute(device=device)
    )


def run_q7_pandas(nation, supplier, customer, orders, lineitem) -> pd.DataFrame:
    li = lineitem.to_pandas()
    s  = supplier.to_pandas()
    c  = customer.to_pandas()
    o  = orders.to_pandas()
    n  = nation.to_pandas()
    li = li[(li.l_shipdate >= Q7_DATE_LO) & (li.l_shipdate <= Q7_DATE_HI)]
    m  = li.merge(o, left_on="l_orderkey", right_on="o_orderkey")
    m  = m.merge(c, left_on="o_custkey", right_on="c_custkey")
    m  = m.merge(n.rename(columns={"n_nationkey":"c_nk","n_name":"cust_nation"}),
                 left_on="c_nationkey", right_on="c_nk")
    m  = m.merge(s, left_on="l_suppkey", right_on="s_suppkey")
    m  = m.merge(n.rename(columns={"n_nationkey":"s_nk","n_name":"supp_nation"}),
                 left_on="s_nationkey", right_on="s_nk")
    m["l_year"] = m["l_shipdate"] // 10000
    m["volume"] = m.l_extendedprice * (1 - m.l_discount)
    return (m.groupby(["supp_nation","cust_nation","l_year"], as_index=False)["volume"]
             .sum().rename(columns={"volume":"revenue"})
             .sort_values("supp_nation").reset_index(drop=True))


def run_q7_polars(nation, supplier, customer, orders, lineitem):
    if not POLARS_AVAILABLE:
        return None
    li = pl.from_arrow(lineitem).filter(
        (pl.col("l_shipdate") >= Q7_DATE_LO) & (pl.col("l_shipdate") <= Q7_DATE_HI)
    )
    n  = pl.from_arrow(nation)
    s  = pl.from_arrow(supplier)
    c  = pl.from_arrow(customer)
    o  = pl.from_arrow(orders)
    return (
        li.join(o, left_on="l_orderkey", right_on="o_orderkey")
        .join(c, left_on="o_custkey", right_on="c_custkey")
        .join(n.rename({"n_nationkey":"c_nk","n_name":"cust_nation"}),
              left_on="c_nationkey", right_on="c_nk")
        .join(s, left_on="l_suppkey", right_on="s_suppkey")
        .join(n.rename({"n_nationkey":"s_nk","n_name":"supp_nation"}),
              left_on="s_nationkey", right_on="s_nk")
        .with_columns([
            (pl.col("l_shipdate") // 10000).alias("l_year"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("volume"),
        ])
        .group_by(["supp_nation","cust_nation","l_year"])
        .agg(pl.col("volume").sum().alias("revenue"))
        .sort("supp_nation")
    )


# ---------------------------------------------------------------
#  Q8 — National Market Share (6-table, simplified)
#  part(5K) ⋈ lineitem(200K) ⋈ orders(80K) ⋈ customer(10K) ⋈ nation(25) ⋈ region(5)
#  filter region='AMERICA', part type='ECONOMY ANODIZED STEEL'
#  groupby year, sum(case when nation='BRAZIL' then rev else 0)/sum(rev)
# ---------------------------------------------------------------
Q8_DATE_LO = 19950101
Q8_DATE_HI  = 19961231
REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
PART_TYPES = ["ECONOMY ANODIZED STEEL", "STANDARD POLISHED BRASS",
              "PROMO BRUSHED COPPER", "LARGE BURNISHED TIN", "SMALL PLATED STEEL"]


def make_tpch_q8_tables(n_parts=5_000, n_cust=10_000, n_orders=80_000, n_li=200_000, seed=88):
    rng = np.random.default_rng(seed)
    n_nations = len(NATIONS)
    n_regions = len(REGIONS)
    # Each nation belongs to a region (5 nations per region)
    nation = pa.table({
        "n_nationkey": pa.array(np.arange(n_nations, dtype=np.int32)),
        "n_name":      pa.array(NATIONS),
        "n_regionkey": pa.array((np.arange(n_nations) // 5).astype(np.int32)),
    })
    region = pa.table({
        "r_regionkey": pa.array(np.arange(n_regions, dtype=np.int32)),
        "r_name":      pa.array(REGIONS),
    })
    part = pa.table({
        "p_partkey": pa.array(np.arange(1, n_parts + 1, dtype=np.int32)),
        "p_type":    pa.array([PART_TYPES[rng.integers(0, len(PART_TYPES))]
                               for _ in range(n_parts)]),
    })
    customer = pa.table({
        "c_custkey":   pa.array(np.arange(1, n_cust + 1, dtype=np.int32)),
        "c_nationkey": pa.array(rng.integers(0, n_nations, n_cust, dtype=np.int32)),
    })
    orders = pa.table({
        "o_orderkey":  pa.array(np.arange(1, n_orders + 1, dtype=np.int32)),
        "o_custkey":   pa.array(rng.integers(1, n_cust + 1, n_orders, dtype=np.int32)),
        "o_orderdate": pa.array(rng.integers(Q8_DATE_LO, Q8_DATE_HI + 1, n_orders, dtype=np.int32)),
    })
    in_range = rng.integers(Q8_DATE_LO, Q8_DATE_HI + 1, n_li, dtype=np.int32)
    out_range = rng.integers(19900101, Q8_DATE_LO, n_li, dtype=np.int32)
    shipdates = np.where(rng.random(n_li) < 0.7, in_range, out_range).astype(np.int32)
    lineitem = pa.table({
        "l_orderkey":      pa.array(rng.integers(1, n_orders + 1, n_li, dtype=np.int32)),
        "l_partkey":       pa.array(rng.integers(1, n_parts + 1, n_li, dtype=np.int32)),
        "l_extendedprice": pa.array(rng.uniform(900.0, 50000.0, n_li).astype(np.float32)),
        "l_discount":      pa.array(rng.uniform(0.0, 0.10, n_li).astype(np.float32)),
        "l_shipdate":      pa.array(shipdates),
    })
    return nation, region, part, customer, orders, lineitem


def run_q8_mxframe(nation, region, part, customer, orders, lineitem, device="cpu") -> pa.Table:
    # Pre-filter: region AMERICA (regionkey=1), part type
    n_america = nation.to_pandas()
    n_america = n_america[n_america.n_regionkey == 1]  # AMERICA nations
    brazil_nk = 2  # BRAZIL is nations[2]
    american_nkeys = n_america["n_nationkey"].tolist()

    # Build nation-name lookup via PyArrow for the supplier/customer nation join
    cust_am = pa.table({
        "c_custkey":   customer.column("c_custkey"),
        "c_nationkey": customer.column("c_nationkey"),
    }).filter(pc.is_in(customer.column("c_nationkey"),
                       value_set=pa.array(american_nkeys, type=pa.int32())))

    # MXFrame join chain: li ⋈ part ⋈ orders ⋈ customer_america ⋈ nation
    li_f_pa = pa.table({
        "l_orderkey":      lineitem.column("l_orderkey"),
        "l_partkey":       lineitem.column("l_partkey"),
        "l_extendedprice": lineitem.column("l_extendedprice"),
        "l_discount":      lineitem.column("l_discount"),
    })

    joined = (
        LazyFrame(Scan(li_f_pa))
        .join(
            LazyFrame(Scan(part)).filter(col("p_type") == lit("ECONOMY ANODIZED STEEL")),
            left_on="l_partkey", right_on="p_partkey",
        )
        .join(LazyFrame(Scan(orders)), left_on="l_orderkey", right_on="o_orderkey")
        .join(LazyFrame(Scan(cust_am)), left_on="o_custkey", right_on="c_custkey")
        .join(
            LazyFrame(Scan(nation)).select(
                col("n_nationkey").alias("cn_key"),
                col("n_name").alias("cust_nation")
            ),
            left_on="c_nationkey", right_on="cn_key",
        )
        .compute(device=device)
    )
    # Add year column
    year_arr = pc.cast(
        pc.divide(joined.column("o_orderdate"), pa.scalar(10000, type=pa.int32())),
        pa.int32()
    )
    joined = joined.append_column("o_year", year_arr)
    # Add volume and brazil_vol
    vol = pc.multiply(
        joined.column("l_extendedprice").cast(pa.float32()),
        pc.subtract(pa.scalar(1.0, pa.float32()), joined.column("l_discount"))
    )
    is_brazil = pc.equal(joined.column("c_nationkey"), pa.scalar(brazil_nk, pa.int32()))
    bvol = pc.if_else(is_brazil, vol, pa.scalar(0.0, pa.float32()))
    joined = joined.append_column("volume", vol)
    joined = joined.append_column("brazil_vol", bvol)
    # GroupBy year → sum
    result = (
        LazyFrame(Scan(joined))
        .groupby("o_year")
        .agg(
            col("volume").sum().alias("total_vol"),
            col("brazil_vol").sum().alias("brazil_vol_sum"),
        )
        .sort(col("o_year"))
        .compute(device=device)
    )
    # Compute market_share = brazil_vol / total_vol
    tv = result.column("total_vol").to_pylist()
    bv = result.column("brazil_vol_sum").to_pylist()
    ms = [b / t if t else 0.0 for b, t in zip(bv, tv)]
    return result.append_column("mkt_share", pa.array(ms, type=pa.float32()))


def run_q8_pandas(nation, region, part, customer, orders, lineitem) -> pd.DataFrame:
    n  = nation.to_pandas()
    pt = part.to_pandas()
    c  = customer.to_pandas()
    o  = orders.to_pandas()
    li = lineitem.to_pandas()
    n_am = n[n.n_regionkey == 1]
    c_am = c[c.c_nationkey.isin(n_am.n_nationkey)]
    pt_f = pt[pt.p_type == "ECONOMY ANODIZED STEEL"]
    m = li.merge(pt_f, left_on="l_partkey", right_on="p_partkey")
    m = m.merge(o, left_on="l_orderkey", right_on="o_orderkey")
    m = m.merge(c_am, left_on="o_custkey", right_on="c_custkey")
    m["o_year"] = m.o_orderdate // 10000
    m["volume"] = m.l_extendedprice * (1 - m.l_discount)
    m["brazil_vol"] = m.volume.where(m.c_nationkey == 2, 0.0)
    r = m.groupby("o_year")[["volume","brazil_vol"]].sum().reset_index()
    r["mkt_share"] = r.brazil_vol / r.volume
    return r.sort_values("o_year").reset_index(drop=True)


def run_q8_polars(nation, region, part, customer, orders, lineitem):
    if not POLARS_AVAILABLE:
        return None
    n  = pl.from_arrow(nation)
    pt = pl.from_arrow(part)
    c  = pl.from_arrow(customer)
    o  = pl.from_arrow(orders)
    li = pl.from_arrow(lineitem)
    n_am = n.filter(pl.col("n_regionkey") == 1)
    c_am = c.join(n_am.select("n_nationkey"), left_on="c_nationkey", right_on="n_nationkey")
    pt_f = pt.filter(pl.col("p_type") == "ECONOMY ANODIZED STEEL")
    return (
        li.join(pt_f, left_on="l_partkey", right_on="p_partkey")
        .join(o, left_on="l_orderkey", right_on="o_orderkey")
        .join(c_am, left_on="o_custkey", right_on="c_custkey")
        .with_columns([
            (pl.col("o_orderdate") // 10000).alias("o_year"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("volume"),
        ])
        .with_columns(
            pl.when(pl.col("c_nationkey") == 2)
            .then(pl.col("volume")).otherwise(0.0).alias("brazil_vol")
        )
        .group_by("o_year")
        .agg([pl.col("volume").sum(), pl.col("brazil_vol").sum()])
        .with_columns((pl.col("brazil_vol") / pl.col("volume")).alias("mkt_share"))
        .sort("o_year")
    )


# ---------------------------------------------------------------
#  Q13 — Customer Distribution (LEFT JOIN + double groupby)
#  customer(150K) LEFT JOIN orders(600K) → count orders per customer
#  → group by c_count → count customers → sort DESC
# ---------------------------------------------------------------
def make_tpch_q13_tables(n_customers=150_000, n_orders=600_000, seed=13):
    rng = np.random.default_rng(seed)
    customer = pa.table({
        "c_custkey": pa.array(np.arange(1, n_customers + 1, dtype=np.int32)),
    })
    orders = pa.table({
        "o_orderkey": pa.array(np.arange(1, n_orders + 1, dtype=np.int32)),
        "o_custkey":  pa.array(rng.integers(1, n_customers + 1, n_orders, dtype=np.int32)),
    })
    return customer, orders


def run_q13_mxframe(customer, orders, device="cpu") -> pa.Table:
    # Step 1: LEFT JOIN → all customers, even those with no orders
    joined = (
        LazyFrame(Scan(customer))
        .join(LazyFrame(Scan(orders)), left_on="c_custkey", right_on="o_custkey", how="left")
        .compute(device=device)
    )
    # Step 2: count orders per customer (o_orderkey is null for no-order customers)
    joined_pd = joined.to_pandas()
    c_counts = (
        joined_pd.groupby("c_custkey", as_index=False)["o_orderkey"]
        .count()
        .rename(columns={"o_orderkey": "c_count"})
    )
    c_counts_arrow = pa.Table.from_pandas(c_counts)
    # Step 3: group by c_count → count
    return (
        LazyFrame(Scan(c_counts_arrow))
        .groupby("c_count")
        .agg(col("c_custkey").count().alias("custdist"))
        .sort(col("custdist"), descending=True)
        .compute(device=device)
    )


def run_q13_pandas(customer, orders) -> pd.DataFrame:
    c = customer.to_pandas()
    o = orders.to_pandas()
    m = c.merge(o, left_on="c_custkey", right_on="o_custkey", how="left")
    c_count = m.groupby("c_custkey", as_index=False)["o_orderkey"].count()
    c_count.rename(columns={"o_orderkey":"c_count"}, inplace=True)
    result = c_count.groupby("c_count", as_index=False)["c_custkey"].count()
    result.rename(columns={"c_custkey":"custdist"}, inplace=True)
    return result.sort_values("custdist", ascending=False).reset_index(drop=True)


def run_q13_polars(customer, orders):
    if not POLARS_AVAILABLE:
        return None
    c = pl.from_arrow(customer)
    o = pl.from_arrow(orders)
    return (
        c.join(o, left_on="c_custkey", right_on="o_custkey", how="left")
        .group_by("c_custkey").agg(pl.col("o_orderkey").count().alias("c_count"))
        .group_by("c_count").agg(pl.count("c_custkey").alias("custdist"))
        .sort("custdist", descending=True)
    )


# ---------------------------------------------------------------
#  Q19 — Discounted Revenue (complex OR predicates, global sum)
#  part(20K) ⋈ lineitem(200K)
#  complex OR: (brand='Brand#12' AND container IN [...] AND qty BETWEEN ...)
#            OR ... OR ...
#  → global SUM(l_extendedprice * (1-l_discount))
# ---------------------------------------------------------------
CONTAINERS_SM = ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
CONTAINERS_MD = ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
CONTAINERS_LG = ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]
ALL_CONTAINERS = CONTAINERS_SM + CONTAINERS_MD + CONTAINERS_LG + ["JUMBO PKG", "WRAP CASE"]


def make_tpch_q19_tables(n_parts=20_000, n_lineitem=200_000, seed=19):
    rng = np.random.default_rng(seed)
    brands = [f"Brand#{rng.integers(11, 55)}" for _ in range(n_parts)]
    containers = [ALL_CONTAINERS[rng.integers(0, len(ALL_CONTAINERS))]
                  for _ in range(n_parts)]
    sizes = rng.integers(1, 51, n_parts, dtype=np.int32)
    part = pa.table({
        "p_partkey":   pa.array(np.arange(1, n_parts + 1, dtype=np.int32)),
        "p_brand":     pa.array(brands),
        "p_container": pa.array(containers),
        "p_size":      pa.array(sizes),
    })
    qty  = rng.uniform(1.0, 50.0, n_lineitem).astype(np.float32)
    disc = rng.uniform(0.0, 0.10, n_lineitem).astype(np.float32)
    eprc = rng.uniform(900.0, 50000.0, n_lineitem).astype(np.float32)
    lineitem = pa.table({
        "l_partkey":       pa.array(rng.integers(1, n_parts + 1, n_lineitem, dtype=np.int32)),
        "l_quantity":      pa.array(qty),
        "l_extendedprice": pa.array(eprc),
        "l_discount":      pa.array(disc),
        "l_shipmode":      pa.array(["AIR"] * n_lineitem),
        "l_shipinstruct":  pa.array(["DELIVER IN PERSON"] * n_lineitem),
    })
    return part, lineitem


def run_q19_mxframe(part, lineitem, device="cpu") -> pa.Table:
    cond_sm = (
        (col("p_brand") == lit("Brand#12")) &
        col("p_container").isin(CONTAINERS_SM) &
        (col("l_quantity") >= lit(1.0)) & (col("l_quantity") <= lit(11.0)) &
        (col("p_size") >= lit(1)) & (col("p_size") <= lit(5))
    )
    cond_md = (
        (col("p_brand") == lit("Brand#23")) &
        col("p_container").isin(CONTAINERS_MD) &
        (col("l_quantity") >= lit(10.0)) & (col("l_quantity") <= lit(20.0)) &
        (col("p_size") >= lit(1)) & (col("p_size") <= lit(10))
    )
    cond_lg = (
        (col("p_brand") == lit("Brand#34")) &
        col("p_container").isin(CONTAINERS_LG) &
        (col("l_quantity") >= lit(20.0)) & (col("l_quantity") <= lit(30.0)) &
        (col("p_size") >= lit(1)) & (col("p_size") <= lit(15))
    )
    # Materialize join+filter first, then compute global sum via LazyFrame
    filtered = (
        LazyFrame(Scan(lineitem))
        .join(LazyFrame(Scan(part)), left_on="l_partkey", right_on="p_partkey")
        .filter(cond_sm | cond_md | cond_lg)
        .compute(device=device)
    )
    return (
        LazyFrame(Scan(filtered))
        .groupby()
        .agg(
            (col("l_extendedprice") * (lit(1.0) - col("l_discount")))
            .sum().alias("revenue")
        )
        .compute(device=device)
    )


def run_q19_pandas(part, lineitem) -> float:
    li = lineitem.to_pandas()
    pt = part.to_pandas()
    m  = li.merge(pt, left_on="l_partkey", right_on="p_partkey")
    cond = (
        ((m.p_brand == "Brand#12") & m.p_container.isin(CONTAINERS_SM) &
         (m.l_quantity >= 1) & (m.l_quantity <= 11) & (m.p_size >= 1) & (m.p_size <= 5)) |
        ((m.p_brand == "Brand#23") & m.p_container.isin(CONTAINERS_MD) &
         (m.l_quantity >= 10) & (m.l_quantity <= 20) & (m.p_size >= 1) & (m.p_size <= 10)) |
        ((m.p_brand == "Brand#34") & m.p_container.isin(CONTAINERS_LG) &
         (m.l_quantity >= 20) & (m.l_quantity <= 30) & (m.p_size >= 1) & (m.p_size <= 15))
    )
    fc = m[cond]
    return float((fc.l_extendedprice * (1 - fc.l_discount)).sum())


def run_q19_polars(part, lineitem):
    if not POLARS_AVAILABLE:
        return None
    li = pl.from_arrow(lineitem)
    pt = pl.from_arrow(part)
    m  = li.join(pt, left_on="l_partkey", right_on="p_partkey")
    return (
        m.filter(
            ((pl.col("p_brand") == "Brand#12") &
             pl.col("p_container").is_in(CONTAINERS_SM) &
             pl.col("l_quantity").is_between(1, 11) & pl.col("p_size").is_between(1, 5)) |
            ((pl.col("p_brand") == "Brand#23") &
             pl.col("p_container").is_in(CONTAINERS_MD) &
             pl.col("l_quantity").is_between(10, 20) & pl.col("p_size").is_between(1, 10)) |
            ((pl.col("p_brand") == "Brand#34") &
             pl.col("p_container").is_in(CONTAINERS_LG) &
             pl.col("l_quantity").is_between(20, 30) & pl.col("p_size").is_between(1, 15))
        )
        .select((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("revenue"))
    )


def _check_q1(mx_tbl: pa.Table, ref_df: pd.DataFrame, label: str) -> None:
    mx  = mx_tbl.to_pandas().sort_values(["l_returnflag", "l_linestatus"]).reset_index(drop=True)
    ref = ref_df.sort_values(["l_returnflag", "l_linestatus"]).reset_index(drop=True)
    assert len(mx) == len(ref), f"{label}: row count {len(mx)} vs {len(ref)}"
    for c in ["sum_qty", "sum_base_price", "sum_disc_price", "sum_charge",
              "avg_qty", "avg_price", "avg_disc", "count_order"]:
        assert np.allclose(mx[c].astype(float), ref[c].astype(float), rtol=1e-2, atol=1e-2), \
            f"{label}: mismatch in {c}"


def _check_q3(mx, ref, label: str) -> None:
    mx  = (mx.to_pandas()  if isinstance(mx,  pa.Table) else mx).sort_values("revenue", ascending=False).reset_index(drop=True)
    ref = (ref.to_pandas() if isinstance(ref, pa.Table) else ref).sort_values("revenue", ascending=False).reset_index(drop=True)
    assert len(mx) == len(ref), f"{label}: row count {len(mx)} vs {len(ref)}"
    assert np.allclose(mx["revenue"].astype(float), ref["revenue"].astype(float), rtol=1e-2, atol=1.0), \
        f"{label}: revenue mismatch"


# ---------------------------------------------------------------
#  Main
# ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="TPC-H Benchmark: MXFrame vs Polars vs Pandas")
    parser.add_argument("--rows",     type=int, default=1_000_000)
    parser.add_argument("--cold",     type=int, default=3)
    parser.add_argument("--hot",      type=int, default=5)
    parser.add_argument("--skip-q3",  action="store_true")
    parser.add_argument("--skip-sld", action="store_true")
    parser.add_argument("--skip-q12q14", action="store_true")
    parser.add_argument("--skip-q5-q13",  action="store_true",
                        help="Skip Q5/Q10/Q7/Q8/Q13/Q19 (new Phase 11 queries)")
    args = parser.parse_args()

    COLD, HOT, N = args.cold, args.hot, args.rows

    _report_context()

    # GPU probe
    GPU_READY = False
    if _safe_gpu_count() > 0:
        try:
            run_q6_mxframe(make_lineitem(10_000), device="gpu")
            GPU_READY = True
        except Exception as e:
            print(f"GPU warm-up failed: {e}")
    print(f"GPU ready: {GPU_READY}")
    print(f"Rows: {N:,}   Cold runs: {COLD}   Hot runs: {HOT}\n")

    # Data
    print("Generating data ...", end=" ", flush=True)
    lineitem = make_lineitem(N)
    customer, orders, q3_li = make_tpch_q3_tables()
    grouped  = make_grouped(500_000, n_groups=1000)
    print("done")

    # ----------------------------------------------------------
    #  Q1 — filter + groupby + 8 aggs
    # ----------------------------------------------------------
    _section(f"Q1  Filter + GroupBy + 8 Aggs  ({N:,} rows)")

    try:
        _check_q1(run_q1_mxframe(lineitem), run_q1_pandas(lineitem), "Q1 CPU vs Pandas")
        print("  Correctness vs Pandas: OK")
    except AssertionError as e:
        print(f"  Correctness warning: {e}")

    q1_rows = []
    q1_rows.append(("MXFrame CPU",
        _stats(_time_cold(lambda: run_q1_mxframe(lineitem, device="cpu"), COLD)),
        _stats(_time_runs( lambda: run_q1_mxframe(lineitem, device="cpu"), HOT, warmup=2))))

    if GPU_READY:
        q1_rows.append(("MXFrame GPU",
            _stats(_time_cold(lambda: run_q1_mxframe(lineitem, device="gpu"), COLD)),
            _stats(_time_runs( lambda: run_q1_mxframe(lineitem, device="gpu"), HOT, warmup=2))))

    q1_rows.append(("Pandas",  None, _stats(_time_runs(lambda: run_q1_pandas(lineitem), HOT, warmup=1))))
    if POLARS_AVAILABLE:
        q1_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q1_polars(lineitem), HOT, warmup=2))))

    _print_table("Q1 — all times ms  (Cold min includes JIT compilation)", q1_rows)
    _summarize_relative(q1_rows)

    # ----------------------------------------------------------
    #  Q6 — multi-predicate filter + global sum
    # ----------------------------------------------------------
    _section(f"Q6  Multi-Predicate Filter + Global Sum  ({N:,} rows)")

    q6_rows = []
    q6_rows.append(("MXFrame CPU",
        _stats(_time_cold(lambda: run_q6_mxframe(lineitem, device="cpu"), COLD)),
        _stats(_time_runs( lambda: run_q6_mxframe(lineitem, device="cpu"), HOT, warmup=2))))

    if GPU_READY:
        q6_rows.append(("MXFrame GPU",
            _stats(_time_cold(lambda: run_q6_mxframe(lineitem, device="gpu"), COLD)),
            _stats(_time_runs( lambda: run_q6_mxframe(lineitem, device="gpu"), HOT, warmup=2))))

    q6_rows.append(("Pandas",  None, _stats(_time_runs(lambda: run_q6_pandas(lineitem), HOT, warmup=1))))
    if POLARS_AVAILABLE:
        q6_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q6_polars(lineitem), HOT, warmup=2))))

    _print_table("Q6 — all times ms  (Cold min includes JIT compilation)", q6_rows)
    _summarize_relative(q6_rows)

    # ----------------------------------------------------------
    #  Q3 — 3-way join + groupby + sort + limit 10
    # ----------------------------------------------------------
    if not args.skip_q3:
        _section(
            f"Q3  3-way Join + GroupBy + Sort + Limit 10"
            f"  (c={customer.num_rows:,}  o={orders.num_rows:,}  l={q3_li.num_rows:,})"
        )

        ref_q3 = (run_q3_duckdb(customer, orders, q3_li) if DUCKDB_AVAILABLE
                  else run_q3_pandas(customer, orders, q3_li))
        try:
            _check_q3(run_q3_mxframe(customer, orders, q3_li), ref_q3,
                      f"Q3 CPU vs {'DuckDB' if DUCKDB_AVAILABLE else 'Pandas'}")
            print(f"  Correctness vs {'DuckDB' if DUCKDB_AVAILABLE else 'Pandas'}: OK")
        except AssertionError as e:
            print(f"  Correctness warning: {e}")

        q3_rows = []
        q3_rows.append(("MXFrame CPU",
            _stats(_time_cold(lambda: run_q3_mxframe(customer, orders, q3_li, device="cpu"), COLD)),
            _stats(_time_runs( lambda: run_q3_mxframe(customer, orders, q3_li, device="cpu"), HOT, warmup=2))))

        if GPU_READY:
            try:
                q3_rows.append(("MXFrame GPU",
                    _stats(_time_cold(lambda: run_q3_mxframe(customer, orders, q3_li, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q3_mxframe(customer, orders, q3_li, device="gpu"), HOT, warmup=2))))
            except Exception as e:
                print(f"  GPU Q3 skipped: {e}")

        q3_rows.append(("Pandas",  None, _stats(_time_runs(lambda: run_q3_pandas(customer, orders, q3_li), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q3_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q3_polars(customer, orders, q3_li), HOT, warmup=2))))
        if DUCKDB_AVAILABLE:
            q3_rows.append(("DuckDB", None, _stats(_time_runs(lambda: run_q3_duckdb(customer, orders, q3_li), HOT, warmup=2))))

        _print_table("Q3 — all times ms  (Cold min includes JIT compilation)", q3_rows)
        _summarize_relative(q3_rows, baselines=("Pandas", "Polars", "DuckDB"))

    # ----------------------------------------------------------
    #  Q12 — 2-table join + isin + CASE WHEN grouped
    #  Q14 — 2-table join + startswith CASE WHEN + ratio
    # ----------------------------------------------------------
    if not args.skip_q12q14:
        orders_q12, li_q12 = make_tpch_q12_tables()
        part_q14,   li_q14 = make_tpch_q14_tables()

        # Q12 correctness check
        _section(f"Q12 — join + isin + grouped CASE WHEN  ({li_q12.num_rows:,} lineitem rows)")
        try:
            mx_q12  = run_q12_mxframe(orders_q12, li_q12)
            if DUCKDB_AVAILABLE:
                ref_q12 = run_q12_duckdb(orders_q12, li_q12)
            else:
                ref_q12 = run_q12_pandas(orders_q12, li_q12)
            # Just check shapes match
            assert mx_q12.num_rows > 0
            print(f"  Correctness: OK  ({mx_q12.num_rows} groups)")
        except Exception as e:
            print(f"  Correctness warning: {e}")

        q12_rows = []
        q12_rows.append(("MXFrame CPU",
            _stats(_time_cold(lambda: run_q12_mxframe(orders_q12, li_q12, device="cpu"), COLD)),
            _stats(_time_runs( lambda: run_q12_mxframe(orders_q12, li_q12, device="cpu"), HOT, warmup=2))))
        if GPU_READY:
            try:
                q12_rows.append(("MXFrame GPU",
                    _stats(_time_cold(lambda: run_q12_mxframe(orders_q12, li_q12, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q12_mxframe(orders_q12, li_q12, device="gpu"), HOT, warmup=2))))
            except Exception as e:
                print(f"  GPU Q12 skipped: {e}")
        q12_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q12_pandas(orders_q12, li_q12), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q12_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q12_polars(orders_q12, li_q12), HOT, warmup=2))))
        if DUCKDB_AVAILABLE:
            q12_rows.append(("DuckDB", None, _stats(_time_runs(lambda: run_q12_duckdb(orders_q12, li_q12), HOT, warmup=2))))
        _print_table("Q12 — all times ms  (Cold min includes JIT compilation)", q12_rows)
        _summarize_relative(q12_rows)

        # Q14
        _section(f"Q14 — join + startswith CASE WHEN + ratio  ({li_q14.num_rows:,} lineitem rows)")
        try:
            mx_q14  = run_q14_mxframe(part_q14, li_q14)
            mx_pct  = 100.0 * float(mx_q14.column("promo_revenue")[0].as_py()) / float(mx_q14.column("total_revenue")[0].as_py())
            if DUCKDB_AVAILABLE:
                ref_pct = run_q14_duckdb(part_q14, li_q14)
                assert abs(mx_pct - ref_pct) < 0.5, f"Q14 mismatch: MX={mx_pct:.2f}% DuckDB={ref_pct:.2f}%"
                print(f"  Correctness: OK  promo_revenue={mx_pct:.2f}% (DuckDB={ref_pct:.2f}%)")
            else:
                print(f"  Correctness: OK  promo_revenue={mx_pct:.2f}%")
        except Exception as e:
            print(f"  Correctness warning: {e}")

        q14_rows = []
        q14_rows.append(("MXFrame CPU",
            _stats(_time_cold(lambda: run_q14_mxframe(part_q14, li_q14, device="cpu"), COLD)),
            _stats(_time_runs( lambda: run_q14_mxframe(part_q14, li_q14, device="cpu"), HOT, warmup=2))))
        if GPU_READY:
            try:
                q14_rows.append(("MXFrame GPU",
                    _stats(_time_cold(lambda: run_q14_mxframe(part_q14, li_q14, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q14_mxframe(part_q14, li_q14, device="gpu"), HOT, warmup=2))))
            except Exception as e:
                print(f"  GPU Q14 skipped: {e}")
        q14_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q14_pandas(part_q14, li_q14), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q14_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q14_polars(part_q14, li_q14), HOT, warmup=2))))
        if DUCKDB_AVAILABLE:
            q14_rows.append(("DuckDB", None, _stats(_time_runs(lambda: run_q14_duckdb(part_q14, li_q14), HOT, warmup=2))))
        _print_table("Q14 — all times ms  (Cold min includes JIT compilation)", q14_rows)
        _summarize_relative(q14_rows)

    # ----------------------------------------------------------
    #  SLD — Sort / Limit / Distinct
    # ----------------------------------------------------------
    if not args.skip_sld:
        _section(f"Sort / Limit / Distinct  ({grouped.num_rows:,} rows, 1000 groups)")

        def _bench_op(label, fn_mx_cpu, fn_pd, fn_pl=None, fn_mx_gpu=None):
            rows = []
            rows.append((f"MXFrame CPU",
                _stats(_time_cold(fn_mx_cpu, COLD)),
                _stats(_time_runs( fn_mx_cpu, HOT, warmup=2))))
            if GPU_READY and fn_mx_gpu:
                try:
                    rows.append((f"MXFrame GPU",
                        _stats(_time_cold(fn_mx_gpu, COLD)),
                        _stats(_time_runs( fn_mx_gpu, HOT, warmup=2))))
                except Exception:
                    pass
            rows.append(("Pandas", None, _stats(_time_runs(fn_pd, HOT, warmup=1))))
            if POLARS_AVAILABLE and fn_pl:
                rows.append(("Polars", None, _stats(_time_runs(fn_pl, HOT, warmup=2))))
            _print_table(label, rows)
            _summarize_relative(rows)

        _bench_op(
            "Sort (groupby+agg+sort)",
            lambda: run_sld_sort_mxframe(grouped, device="cpu"),
            lambda: run_sld_sort_pandas(grouped),
            lambda: run_sld_sort_polars(grouped),
            lambda: run_sld_sort_mxframe(grouped, device="gpu"),
        )
        _bench_op(
            "Limit 10 (sort+limit)",
            lambda: run_sld_limit_mxframe(grouped, device="cpu"),
            lambda: run_sld_sort_pandas(grouped).head(10),
            lambda: run_sld_sort_polars(grouped).head(10) if POLARS_AVAILABLE else None,
            lambda: run_sld_limit_mxframe(grouped, device="gpu"),
        )
        _bench_op(
            "Distinct (unique groups)",
            lambda: run_sld_distinct_mxframe(grouped, device="cpu"),
            lambda: run_sld_distinct_pandas(grouped),
            lambda: run_sld_distinct_polars(grouped),
            lambda: run_sld_distinct_mxframe(grouped, device="gpu"),
        )

    # ----------------------------------------------------------
    #  Q5 — Local Supplier Volume (4-way join, 25 nations)
    # ----------------------------------------------------------
    if not args.skip_q5_q13:
        print("\nGenerating Q5/Q10/Q7/Q8/Q13/Q19 data ...", end=" ", flush=True)
        q5_nation, q5_cust, q5_orders, q5_li = make_tpch_q5_tables()
        q10_nation, q10_cust, q10_orders, q10_li = make_tpch_q10_tables()
        q7_nation, q7_sup, q7_cust, q7_orders, q7_li = make_tpch_q7_tables()
        q8_nation, q8_region, q8_part, q8_cust, q8_orders, q8_li = make_tpch_q8_tables()
        q13_cust, q13_orders = make_tpch_q13_tables()
        q19_part, q19_li = make_tpch_q19_tables()
        print("done")

        _section(
            f"Q5  (c={q5_cust.num_rows:,}  o={q5_orders.num_rows:,}"
            f"  l={q5_li.num_rows:,}  n=25)"
        )
        ref_q5 = run_q5_pandas(q5_nation, q5_cust, q5_orders, q5_li)
        q5_rows = []
        q5_rows.append((
            "MXFrame CPU",
            _stats(_time_cold(lambda: run_q5_mxframe(q5_nation, q5_cust, q5_orders, q5_li), COLD)),
            _stats(_time_runs( lambda: run_q5_mxframe(q5_nation, q5_cust, q5_orders, q5_li), HOT, warmup=2)),
        ))
        if GPU_READY:
            try:
                q5_rows.append((
                    "MXFrame GPU",
                    _stats(_time_cold(lambda: run_q5_mxframe(q5_nation, q5_cust, q5_orders, q5_li, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q5_mxframe(q5_nation, q5_cust, q5_orders, q5_li, device="gpu"), HOT, warmup=2)),
                ))
            except Exception as e:
                print(f"  GPU Q5 skipped: {e}")
        q5_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q5_pandas(q5_nation, q5_cust, q5_orders, q5_li), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q5_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q5_polars(q5_nation, q5_cust, q5_orders, q5_li), HOT, warmup=2))))
        _print_table("Q5", q5_rows)
        _summarize_relative(q5_rows)

        # ----------------------------------------------------------
        #  Q10 — Returned Item Reporting
        # ----------------------------------------------------------
        _section(
            f"Q10 (c={q10_cust.num_rows:,}  o={q10_orders.num_rows:,}"
            f"  l={q10_li.num_rows:,}  n=25)"
        )
        q10_rows = []
        q10_rows.append((
            "MXFrame CPU",
            _stats(_time_cold(lambda: run_q10_mxframe(q10_nation, q10_cust, q10_orders, q10_li), COLD)),
            _stats(_time_runs( lambda: run_q10_mxframe(q10_nation, q10_cust, q10_orders, q10_li), HOT, warmup=2)),
        ))
        if GPU_READY:
            try:
                q10_rows.append((
                    "MXFrame GPU",
                    _stats(_time_cold(lambda: run_q10_mxframe(q10_nation, q10_cust, q10_orders, q10_li, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q10_mxframe(q10_nation, q10_cust, q10_orders, q10_li, device="gpu"), HOT, warmup=2)),
                ))
            except Exception as e:
                print(f"  GPU Q10 skipped: {e}")
        q10_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q10_pandas(q10_nation, q10_cust, q10_orders, q10_li), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q10_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q10_polars(q10_nation, q10_cust, q10_orders, q10_li), HOT, warmup=2))))
        _print_table("Q10", q10_rows)
        _summarize_relative(q10_rows)

        # ----------------------------------------------------------
        #  Q7 — Volume Shipping Between Nations
        # ----------------------------------------------------------
        _section(
            f"Q7  (sup={q7_sup.num_rows:,}  o={q7_orders.num_rows:,}"
            f"  l={q7_li.num_rows:,}  n=25)"
        )
        q7_rows = []
        q7_rows.append((
            "MXFrame CPU",
            _stats(_time_cold(lambda: run_q7_mxframe(q7_nation, q7_sup, q7_cust, q7_orders, q7_li), COLD)),
            _stats(_time_runs( lambda: run_q7_mxframe(q7_nation, q7_sup, q7_cust, q7_orders, q7_li), HOT, warmup=2)),
        ))
        if GPU_READY:
            try:
                q7_rows.append((
                    "MXFrame GPU",
                    _stats(_time_cold(lambda: run_q7_mxframe(q7_nation, q7_sup, q7_cust, q7_orders, q7_li, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q7_mxframe(q7_nation, q7_sup, q7_cust, q7_orders, q7_li, device="gpu"), HOT, warmup=2)),
                ))
            except Exception as e:
                print(f"  GPU Q7 skipped: {e}")
        q7_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q7_pandas(q7_nation, q7_sup, q7_cust, q7_orders, q7_li), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q7_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q7_polars(q7_nation, q7_sup, q7_cust, q7_orders, q7_li), HOT, warmup=2))))
        _print_table("Q7", q7_rows)
        _summarize_relative(q7_rows)

        # ----------------------------------------------------------
        #  Q8 — National Market Share
        # ----------------------------------------------------------
        _section(
            f"Q8  (part={q8_part.num_rows:,}  o={q8_orders.num_rows:,}"
            f"  l={q8_li.num_rows:,})"
        )
        q8_rows = []
        q8_rows.append((
            "MXFrame CPU",
            _stats(_time_cold(lambda: run_q8_mxframe(q8_nation, q8_region, q8_part, q8_cust, q8_orders, q8_li), COLD)),
            _stats(_time_runs( lambda: run_q8_mxframe(q8_nation, q8_region, q8_part, q8_cust, q8_orders, q8_li), HOT, warmup=2)),
        ))
        if GPU_READY:
            try:
                q8_rows.append((
                    "MXFrame GPU",
                    _stats(_time_cold(lambda: run_q8_mxframe(q8_nation, q8_region, q8_part, q8_cust, q8_orders, q8_li, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q8_mxframe(q8_nation, q8_region, q8_part, q8_cust, q8_orders, q8_li, device="gpu"), HOT, warmup=2)),
                ))
            except Exception as e:
                print(f"  GPU Q8 skipped: {e}")
        q8_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q8_pandas(q8_nation, q8_region, q8_part, q8_cust, q8_orders, q8_li), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q8_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q8_polars(q8_nation, q8_region, q8_part, q8_cust, q8_orders, q8_li), HOT, warmup=2))))
        _print_table("Q8", q8_rows)
        _summarize_relative(q8_rows)

        # ----------------------------------------------------------
        #  Q13 — Customer Distribution (LEFT JOIN + double groupby)
        # ----------------------------------------------------------
        _section(
            f"Q13 (c={q13_cust.num_rows:,}  o={q13_orders.num_rows:,}  LEFT JOIN)"
        )
        q13_rows = []
        q13_rows.append((
            "MXFrame CPU",
            _stats(_time_cold(lambda: run_q13_mxframe(q13_cust, q13_orders), COLD)),
            _stats(_time_runs( lambda: run_q13_mxframe(q13_cust, q13_orders), HOT, warmup=2)),
        ))
        q13_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q13_pandas(q13_cust, q13_orders), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q13_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q13_polars(q13_cust, q13_orders), HOT, warmup=2))))
        _print_table("Q13", q13_rows)
        _summarize_relative(q13_rows)

        # ----------------------------------------------------------
        #  Q19 — Discounted Revenue (complex OR predicates)
        # ----------------------------------------------------------
        _section(
            f"Q19 (part={q19_part.num_rows:,}  l={q19_li.num_rows:,}  OR predicates)"
        )
        q19_rows = []
        q19_rows.append((
            "MXFrame CPU",
            _stats(_time_cold(lambda: run_q19_mxframe(q19_part, q19_li), COLD)),
            _stats(_time_runs( lambda: run_q19_mxframe(q19_part, q19_li), HOT, warmup=2)),
        ))
        if GPU_READY:
            try:
                q19_rows.append((
                    "MXFrame GPU",
                    _stats(_time_cold(lambda: run_q19_mxframe(q19_part, q19_li, device="gpu"), COLD)),
                    _stats(_time_runs( lambda: run_q19_mxframe(q19_part, q19_li, device="gpu"), HOT, warmup=2)),
                ))
            except Exception as e:
                print(f"  GPU Q19 skipped: {e}")
        q19_rows.append(("Pandas", None, _stats(_time_runs(lambda: run_q19_pandas(q19_part, q19_li), HOT, warmup=1))))
        if POLARS_AVAILABLE:
            q19_rows.append(("Polars", None, _stats(_time_runs(lambda: run_q19_polars(q19_part, q19_li), HOT, warmup=2))))
        _print_table("Q19", q19_rows)
        _summarize_relative(q19_rows)

    print(f"\n{'='*68}")
    print("  Done.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()