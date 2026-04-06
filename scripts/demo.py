#!/usr/bin/env python3
"""
MXFrame live demo — 5 TPC-H queries at 1M and 10M rows (CPU + GPU).

Run:
    pixi run python scripts/demo.py
    pixi run python scripts/demo.py --scale 10_000_000
    pixi run python scripts/demo.py --scale 100_000_000  # ~30 GB RAM, allow 5 min

Each query runs cold (fresh JIT), then hot (steady-state cache warm).
Results are validated against Pandas before timing is shown.
"""

import argparse
import os
import shutil
import time
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from max import driver as _driver

# ── MXFrame public API ─────────────────────────────────────────────────────
from mxframe import LazyFrame, col, lit, when
from mxframe.lazy_frame import Scan
from mxframe.custom_ops import clear_cache


# ═══════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════
HOT_RUNS  = 5
COLD_RUNS = 1

# Date constants (int32: days since 1970-01-01 OR YYYYMMDD depending on query)
CUTOFF_Q1   = 10471   # 1998-09-02
DISC_LO     = 0.05
DISC_HI     = 0.07
QTY_HI      = 24.0
DATE_LO_Q6  = 8766    # 1994-01-01
DATE_HI_Q6  = 9131    # 1995-01-01
DATE_LO_Q14 = 9374    # 1995-09-01
DATE_HI_Q14 = 9404    # 1995-10-01
DATE_LO_Q12 = 8761    # 1994-01-01
DATE_HI_Q12 = 9126    # 1995-01-01

_Q9_NATIONS = [
    "ALGERIA","ARGENTINA","BRAZIL","CANADA","EGYPT","ETHIOPIA","FRANCE",
    "GERMANY","INDIA","INDONESIA","IRAN","IRAQ","JAPAN","JORDAN","KENYA",
    "MOROCCO","MOZAMBIQUE","PERU","CHINA","ROMANIA","SAUDI ARABIA",
    "VIETNAM","RUSSIA","UNITED KINGDOM","UNITED STATES",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Data generation (scales with --scale)
# ═══════════════════════════════════════════════════════════════════════════

def make_lineitem(n: int, seed: int = 42) -> pa.Table:
    """Flat lineitem table for Q1 and Q6."""
    rng = np.random.default_rng(seed)
    rf  = np.array(["A", "N", "R"],   dtype=object)[rng.integers(0, 3, size=n)]
    ls  = np.array(["F", "O"],         dtype=object)[rng.integers(0, 2, size=n)]
    return pa.table({
        "l_returnflag":    rf.tolist(),
        "l_linestatus":    ls.tolist(),
        "l_quantity":      rng.uniform(1.0,   50.0,      n).astype(np.float32),
        "l_extendedprice": rng.uniform(900.0, 100_000.0, n).astype(np.float32),
        "l_discount":      rng.uniform(0.0,   0.10,      n).astype(np.float32),
        "l_tax":           rng.uniform(0.0,   0.08,      n).astype(np.float32),
        "l_shipdate":      rng.integers(8_000, 10_550,   n, dtype=np.int32),
    })


def make_q12_tables(n: int, seed: int = 77):
    """Orders + lineitem for Q12 (join + conditional groupby)."""
    rng      = np.random.default_rng(seed)
    n_orders = max(n // 6, 1)
    priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
    shipmodes  = ["AIR", "SHIP", "TRUCK", "RAIL", "MAIL", "FOB", "REG AIR"]
    orders = pa.table({
        "o_orderkey":     pa.array(np.arange(n_orders, dtype=np.int32)),
        "o_orderpriority": rng.choice(priorities, n_orders).tolist(),
    })
    # Ensure receiptdate > commitdate > shipdate for some rows (Q12 predicate)
    base     = rng.integers(8400, 9500, n, dtype=np.int32)
    shipdate = base - rng.integers(5, 30, n, dtype=np.int32)
    commit   = base - rng.integers(1, 5,  n, dtype=np.int32)
    lineitem = pa.table({
        "l_orderkey":   pa.array(rng.integers(0, n_orders, n, dtype=np.int32)),
        "l_shipmode":   rng.choice(shipmodes, n).tolist(),
        "l_shipdate":   pa.array(shipdate),
        "l_commitdate": pa.array(commit),
        "l_receiptdate":pa.array(base),
    })
    return orders, lineitem


def make_q14_tables(n: int, seed: int = 88):
    """Part + lineitem for Q14 (promo revenue %)."""
    rng      = np.random.default_rng(seed)
    n_parts  = max(n // 10, 1)
    p_types  = ["PROMO ANODIZED STEEL","STANDARD POLISHED BRASS",
                "LARGE BURNISHED COPPER","ECONOMY ANODIZED STEEL",
                "PROMO BURNISHED NICKEL","MEDIUM PLATED TIN"]
    part = pa.table({
        "p_partkey": pa.array(np.arange(n_parts, dtype=np.int32)),
        "p_type":    rng.choice(p_types, n_parts).tolist(),
    })
    lineitem = pa.table({
        "l_partkey":       pa.array(rng.integers(0, n_parts, n, dtype=np.int32)),
        "l_extendedprice": rng.uniform(900.0, 100_000.0, n).astype(np.float32),
        "l_discount":      rng.uniform(0.0,   0.10,      n).astype(np.float32),
        "l_shipdate":      rng.integers(9_000, 9_800,    n, dtype=np.int32),
    })
    return part, lineitem


def make_q9_tables(n: int, seed: int = 9):
    """6-table star schema for Q9 (5-join profit by nation/year)."""
    rng        = np.random.default_rng(seed)
    n_parts    = max(n // 40, 500)
    n_suppliers= max(n // 200, 100)
    n_orders   = max(n // 8, 1000)
    nation     = pa.table({
        "n_nationkey": pa.array(np.arange(25, dtype=np.int32)),
        "n_name":      pa.array(_Q9_NATIONS),
    })
    supplier = pa.table({
        "s_suppkey":   pa.array(np.arange(n_suppliers, dtype=np.int32)),
        "s_nationkey": pa.array(rng.integers(0, 25, n_suppliers, dtype=np.int32)),
    })
    part_names = [
        f"maroon green dodger {i}" if i % 3 == 0 else f"blue ivory lace {i}"
        for i in range(n_parts)
    ]
    part = pa.table({
        "p_partkey": pa.array(np.arange(n_parts, dtype=np.int32)),
        "p_name":    pa.array(part_names),
    })
    ps_pk       = np.repeat(np.arange(n_parts, dtype=np.int32), 4)
    ps_sk       = (ps_pk + np.tile(np.arange(4, dtype=np.int32), n_parts)) % n_suppliers
    partsupp    = pa.table({
        "ps_partkey":    pa.array(ps_pk),
        "ps_suppkey":    pa.array(ps_sk.astype(np.int32)),
        "ps_supplycost": rng.uniform(1.0, 100.0, len(ps_pk)).astype(np.float32),
    })
    orders = pa.table({
        "o_orderkey":  pa.array(np.arange(n_orders, dtype=np.int32)),
        "o_orderdate": pa.array(rng.integers(19900101, 20000101, n_orders, dtype=np.int32)),
    })
    idx     = rng.integers(0, len(ps_pk), n)
    lineitem = pa.table({
        "l_orderkey":      pa.array(rng.integers(0, n_orders, n, dtype=np.int32)),
        "l_partkey":       pa.array(ps_pk[idx]),
        "l_suppkey":       pa.array(ps_sk[idx].astype(np.int32)),
        "l_extendedprice": rng.uniform(10.0, 1000.0, n).astype(np.float32),
        "l_discount":      rng.uniform(0.0,  0.10,   n).astype(np.float32),
        "l_quantity":      rng.uniform(1.0,  50.0,   n).astype(np.float32),
    })
    return nation, supplier, partsupp, part, orders, lineitem


# ═══════════════════════════════════════════════════════════════════════════
#  Query definitions (MXFrame API)
# ═══════════════════════════════════════════════════════════════════════════

# Q1 — Filter + grouped 8-agg on flat table (pure Mojo group_sum/mean/count)
def q1(lineitem, device="cpu"):
    return (
        LazyFrame(Scan(lineitem))
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


# Q6 — Multi-predicate filter + global sum (single-pass scan)
def q6(lineitem, device="cpu"):
    return (
        LazyFrame(Scan(lineitem))
        .filter(
            (col("l_shipdate") >= lit(DATE_LO_Q6))
            & (col("l_shipdate") <  lit(DATE_HI_Q6))
            & (col("l_discount") >= lit(DISC_LO))
            & (col("l_discount") <= lit(DISC_HI))
            & (col("l_quantity") <  lit(QTY_HI))
        )
        .groupby()
        .agg((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
        .compute(device=device)
    )


# Q12 — Join + isin filter + conditional groupby (CASE WHEN in agg)
def q12(orders, lineitem, device="cpu"):
    return (
        LazyFrame(Scan(lineitem))
        .join(LazyFrame(Scan(orders)), left_on="l_orderkey", right_on="o_orderkey")
        .filter(col("l_shipmode").isin(["MAIL", "SHIP"]))
        .filter(col("l_commitdate") < col("l_receiptdate"))
        .filter(col("l_shipdate")   < col("l_commitdate"))
        .filter(col("l_receiptdate") >= lit(DATE_LO_Q12))
        .filter(col("l_receiptdate") <  lit(DATE_HI_Q12))
        .groupby("l_shipmode")
        .agg(
            when(col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]),
                 lit(1), lit(0)).sum().alias("high_line_count"),
            when(~col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]),
                 lit(1), lit(0)).sum().alias("low_line_count"),
        )
        .sort(col("l_shipmode"))
        .compute(device=device)
    )


# Q14 — Join + startswith filter + promo revenue % (global agg)
def q14(part, lineitem, device="cpu"):
    promo = when(col("p_type").startswith("PROMO"),
                 col("l_extendedprice") * (lit(1.0) - col("l_discount")), lit(0.0))
    total = col("l_extendedprice") * (lit(1.0) - col("l_discount"))
    return (
        LazyFrame(Scan(lineitem))
        .join(LazyFrame(Scan(part)), left_on="l_partkey", right_on="p_partkey")
        .filter(col("l_shipdate") >= lit(DATE_LO_Q14))
        .filter(col("l_shipdate") <  lit(DATE_HI_Q14))
        .groupby()
        .agg(
            promo.sum().alias("promo_revenue"),
            total.sum().alias("total_revenue"),
        )
        .compute(device=device)
    )


# Q9 — 5 joins + profit expression + year() groupby key (star-schema OLAP)
def q9(nation, supplier, partsupp, part, orders, lineitem, device="cpu"):
    return (
        LazyFrame(Scan(lineitem))
        .join(
            LazyFrame(Scan(part)).filter(col("p_name").contains("green")),
            left_on="l_partkey", right_on="p_partkey",
        )
        .join(LazyFrame(Scan(partsupp)),
              left_on=["l_partkey", "l_suppkey"],
              right_on=["ps_partkey", "ps_suppkey"])
        .join(LazyFrame(Scan(supplier)), left_on="l_suppkey",   right_on="s_suppkey")
        .join(LazyFrame(Scan(nation)),   left_on="s_nationkey", right_on="n_nationkey")
        .join(LazyFrame(Scan(orders)),   left_on="l_orderkey",  right_on="o_orderkey")
        .groupby("n_name", col("o_orderdate").year().alias("o_year"))
        .agg(
            (col("l_extendedprice") * (lit(1.0) - col("l_discount"))
             - col("ps_supplycost") * col("l_quantity"))
            .sum().alias("sum_profit")
        )
        .sort(col("n_name"))
        .compute(device=device)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Timing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000.0


def _hot_times(fn, n=HOT_RUNS):
    import gc
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    try:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    finally:
        if was: gc.enable()
        gc.collect()
    return times


def _fmt(ms):      return f"{ms:8.1f}ms"
def _speedup(a, b): return f"{b/a:.1f}x faster" if a < b else f"{a/b:.1f}x slower"


# ═══════════════════════════════════════════════════════════════════════════
#  Query runners (with Pandas validation)
# ═══════════════════════════════════════════════════════════════════════════

def _validate_scalar(name, mxf_val, pan_val, tol=0.01):
    try:
        a, b = float(mxf_val), float(pan_val)
        ok = abs(a - b) / (abs(b) + 1e-9) < tol
    except Exception:
        a, b, ok = mxf_val, pan_val, False
    else:
        mxf_val, pan_val = round(a, 4), round(b, 4)
    status = "✓" if ok else "✗ MISMATCH"
    print(f"  Correctness: {status}  (MXFrame={mxf_val}  Pandas={pan_val})")


def _validate_rows(name, mxf_tbl, pan_df):
    ok = mxf_tbl.num_rows == len(pan_df)
    status = "✓" if ok else "✗ MISMATCH"
    print(f"  Correctness: {status}  ({mxf_tbl.num_rows} rows  Pandas={len(pan_df)} rows)")


def _header(title, rows):
    print()
    print("─" * 60)
    print(f"  {title}  ({rows:,} rows)")
    print("─" * 60)


def _row(label, cold_ms, hot_times, pan_hot):
    hot_min = min(hot_times)
    hot_med = sorted(hot_times)[len(hot_times) // 2]
    sp      = _speedup(hot_min, pan_hot)
    print(f"  {label:<14}  cold={_fmt(cold_ms)}  hot_min={_fmt(hot_min)}"
          f"  hot_med={_fmt(hot_med)}  [{sp} vs Pandas]")


# ═══════════════════════════════════════════════════════════════════════════
#  Per-query demo blocks
# ═══════════════════════════════════════════════════════════════════════════

def demo_q1(lineitem, device):
    _header("Q1  Filter + 8-agg GroupBy", lineitem.num_rows)
    # Pandas reference
    pdf = lineitem.to_pandas()
    q    = pdf[pdf.l_shipdate <= CUTOFF_Q1]
    q    = q.assign(dp=q.l_extendedprice * (1 - q.l_discount))
    q    = q.assign(ch=q.dp * (1 + q.l_tax))
    pan  = (q.groupby(["l_returnflag","l_linestatus"], as_index=False)
             .agg(sum_qty=("l_quantity","sum"),
                  sum_base_price=("l_extendedprice","sum"),
                  sum_disc_price=("dp","sum"),
                  sum_charge=("ch","sum"),
                  avg_qty=("l_quantity","mean"),
                  avg_price=("l_extendedprice","mean"),
                  avg_disc=("l_discount","mean"),
                  count_order=("l_quantity","count")))

    clear_cache()
    res, cold_ms = _timed(lambda: q1(lineitem, device))
    _validate_rows("Q1", res, pan)

    pan_hot  = min(_hot_times(lambda: q1.__wrapped__(pdf) if hasattr(q1, "__wrapped__") else _pandas_q1(pdf)))
    mxf_hot  = _hot_times(lambda: q1(lineitem, device))
    pan_hot2 = min(_hot_times(lambda: _pandas_q1(pdf)))

    _row(f"MXFrame {device.upper()}", cold_ms, mxf_hot, pan_hot2)
    print(f"  Pandas        hot_min={_fmt(pan_hot2)}")
    return res


def _pandas_q1(pdf):
    q = pdf[pdf.l_shipdate <= CUTOFF_Q1].copy()
    q["dp"] = q.l_extendedprice * (1 - q.l_discount)
    q["ch"] = q["dp"] * (1 + q.l_tax)
    return q.groupby(["l_returnflag","l_linestatus"], as_index=False).agg(
        sum_qty=("l_quantity","sum"),
        sum_base_price=("l_extendedprice","sum"),
        sum_disc_price=("dp","sum"),
        sum_charge=("ch","sum"),
        avg_qty=("l_quantity","mean"),
        avg_price=("l_extendedprice","mean"),
        avg_disc=("l_discount","mean"),
        count_order=("l_quantity","count"),
    )


def demo_q6(lineitem, device):
    _header("Q6  5-predicate Filter + Global Sum", lineitem.num_rows)
    pdf     = lineitem.to_pandas()
    pan_rev = (pdf[(pdf.l_shipdate >= DATE_LO_Q6) & (pdf.l_shipdate < DATE_HI_Q6)
                   & (pdf.l_discount >= DISC_LO) & (pdf.l_discount <= DISC_HI)
                   & (pdf.l_quantity < QTY_HI)]
               ["l_extendedprice"].mul(
                pdf[(pdf.l_shipdate >= DATE_LO_Q6) & (pdf.l_shipdate < DATE_HI_Q6)
                    & (pdf.l_discount >= DISC_LO) & (pdf.l_discount <= DISC_HI)
                    & (pdf.l_quantity < QTY_HI)]["l_discount"]).sum())

    clear_cache()
    res, cold_ms = _timed(lambda: q6(lineitem, device))
    mxf_rev = res.column("revenue")[0].as_py()
    _validate_scalar("Q6", round(mxf_rev, 2), round(pan_rev, 2), tol=0.005)

    mxf_hot  = _hot_times(lambda: q6(lineitem, device))
    pan_hot  = min(_hot_times(lambda: _pandas_q6(pdf)))

    _row(f"MXFrame {device.upper()}", cold_ms, mxf_hot, pan_hot)
    print(f"  Pandas        hot_min={_fmt(pan_hot)}")


def _pandas_q6(pdf):
    m = pdf[(pdf.l_shipdate >= DATE_LO_Q6) & (pdf.l_shipdate < DATE_HI_Q6)
            & (pdf.l_discount >= DISC_LO) & (pdf.l_discount <= DISC_HI)
            & (pdf.l_quantity < QTY_HI)]
    return (m.l_extendedprice * m.l_discount).sum()


def demo_q12(orders, lineitem, device):
    n = lineitem.num_rows
    _header("Q12  Join + isin + CASE WHEN GroupBy", n)
    clear_cache()
    res, cold_ms = _timed(lambda: q12(orders, lineitem, device))

    # Pandas reference
    pdf = pd.merge(lineitem.to_pandas(), orders.to_pandas(),
                   left_on="l_orderkey", right_on="o_orderkey")
    pdf = pdf[pdf.l_shipmode.isin(["MAIL","SHIP"])
              & (pdf.l_commitdate < pdf.l_receiptdate)
              & (pdf.l_shipdate   < pdf.l_commitdate)
              & (pdf.l_receiptdate >= DATE_LO_Q12)
              & (pdf.l_receiptdate <  DATE_HI_Q12)]
    pan = (pdf.assign(hi=pdf.o_orderpriority.isin(["1-URGENT","2-HIGH"]).astype(int),
                      lo=(~pdf.o_orderpriority.isin(["1-URGENT","2-HIGH"])).astype(int))
             .groupby("l_shipmode")
             .agg(high_line_count=("hi","sum"), low_line_count=("lo","sum"))
             .reset_index())

    _validate_rows("Q12", res, pan)

    mxf_hot = _hot_times(lambda: q12(orders, lineitem, device))
    pan_hot = min(_hot_times(lambda: _pandas_q12(orders, lineitem)))

    _row(f"MXFrame {device.upper()}", cold_ms, mxf_hot, pan_hot)
    print(f"  Pandas        hot_min={_fmt(pan_hot)}")


def _pandas_q12(orders, lineitem):
    pdf = pd.merge(lineitem.to_pandas(), orders.to_pandas(),
                   left_on="l_orderkey", right_on="o_orderkey")
    pdf = pdf[pdf.l_shipmode.isin(["MAIL","SHIP"])
              & (pdf.l_commitdate < pdf.l_receiptdate)
              & (pdf.l_shipdate   < pdf.l_commitdate)
              & (pdf.l_receiptdate >= DATE_LO_Q12)
              & (pdf.l_receiptdate <  DATE_HI_Q12)]
    return (pdf.assign(hi=pdf.o_orderpriority.isin(["1-URGENT","2-HIGH"]).astype(int))
              .groupby("l_shipmode").agg(high_line_count=("hi","sum")).reset_index())


def demo_q14(part, lineitem, device):
    _header("Q14  Join + startswith + Promo Revenue %", lineitem.num_rows)
    clear_cache()
    res, cold_ms = _timed(lambda: q14(part, lineitem, device))

    mxf_promo = res.column("promo_revenue")[0].as_py()
    mxf_total = res.column("total_revenue")[0].as_py()
    mxf_pct   = 100.0 * mxf_promo / mxf_total if mxf_total > 0 else 0.0

    pdf   = pd.merge(lineitem.to_pandas(), part.to_pandas(),
                     left_on="l_partkey", right_on="p_partkey")
    pdf   = pdf[(pdf.l_shipdate >= DATE_LO_Q14) & (pdf.l_shipdate < DATE_HI_Q14)]
    p_rev = (pdf[pdf.p_type.str.startswith("PROMO")]
             .eval("l_extendedprice * (1 - l_discount)").sum())
    t_rev = pdf.eval("l_extendedprice * (1 - l_discount)").sum()
    pan_pct = 100.0 * p_rev / t_rev if t_rev > 0 else 0.0

    _validate_scalar("Q14", round(mxf_pct, 2), round(pan_pct, 2), tol=0.005)

    mxf_hot = _hot_times(lambda: q14(part, lineitem, device))
    pan_hot = min(_hot_times(lambda: _pandas_q14(part, lineitem)))

    _row(f"MXFrame {device.upper()}", cold_ms, mxf_hot, pan_hot)
    print(f"  Pandas        hot_min={_fmt(pan_hot)}")


def _pandas_q14(part, lineitem):
    pdf = pd.merge(lineitem.to_pandas(), part.to_pandas(),
                   left_on="l_partkey", right_on="p_partkey")
    pdf = pdf[(pdf.l_shipdate >= DATE_LO_Q14) & (pdf.l_shipdate < DATE_HI_Q14)]
    p_r = (pdf[pdf.p_type.str.startswith("PROMO")]
           .eval("l_extendedprice * (1 - l_discount)").sum())
    t_r = pdf.eval("l_extendedprice * (1 - l_discount)").sum()
    return 100.0 * p_r / t_r if t_r > 0 else 0.0


def demo_q9(nation, supplier, partsupp, part, orders, lineitem, device):
    _header("Q9  5-Join Star Schema + profit expr + year() GroupBy", lineitem.num_rows)
    clear_cache()
    try:
        res, cold_ms = _timed(lambda: q9(nation, supplier, partsupp, part, orders, lineitem, device))
    except RuntimeError as e:
        if "Failed to compile" in str(e) or "mojo_pkg" in str(e):
            mod = "/tmp/.modular"
            shutil.rmtree(mod, ignore_errors=True)
            print(f"  [Mojo pkg cache corrupted — cleared {mod}, retrying...]")
            clear_cache()
            res, cold_ms = _timed(lambda: q9(nation, supplier, partsupp, part, orders, lineitem, device))
        else:
            raise

    # Sanity: expect nations × years groups (row count only, not full pandas compare)
    print(f"  Groups returned: {res.num_rows}  (expect ~25 nations × 10 years)")

    mxf_hot = _hot_times(lambda: q9(nation, supplier, partsupp, part, orders, lineitem, device))
    pan_hot = min(_hot_times(lambda: _pandas_q9(nation, supplier, partsupp, part, orders, lineitem)))

    _row(f"MXFrame {device.upper()}", cold_ms, mxf_hot, pan_hot)
    print(f"  Pandas        hot_min={_fmt(pan_hot)}")


def _pandas_q9(nation, supplier, partsupp, part, orders, lineitem):
    n, s, ps, pt, o, li = (t.to_pandas() for t in
                            (nation, supplier, partsupp, part, orders, lineitem))
    pt_f = pt[pt.p_name.str.contains("green")]
    m    = (li.merge(pt_f, left_on="l_partkey", right_on="p_partkey")
              .merge(ps,   left_on=["l_partkey","l_suppkey"],
                           right_on=["ps_partkey","ps_suppkey"])
              .merge(s,    left_on="l_suppkey",   right_on="s_suppkey")
              .merge(n,    left_on="s_nationkey", right_on="n_nationkey")
              .merge(o,    left_on="l_orderkey",  right_on="o_orderkey"))
    m["o_year"]  = m.o_orderdate // 10000
    m["profit"]  = m.l_extendedprice * (1 - m.l_discount) - m.ps_supplycost * m.l_quantity
    return m.groupby(["n_name","o_year"])["profit"].sum().reset_index()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="MXFrame demo — 5 TPC-H queries")
    ap.add_argument("--scale",  type=int,  default=1_000_000,
                    help="lineitem row count (default 1_000_000)")
    ap.add_argument("--device", default="cpu", choices=["cpu","gpu","both"],
                    help="execution device (default cpu)")
    ap.add_argument("--clear-cache", action="store_true",
                    help="wipe Mojo package cache before running (fixes corrupted cache)")
    args = ap.parse_args()

    N      = args.scale
    device = args.device

    if args.clear_cache:
        for _cache in ["/tmp/.modular", os.path.expanduser("~/.modular/cache")]:
            shutil.rmtree(_cache, ignore_errors=True)
        print("  Mojo package cache cleared.\n")

    # GPU availability check
    gpu_count = 0
    try:
        gpu_count = int(_driver.accelerator_count())
    except Exception:
        pass
    has_gpu = gpu_count > 0

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           MXFrame — TPC-H Query Demo                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Scale   : {N:,} rows")
    print(f"  Device  : {device}  (GPU available: {has_gpu})")
    print(f"  Hot runs: {HOT_RUNS}")
    print()

    if device == "gpu" and not has_gpu:
        print("  ⚠  No GPU found — falling back to CPU.")
        device = "cpu"

    devices = ["cpu", "gpu"] if (device == "both" and has_gpu) else [device if device != "both" else "cpu"]

    # ── Data generation ─────────────────────────────────────────────────
    print("Generating data ...")
    t0 = time.perf_counter()

    li_flat  = make_lineitem(N)
    od12, li12 = make_q12_tables(N)
    pt14, li14 = make_q14_tables(N)
    nation, supplier, partsupp, part, orders, li9 = make_q9_tables(N)

    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"  done in {gen_ms:.0f}ms")
    print(f"  lineitem (flat)  : {li_flat.num_rows:>12,} rows  "
          f"({li_flat.nbytes / 1e6:.0f} MB)")
    print(f"  lineitem (Q9)    : {li9.num_rows:>12,} rows")
    print(f"  lineitem (Q12)   : {li12.num_rows:>12,} rows")
    print(f"  lineitem (Q14)   : {li14.num_rows:>12,} rows")

    for dev in devices:
        dev_lbl = f"[{dev.upper()}]"
        print(f"\n{'═'*60}")
        print(f"  Running on {dev.upper()}")
        print(f"{'═'*60}")

        demo_q1(li_flat, dev)
        demo_q6(li_flat, dev)
        demo_q12(od12, li12, dev)
        demo_q14(pt14, li14, dev)
        demo_q9(nation, supplier, partsupp, part, orders, li9, dev)

    print()
    print("═" * 60)
    print("  Done.")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
