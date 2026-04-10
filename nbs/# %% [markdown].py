# %% [markdown]
# # Phase 5 Benchmark: TPC-H Q3 (Shipping Priority)
# 
# Three-way inner join + filter + groupby + aggregation + sort + limit.
# 
# **Query:** For a given market segment ("BUILDING"), find the top-10 unshipped orders
# by revenue. Revenue = `l_extendedprice * (1 - l_discount)`.
# 
# ```sql
# SELECT l_orderkey, o_orderdate, o_shippriority,
#        SUM(l_extendedprice * (1 - l_discount)) AS revenue
# FROM   customer JOIN orders    ON c_custkey = o_custkey
#                 JOIN lineitem  ON l_orderkey = o_orderkey
# WHERE  c_mktsegment = 'BUILDING'
#   AND  o_orderdate  < DATE '1995-03-15'   -- encoded as int
#   AND  l_shipdate   > DATE '1995-03-15'   -- encoded as int
# GROUP BY l_orderkey, o_orderdate, o_shippriority
# ORDER BY revenue DESC, o_orderdate ASC
# LIMIT 10;
# ```

# %%
#| hide
#| eval: false

import os
import platform
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame
from mxframe.custom_ops import clear_cache

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("Polars not installed: skipping Polars baseline.")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("DuckDB not installed: skipping DuckDB reference.")

from max import driver as _driver

# %%
#| hide
#| eval: false

def _safe_gpu_count() -> int:
    try:
        return int(_driver.accelerator_count())
    except Exception:
        return 0

def report_runtime_versions() -> None:
    print("Runtime versions")
    print("================")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"pyarrow: {pa.__version__}")
    if POLARS_AVAILABLE:
        print(f"polars: {pl.__version__}")
    else:
        print("polars: not installed")
    if DUCKDB_AVAILABLE:
        print(f"duckdb: {duckdb.__version__}")
    else:
        print("duckdb: not installed")

def report_benchmark_context(row_counts: dict[str, int]) -> None:
    print("Benchmark context")
    print("=================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"MAX accelerators: {_safe_gpu_count()}")
    print(f"Polars available: {POLARS_AVAILABLE}")
    print(f"DuckDB available: {DUCKDB_AVAILABLE}")
    for name, count in row_counts.items():
        print(f"{name:>8}: {count:,} rows")

def summarize_relative(rows, baselines=("Pandas", "Polars", "DuckDB")):
    stats = {name: s for name, s in rows}
    for baseline in baselines:
        if baseline not in stats:
            continue
        print(f"\nRelative to {baseline} (min-ms ratio, <1 is faster):")
        for name, sample in rows:
            ratio = sample["min"] / stats[baseline]["min"]
            print(f"  {name:<24} {ratio:.2f}x")

def print_takeaway(rows, target="Pandas", label="MXFrame CPU (hot)", close=1.15, beat=0.95):
    stats = {name: s for name, s in rows}
    if target not in stats or label not in stats:
        return
    ratio = stats[label]["min"] / stats[target]["min"]
    if ratio <= beat:
        verdict = "beats"
    elif ratio <= close:
        verdict = "is close to"
    else:
        verdict = "lags"
    print(f"\nTakeaway: {label} {verdict} {target} ({ratio:.2f}x).")

# %% [markdown]
# ## 1) Synthetic TPC-H Data (customer / orders / lineitem)

# %%
#| hide
#| eval: false


DATE_1995_03_15 = 9204  # days since 1970-01-01

def make_tpch_tables(n_customers=15_000, n_orders=150_000, n_lineitem=600_000, seed=42):
    rng = np.random.default_rng(seed)

    # -- Customer --
    segments = np.array(["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"])
    c_custkey = np.arange(1, n_customers + 1, dtype=np.int32)
    c_mktsegment = rng.choice(segments, size=n_customers)
    customer = pa.table({
        "c_custkey": c_custkey,
        "c_mktsegment": c_mktsegment.tolist(),
    })

    # -- Orders --
    o_orderkey = np.arange(1, n_orders + 1, dtype=np.int32)
    o_custkey = rng.integers(1, n_customers + 1, size=n_orders, dtype=np.int32)
    o_orderdate = rng.integers(8800, 9300, size=n_orders, dtype=np.int32)  # ~1994-1995
    o_shippriority = rng.integers(0, 5, size=n_orders, dtype=np.int32)
    orders = pa.table({
        "o_orderkey": o_orderkey,
        "o_custkey": o_custkey,
        "o_orderdate": o_orderdate,
        "o_shippriority": o_shippriority,
    })

    # -- Lineitem --
    l_orderkey = rng.integers(1, n_orders + 1, size=n_lineitem, dtype=np.int32)
    l_extendedprice = rng.uniform(900.0, 100_000.0, size=n_lineitem).astype(np.float32)
    l_discount = rng.uniform(0.0, 0.10, size=n_lineitem).astype(np.float32)
    l_shipdate = rng.integers(8900, 9400, size=n_lineitem, dtype=np.int32)  # ~1994-1995
    lineitem = pa.table({
        "l_orderkey": l_orderkey,
        "l_extendedprice": l_extendedprice,
        "l_discount": l_discount,
        "l_shipdate": l_shipdate,
    })

    return customer, orders, lineitem

customer, orders, lineitem = make_tpch_tables()
print(f"customer:  {customer.num_rows:>10,} rows  cols={customer.column_names}")
print(f"orders:    {orders.num_rows:>10,} rows  cols={orders.column_names}")
print(f"lineitem:  {lineitem.num_rows:>10,} rows  cols={lineitem.column_names}")
report_benchmark_context({
    "customer": customer.num_rows,
    "orders": orders.num_rows,
    "lineitem": lineitem.num_rows,
})
report_runtime_versions()


# %% [markdown]
# ## 2) Query Implementations

# %%
#| hide
#| eval: false

# -- MXFrame Q3 (CPU or GPU) --
def run_q3_mxframe(customer, orders, lineitem, device="cpu"):
    lf_c = LazyFrame(customer).filter(col("c_mktsegment") == lit("BUILDING"))
    lf_o = LazyFrame(orders).filter(col("o_orderdate") < lit(DATE_1995_03_15))
    lf_l = LazyFrame(lineitem).filter(col("l_shipdate") > lit(DATE_1995_03_15))

    # After join, right join key (l_orderkey) is dropped; o_orderkey == l_orderkey.
    result = (
        lf_o
        .join(lf_c, left_on="o_custkey", right_on="c_custkey")
        .join(lf_l, left_on="o_orderkey", right_on="l_orderkey")
        .groupby("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(
            (col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"),
        )
        .sort("revenue", descending=True)
        .limit(10)
        .compute(device=device)
    )
    return result


# -- Pandas Q3 --
def run_q3_pandas(customer, orders, lineitem):
    c = customer.to_pandas()
    o = orders.to_pandas()
    l = lineitem.to_pandas()

    c = c[c["c_mktsegment"] == "BUILDING"]
    o = o[o["o_orderdate"] < DATE_1995_03_15]
    l = l[l["l_shipdate"] > DATE_1995_03_15]

    merged = o.merge(c, left_on="o_custkey", right_on="c_custkey")
    merged = merged.merge(l, left_on="o_orderkey", right_on="l_orderkey")
    merged["revenue"] = merged["l_extendedprice"] * (1 - merged["l_discount"])
    result = (
        merged.groupby(["o_orderkey", "o_orderdate", "o_shippriority"])["revenue"]
        .sum()
        .reset_index()
        .sort_values(["revenue", "o_orderdate"], ascending=[False, True])
        .head(10)
    )
    return result


# -- Polars Q3 --
def run_q3_polars(customer, orders, lineitem):
    if not POLARS_AVAILABLE:
        return None
    c = pl.from_arrow(customer).filter(pl.col("c_mktsegment") == "BUILDING")
    o = pl.from_arrow(orders).filter(pl.col("o_orderdate") < DATE_1995_03_15)
    l = pl.from_arrow(lineitem).filter(pl.col("l_shipdate") > DATE_1995_03_15)

    result = (
        o.join(c, left_on="o_custkey", right_on="c_custkey")
        .join(l, left_on="o_orderkey", right_on="l_orderkey")
        .with_columns((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue"))
        .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
        .agg(pl.col("revenue").sum())
        .sort(["revenue", "o_orderdate"], descending=[True, False])
        .head(10)
    )
    return result.to_arrow()


# -- DuckDB Q3 (correctness reference) --
def run_q3_duckdb(customer, orders, lineitem):
    if not DUCKDB_AVAILABLE:
        return None
    con = duckdb.connect()
    con.register("customer_tbl", customer)
    con.register("orders_tbl", orders)
    con.register("lineitem_tbl", lineitem)
    result = con.sql(f"""
        SELECT o_orderkey, o_orderdate, o_shippriority,
               SUM(CAST(l_extendedprice AS DOUBLE) * (1 - CAST(l_discount AS DOUBLE))) AS revenue
        FROM   customer_tbl
        JOIN   orders_tbl   ON c_custkey = o_custkey
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

# %% [markdown]
# ## 3) Correctness Preview

# %%
#| hide
#| eval: false


def _assert_q3_close(mx_tbl, ref_tbl, label, rtol=1e-2, atol=0.5):
    """Assert MXFrame Q3 output matches reference output."""
    mx_df = mx_tbl.to_pandas() if isinstance(mx_tbl, pa.Table) else mx_tbl
    ref_df = ref_tbl.to_pandas() if isinstance(ref_tbl, pa.Table) else ref_tbl

    # Sort both by revenue DESC for stable comparison
    mx_df = mx_df.sort_values("revenue", ascending=False).reset_index(drop=True)
    ref_df = ref_df.sort_values("revenue", ascending=False).reset_index(drop=True)

    assert len(mx_df) == len(ref_df), f"{label}: row-count mismatch {len(mx_df)} vs {len(ref_df)}"

    # Check integer columns (exact)
    for key in ["o_orderkey", "o_orderdate", "o_shippriority"]:
        if key in mx_df.columns and key in ref_df.columns:
            assert (mx_df[key].values == ref_df[key].values).all(), \
                f"{label}: key mismatch in {key}"

    # Check revenue (float, with tolerance — float32 vs float64 accumulation differences)
    mx_rev = mx_df["revenue"].astype(float).to_numpy()
    ref_rev = ref_df["revenue"].astype(float).to_numpy()
    if not np.allclose(mx_rev, ref_rev, rtol=rtol, atol=atol):
        max_diff = np.abs(mx_rev - ref_rev).max()
        raise AssertionError(f"{label}: revenue mismatch, max diff={max_diff:.6f}")


# Build reference result
ref_label = "DuckDB" if DUCKDB_AVAILABLE else "Pandas"
ref_table = run_q3_duckdb(customer, orders, lineitem) if DUCKDB_AVAILABLE else run_q3_pandas(customer, orders, lineitem)
print(f"{ref_label} Q3 reference (top 10):")
print(ref_table.to_pandas().to_string(index=False) if isinstance(ref_table, pa.Table) else ref_table.to_string(index=False))
print()

# MXFrame CPU result
mx_result = run_q3_mxframe(customer, orders, lineitem)
print(f"MXFrame Q3 CPU result: {mx_result.num_rows} rows, cols={mx_result.column_names}")
print(mx_result.to_pandas().to_string(index=False))
print()

# Correctness assertion
_assert_q3_close(mx_result, ref_table, f"Q3 MXFrame CPU vs {ref_label}")
print(f"✓ Q3 MXFrame CPU correctness check passed against {ref_label}.")

# MXFrame GPU correctness (if available)
if _driver.accelerator_count() > 0:
    try:
        mx_gpu = run_q3_mxframe(customer, orders, lineitem, device="gpu")
        _assert_q3_close(mx_gpu, ref_table, f"Q3 MXFrame GPU vs {ref_label}")
        print(f"✓ Q3 MXFrame GPU correctness check passed against {ref_label}.")
    except Exception as e:
        print(f"  GPU correctness skipped: {e}")

# %% [markdown]
# ## 4) Benchmark
# 
# - **Cold** = `clear_cache()` before every timed run (includes JIT compilation of MAX graph + Mojo kernels)
# - **Hot** = graph already compiled, reuse cached model (this is the steady-state for repeated queries)

# %%
#| hide
#| eval: false


COLD_RUNS = 3
HOT_RUNS  = 5

def _time_once(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out

def _time_runs(fn, runs, warmup=0):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples

def _stats(samples):
    arr = np.array(samples)
    return {"min": arr.min(), "avg": arr.mean(), "median": np.median(arr)}

def _print_table(title, rows):
    print(f"\n{title}")
    print("=" * len(title))
    print(f"  {'Engine':<24} {'Min (ms)':>10} {'Avg (ms)':>10} {'Median (ms)':>12}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*12}")
    for name, s in rows:
        print(f"  {name:<24} {s['min']:>10.1f} {s['avg']:>10.1f} {s['median']:>12.1f}")


# ── GPU readiness ──
GPU_READY = False
if _driver.accelerator_count() > 0:
    try:
        _ = run_q3_mxframe(customer, orders, lineitem, device="gpu")
        GPU_READY = True
    except Exception as e:
        print(f"GPU not usable: {e}")

print(f"GPU ready: {GPU_READY}")
print(f"Rows: customer={customer.num_rows:,}  orders={orders.num_rows:,}  lineitem={lineitem.num_rows:,}")
print(f"Cold runs: {COLD_RUNS}   Hot runs: {HOT_RUNS}")

# ─────────────────────────────────────────────────
#  Q3 COLD (cache cleared each run)
# ─────────────────────────────────────────────────
print("\n" + "━" * 60)
print("  Q3 — 3-way Join + GroupBy + Agg + Sort + Limit 10")
print("━" * 60)

q3_cold = []
cold_samples = []
for _ in range(COLD_RUNS):
    clear_cache()
    ms, _ = _time_once(lambda: run_q3_mxframe(customer, orders, lineitem, device="cpu"))
    cold_samples.append(ms)
q3_cold.append(("MXFrame CPU (cold)", _stats(cold_samples)))

if GPU_READY:
    cold_samples = []
    for _ in range(COLD_RUNS):
        clear_cache()
        ms, _ = _time_once(lambda: run_q3_mxframe(customer, orders, lineitem, device="gpu"))
        cold_samples.append(ms)
    q3_cold.append(("MXFrame GPU (cold)", _stats(cold_samples)))

_print_table("Q3 COLD — includes JIT compilation", q3_cold)

# ─────────────────────────────────────────────────
#  Q3 HOT (cache warm)
# ─────────────────────────────────────────────────
q3_hot = []

# MXFrame CPU hot
_ = run_q3_mxframe(customer, orders, lineitem, device="cpu")
hot_samples = _time_runs(lambda: run_q3_mxframe(customer, orders, lineitem, device="cpu"), runs=HOT_RUNS)
q3_hot.append(("MXFrame CPU (hot)", _stats(hot_samples)))

# MXFrame GPU hot
if GPU_READY:
    _ = run_q3_mxframe(customer, orders, lineitem, device="gpu")
    hot_samples = _time_runs(lambda: run_q3_mxframe(customer, orders, lineitem, device="gpu"), runs=HOT_RUNS)
    q3_hot.append(("MXFrame GPU (hot)", _stats(hot_samples)))

# Pandas
pd_samples = _time_runs(lambda: run_q3_pandas(customer, orders, lineitem), runs=HOT_RUNS, warmup=1)
q3_hot.append(("Pandas", _stats(pd_samples)))

# Polars
if POLARS_AVAILABLE:
    pl_samples = _time_runs(lambda: run_q3_polars(customer, orders, lineitem), runs=HOT_RUNS, warmup=2)
    q3_hot.append(("Polars", _stats(pl_samples)))

# DuckDB
if DUCKDB_AVAILABLE:
    dk_samples = _time_runs(lambda: run_q3_duckdb(customer, orders, lineitem), runs=HOT_RUNS, warmup=2)
    q3_hot.append(("DuckDB", _stats(dk_samples)))

_print_table("Q3 HOT — steady-state, no compilation", q3_hot)
summarize_relative(q3_hot)
print_takeaway(q3_hot, target="Pandas", label="MXFrame CPU (hot)")
if GPU_READY:
    print_takeaway(q3_hot, target="Pandas", label="MXFrame GPU (hot)")


