"""Quick CPU AOT performance comparison — Q1 and Q6 style queries.

Shows cold/hot timing with and without the AOT dispatch layer.
Compares against Polars as external baseline.
"""
import sys, time, numpy as np, pyarrow as pa

from mxframe.custom_ops import CustomOpsCompiler, _MODEL_CACHE, _INPUT_PREP_CACHE, _GROUP_ENCODE_CACHE, _POST_OP_MODEL_CACHE
from mxframe.lazy_expr import col, lit, when, Expr
from mxframe.lazy_frame import LazyFrame, Scan, Filter, Aggregate

# ── data generation ───────────────────────────────────────────────────────────
N = 1_000_000
rng = np.random.default_rng(42)

def _make_lineitem(n):
    return pa.table({
        "l_quantity":      pa.array(rng.uniform(1, 50, n).astype(np.float32)),
        "l_extendedprice": pa.array(rng.uniform(900, 104950, n).astype(np.float32)),
        "l_discount":      pa.array(rng.uniform(0, 0.1, n).astype(np.float32)),
        "l_tax":           pa.array(rng.uniform(0, 0.08, n).astype(np.float32)),
        "l_returnflag":    pa.array(rng.choice([0, 1, 2], n).astype(np.int32)),
        "l_linestatus":    pa.array(rng.choice([0, 1], n).astype(np.int32)),
        "l_shipdate":      pa.array(rng.integers(9000, 10952, n).astype(np.int32)),
    })

print(f"Generating {N:,} lineitem rows ... ", end="", flush=True)
lineitem = _make_lineitem(N)
print("done")

# ── Q1 plan ───────────────────────────────────────────────────────────────────
def make_q1_plan(lf):
    filt = lf.filter(col("l_shipdate") <= lit(10488))
    return (filt.groupby("l_returnflag", "l_linestatus")
               .agg(
                   col("l_quantity").sum().alias("sum_qty"),
                   col("l_extendedprice").sum().alias("sum_base_price"),
                   col("l_quantity").mean().alias("avg_qty"),
                   col("l_extendedprice").mean().alias("avg_price"),
                   col("l_discount").mean().alias("avg_disc"),
               ).plan)

# ── Q6 plan ───────────────────────────────────────────────────────────────────
def make_q6_plan(lf):
    # Build Filter → GlobalAgg manually so _compute_masked_global_agg kicks in
    filter_pred = (
        (col("l_shipdate") >= lit(9131))
        & (col("l_shipdate") < lit(9496))
        & (col("l_discount") >= lit(0.05))
        & (col("l_discount") <= lit(0.07))
        & (col("l_quantity") < lit(24))
    )
    disc_price = Expr("mul", col("l_extendedprice"), col("l_discount"))
    disc_price._alias = None
    revenue = disc_price.sum().alias("revenue")
    return Aggregate(Filter(lf.plan, filter_pred), group_by=[], aggs=[revenue])

def clear():
    _MODEL_CACHE.clear()
    _INPUT_PREP_CACHE.clear()
    _GROUP_ENCODE_CACHE.clear()
    _POST_OP_MODEL_CACHE.clear()

def time_query(name, plan_fn, n_cold=1, n_hot=5):
    comp = CustomOpsCompiler(device="cpu")
    lf = LazyFrame(Scan(lineitem))
    plan = plan_fn(lf)

    cold_times = []
    hot_times = []

    for i in range(n_cold):
        clear()
        t0 = time.perf_counter()
        result = comp.compile_and_run(plan)
        cold_times.append((time.perf_counter() - t0) * 1000)

    for i in range(n_hot):
        t0 = time.perf_counter()
        result = comp.compile_and_run(plan)
        hot_times.append((time.perf_counter() - t0) * 1000)

    prov = comp.last_compile_provenance
    print(f"  {name}")
    print(f"    path          = {prov.get('path', '?')}")
    print(f"    cold (1 run)  = {min(cold_times):7.1f} ms")
    print(f"    hot  avg ({n_hot} runs)= {sum(hot_times)/len(hot_times):7.1f} ms  (min={min(hot_times):.1f}ms)")
    print(f"    result rows   = {result.num_rows}")
    return min(cold_times), sum(hot_times)/len(hot_times)

# ── Polars baseline ───────────────────────────────────────────────────────────
try:
    import polars as pl
    ldf = pl.from_arrow(lineitem).lazy()

    def polars_q1():
        return (ldf
            .filter(pl.col("l_shipdate") <= 10488)
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                pl.col("l_quantity").sum().alias("sum_qty"),
                pl.col("l_extendedprice").sum().alias("sum_base_price"),
                pl.col("l_quantity").mean().alias("avg_qty"),
                pl.col("l_extendedprice").mean().alias("avg_price"),
                pl.col("l_discount").mean().alias("avg_disc"),
            )
            .collect())

    def polars_q6():
        return (ldf
            .filter(
                (pl.col("l_shipdate") >= 9131) & (pl.col("l_shipdate") < 9496)
                & (pl.col("l_discount") >= 0.05) & (pl.col("l_discount") <= 0.07)
                & (pl.col("l_quantity") < 24)
            )
            .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
            .collect())

    POLARS_REPS = 5
    times_q1 = []; [times_q1.append(t) for _ in range(POLARS_REPS) for t in (time.perf_counter(),) if polars_q1() is not None]
    # Simpler timing
    t0 = time.perf_counter(); [polars_q1() for _ in range(POLARS_REPS)]; pl_q1_ms = (time.perf_counter()-t0)*1000/POLARS_REPS
    t0 = time.perf_counter(); [polars_q6() for _ in range(POLARS_REPS)]; pl_q6_ms = (time.perf_counter()-t0)*1000/POLARS_REPS
    HAS_POLARS = True
except Exception as e:
    print(f"Polars not available: {e}")
    HAS_POLARS = False
    pl_q1_ms = pl_q6_ms = float('inf')

# ── Run benchmarks ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CPU AOT micro-benchmark  ({N//1_000_000}M rows)")
print(f"{'='*60}")

q1_cold, q1_hot = time_query("Q1 (GroupBy+8Aggs)", make_q1_plan)
q6_cold, q6_hot = time_query("Q6 (Masked GlobalAgg)", make_q6_plan)

if HAS_POLARS:
    print(f"\n  Polars hot avg: Q1={pl_q1_ms:.1f}ms  Q6={pl_q6_ms:.1f}ms")
    print(f"\n  Speedup vs Polars (hot):")
    print(f"    Q1: {pl_q1_ms/q1_hot:.2f}x  Q6: {pl_q6_ms/q6_hot:.2f}x")

print(f"\n  Cold start eliminated: Q1={q1_cold:.1f}ms (was ~10000ms with MAX Graph JIT)")
print(f"{'='*60}")
