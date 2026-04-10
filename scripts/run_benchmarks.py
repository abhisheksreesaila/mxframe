import argparse
import time
import pyarrow as pa
import numpy as np

# mxframe imports
from mxframe.lazy_frame import LazyFrame
from mxframe.lazy_expr import col, lit

def generate_synthetic_data(num_rows):
    print(f"Generating synthetic PyArrow data for {num_rows:,} rows...")
    # TPC-H Q1 approximate synthetic structure
    np.random.seed(42)
    return pa.table({
        'l_returnflag': np.random.choice(['A', 'N', 'R'], num_rows),
        'l_linestatus': np.random.choice(['F', 'O'], num_rows),
        'l_quantity': np.random.uniform(1.0, 50.0, num_rows),
        'l_extendedprice': np.random.uniform(100.0, 100000.0, num_rows),
        'l_discount': np.random.uniform(0.00, 0.10, num_rows),
        'l_tax': np.random.uniform(0.00, 0.08, num_rows),
        'l_shipdate': np.random.choice(['1998-09-01', '1998-10-01', '1998-11-01'], num_rows)
    })

def run_q1(table, engine="mxframe_cpu"):
    print(f"\n--- Running Q1 on {engine} ---")
    start = time.perf_counter()
    
    if engine == "mxframe_cpu":
        lf = LazyFrame(table)
        res = (
            lf.filter(col("l_shipdate") <= lit('1998-09-02'))
            .groupby(col("l_returnflag"), col("l_linestatus"))
            .agg(
                col("l_quantity").sum().alias("sum_qty"),
                col("l_extendedprice").sum().alias("sum_base_price"),
                col("l_discount").sum().alias("sum_disc"),
                col("l_tax").sum().alias("sum_charge"),
                col("l_quantity").mean().alias("avg_qty"),
                col("l_extendedprice").mean().alias("avg_price"),
                col("l_discount").mean().alias("avg_disc"),
                col("l_quantity").count().alias("count_order")
            )
            .compute(device="cpu")
        )
    elif engine == "mxframe_gpu":
        lf = LazyFrame(table)
        res = (
            lf.filter(col("l_shipdate") <= lit('1998-09-02'))
            .groupby(col("l_returnflag"), col("l_linestatus"))
            .agg(
                col("l_quantity").sum().alias("sum_qty")
                # Add full aggregations as supported by GPU kernels
            )
            .compute(device="gpu")
        )
    elif engine == "pandas":
        import pandas as pd
        df = table.to_pandas()
        res = df[df["l_shipdate"] <= '1998-09-02'].groupby(["l_returnflag", "l_linestatus"]).agg({
            "l_quantity": ["sum", "mean", "count"],
            "l_extendedprice": ["sum", "mean"],
            "l_discount": ["sum", "mean"],
            "l_tax": ["sum"]
        })
    elif engine == "polars":
        import polars as pl
        df = pl.from_arrow(table)
        res = (
            df.filter(pl.col("l_shipdate") <= '1998-09-02')
            .group_by(["l_returnflag", "l_linestatus"])
            .agg([
                pl.col("l_quantity").sum().alias("sum_qty"),
                pl.col("l_extendedprice").sum().alias("sum_base_price")
            ])
        )
        
    duration = (time.perf_counter() - start) * 1000
    print(f"Completed in {duration:.2f} ms")
    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mxframe TPC-H Benchmarks")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor: 1=1M, 10=10M, 100=100M rows")
    parser.add_argument("--engine", type=str, default="all", choices=["all", "mxframe_cpu", "mxframe_gpu", "pandas", "polars"])
    
    args = parser.parse_args()
    num_rows = args.scale * 1_000_000
    
    table = generate_synthetic_data(num_rows)
    
    engines = ["mxframe_cpu", "mxframe_gpu", "pandas", "polars"] if args.engine == "all" else [args.engine]
    
    print(f"\n=============================================")
    print(f"    MXFRAME BENCHMARK RUNNER (Rows: {num_rows:,})    ")
    print(f"=============================================")
    
    for eng in engines:
        run_q1(table, eng)
