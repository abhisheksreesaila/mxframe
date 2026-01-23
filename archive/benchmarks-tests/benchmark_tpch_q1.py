#!/usr/bin/env python3
"""
TPC-H Q1 Benchmark: MXFrame vs Polars

Compares the performance of:
1. MaxNativeQ1 (Fully MAX Engine)
2. CompiledQ1 (Pre-compiled MAX + numpy)
3. Polars (CPU)

Usage:
    python benchmark_tpch_q1.py [--rows N] [--iterations N]
"""
import numpy as np
import time
import argparse
from datetime import date

# Import MXFrame components
from max_main import (
    generate_tpch_lineitem,
    tpch_q1,
    CompiledQ1,
    MaxNativeQ1,
)

def benchmark_polars(lineitem, verbose=False):
    """Run TPC-H Q1 using Polars."""
    import polars as pl
    
    # Convert to Polars DataFrame
    # Convert epoch days to actual dates for Polars
    epoch = date(1970, 1, 1)
    dates = [date.fromordinal(epoch.toordinal() + int(d)) for d in lineitem['l_shipdate']]
    
    df = pl.DataFrame({
        'l_shipdate': dates,
        'l_returnflag': lineitem['l_returnflag'].tolist(),
        'l_linestatus': lineitem['l_linestatus'].tolist(),
        'l_quantity': lineitem['l_quantity'],
        'l_extendedprice': lineitem['l_extendedprice'],
        'l_discount': lineitem['l_discount'],
        'l_tax': lineitem['l_tax'],
    })
    
    # TPC-H Q1 query
    cutoff_date = date(1998, 9, 2)  # 1998-12-01 - 90 days
    
    t0 = time.perf_counter()
    
    result = (
        df
        .filter(pl.col('l_shipdate') <= cutoff_date)
        .group_by(['l_returnflag', 'l_linestatus'])
        .agg([
            pl.col('l_quantity').sum().alias('sum_qty'),
            pl.col('l_extendedprice').sum().alias('sum_base_price'),
            (pl.col('l_extendedprice') * (1 - pl.col('l_discount'))).sum().alias('sum_disc_price'),
            (pl.col('l_extendedprice') * (1 - pl.col('l_discount')) * (1 + pl.col('l_tax'))).sum().alias('sum_charge'),
            pl.col('l_quantity').mean().alias('avg_qty'),
            pl.col('l_extendedprice').mean().alias('avg_price'),
            pl.col('l_discount').mean().alias('avg_disc'),
            pl.col('l_quantity').count().alias('count_order'),
        ])
        .sort(['l_returnflag', 'l_linestatus'])
    )
    
    elapsed = time.perf_counter() - t0
    
    if verbose:
        print(result)
    
    return result, elapsed


def benchmark_polars_lazy(lineitem, verbose=False):
    """Run TPC-H Q1 using Polars lazy execution."""
    import polars as pl
    
    # Convert to Polars DataFrame
    epoch = date(1970, 1, 1)
    dates = [date.fromordinal(epoch.toordinal() + int(d)) for d in lineitem['l_shipdate']]
    
    df = pl.DataFrame({
        'l_shipdate': dates,
        'l_returnflag': lineitem['l_returnflag'].tolist(),
        'l_linestatus': lineitem['l_linestatus'].tolist(),
        'l_quantity': lineitem['l_quantity'],
        'l_extendedprice': lineitem['l_extendedprice'],
        'l_discount': lineitem['l_discount'],
        'l_tax': lineitem['l_tax'],
    })
    
    cutoff_date = date(1998, 9, 2)
    
    t0 = time.perf_counter()
    
    result = (
        df.lazy()
        .filter(pl.col('l_shipdate') <= cutoff_date)
        .group_by(['l_returnflag', 'l_linestatus'])
        .agg([
            pl.col('l_quantity').sum().alias('sum_qty'),
            pl.col('l_extendedprice').sum().alias('sum_base_price'),
            (pl.col('l_extendedprice') * (1 - pl.col('l_discount'))).sum().alias('sum_disc_price'),
            (pl.col('l_extendedprice') * (1 - pl.col('l_discount')) * (1 + pl.col('l_tax'))).sum().alias('sum_charge'),
            pl.col('l_quantity').mean().alias('avg_qty'),
            pl.col('l_extendedprice').mean().alias('avg_price'),
            pl.col('l_discount').mean().alias('avg_disc'),
            pl.col('l_quantity').count().alias('count_order'),
        ])
        .sort(['l_returnflag', 'l_linestatus'])
        .collect()
    )
    
    elapsed = time.perf_counter() - t0
    
    if verbose:
        print(result)
    
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description='TPC-H Q1 Benchmark')
    parser.add_argument('--rows', type=int, default=100_000, help='Number of rows (default: 100K)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations (default: 5)')
    parser.add_argument('--skip-numpy', action='store_true', help='Skip numpy-fallback CompiledQ1')
    args = parser.parse_args()
    
    n_rows = args.rows
    n_iter = args.iterations
    
    print("=" * 70)
    print(f"TPC-H Q1 Benchmark: {n_rows:,} rows, {n_iter} iterations")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating {n_rows:,} rows of synthetic TPC-H lineitem data...")
    t0 = time.perf_counter()
    lineitem = generate_tpch_lineitem(n_rows)
    gen_time = time.perf_counter() - t0
    print(f"Data generation: {gen_time*1000:.1f}ms")
    
    results = {}
    
    # === MaxNativeQ1 (Fully MAX) ===
    print("\n--- MaxNativeQ1 (Fully MAX Engine) ---")
    
    print("Compiling fused graph...")
    t0 = time.perf_counter()
    native_q1 = MaxNativeQ1(n_rows, verbose=True)
    compile_time_native = time.perf_counter() - t0
    
    native_times = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        native_results, timings = native_q1.execute(lineitem)
        exec_time = time.perf_counter() - t0
        native_times.append(exec_time)
        print(f"  Iteration {i+1}: {exec_time*1000:.2f}ms")
    
    min_native = min(native_times)
    print(f"Best execution: {min_native*1000:.2f}ms")
    results['max_native'] = min_native
    
    print("\nResults:")
    for row in native_results:
        print(f"  ({row['l_returnflag']}, {row['l_linestatus']}): "
              f"sum_qty={row['sum_qty']:.0f}, count={row['count_order']}")
    
    # === CompiledQ1 (numpy fallbacks) ===
    if not args.skip_numpy:
        print("\n--- CompiledQ1 (MAX + numpy) ---")
        
        t0 = time.perf_counter()
        compiled_q1 = CompiledQ1(n_rows, verbose=True)
        compile_time_compiled = time.perf_counter() - t0
        
        compiled_times = []
        for i in range(n_iter):
            t0 = time.perf_counter()
            _, _ = compiled_q1.execute(lineitem)
            exec_time = time.perf_counter() - t0
            compiled_times.append(exec_time)
            print(f"  Iteration {i+1}: {exec_time*1000:.2f}ms")
        
        min_compiled = min(compiled_times)
        print(f"Best execution: {min_compiled*1000:.2f}ms")
        results['compiled_numpy'] = min_compiled
    
    # === Polars benchmark ===
    print("\n--- Polars (Lazy) ---")
    polars_times = []
    for i in range(n_iter):
        try:
            polars_result, polars_time = benchmark_polars_lazy(lineitem, verbose=False)
            polars_times.append(polars_time)
            print(f"  Iteration {i+1}: {polars_time*1000:.2f}ms")
        except Exception as e:
            print(f"Polars error: {e}")
            break
    
    if polars_times:
        min_polars = min(polars_times)
        print(f"Best execution: {min_polars*1000:.2f}ms")
        results['polars'] = min_polars
    
    # === Summary ===
    print("\n" + "=" * 70)
    print(f"SUMMARY (Best of {n_iter} iterations)")
    print("=" * 70)
    
    polars_baseline = results.get('polars', 1.0)
    
    print(f"\n{'Engine':<25} {'Time (ms)':<12} {'vs Polars':<20}")
    print("-" * 60)
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    for name, elapsed in sorted_results:
        speedup = polars_baseline / elapsed if elapsed else 0
        if speedup >= 1:
            comparison = f"{speedup:.2f}x faster"
        else:
            comparison = f"{1/speedup:.2f}x slower"
        print(f"{name:<25} {elapsed*1000:<12.2f} {comparison}")
    
    print()


if __name__ == "__main__":
    main()
