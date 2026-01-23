#!/usr/bin/env python3
"""
TPC-H Q1 Benchmark: MAX Engine vs Polars
=========================================
Usage:
    python benchmark_q1_full.py --scale 1          # 6M rows
    python benchmark_q1_full.py --scale 10         # 60M rows
    python benchmark_q1_full.py --synthetic 100000 --force-cpu
"""

import os
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

import argparse
import time
import gc
import numpy as np

GPU_THRESHOLD = 500_000  # Auto-use GPU above this
DATE_CUTOFF = 10471      # 1998-09-02 as epoch days


def generate_tpch_data_duckdb(scale_factor: float = 1.0):
    """Generate official TPC-H data using DuckDB."""
    import duckdb
    
    print(f"📊 Generating TPC-H data (SF={scale_factor})...")
    t0 = time.perf_counter()
    
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={scale_factor})")
    
    df = con.execute("""
        SELECT l_shipdate, l_returnflag, l_linestatus,
               l_quantity, l_extendedprice, l_discount, l_tax
        FROM lineitem
    """).fetchdf()
    
    print(f"✅ Generated {len(df):,} rows in {time.perf_counter() - t0:.1f}s")
    
    # Encode for MAX Engine
    l_shipdate = (df['l_shipdate'].values.astype('datetime64[D]') - 
                  np.datetime64('1970-01-01')).astype(np.int32)
    rf_map = {'A': 0, 'N': 1, 'R': 2}
    ls_map = {'F': 0, 'O': 1}
    
    return {
        'l_shipdate': l_shipdate,
        'l_returnflag_enc': df['l_returnflag'].map(rf_map).values.astype(np.int32),
        'l_linestatus_enc': df['l_linestatus'].map(ls_map).values.astype(np.int32),
        'l_quantity': df['l_quantity'].values.astype(np.float32),
        'l_extendedprice': df['l_extendedprice'].values.astype(np.float32),
        'l_discount': df['l_discount'].values.astype(np.float32),
        'l_tax': df['l_tax'].values.astype(np.float32),
    }, df


def generate_tpch_data_synthetic(n_rows: int, seed: int = 42):
    """Generate synthetic TPC-H-like data (faster, for quick tests)."""
    import pandas as pd
    np.random.seed(seed)
    
    print(f"📊 Generating synthetic data ({n_rows:,} rows)...")
    t0 = time.perf_counter()
    
    l_shipdate = np.random.randint(8035, 10562, size=n_rows).astype(np.int32)
    l_returnflag_enc = np.random.choice([0, 1, 2], n_rows, p=[0.25, 0.50, 0.25]).astype(np.int32)
    l_linestatus_enc = (l_shipdate >= 9299).astype(np.int32)
    l_quantity = np.random.uniform(1, 50, n_rows).astype(np.float32)
    part_price = np.random.uniform(2, 200, n_rows).astype(np.float32)
    l_extendedprice = (l_quantity * part_price).astype(np.float32)
    l_discount = np.random.uniform(0, 0.10, n_rows).astype(np.float32)
    l_tax = np.random.uniform(0, 0.08, n_rows).astype(np.float32)
    
    print(f"✅ Generated {n_rows:,} rows in {time.perf_counter() - t0:.1f}s")
    
    # Create DataFrame for Polars
    df = pd.DataFrame({
        'l_shipdate': pd.to_datetime('1970-01-01') + pd.to_timedelta(l_shipdate, unit='D'),
        'l_returnflag': np.where(l_returnflag_enc == 0, 'A', 
                                 np.where(l_returnflag_enc == 1, 'N', 'R')),
        'l_linestatus': np.where(l_linestatus_enc == 0, 'F', 'O'),
        'l_quantity': l_quantity,
        'l_extendedprice': l_extendedprice,
        'l_discount': l_discount,
        'l_tax': l_tax,
    })
    
    return {
        'l_shipdate': l_shipdate,
        'l_returnflag_enc': l_returnflag_enc,
        'l_linestatus_enc': l_linestatus_enc,
        'l_quantity': l_quantity,
        'l_extendedprice': l_extendedprice,
        'l_discount': l_discount,
        'l_tax': l_tax,
    }, df


def run_max_q1(data: dict, use_gpu: bool = None, n_warmup: int = 2, n_runs: int = 5):
    """Run TPC-H Q1 using MAX Engine."""
    from max import engine, driver
    from max.graph import Graph, TensorType, ops, DeviceRef
    from max.dtype import DType
    
    n_rows = len(data['l_shipdate'])
    if use_gpu is None:
        use_gpu = n_rows >= GPU_THRESHOLD
    
    device_name = "GPU" if use_gpu else "CPU"
    print(f"\n🚀 MAX Engine ({device_name}) - {n_rows:,} rows")
    print("-" * 50)
    
    device = driver.Accelerator() if use_gpu else driver.CPU()
    device_ref = DeviceRef.GPU() if use_gpu else DeviceRef.CPU()
    session = engine.InferenceSession(devices=[device])
    
    class FusedQ1Graph:
        def __init__(self, device_ref):
            self.device_ref = device_ref
        
        def __call__(self, shipdate, rf_enc, ls_enc, qty, price, disc, tax):
            one_f = ops.constant(1.0, dtype=DType.float32, device=self.device_ref)
            cutoff = ops.constant(DATE_CUTOFF, dtype=DType.int32, device=self.device_ref)
            
            date_mask = ops.cast(ops.greater_equal(cutoff, shipdate), DType.float32)
            disc_price = ops.mul(price, ops.sub(one_f, disc))
            charge = ops.mul(disc_price, ops.add(one_f, tax))
            
            two = ops.constant(2, dtype=DType.int32, device=self.device_ref)
            raw_group = ops.add(ops.mul(rf_enc, two), ls_enc)
            
            results = []
            for g, raw_vals in enumerate([[0, 1], [2], [3], [4, 5]]):
                group_mask = None
                for rv in raw_vals:
                    eq = ops.cast(ops.equal(raw_group, ops.constant(rv, DType.int32, self.device_ref)), DType.float32)
                    group_mask = eq if group_mask is None else ops.add(group_mask, eq)
                
                mask = ops.mul(date_mask, group_mask)
                results.extend([
                    ops.sum(ops.mul(mask, qty)),
                    ops.sum(ops.mul(mask, price)),
                    ops.sum(ops.mul(mask, disc)),
                    ops.sum(ops.mul(mask, disc_price)),
                    ops.sum(ops.mul(mask, charge)),
                    ops.sum(mask),
                ])
            return tuple(results)
    
    print("⚙️  Compiling...")
    t0 = time.perf_counter()
    graph = Graph("max_q1", FusedQ1Graph(device_ref), input_types=[
        TensorType(DType.int32, (n_rows,), device_ref),
        TensorType(DType.int32, (n_rows,), device_ref),
        TensorType(DType.int32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
    ])
    model = session.load(graph)
    compile_time = time.perf_counter() - t0
    print(f"✅ Compiled in {compile_time:.1f}s")
    
    # Prepare tensors
    keys = ['l_shipdate', 'l_returnflag_enc', 'l_linestatus_enc',
            'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax']
    if use_gpu:
        inputs = [driver.Tensor.from_numpy(data[k]).to(device) for k in keys]
    else:
        inputs = [driver.Tensor(data[k], device) for k in keys]
    
    def execute():
        outputs = model.execute(*inputs)
        if use_gpu:
            return [float(o.to(driver.CPU()).to_numpy().flat[0]) for o in outputs]
        return [float(o.to_numpy().flat[0]) for o in outputs]
    
    # Warmup
    print(f"🔥 Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        execute()
    
    # Benchmark
    print(f"⏱️  Benchmarking ({n_runs} runs)...")
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        execute()
        times.append(time.perf_counter() - t0)
        print(f"   Run {i+1}: {times[-1]*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    best_time = min(times)
    print(f"\n📊 MAX Engine ({device_name}): avg={avg_time*1000:.2f}ms, best={best_time*1000:.2f}ms")
    
    return {'engine': f'MAX ({device_name})', 'avg': avg_time, 'best': best_time, 'compile': compile_time}


def run_polars_q1(df, use_gpu: bool = False, n_warmup: int = 2, n_runs: int = 5):
    """Run TPC-H Q1 using Polars."""
    import polars as pl
    
    name = "Polars GPU" if use_gpu else "Polars CPU"
    print(f"\n🐻‍❄️ {name} - {len(df):,} rows")
    print("-" * 50)
    
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    
    date_filter = pl.lit("1998-09-02").str.to_date()
    
    def query():
        lf = (df.lazy()
              .filter(pl.col("l_shipdate") <= date_filter)
              .group_by(["l_returnflag", "l_linestatus"])
              .agg([
                  pl.sum("l_quantity").alias("sum_qty"),
                  pl.sum("l_extendedprice").alias("sum_base_price"),
                  (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("sum_disc_price"),
                  (pl.col("l_extendedprice") * (1 - pl.col("l_discount")) * (1 + pl.col("l_tax"))).sum().alias("sum_charge"),
                  pl.len().alias("count_order"),
              ])
              .sort(["l_returnflag", "l_linestatus"]))
        return lf.collect(engine="gpu") if use_gpu else lf.collect()
    
    # Warmup
    print(f"🔥 Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        query()
    
    # Benchmark
    print(f"⏱️  Benchmarking ({n_runs} runs)...")
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        query()
        times.append(time.perf_counter() - t0)
        print(f"   Run {i+1}: {times[-1]*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    best_time = min(times)
    print(f"\n📊 {name}: avg={avg_time*1000:.2f}ms, best={best_time*1000:.2f}ms")
    
    return {'engine': name, 'avg': avg_time, 'best': best_time, 'compile': 0}


def run_benchmark(scale_factor: float = None, synthetic_rows: int = None,
                  force_gpu: bool = False, force_cpu: bool = False,
                  include_polars_gpu: bool = True):
    """Run full benchmark suite."""
    
    print("=" * 70)
    print("🏁 TPC-H Q1 Benchmark: MAX Engine vs Polars")
    print("=" * 70)
    
    # Generate data
    if synthetic_rows:
        data, df = generate_tpch_data_synthetic(synthetic_rows)
    else:
        data, df = generate_tpch_data_duckdb(scale_factor or 1.0)
    
    n_rows = len(data['l_shipdate'])
    
    # Determine GPU usage
    if force_cpu:
        use_gpu = False
    elif force_gpu:
        use_gpu = True
    else:
        use_gpu = n_rows >= GPU_THRESHOLD
    
    print(f"\n📋 Configuration:")
    print(f"   Rows: {n_rows:,}")
    print(f"   Device: {'GPU' if use_gpu else 'CPU'} (threshold: {GPU_THRESHOLD:,})")
    
    results = []
    
    # Run MAX Engine
    max_result = run_max_q1(data, use_gpu=use_gpu)
    results.append(max_result)
    gc.collect()
    
    # Run Polars CPU
    polars_cpu_result = run_polars_q1(df, use_gpu=False)
    results.append(polars_cpu_result)
    gc.collect()
    
    # Run Polars GPU (optional)
    if include_polars_gpu and use_gpu:
        try:
            polars_gpu_result = run_polars_q1(df, use_gpu=True)
            results.append(polars_gpu_result)
        except Exception as e:
            print(f"\n⚠️  Polars GPU failed: {e}")
        gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\nData: {n_rows:,} rows")
    print(f"\n{'Engine':<20} {'Avg Time':>12} {'Best Time':>12} {'vs Polars CPU':>15}")
    print("-" * 62)
    
    baseline = polars_cpu_result['avg']
    for r in results:
        speedup = baseline / r['avg']
        speedup_str = f"{speedup:.1f}x faster" if speedup > 1 else f"{1/speedup:.1f}x slower"
        print(f"{r['engine']:<20} {r['avg']*1000:>10.2f}ms {r['best']*1000:>10.2f}ms {speedup_str:>15}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPC-H Q1 Benchmark: MAX Engine vs Polars")
    parser.add_argument("--scale", type=float, default=None,
                        help="TPC-H scale factor (0.1=600K, 1=6M, 10=60M rows)")
    parser.add_argument("--synthetic", type=int, default=None,
                        help="Use synthetic data with N rows instead of TPC-H")
    parser.add_argument("--force-gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--no-polars-gpu", action="store_true", help="Skip Polars GPU benchmark")
    
    args = parser.parse_args()
    
    # Default to scale 1 if nothing specified
    if args.scale is None and args.synthetic is None:
        args.scale = 1.0
    
    run_benchmark(
        scale_factor=args.scale,
        synthetic_rows=args.synthetic,
        force_gpu=args.force_gpu,
        force_cpu=args.force_cpu,
        include_polars_gpu=not args.no_polars_gpu,
    )
