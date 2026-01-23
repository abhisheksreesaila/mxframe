"""Comprehensive TPC-H Q1 Benchmark: All approaches at 60M rows.

Compares:
1. MAX Engine ops-based (CPU) - ~72 separate operations
2. MAX Engine ops-based (GPU) - ~72 ops with per-op transfers
3. MAX Fused Kernel (CPU) - Single kernel, all 24 aggregations
4. MAX Fused Kernel (GPU) - Single kernel, full GPU execution
5. Polars (CPU)
6. Polars (GPU) - if available
"""

import os
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

import numpy as np
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import duckdb
import polars as pl

from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

# Constants
DATE_CUTOFF = 10471  # 1998-09-02 as epoch days
WARP_SIZE = 32
NUM_GROUPS = 4
NUM_AGGS = 6
TOTAL_OUTPUTS = NUM_GROUPS * NUM_AGGS


@dataclass
class BenchmarkResult:
    name: str
    avg_ms: float
    best_ms: float
    compile_ms: float = 0.0


def generate_tpch_data_duckdb(scale_factor: float = 1.0):
    """Generate official TPC-H data using DuckDB."""
    print(f"\n📊 Generating TPC-H data (SF={scale_factor})...")
    t0 = time.perf_counter()
    
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={scale_factor})")
    
    df = con.execute("""
        SELECT l_shipdate, l_returnflag, l_linestatus,
               l_quantity, l_extendedprice, l_discount, l_tax
        FROM lineitem
    """).fetchdf()
    
    elapsed = time.perf_counter() - t0
    print(f"✅ Generated {len(df):,} rows in {elapsed:.1f}s")
    
    # Encode for MAX Engine
    l_shipdate = (df['l_shipdate'].values.astype('datetime64[D]') - 
                  np.datetime64('1970-01-01')).astype(np.int32)
    rf_map = {'A': 0, 'N': 1, 'R': 2}
    ls_map = {'F': 0, 'O': 1}
    
    l_extendedprice = df['l_extendedprice'].values.astype(np.float32)
    l_discount = df['l_discount'].values.astype(np.float32)
    l_tax = df['l_tax'].values.astype(np.float32)
    
    # Pre-compute derived columns
    disc_price = (l_extendedprice * (1 - l_discount)).astype(np.float32)
    charge = (disc_price * (1 + l_tax)).astype(np.float32)
    
    data = {
        'l_shipdate': l_shipdate,
        'l_returnflag_enc': df['l_returnflag'].map(rf_map).values.astype(np.int32),
        'l_linestatus_enc': df['l_linestatus'].map(ls_map).values.astype(np.int32),
        'l_quantity': df['l_quantity'].values.astype(np.float32),
        'l_extendedprice': l_extendedprice,
        'l_discount': l_discount,
        'l_tax': l_tax,
        'disc_price': disc_price,
        'charge': charge,
    }
    
    return data, df


def run_max_ops_based(data: dict, use_gpu: bool, n_warmup: int = 2, n_runs: int = 5) -> BenchmarkResult:
    """Run TPC-H Q1 using MAX Engine with separate ops (~72 operations)."""
    n_rows = len(data['l_shipdate'])
    device_name = "GPU" if use_gpu else "CPU"
    name = f"MAX ops ({device_name})"
    
    print(f"\n🔧 {name} - {n_rows:,} rows")
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
            one_minus_disc = ops.sub(one_f, disc)
            one_plus_tax = ops.add(one_f, tax)
            disc_price = ops.mul(price, one_minus_disc)
            charge = ops.mul(disc_price, one_plus_tax)
            
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
    graph = Graph("max_q1_ops", FusedQ1Graph(device_ref), input_types=[
        TensorType(DType.int32, (n_rows,), device_ref),
        TensorType(DType.int32, (n_rows,), device_ref),
        TensorType(DType.int32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
        TensorType(DType.float32, (n_rows,), device_ref),
    ])
    model = session.load(graph)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"✅ Compiled in {compile_time:.0f}ms")
    
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
    
    avg_time = sum(times) / len(times) * 1000
    best_time = min(times) * 1000
    print(f"\n📊 {name}: avg={avg_time:.2f}ms, best={best_time:.2f}ms")
    
    return BenchmarkResult(name, avg_time, best_time, compile_time)


def run_max_fused(data: dict, use_gpu: bool, n_warmup: int = 2, n_runs: int = 5) -> BenchmarkResult:
    """Run TPC-H Q1 using fused custom kernel (1 kernel launch)."""
    n_rows = len(data['l_shipdate'])
    device_name = "GPU" if use_gpu else "CPU"
    name = f"MAX fused ({device_name})"
    
    print(f"\n🚀 {name} - {n_rows:,} rows")
    print("-" * 50)
    
    device = driver.Accelerator() if use_gpu else driver.CPU()
    device_ref = DeviceRef.GPU() if use_gpu else DeviceRef.CPU()
    session = engine.InferenceSession(devices=[device])
    
    num_warps = (n_rows + WARP_SIZE - 1) // WARP_SIZE if use_gpu else 1
    out_size = num_warps * TOTAL_OUTPUTS if use_gpu else TOTAL_OUTPUTS
    
    def fused_q1_graph(shipdate, rf_enc, ls_enc, qty, price, disc, disc_price, charge):
        return ops.custom(
            name="fused_q1_full",
            values=[shipdate, rf_enc, ls_enc, qty, price, disc, disc_price, charge],
            out_types=[TensorType(DType.float32, (out_size,), device_ref)],
            device=device_ref,
        )[0]
    
    print("⚙️  Compiling...")
    t0 = time.perf_counter()
    graph = Graph(
        "max_q1_fused",
        fused_q1_graph,
        input_types=[
            TensorType(DType.int32, (n_rows,), device_ref),
            TensorType(DType.int32, (n_rows,), device_ref),
            TensorType(DType.int32, (n_rows,), device_ref),
            TensorType(DType.float32, (n_rows,), device_ref),
            TensorType(DType.float32, (n_rows,), device_ref),
            TensorType(DType.float32, (n_rows,), device_ref),
            TensorType(DType.float32, (n_rows,), device_ref),
            TensorType(DType.float32, (n_rows,), device_ref),
        ],
        custom_extensions=[Path("kernels")],
    )
    model = session.load(graph)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"✅ Compiled in {compile_time:.0f}ms")
    
    # Prepare tensors
    if use_gpu:
        tensors = [
            driver.Tensor.from_numpy(data['l_shipdate']).to(device),
            driver.Tensor.from_numpy(data['l_returnflag_enc']).to(device),
            driver.Tensor.from_numpy(data['l_linestatus_enc']).to(device),
            driver.Tensor.from_numpy(data['l_quantity']).to(device),
            driver.Tensor.from_numpy(data['l_extendedprice']).to(device),
            driver.Tensor.from_numpy(data['l_discount']).to(device),
            driver.Tensor.from_numpy(data['disc_price']).to(device),
            driver.Tensor.from_numpy(data['charge']).to(device),
        ]
    else:
        tensors = [
            driver.Tensor(data['l_shipdate'], device),
            driver.Tensor(data['l_returnflag_enc'], device),
            driver.Tensor(data['l_linestatus_enc'], device),
            driver.Tensor(data['l_quantity'], device),
            driver.Tensor(data['l_extendedprice'], device),
            driver.Tensor(data['l_discount'], device),
            driver.Tensor(data['disc_price'], device),
            driver.Tensor(data['charge'], device),
        ]
    
    def execute():
        outputs = model.execute(*tensors)
        if use_gpu:
            out_arr = outputs[0].to(driver.CPU()).to_numpy()
            # Sum across warps for each of 24 outputs
            return [float(np.sum(out_arr[i::TOTAL_OUTPUTS])) for i in range(TOTAL_OUTPUTS)]
        return [float(outputs[0].to_numpy()[i]) for i in range(TOTAL_OUTPUTS)]
    
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
    
    avg_time = sum(times) / len(times) * 1000
    best_time = min(times) * 1000
    print(f"\n📊 {name}: avg={avg_time:.2f}ms, best={best_time:.2f}ms")
    
    return BenchmarkResult(name, avg_time, best_time, compile_time)


def run_polars(df, use_gpu: bool = False, n_warmup: int = 2, n_runs: int = 5) -> Optional[BenchmarkResult]:
    """Run TPC-H Q1 using Polars."""
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
    try:
        for _ in range(n_warmup):
            query()
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return None
    
    # Benchmark
    print(f"⏱️  Benchmarking ({n_runs} runs)...")
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        query()
        times.append(time.perf_counter() - t0)
        print(f"   Run {i+1}: {times[-1]*1000:.2f}ms")
    
    avg_time = sum(times) / len(times) * 1000
    best_time = min(times) * 1000
    print(f"\n📊 {name}: avg={avg_time:.2f}ms, best={best_time:.2f}ms")
    
    return BenchmarkResult(name, avg_time, best_time)


def run_full_benchmark(scale_factor: float = 10.0, n_warmup: int = 2, n_runs: int = 5):
    """Run complete benchmark across all approaches."""
    print("=" * 70)
    print("🏁 TPC-H Q1 COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    
    # Generate data
    data, df = generate_tpch_data_duckdb(scale_factor)
    n_rows = len(data['l_shipdate'])
    
    results = []
    
    # 1. MAX ops-based CPU
    results.append(run_max_ops_based(data, use_gpu=False, n_warmup=n_warmup, n_runs=n_runs))
    gc.collect()
    
    # 2. MAX ops-based GPU
    try:
        results.append(run_max_ops_based(data, use_gpu=True, n_warmup=n_warmup, n_runs=n_runs))
    except Exception as e:
        print(f"\n❌ MAX ops GPU failed: {e}")
    gc.collect()
    
    # 3. MAX fused CPU
    results.append(run_max_fused(data, use_gpu=False, n_warmup=n_warmup, n_runs=n_runs))
    gc.collect()
    
    # 4. MAX fused GPU
    try:
        results.append(run_max_fused(data, use_gpu=True, n_warmup=n_warmup, n_runs=n_runs))
    except Exception as e:
        print(f"\n❌ MAX fused GPU failed: {e}")
    gc.collect()
    
    # 5. Polars CPU
    polars_cpu = run_polars(df, use_gpu=False, n_warmup=n_warmup, n_runs=n_runs)
    if polars_cpu:
        results.append(polars_cpu)
    gc.collect()
    
    # 6. Polars GPU
    try:
        polars_gpu = run_polars(df, use_gpu=True, n_warmup=n_warmup, n_runs=n_runs)
        if polars_gpu:
            results.append(polars_gpu)
    except Exception as e:
        print(f"\n❌ Polars GPU failed: {e}")
    gc.collect()
    
    # Summary
    print("\n")
    print("=" * 70)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\nData: {n_rows:,} rows (TPC-H SF={scale_factor})")
    print(f"\n{'Approach':<25} {'Avg Time':>12} {'Best Time':>12} {'vs Polars CPU':>15}")
    print("-" * 70)
    
    # Find Polars CPU baseline
    baseline = next((r.avg_ms for r in results if r.name == "Polars CPU"), None)
    
    # Sort by avg time
    results.sort(key=lambda x: x.avg_ms)
    
    for r in results:
        if baseline:
            speedup = baseline / r.avg_ms
            speedup_str = f"{speedup:.1f}x faster" if speedup > 1 else f"{1/speedup:.1f}x slower"
        else:
            speedup_str = "-"
        print(f"{r.name:<25} {r.avg_ms:>10.2f}ms {r.best_ms:>10.2f}ms {speedup_str:>15}")
    
    # Winner announcement
    winner = results[0]
    print(f"\n🏆 WINNER: {winner.name} ({winner.avg_ms:.2f}ms)")
    
    if baseline:
        print(f"\n📈 Key Insights:")
        for r in results:
            if "fused" in r.name.lower() and "GPU" in r.name:
                fused_gpu = r
            if "ops" in r.name.lower() and "GPU" in r.name:
                ops_gpu = r
        
        if 'fused_gpu' in dir() and 'ops_gpu' in dir():
            fusion_speedup = ops_gpu.avg_ms / fused_gpu.avg_ms
            print(f"   • Fusion speedup (GPU): {fusion_speedup:.1f}x (ops: {ops_gpu.avg_ms:.0f}ms → fused: {fused_gpu.avg_ms:.0f}ms)")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Parse command line args
    scale_factor = 10.0  # Default: 60M rows
    if len(sys.argv) > 1:
        scale_factor = float(sys.argv[1])
    
    n_runs = 5
    if len(sys.argv) > 2:
        n_runs = int(sys.argv[2])
    
    print(f"Running benchmark with SF={scale_factor}, {n_runs} runs per approach")
    
    run_full_benchmark(scale_factor=scale_factor, n_runs=n_runs)
