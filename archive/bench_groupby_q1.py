"""TPC-H Q1 style benchmark with MXFrame GroupBy."""
import numpy as np
import time
from mxframe import MXFrame, group_by, agg_sum, agg_mean, agg_count

np.random.seed(42)

def bench(fn, warmup=2, runs=5):
    """Run function and return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


# TPC-H Q1 pattern: group by (returnflag, linestatus)
for N in [10_000, 100_000, 1_000_000]:
    print(f"\n{'='*60}")
    print(f"N = {N:,} rows")
    print(f"{'='*60}")
    
    # Create lineitem-like data
    flags = np.random.choice(['A', 'N', 'R'], N)
    statuses = np.random.choice(['F', 'O'], N)
    quantities = np.random.randint(1, 50, N).astype(np.float32)
    prices = np.random.uniform(1.0, 100.0, N).astype(np.float32)
    
    lineitem = MXFrame({
        'l_returnflag': flags.tolist(),
        'l_linestatus': statuses.tolist(),
        'l_quantity': quantities.tolist(),
        'l_extendedprice': prices.tolist(),
    })
    
    # CPU benchmark
    def run_cpu():
        return lineitem.group_by('l_returnflag', 'l_linestatus', device='cpu').agg(
            sum_qty=agg_sum('l_quantity'),
            sum_price=agg_sum('l_extendedprice'),
            count_order=agg_count('l_quantity'),
        )
    
    # GPU benchmark
    def run_gpu():
        return lineitem.group_by('l_returnflag', 'l_linestatus', device='gpu').agg(
            sum_qty=agg_sum('l_quantity'),
            sum_price=agg_sum('l_extendedprice'),
            count_order=agg_count('l_quantity'),
        )
    
    cpu_time = bench(run_cpu)
    gpu_time = bench(run_gpu)
    
    print(f"CPU: {cpu_time:.2f} ms")
    print(f"GPU: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Show result for verification
    result = run_cpu()
    print(f"\nResult ({result.num_rows} groups):")
    print(result.to_pandas())
