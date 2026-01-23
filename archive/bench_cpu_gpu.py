"""Benchmark CPU vs GPU vs NumPy for max_sum."""
import numpy as np
import time
from mxframe import max_sum
from mxframe.max_ops import MAXSession

def bench(fn, *args, warmup=3, runs=10):
    """Run function and return average time in ms."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)

sizes = [100_000, 1_000_000, 10_000_000]

print("=" * 60)
print("Benchmark: max_sum CPU vs GPU vs NumPy")
print("=" * 60)

for size in sizes:
    arr = np.random.randn(size).astype(np.float32)
    MAXSession.reset()  # Clear cache for fair comparison
    
    # NumPy baseline
    t_numpy = bench(lambda: arr.sum())
    
    # MAX CPU
    t_cpu = bench(lambda: max_sum(arr, device="cpu"))
    
    # MAX GPU
    t_gpu = bench(lambda: max_sum(arr, device="gpu"))
    
    print(f"\nSize: {size:>12,} elements")
    print(f"  NumPy:    {t_numpy:>8.3f} ms")
    print(f"  MAX CPU:  {t_cpu:>8.3f} ms  ({t_numpy/t_cpu:.2f}x vs NumPy)")
    print(f"  MAX GPU:  {t_gpu:>8.3f} ms  ({t_numpy/t_gpu:.2f}x vs NumPy)")

print("\n" + "=" * 60)
print("Note: GPU overhead dominates at small sizes. GPU wins at large sizes.")
print("=" * 60)
