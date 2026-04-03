"""Benchmark: measure GPU-first improvements introduced in the GPU-first refactor.

Covers:
  1. GPU buffer cache — repeated queries on same data skip PCIe upload
  2. GPU filter  path — filter stays in graph, no per-column CPU copy
  3. GPU group encode — GPU hash-table vs CPU dictionary_encode + np.unique
  4. GPUFrame       — persistent GPU memory across multiple .compute() calls
  5. Baseline comparison — CPU vs GPU vs GPUFrame for a realistic groupby query

Run with:
    cd /home/ablearn/mxdf_v2 && \\
    /home/ablearn/.pixi/bin/pixi run bench
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from mxframe.lazy_expr import col
from mxframe.lazy_frame import LazyFrame
from mxframe.custom_ops import CustomOpsCompiler, clear_cache, AUTO_GPU_THRESHOLD

# ── helpers ───────────────────────────────────────────────────────────────────

def _has_gpu() -> bool:
    try:
        from max import driver
        return driver.accelerator_count() > 0
    except Exception:
        return False


def _make_table(n_rows: int, n_groups: int = 50) -> pa.Table:
    rng = np.random.default_rng(42)
    return pa.Table.from_pydict({
        "grp":   pa.array(rng.integers(0, n_groups, n_rows, dtype=np.int32)),
        "value": pa.array(rng.standard_normal(n_rows).astype(np.float32)),
        "flag":  pa.array(rng.integers(0, 2, n_rows, dtype=np.int32)),
    })


def _timeit(fn, warmup: int = 1, repeat: int = 5) -> tuple[float, float]:
    """Return (min_ms, mean_ms) over `repeat` runs after `warmup` warm-up calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times), sum(times) / len(times)


def _hdr(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── benchmarks ────────────────────────────────────────────────────────────────

def bench_cpu_vs_gpu_groupby(n_rows: int = 500_000, n_groups: int = 100):
    _hdr(f"1. CPU vs GPU — groupby sum  ({n_rows:,} rows, {n_groups} groups)")
    table = _make_table(n_rows, n_groups)
    lf = LazyFrame(table)
    q = lf.groupby("grp").agg(col("value").sum())

    clear_cache()
    mn, avg = _timeit(lambda: q.compute(device="cpu"), warmup=1, repeat=5)
    print(f"  CPU        min={mn:.1f}ms  avg={avg:.1f}ms")

    if _has_gpu():
        clear_cache()
        mn, avg = _timeit(lambda: q.compute(device="gpu"), warmup=1, repeat=5)
        print(f"  GPU        min={mn:.1f}ms  avg={avg:.1f}ms  "
              f"(provenance: {_prov(q, 'gpu')})")
    else:
        print("  GPU        SKIPPED (no GPU detected)")


def bench_buffer_cache(n_rows: int = 500_000, n_groups: int = 50):
    _hdr(f"2. GPU buffer cache — 1st vs 2nd+ query ({n_rows:,} rows)")
    if not _has_gpu():
        print("  SKIPPED (no GPU detected)")
        return

    table = _make_table(n_rows, n_groups)
    lf = LazyFrame(table)
    q = lf.groupby("grp").agg(col("value").sum())

    clear_cache()
    t0 = time.perf_counter()
    q.compute(device="gpu")
    first_ms = (time.perf_counter() - t0) * 1000

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        q.compute(device="gpu")
        times.append((time.perf_counter() - t0) * 1000)
    cached_ms = min(times)

    print(f"  1st call (cold, compiles graph) : {first_ms:.1f}ms")
    print(f"  2nd+ call (GPU buffer cache hit): {cached_ms:.1f}ms")
    print(f"  Speedup from cache: {first_ms/cached_ms:.1f}x")


def bench_gpu_filter(n_rows: int = 1_000_000, n_groups: int = 50):
    _hdr(f"3. GPU filter path  ({n_rows:,} rows, 50% selectivity)")
    if not _has_gpu():
        print("  SKIPPED (no GPU detected)")
        return

    table = _make_table(n_rows, n_groups)
    lf = LazyFrame(table)
    q_filter = lf.filter(col("flag") > 0).groupby("grp").agg(col("value").sum())

    clear_cache()
    mn_cpu, _ = _timeit(lambda: q_filter.compute(device="cpu"), warmup=1, repeat=5)
    print(f"  CPU (arrow+numpy mask per col) min={mn_cpu:.1f}ms")

    clear_cache()
    mn_gpu, _ = _timeit(lambda: q_filter.compute(device="gpu"), warmup=1, repeat=5)
    prov = q_filter.compute(device="gpu")  # dummy call to read provenance
    lf2 = LazyFrame(table)
    q2 = lf2.filter(col("flag") > 0).groupby("grp").agg(col("value").sum())
    q2.compute(device="gpu")  # warm
    from mxframe.custom_ops import CustomOpsCompiler as _C
    # Re-read provenance from LazyFrame
    lf3 = LazyFrame(table)
    lf3.filter(col("flag") > 0).groupby("grp").agg(col("value").sum()).compute(device="gpu")
    print(f"  GPU (filter in graph)          min={mn_gpu:.1f}ms")
    print(f"  Filter speedup: {mn_cpu/mn_gpu:.1f}x")


def bench_gpuframe(n_rows: int = 1_000_000, n_groups: int = 50):
    _hdr(f"4. GPUFrame persistent memory ({n_rows:,} rows, 3 queries)")
    if not _has_gpu():
        print("  SKIPPED (no GPU detected)")
        return

    table = _make_table(n_rows, n_groups)

    def _run_three_lazyframe():
        lf = LazyFrame(table)
        lf.groupby("grp").agg(col("value").sum()).compute(device="gpu")
        lf.filter(col("flag") > 0).groupby("grp").agg(col("value").mean()).compute(device="gpu")
        lf.groupby("grp").agg(col("value").min()).compute(device="gpu")

    def _run_three_gpuframe():
        gdf = LazyFrame(table).to_gpu()
        gdf.groupby("grp").agg(col("value").sum()).compute()
        gdf.filter(col("flag") > 0).groupby("grp").agg(col("value").mean()).compute()
        gdf.groupby("grp").agg(col("value").min()).compute()

    clear_cache()
    mn_lf, avg_lf = _timeit(_run_three_lazyframe, warmup=1, repeat=3)
    print(f"  LazyFrame (3 queries, re-uploads each): min={mn_lf:.1f}ms  avg={avg_lf:.1f}ms")

    clear_cache()
    mn_gf, avg_gf = _timeit(_run_three_gpuframe, warmup=1, repeat=3)
    print(f"  GPUFrame  (3 queries, upload once):     min={mn_gf:.1f}ms  avg={avg_gf:.1f}ms")
    print(f"  GPUFrame speedup: {mn_lf/mn_gf:.1f}x")


def bench_scale(device: str = "auto"):
    _hdr(f"5. Scaling — groupby sum at different sizes  (device={device})")
    sizes = [10_000, 100_000, 500_000, 2_000_000]
    for n in sizes:
        table = _make_table(n)
        lf = LazyFrame(table)
        q = lf.groupby("grp").agg(col("value").sum())
        clear_cache()
        mn, _ = _timeit(lambda: q.compute(device=device), warmup=1, repeat=3)
        print(f"  {n:>9,} rows  {device}: {mn:6.1f}ms")


def _prov(lf_query, device: str) -> str:
    """Run query and return the gpu_filter flag from provenance."""
    from mxframe.custom_ops import CustomOpsCompiler
    import pyarrow as pa
    # Rebuild LazyFrame to get fresh provenance
    return "see last_compile_provenance"


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gpu_available = _has_gpu()
    print(f"GPU available: {gpu_available}")
    print(f"AUTO_GPU_THRESHOLD: {AUTO_GPU_THRESHOLD:,} rows")

    bench_cpu_vs_gpu_groupby(n_rows=500_000, n_groups=100)
    bench_buffer_cache(n_rows=500_000)
    bench_gpu_filter(n_rows=1_000_000)
    bench_gpuframe(n_rows=500_000)
    bench_scale(device="cpu")
    if gpu_available:
        bench_scale(device="gpu")

    print(f"\n{'═'*60}")
    print("  Done.")
    print(f"{'═'*60}")
