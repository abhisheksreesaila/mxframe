"""Benchmark MXFrame AOT GPU UTF-8 predicates against Arrow and Polars."""

import argparse
import statistics
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl

from mxframe.aot_kernels import AOTKernelsGPU


def median_ms(fn, runs: int) -> float:
    fn()
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    values = rng.choice(
        [
            "PROMO forest green", "PROMO BRASS", "BRUSHED COPPER",
            "STANDARD green TIN", "ECONOMY NICKEL",
        ],
        size=args.rows,
    ).tolist()
    arrow_array = pa.array(values)
    polars_series = pl.Series("value", values)
    gpu = AOTKernelsGPU()

    benchmarks = [
        (
            "equal('PROMO BRASS')",
            lambda: pc.equal(arrow_array, "PROMO BRASS"),
            lambda: polars_series == "PROMO BRASS",
            lambda: gpu.utf8_equal_mask(arrow_array, "PROMO BRASS"),
        ),
        (
            "isin(2 literals)",
            lambda: pc.is_in(arrow_array, value_set=pa.array(["PROMO BRASS", "ECONOMY NICKEL"])),
            lambda: polars_series.is_in(["PROMO BRASS", "ECONOMY NICKEL"]),
            lambda: gpu.utf8_isin_mask(arrow_array, ["PROMO BRASS", "ECONOMY NICKEL"]),
        ),
        (
            "startswith('PROMO')",
            lambda: pc.starts_with(arrow_array, pattern="PROMO"),
            lambda: polars_series.str.starts_with("PROMO"),
            lambda: gpu.utf8_startswith_mask(arrow_array, "PROMO"),
        ),
        (
            "contains('green')",
            lambda: pc.match_substring(arrow_array, pattern="green"),
            lambda: polars_series.str.contains("green", literal=True),
            lambda: gpu.utf8_contains_mask(arrow_array, "green"),
        ),
    ]

    print(f"UTF-8 predicates: {args.rows:,} rows, median of {args.runs}")
    print(f"{'operation':<24} {'Arrow':>10} {'Polars':>10} {'MX GPU':>10}")
    for name, arrow_fn, polars_fn, gpu_fn in benchmarks:
        arrow_ms = median_ms(arrow_fn, args.runs)
        polars_ms = median_ms(polars_fn, args.runs)
        gpu_ms = median_ms(gpu_fn, args.runs)
        print(f"{name:<24} {arrow_ms:>9.3f}ms {polars_ms:>9.3f}ms {gpu_ms:>9.3f}ms")


if __name__ == "__main__":
    main()