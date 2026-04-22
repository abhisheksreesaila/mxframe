#!/usr/bin/env python3
"""
Simple 4-column TPC-H benchmark: Pandas | Polars | MX CPU | MX GPU

One warmup call (to prime AOT load + join cache), then N timed hot runs.
No cold/cache-clearing — AOT kernels are pre-compiled and ready.

Usage:
    pixi run python3 scripts/bench_simple.py [--rows N] [--runs N]
"""
import argparse
import os
import sys
import time
import faulthandler
import numpy as np
import pyarrow as pa
from max import driver as _driver

# Enable faulthandler so SIGILL/SIGSEGV produce a Python traceback in CI.
faulthandler.enable()

# ── Import all query functions from the existing benchmark file ────────────
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_tpch as bm

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


# ── Timing helpers ─────────────────────────────────────────────────────────

def _time(fn, runs: int) -> float:
    """Warmup once, then take the min over `runs` timed calls."""
    fn()  # warmup — primes join cache, GPU buffers, etc.
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _safe_gpu_count() -> int:
    try:
        return int(_driver.accelerator_count())
    except Exception:
        return 0


# ── Table printing ─────────────────────────────────────────────────────────

def _print_header():
    print(f"\n{'Query':<30} {'Pandas':>10} {'Polars':>10} {'MX CPU':>10} {'MX GPU':>10}")
    print(f"{'—'*30} {'—'*10} {'—'*10} {'—'*10} {'—'*10}")


def _print_row(label: str, pandas_ms, polars_ms, cpu_ms, gpu_ms):
    def fmt(v):
        return f"{v:>9.1f}ms" if v is not None else f"{'—':>10}"
    print(f"  {label:<28} {fmt(pandas_ms)} {fmt(polars_ms)} {fmt(cpu_ms)} {fmt(gpu_ms)}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simple 4-column TPC-H benchmark")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of timed hot runs per engine (min is reported)")
    parser.add_argument("--queries", type=str, default=None,
                        help="Comma-separated list of query numbers to run, e.g. 1,3,6,12")
    args = parser.parse_args()

    N    = args.rows
    RUNS = args.runs
    only = None
    if args.queries:
        only = set(int(q.strip()) for q in args.queries.split(","))

    GPU_READY = _safe_gpu_count() > 0

    print(f"\nSimple TPC-H Benchmark  ({N:,} rows, {RUNS} hot runs, min reported)")
    print(f"GPU: {'ready' if GPU_READY else 'not available'}\n")

    # ── Generate data ──────────────────────────────────────────────────────
    print("Generating data ...", flush=True)
    def _gen(label, fn):
        print(f"  {label} ...", flush=True)
        return fn()

    lineitem = _gen("lineitem", lambda: bm.make_lineitem(N))
    cust_q3, ord_q3, li_q3 = _gen("q3", bm.make_tpch_q3_tables)
    ord_q12, li_q12         = _gen("q12", bm.make_tpch_q12_tables)
    part_q14, li_q14        = _gen("q14", bm.make_tpch_q14_tables)
    nat_q5, cust_q5, ord_q5, li_q5 = _gen("q5", bm.make_tpch_q5_tables)
    nat_q10, cust_q10, ord_q10, li_q10 = _gen("q10", bm.make_tpch_q10_tables)
    nat_q7, sup_q7, cust_q7, ord_q7, li_q7 = _gen("q7", bm.make_tpch_q7_tables)
    nat_q8, reg_q8, part_q8, cust_q8, ord_q8, li_q8 = _gen("q8", bm.make_tpch_q8_tables)
    cust_q13, ord_q13       = _gen("q13", bm.make_tpch_q13_tables)
    part_q19, li_q19        = _gen("q19", bm.make_tpch_q19_tables)
    ord_q4, li_q4           = _gen("q4", bm.make_tpch_q4_tables)
    nat_q9, sup_q9, ps_q9, part_q9, ord_q9, li_q9 = _gen("q9", bm.make_tpch_q9_tables)
    nat_q11, sup_q11, ps_q11 = _gen("q11", bm.make_tpch_q11_tables)
    cust_q18, ord_q18, li_q18 = _gen("q18", bm.make_tpch_q18_tables)
    part_q16, sup_q16, ps_q16 = _gen("q16", bm.make_tpch_q16_tables)
    part_q17, li_q17          = _gen("q17", bm.make_tpch_q17_tables)
    nat_q2, reg_q2, part_q2, sup_q2, ps_q2 = _gen("q2", bm.make_tpch_q2_tables)
    sup_q15, li_q15           = _gen("q15", bm.make_tpch_q15_tables)
    nat_q20, part_q20, sup_q20, ps_q20, li_q20 = _gen("q20", bm.make_tpch_q20_tables)
    nat_q21, sup_q21, ord_q21, li_q21 = _gen("q21", bm.make_tpch_q21_tables)
    cust_q22, ord_q22         = _gen("q22", bm.make_tpch_q22_tables)
    print("done\n")

    def skip(n): return only is not None and n not in only

    rows = []  # (label, pandas_ms, polars_ms, cpu_ms, gpu_ms)

    def bench(label, qn, pd_fn, pl_fn, cpu_fn, gpu_fn):
        if skip(qn):
            return
        pandas_ms = _time(pd_fn, RUNS)
        polars_ms = _time(pl_fn, RUNS) if (POLARS_AVAILABLE and pl_fn) else None
        cpu_ms    = _time(cpu_fn, RUNS)
        gpu_ms    = None
        if GPU_READY and gpu_fn:
            try:
                gpu_ms = _time(gpu_fn, RUNS)
            except Exception as e:
                gpu_ms = None
        rows.append((label, pandas_ms, polars_ms, cpu_ms, gpu_ms))
        _print_row(label, pandas_ms, polars_ms, cpu_ms, gpu_ms)

    _print_header()

    bench("Q1  filter+groupby+8agg", 1,
        lambda: bm.run_q1_pandas(lineitem),
        lambda: bm.run_q1_polars(lineitem),
        lambda: bm.run_q1_mxframe(lineitem, device="cpu"),
        lambda: bm.run_q1_mxframe(lineitem, device="gpu"))

    bench("Q3  3-join+groupby", 3,
        lambda: bm.run_q3_pandas(cust_q3, ord_q3, li_q3),
        lambda: bm.run_q3_polars(cust_q3, ord_q3, li_q3),
        lambda: bm.run_q3_mxframe(cust_q3, ord_q3, li_q3, device="cpu"),
        lambda: bm.run_q3_mxframe(cust_q3, ord_q3, li_q3, device="gpu"))

    bench("Q4  EXISTS semi-join", 4,
        lambda: bm.run_q4_pandas(ord_q4, li_q4),
        lambda: bm.run_q4_polars(ord_q4, li_q4),
        lambda: bm.run_q4_mxframe(ord_q4, li_q4, device="cpu"),
        lambda: bm.run_q4_mxframe(ord_q4, li_q4, device="gpu"))

    bench("Q5  5-join+groupby", 5,
        lambda: bm.run_q5_pandas(nat_q5, cust_q5, ord_q5, li_q5),
        lambda: bm.run_q5_polars(nat_q5, cust_q5, ord_q5, li_q5),
        lambda: bm.run_q5_mxframe(nat_q5, cust_q5, ord_q5, li_q5, device="cpu"),
        lambda: bm.run_q5_mxframe(nat_q5, cust_q5, ord_q5, li_q5, device="gpu"))

    bench("Q6  filter+global sum", 6,
        lambda: bm.run_q6_pandas(lineitem),
        lambda: bm.run_q6_polars(lineitem),
        lambda: bm.run_q6_mxframe(lineitem, device="cpu"),
        lambda: bm.run_q6_mxframe(lineitem, device="gpu"))

    bench("Q7  nation join+groupby", 7,
        lambda: bm.run_q7_pandas(nat_q7, sup_q7, cust_q7, ord_q7, li_q7),
        lambda: bm.run_q7_polars(nat_q7, sup_q7, cust_q7, ord_q7, li_q7),
        lambda: bm.run_q7_mxframe(nat_q7, sup_q7, cust_q7, ord_q7, li_q7, device="cpu"),
        lambda: bm.run_q7_mxframe(nat_q7, sup_q7, cust_q7, ord_q7, li_q7, device="gpu"))

    bench("Q8  market share", 8,
        lambda: bm.run_q8_pandas(nat_q8, reg_q8, part_q8, cust_q8, ord_q8, li_q8),
        lambda: bm.run_q8_polars(nat_q8, reg_q8, part_q8, cust_q8, ord_q8, li_q8),
        lambda: bm.run_q8_mxframe(nat_q8, reg_q8, part_q8, cust_q8, ord_q8, li_q8, device="cpu"),
        lambda: bm.run_q8_mxframe(nat_q8, reg_q8, part_q8, cust_q8, ord_q8, li_q8, device="gpu"))

    bench("Q9  6-join+profit", 9,
        lambda: bm.run_q9_pandas(nat_q9, sup_q9, ps_q9, part_q9, ord_q9, li_q9),
        lambda: bm.run_q9_polars(nat_q9, sup_q9, ps_q9, part_q9, ord_q9, li_q9),
        lambda: bm.run_q9_mxframe(nat_q9, sup_q9, ps_q9, part_q9, ord_q9, li_q9, device="cpu"),
        lambda: bm.run_q9_mxframe(nat_q9, sup_q9, ps_q9, part_q9, ord_q9, li_q9, device="gpu"))

    bench("Q10 returned items", 10,
        lambda: bm.run_q10_pandas(nat_q10, cust_q10, ord_q10, li_q10),
        lambda: bm.run_q10_polars(nat_q10, cust_q10, ord_q10, li_q10),
        lambda: bm.run_q10_mxframe(nat_q10, cust_q10, ord_q10, li_q10, device="cpu"),
        lambda: bm.run_q10_mxframe(nat_q10, cust_q10, ord_q10, li_q10, device="gpu"))

    bench("Q11 HAVING threshold", 11,
        lambda: bm.run_q11_pandas(nat_q11, sup_q11, ps_q11),
        lambda: bm.run_q11_polars(nat_q11, sup_q11, ps_q11),
        lambda: bm.run_q11_mxframe(nat_q11, sup_q11, ps_q11, device="cpu"),
        lambda: bm.run_q11_mxframe(nat_q11, sup_q11, ps_q11, device="gpu"))

    bench("Q12 date filter+join", 12,
        lambda: bm.run_q12_pandas(ord_q12, li_q12),
        lambda: bm.run_q12_polars(ord_q12, li_q12),
        lambda: bm.run_q12_mxframe(ord_q12, li_q12, device="cpu"),
        lambda: bm.run_q12_mxframe(ord_q12, li_q12, device="gpu"))

    bench("Q13 LEFT JOIN cust orders", 13,
        lambda: bm.run_q13_pandas(cust_q13, ord_q13),
        lambda: bm.run_q13_polars(cust_q13, ord_q13),
        lambda: bm.run_q13_mxframe(cust_q13, ord_q13, device="cpu"),
        lambda: bm.run_q13_mxframe(cust_q13, ord_q13, device="gpu"))

    bench("Q14 promo revenue", 14,
        lambda: bm.run_q14_pandas(part_q14, li_q14),
        lambda: bm.run_q14_polars(part_q14, li_q14),
        lambda: bm.run_q14_mxframe(part_q14, li_q14, device="cpu"),
        lambda: bm.run_q14_mxframe(part_q14, li_q14, device="gpu"))

    bench("Q15 argmax supplier", 15,
        lambda: bm.run_q15_pandas(sup_q15, li_q15),
        lambda: bm.run_q15_polars(sup_q15, li_q15),
        lambda: bm.run_q15_mxframe(sup_q15, li_q15, device="cpu"),
        lambda: bm.run_q15_mxframe(sup_q15, li_q15, device="gpu"))

    bench("Q16 distinct suppkey count", 16,
        lambda: bm.run_q16_pandas(part_q16, sup_q16, ps_q16),
        lambda: bm.run_q16_polars(part_q16, sup_q16, ps_q16),
        lambda: bm.run_q16_mxframe(part_q16, sup_q16, ps_q16, device="cpu"),
        lambda: bm.run_q16_mxframe(part_q16, sup_q16, ps_q16, device="gpu"))

    bench("Q17 2-pass avg qty", 17,
        lambda: bm.run_q17_pandas(part_q17, li_q17),
        lambda: bm.run_q17_polars(part_q17, li_q17),
        lambda: bm.run_q17_mxframe(part_q17, li_q17, device="cpu"),
        lambda: bm.run_q17_mxframe(part_q17, li_q17, device="gpu"))

    bench("Q18 large vol customers", 18,
        lambda: bm.run_q18_pandas(cust_q18, ord_q18, li_q18),
        lambda: bm.run_q18_polars(cust_q18, ord_q18, li_q18),
        lambda: bm.run_q18_mxframe(cust_q18, ord_q18, li_q18, device="cpu"),
        lambda: bm.run_q18_mxframe(cust_q18, ord_q18, li_q18, device="gpu"))

    bench("Q19 discounted revenue", 19,
        lambda: bm.run_q19_pandas(part_q19, li_q19),
        lambda: bm.run_q19_polars(part_q19, li_q19),
        lambda: bm.run_q19_mxframe(part_q19, li_q19, device="cpu"),
        lambda: bm.run_q19_mxframe(part_q19, li_q19, device="gpu"))

    bench("Q20 semi-join chain", 20,
        lambda: bm.run_q20_pandas(nat_q20, part_q20, sup_q20, ps_q20, li_q20),
        lambda: bm.run_q20_polars(nat_q20, part_q20, sup_q20, ps_q20, li_q20),
        lambda: bm.run_q20_mxframe(nat_q20, part_q20, sup_q20, ps_q20, li_q20, device="cpu"),
        lambda: bm.run_q20_mxframe(nat_q20, part_q20, sup_q20, ps_q20, li_q20, device="gpu"))

    bench("Q21 EXISTS+NOT EXISTS", 21,
        lambda: bm.run_q21_pandas(nat_q21, sup_q21, ord_q21, li_q21),
        lambda: bm.run_q21_polars(nat_q21, sup_q21, ord_q21, li_q21),
        lambda: bm.run_q21_mxframe(nat_q21, sup_q21, ord_q21, li_q21, device="cpu"),
        lambda: bm.run_q21_mxframe(nat_q21, sup_q21, ord_q21, li_q21, device="gpu"))

    bench("Q22 phone code anti-join", 22,
        lambda: bm.run_q22_pandas(cust_q22, ord_q22),
        lambda: bm.run_q22_polars(cust_q22, ord_q22),
        lambda: bm.run_q22_mxframe(cust_q22, ord_q22, device="cpu"),
        lambda: bm.run_q22_mxframe(cust_q22, ord_q22, device="gpu"))

    # ── Summary vs Polars ──────────────────────────────────────────────────
    if POLARS_AVAILABLE and rows:
        print(f"\n{'Query':<30} {'MX CPU vs Polars':>18} {'MX GPU vs Polars':>18}")
        print(f"{'—'*30} {'—'*18} {'—'*18}")
        for label, pandas_ms, polars_ms, cpu_ms, gpu_ms in rows:
            if polars_ms is None or polars_ms == 0:
                continue
            def ratio_str(ms):
                if ms is None:
                    return f"{'—':>18}"
                r = ms / polars_ms
                s = f"{1/r:.2f}x faster" if r < 1 else f"{r:.2f}x slower"
                return f"{s:>18}"
            print(f"  {label:<28} {ratio_str(cpu_ms)} {ratio_str(gpu_ms)}")

    print(f"\n{'='*70}")
    print("  All times in ms (min over hot runs).")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
