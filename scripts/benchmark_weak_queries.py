"""Benchmark optimized Q4/Q6/Q13/Q21 paths against Polars and optional cuDF."""

import argparse
import statistics
import time

import numpy as np

from benchmark_tpch import (
    Q4_DATE_HI,
    Q4_DATE_LO,
    Q6_DATE_HI,
    Q6_DATE_LO,
    Q6_DISC_HI,
    Q6_DISC_LO,
    Q6_QTY_HI,
    Q21_NATION_KEY,
    make_lineitem,
    make_tpch_q4_tables,
    make_tpch_q13_tables,
    make_tpch_q21_tables,
    run_q4_mxframe,
    run_q4_polars,
    run_q6_mxframe,
    run_q6_polars,
    run_q13_mxframe,
    run_q13_polars,
    run_q21_mxframe,
    run_q21_polars,
)

try:
    import cudf
    import cupy
    CUDF_AVAILABLE = True
except ImportError:
    cudf = None
    cupy = None
    CUDF_AVAILABLE = False


def median_ms(fn, runs: int) -> float:
    fn()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        if cupy is not None:
            cupy.cuda.get_current_stream().synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def q6_cudf(table):
    frame = cudf.DataFrame.from_arrow(table)
    filtered = frame[
        (frame.l_shipdate >= Q6_DATE_LO)
        & (frame.l_shipdate < Q6_DATE_HI)
        & (frame.l_discount >= Q6_DISC_LO)
        & (frame.l_discount <= Q6_DISC_HI)
        & (frame.l_quantity < Q6_QTY_HI)
    ]
    return (filtered.l_extendedprice * filtered.l_discount).sum()


def q4_cudf(orders, lineitem):
    order_frame = cudf.DataFrame.from_arrow(orders)
    lineitem_frame = cudf.DataFrame.from_arrow(lineitem)
    valid = lineitem_frame[lineitem_frame.l_commitdate < lineitem_frame.l_receiptdate]
    valid = valid[["l_orderkey"]].drop_duplicates()
    selected = order_frame[
        (order_frame.o_orderdate >= Q4_DATE_LO)
        & (order_frame.o_orderdate < Q4_DATE_HI)
    ]
    joined = selected.merge(valid, left_on="o_orderkey", right_on="l_orderkey", how="inner")
    return joined.groupby("o_orderpriority").size()


def q13_cudf(customer, orders):
    customer_frame = cudf.DataFrame.from_arrow(customer)
    order_frame = cudf.DataFrame.from_arrow(orders)
    joined = customer_frame.merge(
        order_frame, left_on="c_custkey", right_on="o_custkey", how="left")
    counts = joined.groupby("c_custkey", dropna=False).o_orderkey.count()
    return counts.value_counts()


def q21_cudf(supplier, orders, lineitem):
    supplier_frame = cudf.DataFrame.from_arrow(supplier)
    order_frame = cudf.DataFrame.from_arrow(orders)
    lineitem_frame = cudf.DataFrame.from_arrow(lineitem)
    target = supplier_frame[supplier_frame.s_nationkey == Q21_NATION_KEY][["s_suppkey"]]
    finished = order_frame[order_frame.o_orderstatus == "F"][["o_orderkey"]]
    late = lineitem_frame[lineitem_frame.l_receiptdate > lineitem_frame.l_commitdate]
    distinct_pairs = late[["l_orderkey", "l_suppkey"]].drop_duplicates()
    one_supplier = distinct_pairs.groupby("l_orderkey").size()
    one_supplier = one_supplier[one_supplier == 1].index.to_frame(index=False)
    result = late.merge(finished, left_on="l_orderkey", right_on="o_orderkey")
    result = result.merge(one_supplier, on="l_orderkey")
    result = result.merge(target, left_on="l_suppkey", right_on="s_suppkey")
    return result.groupby("l_suppkey").size().nlargest(100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=(1, 10), default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    scale = args.scale

    lineitem = make_lineitem(1_000_000 * scale)
    orders4, lineitem4 = make_tpch_q4_tables(150_000 * scale, 600_000 * scale)
    customer13, orders13 = make_tpch_q13_tables(150_000 * scale, 600_000 * scale)
    _, supplier21, orders21, lineitem21 = make_tpch_q21_tables(
        2_000 * scale, 60_000 * scale, 200_000 * scale)

    rows = [
        ("Q4", lambda: run_q4_mxframe(orders4, lineitem4, "gpu"),
         lambda: run_q4_polars(orders4, lineitem4),
         lambda: q4_cudf(orders4, lineitem4)),
        ("Q6", lambda: run_q6_mxframe(lineitem, "gpu"),
         lambda: run_q6_polars(lineitem), lambda: q6_cudf(lineitem)),
        ("Q13", lambda: run_q13_mxframe(customer13, orders13, "gpu"),
         lambda: run_q13_polars(customer13, orders13),
         lambda: q13_cudf(customer13, orders13)),
        ("Q21", lambda: run_q21_mxframe(None, supplier21, orders21, lineitem21, "gpu"),
         lambda: run_q21_polars(None, supplier21, orders21, lineitem21),
         lambda: q21_cudf(supplier21, orders21, lineitem21)),
    ]

    print(f"Weak-query benchmark, scale={scale}, median of {args.runs} warm runs")
    print(f"RAPIDS cuDF: {'available' if CUDF_AVAILABLE else 'unavailable (not installed)'}")
    print(f"{'query':<8}{'MX GPU':>12}{'Polars':>12}{'cuDF':>12}")
    for query, mx_fn, polars_fn, cudf_fn in rows:
        mx_time = median_ms(mx_fn, args.runs)
        polars_time = median_ms(polars_fn, args.runs)
        cudf_text = "N/A"
        if CUDF_AVAILABLE:
            cudf_text = f"{median_ms(cudf_fn, args.runs):.3f}ms"
        print(f"{query:<8}{mx_time:>11.3f}ms{polars_time:>11.3f}ms{cudf_text:>12}")


if __name__ == "__main__":
    main()