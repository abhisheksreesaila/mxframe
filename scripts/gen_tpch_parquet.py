#!/usr/bin/env python3
"""
Generate official TPC-H data using DuckDB's built-in dbgen extension.

Writes 8 Parquet files (one per TPC-H table) to an output directory.
Date columns (l_shipdate, o_orderdate, …) are cast to int32 (days since
1970-01-01) so they are directly compatible with the existing benchmark
query functions in benchmark_tpch.py.

Usage:
    pixi run python3 scripts/gen_tpch_parquet.py --sf 1
    pixi run python3 scripts/gen_tpch_parquet.py --sf 0.1 --out-dir data/tpch_sf0.1

Scale factor sizes (approximate):
    SF 0.01  →    60 K lineitem rows   |  quick smoke test  | ~  2 MB
    SF 0.1   →   600 K lineitem rows   |  fast laptop run   | ~ 20 MB
    SF 1     →   6.0 M lineitem rows   |  standard bench    | ~200 MB
    SF 10    →  60 M  lineitem rows    |  large bench       |   ~2 GB

Requirements:
    pip install duckdb  (or: pixi add duckdb)

Data lineage:
    DuckDB's dbgen is a port of the official TPC-H data generator.
    The distributions (uniform, Zipfian, string vocabulary) exactly match
    the TPC-H specification v3.0.1. This data may be used to reproduce
    and publish benchmark results.

    Reference: https://duckdb.org/docs/extensions/tpch
    TPC-H spec: https://www.tpc.org/tpch/
"""
import argparse
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# TPC-H table definitions: which columns need type conversion
# ---------------------------------------------------------------------------
_DATE_COLUMNS: dict[str, list[str]] = {
    "lineitem": ["l_shipdate", "l_commitdate", "l_receiptdate"],
    "orders":   ["o_orderdate"],
    # customer, part, supplier, partsupp, nation, region — no dates
}

# All DECIMAL columns in TPC-H → cast to float32 to match benchmark functions.
# DuckDB generates DECIMAL(15,2) for all monetary/quantity fields.
_DECIMAL_TO_F32: dict[str, list[str]] = {
    "lineitem":  ["l_quantity", "l_extendedprice", "l_discount", "l_tax"],
    "orders":    ["o_totalprice"],
    "customer":  ["c_acctbal"],
    "supplier":  ["s_acctbal"],
    "partsupp":  ["ps_availqty", "ps_supplycost"],
    "part":      [],
    "nation":    [],
    "region":    [],
}

# Column name → int32 alias in the SELECT so DuckDB returns it as a plain
# integer (days since 1970-01-01).  We do the conversion in Python via
# PyArrow cast so we don't rely on a specific DuckDB date function name.
def _col_exprs(table_name: str, duckdb_cols: list[str]) -> str:
    # Just select all columns by name; date casting is done post-hoc in Python.
    return ", ".join(duckdb_cols)


def generate(sf: float, out_dir: str) -> None:
    try:
        import duckdb
    except ImportError:
        print(
            "ERROR: duckdb is not installed.\n"
            "  pixi add duckdb        (recommended)\n"
            "  pip install duckdb     (alternative)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Connecting to DuckDB …")
    con = duckdb.connect()

    # Load the TPC-H extension and generate all 8 tables in memory
    print(f"Running dbgen at SF={sf} (this may take a moment) …")
    t0 = time.perf_counter()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"  dbgen done in {gen_ms:.0f} ms")

    tpch_tables = [
        "lineitem", "orders", "customer",
        "part", "supplier", "partsupp",
        "nation", "region",
    ]

    for tname in tpch_tables:
        t_start = time.perf_counter()

        # Get column list from DuckDB catalog
        cols_result = con.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{tname}' ORDER BY ordinal_position"
        ).fetchall()
        col_names = [r[0] for r in cols_result]

        # Build SELECT with all columns (date casting done in Python below)
        select_expr = _col_exprs(tname, col_names)
        # .arrow() returns RecordBatchReader in some DuckDB versions; materialise it.
        raw_result = con.execute(f"SELECT {select_expr} FROM {tname}").arrow()
        arrow_table = raw_result.read_all() if hasattr(raw_result, "read_all") else raw_result

        # Cast date32 columns → int32 (days since 1970-01-01) so they are
        # compatible with the benchmark query functions that compare dates as
        # plain integers.  PyArrow date32 internally stores days since epoch,
        # so this is a zero-copy type reinterpretation.
        import pyarrow.compute as pc
        for dcol in _DATE_COLUMNS.get(tname, []):
            if dcol in arrow_table.schema.names:
                field_idx = arrow_table.schema.get_field_index(dcol)
                col_type = arrow_table.schema.field(field_idx).type
                if col_type != pa.int32():
                    # date32 → int32: cast via view (days since epoch unchanged)
                    raw = arrow_table.column(dcol).cast(pa.int32())
                    arrow_table = arrow_table.set_column(field_idx, dcol, raw)

        # Cast DECIMAL columns → float32 so they match what the benchmark
        # functions expect (all arithmetic uses float32 kernels).
        for fcol in _DECIMAL_TO_F32.get(tname, []):
            if fcol in arrow_table.schema.names:
                field_idx = arrow_table.schema.get_field_index(fcol)
                if arrow_table.schema.field(field_idx).type != pa.float32():
                    raw = arrow_table.column(fcol).cast(pa.float32())
                    arrow_table = arrow_table.set_column(field_idx, fcol, raw)

        out_path = os.path.join(out_dir, f"{tname}.parquet")
        pq.write_table(arrow_table, out_path, compression="snappy")

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        nrows = len(arrow_table)
        print(f"  {tname:<12} {nrows:>10,} rows  →  {out_path}  ({elapsed_ms:.0f} ms)")

    con.close()
    print(f"\nAll 8 tables written to: {out_dir}")
    print(f"Run the real-data benchmark with:")
    print(f"  pixi run python3 scripts/bench_real_tpch.py --data-dir {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate official TPC-H Parquet files using DuckDB dbgen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sf", type=float, default=1.0,
        help="TPC-H scale factor (default: 1.0 = ~6M lineitem rows, ~200 MB)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: data/tpch_sf<SF>)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or f"data/tpch_sf{args.sf}".replace(".0", "")
    # Resolve relative to repo root (parent of scripts/)
    if not os.path.isabs(out_dir):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(repo_root, out_dir)

    generate(args.sf, out_dir)


if __name__ == "__main__":
    main()
