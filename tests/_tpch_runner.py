"""Run one omxcore TPC-H benchmark query in a fresh process and save the result.

    python _tpch_runner.py <data_dir> <query number> <device> <out.arrow>

A fresh process per query keeps a GPU fault or a leak in one query from taking down the
others (see test_tpch_duckdb.py).  Scalar results (Q17) are saved as a one-row table.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conftest  # noqa: F401,E402  (makes `import omxcore` resolve to this checkout)

QUERIES = {
    1: ["lineitem"], 2: ["nation", "region", "part", "supplier", "partsupp"],
    3: ["customer", "orders", "lineitem"], 4: ["orders", "lineitem"],
    5: ["nation", "customer", "orders", "lineitem"], 6: ["lineitem"],
    7: ["nation", "supplier", "customer", "orders", "lineitem"],
    8: ["nation", "region", "part", "customer", "orders", "lineitem"],
    9: ["nation", "supplier", "partsupp", "part", "orders", "lineitem"],
    10: ["nation", "customer", "orders", "lineitem"], 11: ["nation", "supplier", "partsupp"],
    12: ["orders", "lineitem"], 13: ["customer", "orders"], 14: ["part", "lineitem"],
    15: ["supplier", "lineitem"], 16: ["part", "supplier", "partsupp"],
    17: ["part", "lineitem"], 18: ["customer", "orders", "lineitem"],
    19: ["part", "lineitem"], 20: ["nation", "part", "supplier", "partsupp", "lineitem"],
    21: ["nation", "supplier", "orders", "lineitem"], 22: ["customer", "orders"],
}


# tables a query takes as keywords, beyond the positional ones above
KEYWORD_TABLES = {5: ["supplier"], 8: ["supplier"]}
SCALAR_NAME = {6: "revenue", 17: "avg_yearly", 19: "revenue"}


def run_query(data_dir, q, device):
    import pyarrow as pa
    import omxcore as mx
    sys.path.insert(0, str(HERE.parent / "scripts"))
    import benchmark_tpch as bm
    dictionary = False if device in ("pandas", "polars") else "auto"   # baselines get plain strings
    tables = [mx.read_parquet_table(Path(data_dir) / f"{n}.parquet", dictionary=dictionary) for n in QUERIES[q]]
    extra = {n: mx.read_parquet_table(Path(data_dir) / f"{n}.parquet", dictionary=dictionary)
             for n in KEYWORD_TABLES.get(q, [])}
    if device in ("pandas", "polars"):            # the baseline engines take no device
        out = getattr(bm, f"run_q{q}_{device}")(*tables, **extra)
        if hasattr(out, "to_arrow"):              # polars DataFrame
            out = out.to_arrow()
        elif hasattr(out, "to_dict"):             # pandas DataFrame
            out = pa.Table.from_pandas(out.reset_index(drop=True), preserve_index=False)
    else:
        out = getattr(bm, f"run_q{q}_mxframe")(*tables, device=device, **extra)
    if not isinstance(out, pa.Table):
        out = pa.table({SCALAR_NAME.get(q, "value"): [float(out)]})
    return out


if __name__ == "__main__":
    import pyarrow as pa
    data_dir, q, device, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    result = run_query(data_dir, q, device)
    with pa.OSFile(out_path, "wb") as sink, pa.ipc.new_file(sink, result.schema) as w:
        w.write_table(result)
