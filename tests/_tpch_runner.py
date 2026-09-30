"""Run one mxframe TPC-H benchmark query in a fresh process and save the result.

    python _tpch_runner.py <data_dir> <query number> <device> <out.arrow>

A fresh process per query keeps a GPU fault or a leak in one query from taking down the
others (see test_tpch_duckdb.py).  Scalar results (Q17) are saved as a one-row table.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conftest  # noqa: F401,E402  (makes `import mxframe` resolve to this checkout)

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


def run_query(data_dir, q, device):
    import pyarrow as pa
    import mxframe as mx
    sys.path.insert(0, str(HERE.parent / "scripts"))
    import benchmark_tpch as bm
    tables = [mx.read_parquet_table(Path(data_dir) / f"{n}.parquet") for n in QUERIES[q]]
    out = getattr(bm, f"run_q{q}_mxframe")(*tables, device=device)
    if not isinstance(out, pa.Table):
        out = pa.table({"value": [float(out)]})
    return out


if __name__ == "__main__":
    import pyarrow as pa
    data_dir, q, device, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    result = run_query(data_dir, q, device)
    with pa.OSFile(out_path, "wb") as sink, pa.ipc.new_file(sink, result.schema) as w:
        w.write_table(result)
