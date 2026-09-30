"""One benchmark run in a FRESH process:  python -m tpch_user.worker --engine E --query N ...

Reports (as `@@RESULT {json}` lines on stdout):
  import_ms  time to import the engine (+ pyarrow.parquet where a reader needs it); NOT in the headline
  first_ms   HEADLINE: Parquet read + query + materialized result, first thing this process does
  repeat_ms  the same read+query again in the same process (secondary; warm caches / JIT)
The first result is written (canonical column names/order) to --result as Parquet so the
parent can compare it with the DuckDB reference.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
import traceback

ENGINES = {
    #  name      module     device  extra imports timed as part of "import"
    "pandas":  ("q_pandas", None, ["pandas", "pyarrow.parquet"]),
    "polars":  ("q_polars", None, ["polars"]),
    "duckdb":  ("q_duckdb", None, ["duckdb"]),
    "mx-cpu":  ("q_mx", "cpu", ["pyarrow.parquet", "pyarrow.compute"]),
    "mx-gpu":  ("q_mx", "gpu", ["pyarrow.parquet", "pyarrow.compute"]),
    "mx-auto": ("q_mx", "auto", ["pyarrow.parquet", "pyarrow.compute"]),
}


def emit(d: dict) -> None:
    print("@@RESULT " + json.dumps(d), flush=True)


def canonicalize(tbl, names: list[str]):
    """Select / rename the result columns to the canonical output of the query."""
    if all(n in tbl.column_names for n in names):
        return tbl.select(names)
    if tbl.num_columns == len(names):
        return tbl.rename_columns(names)
    raise ValueError(f"result columns {tbl.column_names} do not match expected {names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--query", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--skip-repeat-over-ms", type=float, default=float("inf"))
    a = ap.parse_args()

    modname, device, extra = ENGINES[a.engine]
    out: dict = {"engine": a.engine, "query": a.query}
    try:
        t0 = time.perf_counter()
        for m in extra:
            importlib.import_module(m)
        mod = importlib.import_module(f"tpch_user.{modname}")
        out["import_ms"] = (time.perf_counter() - t0) * 1000.0
        from tpch_user import spec
        if modname == "q_mx":
            import mxframe
            out["mxframe_path"] = os.path.dirname(os.path.abspath(mxframe.__file__))

        # ---- headline: first run in a fresh process, nothing warmed up -------------------
        t0 = time.perf_counter()
        res = mod.run(a.query, a.data_dir, device)
        first_ms = (time.perf_counter() - t0) * 1000.0
        out["first_ms"] = first_ms

        import pyarrow.parquet as pq
        tbl = canonicalize(mod.to_arrow(res), spec.OUTPUT[a.query])
        out["rows"] = tbl.num_rows
        pq.write_table(tbl, a.result)
        emit(out)

        # ---- secondary: same read+query again in this process ----------------------------
        del res, tbl
        gc.collect()
        if first_ms <= a.skip_repeat_over_ms:
            try:
                t0 = time.perf_counter()
                res = mod.run(a.query, a.data_dir, device)
                emit({"repeat_ms": (time.perf_counter() - t0) * 1000.0})
            except BaseException as e:  # noqa: BLE001
                emit({"repeat_ms": None, "repeat_note": f"repeat failed: {type(e).__name__}: {str(e)[:200]}"})
        else:
            emit({"repeat_ms": None, "repeat_note": "skipped (first run too slow)"})
    except BaseException as e:  # noqa: BLE001 - report everything, the parent decides
        tb = traceback.format_exc().strip().splitlines()
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["error_where"] = " | ".join(tb[-4:-1])[:400]
        emit(out)
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown (GPU runtimes can hang or crash on exit)


if __name__ == "__main__":
    main()
