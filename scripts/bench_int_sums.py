"""F-060: does adding integer sums in int64 cost anything? TPC-H Q1 / Q6 / Q18 with integer columns, before and after.

    pixi run python scripts/bench_int_sums.py --sf 1 [--repeat 7] [--label after]

Run it on a QUIET machine (load average under 2, nothing else on the GPU), once on `main` (before F-060: the
kernels add integers in float64) and once on the F-060 build (int64), and compare the two lines per query.

The TPC-H files store DECIMAL columns as float32, so their sums are float sums and F-060 cannot touch them. This script
reads `lineitem.parquet` and turns the columns into the integers a fixed-point schema would hold: l_quantity as
int32 (units), l_extendedprice as int64 (cents), l_discount and l_tax as int32 (percent). Then:

  Q1   group by (returnflag, linestatus): sum(qty), sum(price), sum(price * (100 - disc)), sum(price * (100 - disc) *
       (100 + tax)), avg(qty), count(*)                         (four int sums, 4 groups)
  Q6   sum(price * disc) where shipdate in 1994, disc between 5 and 7, qty < 24    (one int sum, global)
  Q18  the inner query: group by l_orderkey having sum(qty) > 300         (an int sum over 1.5M groups at SF1)

One fresh process per query; the first compute is a warm-up (imports, kernel load), then the minimum and the median
of `--repeat` computes, in milliseconds, with the result type of the first sum. `--device cpu` (default) or `gpu`."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/home/abhisheksreesaila/Projects/mxframe/data")

QUERIES = {
    "Q1": ("select l_returnflag, l_linestatus, sum(qty) as sq, sum(price) as sp, sum(price * (100 - disc)) as sd, "
           "sum(price * (100 - disc) * (100 + tax)) as sc, avg(qty) as aq, count(*) as n "
           "from li where ship <= 10471 group by l_returnflag, l_linestatus"),
    "Q6": ("select sum(price * disc) as revenue from li "
           "where ship >= 8766 and ship < 9131 and disc between 5 and 7 and qty < 24"),
    "Q18": "select l_orderkey, sum(qty) as sq from li group by l_orderkey having sum(qty) > 300",
}


def child(sf, name, repeat, device):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import mxframe as mx
    t = pq.read_table(DATA / f"tpch_sf{sf}" / "lineitem.parquet")
    li = pa.table({
        "l_orderkey": t.column("l_orderkey"), "l_returnflag": t.column("l_returnflag"),
        "l_linestatus": t.column("l_linestatus"),
        "qty": pa.array(np.rint(t.column("l_quantity").to_numpy()).astype(np.int32)),
        "price": pa.array(np.rint(t.column("l_extendedprice").to_numpy().astype(np.float64) * 100).astype(np.int64)),
        "disc": pa.array(np.rint(t.column("l_discount").to_numpy().astype(np.float64) * 100).astype(np.int32)),
        "tax": pa.array(np.rint(t.column("l_tax").to_numpy().astype(np.float64) * 100).astype(np.int32)),
        "ship": t.column("l_shipdate"),
    })
    q = QUERIES[name]
    out = mx.sql(q, li=li).compute(device=device)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = mx.sql(q, li=li).compute(device=device)
        times.append((time.perf_counter() - t0) * 1000)
    sum_cols = [f.name for f in out.schema if f.name not in ("l_returnflag", "l_linestatus", "l_orderkey", "n", "aq")]
    print(json.dumps({"query": name, "min_ms": round(min(times), 2), "median_ms": round(statistics.median(times), 2),
                      "rows": out.num_rows, "sum_type": str(out.schema.field(sum_cols[0]).type) if sum_cols else None,
                      "loadavg": os.getloadavg()[0]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", default="1")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--label", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    if a.child:
        return child(a.sf, a.child, a.repeat, a.device)
    print(f"# F-060 int-sum bench {a.label}: sf{a.sf}, device {a.device}, loadavg {os.getloadavg()[0]:.2f}"
          f" (run on a quiet machine: load under 2)")
    for name in QUERIES:
        r = subprocess.run([sys.executable, __file__, "--sf", a.sf, "--repeat", str(a.repeat), "--device", a.device,
                            "--child", name], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout.strip() or r.stderr.strip()[-400:])


if __name__ == "__main__":
    main()
