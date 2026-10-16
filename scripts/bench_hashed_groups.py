"""F-069: rough A/B of the hashed native group-by against the paths that used to answer these shapes.

    pixi run python scripts/bench_hashed_groups.py [--rows 6000000] [--repeat 5] [--old-lib PATH]

One fresh process per (shape, variant), engine-only first run excluded (a warm-up compute, then the median of
`--repeat` timed computes with the engine's caches cleared). Variants:
  new     this build (hashed group-by in the executor)
  old     `--old-lib`: the executor library from before F-069 (MXFRAME_EXEC_CPU_LIB), when given
  arrow   MXFRAME_NATIVE=0, the Arrow / NumPy path a declined plan falls back to

The numbers are noisy on a loaded machine (every run prints `loadavg`); read them as a ratio, not as a benchmark."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

SHAPES = {
    "float key, 50 groups": "select f, sum(v) as s, count(*) as n from t group by f",
    "float key, 5k groups": "select f5, sum(v) as s, count(*) as n from t group by f5",
    "float key, 1M groups": "select f1m, sum(v) as s, count(*) as n from t group by f1m",
    "wide int64 pair, 1M groups": "select a, b, sum(v) as s, count(*) as n from t group by a, b",
    "wide int64 key, 3 columns, 5k groups": "select a5, b5, c5, sum(v) as s from t group by a5, b5, c5",
    "float key + wide key, filter": "select f5, a, sum(v) as s from t where v > 0.5 group by f5, a",
    "fused join, float partner key": "select w, sum(v) as s from t, d where k = dk group by w",
    "fused join, wide fact keys": "select a, b, sum(v) as s from t, d where k = dk group by a, b",
}


def data(n):
    import numpy as np
    import pyarrow as pa
    from mxframe.np_arrow import array as A
    r = np.random.default_rng(1)
    wide = r.integers(-2**62, 2**62, 1_000_000)
    k = r.integers(0, 5000, n)
    t = pa.table({
        "f": A(np.round(r.random(50), 6)[r.integers(0, 50, n)]),
        "f5": A(np.round(r.random(5000) * 1000, 6)[r.integers(0, 5000, n)]),
        "f1m": A(r.random(1_000_000)[r.integers(0, 1_000_000, n)]),
        "a": A(wide[r.integers(0, 1_000_000, n)]),
        "b": A(wide[r.integers(0, 1_000, n)]),
        "a5": A(wide[:5000][k]),
        "b5": A(wide[5000:10000][k]),
        "c5": A(wide[10000:15000][k]),
        "k": A(k.astype(np.int64)),
        "v": A(r.random(n)),
    })
    d = pa.table({"dk": A(np.arange(5000, dtype=np.int64)), "w": A(r.random(5000))})
    return t, d


def one(shape, n, repeat):
    import conftest  # noqa: F401  (import mxframe from this checkout)
    import mxframe as mx
    import mxframe.native_exec as nx
    t, d = data(n)
    sql = SHAPES[shape]
    mx.sql(sql, t=t, d=d).compute(device="cpu")                     # warm-up (library load, plan)
    times = []
    for _ in range(repeat):
        mx.clear_cache()
        t0 = time.perf_counter()
        r = mx.sql(sql, t=t, d=d).compute(device="cpu")
        times.append((time.perf_counter() - t0) * 1000)
    print("@@B " + json.dumps({"shape": shape, "ms": statistics.median(times), "min": min(times), "rows": r.num_rows,
                                "group_path": nx.last_group_path(), "path": nx.last_path(),
                                "loadavg": os.getloadavg()[0]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6_000_000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--old-lib")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--one")
    a = ap.parse_args()
    if a.one:
        return one(a.one, a.rows, a.repeat)
    variants = {"new": {}, "arrow": {"MXFRAME_NATIVE": "0"}}
    if a.old_lib:
        variants = {"new": {}, "old": {"MXFRAME_EXEC_CPU_LIB": a.old_lib}, "arrow": {"MXFRAME_NATIVE": "0"}}
    names = [s for s in SHAPES if not a.shapes or any(x in s for x in a.shapes.split(","))]
    print(f"{'shape':40s} " + " ".join(f"{v:>14s}" for v in variants))
    for shape in names:
        cells = []
        for v, env in variants.items():
            p = subprocess.run([sys.executable, __file__, "--one", shape, "--rows", str(a.rows), "--repeat", str(a.repeat)],
                               capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", **env))
            ls = [x for x in p.stdout.splitlines() if x.startswith("@@B ")]
            if not ls:
                cells.append("error")
                continue
            r = json.loads(ls[-1][4:])
            cells.append(f"{r['ms']:7.0f} ms {r['group_path'] or r['path'] or '-':>5s}")
        print(f"{shape:40s} " + " ".join(f"{c:>14s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
