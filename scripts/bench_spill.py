"""Out-of-memory benchmark (F-056 / F-058): SF10 queries forced to stream, spill and Grace-join.

Each (query, engine, budget) runs in a fresh process. omxcore reads with `scan_parquet` under
`MXFRAME_HOST_MEM_BUDGET`; the clock covers read + query + result (a streamed query cannot stage its data,
that is the point). DuckDB is the reference, run once without a limit and once with `memory_limit` set to the
same budget. Every omxcore result is checked against DuckDB's. Peak RSS is the child's ru_maxrss.

    pixi run python scripts/bench_spill.py --budgets 2G,none --devices cpu,gpu,auto --gpu 0
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MXFRAME_TPCH_DATA", ROOT / "data"))
DATA = DATA_ROOT / os.environ.get("BENCH_SPILL_DIR", "tpch_sf10_exact")
TABLES = ("lineitem", "orders", "customer")
# (the data stores dates as int32 days since 1970: 8401 = 1993-01-01)

# (id, description, SQL, sort keys for the check, kind) - kind "rows" = compute(), "sink" = sink_parquet
QUERIES = [
    ("J1", "lineitem x orders, 8 groups",
     "SELECT o_orderpriority, l_returnflag, count(*) AS n, sum(l_quantity) AS q, "
     "sum(l_extendedprice * (1 - l_discount)) AS rev, min(o_totalprice) AS lo, max(l_extendedprice) AS hi "
     "FROM lineitem JOIN orders ON l_orderkey = o_orderkey "
     "WHERE l_shipdate > 8401 GROUP BY o_orderpriority, l_returnflag",
     ("o_orderpriority", "l_returnflag"), "rows"),
    ("J2", "lineitem x orders x customer, by segment",
     "SELECT c_mktsegment, count(*) AS n, sum(l_extendedprice * (1 - l_discount)) AS rev "
     "FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey "
     "GROUP BY c_mktsegment",
     ("c_mktsegment",), "rows"),
    ("J3", "lineitem x orders, 15M groups, top 20",
     "SELECT o_orderkey, o_orderdate, sum(l_extendedprice * (1 - l_discount)) AS rev "
     "FROM lineitem JOIN orders ON l_orderkey = o_orderkey "
     "GROUP BY o_orderkey, o_orderdate ORDER BY rev DESC, o_orderkey LIMIT 20",
     (), "rows"),
    ("J4", "count distinct l_orderkey per l_suppkey (100k groups)",
     "SELECT l_suppkey, count(DISTINCT l_orderkey) AS n FROM lineitem GROUP BY l_suppkey "
     "ORDER BY n DESC, l_suppkey LIMIT 50",
     (), "rows"),
    ("J5", "join result to Parquet (~54M rows)",
     "SELECT l_orderkey, l_partkey, l_extendedprice, o_orderdate, o_totalprice "
     "FROM lineitem JOIN orders ON l_orderkey = o_orderkey WHERE l_discount > 0.0",
     (), "sink"),
]


class _MemPeak:
    """Peak anonymous and file-backed RSS (from /proc/self/status, every 20 ms): ru_maxrss also counts the
    pages of memory-mapped Parquet files, which the kernel can drop, so it overstates what a query holds."""

    def __init__(self):
        import threading
        self.anon = self.file = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(0.02):
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        self.anon = max(self.anon, int(line.split()[1]) // 1024)
                    elif line.startswith("RssFile:"):
                        self.file = max(self.file, int(line.split()[1]) // 1024)

    def stop(self):
        self._stop.set()
        self._t.join()
        return {"anon_mb": self.anon, "file_mb": self.file}


def worker(args):
    peak = _MemPeak()
    t0 = time.perf_counter()
    q = next(x for x in QUERIES if x[0] == args.query)
    out = Path(args.result)
    if args.engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        con.execute(f"SET threads={os.cpu_count()}")
        if args.budget != "none":
            con.execute(f"SET memory_limit='{args.budget}B'")
            con.execute(f"SET temp_directory='{Path.home() / '.cache/mxbench-f058/duck_tmp'}'")
        for t in TABLES:
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{DATA / (t + '.parquet')}')")
        t1 = time.perf_counter()
        if q[4] == "sink":
            con.execute(f"COPY ({q[2]}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
            rows = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
            ms = (time.perf_counter() - t1) * 1000
        else:
            res = con.sql(q[2]).arrow()
            res = res.read_all() if hasattr(res, "read_all") else res
            ms = (time.perf_counter() - t1) * 1000
            import pyarrow.parquet as pq
            pq.write_table(res, out)
            rows = res.num_rows
        info = {}
    else:
        if args.budget != "none":
            os.environ["MXFRAME_HOST_MEM_BUDGET"] = args.budget
        import warnings
        import omxcore as mx
        from omxcore.io_parquet import scan_parquet
        lf = mx.sql(q[2], **{t: scan_parquet(str(DATA / (t + ".parquet"))) for t in TABLES})
        t1 = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if q[4] == "sink":
                rows = lf.sink_parquet(str(out), device=args.device)
                ms = (time.perf_counter() - t1) * 1000
            else:
                res = lf.compute(device=args.device)
                ms = (time.perf_counter() - t1) * 1000
                import pyarrow.parquet as pq
                pq.write_table(res, out)
                rows = res.num_rows
        p = getattr(lf, "last_compile_provenance", None) or {}
        s = p.get("streamed") or {}
        info = {"path": p.get("path"), "batches": s.get("batches"), "devices": s.get("devices"),
                "grace": s.get("grace"), "spill": s.get("spill"),
                "warnings": sorted({str(x.message)[:160] for x in w})[:5]}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("RESULT " + json.dumps({"ms": ms, "rows": rows, "rss_mb": rss, **peak.stop(),
                                  "total_s": time.perf_counter() - t0, **info},
                                 default=str))


def same(got_path, want_path, keys):
    import math
    import pyarrow.parquet as pq
    g, w = pq.read_table(got_path), pq.read_table(want_path)
    if g.num_rows != w.num_rows:
        return f"rows {g.num_rows} vs {w.num_rows}"
    if g.num_rows > 1_000_000:                         # a sink: compare aggregates of every column
        bad = []
        for name in w.column_names:
            import pyarrow.compute as pc
            a, b = g.column(name), w.column(name)
            if pc.count(a).as_py() != pc.count(b).as_py():
                bad.append(name)
                continue
            if a.type == b.type and (str(a.type).startswith(("int", "double", "decimal", "float"))):
                sa, sb = float(pc.sum(a).as_py()), float(pc.sum(b).as_py())
                if not math.isclose(sa, sb, rel_tol=1e-9):
                    bad.append(f"{name} sum {sa} vs {sb}")
        return "; ".join(bad) or "ok"
    gl, wl = g.to_pylist(), w.to_pylist()
    if keys:
        k = lambda r: tuple((r[c] is None, r[c]) for c in keys)    # noqa: E731
        gl, wl = sorted(gl, key=k), sorted(wl, key=k)
    for gr, wr in zip(gl, wl):
        for c in w.column_names:
            a, b = gr.get(c), wr.get(c)
            if isinstance(b, float) or isinstance(a, float) or type(b).__name__ == "Decimal":
                if a is None or b is None or not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9):
                    return f"{c}: {a} vs {b}"
            elif a != b:
                return f"{c}: {a!r} vs {b!r}"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=",".join(q[0] for q in QUERIES))
    ap.add_argument("--budgets", default="2G,none")
    ap.add_argument("--devices", default="cpu,gpu,auto")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=str(ROOT / "docs/bench/f058-spill.jsonl"))
    ap.add_argument("--work", default=str(Path.home() / ".cache/mxbench-f058/spill"))
    ap.add_argument("--sf", default="10")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--query"), ap.add_argument("--engine"), ap.add_argument("--device")
    ap.add_argument("--budget"), ap.add_argument("--result")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BENCH_SPILL_DIR=f"tpch_sf{a.sf}_exact", CUDA_VISIBLE_DEVICES=a.gpu, MXFRAME_SPILL_DIR=str(work / "spilldir"))
    out = open(a.out, "a")

    def run(qid, engine, device, budget):
        res = work / f"{qid}-{engine}-{device}-{budget}.parquet"
        cmd = [sys.executable, __file__, "--worker", "--query", qid, "--engine", engine, "--device", device or "-",
               "--budget", budget, "--result", str(res)]
        t = time.time()
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout)
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
            r = json.loads(line[7:]) if line else {"error": (p.stderr or p.stdout)[-800:]}
        except subprocess.TimeoutExpired:
            r = {"error": f"timeout {a.timeout}s"}
        r.update(query=qid, engine=engine, device=device, budget=budget, at=time.strftime("%H:%M:%S", time.localtime(t)),
                 load1=os.getloadavg()[0], result=str(res))
        return r

    for qid in a.queries.split(","):
        q = next(x for x in QUERIES if x[0] == qid)
        ref = run(qid, "duckdb", None, "none")
        print(json.dumps(ref), flush=True), out.write(json.dumps(ref) + "\n"), out.flush()
        for budget in a.budgets.split(","):
            if budget != "none":
                r = run(qid, "duckdb", None, budget)
                r["check"] = same(r["result"], ref["result"], q[3]) if "error" not in r else "-"
                print(json.dumps(r), flush=True), out.write(json.dumps(r) + "\n"), out.flush()
            for dev in a.devices.split(","):
                r = run(qid, "mx", dev, budget)
                r["check"] = same(r["result"], ref["result"], q[3]) if "error" not in r and "error" not in ref else "-"
                print(json.dumps(r), flush=True), out.write(json.dumps(r) + "\n"), out.flush()
                Path(r["result"]).unlink(missing_ok=True)
        Path(ref["result"]).unlink(missing_ok=True)
        for f in work.glob(f"{qid}-*.parquet"):
            f.unlink()


if __name__ == "__main__":
    main()
