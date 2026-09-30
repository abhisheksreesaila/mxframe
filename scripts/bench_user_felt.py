#!/usr/bin/env python3
"""User-felt TPC-H scoreboard (ticket F-001).

What a user feels: a fresh Python process, data starting as Parquet on disk, the clock
around (Parquet read + query + materialized result).  No warm-up, ever.  One subprocess
per (engine, query, scale factor).  `import` time is reported separately; a repeat run in
the same process is a secondary number.  Every result is checked against DuckDB.

    pixi run python scripts/bench_user_felt.py --sf 1,10 --queries 1-22 \
        --engines pandas,polars,duckdb,mx-cpu,mx-gpu,mx-auto --timeout 300 --out docs/bench

Environment: the subprocesses use CUDA_VISIBLE_DEVICES=<--gpu> (default 1) and import
mxframe from THIS checkout.  Data: $MXFRAME_TPCH_DATA or <repo>/data or the main checkout's
data/ (a worktree has none); files are <data>/tpch_sf<SF>/<table>.parquet.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tpch_user import spec  # noqa: E402
from tpch_user.compare import compare_tables  # noqa: E402
from tpch_user.report import render_markdown, write_csv, environment_info  # noqa: E402
from tpch_user.runner import REPO_ROOT, run_one, warm_page_cache  # noqa: E402

ALL_ENGINES = ["pandas", "polars", "duckdb", "mx-cpu", "mx-gpu", "mx-auto"]


def parse_queries(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [q for q in out if q not in spec.OUTPUT]
    if bad:
        raise SystemExit(f"unknown queries: {bad}")
    return out


def find_data_root(cli: str | None) -> Path:
    cands = [cli, os.environ.get("MXFRAME_TPCH_DATA"), str(REPO_ROOT / "data")]
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        if common:
            cands.append(str(Path(common).parent / "data"))
    except OSError:
        pass
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c)
    raise SystemExit("TPC-H data not found; pass --data-root (expects <root>/tpch_sf<SF>/*.parquet)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sf", default="1", help="scale factors, e.g. 1,10")
    ap.add_argument("--queries", default="1-22")
    ap.add_argument("--engines", default=",".join(ALL_ENGINES))
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per subprocess")
    ap.add_argument("--out", default="docs/bench")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--gpu", default="1", help="value of CUDA_VISIBLE_DEVICES for every subprocess")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--name", default=None, help="output basename (default user-felt-<date>)")
    a = ap.parse_args()

    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ALL_ENGINES]
    if unknown:
        raise SystemExit(f"unknown engines: {unknown}; choose from {ALL_ENGINES}")
    queries = parse_queries(a.queries)
    sfs = [s.strip() for s in a.sf.split(",") if s.strip()]
    root = find_data_root(a.data_root)
    out_dir = Path(a.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = a.name or f"user-felt-{a.date}"

    # the DuckDB reference always runs first (it is only *reported* when requested)
    order = ["duckdb"] + [e for e in engines if e != "duckdb"]
    tmp = Path(tempfile.mkdtemp(prefix="bench-user-felt-"))
    rows: list[dict] = []
    t_start = time.time()
    try:
        for sf in sfs:
            data_dir = root / f"tpch_sf{sf}"
            if not data_dir.is_dir():
                print(f"[skip] {data_dir} not found", flush=True)
                continue
            tables = sorted({t for q in queries for t in spec.TABLES[q]})
            warm_page_cache(str(data_dir), tables)
            for q in queries:
                ref_path = tmp / f"ref_sf{sf}_q{q}.parquet"
                ref_ok = False
                for eng in order:
                    res_path = ref_path if eng == "duckdb" else tmp / f"{eng}_sf{sf}_q{q}.parquet"
                    rec = run_one(eng, q, str(data_dir), str(res_path), a.timeout, a.gpu)
                    mxp = rec.get("mxframe_path")
                    if mxp and os.path.realpath(mxp) != os.path.realpath(REPO_ROOT):
                        raise SystemExit(f"subprocess imported mxframe from {mxp}, not {REPO_ROOT}")
                    rec.update(engine=eng, sf=sf, query=q, correct=None, max_rel_err=None,
                               verdict="unchecked" if eng != "duckdb" else "ref")
                    if eng == "duckdb":
                        ref_ok = "error" not in rec and res_path.exists()
                    elif "error" in rec:
                        rec["verdict"] = "error"
                    elif ref_ok and res_path.exists():
                        import pyarrow.parquet as pq
                        cmp = compare_tables(pq.read_table(res_path), pq.read_table(ref_path),
                                             topn_metric=spec.TOPN_METRIC.get(q))
                        rec.update(correct=cmp.ok, max_rel_err=cmp.max_rel_err, verdict=cmp.verdict)
                        if cmp.message:
                            rec["compare_note"] = cmp.message
                    if eng != "duckdb" or "duckdb" in engines:
                        rows.append(rec)
                    if res_path.exists() and eng != "duckdb":
                        res_path.unlink()
                    status = rec.get("error") or f"first={rec.get('first_ms', 0):.0f}ms {rec['verdict']}"
                    print(f"SF{sf} Q{q:<2} {eng:<8} {status}", flush=True)
                if ref_path.exists():
                    ref_path.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    meta = environment_info(a, root, time.time() - t_start)
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"
    write_csv(rows, csv_path)
    md_path.write_text(render_markdown(rows, meta, engines, queries, sfs))
    print(f"\nwrote {md_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
