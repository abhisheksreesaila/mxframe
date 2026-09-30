#!/usr/bin/env python3
"""User-felt TPC-H scoreboard (ticket F-001).

What a user feels: a fresh Python process, data starting as Parquet on disk, the clock
around (Parquet read + query + materialized result).  No warm-up, ever.  One subprocess
per (engine, query, scale factor).  `import` time is reported separately; a repeat run in
the same process is a secondary number.  Every result is checked against DuckDB.

    pixi run python scripts/bench_user_felt.py --sf 1,10 --queries 1-22 \
        --engines pandas,polars,duckdb,mx-cpu,mx-gpu,mx-auto --timeout 300 --out docs/bench

Every measurement is appended to <out>/<name>.csv (and flushed) the moment it exists, so a crash
or Ctrl-C loses nothing; the markdown is rendered from the CSV at the end.  `--resume` skips the
(engine, sf, query) rows already in the CSV; `--report-only` just re-renders the markdown.

Environment: the subprocesses use CUDA_VISIBLE_DEVICES=<--gpu> (default 1) and import
mxframe from THIS checkout.  Data: $MXFRAME_TPCH_DATA or <repo>/data or the main checkout's
data/ (a worktree has none); files are <data>/tpch_sf<SF>/<table>.parquet.
"""
from __future__ import annotations

import argparse
import csv
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
from tpch_user.report import (CSV_FIELDS, ENGINE_ORDER, environment_info, new_row, read_csv,  # noqa: E402
                              render_markdown)
from tpch_user.runner import (REPO_ROOT, check_mxframe_path, resolve_gpu, run_one,  # noqa: E402
                              warm_page_cache)

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


def compare_result(path, ref_path, q: int):
    """Compare a result Parquet with the reference; returns (verdict, correct, max_rel_err, note)."""
    import pyarrow.parquet as pq
    try:
        cmp = compare_tables(pq.read_table(path), pq.read_table(ref_path),
                             topn_metric=spec.TOPN_METRIC.get(q), order_by=spec.ORDER_BY.get(q))
    except Exception as e:  # noqa: BLE001 - a broken comparison must not lose the measurement
        return "unchecked", None, None, f"comparison failed: {type(e).__name__}: {str(e)[:150]}"
    return cmp.verdict, cmp.ok, cmp.max_rel_err, cmp.message


def open_log(csv_path: Path, resume: bool):
    """(file, writer, rows already present). Without --resume the CSV starts empty."""
    existing = read_csv(csv_path) if resume and csv_path.exists() else []
    f = open(csv_path, "a" if existing else "w", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if not existing:
        w.writeheader()
        f.flush()
    return f, w, existing


def write_report(csv_path: Path, md_path: Path, a, root, gpu_desc: str, elapsed: float,
                 queries: list[int], sfs: list[str]) -> None:
    rows = read_csv(csv_path)
    if not rows:
        return
    engines = [e for e in ENGINE_ORDER if any(r["engine"] == e for r in rows)]
    qs = sorted({r["query"] for r in rows} | set(queries))
    sf_all = sorted({r["sf"] for r in rows} | set(sfs), key=float)
    meta = environment_info(a, root, elapsed, gpu_desc)
    md_path.write_text(render_markdown(rows, meta, engines, qs, sf_all))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sf", default="1", help="scale factors, e.g. 1,10")
    ap.add_argument("--queries", default="1-22")
    ap.add_argument("--engines", default=",".join(ALL_ENGINES))
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per subprocess")
    ap.add_argument("--out", default="docs/bench")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--gpu", default="1", help="nvidia-smi index (or GPU-UUID) of the GPU the workers may use")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--name", default=None, help="output basename (default user-felt-<date>)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing CSV and skip (engine, sf, query) rows already in it")
    ap.add_argument("--report-only", action="store_true", help="re-render the markdown from the CSV and exit")
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
    csv_path, md_path = out_dir / f"{base}.csv", out_dir / f"{base}.md"
    cuda_dev, gpu_desc = resolve_gpu(a.gpu)

    if a.report_only:
        write_report(csv_path, md_path, a, root, gpu_desc, 0.0, queries, sfs)
        print(f"wrote {md_path}")
        return

    # fail BEFORE the loop: data present, and workers import mxframe from THIS checkout
    data_dirs = {sf: root / f"tpch_sf{sf}" for sf in sfs}
    missing = [str(d) for d in data_dirs.values() if not d.is_dir()]
    if missing:
        raise SystemExit(f"TPC-H data not found: {missing}")
    if any(e.startswith("mx-") for e in engines):
        mxp = check_mxframe_path()
        if os.path.realpath(mxp) != os.path.realpath(REPO_ROOT):
            raise SystemExit(f"workers import mxframe from {mxp}, not {REPO_ROOT}")

    # the DuckDB reference always runs first (it is only *reported* when requested)
    order = ["duckdb"] + [e for e in engines if e != "duckdb"]
    tmp = Path(tempfile.mkdtemp(prefix="bench-user-felt-"))
    f, writer, existing = open_log(csv_path, a.resume)
    done = {(r["engine"], r["sf"], r["query"]) for r in existing}
    t_start = time.time()
    try:
        for sf in sfs:
            data_dir = data_dirs[sf]
            tables = sorted({t for q in queries for t in spec.TABLES[q]})
            warm_page_cache(str(data_dir), tables)
            for q in queries:
                todo = [e for e in engines if (e, sf, q) not in done]
                if not todo:
                    continue
                ref_path = tmp / f"ref_sf{sf}_q{q}.parquet"
                ref_ok = False
                for eng in order:
                    if eng == "duckdb":
                        if not any(e != "duckdb" for e in todo) and "duckdb" not in todo:
                            continue
                    elif eng not in todo:
                        continue
                    res_path = ref_path if eng == "duckdb" else tmp / f"{eng}_sf{sf}_q{q}.parquet"
                    rep_path = Path(str(res_path) + ".repeat")
                    load1 = os.getloadavg()[0]
                    rec = run_one(eng, q, str(data_dir), str(res_path), a.timeout, cuda_dev)
                    rec.update(engine=eng, sf=sf, query=q, correct=None, max_rel_err=None,
                               load1=load1, verdict="ref" if eng == "duckdb" else "unchecked",
                               repeat_verdict="")
                    mxp = rec.pop("mxframe_path", None)
                    if mxp and os.path.realpath(mxp) != os.path.realpath(REPO_ROOT):
                        rec["error"] = f"imported mxframe from {mxp}, not {REPO_ROOT}"
                    if eng == "duckdb":
                        ref_ok = "error" not in rec and res_path.exists()
                        if rec.get("repeat_ms") is not None and ref_ok and rep_path.exists():
                            rec["repeat_verdict"] = compare_result(rep_path, ref_path, q)[0]
                    elif "error" in rec:
                        rec["verdict"] = "error"
                    elif ref_ok and res_path.exists():
                        rec["verdict"], rec["correct"], rec["max_rel_err"], note = \
                            compare_result(res_path, ref_path, q)
                        if note:
                            rec["compare_note"] = note
                        if rec.get("repeat_ms") is not None and rep_path.exists():
                            rec["repeat_verdict"] = compare_result(rep_path, ref_path, q)[0]
                    elif res_path.exists():
                        rec["compare_note"] = "not verified: the DuckDB reference failed for this query"
                    if eng in todo:
                        writer.writerow(new_row(rec))
                        f.flush()
                        os.fsync(f.fileno())
                    for p in (res_path, rep_path):
                        if eng != "duckdb" and p.exists():
                            p.unlink()
                    status = rec.get("error") or f"first={rec.get('first_ms', 0):.0f}ms {rec['verdict']}"
                    print(f"SF{sf} Q{q:<2} {eng:<8} {status}", flush=True)
                for p in (ref_path, Path(str(ref_path) + ".repeat")):
                    if p.exists():
                        p.unlink()
    finally:
        f.close()
        shutil.rmtree(tmp, ignore_errors=True)
        write_report(csv_path, md_path, a, root, gpu_desc, time.time() - t_start, queries, sfs)
        print(f"\nwrote {md_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
