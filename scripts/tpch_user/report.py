"""Markdown + CSV output for the user-felt scoreboard (pure formatting, no engines)."""
from __future__ import annotations

import csv
import datetime
import os
import statistics
import subprocess
from importlib import metadata
from pathlib import Path

from . import spec

CSV_FIELDS = ["engine", "sf", "query", "import_ms", "first_ms", "repeat_ms", "rows", "correct",
              "max_rel_err", "verdict", "error", "notes"]
MX = ["mx-cpu", "mx-gpu", "mx-auto"]
NAMES = {"pandas": "pandas", "polars": "Polars", "duckdb": "DuckDB", "mx-cpu": "mx-cpu",
         "mx-gpu": "mx-gpu", "mx-auto": "mx-auto"}


# ------------------------------------------------------------------ helpers (pure) -----------
def is_valid(rec: dict | None) -> bool:
    """Has a headline time and is not WRONG (a wrong answer cannot win)."""
    return bool(rec) and "error" not in rec and rec.get("first_ms") is not None \
        and rec.get("verdict") != "wrong"


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "-"
    if ms >= 100:
        return f"{ms:,.0f}"
    if ms >= 10:
        return f"{ms:.0f}"
    return f"{ms:.1f}"


def _short_error(err: str) -> str:
    if err.startswith("TIMEOUT"):
        return "TIMEOUT"
    if err.startswith("CRASH"):
        return "CRASH"
    return "ERR"


def fmt_cell(rec: dict | None, fastest: bool = False) -> str:
    if rec is None:
        return "n/a"
    if "error" in rec:
        return _short_error(rec["error"])
    ms = fmt_ms(rec.get("first_ms"))
    v = rec.get("verdict")
    if v == "wrong":
        return f"{ms} WRONG"
    text = ms
    if v == "drift":
        text += f" †{rec['max_rel_err']:.0e}".replace("e-0", "e-")
    return f"**{text}**" if fastest else text


def win_counts(rows: list[dict], sf: str, queries: list[int], a: str, b: str) -> tuple[int, int]:
    """(queries where engine `a` is faster than `b`, queries where both have a valid time)."""
    by = {(r["engine"], r["query"]): r for r in rows if r["sf"] == sf}
    wins = comparable = 0
    for q in queries:
        ra, rb = by.get((a, q)), by.get((b, q))
        if is_valid(ra) and is_valid(rb):
            comparable += 1
            wins += ra["first_ms"] < rb["first_ms"]
    return wins, comparable


def losses_vs(rows: list[dict], sf: str, queries: list[int], mx_engines: list[str],
              rival: str = "polars", top: int = 10) -> list[tuple]:
    """(ratio mx/rival, query, mx engine, mx ms, rival ms) where the best valid mx engine is slower."""
    by = {(r["engine"], r["query"]): r for r in rows if r["sf"] == sf}
    out = []
    for q in queries:
        rv = by.get((rival, q))
        cands = [by.get((e, q)) for e in mx_engines]
        cands = [c for c in cands if is_valid(c)]
        if not cands or not is_valid(rv):
            continue
        best = min(cands, key=lambda r: r["first_ms"])
        if best["first_ms"] > rv["first_ms"]:
            out.append((best["first_ms"] / rv["first_ms"], q, best["engine"], best["first_ms"], rv["first_ms"]))
    return sorted(out, reverse=True)[:top]


# ------------------------------------------------------------------ CSV -----------------------
def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (float(r["sf"]), r["query"], r["engine"])):
            notes = "; ".join(str(r[k]) for k in ("repeat_note", "compare_note") if r.get(k))
            w.writerow({**{k: r.get(k) for k in CSV_FIELDS}, "notes": notes,
                        "error": r.get("error", "")})


# ------------------------------------------------------------------ environment ----------------
def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "?"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment_info(args, data_root, elapsed_s: float) -> dict:
    from .runner import REPO_ROOT
    gpu_name = _run(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"]) or "unknown"
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "date": args.date, "cores": os.cpu_count(), "cpu": cpu, "gpu": gpu_name,
        "gpu_index": args.gpu, "timeout": args.timeout, "elapsed_min": elapsed_s / 60.0,
        "data_root": str(data_root),
        "git": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]),
        "versions": {p: _ver(p) for p in ("pandas", "polars", "duckdb", "pyarrow", "max")},
        "loadavg": " ".join(f"{x:.1f}" for x in os.getloadavg()),
    }


# ------------------------------------------------------------------ markdown -------------------
def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def render_markdown(rows: list[dict], meta: dict, engines: list[str], queries: list[int],
                    sfs: list[str]) -> str:
    L: list[str] = []
    a = L.append
    v = meta["versions"]
    a(f"# User-felt TPC-H scoreboard - {meta['date']}\n")
    a("**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, "
      "the clock around *Parquet read + query + materialized result*. No warm-up. "
      "`import` time is reported separately and is not in the headline. "
      "Numbers are milliseconds; **bold** = fastest correct engine for that query.\n")
    a("Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict "
      "1e-6 check), `WRONG` = differs from DuckDB (rows/keys or relative error above 1e-2; excluded from "
      "\"fastest\" and from win counts), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result "
      f"(timeout {meta['timeout']:.0f}s per subprocess).\n")

    for sf in sfs:
        srows = [r for r in rows if r["sf"] == sf]
        if not srows:
            continue
        by = {(r["engine"], r["query"]): r for r in srows}
        a(f"\n## Scale factor {sf}: headline (first run in a fresh process), ms\n")
        body = []
        fastest_count = {e: 0 for e in engines}
        for q in queries:
            valid = [(by[(e, q)]["first_ms"], e) for e in engines if is_valid(by.get((e, q)))]
            best = min(valid)[1] if valid else None
            if best:
                fastest_count[best] += 1
            body.append([f"Q{q} {spec.TITLES[q]}"] + [fmt_cell(by.get((e, q)), fastest=(e == best))
                                                        for e in engines])
        a(_table(["query"] + [NAMES[e] for e in engines], body))
        a("\nFastest engine per query: " + ", ".join(f"{NAMES[e]} {n}" for e, n in fastest_count.items()) + ".\n")

        a(f"### Wins at SF{sf} (headline, both results correct)\n")
        n = len(queries)
        for m in [e for e in MX if e in engines]:
            parts = []
            for rival in ("polars", "pandas", "duckdb"):
                if rival in engines:
                    w, c = win_counts(rows, sf, queries, m, rival)
                    parts.append(f"**{m} beats {NAMES[rival]} on {w}/{n}** ({c} comparable)")
            a("- " + "; ".join(parts))
        a("")

        if any(e in engines for e in MX) and "polars" in engines:
            for label, mxs in (("mx-auto", ["mx-auto"]), ("best of mx-cpu / mx-gpu / mx-auto", MX)):
                mxs = [e for e in mxs if e in engines]
                lost = losses_vs(rows, sf, queries, mxs)
                if lost:
                    a(f"### Where {label} loses to Polars at SF{sf} (worst first)\n")
                    a(_table(["query", "engine", "MXFrame ms", "Polars ms", "slower by"],
                             [[f"Q{q} {spec.TITLES[q]}", e, fmt_ms(m), fmt_ms(p), f"{r:.1f}x"]
                              for r, q, e, m, p in lost]))
                    a("")

        a(f"### SF{sf}: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)\n")
        body = []
        for q in queries:
            body.append([f"Q{q}"] + [fmt_ms(by[(e, q)].get("repeat_ms")) if by.get((e, q)) and "error" not in by[(e, q)]
                                     else "-" for e in engines])
        a(_table(["query"] + [NAMES[e] for e in engines], body))
        a("\n> Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically "
          "fast; that is why the headline is a fresh process.\n")

        a(f"### SF{sf}: import time, ms (median over queries; not in the headline)\n")
        imp = []
        for e in engines:
            xs = [r["import_ms"] for r in srows if r["engine"] == e and r.get("import_ms") is not None]
            imp.append(fmt_ms(statistics.median(xs)) if xs else "-")
        a(_table([""] + [NAMES[e] for e in engines], [["import"] + imp]))
        a("")

        bad = [r for r in srows if r["engine"] != "duckdb" and (r.get("verdict") in ("wrong", "drift") or "error" in r)]
        if bad:
            a(f"### SF{sf}: correctness against DuckDB and failures\n")
            body = []
            for r in sorted(bad, key=lambda r: (r["query"], r["engine"])):
                what = r.get("error") or r.get("compare_note") or ""
                mre = r.get("max_rel_err")
                body.append([f"Q{r['query']}", r["engine"], r["verdict"],
                             "-" if mre is None else ("inf" if mre == float("inf") else f"{mre:.2e}"),
                             what.replace("|", "/")[:160]])
            a("| query | engine | verdict | max rel. error | note |\n|---|---|---|---:|---|")
            a("\n".join("| " + " | ".join(r) + " |" for r in body))
            a("")
        ok_drift = [r for r in srows if r["engine"] != "duckdb" and r.get("max_rel_err") is not None
                    and r["max_rel_err"] != float("inf")]
        worst = {}
        for r in ok_drift:
            worst[r["engine"]] = max(worst.get(r["engine"], 0.0), r["max_rel_err"])
        if worst:
            a("Largest relative error vs DuckDB (over all queries where rows/keys matched): "
              + ", ".join(f"{NAMES[e]} {x:.1e}" for e, x in worst.items()) + ".\n")

    a("\n## Methodology\n")
    a("- **One subprocess per (engine, query, scale factor)**; the worker imports the engine "
      "(timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and "
      "once more in the same process (`repeat_ms`).")
    a("- **Data**: DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read "
      "from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).")
    a("- **Column pruning**: every engine reads only the columns its query uses. Polars (`scan_parquet`) and "
      "DuckDB prune automatically; pandas gets `pd.read_parquet(columns=...)`; MXFrame gets "
      "`mx.read_parquet_table(columns=..., dictionary=\"auto\")` (the `scan_parquet` reader, eager) through one helper "
      "(`scripts/tpch_user/readers.py::mx_read`); tables the harness itself touches with PyArrow string kernels "
      "are read with `dictionary=False`.")
    a("- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB "
      "(SQL over `read_parquet`, also the correctness reference), MXFrame `device=\"cpu\"|\"gpu\"|\"auto\"` "
      "through the public `LazyFrame` API (plus PyArrow where the LazyFrame layer cannot express a step).")
    a("- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; "
      "float columns at 1e-6 relative, sorted by key columns; LIMIT queries tolerate ties at the cutoff). "
      "inputs are float32; an engine that accumulates in float32 "
      "drifts on large sums (`†`), DuckDB sums in double.")
    a(f"- **Hardware**: {meta['cpu']} ({meta['cores']} cores); GPU: {meta['gpu']} "
      f"(`CUDA_VISIBLE_DEVICES={meta['gpu_index']}`). **The machine was shared with other jobs during this run "
      f"(load average at report time: {meta['loadavg']}), so timings are indicative.**")
    a(f"- **Versions**: " + ", ".join(f"{k} {x}" for k, x in v.items()) + f"; mxframe at git `{meta['git']}`. "
      f"Run took {meta['elapsed_min']:.0f} min. Data root: `{meta['data_root']}`.")
    a("\n### Query semantics\n")
    a("All engines and the DuckDB reference compute the same query. Where it differs from the official "
      "TPC-H text (the mxframe implementations are simplified) the difference is listed here; this is "
      "not a TPC-certified benchmark.\n")
    for q, text in sorted(spec.DEVIATIONS.items()):
        a(f"- Q{q}: {text}")
    a("")
    return "\n".join(L) + "\n"
