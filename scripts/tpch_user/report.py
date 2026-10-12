"""Markdown + CSV output for the user-felt scoreboard (pure formatting, no engines).

The CSV is the source of truth: the driver appends one row per measurement as soon as it is
made, and the markdown is rendered from the CSV at the end (or with --report-only).
"""
from __future__ import annotations

import csv
import glob
import os
import statistics
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from . import spec

CSV_FIELDS = ["engine", "sf", "query", "import_ms", "first_ms", "repeat_ms", "rows", "correct",
              "max_rel_err", "verdict", "repeat_verdict", "load1", "error", "repeat_note",
              "compare_note", "mode", "load_ms"]
_FLOATS = ("import_ms", "first_ms", "repeat_ms", "max_rel_err", "load1", "load_ms")
MODES = ("user-felt", "engine")
MX = ["mx-cpu", "mx-gpu", "mx-auto"]
ENGINE_ORDER = ["pandas", "polars", "duckdb", "mx-cpu", "mx-gpu", "mx-auto"]
NAMES = {"pandas": "pandas", "polars": "Polars", "duckdb": "DuckDB", "mx-cpu": "mx-cpu",
         "mx-gpu": "mx-gpu", "mx-auto": "mx-auto"}
VALID = ("ok", "drift", "ref")     # results that may count as fastest / as a win


# ------------------------------------------------------------------ helpers (pure) -----------
def is_valid(rec: dict | None) -> bool:
    """Has a headline time and a result verified against DuckDB (ok / drift / the reference
    itself). WRONG, errored and UNCHECKED (reference failed) results can never win."""
    return bool(rec) and not rec.get("error") and rec.get("first_ms") is not None \
        and rec.get("verdict") in VALID


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


def _mark(ms: float | None, verdict: str | None, err: float | None = None) -> str:
    text = fmt_ms(ms)
    if verdict == "wrong":
        return f"{text} WRONG"
    if verdict in ("unchecked", None, ""):
        return f"{text} ?"
    if verdict == "drift" and err is not None:
        return text + f" †{err:.0e}".replace("e-0", "e-")
    return text


def fmt_cell(rec: dict | None, fastest: bool = False, key: str = "first_ms") -> str:
    if rec is None:
        return "n/a"
    if rec.get("error"):
        return _short_error(rec["error"])
    text = _mark(rec.get(key), rec.get("verdict"), rec.get("max_rel_err"))
    return f"**{text}**" if fastest else text


def fmt_sum_cell(rec: dict | None) -> str:
    """import + first."""
    if rec is None:
        return "n/a"
    if rec.get("error"):
        return _short_error(rec["error"])
    tot = None if rec.get("first_ms") is None else (rec.get("import_ms") or 0.0) + rec["first_ms"]
    return _mark(tot, rec.get("verdict"), rec.get("max_rel_err"))


def fmt_repeat_cell(rec: dict | None) -> str:
    if rec is None or rec.get("error") or rec.get("repeat_ms") is None:
        return "-"
    v = rec.get("repeat_verdict")
    if v in ("ok", "drift"):
        return fmt_ms(rec["repeat_ms"])
    return _mark(rec["repeat_ms"], v if v else "unchecked")


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


def totals(by: dict, engines: list[str], queries: list[int], with_import: bool = False) -> dict:
    """Sum of times over the queries where EVERY engine in the table has a valid result."""
    qs = [q for q in queries if all(is_valid(by.get((e, q))) for e in engines)]
    return {e: sum(by[(e, q)]["first_ms"] + (by[(e, q)].get("import_ms") or 0.0 if with_import else 0.0)
                   for q in qs) for e in engines}, len(qs)


# ------------------------------------------------------------------ CSV -----------------------
def new_row(rec: dict) -> dict:
    return {k: rec.get(k) for k in CSV_FIELDS}


def read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in _FLOATS:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else None
            r["query"] = int(r["query"])
            r["rows"] = int(float(r["rows"])) if r.get("rows") not in (None, "") else None
            r["correct"] = {"True": True, "False": False}.get(r.get("correct"))
            for k in ("error", "repeat_note", "compare_note", "repeat_verdict"):
                r[k] = r.get(k) or ""
            r["mode"] = r.get("mode") or "user-felt"      # CSVs from before F-024 have no mode
            rows.append(r)
    return rows


def mode_of(rows: list[dict]) -> str:
    """The scoreboard mode of a CSV's rows ("user-felt" or "engine"); a mix is an error."""
    modes = {r.get("mode") or "user-felt" for r in rows}
    if len(modes) > 1:
        raise ValueError(f"rows from more than one mode: {sorted(modes)}")
    return modes.pop() if modes else "user-felt"


# ------------------------------------------------------------------ environment ----------------
def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        pass
    # conda packages without pip metadata (MAX, Mojo): read the conda-meta record
    hits = sorted(glob.glob(os.path.join(sys.prefix, "conda-meta", f"{pkg}-[0-9]*.json")))
    if hits:
        return os.path.basename(hits[-1])[len(pkg) + 1:].rsplit("-", 2)[0]
    return "?"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment_info(args, data_root, elapsed_s: float, gpu_desc: str = "") -> dict:
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
        "gpu_desc": gpu_desc or f"index {args.gpu}", "timeout": args.timeout,
        "elapsed_min": elapsed_s / 60.0, "data_root": str(data_root),
        "git": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]),
        "versions": {p: _ver(p) for p in ("pandas", "polars", "duckdb", "pyarrow", "numpy", "max", "mojo")},
        "loadavg": " ".join(f"{x:.1f}" for x in os.getloadavg()),
    }


# ------------------------------------------------------------------ markdown -------------------
def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _outside_flag(q: int) -> str:
    return "PyArrow/NumPy" if q in spec.OUTSIDE_MX else "-"


def render_markdown(rows: list[dict], meta: dict, engines: list[str], queries: list[int],
                    sfs: list[str]) -> str:
    L: list[str] = []
    a = L.append
    v = meta["versions"]
    mx_shown = any(e in engines for e in MX)
    engine_only = mode_of(rows) == "engine"
    if engine_only:
        a(f"# Engine-only TPC-H scoreboard - {meta['date']}\n")
        a("**Engine-only:** a fresh Python process; each engine first loads the columns the query uses "
          "from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table "
          "under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized "
          "result* only. No warm-up. This isolates the query engine from file reading; the user-felt "
          "scoreboard (clock around read + query) stays the headline for what a user waits for. "
          "Load and `import` times are reported separately. "
          "Numbers are milliseconds; **bold** = fastest verified engine for that query.\n")
    else:
        a(f"# User-felt TPC-H scoreboard - {meta['date']}\n")
        a("**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, "
          "the clock around *Parquet read + query + materialized result*. No warm-up. "
          "`import` time is reported separately (and summed in the \"import + first\" tables). "
          "Numbers are milliseconds; **bold** = fastest verified engine for that query.\n")
    a("Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict "
      "1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's "
      "ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not "
      "verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result "
      f"(timeout {meta['timeout']:.0f}s per subprocess). `outside mx` = the mx engines do part of that query with "
      "PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other "
      "mx query is pure mxframe.\n")

    for sf in sfs:
        srows = [r for r in rows if r["sf"] == sf]
        if not srows:
            continue
        by = {(r["engine"], r["query"]): r for r in srows}
        eng = [e for e in engines if any(r["engine"] == e for r in srows)]     # engines run at this SF
        a(f"\n## Scale factor {sf}: headline (first run in a fresh process), ms\n")
        body = []
        fastest_count = {e: 0 for e in eng}
        for q in queries:
            valid = [(by[(e, q)]["first_ms"], e) for e in eng if is_valid(by.get((e, q)))]
            best = min(valid)[1] if valid else None
            if best:
                fastest_count[best] += 1
            body.append([f"Q{q} {spec.TITLES[q]}"]
                        + [fmt_cell(by.get((e, q)), fastest=(e == best)) for e in eng]
                        + ([_outside_flag(q)] if mx_shown else []))
        tot, nq = totals(by, eng, queries)
        body.append([f"**sum over the {nq} queries valid for every engine**"]
                    + [f"**{fmt_ms(tot[e])}**" for e in eng] + (["" ] if mx_shown else []))
        a(_table(["query"] + [NAMES[e] for e in eng] + (["outside mx"] if mx_shown else []), body))
        a("\nFastest verified engine per query: "
          + ", ".join(f"{NAMES[e]} {n}" for e, n in fastest_count.items()) + ".\n")

        a(f"### Wins at SF{sf} (headline, both results verified)\n")
        n = len(queries)
        for m in [e for e in MX if e in eng]:
            parts = []
            for rival in ("polars", "pandas", "duckdb"):
                if rival in eng:
                    w, c = win_counts(rows, sf, queries, m, rival)
                    parts.append(f"**{m} beats {NAMES[rival]} on {w}/{n}** ({c} comparable)")
            a("- " + "; ".join(parts))
        a("")

        if mx_shown and "polars" in eng:
            for label, mxs in (("mx-auto", ["mx-auto"]), ("best of mx-cpu / mx-gpu / mx-auto", MX)):
                mxs = [e for e in mxs if e in eng]
                lost = losses_vs(rows, sf, queries, mxs)
                if lost and (label == "mx-auto" or len(mxs) > 1):
                    a(f"### Where {label} loses to Polars at SF{sf} (worst first)\n")
                    a(_table(["query", "engine", "MXFrame ms", "Polars ms", "slower by"],
                             [[f"Q{q} {spec.TITLES[q]}", e, fmt_ms(m), fmt_ms(p), f"{r:.1f}x"]
                              for r, q, e, m, p in lost]))
                    a("")

        a(f"### SF{sf}: import + first run, ms (headline + engine import time)\n")
        body = []
        for q in queries:
            body.append([f"Q{q}"] + [fmt_sum_cell(by.get((e, q))) for e in eng])
        tot, nq = totals(by, eng, queries, with_import=True)
        body.append([f"**sum ({nq} queries)**"] + [f"**{fmt_ms(tot[e])}**" for e in eng])
        a(_table(["query"] + [NAMES[e] for e in eng], body))
        if mx_shown:
            a("\n> `import mxframe` starts a background thread that warms the CPU kernel libraries while "
              "the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline "
              "unless it contends with the query for cores; the other eng have no equivalent.\n")

        if engine_only:
            a(f"### SF{sf}: load time, ms (Parquet -> in-memory tables, outside the clock)\n")
            body = [[f"Q{q}"] + [fmt_ms((by.get((e, q)) or {}).get("load_ms")) for e in eng] for q in queries]
            a(_table(["query"] + [NAMES[e] for e in eng], body))
            a("\n> Each engine reads only the columns the query uses, but no engine can skip row groups "
              "with the query's filters here (Polars and DuckDB do in the user-felt run).\n")

        a(f"### SF{sf}: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)\n")
        body = [[f"Q{q}"] + [fmt_repeat_cell(by.get((e, q))) for e in eng] for q in queries]
        a(_table(["query"] + [NAMES[e] for e in eng], body))
        a("\n> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). "
          "Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically "
          "fast; that is why the headline is a fresh process.\n")

        a(f"### SF{sf}: import time, ms (median over queries; not in the headline)\n")
        imp = []
        for e in eng:
            xs = [r["import_ms"] for r in srows if r["engine"] == e and r.get("import_ms") is not None]
            imp.append(fmt_ms(statistics.median(xs)) if xs else "-")
        a(_table([""] + [NAMES[e] for e in eng], [["import"] + imp]))
        a("")

        bad = [r for r in srows if r["engine"] != "duckdb"
               and (r.get("verdict") in ("wrong", "drift", "unchecked") or r.get("repeat_verdict") == "wrong"
                    or r.get("error"))]
        if bad:
            a(f"### SF{sf}: correctness against DuckDB and failures\n")
            body = []
            for r in sorted(bad, key=lambda r: (r["query"], r["engine"])):
                what = r.get("error") or r.get("compare_note") or ""
                if r.get("repeat_verdict") == "wrong":
                    what = (what + " | repeat run WRONG").strip(" |")
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

    loads = [r["load1"] for r in rows if r.get("load1") is not None]
    load_txt = (f"1-minute load average at measurement time: median {statistics.median(loads):.1f}, "
                f"max {max(loads):.1f} over {len(loads)} runs; " if loads else "")
    off = sorted((r["query"], r["compare_note"]) for r in rows
                 if r["engine"] == "duckdb" and r["sf"] == "1" and r.get("compare_note", "").startswith("official"))
    if off:
        a("\n## Sanity check of the DuckDB reference against published TPC-H SF1 answers\n")
        a("The harness compares every engine with DuckDB, so the reference itself is checked against the "
          "published SF1 answers (full result, or the top rows for LIMIT queries): "
          + ", ".join(f"Q{q} {n.split(': ', 1)[1]}" for q, n in off) + ".\n")
    if any(r["sf"] != "1" for r in rows) and not any(r["engine"] == "mx-gpu" and r["sf"] != "1" for r in rows):
        a("\n**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the "
          "join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.\n")
    a("\n## Methodology\n")
    if engine_only:
        a("- **One subprocess per (engine, query, scale factor)**; the worker imports the engine "
          "(timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), "
          "then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in "
          "the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed "
          "by table identity cannot answer it.")
    else:
        a("- **One subprocess per (engine, query, scale factor)**; the worker imports the engine "
          "(timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and "
          "once more in the same process (`repeat_ms`).")
    a(f"- **Data** ({meta.get('data_variant', 'exact')}): DuckDB `dbgen` Parquet (dates as int32 days since epoch, "
      "decimals as float32), read "
      "from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).")
    a("- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB "
      "and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as "
      "dictionaries) prune to the columns each query references; pandas uses "
      "`pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).")
    a("- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB "
      "(SQL over `read_parquet`, also the correctness reference), MXFrame `device=\"cpu\"|\"gpu\"|\"auto\"` "
      "through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it "
      "with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed "
      "region and listed per query below.")
    a("- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; "
      "float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the "
      "cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an "
      "engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.")
    a(f"- **Hardware**: {meta['cpu']} ({meta['cores']} cores); GPU: {meta['gpu']} "
      f"({meta['gpu_desc']}; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same "
      f"card nvidia-smi calls by that index). **The machine was shared with other jobs during this run "
      f"({load_txt}load average at report time: {meta['loadavg']}), so timings are indicative.**")
    a(f"- **Versions**: " + ", ".join(f"{k} {x}" for k, x in v.items()) + f"; mxframe at git `{meta['git']}`. "
      f"Last invocation took {meta['elapsed_min']:.0f} min. Data root: `{meta['data_root']}`.")
    a("\n### Query semantics\n")
    a("All engines and the DuckDB reference compute the same query. Where it differs from the official "
      "TPC-H text the difference is listed here; this is not a TPC-certified benchmark.\n")
    for q, text in sorted(spec.DEVIATIONS.items()):
        a(f"- Q{q}: {text}")
    a("")
    a("**Work the mx engines do outside mxframe** (PyArrow / NumPy inside the timed region; all other "
      "queries are pure mxframe):\n")
    for q, text in sorted(spec.OUTSIDE_MX.items()):
        a(f"- Q{q}: {text}")
    a("")
    return "\n".join(L) + "\n"
