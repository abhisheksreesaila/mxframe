"""F-024: the engine-only scoreboard mode. Each engine loads the columns its query uses into
memory outside the clock, then runs the query on that. On a generated SF0.01 dataset the
engine-only results must equal the user-felt ones, and the timed part must not touch the files."""
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from tpch_user import report, spec  # noqa: E402
from tpch_user.compare import compare_tables  # noqa: E402

ENGINES = [("pandas", "q_pandas", None), ("polars", "q_polars", None), ("duckdb", "q_duckdb", None),
           ("mx-cpu", "q_mx", "cpu")]
QUERIES = sorted(spec.OUTPUT)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    sys.path.insert(0, str(SCRIPTS))
    gen = importlib.import_module("gen_tpch_parquet")
    d = tmp_path_factory.mktemp("tpch") / "tpch_sf0.01_exact"
    gen.generate(0.01, str(d))
    return str(d)


def _arrow(mod, res, q):
    tbl = mod.to_arrow(res)
    names = spec.OUTPUT[q]
    return tbl.select(names) if all(n in tbl.column_names for n in names) else tbl.rename_columns(names)


@pytest.mark.parametrize("engine,modname,device", ENGINES, ids=[e[0] for e in ENGINES])
def test_engine_only_equals_user_felt_and_never_reads_the_files(data_dir, tmp_path, engine, modname, device):
    mod = importlib.import_module(f"tpch_user.{modname}")
    for q in QUERIES:
        want = _arrow(mod, mod.run(q, data_dir, device), q)
        tables = mod.load_mem(q, data_dir)
        # the files are gone while the timed part runs: it can only use what was loaded
        moved = tmp_path / f"away_{q}"
        shutil.move(data_dir, moved)
        try:
            got = _arrow(mod, mod.run_mem(q, tables, device), q)
        finally:
            shutil.move(str(moved), data_dir)
        cmp = compare_tables(got, want, topn_metric=spec.TOPN_METRIC.get(q), order_by=spec.ORDER_BY.get(q))
        assert cmp.ok, f"{engine} Q{q}: {cmp.message}"


def _worker(engine, q, data_dir, result, mode):
    from tpch_user.runner import REPO_ROOT, parse_output, worker_env
    p = subprocess.run([sys.executable, "-m", "tpch_user.worker", "--engine", engine, "--query", str(q),
                        "--data-dir", data_dir, "--result", result, "--mode", mode],
                       capture_output=True, text=True, env=worker_env("-1"), cwd=str(REPO_ROOT), timeout=300)
    return parse_output(p.stdout), p


@pytest.mark.parametrize("engine", ["polars", "mx-cpu"])
def test_worker_engine_mode_reports_load_and_reloads_for_the_repeat(data_dir, tmp_path, engine):
    rec, p = _worker(engine, 3, data_dir, str(tmp_path / "r.parquet"), "engine")
    assert "error" not in rec, (rec, p.stdout[-2000:], p.stderr[-2000:])
    assert rec["mode"] == "engine"
    assert rec["load_ms"] > 0 and rec["first_ms"] > 0 and rec["repeat_ms"] > 0
    assert rec["repeat_reloaded"] is True
    assert (tmp_path / "r.parquet").exists() and (tmp_path / "r.parquet.repeat").exists()
    # user-felt mode is unchanged: no load step
    rec, p = _worker(engine, 3, data_dir, str(tmp_path / "u.parquet"), "user-felt")
    assert "error" not in rec, (rec, p.stdout[-2000:])
    assert rec["mode"] == "user-felt" and "load_ms" not in rec


def _rows(mode):
    return [report.new_row({"engine": e, "sf": "1", "query": 1, "first_ms": ms, "load_ms": ld,
                            "import_ms": 10.0, "verdict": v, "mode": mode})
            for e, ms, ld, v in (("duckdb", 20.0, 50.0, "ref"), ("polars", 30.0, 60.0, "ok"),
                                 ("mx-cpu", 25.0, 90.0, "ok"))]


def _meta():
    return {"date": "2026-10-12", "versions": {"polars": "x"}, "timeout": 60, "cpu": "c", "cores": 1,
            "gpu": "g", "gpu_desc": "d", "git": "abc", "elapsed_min": 1.0, "data_root": "/d",
            "loadavg": "0.1"}


def test_report_names_the_mode_and_shows_load_times(tmp_path):
    import csv
    p = tmp_path / "e.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=report.CSV_FIELDS)
        w.writeheader()
        for r in _rows("engine"):
            w.writerow(r)
    rows = report.read_csv(p)
    assert rows[0]["load_ms"] == 50.0 and report.mode_of(rows) == "engine"
    md = report.render_markdown(rows, _meta(), ["polars", "duckdb", "mx-cpu"], [1], ["1"])
    assert md.startswith("# Engine-only TPC-H scoreboard")
    assert "query + materialized result" in md and "outside the clock" in md
    assert "load time" in md.lower() and "90" in md
    uf = report.render_markdown([dict(r, mode="user-felt", load_ms=None) for r in rows], _meta(),
                                ["polars", "duckdb", "mx-cpu"], [1], ["1"])
    assert uf.startswith("# User-felt TPC-H scoreboard") and "load time" not in uf.lower()


def test_old_csv_without_mode_or_load_columns_reads_as_user_felt(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text("engine,sf,query,import_ms,first_ms,repeat_ms,rows,correct,max_rel_err,verdict,"
                 "repeat_verdict,load1,error,repeat_note,compare_note\n"
                 "polars,1,1,77.0,2164.1,1842.3,4,True,0.0,ok,ok,2.4,,,\n")
    rows = report.read_csv(p)
    assert rows[0]["load_ms"] is None and report.mode_of(rows) == "user-felt"


def test_resume_refuses_a_csv_from_the_other_mode_or_an_old_header(tmp_path):
    import bench_user_felt as b
    p = tmp_path / "x.csv"
    p.write_text("engine,sf,query,import_ms,first_ms,repeat_ms,rows,correct,max_rel_err,verdict,"
                 "repeat_verdict,load1,error,repeat_note,compare_note\n"
                 "polars,1,1,77.0,2164.1,1842.3,4,True,0.0,ok,ok,2.4,,,\n")
    with pytest.raises(SystemExit, match="engine"):
        b.open_log(p, resume=True, mode="engine")
    f, w, existing = b.open_log(p, resume=True, mode="user-felt")    # old user-felt CSV: still resumable
    w.writerow(report.new_row({"engine": "mx-cpu", "sf": "1", "query": 2, "first_ms": 5.0,
                               "verdict": "ok", "mode": "user-felt"}))
    f.close()
    rows = report.read_csv(p)
    assert len(existing) == 1 and len(rows) == 2 and rows[1]["first_ms"] == 5.0
