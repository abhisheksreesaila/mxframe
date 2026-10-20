"""F-080 / F-076: every TPC-H query written as SQL runs as natively as its DataFrame form.

Each query's SQL text (scripts/tpch_sql/queries.py) goes through `mx.sql` in a fresh process under
`scripts/native_coverage.py` (generated SF 0.17 data: plain strings): every compute() took a native path and
no PyArrow compute call touched >= 1000 rows, like tests/test_native_strings_tpch.py for the DataFrame forms.
The answers are DuckDB's: tests/test_sql_tpch.py compares all 22 SQL texts with DuckDB."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import omxcore.native_exec as nx

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

NATIVE_PATHS = ("native@", "join_mojo_shortcut@", "pyarrow_shortcut@")
# declines that touch no rows: a filter over a one-row subquery result, an aggregate over an empty input,
# Q16's wide key space (the star scan + mx_execute answer it natively)
ALLOWED_DECLINES = {"nothing to filter", "empty input", "group key space too wide for direct addressing"}


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tpch_sql_cov")
    importlib.import_module("gen_tpch_parquet").generate(0.17, str(root / "tpch_sf0.17"))
    return root


def _coverage(q, root):
    env = dict(os.environ, MXFRAME_TPCH_DATA=str(root), MXFRAME_STRICT_NATIVE="1", CUDA_VISIBLE_DEVICES="-1")
    cmd = [sys.executable, str(ROOT / "scripts" / "native_coverage.py"), "--one", str(q), "--sf", "0.17", "--sql"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@COV ")]
    assert lines, p.stderr[-2000:]
    return json.loads(lines[-1][6:])


@pytest.mark.parametrize("q", range(1, 23))
def test_sql_query_runs_fully_native(data_root, q):
    r = _coverage(q, data_root)
    if q == 21 and "raised" in r:
        pytest.xfail(f"Q21's correlated `<>` is not decorrelated yet (F-112): {r['raised']}")
    assert "raised" not in r, r
    assert r["arrow"] == [], r
    assert all(p.startswith(NATIVE_PATHS) for p in r["paths"]), r["paths"]
    assert set(r["declines"]) <= ALLOWED_DECLINES, r["declines"]
