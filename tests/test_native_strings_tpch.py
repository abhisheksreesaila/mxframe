"""F-034: the TPC-H queries that need general strings run entirely in the native executor.

Q9 / Q16 / Q20 filter plain utf8 columns with LIKE / contains / startswith; Q7 joins renamed copies of
`nation` (a renaming select over a leaf) and groups by their names; Q9's group key passes through a
`with_columns`. Each query runs in a fresh process under `scripts/native_coverage.py` (the criterion of
F-033: every compute() took a native path and no PyArrow compute call touched >= 1000 rows), on generated
SF 0.17 data (plain strings there; at SF 0.017 the reader dictionary-encodes `p_name`), and the answers are DuckDB's (`test_tpch_small_correct.py` compares all 22 for every engine)."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mxframe.native_exec as nx

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

NATIVE_PATHS = ("native@", "join_mojo_shortcut@", "pyarrow_shortcut@")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tpch_cov")
    importlib.import_module("gen_tpch_parquet").generate(0.17, str(root / "tpch_sf0.17"))
    return root


def _coverage(q, root, sql=False, sf="0.17"):
    env = dict(os.environ, MXFRAME_TPCH_DATA=str(root), MXFRAME_STRICT_NATIVE="1", CUDA_VISIBLE_DEVICES="-1")
    cmd = [sys.executable, str(ROOT / "scripts" / "native_coverage.py"), "--one", str(q), "--sf", sf]
    p = subprocess.run(cmd + (["--sql"] if sql else []), capture_output=True, text=True, env=env, cwd=ROOT)
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@COV ")]
    assert lines, p.stderr[-2000:]
    return json.loads(lines[-1][6:])


@pytest.mark.parametrize("q,allowed_declines", [
    (7, set()),                                                  # renamed partner leaves keep their codes
    (9, set()),                                                  # LIKE on part, a group key through with_columns
    (20, set()),                                                 # startswith on part
])
def test_query_runs_fully_native(data_root, q, allowed_declines):
    r = _coverage(q, data_root)
    assert r["arrow"] == [], r
    assert all(p.startswith(NATIVE_PATHS) for p in r["paths"]), r["paths"]
    assert set(r["declines"]) <= allowed_declines, r["declines"]


def test_sql_q2_suffix_like_runs_fully_native(data_root):
    r = _coverage(2, data_root, sql=True)
    assert "raised" not in r, r
    assert r["arrow"] == [], r
    assert all(p.startswith(NATIVE_PATHS) for p in r["paths"]), r["paths"]


SF1 = Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data")) / "tpch_sf1"


@pytest.mark.skipif(not (SF1 / "supplier.parquet").exists(), reason="TPC-H SF1 data not present")
def test_q16_not_like_on_plain_supplier_comments_runs_fully_native():
    # (the star scan + mx_execute answer it after v3 declines the wide key space; below SF1 the reader
    # dictionary-encodes s_comment, which was always native)
    r = _coverage(16, SF1.parent, sf="1")
    assert r["arrow"] == [], r
    assert all(p.startswith(NATIVE_PATHS) for p in r["paths"]), r["paths"]
    assert set(r["declines"]) <= {"group key space too wide for direct addressing"}, r["declines"]
