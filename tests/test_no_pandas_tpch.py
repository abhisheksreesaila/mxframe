"""F-026: no TPC-H query imports pandas on mx-cpu. PyArrow imports pandas the first time some
conversions run (110-160 ms per process); at 12k rows that was 30x the query itself. Each query
runs in a fresh process on generated SF0.002 data, both as the benchmark runs it (scan_parquet)
and on in-memory tables (engine-only mode)."""
import importlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from tpch_user import spec  # noqa: E402
from tpch_user.runner import REPO_ROOT, worker_env  # noqa: E402

QUERIES = sorted(spec.OUTPUT)

_RUN = """
import json, sys
import mxframe, pyarrow.parquet, pyarrow.compute       # what the benchmark worker imports first
from tpch_user import q_mx
q, d, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
out = {"before": "pandas" in sys.modules}
try:                    # a query that fails (F-027) still says whether it imported pandas
    if mode == "engine":
        t = q_mx.load_mem(q, d)
        out["after_load"] = "pandas" in sys.modules
        q_mx.run_mem(q, t, "cpu")
    else:
        q_mx.run(q, d, "cpu")
except Exception as e:
    out["query_error"] = f"{type(e).__name__}: {e}"[:200]
out["after_query"] = "pandas" in sys.modules
print("@@" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    gen = importlib.import_module("gen_tpch_parquet")
    d = tmp_path_factory.mktemp("tpch") / "tpch_sf0.002_exact"
    gen.generate(0.002, str(d))
    return str(d)


def _run(q, d, mode):
    p = subprocess.run([sys.executable, "-c", _RUN, str(q), d, mode], capture_output=True, text=True,
                       env=worker_env("-1"), cwd=str(REPO_ROOT), timeout=300)
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@")]
    if not lines:
        return q, mode, {"error": (p.stderr or p.stdout)[-400:]}
    return q, mode, json.loads(lines[-1][2:])


@pytest.mark.parametrize("mode", ["user-felt", "engine"])
def test_no_query_imports_pandas(data_dir, mode):
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda q: _run(q, data_dir, mode), QUERIES))
    errors = {q: r["error"] for q, _, r in results if "error" in r}
    assert not errors, errors
    assert not any(r["before"] for _, _, r in results), "pandas imported before the query started"
    if mode == "engine":
        assert not [q for q, _, r in results if r.get("after_load")], "loading imported pandas"
    bad = [q for q, _, r in results if r["after_query"]]
    assert not bad, f"queries that import pandas ({mode}): {bad}"
