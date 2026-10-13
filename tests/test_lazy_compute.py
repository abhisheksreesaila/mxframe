"""F-035: `import pyarrow.compute` costs ~20-25 ms per process, and a plan that runs natively never
calls it. mxframe defers it to first use (mxframe/_lazy_pc.py): importing mxframe, the background
warm-up, reading Parquet and a natively executed query must not load it."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from tpch_user.runner import REPO_ROOT, worker_env  # noqa: E402


@pytest.fixture(scope="module")
def tpch_tiny_dir(tmp_path_factory):
    gen = importlib.import_module("gen_tpch_parquet")
    d = tmp_path_factory.mktemp("tpch") / "tpch_sf0.002_exact"
    gen.generate(0.002, str(d))
    return str(d)

_RUN = """
import json, sys
import mxframe
from mxframe import native_exec
mxframe._warm_thread and mxframe._warm_thread.join()
out = {"after_import": "pyarrow.compute" in sys.modules}
from tpch_user import q_mx
q = int(sys.argv[2])
t = q_mx.load_mem(q, sys.argv[1])
out["after_load"] = "pyarrow.compute" in sys.modules
r = q_mx.run_mem(q, t, "cpu")
out["native"] = native_exec.last_path()
out["after_query"] = "pyarrow.compute" in sys.modules
out["result"] = r.to_pydict()
print("@@" + json.dumps(out))
"""


def _run(data_dir, q, env_extra):
    env = worker_env("-1")
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", _RUN, str(data_dir), str(q)], capture_output=True, text=True,
                       env=env, cwd=str(REPO_ROOT), timeout=120)
    line = [l for l in p.stdout.splitlines() if l.startswith("@@")]
    assert line, p.stdout + p.stderr
    return json.loads(line[0][2:])


def test_native_query_never_imports_pyarrow_compute(tpch_tiny_dir):
    out = _run(tpch_tiny_dir, 6, {"MXFRAME_NATIVE": "1"})
    assert out["native"] == "native"
    assert not (out["after_import"] or out["after_load"] or out["after_query"])
    old = _run(tpch_tiny_dir, 6, {"MXFRAME_NATIVE": "0"})          # same answer on the Python path
    assert old["result"]["revenue"][0] == pytest.approx(out["result"]["revenue"][0], rel=1e-12)


def test_python_path_loads_compute_when_it_needs_it(tpch_tiny_dir):
    """Q1 on the Python path sorts with Arrow compute: it loads on that first use, and works."""
    out = _run(tpch_tiny_dir, 1, {"MXFRAME_NATIVE": "0"})
    assert out["after_import"] is False and out["after_load"] is False
    assert out["after_query"] is True
    nat = _run(tpch_tiny_dir, 1, {"MXFRAME_NATIVE": "1"})
    assert nat["native"] == "native" and nat["after_query"] is False
    assert out["result"].keys() == nat["result"].keys()
    for c, vals in out["result"].items():
        assert vals == pytest.approx(nat["result"][c], rel=1e-12) if isinstance(vals[0], float) else vals == nat["result"][c]


@pytest.mark.xfail(strict=True, reason="the join's final gather is Arrow `take`, which imports pyarrow.compute "
                                       "itself; passes once the gather is native (F-032)")
def test_join_query_with_dictionary_predicates_never_imports_pyarrow_compute(tpch_tiny_dir):
    """Q3: leaf filters (a dictionary predicate on c_mktsegment) run natively, string membership is
    computed over the dictionary values in Python, the final aggregate is native (F-032)."""
    out = _run(tpch_tiny_dir, 3, {"MXFRAME_NATIVE": "1"})
    assert not (out["after_import"] or out["after_load"] or out["after_query"]), out
    old = _run(tpch_tiny_dir, 3, {"MXFRAME_NATIVE": "0"})
    assert out["result"] == old["result"] or all(
        (a == pytest.approx(b, rel=1e-12)) for c in out["result"] for a, b in zip(out["result"][c], old["result"][c]))
