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
t = q_mx.load_mem(6, sys.argv[1])
out["after_load"] = "pyarrow.compute" in sys.modules
q_mx.run_mem(6, t, "cpu")
out["native"] = native_exec.last_path()
out["after_query"] = "pyarrow.compute" in sys.modules
print("@@" + json.dumps(out))
"""


def _run(data_dir, env_extra):
    env = worker_env("-1")
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", _RUN, str(data_dir)], capture_output=True, text=True,
                       env=env, cwd=str(REPO_ROOT), timeout=120)
    line = [l for l in p.stdout.splitlines() if l.startswith("@@")]
    assert line, p.stdout + p.stderr
    return json.loads(line[0][2:])


def test_native_query_never_imports_pyarrow_compute(tpch_tiny_dir):
    out = _run(tpch_tiny_dir, {"MXFRAME_NATIVE": "1"})
    assert out == {"after_import": False, "after_load": False, "native": "native_m0", "after_query": False}


def test_python_path_query_still_works_and_loads_compute_lazily(tpch_tiny_dir):
    out = _run(tpch_tiny_dir, {"MXFRAME_NATIVE": "0"})
    assert out["after_import"] is False and out["after_load"] is False
