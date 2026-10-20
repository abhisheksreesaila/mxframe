"""F-035: `import pyarrow.compute` costs ~20-25 ms per process, and a plan that runs natively never
calls it. omxcore defers it to first use (omxcore/_lazy_pc.py): importing omxcore, reading Parquet
and a natively executed query must not load it. The tests turn the background warm-up off so
they see what the query itself imports."""
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
import omxcore
from omxcore import native_exec
omxcore._warm_thread and omxcore._warm_thread.join()
out = {"after_import": "pyarrow.compute" in sys.modules, "core_loaded": "pyarrow._compute" in sys.modules,
       "lib_loaded": native_exec._lib is not None}
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
    out = _run(tpch_tiny_dir, 6, {"MXFRAME_NATIVE": "1", "MXFRAME_WARM": "0"})
    assert out["native"] == "native"
    assert not (out["after_import"] or out["after_load"] or out["after_query"])
    old = _run(tpch_tiny_dir, 6, {"MXFRAME_NATIVE": "0"})          # same answer on the Python path
    assert old["result"]["revenue"][0] == pytest.approx(out["result"]["revenue"][0], rel=1e-12)


def test_python_path_runs_without_pyarrow_compute(tpch_tiny_dir):
    """Q1 on the Python path sorts and gathers with Arrow compute functions: `_lazy_pc.pc` calls
    them through pyarrow._compute, so pyarrow.compute never loads (F-032 M2f), and the answer is
    the native one."""
    out = _run(tpch_tiny_dir, 1, {"MXFRAME_NATIVE": "0", "MXFRAME_WARM": "0"})
    assert out["after_import"] is False and out["after_load"] is False
    assert out["after_query"] is False
    nat = _run(tpch_tiny_dir, 1, {"MXFRAME_NATIVE": "1", "MXFRAME_WARM": "0"})
    assert nat["native"] == "native" and nat["after_query"] is False
    assert out["result"].keys() == nat["result"].keys()
    for c, vals in out["result"].items():
        assert vals == pytest.approx(nat["result"][c], rel=1e-12) if isinstance(vals[0], float) else vals == nat["result"][c]


def test_join_query_with_dictionary_predicates_never_imports_pyarrow_compute(tpch_tiny_dir):
    """Q3: leaf filters (a dictionary predicate on c_mktsegment) run natively, string membership is
    computed over the dictionary values in Python, the join's gather and the final aggregate are
    native (F-032)."""
    out = _run(tpch_tiny_dir, 3, {"MXFRAME_NATIVE": "1", "MXFRAME_WARM": "0"})
    assert not (out["after_import"] or out["after_load"] or out["after_query"]), out
    old = _run(tpch_tiny_dir, 3, {"MXFRAME_NATIVE": "0"})
    assert out["result"] == old["result"] or all(
        (a == pytest.approx(b, rel=1e-12)) for c in out["result"] for a, b in zip(out["result"][c], old["result"][c]))


# F-032 (M2f): `pc.<name>` for a generated compute function calls pyarrow._compute directly; only
# the functions pyarrow.compute defines in Python (cast, take, fill_null, ...) load the module.

def _parity_cases():
    import pyarrow as pa
    a = pa.array([3, 1, None, 2], pa.int64())
    b = pa.array([1.5, 2.0, 3.0, None])
    s = pa.array(["apple", "banana", None, "cherry"])
    m = pa.array([True, False, None, True])
    t = pa.table({"x": [2, 1, 2, 0], "y": ["b", "a", "a", "c"]})
    return [
        ("add", (a, 1), {}), ("subtract", (a, b), {}), ("multiply", (b, 2.0), {}), ("divide", (b, 2.0), {}),
        ("equal", (a, 2), {}), ("greater", (a, 1), {}), ("greater_equal", (b, 2.0), {}),
        ("and_kleene", (m, True), {}), ("or_kleene", (m, False), {}), ("and_", (m, m), {}), ("invert", (m,), {}),
        ("if_else", (m, a, 0), {}), ("is_in", (s,), {"value_set": pa.array(["apple", "cherry"])}),
        ("match_substring", (s, "an"), {}), ("starts_with", (s,), {"pattern": "ch"}),
        ("sort_indices", (t,), {"sort_keys": [("x", "ascending"), ("y", "descending")]}),
        ("min_max", (a,), {}), ("sum", (b,), {}), ("count_distinct", (a,), {}), ("year", (pa.array([0, 400], pa.date32()),), {}),
        ("cast", (a, pa.float64()), {}), ("cast", (b, pa.int64()), {"safe": False}),
        ("cast", (pa.array(["x", "y", "x"]).dictionary_encode(), pa.string()), {}),
        ("take", (a, pa.array([0, 3])), {}), ("take", (t, pa.array([3, 0])), {}),
        ("fill_null", (a, 0), {}), ("fill_null", (m, pa.scalar(False)), {}),
        ("filter", (t, pa.array([True, False, None, True])), {}), ("filter", (a, m), {"null_selection_behavior": "emit_null"}),
        ("dictionary_encode", (pa.chunked_array([s, s]),), {}),
    ]


@pytest.mark.parametrize("name,args,kwargs", _parity_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_fast_shim_matches_pyarrow_compute(name, args, kwargs, monkeypatch):
    import pyarrow.compute as real
    from omxcore import _lazy_pc
    want = getattr(real, name)(*args, **kwargs)
    # once pyarrow.compute is loaded the shim hands every name to it: unload it so the fast path runs
    monkeypatch.delitem(sys.modules, "pyarrow.compute")
    f = _lazy_pc._wrapper(name)
    assert f is not None and getattr(f, "__module__", None) == "omxcore._lazy_pc"
    got = f(*args, **kwargs)
    assert "pyarrow.compute" not in sys.modules
    assert got.equals(want)


_SHIM = """
import json, sys
class _NoPandas:   # pyarrow imports pandas (which imports pyarrow.compute) when it converts Python values
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "pandas":
            raise ImportError(name)
sys.meta_path.insert(0, _NoPandas())
import pyarrow as pa
from omxcore._lazy_pc import pc
out = {}
a = pa.array([3, 1, 2])
pc.add(a, 1); pc.and_kleene(pa.array([True]), True); pc.sort_indices(pa.table({"x": a}), sort_keys=[("x", "ascending")])
pc.is_in(pa.array(["a"]), value_set=pa.array(["a"])); pc.match_substring(pa.array(["ab"]), "b")
out["generic"] = "pyarrow.compute" in sys.modules
pc.cast(a, pa.float64()); pc.take(a, pc.subtract(a, a)); pc.fill_null(a, a); pc.filter(pa.table({"x": a}), pc.greater(a, a))
pc.dictionary_encode(pa.chunked_array([a]))
out["rebuilt"] = "pyarrow.compute" in sys.modules
pc.index(a, a[0])
out["index"] = "pyarrow.compute" in sys.modules
print("@@" + json.dumps(out))
"""


def test_generated_functions_do_not_import_pyarrow_compute():
    p = subprocess.run([sys.executable, "-c", _SHIM], capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60)
    line = [l for l in p.stdout.splitlines() if l.startswith("@@")]
    assert line, p.stdout + p.stderr
    assert json.loads(line[0][2:]) == {"generic": False, "rebuilt": False, "index": True}


def test_shim_errors_like_pyarrow_compute(monkeypatch):
    import pyarrow as pa
    from omxcore._lazy_pc import _LazyCompute
    monkeypatch.delitem(sys.modules, "pyarrow.compute", raising=False)
    with pytest.raises(AttributeError):
        _LazyCompute().no_such_function
    with pytest.raises(TypeError):
        _LazyCompute().add(pa.array([1]))                                      # wrong arity
    with pytest.raises(TypeError):
        _LazyCompute().is_in(pa.array([1]), value_set=pa.array([1]), options={"value_set": pa.array([1])})


def test_background_warm_up_loads_compute_and_the_native_library(tpch_tiny_dir):
    """The warm-up thread (on by default) loads pyarrow._compute and the executor library, so the
    first query imports neither, and it does not load pyarrow.compute."""
    out = _run(tpch_tiny_dir, 6, {"MXFRAME_NATIVE": "1", "MXFRAME_WARM": "1"})
    assert out["core_loaded"] and out["lib_loaded"]
    assert not (out["after_import"] or out["after_query"]) and out["native"] == "native"
    off = _run(tpch_tiny_dir, 6, {"MXFRAME_NATIVE": "1", "MXFRAME_WARM": "0"})
    assert not off["lib_loaded"]
