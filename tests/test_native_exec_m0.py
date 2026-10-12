"""F-030 (M0): one call into the Mojo executor runs a filter + group-by + aggregate + sort plan.

Python compiles the plan once and makes ONE native call (`mx_execute`); inputs go in through the
Arrow C Data Interface and the result comes back as Arrow buffers the executor allocated (no NumPy
on the way in or out). Group keys, aggregate finishing (mean, count), sorting and building the
output (string keys included) all happen in Mojo. Anything outside M0 falls back to the existing
path with the same answer. Opt-in with MXFRAME_NATIVE=1 during the spike."""
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from tpch_user import spec  # noqa: E402
from tpch_user.compare import compare_tables  # noqa: E402

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    calls = []
    real = nx._call

    def counting(*a):
        calls.append(1)
        return real(*a)

    monkeypatch.setattr(nx, "_call", counting)
    return calls


def _run(lf_fn, native_on, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1" if native_on else "0")
    return lf_fn()


@pytest.fixture(scope="module")
def tpch(tmp_path_factory):
    gen = importlib.import_module("gen_tpch_parquet")
    d = tmp_path_factory.mktemp("tpch") / "tpch_sf0.01_exact"
    gen.generate(0.01, str(d))
    return str(d)


@pytest.mark.parametrize("q", [1, 6])
def test_q1_q6_one_native_call_equal_to_duckdb_and_the_old_path(tpch, native, monkeypatch, q):
    q_mx = importlib.import_module("tpch_user.q_mx")
    q_duckdb = importlib.import_module("tpch_user.q_duckdb")
    tables = q_mx.load_mem(q, tpch)
    got = q_mx.run_mem(q, tables, "cpu")
    assert len(native) == 1                                   # the whole plan: one call
    assert nx.last_path() == "native_m0"
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = q_mx.run_mem(q, tables, "cpu")
    assert got.schema == old.schema
    assert compare_tables(got, old, rtol=1e-12).ok
    want = q_duckdb.run(q, tpch)
    cmp = compare_tables(got.rename_columns(spec.OUTPUT[q]), want, order_by=spec.ORDER_BY.get(q))
    assert cmp.ok, cmp.message


def _table(n=20_000, seed=1, chunks=1):
    r = np.random.default_rng(seed)
    flags = pa.array(["A", "N", "R"])
    t = pa.table({
        "f": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, n).astype(np.int32)), flags),
        "g": A(r.integers(-5, 7, n).astype(np.int32)),
        "h": A(r.integers(0, 4, n).astype(np.int64)),
        "v": A(r.random(n) * 100),
        "w": A(r.random(n).astype(np.float32)),
        "d": A(r.integers(0, 1000, n).astype(np.int32)),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        t = pa.concat_tables([t.slice(a, b - a) for a, b in zip(cuts[:-1], cuts[1:])])
    return t


PLANS = {
    "dict key, sort": lambda t: LazyFrame(t).filter(col("d") < lit(700)).groupby("f")
        .agg(col("v").sum().alias("s"), col("v").mean().alias("m"), col("w").count().alias("n")).sort("f"),
    "int keys desc": lambda t: LazyFrame(t).groupby("g", "h")
        .agg((col("v") * lit(2.0) - col("w")).sum().alias("e"), col("v").count().alias("n"))
        .sort("g", "h", descending=[True, False]),
    "mixed keys": lambda t: LazyFrame(t).filter(col("v") > lit(10.0)).groupby("h", "f")
        .agg(col("w").mean().alias("m")).sort("h", "f"),
    "global": lambda t: LazyFrame(t).filter((col("d") >= lit(100)) & (col("d") < lit(200)))
        .groupby().agg((col("v") * col("w")).sum().alias("r"), col("v").count().alias("n")),
    "global over no rows": lambda t: LazyFrame(t).filter(col("d") > lit(5000))
        .groupby().agg(col("v").sum().alias("s"), col("v").mean().alias("m"), col("v").count().alias("n")),
}


@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("name", list(PLANS))
def test_shapes_match_the_old_path(native, monkeypatch, name, chunks):
    t = _table(chunks=chunks)
    got = PLANS[name](t).compute(device="cpu")
    assert nx.last_path() == "native_m0", nx.last_decline()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = PLANS[name](t).compute(device="cpu")
    assert got.schema == old.schema, (got.schema, old.schema)
    if name == "global over no rows":
        assert got.to_pydict() == old.to_pydict() == {"s": [None], "m": [None], "n": [0]}
    else:
        assert got.column_names == old.column_names and got.num_rows == old.num_rows
        for c in got.column_names:
            g, o = got.column(c).to_pylist(), old.column(c).to_pylist()
            if pa.types.is_floating(got.schema.field(c).type):
                np.testing.assert_allclose(g, o, rtol=1e-12, err_msg=c)
            else:
                assert g == o, c


def test_sliced_input_with_offsets(native, monkeypatch):
    t = _table(n=30_000).slice(1234, 20_000)
    got = PLANS["dict key, sort"](t).compute(device="cpu")
    assert nx.last_path() == "native_m0"
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = PLANS["dict key, sort"](t).compute(device="cpu")
    np.testing.assert_allclose(got.column("s").to_pylist(), old.column("s").to_pylist(), rtol=1e-12)
    assert got.column("n").to_pylist() == old.column("n").to_pylist()


DECLINED = {
    "join": lambda t: LazyFrame(t).join(LazyFrame(t.select(["h", "d"]).slice(0, 10)), on="h")
        .groupby("f").agg(col("v").sum().alias("s")).sort("f"),
    "having": lambda t: LazyFrame(t).groupby("f").agg(col("v").sum().alias("s"))
        .filter(col("s") > lit(1.0)).sort("f"),
    "sort by an aggregate": lambda t: LazyFrame(t).groupby("f").agg(col("v").sum().alias("s")).sort("s"),
    "no aggregate": lambda t: LazyFrame(t).filter(col("d") < lit(5)).select("f", "v"),
}


@pytest.mark.parametrize("name", list(DECLINED))
def test_outside_m0_falls_back_with_the_same_answer(native, monkeypatch, name):
    t = _table()
    got = DECLINED[name](t).compute(device="cpu")
    assert nx.last_path() != "native_m0"
    assert len(native) == 0                                   # declined before any native call
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = DECLINED[name](t).compute(device="cpu")
    assert got.equals(old)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MXFRAME_NATIVE", raising=False)
    assert not nx.enabled()


def test_no_pandas_and_no_numpy_conversion_of_the_input():
    code = """
import os, sys
os.environ["MXFRAME_NATIVE"] = "1"
import numpy as np, pyarrow as pa
import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A
n = 50_000
r = np.random.default_rng(0)
t = pa.table({"f": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, n).astype(np.int32)), A(["A", "N", "R"])),
              "v": A(r.random(n)), "d": A(r.integers(0, 100, n).astype(np.int32))})
LazyFrame(t).filter(col("d") < lit(50)).groupby("f").agg(col("v").sum().alias("s")).sort("f").compute(device="cpu")

print("@@", nx.last_path(), "pandas" in sys.modules)
"""
    root = Path(__file__).resolve().parent
    code = f"import sys; sys.path.insert(0, {str(root)!r}); import conftest\n" + code
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert "@@ native_m0 False" in p.stdout, (p.stdout[-500:], p.stderr[-1500:])


def test_chunks_with_different_dictionaries(native, monkeypatch):
    """Each Parquet row group carries its own dictionary, in its own order, sometimes with other
    entries: the executor merges them."""
    r = np.random.default_rng(9)
    parts = []
    for words in (["N", "R", "A"], ["N", "A", "R"], ["R", "Q"], ["A"]):
        n = 5000
        f = pa.DictionaryArray.from_arrays(A(r.integers(0, len(words), n).astype(np.int32)), A(words))
        parts.append(pa.table({"f": f, "v": A(r.random(n)), "d": A(r.integers(0, 100, n).astype(np.int32))}))
    t = pa.concat_tables(parts)
    assert t.column("f").num_chunks == 4
    lf = lambda: (LazyFrame(t).filter(col("d") < lit(80)).groupby("f")
                  .agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("f", descending=[True]))
    got = lf().compute(device="cpu")
    assert nx.last_path() == "native_m0", nx.last_decline()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = lf().compute(device="cpu")
    assert got.column("f").to_pylist() == old.column("f").to_pylist() == ["R", "Q", "N", "A"]
    assert got.column("n").to_pylist() == old.column("n").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), old.column("s").to_pylist(), rtol=1e-12)


def test_columns_chunked_differently(native, monkeypatch):
    """MXFrame's reader returns numeric columns as one chunk and dictionary columns per row group."""
    t = _table(n=30_000)
    f = t.column("f")
    cut = pa.chunked_array([f.chunk(0).slice(0, 7000), f.chunk(0).slice(7000, 13000), f.chunk(0).slice(20000)])
    v = t.column("v")
    v2 = pa.chunked_array([v.chunk(0).slice(0, 11000), v.chunk(0).slice(11000)])
    t2 = t.set_column(0, "f", cut).set_column(3, "v", v2)
    got = PLANS["dict key, sort"](t2).compute(device="cpu")
    assert nx.last_path() == "native_m0", nx.last_decline()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = PLANS["dict key, sort"](t).compute(device="cpu")
    assert got.column("n").to_pylist() == old.column("n").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), old.column("s").to_pylist(), rtol=1e-12)
