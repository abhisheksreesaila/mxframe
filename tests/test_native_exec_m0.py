"""F-030 (M0): one call into the Mojo executor runs a filter + group-by + aggregate + sort plan.

Python compiles the plan once and makes ONE native call (`mx_execute`); inputs go in through the
Arrow C Data Interface and the result comes back as Arrow buffers the executor allocated (no NumPy
on the way in or out). Group keys, aggregate finishing (mean, count), sorting and building the
output (string keys included) all happen in Mojo. Anything outside M0 falls back to the existing
path with the same answer. On by default since F-031 (MXFRAME_NATIVE=0 turns it off)."""
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
    monkeypatch.setitem(nx._last, "path", None)      # no stale path from an earlier test
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
    assert nx.last_path() == "native"
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
    assert nx.last_path() == "native", nx.last_decline()
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
    assert nx.last_path() == "native"
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = PLANS["dict key, sort"](t).compute(device="cpu")
    np.testing.assert_allclose(got.column("s").to_pylist(), old.column("s").to_pylist(), rtol=1e-12)
    assert got.column("n").to_pylist() == old.column("n").to_pylist()


DECLINED = {
    "no aggregate": lambda t: LazyFrame(t).filter(col("d") < lit(5)).select("f", "v"),
}


@pytest.mark.parametrize("name", list(DECLINED))
def test_outside_m0_falls_back_with_the_same_answer(native, monkeypatch, name):
    t = _table()
    got = DECLINED[name](t).compute(device="cpu")
    assert nx.last_path() != "native"
    assert len(native) == 0                                   # declined before any native call
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = DECLINED[name](t).compute(device="cpu")
    assert got.equals(old)


def test_aggregate_after_a_join_runs_natively_on_the_materialized_join(native, monkeypatch):
    """The join is materialized first (Python orchestration until M2); the aggregate over its result
    is a single-table plan, which the executor runs since M1."""
    t = _table()
    plan = lambda: (LazyFrame(t).join(LazyFrame(t.select(["h", "d"]).slice(0, 10)), on="h")
                    .groupby("f").agg(col("v").sum().alias("s")).sort("f").compute(device="cpu"))
    got = plan()
    assert nx.last_path() == "native" and len(native) == 1
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = plan()
    np.testing.assert_allclose(got.column("s").to_pylist(), old.column("s").to_pylist(), rtol=1e-12)
    assert got.column("f").to_pylist() == old.column("f").to_pylist()


def test_on_by_default_and_opt_out(monkeypatch):
    monkeypatch.delenv("MXFRAME_NATIVE", raising=False)
    assert nx.enabled()                                       # default since F-031
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
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
    assert "@@ native False" in p.stdout, (p.stdout[-500:], p.stderr[-1500:])


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
    assert nx.last_path() == "native", nx.last_decline()
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
    assert nx.last_path() == "native", nx.last_decline()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = PLANS["dict key, sort"](t).compute(device="cpu")
    assert got.column("n").to_pylist() == old.column("n").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), old.column("s").to_pylist(), rtol=1e-12)


# ---- F-030 review: the multi-threaded and large paths behind the headline ----------------------

def _check_same(lf_fn, monkeypatch, expect_native=True):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    got = lf_fn().compute(device="cpu")
    path, why = nx.last_path(), nx.last_decline()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = lf_fn().compute(device="cpu")
    assert (path == "native") == expect_native, why
    assert got.schema == old.schema and got.num_rows == old.num_rows
    for c in got.column_names:
        g, o = got.column(c).to_pylist(), old.column(c).to_pylist()
        if pa.types.is_floating(got.schema.field(c).type):
            np.testing.assert_allclose(g, o, rtol=1e-12, err_msg=c)
        else:
            assert g == o, c
    return why


def _big(n=600_000, seed=4, chunks=7):
    r = np.random.default_rng(seed)
    t = pa.table({
        "f": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, n).astype(np.int32)), A(["A", "N", "R"])),
        "g": A(r.integers(0, 50, n).astype(np.int32)),
        "big": A(r.integers(0, 20_000, n).astype(np.int64)),
        "v": A(r.random(n) * 100), "q": A(r.integers(-1000, 1000, n).astype(np.int32)),
        "d": A(r.integers(0, 1000, n).astype(np.int32)),
    })
    cuts = np.linspace(0, n, chunks + 1).astype(int)
    return pa.concat_tables([t.slice(a, b - a) for a, b in zip(cuts[:-1], cuts[1:])])


def test_many_morsels_and_threads_with_min_max(monkeypatch):
    t = _big()                       # ~10 morsels of 64k rows: several workers, then the merge
    _check_same(lambda: LazyFrame(t).filter(col("d") < lit(900)).groupby("f", "g")
                .agg(col("v").sum().alias("s"), col("v").mean().alias("m"), col("q").min().alias("lo"),
                     col("q").max().alias("hi"), col("v").count().alias("n")).sort("f", "g"), monkeypatch)
    _check_same(lambda: LazyFrame(t).filter(col("d") < lit(500)).groupby()
                .agg(col("v").sum().alias("s"), col("q").min().alias("lo"), col("q").max().alias("hi")), monkeypatch)


def test_more_than_4096_groups(monkeypatch):
    t = _big()                       # 20k groups: the group-id array path, not the per-morsel one
    _check_same(lambda: LazyFrame(t).groupby("big").agg(col("v").sum().alias("s"), col("v").count().alias("n"))
                .sort("big"), monkeypatch)


def test_large_dictionaries_merge_natively(native, monkeypatch):
    """M0 capped merged dictionaries at 4096 strings (linear merge, declined past it); since M1 the
    merge hashes the strings, so 5000 distinct strings over chunks with different dictionaries run."""
    n = 20_000
    r = np.random.default_rng(2)
    parts = []
    for shift in (0, 1700, 3400):
        words = [f"w{(i + shift) % 5000:05d}" for i in range(5000)]
        parts.append(pa.table({"f": pa.DictionaryArray.from_arrays(A(r.integers(0, 5000, n).astype(np.int32)), A(words)),
                               "v": A(r.random(n))}))
    t = pa.concat_tables(parts)
    _check_same(lambda: LazyFrame(t).groupby("f").agg(col("v").sum().alias("s")).sort("f", descending=[True]),
                monkeypatch)


def test_dictionaries_repeating_in_non_adjacent_chunks(monkeypatch):
    r = np.random.default_rng(6)
    da, db = A(["A", "B", "C"]), A(["C", "A"])
    parts = []
    for d in [da, db, da, da, db] * 4:
        n = 3000
        parts.append(pa.table({"f": pa.DictionaryArray.from_arrays(A(r.integers(0, len(d), n).astype(np.int32)), d),
                               "v": A(r.random(n))}))
    t = pa.concat_tables(parts)
    _check_same(lambda: LazyFrame(t).groupby("f").agg(col("v").sum().alias("s"), col("v").count().alias("n"))
                .sort("f"), monkeypatch)


def test_an_executor_exception_falls_back(native, monkeypatch):
    def boom():
        raise OSError("library could not be loaded")
    monkeypatch.setattr(nx, "_load", boom)
    t = _table()
    got = PLANS["dict key, sort"](t).compute(device="cpu")
    assert nx.last_path() is None and "OSError" in nx.last_decline()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    assert got.equals(PLANS["dict key, sort"](t).compute(device="cpu"))


def test_invalid_dictionary_codes_are_declined_not_read():
    """A dictionary index past its dictionary (invalid Arrow) must not be read through: Mojo declines."""
    code = """
import os, sys
os.environ["MXFRAME_NATIVE"] = "1"
import numpy as np, pyarrow as pa
import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A
idx = A(np.array([0, 1, 7, 1] * 1000, np.int32))         # 7 is outside a 3-entry dictionary
f = pa.DictionaryArray.from_buffers(pa.dictionary(pa.int32(), pa.string()), 4000, idx.buffers(), A(["A", "N", "R"]))
t = pa.table({"f": f, "v": A(np.ones(4000))})
try:
    LazyFrame(t).groupby("f").agg(col("v").sum().alias("s")).sort("f").compute(device="cpu")
except Exception:
    pass                                                     # the old path may reject it too
print("@@", nx.last_decline())
"""
    root = Path(__file__).resolve().parent
    code = f"import sys; sys.path.insert(0, {str(root)!r}); import conftest\n" + code
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert "@@ executor declined (code -11)" in p.stdout, (p.returncode, p.stdout[-500:], p.stderr[-1500:])
