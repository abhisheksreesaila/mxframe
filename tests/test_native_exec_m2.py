"""F-032 (M2a): leaves under a join are filtered in Mojo.

A ``Filter*(Scan)`` leaf of a join runs as one native call (`mx_filter`): the predicate runs on the
bytecode interpreter per morsel, in parallel, and only the surviving rows of the columns the join
needs are written out (numbers, dates, decimals, strings; dictionary strings decoded on the way out).
Dictionary columns io_parquet used to decode in full because the plan has a join are now decoded
after the leaf's filter, on the surviving rows only, whichever path filters."""
import importlib
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



@pytest.fixture(autouse=True)
def _star_scan(monkeypatch):
    """These tests check the star scan's own mechanics (what it writes, which columns it keeps coded);
    an aggregate over the join would otherwise skip it (F-046a slice 3, tests/test_fused_join_agg.py)."""
    monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "0")

@pytest.fixture
def filters(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    calls = []
    real = nx._call_filter

    def counting(*a):
        calls.append(1)
        return real(*a)

    monkeypatch.setattr(nx, "_call_filter", counting)
    return calls


def _fact(n=120_000, seed=1, chunks=3):
    """A fact table whose chunks each carry their own dictionary (Parquet row groups)."""
    r = np.random.default_rng(seed)
    parts = []
    for words in (["MAIL", "SHIP", "AIR"], ["SHIP", "TRUCK", "MAIL", "RAIL"], ["AIR", "MAIL"])[:chunks]:
        m = n // chunks
        parts.append(pa.table({
            "ok": A(r.integers(0, 5000, m).astype(np.int64)),
            "mode": pa.DictionaryArray.from_arrays(A(r.integers(0, len(words), m).astype(np.int32)), A(words)),
            "note": pa.array([f"n{i % 97}" for i in range(m)], type=pa.string()),
            "ship": A(r.integers(8000, 10000, m).astype(np.int32)),
            "shipd": pa.array(r.integers(8000, 10000, m).astype(np.int32), type=pa.date32()),
            "price": pa.array(r.integers(100, 100000, m).astype(np.int64)).cast(pa.decimal128(19, 0)),
            "qty": A(r.integers(1, 50, m).astype(np.float32)),
            "v": A(r.random(m) * 100),
        }))
    return pa.concat_tables(parts)


def _dim(seed=2, n=5000):
    r = np.random.default_rng(seed)
    return pa.table({
        "ok": A(np.arange(n, dtype=np.int64)),
        "pri": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, n).astype(np.int32)),
                                              A(["1-URGENT", "2-HIGH", "3-LOW"])),
        "day": A(r.integers(8000, 10000, n).astype(np.int32)),
    })


def _same(got, old):
    assert got.schema == old.schema, (got.schema, old.schema)
    assert got.num_rows == old.num_rows
    for c in got.column_names:
        g, o = got.column(c).to_pylist(), old.column(c).to_pylist()
        if pa.types.is_floating(got.schema.field(c).type):
            np.testing.assert_allclose(g, o, rtol=1e-12, err_msg=c)
        else:
            assert g == o, c


def _both(plan, monkeypatch):
    got = plan().compute(device="cpu")
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = plan().compute(device="cpu")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    return got, old


PLANS = {
    "string predicate, aggregate on top": lambda f, d: (
        LazyFrame(f).filter(col("mode").isin(["MAIL", "SHIP"]) & (col("qty") < lit(30.0)))
        .join(LazyFrame(d).filter(col("day") < lit(9500)), on="ok")
        .groupby("mode", "pri").agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("mode", "pri")),
    "rows out (strings, dates, decimals)": lambda f, d: (
        LazyFrame(f).filter((col("ship") >= lit(9900)) & (col("qty") < lit(40.0)))
        .join(LazyFrame(d), on="ok").select("ok", "mode", "note", "shipd", "price", "pri", "qty")
        .sort("ok", "shipd", "note", "qty")),
    "dictionary predicate, string and dictionary keys above": lambda f, d: (
        LazyFrame(f).filter(col("mode").startswith("S") | (col("mode") == lit("AIR")))
        .join(LazyFrame(d), on="ok").groupby("note", "mode").agg(col("v").sum().alias("s")).sort("note", "mode")),
    "nothing survives": lambda f, d: (
        LazyFrame(f).filter(col("qty") > lit(1000.0)).join(LazyFrame(d), on="ok")
        .groupby("pri").agg(col("v").sum().alias("s")).sort("pri")),
}


@pytest.mark.parametrize("name", list(PLANS))
def test_leaf_filters_run_natively_with_the_same_answer(filters, monkeypatch, name):
    f, d = _fact(), _dim()
    got, old = _both(lambda: PLANS[name](f, d), monkeypatch)
    assert len(filters) >= 1, nx.last_filter()
    _same(got, old)


def test_morsels_cross_chunks_with_string_outputs(filters, monkeypatch):
    """Morsels of 64k rows span chunk boundaries: the string bytes of one morsel continue across
    chunks (pass 2), and every chunk keeps its own dictionary."""
    f = _fact(n=300_000, chunks=3)
    d = _dim()
    plan = lambda: (LazyFrame(f).filter(col("mode") != lit("TRUCK")).join(LazyFrame(d), on="ok")
                    .groupby("note", "mode", "pri").agg(col("v").sum().alias("s"), col("v").count().alias("n"))
                    .sort("note", "mode", "pri"))
    got, old = _both(plan, monkeypatch)
    assert len(filters) >= 1
    _same(got, old)


def test_nulls_in_a_leaf_fall_back(filters, monkeypatch):
    f = _fact(n=9000)
    v = f.column("v").to_pylist()
    v[5] = None
    f = f.set_column(f.schema.get_field_index("v"), "v", pa.array(v, type=pa.float64()))
    plan = lambda: (LazyFrame(f).filter(col("qty") < lit(20.0)).join(LazyFrame(_dim()), on="ok")
                    .groupby("pri").agg(col("v").sum().alias("s")).sort("pri"))
    got = plan().compute(device="cpu")
    assert len(filters) == 0 and "nulls" in (nx.last_filter()[1] or ""), nx.last_filter()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    _same(got, plan().compute(device="cpu"))


def test_deferred_dictionary_columns_are_decoded_after_the_filter(monkeypatch):
    """On the Python path too: resolve_plan no longer decodes a join leaf's dictionary column in
    full; the leaf decodes it after its filter, so only surviving rows are decoded."""
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    import mxframe.arrow_dict as ad
    import mxframe.join_engine as je
    sizes = []
    real = ad.decode

    def spy(x, *a, **k):
        sizes.append(len(x))
        return real(x, *a, **k)

    monkeypatch.setattr(ad, "decode", spy)
    f, d = _fact(), _dim()
    out = (LazyFrame(f).filter(col("qty") < lit(5.0)).join(LazyFrame(d), on="ok")
           .groupby("mode").agg(col("v").sum().alias("s")).sort("mode").compute(device="cpu"))
    assert pa.types.is_string(out.schema.field("mode").type)
    assert sizes and max(sizes) < f.num_rows // 5, sizes          # ~8% of the fact rows survive
    assert je  # (imported for the patch target's module)


@pytest.fixture(scope="module")
def tpch(tmp_path_factory):
    gen = importlib.import_module("gen_tpch_parquet")
    d = tmp_path_factory.mktemp("tpch") / "tpch_sf0.01_exact"
    gen.generate(0.01, str(d))
    return str(d)


def test_tpch_join_queries_filter_their_leaves_natively(tpch, filters, monkeypatch):
    q_mx = importlib.import_module("tpch_user.q_mx")
    q_duckdb = importlib.import_module("tpch_user.q_duckdb")
    native = []
    for q in range(1, 23):
        before = len(filters)
        got = q_mx.run_mem(q, q_mx.load_mem(q, tpch), "cpu")
        if len(filters) > before:
            native.append(q)
        cmp = compare_tables(got.rename_columns(spec.OUTPUT[q]), q_duckdb.run(q, tpch), order_by=spec.ORDER_BY.get(q))
        assert cmp.ok, (q, cmp.message)
    for q in (3, 5, 10, 12, 14, 19):
        assert q in native, (q, native)


@pytest.mark.parametrize("native_on", ["1", "0"])
def test_case_when_column_above_a_join_is_decoded_on_the_joined_rows_only(monkeypatch, native_on):
    """Q12's o_orderpriority: a dictionary column used in CASE WHEN above the join (a use the
    Python path can't take on dictionaries) is no longer decoded in full up front: the join decodes
    it after its final gather, so only joined rows are decoded."""
    from mxframe.lazy_expr import when
    import mxframe.arrow_dict as ad
    monkeypatch.setenv("MXFRAME_NATIVE", native_on)
    sizes = []
    real = ad.decode

    def spy(x, *a, **k):
        sizes.append(len(x))
        return real(x, *a, **k)

    monkeypatch.setattr(ad, "decode", spy)
    f, d = _fact(), _dim(n=50_000)            # fact keys are < 5000: ~4.9k joined rows of 50k
    plan = lambda: (LazyFrame(f).filter(col("qty") < lit(3.0)).join(LazyFrame(d), on="ok")
                    .groupby("mode").agg(when(col("pri").isin(["1-URGENT", "2-HIGH"]), lit(1.0), lit(0.0))
                                         .sum().alias("high")).sort("mode"))
    got = plan().compute(device="cpu")
    # never the 50,000-row dimension in full (with the executor on, the native gather decodes the
    # joined rows itself and Arrow decodes nothing)
    assert max(sizes, default=0) < 10_000, sizes
    assert sizes or native_on == "1"
    monkeypatch.setattr(ad, "decode", real)
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    monkeypatch.setattr("mxframe.io_parquet._join_leaf_filters", lambda plan: {})   # reference: decode up front
    _same(got, plan().compute(device="cpu"))



def test_int64_columns_past_2_53_are_declined_not_rounded(filters, monkeypatch):
    """The interpreter computes in float64: an int64 value column beyond +-2^53 must not be compared
    or summed there (2^53+1 > 2^53 would be false). The executor declines; the answer is exact."""
    n = 50_000
    x = np.full(n, (1 << 53) + 1, dtype=np.int64)
    x[::7] = 5
    f = pa.table({"ok": A((np.arange(n) % 5000).astype(np.int64)), "x": A(x), "v": A(np.ones(n))})
    d = _dim()
    plan = lambda: (LazyFrame(f).filter(col("x") > lit(1 << 53)).join(LazyFrame(d), on="ok")
                    .groupby("pri").agg(col("v").sum().alias("s")).sort("pri"))
    got = plan().compute(device="cpu")
    assert "code -16" in (nx.last_filter()[1] or ""), nx.last_filter()
    assert sum(got.column("s").to_pylist()) == n - len(x[::7])
    single = LazyFrame(f).filter(col("x") > lit(1 << 53)).groupby().agg(col("v").sum().alias("s"))
    got1 = single.compute(device="cpu")
    assert got1.column("s").to_pylist() == [float(n - len(x[::7]))]
    assert nx.last_path() != "native" and "code -16" in (nx.last_decline() or ""), nx.last_decline()


@pytest.mark.parametrize("shape", ["limit", "sort-limit"])
def test_leaves_other_than_filter_scan_are_decoded_before_the_join(monkeypatch, shape):
    """Only Filter*(Scan) join inputs defer their dictionary decode (leaf_table decodes those);
    any other leaf shape keeps resolve_plan's up-front decode, so nothing above the join ever
    sees a dictionary column it didn't before."""
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    f, d = _fact(n=30_000), _dim()
    left = LazyFrame(f).limit(20_000) if shape == "limit" else LazyFrame(f).sort("v").limit(20_000)
    out = left.join(LazyFrame(d), on="ok").select("ok", "mode", "pri").compute(device="cpu")
    assert pa.types.is_string(out.schema.field("mode").type)
    assert pa.types.is_string(out.schema.field("pri").type)


def test_left_join_with_deferred_columns_matches(monkeypatch, filters):
    f, d = _fact(), _dim()
    plan = lambda: (LazyFrame(f).filter(col("mode").isin(["MAIL", "AIR"]))
                    .join(LazyFrame(d).filter(col("day") < lit(9000)), on="ok", how="left")
                    .groupby("mode").agg(col("v").sum().alias("s"), col("day").count().alias("n")).sort("mode"))
    got, old = _both(plan, monkeypatch)
    _same(got, old)


def test_parquet_source_with_join(tmp_path, filters, monkeypatch):
    """scan_parquet: dictionary columns read as dictionaries, filtered natively at the leaf."""
    import pyarrow.parquet as pq
    import mxframe as mx
    f, d = _fact(), _dim()
    pq.write_table(f.cast(pa.schema([fld if not pa.types.is_dictionary(fld.type) else pa.field(fld.name, pa.string())
                                     for fld in f.schema])), tmp_path / "f.parquet", row_group_size=30_000)
    pq.write_table(d.cast(pa.schema([fld if not pa.types.is_dictionary(fld.type) else pa.field(fld.name, pa.string())
                                     for fld in d.schema])), tmp_path / "d.parquet")
    # (an `isin` on a string column is pushed into the Parquet reader; a float comparison stays a Filter)
    plan = lambda: (mx.scan_parquet(str(tmp_path / "f.parquet"))
                    .filter(col("mode").isin(["MAIL", "SHIP"]) & (col("qty") < lit(30.0)))
                    .join(mx.scan_parquet(str(tmp_path / "d.parquet")), on="ok")
                    .groupby("mode", "pri").agg(col("v").sum().alias("s")).sort("mode", "pri"))
    got, old = _both(plan, monkeypatch)
    assert len(filters) >= 1, nx.last_filter()
    _same(got, old)


@pytest.fixture
def gathers(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    calls = []
    real = nx._call_gather

    def counting(*a):
        calls.append(1)
        return real(*a)

    monkeypatch.setattr(nx, "_call_gather", counting)
    return calls


def test_join_output_is_gathered_natively(gathers, monkeypatch):
    """The join's final gather: fixed-width columns (int, float, date, decimal), utf8 with empty
    strings, and deferred dictionary columns (decoded in the gather), from several chunks. (Joins
    the star path takes, M2d, gather nothing: pinned off here.)"""
    monkeypatch.setenv("MXFRAME_STAR", "0")
    f, d = _fact(n=90_000), _dim()
    note = f.column("note").to_pylist()
    note[::5] = [""] * len(note[::5])
    f = f.set_column(f.schema.get_field_index("note"), "note", pa.array(note, type=pa.string()))
    plan = lambda: (LazyFrame(f).filter(col("qty") < lit(25.0)).join(LazyFrame(d), on="ok")
                    .groupby("mode", "note", "pri").agg(col("v").sum().alias("s"), col("price").count().alias("n"))
                    .sort("mode", "note", "pri"))
    got, old = _both(plan, monkeypatch)
    assert len(gathers) >= 1
    _same(got, old)
    rows = lambda: (LazyFrame(f).filter(col("qty") < lit(3.0)).join(LazyFrame(d), on="ok")
                    .select("ok", "note", "shipd", "price", "qty", "pri").sort("ok", "note", "qty", "shipd"))
    got, old = _both(rows, monkeypatch)
    _same(got, old)


def test_native_gather_matches_arrow_take_on_odd_chunks():
    import mxframe.custom_ops as co
    r = np.random.default_rng(3)
    words = np.array(["", "a", "bb", "ccc", "Zz"], dtype=object)
    parts = []
    for m in (30_000, 1, 17, 50_000):
        parts.append(pa.table({"s": pa.array(words[r.integers(0, 5, m)].tolist(), type=pa.string()),
                               "i": A(r.integers(-9, 9, m).astype(np.int64)),
                               "f": A(r.random(m).astype(np.float32)),
                               "d": pa.array(r.integers(0, 1000, m).astype(np.int64)).cast(pa.decimal128(38, 2)),
                               "t": pa.array(r.integers(0, 10**9, m), type=pa.timestamp("us", tz="UTC"))}))
    t = pa.concat_tables(parts)
    comp = co.CustomOpsCompiler(device="cpu")
    for dtype in (np.int32, np.int64):
        idx = r.integers(0, t.num_rows, 70_000).astype(dtype)
        got = nx.try_gather(comp, t, idx)
        assert got is not None
        assert got.equals(t.take(pa.array(idx.astype(np.int64))).combine_chunks())
    assert nx.try_gather(comp, t, np.array([t.num_rows], dtype=np.int64)) is None      # out of range: declined



def test_fused_pipeline_declines_int64_past_2_53_too(monkeypatch):
    """With the executor off, the older fused pipeline (also float64) must decline as well."""
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    import mxframe.custom_ops as co
    n = 20_000
    x = np.full(n, (1 << 53) + 1, dtype=np.int64)
    x[::4] = 3
    t = pa.table({"g": A((np.arange(n) % 3).astype(np.int32)), "x": A(x), "v": A(np.ones(n))})
    got = (LazyFrame(t).filter(col("x") > lit(1 << 53)).groupby("g").agg(col("v").sum().alias("s"))
           .sort("g").compute(device="cpu"))
    assert sum(got.column("s").to_pylist()) == n - len(x[::4])


def test_native_gather_on_sliced_misaligned_chunks_and_per_chunk_dictionaries():
    """Columns chunked at different boundaries (cut into common segments, offsets > 0) and a
    dictionary column with a different dictionary per chunk, decoded by the gather."""
    import mxframe.custom_ops as co
    r = np.random.default_rng(12)
    n = 40_000
    vals = A(r.integers(-1000, 1000, n).astype(np.int64))
    strs = pa.array([f"s{i % 13}" * (i % 3) for i in range(n)], type=pa.string())
    a = pa.chunked_array([vals.slice(0, 7_000), vals.slice(7_000, 20_000), vals.slice(27_000)])
    b = pa.chunked_array([strs.slice(0, 15_000), strs.slice(15_000)])
    dparts = []
    for lo, hi, words in ((0, 10_000, ["x", "y"]), (10_000, 31_000, ["y", "zz", ""]), (31_000, n, ["q"])):
        dparts.append(pa.DictionaryArray.from_arrays(A(r.integers(0, len(words), hi - lo).astype(np.int32)), A(words)))
    c = pa.chunked_array(dparts)
    t = pa.table({"a": a, "b": b, "c": c}).slice(1_234, 35_000)
    comp = co.CustomOpsCompiler(device="cpu")
    idx = r.integers(0, t.num_rows, 25_000).astype(np.int64)
    assert nx.try_gather(comp, t, idx) is None                       # a dictionary column: only when decoding
    got = nx.try_gather(comp, t, idx, decode_dicts=True)
    want = t.take(pa.array(idx))
    assert got.column("a").to_pylist() == want.column("a").to_pylist()
    assert got.column("b").to_pylist() == want.column("b").to_pylist()
    assert got.column("c").to_pylist() == want.column("c").to_pylist()
    assert pa.types.is_string(got.schema.field("c").type)
