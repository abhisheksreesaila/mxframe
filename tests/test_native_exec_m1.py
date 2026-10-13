"""F-031 (M1): the Mojo executor runs whole single-table aggregate pipelines.

M0 ran filter + group-by + aggregate + sort-by-the-keys. M1 adds, still in the one `mx_execute` call:
grouped results without an ORDER BY (emitted in key order), ORDER BY any mix of keys and aggregates
with LIMIT (top-k), simple HAVING (aggregate <op> constant, AND-ed), and key ranges past the dense
2^20 groups (composite codes compacted through a presence table). Join results are materialized
before the aggregate, so this is the final stage of most TPC-H queries."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.lazy_expr import when
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


def _table(n=20_000, seed=1, chunks=1, key_range=None):
    r = np.random.default_rng(seed)
    t = pa.table({
        "f": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, n).astype(np.int32)), A(["A", "N", "R"])),
        "g": A(r.integers(-5, 7, n).astype(np.int32)),
        "k": A(r.integers(0, key_range or 1000, n).astype(np.int64)),
        "v": A(r.random(n) * 100),
        "w": A(r.random(n).astype(np.float32)),
        "d": A(r.integers(0, 1000, n).astype(np.int32)),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        t = pa.concat_tables([t.slice(a, b - a) for a, b in zip(cuts[:-1], cuts[1:])])
    return t


def _same(got, old, ordered=True):
    assert got.schema == old.schema, (got.schema, old.schema)
    assert got.num_rows == old.num_rows, (got.num_rows, old.num_rows)
    if not ordered:
        keys = [c for c in got.column_names if not pa.types.is_floating(got.schema.field(c).type)]
        got = got.take(pa.compute.sort_indices(got, [(k, "ascending") for k in keys]))
        old = old.take(pa.compute.sort_indices(old, [(k, "ascending") for k in keys]))
    for c in got.column_names:
        g, o = got.column(c).to_pylist(), old.column(c).to_pylist()
        if pa.types.is_floating(got.schema.field(c).type):
            np.testing.assert_allclose(g, o, rtol=1e-12, err_msg=c)
        else:
            assert g == o, c


def _both(plan, t, monkeypatch):
    got = plan(t).compute(device="cpu")
    path, decline = nx.last_path(), nx.last_decline()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = plan(t).compute(device="cpu")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    return got, old, path, decline


ORDERED = {
    "sort by an aggregate desc, limit": lambda t: LazyFrame(t).filter(col("d") < lit(900)).groupby("k")
        .agg(col("v").sum().alias("s"), col("w").count().alias("n")).sort("s", descending=[True]).limit(10),
    "sort by aggregate then key": lambda t: LazyFrame(t).groupby("g", "f")
        .agg(col("v").count().alias("n"), col("v").mean().alias("m")).sort("n", "g", "f", descending=[True, False, True]),
    "sort by key then aggregate, big limit": lambda t: LazyFrame(t).groupby("g", "k")
        .agg(col("v").sum().alias("s")).sort("g", "s", descending=[False, True]).limit(5000),
    "sort by key, limit": lambda t: LazyFrame(t).groupby("k").agg(col("v").max().alias("x")).sort("k").limit(7),
    "having, sorted": lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("n"))
        .filter((col("s") > lit(1000.0)) & (col("n") >= lit(19))).sort("s", descending=[True]),
    "having on a dict key group": lambda t: LazyFrame(t).groupby("f").agg(col("v").sum().alias("s"))
        .filter(col("s") > lit(1.0)).sort("f"),
}


@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("name", list(ORDERED))
def test_sort_limit_having_match_the_old_path(native, monkeypatch, name, chunks):
    got, old, path, decline = _both(ORDERED[name], _table(chunks=chunks), monkeypatch)
    assert path == "native", decline
    _same(got, old)


UNORDERED = {
    "grouped, no sort": lambda t: LazyFrame(t).filter(col("d") < lit(500)).groupby("g", "f")
        .agg(col("v").sum().alias("s"), col("v").count().alias("n")),
    "having, no sort": lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).filter(col("s") < lit(900.0)),
}


@pytest.mark.parametrize("name", list(UNORDERED))
def test_grouped_without_sort_has_the_same_rows(native, monkeypatch, name):
    got, old, path, decline = _both(UNORDERED[name], _table(chunks=2), monkeypatch)
    assert path == "native", decline
    _same(got, old, ordered=False)


def test_limit_without_sort_returns_the_first_groups_in_key_order(native, monkeypatch):
    t = _table()
    got = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).limit(5).compute(device="cpu")
    assert nx.last_path() == "native", nx.last_decline()
    assert got.column("k").to_pylist() == sorted(set(t.column("k").to_pylist()))[:5]
    full = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).compute(device="cpu")
    sums = dict(zip(full.column("k").to_pylist(), full.column("s").to_pylist()))
    for k, s in zip(got.column("k").to_pylist(), got.column("s").to_pylist()):
        assert s == pytest.approx(sums[k], rel=1e-12)


@pytest.mark.parametrize("chunks", [1, 4])
def test_key_range_past_the_dense_limit(native, monkeypatch, chunks):
    """2M distinct key values possible (> 2^20 dense groups): compacted through a presence table."""
    t = _table(n=300_000, chunks=chunks, key_range=2_000_000)
    plan = lambda t: (LazyFrame(t).filter(col("d") < lit(800)).groupby("k")
                      .agg(col("v").sum().alias("s"), col("w").count().alias("n")).sort("k"))
    got, old, path, decline = _both(plan, t, monkeypatch)
    assert path == "native", decline
    _same(got, old)
    # two keys: 150k x 12 = 1.8M combinations, also past 2^20 (the cap is 8x the rows)
    t2 = _table(n=300_000, chunks=chunks, key_range=150_000)
    plan2 = lambda t: LazyFrame(t).groupby("k", "g").agg(col("v").sum().alias("s")).sort("s", descending=[True]).limit(20)
    got, old, path, decline = _both(plan2, t2, monkeypatch)
    assert path == "native", decline
    _same(got, old)


@pytest.mark.parametrize("chunks", [1, 3])
def test_sparse_key_tuples_go_through_the_hash_table(native, monkeypatch, chunks):
    """Keys spread over 2^40 values (and composites of them) are too sparse to address: hashed."""
    t = _table(n=50_000, chunks=chunks, key_range=1 << 40)
    t = t.set_column(t.schema.get_field_index("k"), "k",
                     A((np.asarray(t.column("k").to_numpy()) % 7000) * 157_000_003))   # 7000 sparse values
    for plan, ordered in (
            (lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).sort("k"), True),
            (lambda t: LazyFrame(t).filter(col("d") < lit(700)).groupby("k", "g", "f")
                .agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("s", descending=[True]).limit(50), True),
            (lambda t: LazyFrame(t).groupby("g", "k").agg(col("w").max().alias("x")), False)):
        got, old, path, decline = _both(plan, t, monkeypatch)
        assert path == "native", decline
        _same(got, old, ordered=ordered)


def _strings(n=30_000, seed=5, chunks=1, distinct=900):
    r = np.random.default_rng(seed)
    names = np.array([f"Customer#{i:09d}" for i in range(distinct)], dtype=object)
    t = pa.table({
        "name": pa.array(names[r.integers(0, distinct, n)].tolist(), type=pa.string()),
        "nation": pa.array(np.array(["FRANCE", "GERMANY", "PERU", "ALGERIA", ""], dtype=object)[r.integers(0, 5, n)].tolist(),
                           type=pa.string()),
        "k": A(r.integers(0, 50, n).astype(np.int64)),
        "v": A(r.random(n) * 10),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        t = pa.concat_tables([t.slice(a, b - a) for a, b in zip(cuts[:-1], cuts[1:])])
    return t


@pytest.mark.parametrize("chunks", [1, 4])
def test_plain_string_keys(native, monkeypatch, chunks):
    t = _strings(chunks=chunks)
    plans = {
        "by nation, sorted": (lambda t: LazyFrame(t).groupby("nation").agg(col("v").sum().alias("s")).sort("nation"), True),
        "name x nation, top 20": (lambda t: LazyFrame(t).groupby("name", "nation")
            .agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("s", "name", descending=[True, False]).limit(20), True),
        "name desc, k": (lambda t: LazyFrame(t).groupby("name", "k").agg(col("v").max().alias("x"))
            .sort("name", "k", descending=[True, False]), True),
        "unsorted": (lambda t: LazyFrame(t).groupby("nation", "k").agg(col("v").mean().alias("m")), False),
    }
    for name, (plan, ordered) in plans.items():
        got, old, path, decline = _both(plan, t, monkeypatch)
        assert path == "native", (name, decline)
        _same(got, old, ordered=ordered)


DECLINED = {
    "having on an expression": lambda t: LazyFrame(t).groupby("f").agg(col("v").sum().alias("s"))
        .filter(col("s") * lit(2.0) > lit(1.0)).sort("f"),
    "having with or": lambda t: LazyFrame(t).groupby("f").agg(col("v").sum().alias("s"), col("v").count().alias("n"))
        .filter((col("s") > lit(1.0)) | (col("n") > lit(3))).sort("f"),
}


@pytest.mark.parametrize("name", list(DECLINED))
def test_outside_m1_falls_back_with_the_same_answer(native, monkeypatch, name):
    t = _table()
    got = DECLINED[name](t).compute(device="cpu")
    assert nx.last_path() != "native"
    assert len(native) == 0
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    assert got.equals(DECLINED[name](t).compute(device="cpu"))


@pytest.fixture(scope="module")
def tpch(tmp_path_factory):
    gen = importlib.import_module("gen_tpch_parquet")
    d = tmp_path_factory.mktemp("tpch") / "tpch_sf0.01_exact"
    gen.generate(0.01, str(d))
    return str(d)


def test_tpch_all_22_correct_with_the_executor_and_most_final_stages_native(tpch, native, monkeypatch):
    q_mx = importlib.import_module("tpch_user.q_mx")
    q_duckdb = importlib.import_module("tpch_user.q_duckdb")
    native_queries = []
    for q in range(1, 23):
        before = len(native)
        got = q_mx.run_mem(q, q_mx.load_mem(q, tpch), "cpu")
        if len(native) > before:
            native_queries.append(q)
        want = q_duckdb.run(q, tpch)
        cmp = compare_tables(got.rename_columns(spec.OUTPUT[q]), want, order_by=spec.ORDER_BY.get(q))
        assert cmp.ok, (q, cmp.message)
    assert len(native_queries) >= 12, native_queries


def _dict_strings(n=40_000, seed=8):
    """Every chunk has its own dictionary, in its own order, some with extra entries (Parquet row groups)."""
    r = np.random.default_rng(seed)
    parts = []
    for words in (["MAIL", "SHIP", "AIR", "RAIL"], ["SHIP", "TRUCK", "MAIL"], ["AIR", "REG AIR", "FOB", "MAIL", "SHIP"]):
        m = n // 3
        parts.append(pa.table({
            "mode": pa.DictionaryArray.from_arrays(A(r.integers(0, len(words), m).astype(np.int32)), A(words)),
            "pri": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, m).astype(np.int32)),
                                                  A(["1-URGENT", "2-HIGH", "3-MEDIUM"][::(1 if len(words) % 2 else -1)])),
            "k": A(r.integers(0, 40, m).astype(np.int32)),
            "v": A(r.random(m) * 10),
        }))
    return pa.concat_tables(parts)


STRING_PREDICATES = {
    "isin": lambda t: LazyFrame(t).filter(col("mode").isin(["MAIL", "SHIP"])).groupby("k")
        .agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("k"),
    "eq and startswith": lambda t: LazyFrame(t).filter((col("mode") == lit("AIR")) | col("pri").startswith("1-"))
        .groupby("mode").agg(col("v").sum().alias("s")).sort("mode"),
    "not isin, contains": lambda t: LazyFrame(t).filter(~col("mode").isin(["TRUCK"]) & col("mode").contains("AI"))
        .groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n")),
}


@pytest.mark.parametrize("name", list(STRING_PREDICATES))
def test_string_predicates_on_per_chunk_dictionaries(native, monkeypatch, name):
    t = _dict_strings()
    assert t.column("mode").num_chunks == 3
    got, old, path, decline = _both(STRING_PREDICATES[name], t, monkeypatch)
    assert path == "native", decline
    _same(got, old)


def test_case_when_on_a_dictionary_column_is_decoded_first_and_still_correct(native, monkeypatch):
    """io_parquet.resolve_plan decodes a dictionary column used inside CASE WHEN before execution (the
    Python path's evaluators need plain strings), so the executor sees utf8 there and declines the
    predicate; the answer is the old path's. (M2/M4: hand dictionaries to the executor as they are.)"""
    t = _dict_strings()
    plan = lambda t: LazyFrame(t).groupby("k").agg(
        when(col("pri").isin(["1-URGENT", "2-HIGH"]), lit(1.0), lit(0.0)).sum().alias("high")).sort("k")
    got, old, path, decline = _both(plan, t, monkeypatch)
    _same(got, old)


# ── order: key order on every path (pandas' groupby default); ORDER BY ties broken by key order ──
# The old path emits groups in first appearance (SQL leaves the order of an unsorted GROUP BY, and
# of ties, undefined); the executor's order is the same whichever grouping path a query takes.

def _keys_table(keys, seed=3):
    r = np.random.default_rng(seed)
    n = len(keys)
    return pa.table({"k": A(np.asarray(keys, dtype=np.int64)), "v": A(np.round(r.random(n) * 4)),
                     "w": A(np.ones(n))})


@pytest.mark.parametrize("spread", ["dense", "compact", "hash"])
def test_unsorted_groups_come_out_in_key_order_on_every_path(native, monkeypatch, spread):
    r = np.random.default_rng(11)
    base = {"dense": 1, "compact": 3, "hash": 1 << 41}[spread]
    keys = r.permutation(np.repeat(np.arange(4000) * base, 5))
    if spread == "compact":
        keys = np.concatenate([keys, [0, 4_000_000]])          # range ~4M > 2^20, under 8x rows
    t = _keys_table(keys)
    got = LazyFrame(t).groupby("k").agg(col("w").sum().alias("n")).compute(device="cpu")
    assert nx.last_path() == "native", nx.last_decline()
    ks = got.column("k").to_pylist()
    assert ks == sorted(ks)
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = LazyFrame(t).groupby("k").agg(col("w").sum().alias("n")).compute(device="cpu")
    _same(got, old, ordered=False)


@pytest.mark.parametrize("spread", ["dense", "hash"])
def test_order_by_ties_and_limit_follow_key_order(native, monkeypatch, spread):
    base = 1 if spread == "dense" else (1 << 41)
    keys = np.random.default_rng(5).permutation(np.repeat(np.arange(300) * base, 4))
    t = _keys_table(keys)
    got = (LazyFrame(t).groupby("k").agg(col("w").sum().alias("n")).sort("n", descending=[True])
           .limit(7).compute(device="cpu"))
    assert nx.last_path() == "native", nx.last_decline()
    assert got.column("k").to_pylist() == [i * base for i in range(7)]       # every n ties at 4
    first = LazyFrame(t).groupby("k").agg(col("w").sum().alias("n")).limit(3).compute(device="cpu")
    assert first.column("k").to_pylist() == [0, base, 2 * base]


def test_sort_by_int64_keys_past_2_53_is_exact(native, monkeypatch):
    big = 1 << 60
    keys = [big + 1, 5, big, big + 3, big + 2, 0, big + 1, -big]
    t = pa.table({"k": A(np.array(keys, dtype=np.int64)), "v": A(np.arange(len(keys), dtype=np.float64))})
    for desc in (False, True):
        got = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).sort("k", descending=[desc]).compute(device="cpu")
        assert nx.last_path() == "native", nx.last_decline()
        want = sorted(set(keys), reverse=desc)
        assert got.column("k").to_pylist() == want
    got = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).compute(device="cpu")
    assert got.column("k").to_pylist() == sorted(set(keys))


@pytest.mark.parametrize("spread", ["dense", "hash"])
def test_string_keys_come_out_in_byte_order(native, monkeypatch, spread):
    """String keys are numbered by first appearance in the dictionaries; output is still in byte
    order, on the dense path and on the hash path (a wide second key)."""
    r = np.random.default_rng(4)
    words = ["z", "m", "a", "b", "Zz", "ab", ""]
    n = 7000
    s_codes = r.integers(0, len(words), n).astype(np.int32)
    second = r.integers(0, 3, n).astype(np.int64) * (1 << 41 if spread == "hash" else 1)
    t = pa.table({"s": pa.DictionaryArray.from_arrays(A(s_codes), A(words)), "i": A(second),
                  "v": A(np.ones(n))})
    got = LazyFrame(t).groupby("s", "i").agg(col("v").sum().alias("n")).compute(device="cpu")
    assert nx.last_path() == "native", nx.last_decline()
    pairs = list(zip(got.column("s").to_pylist(), got.column("i").to_pylist()))
    assert pairs == sorted(pairs)
    plain = pa.table({"s": pa.array([words[c] for c in s_codes], type=pa.string()), "v": A(np.ones(n))})
    got2 = LazyFrame(plain).groupby("s").agg(col("v").sum().alias("n")).limit(3).compute(device="cpu")
    assert got2.column("s").to_pylist() == sorted(words)[:3]
