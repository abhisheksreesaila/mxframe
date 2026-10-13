"""F-032 (M2d): join partners fused into the leaf filter (`mx_filter` plan v2).

A scanned table is filtered and each of its rows looks its partner row up in a map Mojo builds over
the partner's join key (a direct array for compact ranges, a partitioned hash table otherwise), in
the same pass. Only joined rows are written: the scanned table's columns, then each partner's.
Partner keys must be unique; a duplicate declines and the caller joins as before."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


@pytest.fixture
def comp(monkeypatch):
    import mxframe.custom_ops as co
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    return co.CustomOpsCompiler(device="cpu")


def _fact(n, seed=1, key_hi=5000, chunks=(1,)):
    r = np.random.default_rng(seed)
    parts, start = [], 0
    sizes = [n * c // sum(chunks) for c in chunks]
    words = np.array(["", "a", "bb", "ccc", "Zz"], dtype=object)
    for i, m in enumerate(sizes):
        parts.append(pa.table({
            "k": A(r.integers(-10, key_hi, m).astype(np.int64)),
            "k2": A(r.integers(0, 40, m).astype(np.int32)),
            "q": A(r.integers(0, 100, m).astype(np.int32)),
            "s": pa.array(words[r.integers(0, 5, m)].tolist(), type=pa.string()),
            "m": pa.DictionaryArray.from_arrays(A(r.integers(0, 2, m).astype(np.int32)),
                                                A([f"x{i}", f"y{i}"])),
            "v": A(r.random(m)),
        }))
        start += m
    return pa.concat_tables(parts)


def _dim(keys, seed=2):
    r = np.random.default_rng(seed)
    n = len(keys)
    return pa.table({
        "dk": A(np.asarray(keys, dtype=np.int64)),
        "name": pa.array([f"name-{x}" for x in keys], type=pa.string()),
        "pri": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, n).astype(np.int32)), A(["U", "H", "L"])),
        "day": pa.array(r.integers(8000, 10000, n).astype(np.int32), type=pa.date32()),
        "w": A(r.random(n)),
    })


def _reference(fact, preds_mask, lookups):
    """Rows of `fact` passing the mask whose keys all find a partner, as python lists."""
    fk = {n: fact.column(n).to_pylist() for n in fact.column_names}
    maps = []
    for probe, build, bkeys, outs in lookups:
        bk = list(zip(*[build.column(c).to_pylist() for c in bkeys]))
        bo = {c: build.column(c).to_pylist() for c in outs}
        maps.append((probe, {key: i for i, key in enumerate(bk)}, bo, outs))
    rows = []
    for i in range(fact.num_rows):
        if not preds_mask[i]:
            continue
        hit = []
        for probe, m, bo, outs in maps:
            j = m.get(tuple(fk[p][i] for p in probe))
            if j is None:
                break
            hit.append([bo[c][j] for c in outs])
        else:
            rows.append((i, hit))
    return rows


def _check(got, fact, names, rows, lookups):
    assert got is not None, nx.last_filter()
    assert got.num_rows == len(rows)
    for n in names:
        want = [fact.column(n)[i].as_py() for i, _ in rows]
        assert got.column(n).to_pylist() == want, n
    for j, (_, _, _, outs) in enumerate(lookups):
        for t, c in enumerate(outs):
            want = [hit[j][t] for _, hit in rows]
            assert got.column(f"L{j}.{c}").to_pylist() == want, c


def _lk(probe, build, bkeys, outs, j):
    return nx.Lookup(probe, build, bkeys, outs, [f"L{j}.{c}" for c in outs])


@pytest.mark.parametrize("dim_keys", [
    "dense",          # compact range: direct array
    "sparse",         # wide range: partitioned hash table
])
def test_one_partner_with_a_predicate(comp, dim_keys):
    fact = _fact(200_000, chunks=(3, 1, 2))
    keys = np.arange(0, 5000, 3) if dim_keys == "dense" else np.arange(0, 5000, 3) * 1_000_003
    if dim_keys == "sparse":
        fact = fact.set_column(0, "k", A(fact.column("k").to_numpy() * 1_000_003))
    dim = _dim(keys)
    lookups = [(["k"], dim, ["dk"], ["name", "pri", "day", "w"])]
    got = nx.try_filter(comp, fact, [col("q") < lit(60)], {"k", "s", "m", "v"},
                        [_lk(*l, j) for j, l in enumerate(lookups)])
    mask = np.asarray(fact.column("q").to_pylist()) < 60
    _check(got, fact, ["k", "s", "m", "v"], _reference(fact, mask, lookups), lookups)
    assert nx.last_filter()[0] == "native_lookup"


def test_two_partners_composite_key_and_no_predicate(comp):
    fact = _fact(150_000, key_hi=3000, chunks=(1, 1))
    dim = _dim(np.arange(0, 3000, 2))
    r = np.random.default_rng(5)
    pairs = sorted({(int(a), int(b)) for a, b in zip(r.integers(0, 3000, 40_000), r.integers(0, 40, 40_000))})
    ps = pa.table({"pk": A(np.array([a for a, _ in pairs], dtype=np.int64)),
                   "sk": A(np.array([b for _, b in pairs], dtype=np.int32)),
                   "cost": A(r.random(len(pairs)))})
    lookups = [(["k"], dim, ["dk"], ["name"]), (["k", "k2"], ps, ["pk", "sk"], ["cost"])]
    got = nx.try_filter(comp, fact, [], {"q"}, [_lk(*l, j) for j, l in enumerate(lookups)])
    _check(got, fact, ["q"], _reference(fact, np.ones(fact.num_rows, bool), lookups), lookups)


def test_duplicate_partner_keys_decline(comp):
    fact = _fact(10_000)
    dim = _dim([1, 2, 3, 2])
    got = nx.try_filter(comp, fact, [], {"q"}, [_lk(["k"], dim, ["dk"], ["w"], 0)])
    assert got is None
    assert "-18" in nx.last_filter()[1]
    # a wide range (hash table) too
    dim = _dim([1, 2 << 40, 3, 2 << 40])
    assert nx.try_filter(comp, fact, [], {"q"}, [_lk(["k"], dim, ["dk"], ["w"], 0)]) is None


def test_partner_split_into_chunks_and_dictionaries(comp):
    """A partner arriving in several chunks with different dictionaries is combined first."""
    fact = _fact(50_000, key_hi=1000)
    a, b = _dim(np.arange(0, 500)), _dim(np.arange(500, 1000), seed=9)
    b = b.set_column(2, "pri", pa.DictionaryArray.from_arrays(A(np.zeros(500, np.int32)), A(["ZZ"])))
    dim = pa.concat_tables([a, b])
    lookups = [(["k"], dim, ["dk"], ["pri", "name"])]
    got = nx.try_filter(comp, fact, [col("v") > lit(0.5)], {"v"}, [_lk(*l, 0) for l in lookups])
    mask = np.asarray(fact.column("v").to_pylist()) > 0.5
    _check(got, fact, ["v"], _reference(fact, mask, lookups), lookups)


def test_unsupported_keys_decline(comp):
    fact = _fact(5_000)
    dim = _dim(np.arange(10))
    dim_f = dim.set_column(0, "dk", A(np.arange(10, dtype=np.float64)))
    assert nx.try_filter(comp, fact, [], {"q"}, [_lk(["k"], dim_f, ["dk"], ["w"], 0)]) is None
    dim_n = dim.set_column(0, "dk", pa.array([None] + list(range(1, 10)), type=pa.int64()))
    assert nx.try_filter(comp, fact, [], {"q"}, [_lk(["k"], dim_n, ["dk"], ["w"], 0)]) is None


# ── plans: a join tree runs as one scan of its largest Filter*(Scan) leaf ────────────────────

from mxframe import LazyFrame  # noqa: E402
import mxframe.join_engine as je  # noqa: E402


def _tables(n=120_000):
    r = np.random.default_rng(11)
    nat = pa.table({"n_key": A(np.arange(25, dtype=np.int32)),
                    "n_name": pa.array([f"N{i:02d}" for i in range(25)], type=pa.string()),
                    "n_reg": A((np.arange(25) % 5).astype(np.int32))})
    cust = pa.table({"c_key": A(np.arange(1, 3001, dtype=np.int64)),
                     "c_nat": A(r.integers(0, 25, 3000).astype(np.int32)),
                     "c_seg": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, 3000).astype(np.int32)),
                                                             A(["AUTO", "BUILD", "MACH"]))})
    orders = pa.table({"o_key": A(np.arange(1, 40_001, dtype=np.int64) * 4),
                       "o_cust": A(r.integers(1, 3001, 40_000).astype(np.int64)),
                       "o_day": pa.array(r.integers(9000, 9400, 40_000).astype(np.int32), type=pa.date32())})
    supp = pa.table({"s_key": A(np.arange(1, 201, dtype=np.int64)),
                     "s_nat": A(r.integers(0, 25, 200).astype(np.int32)),
                     "tag": A(np.arange(200, dtype=np.int32))})
    ps = pa.table({"ps_part": A(np.repeat(np.arange(1, 1001, dtype=np.int64), 4)),
                   "ps_supp": A(np.tile(np.arange(1, 5, dtype=np.int64), 1000) * 50),
                   "ps_cost": A(r.random(4000) * 100)})
    parts = []
    for c in range(3):
        m = n // 3
        parts.append(pa.table({
            "l_order": A(r.integers(1, 40_001, m).astype(np.int64) * 4 + (r.random(m) < 0.1)),   # 10% miss
            "l_part": A(r.integers(1, 1001, m).astype(np.int64)),
            "l_supp": A(r.integers(1, 5, m).astype(np.int64) * 50),
            "l_qty": A(r.integers(1, 50, m).astype(np.int32)),
            "l_price": A(r.random(m) * 1000),
            "l_mode": pa.DictionaryArray.from_arrays(A(r.integers(0, 2, m).astype(np.int32)),
                                                     A([f"AIR{c}", "MAIL"])),
            "tag": A(np.zeros(m, np.int32)),
        }))
    return dict(nat=nat, cust=cust, orders=orders, supp=supp, ps=ps, li=pa.concat_tables(parts))


STAR_PLANS = {
    "snowflake: lineitem -> orders -> customer -> nation, group by a far string": lambda t: (
        LazyFrame(t["li"]).filter(col("l_qty") < lit(30))
        .join(LazyFrame(t["orders"]).filter(col("o_cust") < lit(2500)), left_on="l_order", right_on="o_key")
        .join(LazyFrame(t["cust"]), left_on="o_cust", right_on="c_key")
        .join(LazyFrame(t["nat"]), left_on="c_nat", right_on="n_key")
        .groupby("n_name", "c_seg").agg(col("l_price").sum().alias("rev"), col("l_qty").count().alias("n"))
        .sort("n_name", "c_seg")),
    "two partners, composite key, condition across partners": lambda t: (
        LazyFrame(t["li"])
        .join(LazyFrame(t["ps"]), left_on=["l_part", "l_supp"], right_on=["ps_part", "ps_supp"])
        .join(LazyFrame(t["orders"]), left_on="l_order", right_on="o_key")
        .join(LazyFrame(t["supp"]), left_on="l_supp", right_on="s_key")
        .join(LazyFrame(t["cust"]), left_on="o_cust", right_on="c_key")
        .filter(col("c_nat") == col("s_nat"))
        .groupby("s_nat").agg((col("l_price") - col("ps_cost")).sum().alias("p")).sort("s_nat")),
    "condition on one renamed partner column (a one-leaf subtree)": lambda t: (
        LazyFrame(t["li"]).join(LazyFrame(t["supp"]), left_on="l_supp", right_on="s_key")
        .filter(col("tag_right") < lit(100))
        .join(LazyFrame(t["orders"]), left_on="l_order", right_on="o_key")
        .groupby("l_qty").agg(col("l_price").count().alias("n")).sort("l_qty")),
    "condition inside a two-leaf subtree": lambda t: (
        LazyFrame(t["li"]).join(LazyFrame(t["orders"]), left_on="l_order", right_on="o_key")
        .join(LazyFrame(t["cust"]), left_on="o_cust", right_on="c_key")
        .filter(col("c_nat") * lit(100) < col("o_cust"))
        .groupby("c_nat").agg(col("l_price").sum().alias("s")).sort("c_nat")),
    "no output column from the scanned table": lambda t: (
        LazyFrame(t["li"]).filter(col("l_qty") > lit(40))
        .join(LazyFrame(t["orders"]), left_on="l_order", right_on="o_key")
        .groupby("o_cust").agg(col("o_day").count().alias("n")).sort("o_cust")),
    "partner filtered on a dictionary string, narrowed to its key (.filter().select(key))": lambda t: (
        LazyFrame(t["li"]).join(LazyFrame(t["orders"]), left_on="l_order", right_on="o_key")
        .join(LazyFrame(t["cust"]).filter(col("c_seg") == lit("BUILD")).select("c_key"),
              left_on="o_cust", right_on="c_key")
        .groupby("l_mode").agg(col("l_price").sum().alias("s")).sort("l_mode")),
    "rows out, strings from both sides": lambda t: (
        LazyFrame(t["li"]).filter(col("l_qty") == lit(7))
        .join(LazyFrame(t["orders"]), left_on="l_order", right_on="o_key")
        .join(LazyFrame(t["cust"]), left_on="o_cust", right_on="c_key")
        .select("l_order", "l_mode", "o_day", "c_seg", "l_price").sort("l_order", "l_price")),
}


def _run_both(plan, t, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    got = plan(t).compute(device="cpu")
    path = nx.last_filter()[0]
    monkeypatch.setenv("MXFRAME_STAR", "0")
    old = plan(t).compute(device="cpu")
    monkeypatch.delenv("MXFRAME_STAR")
    return got, old, path


def _same(got, old):
    assert got.column_names == old.column_names
    assert got.num_rows == old.num_rows
    for c in got.column_names:
        g, o = got.column(c).to_pylist(), old.column(c).to_pylist()
        if pa.types.is_floating(got.schema.field(c).type):
            np.testing.assert_allclose(g, o, rtol=1e-9, err_msg=c)
        else:
            assert [str(x) for x in g] == [str(x) for x in o], c


@pytest.mark.parametrize("name", list(STAR_PLANS))
def test_star_plans_match_the_leaf_by_leaf_join(monkeypatch, name):
    import mxframe.custom_ops as co
    t = _tables()
    star = []
    real = je._execute_star
    monkeypatch.setattr(je, "_execute_star", lambda *a: star.append(r := real(*a)) or r)
    co._JOIN_RESULT_CACHE.clear()
    got, old, path = _run_both(STAR_PLANS[name], t, monkeypatch)
    assert star and star[0] is not None, nx.last_filter()
    assert got.num_rows > 0
    _same(got, old)


def test_star_with_duplicate_partner_keys_falls_back(monkeypatch):
    """A partner whose join key repeats (many-to-many) declines natively; the join runs leaf by leaf."""
    import mxframe.custom_ops as co
    t = _tables()
    dup = pa.concat_tables([t["supp"], t["supp"]])
    plan = lambda t: (LazyFrame(t["li"]).filter(col("l_qty") < lit(10))
                      .join(LazyFrame(dup), left_on="l_supp", right_on="s_key")
                      .groupby("s_nat").agg(col("l_price").sum().alias("s")).sort("s_nat"))
    co._JOIN_RESULT_CACHE.clear()
    got, old, _ = _run_both(plan, t, monkeypatch)
    _same(got, old)
    li = t["li"]
    q = np.asarray(li.column("l_qty").to_pylist())
    assert got.column("s").to_pylist()  # non-empty
    total = sum(got.column("s").to_pylist())
    np.testing.assert_allclose(total, 2 * np.asarray(li.column("l_price").to_pylist())[q < 10].sum(), rtol=1e-9)


def test_composite_key_in_the_hash_table(comp):
    """A two-column key whose combined range is too wide for a direct array."""
    fact = _fact(60_000, key_hi=3000)
    fact = fact.set_column(1, "k2", A(fact.column("k2").to_numpy().astype(np.int64) * 1_000_000_007))
    r = np.random.default_rng(6)
    pairs = sorted({(int(a), int(b) * 1_000_000_007) for a, b in zip(r.integers(0, 3000, 30_000),
                                                                      r.integers(0, 40, 30_000))})
    ps = pa.table({"pk": A(np.array([a for a, _ in pairs], dtype=np.int64)),
                   "sk": A(np.array([b for _, b in pairs], dtype=np.int64)),
                   "cost": A(r.random(len(pairs)))})
    lookups = [(["k", "k2"], ps, ["pk", "sk"], ["cost"])]
    got = nx.try_filter(comp, fact, [col("q") < lit(70)], {"q"}, [_lk(*l, 0) for l in lookups])
    mask = np.asarray(fact.column("q").to_pylist()) < 70
    _check(got, fact, ["q"], _reference(fact, mask, lookups), lookups)


@pytest.mark.parametrize("probe_t,build_t", [(pa.int32(), pa.int64()), (pa.int64(), pa.int32()),
                                             (pa.date32(), pa.date32())])
def test_key_widths_and_dates(comp, probe_t, build_t):
    fact = _fact(40_000, key_hi=2000)
    fact = fact.set_column(0, "k", fact.column("k").cast(pa.int32()).cast(probe_t))
    dim = _dim(np.arange(0, 2000, 7))
    dim = dim.set_column(0, "dk", dim.column("dk").cast(pa.int32()).cast(build_t))
    lookups = [(["k"], dim, ["dk"], ["w", "name"])]
    got = nx.try_filter(comp, fact, [], {"q"}, [_lk(*l, 0) for l in lookups])
    _check(got, fact, ["q"], _reference(fact, np.ones(fact.num_rows, bool), lookups), lookups)


def test_date_key_against_an_integer_key_declines(comp):
    fact = _fact(5_000)
    dim = _dim(np.arange(10))
    dim = dim.set_column(0, "dk", dim.column("dk").cast(pa.int32()).cast(pa.date32()))
    assert nx.try_filter(comp, fact, [], {"q"}, [_lk(["k"], dim, ["dk"], ["w"], 0)]) is None


# ── semi joins: a partner that is the distinct keys of a filtered table ──────────────────────

def test_semi_lookup_allows_repeated_keys(comp):
    fact = _fact(50_000, key_hi=3000)
    for keys in (np.repeat(np.arange(0, 3000, 5), 3), np.repeat(np.arange(0, 3000, 5), 3) * 1_000_003):
        f = fact if keys.max() < 10**6 else fact.set_column(0, "k", A(fact.column("k").to_numpy() * 1_000_003))
        dim = pa.table({"dk": A(keys.astype(np.int64))})
        got = nx.try_filter(comp, f, [], {"q", "k"}, [nx.Lookup(["k"], dim, ["dk"], [], [], semi=True)])
        assert got is not None, nx.last_filter()
        present = set(keys.tolist())
        ks = f.column("k").to_pylist()
        want = [i for i in range(f.num_rows) if ks[i] in present]
        assert got.column("k").to_pylist() == [ks[i] for i in want]
        assert got.column("q").to_pylist() == [f.column("q")[i].as_py() for i in want]


SEMI_PLANS = {
    "filtered distinct keys, aggregate on top": lambda t: (
        LazyFrame(t["orders"]).filter(col("o_cust") < lit(2000))
        .join(LazyFrame(t["li"]).filter(col("l_qty") < lit(5)).select("l_order").distinct(),
              left_on="o_key", right_on="l_order")
        .groupby("o_day").agg(col("o_cust").count().alias("n")).sort("o_day")),
    "unfiltered distinct keys, rows out": lambda t: (
        LazyFrame(t["orders"]).filter(col("o_cust") < lit(300))
        .join(LazyFrame(t["li"]).select("l_order").distinct(), left_on="o_key", right_on="l_order")
        .select("o_key", "o_day").sort("o_key")),
    "two-column distinct keys": lambda t: (
        LazyFrame(t["li"]).filter(col("l_qty") > lit(45))
        .join(LazyFrame(t["ps"]).filter(col("ps_cost") < lit(50.0)).select("ps_part", "ps_supp").distinct(),
              left_on=["l_part", "l_supp"], right_on=["ps_part", "ps_supp"])
        .groupby("l_supp").agg(col("l_price").sum().alias("s")).sort("l_supp")),
}


@pytest.mark.parametrize("name", list(SEMI_PLANS))
def test_semi_plans_match_and_skip_the_distinct(monkeypatch, name):
    import mxframe.custom_ops as co
    t = _tables()
    semis, star = [], []
    real = je._semi_partner
    monkeypatch.setattr(je, "_semi_partner", lambda *a: semis.append(r := real(*a)) or r)
    real_star = je._execute_star
    monkeypatch.setattr(je, "_execute_star", lambda *a: star.append(r := real_star(*a)) or r)
    co._JOIN_RESULT_CACHE.clear()
    got, old, _ = _run_both(SEMI_PLANS[name], t, monkeypatch)
    assert semis and semis[0] is not None
    assert star and star[0] is not None, nx.last_filter()      # the semi lookup ran, no fallback
    assert got.num_rows > 0
    _same(got, old)


def test_distinct_of_more_than_the_keys_is_not_a_semi_join(monkeypatch):
    import mxframe.custom_ops as co
    t = _tables()
    plan = lambda t: (LazyFrame(t["orders"])
                      .join(LazyFrame(t["li"]).filter(col("l_qty") < lit(3)).select("l_order", "l_qty").distinct(),
                            left_on="o_key", right_on="l_order")
                      .groupby("l_qty").agg(col("o_cust").count().alias("n")).sort("l_qty"))
    semis = []
    real = je._semi_partner
    monkeypatch.setattr(je, "_semi_partner", lambda *a: semis.append(r := real(*a)) or r)
    co._JOIN_RESULT_CACHE.clear()
    got, old, _ = _run_both(plan, t, monkeypatch)
    _same(got, old)
    assert all(r is None for r in semis)          # never taken as a semi partner


def test_a_selected_partner_is_filtered_natively_on_its_dictionary(monkeypatch):
    """Q17's `part.filter(brand, container).select(p_partkey)`: the Project only narrows the columns,
    so the partner is still a Filter*(Scan) leaf: its predicate runs in `mx_filter` on the
    dictionary codes (not decoded first, not evaluated in Python). F-032 (M2g)."""
    import mxframe.custom_ops as co
    t = _tables()
    seen = []
    real = nx.try_filter

    def spy(comp, table, preds, needed, lookups=(), **kw):
        seen.append((table.num_rows, table.schema))
        return real(comp, table, preds, needed, lookups, **kw)

    monkeypatch.setattr(nx, "try_filter", spy)
    co._JOIN_RESULT_CACHE.clear()
    plan = STAR_PLANS["partner filtered on a dictionary string, narrowed to its key (.filter().select(key))"]
    got, old, path = _run_both(plan, t, monkeypatch)
    cust = [sch for n, sch in seen if n == 3000]
    assert cust and pa.types.is_dictionary(cust[0].field("c_seg").type)
    _same(got, old)


def test_a_selected_partner_read_from_parquet_keeps_its_strings_encoded(monkeypatch, tmp_path):
    """The same shape read through scan_parquet (M2g review): io_parquet treats
    `.filter().select(key)` under a join as a Filter*(Scan) leaf, so the partner's string column is
    read as a dictionary, filtered on its codes, and never decoded; a second select on top is not
    seen through (one Project, as in the join engine) and still answers the same."""
    import pyarrow.parquet as pq
    import mxframe as mx
    import mxframe.custom_ops as co
    t = _tables()
    for n in ("li", "orders", "cust"):
        pq.write_table(t[n], tmp_path / f"{n}.parquet")
    seen = []
    real = nx.try_filter

    def spy(comp, table, preds, needed, lookups=(), **kw):
        seen.append((table.num_rows, table.schema))
        return real(comp, table, preds, needed, lookups, **kw)

    monkeypatch.setattr(nx, "try_filter", spy)

    def plan(double):
        cust = mx.scan_parquet(str(tmp_path / "cust.parquet")).filter(col("c_seg") == lit("BUILD")).select("c_key", "c_nat")
        cust = cust.select("c_key") if double else cust
        return (mx.scan_parquet(str(tmp_path / "li.parquet"))
                .join(mx.scan_parquet(str(tmp_path / "orders.parquet")), left_on="l_order", right_on="o_key")
                .join(cust, left_on="o_cust", right_on="c_key")
                .groupby("l_mode").agg(col("l_price").sum().alias("s")).sort("l_mode"))

    want = (LazyFrame(t["li"]).join(LazyFrame(t["orders"]), left_on="l_order", right_on="o_key")
            .join(LazyFrame(t["cust"]).filter(col("c_seg") == lit("BUILD")).select("c_key"), left_on="o_cust", right_on="c_key")
            .groupby("l_mode").agg(col("l_price").sum().alias("s")).sort("l_mode"))
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    ref = want.compute(device="cpu")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    for double in (False, True):
        seen.clear()
        co._JOIN_RESULT_CACHE.clear()
        got = plan(double).compute(device="cpu")
        _same(got, ref)
        if not double:
            cust = [sch for n, sch in seen if n == 3000]
            assert cust and pa.types.is_dictionary(cust[0].field("c_seg").type)


def test_a_filtered_string_also_selected_after_the_join_stays_encoded_in_its_leaf(monkeypatch):
    """Q16's part: `p_brand != ...` in the leaf's filter and `p_brand` selected after the join. A bare
    projected column is decoded after the join's gather, so it does not force decoding the whole
    leaf first (io_parquet `bad_expr`): the leaf is filtered natively on its codes (M2h)."""
    import mxframe.custom_ops as co
    t = _tables()
    seen = []
    real = nx.try_filter

    def spy(comp, table, preds, needed, lookups=(), **kw):
        seen.append((table.num_rows, table.schema))
        return real(comp, table, preds, needed, lookups, **kw)

    monkeypatch.setattr(nx, "try_filter", spy)
    co._JOIN_RESULT_CACHE.clear()

    def plan(t):
        return (LazyFrame(t["orders"]).join(LazyFrame(t["cust"]).filter(col("c_seg") != lit("AUTO")),
                                            left_on="o_cust", right_on="c_key")
                .select("c_seg", "o_cust").distinct())

    got, old, _ = _run_both(plan, t, monkeypatch)
    cust = [sch for n, sch in seen if n == 3000]
    assert cust and pa.types.is_dictionary(cust[0].field("c_seg").type)
    assert got.schema.field("c_seg").type == pa.string()
    _same(got, old)


def test_a_condition_inside_a_nested_star_partner(monkeypatch):
    """Q8's shape with a condition inside the partner subtree (M2h review): orders -> customer runs
    as a star of its own (a nested `_execute_star`) with the condition applied there, and the answer
    equals the leaf-by-leaf join's."""
    import mxframe.custom_ops as co
    t = _tables()
    calls = []
    real = je._execute_star
    monkeypatch.setattr(je, "_execute_star", lambda *a: calls.append(r := real(*a)) or r)
    co._JOIN_RESULT_CACHE.clear()

    def plan(t):
        sub = (LazyFrame(t["orders"]).join(LazyFrame(t["cust"]), left_on="o_cust", right_on="c_key")
               .filter(col("c_nat") * lit(100) < col("o_cust")))
        return (LazyFrame(t["li"]).filter(col("l_qty") < lit(40)).join(sub, left_on="l_order", right_on="o_key")
                .groupby("c_nat").agg(col("l_price").sum().alias("s"), col("l_qty").count().alias("n")).sort("c_nat"))

    got, old, _ = _run_both(plan, t, monkeypatch)
    assert len([c for c in calls if c is not None]) >= 2, calls      # the outer star and the nested one
    assert got.num_rows > 0
    _same(got, old)
