"""F-010: mx.scan_parquet (column pruning, predicate pushdown) and dictionary strings."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
from mxframe import LazyFrame, col, lit
from mxframe.io_parquet import resolve_plan
from mxframe.lazy_frame import Scan, ParquetScan, Filter

DEV = "cpu"


# ── helpers ────────────────────────────────────────────────────────────────

def _rows(t: pa.Table):
    """Order-independent, tolerance-friendly row list."""
    cols = [t.column(n).to_pylist() for n in t.column_names]
    rows = list(zip(*cols))
    return sorted(rows, key=lambda r: tuple("" if v is None else str(v) for v in r))


def assert_same(a: pa.Table, b: pa.Table):
    assert a.column_names == b.column_names
    assert [str(t) for t in a.schema.types] == [str(t) for t in b.schema.types]
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            if isinstance(u, float) and isinstance(v, float):
                assert u == pytest.approx(v, rel=1e-5, abs=1e-6)
            else:
                assert u == v


def _base_table(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    return pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "grp": pa.array(rng.choice(["alpha", "beta", "gamma", "delta"], n)),
        "mode": pa.array(rng.choice(["AIR", "RAIL", "SHIP", "MAIL", "TRUCK"], n)),
        "qty": pa.array(rng.integers(1, 50, n).astype(np.int32)),
        "price": pa.array(rng.uniform(1, 100, n).astype(np.float32)),
        "disc": pa.array(rng.uniform(0, 0.1, n).astype(np.float32)),
        "note": pa.array([f"free text {i}" for i in range(n)]),
    })


@pytest.fixture
def parquet_file(tmp_path):
    p = tmp_path / "t.parquet"
    # small row groups so pushdown / multi-chunk dictionaries are exercised
    pq.write_table(_base_table(), p, row_group_size=500, use_dictionary=True)
    return str(p)


def _leaf(plan):
    while not isinstance(plan, Scan):
        plan = plan.input
    return plan


def _dict_table(chunks):
    """Table whose 'k' column has one chunk per entry, each with its own dictionary."""
    ks, vs = [], []
    for values in chunks:
        ks.append(pa.array(values).dictionary_encode())
        vs.append(pa.array(np.arange(len(values), dtype=np.float32) + 1))
    return pa.table({"k": pa.chunked_array(ks), "v": pa.chunked_array(vs)})


# ── lazy source + column pruning ───────────────────────────────────────────

def test_scan_parquet_is_lazy_and_has_schema(parquet_file):
    lf = mx.scan_parquet(parquet_file)
    assert isinstance(lf.plan, ParquetScan)
    assert lf.schema.names == _base_table().column_names
    assert "ParquetScan" in lf.explain(optimized=False)


def test_only_referenced_columns_are_read(parquet_file):
    lf = (mx.scan_parquet(parquet_file)
          .filter(col("qty") > 10)
          .groupby("grp").agg((col("price") * col("disc")).sum().alias("s")))
    scan = _leaf(resolve_plan(lf.plan))
    assert sorted(scan.table.column_names) == ["disc", "grp", "price", "qty"]


def test_columns_used_in_join_sort_and_window_are_read(parquet_file):
    other = LazyFrame(pa.table({"grp": ["alpha", "beta"], "w": [1.0, 2.0]}))
    lf = (mx.scan_parquet(parquet_file).join(other, on="grp")
          .groupby("mode").agg((col("price") * col("w")).sum().alias("s")))
    plan = resolve_plan(lf.plan)
    left = plan.input.left if hasattr(plan.input, "left") else plan.input.input.left
    assert sorted(left.table.column_names) == ["grp", "mode", "price"]


def test_passthrough_reads_everything(parquet_file):
    # nothing cuts the columns off before the output -> all of them are needed
    for lf in (mx.scan_parquet(parquet_file).filter(col("qty") > 3),
               mx.scan_parquet(parquet_file).sort("id").limit(5),
               mx.scan_parquet(parquet_file).distinct(),
               mx.scan_parquet(parquet_file).with_columns((col("qty") * 2).alias("q2"))):
        scan = _leaf(resolve_plan(lf.plan))
        assert scan.table.column_names == _base_table().column_names


def test_explain_shows_pruned_columns_and_filters(parquet_file):
    lf = (mx.scan_parquet(parquet_file).filter(col("qty") > 10)
          .groupby("grp").agg(col("price").sum().alias("s")))
    text = lf.explain(optimized=False)
    assert "columns=['grp', 'qty', 'price']" in text     # file order, only what is needed
    assert "ParquetScan" in text and "('qty', 'gt', 10)" in text
    assert "'note'" not in text


def test_scan_parquet_results_equal_read_table(parquet_file):
    def q(lf):
        return (lf.filter((col("qty") >= 5) & col("mode").isin(["AIR", "RAIL"]))
                .groupby("grp", "mode")
                .agg(col("price").sum().alias("s"), col("qty").count().alias("n"),
                     col("disc").mean().alias("d"))
                .compute(device=DEV))
    assert_same(q(mx.scan_parquet(parquet_file)), q(LazyFrame(pq.read_table(parquet_file))))


def test_lazyframe_reuse_recomputes_from_file(parquet_file):
    lf = mx.scan_parquet(parquet_file).groupby("grp").agg(col("qty").sum().alias("s"))
    a = lf.compute(device=DEV)
    b = lf.compute(device=DEV)
    assert_same(a, b)


# ── predicate pushdown ─────────────────────────────────────────────────────

def _pushed(lf):
    return resolve_plan(lf.plan, read=False)


def test_simple_comparisons_are_pushed_and_filter_kept(parquet_file):
    lf = (mx.scan_parquet(parquet_file)
          .filter((col("id") >= 3500) & (col("mode") == "AIR"))
          .groupby("grp").agg(col("qty").sum().alias("s")))
    plan = _pushed(lf)
    scan = plan.input.input           # Aggregate -> Filter -> ParquetScan
    assert isinstance(plan.input, Filter)          # the engine still filters
    assert set(scan.filters) == {("id", "ge", 3500), ("mode", "eq", "AIR")}


def test_isin_and_reversed_comparison_pushed(parquet_file):
    lf = mx.scan_parquet(parquet_file).filter(col("mode").isin(["AIR", "SHIP"]) & (lit(100) < col("qty")))
    scan = _pushed(lf).input
    assert ("mode", "isin", ("AIR", "SHIP")) in scan.filters
    assert ("qty", "gt", 100) in scan.filters


def test_float_and_complex_predicates_not_pushed(parquet_file):
    lf = (mx.scan_parquet(parquet_file)
          .filter((col("price") > 50.0) & ((col("qty") > 1) | (col("qty") < 0))
                  & (col("qty") + 1 > 3) & col("note").startswith("free")))
    assert _pushed(lf).input.filters is None


def test_filter_not_directly_above_scan_is_not_pushed(parquet_file):
    lf = (mx.scan_parquet(parquet_file).select("id", "qty").filter(col("qty") > 3))
    plan = _pushed(lf)
    assert plan.input.input.filters is None


def test_pushdown_skips_row_groups(parquet_file):
    lf = mx.scan_parquet(parquet_file).filter(col("id") >= 3500).groupby("grp").agg(col("id").count().alias("n"))
    scan = _leaf(resolve_plan(lf.plan))
    assert scan.table.num_rows == 500                       # 1 of 8 row groups read
    out = lf.compute(device=DEV)
    assert sum(out.column("n").to_pylist()) == 500


def test_pushdown_matches_unpushed_results(parquet_file):
    def q(lf):
        return (lf.filter((col("id") < 1200) & (col("grp") != "beta") & (col("qty") <= 25))
                .groupby("mode").agg(col("price").sum().alias("s"), col("id").count().alias("n"))
                .compute(device=DEV))
    assert_same(q(mx.scan_parquet(parquet_file)), q(LazyFrame(pq.read_table(parquet_file))))


def test_filter_that_eliminates_everything(parquet_file):
    lf = (mx.scan_parquet(parquet_file).filter(col("id") > 10_000_000)
          .groupby("grp").agg(col("qty").sum().alias("s")))
    assert lf.compute(device=DEV).num_rows == 0
    lf2 = mx.scan_parquet(parquet_file).filter(col("mode") == "NO SUCH MODE").select("id", "qty")
    assert lf2.compute(device=DEV).num_rows == 0
    lf3 = mx.scan_parquet(parquet_file).filter(col("mode").isin(["nope"])).sort("id")
    assert lf3.compute(device=DEV).num_rows == 0


# ── dictionary columns ─────────────────────────────────────────────────────

def test_auto_reads_low_cardinality_strings_as_dictionaries(parquet_file):
    lf = mx.scan_parquet(parquet_file).groupby("grp", "mode").agg(col("qty").sum().alias("s"))
    scan = _leaf(resolve_plan(lf.plan))
    assert pa.types.is_dictionary(scan.table.schema.field("grp").type)
    assert pa.types.is_dictionary(scan.table.schema.field("mode").type)


def test_high_cardinality_strings_stay_plain_in_auto(tmp_path):
    n = 20000
    t = pa.table({"u": [f"user-{i:08d}-padding-padding" for i in range(n)],
                  "v": np.arange(n, dtype=np.float32)})
    p = str(tmp_path / "u.parquet")
    pq.write_table(t, p)
    scan = _leaf(resolve_plan(mx.scan_parquet(p).groupby("u").agg(col("v").sum().alias("s")).plan))
    assert scan.table.schema.field("u").type == pa.string()


def test_dictionary_option(parquet_file):
    q = lambda **kw: _leaf(resolve_plan(  # noqa: E731
        mx.scan_parquet(parquet_file, **kw).groupby("grp").agg(col("qty").sum().alias("s")).plan)).table
    assert q(dictionary=False).schema.field("grp").type == pa.string()
    assert pa.types.is_dictionary(q(dictionary=True).schema.field("grp").type)
    assert pa.types.is_dictionary(q(dictionary=["grp"]).schema.field("grp").type)
    assert q(dictionary=["mode"]).schema.field("grp").type == pa.string()


def test_dictionary_only_kept_where_engine_handles_it(parquet_file):
    """Group key / filter-only columns stay dictionaries; anything else is decoded."""
    lf = (mx.scan_parquet(parquet_file)
          .filter(col("mode") == "AIR")
          .groupby("grp").agg(col("qty").sum().alias("s")))
    t = _leaf(resolve_plan(lf.plan)).table
    assert pa.types.is_dictionary(t.schema.field("grp").type)
    assert pa.types.is_dictionary(t.schema.field("mode").type)
    # used as a sort key / projected / aggregated -> plain strings
    for lf in (mx.scan_parquet(parquet_file).select("grp", "qty").sort("grp"),
               mx.scan_parquet(parquet_file).select("grp", "qty"),
               mx.scan_parquet(parquet_file).groupby("mode").agg(col("grp").count().alias("n"))):
        t = _leaf(resolve_plan(lf.plan)).table
        assert t.schema.field("grp").type == pa.string()


@pytest.mark.parametrize("dict_mode", [True, False])
def test_group_by_dictionary_keys_single_and_multi(parquet_file, dict_mode):
    def one(lf):
        return lf.groupby("grp").agg(col("qty").sum().alias("s"), col("price").mean().alias("p")).compute(device=DEV)

    def two(lf):
        return (lf.groupby("grp", "mode")
                .agg(col("qty").sum().alias("s"), col("id").count().alias("n")).compute(device=DEV))
    plain = LazyFrame(pq.read_table(parquet_file))
    scan = mx.scan_parquet(parquet_file, dictionary=dict_mode)
    for f in (one, two):
        got = f(scan)
        assert_same(got, f(plain))
        assert got.schema.field("grp").type == pa.string()      # decoded key values
    assert one(scan).schema.field("grp").type == pa.string()


def test_group_by_dictionary_key_with_multiple_chunks_and_different_dictionaries():
    t = _dict_table([["a", "b", "a"], ["c", "a", "c", "b"], ["b", "d"]])
    assert t.column("k").num_chunks == 3
    assert len({tuple(c.dictionary.to_pylist()) for c in t.column("k").chunks}) == 3
    plain = pa.table({"k": pa.array(t.column("k").to_pylist()), "v": t.column("v")})

    def run(tbl):
        return LazyFrame(tbl).groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device=DEV)
    got = run(t)
    assert_same(got, run(plain))
    assert got.schema.field("k").type == pa.string()


def test_multikey_with_different_dictionaries_per_chunk():
    k1 = pa.chunked_array([pa.array(["x", "y", "x"]).dictionary_encode(),
                           pa.array(["z", "x"]).dictionary_encode()])
    k2 = pa.chunked_array([pa.array(["p", "p", "q"]).dictionary_encode(),
                           pa.array(["q", "r"]).dictionary_encode()])
    v = pa.array(np.arange(5, dtype=np.float32) + 1)
    t = pa.table({"k1": k1, "k2": k2, "v": v})
    plain = pa.table({"k1": pa.array(k1.to_pylist()), "k2": pa.array(k2.to_pylist()), "v": v})

    def run(tbl):
        return LazyFrame(tbl).groupby("k1", "k2").agg(col("v").sum().alias("s")).compute(device=DEV)
    assert_same(run(t), run(plain))


def test_dictionary_with_unused_entries_yields_no_empty_groups():
    d = pa.DictionaryArray.from_arrays(pa.array([0, 0, 2], type=pa.int32()), pa.array(["a", "unused", "c"]))
    t = pa.table({"k": d, "v": pa.array([1.0, 2.0, 3.0], type=pa.float32())})
    out = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).compute(device=DEV)
    assert _rows(out) == [("a", 3.0), ("c", 3.0)]


def _string_dict_pair():
    n = 600
    rng = np.random.default_rng(3)
    vals = rng.choice(["MAIL", "SHIP", "AIR", "RAIL", "FOB", "TRUCK PROMO"], n).tolist()
    ids = np.arange(n, dtype=np.int64)
    v = rng.uniform(0, 10, n).astype(np.float32)
    chunks = [pa.array(vals[i:i + 200]).dictionary_encode() for i in range(0, n, 200)]
    d = pa.table({"id": ids, "s": pa.chunked_array(chunks), "v": v})
    p = pa.table({"id": ids, "s": pa.array(vals), "v": v})
    return d, p


@pytest.mark.parametrize("pred", [
    lambda: col("s") == "AIR",
    lambda: col("s") != "AIR",
    lambda: col("s").isin(["AIR", "SHIP", "nope"]),
    lambda: col("s").startswith("T"),
    lambda: col("s").contains("PROMO"),
    lambda: ~col("s").isin(["AIR"]),
    lambda: col("s").isin(["AIR"]) | col("s").startswith("M"),
    lambda: (col("s") == "AIR") & (col("v") > 5),
], ids=["eq", "ne", "isin", "startswith", "contains", "not-isin", "or", "and-num"])
def test_string_filters_on_dictionary_columns(pred):
    d, p = _string_dict_pair()
    agg = lambda lf: (lf.filter(pred()).groupby("s")  # noqa: E731
                      .agg(col("v").sum().alias("t"), col("id").count().alias("n")).compute(device=DEV))
    sel = lambda lf: lf.filter(pred()).select("id", "s", "v").compute(device=DEV)  # noqa: E731
    assert_same(agg(LazyFrame(d)), agg(LazyFrame(p)))
    assert_same(sel(LazyFrame(d)), sel(LazyFrame(p)))


def test_global_aggregate_with_dictionary_filter():
    d, p = _string_dict_pair()
    from mxframe.lazy_frame import Aggregate
    q = lambda t: LazyFrame(Aggregate(LazyFrame(t).filter(col("s").isin(["AIR", "RAIL"])).plan, [],  # noqa: E731
                                      [col("v").sum().alias("t")])).compute(device=DEV)
    assert_same(q(d), q(p))


def test_case_when_and_projection_over_dictionary_columns():
    from mxframe import when
    d, p = _string_dict_pair()
    q = lambda t: (LazyFrame(t).groupby("id").agg(  # noqa: E731
        when(col("s").isin(["AIR", "RAIL"]), 1, 0).sum().alias("hi")).compute(device=DEV))
    assert_same(q(d), q(p))
    sel = lambda t: LazyFrame(t).select("s", (col("v") * 2).alias("v2")).compute(device=DEV)  # noqa: E731
    got = sel(d)
    assert got.schema.field("s").type == pa.string()
    assert_same(got, sel(p))


def test_sort_distinct_limit_on_dictionary_columns():
    d, p = _string_dict_pair()
    for f in (lambda lf: lf.sort("s", "id"),
              lambda lf: lf.sort("s", "id", descending=True).limit(30),
              lambda lf: lf.select("s").distinct(),
              lambda lf: lf.distinct("s"),
              lambda lf: lf.tail(7)):
        assert_same(f(LazyFrame(d)).compute(device=DEV), f(LazyFrame(p)).compute(device=DEV))


def test_join_on_dictionary_string_keys():
    d, p = _string_dict_pair()
    dim_plain = pa.table({"s": ["AIR", "SHIP", "MAIL"], "w": [1.0, 2.0, 3.0]})
    dim_dict = pa.table({"s": pa.array(["AIR", "SHIP", "MAIL"]).dictionary_encode(), "w": dim_plain["w"]})
    for left, right in ((d, dim_dict), (d, dim_plain), (p, dim_dict)):
        got = (LazyFrame(left).join(LazyFrame(right), on="s")
               .groupby("s").agg((col("v") * col("w")).sum().alias("t")).compute(device=DEV))
        exp = (LazyFrame(p).join(LazyFrame(dim_plain), on="s")
               .groupby("s").agg((col("v") * col("w")).sum().alias("t")).compute(device=DEV))
        assert_same(got, exp)
    plain_join = LazyFrame(d).join(LazyFrame(dim_dict), on="s").select("id", "s", "w").compute(device=DEV)
    assert plain_join.schema.field("s").type == pa.string()


def test_window_and_rank_over_dictionary_partition_column():
    d, p = _string_dict_pair()
    q = lambda t: (LazyFrame(t).with_columns(col("v").sum().over("s").alias("tot")).compute(device=DEV))  # noqa: E731
    assert_same(q(d), q(p))


def test_dictionary_table_via_scan_parquet_with_join_decodes(tmp_path):
    fact = pa.table({"k": ["a", "b", "a", "c"] * 50, "v": np.arange(200, dtype=np.float32)})
    dim = pa.table({"k": ["a", "b", "c"], "w": [1.0, 2.0, 3.0]})
    pf, pd_ = str(tmp_path / "f.parquet"), str(tmp_path / "d.parquet")
    pq.write_table(fact, pf)
    pq.write_table(dim, pd_)
    q = lambda f, d: (f.join(d, on="k").groupby("k").agg((col("v") * col("w")).sum().alias("t")).compute(device=DEV))  # noqa: E731
    assert_same(q(mx.scan_parquet(pf), mx.scan_parquet(pd_)), q(LazyFrame(fact), LazyFrame(dim)))


def test_import_does_not_load_parquet_dataset_modules():
    import pathlib
    import subprocess
    import sys
    root = str(pathlib.Path(mxframe_root()).parent)
    code = "import sys; import mxframe; print('pyarrow.dataset' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": root, "PATH": ""})
    assert out.stdout.split()[-1:] == ["False"], out.stderr[-500:]


def mxframe_root():
    import pathlib
    return str(pathlib.Path(mx.__file__).resolve().parent)


def test_groups_absent_after_pushdown_do_not_appear(tmp_path):
    # 'late' only occurs in later row groups; the pushed filter skips them.
    n = 1000
    t = pa.table({"id": np.arange(n, dtype=np.int64),
                  "g": ["early"] * 500 + ["late"] * 500,
                  "v": np.ones(n, dtype=np.float32)})
    p = str(tmp_path / "g.parquet")
    pq.write_table(t, p, row_group_size=250)
    for flt in (col("id") < 400, col("id") >= 400, (col("id") >= 400) & (col("g") == "late")):
        got = mx.scan_parquet(p).filter(flt).groupby("g").agg(col("v").sum().alias("s")).compute(device=DEV)
        exp = LazyFrame(t).filter(flt).groupby("g").agg(col("v").sum().alias("s")).compute(device=DEV)
        assert_same(got, exp)
    only_early = (mx.scan_parquet(p).filter(col("id") < 400)
                  .groupby("g").agg(col("v").sum().alias("s")).compute(device=DEV))
    assert only_early.column("g").to_pylist() == ["early"]


# ── review fixes ───────────────────────────────────────────────────────────

def _write_dict_file(path, values, dictionary=None, vname="x"):
    d = pa.DictionaryArray.from_arrays(pa.array(values, pa.int32()), pa.array(dictionary))
    cols = {"k": d, vname: pa.array(np.arange(len(values), dtype=np.float32) + 1)}
    pq.write_table(pa.table(cols), path)


def test_stored_dictionary_schema_is_decoded_like_plain(tmp_path):
    f, g = str(tmp_path / "f.parquet"), str(tmp_path / "g.parquet")
    _write_dict_file(f, [0, 1, 0, 2], dictionary=["a", "b", "c"])
    _write_dict_file(g, [1, 0], dictionary=["a", "b"], vname="y")
    assert pa.types.is_dictionary(pq.read_table(f).schema.field("k").type)   # premise
    # projection / passthrough are decoded
    assert mx.scan_parquet(f).compute(device=DEV).schema.field("k").type == pa.string()
    # join across two stored-dictionary files
    got = (mx.scan_parquet(f).join(mx.scan_parquet(g), on="k")
           .groupby("k").agg(col("x").sum().alias("s")).compute(device=DEV))
    fp, gp = pq.read_table(f), pq.read_table(g)
    dec = lambda t: pa.table({"k": t["k"].cast(pa.string()), t.column_names[1]: t.column(1)})  # noqa: E731
    exp = (LazyFrame(dec(fp)).join(LazyFrame(dec(gp)), on="k")
           .groupby("k").agg(col("x").sum().alias("s")).compute(device=DEV))
    assert_same(got, exp)


def test_dictionary_with_unused_entries_in_parquet_file(tmp_path):
    f = str(tmp_path / "u.parquet")
    _write_dict_file(f, [0, 0, 2, 2], dictionary=["a", "UNUSED", "c"])
    out = mx.scan_parquet(f).groupby("k").agg(col("x").count().alias("n")).compute(device=DEV)
    assert sorted(out.column("k").to_pylist()) == ["a", "c"]
    out = (mx.scan_parquet(f).filter(col("x") > 0).groupby("k").agg(col("x").count().alias("n"))
           .compute(device=DEV))
    assert sorted(out.column("k").to_pylist()) == ["a", "c"]


def test_engine_caches_do_not_serve_stale_results_for_reused_table_ids(tmp_path):
    """A freed table's id can be reused; cached group encodings must not leak across."""
    import gc
    p = str(tmp_path / "c.parquet")
    pq.write_table(pa.table({"g": ["x", "y", "z"] * 40, "v": np.ones(120, dtype=np.float32)}), p)
    seen_ids, hits = set(), 0
    for i in range(300):
        first = mx.scan_parquet(p).groupby("g").agg(col("v").sum().alias("s")).compute(device=DEV)
        assert first.num_rows == 3
        gc.collect()
        n = 7
        t = pa.table({"g": [f"k{j % n}" for j in range(70)], "v": np.ones(70, dtype=np.float32)})
        if id(t) in seen_ids:
            hits += 1
        seen_ids.add(id(t))
        out = LazyFrame(t).groupby("g").agg(col("v").sum().alias("s")).compute(device=DEV)
        assert out.num_rows == n
        del t, out
    # (id reuse is likely but not guaranteed; the assertions above are what matter)


def test_cache_entries_are_purged_when_table_dies():
    import gc
    from mxframe import custom_ops as co
    t = pa.table({"g": ["a", "b", "a"], "v": np.ones(3, dtype=np.float32)})
    LazyFrame(t).groupby("g").agg(col("v").sum().alias("s")).compute(device=DEV)
    tid = id(t)
    assert any(co._contains_id(k, tid) for k in co._GROUP_ENCODE_CACHE)
    del t
    gc.collect()
    assert not any(co._contains_id(k, tid) for k in co._GROUP_ENCODE_CACHE)
    assert not any(co._contains_id(k, tid) for k in co._INPUT_PREP_CACHE)


def test_other_tables_caches_survive_scan_parquet_compute(parquet_file):
    from mxframe import custom_ops as co
    t = _base_table()
    LazyFrame(t).groupby("grp").agg(col("qty").sum().alias("s")).compute(device=DEV)
    before = len(co._GROUP_ENCODE_CACHE)
    mx.scan_parquet(parquet_file).groupby("grp").agg(col("qty").sum().alias("s")).compute(device=DEV)
    assert len(co._GROUP_ENCODE_CACHE) >= before - 0 and any(
        co._contains_id(k, id(t)) for k in co._GROUP_ENCODE_CACHE)


def test_isin_with_none_matches_arrow_semantics_on_dictionary():
    vals = ["a", None, "b", "a"]
    plain = pa.table({"s": pa.array(vals), "v": np.arange(4, dtype=np.float32)})
    dic = pa.table({"s": pa.array(vals).dictionary_encode(), "v": plain["v"]})
    from mxframe.arrow_dict import dict_predicate
    import pyarrow.compute as pc
    for values in (["a", None], ["a"], [None], ["a", "b"]):
        e = col("s").isin(values)
        exp = pc.is_in(plain["s"], value_set=pa.array(values, pa.string()))
        got = dict_predicate(e, dic)
        assert got.to_pylist() == exp.to_pylist(), values


# ── F-020: a self-join of one scan_parquet frame ────────────────────────────

def _self_join_file(tmp_path):
    import pyarrow.parquet as pq
    p = str(tmp_path / "self.parquet")
    pq.write_table(pa.table({"k": pa.array(np.arange(12) % 4), "x": pa.array(np.arange(12) * 1.5),
                             "y": pa.array(np.arange(12))}), p)
    return p, pq.read_table(p)


@pytest.mark.parametrize("build", [
    lambda a: a.join(a, on="k").select(col("k"), col("x_right")),
    lambda a: a.join(a, on="k").groupby("k").agg(col("x_right").sum().alias("s")).sort("k"),
    lambda a: a.join(a, on="k").filter(col("x_right") > 3.0).select(col("k"), col("y_right")),
    lambda a: a.join(a, on="k").join(a, on="k").select(col("k"), col("x_right"), col("x_right2")),
], ids=["select", "aggregate", "filter", "three-way"])
def test_self_join_of_one_scan_reads_the_renamed_columns(tmp_path, build):
    p, table = _self_join_file(tmp_path)
    got = build(mx.scan_parquet(p)).compute(device="cpu")
    want = build(mx.LazyFrame(table)).compute(device="cpu")
    assert got.column_names == want.column_names
    key = got.column_names
    assert sorted(zip(*[got.column(c).to_pylist() for c in key])) == sorted(zip(*[want.column(c).to_pylist() for c in key]))
