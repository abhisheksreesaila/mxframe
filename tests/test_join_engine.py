"""F-018 join engine: projection pushdown, late materialization, join order, left-join counts.

Every result is judged against DuckDB on the same Arrow tables, never against our own old path.
"""
import numpy as np
import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from omxcore import LazyFrame, col, lit
from omxcore import join_engine
from omxcore.custom_ops import CustomOpsCompiler
from omxcore.lazy_frame import Join, Scan, Filter, Aggregate



@pytest.fixture
def star_scan(monkeypatch):
    """These tests check the star scan's own mechanics (what it writes, which columns it keeps coded);
    an aggregate over the join would otherwise skip it (F-046a slice 3, tests/test_fused_join_agg.py)."""
    monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "0")

def _sql(sql, **tables):
    con = duckdb.connect()
    for name, t in tables.items():
        con.register(name, t)
    return con.execute(sql).to_arrow_table()


def _rows(t, ndigits=6):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    rows = [tuple(round(v, ndigits) if isinstance(v, float) else v for v in r) for r in zip(*cols)]
    return sorted(rows, key=lambda r: tuple((v is None, v) for v in r))


def _i32(a):
    return pa.array(np.asarray(a, dtype=np.int32))


# ── left-join counts (F-016 item owned here) ────────────────────────────────────────────────

@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_count_of_right_column_after_left_join_counts_unmatched_as_zero(device):
    rng = np.random.default_rng(1)
    n, m = 60_000, 40_000
    a = pa.table({"k": _i32(np.arange(n)), "v": pa.array(rng.random(n))})
    b = pa.table({"k": _i32(rng.integers(0, n // 2, m)), "w": pa.array(rng.random(m)),
                  "x": pa.array(np.arange(m, dtype=np.int64))})
    j = LazyFrame(a).join(LazyFrame(b), on="k", how="left")
    got = j.groupby("k").agg(col("x").count().alias("n"), col("w").sum().alias("s")).sort("k").compute(device=device)
    ref = _sql("select a.k, count(b.x) n, sum(b.w) s from a left join b on a.k = b.k group by a.k order by a.k", a=a, b=b)
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()
    assert min(got.column("n").to_pylist()) == 0
    got = j.groupby().agg(col("x").count().alias("n"), col("v").count().alias("nv")).compute(device=device)
    ref = _sql("select count(b.x) n, count(a.v) nv from a left join b on a.k = b.k", a=a, b=b)
    assert got.to_pydict() == ref.to_pydict()


def test_left_join_then_inner_join_keeps_the_unmatched_rows_null():
    a = pa.table({"k": _i32([1, 2, 3, 4]), "v": pa.array([1.0, 2.0, 3.0, 4.0])})
    b = pa.table({"k": _i32([1, 1, 2]), "j": _i32([10, 20, 10]), "w": pa.array([5.0, 6.0, 7.0])})
    c = pa.table({"j": _i32([10, 20]), "z": pa.array([100.0, 200.0])})
    got = (LazyFrame(a).join(LazyFrame(b), on="k", how="left").join(LazyFrame(c), on="j")
           .compute(device="cpu"))
    ref = _sql("select a.k, a.v, b.j, b.w, c.z from a left join b on a.k=b.k join c on b.j=c.j", a=a, b=b, c=c)
    assert _rows(got) == _rows(ref)


# ── correctness of reordered / lean join chains ─────────────────────────────────────────────

def _star(seed, n_fact=30_000):
    rng = np.random.default_rng(seed)
    fact = pa.table({"f_id": _i32(np.arange(n_fact)),
                     "f_a": _i32(rng.integers(0, 500, n_fact)), "f_b": _i32(rng.integers(0, 40, n_fact)),
                     "f_c": _i32(rng.integers(0, 2000, n_fact)),
                     "f_x": pa.array(rng.random(n_fact)), "f_note": pa.array([f"n{i % 7}" for i in range(n_fact)])})
    da = pa.table({"a_id": _i32(np.arange(500)), "a_grp": _i32(np.arange(500) % 9),
                   "name": pa.array([f"a{i}" for i in range(500)])})
    db = pa.table({"b_id": _i32(np.arange(40)), "b_flag": _i32(np.arange(40) % 4), "name": pa.array([f"b{i}" for i in range(40)])})
    dc = pa.table({"c_id": _i32(np.arange(2000)), "c_rank": _i32(rng.integers(0, 100, 2000)),
                   "c_a": _i32(rng.integers(0, 500, 2000))})
    return fact, da, db, dc


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_star_join_matches_duckdb_whatever_the_order(device):
    fact, da, db, dc = _star(3)
    lf = (LazyFrame(fact)
          .join(LazyFrame(da), left_on="f_a", right_on="a_id")
          .join(LazyFrame(db).filter(col("b_flag") == lit(1)), left_on="f_b", right_on="b_id")
          .join(LazyFrame(dc).filter(col("c_rank") < lit(20)), left_on="f_c", right_on="c_id"))
    got = (lf.groupby("a_grp", "b_flag").agg((col("f_x") * col("c_rank")).sum().alias("s"),
                                            col("f_id").count().alias("n"))
           .sort("a_grp", "b_flag").compute(device=device))
    ref = _sql("""select a_grp, b_flag, sum(f_x * c_rank) s, count(f_id) n
                  from fact join da on f_a = a_id join db on f_b = b_id join dc on f_c = c_id
                  where b_flag = 1 and c_rank < 20 group by a_grp, b_flag order by a_grp, b_flag""",
               fact=fact, da=da, db=db, dc=dc)
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), ref.column("s").to_pylist(), rtol=1e-9)


def test_join_output_columns_and_names_do_not_depend_on_the_order():
    fact, da, db, dc = _star(4, n_fact=5_000)
    got = (LazyFrame(fact).join(LazyFrame(da), left_on="f_a", right_on="a_id")
           .join(LazyFrame(db), left_on="f_b", right_on="b_id").compute(device="cpu"))
    # `name` exists on both dimension tables: the later one becomes `name_right`
    assert got.column_names == ["f_id", "f_a", "f_b", "f_c", "f_x", "f_note", "a_grp", "name",
                                "b_flag", "name_right"]
    ref = _sql("""select f_id, f_a, f_b, f_c, f_x, f_note, a_grp, da.name as name, b_flag, db.name as name_right
                  from fact join da on f_a = a_id join db on f_b = b_id""", fact=fact, da=da, db=db)
    assert _rows(got) == _rows(ref)


def test_composite_key_and_snowflake_chain():
    rng = np.random.default_rng(5)
    n = 20_000
    li = pa.table({"p": _i32(rng.integers(0, 300, n)), "s": _i32(rng.integers(0, 50, n)),
                   "q": pa.array(rng.random(n)), "o": _i32(rng.integers(0, 1000, n))})
    ps = pa.table({"p": _i32(np.repeat(np.arange(300), 4)), "s": _i32((np.arange(1200) * 7) % 50),
                   "cost": pa.array(rng.random(1200))})
    ps = pa.Table.from_pydict({k: v for k, v in zip(ps.column_names, ps.columns)})
    orders = pa.table({"o": _i32(np.arange(1000)), "c": _i32(np.arange(1000) % 30)})
    cust = pa.table({"c": _i32(np.arange(30)), "nat": _i32(np.arange(30) % 5)})
    nation = pa.table({"nat": _i32(np.arange(5)), "nname": pa.array(list("ABCDE"))})
    lf = (LazyFrame(li)
          .join(LazyFrame(ps), left_on=["p", "s"], right_on=["p", "s"])
          .join(LazyFrame(orders), on="o")
          .join(LazyFrame(cust), on="c")
          .join(LazyFrame(nation), on="nat"))
    got = lf.groupby("nname").agg((col("q") - col("cost")).sum().alias("t"), col("o").count().alias("n")).sort("nname").compute(device="cpu")
    ref = _sql("""select nname, sum(q - cost) t, count(o) n from li join ps using (p, s) join orders using (o)
                  join cust using (c) join nation using (nat) group by nname order by nname""",
               li=li, ps=ps, orders=orders, cust=cust, nation=nation)
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()
    np.testing.assert_allclose(got.column("t").to_pylist(), ref.column("t").to_pylist(), rtol=1e-9)


def test_string_keys_and_nulls_never_match():
    a = pa.table({"k": pa.array(["x", "y", None, "z", "x"]), "v": pa.array([1.0, 2.0, 3.0, 4.0, 5.0])})
    b = pa.table({"k": pa.array(["x", None, "z", "w"]), "w": pa.array([10.0, 20.0, 30.0, 40.0])})
    c = pa.table({"w": pa.array([10.0, 30.0]), "tag": pa.array(["t1", "t3"])})
    got = LazyFrame(a).join(LazyFrame(b), on="k").join(LazyFrame(c), on="w").compute(device="cpu")
    ref = _sql("select a.k, a.v, b.w, c.tag from a join b on a.k=b.k join c on b.w=c.w", a=a, b=b, c=c)
    assert _rows(got) == _rows(ref)


def test_empty_join_result_and_empty_leaf():
    a = pa.table({"k": _i32([1, 2, 3]), "v": pa.array([1.0, 2.0, 3.0])})
    b = pa.table({"k": _i32([7, 8]), "w": pa.array([1.0, 2.0])})
    c = pa.table({"k": _i32([1, 2]), "z": pa.array([1.0, 2.0])})
    got = LazyFrame(a).join(LazyFrame(b), on="k").join(LazyFrame(c), on="k").compute(device="cpu")
    assert got.num_rows == 0
    got = LazyFrame(a).join(LazyFrame(c).filter(col("z") > lit(100.0)), on="k").compute(device="cpu")
    assert got.num_rows == 0
    got = (LazyFrame(a).join(LazyFrame(c).filter(col("z") > lit(100.0)), on="k")
           .groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    assert got.column("n").to_pylist() == [0]


# ── lean materialization ────────────────────────────────────────────────────────────────────

def test_joins_carry_only_the_columns_read_above_them(star_scan, monkeypatch):
    fact, da, db, dc = _star(6, n_fact=4_000)
    seen = []
    real = join_engine._execute_inner

    def spy(comp, node, needed, *rest):
        out = real(comp, node, needed, *rest)
        seen.append(list(out.column_names))
        return out

    monkeypatch.setattr(join_engine, "_execute_inner", spy)
    lf = (LazyFrame(fact).join(LazyFrame(da), left_on="f_a", right_on="a_id")
          .join(LazyFrame(db), left_on="f_b", right_on="b_id"))
    got = lf.groupby("a_grp").agg(col("f_x").sum().alias("s")).sort("a_grp").compute(device="cpu")
    assert seen and set(seen[-1]) == {"a_grp", "f_x"}, seen
    ref = _sql("select a_grp, sum(f_x) s from fact join da on f_a=a_id join db on f_b=b_id group by a_grp order by a_grp",
               fact=fact, da=da, db=db)
    np.testing.assert_allclose(got.column("s").to_pylist(), ref.column("s").to_pylist(), rtol=1e-9)


def test_needed_columns_flow_through_filter_project_and_sort():
    fact, da, db, dc = _star(7, n_fact=2_000)
    plan = (LazyFrame(fact).join(LazyFrame(da), left_on="f_a", right_on="a_id")
            .filter(col("a_grp") == lit(3)).select("f_id", "f_x").sort("f_id").plan)
    comp = CustomOpsCompiler(device="cpu")
    out = comp._materialize_joins(plan)
    node = out
    while not isinstance(node, Scan):
        node = node.input
    assert set(node.table.column_names) == {"f_id", "f_x", "a_grp"}


# ── join order ──────────────────────────────────────────────────────────────────────────────

def test_plan_order_joins_the_filtered_small_side_before_the_big_one():
    # Q2 shape: partsupp (8M) - supplier (100K) - nation (5 of 25) and partsupp - part (8K of 2M)
    rows = [8_000_000, 100_000, 5, 8_000]
    base = [8_000_000, 100_000, 25, 2_000_000]
    edges = [(0, 1, ["ps_suppkey"], ["s_suppkey"]), (1, 2, ["s_nationkey"], ["n_nationkey"]),
             (0, 3, ["ps_partkey"], ["p_partkey"])]
    start, steps = join_engine.plan_order(rows, base, edges)
    order = [start] + [s[0] for s in steps]
    assert order.index(3) < order.index(1), order      # the tiny filtered part table before the supplier join
    assert sorted(order) == [0, 1, 2, 3]


def test_plan_order_is_a_connected_order_over_every_leaf():
    rows = base = [1000, 100, 10, 50]
    edges = [(0, 1, ["a"], ["a"]), (1, 2, ["b"], ["b"]), (0, 3, ["c"], ["c"])]
    start, steps = join_engine.plan_order(rows, base, edges)
    seen = {start}
    for new, ei, r_leaf in steps:
        a, b = edges[ei][0], edges[ei][1]
        assert r_leaf in seen and new not in seen and {a, b} == {r_leaf, new}
        seen.add(new)
    assert seen == {0, 1, 2, 3}


def test_string_columns_of_a_small_side_are_consumed_by_filters_and_group_keys():
    rng = np.random.default_rng(9)
    n = 200_000
    fact = pa.table({"f_k": _i32(rng.integers(0, 20, n)), "f_x": pa.array(rng.random(n))})
    dim = pa.table({"d_k": _i32(np.arange(20)), "d_name": pa.array([f"nation_{i % 7}" for i in range(20)])})
    lf = LazyFrame(fact).join(LazyFrame(dim), left_on="f_k", right_on="d_k")
    got = (lf.filter(col("d_name") != lit("nation_3")).groupby("d_name")
           .agg(col("f_x").sum().alias("s"), col("f_x").count().alias("n")).sort("d_name").compute(device="cpu"))
    ref = _sql("""select d_name, sum(f_x) s, count(f_x) n from fact join dim on f_k = d_k
                  where d_name <> 'nation_3' group by d_name order by d_name""", fact=fact, dim=dim)
    assert got.column("d_name").to_pylist() == ref.column("d_name").to_pylist()
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), ref.column("s").to_pylist(), rtol=1e-9)
    assert got.schema.field("d_name").type == pa.string()
    got = lf.filter(col("d_name").startswith("nation_1")).groupby().agg(col("f_x").count().alias("n")).compute(device="cpu")
    ref = _sql("select count(f_x) n from fact join dim on f_k = d_k where d_name like 'nation_1%'", fact=fact, dim=dim)
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()
    # the same string column used as the join key of a second join must still match
    got = lf.join(LazyFrame(dim.rename_columns(["d_k2", "d_name2"])), left_on="d_name", right_on="d_name2") \
        .groupby().agg(col("f_x").count().alias("n")).compute(device="cpu")
    ref = _sql("select count(f_x) n from fact join dim on f_k = d_k join dim d2 on dim.d_name = d2.d_name", fact=fact, dim=dim)
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()


def test_threaded_numpy_passes_give_the_same_join(monkeypatch):
    monkeypatch.setattr(join_engine, "PAR_MIN_ROWS", 1000)
    monkeypatch.setattr(join_engine, "_PAR_PIECE", 700)
    fact, da, db, dc = _star(11, n_fact=20_000)
    lf = (LazyFrame(fact).join(LazyFrame(da), left_on="f_a", right_on="a_id")
          .join(LazyFrame(dc).filter(col("c_rank") < lit(50)), left_on="f_c", right_on="c_id"))
    got = lf.groupby("a_grp").agg(col("f_x").sum().alias("s"), col("f_id").count().alias("n")).sort("a_grp").compute(device="cpu")
    ref = _sql("""select a_grp, sum(f_x) s, count(f_id) n from fact join da on f_a = a_id join dc on f_c = c_id
                  where c_rank < 50 group by a_grp order by a_grp""", fact=fact, da=da, dc=dc)
    assert got.column("n").to_pylist() == ref.column("n").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), ref.column("s").to_pylist(), rtol=1e-9)
    assert join_engine._minmax(np.arange(5000)) == (0, 4999)
    a = np.arange(5000, dtype=np.int64)
    idx = np.random.default_rng(0).integers(0, 5000, 4000).astype(np.int32)
    assert (join_engine._take(a, idx) == a[idx]).all()
    assert (join_engine._as_i32(a) == a).all() and join_engine._as_i32(a).dtype == np.int32
