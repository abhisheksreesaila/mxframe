"""F-032 (M2j): semi and anti joins (`how="semi"` / `how="anti"`, as in Polars) and `str_slice`.

A semi join keeps the left rows that have a partner (each once, however many partners), an anti join
the left rows that have none; both return the left columns only, in the left table's row order.
Null keys never match: a semi join drops those rows, an anti join keeps them (SQL EXISTS / NOT
EXISTS). Checked against DuckDB. TPC-H Q22 ("customers who never ordered") needs both features."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
import omxcore.custom_ops as co
from omxcore import LazyFrame, col, lit
from omxcore.np_arrow import array as A


def _gpu_usable() -> bool:
    try:
        return co._max_accelerator() is not None
    except Exception:
        return False


def _tables(n=50_000, seed=5):
    r = np.random.default_rng(seed)
    left = pa.table({
        "k": pa.array(np.where(r.random(n) < 0.02, None, r.integers(0, 8000, n)).tolist(), pa.int64()),
        "s": pa.array([f"c{v % 700}" for v in r.integers(0, 10_000, n)]),
        "j": A(r.integers(0, 5, n).astype(np.int32)),
        "v": A(r.random(n)),
    })
    m = 30_000
    right = pa.table({                                           # repeated keys: a semi join must not repeat rows
        "rk": A(r.integers(0, 5000, m).astype(np.int64)),
        "rs": pa.array([f"c{v % 900}" for v in r.integers(0, 10_000, m)]),
        "rj": A(r.integers(0, 3, m).astype(np.int32)),
        "w": A(r.random(m)),
    })
    return left, right


def _duck(sql, left, right):
    """``sql`` over l and r; ``ORDER BY rowid`` means the left table's row order (a hidden column)."""
    con = duckdb.connect()
    con.register("l", left.append_column("rowid_", A(np.arange(left.num_rows, dtype=np.int64))))
    con.register("r", right)
    out = con.execute(sql.replace("SELECT * FROM l", "SELECT * EXCLUDE (rowid_) FROM l")
                      .replace("ORDER BY rowid", "ORDER BY rowid_")).fetch_arrow_table()
    return out


def _same_rows_in_order(got: pa.Table, want: pa.Table):
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows
    for c in got.column_names:
        assert got.column(c).to_pylist() == want.column(c).to_pylist(), c


CASES = [
    ("int key", ["k"], ["rk"], "l.k = r.rk"),
    ("string key", ["s"], ["rs"], "l.s = r.rs"),
    ("composite key", ["j", "s"], ["rj", "rs"], "l.j = r.rj AND l.s = r.rs"),
]


@pytest.mark.parametrize("how", ["semi", "anti"])
@pytest.mark.parametrize("name,lk,rk,cond", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("device", ["cpu", "auto", pytest.param(
    "gpu", marks=pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU"))])
def test_semi_and_anti_joins_match_duckdb(how, name, lk, rk, cond, device):
    left, right = _tables()
    got = LazyFrame(left).join(LazyFrame(right), left_on=lk, right_on=rk, how=how).compute(device=device)
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    want = _duck(f"SELECT * FROM l WHERE {exists} (SELECT 1 FROM r WHERE {cond}) ORDER BY rowid", left, right)
    _same_rows_in_order(got, want)


def test_filters_above_push_to_the_left_side_and_the_answer_holds():
    left, right = _tables()
    q = (LazyFrame(left).join(LazyFrame(right).filter(col("w") < lit(0.5)), left_on="k", right_on="rk", how="anti")
         .filter((col("j") > lit(1)) & (col("v") < lit(0.9))).select("k", "v"))
    plan = str(q.explain())
    assert plan.index("Filter") > plan.index("Join")          # the left filter moved below the join
    want = _duck("SELECT k, v FROM l WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.rk = l.k AND r.w < 0.5) "
                 "AND j > 1 AND v < 0.9 ORDER BY rowid", left, right)
    _same_rows_in_order(q.compute(device="cpu"), want)


def test_semi_join_feeding_an_aggregate():
    left, right = _tables()
    got = (LazyFrame(left).join(LazyFrame(right), left_on="k", right_on="rk", how="semi")
           .groupby("j").agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("j").compute(device="cpu"))
    want = _duck("SELECT j, sum(v) AS s, count(v) AS n FROM l WHERE EXISTS (SELECT 1 FROM r WHERE r.rk = l.k) "
                 "GROUP BY j ORDER BY j", left, right)
    assert got.column("j").to_pylist() == want.column("j").to_pylist()
    np.testing.assert_allclose(got.column("s").to_pylist(), want.column("s").to_pylist(), rtol=1e-12)
    assert got.column("n").to_pylist() == want.column("n").to_pylist()


def test_semi_and_anti_split_the_left_table():
    left, right = _tables()
    lf, rf = LazyFrame(left), LazyFrame(right)
    semi = lf.join(rf, left_on="k", right_on="rk", how="semi").compute(device="cpu")
    anti = lf.join(rf, left_on="k", right_on="rk", how="anti").compute(device="cpu")
    assert semi.num_rows + anti.num_rows == left.num_rows
    assert semi.column_names == anti.column_names == left.column_names


def test_empty_right_side():
    left, right = _tables()
    empty = right.slice(0, 0)
    assert LazyFrame(left).join(LazyFrame(empty), left_on="k", right_on="rk", how="semi").compute(device="cpu").num_rows == 0
    anti = LazyFrame(left).join(LazyFrame(empty), left_on="k", right_on="rk", how="anti").compute(device="cpu")
    _same_rows_in_order(anti, left)


def test_unknown_join_type_is_rejected():
    left, right = _tables(100)
    with pytest.raises(ValueError, match="join type"):
        LazyFrame(left).join(LazyFrame(right), left_on="k", right_on="rk", how="outer")


@pytest.mark.parametrize("dictionary", [False, True])
def test_str_slice(dictionary):
    vals = ["13-555-1234", "31-777", "2", "", "héllo-wörld", None]
    arr = pa.array(vals)
    if dictionary:
        arr = arr.dictionary_encode()
    t = pa.table({"p": arr, "i": A(np.arange(len(vals), dtype=np.int64))})
    got = LazyFrame(t).with_columns(col("p").str_slice(0, 2).alias("a"), col("p").str_slice(3).alias("b")).compute(device="cpu")
    assert got.column("a").to_pylist() == [None if v is None else v[0:2] for v in vals]
    assert got.column("b").to_pylist() == [None if v is None else v[3:] for v in vals]
    filt = LazyFrame(t).filter(col("p").str_slice(0, 2).isin(["13", "31"])).compute(device="cpu")
    assert filt.column("i").to_pylist() == [0, 1]


# ── M2j review: joins of joins, the fallback path, SQL, reading only the right keys ─────────────

def _dim(n=8000, seed=9):
    r = np.random.default_rng(seed)
    return pa.table({"dk": A(np.arange(n, dtype=np.int64)), "dg": A(r.integers(0, 4, n).astype(np.int32)),
                     "dv": A(r.random(n))})


def test_joins_of_joins_match_duckdb():
    left, right = _tables()
    dim = _dim()
    con = duckdb.connect()
    con.register("l", left.append_column("rowid_", A(np.arange(left.num_rows, dtype=np.int64))))
    con.register("r", right)
    con.register("d", dim)
    L, R, D = LazyFrame(left), LazyFrame(right), LazyFrame(dim)
    cases = [
        # semi, then inner (the inner join's rows come in the order of its largest leaf: compared sorted)
        (L.join(R, left_on="k", right_on="rk", how="semi").join(D, left_on="k", right_on="dk")
         .select("k", "v", "dg"),
         "SELECT l.k, l.v, d.dg FROM l JOIN d ON l.k = d.dk WHERE EXISTS (SELECT 1 FROM r WHERE r.rk = l.k) ORDER BY l.k, l.v"),
        # inner, then anti with a filter on the inner join's right column
        (L.join(D, left_on="k", right_on="dk").filter(col("dg") > lit(1))
         .join(R, left_on="k", right_on="rk", how="anti").select("k", "dg"),
         "SELECT l.k, d.dg FROM l JOIN d ON l.k = d.dk WHERE d.dg > 1 AND NOT EXISTS (SELECT 1 FROM r WHERE r.rk = l.k) ORDER BY rowid_"),
        # left, then semi
        (L.join(D, left_on="k", right_on="dk", how="left").join(R, left_on="k", right_on="rk", how="semi")
         .select("k", "dv"),
         "SELECT l.k, d.dv FROM l LEFT JOIN d ON l.k = d.dk WHERE EXISTS (SELECT 1 FROM r WHERE r.rk = l.k) ORDER BY rowid_"),
        # a right side that is itself a join
        (L.join(R.join(D, left_on="rk", right_on="dk").filter(col("dg") == lit(2)), left_on="k", right_on="rk", how="anti")
         .select("k", "s"),
         "SELECT k, s FROM l WHERE NOT EXISTS (SELECT 1 FROM r JOIN d ON r.rk = d.dk WHERE d.dg = 2 AND r.rk = l.k) ORDER BY rowid_"),
    ]
    for q, sql in cases:
        got = q.compute(device="cpu")
        want = con.execute(sql).fetch_arrow_table()
        assert got.num_rows == want.num_rows, sql
        if sql.endswith("ORDER BY l.k, l.v"):
            got = got.sort_by([("k", "ascending"), ("v", "ascending")])
        for c in got.column_names:
            g, w = got.column(c).to_pylist(), want.column(c).to_pylist()
            if pa.types.is_floating(got.schema.field(c).type):
                np.testing.assert_allclose([np.nan if x is None else x for x in g],
                                           [np.nan if x is None else x for x in w], rtol=1e-12)
            else:
                assert g == w, (sql, c)


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_left_side_with_unknown_columns_takes_the_fallback(how):
    """A left side whose columns the planner cannot name (an unnamed expression) goes through
    `_execute_hash_join`'s semi / anti branch; same answer."""
    left, right = _tables(5000)
    lf = LazyFrame(left).select(col("k"), col("v") * lit(2.0))
    got = lf.join(LazyFrame(right), left_on="k", right_on="rk", how=how).compute(device="cpu")
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    want = _duck(f"SELECT k FROM l WHERE {exists} (SELECT 1 FROM r WHERE r.rk = l.k) ORDER BY rowid", left, right)
    assert got.column("k").to_pylist() == want.column("k").to_pylist()


def test_sql_join_types():
    l = pa.table({"k": pa.array([1, 2, 3, None], pa.int64()), "a": pa.array([10, 20, 30, 40], pa.int64())})
    r = pa.table({"k": pa.array([2, 4, 2], pa.int64()), "w": pa.array([0.5, 0.7, 0.9])})
    run = lambda q: mx.sql(q, l=l, r=r).compute(device="cpu").to_pydict()
    assert run("SELECT l.k, l.a, r.w FROM l LEFT JOIN r ON l.k = r.k ORDER BY a, w") == \
        {"k": [1, 2, 2, 3, None], "a": [10, 20, 20, 30, 40], "w": [None, 0.5, 0.9, None, None]}
    assert run("SELECT * FROM l SEMI JOIN r ON l.k = r.k") == {"k": [2], "a": [20]}
    for q in ("SELECT * FROM l ANTI JOIN r ON l.k = r.k", "SELECT * FROM l LEFT ANTI JOIN r ON l.k = r.k"):
        assert run(q) == {"k": [1, 3, None], "a": [10, 30, 40]}
    assert run("SELECT l.k FROM l JOIN r ON l.k = r.k") == {"k": [2, 2]}
    assert run("SELECT k, a FROM l JOIN r USING (k) ORDER BY a") == {"k": [2, 2], "a": [20, 20]}
    for q in ("SELECT * FROM l FULL OUTER JOIN r ON l.k = r.k", "SELECT * FROM l RIGHT JOIN r ON l.k = r.k",
              "SELECT * FROM l RIGHT SEMI JOIN r ON l.k = r.k",
              "SELECT * FROM l RIGHT ANTI JOIN r ON l.k = r.k"):
        with pytest.raises(ValueError, match="unsupported SQL join"):
            mx.sql(q, l=l, r=r).compute(device="cpu")
    assert run("SELECT COUNT(*) AS n FROM l CROSS JOIN r") == {"n": [12]}          # F-096


def test_only_the_right_keys_are_read_from_parquet(tmp_path):
    import pyarrow.parquet as pq
    from omxcore.io_parquet import resolve_plan
    from omxcore.lazy_frame import ParquetScan
    left, right = _tables(2000)
    pq.write_table(left, tmp_path / "l.parquet")
    pq.write_table(right, tmp_path / "r.parquet")
    for how in ("semi", "anti"):
        for top in (lambda q: q, lambda q: q.sort("k"), lambda q: q.select("s", "k")):
            q = top(mx.scan_parquet(str(tmp_path / "l.parquet")).join(
                mx.scan_parquet(str(tmp_path / "r.parquet")), left_on="k", right_on="rk", how=how))
            from omxcore.optimizer import optimize_plan
            plan = resolve_plan(optimize_plan(q.plan).plan, read=False)
            scans = []

            def walk(n):
                if isinstance(n, ParquetScan):
                    scans.append(n)
                for ch in ("input", "left", "right"):
                    if hasattr(n, ch):
                        walk(getattr(n, ch))
            walk(plan)
            right_scan = [sc for sc in scans if sc.path.endswith("r.parquet")][0]
            assert list(right_scan.columns) == ["rk"], (how, right_scan.columns)
