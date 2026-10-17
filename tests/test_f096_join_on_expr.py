"""F-096: JOIN ... ON with an expression. Equi-joins on computed keys, residual conditions next to the equality
(INNER: a filter after the join; LEFT: the residual decides which right rows match, so a left row whose partners
all fail it comes back once with NULLs), and joins with no equality at all (a cross join with a condition,
small inputs only). Expected answers come from DuckDB native tables. NULLs sit in the keys, in the residual
columns and in the computed keys; k = 2 has two partners, k = 3 two, a key that no partner shares exists on both
sides."""
import duckdb
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import pipeline_kernels as pk

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]


def _i(*v):
    return pa.array(list(v), pa.int64())


def _tables():
    return {
        "t": pa.table({
            "id": _i(1, 2, 3, 4, 5, 6, 7, 8, 9),
            "k": _i(1, 2, 2, 3, 4, None, 5, 6, 7),
            "a": _i(1, 2, 3, 4, 5, 6, None, 8, 9),
            "x": _i(10, 20, 5, 30, 40, 50, 60, None, 15),
            "s": pa.array(["apple", "banana", "cherry", "avocado", "blueberry", None, "cherry", "date", "apricot"]),
        }),
        "u": pa.table({
            "k": _i(1, 2, 2, 3, 3, 4, None, 8, 2),
            "b": _i(2, 3, 4, 5, 6, 9, 7, None, 10),
            "y": _i(5, 25, 10, 20, 35, 50, 1, 2, None),
            "x": _i(1, 2, 3, 4, 5, 6, 7, 8, 9),
            "s2": pa.array(["a", "b", "c", "a", "b", "c", "d", None, "a"]),
        }),
        "v": pa.table({"k": _i(1, 2, 3, 4, 8, None), "q": _i(7, 12, 22, 3, 9, 100)}),
    }


def _norm(tbl):
    cols = tbl.column_names
    return cols, sorted(zip(*[tbl.column(c).to_pylist() for c in cols]), key=repr)


def _expected(q):
    tables = _tables()
    con = duckdb.connect()
    for name, tbl in tables.items():
        con.register(name + "_arrow", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
    return con.execute(q).to_arrow_table()


def _check(q, device):
    exp = _expected(q)
    got = mx.sql(q, **_tables()).compute(device=device)
    assert _norm(got) == _norm(exp)
    return exp.num_rows


# ---- an ON clause, tried with JOIN and LEFT JOIN -------------------------------------------------------------
COMPUTED = [
    "t.a + 1 = u.b",
    "u.b = t.a + 1",
    "(t.a + 1) = (u.b)",
    "t.a * 2 = u.b - 1",
    "t.a % 3 = u.k",
    "t.a / 2 = u.k",
    "u.k + 1 = t.k",
    "substring(t.s, 1, 1) = u.s2",
    "CASE WHEN t.a > 4 THEN t.a ELSE 0 END = u.k",
    "t.a + 1 = u.b AND t.k = u.k",
    "t.k = u.k AND u.y * 2 = t.x",
    "t.a + t.k = u.b + u.k",
]
RESIDUAL = [
    "t.k = u.k AND t.x > u.y",
    "u.y < t.x AND u.k = t.k",
    "t.k = u.k AND t.x >= u.y AND t.x < 40",
    "t.k = u.k AND (t.x > u.y OR t.x < 8)",
    "t.k = u.k AND t.x > 15",                       # a test on the left table only
    "t.k = u.k AND t.x < 15 AND t.a > 1",
    "t.k = u.k AND u.y < 30",                       # on the right table only
    "t.k = u.k AND u.y IS NOT NULL",
    "t.k = u.k AND t.x > u.y + 3",
    "t.k = u.k AND t.x - u.y > 5 AND t.a < 8",
    "t.k = u.k AND t.x <> u.y",
    "t.k = u.k AND t.x > u.x",                      # x is in both tables (the right one is renamed)
    "t.k = u.k AND u.k < 3",                        # the right key in the residual
    "t.k = u.k AND t.k < 3",
    "t.k = u.k AND t.a + u.b > 8",
    "t.k = u.k AND NOT (t.x > u.y)",
    "t.k = u.k AND t.x > 1000",                     # nothing survives
    "t.k = u.k AND t.x > 0",                        # everything survives
    "t.a + 1 = u.b AND t.x > u.y",                  # a computed key and a residual
    "t.k = u.k AND t.a = u.x AND t.x > u.y",
]
THETA = [
    "t.x > u.y",
    "t.x > u.y AND t.a < 5",
    "t.k = u.k OR t.x > u.y",
    "t.k < u.k",
    "t.a + 1 > u.b",
    "t.x BETWEEN u.y AND u.y + 10",
    "t.k <> u.k",
    "t.x > 15",                                     # a condition on the left table only
    "u.y < 10",                                     # on the right table only
    "t.a > 5 AND u.b < 9",
    "t.a = 1 OR u.b = 9",
]


def _selects(on, kind):
    return [f"SELECT t.id, u.b, u.y FROM t {kind} u ON {on}",
            f"SELECT t.id, u.k AS uk, u.x AS ux FROM t {kind} u ON {on}"]


QUERIES = [q for on in COMPUTED + RESIDUAL + THETA for kind in ("JOIN", "LEFT JOIN") for q in _selects(on, kind)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", QUERIES)
def test_join_on_expression_matches_duckdb(q, device):
    _check(q, device)


MORE = [
    # aggregates and WHERE over a LEFT JOIN with a residual: each left row counts once
    "SELECT t.id, COUNT(u.y) AS n, COUNT(*) AS rows_ FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y GROUP BY t.id",
    "SELECT COUNT(*) AS n FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y",
    "SELECT SUM(u.y) AS s, COUNT(*) AS n FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y",
    "SELECT t.id FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y WHERE u.y IS NULL",
    "SELECT t.id FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y WHERE u.y IS NOT NULL",
    "SELECT t.id, u.y FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y WHERE t.a > 2 ORDER BY t.id, u.y",
    "SELECT t.id, u.y FROM t LEFT JOIN u ON t.a + 1 = u.b ORDER BY t.id LIMIT 4",
    "SELECT t.*, u.y FROM t LEFT JOIN u ON t.a + 1 = u.b",
    "SELECT u.k, t.id FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y",
    "SELECT u.k AS uk, u.y FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y AND u.k < 3",
    # an inner join, a filter, an aggregate
    "SELECT u.k, COUNT(*) AS n FROM t JOIN u ON t.k = u.k AND t.x > u.y GROUP BY u.k",
    "SELECT t.k, MAX(t.x - u.y) AS m FROM t JOIN u ON t.a + 1 = u.b AND t.x > u.y GROUP BY t.k",
    # aliases and a self-join
    "SELECT a.id, b.y FROM t a LEFT JOIN u b ON a.k = b.k AND a.x > b.y",
    "SELECT a.id, b.y FROM t AS a JOIN u AS b ON b.y < a.x AND b.k = a.k",
    "SELECT p.id, q.id AS qid FROM t p LEFT JOIN t q ON p.k = q.k AND p.x > q.x",
    "SELECT p.id, q.id AS qid FROM t p JOIN t q ON p.a + 1 = q.a AND p.x < q.x",
    "SELECT p.id, q.id AS qid FROM t p LEFT JOIN t q ON p.a + 1 = q.a",
    "SELECT p.id, q.id AS qid FROM t p JOIN t q ON p.x > q.x AND p.k < q.k",
    # three tables
    "SELECT t.id, u.y, v.q FROM t JOIN u ON t.k = u.k AND t.x > u.y JOIN v ON u.k = v.k",
    "SELECT t.id, u.y, v.q FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y LEFT JOIN v ON v.k = u.k",
    "SELECT t.id, u.y, v.q FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y LEFT JOIN v ON v.k = u.k AND v.q > u.y",
    "SELECT t.id, u.y, v.q FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y LEFT JOIN v ON v.k = t.k AND v.q > u.y",
    "SELECT t.id, u.y, v.q FROM t LEFT JOIN u ON t.a + 1 = u.b LEFT JOIN v ON v.k = u.k",
    "SELECT t.id, u.y, v.q FROM t JOIN u ON t.k = u.k JOIN v ON t.a + 1 = v.q AND u.y < v.q",
    "SELECT t.id, u.y, v.q FROM t JOIN u ON t.k = u.k LEFT JOIN v ON v.q > t.x AND v.q > u.y",
    "SELECT t.id, u.y, v.q FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN v ON v.k = u.k AND u.y IS NOT NULL",
    "SELECT t.id, u.y, v.q FROM t LEFT JOIN u ON t.x > u.y LEFT JOIN v ON v.q < t.x",
    # a text column on the blanked side
    "SELECT t.id, u.s2, t.s FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y",
    "SELECT t.id, u.s2 FROM t LEFT JOIN u ON t.s = 'cherry' AND t.k = u.k AND u.s2 <> 'a'",
    # the equality keeps working next to the new forms
    "SELECT t.id, u.y FROM t JOIN u USING (k)",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", MORE)
def test_more_shapes_match_duckdb(q, device):
    _check(q, device)


@pytest.mark.parametrize("device", DEVICES)
def test_left_join_residual_returns_unmatched_left_rows_once(device):
    """The key case. t.k = 2 has partners (y = 25, 10, None) but t.x = 5 beats none of them; t.k = 6 and 7 have
    partners that do not exist or fail. Every left row is in the result: a row with no surviving partner
    exactly once, with NULLs; a row with partners once per surviving partner."""
    q = "SELECT t.id, u.y FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y ORDER BY t.id, u.y"
    got = mx.sql(q, **_tables()).compute(device=device).to_pylist()
    assert got == _expected(q).to_pylist()
    ids = [r["id"] for r in got]
    assert sorted(set(ids)) == list(range(1, 10))                       # nobody lost
    assert [r["y"] for r in got if r["id"] == 3] == [None]              # k = 2, x = 5: nothing below 5
    assert [r["y"] for r in got if r["id"] == 2] == [10]                # k = 2, x = 20: 10 only (25 and NULL fail)
    assert ids.count(3) == 1 and ids.count(6) == 1 and ids.count(8) == 1
    assert [r["y"] for r in got if r["id"] == 4] == [20]                # k = 3, x = 30: y 20 (35 fails)


@pytest.mark.parametrize("device", DEVICES)
def test_inner_join_residual_drops_the_rows_it_rejects(device):
    q = "SELECT t.id, u.y FROM t JOIN u ON t.k = u.k AND t.x > u.y ORDER BY t.id, u.y"
    got = mx.sql(q, **_tables()).compute(device=device).to_pylist()
    assert got == _expected(q).to_pylist()
    assert 3 not in [r["id"] for r in got]


@pytest.mark.parametrize("device", DEVICES)
def test_computed_key_values(device):
    q = "SELECT t.id, u.b FROM t JOIN u ON t.a + 1 = u.b ORDER BY t.id"
    got = mx.sql(q, **_tables()).compute(device=device).to_pylist()
    assert got == _expected(q).to_pylist()
    assert got == [{"id": i, "b": i + 1} for i in (1, 2, 3, 4, 5, 6, 8, 9)]      # a NULL a (id 7) joins nothing


def test_no_equality_and_large_inputs_is_a_clear_error():
    big = pa.table({"a": pa.array(range(5000), pa.int64())})
    other = pa.table({"b": pa.array(range(5000), pa.int64())})
    with pytest.raises(NotImplementedError, match="no equality"):
        mx.sql("SELECT a, b FROM big JOIN other ON big.a > other.b", big=big, other=other).compute()
    with pytest.raises(NotImplementedError, match="no equality"):
        mx.sql("SELECT a, b FROM big LEFT JOIN other ON big.a > other.b", big=big, other=other).compute()


def test_no_equality_with_small_inputs_runs():
    small = pa.table({"a": pa.array(range(300), pa.int64())})
    other = pa.table({"b": pa.array(range(300), pa.int64())})
    got = mx.sql("SELECT COUNT(*) AS n FROM small JOIN other ON small.a > other.b", small=small, other=other).compute()
    assert got.to_pylist() == [{"n": 300 * 299 // 2}]


def test_a_condition_with_no_column_is_a_clear_error():
    with pytest.raises(NotImplementedError, match="refers to no column"):
        mx.sql("SELECT t.id FROM t JOIN u ON t.k = u.k AND 1 = 1", **_tables()).compute()


def test_comma_join_with_a_non_equality_where():
    for q in ("SELECT t.id, u.y FROM t, u WHERE t.x > u.y",
              "SELECT t.id, u.y FROM t, u WHERE t.x > u.y AND t.a < 4",
              "SELECT t.id, u.y, v.q FROM t, u, v WHERE t.k = u.k AND v.q > u.y"):
        _check(q, "cpu")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [
    "SELECT * FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y",
    "SELECT * FROM t JOIN u ON t.k = u.k AND t.x > u.y",
    "SELECT * FROM t LEFT JOIN u ON t.a + 1 = u.b",
    "SELECT * FROM t JOIN u ON t.a + 1 = u.b",
    "SELECT * FROM t LEFT JOIN u ON t.x > u.y",
])
def test_select_star_has_both_tables_columns_and_no_internal_ones(q, device):
    """DuckDB repeats a name (k, k); mx numbers the repeat (k_1). Values compared by position."""
    exp = _expected(q)
    got = mx.sql(q, **_tables()).compute(device=device)
    assert got.column_names == ["id", "k", "a", "x", "s", "k_1", "b", "y", "x_1", "s2"]
    rows = lambda t: sorted(zip(*[c.to_pylist() for c in t.columns]), key=repr)
    assert rows(got) == rows(exp)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [
    "SELECT t.id, u.y FROM t CROSS JOIN u WHERE t.x > u.y",
    "SELECT t.id, u.y FROM t CROSS JOIN u WHERE t.x > u.y AND t.k = 2",
    "SELECT COUNT(*) AS n FROM t CROSS JOIN u",
    "SELECT COUNT(*) AS n FROM t, u",
    "SELECT t.id, v.q FROM t, v WHERE t.x < v.q",
    "SELECT t.id, u.y, v.q FROM t, u, v WHERE t.x > u.y AND v.q > u.y AND t.k = v.k",
    "SELECT t.id, u.y FROM t, u WHERE t.x > u.y ORDER BY t.id, u.y LIMIT 5",
])
def test_cross_join_with_a_condition(q, device):
    _check(q, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("on", [
    "t.k = u.k AND t.x > u.y",
    "t.k = u.k AND t.x > u.y AND t.a + u.b < 700",
    "t.a + 1 = u.b AND t.x > u.y",
    "t.k = u.k AND (t.x > u.y OR u.y IS NULL)",
])
def test_larger_random_tables_with_nulls_and_duplicate_rows(on, device):
    """20k left rows (duplicates, NULL keys and residual values) against 3k right rows: row counts and sums
    equal DuckDB's, for LEFT and INNER, so the row ids and windows hold up past the toy sizes."""
    import numpy as np
    rng = np.random.default_rng(7)
    n, m = 20_000, 3_000

    def col(size, hi, null_every):
        v = rng.integers(0, hi, size)
        return pa.array([None if i % null_every == 0 else int(x) for i, x in enumerate(v)], pa.int64())
    tabs = {"t": pa.table({"k": col(n, 1500, 11), "a": col(n, 800, 13), "x": col(n, 1000, 17)}),
            "u": pa.table({"k": col(m, 1500, 19), "b": col(m, 800, 23), "y": col(m, 1000, 29)})}
    con = duckdb.connect()
    for name, tbl in tabs.items():
        con.register(name + "_arrow", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
    for kind in ("JOIN", "LEFT JOIN"):
        q = (f"SELECT COUNT(*) AS n, SUM(t.x) AS sx, SUM(u.y) AS sy, SUM(t.a) AS sa, COUNT(u.y) AS cy "
             f"FROM t {kind} u ON {on}")
        assert mx.sql(q, **tabs).compute(device=device).to_pylist() == con.execute(q).to_arrow_table().to_pylist()


@pytest.mark.parametrize("device", DEVICES)
def test_parquet_frames_with_dictionary_text_and_dates(tmp_path, device):
    """The same LEFT JOIN residual over scan_parquet frames, with a dictionary-encoded string and a date on the
    side that gets blanked."""
    import datetime as dt
    import pyarrow.parquet as pq
    t = pa.table({"id": _i(1, 2, 3, 4, 5), "k": _i(1, 2, 2, 3, None), "x": _i(10, 20, 5, 30, 1)})
    d0 = dt.date(2024, 1, 1)
    u = pa.table({"k": _i(1, 2, 2, 3, 9), "y": _i(5, 25, 10, 20, 1),
                  "c": pa.array(["a", "b", "a", "b", "a"]), "d": pa.array([d0 + dt.timedelta(days=i) for i in range(5)])})
    pq.write_table(t, tmp_path / "t.parquet")
    pq.write_table(u, tmp_path / "u.parquet")
    q = "SELECT t.id, u.y, u.c, u.d FROM t LEFT JOIN u ON t.k = u.k AND t.x > u.y"
    con = duckdb.connect()
    con.register("t", t)
    con.register("u", u)
    exp = con.execute(q).to_arrow_table()
    got = mx.sql(q, t=mx.scan_parquet(str(tmp_path / "t.parquet")),
                 u=mx.scan_parquet(str(tmp_path / "u.parquet"))).compute(device=device)
    assert _norm(got) == _norm(exp)
    assert sorted(got.column("id").to_pylist()) == [1, 2, 3, 4, 5]


def test_cross_join_cap_counts_a_left_join_with_an_empty_right_side_as_its_left_rows():
    """A LEFT JOIN to an empty table still yields every left row: its bound is not 0, so 3000 x 3000 is refused."""
    big = pa.table({"a": pa.array(range(3000), pa.int64())})
    other = pa.table({"b": pa.array(range(3000), pa.int64())})
    empty = pa.table({"k": pa.array([], pa.int64())})
    q = ("SELECT d.a, other.b FROM (SELECT big.a FROM big LEFT JOIN empty ON big.a = empty.k) d "
         "JOIN other ON d.a > other.b")
    with pytest.raises(NotImplementedError, match="up to 3000 x 3000 rows"):
        mx.sql(q, big=big, other=other, empty=empty).compute()


def test_cross_join_of_large_inputs_is_a_clear_error():
    big = pa.table({"a": pa.array(range(5000), pa.int64())})
    other = pa.table({"b": pa.array(range(5000), pa.int64())})
    with pytest.raises(NotImplementedError, match="no equality"):
        mx.sql("SELECT a, b FROM big, other WHERE big.a > other.b", big=big, other=other).compute()
