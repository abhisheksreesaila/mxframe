"""F-086: SQL ``IS [NOT] NULL`` (and IS [NOT] DISTINCT FROM, IS [NOT] TRUE / FALSE) in WHERE, SELECT, CASE,
HAVING, subqueries and the anti-join pattern. Expected answers come from DuckDB native tables. A float NaN
is not NULL. Run at a size the Arrow fallback takes and at one the native filter takes, on cpu and gpu."""
import math

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import pipeline_kernels as pk

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]
SIZES = [60, 150_000]


def _t(n):
    r = np.random.default_rng(86)
    ids = np.arange(n, dtype=np.int64)
    f = r.random(n)
    f[r.random(n) < 0.15] = np.nan                                   # NaN is a value
    words = np.array(["red", "green", "blue", "grey"])[r.integers(0, 4, n)]
    g = r.integers(0, 5, n)
    i = r.integers(0, 8, n)
    i_null = r.random(n) < 0.25
    i_null[g == 4] = True                                            # group 4: every i is NULL
    s = pa.array(words, pa.string(), mask=r.random(n) < 0.25)
    d = pa.array(words, pa.string(), mask=r.random(n) < 0.25).dictionary_encode()
    days = pa.array(r.integers(18000, 19000, n).astype(np.int32), pa.date32(), mask=r.random(n) < 0.25)
    return pa.table({
        "id": pa.array(ids),
        "g": pa.array(g.astype(np.int64)),
        "i": pa.array(i.astype(np.int64), mask=i_null),
        "j": pa.array(r.integers(0, 8, n).astype(np.int64), mask=r.random(n) < 0.3),
        "f": pa.array(f, mask=r.random(n) < 0.15),                   # NULLs and NaNs
        "s": s, "d": d, "dt": days,
    })


def _u():
    return pa.table({"k": pa.array([0, 1, 2, 3, 4, 5], pa.int64()),
                     "v": pa.array([10, None, 30, None, 50, 60], pa.int64())})


def _a(n):
    """Same key name as u (k), a float pair with NaNs and NULLs."""
    r = np.random.default_rng(87)
    f = r.integers(0, 3, n).astype(np.float64)
    f2 = r.integers(0, 3, n).astype(np.float64)
    f[r.random(n) < 0.3] = np.nan
    f2[r.random(n) < 0.3] = np.nan
    return pa.table({"id": pa.array(np.arange(n, dtype=np.int64)),
                     "k": pa.array(r.integers(0, 8, n).astype(np.int64), mask=r.random(n) < 0.25),
                     "f": pa.array(f, mask=r.random(n) < 0.2), "f2": pa.array(f2, mask=r.random(n) < 0.2)})


def _w():
    """A column named like the engine's internal copy of u.k."""
    return pa.table({"k": pa.array([0, 1, 7], pa.int64()), "__rk_u_k": pa.array([5, 6, 7], pa.int64()),
                     "__m_u": pa.array([8, 9, 10], pa.int64())})


def _norm(x):
    if isinstance(x, float):
        return "nan" if math.isnan(x) else round(x, 6)
    return x


def _rows(tbl):
    cols = tbl.column_names
    return cols, list(zip(*[[_norm(x) for x in tbl.column(c).to_pylist()] for c in cols]))


def _duck(q, tables):
    con = duckdb.connect()
    for name, tbl in tables.items():
        con.register(name + "_arrow", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
    return con.execute(q).to_arrow_table()


def _check(q, n, device, **extra):
    tables = {"t": _t(n), "u": _u(), **extra}
    exp = _duck(q, tables)
    got = mx.sql(q, **tables).compute(device=device)
    gc, g = _rows(got)
    ec, e = _rows(exp)
    assert gc == ec
    if "ORDER BY" not in q:
        g, e = sorted(g, key=repr), sorted(e, key=repr)
    assert g == e
    return len(e)


COLS = ["i", "f", "s", "d", "dt"]

WHERE = [f"SELECT id FROM t WHERE {c} IS {neg}NULL ORDER BY id" for c in COLS for neg in ("", "NOT ")]
SELECT = [f"SELECT id, {c} IS NULL AS a, {c} IS NOT NULL AS b FROM t ORDER BY id" for c in COLS]
CASE = [
    "SELECT id, CASE WHEN s IS NULL THEN 1 ELSE 0 END AS z FROM t ORDER BY id",
    "SELECT id, CASE WHEN f IS NULL THEN 1 WHEN f IS NOT NULL AND i IS NULL THEN 2 ELSE 3 END AS z FROM t ORDER BY id",
    "SELECT g, SUM(CASE WHEN i IS NULL THEN 1 ELSE 0 END) AS n, SUM(CASE WHEN d IS NOT NULL THEN 1 ELSE 0 END) AS m "
    "FROM t GROUP BY g ORDER BY g",
]
HAVING = [
    "SELECT g, SUM(i) AS si FROM t GROUP BY g HAVING SUM(i) IS NULL ORDER BY g",
    "SELECT g, SUM(i) AS si FROM t GROUP BY g HAVING SUM(i) IS NOT NULL ORDER BY g",
    "SELECT g, MAX(s) AS m FROM t GROUP BY g HAVING MAX(s) IS NOT NULL AND COUNT(*) > 3 ORDER BY g",
    "SELECT g, COUNT(*) AS n FROM t WHERE s IS NOT NULL GROUP BY g HAVING MIN(f) IS NOT NULL ORDER BY g",
]
JOINS = [
    "SELECT t.id FROM t LEFT JOIN u ON t.i = u.k WHERE u.k IS NULL ORDER BY t.id",
    "SELECT t.id FROM t LEFT JOIN u ON t.i = u.k WHERE u.k IS NOT NULL ORDER BY t.id",
    "SELECT t.id FROM t LEFT JOIN u ON t.i = u.k WHERE u.v IS NULL ORDER BY t.id",
    "SELECT t.id, u.k IS NULL AS lonely FROM t LEFT JOIN u ON t.j = u.k ORDER BY t.id",
    "SELECT t.g, COUNT(*) AS n FROM t LEFT JOIN u ON t.j = u.k WHERE u.k IS NULL AND t.s IS NOT NULL "
    "GROUP BY t.g ORDER BY t.g",
]
SUBQUERIES = [
    "SELECT id FROM t WHERE j IN (SELECT k FROM u WHERE v IS NULL) ORDER BY id",
    "SELECT id FROM t WHERE i > (SELECT MIN(k) FROM u WHERE v IS NOT NULL) AND s IS NULL ORDER BY id",
    "SELECT x.id FROM (SELECT id, s FROM t WHERE i IS NOT NULL) x WHERE x.s IS NULL ORDER BY x.id",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.j AND u.v IS NULL) AND d IS NOT NULL ORDER BY id",
]
COMBINED = [
    "SELECT id FROM t WHERE i IS NULL OR f IS NULL ORDER BY id",
    "SELECT id FROM t WHERE (i IS NULL OR f IS NULL) AND NOT (s IS NULL) ORDER BY id",
    "SELECT id FROM t WHERE NOT (i IS NULL AND s IS NULL) ORDER BY id",
    "SELECT id FROM t WHERE NOT i IS NULL ORDER BY id",
    "SELECT id FROM t WHERE i IS NULL OR i > 3 ORDER BY id",
    "SELECT id FROM t WHERE i > 3 AND s IS NOT NULL AND d IS NULL ORDER BY id",
    "SELECT id FROM t WHERE s IS NULL OR s = 'red' ORDER BY id",
    "SELECT id FROM t WHERE NOT (i > 3) AND f IS NOT NULL ORDER BY id",
]
DISTINCT = [
    "SELECT id FROM t WHERE i IS DISTINCT FROM 3 ORDER BY id",
    "SELECT id FROM t WHERE i IS NOT DISTINCT FROM 3 ORDER BY id",
    "SELECT id FROM t WHERE i IS NOT DISTINCT FROM NULL ORDER BY id",
    "SELECT id FROM t WHERE i IS DISTINCT FROM NULL ORDER BY id",
    "SELECT id FROM t WHERE i IS NOT DISTINCT FROM j ORDER BY id",
    "SELECT id FROM t WHERE i IS DISTINCT FROM j ORDER BY id",
    "SELECT id FROM t WHERE s IS DISTINCT FROM 'red' ORDER BY id",
]
TRUTH = [
    "SELECT id FROM t WHERE (i > 3) IS TRUE ORDER BY id",
    "SELECT id FROM t WHERE (i > 3) IS NOT TRUE ORDER BY id",
    "SELECT id FROM t WHERE (i > 3) IS FALSE ORDER BY id",
    "SELECT id FROM t WHERE (i > 3) IS NOT FALSE ORDER BY id",
]

ALL = WHERE + SELECT + CASE + HAVING + JOINS + SUBQUERIES + COMBINED + DISTINCT + TRUTH


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("q", ALL)
def test_matches_duckdb(q, n, device):
    assert _check(q, n, device) > 0 or "HAVING" in q


def test_nan_is_not_null():
    t = pa.table({"f": pa.array([1.0, float("nan"), None], pa.float64())})
    got = mx.sql("SELECT f IS NULL AS a, f IS NOT NULL AS b FROM t", t=t).compute(device="cpu")
    assert got.column("a").to_pylist() == [False, False, True]
    assert got.column("b").to_pylist() == [True, True, False]


@pytest.mark.parametrize("make, where", [
    (lambda c: c("i").is_null(), "i IS NULL"),
    (lambda c: ~c("s").is_null() & (c("i") > 3), "s IS NOT NULL AND i > 3"),
    (lambda c: c("i").is_null() | c("s").is_null(), "i IS NULL OR s IS NULL"),
])
def test_dataframe_filter_on_parquet(make, where, tmp_path):
    """``col(x).is_null()`` through a Parquet scan: row-group pruning and pushed filters must not read it as a comparison."""
    import pyarrow.parquet as pq
    path = str(tmp_path / "t.parquet")
    pq.write_table(_t(200_000).select(["id", "i", "s"]), path, row_group_size=20_000)
    exp = [r[0] for r in duckdb.sql(f"SELECT id FROM '{path}' WHERE {where} ORDER BY id").fetchall()]
    got = mx.scan_parquet(path).filter(make(mx.col)).select(mx.col("id")).compute(device="cpu")
    assert sorted(got.column("id").to_pylist()) == exp


# ---- review fixes: a LEFT JOIN's right key, NaN in DISTINCT FROM, the alias in the join cache, ON, constants ----

RIGHT_KEY = [
    "SELECT a.id, u.k FROM a LEFT JOIN u ON a.k = u.k ORDER BY a.id",
    "SELECT a.id, u.k FROM a LEFT JOIN u ON a.k = u.k WHERE u.k IS NULL ORDER BY a.id",
    "SELECT a.id FROM a LEFT JOIN u ON a.k = u.k WHERE u.k IS NOT DISTINCT FROM NULL ORDER BY a.id",
    "SELECT a.id FROM a LEFT JOIN u ON a.k = u.k WHERE u.k IS DISTINCT FROM NULL ORDER BY a.id",
    "SELECT a.id, u.k AS rk, a.k AS lk FROM a LEFT JOIN u ON a.k = u.k ORDER BY a.id",
    "SELECT u.k, COUNT(*) AS n FROM a LEFT JOIN u ON a.k = u.k GROUP BY u.k ORDER BY u.k",
    "SELECT a.id, u.k, u.v FROM a LEFT JOIN u ON a.k = u.k ORDER BY a.id",
    "SELECT a.id, kk FROM a LEFT JOIN (SELECT k AS kk, v FROM u) x ON a.k = x.kk ORDER BY a.id",
    "SELECT a.id FROM a LEFT JOIN (SELECT k AS kk, v FROM u) x ON a.k = x.kk WHERE x.kk IS NULL ORDER BY a.id",
    "SELECT a.id, x.kk FROM a LEFT JOIN (SELECT k AS kk, v FROM u) x ON a.k = x.kk ORDER BY a.id",
    "SELECT a.id FROM a LEFT JOIN u USING (k) WHERE k IS NULL ORDER BY a.id",
    "SELECT a.id FROM a LEFT JOIN u USING (k) WHERE u.k IS NULL ORDER BY a.id",
    "SELECT a.id, k FROM a LEFT JOIN u USING (k) ORDER BY a.id",
    "SELECT a.id, u.k FROM a LEFT JOIN u USING (k) ORDER BY a.id",
    "SELECT a.id FROM a LEFT JOIN u ON a.k = u.k LEFT JOIN u AS u2 ON a.id = u2.k WHERE u.k IS NULL AND u2.k IS NOT NULL "
    "ORDER BY a.id",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", RIGHT_KEY)
def test_left_join_right_key_is_null_where_unmatched(q, device):
    assert _check(q, 40, device, a=_a(40)) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_left_join_copy_never_in_a_result(device):
    """SELECT * (also inside a subquery) has the columns of the tables (both keys), not the internal copy of the key."""
    a = _a(40)
    for q in ["SELECT * FROM a LEFT JOIN u ON a.k = u.k WHERE u.k IS NULL",
              "SELECT * FROM (SELECT * FROM a LEFT JOIN u ON a.k = u.k WHERE u.k IS NULL) x"]:
        got = mx.sql(q, a=a, u=_u()).compute(device=device)
        assert got.column_names == ["id", "k", "f", "f2", "k_1", "v"]       # both key columns, as DuckDB
        assert got.num_rows == _duck(q, {"a": a, "u": _u()}).num_rows > 0


@pytest.mark.parametrize("device", DEVICES)
def test_left_join_copy_name_does_not_collide(device):
    q = ("SELECT a.id, w.__rk_u_k, w.__m_u FROM a LEFT JOIN w ON a.k = w.k LEFT JOIN u ON a.k = u.k "
         "WHERE u.k IS NULL ORDER BY a.id")
    assert _check(q, 40, device, a=_a(40), w=_w()) > 0


FLOAT_DISTINCT = [
    "SELECT id FROM a WHERE f IS DISTINCT FROM f2 ORDER BY id",
    "SELECT id FROM a WHERE f IS NOT DISTINCT FROM f2 ORDER BY id",
    "SELECT id, f IS DISTINCT FROM f AS x, f IS NOT DISTINCT FROM f2 AS y FROM a ORDER BY id",
    "SELECT id FROM a WHERE f IS DISTINCT FROM 1.0 ORDER BY id",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", FLOAT_DISTINCT)
def test_two_nans_are_not_distinct(q, device):
    assert _check(q, 60, device, a=_a(60)) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_plans_differing_only_in_an_alias_do_not_share_a_join_result(device):
    """The join cache key left out the output names: the second plan got the first one's column names."""
    t = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "k": pa.array([0, 1, 2], pa.int64())})
    u = pa.table({"k": pa.array([0, 1], pa.int64()), "v": pa.array([10, 20], pa.int64())})
    for name in ("p", "q", "p"):
        right = mx.LazyFrame(u).select(mx.col("k"), (mx.col("v") + 0).alias(name))
        got = mx.LazyFrame(t).join(right, left_on="k", right_on="k").compute(device=device)
        assert got.column_names == ["id", "k", name]
        assert got.column(name).to_pylist() == [10, 20]


ON_NULL = [
    "SELECT t.id FROM t LEFT JOIN u ON t.i = u.k AND u.v IS NULL ORDER BY t.id",
    "SELECT t.id, u.v FROM t LEFT JOIN u ON t.i = u.k AND u.v IS NOT NULL ORDER BY t.id",
    "SELECT t.id FROM t JOIN u ON t.i = u.k AND u.v IS NULL ORDER BY t.id",
    "SELECT t.id FROM t LEFT JOIN u ON t.i = u.k AND u.v IS NULL WHERE u.k IS NULL ORDER BY t.id",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", ON_NULL)
def test_is_null_on_the_right_table_in_join_on(q, device):
    assert _check(q, 60, device) > 0


@pytest.mark.parametrize("q", [
    "SELECT id FROM t WHERE 1 IS DISTINCT FROM 2",
    "SELECT id FROM t WHERE NULL IS NULL",
    "SELECT id FROM t WHERE 1 IS NOT DISTINCT FROM NULL",
])
def test_constants_only_raise_a_clear_error(q):
    with pytest.raises(NotImplementedError, match="no column to filter"):
        mx.sql(q, t=_t(10)).compute(device="cpu")
