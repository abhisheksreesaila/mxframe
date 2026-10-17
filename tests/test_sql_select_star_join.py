"""F-102: `SELECT *` over `JOIN ... ON` (and the comma join) keeps both key columns, each with its own type and
values; USING / NATURAL keep one. A key column a LEFT JOIN did not match is NULL. Expected values and types come
from DuckDB native tables. Output names: a native table (and `con.register`) keeps a repeated name (`k`, `k`)
at the top level, but `duckdb.sql(q)` over Python Arrow variables (the replacement scan) and a subquery give
`k_1`. mx keeps `k_1` (F-088's convention, and a result table with two columns of one name is a trap), so the
expected names are renamed that way."""
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
        # t.k 5 and 6 have no partner in u / w / fl; t.k NULL never matches
        "t": pa.table({"i": _i(1, 2, 3, 4, 5, 6, 7), "k": _i(1, 2, 2, 3, 5, 6, None),
                       "v": _i(10, 20, 30, 40, 50, 60, 70)}),
        "u": pa.table({"k": _i(1, 2, 3, 4), "v": _i(100, 200, 300, 400), "w": _i(7, 8, 9, 10)}),
        "w": pa.table({"x": _i(1, 2, 3, 4), "n": _i(11, 22, 33, 44)}),
        # shares exactly one column name (k) with t, for NATURAL JOIN
        "w2": pa.table({"k": _i(1, 2, 3, 4), "n": _i(11, 22, 33, 44)}),
        "fl": pa.table({"x": pa.array([1.0, 2.0, -0.0, 3.0, 4.0], pa.float64()),
                        "n": _i(111, 222, 333, 444, 555)}),
        "s": pa.table({"k": pa.array(["a", "b", "b", None], pa.string()), "v": _i(1, 2, 3, 4)}),
        "r": pa.table({"k": pa.array(["a", "b", "c"], pa.string()), "v": _i(5, 6, 7)}),
    }


def _rows(tbl):
    return list(zip(*[tbl.column(c).to_pylist() for c in tbl.column_names]))


def _run(con, q):
    res = con.sql(q).arrow()
    return res.read_all() if isinstance(res, pa.RecordBatchReader) else res


def _renamed(names):
    """A repeated output name becomes `k_1`, `k_2`: what `duckdb.sql(q)` over Python Arrow variables and a
    subquery return; a native table or `con.register` view keeps `k` twice at the top level."""
    seen, out = set(), []
    for n in names:
        m, i = n, 0
        while m in seen:
            i += 1
            m = f"{n}_{i}"
        seen.add(m)
        out.append(m)
    return out


def _expect(q, tables):
    """Values and types as DuckDB gives them for native tables."""
    con = duckdb.connect()
    for name, tbl in tables.items():
        con.register(name + "_arrow", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
    res = _run(con, q)
    return res.rename_columns(_renamed(res.column_names))


def _check(q, device, typed=True, ordered=False):
    tables = _tables()
    exp = _expect(q, tables)
    got = mx.sql(q, **tables).compute(device=device)
    assert got.column_names == exp.column_names, q
    if typed:
        assert got.schema.types == exp.schema.types, q
    if ordered:
        assert _rows(got) == _rows(exp), q
    else:
        assert sorted(_rows(got), key=repr) == sorted(_rows(exp), key=repr), q


# F-102: SELECT * keeps both key columns; USING / NATURAL keep one
STARS = [
    "SELECT * FROM t JOIN u ON t.k = u.k",
    "SELECT * FROM t LEFT JOIN u ON t.k = u.k",
    "SELECT * FROM t JOIN w ON t.k = w.x",
    "SELECT * FROM t LEFT JOIN w ON t.k = w.x",
    "SELECT * FROM t JOIN w ON w.x = t.k",
    "SELECT * FROM t JOIN u USING (k)",
    "SELECT * FROM t LEFT JOIN u USING (k)",
    "SELECT * FROM t NATURAL JOIN w2",
    "SELECT * FROM t, w WHERE t.k = w.x",
    "SELECT * FROM t, u WHERE t.k = u.k",
    "SELECT * FROM t, u WHERE t.k = u.k AND t.i > 1",
    "SELECT * FROM t JOIN fl ON t.k = fl.x",
    "SELECT * FROM t LEFT JOIN fl ON t.k = fl.x",
    "SELECT * FROM t, fl WHERE t.k = fl.x",
    "SELECT * FROM s JOIN r ON s.k = r.k",
    "SELECT * FROM s LEFT JOIN r ON s.k = r.k",
    "SELECT * FROM (SELECT * FROM t JOIN u ON t.k = u.k)",
    "SELECT * FROM (SELECT * FROM t LEFT JOIN w ON t.k = w.x) WHERE x IS NULL",
    "SELECT * FROM t JOIN u ON t.k = u.k JOIN w ON u.k = w.x",
    "SELECT * FROM t JOIN u ON t.k = u.k JOIN u u2 ON u2.k = t.k",
    "SELECT * FROM t LEFT JOIN u ON t.k = u.k WHERE u.k IS NULL",
]
# the right key by name, and the type each key keeps
KEYS = [
    "SELECT w.x FROM t JOIN w ON t.k = w.x",
    "SELECT fl.x FROM t JOIN fl ON t.k = fl.x",
    "SELECT fl.x, t.k FROM t JOIN fl ON t.k = fl.x",
    "SELECT t.k, fl.x FROM t, fl WHERE t.k = fl.x",
    "SELECT fl.x FROM t, fl WHERE t.k = fl.x",
    "SELECT u.k, t.k FROM t JOIN u ON t.k = u.k",
    "SELECT u.k, t.k FROM t LEFT JOIN u ON t.k = u.k",
    "SELECT w.x, t.k FROM t LEFT JOIN w ON t.k = w.x",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", STARS + KEYS)
def test_star_and_keys(q, device):
    _check(q, device)


@pytest.mark.parametrize("q,expected", [
    ("SELECT * FROM t JOIN u ON t.k = u.k ORDER BY t.i", ["i", "k", "v", "k_1", "v_1", "w"]),
    ("SELECT * FROM t JOIN u USING (k) ORDER BY t.i", ["i", "k", "v", "v_1", "w"]),
    ("SELECT * FROM t JOIN w ON t.k = w.x ORDER BY t.i", ["i", "k", "v", "x", "n"]),
])
def test_star_column_names(q, expected):
    assert mx.sql(q, **_tables()).compute().column_names == expected


def test_left_join_star_right_key_is_null_where_unmatched():
    got = mx.sql("SELECT * FROM t LEFT JOIN w ON t.k = w.x ORDER BY i", **_tables()).compute()
    assert got.column("x").to_pylist() == [1, 2, 2, 3, None, None, None]
    assert got.column("n").to_pylist() == [11, 22, 22, 33, None, None, None]


def test_float_key_keeps_its_own_type_and_values():
    got = mx.sql("SELECT t.k, fl.x FROM t JOIN fl ON t.k = fl.x ORDER BY t.i", **_tables()).compute()
    assert got.schema.types == [pa.int64(), pa.float64()]
    assert got.column("x").to_pylist() == [1.0, 2.0, 2.0, 3.0]


# ORDER BY a right key (a copy) or a renamed column next to a lone star
ORDERED = [
    "SELECT * FROM t JOIN u ON t.k = u.k ORDER BY u.k, t.i",
    "SELECT * FROM t JOIN u ON t.k = u.k ORDER BY u.v DESC, t.i LIMIT 3",
    "SELECT * FROM t LEFT JOIN w ON t.k = w.x ORDER BY w.x, t.i",
    "SELECT * FROM t LEFT JOIN w ON t.k = w.x ORDER BY w.x DESC, t.i",
    "SELECT * FROM t, u WHERE t.k = u.k ORDER BY u.v, t.i",
    "SELECT * FROM t, fl WHERE t.k = fl.x ORDER BY fl.x, t.i",
    "SELECT * FROM t JOIN u USING (k) ORDER BY u.v, t.i",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", ORDERED)
def test_star_order_by_right_columns(q, device):
    _check(q, device, ordered=True)
