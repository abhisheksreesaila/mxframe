"""F-109: a JOIN ... ON equality is resolved by the table qualifier of each side, not by the column name.
``x.v = t.k`` where both tables have ``k`` and ``v`` joined ``t.v = x.k`` (a silent wrong answer). Expected
answers come from DuckDB native tables; NULLs in the keys, duplicates, aliases, three tables, LEFT and INNER."""
import duckdb
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import pipeline_kernels as pk

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]


def _i(*v):
    return pa.array(list(v), pa.int64())


def _tables():
    return {
        # k and v exist in all three tables: t.k = x.v and t.v = x.k pair different rows
        "t": pa.table({"id": _i(1, 2, 3, 4, 5, 6), "k": _i(1, 2, 3, 4, 5, None), "v": _i(1, 1, 2, 9, 5, 3)}),
        "x": pa.table({"z": _i(10, 20, 30, 40, 50, 60), "k": _i(2, 1, 3, 4, 9, None), "v": _i(1, 2, 5, 8, 3, None)}),
        "y": pa.table({"w": _i(100, 200, 300, 400, 500), "k": _i(1, 2, 4, 7, 5), "v": _i(2, 5, 9, 3, 1)}),
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


QUERIES = [
    # the ticket: x.v = t.k joins t.k to x.v, not t.v to x.k
    "SELECT t.id, x.z FROM t JOIN x ON x.v = t.k",
    "SELECT t.id, x.z FROM t JOIN x ON t.k = x.v",
    "SELECT t.id, x.z FROM t JOIN x ON x.k = t.v",
    "SELECT t.id, x.z FROM t JOIN x ON t.v = x.k",
    "SELECT t.id, x.z FROM t JOIN x ON t.k = x.k",
    "SELECT t.id, x.z FROM t JOIN x ON x.v = t.v",
    # both columns of both tables in the output
    "SELECT t.id, t.k AS tk, t.v AS tv, x.k AS xk, x.v AS xv, x.z FROM t JOIN x ON x.v = t.k",
    "SELECT t.id, t.k AS tk, t.v AS tv, x.k AS xk, x.v AS xv, x.z FROM t JOIN x ON t.v = x.k",
    # two equalities, in either operand order
    "SELECT t.id, x.z FROM t JOIN x ON x.v = t.k AND x.k = t.v",
    "SELECT t.id, x.z FROM t JOIN x ON t.k = x.v AND t.v = x.k",
    "SELECT t.id, x.z FROM t JOIN x ON x.k = t.k AND x.v = t.v",
    "SELECT t.id, x.z FROM t JOIN x ON t.k = x.v AND x.k = t.k",
    # LEFT JOIN
    "SELECT t.id, x.z FROM t LEFT JOIN x ON x.v = t.k",
    "SELECT t.id, x.z FROM t LEFT JOIN x ON t.k = x.v",
    "SELECT t.id, x.z FROM t LEFT JOIN x ON t.v = x.k",
    "SELECT t.id, x.k AS xk, x.v AS xv, x.z FROM t LEFT JOIN x ON x.v = t.k",
    "SELECT t.id, x.z FROM t LEFT JOIN x ON x.v = t.k AND x.k = t.v",
    "SELECT t.id, x.z FROM t LEFT JOIN x ON x.v = t.k WHERE x.z IS NULL",
    # aliases, with the alias in either place
    "SELECT a.id, b.z FROM t a JOIN x b ON b.v = a.k",
    "SELECT a.id, b.z FROM t a JOIN x b ON a.k = b.v",
    "SELECT a.id, b.z FROM t AS a LEFT JOIN x AS b ON b.k = a.v",
    "SELECT a.id, b.z FROM t a JOIN x b ON b.v = a.k AND b.k = a.v",
    # an aggregate over the join
    "SELECT t.k, COUNT(*) AS n, SUM(x.z) AS s FROM t JOIN x ON x.v = t.k GROUP BY t.k",
    # three tables
    "SELECT t.id, x.z, y.w FROM t JOIN x ON x.v = t.k JOIN y ON y.k = x.k",
    "SELECT t.id, x.z, y.w FROM t JOIN x ON x.v = t.k JOIN y ON y.v = x.v",
    "SELECT t.id, x.z, y.w FROM t JOIN x ON x.v = t.k JOIN y ON y.v = x.k",
    "SELECT t.id, x.z, y.w FROM t JOIN x ON x.k = t.v JOIN y ON x.v = y.k",
    "SELECT t.id, x.z, y.w FROM t JOIN x ON x.v = t.k JOIN y ON y.k = t.v",
    "SELECT t.id, x.z, y.w FROM t JOIN x ON x.v = t.k JOIN y ON y.v = t.k",
    "SELECT t.id, x.z, y.w FROM t JOIN x ON t.v = x.k JOIN y ON t.k = y.v",
    "SELECT t.id, x.z, y.w FROM t LEFT JOIN x ON x.v = t.k LEFT JOIN y ON y.k = x.k",
    "SELECT t.id, x.z, y.w FROM t LEFT JOIN x ON x.v = t.k JOIN y ON y.v = t.k",
    "SELECT t.id, x.z, y.w FROM t JOIN x ON x.v = t.k LEFT JOIN y ON y.v = x.v",
    "SELECT a.id, b.z, c.w FROM t a JOIN x b ON b.v = a.k JOIN y c ON c.k = b.k",
    # USING keeps working
    "SELECT t.id, x.z FROM t JOIN x USING (k)",
    "SELECT t.id, x.z, y.w FROM t JOIN x USING (k) JOIN y ON y.v = x.v",
    # the second copy of a table
    "SELECT p.id, q.id AS qid FROM t p JOIN t q ON q.v = p.k",
    "SELECT p.id, q.id AS qid FROM t p JOIN t q ON p.k = q.v",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", QUERIES)
def test_on_equality_follows_the_table_qualifier(q, device):
    assert _check(q, device) >= 0


def test_ticket_repro_values():
    """t.k 1, 2, 3, 5 meet x.v 1, 2, 3, 5; mx used to join t.v = x.k (ids 1, 2, 3, 4 with other z)."""
    q = "SELECT t.id, x.z FROM t JOIN x ON x.v = t.k ORDER BY t.id"
    exp = _expected(q)
    got = mx.sql(q, **_tables()).compute(device="cpu")
    assert got.to_pylist() == exp.to_pylist() == [{"id": 1, "z": 10}, {"id": 2, "z": 20}, {"id": 3, "z": 50},
                                                       {"id": 5, "z": 30}]


def test_unqualified_name_in_both_tables_is_ambiguous():
    for q in ("SELECT t.id FROM t JOIN x ON k = t.k", "SELECT t.id FROM t JOIN x ON v = x.k"):
        with pytest.raises(Exception):
            _expected(q)                                     # DuckDB: Ambiguous reference
        with pytest.raises(ValueError, match="ambiguous"):
            mx.sql(q, **_tables()).compute()


def test_unqualified_name_in_one_table_still_works():
    q = "SELECT t.id, x.z FROM t JOIN x ON z = t.id"
    assert _check(q, "cpu") >= 0


@pytest.mark.parametrize("device", DEVICES)
def test_select_star_keeps_both_key_columns(device):
    q = "SELECT * FROM t JOIN x ON x.v = t.k"
    exp = _expected(q)
    got = mx.sql(q, **_tables()).compute(device=device)
    assert got.column_names == ["id", "k", "v", "z", "k_1", "v_1"]          # mx renames a repeated name
    rows = lambda t: sorted(zip(*[c.to_pylist() for c in t.columns]), key=repr)
    assert rows(got) == rows(exp)
