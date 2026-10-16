"""F-104: a later JOIN's ON that names a previous LEFT JOIN's right table (``u.k``, ``u.v``) sees NULL where that
join matched nothing, so nothing matches. Expected answers come from DuckDB native tables. Unmatched rows at
every step: t.k 5..7 are not in u, u.k 3 is not in x, x.k 2 is not in y."""
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
        "t": pa.table({"id": _i(1, 2, 3, 4, 5, 6, 7, 8), "k": _i(0, 1, 2, 3, 5, 6, 7, None)}),
        "u": pa.table({"k": _i(0, 1, 2, 3, 4), "v": _i(10, 20, 30, 40, 50)}),
        "x": pa.table({"k": _i(0, 1, 2, 5, 6, 7), "w": _i(10, 20, 30, 50, 60, 70),
                       "z": _i(100, 200, 300, 500, 600, 700)}),
        "y": pa.table({"k": _i(0, 1, 5, 6), "q": _i(1000, 2000, 5000, 6000)}),
        # a non-key column v in three tables (renamed v_right, v_right2 by the join)
        "a": pa.table({"id": _i(1, 2, 3, 4, 5), "v": _i(1, 2, 3, 4, None)}),
        "b": pa.table({"bk": _i(1, 2, 3, 9), "v": _i(1, 2, 7, 9)}),
        "d": pa.table({"dk": _i(1, 2, 4, 9), "v": _i(1, 3, 4, 9), "dz": _i(111, 222, 444, 999)}),
        "c": pa.table({"ck": _i(1, 3, 4, 9), "v": _i(1, 3, 4, 9), "cz": _i(11, 33, 44, 99)}),
    }


def _norm(tbl):
    cols = tbl.column_names
    return cols, sorted(zip(*[tbl.column(c).to_pylist() for c in cols]), key=repr)


def _check(q, device):
    tables = _tables()
    con = duckdb.connect()
    for name, tbl in tables.items():
        con.register(name + "_arrow", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
    exp = con.execute(q).to_arrow_table()
    got = mx.sql(q, **tables).compute(device=device)
    assert _norm(got) == _norm(exp)
    return exp.num_rows


QUERIES = [
    # the ticket: x joins on u.k, which is NULL for a t.k with no match in u
    "SELECT t.id, u.k, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k",
    "SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON x.k = u.k",
    "SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k WHERE x.z IS NULL",
    "SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k WHERE x.z IS NOT NULL",
    # the later ON on t.k still matches the unmatched rows
    "SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON t.k = x.k",
    # a non-key column of the previous right table
    "SELECT t.id, u.v, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.v = x.w",
    "SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON x.w = u.v",
    # an INNER JOIN after a LEFT JOIN drops the unmatched rows
    "SELECT t.id, u.k, x.z FROM t LEFT JOIN u ON t.k = u.k JOIN x ON u.k = x.k",
    "SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k INNER JOIN x ON u.k = x.k",
    "SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k JOIN x ON u.v = x.w",
    # an INNER JOIN, then a LEFT JOIN on the inner table's key
    "SELECT t.id, u.v, x.z FROM t JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k",
    # three chained LEFT JOINs: each on the previous right key
    "SELECT t.id, u.k AS uk, x.k AS xk, y.q FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k LEFT JOIN y ON x.k = y.k",
    "SELECT t.id, y.q FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k LEFT JOIN y ON x.k = y.k",
    "SELECT t.id, y.q FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON x.k = u.k LEFT JOIN y ON y.k = u.k",
    "SELECT t.id, y.q FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON t.k = x.k LEFT JOIN y ON x.k = y.k",
    "SELECT t.id, y.q FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k JOIN y ON x.k = y.k",
    "SELECT t.id, u.v, x.w, y.q FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.v = x.w LEFT JOIN y ON x.k = y.k",
    # aggregates over the chain
    "SELECT u.k, COUNT(*) AS n, SUM(x.z) AS s FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k GROUP BY u.k",
    # USING forms: USING keeps the left key; u.k is still the NULL-on-unmatched copy
    "SELECT t.id, x.z FROM t LEFT JOIN u USING (k) LEFT JOIN x USING (k)",
    "SELECT t.id, x.z FROM t LEFT JOIN u USING (k) LEFT JOIN x ON u.k = x.k",
    "SELECT t.id, y.q FROM t LEFT JOIN u USING (k) LEFT JOIN x ON u.k = x.k LEFT JOIN y ON x.k = y.k",
    "SELECT t.id, x.z FROM t JOIN u USING (k) LEFT JOIN x USING (k)",
    # the same non-key name in three tables, both operand orders (renamed v_right / v_right2)
    "SELECT a.id, b.bk, c.cz FROM a LEFT JOIN b ON a.id = b.bk LEFT JOIN c ON b.v = c.v",
    "SELECT a.id, b.bk, c.cz FROM a LEFT JOIN b ON a.id = b.bk LEFT JOIN c ON c.v = b.v",
    "SELECT a.id, c.cz FROM a LEFT JOIN b ON a.id = b.bk LEFT JOIN c ON a.v = c.v",
    "SELECT a.id, c.cz FROM a LEFT JOIN b ON a.id = b.bk LEFT JOIN c ON c.v = a.v",
    "SELECT a.id, c.cz FROM a LEFT JOIN b ON a.id = b.bk JOIN c ON c.v = b.v",
    "SELECT a.id, c.cz FROM a JOIN b ON a.id = b.bk LEFT JOIN c ON c.v = b.v AND c.ck = a.id",
    "SELECT a.id, b.bk, c.cz, d.dz FROM a LEFT JOIN b ON a.id = b.bk LEFT JOIN c ON c.v = b.v "
    "LEFT JOIN d ON d.v = c.v",
    "SELECT a.id, b.bk, c.cz, d.dz FROM a LEFT JOIN b ON a.id = b.bk LEFT JOIN c ON c.v = b.v "
    "LEFT JOIN d ON b.v = d.v",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", QUERIES)
def test_later_on_sees_null_where_the_previous_left_join_missed(q, device):
    assert _check(q, device) > 0


def test_ticket_repro_values():
    """DuckDB gives z = NULL for every t.k with no match in u; mx used to join x (it matched x on t.k, giving 500, 600, 700 for t.k 5..7)."""
    got = mx.sql("SELECT t.id, x.z FROM t LEFT JOIN u ON t.k = u.k LEFT JOIN x ON u.k = x.k ORDER BY t.id",
                 **_tables()).compute(device="cpu")
    assert got.column("z").to_pylist() == [100, 200, 300, None, None, None, None, None]
