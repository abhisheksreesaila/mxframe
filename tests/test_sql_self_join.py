"""F-099: a column name that two joined tables share (a self-join `t x JOIN t y`, `u.v` next to `t.v`): `x.v`
and `y.v` resolve to the right copy in every clause (SELECT, WHERE, GROUP BY, HAVING, ORDER BY, a later ON), and
the result names them as DuckDB does (`v`, `v_1`). Expected answers come from DuckDB native tables."""
import pyarrow.parquet as pq
import pytest

import omxcore as mx
from test_sql_select_star_join import DEVICES, _check, _expect, _rows, _tables


# F-099: a self-join, and the same name in several tables
SELF = [
    "SELECT u.*, t.i FROM t JOIN u ON t.k = u.k",
    "SELECT t.*, u.* FROM t LEFT JOIN u ON t.k = u.k",
    "SELECT * FROM t JOIN u ON t.k = u.k WHERE u.v > 100",
    "SELECT x.i, y.v FROM t x JOIN t y ON x.k = y.k",
    "SELECT x.i, x.v, y.v FROM t x JOIN t y ON x.k = y.k",
    "SELECT y.v, x.v FROM t x JOIN t y ON x.k = y.k",
    "SELECT x.i, y.v FROM t x JOIN t y ON x.k = y.k WHERE y.v > 20",
    "SELECT x.i, y.v FROM t x JOIN t y ON x.k = y.k WHERE x.v > 20",
    "SELECT x.i, y.v FROM t x JOIN t y ON x.k = y.k WHERE x.v > 20 AND y.v < 40",
    "SELECT x.i, y.i AS j FROM t x JOIN t y ON x.k = y.k WHERE x.v < y.v",
    "SELECT x.i, y.i AS j FROM t x JOIN t y ON x.k = y.k WHERE y.i <> x.i",
    "SELECT * FROM t x JOIN t y ON x.k = y.k",
    "SELECT * FROM t x LEFT JOIN t y ON x.k = y.k",
    "SELECT * FROM t x LEFT JOIN t y ON x.k = y.i",
    "SELECT x.i, y.v FROM t x JOIN t y ON x.k = y.i",
    "SELECT x.i, y.k, y.v FROM t x LEFT JOIN t y ON x.k = y.i",
    "SELECT x.i, y.v FROM t x LEFT JOIN t y ON x.k = y.i WHERE y.v IS NULL",
    "SELECT y.v, count(*) AS n FROM t x JOIN t y ON x.k = y.k GROUP BY y.v",
    "SELECT y.v AS yv, sum(x.v) AS s FROM t x JOIN t y ON x.k = y.k GROUP BY y.v",
    "SELECT x.k, sum(y.v) AS s, max(y.v) AS m FROM t x JOIN t y ON x.k = y.k GROUP BY x.k",
    "SELECT x.k, count(*) AS n FROM t x JOIN t y ON x.k = y.k GROUP BY x.k HAVING sum(y.v) > 30",
    "SELECT x.i, y.v FROM t x JOIN t y ON x.k = y.k ORDER BY y.v DESC, x.i",
    "SELECT x.i, y.i AS j FROM t x JOIN t y ON x.k = y.k ORDER BY y.i, x.i LIMIT 4",
    "SELECT x.i, y.v + 1 AS z FROM t x JOIN t y ON x.k = y.k ORDER BY y.v + 1, x.i",
    "SELECT x.i, y.i AS j, z.i AS l FROM t x JOIN t y ON x.k = y.k JOIN t z ON z.k = y.k",
    "SELECT x.v, y.v, z.v FROM t x JOIN t y ON x.k = y.k JOIN t z ON z.k = y.k",
    "SELECT x.i, z.v FROM t x JOIN t y ON x.k = y.k JOIN t z ON z.v = y.v",
    "SELECT * FROM t x JOIN t y ON x.k = y.k JOIN t z ON z.k = y.k",
    "SELECT t.v, u.v, u2.v FROM t JOIN u ON t.k = u.k JOIN u u2 ON u2.k = t.k",
    "SELECT t.i, u.v, w2.n FROM t JOIN u ON t.k = u.k JOIN w2 ON w2.k = u.k WHERE u.v > 100",
    "SELECT a.i, b.v FROM t a, t b WHERE a.k = b.k AND b.v > 20",
    "SELECT a.v, b.v FROM t a, t b WHERE a.k = b.k",
    "SELECT a.i, b.v FROM t a, t b WHERE a.k = b.k ORDER BY b.v, a.i",
    "SELECT b.v, count(*) AS n FROM t a, t b WHERE a.k = b.k GROUP BY b.v",
    "SELECT t.i, u.v FROM t JOIN u ON t.k = u.k WHERE u.v > 150 ORDER BY u.v, t.i",
    "SELECT u.v, count(*) AS n FROM t JOIN u ON t.k = u.k GROUP BY u.v",
    "SELECT t.v FROM t JOIN u ON t.k = u.k WHERE u.v > 150",
    "SELECT t.i FROM t LEFT JOIN u ON t.k = u.k WHERE u.v IS NULL",
    "SELECT x.i, y.v FROM t x JOIN (SELECT * FROM t WHERE v > 30) y ON x.k = y.k",
    "SELECT x.i FROM t x WHERE x.v IN (SELECT y.v FROM t y JOIN t z ON y.k = z.k WHERE z.v > 20)",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", SELF)
def test_shared_names(q, device):
    # aggregates: DuckDB widens sums, so only names and values are compared for them
    _check(q, device, typed="sum(" not in q, ordered="ORDER BY" in q)


def test_self_join_over_scan_parquet(tmp_path):
    tbl = _tables()["t"]
    path = str(tmp_path / "t.parquet")
    pq.write_table(tbl, path, row_group_size=3)
    for q in ("SELECT x.i, y.v FROM t x JOIN t y ON x.k = y.k WHERE y.v > 20",
              "SELECT y.v, count(*) AS n FROM t x JOIN t y ON x.k = y.k GROUP BY y.v",
              "SELECT * FROM t x JOIN t y ON x.k = y.k"):
        exp = _expect(q, {"t": tbl})
        got = mx.sql(q, t=mx.scan_parquet(path)).compute()
        assert got.column_names == exp.column_names, q
        assert sorted(_rows(got), key=repr) == sorted(_rows(exp), key=repr), q


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [
    "SELECT * FROM t x JOIN t y ON x.k = y.k ORDER BY x.i, y.i",
    "SELECT * FROM t x JOIN t y ON x.k = y.k ORDER BY y.v DESC, x.i",
    "SELECT * FROM t x LEFT JOIN t y ON x.k = y.i ORDER BY y.v, x.i",
    "SELECT * FROM t JOIN u ON t.k = u.k ORDER BY u.v, t.i",
])
def test_star_order_by_second_copy(q, device):
    _check(q, device, ordered=True)
