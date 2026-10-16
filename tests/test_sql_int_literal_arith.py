"""F-103: in the SQL front end an integer column plus / minus / times an integer literal (or another integer
column) stays an exact integer, like DuckDB. It used to promote the literal to float, so `i + 1` on 2^53 + 1
rounded, integer results came back as doubles, and a global aggregate over them rounded too. `/` stays float,
and an operand that is a float (or of unknown type) still promotes the literal. Every query runs with the
native executor on and off and equals DuckDB."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx

B = (1 << 53) + 1                      # the first integer float64 cannot hold
N = 300


def _table():
    k = np.arange(N)
    return pa.table({
        "i": pa.array(B + k, pa.int64()),
        "s": pa.array(k % 7, pa.int32()),
        "g": pa.array(k % 3, pa.int64()),
        "f": pa.array(k * 0.5, pa.float64()),
        "n": pa.array([None if x % 10 == 0 else B + x for x in k], pa.int64()),
    })


T = _table()


@pytest.fixture(params=["1", "0"], ids=["native", "arrow"])
def mode(request, monkeypatch):
    if request.param == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", request.param)
    return request.param


def _both(q):
    con = duckdb.connect()
    con.register("t", T)
    cur = con.execute(q)
    want = cur.fetchall()
    got = mx.sql(q, t=T).compute()
    return [tuple(r.values()) for r in got.to_pylist()], want, got


QUERIES = {
    "select_add": "select i + 1 as x from t order by i",
    "select_sub": "select i - 1 as x from t order by i",
    "select_mul": "select i * 3 as x from t order by i",
    "literal_first": "select 1 + i as x from t order by i",
    "neg_literal": "select i + -1 as x from t order by i",
    "col_col": "select i + s as x from t order by i",
    "int32_literal": "select s + 1 as x from t order by i",
    "nested": "select (i + 1) - 2 + g * 2 as x from t order by i",
    "nulls": "select n + 1 as x from t order by i",
    "where_eq": "select i from t where i + 1 = 9007199254741000 order by i",
    "where_range": "select i - 1 as x from t where i * 1 > 9007199254741200 order by i",
    "group_by_expr": "select i + 1 as k, count(*) as c from t group by i + 1 order by k",
    "group_by_key": "select g, i + 1 as k, count(*) as c from t where s = 0 group by g, i + 1 order by k",
    "order_by_expr": "select i from t order by i + 1 desc limit 5",
    "subquery_from": "select q + 1 as x from (select i + 1 as q from t) u where q + 1 > 9007199254741200 order by x",
    "subquery_in": "select i from t where i in (select i + 1 from t where s = 1) order by i",
    "cte": "with c as (select i + 1 as q, g from t) select q * 2 as x from c where g = 1 order by q limit 7",
    "sum_expr": "select sum(i + 1) as x from t where g = 1",
    "sum_grouped": "select g, sum(i - 1) as x from t group by g order by g",
    "min_max_expr": "select g, min(s + 1) as lo, max(s * 2 + 1) as hi from t group by g order by g",
    "sum_small": "select sum(s + 1) as x, count(*) as c from t",
    "division_is_float": "select i / 2 as x from t order by i limit 3",
    "float_operand": "select f + 1 as x, 1 - f as y from t order by i limit 4",
    "int_times_float": "select s * 0.5 as x, s + 1.5 as y from t order by i limit 4",
}


@pytest.mark.parametrize("name", QUERIES)
def test_equals_duckdb(name, mode):
    got, want, _ = _both(QUERIES[name])
    assert got == want


@pytest.mark.parametrize("name", ["select_add", "select_sub", "select_mul", "literal_first", "col_col",
                                  "int32_literal", "nested", "nulls", "group_by_expr", "subquery_from"])
def test_result_stays_an_integer_type(name, mode):
    _, _, tbl = _both(QUERIES[name])
    assert all(pa.types.is_integer(c.type) for c in tbl.columns), [str(c.type) for c in tbl.columns]


def test_division_and_float_operands_stay_double(mode):
    for name in ("division_is_float", "float_operand", "int_times_float"):
        _, _, tbl = _both(QUERIES[name])
        assert all(c.type == pa.float64() for c in tbl.columns), (name, [str(c.type) for c in tbl.columns])


def test_the_first_value_past_2_53_is_exact(mode):
    got, _, _ = _both("select i + 1 as x from t order by i limit 1")
    assert got == [(B + 1,)] == [(9007199254740994,)]


def test_overflow_raises_like_duckdb(mode):
    t = pa.table({"i": pa.array([1 << 62, 5], pa.int64())})
    with pytest.raises(Exception, match="verflow"):
        mx.sql("select i * 4 as x from t", t=t).compute()
    with pytest.raises(duckdb.Error):
        duckdb.connect().register("t", t).execute("select i * 4 as x from t")
