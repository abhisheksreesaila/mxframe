"""F-105: SQL CASE takes the common type of its branches and gives NULL when no ELSE matches.

Before: a missing ELSE was 0 (`count(case when p then 1 end)` counted every row, `avg` averaged the zeros), and
every result was forced to float, so a string branch raised "Failed to parse string 'hit' as double" in SELECT, in
a GROUP BY key, in WHERE and in ORDER BY; a date branch raised too; `CASE x WHEN v` ignored `x`; `THEN NULL`,
`THEN TRUE` and `THEN DATE '...'` were unsupported.

DuckDB's typing (probed): equal types stay; integers widen; an integer next to a DECIMAL / DOUBLE / FLOAT column is
fractional (we return double for all of them); text is text; dates are dates; a NULL branch takes the other
branch's type; every branch NULL is INTEGER; text mixed with numbers is an error. A NULL condition is not true.

Expected answers come from a NATIVE DuckDB table, on cpu and gpu, native executor on and off. The native
executor must keep answering `sum(case when .. then x else 0 end)` (TPC-H Q8, Q12, Q14), and must hand a CASE
that can be NULL to Arrow.
"""
import datetime as dt
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import gpu_stream

GPU = gpu_stream.cuda_available()
D0 = dt.date(2024, 1, 1)


def _table():
    n = 12
    return pa.table({
        "id": pa.array(range(n), pa.int64()),
        "p": pa.array([None if k % 5 == 4 else k % 3 for k in range(n)], pa.int64()),          # NULL on 4 and 9
        "i": pa.array([k * 2 for k in range(n)], pa.int64()),
        "j": pa.array([None if k % 4 == 3 else k for k in range(n)], pa.int32()),             # NULL on 3, 7, 11
        "f": pa.array([k * 1.5 for k in range(n)], pa.float64()),
        "r": pa.array([k * 0.5 for k in range(n)], pa.float32()),
        "s": pa.array(["hit", "miss", None, "hit", "other", "hit", "miss", "hit", "other", "miss", "hit", "hit"]),
        "d": pa.array([D0 + dt.timedelta(days=k * 40) for k in range(n)], pa.date32()),
        "g": pa.array([k % 3 for k in range(n)], pa.int32()),
        "k": pa.array(["x", "y", "z"] * 4),                                                    # no NULLs
    })


T = _table()
CON = duckdb.connect()
CON.register("_t", T)
CON.execute("create table t as select * from _t")


@pytest.fixture(params=[(dev, nat) for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-native{p[1]}")
def cfg(request, monkeypatch):
    dev, nat = request.param
    if dev == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    if nat == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", nat)
    return dev, nat == "1"


def _norm(v):
    return float(v) if isinstance(v, Decimal) else v


def _run(q, dev):
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T).compute(device=dev)
    rows = lambda x: [tuple(_norm(c[r]) for c in (x.column(i).to_pylist() for i in range(x.num_columns)))
                      for r in range(x.num_rows)]
    return rows(got), rows(want), got, want


def _kind(t):
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    return "date" if pa.types.is_date(t) else "bool" if pa.types.is_boolean(t) else str(t)


# (query, kinds of the result columns). ORDER BY id / the key makes the order the same as DuckDB's.
SELECT = {
    "no_else_int": ("select id, case when p = 1 then 1 end as c from t order by id", ["int", "int"]),
    "no_else_string": ("select id, case when p = 1 then 'hit' end as c from t order by id", ["int", "string"]),
    "no_else_date": ("select id, case when p = 1 then d end as c from t order by id", ["int", "date"]),
    "no_else_float_col": ("select id, case when p = 1 then f end as c from t order by id", ["int", "float"]),
    "two_whens_no_else": ("select id, case when p = 1 then 10 when p = 2 then 20 end as c from t order by id",
                          ["int", "int"]),
    "null_condition_falls_to_else": ("select id, case when p = 1 then 'one' else 'rest' end as c from t order by id",
                                     ["int", "string"]),
    "string_both": ("select id, case when p = 1 then s else 'none' end as c from t order by id", ["int", "string"]),
    "string_col_else_col": ("select id, case when p = 0 then s when p = 1 then 'b' else s end as c from t order by id",
                            ["int", "string"]),
    "date_col_literal": ("select id, case when p = 1 then d else date '2020-02-29' end as c from t order by id",
                         ["int", "date"]),
    "date_literals": ("select id, case when p = 1 then date '2020-01-02' else date '2021-03-04' end as c "
                      "from t order by id", ["int", "date"]),
    "int_int": ("select id, case when p = 1 then 1 else 2 end as c from t order by id", ["int", "int"]),
    "int_cols": ("select id, case when p = 1 then i else j end as c from t order by id", ["int", "int"]),
    "int_big": ("select id, case when p = 1 then 3000000000 else 1 end as c from t order by id", ["int", "int"]),
    "int_decimal": ("select id, case when p = 1 then 1 else 2.5 end as c from t order by id", ["int", "float"]),
    "int_double": ("select id, case when p = 1 then 1 else 2.5e0 end as c from t order by id", ["int", "float"]),
    "int_float_col": ("select id, case when p = 1 then 1 else f end as c from t order by id", ["int", "float"]),
    "float32_col_int": ("select id, case when p = 1 then r else 0 end as c from t order by id", ["int", "float"]),
    "null_then": ("select id, case when p = 1 then null else 7 end as c from t order by id", ["int", "int"]),
    "null_else_string": ("select id, case when p = 1 then 'a' else null end as c from t order by id",
                         ["int", "string"]),
    "null_else_date": ("select id, case when p = 1 then d else null end as c from t order by id", ["int", "date"]),
    "all_null": ("select id, case when p = 1 then null else null end as c from t order by id", ["int", "int"]),
    "all_null_no_else": ("select id, case when p = 1 then null end as c from t order by id", ["int", "int"]),
    "true_branch": ("select id, case when p = 1 then true end as c from t order by id", ["int", "bool"]),
    "simple_case_string": ("select id, case g when 0 then 'zero' when 1 then 'one' end as c from t order by id",
                           ["int", "string"]),
    "simple_case_else": ("select id, case s when 'hit' then 1 else 0 end as c from t order by id", ["int", "int"]),
    "nested": ("select id, case when p = 1 then case when i > 10 then 'big' end else 'no' end as c "
               "from t order by id", ["int", "string"]),
    "arith_on_case": ("select id, case when p = 1 then i end + 1 as c from t order by id", ["int", "int"]),
}


@pytest.mark.parametrize("name", SELECT)
def test_select_equals_duckdb(name, cfg):
    q, kinds = SELECT[name]
    dev, _ = cfg
    got, want, tbl, _ = _run(q, dev)
    assert got == want
    assert [_kind(f.type) for f in tbl.schema] == kinds, [str(f.type) for f in tbl.schema]


AGG = {
    "count_no_else": "select count(case when p = 1 then 1 end) as c from t",
    "count_no_else_star_same": "select count(*) as a, count(case when p = 1 then 1 end) as c from t",
    "count_all_null_cond": "select count(case when p = 7 then 1 end) as c from t",
    "sum_no_else": "select sum(case when p = 1 then i end) as c from t",
    "sum_no_else_none_match": "select sum(case when p = 7 then i end) as c from t",
    "sum_else_zero_none_match": "select sum(case when p = 7 then i else 0 end) as c from t",
    "avg_no_else": "select avg(case when p = 1 then f end) as c from t",
    "avg_else_zero": "select avg(case when p = 1 then f else 0 end) as c from t",
    "min_max_no_else": "select min(case when p = 1 then i end) as a, max(case when p = 1 then i end) as b from t",
    "count_string_no_else": "select count(case when p = 1 then s end) as c from t",
    "sum_else_zero": "select sum(case when p = 1 then f else 0 end) as c from t",
    "sum_else_zero_int": "select sum(case when p = 1 then i else 0 end) as c from t",
    "sum_ones_zero": "select sum(case when s = 'hit' then 1 else 0 end) as c from t",
    "grouped_count_no_else": "select g, count(case when p = 1 then 1 end) as c from t group by g order by g",
    "grouped_sum_no_else": "select g, sum(case when p <> 0 then i end) as c from t group by g order by g",
    "grouped_avg_no_else": "select g, avg(case when p <> 0 then f end) as c from t group by g order by g",
    "grouped_sum_else_zero": "select g, sum(case when p <> 0 then f else 0 end) as c from t group by g order by g",
    "grouped_ratio": "select g, 100.0 * sum(case when k = 'x' then f else 0 end) / sum(f) as c "
                     "from t group by g order by g",
    "having_no_else": "select g, count(case when p = 1 then 1 end) as c from t group by g "
                      "having count(case when p = 1 then 1 end) > 0 order by g",
}


@pytest.mark.parametrize("name", AGG)
def test_aggregate_equals_duckdb(name, cfg):
    dev, _ = cfg
    got, want, _, _ = _run(AGG[name], dev)
    assert got == want


KEY = {
    "string_key": "select case when i > 10 then 'big' else 'small' end as k, count(*) as n, sum(f) as s from t "
                  "group by 1 order by 1",
    "string_key_no_else": "select case when i > 10 then 'big' end as k, count(*) as n from t group by 1 order by 1",
    "string_key_by_expr": "select case when i > 10 then 'big' end as k, count(*) as n from t "
                          "group by case when i > 10 then 'big' end order by k",
    "int_key_no_else": "select case when p = 1 then 1 end as k, sum(i) as s from t group by 1 order by 1",
    "column_and_case_key": "select g, case when i > 10 then s else 'low' end as k, count(*) as n from t "
                           "group by g, 2 order by g, k",
    "date_key": "select case when p = 1 then d end as k, count(*) as n from t group by 1 order by 1",
    "order_by_case": "select id from t order by case when p = 1 then 0 end, id",
    "order_by_case_desc": "select id from t order by case when p = 1 then 0 end desc nulls last, id",
    "order_by_string_case": "select id from t order by case when s = 'hit' then 'a' else 'b' end, id",
    "where_string": "select id from t where case when p = 1 then 'y' end = 'y' order by id",
    "where_no_else_int": "select id from t where case when p = 1 then 1 end = 1 order by id",
    "where_no_else_not": "select id from t where case when p = 1 then 1 end is null order by id",
    "where_bool": "select id from t where case when p = 1 then true end order by id",
    "where_date": "select id from t where case when p = 1 then d else date '2000-01-01' end > date '2024-06-01' "
                  "order by id",
    "where_string_vs_col": "select id from t where case when p = 1 then s else 'none' end = 'hit' order by id",
}


@pytest.mark.parametrize("name", KEY)
def test_key_order_where_equals_duckdb(name, cfg):
    dev, _ = cfg
    got, want, _, _ = _run(KEY[name], dev)
    assert got == want


def test_text_mixed_with_numbers_raises(cfg):
    with pytest.raises(Exception):
        mx.sql("select case when p = 1 then 'a' else 1 end as c from t", t=T).compute(device=cfg[0])


# -- the native executor keeps the classic shape, and hands a possible NULL to Arrow ---------------------------

NATIVE = ["select sum(case when k = 'x' then f else 0 end) as c from t",
          "select g, sum(case when k = 'x' then f else 0 end) as c, sum(f) as d from t group by g order by g",
          "select g, sum(case when g = 1 then 1 else 0 end) as c from t group by g order by g"]


@pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")
@pytest.mark.parametrize("q", NATIVE)
def test_else_zero_sum_stays_native(q, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    got, want, _, _ = _run(q, "cpu")
    assert nx.last_path() == "native", nx.last_decline()
    assert got == want


@pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")
@pytest.mark.parametrize("q", ["select count(case when g = 1 then 1 end) as c from t",
                               "select g, avg(case when i > 8 then f end) as c from t group by g order by g",
                               "select case when i > 10 then 'big' end as k, count(*) as n from t group by 1 order by 1"])
def test_possible_null_is_not_answered_natively(q, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    monkeypatch.setitem(nx._last, "path", None)             # last_path keeps the previous native run
    got, want, _, _ = _run(q, "cpu")
    assert nx.last_path() != "native"
    assert got == want


# -- the DataFrame API: when() without otherwise is NULL ------------------------------------------------------

def test_when_without_otherwise_is_null(cfg):
    from mxframe import LazyFrame, col, lit
    from mxframe.lazy_expr import when
    dev, _ = cfg
    out = LazyFrame(T).select(col("id"), when(col("p") == lit(1), lit("one")).alias("c")).sort(col("id")).compute(device=dev)
    want = CON.execute("select id, case when p = 1 then 'one' end as c from t order by id").to_arrow_table()
    assert out.column("c").to_pylist() == want.column("c").to_pylist()
