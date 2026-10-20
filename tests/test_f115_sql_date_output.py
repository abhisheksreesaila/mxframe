"""F-115: a DATE constant that reaches a SELECT output is a date32, not a float day number.

Before: ``select id, date '2024-01-01' as c from t`` returned 19723.0 (every ``DATE '..'`` became a day number
because the TPC-H Parquet files store dates as int32 day numbers; it was a date only next to a date32 column).

Rule (DuckDB probed): an output is a DATE; a comparison / BETWEEN against an integer day-number column keeps the day
number (the TPC-H convention), against a date32 column it is a date. ``DATE + INTERVAL`` as an output is a TIMESTAMP
(microseconds) in DuckDB, so it is one here (the ticket said date32; DuckDB's answer wins).

Expected answers come from a NATIVE DuckDB table, on cpu and gpu, native executor on and off.
"""
import datetime as dt

import duckdb
import pyarrow as pa
import pytest

import omxcore as mx
import omxcore.native_exec as nx
from omxcore import gpu_stream

GPU = gpu_stream.cuda_available()
D0 = dt.date(2024, 1, 1)
N = 8
EPOCH = dt.date(1970, 1, 1)


def _table():
    dates = [None if k == 5 else D0 + dt.timedelta(days=k * 40) for k in range(N)]
    return pa.table({
        "id": pa.array(range(N), pa.int64()),
        "g": pa.array([k % 2 for k in range(N)], pa.int32()),
        "d": pa.array(dates, pa.date32()),
        # the TPC-H layout: dates as int32 day numbers
        "n": pa.array([None if x is None else (x - EPOCH).days for x in dates], pa.int32()),
        "v": pa.array([float(k) for k in range(N)], pa.float64()),
    })


T = _table()
CON = duckdb.connect()
CON.register("_t", T)
CON.execute("create table t as select * from _t")
CON.execute("alter table t alter n set data type date using (date '1970-01-01' + n)")      # the day numbers, as the dates they are


@pytest.fixture(params=[(dev, nat) for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-native{p[1]}")
def cfg(request, monkeypatch):
    dev, nat = request.param
    if dev == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    if nat == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", nat)
    return dev


def _rows(x):
    cols = [x.column(i).to_pylist() for i in range(x.num_columns)]
    return [tuple(c[r] for c in cols) for r in range(x.num_rows)]


# (query, types of the result columns that are dates: the rest are checked by value only)
QUERIES = {
    "bare": "select id, date '2024-01-01' as c from t order by id",
    "bare_only": "select date '2024-01-01' as c from t order by id",
    "cast_form": "select id, cast('2024-01-01' as date) as c from t order by id",
    "two_dates": "select id, date '2024-01-01' as a, date '2025-06-30' as b from t order by id",
    "next_to_date_column": "select id, d, date '2024-01-01' as c from t order by id",
    "plus_interval_day": "select id, date '2024-01-01' + interval 30 day as c from t order by id",
    "minus_interval_month": "select id, date '2024-03-31' - interval 1 month as c from t order by id",
    "case_literal": "select id, case when id < 4 then date '2024-01-01' else date '2025-01-01' end as c from t order by id",
    "case_null_else": "select id, case when id < 4 then date '2024-01-01' end as c from t order by id",
    "coalesce_date_column": "select id, coalesce(d, date '2020-01-01') as c from t order by id",
    "case_condition_on_day_column": "select id, case when n >= date '2024-03-01' then 'late' else 'early' end as c "
                                    "from t order by id",
    "output_next_to_day_compare": "select id, date '2024-01-01' as c from t where n >= date '2024-03-01' order by id",
    "subquery_output": "select id, c from (select id, date '2024-01-01' as c from t) s order by id",
    "subquery_compare_on_output": "select id from (select id, date '2024-03-01' as c from t) s where c > date '2024-01-01' "
                                  "order by id",
    "cte_output": "with u as (select id, date '2024-01-01' as c from t) select id, c from u order by id",
    "order_by_output": "select id, date '2024-01-01' as c from t order by c, id",
    "next_to_day_column": "select id, date '2024-01-01' as c from t order by n, id",
    # a date leaving a derived table, then meeting the int32 day column (all worked before F-115)
    "derived_gt": "select id from (select id, n, date '2024-03-01' as x from t) where x > n order by id",
    "derived_ge": "select id from (select id, n, date '2024-03-01' as x from t) where n >= x order by id",
    "derived_eq": "select id from (select id, n, date '2024-03-01' as x from t) where x = n order by id",
    "derived_ne": "select id from (select id, n, date '2024-03-01' as x from t) where x <> n order by id",
    "derived_lt_out": "select id, x from (select id, n, date '2024-03-01' as x from t) where n < x order by id",
    "derived_group": "select g, count(*) as c from (select g, n, date '2024-03-01' as x from t) where n > x "
                     "group by g order by g",
    "derived_interval": "select id from (select id, n, date '2024-03-01' + interval 1 day as x from t) where n < x "
                        "order by id",
    "derived_minus": "select id, x - n as c from (select id, n, date '2024-03-05' as x from t) order by id",
    "derived_case": "select id from (select id, n, case when g = 0 then date '2024-03-01' else date '2024-04-10' end as x "
                    "from t) where n < x order by id",
    "case_of_dates_vs_day": "select id from t where n < case when g = 0 then date '2024-03-01' else date '2024-04-10' end "
                            "order by id",
    # comparisons keep the day-number convention
    "where_day_column_ge": "select id from t where n >= date '2024-03-01' order by id",
    "where_day_column_between": "select id from t where n between date '2024-02-01' and date '2024-06-01' order by id",
    "where_day_column_interval": "select id from t where n < date '2024-01-01' + interval 3 month order by id",
    "where_date_column_ge": "select id from t where d >= date '2024-03-01' order by id",
    "where_date_literal_first": "select id from t where date '2024-03-01' <= n order by id",
    "having_day": "select g, count(*) as k from t group by g having min(n) < date '2024-02-01' order by g",
    "group_sum_where_day": "select g, sum(v) as s from t where n < date '2024-06-01' group by g order by g",
}


@pytest.mark.parametrize("name", list(QUERIES))
def test_matches_duckdb(cfg, name):
    q = QUERIES[name]
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T).compute(device=cfg)
    assert got.column_names == want.column_names
    assert _rows(got) == _rows(want), (_rows(got), _rows(want))
    for g, w in zip(got.schema, want.schema):
        if pa.types.is_date(w.type):
            assert pa.types.is_date32(g.type), (g.name, g.type, w.type)


def test_select_date_returns_a_date():
    out = mx.sql("select id, date '2024-01-01' as c from t order by id", t=T).compute()
    assert out.schema.field("c").type == pa.date32()
    assert out.column("c").to_pylist() == [dt.date(2024, 1, 1)] * N


def test_int32_day_column_still_compares_by_day_number():
    out = mx.sql("select id from t where n >= date '2024-03-01' order by id", t=T).compute()
    assert out.column("id").to_pylist() == [2, 3, 4, 6, 7]
    out = mx.sql("select id, date '2024-03-01' as c from t where n >= date '2024-03-01' order by id", t=T).compute()
    assert out.schema.field("c").type == pa.date32()
    assert out.column("id").to_pylist() == [2, 3, 4, 6, 7]


def test_coalesce_of_constants_only_is_refused_like_the_other_types():
    with pytest.raises(NotImplementedError, match="constants only"):
        mx.sql("select id, coalesce(null, date '2024-01-01') as c from t", t=T)
