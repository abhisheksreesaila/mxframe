"""F-116: SQL constants beside aggregates, DATE lists in IN, and a DATE constant as a subquery / MIN / EXTRACT.

Before: ``select g, 1 as c, count(*) from t group by g`` raised "must appear in the GROUP BY"; ``d IN (DATE '..',
DATE '..')`` raised "Expected scalar literal in IN list, got Cast"; ``(select date '..')``, ``min(date '..')`` and
``extract(year from date '..')`` raised or came back as a float day number.

DuckDB (probed): a select item with no column of the input is a constant, one value on every output row (also for
an aggregate over no rows); an item built only from group keys and aggregates (``gi + 1``, ``count(*) * 2``) is
fine; a bare column that is neither is an error. HAVING and ORDER BY may name the constant's alias.

``dd`` is an int32 day number (the TPC-H Parquet files store dates so); DuckDB sees it as a DATE.
Expected answers come from a NATIVE DuckDB table, on cpu and gpu, native executor on and off.
"""
import datetime as dt
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest

import omxcore as mx
import omxcore.native_exec as nx
from omxcore import gpu_stream

GPU = gpu_stream.cuda_available()
D0 = dt.date(2024, 1, 1)
EPOCH = dt.date(1970, 1, 1)


def _table():
    n = 12
    dates = [None if k % 5 == 2 else D0 + dt.timedelta(days=k * 40) for k in range(n)]
    return pa.table({
        "id": pa.array(range(n), pa.int64()),
        "g": pa.array(["a", "a", "b", None, "b", "a", "c", None, "b", "a", "c", "b"]),
        "gi": pa.array([None if k % 6 == 5 else k % 3 for k in range(n)], pa.int64()),
        "x": pa.array([None if k % 4 == 3 else k * 2 for k in range(n)], pa.int64()),
        "f": pa.array([None if k % 5 == 1 else k * 1.5 - 6 for k in range(n)], pa.float64()),
        "d": pa.array(dates, pa.date32()),
        "dd": pa.array([None if d is None else (d - EPOCH).days for d in dates], pa.int32()),
    })


T = _table()
CON = duckdb.connect()
CON.register("_t", T)
CON.execute("create table t as select id, g, gi, x, f, d, DATE '1970-01-01' + dd as dd from _t")


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


def _rows(x):
    cols = [x.column(i).to_pylist() for i in range(x.num_columns)]
    return [tuple(_norm(c[r]) for c in cols) for r in range(x.num_rows)]


def _kind(t):
    if pa.types.is_null(t):
        return "null"
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    return "date" if pa.types.is_date(t) else str(t)


def _check(q, dev):
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T).compute(device=dev)
    assert got.column_names == want.column_names, q
    assert _rows(got) == _rows(want), q
    # an all-NULL column is typed (INTEGER) by DuckDB and null by the engine: the values decide
    assert [_kind(t) for t in got.schema.types] == [_kind(t) if not pa.types.is_integer(t) or w.null_count < len(w)
                                                   else "null"
                                                   for t, w in zip(want.schema.types, want.columns)] \
        or all(k == "null" or k == _kind(w) for k, w in zip((_kind(t) for t in got.schema.types), want.schema.types)), q
    return want


# ---------------------------------------------------------------- item 1: constants beside aggregates

CONSTANTS = {
    "int": "1", "negative_int": "-7", "big_int": "3000000000", "float": "1.5", "double": "1e0", "string": "'a'",
    "empty_string": "''", "date": "date '2020-01-02'", "null": "null", "true": "true", "false": "false",
    "int_expr": "1 + 1", "string_with_quote": "'it''s'",
}


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_beside_grouped_count(cfg, name):
    q = f"select g, {CONSTANTS[name]} as c, count(*) as n from t group by g order by g"
    want = _check(q, cfg[0])
    assert want.num_rows == 4                                  # a, b, c and the NULL group


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_beside_global_aggregate(cfg, name):
    want = _check(f"select {CONSTANTS[name]} as c, count(*) as n, max(x) as s from t", cfg[0])
    assert want.num_rows == 1


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_beside_aggregate_over_no_rows(cfg, name):
    want = _check(f"select {CONSTANTS[name]} as c, count(*) as n, max(x) as s from t where id > 100", cfg[0])
    assert _rows(want)[0][1:] == (0, None)                     # one row: the constant, 0 rows, no sum


def test_constant_first_in_select_list(cfg):
    _check("select 'x' as tag, g, max(x) as s from t group by g order by g", cfg[0])


def test_several_constants_every_type_together(cfg):
    q = ("select g, 1 as i, 1.5 as f, 'a' as s, date '2020-01-02' as d, null as z, true as b, count(*) as n, "
         "max(x) as s2 from t group by g order by g")
    _check(q, cfg[0])


def test_constant_between_aggregates(cfg):
    _check("select g, count(*) as n, 5 as c, max(x) as s, 'k' as k, max(f) as m from t group by g order by g", cfg[0])


def test_constant_beside_aggregate_without_alias_still_needs_a_name(cfg):
    with pytest.raises(ValueError, match="alias"):
        mx.sql("select g, 1, count(*) as n from t group by g", t=T).compute(device=cfg[0])


@pytest.mark.parametrize("q", [
    "select g, 1 as c, count(*) as n from t group by g having c = 1 order by g",
    "select g, 1 as c, count(*) as n from t group by g having c = 2 order by g",
    "select g, 1 as c, count(*) as n from t group by g having c = 1 and count(*) > 2 order by g",
    "select g, 'a' as c, count(*) as n from t group by g having c = 'a' order by g",
    "select g, 1 as c, count(*) as n from t group by g order by c, g",
    "select g, 1 as c, count(*) as n from t group by g order by c desc, n desc, g",
    "select g, 1 as c, count(*) as n from t group by g order by 2, 3 desc, 1",
    "select g, 1 as c, count(*) as n from t group by g order by n desc, g limit 2",
    "select 1 as c, count(*) as n from t having c = 1",
    "select 1 as c, count(*) as n from t having c = 2",
    "select g, 2.5 as c, count(*) as n from t group by g having c > 2 order by g",
])
def test_having_and_order_by_on_a_constant(cfg, q):
    _check(q, cfg[0])


# expressions of group keys and aggregates (not constants, not aggregates alone)
EXPRESSIONS = {
    "key_plus_one": "select gi, gi + 1 as h, count(*) as n from t group by gi order by gi",
    "key_times_aggregate": "select gi, gi * 2 + count(*) as h from t group by gi order by gi",
    "aggregate_times_two": "select g, count(*) * 2 as h from t group by g order by g",
    "key_and_constant": "select gi, gi + 10 as h, 1 as c, max(x) as s from t group by gi order by gi",
    "two_keys": "select g, gi, gi + 1 as h, count(*) as n from t group by g, gi order by g, gi",
    "key_minus_key": "select gi, gi - gi as h, count(*) as n from t group by gi order by gi",
    "expression_key": "select gi + 1 as h, count(*) as n from t group by gi + 1 order by h",
    "key_float": "select gi, gi * 1.5 as h, count(*) as n from t group by gi order by gi",
    "having_on_it": "select gi, gi + 1 as h, count(*) as n from t group by gi having h > 1 order by gi",
    "order_by_it": "select gi, gi + 1 as h, count(*) as n from t group by gi order by h desc, gi",
}


@pytest.mark.parametrize("name", EXPRESSIONS)
def test_expressions_of_keys_and_aggregates(cfg, name):
    _check(EXPRESSIONS[name], cfg[0])


@pytest.mark.parametrize("q", [
    "select g, x, count(*) as n from t group by g",                       # x is neither grouped nor aggregated
    "select g, x + 1 as h, count(*) as n from t group by g",
    "select g, 1 + x as h, count(*) as n from t group by g",
    "select 1 as c, x, count(*) as n from t",
])
def test_a_column_that_is_neither_grouped_nor_aggregated_still_raises(cfg, q):
    with pytest.raises(Exception):
        CON.execute(q)                                                  # DuckDB refuses it too
    with pytest.raises(ValueError, match="must appear in the GROUP BY"):
        mx.sql(q, t=T).compute(device=cfg[0])


# ---------------------------------------------------------------- item 2: DATE lists in IN

DATE_IN = {
    "date32_where": "select id from t where d in (date '2024-01-01', date '2024-04-10', date '2025-03-26') order by id",
    "date32_single": "select id from t where d in (date '2024-01-01') order by id",
    "date32_not_in": "select id from t where d not in (date '2024-01-01', date '2024-04-10') order by id",
    "date32_none_match": "select count(*) as n from t where d in (date '1999-01-01', date '1999-01-02')",
    "date32_with_other_filter": "select id from t where d in (date '2024-01-01', date '2024-02-10') and id < 5 order by id",
    "date32_interval_constant": "select id from t where d in (date '2024-01-01' + interval 40 day, date '2024-01-01') order by id",
    "date32_group": "select g, count(*) as n from t where d in (date '2024-01-01', date '2024-04-10', "
                    "date '2024-05-20') group by g order by g",
    "date32_having": "select g, count(*) as n from t group by g having min(d) in (date '2024-01-01', date '2024-04-10') "
                     "order by g",
    "int32_where": "select id from t where dd in (date '2024-01-01', date '2024-04-10', date '2025-03-26') order by id",
    "int32_single": "select id from t where dd in (date '2024-01-01') order by id",
    "int32_not_in": "select id from t where dd not in (date '2024-01-01', date '2024-04-10') order by id",
    "int32_none_match": "select count(*) as n from t where dd in (date '1999-01-01', date '1999-01-02')",
    "int32_group": "select g, count(*) as n from t where dd in (date '2024-01-01', date '2024-04-10', "
                   "date '2024-05-20') group by g order by g",
    "int32_epoch": "select count(*) as n from t where dd in (date '1970-01-01', date '2024-01-01')",
}


@pytest.mark.parametrize("name", DATE_IN)
def test_date_in_list(cfg, name):
    want = _check(DATE_IN[name], cfg[0])
    if "none_match" not in name and "having" not in name:
        assert want.num_rows > 0                                # the case matches something


def test_date_in_list_not_in_keeps_null_dates_out(cfg):
    got = mx.sql("select id from t where d not in (date '2024-01-01') order by id", t=T).compute(device=cfg[0])
    nulls = [k for k in range(12) if k % 5 == 2]
    assert all(r[0] not in nulls for r in _rows(got))           # a NULL date is neither IN nor NOT IN


# ---------------------------------------------------------------- item 3: DATE constants

BARE = {
    "subquery_in_select": "select id, (select date '2020-01-02') as c from t order by id",
    "subquery_aliased_inside": "select id, (select date '2020-01-02' as e) as c from t order by id",
    "subquery_vs_date32": "select id from t where d = (select date '2024-01-01') order by id",
    "subquery_vs_date32_gt": "select id from t where d > (select date '2024-06-01') order by id",
    "subquery_vs_int32": "select id from t where dd = (select date '2024-01-01') order by id",
    "subquery_vs_int32_lt": "select id from t where dd < (select date '2024-06-01') order by id",
    "subquery_beside_aggregate": "select g, (select date '2020-01-02') as c, count(*) as n from t group by g order by g",
    "subquery_plus_interval_is_a_timestamp": "select id, (select date '2020-01-02' + interval 1 day) as c from t order by id",
    "subquery_plus_interval_vs_date32": "select id from t where d < (select date '2024-01-01' + interval 90 day) order by id",
    "min_date": "select min(date '2020-01-02') as m from t",
    "max_date": "select max(date '2020-01-02') as m from t",
    "min_date_grouped": "select g, min(date '2020-01-02') as m, count(*) as n from t group by g order by g",
    "min_date_no_rows": "select min(date '2020-01-02') as m, max(date '2020-01-02') as x from t where id > 100",
    "min_date_with_column_max": "select min(date '2020-01-02') as m, max(d) as x from t",
    "extract_year": "select id, extract(year from date '2020-01-02') as y from t order by id",
    "extract_year_after_interval": "select id, extract(year from date '2020-01-02' + interval 1 year) as y from t order by id",
    "extract_year_beside_aggregate": "select g, extract(year from date '2021-06-30') as y, count(*) as n from t "
                                     "group by g order by g",
    "extract_year_where": "select id from t where extract(year from d) = extract(year from date '2024-01-01') order by id",
}


@pytest.mark.parametrize("name", BARE)
def test_date_constant_forms(cfg, name):
    want = _check(BARE[name], cfg[0])
    assert want.num_rows > 0 or "no_rows" in name or name.startswith("subquery_vs")


# ---------------------------------------------------------------- reviewer round 1

@pytest.mark.parametrize("q", [
    "select id, (select date '2020-01-01' where 1 = 0) as c from t order by id",       # DuckDB: NULL on every row
    "select count(*) as n from t where d = (select date '2020-01-01' limit 0)",         # DuckDB: no row matches
    "select count(*) as n from t where d = (select date '2024-01-01' having 1 = 0)",
])
def test_a_date_subquery_with_other_clauses_is_refused_not_answered_as_the_date(cfg, q):
    CON.execute(q)                                                      # DuckDB accepts it
    with pytest.raises(NotImplementedError, match="scalar subquery"):
        mx.sql(q, t=T).compute(device=cfg[0])


def test_min_of_a_date_constant_in_a_window_is_refused(cfg):
    with pytest.raises(NotImplementedError):
        mx.sql("select g, min(date '2020-01-02') over (partition by g) as m from t", t=T).compute(device=cfg[0])


@pytest.mark.parametrize("q", [
    "select g, 1 as c, count(*) as n from t group by g, c order by g",
    "select g, 1 as c, count(*) as n from t group by g, 2 order by g",
    "select g, 1 as c, count(*) as n from t group by 1, 2 order by g",
    "select g, 1 as c, count(*) as n from t group by all order by g",
    "select 1 as c, count(*) as n from t group by c",
    "select g, count(*) as n from t group by g, 1 order by g",
    "select g, 'a' as c, null as z, count(*) as n from t group by g, c, z order by g",
])
def test_group_by_naming_a_constant_is_no_key(cfg, q):
    _check(q, cfg[0])
