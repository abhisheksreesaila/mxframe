"""F-108: SQL TIMESTAMP literals, temporal HAVING, and MIN / MAX of computed text / date / timestamp values.

Before: ``TIMESTAMP '...'`` and ``CAST('...' AS TIMESTAMP)`` raised "Unsupported SQL expression type: Cast";
``HAVING min(ts) > DATE '...'`` raised ArrowNotImplementedError (a timestamp against a day number);
``MIN`` / ``MAX`` of a CASE / COALESCE of strings, dates or timestamps (and of ``d + INTERVAL ...``) cast the
argument to float64 ("could not convert 'hit' to double"; a date or a ns timestamp cannot live in a float64) and a
64-bit integer past 2^53 raised; a timestamp branch next to a date branch in CASE raised "no common type".

DuckDB's rules (probed, session time zone UTC): ``TIMESTAMP '..'`` is a TIMESTAMP (microseconds; ``TIMESTAMP_S`` /
``_MS`` / ``_NS`` have their own unit); a timestamp column is compared with it in the finer unit, a zoned column
treats the literal as UTC; a DATE next to a timestamp is a timestamp of that timestamp's unit and zone (in a compare
and in CASE / COALESCE); two timestamps in CASE / COALESCE take the finer unit, a zone and no zone do not mix; a
DATE plus an INTERVAL is a TIMESTAMP (microseconds); MIN / MAX of a text, date or timestamp expression has that type.

Expected answers come from a NATIVE DuckDB table, on cpu and gpu, native executor on and off. Timestamps are
compared as integer nanoseconds (the instant), and their units as strings.
"""
import datetime as dt

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import gpu_stream

GPU = gpu_stream.cuda_available()
D0 = dt.date(2024, 1, 1)
N = 12
NS = {"s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3, "ns": 1}


def _instants(unit, tz=None, step_hours=7, extra_ns=0):
    """12 timestamps from 2024-01-01 10:00 (about 1.7e18 ns: past 2^53); NULL on 3, 8; ``extra_ns`` makes them
    differ by single nanoseconds."""
    base = int(np.datetime64("2024-01-01T10:00:00", "ns").astype("int64"))
    f = NS[unit]
    vals = [None if k in (3, 8) else (base + k * step_hours * 3600 * 10 ** 9 + k * extra_ns) // f for k in range(N)]
    return pa.array(vals, pa.int64()).cast(pa.timestamp(unit, tz))


def _table():
    return pa.table({
        "id": pa.array(range(N), pa.int64()),
        "g": pa.array([k % 3 for k in range(N)], pa.int32()),
        "tn": _instants("ns", extra_ns=1),                           # 1 ns apart is visible only in int64
        "tu": _instants("us"),
        "tm": _instants("ms"),
        "tsec": _instants("s"),
        "zn": _instants("ns", "UTC"),                              # (DuckDB keeps microseconds for a zoned column)
        "zu": _instants("us", "UTC"),
        "zm": _instants("ms", "America/New_York"),
        "zs": _instants("s", "Asia/Kolkata"),
        "d": pa.array([None if k % 5 == 2 else D0 + dt.timedelta(days=k * 40) for k in range(N)], pa.date32()),
        "big": pa.array([None if k % 6 == 5 else 2 ** 60 + k for k in range(N)], pa.int64()),
        "s": pa.array(["hit", "miss", None, "hit", "other", "hit", "miss", None, "other", "miss", "hit", ""]),
        "ls": pa.array(["hit", None, "x", "hit", "other", None, "miss", "y", "other", "miss", "hit", "z"],
                       pa.large_string()),
        "ds": pa.array(["a", None, "c", "a", None, "b", "c", "a", "b", "c", "a", "b"]).dictionary_encode(),
    })


T = _table()
TS_COLS = ["tn", "tu", "tm", "tsec", "zn", "zu", "zm", "zs"]
CON = duckdb.connect()
CON.execute("set TimeZone='UTC'")
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


def _cell(col, i):
    t = col.type
    v = col[i]
    if not v.is_valid:
        return None
    if pa.types.is_timestamp(t):
        return v.value * NS[t.unit]                       # the instant, in ns
    return v.as_py()


def _rows(x):
    return [tuple(_cell(x.column(c), r) for c in range(x.num_columns)) for r in range(x.num_rows)]


def _kind(t):
    if pa.types.is_timestamp(t):
        return "ts:tz" if t.tz else f"ts:{t.unit}"          # (DuckDB has one zoned type, microseconds; we keep the column's unit)
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    return "date" if pa.types.is_date(t) else "bool" if pa.types.is_boolean(t) else str(t)


def _run(q, dev, **tables):
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, **(tables or {"t": T})).compute(device=dev)
    return _rows(got), _rows(want), got, want


# -- TIMESTAMP literals in WHERE / BETWEEN / IN, every unit, with and without a zone -----------------------------

FORMS = [
    "{c} > timestamp '2024-01-02 00:00:00'",
    "{c} >= timestamp '2024-01-01 10:00:00'",
    "{c} < TIMESTAMP '2024-01-02 03:00:00'",
    "{c} <= cast('2024-01-02 03:00:00' as timestamp)",
    "{c} = timestamp '2024-01-01 10:00:00'",
    "{c} <> timestamp '2024-01-01 10:00:00'",
    "timestamp '2024-01-02 00:00:00' < {c}",
    "{c} between timestamp '2024-01-01 12:00:00' and timestamp '2024-01-02 12:00:00'",
    "{c} not between timestamp '2024-01-01 12:00:00' and timestamp '2024-01-02 12:00:00'",
    "{c} > timestamp '2024-01-01 10:00:00.5'",                       # finer than a seconds column
    "{c} < timestamp '2024-01-01 10:00:00.000001'",
    "{c} > cast('2024-01-02' as timestamp)",
    "{c} in (timestamp '2024-01-01 10:00:00', timestamp '2024-01-01 17:00:00')",
    "{c} not in (timestamp '2024-01-01 10:00:00', timestamp '2024-01-01 17:00:00')",
    "{c} > date '2024-01-02'",                                       # a DATE is promoted to a timestamp
    "{c} between date '2024-01-01' and date '2024-01-02'",
    "{c} >= timestamp '2024-01-01 10:00:00' and {c} < timestamp '2024-01-03 00:00:00'",
    "{c} < timestamp '2024-01-01 10:00:00' or {c} > timestamp '2024-01-03 00:00:00'",
    "not ({c} > timestamp '2024-01-02 00:00:00')",
    "{c} > timestamp '2099-01-01 00:00:00'",
    "{c} < timestamp '2099-01-01 00:00:00'",
]


@pytest.mark.parametrize("form", range(len(FORMS)))
@pytest.mark.parametrize("c", TS_COLS)
def test_where_timestamp_literal_equals_duckdb(c, form, cfg):
    q = f"select id from t where {FORMS[form].format(c=c)} order by id"
    got, want, _, _ = _run(q, cfg[0])
    assert got == want, (q, got, want)


NS_FORMS = [
    "{c} = timestamp_ns '2024-01-01 10:00:00.000000000'",
    "{c} = timestamp_ns '2024-01-01 17:00:00.000000001'",
    "{c} = cast('2024-01-01 17:00:00.000000001' as timestamp_ns)",
    "{c} > timestamp_ns '2024-01-01 17:00:00.000000001'",
    "{c} >= timestamp_ns '2024-01-01 17:00:00.000000001'",
    "{c} < timestamp_ns '2024-01-01 17:00:00.000000002'",
    "{c} in (timestamp_ns '2024-01-01 10:00:00', timestamp_ns '2024-01-01 17:00:00.000000001')",
    "{c} between timestamp_ns '2024-01-01 17:00:00.000000001' and timestamp_ns '2024-01-01 17:00:00.000000002'",
]


@pytest.mark.parametrize("form", range(len(NS_FORMS)))
@pytest.mark.parametrize("c", ["tn", "tu"])
def test_where_nanosecond_literal_past_2_53_equals_duckdb(c, form, cfg):
    """Single nanoseconds at 1.7e18 ns are not representable in float64: the compare stays in the integer unit."""
    q = f"select id from t where {NS_FORMS[form].format(c=c)} order by id"
    got, want, _, _ = _run(q, cfg[0])
    assert got == want, (q, got, want)


def test_the_nanosecond_row_is_the_only_match(cfg):
    got, _, _, _ = _run("select id from t where tn = timestamp_ns '2024-01-01 17:00:00.000000001'", cfg[0])
    assert got == [(1,)]


UNITS = [("s", "timestamp_s '2024-01-01 10:00:00'"), ("ms", "timestamp_ms '2024-01-01 10:00:00.123'"),
         ("us", "timestamp '2024-01-01 10:00:00.123456'"), ("ns", "timestamp_ns '2024-01-01 10:00:00.123456789'")]


@pytest.mark.parametrize("unit,lit", UNITS)
@pytest.mark.parametrize("c", TS_COLS)
def test_literal_of_every_unit_against_every_column(c, unit, lit, cfg):
    if c.startswith("z") and unit != "us":
        pytest.skip("DuckDB has no implicit cast between a zoned timestamp and TIMESTAMP_S / _MS / _NS")
    for op in (">", "<=", "="):
        q = f"select id from t where {c} {op} {lit} order by id"
        got, want, _, _ = _run(q, cfg[0])
        assert got == want, (q, got, want)


@pytest.mark.parametrize("c", ["tu", "zu"])
def test_date_column_against_timestamp_literal(c, cfg):
    for q in ("select id from t where d = timestamp '2024-01-01 00:00:00' order by id",
              "select id from t where d > timestamp '2024-01-01 00:00:01' order by id",
              "select id from t where d <= timestamp '2024-04-09 23:59:59' order by id",
              "select id from t where d between timestamp '2024-01-01 00:00:00' and timestamp '2024-05-01 00:00:00' order by id",
              f"select id from t where {c} > d order by id"):
        got, want, _, _ = _run(q, cfg[0])
        assert got == want, (q, got, want)


# -- SELECT, GROUP BY, ORDER BY, CASE, COALESCE, NULLIF with timestamps -----------------------------------------

SELECT = {
    "literal_us": ("select id, timestamp '2024-01-01 10:00:00' as c from t order by id", ["int", "ts:us"]),
    "literal_cast": ("select id, cast('2024-01-01 10:00:00' as timestamp) as c from t order by id", ["int", "ts:us"]),
    "literal_ns": ("select id, timestamp_ns '2024-01-01 10:00:00.000000007' as c from t order by id", ["int", "ts:ns"]),
    "literal_s": ("select id, timestamp_s '2024-01-01 10:00:00' as c from t order by id", ["int", "ts:s"]),
    "compare_in_select": ("select id, tn > timestamp '2024-01-02 00:00:00' as c from t order by id", ["int", "bool"]),
    "case_literal_else": ("select id, case when tu > timestamp '2024-01-02' then tu else timestamp '2000-01-01' end as c "
                          "from t order by id", ["int", "ts:us"]),
    "case_ns_and_us_literal": ("select id, case when id < 6 then tn else timestamp '2000-01-01' end as c from t order by id",
                               ["int", "ts:ns"]),
    "case_no_else": ("select id, case when id < 6 then tm end as c from t order by id", ["int", "ts:ms"]),
    "case_s_and_ns": ("select id, case when id < 6 then tsec else tn end as c from t order by id", ["int", "ts:ns"]),
    "case_ms_and_us": ("select id, case when id < 6 then tm else tu end as c from t order by id", ["int", "ts:us"]),
    "case_s_and_us": ("select id, case when id < 6 then tsec else tu end as c from t order by id", ["int", "ts:us"]),
    "case_zoned_units": ("select id, case when id < 6 then zs else zn end as c from t order by id", ["int", "ts:tz"]),
    "case_timestamp_and_date": ("select id, case when id < 6 then tu else d end as c from t order by id", ["int", "ts:us"]),
    "case_date_and_timestamp": ("select id, case when id < 6 then d else tu end as c from t order by id", ["int", "ts:us"]),
    "case_seconds_and_date": ("select id, case when id < 6 then tsec else d end as c from t order by id", ["int", "ts:s"]),
    "case_ns_and_date": ("select id, case when id < 6 then tn else d end as c from t order by id", ["int", "ts:ns"]),
    "case_zoned_and_date": ("select id, case when id < 6 then zm else d end as c from t order by id", ["int", "ts:tz"]),
    "case_date_literal_and_timestamp": ("select id, case when id < 6 then date '2020-02-29' else tu end as c from t order by id",
                                        ["int", "ts:us"]),
    "case_timestamp_literal_and_date": ("select id, case when id < 6 then timestamp '2020-02-29 12:30:00' else d end as c "
                                        "from t order by id", ["int", "ts:us"]),
    "coalesce_literal": ("select id, coalesce(tu, timestamp '1999-12-31 23:59:59') as c from t order by id", ["int", "ts:us"]),
    "coalesce_columns": ("select id, coalesce(tm, tn) as c from t order by id", ["int", "ts:ns"]),
    "coalesce_s_and_us": ("select id, coalesce(tsec, tu) as c from t order by id", ["int", "ts:us"]),
    "coalesce_date": ("select id, coalesce(tsec, d) as c from t order by id", ["int", "ts:s"]),
    "coalesce_date_first": ("select id, coalesce(d, tn) as c from t order by id", ["int", "ts:ns"]),
    "coalesce_date_literal": ("select id, coalesce(tu, date '2000-01-01') as c from t order by id", ["int", "ts:us"]),
    "coalesce_zoned": ("select id, coalesce(zm, zn) as c from t order by id", ["int", "ts:tz"]),
    "coalesce_ns_exact": ("select id, coalesce(tn, timestamp_ns '2000-01-01 00:00:00.000000009') as c from t order by id",
                          ["int", "ts:ns"]),
    "nullif_literal": ("select id, nullif(tu, timestamp '2024-01-01 10:00:00') as c from t order by id", ["int", "ts:us"]),
    "nullif_ns": ("select id, nullif(tn, timestamp_ns '2024-01-01 17:00:00.000000001') as c from t order by id", ["int", "ts:ns"]),
    "is_null_coalesce": ("select id, coalesce(tu, tm) is null as c from t order by id", ["int", "bool"]),
    "is_null_case": ("select id, case when id < 6 then tu end is null as c from t order by id", ["int", "bool"]),
    "date_plus_interval_day": ("select id, d + interval 1 day as c from t order by id", ["int", "ts:us"]),
    "date_minus_interval_hours": ("select id, d - interval 36 hour as c from t order by id", ["int", "ts:us"]),
    "timestamp_plus_interval": ("select id, tu + interval 90 minute as c from t order by id", ["int", "ts:us"]),
    "timestamp_s_plus_interval": ("select id, tsec + interval 1 second as c from t order by id", ["int", "ts:us"]),
    "zoned_plus_interval": ("select id, zm - interval 1 day as c from t order by id", ["int", "ts:tz"]),
}


@pytest.mark.parametrize("name", SELECT)
def test_select_equals_duckdb(name, cfg):
    q, kinds = SELECT[name]
    got, want, tbl, wtbl = _run(q, cfg[0])
    assert got == want, (got, want)
    assert [_kind(f.type) for f in tbl.schema] == kinds == [_kind(f.type) for f in wtbl.schema]


def test_timestamp_branch_in_the_nanosecond_row_stays_exact(cfg):
    got, want, _, _ = _run(SELECT["case_ns_and_date"][0], cfg[0])
    assert got == want and got[1][1] == int(np.datetime64("2024-01-01T17:00:00.000000001", "ns").astype("int64"))


def test_zoned_and_naive_timestamps_do_not_mix_in_case(cfg):
    q = "select case when id < 6 then zn else tn end as c from t"
    with pytest.raises(Exception):
        CON.execute(q)                                    # DuckDB: Cannot mix values of type ...
    with pytest.raises(Exception):
        mx.sql(q, t=T).compute(device=cfg[0])


KEYS = {
    "group_by_column": "select tn, count(*) as n from t group by tn order by tn",
    "group_by_zoned": "select zm, count(*) as n from t group by zm order by zm",
    "group_by_s": "select tsec, count(*) as n from t group by tsec order by tsec",
    "group_by_case_literal": "select case when tu > timestamp '2024-01-02' then tu end as k, count(*) as n from t "
                             "group by 1 order by 1",
    "group_by_coalesce": "select coalesce(tu, timestamp '2000-01-01') as k, count(*) as n from t group by 1 order by 1",
    "group_by_coalesce_ns": "select coalesce(tn, timestamp_ns '2000-01-01 00:00:00.000000001') as k, count(*) as n from t "
                            "group by 1 order by 1",
    "group_by_timestamp_and_date": "select case when id < 6 then tu else d end as k, count(*) as n from t group by 1 order by 1",
    "group_by_filtered": "select g, count(*) as n from t where tn >= timestamp '2024-01-01 10:00:00' and "
                         "tn < timestamp '2024-01-03 00:00:00' group by g order by g",
    "group_by_filtered_zoned": "select g, count(*) as n, sum(id) as s from t where zm > timestamp '2024-01-02' group by g order by g",
    "order_by_coalesce": "select id from t order by coalesce(tn, timestamp_ns '2000-01-01'), id",
    "order_by_coalesce_desc": "select id from t order by coalesce(tu, timestamp '2999-01-01') desc, id",
    "order_by_case": "select id from t order by case when id < 6 then tu else d end, id",
    "order_by_nullif": "select id from t order by nullif(tu, timestamp '2024-01-01 10:00:00'), id",
    "order_by_interval": "select id from t order by d + interval 1 day desc, id",
    "where_coalesce": "select id from t where coalesce(tu, timestamp '1999-01-01') < timestamp '2024-01-02' order by id",
    "where_coalesce_date_literal": "select id from t where coalesce(tu, date '1999-01-01') < date '2024-01-02' order by id",
    "where_case": "select id from t where case when id < 6 then tu else d end > timestamp '2024-01-02' order by id",
    "where_nullif": "select id from t where nullif(tu, timestamp '2024-01-01 10:00:00') is null order by id",
    "where_interval": "select id from t where d + interval 1 day > date '2024-02-01' order by id",
    "where_interval_timestamp": "select id from t where tu + interval 1 day > timestamp '2024-01-02 12:00:00' order by id",
    "limit": "select id, tn from t where tn > timestamp '2024-01-01 12:00:00' order by tn desc limit 3",
}


@pytest.mark.parametrize("name", KEYS)
def test_keys_order_where_equal_duckdb(name, cfg):
    got, want, _, _ = _run(KEYS[name], cfg[0])
    assert got == want, (got, want)


# -- HAVING -------------------------------------------------------------------------------------------------

HAVING = {}
for _c in TS_COLS:
    HAVING[f"min_{_c}_gt_date"] = f"select g, min({_c}) as m from t group by g having min({_c}) > date '2024-01-01' order by g"
    HAVING[f"max_{_c}_le_date"] = f"select g, max({_c}) as m from t group by g having max({_c}) <= date '2024-01-03' order by g"
    HAVING[f"min_{_c}_gt_timestamp"] = (f"select g, min({_c}) as m from t group by g "
                                        f"having min({_c}) > timestamp '2024-01-01 10:00:00' order by g")
HAVING.update({
    "min_gt_date_global": "select min(tu) as m from t having min(tu) > date '2024-01-01'",
    "min_eq_date": "select g, min(tu) as m from t group by g having min(tu) = date '2024-01-01' order by g",
    "min_between_dates": "select g, min(tu) as m from t group by g having min(tu) between date '2024-01-01' and date '2024-01-02' order by g",
    "min_between_timestamps": "select g, min(tn) as m from t group by g "
                              "having min(tn) between timestamp '2024-01-01 10:00:00' and timestamp '2024-01-02 00:00:00' order by g",
    "min_gt_date_and_count": "select g, min(tu) as m, count(*) as n from t group by g having min(tu) >= date '2024-01-01' and count(*) > 3 order by g",
    "date_literal_on_the_left": "select g, max(tu) as m from t group by g having date '2024-01-02' < max(tu) order by g",
    "min_date_gt_timestamp": "select g, min(d) as m from t group by g having min(d) > timestamp '2024-01-01 00:00:01' order by g",
    "min_date_ge_date": "select g, min(d) as m from t group by g having min(d) >= date '2024-02-10' order by g",
    "min_in_literals": "select g, min(tu) as m from t group by g having min(tu) in (timestamp '2024-01-01 10:00:00') order by g",
    "order_by_min": "select g, min(tn) as m from t group by g order by min(tn) desc",
    "having_not": "select g, min(tu) as m from t group by g having not (min(tu) > date '2024-01-01') order by g",
    "having_coalesce_min": "select g, min(tu) as m from t group by g having coalesce(min(tu), timestamp '2999-01-01') > date '2024-01-01' order by g",
})


@pytest.mark.parametrize("name", HAVING)
def test_having_equals_duckdb(name, cfg):
    got, want, tbl, wtbl = _run(HAVING[name], cfg[0])
    assert got == want, (got, want)
    assert [_kind(f.type) for f in tbl.schema] == [_kind(f.type) for f in wtbl.schema]


# -- MIN / MAX of computed values -----------------------------------------------------------------------------

AGG = {
    # strings
    "max_case_string": "select max(case when id > 3 then s end) as m from t",
    "min_case_string": "select min(case when id > 3 then s end) as m from t",
    "max_case_string_else": "select max(case when id > 3 then s else 'zz' end) as m from t",
    "min_case_string_columns": "select min(case when id > 3 then s else ls end) as m from t",
    "max_case_large_string": "select max(case when id > 3 then ls end) as m from t",
    "max_case_dictionary": "select max(case when id > 3 then ds end) as m from t",
    "min_case_dictionary": "select min(case when id > 3 then ds else 'zz' end) as m from t",
    "min_max_coalesce_string": "select min(coalesce(s, 'zzz')) as a, max(coalesce(s, 'A')) as b from t",
    "max_coalesce_three": "select max(coalesce(s, ls, ds, 'none')) as m from t",
    "min_nullif_string": "select min(nullif(s, '')) as a, max(nullif(s, 'miss')) as b from t",
    "min_case_string_none_match": "select min(case when id > 99 then s end) as m from t",
    "grouped_max_case_string": "select g, max(case when id > 3 then s end) as m from t group by g order by g",
    "grouped_min_case_string": "select g, min(case when id > 3 then s else 'zz' end) as m from t group by g order by g",
    "grouped_min_max_coalesce": "select g, min(coalesce(s, 'zzz')) as a, max(coalesce(s, 'A')) as b from t group by g order by g",
    "grouped_dictionary": "select g, max(case when id > 3 then ds end) as m from t group by g order by g",
    "grouped_with_sum": "select g, max(case when id > 3 then s end) as m, count(*) as t from t group by g order by g",
    "grouped_none_match": "select g, max(case when id > 99 then s end) as m from t group by g order by g",
    "filtered": "select g, max(case when id > 3 then s end) as m from t where id <> 4 group by g order by g",
    "having_string": "select g, min(case when id > 3 then s else 'zz' end) as m from t group by g "
                     "having min(case when id > 3 then s else 'zz' end) > 'a' order by g",
    "having_string_alias_agg": "select g, max(case when id > 3 then s end) as m from t group by g having max(case when id > 3 then s end) = 'miss' order by g",
    "order_by_string_agg": "select g, max(case when id > 3 then s end) as m from t group by g order by max(case when id > 3 then s end) desc, g",
    "limit_string_agg": "select g, max(coalesce(s, 'A')) as m from t group by g order by m desc, g limit 2",
    # dates
    "max_case_date": "select max(case when id > 3 then d end) as m from t",
    "min_case_date_else": "select min(case when id > 3 then d else date '2999-12-31' end) as m from t",
    "min_coalesce_date": "select min(coalesce(d, date '1999-01-01')) as m from t",
    "max_coalesce_date": "select max(coalesce(d, date '1999-01-01')) as m from t",
    "grouped_max_case_date": "select g, max(case when id > 3 then d end) as m from t group by g order by g",
    "grouped_min_coalesce_date": "select g, min(coalesce(d, date '1999-01-01')) as m from t group by g order by g",
    "grouped_nullif_date": "select g, min(nullif(d, date '2024-01-01')) as m from t group by g order by g",
    "having_date_case": "select g, max(case when id > 3 then d end) as m from t group by g having max(case when id > 3 then d end) > date '2024-06-01' order by g",
    "min_date_interval": "select min(d + interval 1 day) as m from t",
    "grouped_max_date_interval": "select g, max(d + interval 1 day) as m from t group by g order by g",
    "grouped_min_date_interval_hours": "select g, min(d - interval 36 hour) as m from t group by g order by g",
    "having_date_interval": "select g, min(d + interval 1 day) as m from t group by g having min(d + interval 1 day) > date '2024-01-02' order by g",
    # timestamps, every unit
    "min_case_timestamp": "select min(case when id > 3 then tu end) as m from t",
    "grouped_max_coalesce_ns": "select g, max(coalesce(tn, timestamp_ns '2000-01-01 00:00:00.000000009')) as m from t group by g order by g",
    "grouped_min_coalesce_ns": "select g, min(coalesce(tn, timestamp_ns '2200-01-01 00:00:00.000000009')) as m from t group by g order by g",
    "grouped_case_ms": "select g, max(case when id > 3 then tm end) as m from t group by g order by g",
    "grouped_case_s": "select g, min(case when id > 3 then tsec end) as m from t group by g order by g",
    "grouped_case_zoned": "select g, max(case when id > 3 then zm end) as m from t group by g order by g",
    "grouped_case_zoned_ns": "select g, max(case when id > 3 then zn end) as m from t group by g order by g",
    "grouped_coalesce_zoned": "select g, min(coalesce(zs, zn)) as m from t group by g order by g",
    "timestamp_and_date_branches": "select g, max(case when id > 3 then tu else d end) as m from t group by g order by g",
    "date_and_timestamp_branches": "select g, min(case when id > 3 then d else tn end) as m from t group by g order by g",
    "coalesce_date_and_timestamp": "select g, max(coalesce(tsec, d)) as m from t group by g order by g",
    "global_timestamp_and_date": "select min(case when id > 3 then tn else d end) as a, max(case when id > 3 then tn else d end) as b from t",
    "timestamp_interval": "select g, max(tn + interval 1 day) as m from t group by g order by g",
    "timestamp_s_interval": "select g, min(tsec + interval 1 second) as m from t group by g order by g",
    "zoned_interval": "select g, max(zm + interval 1 day) as m from t group by g order by g",
    "having_timestamp_case": "select g, max(case when id > 3 then tn end) as m from t group by g having max(case when id > 3 then tn end) > date '2024-01-02' order by g",
    "order_by_timestamp_agg": "select g, min(coalesce(tn, timestamp_ns '2200-01-01 00:00:00')) as m from t group by g order by m desc",
    # 64-bit integers past 2^53
    "min_max_coalesce_big": "select min(coalesce(big, 5)) as a, max(coalesce(big, 5)) as b from t",
    "min_max_case_big": "select min(case when id > 3 then big end) as a, max(case when id > 3 then big end) as b from t",
    "grouped_max_coalesce_big": "select g, max(coalesce(big, 5)) as m from t group by g order by g",
    "grouped_min_case_big_else": "select g, min(case when id > 3 then big else 9223372036854775807 end) as m from t group by g order by g",
    "grouped_big_two_columns": "select g, max(coalesce(big, id)) as m, min(coalesce(big, id)) as n from t group by g order by g",
}


@pytest.mark.parametrize("name", AGG)
def test_min_max_of_computed_value_equals_duckdb(name, cfg):
    got, want, tbl, wtbl = _run(AGG[name], cfg[0])
    assert got == want, (got, want)
    assert [_kind(f.type) for f in tbl.schema] == [_kind(f.type) for f in wtbl.schema]


@pytest.mark.parametrize("name", ["max_case_string", "grouped_min_max_coalesce", "grouped_dictionary"])
def test_string_results_are_strings(name, cfg):
    q = AGG[name] if name in AGG else AGG["grouped_min_max_coalesce"]
    _, _, tbl, _ = _run(q, cfg[0])
    assert pa.types.is_string(tbl.schema[-1].type) or pa.types.is_large_string(tbl.schema[-1].type)


def test_numeric_min_max_is_untouched(cfg):
    q = "select g, min(id * 2) as a, max(case when id > 3 then id else 0 end) as b, sum(id) as c from t group by g order by g"
    got, want, _, _ = _run(q, cfg[0])
    assert got == want


# -- a Parquet scan, a join, a subquery ---------------------------------------------------------------------------

def test_parquet_scan(tmp_path, cfg):
    path = str(tmp_path / "t.parquet")
    pq.write_table(T, path, row_group_size=4)
    for q in ("select id from t where tn > timestamp_ns '2024-01-01 17:00:00.000000001' order by id",
              "select id from t where tsec between timestamp '2024-01-01 12:00:00' and date '2024-01-03' order by id",
              "select g, min(tu) as m from t group by g having min(tu) > date '2024-01-01' order by g",
              "select g, max(case when id > 3 then s end) as m from t group by g order by g",
              "select g, max(case when id > 3 then zm end) as m from t group by g order by g"):
        want = CON.execute(q).to_arrow_table()
        got = mx.sql(q, t=mx.scan_parquet(path)).compute(device=cfg[0])
        assert _rows(got) == _rows(want), q


U = pa.table({"k": pa.array([0, 1, 2, 7, 9], pa.int64()), "lbl": pa.array(["zero", None, "two", "seven", "nine"])})
CON.register("_u", U)
CON.execute("create table u as select * from _u")


@pytest.mark.parametrize("q", [
    "select t.id from t join u on t.id = u.k where t.tu > timestamp '2024-01-01 10:00:00' order by t.id",
    "select u.lbl, max(case when t.id > 0 then t.tn end) as m from t join u on t.id = u.k group by u.lbl order by u.lbl",
    "select t.g, max(coalesce(u.lbl, 'none')) as m from t left join u on t.id = u.k group by t.g order by t.g",
    "select t.g, min(coalesce(t.tu, u.k)) as m from t left join u on t.id = u.k group by t.g order by t.g" if False else
    "select t.g, min(case when u.k is null then t.d end) as m from t left join u on t.id = u.k group by t.g order by t.g",
    "select id from t where tu > (select min(tu) from t where tu > timestamp '2024-01-02') order by id",
    "select x, count(*) as n from (select case when tu > timestamp '2024-01-02' then 'late' else 'early' end as x from t) q group by x order by x",
])
def test_join_and_subquery(q, cfg):
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T, u=U).compute(device=cfg[0])
    assert _rows(got) == _rows(want), q


@pytest.mark.parametrize("lit", ["timestamp_ns '2999-01-01 00:00:00'", "timestamp '2024-13-01 00:00:00'", "timestamp '2024-01-01 25:00:00'"])
def test_out_of_range_literal_raises(lit, cfg):
    with pytest.raises(Exception):
        CON.execute(f"select id from t where tn > {lit}")
    with pytest.raises(ValueError):
        mx.sql(f"select id from t where tn > {lit}", t=T).compute(device=cfg[0])


def test_invalid_timestamp_literal_raises(cfg):
    with pytest.raises(ValueError):
        mx.sql("select id from t where tu > timestamp 'not a time'", t=T).compute(device=cfg[0])


def test_month_interval_on_a_column_is_not_supported(cfg):
    with pytest.raises(NotImplementedError):
        mx.sql("select d + interval 1 month as c from t", t=T).compute(device=cfg[0])


@pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")
def test_date_filter_still_native(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    got, want, _, _ = _run("select g, sum(id) as s from t where id > 2 group by g order by g", "cpu")
    assert nx.last_path() == "native", nx.last_decline()
    assert got == want
