"""F-100: SQL COALESCE / IFNULL / NULLIF, ``expr IS [NOT] NULL`` over expressions, and ABS / CAST on float32.

Before: ``COALESCE(...)``, ``NULLIF(...)``, ``ABS(...)`` and ``CAST(x AS FLOAT | DOUBLE | INT ...)`` raised
"Unsupported SQL expression type" in every clause.

DuckDB's typing (probed): COALESCE takes the common type of its arguments, as CASE does (F-105): equal types stay,
integers widen, FLOAT next to an integer or a DECIMAL literal stays FLOAT and only a DOUBLE (exponent literal,
float64 column) widens it, text is text, a NULL argument takes the other's type. NULLIF(a, b) has a's type and is
NULL where a = b (NaN = NaN). ABS keeps its argument's type. CAST of a float to an integer rounds half to even
and raises when the value does not fit.

Expected answers come from a NATIVE DuckDB table, on cpu and gpu, native executor on and off.
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
NAN, INF = float("nan"), float("inf")


def _table():
    n = 12
    return pa.table({
        "id": pa.array(range(n), pa.int64()),
        "i": pa.array([None if k % 4 == 3 else k * 2 for k in range(n)], pa.int64()),            # NULL on 3, 7, 11
        "j": pa.array([None if k % 3 == 0 else k for k in range(n)], pa.int32()),                # NULL on 0, 3, 6, 9
        "b": pa.array([None if k % 5 == 2 else k - 6 for k in range(n)], pa.int8()),             # NULL on 2, 7
        "big": pa.array([None if k % 6 == 5 else 2 ** 60 + k for k in range(n)], pa.int64()),     # past 2^53
        "f": pa.array([None if k % 5 == 1 else NAN if k == 4 else INF if k == 7 else k * 1.5 - 6 for k in range(n)],
                      pa.float64()),
        "r": pa.array([None if k % 4 == 2 else k * 0.5 - 3 for k in range(n)], pa.float32()),    # NULL on 2, 6, 10
        "tiny": pa.array([None if k % 3 == 0 else (1e-30 if k % 2 else -1e-30) for k in range(n)], pa.float32()),
        "h": pa.array([None if k == 10 else k - 4.5 + (k % 2) for k in range(n)], pa.float64()),       # ... -0.5, 0.5, 1.5 ...
        "z": pa.array([None if k == 10 else k % 3 for k in range(n)], pa.int64()),               # zero divisors
        "s": pa.array(["hit", "miss", None, "hit", "other", "hit", "miss", None, "other", "miss", "hit", ""]),
        "ls": pa.array(["hit", None, "x", "hit", "other", None, "miss", "y", "other", "miss", "hit", "z"],
                       pa.large_string()),
        "ds": pa.array(["a", None, "c", "a", None, "b", "c", "a", "b", "c", "a", "b"]).dictionary_encode(),
        "d": pa.array([None if k % 5 == 2 else D0 + dt.timedelta(days=k * 40) for k in range(n)], pa.date32()),
        "g": pa.array([k % 3 for k in range(n)], pa.int32()),
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


def _same(a, b):
    """Rows equal; NaN equals NaN."""
    return len(a) == len(b) and all(
        len(x) == len(y) and all(p == q or (p != p and q != q) for p, q in zip(x, y)) for x, y in zip(a, b))


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


# (query, kinds of the result columns). ORDER BY id makes the order the same as DuckDB's.
SELECT = {
    # COALESCE: columns, literals, several arguments, every type
    "int_literal": ("select id, coalesce(i, 5) as c from t order by id", ["int", "int"]),
    "int_negative_literal": ("select id, coalesce(i, -1) as c from t order by id", ["int", "int"]),
    "ifnull": ("select id, ifnull(i, 5) as c from t order by id", ["int", "int"]),
    "two_columns": ("select id, coalesce(i, j) as c from t order by id", ["int", "int"]),
    "three_args": ("select id, coalesce(i, j, 0) as c from t order by id", ["int", "int"]),
    "still_null": ("select id, coalesce(i, j) as c from t where id in (3, 6, 9) order by id", ["int", "int"]),
    "single_arg": ("select id, coalesce(i) as c from t order by id", ["int", "int"]),
    "null_literal_first": ("select id, coalesce(null, i) as c from t order by id", ["int", "int"]),
    "null_literal_last": ("select id, coalesce(i, null) as c from t order by id", ["int", "int"]),
    "int8_and_big_literal": ("select id, coalesce(b, 1000) as c from t order by id", ["int", "int"]),
    "int8_and_int64": ("select id, coalesce(b, i) as c from t order by id", ["int", "int"]),
    "past_2_53": ("select id, coalesce(big, 7) as c from t order by id", ["int", "int"]),
    "past_2_53_two_cols": ("select id, coalesce(big, i) as c from t order by id", ["int", "int"]),
    "int_decimal_literal": ("select id, coalesce(i, 5.5) as c from t order by id", ["int", "float"]),
    "int_double_literal": ("select id, coalesce(i, 1e0) as c from t order by id", ["int", "float"]),
    "int_float_col": ("select id, coalesce(i, f) as c from t order by id", ["int", "float"]),
    "float_col": ("select id, coalesce(f, 0) as c from t order by id", ["int", "float"]),
    "float_nan_is_not_null": ("select id, coalesce(f, -1.0) as c from t order by id", ["int", "float"]),
    "string_literal": ("select id, coalesce(s, 'x') as c from t order by id", ["int", "string"]),
    "empty_string_is_not_null": ("select id, coalesce(s, 'x') as c from t where id = 11", ["int", "string"]),
    "string_columns": ("select id, coalesce(s, ls) as c from t order by id", ["int", "string"]),
    "string_three": ("select id, coalesce(s, ls, 'none') as c from t order by id", ["int", "string"]),
    "large_string": ("select id, coalesce(ls, 'x') as c from t order by id", ["int", "string"]),
    "dictionary": ("select id, coalesce(ds, 'x') as c from t order by id", ["int", "string"]),
    "dictionary_and_large": ("select id, coalesce(ds, ls) as c from t order by id", ["int", "string"]),
    "date_literal": ("select id, coalesce(d, date '2020-01-01') as c from t order by id", ["int", "date"]),
    "date_column_literal_chain": ("select id, coalesce(d, date '2020-01-01') as c from t where id < 6 order by id",
                                  ["int", "date"]),
    "case_inside": ("select id, coalesce(case when g = 0 then i end, 99) as c from t order by id", ["int", "int"]),
    "case_outside": ("select id, case when coalesce(i, 0) > 10 then 'big' else 'small' end as c from t order by id",
                     ["int", "string"]),
    "arithmetic": ("select id, coalesce(i, 0) + coalesce(j, 0) as c from t order by id", ["int", "int"]),
    "arithmetic_float": ("select id, coalesce(f, 0.0) * 2 as c from t order by id", ["int", "float"]),
    "nested": ("select id, coalesce(coalesce(i, j), 100) as c from t order by id", ["int", "int"]),
    "float32_integer": ("select id, coalesce(r, 0) as c from t order by id", ["int", "float"]),
    "float32_decimal": ("select id, coalesce(r, 0.1) as c from t order by id", ["int", "float"]),
    "float32_double": ("select id, coalesce(r, 1e-1) as c from t order by id", ["int", "float"]),
    "float32_float64": ("select id, coalesce(r, f) as c from t order by id", ["int", "float"]),
    "float32_two": ("select id, coalesce(tiny, r) as c from t order by id", ["int", "float"]),
    # NULLIF
    "nullif_int": ("select id, nullif(i, 0) as c from t order by id", ["int", "int"]),
    "nullif_column": ("select id, nullif(i, j) as c from t order by id", ["int", "int"]),
    "nullif_int_decimal": ("select id, nullif(i, 4.0) as c from t order by id", ["int", "int"]),
    "nullif_string": ("select id, nullif(s, 'hit') as c from t order by id", ["int", "string"]),
    "nullif_empty_string": ("select id, nullif(s, '') as c from t order by id", ["int", "string"]),
    "nullif_float_nan": ("select id, nullif(f, 'nan'::double) as c from t order by id", ["int", "float"]),
    "nullif_float32": ("select id, nullif(r, 1.5) as c from t order by id", ["int", "float"]),
    "nullif_date": ("select id, nullif(d, date '2024-02-10') as c from t order by id", ["int", "date"]),
    "nullif_null": ("select id, nullif(i, null) as c from t order by id", ["int", "int"]),
    "nullif_zero_divisor": ("select id, i / nullif(z, 0) as c from t order by id", ["int", "float"]),
    "nullif_zero_divisor_float": ("select id, 10.0 / nullif(z, 0) as c from t order by id", ["int", "float"]),
    "nullif_then_coalesce": ("select id, coalesce(nullif(i, 0), -1) as c from t order by id", ["int", "int"]),
    "nullif_of_coalesce": ("select id, nullif(coalesce(i, 0), 0) as c from t order by id", ["int", "int"]),
    # IS [NOT] NULL over expressions
    "is_null_sum": ("select id, (i + j) is null as c from t order by id", ["int", "bool"]),
    "is_not_null_sum": ("select id, (i + j) is not null as c from t order by id", ["int", "bool"]),
    "is_null_product": ("select id, i * j * 2 is null as c from t order by id", ["int", "bool"]),
    "is_null_coalesce": ("select id, coalesce(i, j) is null as c from t order by id", ["int", "bool"]),
    "is_null_coalesce_total": ("select id, coalesce(i, 0) is null as c from t order by id", ["int", "bool"]),
    "is_null_nullif": ("select id, nullif(i, 0) is null as c from t order by id", ["int", "bool"]),
    "is_null_case": ("select id, case when g = 1 then i end is null as c from t order by id", ["int", "bool"]),
    "is_null_case_string": ("select id, case when g = 1 then s end is not null as c from t order by id",
                            ["int", "bool"]),
    "is_null_division": ("select id, i / z is null as c from t order by id", ["int", "bool"]),
    "is_null_negation": ("select id, -i is null as c from t order by id", ["int", "bool"]),
    "is_null_abs": ("select id, abs(i) is null as c from t order by id", ["int", "bool"]),
    "is_null_cast": ("select id, cast(f as float) is null as c from t order by id", ["int", "bool"]),
    "is_null_float_nan": ("select id, (f + 1) is null as c from t order by id", ["int", "bool"]),
    "is_null_comparison": ("select id, (i > 5) is null as c from t order by id", ["int", "bool"]),
    "is_null_and": ("select id, (i > 5 and j > 5) is null as c from t order by id", ["int", "bool"]),
    "is_null_substring": ("select id, substring(s, 1, 2) is null as c from t order by id", ["int", "bool"]),
    "is_null_date_case": ("select id, case when g = 1 then d end is null as c from t order by id", ["int", "bool"]),
    "is_null_coalesce_string": ("select id, coalesce(s, ls) is null as c from t order by id", ["int", "bool"]),
    # ABS and CAST
    "abs_int": ("select id, abs(i - 10) as c from t order by id", ["int", "int"]),
    "abs_int8": ("select id, abs(b) as c from t order by id", ["int", "int"]),
    "abs_float": ("select id, abs(f) as c from t order by id", ["int", "float"]),
    "abs_float32": ("select id, abs(r) as c from t order by id", ["int", "float"]),
    "abs_coalesce": ("select id, abs(coalesce(i, -5)) as c from t order by id", ["int", "int"]),
    "abs_plus_int": ("select id, abs(i) + 1 as c from t order by id", ["int", "int"]),
    "cast_float32": ("select id, cast(r as float) as c from t order by id", ["int", "float"]),
    "cast_float32_shorthand": ("select id, r::float as c from t order by id", ["int", "float"]),
    "cast_double": ("select id, cast(r as double) as c from t order by id", ["int", "float"]),
    "cast_int_to_float": ("select id, cast(big as float) as c from t order by id", ["int", "float"]),
    "cast_int_to_double": ("select id, cast(i as double) as c from t order by id", ["int", "float"]),
    "cast_double_to_float": ("select id, cast(f as float) as c from t where id <> 7 order by id", ["int", "float"]),
    "cast_half_to_even": ("select id, cast(h as int) as c from t order by id", ["int", "int"]),
    "cast_float32_half_to_even": ("select id, cast(r as int) as c from t order by id", ["int", "int"]),
    "cast_double_to_bigint": ("select id, cast(f as bigint) as c from t where id not in (4, 7) order by id",
                              ["int", "int"]),
    "cast_int_to_smallint": ("select id, cast(i as smallint) as c from t order by id", ["int", "int"]),
    "cast_coalesce": ("select id, cast(coalesce(i, 0) as double) / 4 as c from t order by id", ["int", "float"]),
}


@pytest.mark.parametrize("name", SELECT)
def test_select_equals_duckdb(name, cfg):
    q, kinds = SELECT[name]
    dev, _ = cfg
    got, want, tbl, _ = _run(q, dev)
    assert _same(got, want), (got, want)
    assert [_kind(f.type) for f in tbl.schema] == kinds, [str(f.type) for f in tbl.schema]


FLOAT_TYPE = {"float32_integer": pa.float32(), "float32_decimal": pa.float32(), "float32_double": pa.float64(),
              "float32_float64": pa.float64(), "float32_two": pa.float32(), "nullif_float32": pa.float32(),
              "abs_float32": pa.float32(), "cast_float32": pa.float32(), "cast_float32_shorthand": pa.float32(),
              "cast_double": pa.float64(), "cast_int_to_float": pa.float32(), "cast_double_to_float": pa.float32(),
              "abs_float": pa.float64(), "int_float_col": pa.float64()}


@pytest.mark.parametrize("name", FLOAT_TYPE)
def test_float_result_type_is_duckdbs(name, cfg):
    """FLOAT stays FLOAT through COALESCE next to an integer or DECIMAL literal, ABS, NULLIF and CAST; only a
    DOUBLE widens it. The type is what DuckDB returns."""
    got, want, tbl, wtbl = _run(SELECT[name][0], cfg[0])
    assert _same(got, want)
    assert tbl.column("c").type == FLOAT_TYPE[name] == wtbl.column("c").type


def test_float32_decimal_literal_value(cfg):
    got, _, _, _ = _run(SELECT["float32_decimal"][0], cfg[0])
    assert got[2] == (2, 0.10000000149011612)           # r is NULL on row 2: the DECIMAL literal as a float32


def test_integer_results_keep_exact_values_past_2_53(cfg):
    got, want, tbl, _ = _run(SELECT["past_2_53"][0], cfg[0])
    assert got == want and got[0] == (0, 2 ** 60) and got[5] == (5, 7)
    assert tbl.column("c").type == pa.int64()


def test_text_mixed_with_numbers_raises(cfg):
    with pytest.raises(Exception):
        mx.sql("select coalesce(i, 'x') as c from t", t=T).compute(device=cfg[0])


def test_constants_only_has_no_column_to_compute_on(cfg):
    with pytest.raises(NotImplementedError):
        mx.sql("select coalesce(null, 5) as c from t", t=T).compute(device=cfg[0])


@pytest.mark.parametrize("q", ["select cast(big as int) as c from t", "select cast(i as tinyint) + cast(big as int) as c from t",
                               "select cast(f as int) as c from t where f > 1e30 or f = 'inf'::double"])
def test_cast_out_of_range_raises(q, cfg):
    with pytest.raises(Exception):
        CON.execute(q)                                      # DuckDB raises too
    with pytest.raises(Exception):
        mx.sql(q, t=T).compute(device=cfg[0])


def test_abs_of_the_minimum_integer_raises(cfg):
    m = pa.table({"x": pa.array([-(2 ** 63), 1], pa.int64())})
    with pytest.raises(Exception):
        mx.sql("select abs(x) as c from m", m=m).compute(device=cfg[0])


# -- every clause -------------------------------------------------------------------------------------------

CLAUSES = {
    "where_coalesce": "select id from t where coalesce(i, -1) < 5 order by id",
    "where_coalesce_string": "select id from t where coalesce(s, 'x') = 'x' order by id",
    "where_coalesce_dictionary": "select id from t where coalesce(ds, 'z') in ('a', 'z') order by id",
    "where_coalesce_date": "select id from t where coalesce(d, date '2000-01-01') > date '2024-06-01' order by id",
    "where_coalesce_or": "select id from t where coalesce(i, 100) > 10 or coalesce(j, 100) < 3 order by id",
    "where_nullif": "select id from t where nullif(i, 4) is not null and i > 0 order by id",
    "where_nullif_divisor": "select id from t where i / nullif(z, 0) > 3 order by id",
    "where_is_null_expr": "select id from t where (i + j) is null order by id",
    "where_is_not_null_expr": "select id from t where (i + j) is not null order by id",
    "where_is_null_coalesce": "select id from t where coalesce(i, j) is null order by id",
    "where_not_is_null_expr": "select id from t where not ((i + j) is null) order by id",
    "where_is_null_case": "select id from t where case when g = 1 then i end is null order by id",
    "where_is_null_string_case": "select id from t where case when g = 0 then s end is null order by id",
    "where_is_null_or": "select id from t where (i * 2) is null or (j * 2) is null order by id",
    "where_abs": "select id from t where abs(b) > 3 order by id",
    "where_abs_float32_widened": "select id from t where abs(tiny) > 1e-30 order by id",
    "where_coalesce_float32_widened": "select id from t where coalesce(tiny, 0) > 1e-30 order by id",
    "where_cast_float32_widened": "select id from t where cast(tiny as float) > 1e-30 order by id",
    "where_float32_decimal": "select id from t where coalesce(r, 0) = 0.5 order by id",
    "where_cast_float32_decimal": "select id from t where cast(r as float) = 0.5 order by id",
    "group_by_coalesce": "select coalesce(i, -1) as k, count(*) as n from t group by 1 order by 1",
    "group_by_coalesce_expr": "select coalesce(i, -1) as k, sum(f) as s from t group by coalesce(i, -1) order by k",
    "group_by_coalesce_string": "select coalesce(s, 'none') as k, count(*) as n, sum(i) as t from t group by 1 order by 1",
    "group_by_coalesce_dictionary": "select coalesce(ds, 'none') as k, count(*) as n from t group by 1 order by 1",
    "group_by_coalesce_date": "select coalesce(d, date '2000-01-01') as k, count(*) as n from t group by 1 order by 1",
    "group_by_nullif": "select nullif(g, 1) as k, count(*) as n from t group by 1 order by 1",
    "group_by_is_null": "select (i + j) is null as k, count(*) as n from t group by 1 order by 1",
    "group_by_is_null_sum": "select g, sum(case when (i + j) is null then 1 else 0 end) as n from t group by g order by g",
    "group_by_two": "select g, coalesce(s, 'none') as k, count(*) as n from t group by g, 2 order by g, k",
    "group_by_abs": "select abs(b) as k, count(*) as n from t group by 1 order by 1",
    "order_by_coalesce": "select id from t order by coalesce(i, 1000), id",
    "order_by_coalesce_desc": "select id from t order by coalesce(i, -1) desc, id",
    "order_by_coalesce_string": "select id from t order by coalesce(s, 'zzz'), id",
    "order_by_nullif": "select id from t order by nullif(i, 0), id",
    "order_by_is_null": "select id from t order by (i + j) is null, id",
    "order_by_abs": "select id from t order by abs(b), id",
    "order_by_alias": "select id, coalesce(i, 100) as c from t order by c, id",
    "limit_after_coalesce": "select id, coalesce(i, 100) as c from t order by c desc, id limit 4",
}


@pytest.mark.parametrize("name", CLAUSES)
def test_clause_equals_duckdb(name, cfg):
    got, want, _, _ = _run(CLAUSES[name], cfg[0])
    assert _same(got, want), (got, want)


AGG = {
    "sum": "select sum(coalesce(i, 0)) as c from t",
    "sum_float": "select sum(coalesce(f, 0.0)) as c from t where f is null or f < 100 and f = f",
    "sum_columns": "select sum(coalesce(i, j)) as c from t",
    "avg": "select avg(coalesce(i, 0)) as c from t",
    "avg_vs_plain": "select avg(i) as a, avg(coalesce(i, 0)) as c from t",
    "count_coalesce": "select count(coalesce(i, j)) as c from t",
    "count_nullif": "select count(nullif(i, 0)) as c from t",
    "count_is_null_expr": "select count(*) as c from t where (i + j) is null",
    "count_case_is_null": "select sum(case when (i + j) is null then 1 else 0 end) as c from t",
    "min_max": "select min(coalesce(i, 100)) as a, max(coalesce(i, -100)) as b from t",
    "sum_abs": "select sum(abs(b)) as c from t",
    "sum_nullif": "select sum(nullif(i, 4)) as c from t",
    "ratio_nullif": "select sum(i) / nullif(sum(z), 0) as c from t",
    "ratio_nullif_zero": "select sum(i) / nullif(sum(z) - sum(z), 0) as c from t",
    "grouped_sum": "select g, sum(coalesce(i, 0)) as c from t group by g order by g",
    "grouped_avg": "select g, avg(coalesce(f, 0.0)) as c from t where f = f group by g order by g",
    "grouped_count": "select g, count(coalesce(i, j)) as c from t group by g order by g",
    "grouped_sum_float32": "select g, sum(coalesce(r, 0)) as c from t group by g order by g",
    "grouped_abs": "select g, sum(abs(b)) as c from t group by g order by g",
    "having_coalesce": "select g, sum(coalesce(i, 0)) as c from t group by g having sum(coalesce(i, 0)) > 20 order by g",
    "having_is_null": "select g, count(*) as n from t group by g having count(*) > 1 and sum(coalesce(i, 0)) is not null "
                      "order by g",
    "having_nullif": "select g, sum(i) as c from t group by g having nullif(sum(coalesce(i, 0)), 0) > 20 order by g",
    "having_coalesce_min": "select g, min(i) as c from t group by g having coalesce(min(i), -1) >= 0 order by g",
    "having_coalesce_of_agg": "select g, max(i) as c from t where id > 8 group by g having coalesce(max(i), -1) > 10 order by g",
    "order_by_agg_coalesce": "select g, sum(i) as c from t group by g order by coalesce(sum(i), 0) desc, g",
}


@pytest.mark.parametrize("name", AGG)
def test_aggregate_equals_duckdb(name, cfg):
    got, want, _, _ = _run(AGG[name], cfg[0])
    assert _same(got, want), (got, want)


# -- joins and subqueries -----------------------------------------------------------------------------------

U = pa.table({"k": pa.array([0, 1, 2, 7], pa.int64()), "w": pa.array([None, 5, None, 9], pa.int64()),
              "lbl": pa.array(["zero", None, "two", "seven"])})
CON.register("_u", U)
CON.execute("create table u as select * from _u")


def _run2(q, dev):
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T, u=U).compute(device=dev)
    rows = lambda x: [tuple(_norm(c[r]) for c in (x.column(i).to_pylist() for i in range(x.num_columns)))
                      for r in range(x.num_rows)]
    return rows(got), rows(want)


JOIN = {
    "left_join_coalesce": "select t.id, coalesce(u.w, -1) as c from t left join u on t.id = u.k order by t.id",
    "left_join_coalesce_string": "select t.id, coalesce(u.lbl, 'none') as c from t left join u on t.id = u.k order by t.id",
    "left_join_is_null_expr": "select t.id from t left join u on t.id = u.k where (u.w + 1) is null order by t.id",
    "left_join_coalesce_two_tables": "select t.id, coalesce(u.w, t.i, 0) as c from t left join u on t.id = u.k order by t.id",
    "left_join_group": "select coalesce(u.lbl, 'none') as k, count(*) as n from t left join u on t.id = u.k group by 1 order by 1",
    "left_join_sum": "select sum(coalesce(u.w, t.i, 0)) as c from t left join u on t.id = u.k",
    "inner_join_nullif": "select t.id, nullif(u.w, 5) as c from t join u on t.id = u.k order by t.id",
    "subquery": "select id from t where coalesce(i, -1) > (select avg(coalesce(i, 0)) from t) order by id",
    "derived_table": "select x, count(*) as n from (select coalesce(s, 'none') as x from t) q group by x order by x",
    "derived_table_is_null": "select id from (select id, i + j as ij from t) q where ij is null order by id",
}


@pytest.mark.parametrize("name", JOIN)
def test_join_and_subquery_equal_duckdb(name, cfg):
    got, want = _run2(JOIN[name], cfg[0])
    assert _same(got, want), (got, want)


# -- the native executor still answers what it did, and hands COALESCE to Arrow -------------------------------

@pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")
def test_aggregate_without_coalesce_stays_native(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    got, want, _, _ = _run("select g, sum(id) as s from t group by g order by g", "cpu")
    assert nx.last_path() == "native", nx.last_decline()
    assert _same(got, want)
