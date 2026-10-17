"""F-118: a constant in a plain SELECT list keeps its type.

Before: ``select id, 1 as c from t`` returned c as float32 (the engine broadcast an integer literal as a float);
``true`` raised "Unsupported SQL expression type: Boolean".

DuckDB (probed): an integer literal is INTEGER (int32) when it fits, else BIGINT; TRUE / FALSE are BOOLEAN; a
decimal literal is a DECIMAL (a float here); a string is VARCHAR; DATE '..' is a DATE; a bare NULL is INTEGER
(all NULL); arithmetic on integer literals is INTEGER. Expected answers come from a native DuckDB table, column
types compared (INTEGER is not BIGINT).
"""
import pyarrow as pa
import pytest

import mxframe as mx
from _sql_reference import T, check, cfg  # noqa: F401  (cfg is the fixture)

CONSTANTS = {
    "int": "1", "negative": "-7", "int32_min": "-2147483648", "above_int32_min": "-2147483647", "zero": "0", "big": "3000000000", "int32_max": "2147483647",
    "past_int32": "2147483648", "float": "1.5", "double": "1e0", "string": "'a'", "empty_string": "''",
    "quoted": "'it''s'", "date": "date '2020-01-02'", "true": "true", "false": "false", "null": "null",
    "sum": "1 + 1", "product": "2 * 3", "paren": "(5)", "neg_paren": "-(1)", "plus": "+1",
}


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_next_to_a_column(cfg, name):
    check(f"select id, {CONSTANTS[name]} as c from t order by id limit 3", cfg[0])


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_over_no_rows_keeps_its_type(cfg, name):
    want = check(f"select id, {CONSTANTS[name]} as c from t where id > 100", cfg[0])
    assert want.num_rows == 0


def test_every_constant_together(cfg):
    q = ("select id, 1 as i, 3000000000 as big, 1.5 as f, 'a' as s, date '2020-01-02' as d, true as b, false as b2, "
         "null as z, 1 + 1 as e from t order by id")
    assert check(q, cfg[0]).num_rows == 12


def test_constant_only_select(cfg):
    check("select 1 as c, 'a' as s, true as b from t limit 2", cfg[0])


def test_the_values_not_only_the_types(cfg):
    got = mx.sql("select id, 1 as c, true as b from t order by id limit 2", t=T).compute(device=cfg[0])
    assert got.schema.types == [pa.int64(), pa.int32(), pa.bool_()]
    assert got.to_pylist() == [{"id": 0, "c": 1, "b": True}, {"id": 1, "c": 1, "b": True}]


@pytest.mark.parametrize("q", [
    "select id, 1 as c from t where x > 3 order by id",
    "select id, 1 as c from t order by c, id limit 3",                       # ORDER BY the constant
    "select id, 1 as c from t order by id desc limit 2",
    "select id, 1 as c, x * 2 as h from t order by id limit 3",              # next to a computed column
    "select x + 1 as h, 1 as c from t order by h, c limit 4",
    "select id, 1 as c from t order by id limit 2 offset 3",
    "select distinct g, 1 as c from t order by g",
])
def test_constant_with_filter_order_distinct(cfg, q):
    check(q, cfg[0])


@pytest.mark.parametrize("q", [
    "select id, case when x > 3 then 1 else 0 end as k from t order by id",
    "select id, coalesce(x, 0) as k from t order by id",
    "select id, case when x > 3 then true else false end as k from t order by id",
    "select id, case when x > 3 then 'a' else 'b' end as k from t order by id",
    "select id, case when x > 3 then 1.5 else 2 end as k from t order by id",
])
def test_constants_inside_case_and_coalesce_keep_their_values(cfg, q):
    check(q, cfg[0], types=False)       # DuckDB types these INTEGER; the values are what is checked here


def test_arithmetic_on_a_constant_column_is_unchanged(cfg):
    # the constant column is no longer a float: id + 1, x * 2 and 1 / 2 keep DuckDB's types and values
    check("select id + 1 as a, x * 2 as b, 7 / 2 as c, 7 % 4 as e from t order by id limit 3", cfg[0])


def test_a_constant_beside_aggregates_has_the_same_type(cfg):
    check("select g, 1 as c, count(*) as n from t group by g order by g", cfg[0])
    check("select 1 as c, true as b, count(*) as n from t", cfg[0])
