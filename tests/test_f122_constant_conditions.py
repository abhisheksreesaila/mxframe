"""F-122: constant conditions in WHERE, HAVING, ON, CASE and under NOT.

Before: ``select * from t where 1 = 1`` raised ``ArrowTypeError: Filter should be array-like``; ``WHERE TRUE`` /
``FALSE`` / ``NULL`` raised "Unsupported SQL expression type: Boolean"; ``HAVING 1 = 0`` and ``ON ... AND 1 = 1``
raised the same family of errors.

DuckDB (probed): a constant condition is evaluated in three-valued logic: true keeps every row, false and NULL
keep none, and the result still has the query's columns and types (also under an aggregate: ``count(*)`` over no
rows is one row of 0). ``WHERE 1`` is true, ``WHERE 0`` false (a number is its truth value). ``x OR TRUE`` is true
(also for a NULL x), ``x AND FALSE`` false, ``NOT NULL`` NULL. In a CASE a false or NULL condition falls to the
next branch. In an ON a false or NULL condition matches nothing: an INNER JOIN is empty, a LEFT JOIN keeps its
left rows with NULLs. Expected answers come from a native DuckDB table.
"""
import pytest

from _sql_reference import check, cfg  # noqa: F401  (cfg is the fixture)

TRUE_ = ["1 = 1", "true", "1", "2 > 1", "'a' = 'a'", "not false", "not (1 = 0)", "1 = 1 and true", "false or true",
         "null is null", "1 is not null", "not null is not null", "true is true", "-1 < 0",
         "1.5 > 1", "0.1 + 0.2 = 0.3", "1 in (1, 2)", "'true'", "'a' in ('a')", "'a' < 'b'", "(1 = 1)", "((true))", "not not true", "null or true", "1 <> 2", "2 >= 2", "1 <= 1"]
FALSE_ = ["1 = 0", "false", "0", "1 > 2", "'a' = 'b'", "not true", "not (1 = 1)", "1 = 1 and false", "false and true",
          "null is not null", "1 is null", "true is false", "null is true", "(1 = 0)", "0.0 = 1", "1 > 1",
          "not 1", "'b' < 'a'", "not not false", "0.1e0 + 0.2e0 = 0.3e0", "1e0 = 2", "1 in (3)", "'false'"]
NULL_ = ["null", "not null", "1 = null", "null = null", "null and true", "null or false", "null > 1", "null <> null",
         "(null)", "not (null = 1)", "not not null", "null and null"]


@pytest.mark.parametrize("cond", TRUE_)
def test_a_true_constant_keeps_every_row(cfg, cond):
    want = check(f"select id, g, gi, x, f, d from t where {cond} order by id", cfg[0])
    assert want.num_rows == 12


@pytest.mark.parametrize("cond", FALSE_ + NULL_)
def test_a_false_or_null_constant_keeps_no_row_and_the_columns(cfg, cond):
    want = check(f"select id, g, gi, x, f, d from t where {cond}", cfg[0])
    assert want.num_rows == 0 and want.column_names == ["id", "g", "gi", "x", "f", "d"]


@pytest.mark.parametrize("cond", ["1 = 1", "true", "1 = 0", "false", "null", "not null", "1", "0"])
def test_in_a_projection_and_with_order_and_limit(cfg, cond):
    check(f"select id, x * 2 as h from t where {cond} order by id limit 3", cfg[0])
    check(f"select id from t where {cond} order by id desc", cfg[0])
    check(f"select distinct g from t where {cond} order by g", cfg[0])


@pytest.mark.parametrize("cond", ["1 = 1", "true", "1 = 0", "false", "null", "not true", "1"])
def test_constant_where_with_aggregates(cfg, cond):
    check(f"select count(*) as n, max(x) as s, max(f) as m from t where {cond}", cfg[0])
    check(f"select g, count(*) as n, max(x) as s from t where {cond} group by g order by g", cfg[0])
    check(f"select gi, count(*) as n from t where {cond} group by gi order by gi", cfg[0])


@pytest.mark.parametrize("q", [
    "select id from t where 1 = 1 and id > 9 order by id",
    "select id from t where id > 9 and 1 = 1 order by id",
    "select id from t where 1 = 0 or id > 9 order by id",
    "select id from t where id > 9 or 1 = 0 order by id",
    "select id from t where id > 9 or true order by id",                         # true for every row, also NULL ids
    "select id from t where x > 3 or true order by id",                          # x NULL: NULL or TRUE is TRUE
    "select id from t where x > 3 or 1 = 1 order by id",
    "select id from t where x > 3 and false order by id",
    "select id from t where x > 3 and 1 = 0 order by id",
    "select id from t where x > 3 and null order by id",
    "select id from t where x > 3 or null order by id",
    "select id from t where x > 3 and true order by id",
    "select id from t where x > 3 or false order by id",
    "select id from t where not (x > 3 and false) order by id",                  # NOT (x AND FALSE) = NOT FALSE = TRUE
    "select id from t where not (x > 3 or true) order by id",
    "select id from t where not (x > 3 or false) order by id",                   # NULL x: NOT NULL is NULL, not kept
    "select id from t where not (x > 3 and true) order by id",
    "select id from t where not (x > 3 or null) order by id",                    # NOT (x OR NULL): never TRUE (F-122 review)
    "select id from t where not (x > 3 and null) order by id",
    "select id from t where not (null or null) order by id",
    "select id from t where not (x is null and null) order by id",
    "select id from t where not (not (x > 3 or null)) order by id",
    "select id from t where not (1 = 1 and x > 3) order by id",
    "select id from t where (x > 3 and 1 = 1) or (x < 3 and 1 = 0) order by id",
    "select id from t where 1 = 1 and (g = 'a' or 1 = 0) and not (1 = 0) order by id",
    "select id from t where x is null and 1 = 1 order by id",
    "select id from t where g in ('a', 'b') and 1 = 1 order by id",
    "select id from t where 1 = 1 and g like 'a%' order by id",
    "select id from t where 1 = 1 and x between 2 and 6 order by id",
    "select id from t where 1 = 1 and x > (select min(x) from t) order by id",
    "select id from t where 1 = 1 and exists (select 1 from u where u.k = t.gi) order by id",
    "select id from t where 1 = 0 and exists (select 1 from u where u.k = t.gi) order by id",
    "select id from t where 1 = 1 and gi in (select k from u) order by id",
    "select id from t where gi in (select k from u where 1 = 1) order by id",
    "select id from t where gi in (select k from u where 1 = 0) order by id",
    "select id from t where exists (select 1 from u where 1 = 0 and u.k = t.gi) order by id",
    "select id from t where not exists (select 1 from u where 1 = 0 and u.k = t.gi) order by id",
    "select id from t where x > (select min(x) from t where 1 = 0) order by id",       # NULL comparison: nothing
    "select t.id from t, u where 1 = 1 and t.gi = u.k order by 1",
    "select * from (select id, x from t where 1 = 0) order by id",
    "select * from (select id, x from t where 1 = 1) where x > 8 order by id",
    "select id from t where 1 = 1 and 2 = 2 and 3 = 3 order by id limit 3",
])
def test_constants_mixed_with_real_conditions(cfg, q):
    check(q, cfg[0])


# ---------------------------------------------------------------- HAVING

@pytest.mark.parametrize("cond", ["1 = 1", "true", "1", "1 = 0", "false", "0", "null", "not null", "not true"])
def test_constant_having(cfg, cond):
    check(f"select g, count(*) as n from t group by g having {cond} order by g", cfg[0])
    check(f"select count(*) as n from t having {cond}", cfg[0])
    check(f"select gi, max(x) as s from t group by gi having {cond} order by gi", cfg[0])


@pytest.mark.parametrize("q", [
    "select g, count(*) as n from t group by g having n > 2 and 1 = 1 order by g",
    "select g, count(*) as n from t group by g having n > 2 or 1 = 0 order by g",
    "select g, count(*) as n from t group by g having n > 2 or true order by g",
    "select g, count(*) as n from t group by g having n > 2 and false order by g",
    "select g, count(*) as n from t group by g having not (n > 2 and false) order by g",
    "select g, count(*) as n from t group by g having 1 = 1 and count(*) > 2 order by g",
])
def test_having_constants_mixed_with_real_conditions(cfg, q):
    check(q, cfg[0])


# ---------------------------------------------------------------- CASE

@pytest.mark.parametrize("q", [
    "select id, case when 1 = 1 then 'a' else 'b' end as c from t order by id limit 3",
    "select id, case when true then 'a' else 'b' end as c from t order by id limit 3",
    "select id, case when 1 = 0 then 'a' else 'b' end as c from t order by id limit 3",
    "select id, case when false then 'a' else 'b' end as c from t order by id limit 3",
    "select id, case when null then 'a' else 'b' end as c from t order by id limit 3",
    "select id, case when x > 3 then 'big' when 1 = 1 then 'any' else 'none' end as c from t order by id",
    "select id, case when 1 = 0 then 'x' when x > 3 then 'big' else 'small' end as c from t order by id",
    "select id, case when x > 3 then 'big' when 1 = 0 then 'never' else 'small' end as c from t order by id",
    "select id, case when null then 1 when x > 3 then 2 else 3 end as c from t order by id",
    "select id, case when x > 3 and 1 = 1 then 'big' else 'small' end as c from t order by id",
    "select id, case when x > 3 or true then 'big' else 'small' end as c from t order by id",
    "select id, case when not (1 = 1) then 1 else x end as c from t order by id",
    "select id, case when 1 = 1 then x else 0 end as c from t order by id",
    "select id, case when 1 = 1 then x end as c from t order by id",
    "select id, case when 1 = 0 then x else f end as c from t order by id",
    "select id, case when 1 = 0 then x end as c from t order by id",
    "select id from t where case when 1 = 1 then x > 3 else false end order by id",
    "select id from t where case when 1 = 0 then true else x > 3 end order by id",
    "select g, sum(case when 1 = 1 then x else 0 end) as s from t group by g order by g",
    "select g, sum(case when 1 = 0 then x else 1 end) as s from t group by g order by g",
])
def test_constant_conditions_in_case(cfg, q):
    check(q, cfg[0], types=False)


# ---------------------------------------------------------------- ON

@pytest.mark.parametrize("on", [
    "t.gi = u.k and 1 = 1", "t.gi = u.k and true", "1 = 1 and t.gi = u.k", "t.gi = u.k and not (1 = 0)",
    "t.gi = u.k and 1 = 0", "t.gi = u.k and false", "t.gi = u.k and null", "1 = 0 and t.gi = u.k",
    "1 = 1", "true", "1 = 0", "false", "null",
])
@pytest.mark.parametrize("how", ["join", "left join"])
def test_constant_conditions_in_on(cfg, how, on):
    check(f"select t.id, u.w from t {how} u on {on} where t.id < 4 order by t.id, u.w", cfg[0])


def test_on_true_is_a_cross_join(cfg):
    want = check("select t.id, u.w from t join u on 1 = 1 where t.id < 2 order by t.id, u.w", cfg[0])
    assert want.num_rows == 10


def test_a_false_on_of_a_left_join_keeps_the_left_rows(cfg):
    want = check("select t.id, u.w from t left join u on 1 = 0 order by t.id", cfg[0])
    assert want.num_rows == 12 and set(want.column("w").to_pylist()) == {None}


# ---------------------------------------------------------------- the same condition elsewhere

def test_a_constant_condition_in_a_subquery_and_a_cte(cfg):
    check("with c as (select id, x from t where 1 = 1) select id from c where x > 8 order by id", cfg[0])
    check("with c as (select id, x from t where 1 = 0) select id from c order by id", cfg[0])
    check("select id from t where id in (select id from t where 1 = 1 and x > 8) order by id", cfg[0])
    check("select (select count(*) from t where 1 = 0) as n from u limit 1", cfg[0], types=False)
    check("select (select count(*) from t where 1 = 1) as n from u limit 1", cfg[0], types=False)


def test_a_false_on_does_not_build_the_cross_product_of_big_inputs():
    # 3000 x 3000 is past the 4M cross-join cap; the right side is already empty, so the cap does not apply
    import duckdb
    import numpy as np
    import pyarrow as pa
    import mxframe as mx
    n = 3000
    a = pa.table({"k": pa.array(np.arange(n)), "v": pa.array(np.arange(n))})
    b = pa.table({"k": pa.array(np.arange(n)), "w": pa.array(np.arange(n))})
    con = duckdb.connect()
    con.register("a", a)
    con.register("b", b)
    for q in ["select a.k, b.w from a left join b on false order by a.k",
              "select a.k, b.w from a join b on false",
              "select a.k, b.w from a left join b on a.k = b.k and false order by a.k",
              "select a.k, b.w from a join b on a.k = b.k and 1 = 0",
              "select a.k, b.w from a left join b on a.k = b.k and null order by a.k",
              "select a.k, b.w from a left join b on a.k = b.k and 1 = 1 order by a.k"]:
        want = con.execute(q).to_arrow_table()
        got = mx.sql(q, a=a, b=b).compute()
        assert got.column_names == want.column_names and got.num_rows == want.num_rows, q
        assert got.to_pylist() == want.to_pylist(), q
