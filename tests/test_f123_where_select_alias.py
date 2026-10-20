"""F-123: a SELECT alias used in the WHERE of the same query.

Before: ``select x as kk from t where kk > 5`` raised ``KeyError 'kk'``.

DuckDB (probed): a bare name in WHERE is a FROM column when one has that name (``select id as x ... where x > 5``
filters ``t.x``), else the SELECT alias, matched case-insensitively. The alias of an aggregate or window call is
no WHERE condition (an error). GROUP BY and HAVING already took an alias (GROUP BY: only when no FROM column has
the name; HAVING: the alias wins), and still do. Expected answers come from a native DuckDB table.
"""
import pytest

import omxcore as mx
from _sql_reference import CON, T, check, refused, cfg  # noqa: F401  (cfg is the fixture)

ALIAS_IN_WHERE = [
    "select x as kk from t where kk > 5 order by id",
    "select x * 2 as kk from t where kk > 5 and id > 2 order by id",
    "select x as id2 from t where id2 > 5 order by id",
    "select x as kk from t where KK > 5 order by id",                                  # case-insensitive
    "select x as KK from t where kk > 5 order by id",
    "select x as kk from t where kk is null",
    "select x as kk from t where kk is not null order by 1",
    "select x as kk from t where kk in (2, 4, 99) order by 1",
    "select x as kk from t where kk not in (2, 4) order by 1",
    "select x as kk from t where kk between 2 and 6 order by 1",
    "select x as kk from t where kk > 2 and kk < 9 or kk = 0 order by 1",
    "select x as kk from t where not (kk > 2) order by 1",
    "select case when x > 3 then x else 0 end as kk from t where kk = 8 order by id",
    "select id, case when x > 3 then 'big' else 'small' end as sz from t where sz = 'big' order by id",
    "select id, coalesce(x, -1) as xx from t where xx < 0 order by id",
    "select g as gg, id from t where gg = 'a' order by id",
    "select g as gg, id from t where gg is null order by id",
    "select id, f + 1 as ff from t where ff > 0 order by id",
    "select id, d as dd2 from t where dd2 > date '2024-06-01' order by id",
    "select x as kk from t where kk > 3 order by kk desc limit 2",
    "select x as kk from t where kk > 3 limit 2",
    "select distinct g as gg from t where gg is not null order by gg",
    "select x as kk, g from t where kk > 2 and g = 'a' order by kk",
    "select x as kk from t where kk > 2 and id < (select max(id) from t where id < 10) order by 1",
    "select t.x as kk from t join u on t.gi = u.k where kk > 3 order by id",
    "select t.x as kk, u.w from t join u on t.gi = u.k where kk > 3 and w > 10 order by t.id",
    "select x as kk from t where kk > 2 and exists (select 1 from u where u.k = t.gi) order by 1",
    "select x as kk from t where kk > 2 and gi in (select k from u) order by 1",
    "select * from (select x as kk from t) where kk > 5 order by kk",              # the plain column, as ever
]


@pytest.mark.parametrize("q", ALIAS_IN_WHERE)
def test_select_alias_in_where(cfg, q):
    check(q, cfg[0])


FROM_COLUMN_WINS = [
    "select id as x from t where x > 5 order by id",                    # x is t.x, not the alias of id
    "select id as X from t where x > 2 order by 1",
    "select x as id from t where id > 8 order by 1",                    # id is t.id (the alias would be x)
    "select f as x, id from t where x > 6 order by id",
    "select id + 100 as gi, id from t where gi = 1 order by id",
]


@pytest.mark.parametrize("q", FROM_COLUMN_WINS)
def test_a_from_column_wins_over_the_alias(cfg, q):
    check(q, cfg[0])


def test_the_alias_filters_before_the_projection_and_changes_nothing_else(cfg):
    want = check("select x as kk, id from t where kk > 5 order by id", cfg[0])
    assert want.column_names == ["kk", "id"] and want.num_rows == 6


@pytest.mark.parametrize("q", [
    "select count(*) as n from t where n > 1",
    "select sum(x) as s from t where s > 1",
    "select g, max(x) as m from t where m > 1 group by g",
    "select x, sum(x) over () as s from t where s > 1",
    "select x, count(*) over (partition by g) as s from t where s > 1",
])
def test_the_alias_of_an_aggregate_or_window_is_no_where_condition(cfg, q):
    refused(q, cfg[0], match="WHERE")


def test_an_alias_nobody_defined_is_still_an_unknown_column(cfg):
    with pytest.raises(Exception):
        CON.execute("select x as kk from t where nope > 1")
    with pytest.raises(Exception):
        mx.sql("select x as kk from t where nope > 1", t=T).compute(device=cfg[0])


@pytest.mark.parametrize("q", [
    "select g as gg, count(*) as n from t group by gg order by gg",                       # GROUP BY an alias
    "select gi + 1 as k, count(*) as n from t group by k order by k",
    "select g, count(*) as n from t group by g having n > 2 order by g",                   # HAVING an alias
    "select g as gg, count(*) as n from t group by g having gg = 'a'",
    "select gi + 1 as k, count(*) as n from t group by k having k > 1 order by k",
    "select g, count(*) as x from t group by g having x > 1 order by g",                   # HAVING: the alias beats t.x
    "select g as gg, count(*) as n from t where gg is not null group by gg order by gg",   # all three together
    "select gi + 1 as k, count(*) as n from t where k > 1 group by k having n > 1 order by k",
])
def test_group_by_and_having_aliases_keep_working(cfg, q):
    check(q, cfg[0])


@pytest.mark.parametrize("q", [
    "select g, count(*) as x from t where x > 1 group by g",           # x: t.x is a column, so the alias is out
])
def test_where_with_a_name_that_is_a_column_and_an_aggregate_alias(cfg, q):
    check(q, cfg[0])


@pytest.mark.parametrize("q", [
    "select kk + 1 as kk from t where kk > 5",                  # kk is defined in terms of itself and is no column
    "select id, kk + 1 as kk from t order by kk + 1",
])
def test_an_alias_defined_by_itself_is_a_clear_error(cfg, q):
    refused(q, cfg[0], match="in terms of itself")
