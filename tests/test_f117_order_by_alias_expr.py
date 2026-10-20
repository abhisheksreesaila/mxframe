"""F-117: ORDER BY an expression that uses a SELECT alias.

Before: ``select g, 1 as c, count(*) as n from t group by g order by c + 1`` raised ``KeyError '__calc1'``, and
``select id, x * 2 as h from t order by h + 1`` raised ``KeyError 'h'``.

DuckDB (probed): inside an ORDER BY expression a bare name is a FROM column when one has that name, else the
SELECT alias (``order by h + 1``, ``-h``, ``h * h``, ``coalesce(s, 0)``); a bare ``order by h`` is the alias
first. An alias used in an expression that is also a FROM column's name is the column (and in a grouped query
then not grouped, so an error). Expected answers come from a native DuckDB table.
"""
import pytest

from _sql_reference import check, refused, cfg  # noqa: F401  (cfg is the fixture)

PLAIN = [
    "select id, x * 2 as h from t order by h + 1, id",
    "select id, x * 2 as h from t order by -h, id",
    "select id, x * 2 as h from t order by h * h desc, id",
    "select id, x as id2 from t order by id2 + 1, id",
    "select id, 1 as c from t order by c + id desc",
    "select id, f * 2 as h from t order by coalesce(h, 0) + 1, id",
    "select id, f as x from t order by x + 1, id",                     # x is a FROM column: it wins in an expression
    "select id, x * 2 as h from t order by h + 1 desc, h, id",         # an expression and the bare alias together
    "select id, x * 2 as h, x as x2 from t order by h + x2, id",       # two aliases
    "select id, x * 2 as h from t order by h + id, id limit 4",
    "select id, x * 2 as h from t order by h + 1, id limit 3 offset 2",
    "select id, x * 2 as h from t where id > 2 order by h - 1, id",
]


@pytest.mark.parametrize("q", PLAIN)
def test_order_by_expression_of_an_alias_in_a_plain_select(cfg, q):
    check(q, cfg[0], types=False)


GROUPED = [
    "select g, 1 as c, count(*) as n from t group by g order by c + 1, g",
    "select g, 1 as c, count(*) as n from t group by g order by n * 2 + c desc, g",
    "select g, count(*) as h from t group by g order by h * 2, g",
    "select g, count(*) as h, 1 as c from t group by g order by h + c, g",
    "select g, count(*) + 1 as h from t group by g order by h * 2 desc, g",
    "select g, sum(x) as s from t group by g order by coalesce(s, 0) + 1 desc, g",
    "select g, sum(x) as s from t group by g order by s + 1 desc, g",
    "select gi, gi + 1 as h, count(*) as n from t group by gi order by h * 2 desc, gi",
    "select gi + 1 as k, count(*) as n from t group by gi order by k + 1, n desc",
    "select g, count(*) as n from t group by g order by n + sum(x), g",                # an aggregate the SELECT lacks
    "select g, 5 as c, count(*) as n from t group by g order by c - n, g limit 2",
    "select 1 as c, count(*) as n from t order by c + 1",                              # one row, no GROUP BY
    "select g, 1 as c, count(*) as n from t group by g having n > 2 order by c + n desc, g",
]


@pytest.mark.parametrize("q", GROUPED)
def test_order_by_expression_of_an_alias_in_a_grouped_select(cfg, q):
    check(q, cfg[0], types=False)


def test_the_hidden_sort_key_is_not_in_the_result(cfg):
    want = check("select g, 1 as c, count(*) as n from t group by g order by c + 1, g", cfg[0], types=False)
    assert want.column_names == ["g", "c", "n"]


def test_a_from_column_wins_over_an_alias_in_a_grouped_expression(cfg):
    # x is the FROM column t.x, not the alias of count(*): not grouped, an error in both
    refused("select g, count(*) as x from t group by g order by x + 1, g", cfg[0], match="GROUP BY")


def test_distinct_order_by_expression_of_an_alias_still_names_the_problem(cfg):
    # DuckDB accepts this; the engine refuses with its usual DISTINCT message (follow-up, not F-117)
    with pytest.raises(ValueError, match="DISTINCT"):
        import omxcore as mx
        from _sql_reference import T
        mx.sql("select distinct g, 1 as c from t order by c + 1, g", t=T).compute(device=cfg[0])
