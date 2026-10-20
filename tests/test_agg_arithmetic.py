"""F-068: arithmetic, CASE, constants and HAVING over aggregates (SQL `100 * SUM(CASE ...) / SUM(...)`,
`SUM(x) / 7.0`) run the aggregate in the native executor (single table, fused join, star) and the small
per-group arithmetic after it; the plan is no longer declined whole. Every query is checked against
DuckDB (`/` is float division there, x / 0 is inf, an empty global aggregate is NULL) and against the
path: the executor's aggregate hooks ran and none declined."""
import math
from datetime import date

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
import omxcore.native_exec as nx

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

N = 20_000


def _days(y, m, d):
    return (date(y, m, d) - date(1970, 1, 1)).days


def _tables():
    rng = np.random.default_rng(68)
    kinds = ["PROMO BRUSHED", "PROMO PLATED", "STANDARD ANODIZED", "ECONOMY POLISHED", "LARGE BURNISHED"]
    d = pa.table({"dk": pa.array(np.arange(1, 201), pa.int64()),
                  "cat": pa.array([kinds[i % 5] for i in range(200)]),
                  "nat": pa.array([["BRAZIL", "PERU", "CHILE"][i % 3] for i in range(200)])})
    day = rng.integers(_days(1994, 1, 1), _days(1997, 1, 1), N).astype(np.int32)
    f = pa.table({"fk": pa.array(rng.integers(1, 201, N), pa.int64()),
                  "g": pa.array(rng.integers(0, 7, N), pa.int32()),
                  "q": pa.array(rng.integers(1, 50, N), pa.int32()),
                  "a": pa.array(np.round(rng.uniform(1, 100, N), 2), pa.float64()),
                  "b": pa.array(np.round(rng.uniform(0, 0.1, N), 2), pa.float64()),
                  "z": pa.array(np.where(rng.integers(0, 7, N) >= 5, 0.0, 1.0), pa.float64()),   # group sums that are 0
                  "day": pa.array(day)})
    # groups whose sums are exactly zero / tied: g = 7 has z = 0 and a = 0, g = 8 and g = 9 tie on every sum
    extra = pa.table({"fk": pa.array([1, 2, 3, 4, 5, 6], pa.int64()), "g": pa.array([7, 7, 8, 8, 9, 9], pa.int32()),
                      "q": pa.array([1, 2, 3, 4, 3, 4], pa.int32()),
                      "a": pa.array([0.0, 0.0, 10.0, 20.0, 10.0, 20.0], pa.float64()),
                      "b": pa.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25], pa.float64()),
                      "z": pa.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], pa.float64()),
                      "day": pa.array([day[0]] * 6, pa.int32())})
    return {"f": pa.concat_tables([f, extra]), "d": d}


TABLES = _tables()


def _duck(sql):
    con = duckdb.connect()
    con.register("_f", TABLES["f"])
    con.execute("create view f as select * exclude (day), date '1970-01-01' + day as day from _f")
    con.register("d", TABLES["d"])
    return [tuple(r) for r in con.execute(sql).fetchall()]


@pytest.fixture
def hooks(monkeypatch):
    """Outcome of every aggregate hook call: ('execute' | 'join', ran)."""
    log = []
    for name, tag in (("try_execute", "execute"), ("try_execute_join", "join")):
        real = getattr(nx, name)

        def wrap(*a, _real=real, _tag=tag, **k):
            r = _real(*a, **k)
            log.append((_tag, r is not None, nx.last_decline() if r is None else None))
            return r
        monkeypatch.setattr(nx, name, wrap)
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    return log


def _eq(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) or isinstance(y, float):
        if math.isnan(x) or math.isnan(y):
            return math.isnan(x) and math.isnan(y)
        if math.isinf(x) or math.isinf(y):
            return x == y
        return x == pytest.approx(y, rel=1e-9, abs=1e-12)
    return x == y


def _check(sql, hooks, ordered=False, join=None, nrows=None):
    """A derived table's schema is asked first on zero rows (LazyFrame.schema): those calls decline 'empty input'."""
    exp = _duck(sql)
    got_t = mx.sql(sql, **TABLES).compute(device="cpu")
    got = [tuple(got_t.column(i)[r].as_py() for i in range(got_t.num_columns)) for r in range(got_t.num_rows)]
    if nrows is not None:
        assert len(exp) == nrows, "the fixture no longer produces the rows this test is about"
    if not ordered:
        key = lambda r: tuple((x is None, repr(x)) for x in r)
        got, exp = sorted(got, key=key), sorted(exp, key=key)
    assert len(got) == len(exp), (got, exp)
    for g, e in zip(got, exp):
        assert len(g) == len(e) and all(_eq(x, y) for x, y in zip(g, e)), (g, e)
    hooks = [h for h in hooks if h[1] or h[2] != "empty input"]
    declined = [h for h in hooks if not h[1]]
    assert not declined, declined
    assert hooks, "no native aggregate call was made"
    if join is not None:
        assert any(h[0] == "join" for h in hooks) == join, hooks
    return got


# ── the Q14 shape: a ratio of two sums over a join, strings tested in the aggregate ─────────────────

def test_global_ratio_of_case_sums_over_a_join(hooks):
    got = _check("select 100.00 * sum(case when cat like 'PROMO%' then a * (1 - b) else 0 end) "
                 "/ sum(a * (1 - b)) as promo from f, d where fk = dk and q >= 10", hooks, join=True)
    assert 20 < got[0][0] < 60


def test_grouped_ratio_with_a_partner_string_test_and_a_year_key(hooks):
    # Q8: EXTRACT(YEAR) of a day column as the key, CASE on a partner's string inside the sum
    got = _check("select extract(year from day) as y, sum(case when nat = 'BRAZIL' then a else 0 end) / sum(a) as share "
                 "from f, d where fk = dk and g < 7 group by y order by y", hooks, ordered=True, join=True)
    assert [r[0] for r in got] == [1994, 1995, 1996]


def test_year_key_on_a_single_table(hooks):
    _check("select extract(year from day) as y, sum(a) / count(*) as m from f group by y order by y",
           hooks, ordered=True)


# ── single table: grouped and global ratios, constants, CASE over aggregates ─────────────────────────

def test_grouped_ratio(hooks):
    _check("select g, sum(a) / sum(q) as r from f group by g order by g", hooks, ordered=True, nrows=10)


def test_constant_divisor_and_multiplier(hooks):
    _check("select sum(a) / 7.0 as avg_yearly, sum(q) * 3 + 1 as t, 100 * sum(a) as p from f", hooks, nrows=1)


def test_sum_over_sum_of_integers_is_float_division(hooks):
    got = _check("select g, sum(q) / 2 as h from f group by g order by g", hooks, ordered=True)
    assert got[0][1] == pytest.approx(_duck("select sum(q) / 2.0 from f where g = 0")[0][0])
    assert isinstance(got[0][1], float)


def test_case_over_aggregates(hooks):
    _check("select g, case when sum(z) = 0 then -1.0 when sum(a) > 100000 then sum(a) / sum(z) "
           "else sum(a) * 2 / sum(q) end as c from f group by g order by g", hooks, ordered=True)


def test_division_by_a_zero_aggregate_is_inf_or_nan_like_duckdb(hooks):
    got = _check("select g, sum(a) / sum(z) as r from f group by g order by g", hooks, ordered=True)
    by = {g: r for g, r in got}
    assert math.isinf(by[7]) or math.isnan(by[7])        # the group of zero sums: DuckDB's value, checked above


def test_global_ratio_of_nothing_is_null(hooks):
    got = _check("select sum(a) / sum(b) as r, sum(a) - sum(b) as d from f where g < 0", hooks, nrows=1)
    assert got == [(None, None)]


def test_zero_over_zero(hooks):
    _check("select sum(a) / sum(a) as r from f where g = 7", hooks, nrows=1)


def test_integer_sums_past_2_pow_53_stay_exact(hooks):
    big = pa.table({"k": pa.array([1, 1, 2, 2], pa.int64()),
                    "v": pa.array([2 ** 60, 2 ** 60 + 1, 5, 7], pa.int64())})
    con = duckdb.connect()
    con.register("big", big)
    sql = "select k, sum(v) as s, sum(v) / 2 as h from big group by k order by k"
    exp = con.execute(sql).fetchall()
    got = mx.sql(sql, big=big).compute(device="cpu")
    assert exp[0][1] == 2 ** 61 + 1
    assert [tuple(r) for r in zip(*(got.column(i).to_pylist() for i in range(3)))] == [tuple(r) for r in exp]


# ── HAVING, ORDER BY and LIMIT on such expressions ───────────────────────────────────────────────────

def test_having_on_a_ratio(hooks):
    got = _check("select g, sum(a) / sum(q) as r from f group by g having sum(a) / sum(q) > 2.0 order by g",
                 hooks, ordered=True)
    assert 0 < len(got) < 10


def test_having_on_a_ratio_of_a_join(hooks):
    _check("select g, sum(case when cat like 'PROMO%' then a else 0 end) / sum(a) as r from f, d "
           "where fk = dk and g < 7 group by g having sum(case when cat like 'PROMO%' then a else 0 end) / sum(a) > 0.39 "
           "order by g", hooks, ordered=True, join=True)


def test_order_by_a_ratio_with_limit(hooks):
    got = _check("select g, sum(a) / sum(q) as r from f group by g order by r desc, g limit 3", hooks, ordered=True)
    assert len(got) == 3


def test_ties_on_the_ratio_break_by_the_group_key(hooks):
    # g = 8 and g = 9 have the same ratio (and so do they in DuckDB's ORDER BY r, g)
    got = _check("select g, sum(a) / sum(b) as r from f where g >= 8 group by g order by r, g", hooks, ordered=True)
    assert [r[0] for r in got] == [8, 9]


def test_order_by_a_ratio_over_a_join_keeps_ties_in_key_order(hooks):
    _check("select g, sum(a) / sum(b) as r from f, d where fk = dk and g >= 8 group by g order by r desc, g",
           hooks, ordered=True, join=True)


# ── the three TPC-H queries written as SQL, small scale factor ───────────────────────────────────────

@pytest.mark.parametrize("q", [8, 14, 17])
def test_tpch_sql_q8_q14_q17_equal_duckdb_and_aggregate_natively(q, hooks):
    import sys
    from pathlib import Path
    from conftest import TPCH_DATA
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from tpch_sql import run
    if not (TPCH_DATA / "tpch_sf0.017_exact" / "lineitem.parquet").exists():
        pytest.skip("TPC-H small data not present")
    out = run.one(q, "0.017")
    assert out["status"] == "equal", out
    assert hooks and all(h[1] for h in hooks), [h for h in hooks if not h[1]]


def test_float_arithmetic_over_the_groups_does_not_touch_arrow_compute(hooks, monkeypatch):
    import omxcore.custom_ops as co

    def boom(*a, **k):
        raise AssertionError("the per-group arithmetic went through Arrow's checked kernels")
    monkeypatch.setattr(co, "_checked_arith", boom)
    _check("select g, 0.2 * sum(a) + 1.5 - sum(b) as t, sum(a) / sum(z) as r from f group by g order by g",
           hooks, ordered=True, nrows=10)


def test_a_literal_column_next_to_a_computed_one_still_broadcasts(hooks):
    t = pa.table({"k": pa.array([1, 2], pa.int64()), "y": pa.array([1.5, 2.5], pa.float64())})
    got = mx.sql("select k, y * 2.0 as d, (select max(y) from u) as m from t", t=t, u=t).compute(device="cpu")
    assert got.to_pydict() == {"k": [1, 2], "d": [3.0, 5.0], "m": [2.5, 2.5]}


def test_a_dictionary_column_passes_through_a_float_project_decoded():
    from omxcore.custom_ops import _np_float_project
    t = pa.table({"k": pa.array(["a", "b"]).dictionary_encode(), "y": pa.array([1.5, 2.5], pa.float64())})
    got = _np_float_project([mx.col("k"), (mx.col("y") * 2.0).alias("d")], t)
    assert got.schema.field("k").type == pa.string()
    assert got.to_pydict() == {"k": ["a", "b"], "d": [3.0, 5.0]}


def test_second_aggregate_over_an_aggregate_over_a_join(hooks):
    got = _check("select max(r) as top, min(r) as bot, sum(r) / count(*) as m from "
                 "(select g, sum(a) / sum(q) as r from f, d where fk = dk and g < 7 group by g) t",
                 hooks, nrows=1, join=True)
    assert got[0][0] > got[0][1]


def test_filter_over_a_global_aggregate_over_a_join(hooks):
    got = _check("select * from (select sum(a) / sum(q) as r from f, d where fk = dk) t where r > 0",
                 hooks, nrows=1, join=True)
    assert got[0][0] > 0
    _check("select * from (select sum(a) / sum(q) as r from f, d where fk = dk) t where r < 0", hooks)
