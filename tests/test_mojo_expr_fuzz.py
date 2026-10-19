"""F-156: random expressions equal DuckDB's.

A seeded generator builds typed expression trees from the pieces the engine now has (CASE, COALESCE, NULLIF, arithmetic, comparisons, IS DISTINCT FROM, IN,
LIKE, text functions, ||, EXTRACT, intervals, date_trunc, casts) over the columns of tests/test_mojo_exprs.py's data (NULLs, zeros, text with accents, dates). Each
expression is a select item (every row), and the same expression is a WHERE, a group key and an aggregate argument; the answers are compared with DuckDB's over a table
and over an edited frame on a streamed file. A failure prints the expression, so it can be repeated by hand.
"""
import random

import pytest

from test_mojo_exprs import data, run, MODES  # noqa: F401  (data is a session fixture)
from test_mojo_sql import probe  # noqa: F401

SEEDS = range(250)
PER_SEED = 6


class Gen:
    """(sql for the engine, sql for DuckDB) of typed expressions: DuckDB differs where the engine returns a DATE for date + INTERVAL and date_trunc."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def pick(self, *options):
        return self.r.choice(options)

    def num(self, depth):
        r = self.r
        if depth <= 0 or r.random() < 0.25:
            return self.pick("x", "y", "id", "1", "-3", "0", "7", "length(s)", "EXTRACT(month FROM dt)", "datediff('day', dt, dt2)", "x + y")
        k = r.randint(0, 9)
        a = self.num(depth - 1)
        b = self.num(depth - 1)
        if k == 0:
            return f"({a} + {b})"
        if k == 1:
            return f"({a} - {b})"
        if k == 2:
            return f"({a} * {self.pick('2', '-1', 'y', '3')})"
        if k == 3:
            return f"({a} // {b})"
        if k == 4:
            return f"({a} % {b})"
        if k == 5:
            return f"CASE WHEN {self.cond(depth - 1)} THEN {a} ELSE {b} END"
        if k == 6:
            return f"COALESCE({a}, {b})"
        if k == 7:
            return f"NULLIF({a}, {b})"
        if k == 8:
            return f"CASE WHEN {self.cond(depth - 1)} THEN {a} END"
        return f"length({self.text(depth - 1)})"

    def dec(self, depth):
        r = self.r
        if depth <= 0 or r.random() < 0.3:
            return self.pick("d", "e", "d * 2", "CAST(x AS DECIMAL(10,2))", "0.5", "-1.25")
        k = r.randint(0, 4)
        a = self.dec(depth - 1)
        if k == 0:
            return f"({a} + {self.pick('d', 'e', '1')})"
        if k == 1:
            return f"CASE WHEN {self.cond(depth - 1)} THEN {a} ELSE {self.dec(depth - 1)} END"
        if k == 2:
            return f"COALESCE({a}, {self.dec(depth - 1)})"
        if k == 3:
            return f"NULLIF({a}, {self.pick('0', 'd')})"
        return f"({a} % {self.pick('3', 'y', 'd')})"

    def text(self, depth):
        r = self.r
        if depth <= 0 or r.random() < 0.3:
            return self.pick("s", "s2", "'x'", "''", "'West'")
        k = r.randint(0, 9)
        a = self.text(depth - 1)
        if k == 0:
            return f"lower({a})"
        if k == 1:
            return f"upper({a})"
        if k == 2:
            return f"trim({a})"
        if k == 3:
            return f"substr({a}, {self.pick('1', '2', '-2', '0')}, {self.pick('1', '3', '-1')})"
        if k == 4:
            return f"({a} || {self.text(depth - 1)})"
        if k == 5:
            return f"CASE WHEN {self.cond(depth - 1)} THEN {a} ELSE {self.text(depth - 1)} END"
        if k == 6:
            return f"COALESCE({a}, {self.text(depth - 1)})"
        if k == 7:
            return f"NULLIF({self.pick('s', 's2')}, {a})"                  # (DuckDB folds the NULLIF of two constants into an untyped NULL)
        if k == 8:
            return f"left({a}, {self.pick('1', '2', '-1')})"
        return f"replace({a}, 'a', 'AA')"

    def date(self, depth):
        r = self.r
        if depth <= 0 or r.random() < 0.3:
            return self.pick("dt", "dt2", "DATE '2000-02-29'")
        k = r.randint(0, 5)
        a = self.date(depth - 1)
        if k == 0:
            return f"CAST({a} + INTERVAL '{self.pick('1', '3', '12', '-1')}' MONTH AS DATE)"
        if k == 1:
            return f"CAST({a} - INTERVAL '{self.pick('1', '30', '2')}' {self.pick('DAY', 'WEEK', 'YEAR')} AS DATE)"
        if k == 2:
            return f"CAST(date_trunc({self.pick(chr(39) + 'month' + chr(39), chr(39) + 'year' + chr(39), chr(39) + 'week' + chr(39), chr(39) + 'quarter' + chr(39))}, {a}) AS DATE)"
        if k == 3:
            return f"CASE WHEN {self.cond(depth - 1)} THEN {a} ELSE {self.date(depth - 1)} END"
        if k == 4:
            return f"COALESCE({a}, {self.date(depth - 1)})"
        return f"({a} + {self.pick('7', '-30', '1')})"

    def cond(self, depth):
        r = self.r
        k = r.randint(0, 13)
        if depth <= 0:
            k = r.choice([0, 1, 2, 3, 9, 12])
        if k == 0:
            return f"{self.num(max(depth - 1, 0))} {self.pick('<', '<=', '=', '<>', '>', '>=')} {self.num(max(depth - 1, 0))}"
        if k == 1:
            return f"{self.dec(max(depth - 1, 0))} {self.pick('<', '=', '>')} {self.dec(max(depth - 1, 0))}"
        if k == 2:
            return f"{self.text(max(depth - 1, 0))} {self.pick('=', '<>')} {self.text(max(depth - 1, 0))}"
        if k == 3:
            return f"{self.date(max(depth - 1, 0))} {self.pick('<', '=', '>=')} {self.date(max(depth - 1, 0))}"
        if k == 4:
            return f"({self.cond(depth - 1)} AND {self.cond(depth - 1)})"
        if k == 5:
            return f"({self.cond(depth - 1)} OR {self.cond(depth - 1)})"
        if k == 6:
            return f"NOT {self.cond(depth - 1)}"
        if k == 7:
            return f"{self.num(depth - 1)} IS DISTINCT FROM {self.num(depth - 1)}"
        if k == 8:
            return f"{self.text(depth - 1)} {self.pick('LIKE', 'ILIKE', 'NOT LIKE')} {self.pick(chr(39) + '%a%' + chr(39), chr(39) + 'W%' + chr(39), chr(39) + '_a' + chr(39), chr(39) + '%' + chr(39))}"
        if k == 9:
            return f"{self.pick('x', 'y', 'dt', 's', 'd', 'b')} IS {self.pick('', 'NOT ')}NULL"
        if k == 10:
            return f"{self.num(depth - 1)} IN (1, 2, 3, NULL, 7)"
        if k == 11:
            return f"{self.text(depth - 1)} IN ('West', 'east', 'x')"
        if k == 12:
            return self.pick("b", "NOT b", "b IS TRUE", "b IS NOT FALSE")
        return f"starts_with({self.text(depth - 1)}, 'W')"


def duck_of(expr):
    """The oracle's text of an engine expression: dates through CAST (DuckDB makes TIMESTAMPs of them), ASCII-lowered text order is not used (equality only)."""
    return expr


def check_expr(probe, data, mode, expr, kind):
    sql = f"SELECT id, {expr} AS v FROM t ORDER BY id"
    run(probe, data, sql, mode, ordered=True)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_expressions_equal_duckdb(probe, data, seed):
    g = Gen(seed)
    for k in range(PER_SEED):
        kind = g.pick("num", "num", "dec", "text", "text", "date", "cond")
        expr = {"num": g.num, "dec": g.dec, "text": g.text, "date": g.date, "cond": g.cond}[kind](3)
        for mode in ("table", "framescan"):
            try:
                check_expr(probe, data, mode, expr, kind)
            except AssertionError as e:
                raise AssertionError(f"seed {seed}, {mode}, expression `{expr}`: {str(e)[:900]}") from None
        # the same expression as a condition, a group key and an aggregate argument
        # casts of the expression: as a key, a DISTINCT key and under min / max (a converted value is an integer, whatever it was)
        if kind in ("num", "dec"):
            cast_ty = g.pick("BIGINT", "DECIMAL(12,1)", "DOUBLE")
            ce = f"TRY_CAST({expr} AS {cast_ty})"
            run(probe, data, f"SELECT {ce} AS k, count(*) FROM t GROUP BY 1", "table")
            run(probe, data, f"SELECT DISTINCT {ce} FROM t", "framescan")
            run(probe, data, f"SELECT g, min({ce}), max({ce}), count(DISTINCT {ce}) FROM t GROUP BY g", "table")
        elif kind == "date":
            run(probe, data, f"SELECT CAST(EXTRACT(dow FROM {expr}) AS BIGINT) AS k, count(*) FROM t GROUP BY 1", "table")
        if kind == "cond":
            for mode in ("table", "frame"):
                run(probe, data, f"SELECT count(*), sum(x) FROM t WHERE {expr}", mode)
                run(probe, data, f"SELECT g, count(*) FROM t WHERE {expr} GROUP BY g", mode)
        elif kind in ("num", "dec"):
            run(probe, data, f"SELECT g, sum({expr}), count({expr}), min({expr}), max({expr}) FROM t GROUP BY g", "table")
        elif kind == "text":
            run(probe, data, f"SELECT {expr} AS k, count(*) FROM t GROUP BY 1", "table")
