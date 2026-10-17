"""F-127: SQL arithmetic gaps, every answer checked against DuckDB.

* a NULL literal in `+ - * / %` is a typed NULL (`a + NULL` is NULL of a's type, `/` a DOUBLE);
* `a // b`: integers truncate toward zero in their own type, a float or decimal operand divides, a zero divisor
  (integer or float) gives NULL;
* `power` / `**`, `sqrt`, `ln`: a DOUBLE; `sqrt` of a negative and `ln` of zero or a negative RAISE (DuckDB does),
  also under a WHERE that keeps the bad rows out (it must not raise then);
* `count()` / `sum` / `avg` / `min` / `max` over every expression that yields NULL skip those rows (the F-125 class:
  `custom_ops._can_be_null` knows each NULL source), global, grouped, filtered, as a GROUP BY key.
cpu and gpu, native on and off, a size for the Arrow fallback and one for the native / fused paths, with and without
NULLs in the inputs."""
import math

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import pipeline_kernels as pk
from mxframe.custom_ops import _can_be_null
from mxframe.lazy_expr import col, lit

DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU"))]


def _t(n, nulls):
    r = np.random.default_rng(127)
    mask = (lambda p: r.random(n) < p) if nulls else (lambda p: None)
    return pa.table({
        "i": pa.array(np.arange(n), pa.int64()),
        "g": pa.array(r.integers(0, 5, n), pa.int64(), mask=mask(0.1)),
        "a": pa.array(r.integers(-100, 100, n), pa.int64(), mask=mask(0.1)),
        "b": pa.array(r.integers(-3, 4, n), pa.int64(), mask=mask(0.1)),                 # divisors: zeros and negatives
        "s": pa.array(r.integers(-100, 100, n), pa.int32(), mask=mask(0.1)),
        "t": pa.array(r.integers(-2, 3, n), pa.int8(), mask=mask(0.1)),
        "f": pa.array(r.random(n) * 20 - 10, pa.float64(), mask=mask(0.1)),            # negative too
        "z": pa.array(r.integers(0, 3, n) * 0.5, pa.float64(), mask=mask(0.1)),         # a third of the divisors are 0.0
        "p": pa.array(r.random(n) * 50 + 0.5, pa.float64(), mask=mask(0.1)),           # always positive: sqrt / ln are safe
    })


AGGS = "count({e}) AS c, sum({e}) AS s, avg({e}) AS v, min({e}) AS mn, max({e}) AS mx"
NULL_EXPRS = [                                           # each yields NULL on rows whatever the columns hold
    "a + NULL", "a * NULL", "a - NULL", "NULL + a", "a % NULL", "f * NULL", "a / NULL", "s + NULL", "a // NULL",
    "a // 0", "f // 0", "a // b", "f // z", "a // (b - b)", "(a + NULL) + 1", "coalesce(a + NULL, 7)",
    "(a // b) + 1", "(a // b) * 2", "coalesce(a // b, -1)", "nullif(a, 5) // b", "power(a, NULL)",
    "try_cast(a * 3 AS tinyint)", "try_cast(f * 1e9 AS integer)", "try_cast(f / z AS integer)", "try_cast(f * 1e30 AS float)",
    "abs(try_cast(a * 3 AS tinyint))", "coalesce(try_cast(a * 3 AS tinyint), t)",
]
VALUE_EXPRS = [                                          # no NULL of their own, still new
    "a // 2", "a // -3", "s // 4", "t // 2", "f // 2.5", "a // 2.5", "f // z", "power(p, 2)", "power(a, 2)",
    "power(p, 0.5)", "power(p, 0.1)", "p ** 3", "power(2, a / 10)", "sqrt(p)", "sqrt(a * a)", "ln(p)", "ln(p * p)",
    "sqrt(p) + ln(p)", "power(p, -1)", "try_cast(a AS smallint)", "try_cast(f AS integer)", "try_cast(p AS double)",
]
EXPRS = NULL_EXPRS + VALUE_EXPRS
QUERIES = (
    [f"SELECT {AGGS.format(e=e)} FROM t" for e in EXPRS]
    + [f"SELECT g, {AGGS.format(e=e)} FROM t GROUP BY g ORDER BY g" for e in EXPRS]
    + [f"SELECT g, {AGGS.format(e=e)} FROM t WHERE f > 2 GROUP BY g ORDER BY g"
       for e in ("a // b", "a + NULL", "a // 0", "sqrt(p)", "power(a, 2)")]
    + [f"SELECT i, {e} AS x FROM t ORDER BY i" for e in EXPRS]
    + [
        "SELECT a // b AS k, count(*) AS n FROM t GROUP BY 1 ORDER BY 1",
        "SELECT a // 0 AS k, count(*) AS n FROM t GROUP BY 1",
        "SELECT a + NULL AS k, count(*) AS n FROM t GROUP BY 1",
        "SELECT g, a // b AS k, count(*) AS n, count(a // b) AS c FROM t GROUP BY 1, 2 ORDER BY 1, 2",
        "SELECT a // 7 AS k, sum(f) AS s FROM t GROUP BY 1 ORDER BY 1",
        "SELECT i FROM t WHERE a // b > 3 ORDER BY i",
        "SELECT i FROM t WHERE a // 3 = 2 ORDER BY i",
        "SELECT i FROM t WHERE a + NULL > 3 ORDER BY i",
        "SELECT i FROM t WHERE a + NULL IS NULL ORDER BY i",
        "SELECT i FROM t WHERE sqrt(p) > 3 ORDER BY i",
        "SELECT g, count(*) AS n FROM t WHERE power(p, 2) < 100 GROUP BY g ORDER BY g",
        "SELECT g, count(*) AS n FROM t WHERE a // b IS NULL GROUP BY g ORDER BY g",
        "SELECT g, sum(a // 2) AS s FROM t WHERE a // b > 0 GROUP BY g ORDER BY g",
        "SELECT i, a % 2.5 AS m FROM t WHERE g = 1 ORDER BY i LIMIT 7",
        "SELECT g, sum(f % 2.5) AS s, count(f % 2.5) AS c FROM t GROUP BY g ORDER BY g",
        "SELECT g, avg(f % 2.5) AS s FROM t WHERE p > 5 GROUP BY g ORDER BY g",
        "SELECT g, min(f % 2.5) AS lo, max(f % 2.5) AS hi FROM t GROUP BY g ORDER BY g",
        "SELECT f % 2.5 AS k, count(*) AS n FROM t WHERE i < 40 GROUP BY 1 ORDER BY 1",
    ])


def _rows(tbl):
    d = tbl.to_pydict()
    rows = zip(*[d[c] for c in tbl.column_names])
    return [tuple("nan" if isinstance(x, float) and math.isnan(x) else x if x is None or isinstance(x, str)
                  else float(f"{float(x):.9g}") for x in row) for row in rows]


def _duck(q, t):
    exp = duckdb.sql(q).arrow()
    return exp.read_all() if isinstance(exp, pa.RecordBatchReader) else exp


@pytest.mark.parametrize("native", ["1", "0"])
@pytest.mark.parametrize("nulls", [False, True])
@pytest.mark.parametrize("n", [60, 20_000])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", QUERIES)
def test_matches_duckdb(q, device, n, nulls, native, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    t = _t(n, nulls)                                         # noqa: F841 (DuckDB finds it by name)
    exp = _duck(q, t)
    got = mx.sql(q, t=t).compute(device=device)
    assert got.column_names == exp.column_names
    g, e = _rows(got), _rows(exp)
    if "ORDER BY" not in q:
        g, e = sorted(g, key=repr), sorted(e, key=repr)
    assert g == e
    if q.startswith("SELECT i, ") and "WHERE" not in q:      # a plain projection: the column types are DuckDB's too
        assert got.schema.types[1] == exp.schema.types[1], q


# ---- the typed NULL, in the types DuckDB gives it ---------------------------------------------------------------

@pytest.mark.parametrize("e", ["a + NULL", "NULL * s", "t - NULL", "f + NULL", "a / NULL", "s / NULL", "a % NULL",
                               "s % NULL", "a // NULL", "s // NULL", "NULL // t", "power(a, NULL)"])
def test_a_null_literal_keeps_the_other_operands_type(e):
    t = _t(30, False)                                        # noqa: F841
    q = f"SELECT {e} AS x FROM t"
    exp, got = _duck(q, t), mx.sql(q, t=t).compute(device="cpu")
    assert got.schema.types == exp.schema.types
    assert got.to_pydict() == exp.to_pydict() == {"x": [None] * 30}


def test_try_cast_is_null_where_the_value_does_not_fit():
    t = pa.table({"a": pa.array([7, 300, -300, None, 127, -128, 128], pa.int64()),
                  "f": pa.array([1.5, 1e10, float("nan"), None, 2.5, -2147483648.4, float("inf")])})
    q = ("SELECT try_cast(a AS tinyint) AS x, try_cast(f AS integer) AS y, try_cast(f AS bigint) AS z, "
         "try_cast(f * 1e300 AS float) AS w FROM t")
    exp, got = _duck(q, t), mx.sql(q, t=t).compute(device="cpu")
    assert got.schema.types == exp.schema.types
    assert _rows(got) == _rows(exp)
    assert got.to_pydict()["x"] == [7, None, None, None, 127, -128, None]
    assert got.to_pydict()["y"] == [2, None, None, None, 2, None, None]          # -2147483648.4 is below the INT32 range
    assert got.to_pydict()["z"] == [2, 10000000000, None, None, 2, -2147483648, None]


# ---- integer division: DuckDB's values, hand-checked ------------------------------------------------------------

def test_integer_division_truncates_toward_zero_and_a_zero_divisor_is_null():
    t = pa.table({"a": pa.array([7, -7, 7, -7, 0, 5, None, 9, -9], pa.int64()),
                  "b": pa.array([2, 2, -2, -2, 3, 0, 3, None, 4], pa.int64())})
    got = mx.sql("SELECT a // b AS q, a // 2 AS h, a // 0 AS z FROM t", t=t).compute(device="cpu").to_pydict()
    assert got == {"q": [3, -3, -3, 3, 0, None, None, None, -2], "h": [3, -3, 3, -3, 0, 2, None, 4, -4], "z": [None] * 9}


def test_integer_division_keeps_the_column_type_a_float_operand_divides():
    t = pa.table({"s": pa.array([7, -7, 0], pa.int32()), "t": pa.array([2, -2, 0], pa.int8()),
                  "f": pa.array([7.5, -7.5, 2.0])})
    got = mx.sql("SELECT s // 2 AS a, s // t AS b, f // 2.5 AS c, s // 2.5 AS d, f // 0.0 AS e FROM t", t=t).compute(device="cpu")
    assert [str(x) for x in got.schema.types] == ["int32", "int32", "double", "double", "double"]
    assert got.to_pydict() == {"a": [3, -3, 0], "b": [3, 3, None], "c": [3.0, -3.0, 0.8], "d": [2.8, -2.8, 0.0], "e": [None] * 3}


def test_integer_division_of_two_constants():
    t = pa.table({"a": pa.array([1], pa.int64())})
    got = mx.sql("SELECT a, -7 // 2 AS x, 7 // -2 AS y, 7 // 0 AS z FROM t", t=t).compute(device="cpu").to_pydict()
    assert got == {"a": [1], "x": [-3], "y": [-3], "z": [None]}


def test_integer_division_overflow_raises_like_duckdb():
    t = pa.table({"a": pa.array([-2 ** 63, 5], pa.int64()), "b": pa.array([-1, 1], pa.int64())})
    with pytest.raises(duckdb.Error, match="Overflow in division"):
        duckdb.sql("SELECT a // b FROM t").arrow()
    with pytest.raises(OverflowError, match="Overflow in division"):
        mx.sql("SELECT a // b AS x FROM t", t=t).compute(device="cpu")


# ---- sqrt / ln raise where DuckDB raises ------------------------------------------------------------------------

BAD = [("SELECT sqrt(f) AS x FROM t", "cannot take square root of a negative number"),
       ("SELECT ln(f) AS x FROM t", "cannot take logarithm of a negative number"),
       ("SELECT ln(z) AS x FROM t WHERE z <= 0.5", "cannot take logarithm of zero"),
       ("SELECT sum(sqrt(f)) AS x FROM t", "cannot take square root of a negative number"),
       ("SELECT g, sum(ln(f)) AS x FROM t GROUP BY g", "cannot take logarithm of a negative number"),
       ("SELECT g, sum(sqrt(a)) AS x FROM t WHERE i > 3 GROUP BY g", "cannot take square root of a negative number")]


@pytest.mark.parametrize("n", [60, 20_000])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q,message", BAD)
def test_sqrt_and_ln_of_a_rejected_value_raise(q, message, device, n):
    t = _t(n, True)                                          # noqa: F841
    with pytest.raises(duckdb.Error, match=message):
        duckdb.sql(q).arrow()
    with pytest.raises(ValueError, match=message):
        mx.sql(q, t=t).compute(device=device)


GUARDED = [
    "SELECT i, sqrt(f) AS x FROM t WHERE f >= 0 ORDER BY i",
    "SELECT sum(sqrt(f)) AS x, count(sqrt(f)) AS c FROM t WHERE f >= 0",
    "SELECT g, sum(sqrt(f)) AS x FROM t WHERE f >= 0 GROUP BY g ORDER BY g",
    "SELECT g, sum(ln(f)) AS x FROM t WHERE f > 0 GROUP BY g ORDER BY g",
    "SELECT g, avg(ln(z)) AS x FROM t WHERE z > 0 GROUP BY g ORDER BY g",
    "SELECT i FROM t WHERE f > 0 AND ln(f) > 1 ORDER BY i",
    "SELECT g, sum(p) AS x FROM t WHERE f >= 0 AND sqrt(f) > 1 GROUP BY g ORDER BY g",
    "SELECT g, count(*) AS c FROM t WHERE a >= 0 GROUP BY g HAVING sum(sqrt(a)) > 20 ORDER BY g",
]


@pytest.mark.parametrize("native", ["1", "0"])
@pytest.mark.parametrize("n", [60, 20_000])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", GUARDED)
def test_sqrt_and_ln_do_not_raise_for_rows_a_where_removed(q, device, n, native, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    t = _t(n, True)                                          # noqa: F841
    exp, got = _duck(q, t), mx.sql(q, t=t).compute(device=device)
    assert got.column_names == exp.column_names
    assert _rows(got) == _rows(exp)


# ---- NaN / inf / overflow in power: values, not errors ----------------------------------------------------------

def test_power_edge_values_are_doubles_like_duckdb():
    t = pa.table({"a": pa.array([4.0, -8.0, 0.0, 2.0, None]), "n": pa.array([0.5, 1.0 / 3, -1.0, 1000.0, 2.0])})
    q = "SELECT power(a, n) AS x, power(a, 0.5) AS y, a ** 2 AS z FROM t"
    exp, got = _duck(q, t), mx.sql(q, t=t).compute(device="cpu")
    assert got.schema.types == exp.schema.types
    assert _rows(got) == _rows(exp)


# ---- the NULL sources are registered in _can_be_null ------------------------------------------------------------

def test_every_null_source_is_registered_in_can_be_null():
    from mxframe.lazy_expr import Expr
    a, b = col("a"), col("b")
    assert _can_be_null(Expr("nullify", a)) is True
    assert _can_be_null(Expr("idiv", a, b)) is True
    assert _can_be_null(Expr("idiv", a, lit(0))) is True
    assert _can_be_null(Expr("idiv", a, lit(2))) is False
    assert _can_be_null(Expr("add", Expr("idiv", a, b), lit(1))) is True
    assert _can_be_null(Expr("mod", a, b)) is True and _can_be_null(Expr("mod", a, lit(3))) is False
    assert _can_be_null(Expr("sqrt", a)) is False and _can_be_null(Expr("ln", a)) is False      # they raise instead
    assert _can_be_null(Expr("pow", a, lit(2.0))) is False
    assert _can_be_null(a.cast("int8", try_=True)) is True and _can_be_null(a.cast("int8")) is False     # TRY_CAST


# ---- the grouped decimal modulo (the review's shape) ------------------------------------------------------------

def test_a_grouped_decimal_modulo_equals_duckdb():
    t = pa.table({"g": pa.array([1, 1, 2, 2, 3, None], pa.int64()), "a": pa.array([7, -7, 0, 5, None, 9], pa.int64()),
                  "f": pa.array([7.5, -7.5, 0.0, 2.0, None, -1.0])})
    for q in ("SELECT g, sum(f % 2.5) AS s FROM t GROUP BY g ORDER BY g",
              "SELECT g, sum(a % 2.5) AS s FROM t GROUP BY g ORDER BY g",
              "SELECT sum(f % 2.5) AS s FROM t", "SELECT g, max(a % 2.5) AS s FROM t GROUP BY g ORDER BY g"):
        assert _rows(mx.sql(q, t=t).compute(device="cpu")) == _rows(_duck(q, t)), q
