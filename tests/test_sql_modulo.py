"""F-094: SQL `%` and MOD(a, b). DuckDB's result follows the dividend's sign (C fmod: -7 % 3 = -1, 7 % -3 = 1),
a zero divisor gives NULL (integers and floats), NULL in gives NULL out, NaN stays NaN, `x % inf` is x. Every
shape is checked against DuckDB on the cpu and, where a CUDA GPU exists, on the gpu."""
import math
from decimal import Decimal

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import gpu_stream

NAN, INF = float("nan"), float("inf")


def _t():
    a = [-7, 7, 7, None, 5, -6, 7, 0, -1, 9, 100, -100, 13, -13, 2, 3]
    b = [3, -3, 0, 3, None, 3, -3, 5, 2, 0, -7, 7, 13, -13, -2, 1]
    x = [-7.5, 7.5, 7.0, None, 5.0, -6.0, NAN, INF, -INF, 0.0, 5.5, -5.5, 1e300, -0.0, 2.25, 8.0]
    y = [2.0, -2.0, 0.0, 2.0, None, 3.0, INF, 2.0, 2.0, 0.0, -2.5, 2.5, 7.0, 1.0, NAN, -INF]
    n = len(a)
    return pa.table({"i": pa.array(range(n), pa.int64()), "a": pa.array(a, pa.int64()),
                     "b": pa.array(b, pa.int64()), "x": pa.array(x, pa.float64()),
                     "y": pa.array(y, pa.float64()),
                     "g": pa.array([k % 4 for k in range(n)], pa.int64())})


def _big():
    r = np.random.default_rng(3)
    n = 3000
    return pa.table({"i": pa.array(range(n), pa.int64()), "a": pa.array(r.integers(-500, 500, n), pa.int64()),
                     "v": pa.array(r.integers(1, 100, n), pa.int64())})


def _typed():
    """Integer widths and signedness: int32 and uint64 columns, values past 2**53 and 2**63."""
    big = 2 ** 60 + 3
    return pa.table({
        "i": pa.array(range(8), pa.int64()),
        "i32": pa.array([7, -7, 0, None, 2 ** 30, -(2 ** 30), 10, 3], pa.int32()),
        "d32": pa.array([3, 0, -3, 2, None, 5, 0, 1], pa.int32()),
        "u64": pa.array([2 ** 64 - 1, 2 ** 63, 5, None, 2 ** 63 + 7, 0, 18, 2 ** 64 - 2], pa.uint64()),
        "s64": pa.array([10, -10, 3, 7, 0, -1, None, 2 ** 62], pa.int64()),
        "b64": pa.array([big, -big, 2 ** 53 + 1, -(2 ** 53) - 1, 2 ** 62, 9, None, 2 ** 63 - 2], pa.int64()),
    })


def _dec(v):
    """A whole DECIMAL (DuckDB's HUGEINT) as an int, a fractional one as the nearest double."""
    return int(v) if v == v.to_integral_value() else float(v)


def _norm(rows):
    """Rows with NaN as a string and DuckDB's DECIMAL as the nearest double (what the engine returns)."""
    return [tuple("nan" if isinstance(v, float) and math.isnan(v) else _dec(v) if isinstance(v, Decimal) else v
                  for v in row) for row in rows]


def _check(q, device="cpu"):
    t, big, ty = _t(), _big(), _typed()                         # noqa: F841 (DuckDB finds them by name)
    exp = duckdb.sql(q).arrow()
    exp = exp.read_all() if isinstance(exp, pa.RecordBatchReader) else exp
    got = mx.sql(q, t=t, big=big, ty=ty).compute(device=device)
    assert got.column_names == exp.column_names, q
    for g, e in zip(got.schema, exp.schema):                    # 1 == 1.0 in Python: compare the kinds too
        if pa.types.is_integer(e.type) or pa.types.is_floating(e.type):
            assert (pa.types.is_integer(g.type), pa.types.is_floating(g.type)) == \
                (pa.types.is_integer(e.type), pa.types.is_floating(e.type)), (q, g, e)
    assert _norm(zip(*got.to_pydict().values())) == _norm(zip(*exp.to_pydict().values())), q


QUERIES = [
    # SELECT: sign follows the dividend, zero divisor and NULL give NULL
    "SELECT i, a % b AS r FROM t ORDER BY i",
    "SELECT i, MOD(a, b) AS r FROM t ORDER BY i",
    "SELECT i, a % 3 AS r FROM t ORDER BY i",
    "SELECT i, a % -3 AS r FROM t ORDER BY i",
    "SELECT i, 17 % b AS r FROM t ORDER BY i",
    "SELECT i, a % 0 AS r FROM t ORDER BY i",
    "SELECT i, x % y AS r FROM t ORDER BY i",
    "SELECT i, MOD(x, y) AS r FROM t ORDER BY i",
    "SELECT i, x % 2 AS r FROM t ORDER BY i",
    "SELECT i, x % 2.5 AS r FROM t ORDER BY i",
    "SELECT i, a % y AS r FROM t ORDER BY i",                   # int % float
    "SELECT i, x % b AS r FROM t ORDER BY i",                   # float % int
    "SELECT i, (a + b) % (b - a) AS r FROM t ORDER BY i",
    "SELECT i, a % b + b AS r FROM t ORDER BY i",
    "SELECT i, a % b % 4 AS r FROM t ORDER BY i",
    # WHERE
    "SELECT i FROM t WHERE a % 2 = 0 ORDER BY i",
    "SELECT i FROM t WHERE a % 2 = 1 ORDER BY i",               # -7 % 2 is -1: not 1
    "SELECT i FROM t WHERE a % 2 = -1 ORDER BY i",
    "SELECT i FROM t WHERE a % b = 1 ORDER BY i",
    "SELECT i FROM t WHERE MOD(a, b) < 0 ORDER BY i",
    "SELECT i FROM t WHERE x < 100 AND x > -100 AND x % 2 > 0.5 ORDER BY i",
    "SELECT i FROM t WHERE x % y = 0 ORDER BY i",
    "SELECT i FROM big WHERE a % 7 = 3 ORDER BY i",
    "SELECT i FROM big WHERE a % 7 = -3 ORDER BY i",
    "SELECT count(*) AS n FROM big WHERE a % 2 = 0",
    "SELECT sum(v) AS s FROM big WHERE a % 10 = 0",
    # GROUP BY
    "SELECT a % 3 AS m, count(*) AS n FROM t GROUP BY a % 3 ORDER BY m",
    "SELECT i % 10 AS m, count(*) AS n, sum(v) AS s FROM big GROUP BY i % 10 ORDER BY m",
    "SELECT a % 10 AS m, count(*) AS n, sum(v) AS s FROM big GROUP BY a % 10 ORDER BY m",
    "SELECT a % 10 AS m, sum(v) AS s FROM big GROUP BY 1 ORDER BY 1",
    "SELECT g, sum(a % 3) AS s, count(a % 3) AS n FROM t GROUP BY g ORDER BY g",
    "SELECT g, sum(x % 2) AS s FROM t WHERE x < 100 AND x > -100 GROUP BY g ORDER BY g",
    # HAVING
    "SELECT a % 10 AS m, count(*) AS n FROM big GROUP BY a % 10 HAVING a % 10 > 3 ORDER BY m",
    "SELECT g, sum(a) AS s FROM t GROUP BY g HAVING sum(a) % 2 = 0 ORDER BY g",
    "SELECT g, sum(v) AS s FROM (SELECT i % 5 AS g, v FROM big) GROUP BY g HAVING sum(v) % 3 = 1 ORDER BY g",
    "SELECT i % 7 AS g, count(*) AS n FROM big GROUP BY i % 7 HAVING count(*) % 2 = 1 ORDER BY g",
    # ORDER BY an expression, DISTINCT
    "SELECT i FROM t ORDER BY a % 3, i",
    "SELECT DISTINCT a % 4 AS m FROM t ORDER BY m",
    # computed integers stay integers: from a subquery, an aggregate, a cast column
    "SELECT i, m % 0 AS r FROM (SELECT i, i32 + s64 AS m FROM ty) ORDER BY i",
    "SELECT i, m % 3 AS r FROM (SELECT i, i32 + d32 AS m FROM ty) ORDER BY i",
    "SELECT i, m % d32 AS r FROM (SELECT i, i32 - s64 AS m, d32 FROM ty) ORDER BY i",
    "SELECT sum(i32) % 2 AS r FROM ty",
    "SELECT sum(i32) % 0 AS r FROM ty",
    "SELECT i % 3 AS g, sum(s64) % 4 AS r FROM ty GROUP BY i % 3 ORDER BY g",
    "SELECT count(*) % 3 AS r FROM big",
    "SELECT i, i32 % 0 AS r FROM ty ORDER BY i",
    "SELECT i, i32 % d32 AS r FROM ty ORDER BY i",
    "SELECT i, i32 % 7 AS r FROM ty ORDER BY i",
    "SELECT i, 100 % d32 AS r FROM ty ORDER BY i",
    # past 2**53 and 2**63
    "SELECT i, b64 % 10 AS r FROM ty ORDER BY i",
    "SELECT i, b64 % -7 AS r FROM ty ORDER BY i",
    "SELECT i, b64 % s64 AS r FROM ty ORDER BY i",
    "SELECT i, (b64 - i) % 10 AS r FROM ty ORDER BY i",
    "SELECT i, u64 % 10 AS r FROM ty ORDER BY i",
    "SELECT i, u64 % 1000000007 AS r FROM ty ORDER BY i",
    "SELECT i, u64 % u64 AS r FROM ty ORDER BY i",
    "SELECT i, MOD(u64, 3) AS r FROM ty ORDER BY i",
    "SELECT i FROM ty WHERE u64 % 10 = 5 ORDER BY i",
    "SELECT u64 % 10 AS g, count(*) AS n FROM ty GROUP BY u64 % 10 ORDER BY g",
    # a decimal literal is exact in DuckDB, also next to integers past 2**53; a decimal zero divisor is NULL
    "SELECT i, b64 % 2.5 AS r FROM ty ORDER BY i",
    "SELECT i, b64 % 0.1 AS r FROM ty ORDER BY i",
    "SELECT i, MOD(i32, 0.0) AS r FROM ty ORDER BY i",
    "SELECT i, i32 % 0.5 AS r FROM ty ORDER BY i",
    "SELECT i, 7.5 % d32 AS r FROM ty ORDER BY i",
    "SELECT i, -7.5 % s64 AS r FROM ty ORDER BY i",
    "SELECT i, i32 % -2.5 AS r FROM ty ORDER BY i",
    "SELECT i FROM ty WHERE b64 % 2.5 = 0.5 ORDER BY i",
    "SELECT i, x % 0.0 AS r FROM t ORDER BY i",                  # a double column stays IEEE: NaN
    "SELECT i, b64 % 1e3 AS r FROM ty ORDER BY i",              # an exponent makes it DOUBLE: b64 rounds first
    "SELECT i, b64 % 1000.0 AS r FROM ty ORDER BY i",           # a point makes it DECIMAL: exact
    "SELECT i, u64 % 2.5 AS r FROM ty ORDER BY i",
    "SELECT i, b64 % 9.200000001 AS r FROM ty ORDER BY i",      # k * s = 9.2e18, the last that fits int64
    "SELECT i, u64 % 9.200000001 AS r FROM ty ORDER BY i",
    "SELECT i, 9.5 % s64 AS r FROM ty ORDER BY i",
    "SELECT i, 9.5 % b64 AS r FROM ty ORDER BY i",
    # mixed signed / unsigned
    "SELECT i, u64 % s64 AS r FROM ty ORDER BY i",
    "SELECT i, s64 % u64 AS r FROM ty ORDER BY i",
    "SELECT i, u64 % -10 AS r FROM ty ORDER BY i",
    "SELECT i, u64 % i32 AS r FROM ty ORDER BY i",
    "SELECT i, u64 % b64 AS r FROM ty ORDER BY i",
]


@pytest.mark.parametrize("q", QUERIES)
def test_modulo_matches_duckdb(q):
    _check(q)


@pytest.mark.skipif(not gpu_stream.cuda_available(), reason="no CUDA GPU")
@pytest.mark.parametrize("q", QUERIES)
def test_modulo_matches_duckdb_on_gpu(q):
    _check(q, device="gpu")


def test_values_a_user_reads():
    got = mx.sql("SELECT -7 % 3 AS a, 7 % -3 AS b, 7 % 0 AS c, MOD(-7.5, 2) AS d FROM t LIMIT 1", t=_t()).compute(device="cpu")
    assert got.to_pydict() == {"a": [-1], "b": [1], "c": [None], "d": [-1.5]}


def test_decimal_literal_past_the_exact_range_raises_instead_of_rounding():
    with pytest.raises(OverflowError, match="too many digits"):         # k * s = 9.3e18 > int64
        mx.sql("SELECT b64 % 9.300000001 AS r FROM ty", ty=_typed()).compute(device="cpu")
