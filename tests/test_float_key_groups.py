"""F-077: a NULL in a float group key sends the grouping to the Arrow fallback; -0.0 and 0.0 must be
one group (reported as +0.0) and every NaN one group, whatever its payload, like DuckDB. Keys with
-inf / +inf / NULL stay apart, and ORDER BY puts -inf ... inf, NaN, NULL in DuckDB's order."""
import math
import struct

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, sql
from mxframe.np_arrow import array as A

BITS = {np.float64: ("<Q", "<d"), np.float32: ("<I", "<f")}
NAN_PAYLOADS = {np.float64: (0x7FF8000000000000, 0xFFF8000000001234, 0x7FF0000000000001),
                np.float32: (0x7FC00000, 0xFFC01234, 0x7F800001)}


def _nans(dt):
    ib, fb = BITS[dt]
    return [struct.unpack(fb, struct.pack(ib, p))[0] for p in NAN_PAYLOADS[dt]]


def _keys(dt):
    """Float keys with NULL, -0.0, 0.0, NaNs with three different payloads and +-inf."""
    n1, n2, n3 = _nans(dt)
    vals = [-0.0, 0.0, n1, 1.5, None, n2, math.inf, -math.inf, 0.0, n3, -0.0, None, 1.5]
    mask = np.array([v is None for v in vals])
    data = np.array([0.0 if v is None else v for v in vals], dtype=dt)
    return np.ma.array(data, mask=mask)


def _table(dt):
    k = _keys(dt)
    n = len(k)
    return pa.table({"k": A(k, type=pa.from_numpy_dtype(dt)), "v": A(np.arange(1, n + 1, dtype=np.int64)),
                     "g": A((np.arange(n) % 2).astype(np.int32)), "id": A(np.arange(n, dtype=np.int64))})


def _canon(x, zero_sign=False):
    """Comparable form of a float key: NaN equals NaN; the sign of a zero only when asked (a group
    reports the zero of one of its rows, and which one is up to the engine)."""
    if x is None:
        return "null"
    if math.isnan(x):
        return "nan"
    return repr(float(x)) if zero_sign or x != 0 else "0.0"


def _rows(table, names, zero_sign=False):
    d = table.to_pydict()
    return [tuple(_canon(x, zero_sign) if n == "k" else x for n, x in zip(names, r))
            for r in zip(*[d[n] for n in names])]


def _duck(query, **tables):
    con = duckdb.connect()
    for name, t in tables.items():
        con.register(name, t)
    return con.execute(query).fetch_arrow_table()


def _same(got, ref, names, ordered=False, zero_sign=False):
    a, b = _rows(got, names, zero_sign), _rows(ref, names, zero_sign)
    if not ordered:
        a, b = sorted(a, key=str), sorted(b, key=str)
    assert a == b, f"\nmxframe: {a}\nduckdb:  {b}"


DTYPES = [np.float64, np.float32]


@pytest.mark.parametrize("dt", DTYPES)
def test_single_table(dt):
    t = _table(dt)
    got = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu")
    ref = _duck("select k, sum(v)::double as s, count(v) as n from t group by k", t=t)
    assert got.num_rows == ref.num_rows == 6
    _same(got, ref, ["k", "s", "n"])


@pytest.mark.parametrize("dt", DTYPES)
def test_two_keys(dt):
    t = _table(dt)
    got = LazyFrame(t).groupby("k", "g").agg(col("v").sum().alias("s")).compute(device="cpu")
    ref = _duck("select k, g, sum(v)::double as s from t group by k, g", t=t)
    assert got.num_rows == ref.num_rows
    _same(got, ref, ["k", "g", "s"])


@pytest.mark.parametrize("dt", DTYPES)
def test_fused_join_partner_column(dt):
    t = _table(dt)
    fact = pa.table({"id": A(np.arange(40, dtype=np.int64) % 13), "v": A(np.arange(40, dtype=np.float64))})
    dim = t.select(["id", "k"])
    got = (LazyFrame(fact).join(LazyFrame(dim), on="id").groupby("k")
           .agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    ref = _duck("select k, sum(v) as s, count(v) as n from fact join dim using (id) group by k", fact=fact, dim=dim)
    assert got.num_rows == ref.num_rows == 6
    _same(got, ref, ["k", "s", "n"])


@pytest.mark.parametrize("dt", DTYPES)
def test_sql_group_by(dt):
    t = _table(dt)
    got = sql("select k, sum(v) as s, count(v) as n from t group by k", t=t).compute(device="cpu")
    ref = _duck("select k, sum(v)::double as s, count(v) as n from t group by k", t=t)
    assert got.num_rows == 6
    _same(got, ref, ["k", "s", "n"])


@pytest.mark.parametrize("dt", DTYPES)
def test_distinct(dt):
    t = _table(dt).select(["k"])
    got = LazyFrame(t).distinct().compute(device="cpu")
    ref = _duck("select distinct k from t", t=t)
    assert got.num_rows == ref.num_rows == 6
    _same(got, ref, ["k"])


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("desc", [False, True])
def test_order_by_key(dt, desc):
    t = _table(dt)
    got = (LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).sort("k", descending=desc)
           .compute(device="cpu"))
    ref = _duck(f"select k, sum(v)::double as s from t group by k order by k {'desc' if desc else ''}", t=t)
    _same(got, ref, ["k", "s"], ordered=True)


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("first_zero", [-0.0, 0.0])
def test_group_reports_the_zero_of_its_first_row_like_duckdb(dt, first_zero):
    other = 0.0 if first_zero == -0.0 else -0.0
    k = np.ma.array(np.array([first_zero, other, other, 2.0], dtype=dt), mask=[False, False, False, True])
    t = pa.table({"k": A(k, type=pa.from_numpy_dtype(dt)), "v": A(np.arange(1, 5, dtype=np.int64))})
    got = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).compute(device="cpu")
    ref = _duck("select k, sum(v)::double as s from t group by k", t=t)
    assert got.num_rows == ref.num_rows == 2
    _same(got, ref, ["k", "s"], zero_sign=True)


@pytest.mark.parametrize("dt", DTYPES)
def test_no_null_float_key_groups_the_same_way(dt):
    """The fused pipeline groups through Arrow dictionary encoding too: same bug without a NULL."""
    t = _table(dt)
    t = t.filter(A(~np.asarray(t.column("k").is_null().to_numpy(zero_copy_only=False))))
    got = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).compute(device="cpu")
    ref = _duck("select k, sum(v)::double as s from t group by k", t=t)
    assert got.num_rows == ref.num_rows == 5
    _same(got, ref, ["k", "s"])
    got2 = LazyFrame(t).groupby("k", "g").agg(col("v").sum().alias("s")).compute(device="cpu")
    ref2 = _duck("select k, g, sum(v)::double as s from t group by k, g", t=t)
    assert got2.num_rows == ref2.num_rows
    _same(got2, ref2, ["k", "g", "s"])
