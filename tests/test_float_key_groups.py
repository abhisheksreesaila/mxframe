"""F-077: -0.0 and 0.0 are one group and every NaN is one group, whatever its payload, like DuckDB,
also when a NULL in the key sends the grouping to the Arrow fallback. A group reports the key value of
its first row, bit for bit (DuckDB does not normalise a zero or a NaN): the bit tests compare the sign
of a zero and the NaN payload. Keys with -inf / +inf / NULL stay apart, and ORDER BY puts
-inf ... inf, NaN, NULL in DuckDB's order."""
import math
import struct

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx

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


# ---- the key value a group reports, bit for bit, on every grouping path -------------------------

ODD_NAN = {np.float64: 0xFFF8000000001234, np.float32: 0xFFC01234}
UINT = {np.float64: np.uint64, np.float32: np.uint32}


def _bits(table, name):
    """Raw bits of a float column (None for NULL): tells -0.0 from 0.0 and one NaN payload from another."""
    a = table.column(name).combine_chunks()
    x = np.frombuffer(a.buffers()[1], dtype=UINT[np.float64 if a.type == pa.float64() else np.float32],
                      count=a.offset + len(a))[a.offset:]
    return [None if not ok else int(b) for b, ok in zip(x, a.is_valid().to_pylist())]


def _bit_rows(table, names):
    d = {n: (_bits(table, n) if n == "k" else table.column(n).to_pylist()) for n in names}
    return sorted(zip(*[d[n] for n in names]), key=str)


def _native_duck(query, **tables):
    """DuckDB over its own tables: a registered Arrow table is scanned on another path."""
    con = duckdb.connect()
    for name, t in tables.items():
        con.register("_" + name, t)
        con.execute(f"create table {name} as select * from _{name}")
    return con.execute(query).fetch_arrow_table()


def _first_row_table(dt, first_bits, with_null, n=1):
    """First row is the value under test; the other forms of the same key follow it (x n)."""
    u = UINT[dt]
    zero_neg = u(1) << u(np.dtype(dt).itemsize * 8 - 1)
    nan_a = u(0x7FF8000000000000 if dt is np.float64 else 0x7FC00000)
    keys = [u(first_bits), zero_neg, u(0), nan_a, u(ODD_NAN[dt]), u(first_bits)] * n + [u(0)] * 2
    k = np.array(keys, dtype=u).view(dt)
    mask = np.zeros(len(k), bool)
    if with_null:
        mask[-1] = True
    k = np.ma.array(k, mask=mask)
    return pa.table({"k": A(k, type=pa.from_numpy_dtype(dt)), "v": A(np.arange(1, len(k) + 1, dtype=np.int64)),
                     "id": A(np.arange(len(k), dtype=np.int64))})


FIRSTS = {"neg_zero": lambda dt: UINT[dt](1) << UINT[dt](np.dtype(dt).itemsize * 8 - 1),
          "pos_zero": lambda dt: UINT[dt](0),
          "odd_nan": lambda dt: UINT[dt](ODD_NAN[dt])}


@pytest.fixture(params=["fused", "native"])
def grouping(request, monkeypatch):
    """Which engine groups: the fused pipeline's Arrow key encoding, or the native hashed path."""
    monkeypatch.setenv("MXFRAME_NATIVE", "0" if request.param == "fused" else "1")
    return request.param


def _assert_path(grouping, lf):
    if grouping == "fused":
        assert lf.last_compile_provenance["path"] == "fused_pipeline_cpu", lf.last_compile_provenance
    else:
        assert nx.last_group_path() == "hash"


@pytest.mark.parametrize("first", list(FIRSTS))
@pytest.mark.parametrize("dt", DTYPES)
def test_group_reports_its_first_rows_bits_without_null(dt, first, grouping, request):
    if grouping == "native" and first != "pos_zero":
        request.applymarker(pytest.mark.xfail(strict=True, reason=(
            "F-069's native hashed path stores the canonical key (+0.0, one NaN) in its hash records "
            "and reports that, where DuckDB reports the first row's bits: a kernel change, see the "
            "follow-up in docs/plan.md F-077")))
    if grouping == "native" and dt is np.float32:
        pytest.skip("the native hashed path takes float64 keys")
    if not nx.available() and grouping == "native":
        pytest.skip("libmxexec_cpu.so not built")
    t = _first_row_table(dt, FIRSTS[first](dt), with_null=False, n=30_000 if grouping == "native" else 1)
    lf = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"))
    got = lf.compute(device="cpu")
    ref = _native_duck("select k, sum(v)::double as s from t group by k", t=t)
    _assert_path(grouping, lf)
    assert got.num_rows == ref.num_rows == 2
    assert _bit_rows(got, ["k", "s"]) == _bit_rows(ref, ["k", "s"])


@pytest.mark.parametrize("first", list(FIRSTS))
@pytest.mark.parametrize("dt", DTYPES)
def test_group_reports_its_first_rows_bits_with_null(dt, first):
    """A NULL key row: the native path declines, the fused encoder and the Arrow fallback answer."""
    t = _first_row_table(dt, FIRSTS[first](dt), with_null=True)
    got = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).compute(device="cpu")
    ref = _native_duck("select k, sum(v)::double as s from t group by k", t=t)
    assert got.num_rows == ref.num_rows == 3
    assert _bit_rows(got, ["k", "s"]) == _bit_rows(ref, ["k", "s"])
    got = LazyFrame(t.select(["k"])).distinct().compute(device="cpu")
    ref = _native_duck("select distinct k from t", t=t)
    assert _bit_rows(got, ["k"]) == _bit_rows(ref, ["k"])


@pytest.mark.parametrize("first", list(FIRSTS))
@pytest.mark.parametrize("dt", DTYPES)
def test_fused_join_partner_key_without_null(dt, first, monkeypatch):
    """The partner column of a fused join reaches the fused pipeline's key encoder (no NULL, so no fallback)."""
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    dim = _first_row_table(dt, FIRSTS[first](dt), with_null=False).select(["id", "k"])
    fact = pa.table({"id": A(np.arange(40, dtype=np.int64) % dim.num_rows), "v": A(np.arange(40, dtype=np.float64))})
    lf = (LazyFrame(fact).join(LazyFrame(dim), on="id").groupby("k").agg(col("v").sum().alias("s")))
    got = lf.compute(device="cpu")
    monkeypatch.delenv("MXFRAME_NATIVE")
    ref = _native_duck("select k, sum(v) as s from fact join dim using (id) group by k", fact=fact, dim=dim)
    assert lf.last_compile_provenance["path"].startswith("fused_pipeline"), lf.last_compile_provenance
    assert got.num_rows == ref.num_rows == 2
    assert _bit_rows(got, ["k", "s"]) == _bit_rows(ref, ["k", "s"])


def test_an_all_null_float_key_encodes_without_error():
    from mxframe import arrow_dict
    enc = arrow_dict.encode_key(pa.chunked_array([pa.array([None, None], pa.float64())]))
    assert len(enc.dictionary) == 0 and enc.indices.null_count == 2
