"""Shared by the F-117 / F-118 / F-122 / F-123 SQL tests: one table, the same rows in a NATIVE DuckDB table, a
`check(q, dev)` that runs the query in both and compares column names, values and column types, and the
`cfg` fixture (cpu / gpu x native executor on / off). Expected answers always come from DuckDB.

`dd` is an int32 day number (the TPC-H Parquet files store dates so); DuckDB sees it as a DATE.
"""
import datetime as dt
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import gpu_stream

GPU = gpu_stream.cuda_available()
D0 = dt.date(2024, 1, 1)
EPOCH = dt.date(1970, 1, 1)


def _table():
    n = 12
    dates = [None if k % 5 == 2 else D0 + dt.timedelta(days=k * 40) for k in range(n)]
    return pa.table({
        "id": pa.array(range(n), pa.int64()),
        "g": pa.array(["a", "a", "b", None, "b", "a", "c", None, "b", "a", "c", "b"]),
        "gi": pa.array([None if k % 6 == 5 else k % 3 for k in range(n)], pa.int64()),
        "x": pa.array([None if k % 4 == 3 else k * 2 for k in range(n)], pa.int64()),
        "f": pa.array([None if k % 5 == 1 else k * 1.5 - 6 for k in range(n)], pa.float64()),
        "d": pa.array(dates, pa.date32()),
        "dd": pa.array([None if d is None else (d - EPOCH).days for d in dates], pa.int32()),
    })


T = _table()
U = pa.table({"k": pa.array([0, 1, 2, 5, None], pa.int64()), "w": pa.array([10, 20, 30, 40, 50], pa.int64())})
CON = duckdb.connect()
CON.register("_t", T)
CON.execute("create table t as select id, g, gi, x, f, d, DATE '1970-01-01' + dd as dd from _t")
CON.register("_u", U)
CON.execute("create table u as select * from _u")


@pytest.fixture(params=[(dev, nat) for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-native{p[1]}")
def cfg(request, monkeypatch):
    dev, nat = request.param
    if dev == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    if nat == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", nat)
    return dev, nat == "1"


def _norm(v):
    return float(v) if isinstance(v, Decimal) else v


def rows(x):
    cols = [x.column(i).to_pylist() for i in range(x.num_columns)]
    return [tuple(_norm(c[r]) for c in cols) for r in range(x.num_rows)]


def kind(t):
    """The type as the user sees it: integers keep their width (INTEGER is not BIGINT), DECIMAL and DOUBLE are
    both a float, a NULL-typed column is whatever DuckDB says (it has no values to tell)."""
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_integer(t):
        return f"int{t.bit_width}"
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    return "date" if pa.types.is_date(t) else str(t)


def check(q, dev, types=True):
    """Run ``q`` on DuckDB and on mxframe: the same names, rows (in order) and types. Returns DuckDB's table."""
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T, u=U).compute(device=dev)
    assert got.column_names == want.column_names, q
    assert rows(got) == rows(want), q
    if types:
        for name, g, w in zip(want.column_names, got.schema.types, want.schema.types):
            if pa.types.is_null(g) and w.bit_width == 32 and pa.types.is_integer(w):   # a bare NULL: no values to type
                continue
            assert kind(g) == kind(w), f"{q}: column {name} is {g}, DuckDB gives {w}"
    return want


def refused(q, dev, match=None):
    """DuckDB raises on ``q`` and so does mxframe (a ValueError / NotImplementedError saying why)."""
    with pytest.raises(Exception):
        CON.execute(q)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        mx.sql(q, t=T, u=U).compute(device=dev)
