"""F-110: float32 arithmetic keeps a FLOAT result, rounded to float32 after every operation, as DuckDB does.

DuckDB types FLOAT + INTEGER / DECIMAL literal / FLOAT as FLOAT (the other operand is cast to float32 and the result
is rounded to float32 per operation); only a DOUBLE operand (exponent literal, float64 column) makes DOUBLE. sqrt,
ln, power, exp are DOUBLE functions of a FLOAT; abs, coalesce, CASE, cast(.. as float) keep FLOAT; SUM / AVG of a
FLOAT are DOUBLE, MIN / MAX FLOAT. The engines used to compute float32 expressions in double: `g * 0.5 > 0.05`,
`g + 0.1 > 0.2` and `g * 2 = 0.2` gave other rows than DuckDB on the rows near a float32 rounding edge.

Expected answers come from a NATIVE DuckDB table. Every case runs on cpu and gpu, native on and off.
"""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import gpu_stream

GPU = gpu_stream.cuda_available()


def _table():
    rng = np.random.default_rng(110)
    special = np.array([0.1, 0.10000001, 0.09999999, 0.2, 0.05, 0.25, 0.3, 0.1 + 1e-8, 16777216.0, 16777215.0,
                        1e-30, 0.0, -0.1, -0.05, 1.0, 3.0, 0.0333, 0.33333334], np.float32)
    g = np.concatenate([special, rng.random(3000).astype(np.float32),
                        (rng.random(1000) * 1e4).astype(np.float32)])
    n = len(g)
    h = np.concatenate([special[::-1], rng.random(n - len(special)).astype(np.float32)])
    return pa.table({
        "k": pa.array(np.arange(n), pa.int64()),
        "g": pa.array(g, pa.float32()),
        "h": pa.array(h, pa.float32()),
        "i": pa.array(rng.integers(-30, 30, n), pa.int32()),
        "b": pa.array(rng.integers(-(1 << 40), 1 << 40, n), pa.int64()),
        "f": pa.array(g.astype(np.float64) * 1.0000001, pa.float64()),
    })


T = _table()
CON = duckdb.connect()
CON.register("_t", T)
CON.execute("create table t as select * from _t")


@pytest.fixture(params=[(dev, nat) for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-native{p[1]}")
def cfg(request, monkeypatch):
    dev, nat = request.param
    if dev == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    if nat == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", nat)
    return dev


def _rows(t):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    return [tuple(c[r] for c in cols) for r in range(t.num_rows)]


def _check(dev, q, rel=0.0):
    want = CON.execute(q).fetchall()
    got = _rows(mx.sql(q, t=T).compute(device=dev))
    assert len(got) == len(want), (q, len(got), len(want))
    for g, w in zip(got, want):
        assert len(g) == len(w) and all(
            a == pytest.approx(b, rel=rel, abs=0) if isinstance(b, float) and a is not None and rel else a == b
            for a, b in zip(g, w)), (q, g, w)


# ── WHERE: a FLOAT expression against a DECIMAL / INTEGER literal compares in float32 ─────────────────────────

WHERES = [
    "g * 0.5 > 0.05", "g * 0.5 >= 0.05", "g + 0.1 > 0.2", "g + 0.1 = 0.2", "g * 2 = 0.2", "g * 2 <> 0.2",
    "g - 0.05 >= 0.05", "g / 3 < 0.0333", "g / 3 = 0.33333334", "0.2 < g + 0.1", "0.05 >= g * 0.5",
    "g * g > 0.01", "g + h > 0.3", "g * h <= 0.05", "g - h < 0", "(g + 0.1) * 3 > 0.9", "g * 0.5 + 0.1 > 0.15",
    "g * i > 0.5", "g + i = 0.1", "g + 16777217 > 16777217", "g * 3 > 16777216",
    "-g * 0.5 < -0.05", "g * 0.5 between 0.05 and 0.1", "g * 2 in (0.2, 0.4, 0.5)", "g + 0.1 not in (0.2)",
    "g * 0.5 > 0.05 and h + 0.1 < 0.3", "g * 0.5 > 0.05 or g + 0.1 = 0.2",
    "g * 1e0 > 0.1", "g * 0.5 > 1e-30", "(g * 0.5) * 1e0 = 0.05",         # a DOUBLE operand widens
    "g * f > 0.05", "g + b > 0.1", "case when g * 0.5 > 0.05 then 1 else 0 end = 1",
    "abs(g * 0.5) > 0.05", "coalesce(g + 0.1, 0) = 0.2", "cast(g * 0.5 as float) > 0.05",
]


@pytest.mark.parametrize("w", WHERES)
def test_where(cfg, w):
    _check(cfg, f"select k from t where {w} order by k")


# ── SELECT: the FLOAT value itself, and its type ──────────────────────────────────────────────────────────────

SELECTS = [
    "g * 0.5", "g + 0.1", "g - 0.05", "g / 3", "g * 2", "g * g", "g + h", "g * h", "g - h", "g / (h + 1)",
    "(g + 0.1) * 3", "g * i", "g + i", "i + g", "g + 16777217", "-g * 0.5", "g * 0.5 + 0.1", "0.1 + g", "3 / (g + 1)",
    "g * 1e0", "g * 0.5 * 1e0", "g + f", "g + b", "abs(g * 0.5)", "coalesce(g + 0.1, 0)", "cast(g * 0.5 as float)",
    "case when g > 0.5 then g * 0.5 else g + 0.1 end",
    "sqrt(g * 2 + 1)", "ln(g + 1.1)", "power(g + 0.1, 2)", "sqrt(abs(g) + h)",
]


@pytest.mark.parametrize("e", SELECTS)
def test_select_value_and_type(cfg, e):
    q = f"select k, {e} as v from t order by k"
    _check(cfg, q)
    want = CON.execute(f"select typeof({e}) from t limit 1").fetchone()[0]
    got = mx.sql(q, t=T).compute(device=cfg).schema.field("v").type
    assert {"FLOAT": pa.float32(), "DOUBLE": pa.float64()}[want] == got, (e, want, got)


# ── aggregates: SUM / AVG of a FLOAT are DOUBLE (of the rounded values), MIN / MAX FLOAT ──────────────────────

AGGS = [
    "sum(g * 0.5)", "sum(g + 0.1)", "sum(g * g)", "sum(g * h)", "avg(g * 0.5)", "min(g * 0.5)", "max(g + 0.1)",
    "min(g / 3)", "max(g * i)", "sum(g * 2 - h)", "count(case when g * 0.5 > 0.05 then 1 end)",
    "sum(case when g * 0.5 > 0.05 then g + 0.1 else 0 end)", "sum(case when g + 0.1 > 0.2 then g * 0.5 end)",
]


@pytest.mark.parametrize("a", AGGS)
def test_aggregate(cfg, a):
    _check(cfg, f"select {a} as v from t", rel=1e-9)
    _check(cfg, f"select abs(i) % 4 as gk, {a} as v from t group by 1 order by 1", rel=1e-9)


def test_where_with_aggregate(cfg):
    _check(cfg, "select count(*), sum(g + 0.1) as s from t where g * 0.5 > 0.05 and g + 0.1 > 0.2", rel=1e-9)
    _check(cfg, "select count(*) from t where g * 2 = 0.2")
    _check(cfg, "select abs(i) % 3 as gk, count(*) from t where g + 0.1 > 0.2 group by 1 having sum(g * 0.5) > 0.5 order by 1")


def test_boundary_rows_are_in_the_data():
    """The cases above only mean something when the double and the float32 computation disagree on some rows."""
    g = T.column("g").to_numpy().astype(np.float32)
    d = (g.astype(np.float64) * 0.5 > 0.05)
    f = (g * np.float32(0.5) > np.float32(0.05))
    assert (d != f).sum() >= 1
    d = (g.astype(np.float64) + 0.1 > 0.2)
    f = (g + np.float32(0.1) > np.float32(0.2))
    assert (d != f).sum() >= 1
