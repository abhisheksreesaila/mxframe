"""F-106: SQL numeric literals take their SQL meaning, and a float32 column meets a DOUBLE literal in double.

* integer literals stay integers (F-103), a number with a point and no exponent is a DECIMAL (F-094), and any
  exponent form (`1e3`, `1E-5`, `2E5`, `1.5e2`) is a DOUBLE: `x > 1e-310` used to raise ValueError (the front
  end looked for a "." to tell int from float).
* DuckDB casts a DECIMAL or INTEGER literal to a float32 column's type (`x = 0.1` matches float32(0.1),
  `x = 16777217` matches 16777216) but widens the column to DOUBLE against a DOUBLE literal, so
  `x between -1e-30 and 1e-30` drops float32(1e-30) (it is 1.0000000031710769e-30). The literal used to be cast
  to float32 on every path.

Expected answers come from a NATIVE DuckDB table. Every case runs on cpu and gpu, native on and off.
"""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import gpu_stream

GPU = gpu_stream.cuda_available()
N = 40


def _table():
    k = np.arange(N)
    special = [1e-30, -1e-30, 0.0, 0.1, 16777216.0, 0.07, 1.5, 1e-39, 2.0e5, 1000.0, 100000.0, -0.5]
    g = np.array(special + [float(x) for x in k[len(special):]], dtype=np.float32)
    return pa.table({
        "i": pa.array(k, pa.int64()),
        "s": pa.array(k - 20, pa.int32()),
        "f": pa.array((k - 20) * 250.5, pa.float64()),
        "g": pa.array(g, pa.float32()),
        "h": pa.array(g.astype(np.float64) * 1.0, pa.float64()),
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


def _check(dev, q):
    want = CON.execute(q).fetchall()
    got = _rows(mx.sql(q, t=T).compute(device=dev))
    assert len(got) == len(want), (q, got, want)
    for g, w in zip(got, want):
        assert len(g) == len(w) and all(
            a == pytest.approx(b, rel=1e-12, abs=0) if isinstance(b, float) and a is not None else a == b
            for a, b in zip(g, w)), (q, g, w)


# ── literal forms ─────────────────────────────────────────────────────────────────────────────────

LITERALS = ["1e3", "1E-5", "2E5", "1.5e2", ".5", "5.", "1_000", "1e+2", "0.5e1"]


@pytest.mark.parametrize("v", LITERALS)
def test_literal_in_where(cfg, v):
    _check(cfg, f"select i from t where f > {v} order by i")
    _check(cfg, f"select i from t where {v} >= f order by i")
    _check(cfg, f"select i from t where f between -{v} and {v} order by i")


@pytest.mark.parametrize("v", LITERALS)
def test_literal_in_select_and_arithmetic(cfg, v):
    _check(cfg, f"select {v} as c from t order by i limit 3")
    _check(cfg, f"select i, f + {v} as a, f * {v} as b, {v} - f as c, f / {v} as d from t order by i")
    _check(cfg, f"select i, s + {v} as a, s * {v} as b from t order by i")
    _check(cfg, f"select sum(f * {v}) as a, max(f + {v}) as b from t")


@pytest.mark.parametrize("v", ["1e-310", "1e-300", "1E300", "2E5"])
def test_scientific_without_a_dot_in_where(cfg, v):
    _check(cfg, f"select i from t where h > {v} order by i")
    _check(cfg, f"select i from t where h < -{v} order by i")


def test_literal_in_in_list(cfg):
    _check(cfg, "select i from t where f in (1e3, 2E5, -250.5, 0.5) order by i")
    _check(cfg, "select i from t where s in (1, 2e0, 3) order by i")


@pytest.mark.parametrize("v", ["9223372036854775807", "9223372036854775808", "99999999999999999999",
                               "-99999999999999999999"])
def test_huge_integer_literals(cfg, v):
    _check(cfg, f"select i from t where i < {v} order by i limit 5")
    _check(cfg, f"select i from t where f > {v} order by i")


# ── int64 columns at the limits (review round 1) ──────────────────────────────────────────────────

BIG = pa.table({"i": pa.array([1, 5, 2**53 + 1, 9223372036854775806, 9223372036854775807, -9223372036854775808, None],
                              pa.int64()),
                "j": pa.array([1, 2, 3, 4, 5, 6, None], pa.int32()),
                "f": pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None], pa.float64())})
BIGCON = duckdb.connect()
BIGCON.register("_b", BIG)
BIGCON.execute("create table t as select * from _b")

BIG_WHERE = [f"{c} {op} {v}" for v in ("-9223372036854775808", "9223372036854775807", "9223372036854775808",
                                       "-9223372036854775809", "99999999999999999999", "-99999999999999999999")
             for c in ("i", "j") for op in ("=", "<>", "<", "<=", ">", ">=")] + [
    "i in (9223372036854775808, 1)", "i in (-9223372036854775808, 5)", "i not in (9223372036854775808, 1)",
    "j in (99999999999999999999, 2)", "i between 1 and 9223372036854775808",
    "i between -9223372036854775808 and 5", "9223372036854775808 > i", "-9223372036854775808 = i",
    "i < 1e3", "i > 1e3", "i = 9007199254740992e0", "i > 9007199254740992e0", "i >= 9007199254740992e0",
    "i < 9.3e18", "i > -9.3e18", "i = -9223372036854775808.0",
    "i > -9223372036854775808.0", "i <= 9223372036854775807.0", "i + 0 > -9223372036854775808",
    "i = 9007199254740993",
]


@pytest.mark.parametrize("w", BIG_WHERE)
def test_int64_limits(cfg, w):
    q = f"select i from t where {w} order by i"
    want = BIGCON.execute(q).fetchall()
    got = _rows(mx.sql(q, t=BIG).compute(device=cfg))
    assert got == want, (q, got, want)


# ── DECIMAL literals against int64 columns are exact; DOUBLE ones compare in double ────────────────

DEC = pa.table({"i": pa.array([1, 2, 3, 2**53, 2**53 + 1, 9223372036854775806, 9223372036854775807, None], pa.int64()),
                "j": pa.array([1, 2, 3, 4, 5, 6, 7, None], pa.int32())})
DECCON = duckdb.connect()
DECCON.register("_d", DEC)
DECCON.execute("create table t as select * from _d")

DEC_WHERE = [f"{c} {op} {v}" for v in ("9007199254740992.0", "9007199254740993.0", "9223372036854775806.0",
                                       "9223372036854775807.0", "9223372036854775807.5", "2.5", "-2.5", "2.0", ".5", "3.")
             for c in ("i", "j") for op in ("=", "<>", "<", "<=", ">", ">=")] + [
    "2.5 > i", "2.5 = i", "9007199254740992.0 < i", "i between 2.5 and 9007199254740992.0",
    "i in (9007199254740992.0, 2.5, 3.0)", "i not in (9007199254740992.0, 2.5)",
    "i + 0 = 9007199254740992.0", "i + 0 > 9007199254740992.0", "i * 1 < 9223372036854775807.0",
    "i + 0 = 9223372036854775806.0", "i - 0 < 2.5", "i = 9007199254740992e0", "i > 9007199254740992e0",
    "i < 9223372036854775807e0", "i >= 9223372036854775807e0", "i = 2e0", "i < 2.5e0",
]


@pytest.mark.parametrize("w", DEC_WHERE)
def test_decimal_literals_exact(cfg, w):
    q = f"select i from t where {w} order by i"
    want = DECCON.execute(q).fetchall()
    got = _rows(mx.sql(q, t=DEC).compute(device=cfg))
    assert got == want, (q, got, want)


# ── float32 column against literals ───────────────────────────────────────────────────────────────

F32_WHERE = [
    "g between -1e-30 and 1e-30",                 # DOUBLE bounds: float32(1e-30) is outside
    "g > 1e-30", "g >= 1e-30", "g < 1e-30", "g <= -1e-30", "g = 1e-30", "g <> 1e-30",
    "g = 0.1", "g > 0.1", "g >= 0.1", "g < 0.1", "g <> 0.1",           # DECIMAL: compared as float32
    "g = 1e-1", "g > 1e-1", "g >= 1e-1", "g < 1e-1", "g <= 1e-1",     # DOUBLE: the column widens
    "g = 0.07", "g <= 0.07", "g <= 7e-2",
    "g = 16777217", "g < 16777217", "g > 16777217", "g >= 16777217",   # INTEGER: compared as float32
    "g = 16777217.0", "g = 16777217e0", "g < 16777217e0", "g >= 16777217e0",
    "1e-30 > g", "0.1 = g", "1e-1 = g",
    "g between 1e-1 and 2e5", "g between 0.1 and 2E5", "g not between 1e-30 and 1e-1",
    "g > 1e-310", "g > 1e-39", "g = 1e-39",
    "g in (0.1, 1.5)", "g in (1e-1, 1.5)", "g in (1e-30, 0.0)",
    "(g > 1e-1 and g < 2) or g = 1e-30",
    "g > 1e-30 * 1", "-g < -1e-30", "(g) > (1e-30)", "-g > -1e-30", "g * 1 > 1e-30", "g + 0 > 1e-30", "1e-30 * 1 < g",
    "g * 0.5 > 1e-30", "g > h", "g > h * 1.0 - 1e-30", "(-g) between -1e-30 and 1e-30",
]


@pytest.mark.parametrize("w", F32_WHERE)
def test_float32_filter(cfg, w):
    _check(cfg, f"select i from t where {w} order by i")


@pytest.mark.parametrize("w", ["g between -1e-30 and 1e-30", "g = 0.1", "g = 1e-1", "g < 16777217e0", "g > 1e-30"])
def test_float32_filter_aggregate(cfg, w):
    _check(cfg, f"select count(*) as c, sum(h) as s from t where {w}")


@pytest.mark.parametrize("e", ["g + 1e-30", "g + 0.1", "g * 1e-1", "g - 1e0", "g + 16777217", "g * 2e0", "g * 0.07"])
def test_float32_arithmetic(cfg, e):
    _check(cfg, f"select i, {e} as x from t order by i")


def test_float32_case_when(cfg):
    _check(cfg, "select i, case when g > 1e-30 then 1 else 0 end as c from t order by i")
    _check(cfg, "select i, case when g = 0.1 then 1 else 0 end as c from t order by i")


# ── parquet pushdown ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("w", ["g between -1e-30 and 1e-30", "g > 1e-30", "g = 1e-1", "g = 0.1", "g <= 1e-30",
                               "g < 16777217e0", "g > 1e-310"])
def test_parquet_pushdown(cfg, tmp_path, w):
    path = str(tmp_path / "t.parquet")
    pq.write_table(T, path, row_group_size=7)
    q = f"select i from t where {w} order by i"
    want = CON.execute(q).fetchall()
    got = _rows(mx.sql(q, t=mx.scan_parquet(path)).compute(device=cfg))
    assert got == want
