"""F-079: float join keys compare like SQL: -0.0 equals 0.0, every NaN (any sign or payload) equals every
other NaN, +-inf match only themselves, a NULL key never matches, and float32 meets float64 after the
float32 side widens (0.1f does not equal 0.1). Checked against DuckDB on native DuckDB tables for inner,
left, semi (EXISTS / IN) and anti (NOT EXISTS / NOT IN) joins, on every join path, with the key columns
reported as each side holds them (a join does not normalise a zero or a NaN).

Before the fix `dictionary_encode` compared the float bits, so -0.0 and 0.0 and NaNs of different payloads
never met: 5 of 10 inner pairs, 5 of 6 semi rows. The paths are forced and asserted (a test that silently
ran another path proves nothing):
  native   join_engine.encode_keys -> Mojo/NumPy kernels (the `join_mojo_shortcut` provenance)
  arrow    the legacy `_execute_hash_join` (`_encode_join_key_pair`), forced by declining the native step
  gpu      device="gpu" (the GPU hash join gets the dense codes, never the float bits)
  fused    a join feeding a group-by, native and fused-pipeline aggregates: float keys decline the native
           lookup, so the join is materialized through the same encoder"""
import math
import struct

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.custom_ops as co
import mxframe.join_engine as je
import mxframe.native_exec as nx
from mxframe import LazyFrame, col, gpu_budget, sql
from mxframe.np_arrow import array as A

BITS = {np.float64: ("<Q", "<d"), np.float32: ("<I", "<f")}
NAN_PAYLOADS = {np.float64: (0x7FF8000000000000, 0xFFF8000000001234, 0x7FF0000000000001),
                np.float32: (0x7FC00000, 0xFFC01234, 0x7F800001)}      # quiet, negative payload, signalling


def _gpu_usable() -> bool:
    try:
        return co._max_accelerator() is not None
    except Exception:
        return False


def _nans(dt):
    ib, fb = BITS[dt]
    return [struct.unpack(fb, struct.pack(ib, p))[0] for p in NAN_PAYLOADS[dt]]


def _col(vals, dt):
    mask = np.array([v is None for v in vals])
    data = np.array([0.0 if v is None else v for v in vals], dtype=dt)
    return A(np.ma.array(data, mask=mask), type=pa.from_numpy_dtype(dt))


def _tables(ldt, rdt, filler=0):
    """l(k, id) and r(rk, rv, rid): -0.0 / 0.0 both ways, three NaN payloads over both sides, NULLs on
    both sides, +-inf, a key with no partner. ``filler`` distinct keys push the join past the tiny sizes."""
    ln1, ln2, _ = _nans(ldt)
    _, rn2, rn3 = _nans(rdt)
    lv = [-0.0, 0.0, ln1, ln2, None, math.inf, -math.inf, 1.5, 0.1] + [float(i) + 10 for i in range(filler)]
    rv = [0.0, -0.0, rn3, None, math.inf, -math.inf, rn2, 0.1] + [float(i) + 10 for i in range(0, filler, 3)]
    r = _col(rv, rdt)
    return (pa.table({"k": _col(lv, ldt), "id": A(np.arange(len(lv), dtype=np.int64))}),
            pa.table({"rk": r, "rv": r, "rid": A(np.arange(len(rv), dtype=np.int64))}))


def _c(x):
    """Comparable form of a float: NaN equals NaN (its payload is not reported), the sign of a zero counts."""
    if x is None:
        return "null"
    return "nan" if math.isnan(x) else repr(float(x))


def _norm(rows, float_cols):
    return sorted((tuple(_c(v) if i in float_cols else v for i, v in enumerate(r)) for r in rows), key=str)


DUCK = {
    "inner": ("select l.id, r.rid, l.k, r.rv from l join r on l.k = r.rk", (2, 3)),
    "left": ("select l.id, r.rid, l.k, r.rv from l left join r on l.k = r.rk", (2, 3)),
    "semi": ("select id, k from l where exists (select 1 from r where r.rk = l.k)", (1,)),
    "anti": ("select id, k from l where not exists (select 1 from r where r.rk = l.k)", (1,)),
}
COLS = {"inner": ["id", "rid", "k", "rv"], "left": ["id", "rid", "k", "rv"], "semi": ["id", "k"], "anti": ["id", "k"]}


def _duck(query, **tables):
    """The query over native DuckDB tables (not registered Arrow ones)."""
    con = duckdb.connect()
    for name, t in tables.items():
        con.register(name + "_arrow", t)
        con.execute(f"create table {name} as select * from {name}_arrow")
    return con.execute(query).fetchall()


class Spy:
    """Which join steps ran: the dense key encoder, the legacy Arrow join, the GPU hash join."""

    def __init__(self, monkeypatch):
        self.calls = []
        for mod, name in ((je, "encode_keys"), (co.CustomOpsCompiler, "_execute_hash_join"),
                          (co.CustomOpsCompiler, "_hash_join_left_mojo_gpu"), (gpu_budget, "run_piece")):
            self._wrap(monkeypatch, mod, name)

    def _wrap(self, monkeypatch, mod, name):
        orig = getattr(mod, name)

        def spy(*a, **k):
            self.calls.append(name if name != "run_piece" else f"run_piece:{a[0]}")
            return orig(*a, **k)

        monkeypatch.setattr(mod, name, spy)


PATHS = ["native", "arrow", pytest.param("gpu", marks=pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU"))]


def _run(path, monkeypatch, build):
    """``build()`` -> LazyFrame; computed on ``path`` with the path asserted afterwards."""
    spy = Spy(monkeypatch)
    if path == "arrow":
        for step in ("_execute_filtering", "_execute_left", "_execute_inner"):
            monkeypatch.setattr(je, step, lambda *a, **k: None)
    q = build()
    out = q.compute(device="gpu" if path == "gpu" else "cpu")
    if path == "arrow":
        assert "_execute_hash_join" in spy.calls and "encode_keys" not in spy.calls
    else:
        assert "encode_keys" in spy.calls and "_execute_hash_join" not in spy.calls
        assert q.last_compile_provenance["path"] == "join_mojo_shortcut"
    return out, spy


def _same(out, how, want):
    d = out.to_pydict()
    cols = COLS[how]
    got = list(zip(*[d[c] for c in cols]))
    assert _norm(got, DUCK[how][1]) == _norm(want, DUCK[how][1])


DTYPES = [(np.float64, np.float64), (np.float32, np.float32), (np.float32, np.float64), (np.float64, np.float32)]
IDS = ["f64-f64", "f32-f32", "f32-f64", "f64-f32"]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("how", list(DUCK))
@pytest.mark.parametrize("dts", DTYPES, ids=IDS)
def test_join_matches_duckdb(dts, how, path, monkeypatch):
    l, r = _tables(*dts)
    want = _duck(DUCK[how][0], l=l, r=r)
    out, _ = _run(path, monkeypatch, lambda: LazyFrame(l).join(LazyFrame(r), left_on="k", right_on="rk", how=how))
    _same(out, how, want)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("how", list(DUCK))
def test_join_matches_duckdb_with_many_rows(how, path, monkeypatch):
    l, r = _tables(np.float64, np.float64, filler=30_000)
    want = _duck(DUCK[how][0], l=l, r=r)
    out, _ = _run(path, monkeypatch, lambda: LazyFrame(l).join(LazyFrame(r), left_on="k", right_on="rk", how=how))
    _same(out, how, want)


def test_the_pinned_answers():
    """The rules themselves, from DuckDB: both zeros and every NaN pair, no NULL, inf only with itself."""
    l, r = _tables(np.float64, np.float64)
    pairs = sorted(_duck("select l.id, r.rid from l join r on l.k = r.rk", l=l, r=r))
    assert pairs == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 6), (3, 2), (3, 6), (5, 4), (6, 5), (8, 7)]
    assert sorted(x for (x,) in _duck("select id from l where k not in (select rk from r)", l=l, r=r)) == []


@pytest.mark.parametrize("dt", [np.float64, np.float32])
def test_join_on_a_float_key_and_an_integer_key(dt):
    """A composite key: the float part meets like SQL, the integer part exactly."""
    n1, n2, _ = _nans(dt)
    lt = pa.table({"k": _col([-0.0, n1, 1.0, 0.0, n2], dt), "j": A(np.array([1, 1, 1, 2, 2], dtype=np.int64)),
                   "id": A(np.arange(5, dtype=np.int64))})
    rt = pa.table({"rk": _col([0.0, n2, 1.0, 0.0], dt), "rj": A(np.array([1, 1, 2, 2], dtype=np.int64)),
                   "rid": A(np.arange(4, dtype=np.int64))})
    want = sorted(_duck("select id, rid from l join r on l.k = r.rk and l.j = r.rj", l=lt, r=rt))
    assert want == [(0, 0), (1, 1), (3, 3)]
    got = LazyFrame(lt).join(LazyFrame(rt), left_on=["k", "j"], right_on=["rk", "rj"]).compute(device="cpu")
    assert sorted(zip(got.column("id").to_pylist(), got.column("rid").to_pylist())) == want


@pytest.mark.parametrize("env", ["1", "0"], ids=["native-aggregate", "fused-pipeline-aggregate"])
@pytest.mark.parametrize("dt", [np.float64, np.float32])
def test_join_feeding_an_aggregate(dt, env, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", env)
    l, r = _tables(dt, dt)
    fact = pa.table({"k": l.column("k"), "v": A(np.arange(l.num_rows, dtype=np.float64) + 1)})
    dim = pa.table({"k": r.column("rk"), "d": A((np.arange(r.num_rows) % 3).astype(np.int64))})
    want = _duck("select d, sum(v) as s, count(*) as n from fact join dim using (k) group by d order by d",
                 fact=fact, dim=dim)
    spy = Spy(monkeypatch)
    q = (LazyFrame(fact).join(LazyFrame(dim), on="k").groupby("d")
         .agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("d"))
    out = q.compute(device="cpu")
    assert "encode_keys" in spy.calls                       # the float join was materialized through the shared encoder
    assert q.last_compile_provenance["path"] == ("native" if env == "1" else "fused_pipeline_cpu")
    assert list(zip(*[out.to_pydict()[c] for c in ("d", "s", "n")])) == want


@pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU")
@pytest.mark.parametrize("dt", [np.float64, np.float32])
def test_join_feeding_an_aggregate_on_the_gpu(dt, monkeypatch):
    l, r = _tables(dt, dt)
    fact = pa.table({"k": l.column("k"), "v": A(np.arange(l.num_rows, dtype=np.float64) + 1)})
    dim = pa.table({"k": r.column("rk"), "d": A((np.arange(r.num_rows) % 3).astype(np.int64))})
    want = _duck("select d, sum(v) as s, count(*) as n from fact join dim using (k) group by d order by d",
                 fact=fact, dim=dim)
    spy = Spy(monkeypatch)
    q = (LazyFrame(fact).join(LazyFrame(dim), on="k").groupby("d")
         .agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("d"))
    out = q.compute(device="gpu")
    assert "encode_keys" in spy.calls and "run_piece:join" in spy.calls
    assert list(zip(*[out.to_pydict()[c] for c in ("d", "s", "n")])) == want


SQL = [
    ("in", "select id from l where k in (select rk from r) order by id"),
    ("exists", "select id from l where exists (select 1 from r where r.rk = l.k) order by id"),
    ("not in", "select id from l where k not in (select rk from r) order by id"),
    ("not in, no NULL on the right", "select id from l where k not in (select rk from r where rk = rk) order by id"),
    ("not exists", "select id from l where not exists (select 1 from r where r.rk = l.k) order by id"),
    ("join", "select id, rid from l join r on l.k = r.rk order by id, rid"),
    ("left join", "select id, rid from l left join r on l.k = r.rk order by id, rid"),
]


@pytest.mark.parametrize("dt", [np.float64, np.float32])
@pytest.mark.parametrize("name,query", SQL, ids=[s[0] for s in SQL])
def test_sql_forms_match_duckdb(name, query, dt, monkeypatch):
    l, r = _tables(dt, dt)
    want = _duck(query, l=l, r=r)
    spy = Spy(monkeypatch)
    out = sql(query, l=l, r=r).compute(device="cpu")
    assert "encode_keys" in spy.calls
    assert list(zip(*out.to_pydict().values())) == want
