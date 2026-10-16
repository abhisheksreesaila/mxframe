"""F-082: denormal (subnormal) floats are values, not zero, on every path, like DuckDB.

Mojo's `parallelize` / `sync_parallelize` ran every worker with the CPU's flush-to-zero and denormals-are-zero bits
set (MXCSR 0x9FE0), so a kernel that fanned out over threads read 5e-324 as 0: `x > 0.0` counted 0 of 100,000 rows
and `sum(x)` was 0 (fused pipeline, native executor above one thread's worth of rows, hash aggregate, ...).
kernels_aot/_fpenv.mojo now resets MXCSR at the start of every task.

Every expected answer comes from native DuckDB. Each case runs float64 and float32, on cpu (native executor on and
off) and gpu, with enough rows (200,000) that the kernels start worker threads, and asserts the path that answered.
Before the fix the cpu cases that fan out over threads failed on main (counts of 0, sums of 0).
"""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import gpu_stream
from mxframe.custom_ops import CustomOpsCompiler
import mxframe.native_exec as nx

GPU = gpu_stream.cuda_available()
N = 200_000

# one cycle of 8 values per dtype: the smallest and the largest denormal, both signs, zero, and normal values
CYCLE = {
    "float64": [5e-324, 2.2e-308 / 2, -5e-324, 0.0, 1.5, -2.2e-308 / 3, 1.0e-200, 3.0],
    "float32": [1e-45, 1.1e-38, -1e-45, 0.0, 1.5, -5e-40, 1.0e-30, 3.0],
}
# only the denormals and the zeros (the cycle's other values are normal): a sum of them is exact and is 0 if they are flushed
TINY = {"float64": "x < 1.0e-250 and x > -1.0e-250", "float32": "x < 1.0e-37 and x > -1.0e-37"}


def make_table(dtype, nulls):
    dt = np.float64 if dtype == "float64" else np.float32
    x = np.resize(np.array(CYCLE[dtype], dt), N)
    mask = (np.arange(N) % 11 == 0) if nulls else None
    return pa.table({
        "i": pa.array(np.arange(N), pa.int32()),
        "g": pa.array(np.arange(N) % 3, pa.int32()),
        "k": pa.array(np.arange(N) % 5000, pa.int32()),
        "x": pa.array(x, mask=mask),
    })


@pytest.fixture(scope="module")
def world():
    out = {}
    for dtype in ("float64", "float32"):
        for nulls in (False, True):
            t = make_table(dtype, nulls)
            con = duckdb.connect()
            con.register("_t", t)
            con.execute("create table t as select * from _t")
            out[(dtype, nulls)] = (t, con)
    return out


@pytest.fixture(params=[(d, nl, dev, nat) for d in ("float64", "float32") for nl in (False, True)
                        for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-{'nulls' if p[1] else 'nonull'}-{p[2]}-native{p[3]}")
def cfg(request, world, monkeypatch):
    dtype, nulls, device, native = request.param
    if device == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    if device == "gpu" and native == "0":
        pytest.skip("the native executor is not on the gpu path")
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    table, con = world[(dtype, nulls)]
    paths = []
    run = CustomOpsCompiler._compile_and_run

    def recording(self, *a, **k):
        out = run(self, *a, **k)
        paths.append((self.last_compile_provenance or {}).get("path"))
        return out

    monkeypatch.setattr(CustomOpsCompiler, "_compile_and_run", recording)
    return dict(dtype=dtype, nulls=nulls, device=device, native=native == "1", table=table, con=con, paths=paths)


def _rows(t):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    return [tuple(c[r] for c in cols) for r in range(t.num_rows)]


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        return a == pytest.approx(b, rel=1e-6, abs=0.0)        # abs=0: a denormal must not pass as zero
    return a == b


def check(cfg, sql, path):
    """The answer equals DuckDB's exactly (floats to 1e-6 relative, no absolute slack) and `path` is among the
    paths that ran."""
    exp = [tuple(r) for r in cfg["con"].execute(sql).fetchall()]
    nx._filter_last.update(path=None, decline=None)
    cfg["paths"].clear()
    got = _rows(mx.sql(sql, t=cfg["table"]).compute(device=cfg["device"]))
    assert len(got) == len(exp), (got[:5], exp[:5])
    for g, e in zip(got, exp):
        assert len(g) == len(e) and all(_eq(a, b) for a, b in zip(g, e)), (g, e)
    if path:
        assert path in cfg["paths"], (cfg["paths"], path)
    return got


def _agg_path(cfg):
    if cfg["nulls"]:
        return "arrow_null_semantics"
    if cfg["device"] == "gpu":
        return "fused_pipeline_gpu"
    return "native" if cfg["native"] else "fused_pipeline_cpu"


PREDICATES = ["x > 0.0", "x < 0.0", "x = 0.0", "x <> 0.0", "x >= 0.0", "x <= 0.0", "x > 1.0e-310", "x < -1.0e-310",
              "x between -2.0e-30 and 2.0e-30", "x in (0.0, 1.5)", "not (x > 0.0)"]


@pytest.mark.parametrize("pred", PREDICATES)
def test_filter_count_and_sum(cfg, pred):
    check(cfg, f"select count(*) c, sum(i) s from t where {pred}", _agg_path(cfg))


@pytest.mark.parametrize("pred", PREDICATES)
def test_grouped_filter_count(cfg, pred):
    check(cfg, f"select g, count(*) c from t where {pred} group by g order by g", _agg_path(cfg))


@pytest.mark.parametrize("pred", ["x > 0.0", "x < 0.0", "x = 0.0"])
def test_many_groups(cfg, pred):
    check(cfg, f"select k, count(*) c, sum(i) s from t where {pred} group by k order by k limit 20", None)


def test_sum_of_denormals(cfg):
    check(cfg, f"select sum(x) s, count(*) c from t where {TINY[cfg["dtype"]]}", _agg_path(cfg))


def test_sum_of_everything(cfg):
    check(cfg, "select sum(x) s from t", _agg_path(cfg))


def test_grouped_sum_of_denormals(cfg):
    check(cfg, f"select g, sum(x) s from t where {TINY[cfg["dtype"]]} group by g order by g", _agg_path(cfg))


def test_many_groups_sum_of_denormals(cfg):
    check(cfg, f"select k, sum(x) s from t where {TINY[cfg["dtype"]]} group by k order by k limit 20", None)


def test_min_and_max(cfg):
    check(cfg, "select min(x) lo, max(x) hi from t", None)
    check(cfg, f"select g, min(x) lo, max(x) hi from t where {TINY[cfg["dtype"]]} group by g order by g", None)


def test_row_filter_keeps_denormal_rows(cfg):
    """A plain filter (the native leaf filter, or Arrow's compute) returns the rows holding a denormal."""
    check(cfg, "select i from t where x > 0.0 and x < 1.0e-30 order by i limit 50", "pyarrow_shortcut")
    check(cfg, "select count(*) c from t where x < 0.0 and x > -1.0e-30", None)

