"""F-089: empty input / LIMIT 0 on the GPU device. cuMemAlloc(0) used to raise; every GPU op over no rows
must return the CPU result (same schema, same rows) without touching the device. Skipped without a GPU."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import LazyFrame
from mxframe.aot_kernels import GPU_AOT_AVAILABLE, get_aot_gpu

pytestmark = pytest.mark.skipif(not GPU_AOT_AVAILABLE or get_aot_gpu() is None, reason="no GPU")


def _t(n=200):
    r = np.random.default_rng(89)
    return pa.table({"k": pa.array(r.integers(0, 5, n).astype(np.int32)), "v": pa.array(r.random(n)),
                     "d": pa.array(r.integers(0, 100, n).astype(np.int32))})


def _u(t):
    return pa.table({"j": t["k"], "w": t["v"], "e": t["d"]})


def _empty(t):
    return t.slice(0, 0)


Q = {
    "limit 0 sort int": lambda t: mx.sql("SELECT k, v FROM t ORDER BY d LIMIT 0", t=t),
    "limit 0 sort float": lambda t: mx.sql("SELECT k, v FROM t ORDER BY v DESC LIMIT 0", t=t),
    "limit 0 api": lambda t: LazyFrame(t).sort("v").limit(0),
    "empty sort": lambda t: LazyFrame(_empty(t)).sort("v"),
    "empty sort limit": lambda t: mx.sql("SELECT k, v FROM t ORDER BY v LIMIT 5", t=_empty(t)),
    "filter none then sort": lambda t: mx.sql("SELECT k, v FROM t WHERE d > 1000 ORDER BY v", t=t),
    "filter none then topn": lambda t: mx.sql("SELECT k, v FROM t WHERE d > 1000 ORDER BY v LIMIT 3", t=t),
    "filter none then global agg": lambda t: mx.sql("SELECT sum(v) AS s, count(*) AS n FROM t WHERE d > 1000", t=t),
    "filter none then group by": lambda t: mx.sql("SELECT k, sum(v) AS s FROM t WHERE d > 1000 GROUP BY k", t=t),
    "empty group by": lambda t: mx.sql("SELECT k, sum(v) AS s, count(*) AS n FROM t GROUP BY k", t=_empty(t)),
    "group by then limit 0": lambda t: mx.sql("SELECT k, sum(v) AS s FROM t GROUP BY k ORDER BY s LIMIT 0", t=t),
    "join empty right": lambda t: mx.sql("SELECT a.k, b.w FROM t a JOIN u b ON a.k = b.j", t=t, u=_u(_empty(t))),
    "join empty left": lambda t: mx.sql("SELECT a.k, b.w FROM t a JOIN u b ON a.k = b.j", t=_empty(t), u=_u(t)),
    "join filtered-empty then sort": lambda t: mx.sql(
        "SELECT a.k, b.w FROM t a JOIN u b ON a.k = b.j WHERE b.e > 1000 ORDER BY b.w", t=t, u=_u(t)),
    "left join empty right": lambda t: mx.sql(
        "SELECT a.k, b.w FROM t a LEFT JOIN u b ON a.k = b.j ORDER BY a.d LIMIT 3", t=t, u=_u(_empty(t))),
}


@pytest.mark.parametrize("device", ["gpu", "auto"])
@pytest.mark.parametrize("name", list(Q))
def test_empty_on_gpu_matches_cpu(name, device):
    t = _t()
    cpu = Q[name](t).compute(device="cpu")
    out = Q[name](t).compute(device=device)
    assert out.schema == cpu.schema, (out.schema, cpu.schema)
    assert out.to_pylist() == cpu.to_pylist()


def test_zero_byte_gpu_allocation_is_a_null_handle():
    cu = get_aot_gpu()._cu
    assert cu.malloc(0) == 0
    cu.free(0)      # freeing the null handle is harmless
    assert cu.malloc(64) != 0
