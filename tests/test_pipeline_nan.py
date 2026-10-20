"""F-008 review: min / max over NaN follow DuckDB (NaN sorts above every number: min skips it unless
the group is all NaN, max is NaN as soon as one value is), on both devices."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

from omxcore import pipeline_kernels as pk
from omxcore.lazy_expr import col
from omxcore.pipeline_program import compile_pipeline, finalize

NAN = float("nan")
GIDS = np.array([0, 0, 0, 1, 1, 2, 3, 3, 4, 4], dtype=np.int32)
VALUES = [1.0, NAN, 3.0, NAN, NAN, 2.0, float("inf"), 5.0, float("-inf"), NAN]


def _gpu_ok():
    if not pk.gpu_available():
        return False
    try:
        pk._get_cuda()
        return True
    except Exception:
        return False


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    if request.param == "cpu" and not pk.cpu_available():
        pytest.skip("libmxpipeline_cpu.so not built")
    if request.param == "gpu" and not _gpu_ok():
        pytest.skip("no GPU")
    return request.param


def _same(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert np.array_equal(np.isnan(a), np.isnan(b)), (a, b)
    assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]), (a, b)


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
def test_grouped_min_max_match_duckdb(device, dtype):
    t = pa.table({"g": pa.array(GIDS), "x": pa.array(VALUES, type=dtype)})
    c = duckdb.connect()
    c.register("t", t)
    want = c.sql("select min(x), max(x) from t group by g order by g").fetchall()
    prog = compile_pipeline([], [col("x").min().alias("lo"), col("x").max().alias("hi")], t)
    run = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    parts = run(prog, prog.bind_chunks(t), GIDS, 5, 0, t.num_rows)
    _, cols = finalize(prog, parts, grouped=True)
    _same(cols["lo"], [w[0] for w in want])
    _same(cols["hi"], [w[1] for w in want])


@pytest.mark.parametrize("keep", [[0, 1, 2], [3, 4, 5, 6], [1, 3]])
def test_global_min_max_match_duckdb(device, keep):
    t = pa.table({"x": pa.array([VALUES[i] for i in keep], type=pa.float32())})
    c = duckdb.connect()
    c.register("t", t)
    want = c.sql("select min(x), max(x) from t").fetchone()
    prog = compile_pipeline([], [col("x").min().alias("lo"), col("x").max().alias("hi")], t)
    run = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    parts = run(prog, prog.bind_chunks(t), None, 1, 0, t.num_rows)
    _, cols = finalize(prog, parts, grouped=False)
    _same(cols["lo"], [want[0]])
    _same(cols["hi"], [want[1]])
