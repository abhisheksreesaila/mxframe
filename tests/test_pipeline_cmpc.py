"""F-008: column-vs-constant compares run in the column's own type (CMPC) and still
equal the float64 comparison, including right at the rounding boundaries."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe.lazy_expr import col, lit
from mxframe import pipeline_kernels as pk
from mxframe.pipeline_program import compile_pipeline


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
        pytest.skip("no GPU or libmxpipeline_gpu.so not built")
    return request.param


def boundary_table():
    lits = [0.05, 0.06, 0.07, 24.0, 5.0, 5.5, -0.5, 0.0, 1e30, -1e30, 3e9, 2.0 ** 40, 2.0 ** 53]
    f32 = [np.float32(0), np.float32(-0.0), np.float32(np.inf), np.float32(-np.inf), np.float32(3.4028235e38)]
    for c in lits:
        if abs(c) < 3.4e38:
            f = np.float32(c)
            f32 += [np.nextafter(f, np.float32(-np.inf)), f, np.nextafter(f, np.float32(np.inf))]
    ints = [-3, -2, -1, 0, 1, 2, 3, 5, 6, 24, 25, 2 ** 31 - 1, -2 ** 31, 2 ** 31 - 2]
    i64 = ints + [2 ** 40, 2 ** 40 + 1, 2 ** 53, 2 ** 53 + 2, -2 ** 53, 2 ** 62]
    f64 = [0.05, 0.06, 0.07, 0.05 + 1e-17, np.nextafter(0.05, 1), np.nextafter(0.05, 0), 5.0, 5.5, -0.5, 1e30, 3e9]
    n = 64
    table = pa.table({
        "f32": pa.array(np.resize(np.array(f32, dtype=np.float32), n)),
        "i32": pa.array(np.resize(np.array(ints, dtype=np.int32), n)),
        "i64": pa.array(np.resize(np.array(i64, dtype=np.int64), n)),
        "f64": pa.array(np.resize(np.array(f64, dtype=np.float64), n)),
        "flag": pa.array(np.resize(np.array([True, False, False], dtype=bool), n)),
    })
    return table, lits + [float("inf"), float("-inf"), 1, 0, 2 ** 31]


NP_OPS = {"ge": np.greater_equal, "gt": np.greater, "le": np.less_equal, "lt": np.less,
          "eq": np.equal, "ne": np.not_equal}
DUNDER = {"ge": "__ge__", "gt": "__gt__", "le": "__le__", "lt": "__lt__", "eq": "__eq__", "ne": "__ne__"}
MIRROR = {"ge": "__le__", "gt": "__lt__", "le": "__ge__", "lt": "__gt__", "eq": "__eq__", "ne": "__ne__"}


@pytest.mark.parametrize("op", list(NP_OPS))
@pytest.mark.parametrize("column", ["f32", "i32", "i64", "f64", "flag"])
def test_column_vs_constant_matches_float64_reference(device, column, op):
    t, lits = boundary_table()
    x = t.column(column).to_numpy().astype(np.float64)
    run = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    for c in lits:
        want = int(np.count_nonzero(NP_OPS[op](x, float(c))))
        for pred in (getattr(col(column), DUNDER[op])(lit(c)),      # column <op> literal
                     getattr(lit(c), MIRROR[op])(col(column))):     # literal <mirrored op> column
            prog = compile_pipeline([pred], [col(column).count().alias("n")], t)
            assert prog is not None
            parts = run(prog, prog.bind(t), None, 1, 0, t.num_rows)
            assert int(parts.cnt[0]) == want, (column, op, c)


def test_cmpc_replaces_load_and_compare():
    t, _ = boundary_table()
    prog = compile_pipeline([col("f32") > lit(5.0)], [col("f64").sum()], t)
    assert prog.op_names().count("CMPC") == 1 and prog.op_names().count("LOAD_COL") == 1
    both = compile_pipeline([col("f32") > col("f64")], [col("f64").sum()], t)   # column vs column: generic ops
    assert "CMPC" not in both.op_names() and "GT" in both.op_names()


def test_dead_columns_are_not_bound():
    t, _ = boundary_table()
    # the constant-false conjunct settles the predicate: no predicate column is read at all
    prog = compile_pipeline([col("i32") > lit(1), lit(0) > lit(1)], [col("f64").sum()], t)
    assert prog.columns == ["f64"]
