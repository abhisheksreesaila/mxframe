"""F-008: column-vs-constant compares run in the column's own type (CMPC): a float32 column
takes the literal as float32 (so TPC-H Q6's `l_discount <= 0.07` keeps the 0.07 rows, as DuckDB
does), integer columns compare exactly. Boundary values on both sides of every literal."""
import numpy as np
import pyarrow as pa
import pytest

from omxcore.lazy_expr import col, lit
from omxcore import pipeline_kernels as pk
from omxcore.pipeline_program import compile_pipeline


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


def reference(column, x_native, op, c):
    """The comparison as DuckDB / Polars / NumPy 2 do it: a float32 column takes the literal as float32,
    everything else compares exactly (float64 for float64, exact integers for integer columns)."""
    if column == "f32":
        with np.errstate(over="ignore"):
            return NP_OPS[op](x_native.astype(np.float32), np.float32(c))
    return NP_OPS[op](x_native.astype(np.float64), float(c))


NP_OPS = {"ge": np.greater_equal, "gt": np.greater, "le": np.less_equal, "lt": np.less,
          "eq": np.equal, "ne": np.not_equal}
DUNDER = {"ge": "__ge__", "gt": "__gt__", "le": "__le__", "lt": "__lt__", "eq": "__eq__", "ne": "__ne__"}
MIRROR = {"ge": "__le__", "gt": "__lt__", "le": "__ge__", "lt": "__gt__", "eq": "__eq__", "ne": "__ne__"}


@pytest.mark.parametrize("op", list(NP_OPS))
@pytest.mark.parametrize("column", ["f32", "i32", "i64", "f64", "flag"])
def test_column_vs_constant_uses_the_columns_type(device, column, op):
    t, lits = boundary_table()
    x = t.column(column).to_numpy()
    run = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    for c in lits:
        want = int(np.count_nonzero(reference(column, x, op, c)))
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


def test_between_on_float32_column_keeps_boundary_rows(device):
    """between() follows the same column-type rule as the two compares it stands for."""
    t = pa.table({"d": pa.array(np.array([0.05, 0.06, 0.07, 0.07, 0.08, 0.04], dtype=np.float32)),
                  "v": pa.array(np.ones(6))})
    prog = compile_pipeline([col("d").between(0.05, 0.07)], [col("v").sum().alias("s")], t)
    assert "BETWEEN" not in prog.op_names()
    run = pk.run_pipeline_cpu if device == "cpu" else pk.run_pipeline_gpu
    assert run(prog, prog.bind(t), None, 1, 0, 6).vals[0, 0] == 4.0


def test_dead_columns_are_not_bound():
    t, _ = boundary_table()
    # the constant-false conjunct settles the predicate: no predicate column is read at all
    prog = compile_pipeline([col("i32") > lit(1), lit(0) > lit(1)], [col("f64").sum()], t)
    assert prog.columns == ["f64"]


def test_float32_column_keeps_rows_equal_to_a_decimal_literal():
    """0.07 stored as float32 is 0.07000000029..., above the float64 literal 0.07: comparing in
    float64 would drop every such row (TPC-H Q6 then loses a third of its revenue)."""
    t = pa.table({"d": pa.array(np.array([0.05, 0.06, 0.07, 0.08], dtype=np.float32)),
                  "v": pa.array(np.ones(4))})
    prog = compile_pipeline([col("d") >= lit(0.05), col("d") <= lit(0.07)], [col("v").sum().alias("s")], t)
    assert pk.cpu_available()
    parts = pk.run_pipeline_cpu(prog, prog.bind(t), None, 1, 0, 4)
    assert parts.vals[0, 0] == 3.0
