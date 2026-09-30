"""F-018: a disjunction over both sides of a join filters each side by what it implies."""
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit
from mxframe.lazy_frame import Filter, Join, Scan
from mxframe.optimizer import optimize_plan


def _i32(a):
    return pa.array(a, pa.int32())


def _tables():
    a = pa.table({"k": _i32([1, 2, 3, 4]), "x": _i32([1, 2, 3, 4])})
    b = pa.table({"k": _i32([1, 2, 3, 4]), "y": _i32([10, 20, 30, 40])})
    return a, b


def _rows(t):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    return sorted(zip(*cols))


def test_disjunction_over_both_sides_is_pushed_as_per_side_filters():
    a, b = _tables()
    pred = ((col("x") == lit(1)) & (col("y") == lit(20))) | ((col("x") == lit(3)) & (col("y") == lit(40)))
    lf = LazyFrame(a).join(LazyFrame(b), on="k").filter(pred)
    plan = optimize_plan(lf.plan).plan
    assert isinstance(plan, Filter) and isinstance(plan.input, Join)          # the exact predicate stays on top
    assert isinstance(plan.input.left, Filter) and isinstance(plan.input.right, Filter)
    assert lf.compute(device="cpu").num_rows == 0        # the pushed filters must not invent a row
    pred2 = ((col("x") == lit(1)) & (col("y") == lit(10))) | ((col("x") == lit(3)) & (col("y") == lit(30)))
    got = LazyFrame(a).join(LazyFrame(b), on="k").filter(pred2).compute(device="cpu")
    assert _rows(got) == [(1, 1, 10), (3, 3, 30)]


def test_disjunct_without_a_condition_on_a_side_pushes_nothing_to_it():
    a, b = _tables()
    pred = ((col("x") == lit(1)) & (col("y") == lit(10))) | (col("y") == lit(20))
    lf = LazyFrame(a).join(LazyFrame(b), on="k").filter(pred)
    plan = optimize_plan(lf.plan).plan
    assert isinstance(plan.input.left, Scan)           # the second disjunct says nothing about x
    assert isinstance(plan.input.right, Filter)
    assert _rows(lf.compute(device="cpu")) == [(1, 1, 10), (2, 2, 20)]
