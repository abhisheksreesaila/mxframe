"""Optimizer: push filter conjuncts below joins (predicate pushdown)."""
import numpy as np
import pyarrow as pa
import pytest

from omxcore import LazyFrame, col, lit
from omxcore.lazy_frame import Filter, Join, Scan
from omxcore.optimizer import optimize_plan


def tables(n=20_000, m=500, seed=0):
    rng = np.random.default_rng(seed)
    left = pa.table({"lk": rng.integers(0, m, n).astype(np.int32), "d": rng.integers(0, 100, n).astype(np.int32),
                     "v": rng.random(n), "shared": rng.integers(0, 3, n).astype(np.int32)})
    right = pa.table({"rk": np.arange(m, dtype=np.int32), "t": rng.integers(0, 10, m).astype(np.int32),
                      "shared": rng.integers(0, 3, m).astype(np.int32)})
    return left, right


def _find(plan, kind):
    out = []
    def walk(p):
        if isinstance(p, kind):
            out.append(p)
        for a in ("input", "left", "right"):
            if hasattr(p, a):
                walk(getattr(p, a))
    walk(plan)
    return out


def test_single_side_conjuncts_move_below_the_join():
    left, right = tables()
    lf = (LazyFrame(left).join(LazyFrame(right), left_on="lk", right_on="rk")
          .filter((col("d") < lit(10)) & (col("t") == lit(3)) & (col("v") > col("d"))))
    opt = optimize_plan(lf.plan)
    joins = _find(opt.plan, Join)
    assert len(joins) == 1
    j = joins[0]
    assert isinstance(j.left, Filter) and isinstance(j.right, Filter)
    assert "push_filters_through_joins" in opt.trace


def test_ambiguous_names_and_left_join_right_side_stay_above():
    left, right = tables()
    lf = LazyFrame(left).join(LazyFrame(right), left_on="lk", right_on="rk").filter(col("shared") == lit(1))
    j = _find(optimize_plan(lf.plan).plan, Join)[0]
    # a duplicate name is the LEFT column (the right one is `shared_right`): it sinks to the left only
    assert isinstance(j.left, Filter) and not isinstance(j.right, Filter)
    lf = LazyFrame(left).join(LazyFrame(right), left_on="lk", right_on="rk").filter(col("shared_right") == lit(1))
    j = _find(optimize_plan(lf.plan).plan, Join)[0]
    assert not isinstance(j.left, Filter) and not isinstance(j.right, Filter)     # renamed: stays above
    lf = LazyFrame(left).join(LazyFrame(right), left_on="lk", right_on="rk", how="left").filter(col("t") == lit(3))
    j = _find(optimize_plan(lf.plan).plan, Join)[0]
    assert not isinstance(j.right, Filter)


@pytest.mark.parametrize("how", ["inner", "left"])
def test_results_identical_with_and_without_pushdown(how):
    left, right = tables()
    lf = (LazyFrame(left).join(LazyFrame(right), left_on="lk", right_on="rk", how=how)
          .filter(col("d") < lit(30)).filter(col("t") >= lit(2))
          .groupby("t").agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("t"))
    a = lf.compute(device="cpu", optimize=True)
    b = lf.compute(device="cpu", optimize=False)
    assert a.column("t").to_pylist() == b.column("t").to_pylist()
    np.testing.assert_allclose(a.column("s").to_numpy(), b.column("s").to_numpy(), rtol=1e-12)
    np.testing.assert_array_equal(a.column("n").to_numpy(), b.column("n").to_numpy())


def test_pushdown_through_chained_joins():
    left, right = tables()
    left, right = left.drop(["shared"]), right.drop(["shared"])   # engine can't join duplicate names (F-011)
    third = pa.table({"t": np.arange(10, dtype=np.int32), "name": [f"n{i}" for i in range(10)]})
    lf = (LazyFrame(left).join(LazyFrame(right), left_on="lk", right_on="rk")
          .join(LazyFrame(third), on="t").filter(col("d") < lit(5)))
    opt = optimize_plan(lf.plan).plan
    inner = _find(opt, Join)[-1]          # the lk/rk join
    assert isinstance(inner.left, Filter)
    a = lf.compute(device="cpu").sort_by("v")
    b = lf.compute(device="cpu", optimize=False).sort_by("v")
    assert a.num_rows == b.num_rows and a.column("v").equals(b.column("v"))


def test_window_and_aggregate_predicates_are_not_pushed():
    from omxcore.optimizer import _expr_columns
    assert _expr_columns(col("v").sum().over("t") > lit(10)) is None
    assert _expr_columns(col("v") > col("v").mean()) is None
