"""F-027: aggregates over no rows follow SQL. A global SUM / MEAN / MIN / MAX over zero rows is one
row of NULL, COUNT is 0; a grouped aggregate over zero rows is zero groups. Every aggregate path is
covered: a plan the fused pipeline takes, the masked global path, expressions, and an empty input."""
import numpy as np
import pyarrow as pa
import pytest

from omxcore import LazyFrame, col, lit
from omxcore.np_arrow import array as A


def _t(n=1000):
    r = np.random.default_rng(27)
    return pa.table({"k": A(r.integers(0, 5, n).astype(np.int32)), "v": A(r.random(n)),
                     "f": A(r.random(n).astype(np.float32)), "d": A(r.integers(0, 100, n).astype(np.int32))})


def _global(lf):
    return lf.groupby().agg(col("v").sum().alias("s"), col("v").mean().alias("m"), col("f").min().alias("lo"),
                            col("f").max().alias("hi"), col("v").count().alias("n"),
                            (col("v") * lit(2.0) - col("f")).sum().alias("e"))


NONE = {"s": [None], "m": [None], "lo": [None], "hi": [None], "n": [0], "e": [None]}

PLANS = {
    "filter keeps nothing": lambda t: _global(LazyFrame(t).filter(col("d") > lit(1000))),
    "two filters keep nothing": lambda t: _global(LazyFrame(t).filter(col("d") > lit(50))
                                                  .filter(col("d") < lit(10))),
    "empty input table": lambda t: _global(LazyFrame(t.slice(0, 0))),
    "zero-chunk input": lambda t: _global(LazyFrame(pa.table({c: pa.chunked_array([], type=t.schema.field(c).type)
                                                              for c in t.column_names}))),
}


@pytest.mark.parametrize("name", list(PLANS))
def test_global_aggregate_over_no_rows_is_sql_null(name):
    out = PLANS[name](_t()).compute(device="cpu").to_pydict()
    assert out == NONE, out


def test_single_aggregate_over_no_rows():
    t = _t()
    empty = LazyFrame(t).filter(col("d") > lit(1000))
    assert empty.groupby().agg(col("v").sum().alias("s")).compute(device="cpu").to_pydict() == {"s": [None]}
    assert empty.groupby().agg(col("v").count().alias("n")).compute(device="cpu").to_pydict() == {"n": [0]}


def test_grouped_aggregate_over_no_rows_is_no_groups():
    out = (LazyFrame(_t()).filter(col("d") > lit(1000)).groupby("k")
           .agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    assert out.num_rows == 0 and out.column_names == ["k", "s", "n"]


def _with_nulls(n=1000):
    """The same table with nulls in v and f: aggregates go through `_aggregate_with_nulls`."""
    t = _t(n)
    r = np.random.default_rng(5)
    v = np.ma.array(t.column("v").to_numpy(), mask=r.random(n) < 0.3)
    f = np.ma.array(t.column("f").to_numpy(), mask=r.random(n) < 0.3)
    return t.set_column(1, "v", A(v)).set_column(2, "f", A(f))


@pytest.mark.parametrize("name", list(PLANS))
def test_global_aggregate_over_no_rows_with_nulls_is_sql_null(name):
    out = PLANS[name](_with_nulls()).compute(device="cpu").to_pydict()
    assert out == NONE, out


def test_grouped_over_no_rows_with_nulls_and_an_all_null_group():
    t = _with_nulls()
    out = (LazyFrame(t).filter(col("d") > lit(1000)).groupby("k")
           .agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    assert out.num_rows == 0
    k = pa.table({"k": A(np.array([1, 1, 2], np.int32)),
                  "v": A(np.ma.array(np.array([1.0, 2.0, 3.0]), mask=[1, 1, 0]))})
    out = (LazyFrame(k).groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("n"))
           .sort("k").compute(device="cpu").to_pydict())
    assert out == {"k": [1, 2], "s": [None, 3.0], "n": [0, 1]}


def test_some_rows_still_sum_normally():
    t = _t()
    out = LazyFrame(t).filter(col("d") > lit(98)).groupby().agg(col("v").sum().alias("s")).compute(device="cpu")
    d, v = t.column("d").to_numpy(), t.column("v").to_numpy()
    assert out.column("s")[0].as_py() == pytest.approx(v[d > 98].sum(), rel=1e-12)
