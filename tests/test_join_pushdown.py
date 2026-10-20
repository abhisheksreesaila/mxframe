"""F-018: a disjunction over both sides of a join filters each side by what it implies."""
import pyarrow as pa
import pytest

from omxcore import LazyFrame, col, lit
from omxcore.lazy_frame import Filter, Join, Scan
from omxcore.optimizer import optimize_plan


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


# ── projection pushdown: with_columns passes every column through; the parent reads a few ────

def _write(tmp_path, name, table):
    import pyarrow.parquet as pq
    path = tmp_path / f"{name}.parquet"
    pq.write_table(table, path)
    return str(path)


def _leaves(plan):
    from omxcore.lazy_frame import ParquetScan
    if isinstance(plan, ParquetScan):
        yield plan
    elif isinstance(plan, Join):
        yield from _leaves(plan.left)
        yield from _leaves(plan.right)
    elif hasattr(plan, "input"):
        yield from _leaves(plan.input)


def test_with_columns_after_a_join_reads_only_the_columns_the_aggregate_needs(tmp_path):
    import omxcore as mx
    from omxcore.io_parquet import resolve_plan
    fact = pa.table({"k": _i32([1, 2, 3, 4]), "d": _i32([10, 20, 30, 40]), "x": pa.array([1.0, 2.0, 3.0, 4.0]),
                     "pad1": pa.array(["a"] * 4), "pad2": pa.array([1.5] * 4)})
    dim = pa.table({"k": _i32([1, 2, 3, 4]), "name": pa.array(["p", "q", "p", "q"]),
                    "pad3": pa.array(["b"] * 4), "pad4": _i32([7] * 4)})
    lf = (mx.scan_parquet(_write(tmp_path, "fact", fact)).join(mx.scan_parquet(_write(tmp_path, "dim", dim)), on="k")
          .with_columns((col("d") - lit(10)).alias("bucket"))
          .groupby("name", "bucket").agg(col("x").sum().alias("s")))
    opt = optimize_plan(lf.plan)
    assert "prune_unused_columns" in opt.trace
    read = {tuple(sorted(leaf.columns)) for leaf in _leaves(resolve_plan(opt.plan, read=False))}
    assert read == {("d", "k", "x"), ("k", "name")}, read
    got = lf.compute(device="cpu")
    ref = {("p", 0): 1.0, ("q", 10): 2.0, ("p", 20): 3.0, ("q", 30): 4.0}
    rows = {(n, int(b)): s for n, b, s in zip(*[got.column(c).to_pylist() for c in ("name", "bucket", "s")])}
    assert rows == ref, rows


def test_pruning_keeps_what_a_select_at_the_top_returns():
    a, b = _tables()
    lf = LazyFrame(a).join(LazyFrame(b), on="k").with_columns((col("x") + lit(1)).alias("z"))
    got = lf.compute(device="cpu")
    assert got.column_names == ["k", "x", "y", "z"]
    got = lf.select("k", "z").compute(device="cpu")
    assert got.column_names == ["k", "z"]
    assert _rows(got) == [(1, 2), (2, 3), (3, 4), (4, 5)]


def test_pruning_below_a_join_keeps_the_right_suffix_names():
    A = pa.table({"k": [1, 2, 3], "x": [10, 20, 30], "a": [7, 8, 9]})
    B = pa.table({"k2": [1, 2, 2], "x": [100, 200, 201], "x_right": [5, 6, 7]})
    q1 = (LazyFrame(A).select(col("k"), col("x"))
          .join(LazyFrame(B).select(col("k2"), col("x")), left_on="k", right_on="k2").select(col("x_right")))
    q2 = (LazyFrame(A).with_columns((col("a") + lit(1)).alias("b"))
          .join(LazyFrame(B), left_on="k", right_on="k2").select(col("x_right2"), col("b")))
    for q in (q1, q2):
        assert q.compute(device="cpu").to_pydict() == q.compute(device="cpu", optimize=False).to_pydict()
