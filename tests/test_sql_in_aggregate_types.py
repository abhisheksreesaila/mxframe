"""F-090: integer MIN / MAX / SUM keep int64 (COUNT is int64), and IN / NOT IN against an aggregate
subquery works for int and float columns. Expected answers come from DuckDB tables."""
import duckdb
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import LazyFrame, col

DEVICES = ["cpu", "gpu"]


def _t():
    return pa.table({
        "a": pa.array([1, 1, 2, 2, 3, 3, 4], pa.int64()),
        "i": pa.array([1, 5, 2, 7, 3, 3, 9], pa.int64()),
        "x": pa.array([1.0, 5.5, 2.0, 7.0, 3.0, None, 9.0], pa.float64()),
    })


def _duck_types(q, t):
    con = duckdb.connect()
    con.register("t", t)
    return con.execute(q).to_arrow_table()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", ["max", "min", "sum", "count"])
def test_integer_group_aggregate_keeps_int64(fn, device):
    t = _t()
    want = _duck_types(f"SELECT a, {fn}(i) AS m FROM t GROUP BY a ORDER BY a", t)
    got = LazyFrame(t).groupby("a").agg(getattr(col("i"), fn)().alias("m")).compute(device=device)
    got = got.sort_by("a")
    if fn != "sum":              # DuckDB's SUM of int64 is HUGEINT; its value is checked, not its type
        assert got.schema.field("m").type == want.schema.field("m").type
    assert [float(v) for v in got.column("m").to_pylist()] == [float(v) for v in want.column("m").to_pylist()]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", ["max", "min"])
def test_integer_global_aggregate_keeps_int64(fn, device):
    t = _t()
    want = _duck_types(f"SELECT {fn}(i) AS m FROM t", t)
    got = LazyFrame(t).groupby().agg(getattr(col("i"), fn)().alias("m")).compute(device=device)
    assert got.schema.field("m").type == want.schema.field("m").type
    assert got.column("m").to_pylist() == want.column("m").to_pylist()


SUBS = [f"{fn}({c})" for fn in ("max", "min", "sum", "count") for c in ("i", "x")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("neg", ["", "NOT "])
@pytest.mark.parametrize("outer", ["i", "x"])
@pytest.mark.parametrize("sub", SUBS)
def test_in_not_in_aggregate_subquery(sub, outer, neg, device):
    t = _t()
    q = f"SELECT a, i, x FROM t WHERE {outer} {neg}IN (SELECT {sub} FROM t GROUP BY a) ORDER BY a, i"
    want = _duck_types(q, t)
    got = mx.sql(q, t=t).compute(device=device).sort_by([("a", "ascending"), ("i", "ascending")])
    assert repr(got.to_pydict()) == repr(want.to_pydict())


@pytest.mark.parametrize("device", DEVICES)
def test_integer_max_through_filter_and_join_keeps_int64(device):
    t = _t()
    u = pa.table({"a": pa.array([1, 2, 3, 4], pa.int64()), "z": pa.array([10, 20, 30, 40], pa.int64())})
    q = "SELECT t.a, max(t.i) AS m, min(u.z) AS n FROM t JOIN u ON t.a = u.a WHERE t.i > 1 GROUP BY t.a ORDER BY t.a"
    con = duckdb.connect()
    con.register("t", t)
    con.register("u", u)
    want = con.execute(q).to_arrow_table()
    got = mx.sql(q, t=t, u=u).compute(device=device).sort_by("a")
    assert got.schema.field("m").type == want.schema.field("m").type == pa.int64()
    assert got.schema.field("n").type == want.schema.field("n").type == pa.int64()
    assert got.to_pydict() == want.to_pydict()


BIG = 2 ** 60


def _big():
    return pa.table({
        "a": pa.array([1, 1, 2, 2, 2], pa.int64()),
        "i": pa.array([BIG + 1, BIG + 3, -BIG - 1, 5, BIG + 7], pa.int64()),
    })


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", ["max", "min"])
def test_integer_min_max_past_2_53_are_exact(fn, device):
    t = _big()
    want = _duck_types(f"SELECT a, {fn}(i) AS m FROM t GROUP BY a ORDER BY a", t)
    got = LazyFrame(t).groupby("a").agg(getattr(col("i"), fn)().alias("m")).compute(device=device).sort_by("a")
    assert got.column("m").to_pylist() == want.column("m").to_pylist()
    want = _duck_types(f"SELECT {fn}(i) AS m FROM t", t)
    got = LazyFrame(t).groupby().agg(getattr(col("i"), fn)().alias("m")).compute(device=device)
    assert got.column("m").to_pylist() == want.column("m").to_pylist()
    assert got.schema.field("m").type == pa.int64()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("neg", ["", "NOT "])
@pytest.mark.parametrize("fn", ["max", "min"])
def test_in_over_integer_max_past_2_53(fn, neg, device):
    t = _big()
    q = f"SELECT a, i FROM t WHERE i {neg}IN (SELECT {fn}(i) FROM t GROUP BY a) ORDER BY a, i"
    want = _duck_types(q, t)
    got = mx.sql(q, t=t).compute(device=device).sort_by([("a", "ascending"), ("i", "ascending")])
    assert got.to_pydict() == want.to_pydict()


@pytest.mark.parametrize("device", DEVICES)
def test_int_float_join_keeps_each_sides_key_column(device):
    t = pa.table({"i": pa.array([1, 2, 3, 4], pa.int64()), "p": pa.array([10, 20, 30, 40], pa.int64())})
    u = pa.table({"x": pa.array([2.0, 4.0, 4.5], pa.float64()), "q": pa.array([1, 2, 3], pa.int64())})
    q = "SELECT t.i, u.x, p, q FROM t JOIN u ON t.i = u.x ORDER BY t.i"
    con = duckdb.connect()
    con.register("t", t)
    con.register("u", u)
    want = con.execute(q).to_arrow_table()
    got = mx.sql(q, t=t, u=u).compute(device=device).sort_by("i")
    assert got.schema.types == want.schema.types
    assert got.to_pydict() == want.to_pydict()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variant", ["plain", "with_big_int64", "with_nulls"])
@pytest.mark.parametrize("grouped", [True, False])
def test_int32_min_max_keep_int32(variant, grouped, device):
    t = pa.table({
        "a": pa.array([1, 1, 2, 2], pa.int64()),
        "j": pa.array([5, None if variant == "with_nulls" else 9, -3, 7], pa.int32()),
        "b": pa.array([2 ** 60 + 1, 2 ** 60 + 3, 4, 5] if variant == "with_big_int64" else [1, 2, 3, 4], pa.int64()),
    })
    extra = ", max(b) AS mb" if variant == "with_big_int64" else ""
    q = (f"SELECT a, max(j) AS mx, min(j) AS mn{extra} FROM t GROUP BY a ORDER BY a" if grouped
         else f"SELECT max(j) AS mx, min(j) AS mn{extra} FROM t")
    want = _duck_types(q, t)
    got = mx.sql(q, t=t).compute(device=device)
    if grouped:
        got = got.sort_by("a")
    assert got.schema.types == want.schema.types
    assert got.to_pydict() == want.to_pydict()


@pytest.mark.parametrize("device", DEVICES)
def test_hidden_join_key_name_does_not_collide_with_a_user_column(device):
    t = pa.table({"i": pa.array([1, 2, 3], pa.int64())})
    u = pa.table({"x": pa.array([2.0, 3.0], pa.float64()), "__key_x": pa.array([1, 2], pa.int64())})
    q = "SELECT t.i, u.x, __key_x FROM t JOIN u ON t.i = u.x ORDER BY t.i"
    con = duckdb.connect()
    con.register("t", t)
    con.register("u", u)
    want = con.execute(q).to_arrow_table()
    got = mx.sql(q, t=t, u=u).compute(device=device).sort_by("i")
    assert got.schema.types == want.schema.types
    assert got.to_pydict() == want.to_pydict()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variant", ["plain", "with_big_int64", "with_nulls"])
@pytest.mark.parametrize("grouped", [True, False])
def test_int32_min_max_keep_int32(variant, grouped, device):
    t = pa.table({
        "a": pa.array([1, 1, 2, 2], pa.int64()),
        "j": pa.array([5, None if variant == "with_nulls" else 9, -3, 7], pa.int32()),
        "b": pa.array([2 ** 60 + 1, 2 ** 60 + 3, 4, 5] if variant == "with_big_int64" else [1, 2, 3, 4], pa.int64()),
    })
    extra = ", max(b) AS mb" if variant == "with_big_int64" else ""
    q = (f"SELECT a, max(j) AS mx, min(j) AS mn{extra} FROM t GROUP BY a ORDER BY a" if grouped
         else f"SELECT max(j) AS mx, min(j) AS mn{extra} FROM t")
    want = _duck_types(q, t)
    got = mx.sql(q, t=t).compute(device=device)
    if grouped:
        got = got.sort_by("a")
    assert got.schema.types == want.schema.types
    assert got.to_pydict() == want.to_pydict()


@pytest.mark.parametrize("device", DEVICES)
def test_hidden_join_key_name_does_not_collide_with_a_user_column(device):
    t = pa.table({"i": pa.array([1, 2, 3], pa.int64())})
    u = pa.table({"x": pa.array([2.0, 3.0], pa.float64()), "__key_x": pa.array([1, 2], pa.int64())})
    q = "SELECT t.i, u.x, __key_x FROM t JOIN u ON t.i = u.x ORDER BY t.i"
    con = duckdb.connect()
    con.register("t", t)
    con.register("u", u)
    want = con.execute(q).to_arrow_table()
    got = mx.sql(q, t=t, u=u).compute(device=device).sort_by("i")
    assert got.schema.types == want.schema.types
    assert got.to_pydict() == want.to_pydict()
