"""F-091: lag / lead keep NULL as NULL, keep the input dtype, and fill with the default like DuckDB.
Expected values come from DuckDB itself."""
import duckdb
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col

NAN = float("nan")
K = [1, 1, 1, 1, 1, 2, 2, 2, 3, 3]
O = [3.0, None, NAN, 1.0, 2.0, NAN, 2.0, None, 1.0, 5.0]   # distinct within a partition
COLS = {
    "i": pa.array([1, None, 3, 4, None, 6, 7, None, 9, 10], pa.int64()),
    "f": pa.array([1.5, None, 3.0, NAN, 5.0, None, 7.0, 8.0, None, 10.0], pa.float64()),
    "s": pa.array(["a", None, "c", "d", "e", "f", None, "h", "i", "j"], pa.string()),
}
DEFAULTS = {"i": 99, "f": -1.5, "s": "zz"}


def _table():
    return pa.table({"k": pa.array(K, pa.int64()), "o": pa.array(O, pa.float64()),
                     "id": pa.array(range(len(K)), pa.int64()), **COLS})


def _duck(t, sql):
    con = duckdb.connect()
    con.register("t", t)
    return con.execute(f"SELECT {sql} AS r FROM t ORDER BY id").to_arrow_table().column("r")


def _mx(t, expr, device="cpu"):
    out = LazyFrame(t).with_columns(expr.alias("r")).compute(device=device)
    ids = out.column("id").to_pylist()
    order = [ids.index(i) for i in range(t.num_rows)]
    return out.column("r").combine_chunks().take(pa.array(order))


def _same(got, want):
    assert repr(got.to_pylist()) == repr(want.to_pylist()), (got.to_pylist(), want.to_pylist())
    assert got.type == want.type


@pytest.mark.parametrize("c", ["i", "f", "s"])
@pytest.mark.parametrize("fn", ["lag", "lead"])
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("shape", ["whole", "ordered", "partitioned"])
@pytest.mark.parametrize("default", [False, True])
def test_lag_lead_match_duckdb(c, fn, n, shape, default):
    t = _table()
    d = DEFAULTS[c]
    args = (n, d) if default else (n,)
    sql_d = f", {d!r}" if default else ""
    if shape == "whole":      # no partition, no order: table order
        expr, over = getattr(col(c), fn)(*args).over(), "ORDER BY id"
    elif shape == "ordered":
        expr, over = getattr(col(c), fn)(*args).over(order_by="o"), "ORDER BY o"
    else:
        expr, over = getattr(col(c), fn)(*args).over("k", order_by="o"), "PARTITION BY k ORDER BY o"
    _same(_mx(t, expr), _duck(t, f"{fn}({c}, {n}{sql_d}) OVER ({over})"))


def test_lag_of_null_gives_null_not_nan():
    t = pa.table({"id": pa.array([0, 1, 2]), "v": pa.array([1.0, None, 3.0])})
    assert _mx(t, col("v").lag(1).over()).to_pylist() == [None, 1.0, None]
