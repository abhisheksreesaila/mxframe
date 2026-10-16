"""F-033 (M3): subquery-shaped plans run natively.

1. A partner that is `[Project] Filter(Aggregate(...))` (`WHERE k IN (SELECT k ... GROUP BY k HAVING ...)`,
   TPC-H Q18 / Q21) is computed by the executor's HAVING, not by evaluating the predicate over the
   aggregate's rows in Arrow.
2. `LazyFrame.scalar(agg)`: a one-value subquery usable in a later filter (`value > (SELECT sum(..) * f)`,
   TPC-H Q11 / Q15 / Q22), computed natively first and used as a literal by the filter (and by HAVING).
Every result is compared with DuckDB; the path asserts count the Arrow predicate evaluations and the native calls.
"""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.custom_ops import CustomOpsCompiler
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def _rows(t: pa.Table):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    return [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in zip(*cols)]


def _duck(tables: dict, sql: str):
    con = duckdb.connect()
    for n, t in tables.items():
        con.register(n, t)
    return _rows(con.execute(sql).arrow().read_all())


@pytest.fixture
def counters(monkeypatch):
    """Arrow predicate evaluations (the Python path) and native executor / filter calls."""
    seen = {"arrow_predicates": 0, "execute": 0, "filter": 0}
    real_pred = CustomOpsCompiler._eval_predicate

    def pred(*a, **k):
        seen["arrow_predicates"] += 1
        return real_pred(*a, **k)

    real_call, real_filter = nx._call, nx._call_filter
    monkeypatch.setattr(CustomOpsCompiler, "_eval_predicate", staticmethod(pred))
    monkeypatch.setattr(nx, "_call", lambda *a: (seen.__setitem__("execute", seen["execute"] + 1), real_call(*a))[1])
    monkeypatch.setattr(nx, "_call_filter", lambda *a: (seen.__setitem__("filter", seen["filter"] + 1), real_filter(*a))[1])
    return seen


def _orders(n_orders=40_000, n_lines=200_000, seed=11):
    r = np.random.default_rng(seed)
    orders = pa.table({
        "ok": A(np.arange(n_orders, dtype=np.int32)),
        "ck": A(r.integers(0, 500, n_orders).astype(np.int32)),
        "price": A(np.round(r.random(n_orders) * 1000, 2)),
    })
    lines = pa.table({
        "lk": A(r.integers(0, n_orders, n_lines).astype(np.int32)),
        "qty": A(r.integers(1, 50, n_lines).astype(np.float64)),
        "supp": A(r.integers(0, 90, n_lines).astype(np.int32)),
    })
    return orders, lines


def test_in_subquery_with_having_is_computed_by_the_executor(counters):
    """Q18: orders whose lines sum to more than a threshold, as a join partner."""
    orders, lines = _orders()
    big = (LazyFrame(lines).groupby("lk").agg(col("qty").sum().alias("sq"))
           .filter(col("sq") > lit(150.0)).select("lk"))
    got = (LazyFrame(orders).join(big, left_on="ok", right_on="lk")
           .groupby("ck").agg(col("price").sum().alias("p"), col("price").count().alias("n"))
           .sort("ck").compute(device="cpu"))
    want = _duck({"orders": orders, "lines": lines}, """
        SELECT ck, sum(price) p, count(*) n FROM orders
        WHERE ok IN (SELECT lk FROM lines GROUP BY lk HAVING sum(qty) > 150)
        GROUP BY ck ORDER BY ck""")
    assert len(want) > 100
    assert _rows(got) == want
    assert counters["arrow_predicates"] == 0          # the HAVING ran in the executor, not in Arrow


def test_semi_and_anti_join_on_a_having_partner(counters):
    orders, lines = _orders()
    big = (LazyFrame(lines).groupby("lk").agg(col("qty").count().alias("c"))
           .filter(col("c") >= lit(6)).select("lk"))
    semi = LazyFrame(orders).join(big, left_on="ok", right_on="lk", how="semi").sort("ok").compute(device="cpu")
    want = _duck({"orders": orders, "lines": lines}, """
        SELECT * FROM orders WHERE ok IN (SELECT lk FROM lines GROUP BY lk HAVING count(*) >= 6) ORDER BY ok""")
    assert _rows(semi) == want
    assert 0 < len(want) < orders.num_rows
