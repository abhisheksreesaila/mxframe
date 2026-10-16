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
    return list(zip(*cols))


def _same(got, want, rel=1e-9):
    """Rows equal, floats within `rel` (the executor's parallel sums differ from DuckDB's in the last bits)."""
    assert len(got) == len(want), (len(got), len(want))
    for g, w in zip(got, want):
        assert len(g) == len(w)
        for a, b in zip(g, w):
            if isinstance(a, float) or isinstance(b, float):
                assert abs(a - b) <= rel * max(1.0, abs(a), abs(b)), (g, w)
            else:
                assert a == b, (g, w)


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
    _same(_rows(got), want)
    assert counters["arrow_predicates"] == 0          # the HAVING ran in the executor, not in Arrow


def test_semi_and_anti_join_on_a_having_partner(counters):
    orders, lines = _orders()
    big = (LazyFrame(lines).groupby("lk").agg(col("qty").count().alias("c"))
           .filter(col("c") >= lit(6)).select("lk"))
    semi = LazyFrame(orders).join(big, left_on="ok", right_on="lk", how="semi").sort("ok").compute(device="cpu")
    want = _duck({"orders": orders, "lines": lines}, """
        SELECT * FROM orders WHERE ok IN (SELECT lk FROM lines GROUP BY lk HAVING count(*) >= 6) ORDER BY ok""")
    _same(_rows(semi), want)
    assert 0 < len(want) < orders.num_rows
    assert counters["arrow_predicates"] == 0 and counters["execute"] >= 1      # the partner's HAVING ran in the executor
    counters["execute"] = 0
    anti = LazyFrame(orders).join(big, left_on="ok", right_on="lk", how="anti").sort("ok").compute(device="cpu")
    want_anti = _duck({"orders": orders, "lines": lines}, """
        SELECT * FROM orders WHERE ok NOT IN (SELECT lk FROM lines GROUP BY lk HAVING count(*) >= 6) ORDER BY ok""")
    _same(_rows(anti), want_anti)
    assert 0 < len(want_anti) < orders.num_rows and len(want_anti) + len(want) == orders.num_rows
    assert counters["arrow_predicates"] == 0 and counters["execute"] >= 1


# ── scalar subqueries ───────────────────────────────────────────────────────────────────────────

def _stock(n=60_000, seed=3):
    r = np.random.default_rng(seed)
    ps = pa.table({
        "pk": A(r.integers(0, 4000, n).astype(np.int32)),
        "sk": A(r.integers(0, 120, n).astype(np.int32)),
        "cost": A(np.round(r.random(n) * 900 + 1, 2)),
        "qty": A(r.integers(1, 9999, n).astype(np.int32)),
    })
    sup = pa.table({"sk": A(np.arange(120, dtype=np.int32)), "nk": A(r.integers(0, 25, 120).astype(np.int32))})
    return ps, sup


def test_scalar_subquery_feeds_a_having(counters):
    """Q11: groups whose value exceeds a fraction of the grand total (computed natively first)."""
    ps, sup = _stock()
    base = LazyFrame(ps).join(LazyFrame(sup).filter(col("nk") == lit(7)).select("sk"), on="sk")
    got = (base.groupby("pk").agg((col("cost") * col("qty")).sum().alias("v"))
           .filter(col("v") > base.scalar((col("cost") * col("qty")).sum()) * lit(0.0002))
           .sort("v", descending=True).compute(device="cpu"))
    want = _duck({"ps": ps, "sup": sup}, """
        WITH j AS (SELECT ps.pk, ps.cost * ps.qty AS v FROM ps JOIN sup ON ps.sk = sup.sk WHERE sup.nk = 7)
        SELECT pk, sum(v) v FROM j GROUP BY pk HAVING sum(v) > (SELECT sum(v) * 0.0002 FROM j) ORDER BY v DESC""")
    assert 5 < len(want) < 4000
    _same(_rows(got), want)
    assert counters["arrow_predicates"] == 0
    assert counters["execute"] == 2                    # the scalar, then the grouped aggregate with its HAVING


def test_scalar_subquery_in_a_row_filter_with_a_float32_column(counters):
    """Q22: balance above the average of the positive balances (float32 column against a float64 mean)."""
    r = np.random.default_rng(5)
    n = 50_000
    cust = pa.table({"ck": A(np.arange(n, dtype=np.int32)),
                     "bal": A(np.round(r.random(n) * 12000 - 1000, 2).astype(np.float32)),
                     "code": A(r.integers(10, 35, n).astype(np.int32))})
    pos = LazyFrame(cust).filter(col("bal") > lit(0.0))
    got = (LazyFrame(cust).filter(col("bal") > pos.scalar(col("bal").mean()))
           .groupby("code").agg(col("ck").count().alias("n"), col("bal").sum().alias("t"))
           .sort("code").compute(device="cpu"))
    want = _duck({"cust": cust}, """
        SELECT code, count(*) n, sum(bal) t FROM cust
        WHERE bal > (SELECT avg(bal) FROM cust WHERE bal > 0) GROUP BY code ORDER BY code""")
    assert len(want) > 10
    assert nx.last_path() == "native", nx.last_decline()             # the main plan after the subquery
    assert counters["execute"] == 2 and counters["arrow_predicates"] == 0     # the subquery, then the grouped plan
    assert [r[:2] for r in _rows(got)] == [r[:2] for r in want]
    assert all(abs(g[2] - w[2]) < 1e-2 * max(1, abs(w[2])) for g, w in zip(_rows(got), want))


def test_scalar_max_equality_on_a_computed_view_is_exact(counters):
    """Q15: the rows whose total equals the maximum total (float64 equal to the last bit). The view is computed
    once, so both sides read the same table."""
    orders, lines = _orders()
    view = LazyFrame(LazyFrame(lines).groupby("supp").agg((col("qty") * lit(1.1)).sum().alias("rev"))
                     .compute(device="cpu"))
    got = view.filter(col("rev") == view.scalar(col("rev").max())).compute(device="cpu")
    want = _duck({"lines": lines}, """
        WITH v AS (SELECT supp, sum(qty * 1.1) rev FROM lines GROUP BY supp)
        SELECT supp, rev FROM v WHERE rev = (SELECT max(rev) FROM v)""")
    assert len(want) == 1
    _same(_rows(got), want)


@pytest.mark.xfail(strict=True, reason="F-064: executor float sums are not bit-reproducible, so a view recomputed "
                                       "inside the main plan misses its own maximum")
def test_scalar_max_equality_on_an_uncomputed_view_is_exact():
    """The same shape with the view left lazy: the scalar and the main plan each run the aggregate. 6M rows,
    10k groups (the size at which the sums differ between runs; strict: it starts passing when F-064 lands)."""
    r = np.random.default_rng(7)
    n = 6_000_000
    t = pa.table({"s": A(r.integers(0, 10000, n).astype(np.int32)), "p": A(r.random(n) * 1000), "d": A(r.random(n))})
    view = LazyFrame(t).groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"))
    for _ in range(8):
        got = view.filter(col("tot") == view.scalar(col("tot").max())).compute(device="cpu")
        assert got.num_rows == 1


def test_scalar_in_a_computed_column_and_a_count_scalar(counters):
    orders, lines = _orders()
    lf = LazyFrame(lines)
    share = (lf.groupby("supp").agg(col("qty").sum().alias("q"))
             .with_columns((col("q") / lf.scalar(col("qty").sum())).alias("frac")).sort("supp").compute(device="cpu"))
    want = _duck({"lines": lines}, """
        SELECT supp, sum(qty) q, sum(qty) / (SELECT sum(qty) FROM lines) frac FROM lines GROUP BY supp ORDER BY supp""")
    _same(_rows(share), want)
    n = (lf.filter(col("supp") < lf.scalar(col("qty").count()) * lit(0.0004))
         .groupby().agg(col("qty").count().alias("n")).compute(device="cpu"))
    assert n.column("n")[0].as_py() == _duck(
        {"lines": lines}, "SELECT count(*) FROM lines WHERE supp < (SELECT count(*) FROM lines) * 0.0004")[0][0]


def test_scalar_over_no_rows_is_null_and_keeps_no_rows(counters):
    """SQL: `x > (SELECT max(y) FROM empty)` is NULL, so the filter keeps nothing (also under NOT)."""
    orders, lines = _orders()
    lf = LazyFrame(lines)
    none = lf.filter(col("supp") > lit(10_000))
    got = lf.filter(col("supp") > none.scalar(col("qty").max())).compute(device="cpu")
    assert got.num_rows == 0
    got_not = lf.filter(~(col("supp") > none.scalar(col("qty").max()))).compute(device="cpu")
    assert got_not.num_rows == 0


def test_null_subquery_inside_and_or_and_under_not(counters):
    orders, lines = _orders()
    lf = LazyFrame(lines)
    nul = lf.filter(col("supp") > lit(10_000)).scalar(col("qty").max())
    # AND with a NULL term is never true; OR with a NULL term is true where the other term is
    both = lf.filter((col("supp") < lit(30)) & (col("supp") > nul)).compute(device="cpu")
    either = lf.filter((col("supp") < lit(30)) | (col("supp") > nul)).compute(device="cpu")
    assert both.num_rows == 0
    assert either.num_rows == _duck({"lines": lines}, "SELECT count(*) FROM lines WHERE supp < 30")[0][0]
    # NOT over a NULL term would need three-valued logic over rows: refused, never answered wrongly
    with pytest.raises(ValueError, match="NULL"):
        lf.filter(~((col("supp") < lit(30)) | (col("supp") > nul))).compute(device="cpu")
