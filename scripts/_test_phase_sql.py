"""Phase 6 SQL Frontend tests: TPC-H Q1, Q6, Q3, Q12, Q14 via mx.sql()."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pyarrow as pa
import duckdb

import mxframe as mx

PASS = 0
FAIL = 0

def ok(label):
    global PASS; PASS += 1
    print(f"OK  {label}")

def fail(label, exc):
    global FAIL; FAIL += 1
    print(f"FAIL {label}: {exc}")
    import traceback; traceback.print_exc()

# ── helpers ──────────────────────────────────────────────────────────────
def _duckdb(query, **tables):
    con = duckdb.connect()
    for name, tbl in tables.items():
        con.register(name, tbl)
    return con.execute(query).arrow().read_all()

def _close(a: float, b: float, tol: float = 1.0) -> bool:
    return abs(a - b) < tol

def _sort_table(t: pa.Table, by: list[str]) -> pa.Table:
    import pyarrow.compute as pc
    import pyarrow as pa
    idx = pa.Table.from_pydict({col: t.column(col) for col in by})
    order = pa.array(range(len(idx)))
    # Use pyarrow sort_indices
    from pyarrow import compute
    indices = compute.sort_indices(t, sort_keys=[(c, "ascending") for c in by])
    return t.take(indices)

# ── data generators ───────────────────────────────────────────────────────
_rng = np.random.default_rng(42)

def make_lineitem(n=100_000):
    rng = np.random.default_rng(42)
    rf = np.array(["A", "N", "R"])[rng.integers(0, 3, size=n)]
    ls = np.array(["F", "O"])[rng.integers(0, 2, size=n)]
    return pa.table({
        "l_returnflag":    rf.tolist(),
        "l_linestatus":    ls.tolist(),
        "l_quantity":      rng.uniform(1.0, 50.0, size=n).astype(np.float32),
        "l_extendedprice": rng.uniform(900.0, 100_000.0, size=n).astype(np.float32),
        "l_discount":      rng.uniform(0.0, 0.10, size=n).astype(np.float32),
        "l_tax":           rng.uniform(0.0, 0.08, size=n).astype(np.float32),
        "l_shipdate":      rng.integers(8_000, 10_550, size=n).astype(np.int32),
    })

def make_q3_tables(n_c=5_000, n_o=50_000, n_l=200_000):
    rng = np.random.default_rng(99)
    segs = ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"]
    customer = pa.table({
        "c_custkey":    np.arange(1, n_c+1, dtype=np.int32),
        "c_mktsegment": rng.choice(segs, size=n_c).tolist(),
    })
    orders = pa.table({
        "o_orderkey":     np.arange(1, n_o+1, dtype=np.int32),
        "o_custkey":      rng.integers(1, n_c+1, size=n_o, dtype=np.int32),
        "o_orderdate":    rng.integers(8800, 9300, size=n_o, dtype=np.int32),
        "o_shippriority": rng.integers(0, 5, size=n_o, dtype=np.int32),
    })
    lineitem = pa.table({
        "l_orderkey":      rng.integers(1, n_o+1, size=n_l, dtype=np.int32),
        "l_extendedprice": rng.uniform(900.0, 100_000.0, size=n_l).astype(np.float32),
        "l_discount":      rng.uniform(0.0, 0.10, size=n_l).astype(np.float32),
        "l_shipdate":      rng.integers(8900, 9400, size=n_l, dtype=np.int32),
    })
    return customer, orders, lineitem

def make_q12_tables(n_o=20_000, n_l=100_000):
    rng = np.random.default_rng(77)
    priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
    orders = pa.table({
        "o_orderkey":      np.arange(1, n_o+1, dtype=np.int32),
        "o_orderpriority": rng.choice(priorities, size=n_o).tolist(),
    })
    lineitem = pa.table({
        "l_orderkey":    rng.integers(1, n_o+1, size=n_l, dtype=np.int32),
        "l_shipmode":    rng.choice(["MAIL","SHIP","TRUCK","AIR"], size=n_l).tolist(),
        "l_commitdate":  rng.integers(8200, 9000, size=n_l, dtype=np.int32),
        "l_receiptdate": rng.integers(8700, 9500, size=n_l, dtype=np.int32),
        "l_shipdate":    rng.integers(8000, 8900, size=n_l, dtype=np.int32),
    })
    return orders, lineitem

def make_q14_tables(n_p=10_000, n_l=80_000):
    rng = np.random.default_rng(88)
    part_types = ["PROMO STEEL", "PROMO BRASS", "BRUSHED COPPER",
                  "STANDARD STEEL", "ECONOMY TIN", "PROMO NICKEL"]
    part = pa.table({
        "p_partkey": np.arange(1, n_p+1, dtype=np.int32),
        "p_type":    rng.choice(part_types, size=n_p).tolist(),
    })
    lineitem = pa.table({
        "l_partkey":       rng.integers(1, n_p+1, size=n_l, dtype=np.int32),
        "l_extendedprice": rng.uniform(900.0, 100_000.0, size=n_l).astype(np.float32),
        "l_discount":      rng.uniform(0.0, 0.10, size=n_l).astype(np.float32),
        "l_shipdate":      rng.integers(9200, 9600, size=n_l, dtype=np.int32),
    })
    return part, lineitem


# ── tests ─────────────────────────────────────────────────────────────────

def test_q6_sql():
    """Q6 via SQL: filter + global SUM — matches DuckDB."""
    li = make_lineitem()
    q = """
        SELECT SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= 8761
          AND l_shipdate < 9126
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24
    """
    mx_result = mx.sql(q, lineitem=li).compute(device="cpu")
    ref = _duckdb(q, lineitem=li)
    mx_val = float(mx_result.column("revenue")[0].as_py())
    ref_val = float(ref.column("revenue")[0].as_py())
    assert _close(mx_val, ref_val, tol=max(abs(ref_val) * 0.01, 1.0)), \
        f"Q6 revenue mismatch: MX={mx_val:.2f} DuckDB={ref_val:.2f}"
    ok("Q6 via SQL (filter + SUM matches DuckDB)")


def test_q1_sql():
    """Q1 via SQL: filter + groupby + 8 aggs + sort — matches DuckDB."""
    li = make_lineitem()
    q = """
        SELECT l_returnflag, l_linestatus,
               SUM(l_quantity)      AS sum_qty,
               SUM(l_extendedprice) AS sum_base_price,
               SUM(l_extendedprice * (1 - l_discount))           AS sum_disc_price,
               SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
               AVG(l_quantity)      AS avg_qty,
               AVG(l_extendedprice) AS avg_price,
               AVG(l_discount)      AS avg_disc,
               COUNT(*)             AS count_order
        FROM lineitem
        WHERE l_shipdate <= 10471
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    """
    mx_result = mx.sql(q, lineitem=li).compute(device="cpu")
    ref = _duckdb(q, lineitem=li)
    # correctness: compare sum_qty and count_order per group
    mx_s = _sort_table(mx_result, ["l_returnflag", "l_linestatus"])
    ref_s = _sort_table(ref,      ["l_returnflag", "l_linestatus"])
    assert mx_s.num_rows == ref_s.num_rows, \
        f"Q1 row count mismatch: {mx_s.num_rows} vs {ref_s.num_rows}"
    for i in range(mx_s.num_rows):
        mx_cnt  = int(mx_s.column("count_order")[i].as_py())
        ref_cnt = int(ref_s.column("count_order")[i].as_py())
        assert abs(mx_cnt - ref_cnt) <= 1, \
            f"Q1 count mismatch row {i}: {mx_cnt} vs {ref_cnt}"
        mx_qty  = float(mx_s.column("sum_qty")[i].as_py())
        ref_qty = float(ref_s.column("sum_qty")[i].as_py())
        assert _close(mx_qty, ref_qty, tol=max(abs(ref_qty) * 0.01, 1.0)), \
            f"Q1 sum_qty mismatch row {i}: {mx_qty:.2f} vs {ref_qty:.2f}"
    ok(f"Q1 via SQL (grouped agg + sort, {mx_s.num_rows} groups, matches DuckDB)")


def test_q3_sql():
    """Q3 via SQL: 3-way join + filter + groupby + sort + limit."""
    customer, orders, lineitem = make_q3_tables()
    DATE = 9204  # 1995-03-15
    q = f"""
        SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate, o_shippriority
        FROM customer
        JOIN orders   ON c_custkey = o_custkey
        JOIN lineitem ON l_orderkey = o_orderkey
        WHERE c_mktsegment = 'BUILDING'
          AND o_orderdate < {DATE}
          AND l_shipdate  > {DATE}
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue
        LIMIT 10
    """
    mx_result = mx.sql(
        q, customer=customer, orders=orders, lineitem=lineitem
    ).compute(device="cpu")
    ref = _duckdb(
        q, customer=customer, orders=orders, lineitem=lineitem
    )
    assert mx_result.num_rows > 0, "Q3 SQL returned no rows"
    assert mx_result.num_rows <= 10, f"Q3 LIMIT 10 not applied: {mx_result.num_rows}"
    # total revenue should be in the right ballpark vs DuckDB
    mx_rev  = sum(float(v.as_py()) for v in mx_result.column("revenue"))
    ref_rev = sum(float(v.as_py()) for v in ref.column("revenue"))
    assert _close(mx_rev, ref_rev, tol=max(abs(ref_rev) * 0.02, 1.0)), \
        f"Q3 revenue differs: MX={mx_rev:.0f} DuckDB={ref_rev:.0f}"
    ok(f"Q3 via SQL (3-way join + limit, rows={mx_result.num_rows}, revenue matches DuckDB)")


def test_q12_sql():
    """Q12 via SQL: 2-way join + isin filter + CASE WHEN grouped sum."""
    orders, lineitem = make_q12_tables()
    q = """
        SELECT l_shipmode,
               SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
               SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
        FROM lineitem
        JOIN orders ON l_orderkey = o_orderkey
        WHERE l_shipmode IN ('MAIL','SHIP')
          AND l_commitdate  < l_receiptdate
          AND l_shipdate    < l_commitdate
          AND l_receiptdate >= 8761
          AND l_receiptdate  < 9126
        GROUP BY l_shipmode
        ORDER BY l_shipmode
    """
    mx_result = mx.sql(q, lineitem=lineitem, orders=orders).compute(device="cpu")
    ref = _duckdb(q, lineitem=lineitem, orders=orders)
    assert mx_result.num_rows > 0, "Q12 SQL returned no rows"
    mx_s  = _sort_table(mx_result, ["l_shipmode"])
    ref_s = _sort_table(ref,       ["l_shipmode"])
    assert mx_s.num_rows == ref_s.num_rows, \
        f"Q12 row count mismatch: {mx_s.num_rows} vs {ref_s.num_rows}"
    for i in range(mx_s.num_rows):
        mx_h  = int(mx_s.column("high_line_count")[i].as_py())
        ref_h = int(ref_s.column("high_line_count")[i].as_py())
        assert abs(mx_h - ref_h) <= 1, f"Q12 high row {i}: {mx_h} vs {ref_h}"
    ok(f"Q12 via SQL (join + isin + CASE WHEN, {mx_s.num_rows} groups, matches DuckDB)")


def test_q14_sql():
    """Q14 via SQL: 2-way join + LIKE CASE WHEN + ratio."""
    part, lineitem = make_q14_tables()
    q = """
        SELECT
          SUM(CASE WHEN p_type LIKE 'PROMO%'
                   THEN l_extendedprice * (1 - l_discount) ELSE 0 END) AS promo_revenue,
          SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        JOIN part ON l_partkey = p_partkey
        WHERE l_shipdate >= 9374 AND l_shipdate < 9404
    """
    mx_result = mx.sql(q, lineitem=lineitem, part=part).compute(device="cpu")
    ref = _duckdb(q, lineitem=lineitem, part=part)

    mx_promo = float(mx_result.column("promo_revenue")[0].as_py())
    mx_total = float(mx_result.column("total_revenue")[0].as_py())
    ref_promo = float(ref.column("promo_revenue")[0].as_py())
    ref_total = float(ref.column("total_revenue")[0].as_py())

    mx_pct  = 100.0 * mx_promo  / mx_total  if mx_total  else 0.0
    ref_pct = 100.0 * ref_promo / ref_total if ref_total else 0.0
    assert abs(mx_pct - ref_pct) < 0.5, \
        f"Q14 promo% mismatch: MX={mx_pct:.2f}% DuckDB={ref_pct:.2f}%"
    ok(f"Q14 via SQL (LIKE CASE WHEN, promo={mx_pct:.2f}% matches DuckDB={ref_pct:.2f}%)")


def test_simple_filter_sql():
    """Simple WHERE + projection via SQL."""
    tbl = pa.table({"a": [1, 2, 3, 4, 5], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result = mx.sql(
        "SELECT a, b FROM t WHERE a > 2", t=tbl
    ).compute(device="cpu")
    assert result.num_rows == 3, f"Expected 3 rows, got {result.num_rows}"
    ok("Simple filter + projection via SQL")


def test_global_sum_sql():
    """Global SUM via SQL (no GROUP BY)."""
    tbl = pa.table({"v": [1.0, 2.0, 3.0, 4.0]})
    result = mx.sql("SELECT SUM(v) AS total FROM t", t=tbl).compute(device="cpu")
    val = float(result.column("total")[0].as_py())
    assert abs(val - 10.0) < 0.01, f"Expected 10.0 got {val}"
    ok("Global SUM via SQL")


def test_order_limit_sql():
    """ORDER BY + LIMIT via SQL."""
    tbl = pa.table({"k": [3, 1, 4, 1, 5, 9, 2], "v": [30, 10, 40, 10, 50, 90, 20]})
    result = mx.sql(
        "SELECT k, v FROM t ORDER BY v LIMIT 3", t=tbl
    ).compute(device="cpu")
    assert result.num_rows == 3, f"Expected 3 rows, got {result.num_rows}"
    ok("ORDER BY + LIMIT via SQL")


# ── runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TESTS = [
        test_simple_filter_sql,
        test_global_sum_sql,
        test_order_limit_sql,
        test_q6_sql,
        test_q1_sql,
        test_q3_sql,
        test_q12_sql,
        test_q14_sql,
    ]

    for t in TESTS:
        try:
            t()
        except Exception as e:
            fail(t.__name__, e)

    print()
    print(f"{'All SQL Frontend tests passed!' if FAIL == 0 else f'{FAIL} test(s) FAILED'}")
    print(f"  Passed: {PASS}  Failed: {FAIL}")
    sys.exit(0 if FAIL == 0 else 1)
