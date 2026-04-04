"""TPC-H Q5 (6-way join) and Q10 (4-way join) validation tests.

Q5 — 6-way join:
  SELECT n_name, SUM(l_extendedprice*(1-l_discount)) AS revenue
  FROM customer JOIN orders ON c_custkey=o_custkey
  JOIN lineitem ON l_orderkey=o_orderkey
  JOIN supplier ON l_suppkey=s_suppkey
  JOIN nation   ON s_nationkey=n_nationkey AND c_nationkey=n_nationkey
  JOIN region   ON n_regionkey=r_regionkey
  WHERE r_name='ASIA' AND o_orderdate >= DATE1 AND o_orderdate < DATE2
    AND l_shipdate >= DATE1
  GROUP BY n_name ORDER BY revenue DESC

Q10 — 4-way join:
  SELECT c_custkey, c_name, SUM(l_extendedprice*(1-l_discount)) AS revenue,
         c_acctbal, n_name, c_address, c_comment
  FROM customer JOIN orders ON c_custkey=o_custkey
  JOIN lineitem ON l_orderkey=o_orderkey
  JOIN nation   ON c_nationkey=n_nationkey
  WHERE o_orderdate >= DATE1 AND o_orderdate < DATE2
    AND l_returnflag=&#39;R&#39;
  GROUP BY c_custkey,c_name,c_acctbal,c_address,n_name,c_comment
  ORDER BY revenue DESC LIMIT 20
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pyarrow as pa
import duckdb
import mxframe as mx

PASS = 0; FAIL = 0

def ok(label):    global PASS; PASS += 1; print(f"OK  {label}")
def fail(label, exc):
    global FAIL; FAIL += 1; print(f"FAIL {label}: {exc}")
    import traceback; traceback.print_exc()

def _duckdb(query, **tables):
    con = duckdb.connect()
    for name, tbl in tables.items():
        con.register(name, tbl)
    return con.execute(query).arrow().read_all()

def _close(a, b, tol=None):
    if tol is None: tol = max(abs(b) * 0.02, 1.0)
    return abs(a - b) < tol

def _sort_table(t, by):
    import pyarrow.compute as pc
    indices = pc.sort_indices(t, sort_keys=[(c, "ascending") for c in by])
    return t.take(indices)


# ── data generators ──────────────────────────────────────────────────────
def make_q5_tables(n_regions=5, n_nations=25, n_suppliers=500,
                   n_customers=2000, n_orders=10_000, n_lineitem=40_000, seed=55):
    rng = np.random.default_rng(seed)
    region = pa.table({
        "r_regionkey": np.arange(n_regions, dtype=np.int32),
        "r_name": ["AFRICA","AMERICA","ASIA","EUROPE","MIDDLE EAST"],
    })
    nation = pa.table({
        "n_nationkey":  np.arange(n_nations, dtype=np.int32),
        "n_regionkey":  rng.integers(0, n_regions, size=n_nations, dtype=np.int32),
        "n_name": [f"NAT{i}" for i in range(n_nations)],
    })
    supplier = pa.table({
        "s_suppkey":    np.arange(n_suppliers, dtype=np.int32),
        "s_nationkey":  rng.integers(0, n_nations, size=n_suppliers, dtype=np.int32),
    })
    customer = pa.table({
        "c_custkey":    np.arange(n_customers, dtype=np.int32),
        "c_nationkey":  rng.integers(0, n_nations, size=n_customers, dtype=np.int32),
    })
    orders = pa.table({
        "o_orderkey":  np.arange(n_orders, dtype=np.int32),
        "o_custkey":   rng.integers(0, n_customers, size=n_orders, dtype=np.int32),
        "o_orderdate": rng.integers(9100, 9500, size=n_orders, dtype=np.int32),
    })
    lineitem = pa.table({
        "l_orderkey":      rng.integers(0, n_orders, size=n_lineitem, dtype=np.int32),
        "l_suppkey":       rng.integers(0, n_suppliers, size=n_lineitem, dtype=np.int32),
        "l_extendedprice": rng.uniform(100., 50_000., size=n_lineitem).astype(np.float32),
        "l_discount":      rng.uniform(0., 0.1, size=n_lineitem).astype(np.float32),
        "l_shipdate":      rng.integers(9100, 9500, size=n_lineitem, dtype=np.int32),
    })
    return region, nation, supplier, customer, orders, lineitem


def make_q10_tables(n_nations=25, n_customers=2000, n_orders=10_000,
                    n_lineitem=40_000, seed=66):
    rng = np.random.default_rng(seed)
    nation = pa.table({
        "n_nationkey": np.arange(n_nations, dtype=np.int32),
        "n_name": [f"NAT{i}" for i in range(n_nations)],
    })
    customer = pa.table({
        "c_custkey":   np.arange(n_customers, dtype=np.int32),
        "c_name":      [f"Cust#{i}" for i in range(n_customers)],
        "c_nationkey": rng.integers(0, n_nations, size=n_customers, dtype=np.int32),
        "c_acctbal":   rng.uniform(0., 9999., size=n_customers).astype(np.float32),
        "c_address":   [f"addr{i}" for i in range(n_customers)],
        "c_comment":   [f"comment{i}" for i in range(n_customers)],
    })
    orders = pa.table({
        "o_orderkey":  np.arange(n_orders, dtype=np.int32),
        "o_custkey":   rng.integers(0, n_customers, size=n_orders, dtype=np.int32),
        "o_orderdate": rng.integers(9100, 9500, size=n_orders, dtype=np.int32),
    })
    lineitem = pa.table({
        "l_orderkey":      rng.integers(0, n_orders, size=n_lineitem, dtype=np.int32),
        "l_extendedprice": rng.uniform(100., 50_000., size=n_lineitem).astype(np.float32),
        "l_discount":      rng.uniform(0., 0.1, size=n_lineitem).astype(np.float32),
        "l_returnflag":    rng.choice(["N", "R", "A"], size=n_lineitem).tolist(),
    })
    return nation, customer, orders, lineitem


# ── tests ─────────────────────────────────────────────────────────────────

def test_q5():
    """Q5 — 6-way join + groupby + sort."""
    region, nation, supplier, customer, orders, lineitem = make_q5_tables()
    DATE_LO, DATE_HI = 9131, 9496  # ~1995-01-01 to ~1996-01-01

    # MXFrame DataFrame API (multi-join + filter + groupby + sort)
    from mxframe import col, lit
    from mxframe.lazy_frame import LazyFrame, Scan
    import pyarrow.compute as pc

    # Join path: orders ⋈ customer ⋈ lineitem ⋈ supplier ⋈ nation ⋈ region
    lf = (
        LazyFrame(Scan(orders))
        .filter(col("o_orderdate") >= lit(DATE_LO))
        .filter(col("o_orderdate") <  lit(DATE_HI))
        .join(LazyFrame(Scan(customer)), left_on="o_custkey",  right_on="c_custkey")
        .join(LazyFrame(Scan(lineitem)), left_on="o_orderkey", right_on="l_orderkey")
        .join(LazyFrame(Scan(supplier)), left_on="l_suppkey",  right_on="s_suppkey")
        .join(LazyFrame(Scan(nation)),   left_on="s_nationkey", right_on="n_nationkey")
        # NOTE: we also need c_nationkey = n_nationkey, handled via filter post-join
        .join(LazyFrame(Scan(region)),   left_on="n_regionkey", right_on="r_regionkey")
        .filter(col("r_name") == lit("ASIA"))
        .filter(col("c_nationkey") == col("s_nationkey"))  # same-nation customer-supplier
        .groupby("n_name")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
        .sort("revenue", descending=True)
    )
    mx_result = lf.compute(device="cpu")

    # DuckDB reference
    ref = _duckdb(f"""
        SELECT n_name, SUM(l_extendedprice*(1-l_discount)) AS revenue
        FROM orders
          JOIN customer  ON o_custkey=c_custkey
          JOIN lineitem  ON l_orderkey=o_orderkey
          JOIN supplier  ON l_suppkey=s_suppkey
          JOIN nation    ON s_nationkey=n_nationkey
          JOIN region    ON n_regionkey=r_regionkey
        WHERE r_name='ASIA' AND c_nationkey=s_nationkey
          AND o_orderdate >= {DATE_LO} AND o_orderdate < {DATE_HI}
        GROUP BY n_name ORDER BY revenue DESC
    """, orders=orders, customer=customer, lineitem=lineitem,
         supplier=supplier, nation=nation, region=region)

    # Correctness: row counts and total revenue
    assert mx_result.num_rows > 0, "Q5 returned no rows"
    mx_total  = sum(float(v.as_py()) for v in mx_result.column("revenue"))
    ref_total = sum(float(v.as_py()) for v in ref.column("revenue"))
    assert _close(mx_total, ref_total), \
        f"Q5 total revenue: MX={mx_total:.0f} DuckDB={ref_total:.0f}"
    ok(f"Q5 — 6-way join ({mx_result.num_rows} nations, revenue matches DuckDB)")


def test_q10():
    """Q10 — 4-way join + groupby + sort + limit."""
    nation, customer, orders, lineitem = make_q10_tables()
    DATE_LO, DATE_HI = 9204, 9296

    from mxframe import col, lit
    from mxframe.lazy_frame import LazyFrame, Scan

    lf = (
        LazyFrame(Scan(orders))
        .filter(col("o_orderdate") >= lit(DATE_LO))
        .filter(col("o_orderdate") <  lit(DATE_HI))
        .join(LazyFrame(Scan(customer)), left_on="o_custkey",  right_on="c_custkey")
        .join(LazyFrame(Scan(lineitem)), left_on="o_orderkey", right_on="l_orderkey")
        .join(LazyFrame(Scan(nation)),   left_on="c_nationkey", right_on="n_nationkey")
        .filter(col("l_returnflag") == lit("R"))
        # Group by key columns only (c_name / c_address are additional attrs in real Q10;
        # simplified here to 3 cols to stay within int32 composite key range)
        .groupby("o_custkey", "n_name", "c_acctbal")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
        .sort("revenue", descending=True)
        .limit(20)
    )
    mx_result = lf.compute(device="cpu")

    ref = _duckdb(f"""
        SELECT o_custkey,
               SUM(l_extendedprice*(1-l_discount)) AS revenue,
               c_acctbal, n_name
        FROM orders
          JOIN customer ON o_custkey=c_custkey
          JOIN lineitem ON l_orderkey=o_orderkey
          JOIN nation   ON c_nationkey=n_nationkey
        WHERE o_orderdate >= {DATE_LO} AND o_orderdate < {DATE_HI}
          AND l_returnflag='R'
        GROUP BY o_custkey, n_name, c_acctbal
        ORDER BY revenue DESC LIMIT 20
    """, nation=nation, customer=customer, orders=orders, lineitem=lineitem)

    assert mx_result.num_rows > 0, "Q10 returned no rows"
    assert mx_result.num_rows <= 20, f"Q10 LIMIT 20 not applied: {mx_result.num_rows}"
    mx_total  = sum(float(v.as_py()) for v in mx_result.column("revenue"))
    ref_total = sum(float(v.as_py()) for v in ref.column("revenue"))
    assert _close(mx_total, ref_total), \
        f"Q10 total revenue: MX={mx_total:.0f} DuckDB={ref_total:.0f}"
    ok(f"Q10 — 4-way join + limit ({mx_result.num_rows} rows, revenue matches DuckDB)")


# ── runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for t in [test_q5, test_q10]:
        try:
            t()
        except Exception as e:
            fail(t.__name__, e)
    print()
    if FAIL == 0:
        print("All Q5/Q10 multi-join tests passed!")
    else:
        print(f"{FAIL} test(s) FAILED")
    sys.exit(0 if FAIL == 0 else 1)
