"""TPC-H reference answers from DuckDB (F-011).

The omxcore benchmark queries read Parquet written by ``scripts/gen_tpch_parquet.py``:
dates are int32 day numbers (days since 1970-01-01) and DECIMAL columns are float32.
DuckDB reads the very same files through views that turn them back into what TPC-H
says they are (DATE and DECIMAL(15,2)), so the SQL below is the official TPC-H text
(with the parameter values the benchmark uses) and the answers are exact.

Where the reference differs from the official TPC-H text (all on purpose, and the same query):
  Q1/Q3/Q10/Q18/Q21  column aliases follow the benchmark functions (o_orderkey for l_orderkey, ...)
  Q8   also returns the two sums the share is made of (total_vol, brazil_vol_sum)
  Q14  also returns the two sums (promo_revenue, total_revenue) besides the percentage (promo_pct)
  Q17  sum(...) / 7.0 as avg_yearly, unchanged; Q15 reads the revenue view as a CTE
  Everything else (parameters, joins, predicates, GROUP BY / ORDER BY / LIMIT) is the official text;
  Q11's FRACTION is 0.0001 / SF (SF = suppliers / 10,000) as in the specification.

Each entry: (sql, meta) where meta has
  int_cols  columns of the result that must be integers (counts), never floats
  limit     (column, descending) when the query has a LIMIT; rows tied with the last
            kept row on that column may legitimately differ, so they are dropped from
            both sides before comparing
"""
from __future__ import annotations

import pyarrow as pa

DATE_COLUMNS = {"l_shipdate", "l_commitdate", "l_receiptdate", "o_orderdate"}
TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]


def connect(data_dir):
    """A DuckDB connection with one typed view per TPC-H table over ``data_dir``."""
    import duckdb
    con = duckdb.connect()
    for name in TABLES:
        path = f"{data_dir}/{name}.parquet"
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        exprs = []
        for cname, ctype, *_ in cols:
            if cname in DATE_COLUMNS:
                exprs.append(f"DATE '1970-01-01' + CAST({cname} AS INTEGER) AS {cname}")
            elif ctype in ("FLOAT", "DOUBLE"):
                exprs.append(f"CAST({cname} AS DECIMAL(15,2)) AS {cname}")
            else:
                exprs.append(cname)
        con.execute(f"CREATE VIEW {name} AS SELECT {', '.join(exprs)} FROM read_parquet('{path}')")
    return con


def run(con, q: int) -> pa.Table:
    """Run reference query ``q``; DECIMAL results come back as float64, DATE as int32 day numbers."""
    sql, _ = REFERENCE[q]
    t = con.execute(sql).to_arrow_table()
    for i, typ in enumerate(t.schema.types):
        if pa.types.is_decimal(typ):
            t = t.set_column(i, t.schema.field(i).name, t.column(i).cast(pa.float64()))
        elif pa.types.is_date(typ):         # back to the day numbers the Parquet files use
            t = t.set_column(i, t.schema.field(i).name, t.column(i).cast(pa.int32()))
    return t


# The ORDER BY columns of each query, in order (every key of the SQL, so the order is fully
# determined up to rows equal on all of them).  The result must come back in this order.
ORDER_BY = {
    1: ["l_returnflag", "l_linestatus"], 2: ["s_acctbal", "n_name", "s_name", "p_partkey"],
    3: ["revenue", "o_orderdate"], 4: ["o_orderpriority"], 5: ["revenue"],
    7: ["supp_nation", "cust_nation", "l_year"], 8: ["o_year"], 9: ["n_name", "o_year"],
    10: ["revenue"], 11: ["value"], 12: ["l_shipmode"], 13: ["custdist", "c_count"],
    15: ["s_suppkey"], 16: ["supplier_cnt", "p_brand", "p_type", "p_size"],
    18: ["o_totalprice", "o_orderdate"], 20: ["s_name"], 21: ["numwait", "s_name"],
    22: ["cntrycode"],
}

REVENUE = "l_extendedprice * (1 - l_discount)"

REFERENCE = {
    1: (f"""
        SELECT l_returnflag, l_linestatus,
               sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
               sum({REVENUE}) AS sum_disc_price,
               sum({REVENUE} * (1 + l_tax)) AS sum_charge,
               avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price,
               avg(l_discount) AS avg_disc, count(*) AS count_order
        FROM lineitem WHERE l_shipdate <= DATE '1998-09-02'
        GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus""",
        dict(int_cols=["count_order"])),
    2: ("""
        SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
        FROM part, supplier, partsupp, nation, region
        WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 15
          AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
          AND r_name = 'EUROPE'
          AND ps_supplycost = (SELECT min(ps_supplycost) FROM partsupp, supplier, nation, region
                               WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
                                 AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
                                 AND r_name = 'EUROPE')
        ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100""",
        dict(limit=("s_acctbal", True))),
    3: (f"""
        SELECT l_orderkey AS o_orderkey, o_orderdate, o_shippriority, sum({REVENUE}) AS revenue
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey
          AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate LIMIT 10""",
        dict(limit=("revenue", True))),
    4: ("""
        SELECT o_orderpriority, count(*) AS order_count FROM orders
        WHERE o_orderdate >= DATE '1993-07-01' AND o_orderdate < DATE '1993-10-01'
          AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
        GROUP BY o_orderpriority ORDER BY o_orderpriority""",
        dict(int_cols=["order_count"])),
    5: (f"""
        SELECT n_name, sum({REVENUE}) AS revenue
        FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey
          AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
          AND r_name = 'ASIA' AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01'
        GROUP BY n_name ORDER BY revenue DESC""", {}),
    6: ("""
        SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24""", {}),
    7: (f"""
        SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM (
          SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
                 CAST(extract(year FROM l_shipdate) AS INTEGER) AS l_year, {REVENUE} AS volume
          FROM supplier, lineitem, orders, customer, nation n1, nation n2
          WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey
            AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey
            AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
              OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
            AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31') AS shipping
        GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year""", {}),
    8: (f"""
        SELECT o_year, sum(volume) AS total_vol,
               sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) AS brazil_vol_sum,
               sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / sum(volume) AS mkt_share
        FROM (SELECT CAST(extract(year FROM o_orderdate) AS INTEGER) AS o_year,
                     {REVENUE} AS volume, n2.n_name AS nation
              FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
              WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey
                AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey
                AND n1.n_regionkey = r_regionkey AND r_name = 'AMERICA'
                AND s_nationkey = n2.n_nationkey
                AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                AND p_type = 'ECONOMY ANODIZED STEEL') AS all_nations
        GROUP BY o_year ORDER BY o_year""", {}),
    9: ("""
        SELECT nation AS n_name, o_year, sum(amount) AS sum_profit FROM (
          SELECT n_name AS nation, CAST(extract(year FROM o_orderdate) AS INTEGER) AS o_year,
                 l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
          FROM part, supplier, lineitem, partsupp, orders, nation
          WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey
            AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey
            AND p_name LIKE '%green%') AS profit
        GROUP BY nation, o_year ORDER BY nation, o_year DESC""", {}),
    10: (f"""
        SELECT c_custkey AS o_custkey, c_name, sum({REVENUE}) AS revenue, c_acctbal, n_name,
               c_address, c_phone, c_comment
        FROM customer, orders, lineitem, nation
        WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
          AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
          AND l_returnflag = 'R' AND c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
        ORDER BY revenue DESC LIMIT 20""",
        dict(limit=("revenue", True))),
    11: ("""
        SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * 0.0001 / ((SELECT count(*) FROM supplier) / 10000.0)
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY')
        ORDER BY value DESC""", {}),
    12: ("""
        SELECT l_shipmode,
               CAST(sum(CASE WHEN o_orderpriority IN ('1-URGENT', '2-HIGH') THEN 1 ELSE 0 END) AS BIGINT) AS high_line_count,
               CAST(sum(CASE WHEN o_orderpriority NOT IN ('1-URGENT', '2-HIGH') THEN 1 ELSE 0 END) AS BIGINT) AS low_line_count
        FROM orders, lineitem
        WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP')
          AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
          AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1995-01-01'
        GROUP BY l_shipmode ORDER BY l_shipmode""", {}),
    13: ("""
        SELECT c_count, count(*) AS custdist FROM (
          SELECT c_custkey, count(o_orderkey) AS c_count
          FROM customer LEFT OUTER JOIN orders
            ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
          GROUP BY c_custkey) AS c_orders
        GROUP BY c_count ORDER BY custdist DESC, c_count DESC""",
        dict(int_cols=["c_count", "custdist"])),
    14: (f"""
        SELECT sum(CASE WHEN p_type LIKE 'PROMO%' THEN {REVENUE} ELSE 0 END) AS promo_revenue,
               sum({REVENUE}) AS total_revenue,
               100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN {REVENUE} ELSE 0 END) / sum({REVENUE}) AS promo_pct
        FROM lineitem, part
        WHERE l_partkey = p_partkey AND l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01'""", {}),
    15: (f"""
        WITH revenue0 AS (
          SELECT l_suppkey AS supplier_no, sum({REVENUE}) AS total_revenue FROM lineitem
          WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
          GROUP BY l_suppkey)
        SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
        FROM supplier, revenue0
        WHERE s_suppkey = supplier_no AND total_revenue = (SELECT max(total_revenue) FROM revenue0)
        ORDER BY s_suppkey""", {}),
    16: ("""
        SELECT p_brand, p_type, p_size, count(DISTINCT ps_suppkey) AS supplier_cnt
        FROM partsupp, part
        WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45' AND p_type NOT LIKE 'MEDIUM POLISHED%'
          AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
          AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')
        GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_brand, p_type, p_size""",
        dict(int_cols=["supplier_cnt"])),
    17: ("""
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part
        WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)""", {}),
    18: ("""
        SELECT c_name, c_custkey AS o_custkey, o_orderkey AS l_orderkey, o_orderdate, o_totalprice,
               sum(l_quantity) AS sum_qty
        FROM customer, orders, lineitem
        WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300)
          AND c_custkey = o_custkey AND o_orderkey = l_orderkey
        GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
        ORDER BY o_totalprice DESC, o_orderdate LIMIT 100""",
        dict(limit=("o_totalprice", True))),
    19: (f"""
        SELECT sum({REVENUE}) AS revenue FROM lineitem, part
        WHERE p_partkey = l_partkey AND l_shipmode IN ('AIR', 'AIR REG')
          AND l_shipinstruct = 'DELIVER IN PERSON'
          AND ((p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                AND l_quantity >= 1 AND l_quantity <= 11 AND p_size BETWEEN 1 AND 5)
            OR (p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                AND l_quantity >= 10 AND l_quantity <= 20 AND p_size BETWEEN 1 AND 10)
            OR (p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                AND l_quantity >= 20 AND l_quantity <= 30 AND p_size BETWEEN 1 AND 15))""", {}),
    20: ("""
        SELECT s_name, s_address FROM supplier, nation
        WHERE s_suppkey IN (
            SELECT ps_suppkey FROM partsupp
            WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
              AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem
                                 WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
                                   AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'))
          AND s_nationkey = n_nationkey AND n_name = 'CANADA'
        ORDER BY s_name""", {}),
    21: ("""
        SELECT s_name, count(*) AS numwait
        FROM supplier, lineitem l1, orders, nation
        WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F'
          AND l1.l_receiptdate > l1.l_commitdate
          AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
          AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey
                            AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate)
          AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA'
        GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100""",
        dict(int_cols=["numwait"], limit=("numwait", True))),
    22: ("""
        SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal FROM (
          SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal FROM customer
          WHERE substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
            AND c_acctbal > (SELECT avg(c_acctbal) FROM customer WHERE c_acctbal > 0.00
                             AND substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17'))
            AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)) AS custsale
        GROUP BY cntrycode ORDER BY cntrycode""",
        dict(int_cols=["numcust"])),
}
