"""DuckDB SQL versions of the 22 queries: the correctness reference AND a scoreboard engine.

Each SQL computes exactly what the shared implementation computes (see spec.DEVIATIONS), on
the same Parquet files, with dates as int32 days and float32 columns.  Sums are done in
DOUBLE from the stored float32 values, so the reference has no accumulation drift.
Comparisons of FLOAT columns against decimal constants cast the constant to FLOAT (that is
what a float32 engine does and matches the decimal semantics of TPC-H for 2-decimal data).
"""
from __future__ import annotations

from . import spec as S
from .readers import table_path

_REV = "l_extendedprice::DOUBLE * (1 - l_discount::DOUBLE)"
_YEAR = "year(DATE '1970-01-01' + {c})"

SQL: dict[int, str] = {
    1: f"""
SELECT l_returnflag, l_linestatus,
  sum(l_quantity::DOUBLE) AS sum_qty, sum(l_extendedprice::DOUBLE) AS sum_base_price,
  sum({_REV}) AS sum_disc_price, sum({_REV} * (1 + l_tax::DOUBLE)) AS sum_charge,
  avg(l_quantity::DOUBLE) AS avg_qty, avg(l_extendedprice::DOUBLE) AS avg_price,
  avg(l_discount::DOUBLE) AS avg_disc, count(l_quantity) AS count_order
FROM lineitem WHERE l_shipdate <= {S.Q1_CUTOFF}
GROUP BY 1, 2 ORDER BY 1, 2""",
    2: f"""
WITH eu AS (SELECT s_suppkey, s_name, s_acctbal, s_address, s_phone, s_comment, n_name FROM supplier
            JOIN nation ON s_nationkey = n_nationkey WHERE n_regionkey = {S.Q2_REGIONKEY}),
tp AS (SELECT p_partkey, p_mfgr FROM part WHERE p_size = 15 AND p_type LIKE '%BRASS%'),
m AS (SELECT ps_partkey, ps_supplycost, s_acctbal, s_name, n_name, s_address, s_phone, s_comment, p_mfgr
      FROM partsupp JOIN eu ON ps_suppkey = s_suppkey JOIN tp ON ps_partkey = p_partkey),
mc AS (SELECT ps_partkey AS pk, min(ps_supplycost) AS mn FROM m GROUP BY 1)
SELECT s_acctbal, s_name, n_name, ps_partkey, p_mfgr, s_address, s_phone, s_comment
FROM m JOIN mc ON ps_partkey = pk AND ps_supplycost = mn
ORDER BY s_acctbal DESC, n_name, s_name, ps_partkey LIMIT 100""",
    3: f"""
SELECT o_orderkey, o_orderdate, o_shippriority, sum({_REV}) AS revenue
FROM orders JOIN customer ON o_custkey = c_custkey JOIN lineitem ON o_orderkey = l_orderkey
WHERE c_mktsegment = 'BUILDING' AND o_orderdate < {S.Q3_DATE} AND l_shipdate > {S.Q3_DATE}
GROUP BY 1, 2, 3 ORDER BY revenue DESC, o_orderdate, o_orderkey LIMIT 10""",
    4: f"""
SELECT o_orderpriority, count(*) AS order_count FROM orders
WHERE o_orderdate >= {S.Q4_LO} AND o_orderdate < {S.Q4_HI}
  AND EXISTS (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
GROUP BY 1 ORDER BY 1""",
    5: f"""
SELECT n_name, sum({_REV}) AS revenue
FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey
     JOIN supplier ON l_suppkey = s_suppkey AND c_nationkey = s_nationkey
     JOIN nation ON s_nationkey = n_nationkey
WHERE n_regionkey = {S.Q5_REGIONKEY} AND o_orderdate >= {S.Q5_LO} AND o_orderdate < {S.Q5_HI}
GROUP BY 1 ORDER BY 2 DESC""",
    6: f"""
SELECT sum(l_extendedprice::DOUBLE * l_discount::DOUBLE) AS revenue FROM lineitem
WHERE l_shipdate >= {S.Q6_LO} AND l_shipdate < {S.Q6_HI}
  AND l_discount >= CAST(0.05 AS FLOAT) AND l_discount <= CAST(0.07 AS FLOAT) AND l_quantity < 24""",
    7: f"""
SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, {_YEAR.format(c='l_shipdate')} AS l_year,
       sum({_REV}) AS revenue
FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey
     JOIN supplier ON l_suppkey = s_suppkey
     JOIN nation n1 ON s_nationkey = n1.n_nationkey JOIN nation n2 ON c_nationkey = n2.n_nationkey
WHERE l_shipdate >= {S.Q7_LO} AND l_shipdate <= {S.Q7_HI}
  AND ((n1.n_name = '{S.Q7_NATIONS[0]}' AND n2.n_name = '{S.Q7_NATIONS[1]}')
    OR (n1.n_name = '{S.Q7_NATIONS[1]}' AND n2.n_name = '{S.Q7_NATIONS[0]}'))
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""",
    8: f"""
WITH v AS (
  SELECT {_YEAR.format(c='o_orderdate')} AS o_year, {_REV} AS vol, s_nationkey
  FROM lineitem JOIN part ON l_partkey = p_partkey JOIN supplier ON l_suppkey = s_suppkey
       JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey
  WHERE p_type = 'ECONOMY ANODIZED STEEL' AND o_orderdate >= {S.Q8_LO} AND o_orderdate <= {S.Q8_HI}
    AND c_nationkey IN (SELECT n_nationkey FROM nation WHERE n_regionkey = {S.Q8_REGIONKEY}))
SELECT o_year, sum(vol) AS total_vol,
       sum(CASE WHEN s_nationkey = {S.Q8_NATIONKEY} THEN vol ELSE 0 END) AS brazil_vol_sum,
       sum(CASE WHEN s_nationkey = {S.Q8_NATIONKEY} THEN vol ELSE 0 END) / sum(vol) AS mkt_share
FROM v GROUP BY 1 ORDER BY 1""",
    9: f"""
SELECT n_name, {_YEAR.format(c='o_orderdate')} AS o_year,
       sum({_REV} - ps_supplycost::DOUBLE * l_quantity::DOUBLE) AS sum_profit
FROM lineitem JOIN part ON l_partkey = p_partkey
     JOIN partsupp ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey
     JOIN supplier ON l_suppkey = s_suppkey JOIN nation ON s_nationkey = n_nationkey
     JOIN orders ON l_orderkey = o_orderkey
WHERE p_name LIKE '%green%' GROUP BY 1, 2 ORDER BY 1, 2 DESC""",
    10: f"""
SELECT o_custkey, n_name, sum({_REV}) AS revenue
FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey
     JOIN nation ON c_nationkey = n_nationkey
WHERE l_returnflag = 'R' AND o_orderdate >= {S.Q10_LO} AND o_orderdate < {S.Q10_HI}
GROUP BY 1, 2 ORDER BY 3 DESC, 1 LIMIT 20""",
    11: f"""
WITH v AS (SELECT ps_partkey, sum(ps_supplycost::DOUBLE * ps_availqty::DOUBLE) AS value FROM partsupp
           WHERE ps_suppkey IN (SELECT s_suppkey FROM supplier WHERE s_nationkey = {S.Q11_NATIONKEY})
           GROUP BY 1)
SELECT ps_partkey, value FROM v WHERE value > (SELECT sum(value) * @FRAC@ FROM v)
ORDER BY value DESC""",
    12: f"""
SELECT l_shipmode,
  sum(CASE WHEN o_orderpriority IN ('1-URGENT', '2-HIGH') THEN 1 ELSE 0 END)::BIGINT AS high_line_count,
  sum(CASE WHEN o_orderpriority IN ('1-URGENT', '2-HIGH') THEN 0 ELSE 1 END)::BIGINT AS low_line_count
FROM lineitem JOIN orders ON l_orderkey = o_orderkey
WHERE l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
  AND l_receiptdate >= {S.Q12_LO} AND l_receiptdate < {S.Q12_HI}
GROUP BY 1 ORDER BY 1""",
    13: """
SELECT c_count, count(*) AS custdist FROM (
  SELECT c_custkey, count(o_orderkey) AS c_count FROM customer LEFT JOIN orders ON c_custkey = o_custkey
  GROUP BY 1) GROUP BY 1 ORDER BY 2 DESC, 1 DESC""",
    14: f"""
SELECT 100.0 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN {_REV} ELSE 0 END) / sum({_REV}) AS promo_revenue
FROM lineitem JOIN part ON l_partkey = p_partkey
WHERE l_shipdate >= {S.Q14_LO} AND l_shipdate < {S.Q14_HI}""",
    15: f"""
WITH r AS (SELECT l_suppkey, sum({_REV}) AS total_revenue FROM lineitem
           WHERE l_shipdate >= {S.Q15_LO} AND l_shipdate < {S.Q15_HI} GROUP BY 1)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM supplier JOIN r ON s_suppkey = l_suppkey
WHERE total_revenue = (SELECT max(total_revenue) FROM r) ORDER BY 1""",
    16: f"""
SELECT p_brand, p_type, p_size, count(DISTINCT ps_suppkey) AS supplier_cnt
FROM partsupp JOIN part ON p_partkey = ps_partkey
WHERE p_brand <> 'Brand#45' AND p_type NOT LIKE 'MEDIUM POLISHED%'
  AND p_size IN ({', '.join(map(str, S.Q16_SIZES))})
  AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Complaints%')
GROUP BY 1, 2, 3 ORDER BY 4 DESC, 1, 2, 3""",
    17: """
WITH tp AS (SELECT p_partkey FROM part WHERE p_brand = 'Brand#23' AND p_container = 'MED BOX'),
li AS (SELECT l_partkey, l_quantity, l_extendedprice FROM lineitem JOIN tp ON l_partkey = p_partkey),
av AS (SELECT l_partkey AS pk, avg(l_quantity::DOUBLE) AS aq FROM li GROUP BY 1)
SELECT sum(l_extendedprice::DOUBLE) / 7.0 AS revenue
FROM li JOIN av ON l_partkey = pk WHERE l_quantity < 0.2 * aq""",
    18: """
WITH big AS (SELECT l_orderkey FROM lineitem GROUP BY 1 HAVING sum(l_quantity::DOUBLE) > 300)
SELECT o_custkey, l_orderkey, sum(l_quantity::DOUBLE) AS sum_qty
FROM lineitem JOIN big USING (l_orderkey) JOIN orders ON l_orderkey = o_orderkey
     JOIN customer ON o_custkey = c_custkey
GROUP BY 1, 2 ORDER BY 3 DESC, 2 LIMIT 100""",
    19: f"""
SELECT sum({_REV}) AS revenue FROM lineitem JOIN part ON l_partkey = p_partkey
WHERE ((p_brand = 'Brand#12' AND p_container IN ({', '.join(repr(c) for c in S.CONTAINERS_SM)})
       AND l_quantity >= 1 AND l_quantity <= 11 AND p_size BETWEEN 1 AND 5)
   OR (p_brand = 'Brand#23' AND p_container IN ({', '.join(repr(c) for c in S.CONTAINERS_MD)})
       AND l_quantity >= 10 AND l_quantity <= 20 AND p_size BETWEEN 1 AND 10)
   OR (p_brand = 'Brand#34' AND p_container IN ({', '.join(repr(c) for c in S.CONTAINERS_LG)})
       AND l_quantity >= 20 AND l_quantity <= 30 AND p_size BETWEEN 1 AND 15))
  AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON'""",
    20: f"""
WITH forest AS (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%'),
q AS (SELECT l_partkey, l_suppkey, sum(l_quantity::DOUBLE) AS sq FROM lineitem
      WHERE l_shipdate >= {S.Q20_LO} AND l_shipdate < {S.Q20_HI}
        AND l_partkey IN (SELECT p_partkey FROM forest) GROUP BY 1, 2),
ex AS (SELECT DISTINCT ps_suppkey FROM partsupp JOIN q ON ps_partkey = l_partkey AND ps_suppkey = l_suppkey
       WHERE ps_availqty::DOUBLE > 0.5 * sq)
SELECT s_name, s_address FROM supplier
WHERE s_nationkey = {S.Q20_NATIONKEY} AND s_suppkey IN (SELECT ps_suppkey FROM ex) ORDER BY s_name""",
    21: f"""
WITH late AS (SELECT l_orderkey, l_suppkey FROM lineitem WHERE l_receiptdate > l_commitdate),
single AS (SELECT l_orderkey FROM late GROUP BY 1 HAVING count(DISTINCT l_suppkey) = 1),
multi AS (SELECT l_orderkey FROM lineitem GROUP BY 1 HAVING count(DISTINCT l_suppkey) >= 2)
SELECT l_suppkey, count(*) AS numwait
FROM late JOIN orders ON l_orderkey = o_orderkey JOIN single USING (l_orderkey)
     JOIN multi USING (l_orderkey)
WHERE o_orderstatus = 'F'
  AND l_suppkey IN (SELECT s_suppkey FROM supplier WHERE s_nationkey = {S.Q21_NATIONKEY})
GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 100""",
    22: f"""
WITH t AS (SELECT c_custkey, c_acctbal, substr(c_phone, 1, 2) AS cc FROM customer
           WHERE substr(c_phone, 1, 2) IN ({', '.join(repr(c) for c in S.Q22_CODES)})),
a AS (SELECT avg(c_acctbal::DOUBLE) AS ab FROM t WHERE c_acctbal > 0)
SELECT cc AS cntrycode, count(*) AS numcust, sum(c_acctbal::DOUBLE) AS totacctbal FROM t
WHERE c_acctbal::DOUBLE > (SELECT ab FROM a)
  AND NOT EXISTS (SELECT 1 FROM orders WHERE o_custkey = c_custkey)
GROUP BY 1 ORDER BY 1""",
}


def run(q: int, data_dir: str, device=None):
    """Timed unit: Parquet read + query + materialized Arrow result."""
    import duckdb
    S.set_scale_from(data_dir)
    con = duckdb.connect()
    for name in S.TABLES[q]:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{table_path(data_dir, name)}')")
    return _fetch(con.execute(SQL[q].replace('@FRAC@', repr(S.q11_fraction()))))


def load_mem(q: int, data_dir: str):
    """Engine-only mode, untimed: the columns the query uses, as DuckDB in-memory tables
    (its native storage, so the query runs on loaded data, not on a Parquet scan)."""
    import duckdb
    S.set_scale_from(data_dir)
    con = duckdb.connect()
    for name, cols in S.TABLES[q].items():
        con.execute(f"CREATE TABLE {name} AS SELECT {', '.join(cols)} "
                    f"FROM read_parquet('{table_path(data_dir, name)}')")
    return con


def run_mem(q: int, con, device=None):
    """Engine-only timed unit: query + materialized Arrow result on the loaded tables."""
    return _fetch(con.execute(SQL[q].replace('@FRAC@', repr(S.q11_fraction()))))


def _fetch(res):
    out = res.fetch_arrow_table() if hasattr(res, "fetch_arrow_table") else res.arrow()
    return out.read_all() if hasattr(out, "read_all") else out


def to_arrow(result):
    return result
