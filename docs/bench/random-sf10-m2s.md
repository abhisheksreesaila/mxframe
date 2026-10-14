# Random queries on TPC-H SF10: MXFrame vs Polars vs DuckDB

50 seeded random queries (seeds 0..49, `scripts/bench_random/`), engine-only: each (query, engine) in a fresh process, columns loaded untimed, the first run timed (query + materialized result). DuckDB's result is the reference.

| shape | queries | MXFrame correct | faster than Polars | faster than DuckDB | Polars / MXFrame (geo. mean) | DuckDB / MXFrame (geo. mean) | MXFrame: no native decline logged |
|---|---:|---:|---:|---:|---:|---:|---:|
| agg | 13 | 13/13 | 12/13 | 11/13 | 6.96x | 1.84x | 13/13 |
| distinct | 5 | 5/5 | 4/5 | 3/5 | 1.99x | 0.89x | 4/5 |
| global | 4 | 4/4 | 3/4 | 4/4 | 1.77x | 1.26x | 4/4 |
| join | 16 | 16/16 | 16/16 | 7/16 | 4.32x | 0.96x | 16/16 |
| select_distinct | 6 | 6/6 | 2/6 | 0/6 | 0.92x | 0.62x | 6/6 |
| topn | 6 | 6/6 | 6/6 | 2/6 | 8.28x | 0.90x | 6/6 |
| all | 50 | 50/50 | 43/50 | 27/50 | 3.78x | 1.09x | 49/50 |

Total time over queries every engine ran: MXFrame 5093 ms, Polars 35459 ms, DuckDB 5571 ms. Geo. mean ratio > 1: MXFrame faster.

## MXFrame results that differ from DuckDB or failed (0)


## Polars results that differ from DuckDB or failed (0)


## Where MXFrame is slowest relative to the faster of Polars / DuckDB

| query | shape | MXFrame ms | Polars ms | DuckDB ms | fell back to Python for |
|---|---|---:|---:|---:|---|
| q14 | agg | 201.6 | 212.6 | 49.0 |  |
| q12 | distinct | 24.1 | 22.6 | 8.6 | count distinct: empty input; empty input |
| q48 | select_distinct | 26.4 | 10.4 | 11.9 |  |
| q34 | select_distinct | 74.7 | 35.9 | 35.7 |  |
| q30 | select_distinct | 38.4 | 107.1 | 20.5 |  |
| q23 | join | 209.6 | 1773.8 | 118.5 |  |
| q9 | distinct | 75.1 | 109.3 | 44.7 |  |
| q17 | select_distinct | 6.3 | 4.0 | 5.5 |  |
| q33 | join | 105.7 | 360.7 | 68.7 |  |
| q44 | topn | 63.5 | 330.6 | 43.6 |  |
| q35 | select_distinct | 23.5 | 16.6 | 16.8 |  |
| q5 | join | 76.0 | 398.5 | 60.2 |  |

## Why parts of MXFrame queries ran in Python (count of queries)

- 1: count distinct: empty input
- 1: empty input

## Queries

- q0 (topn): `SELECT c_custkey, c_acctbal FROM customer WHERE (c_mktsegment IN ('BUILDING', 'FURNITURE', 'MACHINERY')) AND (c_acctbal < 3166.7) ORDER BY c_acctbal ASC, c_custkey ASC LIMIT 200`
- q1 (join): `SELECT l_linestatus, SUM(l_extendedprice * (1 - l_discount)) AS a0, AVG(l_quantity) AS a1, MAX(l_extendedprice * (1 - l_discount)) AS a2 FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN supplier ON l_suppkey = s_suppkey WHERE (l_extendedprice < 10826.64) AND (o_orderpriority IN ('4-NOT SPECIFIED', '5-LOW')) AND (l_shipinstruct = 'TAKE BACK RETURN') GROUP BY l_linestatus ORDER BY a0 DESC, l_linestatus ASC`
- q2 (agg): `SELECT l_linestatus, l_shipmode, AVG(l_extendedprice) AS a0, AVG(l_extendedprice) AS a1, COUNT(l_tax) AS a2 FROM lineitem WHERE TRUE GROUP BY l_linestatus, l_shipmode`
- q3 (join): `SELECT s_nationkey, p_size, SUM(p_size) AS a0, AVG(l_tax) AS a1 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN part ON l_partkey = p_partkey WHERE TRUE GROUP BY s_nationkey, p_size`
- q4 (join): `SELECT o_orderpriority, o_orderstatus, SUM(o_totalprice) AS a0, COUNT(c_acctbal) AS a1, MIN(o_totalprice) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey WHERE (o_totalprice < 144409.04) AND (o_orderdate >= 8952 AND o_orderdate <= 9240) AND (c_acctbal < 6265.79) GROUP BY o_orderpriority, o_orderstatus HAVING COUNT(c_acctbal) > 10`
- q5 (join): `SELECT p_container, SUM(s_acctbal) AS a0, AVG(ps_availqty) AS a1, AVG(ps_supplycost) AS a2, SUM(ps_supplycost) AS a3 FROM partsupp JOIN part ON ps_partkey = p_partkey JOIN supplier ON ps_suppkey = s_suppkey JOIN nation ON s_nationkey = n_nationkey WHERE TRUE GROUP BY p_container ORDER BY a0 DESC, p_container ASC LIMIT 10`
- q6 (agg): `SELECT p_mfgr, p_size, SUM(p_size) AS a0 FROM part WHERE (p_size < 43) AND (p_mfgr IN ('Manufacturer#1', 'Manufacturer#2', 'Manufacturer#3', 'Manufacturer#5')) GROUP BY p_mfgr, p_size`
- q7 (global): `SELECT MAX(l_extendedprice * (1 - l_discount)) AS a0 FROM lineitem WHERE (l_quantity < 24.0) AND (l_extendedprice >= 5087.0 AND l_extendedprice <= 19458.72) AND (l_returnflag <> 'A')`
- q8 (join): `SELECT p_type, MAX(ps_availqty) AS a0, MAX(s_acctbal) AS a1 FROM partsupp JOIN part ON ps_partkey = p_partkey JOIN supplier ON ps_suppkey = s_suppkey WHERE TRUE GROUP BY p_type ORDER BY a0 DESC, p_type ASC`
- q9 (distinct): `SELECT ps_partkey, COUNT(DISTINCT p_container) AS n FROM partsupp JOIN part ON ps_partkey = p_partkey WHERE (p_mfgr = 'Manufacturer#3') GROUP BY ps_partkey ORDER BY n DESC, ps_partkey ASC`
- q10 (agg): `SELECT c_mktsegment, c_nationkey, SUM(c_acctbal) AS a0, SUM(c_acctbal) AS a1, MAX(c_acctbal) AS a2 FROM customer WHERE (c_acctbal >= 5817.34) AND (c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY')) GROUP BY c_mktsegment, c_nationkey ORDER BY a0 DESC, c_mktsegment ASC, c_nationkey ASC LIMIT 10`
- q11 (distinct): `SELECT p_partkey, COUNT(DISTINCT p_size) AS n FROM part WHERE (p_mfgr IN ('Manufacturer#1', 'Manufacturer#2', 'Manufacturer#4', 'Manufacturer#5')) AND (p_brand IN ('Brand#13', 'Brand#43')) AND (p_size >= 3 AND p_size <= 13) GROUP BY p_partkey ORDER BY n DESC, p_partkey ASC LIMIT 100`
- q12 (distinct): `SELECT c_nationkey, COUNT(DISTINCT o_orderpriority) AS n FROM orders JOIN customer ON o_custkey = c_custkey WHERE (o_orderdate >= 10008) AND (o_orderpriority IN ('1-URGENT', '2-HIGH', '3-MEDIUM', '5-LOW')) AND (o_orderstatus <> 'O') GROUP BY c_nationkey HAVING COUNT(DISTINCT o_orderpriority) > 1 ORDER BY n DESC, c_nationkey ASC`
- q13 (join): `SELECT o_orderstatus, COUNT(c_acctbal) AS a0, SUM(c_acctbal) AS a1 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (n_name <> 'RUSSIA') GROUP BY o_orderstatus HAVING COUNT(c_acctbal) > 20`
- q14 (agg): `SELECT o_custkey, o_orderpriority, o_orderstatus, SUM(o_totalprice) AS a0, MIN(o_totalprice) AS a1, MIN(o_totalprice) AS a2 FROM orders WHERE (o_orderpriority = '4-NOT SPECIFIED') AND (o_totalprice >= 166120.84) GROUP BY o_custkey, o_orderpriority, o_orderstatus ORDER BY a0 DESC, o_custkey ASC, o_orderpriority ASC, o_orderstatus ASC LIMIT 10`
- q15 (join): `SELECT o_orderstatus, l_shipinstruct, l_linestatus, MAX(o_totalprice) AS a0, MAX(l_discount) AS a1, SUM(p_size) AS a2 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN orders ON l_orderkey = o_orderkey JOIN part ON l_partkey = p_partkey WHERE (l_quantity < 43.0) GROUP BY o_orderstatus, l_shipinstruct, l_linestatus ORDER BY a0 DESC, o_orderstatus ASC, l_shipinstruct ASC, l_linestatus ASC LIMIT 100`
- q16 (global): `SELECT MIN(p_size) AS a0, COUNT(p_size) AS a1, SUM(p_size) AS a2 FROM part WHERE (p_container IN ('MED BAG', 'SM CAN')) AND (p_type <> 'MEDIUM BRUSHED COPPER') AND (p_size < 19)`
- q17 (select_distinct): `SELECT DISTINCT n_name, c_mktsegment FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE (c_acctbal < -784.52) AND (c_mktsegment <> 'HOUSEHOLD')`
- q18 (join): `SELECT p_brand, l_shipinstruct, COUNT(l_quantity) AS a0, MIN(p_size) AS a1, SUM(o_totalprice) AS a2 FROM lineitem JOIN part ON l_partkey = p_partkey JOIN supplier ON l_suppkey = s_suppkey JOIN orders ON l_orderkey = o_orderkey WHERE (o_totalprice >= 177206.8 AND o_totalprice <= 212358.46) GROUP BY p_brand, l_shipinstruct ORDER BY a0 DESC, p_brand ASC, l_shipinstruct ASC LIMIT 10`
- q19 (agg): `SELECT l_shipmode, MIN(l_discount) AS a0, MIN(l_extendedprice * (1 - l_discount)) AS a1, AVG(l_extendedprice) AS a2, SUM(l_quantity) AS a3 FROM lineitem WHERE (l_shipinstruct IN ('COLLECT COD', 'DELIVER IN PERSON', 'NONE', 'TAKE BACK RETURN')) GROUP BY l_shipmode ORDER BY a0 ASC, l_shipmode ASC`
- q20 (join): `SELECT o_orderpriority, MAX(c_acctbal) AS a0, MAX(o_totalprice) AS a1, SUM(c_acctbal) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (o_orderdate >= 9336) GROUP BY o_orderpriority`
- q21 (join): `SELECT c_mktsegment, COUNT(c_acctbal) AS a0, MAX(c_acctbal) AS a1 FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE (c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY')) AND (c_acctbal >= 7136.99) GROUP BY c_mktsegment ORDER BY a0 DESC, c_mktsegment ASC LIMIT 10`
- q22 (join): `SELECT o_orderpriority, SUM(c_acctbal) AS a0, AVG(o_totalprice) AS a1, AVG(c_acctbal) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey WHERE (c_mktsegment = 'FURNITURE') GROUP BY o_orderpriority`
- q23 (join): `SELECT l_returnflag, s_nationkey, COUNT(l_tax) AS a0 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey WHERE (l_shipinstruct IN ('DELIVER IN PERSON', 'TAKE BACK RETURN')) AND (s_acctbal >= -787.58) GROUP BY l_returnflag, s_nationkey`
- q24 (topn): `SELECT l_orderkey, l_linenumber, l_quantity, l_discount FROM lineitem WHERE (l_receiptdate < 10180) ORDER BY l_quantity ASC, l_orderkey ASC, l_linenumber ASC LIMIT 10`
- q25 (topn): `SELECT l_orderkey, l_linenumber, l_tax, l_quantity FROM lineitem WHERE (l_extendedprice >= 5087.0) AND (l_quantity >= 37.0) ORDER BY l_tax DESC, l_orderkey ASC, l_linenumber ASC LIMIT 10`
- q26 (join): `SELECT o_orderpriority, SUM(c_acctbal) AS a0, MAX(c_acctbal) AS a1 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (o_orderdate >= 8567 AND o_orderdate <= 9048) AND (c_mktsegment IN ('BUILDING', 'HOUSEHOLD')) AND (c_acctbal >= -784.52 AND c_acctbal <= 1429.5) GROUP BY o_orderpriority ORDER BY a0 DESC, o_orderpriority ASC`
- q27 (distinct): `SELECT o_shippriority, COUNT(DISTINCT c_mktsegment) AS n FROM orders JOIN customer ON o_custkey = c_custkey WHERE TRUE GROUP BY o_shippriority ORDER BY n DESC, o_shippriority ASC LIMIT 20`
- q28 (agg): `SELECT l_shipinstruct, l_linestatus, SUM(l_extendedprice * (1 - l_discount)) AS a0, SUM(l_extendedprice * (1 - l_discount)) AS a1 FROM lineitem WHERE (l_receiptdate < 10084) GROUP BY l_shipinstruct, l_linestatus ORDER BY a0 DESC, l_shipinstruct ASC, l_linestatus ASC LIMIT 10`
- q29 (select_distinct): `SELECT DISTINCT l_shipinstruct FROM lineitem JOIN supplier ON l_suppkey = s_suppkey WHERE (l_discount >= 0.05) AND (l_shipmode = 'AIR')`
- q30 (select_distinct): `SELECT DISTINCT o_orderpriority, o_orderstatus FROM orders WHERE (o_orderpriority = '4-NOT SPECIFIED')`
- q31 (agg): `SELECT p_mfgr, p_brand, SUM(p_size) AS a0, COUNT(p_size) AS a1 FROM part WHERE TRUE GROUP BY p_mfgr, p_brand HAVING COUNT(p_size) > 2`
- q32 (agg): `SELECT o_orderstatus, MIN(o_totalprice) AS a0 FROM orders WHERE (o_orderpriority <> '4-NOT SPECIFIED') GROUP BY o_orderstatus`
- q33 (join): `SELECT n_name, MIN(c_acctbal) AS a0, AVG(o_totalprice) AS a1, SUM(o_totalprice) AS a2, MIN(c_acctbal) AS a3 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'FURNITURE')) AND (c_acctbal >= 3166.7) GROUP BY n_name`
- q34 (select_distinct): `SELECT DISTINCT ps_suppkey FROM partsupp WHERE (ps_availqty >= 4605)`
- q35 (select_distinct): `SELECT DISTINCT ps_suppkey FROM partsupp WHERE (ps_availqty >= 3409) AND (ps_supplycost >= 61.36 AND ps_supplycost <= 260.84)`
- q36 (global): `SELECT SUM(l_quantity) AS a0, SUM(l_extendedprice) AS a1, COUNT(l_tax) AS a2 FROM lineitem WHERE TRUE`
- q37 (agg): `SELECT l_linestatus, l_returnflag, MAX(l_quantity) AS a0 FROM lineitem WHERE (l_extendedprice < 65867.5) AND (l_returnflag IN ('A', 'N', 'R')) GROUP BY l_linestatus, l_returnflag`
- q38 (topn): `SELECT c_custkey, c_acctbal FROM customer WHERE TRUE ORDER BY c_acctbal ASC, c_custkey ASC LIMIT 50`
- q39 (join): `SELECT o_shippriority, o_orderpriority, o_orderstatus, COUNT(o_totalprice) AS a0, MIN(o_totalprice) AS a1, SUM(c_acctbal) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (o_orderstatus = 'P') GROUP BY o_shippriority, o_orderpriority, o_orderstatus`
- q40 (distinct): `SELECT l_suppkey, COUNT(DISTINCT l_returnflag) AS n FROM lineitem WHERE (l_receiptdate >= 8932) AND (l_discount >= 0.06 AND l_discount <= 0.07) GROUP BY l_suppkey ORDER BY n DESC, l_suppkey ASC LIMIT 100`
- q41 (topn): `SELECT ps_partkey, ps_suppkey, ps_supplycost, ps_availqty FROM partsupp WHERE (ps_supplycost >= 380.85 AND ps_supplycost <= 580.1) ORDER BY ps_supplycost ASC, ps_partkey ASC, ps_suppkey ASC LIMIT 200`
- q42 (agg): `SELECT l_suppkey, l_returnflag, l_shipmode, SUM(l_extendedprice * (1 - l_discount)) AS a0, SUM(l_quantity) AS a1, COUNT(l_tax) AS a2, SUM(l_tax) AS a3 FROM lineitem WHERE (l_receiptdate < 10373) AND (l_discount < 0.09) GROUP BY l_suppkey, l_returnflag, l_shipmode ORDER BY a0 ASC, l_suppkey ASC, l_returnflag ASC, l_shipmode ASC LIMIT 100`
- q43 (agg): `SELECT o_custkey, o_shippriority, o_orderstatus, AVG(o_totalprice) AS a0, AVG(o_totalprice) AS a1, MAX(o_totalprice) AS a2, SUM(o_totalprice) AS a3 FROM orders WHERE (o_orderstatus IN ('F', 'O', 'P')) GROUP BY o_custkey, o_shippriority, o_orderstatus`
- q44 (topn): `SELECT l_orderkey, l_linenumber, l_discount, l_tax FROM lineitem WHERE (l_receiptdate < 8450) AND (l_shipdate >= 8143 AND l_shipdate <= 8434) AND (l_quantity < 49.0) ORDER BY l_discount ASC, l_orderkey ASC, l_linenumber ASC LIMIT 200`
- q45 (join): `SELECT c_mktsegment, SUM(c_acctbal) AS a0, MIN(c_acctbal) AS a1, MIN(c_acctbal) AS a2, AVG(c_acctbal) AS a3 FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE TRUE GROUP BY c_mktsegment ORDER BY a0 ASC, c_mktsegment ASC LIMIT 100`
- q46 (agg): `SELECT c_mktsegment, c_nationkey, SUM(c_acctbal) AS a0 FROM customer WHERE TRUE GROUP BY c_mktsegment, c_nationkey`
- q47 (global): `SELECT MIN(l_tax) AS a0 FROM lineitem WHERE (l_extendedprice >= 48251.5) AND (l_returnflag IN ('A', 'N', 'R')) AND (l_shipdate >= 8143 AND l_shipdate <= 8434)`
- q48 (select_distinct): `SELECT DISTINCT ps_suppkey FROM partsupp WHERE (ps_availqty < 1802) AND (ps_supplycost >= 141.43 AND ps_supplycost <= 300.97) ORDER BY ps_suppkey ASC`
- q49 (agg): `SELECT ps_suppkey, COUNT(ps_availqty) AS a0, COUNT(ps_availqty) AS a1, AVG(ps_supplycost) AS a2, AVG(ps_supplycost) AS a3 FROM partsupp WHERE (ps_availqty >= 6603 AND ps_availqty <= 7798) AND (ps_supplycost >= 61.36 AND ps_supplycost <= 260.84) GROUP BY ps_suppkey ORDER BY a0 DESC, ps_suppkey ASC LIMIT 10`
