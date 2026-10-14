# Random queries on TPC-H SF1 (rows shuffled): MXFrame vs Polars vs DuckDB

100 seeded random queries (seeds 0..99, `scripts/bench_random/`), engine-only: each (query, engine) in a fresh process, columns loaded untimed, the first run timed (query + materialized result). DuckDB's result is the reference.

| shape | queries | MXFrame correct | faster than Polars | faster than DuckDB | Polars / MXFrame (geo. mean) | DuckDB / MXFrame (geo. mean) | MXFrame: no native decline logged |
|---|---:|---:|---:|---:|---:|---:|---:|
| agg | 23 | 23/23 | 22/22 | 20/23 | 4.50x | 1.72x | 23/23 |
| distinct | 13 | 13/13 | 13/13 | 7/13 | 1.96x | 1.11x | 11/13 |
| global | 8 | 8/8 | 8/8 | 5/8 | 4.76x | 1.16x | 8/8 |
| join | 35 | 35/35 | 33/35 | 23/35 | 2.59x | 1.13x | 35/35 |
| select_distinct | 11 | 11/11 | 11/11 | 4/11 | 2.14x | 0.99x | 11/11 |
| topn | 10 | 10/10 | 10/10 | 6/10 | 4.65x | 0.98x | 10/10 |
| all | 100 | 100/100 | 97/99 | 65/100 | 3.08x | 1.21x | 98/100 |

Total time over queries every engine ran: MXFrame 1419 ms, Polars 5075 ms, DuckDB 1654 ms. Geo. mean ratio > 1: MXFrame faster.

## MXFrame results that differ from DuckDB or failed (0)


## Polars results that differ from DuckDB or failed (1)

- q69: WRONG first differing row (sorted) 0

## Where MXFrame is slowest relative to the faster of Polars / DuckDB

| query | shape | MXFrame ms | Polars ms | DuckDB ms | fell back to Python for |
|---|---|---:|---:|---:|---|
| q3 | join | 43.8 | 198.4 | 15.8 |  |
| q12 | distinct | 7.2 | 20.6 | 3.0 | count distinct: empty input; empty input |
| q14 | agg | 25.7 | 26.5 | 11.1 |  |
| q64 | distinct | 38.1 | 107.2 | 17.4 |  |
| q34 | select_distinct | 10.2 | 13.1 | 5.3 |  |
| q30 | select_distinct | 9.7 | 17.1 | 5.2 |  |
| q23 | join | 25.5 | 151.0 | 16.2 |  |
| q70 | agg | 9.3 | 10.2 | 5.9 |  |
| q74 | select_distinct | 68.4 | 125.8 | 46.7 |  |
| q33 | join | 20.5 | 25.2 | 14.1 |  |
| q88 | topn | 12.8 | 47.9 | 8.9 |  |
| q62 | join | 38.5 | 186.4 | 27.2 |  |

## Why parts of MXFrame queries ran in Python (count of queries)

- 1: count distinct: empty input
- 1: empty input
- 1: count distinct: column types or nulls

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
- q50 (distinct): `SELECT o_shippriority, COUNT(DISTINCT o_orderpriority) AS n FROM orders JOIN customer ON o_custkey = c_custkey WHERE (o_orderpriority <> '5-LOW') AND (o_orderdate < 8469) AND (c_mktsegment IN ('FURNITURE', 'MACHINERY')) GROUP BY o_shippriority HAVING COUNT(DISTINCT o_orderpriority) > 2 ORDER BY n DESC, o_shippriority ASC LIMIT 100`
- q51 (join): `SELECT l_linestatus, AVG(l_extendedprice * (1 - l_discount)) AS a0 FROM lineitem JOIN orders ON l_orderkey = o_orderkey WHERE (l_shipinstruct IN ('COLLECT COD', 'DELIVER IN PERSON', 'NONE', 'TAKE BACK RETURN')) AND (o_orderdate >= 9240) GROUP BY l_linestatus`
- q52 (join): `SELECT l_returnflag, l_shipinstruct, SUM(l_extendedprice) AS a0 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN part ON l_partkey = p_partkey JOIN nation ON s_nationkey = n_nationkey WHERE (l_quantity >= 12.0) AND (l_shipdate >= 9492) AND (p_mfgr = 'Manufacturer#1') GROUP BY l_returnflag, l_shipinstruct`
- q53 (join): `SELECT p_size, SUM(p_size) AS a0, SUM(p_size) AS a1 FROM part WHERE (p_mfgr IN ('Manufacturer#1', 'Manufacturer#2', 'Manufacturer#4', 'Manufacturer#5')) AND (p_container <> 'MED BAG') AND (p_type = 'LARGE PLATED STEEL') GROUP BY p_size`
- q54 (join): `SELECT p_size, p_brand, p_container, AVG(p_size) AS a0, MAX(p_size) AS a1, AVG(p_size) AS a2 FROM part WHERE (p_type IN ('PROMO BRUSHED TIN', 'SMALL BURNISHED TIN')) AND (p_mfgr IN ('Manufacturer#1', 'Manufacturer#2')) GROUP BY p_size, p_brand, p_container ORDER BY a0 DESC, p_size ASC, p_brand ASC, p_container ASC LIMIT 10`
- q55 (agg): `SELECT o_orderstatus, o_orderpriority, AVG(o_totalprice) AS a0, MAX(o_totalprice) AS a1, MIN(o_totalprice) AS a2, AVG(o_totalprice) AS a3 FROM orders WHERE (o_orderpriority = '2-HIGH') GROUP BY o_orderstatus, o_orderpriority`
- q56 (select_distinct): `SELECT DISTINCT s_nationkey FROM lineitem JOIN supplier ON l_suppkey = s_suppkey WHERE (s_acctbal >= 2328.24 AND s_acctbal <= 3623.7) AND (l_shipinstruct IN ('COLLECT COD', 'DELIVER IN PERSON', 'NONE', 'TAKE BACK RETURN')) ORDER BY s_nationkey ASC`
- q57 (agg): `SELECT ps_suppkey, MIN(ps_supplycost) AS a0, COUNT(ps_supplycost) AS a1, SUM(ps_availqty) AS a2, AVG(ps_supplycost) AS a3 FROM partsupp WHERE TRUE GROUP BY ps_suppkey`
- q58 (join): `SELECT o_shippriority, o_orderstatus, MAX(o_totalprice) AS a0 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (o_orderstatus IN ('F', 'O', 'P')) GROUP BY o_shippriority, o_orderstatus ORDER BY a0 DESC, o_shippriority ASC, o_orderstatus ASC LIMIT 100`
- q59 (join): `SELECT o_shippriority, COUNT(l_extendedprice) AS a0, AVG(l_extendedprice * (1 - l_discount)) AS a1, COUNT(l_quantity) AS a2 FROM lineitem JOIN part ON l_partkey = p_partkey JOIN orders ON l_orderkey = o_orderkey JOIN supplier ON l_suppkey = s_suppkey WHERE (l_quantity >= 26.0) AND (s_acctbal >= 6720.91) GROUP BY o_shippriority ORDER BY a0 DESC, o_shippriority ASC LIMIT 100`
- q60 (join): `SELECT o_orderpriority, SUM(o_totalprice) AS a0, COUNT(o_totalprice) AS a1, MAX(o_totalprice) AS a2, SUM(c_acctbal) AS a3 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (n_name IN ('ARGENTINA', 'BRAZIL', 'INDONESIA')) GROUP BY o_orderpriority`
- q61 (distinct): `SELECT l_suppkey, COUNT(DISTINCT l_shipinstruct) AS n FROM lineitem WHERE (l_returnflag = 'N') AND (l_shipdate >= 9205) GROUP BY l_suppkey ORDER BY n DESC, l_suppkey ASC`
- q62 (join): `SELECT l_shipmode, l_linestatus, MAX(l_tax) AS a0 FROM lineitem JOIN part ON l_partkey = p_partkey WHERE (p_container <> 'WRAP BAG') AND (l_shipmode <> 'FOB') GROUP BY l_shipmode, l_linestatus`
- q63 (distinct): `SELECT p_size, COUNT(DISTINCT p_container) AS n FROM part WHERE (p_container = 'SM BAG') AND (p_type = 'ECONOMY PLATED TIN') GROUP BY p_size HAVING COUNT(DISTINCT p_container) > 3 ORDER BY n DESC, p_size ASC LIMIT 20`
- q64 (distinct): `SELECT s_nationkey, COUNT(DISTINCT l_shipmode) AS n FROM lineitem JOIN supplier ON l_suppkey = s_suppkey WHERE TRUE GROUP BY s_nationkey ORDER BY n DESC, s_nationkey ASC LIMIT 100`
- q65 (topn): `SELECT o_orderkey, o_totalprice, c_acctbal FROM orders JOIN customer ON o_custkey = c_custkey WHERE (o_orderstatus IN ('F', 'O', 'P')) AND (c_acctbal >= 5368.49 AND c_acctbal <= 7136.99) AND (c_mktsegment <> 'HOUSEHOLD') ORDER BY o_totalprice DESC, o_orderkey ASC LIMIT 200`
- q66 (agg): `SELECT o_orderstatus, o_shippriority, o_orderpriority, MAX(o_totalprice) AS a0, COUNT(o_totalprice) AS a1 FROM orders WHERE (o_totalprice >= 200197.57) AND (o_orderstatus = 'P') AND (o_orderpriority IN ('2-HIGH', '4-NOT SPECIFIED')) GROUP BY o_orderstatus, o_shippriority, o_orderpriority`
- q67 (agg): `SELECT l_shipmode, SUM(l_extendedprice * (1 - l_discount)) AS a0 FROM lineitem WHERE (l_extendedprice >= 39600.72 AND l_extendedprice <= 51216.75) AND (l_receiptdate >= 8932 AND l_receiptdate <= 9220) AND (l_shipdate >= 8241 AND l_shipdate <= 8723) GROUP BY l_shipmode`
- q68 (distinct): `SELECT l_orderkey, COUNT(DISTINCT l_returnflag) AS n FROM lineitem WHERE (l_extendedprice >= 30974.34 AND l_extendedprice <= 42480.4) AND (l_shipdate >= 9205 AND l_shipdate <= 9492) AND (l_shipmode = 'AIR') GROUP BY l_orderkey ORDER BY n DESC, l_orderkey ASC`
- q69 (agg): `SELECT l_shipinstruct, l_suppkey, l_linestatus, AVG(l_discount) AS a0, MAX(l_extendedprice) AS a1, MAX(l_quantity) AS a2, SUM(l_extendedprice * (1 - l_discount)) AS a3 FROM lineitem WHERE (l_discount >= 0.05 AND l_discount <= 0.06) GROUP BY l_shipinstruct, l_suppkey, l_linestatus ORDER BY a0 DESC, l_shipinstruct ASC, l_suppkey ASC, l_linestatus ASC LIMIT 100`
- q70 (agg): `SELECT o_custkey, AVG(o_totalprice) AS a0 FROM orders WHERE (o_orderstatus = 'F') AND (o_orderdate >= 8663 AND o_orderdate <= 9144) AND (o_totalprice >= 155188.18 AND o_totalprice <= 200197.57) GROUP BY o_custkey`
- q71 (global): `SELECT AVG(l_extendedprice) AS a0 FROM lineitem WHERE (l_shipdate < 8916) AND (l_shipinstruct = 'DELIVER IN PERSON')`
- q72 (agg): `SELECT l_linestatus, l_returnflag, MAX(l_quantity) AS a0, MIN(l_quantity) AS a1 FROM lineitem WHERE (l_shipmode <> 'SHIP') AND (l_shipdate >= 9109 AND l_shipdate <= 9396) GROUP BY l_linestatus, l_returnflag`
- q73 (join): `SELECT p_mfgr, AVG(l_tax) AS a0, MAX(l_extendedprice) AS a1, MIN(p_size) AS a2 FROM lineitem JOIN part ON l_partkey = p_partkey JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey WHERE (c_mktsegment IN ('BUILDING', 'HOUSEHOLD', 'MACHINERY')) AND (l_shipdate < 8143) GROUP BY p_mfgr`
- q74 (select_distinct): `SELECT DISTINCT l_suppkey, o_shippriority, l_linestatus FROM lineitem JOIN orders ON l_orderkey = o_orderkey WHERE (l_shipdate >= 8434) AND (l_receiptdate >= 9507) ORDER BY l_suppkey ASC, o_shippriority ASC, l_linestatus ASC`
- q75 (distinct): `SELECT n_name, COUNT(DISTINCT c_custkey) AS n FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE (n_name IN ('GERMANY', 'VIETNAM')) AND (c_mktsegment IN ('BUILDING', 'FURNITURE', 'HOUSEHOLD')) AND (c_acctbal >= -784.52) GROUP BY n_name ORDER BY n DESC, n_name ASC`
- q76 (global): `SELECT AVG(p_size) AS a0, SUM(p_size) AS a1, AVG(p_size) AS a2 FROM part WHERE (p_brand IN ('Brand#25', 'Brand#55')) AND (p_container IN ('JUMBO JAR', 'LG PACK', 'MED JAR')) AND (p_size >= 15)`
- q77 (join): `SELECT p_size, p_brand, p_type, MIN(ps_availqty) AS a0 FROM partsupp JOIN part ON ps_partkey = p_partkey WHERE (p_mfgr = 'Manufacturer#3') GROUP BY p_size, p_brand, p_type`
- q78 (join): `SELECT l_returnflag, l_linestatus, l_shipmode, MIN(l_tax) AS a0, COUNT(l_tax) AS a1 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN nation ON s_nationkey = n_nationkey WHERE (l_shipinstruct <> 'TAKE BACK RETURN') AND (l_quantity >= 14.0 AND l_quantity <= 24.0) GROUP BY l_returnflag, l_linestatus, l_shipmode ORDER BY a0 ASC, l_returnflag ASC, l_linestatus ASC, l_shipmode ASC LIMIT 10`
- q79 (join): `SELECT p_size, SUM(p_size) AS a0, MAX(p_size) AS a1, AVG(p_size) AS a2 FROM part WHERE (p_mfgr <> 'Manufacturer#2') AND (p_brand IN ('Brand#13', 'Brand#32')) GROUP BY p_size ORDER BY a0 DESC, p_size ASC LIMIT 10`
- q80 (join): `SELECT c_mktsegment, SUM(c_acctbal) AS a0 FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE (c_mktsegment IN ('HOUSEHOLD', 'MACHINERY')) AND (n_name <> 'ROMANIA') GROUP BY c_mktsegment`
- q81 (select_distinct): `SELECT DISTINCT p_container FROM part WHERE (p_mfgr IN ('Manufacturer#2', 'Manufacturer#5')) AND (p_type IN ('ECONOMY ANODIZED BRASS', 'MEDIUM POLISHED STEEL', 'PROMO BRUSHED COPPER', 'PROMO PLATED COPPER'))`
- q82 (join): `SELECT p_brand, SUM(p_size) AS a0, MIN(p_size) AS a1, SUM(p_size) AS a2, SUM(p_size) AS a3 FROM part WHERE (p_brand IN ('Brand#12', 'Brand#55')) AND (p_mfgr IN ('Manufacturer#1', 'Manufacturer#4', 'Manufacturer#5')) GROUP BY p_brand ORDER BY a0 ASC, p_brand ASC`
- q83 (distinct): `SELECT p_size, COUNT(DISTINCT p_type) AS n FROM part WHERE TRUE GROUP BY p_size HAVING COUNT(DISTINCT p_type) > 3 ORDER BY n DESC, p_size ASC`
- q84 (join): `SELECT l_suppkey, c_mktsegment, o_shippriority, COUNT(l_discount) AS a0, AVG(l_extendedprice * (1 - l_discount)) AS a1, MAX(l_quantity) AS a2, SUM(l_extendedprice * (1 - l_discount)) AS a3 FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey WHERE (c_mktsegment IN ('FURNITURE', 'HOUSEHOLD', 'MACHINERY')) AND (l_receiptdate < 9797) GROUP BY l_suppkey, c_mktsegment, o_shippriority ORDER BY a0 DESC, l_suppkey ASC, c_mktsegment ASC, o_shippriority ASC`
- q85 (join): `SELECT p_brand, p_mfgr, MAX(l_extendedprice * (1 - l_discount)) AS a0, SUM(l_extendedprice) AS a1, SUM(l_discount) AS a2, SUM(l_tax) AS a3 FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN part ON l_partkey = p_partkey WHERE (o_orderstatus IN ('F', 'O', 'P')) AND (l_extendedprice >= 39600.72 AND l_extendedprice <= 48251.5) AND (o_orderpriority = '5-LOW') GROUP BY p_brand, p_mfgr`
- q86 (agg): `SELECT ps_suppkey, SUM(ps_availqty) AS a0, MAX(ps_supplycost) AS a1, SUM(ps_availqty) AS a2, SUM(ps_supplycost) AS a3 FROM partsupp WHERE TRUE GROUP BY ps_suppkey ORDER BY a0 ASC, ps_suppkey ASC`
- q87 (join): `SELECT n_name, o_orderstatus, MIN(o_totalprice) AS a0, COUNT(o_totalprice) AS a1 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY')) AND (c_acctbal >= 9779.74) GROUP BY n_name, o_orderstatus`
- q88 (topn): `SELECT o_orderkey, o_totalprice, c_acctbal FROM orders JOIN customer ON o_custkey = c_custkey WHERE TRUE ORDER BY o_totalprice ASC, o_orderkey ASC LIMIT 10`
- q89 (agg): `SELECT o_shippriority, o_orderstatus, COUNT(o_totalprice) AS a0, SUM(o_totalprice) AS a1, MAX(o_totalprice) AS a2 FROM orders WHERE (o_orderpriority = '2-HIGH') GROUP BY o_shippriority, o_orderstatus HAVING COUNT(o_totalprice) > 19 ORDER BY a0 ASC, o_shippriority ASC, o_orderstatus ASC LIMIT 100`
- q90 (join): `SELECT l_shipinstruct, MAX(l_extendedprice * (1 - l_discount)) AS a0, AVG(l_extendedprice * (1 - l_discount)) AS a1, SUM(l_extendedprice * (1 - l_discount)) AS a2 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN nation ON s_nationkey = n_nationkey WHERE (l_shipmode IN ('MAIL', 'RAIL', 'SHIP')) AND (l_extendedprice < 51216.75) GROUP BY l_shipinstruct ORDER BY a0 ASC, l_shipinstruct ASC`
- q91 (agg): `SELECT l_linestatus, MAX(l_extendedprice) AS a0, SUM(l_tax) AS a1, SUM(l_tax) AS a2 FROM lineitem WHERE (l_shipinstruct IN ('COLLECT COD', 'DELIVER IN PERSON', 'NONE', 'TAKE BACK RETURN')) GROUP BY l_linestatus`
- q92 (topn): `SELECT o_orderkey, o_totalprice FROM orders WHERE (o_totalprice < 25803.23) AND (o_orderdate < 10008) AND (o_orderstatus <> 'P') ORDER BY o_totalprice ASC, o_orderkey ASC LIMIT 10`
- q93 (distinct): `SELECT ps_partkey, COUNT(DISTINCT ps_suppkey) AS n FROM partsupp WHERE TRUE GROUP BY ps_partkey HAVING COUNT(DISTINCT ps_suppkey) > 1 ORDER BY n DESC, ps_partkey ASC`
- q94 (select_distinct): `SELECT DISTINCT l_shipinstruct, l_shipmode FROM lineitem WHERE (l_returnflag = 'R') AND (l_receiptdate >= 9220)`
- q95 (select_distinct): `SELECT DISTINCT l_shipinstruct, l_returnflag FROM lineitem JOIN orders ON l_orderkey = o_orderkey WHERE (l_quantity >= 6.0 AND l_quantity <= 16.0)`
- q96 (global): `SELECT SUM(ps_supplycost) AS a0 FROM partsupp WHERE (ps_availqty >= 9799) AND (ps_supplycost < 899.89)`
- q97 (join): `SELECT c_mktsegment, MIN(c_acctbal) AS a0, SUM(c_acctbal) AS a1, MAX(c_acctbal) AS a2 FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE TRUE GROUP BY c_mktsegment`
- q98 (global): `SELECT SUM(l_quantity) AS a0, MIN(l_extendedprice * (1 - l_discount)) AS a1, COUNT(l_tax) AS a2 FROM lineitem WHERE (l_shipinstruct IN ('COLLECT COD', 'DELIVER IN PERSON', 'TAKE BACK RETURN')) AND (l_quantity >= 4.0)`
- q99 (topn): `SELECT c_custkey, c_acctbal FROM customer WHERE (c_acctbal < 990.7) ORDER BY c_acctbal DESC, c_custkey ASC LIMIT 50`
