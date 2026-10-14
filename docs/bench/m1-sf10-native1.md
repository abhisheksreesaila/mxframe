# Engine-only TPC-H scoreboard - 2026-10-12

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | mx-cpu | outside mx |
|---|---:|---:|
| Q1 filter+groupby+8 aggs | **65** | - |
| Q2 min-cost supplier | **157** | - |
| Q3 3-join+top10 | **778** | - |
| Q4 EXISTS semi-join | **1,145** | - |
| Q5 4-join+groupby | **557** | - |
| Q6 filter+global sum | **30** | - |
| Q7 volume shipping | **811** | - |
| Q8 market share | **510** | - |
| Q9 6-join profit | **1,465** | - |
| Q10 returned items top20 | **1,105** | - |
| Q11 important stock | **83** | - |
| Q12 shipping modes | **1,788** | - |
| Q13 customer distribution | **173** | PyArrow/NumPy |
| Q14 promo revenue | **345** | - |
| Q15 top supplier | **194** | - |
| Q16 part/supplier | **475** | - |
| Q17 small-qty revenue | **249** | - |
| Q18 large volume top100 | **1,957** | - |
| Q19 discounted revenue | **2,159** | - |
| Q20 potential promotion | **459** | - |
| Q21 suppliers waiting | **6,365** | - |
| Q22 global sales opportunity | **135** | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **21,006** |  |

Fastest verified engine per query: mx-cpu 22.

### Wins at SF10 (engine-only, both results verified)

- 

### SF10: import + first run, ms (engine-only + engine import time)

| query | mx-cpu |
|---|---:|
| Q1 | 168 |
| Q2 | 262 |
| Q3 | 883 |
| Q4 | 1,247 |
| Q5 | 658 |
| Q6 | 132 |
| Q7 | 916 |
| Q8 | 616 |
| Q9 | 1,602 |
| Q10 | 1,227 |
| Q11 | 185 |
| Q12 | 1,908 |
| Q13 | 275 |
| Q14 | 447 |
| Q15 | 298 |
| Q16 | 578 |
| Q17 | 351 |
| Q18 | 2,061 |
| Q19 | 2,264 |
| Q20 | 566 |
| Q21 | 6,467 |
| Q22 | 236 |
| **sum (22 queries)** | **23,346** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | mx-cpu |
|---|---:|
| Q1 | 474 |
| Q2 | 109 |
| Q3 | 662 |
| Q4 | 605 |
| Q5 | 673 |
| Q6 | 258 |
| Q7 | 699 |
| Q8 | 1,025 |
| Q9 | 1,164 |
| Q10 | 679 |
| Q11 | 78 |
| Q12 | 712 |
| Q13 | 204 |
| Q14 | 760 |
| Q15 | 371 |
| Q16 | 99 |
| Q17 | 731 |
| Q18 | 598 |
| Q19 | 888 |
| Q20 | 902 |
| Q21 | 635 |
| Q22 | 236 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | mx-cpu |
|---|---:|
| Q1 | 64 |
| Q2 | 127 |
| Q3 | 742 |
| Q4 | 1,127 |
| Q5 | 491 |
| Q6 | 29 |
| Q7 | 767 |
| Q8 | 448 |
| Q9 | 1,416 |
| Q10 | 1,080 |
| Q11 | 58 |
| Q12 | 1,725 |
| Q13 | 145 |
| Q14 | 306 |
| Q15 | 168 |
| Q16 | 429 |
| Q17 | 227 |
| Q18 | 1,798 |
| Q19 | 2,371 |
| Q20 | 432 |
| Q21 | 6,251 |
| Q22 | 108 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | mx-cpu |
|---|---:|
| import | 103 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 8.7e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.3, max 7.1 over 22 runs; load average at report time: 7.1 5.3 5.4), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `8de50e7`. Last invocation took 2 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

### Query semantics

All engines and the DuckDB reference compute the same query. Where it differs from the official TPC-H text the difference is listed here; this is not a TPC-certified benchmark.

- Q2: region EUROPE and nation joined by key (n_regionkey = 3, no region table); 'BRASS' matched anywhere in p_type (TPC-H: suffix)
- Q5: region ASIA by n_regionkey = 2 (no region table)
- Q7: shipping year from l_shipdate; nations FRANCE/GERMANY by name
- Q8: supplier nation BRAZIL by nationkey 2, region AMERICA by n_regionkey = 1 (no region table)
- Q10: no c_name/c_acctbal/c_address/c_phone/c_comment output columns; groups by (o_custkey, n_name); ties on revenue broken by o_custkey
- Q13: the `o_comment NOT LIKE '%special%requests%'` predicate is NOT applied (o_comment is not read)
- Q16: supplier filter is `s_comment LIKE '%Complaints%'` (TPC-H: '%Customer%Complaints%')
- Q18: sorted by sum_qty DESC, l_orderkey (TPC-H: o_totalprice DESC, o_orderdate); no c_name/o_orderdate/o_totalprice output columns
- Q20: Canada by nationkey 3 (no nation table)
- Q21: Saudi Arabia by nationkey 20 (no nation table)
- Q22: the c_acctbal column is float32 in the data, so the average-balance threshold and the sums are computed on float32 inputs (the DuckDB reference sums them in DOUBLE)

**Work the mx engines do outside mxframe** (PyArrow / NumPy inside the timed region; all other queries are pure mxframe):

- Q13: PyArrow: appends the zero-order customer row (customers minus customers with orders); no left join: works around the engine's left-join count bug (F-016)
- Q22: PyArrow/NumPy: phone-prefix substring, NOT EXISTS anti-join

