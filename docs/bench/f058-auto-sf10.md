# Engine-only TPC-H scoreboard - 2026-10-17

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 140 | **67** | 75 | - |
| Q2 min-cost supplier | 30 | **24** | 29 | - |
| Q3 3-join+top10 | **82** | 101 | 110 | - |
| Q4 EXISTS semi-join | 98 | **96** | 101 | - |
| Q5 4-join+groupby | 74 | **64** | 67 | - |
| Q6 filter+global sum | **31** | 33 | 37 | - |
| Q7 volume shipping | 85 | **83** | 85 | - |
| Q8 market share | **62** | 66 | 72 | - |
| Q9 6-join profit | 439 | 230 | **228** | - |
| Q10 returned items top20 | **117** | 122 | 129 | - |
| Q11 important stock | **16** | 38 | 46 | - |
| Q12 shipping modes | **81** | 124 | 128 | - |
| Q13 customer distribution | 238 | **94** | 103 | PyArrow/NumPy |
| Q14 promo revenue | 62 | **28** | 32 | - |
| Q15 top supplier | **51** | 52 | 57 | - |
| Q16 part/supplier | **97** | 99 | 143 | - |
| Q17 small-qty revenue | **30** | 69 | 71 | - |
| Q18 large volume top100 | **314** | 375 | 387 | - |
| Q19 discounted revenue | 82 | **69** | 74 | - |
| Q20 potential promotion | **49** | 66 | 72 | - |
| Q21 suppliers waiting | 2,737 | **801** | 819 | - |
| Q22 global sales opportunity | **43** | 93 | 95 | - |
| **sum over the 22 queries valid for every engine** | **4,959** | **2,796** | **2,959** |  |

Fastest verified engine per query: DuckDB 12, mx-cpu 9, mx-auto 1.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats DuckDB on 10/22** (22 comparable)
- **mx-auto beats DuckDB on 9/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 253 | 175 | 182 |
| Q2 | 143 | 128 | 131 |
| Q3 | 191 | 209 | 214 |
| Q4 | 210 | 199 | 205 |
| Q5 | 184 | 167 | 170 |
| Q6 | 145 | 135 | 139 |
| Q7 | 196 | 190 | 187 |
| Q8 | 175 | 168 | 173 |
| Q9 | 553 | 334 | 367 |
| Q10 | 229 | 225 | 231 |
| Q11 | 125 | 141 | 155 |
| Q12 | 190 | 227 | 232 |
| Q13 | 348 | 201 | 207 |
| Q14 | 176 | 132 | 134 |
| Q15 | 161 | 159 | 160 |
| Q16 | 208 | 203 | 247 |
| Q17 | 143 | 175 | 174 |
| Q18 | 427 | 495 | 495 |
| Q19 | 191 | 172 | 177 |
| Q20 | 160 | 171 | 176 |
| Q21 | 2,848 | 904 | 922 |
| Q22 | 155 | 203 | 198 |
| **sum (22 queries)** | **7,409** | **5,116** | **5,277** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 617 | 488 | 483 |
| Q2 | 244 | 113 | 114 |
| Q3 | 465 | 668 | 673 |
| Q4 | 339 | 604 | 597 |
| Q5 | 507 | 683 | 683 |
| Q6 | 327 | 260 | 265 |
| Q7 | 550 | 719 | 717 |
| Q8 | 713 | 1,023 | 1,025 |
| Q9 | 819 | 1,177 | 1,189 |
| Q10 | 480 | 676 | 683 |
| Q11 | 106 | 79 | 84 |
| Q12 | 478 | 695 | 688 |
| Q13 | 88 | 207 | 208 |
| Q14 | 468 | 774 | 767 |
| Q15 | 385 | 378 | 380 |
| Q16 | 175 | 101 | 100 |
| Q17 | 389 | 743 | 739 |
| Q18 | 245 | 598 | 599 |
| Q19 | 759 | 868 | 882 |
| Q20 | 557 | 913 | 911 |
| Q21 | 425 | 646 | 637 |
| Q22 | 136 | 244 | 249 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 139 | 64 | 66 |
| Q2 | 30 | 17 | 18 |
| Q3 | 81 | 86 | 90 |
| Q4 | 100 | 81 | 87 |
| Q5 | 74 | 53 | 53 |
| Q6 | 32 | 29 | 29 |
| Q7 | 84 | 69 | 71 |
| Q8 | 60 | 54 | 56 |
| Q9 | 436 | 231 | 222 |
| Q10 | 110 | 97 | 100 |
| Q11 | 15 | 24 | 28 |
| Q12 | 80 | 113 | 115 |
| Q13 | 232 | 82 | 97 |
| Q14 | 59 | 23 | 24 |
| Q15 | 48 | 45 | 48 |
| Q16 | 114 | 86 | 118 |
| Q17 | 28 | 63 | 66 |
| Q18 | 310 | 349 | 354 |
| Q19 | 82 | 68 | 66 |
| Q20 | 49 | 58 | - |
| Q21 | 2,733 | 706 | 705 |
| Q22 | 42 | 87 | 87 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 112 | 104 | 103 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.1e-14, mx-auto 1.0e-14.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 4.1, max 8.2 over 66 runs; load average at report time: 7.1 4.6 3.6), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 2 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

