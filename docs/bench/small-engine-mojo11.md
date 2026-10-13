# Engine-only TPC-H scoreboard - 2026-10-12

> **Note (F-035):** run on the uncommitted Mojo 1.1 port (modular 26.6, Mojo 1.1.0), *before* the lazy `pyarrow.compute` change; the "mxframe at git 9b5d66e" line below is the base commit, not 26.4 code. mx engines still pre-imported `pyarrow.compute` in this run.


**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 8.9 | **2.4 †3e-6** | 76 | 12 | - |
| Q2 min-cost supplier | 8.1 | **3.0** | 78 | 5.4 | - |
| Q3 3-join+top10 | 6.9 | **2.2** | 81 | 3.9 | - |
| Q4 EXISTS semi-join | 3.4 | **1.7** | 76 | 3.8 | - |
| Q5 4-join+groupby | 6.4 | **2.6** | 85 | 4.1 | - |
| Q6 filter+global sum | 1.8 | **1.7** | 81 | 11 | - |
| Q7 volume shipping | 10 | **3.0** | 81 | 5.1 | - |
| Q8 market share | 7.2 | **2.1** | 76 | 5.2 | - |
| Q9 6-join profit | 8.8 | **3.6** | 84 | 5.5 | - |
| Q10 returned items top20 | 5.8 | **2.5** | 76 | 5.3 | - |
| Q11 important stock | 2.8 | **1.3** | 77 | 16 | - |
| Q12 shipping modes | 6.0 | **2.9** | 75 | 3.9 | - |
| Q13 customer distribution | 3.8 | **1.8** | 75 | 3.4 | PyArrow/NumPy |
| Q14 promo revenue | 2.7 | **1.6** | 74 | 15 | - |
| Q15 top supplier | 3.8 | **1.7** | 80 | 5.0 | - |
| Q16 part/supplier | 6.5 | **3.4** | 81 | 5.6 | - |
| Q17 small-qty revenue | 4.0 | **1.8** | 74 | 7.5 | - |
| Q18 large volume top100 | 5.5 | **3.0** | 81 | 4.3 | - |
| Q19 discounted revenue | 6.4 | **2.2** | 87 | 8.8 | - |
| Q20 potential promotion | 5.2 | **2.9** | 77 | 4.4 | - |
| Q21 suppliers waiting | 4.5 | **3.0** | 80 | 6.6 | - |
| Q22 global sales opportunity | 5.9 | **2.1** | 76 | 3.8 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **125** | **53** | **1,731** | **145** |  |

Fastest verified engine per query: pandas 0, Polars 22, DuckDB 0, mx-cpu 0.

### Wins at SF0.002 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 13/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.002: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 246 | 80 †3e-6 | 115 | 172 |
| Q2 | 289 | 85 | 122 | 148 |
| Q3 | 306 | 78 | 127 | 131 |
| Q4 | 273 | 77 | 119 | 140 |
| Q5 | 284 | 78 | 125 | 166 |
| Q6 | 267 | 83 | 126 | 161 |
| Q7 | 326 | 87 | 121 | 134 |
| Q8 | 254 | 77 | 115 | 133 |
| Q9 | 330 | 87 | 123 | 138 |
| Q10 | 257 | 77 | 115 | 146 |
| Q11 | 303 | 88 | 124 | 146 |
| Q12 | 296 | 85 | 114 | 148 |
| Q13 | 291 | 78 | 123 | 147 |
| Q14 | 267 | 77 | 113 | 144 |
| Q15 | 290 | 75 | 123 | 158 |
| Q16 | 258 | 87 | 131 | 168 |
| Q17 | 244 | 76 | 116 | 149 |
| Q18 | 282 | 76 | 126 | 168 |
| Q19 | 278 | 76 | 125 | 150 |
| Q20 | 242 | 75 | 116 | 132 |
| Q21 | 246 | 77 | 121 | 164 |
| Q22 | 245 | 74 | 117 | 131 |
| **sum (22 queries)** | **6,072** | **1,755** | **2,658** | **3,275** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.002: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 16 | 18 | 10 | 8.3 |
| Q2 | 18 | 19 | 11 | 19 |
| Q3 | 22 | 19 | 12 | 17 |
| Q4 | 17 | 19 | 8.9 | 17 |
| Q5 | 20 | 21 | 11 | 19 |
| Q6 | 15 | 12 | 9.3 | 6.6 |
| Q7 | 23 | 23 | 11 | 18 |
| Q8 | 20 | 21 | 12 | 18 |
| Q9 | 22 | 24 | 15 | 19 |
| Q10 | 17 | 21 | 11 | 17 |
| Q11 | 18 | 18 | 9.6 | 8.0 |
| Q12 | 19 | 20 | 10 | 19 |
| Q13 | 15 | 19 | 8.8 | 15 |
| Q14 | 20 | 18 | 9.3 | 7.8 |
| Q15 | 18 | 20 | 11 | 16 |
| Q16 | 18 | 21 | 10 | 16 |
| Q17 | 15 | 18 | 9.5 | 17 |
| Q18 | 16 | 19 | 11 | 16 |
| Q19 | 18 | 19 | 11 | 16 |
| Q20 | 17 | 19 | 10 | 16 |
| Q21 | 17 | 19 | 10.0 | 16 |
| Q22 | 15 | 18 | 8.5 | 16 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 7.8 | 2.3 | 1.9 | 1.6 |
| Q2 | 7.0 | 2.0 | 6.3 | 3.3 |
| Q3 | 4.5 | 2.1 | 5.1 | 1.7 |
| Q4 | 2.1 | 2.1 | 4.7 | 1.6 |
| Q5 | 5.2 | 2.4 | 3.6 | 1.9 |
| Q6 | 1.2 | 1.1 | 1.0 | 0.6 |
| Q7 | 9.3 | 2.7 | 3.6 | 3.0 |
| Q8 | 5.9 | 1.7 | 4.5 | 3.1 |
| Q9 | 7.8 | 2.8 | 6.4 | 3.0 |
| Q10 | 4.5 | 1.9 | 4.0 | 2.3 |
| Q11 | 2.8 | 2.2 | 2.1 | 1.2 |
| Q12 | 4.4 | 1.6 | 2.8 | 1.9 |
| Q13 | 3.5 | 1.2 | 2.9 | 1.4 |
| Q14 | 1.7 | 1.1 | 1.6 | 1.8 |
| Q15 | 2.7 | 1.5 | 2.1 | 2.1 |
| Q16 | 6.8 | 1.8 | 5.0 | 2.8 |
| Q17 | 3.0 | 1.5 | 1.4 | 1.8 |
| Q18 | 3.7 | 1.8 | 3.9 | 2.3 |
| Q19 | 5.2 | 2.0 | 2.2 | 3.0 |
| Q20 | 4.5 | 1.7 | 3.3 | 2.3 |
| Q21 | 3.6 | 2.8 | 6.3 | 3.9 |
| Q22 | 4.9 | 1.2 | 5.2 | 1.6 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 271 | 75 | 41 | 141 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 21 | **5.2 †2e-4** | 77 | 14 | - |
| Q2 min-cost supplier | 8.9 | **2.6** | 77 | 5.0 | - |
| Q3 3-join+top10 | 7.3 | **2.7** | 76 | 5.4 | - |
| Q4 EXISTS semi-join | 4.5 | **2.5** | 80 | 5.9 | - |
| Q5 4-join+groupby | 8.9 | **3.2** | 78 | 5.7 | - |
| Q6 filter+global sum | 1.7 | **0.9** | 73 | 11 | - |
| Q7 volume shipping | 14 | **4.1** | 77 | 6.1 | - |
| Q8 market share | 7.8 | **2.5** | 77 | 6.3 | - |
| Q9 6-join profit | 9.9 | **4.5** | 78 | 7.2 | - |
| Q10 returned items top20 | 7.4 | **3.6** | 83 | 5.5 | - |
| Q11 important stock | 2.9 | **1.7** | 73 | 3.6 | - |
| Q12 shipping modes | 6.0 | **2.4** | 77 | 5.9 | - |
| Q13 customer distribution | 5.4 | **2.2** | 76 | 7.0 | PyArrow/NumPy |
| Q14 promo revenue | 2.8 | **1.6** | 72 | 4.0 | - |
| Q15 top supplier | 4.0 | **1.9** | 74 | 5.3 | - |
| Q16 part/supplier | 7.0 | **3.0** | 82 | 5.4 | - |
| Q17 small-qty revenue | 4.5 | **1.9** | 73 | 4.9 | - |
| Q18 large volume top100 | 8.8 | **3.8** | 78 | 9.7 | - |
| Q19 discounted revenue | 19 | **2.8** | 76 | 8.4 | - |
| Q20 potential promotion | 6.2 | **2.9** | 78 | 5.5 | - |
| Q21 suppliers waiting | 16 | **9.5** | 85 | 25 | - |
| Q22 global sales opportunity | 6.9 | **2.3** | 78 | 4.6 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **180** | **68** | **1,698** | **161** |  |

Fastest verified engine per query: pandas 0, Polars 22, DuckDB 0, mx-cpu 0.

### Wins at SF0.017 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 13/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 259 | 79 †2e-4 | 116 | 143 |
| Q2 | 251 | 77 | 117 | 133 |
| Q3 | 244 | 77 | 116 | 176 |
| Q4 | 245 | 80 | 119 | 135 |
| Q5 | 249 | 75 | 120 | 132 |
| Q6 | 248 | 73 | 115 | 140 |
| Q7 | 280 | 78 | 117 | 145 |
| Q8 | 245 | 76 | 116 | 137 |
| Q9 | 249 | 76 | 116 | 138 |
| Q10 | 249 | 76 | 125 | 135 |
| Q11 | 246 | 75 | 112 | 132 |
| Q12 | 246 | 76 | 117 | 133 |
| Q13 | 245 | 75 | 118 | 135 |
| Q14 | 244 | 75 | 112 | 136 |
| Q15 | 242 | 77 | 113 | 136 |
| Q16 | 248 | 78 | 120 | 133 |
| Q17 | 279 | 76 | 113 | 133 |
| Q18 | 247 | 79 | 116 | 139 |
| Q19 | 274 | 78 | 115 | 138 |
| Q20 | 247 | 75 | 117 | 134 |
| Q21 | 255 | 85 | 125 | 160 |
| Q22 | 285 | 79 | 117 | 170 |
| **sum (22 queries)** | **5,576** | **1,696** | **2,574** | **3,092** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.017: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 19 | 19 | 29 | 7.9 |
| Q2 | 18 | 20 | 11 | 17 |
| Q3 | 18 | 20 | 21 | 18 |
| Q4 | 17 | 19 | 15 | 16 |
| Q5 | 20 | 22 | 20 | 20 |
| Q6 | 15 | 13 | 17 | 6.9 |
| Q7 | 20 | 22 | 20 | 20 |
| Q8 | 21 | 23 | 23 | 19 |
| Q9 | 22 | 24 | 26 | 20 |
| Q10 | 21 | 22 | 25 | 18 |
| Q11 | 15 | 19 | 9.5 | 15 |
| Q12 | 20 | 19 | 23 | 18 |
| Q13 | 15 | 18 | 9.8 | 15 |
| Q14 | 17 | 19 | 16 | 17 |
| Q15 | 17 | 20 | 16 | 17 |
| Q16 | 17 | 19 | 10 | 17 |
| Q17 | 17 | 19 | 15 | 16 |
| Q18 | 17 | 21 | 15 | 16 |
| Q19 | 21 | 22 | 32 | 17 |
| Q20 | 19 | 20 | 18 | 17 |
| Q21 | 18 | 20 | 17 | 19 |
| Q22 | 17 | 18 | 9.5 | 17 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 19 | 4.6 | 5.5 | 2.7 |
| Q2 | 7.5 | 1.9 | 5.1 | 2.8 |
| Q3 | 5.8 | 2.3 | 5.3 | 3.6 |
| Q4 | 3.3 | 2.6 | 3.8 | 3.6 |
| Q5 | 7.8 | 2.5 | 5.1 | 3.3 |
| Q6 | 1.1 | 2.6 | 1.2 | 0.9 |
| Q7 | 12 | 3.6 | 5.2 | 4.0 |
| Q8 | 6.6 | 2.1 | 4.7 | 4.0 |
| Q9 | 9.0 | 3.6 | 6.4 | 5.5 |
| Q10 | 5.8 | 3.0 | 5.7 | 3.2 |
| Q11 | 1.9 | 1.3 | 1.9 | 1.6 |
| Q12 | 4.9 | 1.8 | 5.3 | 5.1 |
| Q13 | 4.5 | 2.2 | 3.3 | 1.8 |
| Q14 | 1.9 | 1.1 | 1.8 | 1.7 |
| Q15 | 2.9 | 1.4 | 2.3 | 2.7 |
| Q16 | 5.6 | 2.0 | 5.4 | 3.0 |
| Q17 | 3.5 | 3.7 | 3.1 | 2.8 |
| Q18 | 7.3 | 2.9 | 5.8 | 6.2 |
| Q19 | 16 | 2.3 | 5.3 | 6.8 |
| Q20 | 5.2 | 2.6 | 4.2 | 2.9 |
| Q21 | 13 | 9.5 | 14 | 19 |
| Q22 | 5.7 | 1.9 | 4.6 | 2.1 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 241 | 74 | 40 | 129 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.0e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.1, max 6.5 over 176 runs; load average at report time: 6.5 4.3 4.5), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `9b5d66e`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

