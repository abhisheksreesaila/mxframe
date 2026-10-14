# Engine-only TPC-H scoreboard - 2026-10-12

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 9.0 | **3.0 †3e-6** | 77 | 12 | - |
| Q2 min-cost supplier | 7.7 | **2.6** | 76 | 27 | - |
| Q3 3-join+top10 | 5.5 | **2.0** | 79 | 26 | - |
| Q4 EXISTS semi-join | 2.6 | **1.8** | 77 | 35 | - |
| Q5 4-join+groupby | 5.5 | **2.8** | 75 | 26 | - |
| Q6 filter+global sum | 1.4 | **1.0** | 72 | 11 | - |
| Q7 volume shipping | 9.2 | **2.9** | 75 | 28 | - |
| Q8 market share | 7.1 | **2.2** | 76 | 28 | - |
| Q9 6-join profit | 7.5 | **3.1** | 77 | 28 | - |
| Q10 returned items top20 | 5.8 | **2.5** | 74 | 26 | - |
| Q11 important stock | 2.3 | **1.3** | 74 | 40 | - |
| Q12 shipping modes | 4.5 | **2.4** | 75 | 27 | - |
| Q13 customer distribution | 3.6 | **1.6** | 76 | 35 | PyArrow/NumPy |
| Q14 promo revenue | 2.4 | **1.5** | 79 | 36 | - |
| Q15 top supplier | 3.6 | **1.8** | 73 | 26 | - |
| Q16 part/supplier | 6.2 | **2.6** | 79 | 27 | - |
| Q17 small-qty revenue | 4.0 | **2.0** | 72 | 38 | - |
| Q18 large volume top100 | 4.9 | **2.5** | 77 | 25 | - |
| Q19 discounted revenue | 6.0 | **2.4** | 73 | 43 | - |
| Q20 potential promotion | 5.1 | **2.6** | 75 | 26 | - |
| Q21 suppliers waiting | 4.6 | **3.2** | 79 | 29 | - |
| Q22 global sales opportunity | 5.9 | **2.3** | 77 | 33 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **114** | **50** | **1,668** | **635** |  |

Fastest verified engine per query: pandas 0, Polars 22, DuckDB 0, mx-cpu 0.

### Wins at SF0.002 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 0/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.002: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 251 | 74 †3e-6 | 116 | 113 |
| Q2 | 243 | 74 | 115 | 130 |
| Q3 | 240 | 74 | 122 | 128 |
| Q4 | 241 | 77 | 117 | 137 |
| Q5 | 260 | 75 | 115 | 129 |
| Q6 | 237 | 81 | 111 | 115 |
| Q7 | 246 | 77 | 114 | 133 |
| Q8 | 243 | 75 | 114 | 132 |
| Q9 | 243 | 75 | 119 | 132 |
| Q10 | 242 | 77 | 115 | 130 |
| Q11 | 239 | 73 | 114 | 142 |
| Q12 | 240 | 78 | 113 | 130 |
| Q13 | 268 | 75 | 116 | 138 |
| Q14 | 250 | 74 | 120 | 138 |
| Q15 | 240 | 73 | 112 | 131 |
| Q16 | 244 | 74 | 121 | 131 |
| Q17 | 242 | 76 | 112 | 140 |
| Q18 | 242 | 75 | 118 | 130 |
| Q19 | 243 | 74 | 115 | 144 |
| Q20 | 242 | 74 | 117 | 130 |
| Q21 | 238 | 75 | 119 | 133 |
| Q22 | 239 | 77 | 119 | 136 |
| **sum (22 queries)** | **5,371** | **1,658** | **2,553** | **2,903** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.002: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 16 | 12 | 11 | 6.4 |
| Q2 | 17 | 20 | 9.7 | 16 |
| Q3 | 17 | 19 | 11 | 17 |
| Q4 | 16 | 19 | 9.2 | 6.7 |
| Q5 | 18 | 22 | 11 | 17 |
| Q6 | 14 | 17 | 8.6 | 6.6 |
| Q7 | 18 | 20 | 11 | 18 |
| Q8 | 19 | 22 | 12 | 17 |
| Q9 | 19 | 20 | 12 | 18 |
| Q10 | 17 | 20 | 11 | 18 |
| Q11 | 15 | 18 | 8.2 | 6.6 |
| Q12 | 16 | 17 | 10 | 16 |
| Q13 | 14 | 18 | 8.6 | 7.0 |
| Q14 | 16 | 17 | 9.9 | 6.1 |
| Q15 | 16 | 18 | 8.9 | 14 |
| Q16 | 16 | 19 | 8.7 | 15 |
| Q17 | 16 | 18 | 9.0 | 8.1 |
| Q18 | 16 | 18 | 9.4 | 17 |
| Q19 | 17 | 18 | 11 | 6.2 |
| Q20 | 18 | 19 | 11 | 17 |
| Q21 | 16 | 19 | 9.9 | 15 |
| Q22 | 15 | 17 | 8.7 | 7.7 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 7.8 | 5.9 | 2.1 | 0.9 |
| Q2 | 6.5 | 1.9 | 5.1 | 2.4 |
| Q3 | 4.1 | 1.9 | 3.4 | 1.9 |
| Q4 | 1.8 | 2.2 | 3.3 | 1.4 |
| Q5 | 4.5 | 2.1 | 3.5 | 1.9 |
| Q6 | 1.0 | 3.4 | 0.7 | 0.6 |
| Q7 | 7.9 | 2.1 | 2.8 | 3.2 |
| Q8 | 5.8 | 2.0 | 3.8 | 2.9 |
| Q9 | 6.1 | 2.5 | 5.3 | 3.3 |
| Q10 | 4.4 | 1.9 | 3.6 | 2.2 |
| Q11 | 1.7 | 1.7 | 1.6 | 1.4 |
| Q12 | 3.6 | 1.6 | 2.6 | 1.8 |
| Q13 | 2.7 | 3.5 | 2.7 | 1.5 |
| Q14 | 1.7 | 1.1 | 1.8 | 1.6 |
| Q15 | 2.6 | 1.5 | 1.9 | 1.9 |
| Q16 | 5.2 | 1.8 | 4.3 | 3.0 |
| Q17 | 3.0 | 1.6 | 1.6 | 1.5 |
| Q18 | 3.4 | 1.8 | 3.9 | 1.9 |
| Q19 | 4.8 | 3.1 | 2.0 | 3.2 |
| Q20 | 4.2 | 1.6 | 3.8 | 2.5 |
| Q21 | 3.5 | 2.6 | 6.3 | 4.4 |
| Q22 | 4.7 | 1.7 | 4.2 | 1.5 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 236 | 72 | 40 | 103 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 21 | **5.1 †2e-4** | 77 | 14 | - |
| Q2 min-cost supplier | 8.8 | **2.7** | 77 | 27 | - |
| Q3 3-join+top10 | 7.4 | **2.7** | 77 | 28 | - |
| Q4 EXISTS semi-join | 4.6 | **2.7** | 79 | 28 | - |
| Q5 4-join+groupby | 8.6 | **3.3** | 76 | 27 | - |
| Q6 filter+global sum | 1.7 | **1.0** | 71 | 16 | - |
| Q7 volume shipping | 13 | **4.0** | 76 | 30 | - |
| Q8 market share | 7.8 | **2.8** | 77 | 29 | - |
| Q9 6-join profit | 10 | **4.3** | 77 | 30 | - |
| Q10 returned items top20 | 7.2 | **3.3** | 76 | 30 | - |
| Q11 important stock | 2.6 | **1.8** | 73 | 36 | - |
| Q12 shipping modes | 5.9 | **2.7** | 75 | 27 | - |
| Q13 customer distribution | 6.4 | **2.5** | 75 | 27 | PyArrow/NumPy |
| Q14 promo revenue | 2.8 | **1.7** | 73 | 38 | - |
| Q15 top supplier | 4.0 | **1.8** | 74 | 28 | - |
| Q16 part/supplier | 6.8 | **2.9** | 80 | 30 | - |
| Q17 small-qty revenue | 4.4 | **2.3** | 73 | 36 | - |
| Q18 large volume top100 | 8.9 | **3.5** | 78 | 31 | - |
| Q19 discounted revenue | 17 | **3.0** | 74 | 46 | - |
| Q20 potential promotion | 6.4 | **3.1** | 84 | 36 | - |
| Q21 suppliers waiting | 15 | **7.0** | 87 | 47 | - |
| Q22 global sales opportunity | 6.2 | **2.3** | 75 | 35 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **178** | **66** | **1,684** | **674** |  |

Fastest verified engine per query: pandas 0, Polars 22, DuckDB 0, mx-cpu 0.

### Wins at SF0.017 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 1/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 259 | 79 †2e-4 | 117 | 118 |
| Q2 | 246 | 75 | 115 | 129 |
| Q3 | 242 | 74 | 119 | 132 |
| Q4 | 240 | 76 | 121 | 131 |
| Q5 | 246 | 78 | 117 | 133 |
| Q6 | 237 | 75 | 109 | 117 |
| Q7 | 251 | 78 | 116 | 135 |
| Q8 | 247 | 75 | 119 | 135 |
| Q9 | 248 | 77 | 117 | 135 |
| Q10 | 242 | 76 | 118 | 133 |
| Q11 | 243 | 76 | 113 | 138 |
| Q12 | 243 | 78 | 114 | 132 |
| Q13 | 270 | 75 | 115 | 159 |
| Q14 | 252 | 74 | 123 | 139 |
| Q15 | 238 | 76 | 112 | 130 |
| Q16 | 245 | 74 | 120 | 133 |
| Q17 | 242 | 74 | 112 | 139 |
| Q18 | 246 | 79 | 118 | 159 |
| Q19 | 257 | 78 | 113 | 148 |
| Q20 | 290 | 75 | 129 | 137 |
| Q21 | 253 | 80 | 126 | 152 |
| Q22 | 244 | 77 | 117 | 138 |
| **sum (22 queries)** | **5,483** | **1,680** | **2,578** | **3,001** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.017: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 19 | 20 | 28 | 6.7 |
| Q2 | 19 | 21 | 11 | 18 |
| Q3 | 18 | 21 | 20 | 17 |
| Q4 | 17 | 18 | 16 | 17 |
| Q5 | 20 | 21 | 21 | 17 |
| Q6 | 16 | 15 | 16 | 3.3 |
| Q7 | 20 | 23 | 21 | 19 |
| Q8 | 22 | 23 | 26 | 19 |
| Q9 | 22 | 23 | 27 | 20 |
| Q10 | 20 | 21 | 24 | 18 |
| Q11 | 15 | 19 | 9.2 | 7.0 |
| Q12 | 20 | 19 | 23 | 18 |
| Q13 | 22 | 18 | 9.8 | 16 |
| Q14 | 17 | 19 | 19 | 7.2 |
| Q15 | 17 | 19 | 16 | 17 |
| Q16 | 17 | 20 | 10 | 16 |
| Q17 | 17 | 20 | 15 | 8.7 |
| Q18 | 17 | 20 | 14 | 17 |
| Q19 | 20 | 19 | 31 | 8.4 |
| Q20 | 19 | 21 | 19 | 18 |
| Q21 | 18 | 21 | 18 | 18 |
| Q22 | 15 | 18 | 9.6 | 7.3 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 18 | 4.8 | 6.0 | 1.7 |
| Q2 | 7.3 | 1.9 | 5.8 | 2.7 |
| Q3 | 5.7 | 2.3 | 4.1 | 4.0 |
| Q4 | 3.4 | 2.8 | 4.4 | 3.7 |
| Q5 | 7.2 | 2.8 | 4.9 | 3.1 |
| Q6 | 1.1 | 2.6 | 1.2 | 0.9 |
| Q7 | 12 | 3.4 | 5.4 | 4.9 |
| Q8 | 6.5 | 1.8 | 4.5 | 4.2 |
| Q9 | 8.9 | 3.8 | 6.5 | 5.6 |
| Q10 | 5.7 | 3.1 | 4.9 | 5.0 |
| Q11 | 1.9 | 1.9 | 1.7 | 1.6 |
| Q12 | 4.9 | 2.1 | 3.8 | 3.2 |
| Q13 | 4.9 | 2.2 | 3.1 | 1.8 |
| Q14 | 1.9 | 1.5 | 2.6 | 2.8 |
| Q15 | 2.9 | 1.3 | 2.7 | 2.5 |
| Q16 | 5.8 | 2.3 | 6.2 | 4.9 |
| Q17 | 3.5 | 1.3 | 3.1 | 2.7 |
| Q18 | 7.4 | 3.0 | 5.5 | 6.1 |
| Q19 | 16 | 2.2 | 4.5 | 5.0 |
| Q20 | 5.2 | 2.5 | 4.0 | 11 |
| Q21 | 13 | 11 | 12 | 21 |
| Q22 | 5.1 | 1.9 | 4.1 | 1.6 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 238 | 73 | 40 | 103 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.2e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 8.0, max 10.3 over 176 runs; load average at report time: 10.3 6.4 5.3), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `7576e6c`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

