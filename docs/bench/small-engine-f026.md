# Engine-only TPC-H scoreboard - 2026-10-12

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 8.8 | **2.6 †3e-6** | 72 | 3.5 | - |
| Q2 min-cost supplier | 7.6 | **2.4** | 75 | 4.5 | - |
| Q3 3-join+top10 | 5.4 | **2.3** | 76 | 3.9 | - |
| Q4 EXISTS semi-join | 2.5 | **1.9** | 75 | 3.5 | - |
| Q5 4-join+groupby | 5.4 | **2.9** | 74 | 3.8 | - |
| Q6 filter+global sum | 1.4 | **0.9** | 71 | 24 | - |
| Q7 volume shipping | 9.0 | **2.9** | 74 | 4.9 | - |
| Q8 market share | 7.0 | **2.0** | 75 | 4.8 | - |
| Q9 6-join profit | 7.5 | **3.3** | 77 | 4.9 | - |
| Q10 returned items top20 | 5.8 | **2.2** | 77 | 4.0 | - |
| Q11 important stock | 2.4 | **1.5** | 73 | ERR | - |
| Q12 shipping modes | 4.4 | **2.0** | 106 | 3.8 | - |
| Q13 customer distribution | 3.6 | **1.8** | 74 | 3.4 | PyArrow/NumPy |
| Q14 promo revenue | 2.3 | **1.4** | 71 | 3.6 | - |
| Q15 top supplier | 3.7 | **1.5** | 74 | 4.0 | - |
| Q16 part/supplier | 6.4 | **2.6** | 77 | 4.8 | - |
| Q17 small-qty revenue | 4.2 | **1.9** | 72 | 7.1 WRONG | - |
| Q18 large volume top100 | 4.5 | **2.6** | 76 | 4.4 | - |
| Q19 discounted revenue | 5.9 WRONG | 2.4 WRONG | 72 | **8.6** | - |
| Q20 potential promotion | 5.1 | **2.9** | 74 | 4.8 | - |
| Q21 suppliers waiting | 4.8 | **3.5** | 78 | 6.4 | - |
| Q22 global sales opportunity | 6.3 | **2.1** | 76 | 6.9 | PyArrow/NumPy |
| **sum over the 19 queries valid for every engine** | **102** | **44** | **1,450** | **104** |  |

Fastest verified engine per query: pandas 0, Polars 21, DuckDB 0, mx-cpu 1.

### Wins at SF0.002 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (19 comparable); **mx-cpu beats pandas on 13/22** (19 comparable); **mx-cpu beats DuckDB on 20/22** (20 comparable)

### SF0.002: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 243 | 74 †3e-6 | 113 | 127 |
| Q2 | 240 | 73 | 116 | 130 |
| Q3 | 239 | 74 | 114 | 128 |
| Q4 | 236 | 74 | 113 | 129 |
| Q5 | 236 | 77 | 115 | 129 |
| Q6 | 237 | 73 | 109 | 149 |
| Q7 | 246 | 74 | 112 | 130 |
| Q8 | 248 | 75 | 113 | 129 |
| Q9 | 242 | 77 | 117 | 133 |
| Q10 | 238 | 73 | 118 | 129 |
| Q11 | 239 | 74 | 111 | ERR |
| Q12 | 241 | 76 | 148 | 132 |
| Q13 | 237 | 74 | 112 | 127 |
| Q14 | 235 | 73 | 108 | 131 |
| Q15 | 244 | 72 | 113 | 128 |
| Q16 | 237 | 74 | 114 | 129 |
| Q17 | 239 | 73 | 111 | 132 WRONG |
| Q18 | 240 | 75 | 113 | 131 |
| Q19 | 237 WRONG | 76 WRONG | 110 | 134 |
| Q20 | 240 | 77 | 114 | 134 |
| Q21 | 241 | 75 | 117 | 131 |
| Q22 | 242 | 73 | 115 | 133 |
| **sum (19 queries)** | **4,560** | **1,416** | **2,192** | **2,489** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.002: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 16 | 13 | 9.7 | 29 |
| Q2 | 17 | 19 | 9.9 | 30 |
| Q3 | 17 | 20 | 10 | 29 |
| Q4 | 16 | 18 | 8.9 | 28 |
| Q5 | 17 | 20 | 11 | 30 |
| Q6 | 14 | 13 | 8.4 | 6.0 |
| Q7 | 18 | 20 | 11 | 28 |
| Q8 | 19 | 21 | 12 | 31 |
| Q9 | 19 | 21 | 13 | 32 |
| Q10 | 17 | 19 | 11 | 28 |
| Q11 | 14 | 18 | 7.9 | 28 |
| Q12 | 16 | 17 | 9.6 | 31 |
| Q13 | 14 | 18 | 8.0 | 27 |
| Q14 | 15 | 18 | 9.0 | 28 |
| Q15 | 16 | 19 | 9.1 | 27 |
| Q16 | 15 | 19 | 9.0 | 28 |
| Q17 | 15 | 18 | 8.9 | 25 |
| Q18 | 16 | 19 | 9.1 | 29 |
| Q19 | 16 | 18 | 10 | 29 |
| Q20 | 17 | 19 | 11 | 30 |
| Q21 | 17 | 19 | 9.6 | 31 |
| Q22 | 16 | 17 | 8.4 | 27 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 7.4 | 2.0 | 2.7 | 1.2 |
| Q2 | 6.5 | 1.4 | 5.8 | 2.4 |
| Q3 | 4.2 | 5.1 | 3.4 | 1.7 |
| Q4 | 1.7 | 1.6 | 3.6 | 1.8 |
| Q5 | 4.3 | 2.1 | 3.7 | 2.1 |
| Q6 | 0.9 | 2.7 | 0.7 | 0.6 |
| Q7 | 7.8 | 2.3 | 3.1 | 2.8 |
| Q8 | 5.9 | 2.1 | 4.3 | 2.5 |
| Q9 | 6.3 | 2.5 | 5.2 | 2.9 |
| Q10 | 4.4 | 1.9 | 3.6 | 2.0 |
| Q11 | 1.7 | 2.2 | 1.5 | - |
| Q12 | 3.5 | 1.6 | 3.2 | 1.7 |
| Q13 | 2.7 | 1.2 | 3.0 | 1.4 |
| Q14 | 1.7 | 1.0 | 1.4 | 2.0 |
| Q15 | 2.7 | 1.2 | 2.1 | 1.7 |
| Q16 | 5.2 | 1.7 | 4.3 | 2.2 |
| Q17 | 3.0 | 4.5 | 1.4 | 1.4 WRONG |
| Q18 | 3.4 | 3.2 | 3.3 | 2.0 |
| Q19 | 4.9 WRONG | 8.4 WRONG | 1.7 | 3.1 |
| Q20 | 4.1 | 2.4 | 3.4 | 2.4 |
| Q21 | 3.6 | 2.4 | 6.7 | 3.9 |
| Q22 | 4.9 | 1.5 | 3.6 | 1.6 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 234 | 72 | 39 | 125 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |
| Q11 | mx-cpu | error | - | TypeError: unsupported operand type(s) for *: 'NoneType' and 'float' |
| Q17 | mx-cpu | wrong | inf | max relative error inf / repeat run WRONG |
| Q19 | pandas | wrong | inf | max relative error inf / repeat run WRONG |
| Q19 | polars | wrong | inf | max relative error inf / repeat run WRONG |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 22 | **4.6 †2e-4** | 75 | 25 | - |
| Q2 min-cost supplier | 8.7 | **2.9** | 75 | 5.3 | - |
| Q3 3-join+top10 | 7.4 | **2.7** | 74 | 5.1 | - |
| Q4 EXISTS semi-join | 4.6 | **2.3** | 75 | 5.5 | - |
| Q5 4-join+groupby | 8.7 | **3.3** | 74 | 4.7 | - |
| Q6 filter+global sum | 1.7 | **1.0** | 72 | 2.7 | - |
| Q7 volume shipping | 13 | **4.0** | 76 | 6.1 | - |
| Q8 market share | 7.6 | **2.5** | 75 | 5.4 | - |
| Q9 6-join profit | 9.8 | **4.0** | 78 | 7.1 | - |
| Q10 returned items top20 | 7.2 | **3.8** | 79 | 5.3 | - |
| Q11 important stock | 2.6 | **2.0** | 73 | 7.4 | - |
| Q12 shipping modes | 5.8 | **2.3** | 74 | 6.4 | - |
| Q13 customer distribution | 5.5 | **2.6** | 76 | 4.1 | PyArrow/NumPy |
| Q14 promo revenue | 2.8 | **2.0** | 71 | 3.9 | - |
| Q15 top supplier | 4.0 | **2.1** | 73 | 5.5 | - |
| Q16 part/supplier | 6.7 | **2.9** | 80 | 5.5 | - |
| Q17 small-qty revenue | 4.4 | **1.7** | 73 | 5.0 | - |
| Q18 large volume top100 | 8.8 | **4.2** | 78 | 9.4 | - |
| Q19 discounted revenue | 17 | **2.8** | 77 | 8.0 | - |
| Q20 potential promotion | 6.1 | **3.3** | 75 | 4.9 | - |
| Q21 suppliers waiting | 16 | **7.8** | 84 | 24 | - |
| Q22 global sales opportunity | 6.3 | **2.6** | 76 | 3.8 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **177** | **68** | **1,661** | **160** |  |

Fastest verified engine per query: pandas 0, Polars 22, DuckDB 0, mx-cpu 0.

### Wins at SF0.017 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 12/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 258 | 76 †2e-4 | 113 | 150 |
| Q2 | 241 | 74 | 114 | 131 |
| Q3 | 243 | 75 | 112 | 131 |
| Q4 | 242 | 75 | 113 | 131 |
| Q5 | 245 | 75 | 113 | 131 |
| Q6 | 240 | 75 | 111 | 131 |
| Q7 | 248 | 78 | 116 | 134 |
| Q8 | 243 | 74 | 115 | 131 |
| Q9 | 240 | 75 | 117 | 133 |
| Q10 | 242 | 77 | 118 | 131 |
| Q11 | 239 | 77 | 114 | 135 |
| Q12 | 241 | 74 | 114 | 131 |
| Q13 | 241 | 76 | 117 | 130 |
| Q14 | 237 | 76 | 110 | 131 |
| Q15 | 242 | 75 | 111 | 133 |
| Q16 | 243 | 74 | 120 | 130 |
| Q17 | 239 | 74 | 113 | 129 |
| Q18 | 245 | 76 | 117 | 136 |
| Q19 | 252 | 76 | 117 | 137 |
| Q20 | 239 | 76 | 115 | 133 |
| Q21 | 248 | 79 | 122 | 150 |
| Q22 | 244 | 75 | 113 | 130 |
| **sum (22 queries)** | **5,354** | **1,661** | **2,524** | **2,939** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.017: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 18 | 19 | 28 | 8.4 |
| Q2 | 18 | 20 | 11 | 31 |
| Q3 | 18 | 21 | 19 | 30 |
| Q4 | 17 | 18 | 16 | 29 |
| Q5 | 20 | 21 | 20 | 30 |
| Q6 | 16 | 14 | 17 | 27 |
| Q7 | 20 | 23 | 21 | 32 |
| Q8 | 21 | 23 | 23 | 31 |
| Q9 | 22 | 23 | 27 | 32 |
| Q10 | 20 | 22 | 24 | 29 |
| Q11 | 15 | 18 | 9.4 | 29 |
| Q12 | 19 | 19 | 23 | 28 |
| Q13 | 15 | 17 | 10 | 28 |
| Q14 | 17 | 19 | 16 | 30 |
| Q15 | 17 | 18 | 16 | 28 |
| Q16 | 17 | 19 | 9.4 | 27 |
| Q17 | 17 | 20 | 15 | 28 |
| Q18 | 17 | 20 | 16 | 30 |
| Q19 | 21 | 20 | 33 | 31 |
| Q20 | 19 | 20 | 17 | 28 |
| Q21 | 18 | 21 | 17 | 29 |
| Q22 | 15 | 18 | 9.1 | 26 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 20 | 4.6 | 4.9 | 2.7 |
| Q2 | 7.5 | 1.9 | 4.4 | 2.7 |
| Q3 | 5.6 | 2.4 | 3.6 | 2.9 |
| Q4 | 3.4 | 2.6 | 4.5 | 3.6 |
| Q5 | 7.5 | 2.7 | 5.5 | 3.1 |
| Q6 | 1.1 | 5.2 | 1.1 | 0.8 |
| Q7 | 12 | 4.1 | 5.2 | 4.0 |
| Q8 | 6.4 | 2.2 | 4.6 | 3.9 |
| Q9 | 8.7 | 3.1 | 6.4 | 5.3 |
| Q10 | 5.7 | 2.9 | 6.4 | 3.2 |
| Q11 | 1.9 | 1.5 | 2.2 | 1.5 |
| Q12 | 5.0 | 1.6 | 3.8 | 4.1 |
| Q13 | 4.4 | 2.2 | 3.6 | 1.9 |
| Q14 | 1.9 | 1.3 | 2.2 | 1.7 |
| Q15 | 2.9 | 2.3 | 2.3 | 2.7 |
| Q16 | 5.6 | 2.2 | 5.8 | 3.0 |
| Q17 | 3.5 | 1.6 | 2.5 | 2.7 |
| Q18 | 6.9 | 3.0 | 5.6 | 5.9 |
| Q19 | 15 | 2.5 | 4.9 | 6.7 |
| Q20 | 5.2 | 2.2 | 4.4 | 3.0 |
| Q21 | 13 | 12 | 14 | 19 |
| Q22 | 5.2 | 1.8 | 3.8 | 1.7 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 235 | 72 | 39 | 126 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.3e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 2.6, max 3.0 over 176 runs; load average at report time: 2.1 3.3 2.9), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `c80ac56`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

