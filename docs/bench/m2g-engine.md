# Engine-only TPC-H scoreboard - 2026-10-13

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | mx-cpu | outside mx |
|---|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 260 †3e-3 | **9.9** | - |
| Q2 min-cost supplier | **8.0** | 12 | - |
| Q3 3-join+top10 | **31** | 37 | - |
| Q4 EXISTS semi-join | 78 | **32** | - |
| Q5 4-join+groupby | 48 | **27** | - |
| Q6 filter+global sum | 6.7 | **5.3** | - |
| Q7 volume shipping | 105 | **43** | - |
| Q8 market share | **14** | 33 | - |
| Q9 6-join profit | **69 †2e-6** | 77 | - |
| Q10 returned items top20 | 35 | **32** | - |
| Q11 important stock | **4.2** | 8.1 | - |
| Q12 shipping modes | **17** | 19 | - |
| Q13 customer distribution | 52 | **11** | PyArrow/NumPy |
| Q14 promo revenue | **9.2** | 16 | - |
| Q15 top supplier | **8.6** | 11 | - |
| Q16 part/supplier | **15** | 43 | - |
| Q17 small-qty revenue | **8.6** | 15 | - |
| Q18 large volume top100 | **137** | 165 | - |
| Q19 discounted revenue | 37 | **15** | - |
| Q20 potential promotion | **15** | 18 | - |
| Q21 suppliers waiting | **354** | 587 | - |
| Q22 global sales opportunity | **15** | 33 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **1,327** | **1,248** |  |

Fastest verified engine per query: Polars 14, mx-cpu 8.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 8/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 339 †3e-3 | 114 |
| Q2 | 81 | 114 |
| Q3 | 102 | 139 |
| Q4 | 150 | 135 |
| Q5 | 122 | 128 |
| Q6 | 79 | 106 |
| Q7 | 177 | 148 |
| Q8 | 87 | 160 |
| Q9 | 140 †2e-6 | 179 |
| Q10 | 108 | 135 |
| Q11 | 78 | 116 |
| Q12 | 89 | 121 |
| Q13 | 124 | 112 |
| Q14 | 82 | 123 |
| Q15 | 83 | 112 |
| Q16 | 87 | 147 |
| Q17 | 84 | 117 |
| Q18 | 210 | 267 |
| Q19 | 111 | 117 |
| Q20 | 87 | 121 |
| Q21 | 427 | 691 |
| Q22 | 90 | 152 |
| **sum (22 queries)** | **2,935** | **3,555** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 81 | 62 |
| Q2 | 31 | 27 |
| Q3 | 60 | 77 |
| Q4 | 49 | 70 |
| Q5 | 59 | 85 |
| Q6 | 33 | 33 |
| Q7 | 66 | 93 |
| Q8 | 78 | 133 |
| Q9 | 79 | 137 |
| Q10 | 72 | 75 |
| Q11 | 24 | 26 |
| Q12 | 65 | 74 |
| Q13 | 31 | 29 |
| Q14 | 44 | 91 |
| Q15 | 40 | 43 |
| Q16 | 30 | 26 |
| Q17 | 43 | 88 |
| Q18 | 46 | 67 |
| Q19 | 74 | 99 |
| Q20 | 58 | 100 |
| Q21 | 58 | 83 |
| Q22 | 28 | 26 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 233 | 9.5 |
| Q2 | 7.7 | 9.5 |
| Q3 | 25 | 30 |
| Q4 | 61 | 31 |
| Q5 | 39 | 18 |
| Q6 | 5.6 | 4.9 |
| Q7 | 90 | 24 |
| Q8 | 14 | 22 |
| Q9 | 58 | 63 |
| Q10 | 35 | 26 |
| Q11 | 4.4 | 5.8 |
| Q12 | 19 | 16 |
| Q13 | 42 | 11 |
| Q14 | 8.1 | 8.1 |
| Q15 | 8.0 | 8.6 |
| Q16 | 15 | 44 |
| Q17 | 5.7 | 11 |
| Q18 | 121 | 150 |
| Q19 | 33 | 13 |
| Q20 | 15 | 12 |
| Q21 | 343 | 577 |
| Q22 | 13 | 12 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | mx-cpu |
|---|---:|---:|
| import | 73 | 103 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.8e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | mx-cpu | outside mx |
|---|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,719 WRONG | **62** | - |
| Q2 min-cost supplier | 58 | **27** | - |
| Q3 3-join+top10 | 290 | **261** | - |
| Q4 EXISTS semi-join | 1,015 | **260** | - |
| Q5 4-join+groupby | 501 | **206** | - |
| Q6 filter+global sum | 42 | **29** | - |
| Q7 volume shipping | 1,103 †3e-6 | **198** | - |
| Q8 market share | **136** | 196 | - |
| Q9 6-join profit | 933 †7e-6 | **685** | - |
| Q10 returned items top20 | 328 | **280** | - |
| Q11 important stock | **35** | 40 | - |
| Q12 shipping modes | 153 | **141** | - |
| Q13 customer distribution | 661 | **163** | PyArrow/NumPy |
| Q14 promo revenue | **79** | 90 | - |
| Q15 top supplier | 69 | **61** | - |
| Q16 part/supplier | **110** | 339 | - |
| Q17 small-qty revenue | **58** | 80 | - |
| Q18 large volume top100 | 1,537 | **814** | - |
| Q19 discounted revenue | 450 | **80** | - |
| Q20 potential promotion | 121 | **99** | - |
| Q21 suppliers waiting | **4,703** | 5,581 | - |
| Q22 global sales opportunity | 200 | **133** | PyArrow/NumPy |
| **sum over the 21 queries valid for every engine** | **12,583** | **9,762** |  |

Fastest verified engine per query: Polars 6, mx-cpu 16.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 15/22** (21 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 2,793 WRONG | 182 |
| Q2 | 132 | 135 |
| Q3 | 365 | 363 |
| Q4 | 1,092 | 362 |
| Q5 | 572 | 316 |
| Q6 | 113 | 149 |
| Q7 | 1,177 †3e-6 | 300 |
| Q8 | 212 | 334 |
| Q9 | 1,011 †7e-6 | 799 |
| Q10 | 402 | 397 |
| Q11 | 111 | 185 |
| Q12 | 233 | 246 |
| Q13 | 734 | 263 |
| Q14 | 154 | 198 |
| Q15 | 142 | 165 |
| Q16 | 187 | 442 |
| Q17 | 133 | 186 |
| Q18 | 1,612 | 935 |
| Q19 | 523 | 186 |
| Q20 | 196 | 210 |
| Q21 | 4,775 | 5,687 |
| Q22 | 276 | 248 |
| **sum (21 queries)** | **14,152** | **12,108** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 555 | 515 |
| Q2 | 81 | 112 |
| Q3 | 323 | 668 |
| Q4 | 266 | 602 |
| Q5 | 370 | 700 |
| Q6 | 199 | 269 |
| Q7 | 396 | 733 |
| Q8 | 501 | 1,061 |
| Q9 | 551 | 1,230 |
| Q10 | 421 | 697 |
| Q11 | 58 | 82 |
| Q12 | 465 | 713 |
| Q13 | 79 | 208 |
| Q14 | 281 | 791 |
| Q15 | 257 | 389 |
| Q16 | 69 | 105 |
| Q17 | 255 | 755 |
| Q18 | 201 | 620 |
| Q19 | 572 | 902 |
| Q20 | 374 | 915 |
| Q21 | 338 | 646 |
| Q22 | 79 | 261 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 2,341 WRONG | 62 |
| Q2 | 41 | 21 |
| Q3 | 255 | 230 |
| Q4 | 915 | 214 |
| Q5 | 461 | 175 |
| Q6 | 41 | 28 |
| Q7 | 963 | 196 |
| Q8 | 90 | 183 |
| Q9 | 771 | 624 |
| Q10 | 306 | 231 |
| Q11 | 28 | 27 |
| Q12 | 149 | 137 |
| Q13 | 606 | 150 |
| Q14 | 62 | 88 |
| Q15 | 58 | 63 |
| Q16 | 104 | 306 |
| Q17 | 31 | 81 |
| Q18 | 1,517 | 670 |
| Q19 | 418 | 81 |
| Q20 | 104 | 103 |
| Q21 | 3,999 | 5,497 |
| Q22 | 226 | 118 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | mx-cpu |
|---|---:|---:|
| import | 75 | 108 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.1e-14, Polars 5.5e-02.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.9, max 18.5 over 88 runs; load average at report time: 19.2 10.6 6.9), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `6f76e36`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

