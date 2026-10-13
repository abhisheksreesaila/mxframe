# Engine-only TPC-H scoreboard - 2026-10-13

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 259 †3e-3 | 121 | **11** | - |
| Q2 min-cost supplier | **7.2** | 82 | 8.0 | - |
| Q3 3-join+top10 | 28 | 87 | **19** | - |
| Q4 EXISTS semi-join | 76 | 89 | **19** | - |
| Q5 4-join+groupby | 49 | 86 | **18** | - |
| Q6 filter+global sum | 6.7 | 77 | **6.4** | - |
| Q7 volume shipping | 104 | 92 | **21** | - |
| Q8 market share | **14** | 86 | 17 | - |
| Q9 6-join profit | 73 †2e-6 | 132 | **55** | - |
| Q10 returned items top20 | 39 | 93 | **22** | - |
| Q11 important stock | **5.3** | 75 | 5.8 | - |
| Q12 shipping modes | **17** | 85 | 18 | - |
| Q13 customer distribution | 54 | 100 | **8.6** | PyArrow/NumPy |
| Q14 promo revenue | **9.9** | 87 | 11 | - |
| Q15 top supplier | **8.4** | 80 | 9.2 | - |
| Q16 part/supplier | **17** | 102 | 24 | - |
| Q17 small-qty revenue | **7.6** | 80 | 9.2 | - |
| Q18 large volume top100 | 131 | 116 | **54** | - |
| Q19 discounted revenue | 41 | 92 | **15** | - |
| Q20 potential promotion | 15 | 83 | **13** | - |
| Q21 suppliers waiting | 354 | 308 | **103** | - |
| Q22 global sales opportunity | **13** | 84 | 18 | - |
| **sum over the 22 queries valid for every engine** | **1,329** | **2,236** | **485** |  |

Fastest verified engine per query: Polars 9, DuckDB 0, mx-cpu 13.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 13/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 336 †3e-3 | 164 | 115 |
| Q2 | 80 | 121 | 112 |
| Q3 | 99 | 126 | 121 |
| Q4 | 148 | 128 | 124 |
| Q5 | 123 | 124 | 120 |
| Q6 | 82 | 117 | 109 |
| Q7 | 176 | 132 | 122 |
| Q8 | 89 | 128 | 119 |
| Q9 | 148 †2e-6 | 171 | 159 |
| Q10 | 110 | 133 | 123 |
| Q11 | 79 | 112 | 112 |
| Q12 | 89 | 125 | 122 |
| Q13 | 124 | 138 | 112 |
| Q14 | 85 | 126 | 114 |
| Q15 | 80 | 118 | 110 |
| Q16 | 93 | 143 | 126 |
| Q17 | 82 | 118 | 114 |
| Q18 | 202 | 154 | 155 |
| Q19 | 113 | 130 | 116 |
| Q20 | 86 | 123 | 117 |
| Q21 | 424 | 346 | 207 |
| Q22 | 84 | 122 | 120 |
| **sum (22 queries)** | **2,931** | **3,100** | **2,749** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 65 | 131 | 56 |
| Q2 | 31 | 65 | 31 |
| Q3 | 61 | 118 | 87 |
| Q4 | 50 | 104 | 71 |
| Q5 | 64 | 109 | 93 |
| Q6 | 31 | 64 | 37 |
| Q7 | 64 | 108 | 89 |
| Q8 | 73 | 141 | 124 |
| Q9 | 86 | 170 | 143 |
| Q10 | 69 | 118 | 85 |
| Q11 | 26 | 48 | 25 |
| Q12 | 69 | 134 | 78 |
| Q13 | 28 | 38 | 37 |
| Q14 | 46 | 79 | 86 |
| Q15 | 41 | 71 | 43 |
| Q16 | 26 | 51 | 27 |
| Q17 | 40 | 76 | 87 |
| Q18 | 45 | 67 | 72 |
| Q19 | 78 | 159 | 95 |
| Q20 | 57 | 109 | 106 |
| Q21 | 56 | 97 | 77 |
| Q22 | 24 | 33 | 23 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 209 | 20 | 9.1 |
| Q2 | 7.3 | 10 | 6.9 |
| Q3 | 26 | 16 | 13 |
| Q4 | 62 | 15 | 12 |
| Q5 | 40 | 14 | 12 |
| Q6 | 6.0 | 5.2 | 5.1 |
| Q7 | 89 | 18 | 14 |
| Q8 | 12 | 16 | 11 |
| Q9 | 56 | 59 | 43 |
| Q10 | 33 | 22 | 17 |
| Q11 | 4.1 | 4.8 | 4.7 |
| Q12 | 17 | 14 | 14 |
| Q13 | 42 | 27 | 5.9 |
| Q14 | 7.6 | 13 | 8.1 |
| Q15 | 7.6 | 8.5 | 7.2 |
| Q16 | 14 | 26 | 21 |
| Q17 | 5.1 | 9.1 | 6.3 |
| Q18 | 118 | 33 | 45 |
| Q19 | 39 | 20 | 12 |
| Q20 | 14 | 12 | 10 |
| Q21 | 316 | 249 | 95 |
| Q22 | 14 | 13 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 72 | 39 | 103 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.1e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,645 WRONG | 223 | **67** | - |
| Q2 min-cost supplier | 55 | 99 | **19** | - |
| Q3 3-join+top10 | 300 | 161 | **126** | - |
| Q4 EXISTS semi-join | 974 | 167 | **113** | - |
| Q5 4-join+groupby | 502 | 144 | **110** | - |
| Q6 filter+global sum | 48 | 103 | **31** | - |
| Q7 volume shipping | 1,065 †3e-6 | 160 | **115** | - |
| Q8 market share | 145 | 135 | **111** | - |
| Q9 6-join profit | 862 †7e-6 | 517 | **422** | - |
| Q10 returned items top20 | 348 | 186 | **148** | - |
| Q11 important stock | 32 | 85 | **26** | - |
| Q12 shipping modes | 143 | 167 | **117** | - |
| Q13 customer distribution | 659 | 309 | **97** | PyArrow/NumPy |
| Q14 promo revenue | 84 | 138 | **55** | - |
| Q15 top supplier | 74 | 132 | **48** | - |
| Q16 part/supplier | **98** | 176 | 99 | - |
| Q17 small-qty revenue | 67 | 105 | **63** | - |
| Q18 large volume top100 | 1,490 | **386** | 403 | - |
| Q19 discounted revenue | 447 | 156 | **68** | - |
| Q20 potential promotion | 123 | 121 | **66** | - |
| Q21 suppliers waiting | 3,929 | 2,862 | **962** | - |
| Q22 global sales opportunity | 193 | 116 | **95** | - |
| **sum over the 21 queries valid for every engine** | **11,637** | **6,424** | **3,293** |  |

Fastest verified engine per query: Polars 1, DuckDB 1, mx-cpu 20.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 20/22** (21 comparable); **mx-cpu beats DuckDB on 21/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,716 WRONG | 261 | 168 |
| Q2 | 126 | 137 | 123 |
| Q3 | 372 | 201 | 227 |
| Q4 | 1,045 | 207 | 216 |
| Q5 | 578 | 182 | 211 |
| Q6 | 120 | 141 | 134 |
| Q7 | 1,137 †3e-6 | 199 | 215 |
| Q8 | 217 | 173 | 219 |
| Q9 | 934 †7e-6 | 556 | 525 |
| Q10 | 422 | 223 | 249 |
| Q11 | 104 | 122 | 130 |
| Q12 | 216 | 205 | 221 |
| Q13 | 731 | 348 | 196 |
| Q14 | 155 | 178 | 161 |
| Q15 | 145 | 169 | 151 |
| Q16 | 170 | 214 | 216 |
| Q17 | 139 | 143 | 166 |
| Q18 | 1,565 | 425 | 505 |
| Q19 | 521 | 194 | 172 |
| Q20 | 196 | 159 | 169 |
| Q21 | 4,004 | 2,901 | 1,066 |
| Q22 | 266 | 156 | 199 |
| **sum (21 queries)** | **13,161** | **7,233** | **5,474** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 536 | 624 | 503 |
| Q2 | 83 | 223 | 110 |
| Q3 | 327 | 490 | 659 |
| Q4 | 264 | 325 | 606 |
| Q5 | 360 | 504 | 684 |
| Q6 | 206 | 309 | 279 |
| Q7 | 395 | 562 | 720 |
| Q8 | 469 | 728 | 1,057 |
| Q9 | 527 | 835 | 1,174 |
| Q10 | 429 | 489 | 676 |
| Q11 | 55 | 86 | 78 |
| Q12 | 442 | 476 | 704 |
| Q13 | 76 | 88 | 206 |
| Q14 | 272 | 452 | 772 |
| Q15 | 257 | 384 | 392 |
| Q16 | 68 | 165 | 103 |
| Q17 | 244 | 413 | 755 |
| Q18 | 205 | 247 | 600 |
| Q19 | 579 | 755 | 887 |
| Q20 | 359 | 548 | 930 |
| Q21 | 339 | 432 | 645 |
| Q22 | 79 | 128 | 240 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,094 WRONG | 140 | 64 |
| Q2 | 44 | 30 | 15 |
| Q3 | 249 | 85 | 116 |
| Q4 | 883 | 96 | 99 |
| Q5 | 427 | 70 | 98 |
| Q6 | 42 | 31 | 29 |
| Q7 | 920 | 85 | 106 |
| Q8 | 104 | 62 | 94 |
| Q9 | 776 | 453 | 386 |
| Q10 | 305 | 114 | 146 |
| Q11 | 26 | 15 | 19 |
| Q12 | 146 | 78 | 114 |
| Q13 | 564 | 232 | 86 |
| Q14 | 64 | 62 | 53 |
| Q15 | 57 | 49 | 46 |
| Q16 | 83 | 116 | 88 |
| Q17 | 35 | 28 | 61 |
| Q18 | 1,495 | 314 | 373 |
| Q19 | 406 | 83 | 70 |
| Q20 | 104 | 47 | 65 |
| Q21 | 3,659 | 2,775 | 948 |
| Q22 | 182 | 42 | 91 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 72 | 38 | 103 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.2e-14, Polars 5.5e-02.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.3, max 11.6 over 132 runs; load average at report time: 10.8 6.5 5.6), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `a3d9d01`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

