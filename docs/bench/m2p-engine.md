# Engine-only TPC-H scoreboard - 2026-10-13

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 276 †3e-3 | 21 | **9.8** | - |
| Q2 min-cost supplier | **7.2** | 11 | 9.7 | - |
| Q3 3-join+top10 | 31 | **16** | 20 | - |
| Q4 EXISTS semi-join | 80 | **17** | 19 | - |
| Q5 4-join+groupby | 49 | **17** | 19 | - |
| Q6 filter+global sum | 6.2 | **5.2** | 11 | - |
| Q7 volume shipping | 113 | **18** | 20 | - |
| Q8 market share | **14** | 14 | 18 | - |
| Q9 6-join profit | 75 †2e-6 | 58 | **54** | - |
| Q10 returned items top20 | 36 | 23 | **22** | - |
| Q11 important stock | **4.6** | 6.2 | 14 | - |
| Q12 shipping modes | 19 | **15** | 18 | - |
| Q13 customer distribution | 72 | 34 | **9.8** | PyArrow/NumPy |
| Q14 promo revenue | **9.9** | 17 | 11 | - |
| Q15 top supplier | 8.9 | **8.7** | 9.5 | - |
| Q16 part/supplier | **17** | 25 | 27 | - |
| Q17 small-qty revenue | **8.7** | 9.8 | 9.4 | - |
| Q18 large volume top100 | 143 | **35** | 62 | - |
| Q19 discounted revenue | 48 | 21 | **14** | - |
| Q20 potential promotion | 16 | **12** | 14 | - |
| Q21 suppliers waiting | 373 | 226 | **92** | - |
| Q22 global sales opportunity | **14** | 17 | 18 | - |
| **sum over the 22 queries valid for every engine** | **1,421** | **627** | **500** |  |

Fastest verified engine per query: Polars 7, DuckDB 9, mx-cpu 6.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 13/22** (22 comparable); **mx-cpu beats DuckDB on 9/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 350 †3e-3 | 181 | 115 |
| Q2 | 82 | 122 | 114 |
| Q3 | 104 | 127 | 126 |
| Q4 | 153 | 132 | 153 |
| Q5 | 120 | 131 | 123 |
| Q6 | 77 | 116 | 113 |
| Q7 | 188 | 127 | 120 |
| Q8 | 87 | 125 | 120 |
| Q9 | 149 †2e-6 | 170 | 155 |
| Q10 | 109 | 133 | 125 |
| Q11 | 80 | 116 | 116 |
| Q12 | 95 | 130 | 120 |
| Q13 | 155 | 158 | 119 |
| Q14 | 85 | 135 | 113 |
| Q15 | 80 | 121 | 125 |
| Q16 | 94 | 134 | 129 |
| Q17 | 84 | 123 | 118 |
| Q18 | 216 | 147 | 171 |
| Q19 | 132 | 131 | 121 |
| Q20 | 96 | 131 | 117 |
| Q21 | 448 | 335 | 195 |
| Q22 | 88 | 127 | 121 |
| **sum (22 queries)** | **3,073** | **3,152** | **2,827** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 65 | 138 | 55 |
| Q2 | 32 | 77 | 34 |
| Q3 | 61 | 125 | 83 |
| Q4 | 49 | 124 | 73 |
| Q5 | 63 | 125 | 92 |
| Q6 | 35 | 82 | 34 |
| Q7 | 65 | 126 | 91 |
| Q8 | 76 | 156 | 124 |
| Q9 | 89 | 182 | 136 |
| Q10 | 69 | 132 | 80 |
| Q11 | 25 | 56 | 17 |
| Q12 | 68 | 150 | 77 |
| Q13 | 33 | 55 | 41 |
| Q14 | 42 | 108 | 85 |
| Q15 | 40 | 82 | 55 |
| Q16 | 29 | 58 | 24 |
| Q17 | 45 | 93 | 92 |
| Q18 | 44 | 79 | 83 |
| Q19 | 85 | 186 | 101 |
| Q20 | 59 | 121 | 100 |
| Q21 | 55 | 116 | 78 |
| Q22 | 25 | 43 | 29 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 202 | 24 | 9.2 |
| Q2 | 7.0 | 11 | 7.0 |
| Q3 | 26 | 16 | 12 |
| Q4 | 68 | 17 | 13 |
| Q5 | 42 | 14 | 12 |
| Q6 | 5.6 | 5.2 | 4.9 |
| Q7 | 92 | 18 | 14 |
| Q8 | 15 | 15 | 11 |
| Q9 | 62 | 61 | 49 |
| Q10 | 33 | 23 | 18 |
| Q11 | 4.1 | 4.7 | 4.6 |
| Q12 | 20 | 13 | 15 |
| Q13 | 54 | 29 | 6.9 |
| Q14 | 8.1 | 14 | 7.9 |
| Q15 | 7.6 | 9.0 | 7.8 |
| Q16 | 15 | 22 | 21 |
| Q17 | 6.6 | 9.1 | 6.5 |
| Q18 | 115 | 32 | 46 |
| Q19 | 41 | 19 | 12 |
| Q20 | 16 | 11 | 12 |
| Q21 | 326 | 243 | 70 |
| Q22 | 14 | 14 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 74 | 111 | 103 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 3.7e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,662 WRONG | 140 | **67** | - |
| Q2 min-cost supplier | 54 | 33 | **19** | - |
| Q3 3-join+top10 | 319 | **90** | 148 | - |
| Q4 EXISTS semi-join | 1,009 | **102** | 125 | - |
| Q5 4-join+groupby | 495 | **71** | 118 | - |
| Q6 filter+global sum | 48 | 32 | **31** | - |
| Q7 volume shipping | 1,084 †3e-6 | **85** | 119 | - |
| Q8 market share | 139 | **66** | 109 | - |
| Q9 6-join profit | 871 †7e-6 | 444 | **431** | - |
| Q10 returned items top20 | 350 | **112** | 148 | - |
| Q11 important stock | 32 | **16** | 27 | - |
| Q12 shipping modes | 141 | **78** | 118 | - |
| Q13 customer distribution | 671 | 240 | **93** | PyArrow/NumPy |
| Q14 promo revenue | 85 | 65 | **53** | - |
| Q15 top supplier | 77 | 54 | **50** | - |
| Q16 part/supplier | 101 | **97** | 101 | - |
| Q17 small-qty revenue | 67 | **29** | 64 | - |
| Q18 large volume top100 | 1,547 | **329** | 404 | - |
| Q19 discounted revenue | 472 | 85 | **69** | - |
| Q20 potential promotion | 129 | **51** | 65 | - |
| Q21 suppliers waiting | 4,324 | 3,038 | **810** | - |
| Q22 global sales opportunity | 263 | **52** | 103 | - |
| **sum over the 21 queries valid for every engine** | **12,278** | **5,169** | **3,202** |  |

Fastest verified engine per query: Polars 0, DuckDB 13, mx-cpu 9.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 21/22** (21 comparable); **mx-cpu beats DuckDB on 9/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,734 WRONG | 255 | 174 |
| Q2 | 132 | 149 | 136 |
| Q3 | 400 | 206 | 249 |
| Q4 | 1,088 | 213 | 228 |
| Q5 | 568 | 181 | 225 |
| Q6 | 120 | 141 | 133 |
| Q7 | 1,157 †3e-6 | 197 | 221 |
| Q8 | 211 | 177 | 210 |
| Q9 | 943 †7e-6 | 555 | 535 |
| Q10 | 422 | 224 | 251 |
| Q11 | 104 | 127 | 130 |
| Q12 | 213 | 189 | 219 |
| Q13 | 743 | 350 | 194 |
| Q14 | 156 | 175 | 154 |
| Q15 | 148 | 164 | 153 |
| Q16 | 174 | 206 | 209 |
| Q17 | 140 | 139 | 165 |
| Q18 | 1,626 | 440 | 509 |
| Q19 | 546 | 196 | 170 |
| Q20 | 207 | 161 | 201 |
| Q21 | 4,400 | 3,185 | 942 |
| Q22 | 345 | 166 | 216 |
| **sum (21 queries)** | **13,844** | **7,541** | **5,449** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 537 | 648 | 498 |
| Q2 | 84 | 253 | 142 |
| Q3 | 342 | 508 | 671 |
| Q4 | 279 | 362 | 610 |
| Q5 | 367 | 517 | 702 |
| Q6 | 205 | 343 | 276 |
| Q7 | 398 | 565 | 734 |
| Q8 | 470 | 773 | 1,035 |
| Q9 | 529 | 856 | 1,204 |
| Q10 | 425 | 516 | 682 |
| Q11 | 51 | 92 | 77 |
| Q12 | 445 | 513 | 694 |
| Q13 | 77 | 89 | 209 |
| Q14 | 270 | 474 | 770 |
| Q15 | 248 | 404 | 382 |
| Q16 | 71 | 185 | 106 |
| Q17 | 243 | 420 | 752 |
| Q18 | 207 | 275 | 601 |
| Q19 | 588 | 797 | 885 |
| Q20 | 371 | 619 | 1,013 |
| Q21 | 359 | 487 | 719 |
| Q22 | 84 | 140 | 250 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,128 WRONG | 141 | 64 |
| Q2 | 42 | 37 | 20 |
| Q3 | 258 | 91 | 135 |
| Q4 | 921 | 104 | 118 |
| Q5 | 419 | 70 | 99 |
| Q6 | 42 | 31 | 29 |
| Q7 | 931 | 85 | 109 |
| Q8 | 102 | 62 | 97 |
| Q9 | 798 | 447 | 380 |
| Q10 | 307 | 112 | 148 |
| Q11 | 27 | 15 | 19 |
| Q12 | 142 | 77 | 115 |
| Q13 | 568 | 243 | 88 |
| Q14 | 64 | 63 | 54 |
| Q15 | 57 | 49 | 46 |
| Q16 | 80 | 108 | 86 |
| Q17 | 36 | 28 | 62 |
| Q18 | 1,497 | 328 | 374 |
| Q19 | 435 | 87 | 68 |
| Q20 | 106 | 52 | 73 |
| Q21 | 4,383 | 3,066 | 764 |
| Q22 | 241 | 41 | 97 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 73 | 111 | 103 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.1e-14, Polars 5.5e-02.


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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 7.1, max 17.4 over 132 runs; load average at report time: 16.2 7.2 4.5), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `dc88dca`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

