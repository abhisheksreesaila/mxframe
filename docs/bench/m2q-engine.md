# Engine-only TPC-H scoreboard - 2026-10-13

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 250 †3e-3 | 20 | **11** | - |
| Q2 min-cost supplier | **7.4** | 12 | 8.8 | - |
| Q3 3-join+top10 | 31 | **17** | 19 | - |
| Q4 EXISTS semi-join | 83 | **17** | 18 | - |
| Q5 4-join+groupby | 48 | **16** | 18 | - |
| Q6 filter+global sum | 6.6 | **5.6** | 6.2 | - |
| Q7 volume shipping | 102 | **19** | 21 | - |
| Q8 market share | **13** | 15 | 17 | - |
| Q9 6-join profit | 66 †2e-6 | 60 | **53** | - |
| Q10 returned items top20 | 35 | 23 | **22** | - |
| Q11 important stock | **4.4** | 6.5 | 9.5 | - |
| Q12 shipping modes | 17 | **15** | 20 | - |
| Q13 customer distribution | 57 | 28 | **8.9** | PyArrow/NumPy |
| Q14 promo revenue | **10** | 16 | 11 | - |
| Q15 top supplier | **8.1** | 8.8 | 9.4 | - |
| Q16 part/supplier | **15** | 25 | 25 | - |
| Q17 small-qty revenue | **8.2** | 9.2 | 9.5 | - |
| Q18 large volume top100 | 133 | **35** | 57 | - |
| Q19 discounted revenue | 38 | 22 | **15** | - |
| Q20 potential promotion | 15 | **13** | 14 | - |
| Q21 suppliers waiting | 367 | 242 | **82** | - |
| Q22 global sales opportunity | **13** | 15 | 17 | - |
| **sum over the 22 queries valid for every engine** | **1,326** | **639** | **473** |  |

Fastest verified engine per query: Polars 8, DuckDB 8, mx-cpu 6.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 13/22** (22 comparable); **mx-cpu beats DuckDB on 9/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 325 †3e-3 | 136 | 116 |
| Q2 | 85 | 123 | 114 |
| Q3 | 104 | 131 | 125 |
| Q4 | 158 | 128 | 122 |
| Q5 | 121 | 145 | 124 |
| Q6 | 81 | 118 | 111 |
| Q7 | 176 | 130 | 128 |
| Q8 | 87 | 130 | 122 |
| Q9 | 141 †2e-6 | 172 | 156 |
| Q10 | 109 | 139 | 128 |
| Q11 | 79 | 117 | 115 |
| Q12 | 93 | 128 | 126 |
| Q13 | 131 | 139 | 113 |
| Q14 | 83 | 127 | 118 |
| Q15 | 82 | 123 | 119 |
| Q16 | 90 | 135 | 131 |
| Q17 | 82 | 121 | 115 |
| Q18 | 206 | 146 | 163 |
| Q19 | 115 | 134 | 122 |
| Q20 | 92 | 122 | 123 |
| Q21 | 441 | 354 | 185 |
| Q22 | 90 | 127 | 124 |
| **sum (22 queries)** | **2,970** | **3,125** | **2,799** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 69 | 152 | 58 |
| Q2 | 33 | 76 | 30 |
| Q3 | 62 | 134 | 85 |
| Q4 | 55 | 121 | 70 |
| Q5 | 63 | 124 | 90 |
| Q6 | 36 | 85 | 38 |
| Q7 | 65 | 118 | 93 |
| Q8 | 76 | 151 | 130 |
| Q9 | 83 | 188 | 144 |
| Q10 | 71 | 139 | 91 |
| Q11 | 26 | 57 | 26 |
| Q12 | 64 | 158 | 79 |
| Q13 | 28 | 58 | 36 |
| Q14 | 48 | 90 | 96 |
| Q15 | 41 | 83 | 48 |
| Q16 | 30 | 66 | 23 |
| Q17 | 45 | 90 | 88 |
| Q18 | 46 | 87 | 76 |
| Q19 | 75 | 179 | 102 |
| Q20 | 60 | 131 | 105 |
| Q21 | 58 | 118 | 77 |
| Q22 | 26 | 44 | 23 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 203 | 20 | 9.8 |
| Q2 | 6.3 | 12 | 6.2 |
| Q3 | 25 | 16 | 13 |
| Q4 | 77 | 15 | 15 |
| Q5 | 41 | 16 | 13 |
| Q6 | 5.8 | 4.8 | 4.3 |
| Q7 | 93 | 18 | 14 |
| Q8 | 15 | 16 | 12 |
| Q9 | 64 | 59 | 43 |
| Q10 | 34 | 24 | 20 |
| Q11 | 4.8 | 5.0 | 5.7 |
| Q12 | 18 | 14 | 16 |
| Q13 | 43 | 25 | 6.1 |
| Q14 | 7.3 | 13 | 7.3 |
| Q15 | 8.7 | 9.2 | 7.7 |
| Q16 | 16 | 22 | 22 |
| Q17 | 5.8 | 8.8 | 7.2 |
| Q18 | 124 | 37 | 44 |
| Q19 | 37 | 20 | 13 |
| Q20 | 14 | 12 | 11 |
| Q21 | 351 | 246 | 69 |
| Q22 | 11 | 13 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 74 | 112 | 106 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 3.0e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,673 WRONG | 143 | **67** | - |
| Q2 min-cost supplier | 51 | 30 | **19** | - |
| Q3 3-join+top10 | 295 | **85** | 127 | - |
| Q4 EXISTS semi-join | 994 | **104** | 117 | - |
| Q5 4-join+groupby | 478 | **72** | 114 | - |
| Q6 filter+global sum | 43 | 32 | **30** | - |
| Q7 volume shipping | 1,084 †3e-6 | **87** | 117 | - |
| Q8 market share | 116 | **64** | 111 | - |
| Q9 6-join profit | 839 †7e-6 | 458 | **436** | - |
| Q10 returned items top20 | 358 | **119** | 155 | - |
| Q11 important stock | 32 | **16** | 28 | - |
| Q12 shipping modes | 150 | **80** | 115 | - |
| Q13 customer distribution | 664 | 249 | **96** | PyArrow/NumPy |
| Q14 promo revenue | 80 | 64 | **54** | - |
| Q15 top supplier | 62 | 53 | **49** | - |
| Q16 part/supplier | **101** | 102 | 102 | - |
| Q17 small-qty revenue | 49 | **29** | 66 | - |
| Q18 large volume top100 | 1,527 | **330** | 413 | - |
| Q19 discounted revenue | 436 | 85 | **70** | - |
| Q20 potential promotion | 104 | **51** | 67 | - |
| Q21 suppliers waiting | 3,915 | 2,814 | **744** | - |
| Q22 global sales opportunity | 194 | **42** | 97 | - |
| **sum over the 21 queries valid for every engine** | **11,571** | **4,967** | **3,129** |  |

Fastest verified engine per query: Polars 1, DuckDB 12, mx-cpu 9.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 19/22** (21 comparable); **mx-cpu beats DuckDB on 10/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,748 WRONG | 254 | 177 |
| Q2 | 125 | 143 | 125 |
| Q3 | 369 | 197 | 238 |
| Q4 | 1,070 | 217 | 221 |
| Q5 | 550 | 183 | 239 |
| Q6 | 116 | 141 | 135 |
| Q7 | 1,158 †3e-6 | 197 | 225 |
| Q8 | 189 | 179 | 216 |
| Q9 | 912 †7e-6 | 572 | 546 |
| Q10 | 431 | 229 | 260 |
| Q11 | 110 | 126 | 133 |
| Q12 | 223 | 189 | 218 |
| Q13 | 739 | 359 | 201 |
| Q14 | 153 | 177 | 160 |
| Q15 | 136 | 165 | 156 |
| Q16 | 176 | 215 | 207 |
| Q17 | 122 | 150 | 173 |
| Q18 | 1,600 | 444 | 523 |
| Q19 | 541 | 196 | 175 |
| Q20 | 178 | 161 | 171 |
| Q21 | 3,993 | 2,924 | 848 |
| Q22 | 269 | 156 | 200 |
| **sum (21 queries)** | **13,162** | **7,322** | **5,371** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 547 | 637 | 485 |
| Q2 | 85 | 250 | 118 |
| Q3 | 327 | 506 | 684 |
| Q4 | 264 | 352 | 615 |
| Q5 | 362 | 538 | 694 |
| Q6 | 205 | 338 | 254 |
| Q7 | 401 | 580 | 736 |
| Q8 | 467 | 762 | 1,052 |
| Q9 | 531 | 869 | 1,192 |
| Q10 | 440 | 506 | 711 |
| Q11 | 54 | 92 | 79 |
| Q12 | 449 | 504 | 718 |
| Q13 | 80 | 93 | 210 |
| Q14 | 272 | 483 | 771 |
| Q15 | 253 | 417 | 386 |
| Q16 | 71 | 183 | 103 |
| Q17 | 245 | 420 | 747 |
| Q18 | 203 | 271 | 624 |
| Q19 | 609 | 782 | 877 |
| Q20 | 355 | 579 | 913 |
| Q21 | 337 | 437 | 653 |
| Q22 | 80 | 139 | 243 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,169 WRONG | 141 | 66 |
| Q2 | 44 | 32 | 16 |
| Q3 | 252 | 85 | 126 |
| Q4 | 894 | 97 | 103 |
| Q5 | 424 | 70 | 103 |
| Q6 | 41 | 32 | 30 |
| Q7 | 937 | 87 | 108 |
| Q8 | 91 | 62 | 100 |
| Q9 | 781 | 448 | 394 |
| Q10 | 361 | 114 | 159 |
| Q11 | 27 | 16 | 20 |
| Q12 | 147 | 80 | 114 |
| Q13 | 603 | 237 | 90 |
| Q14 | 64 | 66 | 51 |
| Q15 | 58 | 49 | 47 |
| Q16 | 90 | 117 | 88 |
| Q17 | 30 | 28 | 63 |
| Q18 | 1,523 | 333 | 381 |
| Q19 | 467 | 86 | 68 |
| Q20 | 101 | 49 | 65 |
| Q21 | 3,977 | 2,820 | 666 |
| Q22 | 191 | 41 | 90 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 74 | 112 | 105 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.0e-14, Polars 5.5e-02.


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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 6.2, max 12.9 over 132 runs; load average at report time: 9.8 7.7 7.7), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `1313971`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

