# Engine-only TPC-H scoreboard - 2026-10-13

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 245 †3e-3 | 121 | **10** | - |
| Q2 min-cost supplier | **8.2** | 81 | 12 | - |
| Q3 3-join+top10 | 29 | 89 | **19** | - |
| Q4 EXISTS semi-join | 75 | 89 | **29** | - |
| Q5 4-join+groupby | 53 | 87 | **18** | - |
| Q6 filter+global sum | **6.7** | 75 | 7.0 | - |
| Q7 volume shipping | 112 | 119 | **25** | - |
| Q8 market share | **15** | 89 | 20 | - |
| Q9 6-join profit | 73 †2e-6 | 134 | **59** | - |
| Q10 returned items top20 | 42 | 96 | **28** | - |
| Q11 important stock | **5.1** | 79 | 8.6 | - |
| Q12 shipping modes | 22 | 86 | **20** | - |
| Q13 customer distribution | 58 | 104 | **12** | PyArrow/NumPy |
| Q14 promo revenue | **13** | 91 | 13 | - |
| Q15 top supplier | 10 | 81 | **10** | - |
| Q16 part/supplier | **19** | 103 | 28 | - |
| Q17 small-qty revenue | **7.3** | 84 | 11 | - |
| Q18 large volume top100 | 130 | 115 | **77** | - |
| Q19 discounted revenue | 35 | 95 | **16** | - |
| Q20 potential promotion | 15 | 86 | **14** | - |
| Q21 suppliers waiting | 355 | 319 | **279** | - |
| Q22 global sales opportunity | **12** | 87 | 18 | - |
| **sum over the 22 queries valid for every engine** | **1,339** | **2,311** | **734** |  |

Fastest verified engine per query: Polars 8, DuckDB 0, mx-cpu 14.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 14/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 317 †3e-3 | 166 | 113 |
| Q2 | 83 | 126 | 115 |
| Q3 | 104 | 127 | 123 |
| Q4 | 146 | 128 | 148 |
| Q5 | 125 | 127 | 129 |
| Q6 | 89 | 117 | 112 |
| Q7 | 196 | 162 | 127 |
| Q8 | 100 | 131 | 157 |
| Q9 | 147 †2e-6 | 176 | 182 |
| Q10 | 119 | 137 | 138 |
| Q11 | 85 | 126 | 146 |
| Q12 | 98 | 128 | 139 |
| Q13 | 135 | 146 | 158 |
| Q14 | 94 | 134 | 127 |
| Q15 | 88 | 126 | 116 |
| Q16 | 95 | 142 | 138 |
| Q17 | 79 | 129 | 113 |
| Q18 | 203 | 157 | 180 |
| Q19 | 106 | 133 | 126 |
| Q20 | 86 | 124 | 117 |
| Q21 | 429 | 357 | 397 |
| Q22 | 85 | 126 | 119 |
| **sum (22 queries)** | **3,007** | **3,225** | **3,220** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 70 | 131 | 54 |
| Q2 | 36 | 74 | 29 |
| Q3 | 54 | 112 | 82 |
| Q4 | 49 | 105 | 72 |
| Q5 | 62 | 108 | 94 |
| Q6 | 43 | 65 | 40 |
| Q7 | 72 | 130 | 102 |
| Q8 | 78 | 147 | 136 |
| Q9 | 87 | 176 | 150 |
| Q10 | 74 | 145 | 90 |
| Q11 | 27 | 49 | 27 |
| Q12 | 71 | 145 | 92 |
| Q13 | 32 | 40 | 40 |
| Q14 | 47 | 84 | 99 |
| Q15 | 42 | 69 | 54 |
| Q16 | 27 | 48 | 30 |
| Q17 | 44 | 81 | 81 |
| Q18 | 45 | 78 | 69 |
| Q19 | 77 | 160 | 99 |
| Q20 | 58 | 111 | 107 |
| Q21 | 54 | 98 | 83 |
| Q22 | 25 | 33 | 25 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 197 | 18 | 10 |
| Q2 | 6.3 | 10 | 8.8 |
| Q3 | 27 | 14 | 14 |
| Q4 | 61 | 15 | 23 |
| Q5 | 40 | 14 | 14 |
| Q6 | 6.2 | 5.1 | 5.8 |
| Q7 | 94 | 18 | 15 |
| Q8 | 12 | 15 | 13 |
| Q9 | 70 | 66 | 46 |
| Q10 | 35 | 22 | 23 |
| Q11 | 4.3 | 4.9 | 6.8 |
| Q12 | 19 | 15 | 17 |
| Q13 | 47 | 33 | 5.8 |
| Q14 | 7.9 | 14 | 8.7 |
| Q15 | 8.7 | 8.4 | 7.5 |
| Q16 | 17 | 22 | 26 |
| Q17 | 7.2 | 8.7 | 8.9 |
| Q18 | 113 | 32 | 64 |
| Q19 | 40 | 20 | 13 |
| Q20 | 15 | 12 | 10 |
| Q21 | 312 | 239 | 261 |
| Q22 | 13 | 13 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 75 | 42 | 110 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.8e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,803 WRONG | 222 | **67** | - |
| Q2 min-cost supplier | 61 | 105 | **19** | - |
| Q3 3-join+top10 | 343 | 177 | **145** | - |
| Q4 EXISTS semi-join | 1,085 | **170** | 231 | - |
| Q5 4-join+groupby | 511 | 154 | **116** | - |
| Q6 filter+global sum | 48 | 121 | **31** | - |
| Q7 volume shipping | 1,149 †3e-6 | 155 | **117** | - |
| Q8 market share | 145 | 141 | **109** | - |
| Q9 6-join profit | 1,044 †7e-6 | 512 | **442** | - |
| Q10 returned items top20 | 345 | **188** | 206 | - |
| Q11 important stock | 32 | 88 | **31** | - |
| Q12 shipping modes | 170 | 159 | **119** | - |
| Q13 customer distribution | 652 | 314 | **101** | PyArrow/NumPy |
| Q14 promo revenue | 79 | 138 | **57** | - |
| Q15 top supplier | 73 | 135 | **47** | - |
| Q16 part/supplier | **98** | 184 | 99 | - |
| Q17 small-qty revenue | 67 | 104 | **64** | - |
| Q18 large volume top100 | 1,571 | **395** | 728 | - |
| Q19 discounted revenue | 484 | 157 | **72** | - |
| Q20 potential promotion | 122 | 122 | **67** | - |
| Q21 suppliers waiting | 4,124 | **2,803** | 3,447 | - |
| Q22 global sales opportunity | 199 | **117** | 138 | - |
| **sum over the 21 queries valid for every engine** | **12,402** | **6,440** | **6,387** |  |

Fastest verified engine per query: Polars 1, DuckDB 5, mx-cpu 16.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 20/22** (21 comparable); **mx-cpu beats DuckDB on 17/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,883 WRONG | 260 | 184 |
| Q2 | 145 | 143 | 138 |
| Q3 | 429 | 217 | 267 |
| Q4 | 1,162 | 210 | 367 |
| Q5 | 587 | 192 | 226 |
| Q6 | 139 | 163 | 135 |
| Q7 | 1,224 †3e-6 | 196 | 253 |
| Q8 | 219 | 182 | 231 |
| Q9 | 1,120 †7e-6 | 553 | 545 |
| Q10 | 422 | 235 | 308 |
| Q11 | 109 | 129 | 166 |
| Q12 | 243 | 200 | 220 |
| Q13 | 725 | 352 | 203 |
| Q14 | 150 | 178 | 158 |
| Q15 | 146 | 173 | 151 |
| Q16 | 172 | 222 | 204 |
| Q17 | 139 | 145 | 174 |
| Q18 | 1,643 | 434 | 833 |
| Q19 | 562 | 202 | 177 |
| Q20 | 194 | 160 | 176 |
| Q21 | 4,196 | 2,841 | 3,549 |
| Q22 | 273 | 155 | 241 |
| **sum (21 queries)** | **13,999** | **7,280** | **8,720** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 545 | 622 | 561 |
| Q2 | 92 | 225 | 121 |
| Q3 | 402 | 542 | 720 |
| Q4 | 284 | 315 | 664 |
| Q5 | 389 | 501 | 753 |
| Q6 | 209 | 362 | 272 |
| Q7 | 398 | 621 | 786 |
| Q8 | 470 | 795 | 1,089 |
| Q9 | 565 | 793 | 1,215 |
| Q10 | 435 | 513 | 686 |
| Q11 | 57 | 89 | 90 |
| Q12 | 442 | 588 | 682 |
| Q13 | 77 | 87 | 210 |
| Q14 | 266 | 454 | 778 |
| Q15 | 244 | 399 | 381 |
| Q16 | 70 | 167 | 98 |
| Q17 | 238 | 437 | 803 |
| Q18 | 208 | 253 | 604 |
| Q19 | 590 | 794 | 922 |
| Q20 | 366 | 543 | 948 |
| Q21 | 330 | 422 | 653 |
| Q22 | 81 | 121 | 251 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,491 WRONG | 142 | 69 |
| Q2 | 46 | 34 | 17 |
| Q3 | 260 | 89 | 161 |
| Q4 | 1,022 | 102 | 200 |
| Q5 | 466 | 88 | 99 |
| Q6 | 46 | 36 | 35 |
| Q7 | 1,003 | 90 | 114 |
| Q8 | 95 | 60 | 96 |
| Q9 | 949 | 452 | 403 |
| Q10 | 315 | 126 | 232 |
| Q11 | 26 | 21 | 22 |
| Q12 | 154 | 76 | 116 |
| Q13 | 582 | 225 | 98 |
| Q14 | 59 | 60 | 52 |
| Q15 | 59 | 48 | 48 |
| Q16 | 88 | 114 | 91 |
| Q17 | 36 | 29 | 62 |
| Q18 | 1,501 | 321 | 625 |
| Q19 | 419 | 83 | 75 |
| Q20 | 101 | 51 | 64 |
| Q21 | 3,680 | 2,774 | 3,189 |
| Q22 | 179 | 41 | 127 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 75 | 40 | 105 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.3e-14, Polars 5.5e-02.


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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 6.9, max 17.4 over 132 runs; load average at report time: 16.1 9.7 7.7), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `35c7ccf`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

