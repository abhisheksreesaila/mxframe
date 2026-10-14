# Engine-only TPC-H scoreboard - 2026-10-14

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 242 †3e-3 | 18 | **10** | - |
| Q2 min-cost supplier | **7.3** | 12 | 8.7 | - |
| Q3 3-join+top10 | 32 | **16** | 17 | - |
| Q4 EXISTS semi-join | 75 | **16** | 19 | - |
| Q5 4-join+groupby | 47 | **15** | 19 | - |
| Q6 filter+global sum | 6.4 | **5.0** | 6.5 | - |
| Q7 volume shipping | 107 | **19** | 21 | - |
| Q8 market share | **13** | 15 | 17 | - |
| Q9 6-join profit | 68 †2e-6 | 58 | **53** | - |
| Q10 returned items top20 | 36 | 24 | **24** | - |
| Q11 important stock | **4.2** | 6.3 | 11 | - |
| Q12 shipping modes | 18 | **14** | 20 | - |
| Q13 customer distribution | 54 | 27 | **8.4** | PyArrow/NumPy |
| Q14 promo revenue | **10** | 16 | 11 | - |
| Q15 top supplier | **8.3** | 8.8 | 9.4 | - |
| Q16 part/supplier | **15** | 25 | 24 | - |
| Q17 small-qty revenue | **8.4** | 8.9 | 8.8 | - |
| Q18 large volume top100 | 133 | **35** | 53 | - |
| Q19 discounted revenue | 40 | 21 | **15** | - |
| Q20 potential promotion | 14 | **12** | 14 | - |
| Q21 suppliers waiting | 354 | 241 | **85** | - |
| Q22 global sales opportunity | 16 | **15** | 18 | - |
| **sum over the 22 queries valid for every engine** | **1,309** | **628** | **472** |  |

Fastest verified engine per query: Polars 7, DuckDB 9, mx-cpu 6.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 11/22** (22 comparable); **mx-cpu beats DuckDB on 10/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 313 †3e-3 | 127 | 114 |
| Q2 | 79 | 123 | 124 |
| Q3 | 106 | 127 | 127 |
| Q4 | 146 | 123 | 122 |
| Q5 | 119 | 128 | 121 |
| Q6 | 81 | 113 | 110 |
| Q7 | 178 | 129 | 122 |
| Q8 | 85 | 121 | 118 |
| Q9 | 139 †2e-6 | 167 | 157 |
| Q10 | 107 | 131 | 154 |
| Q11 | 75 | 114 | 112 |
| Q12 | 89 | 123 | 120 |
| Q13 | 126 | 134 | 126 |
| Q14 | 81 | 126 | 143 |
| Q15 | 80 | 116 | 110 |
| Q16 | 86 | 133 | 125 |
| Q17 | 80 | 122 | 110 |
| Q18 | 206 | 143 | 168 |
| Q19 | 115 | 173 | 117 |
| Q20 | 85 | 120 | 124 |
| Q21 | 425 | 349 | 199 |
| Q22 | 88 | 122 | 128 |
| **sum (22 queries)** | **2,889** | **3,063** | **2,853** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 67 | 153 | 53 |
| Q2 | 33 | 74 | 33 |
| Q3 | 60 | 132 | 85 |
| Q4 | 50 | 117 | 70 |
| Q5 | 64 | 125 | 89 |
| Q6 | 32 | 84 | 35 |
| Q7 | 63 | 128 | 89 |
| Q8 | 77 | 154 | 122 |
| Q9 | 86 | 197 | 141 |
| Q10 | 71 | 135 | 90 |
| Q11 | 26 | 65 | 18 |
| Q12 | 64 | 150 | 76 |
| Q13 | 29 | 46 | 36 |
| Q14 | 44 | 90 | 94 |
| Q15 | 40 | 88 | 47 |
| Q16 | 27 | 56 | 27 |
| Q17 | 39 | 80 | 92 |
| Q18 | 46 | 76 | 76 |
| Q19 | 76 | 165 | 93 |
| Q20 | 63 | 122 | 112 |
| Q21 | 58 | 114 | 86 |
| Q22 | 29 | 52 | 26 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 206 | 17 | 9.7 |
| Q2 | 6.8 | 9.5 | 6.6 |
| Q3 | 26 | 16 | 13 |
| Q4 | 62 | 16 | 13 |
| Q5 | 39 | 15 | 13 |
| Q6 | 5.8 | 4.9 | 4.5 |
| Q7 | 91 | 18 | 15 |
| Q8 | 12 | 15 | 11 |
| Q9 | 62 | 59 | 43 |
| Q10 | 32 | 23 | 18 |
| Q11 | 3.8 | 4.7 | 4.1 |
| Q12 | 16 | 14 | 15 |
| Q13 | 43 | 24 | 6.2 |
| Q14 | 7.8 | 13 | 6.8 |
| Q15 | 8.8 | 8.8 | 7.2 |
| Q16 | 12 | 22 | 22 |
| Q17 | 5.4 | 9.6 | 6.3 |
| Q18 | 124 | 37 | 45 |
| Q19 | 36 | 20 | 13 |
| Q20 | 15 | 11 | 11 |
| Q21 | 334 | 237 | 67 |
| Q22 | 16 | 14 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 71 | 108 | 104 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 3.1e-03, mx-cpu 1.8e-15.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,724 WRONG | 138 | **65** | - |
| Q2 min-cost supplier | 51 | 30 | **18** | - |
| Q3 3-join+top10 | 306 | **86** | 124 | - |
| Q4 EXISTS semi-join | 966 | **101** | 114 | - |
| Q5 4-join+groupby | 498 | **75** | 113 | - |
| Q6 filter+global sum | 45 | 33 | **32** | - |
| Q7 volume shipping | 1,100 †3e-6 | **85** | 119 | - |
| Q8 market share | 139 | **61** | 105 | - |
| Q9 6-join profit | 951 †7e-6 | 451 | **426** | - |
| Q10 returned items top20 | 358 | **118** | 150 | - |
| Q11 important stock | 31 | **16** | 24 | - |
| Q12 shipping modes | 150 | **82** | 118 | - |
| Q13 customer distribution | 641 | 235 | **94** | PyArrow/NumPy |
| Q14 promo revenue | 84 | 64 | **56** | - |
| Q15 top supplier | 66 | 52 | **49** | - |
| Q16 part/supplier | **98** | 106 | 101 | - |
| Q17 small-qty revenue | 66 | **28** | 65 | - |
| Q18 large volume top100 | 1,513 | **331** | 410 | - |
| Q19 discounted revenue | 460 | 83 | **70** | - |
| Q20 potential promotion | 121 | **49** | 64 | - |
| Q21 suppliers waiting | 3,843 | 2,814 | **741** | - |
| Q22 global sales opportunity | 191 | **42** | 94 | - |
| **sum over the 21 queries valid for every engine** | **11,677** | **4,943** | **3,087** |  |

Fastest verified engine per query: Polars 1, DuckDB 12, mx-cpu 9.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 20/22** (21 comparable); **mx-cpu beats DuckDB on 10/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,798 WRONG | 246 | 166 |
| Q2 | 123 | 139 | 120 |
| Q3 | 378 | 194 | 225 |
| Q4 | 1,038 | 210 | 214 |
| Q5 | 570 | 186 | 233 |
| Q6 | 116 | 141 | 135 |
| Q7 | 1,173 †3e-6 | 197 | 224 |
| Q8 | 211 | 171 | 208 |
| Q9 | 1,022 †7e-6 | 559 | 526 |
| Q10 | 430 | 227 | 251 |
| Q11 | 101 | 124 | 125 |
| Q12 | 224 | 190 | 237 |
| Q13 | 714 | 348 | 226 |
| Q14 | 156 | 174 | 174 |
| Q15 | 140 | 163 | 155 |
| Q16 | 170 | 219 | 235 |
| Q17 | 137 | 139 | 187 |
| Q18 | 1,590 | 438 | 512 |
| Q19 | 532 | 193 | 172 |
| Q20 | 195 | 157 | 167 |
| Q21 | 3,915 | 2,924 | 887 |
| Q22 | 264 | 154 | 195 |
| **sum (21 queries)** | **13,199** | **7,249** | **5,409** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 538 | 633 | 494 |
| Q2 | 83 | 229 | 108 |
| Q3 | 326 | 518 | 669 |
| Q4 | 259 | 345 | 607 |
| Q5 | 359 | 531 | 689 |
| Q6 | 202 | 337 | 281 |
| Q7 | 411 | 570 | 722 |
| Q8 | 461 | 769 | 1,025 |
| Q9 | 525 | 817 | 1,178 |
| Q10 | 427 | 493 | 678 |
| Q11 | 53 | 98 | 78 |
| Q12 | 440 | 485 | 691 |
| Q13 | 75 | 101 | 212 |
| Q14 | 269 | 485 | 782 |
| Q15 | 250 | 400 | 383 |
| Q16 | 68 | 180 | 108 |
| Q17 | 239 | 421 | 742 |
| Q18 | 203 | 262 | 618 |
| Q19 | 572 | 756 | 873 |
| Q20 | 364 | 559 | 914 |
| Q21 | 336 | 441 | 665 |
| Q22 | 77 | 137 | 243 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,346 WRONG | 137 | 63 |
| Q2 | 44 | 32 | 16 |
| Q3 | 251 | 84 | 123 |
| Q4 | 895 | 96 | 103 |
| Q5 | 418 | 70 | 102 |
| Q6 | 40 | 31 | 28 |
| Q7 | 1,004 | 85 | 105 |
| Q8 | 92 | 60 | 97 |
| Q9 | 810 | 444 | 389 |
| Q10 | 310 | 116 | 148 |
| Q11 | 24 | 16 | 20 |
| Q12 | 137 | 78 | 112 |
| Q13 | 591 | 229 | 92 |
| Q14 | 65 | 61 | 52 |
| Q15 | 58 | 49 | 46 |
| Q16 | 86 | 110 | 86 |
| Q17 | 36 | 28 | 61 |
| Q18 | 1,477 | 327 | 374 |
| Q19 | 412 | 84 | 69 |
| Q20 | 102 | 49 | 63 |
| Q21 | 4,109 | 2,784 | 658 |
| Q22 | 181 | 39 | 89 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 72 | 109 | 103 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 1.0e-14.


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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 4.7, max 14.0 over 132 runs; load average at report time: 13.7 5.5 3.9), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `37265ea`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

