# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-13

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 168 †3e-3 | 19 | **9.3** | - |
| Q2 min-cost supplier | **7.3** | 9.9 | 8.5 | - |
| Q3 3-join+top10 | 26 | **15** | 18 | - |
| Q4 EXISTS semi-join | 77 | **16** | 20 | - |
| Q5 4-join+groupby | 43 | **16** | 22 | - |
| Q6 filter+global sum | 9.4 | **5.2** | 5.4 | - |
| Q7 volume shipping | 113 | 19 | **19** | - |
| Q8 market share | **12** | 14 | 16 | - |
| Q9 6-join profit | 55 †2e-6 | **36** | 58 | - |
| Q10 returned items top20 | 28 | 23 | **22** | - |
| Q11 important stock | **4.2** | 6.4 | 6.2 | - |
| Q12 shipping modes | 66 | **14** | 23 | - |
| Q13 customer distribution | 49 | 27 | **8.4** | PyArrow/NumPy |
| Q14 promo revenue | **9.1** | 13 | 10 | - |
| Q15 top supplier | 9.7 | 9.2 | **9.1** | - |
| Q16 part/supplier | **14** | 21 | 27 | - |
| Q17 small-qty revenue | **6.9** | 8.5 | 9.2 | - |
| Q18 large volume top100 | 117 | **82** | 99 | - |
| Q19 discounted revenue | 75 | 19 | **14** | - |
| Q20 potential promotion | 18 | **12** | 13 | - |
| Q21 suppliers waiting | 377 | 234 | **144** | - |
| Q22 global sales opportunity | **13** | 15 | 18 | - |
| **sum over the 22 queries valid for every engine** | **1,297** | **634** | **578** |  |

Fastest verified engine per query: Polars 7, DuckDB 8, mx-cpu 7.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 14/22** (22 comparable); **mx-cpu beats DuckDB on 10/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 245 †3e-3 | 133 | 112 |
| Q2 | 81 | 120 | 114 |
| Q3 | 101 | 125 | 124 |
| Q4 | 150 | 130 | 124 |
| Q5 | 118 | 131 | 125 |
| Q6 | 82 | 119 | 110 |
| Q7 | 191 | 135 | 129 |
| Q8 | 87 | 127 | 138 |
| Q9 | 128 †2e-6 | 149 | 166 |
| Q10 | 102 | 134 | 129 |
| Q11 | 78 | 118 | 110 |
| Q12 | 140 | 129 | 128 |
| Q13 | 124 | 138 | 113 |
| Q14 | 82 | 127 | 116 |
| Q15 | 83 | 121 | 126 |
| Q16 | 92 | 132 | 131 |
| Q17 | 83 | 121 | 118 |
| Q18 | 192 | 193 | 205 |
| Q19 | 149 | 132 | 120 |
| Q20 | 92 | 122 | 117 |
| Q21 | 451 | 348 | 249 |
| Q22 | 89 | 131 | 121 |
| **sum (22 queries)** | **2,940** | **3,117** | **2,925** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 66 | 150 | 55 |
| Q2 | 32 | 77 | 27 |
| Q3 | 54 | 125 | 87 |
| Q4 | 50 | 122 | 75 |
| Q5 | 60 | 117 | 91 |
| Q6 | 31 | 83 | 36 |
| Q7 | 66 | 127 | 102 |
| Q8 | 78 | 153 | 127 |
| Q9 | 89 | 186 | 143 |
| Q10 | 70 | 131 | 88 |
| Q11 | 25 | 67 | 26 |
| Q12 | 67 | 160 | 81 |
| Q13 | 30 | 56 | 38 |
| Q14 | 44 | 91 | 95 |
| Q15 | 42 | 84 | 51 |
| Q16 | 26 | 69 | 26 |
| Q17 | 44 | 89 | 89 |
| Q18 | 43 | 78 | 76 |
| Q19 | 80 | 183 | 106 |
| Q20 | 60 | 123 | 110 |
| Q21 | 59 | 110 | 88 |
| Q22 | 24 | 44 | 25 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 153 | 19 | 7.7 |
| Q2 | 6.7 | 10 | 6.9 |
| Q3 | 24 | 16 | 13 |
| Q4 | 70 | 16 | 15 |
| Q5 | 34 | 15 | 16 |
| Q6 | 8.6 | 4.7 | 4.0 |
| Q7 | 96 | 18 | 14 |
| Q8 | 11 | 13 | 11 |
| Q9 | 52 | 35 | 49 |
| Q10 | 28 | 22 | 20 |
| Q11 | 3.5 | 5.9 | 4.9 |
| Q12 | 67 | 14 | 16 |
| Q13 | 39 | 28 | 6.1 |
| Q14 | 11 | 14 | 7.0 |
| Q15 | 12 | 9.9 | 7.9 |
| Q16 | 12 | 24 | 23 |
| Q17 | 6.1 | 8.4 | 6.1 |
| Q18 | 109 | 83 | 88 |
| Q19 | 74 | 22 | 12 |
| Q20 | 15 | 12 | 10 |
| Q21 | 369 | 232 | 100 |
| Q22 | 11 | 14 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 74 | 113 | 105 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.13e-03 | max relative error 3.13e-03 |
| Q9 | polars | drift | 1.86e-06 | max relative error 1.86e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.9e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,176 WRONG | 140 | **67** | - |
| Q2 min-cost supplier | 51 | 27 | **23** | - |
| Q3 3-join+top10 | 264 | **90** | 169 | - |
| Q4 EXISTS semi-join | 1,185 | **111** | 124 | - |
| Q5 4-join+groupby | 444 | **94** | 156 | - |
| Q6 filter+global sum | 81 | 32 | **31** | - |
| Q7 volume shipping | 1,164 †4e-6 | **91** | 117 | - |
| Q8 market share | 76 | **61** | 110 | - |
| Q9 6-join profit | 660 †6e-6 | **234** | 548 | - |
| Q10 returned items top20 | 287 | **134** | 163 | - |
| Q11 important stock | 28 | **16** | 27 | - |
| Q12 shipping modes | 629 | **80** | 199 | - |
| Q13 customer distribution | 683 | 226 | **96** | PyArrow/NumPy |
| Q14 promo revenue | 72 | 57 | **56** | - |
| Q15 top supplier | 79 | **59** | 61 | - |
| Q16 part/supplier | **98** | 105 | 102 | - |
| Q17 small-qty revenue | **27** | 29 | 65 | - |
| Q18 large volume top100 | 1,555 | **1,084** | 1,320 | - |
| Q19 discounted revenue | 736 | 87 | **67** | - |
| Q20 potential promotion | 132 | **51** | 74 | - |
| Q21 suppliers waiting | 4,642 | 2,666 | **1,323** | - |
| Q22 global sales opportunity | 167 | **41** | 96 | - |
| **sum over the 21 queries valid for every engine** | **13,060** | **5,375** | **4,926** |  |

Fastest verified engine per query: Polars 2, DuckDB 13, mx-cpu 7.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 18/22** (21 comparable); **mx-cpu beats DuckDB on 8/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,252 WRONG | 251 | 173 |
| Q2 | 126 | 141 | 131 |
| Q3 | 337 | 203 | 274 |
| Q4 | 1,261 | 222 | 233 |
| Q5 | 520 | 206 | 261 |
| Q6 | 155 | 141 | 137 |
| Q7 | 1,237 †4e-6 | 198 | 224 |
| Q8 | 150 | 171 | 215 |
| Q9 | 736 †6e-6 | 351 | 661 |
| Q10 | 363 | 244 | 269 |
| Q11 | 103 | 130 | 129 |
| Q12 | 702 | 192 | 302 |
| Q13 | 756 | 334 | 197 |
| Q14 | 147 | 166 | 158 |
| Q15 | 153 | 168 | 164 |
| Q16 | 171 | 216 | 206 |
| Q17 | 102 | 138 | 168 |
| Q18 | 1,633 | 1,196 | 1,423 |
| Q19 | 807 | 199 | 171 |
| Q20 | 205 | 164 | 182 |
| Q21 | 4,715 | 2,776 | 1,424 |
| Q22 | 240 | 157 | 198 |
| **sum (21 queries)** | **14,619** | **7,711** | **7,128** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 543 | 644 | 525 |
| Q2 | 85 | 247 | 122 |
| Q3 | 350 | 495 | 692 |
| Q4 | 266 | 345 | 613 |
| Q5 | 363 | 529 | 712 |
| Q6 | 207 | 330 | 272 |
| Q7 | 401 | 574 | 731 |
| Q8 | 472 | 759 | 1,072 |
| Q9 | 540 | 848 | 1,212 |
| Q10 | 431 | 504 | 697 |
| Q11 | 54 | 98 | 81 |
| Q12 | 449 | 504 | 708 |
| Q13 | 76 | 98 | 203 |
| Q14 | 275 | 477 | 772 |
| Q15 | 251 | 399 | 382 |
| Q16 | 69 | 175 | 103 |
| Q17 | 246 | 418 | 760 |
| Q18 | 209 | 264 | 604 |
| Q19 | 575 | 799 | 877 |
| Q20 | 358 | 569 | 914 |
| Q21 | 331 | 438 | 637 |
| Q22 | 75 | 136 | 239 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 1,596 WRONG | 141 | 63 |
| Q2 | 53 | 26 | 20 |
| Q3 | 209 | 93 | 160 |
| Q4 | 1,101 | 105 | 105 |
| Q5 | 447 | 93 | 143 |
| Q6 | 77 | 34 | 29 |
| Q7 | 1,102 | 93 | 111 |
| Q8 | 77 | 68 | 101 |
| Q9 | 631 | 238 | 512 |
| Q10 | 271 | 134 | 158 |
| Q11 | 27 | 16 | 24 |
| Q12 | 624 | 85 | 194 |
| Q13 | 618 | 224 | 85 |
| Q14 | 65 | 60 | 51 |
| Q15 | 74 | 59 | 57 |
| Q16 | 94 | 110 | 91 |
| Q17 | 29 | 29 | 63 |
| Q18 | 1,489 | 1,073 | 1,291 |
| Q19 | 722 | 90 | 65 |
| Q20 | 133 | 52 | 69 |
| Q21 | 4,308 | 2,731 | 1,040 |
| Q22 | 172 | 40 | 89 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 74 | 111 | 104 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 3.69e-06 | max relative error 3.69e-06 |
| Q9 | polars | drift | 5.94e-06 | max relative error 5.94e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.3e-14, Polars 5.5e-02.


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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.8, max 9.8 over 132 runs; load average at report time: 8.0 4.4 5.0), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `1313971`. Last invocation took 16 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

