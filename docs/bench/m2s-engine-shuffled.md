# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-14

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 170 †3e-3 | 19 | **10** | - |
| Q2 min-cost supplier | **7.9** | 12 | 11 | - |
| Q3 3-join+top10 | 27 | **16** | 19 | - |
| Q4 EXISTS semi-join | 90 | **15** | 20 | - |
| Q5 4-join+groupby | 48 | **18** | 21 | - |
| Q6 filter+global sum | 9.9 | **4.9** | 5.8 | - |
| Q7 volume shipping | 128 | **18** | 20 | - |
| Q8 market share | **13** | 15 | 17 | - |
| Q9 6-join profit | 62 †2e-6 | **42** | 63 | - |
| Q10 returned items top20 | 37 | **23** | 24 | - |
| Q11 important stock | **3.9** | 7.5 | 6.4 | - |
| Q12 shipping modes | 68 | **13** | 25 | - |
| Q13 customer distribution | 55 | 26 | **9.7** | PyArrow/NumPy |
| Q14 promo revenue | **8.2** | 17 | 11 | - |
| Q15 top supplier | 11 | **9.2** | 9.9 | - |
| Q16 part/supplier | **16** | 22 | 28 | - |
| Q17 small-qty revenue | **7.7** | 8.9 | 9.6 | - |
| Q18 large volume top100 | 118 | **86** | 107 | - |
| Q19 discounted revenue | 74 | 22 | **15** | - |
| Q20 potential promotion | 19 | 13 | **13** | - |
| Q21 suppliers waiting | 426 | 236 | **152** | - |
| Q22 global sales opportunity | **11** | 15 | 18 | - |
| **sum over the 22 queries valid for every engine** | **1,410** | **660** | **614** |  |

Fastest verified engine per query: Polars 7, DuckDB 10, mx-cpu 5.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 14/22** (22 comparable); **mx-cpu beats DuckDB on 8/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 243 †3e-3 | 131 | 152 |
| Q2 | 84 | 130 | 116 |
| Q3 | 103 | 135 | 123 |
| Q4 | 165 | 152 | 125 |
| Q5 | 124 | 164 | 160 |
| Q6 | 83 | 123 | 110 |
| Q7 | 205 | 135 | 131 |
| Q8 | 92 | 138 | 120 |
| Q9 | 138 †2e-6 | 153 | 175 |
| Q10 | 110 | 134 | 128 |
| Q11 | 75 | 124 | 113 |
| Q12 | 143 | 129 | 135 |
| Q13 | 132 | 135 | 128 |
| Q14 | 82 | 129 | 121 |
| Q15 | 85 | 128 | 143 |
| Q16 | 94 | 135 | 147 |
| Q17 | 91 | 124 | 119 |
| Q18 | 196 | 206 | 247 |
| Q19 | 153 | 133 | 151 |
| Q20 | 98 | 125 | 122 |
| Q21 | 502 | 348 | 288 |
| Q22 | 85 | 131 | 123 |
| **sum (22 queries)** | **3,083** | **3,240** | **3,177** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 70 | 164 | 55 |
| Q2 | 35 | 82 | 34 |
| Q3 | 60 | 137 | 81 |
| Q4 | 52 | 104 | 72 |
| Q5 | 60 | 112 | 96 |
| Q6 | 34 | 88 | 40 |
| Q7 | 65 | 130 | 97 |
| Q8 | 78 | 170 | 127 |
| Q9 | 88 | 200 | 149 |
| Q10 | 69 | 138 | 85 |
| Q11 | 28 | 67 | 25 |
| Q12 | 69 | 164 | 85 |
| Q13 | 29 | 51 | 37 |
| Q14 | 49 | 97 | 96 |
| Q15 | 42 | 88 | 48 |
| Q16 | 30 | 58 | 26 |
| Q17 | 42 | 98 | 91 |
| Q18 | 47 | 85 | 89 |
| Q19 | 83 | 192 | 110 |
| Q20 | 65 | 138 | 107 |
| Q21 | 60 | 116 | 89 |
| Q22 | 29 | 43 | 25 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 156 | 19 | 8.1 |
| Q2 | 7.7 | 11 | 6.6 |
| Q3 | 28 | 17 | 13 |
| Q4 | 75 | 17 | 15 |
| Q5 | 43 | 14 | 18 |
| Q6 | 9.2 | 5.3 | 4.1 |
| Q7 | 109 | 17 | 14 |
| Q8 | 11 | 14 | 12 |
| Q9 | 58 | 37 | 48 |
| Q10 | 31 | 20 | 20 |
| Q11 | 3.8 | 5.1 | 4.7 |
| Q12 | 67 | 14 | 20 |
| Q13 | 40 | 24 | 6.8 |
| Q14 | 7.9 | 11 | 8.6 |
| Q15 | 11 | 8.8 | 9.1 |
| Q16 | 13 | 24 | 26 |
| Q17 | 8.0 | 8.7 | 6.9 |
| Q18 | 108 | 85 | 106 |
| Q19 | 73 | 20 | 12 |
| Q20 | 16 | 13 | 10 |
| Q21 | 415 | 248 | 106 |
| Q22 | 10 | 15 | 16 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 76 | 116 | 110 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.13e-03 | max relative error 3.13e-03 |
| Q9 | polars | drift | 1.86e-06 | max relative error 1.86e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 3.1e-03, mx-cpu 2.7e-15.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,039 WRONG | 150 | **65** | - |
| Q2 min-cost supplier | 52 | 27 | **23** | - |
| Q3 3-join+top10 | 251 | **84** | 169 | - |
| Q4 EXISTS semi-join | 1,114 | **100** | 119 | - |
| Q5 4-join+groupby | 457 | **89** | 157 | - |
| Q6 filter+global sum | 81 | 32 | **30** | - |
| Q7 volume shipping | 1,311 †4e-6 | **87** | 116 | - |
| Q8 market share | 75 | **62** | 106 | - |
| Q9 6-join profit | 637 †6e-6 | **232** | 551 | - |
| Q10 returned items top20 | 342 | **129** | 172 | - |
| Q11 important stock | 27 | **15** | 30 | - |
| Q12 shipping modes | 765 | **79** | 201 | - |
| Q13 customer distribution | 688 | 225 | **92** | PyArrow/NumPy |
| Q14 promo revenue | 86 | 59 | **54** | - |
| Q15 top supplier | 78 | **57** | 59 | - |
| Q16 part/supplier | **101** | 103 | 101 | - |
| Q17 small-qty revenue | **28** | 28 | 65 | - |
| Q18 large volume top100 | 1,508 | **1,057** | 1,287 | - |
| Q19 discounted revenue | 729 | 84 | **68** | - |
| Q20 potential promotion | 134 | **50** | 70 | - |
| Q21 suppliers waiting | 4,380 | 2,767 | **1,336** | - |
| Q22 global sales opportunity | 177 | **40** | 95 | - |
| **sum over the 21 queries valid for every engine** | **13,022** | **5,405** | **4,904** |  |

Fastest verified engine per query: Polars 2, DuckDB 13, mx-cpu 7.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 17/22** (21 comparable); **mx-cpu beats DuckDB on 8/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,111 WRONG | 262 | 167 |
| Q2 | 126 | 139 | 159 |
| Q3 | 326 | 192 | 270 |
| Q4 | 1,187 | 209 | 222 |
| Q5 | 530 | 202 | 279 |
| Q6 | 153 | 144 | 133 |
| Q7 | 1,388 †4e-6 | 196 | 219 |
| Q8 | 154 | 172 | 208 |
| Q9 | 709 †6e-6 | 344 | 653 |
| Q10 | 415 | 238 | 289 |
| Q11 | 99 | 124 | 168 |
| Q12 | 837 | 190 | 304 |
| Q13 | 759 | 337 | 194 |
| Q14 | 160 | 167 | 158 |
| Q15 | 151 | 169 | 168 |
| Q16 | 176 | 214 | 204 |
| Q17 | 100 | 136 | 167 |
| Q18 | 1,580 | 1,168 | 1,408 |
| Q19 | 802 | 196 | 171 |
| Q20 | 205 | 162 | 173 |
| Q21 | 4,456 | 2,876 | 1,438 |
| Q22 | 250 | 149 | 196 |
| **sum (21 queries)** | **14,561** | **7,724** | **7,180** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 527 | 713 | 514 |
| Q2 | 81 | 244 | 119 |
| Q3 | 325 | 510 | 669 |
| Q4 | 263 | 366 | 606 |
| Q5 | 371 | 541 | 681 |
| Q6 | 204 | 345 | 274 |
| Q7 | 392 | 576 | 722 |
| Q8 | 474 | 759 | 1,034 |
| Q9 | 535 | 856 | 1,193 |
| Q10 | 432 | 509 | 689 |
| Q11 | 56 | 97 | 77 |
| Q12 | 440 | 505 | 703 |
| Q13 | 77 | 103 | 208 |
| Q14 | 273 | 483 | 769 |
| Q15 | 244 | 398 | 382 |
| Q16 | 68 | 170 | 102 |
| Q17 | 240 | 428 | 746 |
| Q18 | 208 | 259 | 599 |
| Q19 | 574 | 773 | 883 |
| Q20 | 359 | 591 | 913 |
| Q21 | 341 | 440 | 647 |
| Q22 | 79 | 136 | 246 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 1,660 WRONG | 139 | 62 |
| Q2 | 49 | 27 | 21 |
| Q3 | 208 | 85 | 160 |
| Q4 | 1,108 | 102 | 105 |
| Q5 | 426 | 90 | 145 |
| Q6 | 77 | 33 | 29 |
| Q7 | 1,111 | 87 | 109 |
| Q8 | 70 | 62 | 100 |
| Q9 | 621 | 227 | 513 |
| Q10 | 274 | 127 | 159 |
| Q11 | 26 | 15 | 22 |
| Q12 | 624 | 80 | 191 |
| Q13 | 634 | 227 | 83 |
| Q14 | 66 | 59 | 51 |
| Q15 | 74 | 58 | 56 |
| Q16 | 89 | 107 | 91 |
| Q17 | 28 | 28 | 61 |
| Q18 | 1,592 | 1,048 | 1,262 |
| Q19 | 719 | 86 | 68 |
| Q20 | 135 | 50 | 68 |
| Q21 | 4,378 | 2,741 | 1,070 |
| Q22 | 174 | 41 | 90 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 73 | 110 | 103 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 3.69e-06 | max relative error 3.69e-06 |
| Q9 | polars | drift | 5.94e-06 | max relative error 5.94e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 2.4e-14.


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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 7.3, max 14.6 over 132 runs; load average at report time: 6.0 4.3 4.1), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `37265ea`. Last invocation took 16 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

