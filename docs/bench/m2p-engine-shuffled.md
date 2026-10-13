# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-13

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 163 †3e-3 | 20 | **10** | - |
| Q2 min-cost supplier | **7.0** | 11 | 8.4 | - |
| Q3 3-join+top10 | 26 | **15** | 18 | - |
| Q4 EXISTS semi-join | 90 | **16** | 20 | - |
| Q5 4-join+groupby | 48 | **18** | 20 | - |
| Q6 filter+global sum | 10 | **5.7** | 6.1 | - |
| Q7 volume shipping | 106 | **18** | 19 | - |
| Q8 market share | **14** | 18 | 19 | - |
| Q9 6-join profit | 60 †2e-6 | **43** | 60 | - |
| Q10 returned items top20 | 42 | **23** | 26 | - |
| Q11 important stock | **4.7** | 7.3 | 7.1 | - |
| Q12 shipping modes | 66 | **14** | 24 | - |
| Q13 customer distribution | 68 | 29 | **8.3** | PyArrow/NumPy |
| Q14 promo revenue | **8.4** | 12 | 12 | - |
| Q15 top supplier | 10 | **9.4** | 11 | - |
| Q16 part/supplier | **16** | 24 | 27 | - |
| Q17 small-qty revenue | 9.3 | **8.0** | 9.3 | - |
| Q18 large volume top100 | 117 | **87** | 103 | - |
| Q19 discounted revenue | 77 | 24 | **15** | - |
| Q20 potential promotion | 17 | **12** | 13 | - |
| Q21 suppliers waiting | 367 | **240** | 317 | - |
| Q22 global sales opportunity | **12** | 14 | 18 | - |
| **sum over the 22 queries valid for every engine** | **1,338** | **669** | **772** |  |

Fastest verified engine per query: Polars 6, DuckDB 13, mx-cpu 3.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 14/22** (22 comparable); **mx-cpu beats DuckDB on 5/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 243 †3e-3 | 144 | 146 |
| Q2 | 79 | 130 | 150 |
| Q3 | 103 | 138 | 133 |
| Q4 | 175 | 134 | 132 |
| Q5 | 123 | 140 | 157 |
| Q6 | 90 | 126 | 111 |
| Q7 | 178 | 130 | 124 |
| Q8 | 93 | 128 | 166 |
| Q9 | 138 †2e-6 | 155 | 179 |
| Q10 | 118 | 136 | 136 |
| Q11 | 98 | 120 | 142 |
| Q12 | 146 | 130 | 157 |
| Q13 | 148 | 147 | 114 |
| Q14 | 92 | 133 | 136 |
| Q15 | 84 | 119 | 121 |
| Q16 | 90 | 140 | 137 |
| Q17 | 92 | 122 | 117 |
| Q18 | 192 | 198 | 213 |
| Q19 | 158 | 136 | 120 |
| Q20 | 91 | 122 | 114 |
| Q21 | 439 | 348 | 419 |
| Q22 | 90 | 138 | 159 |
| **sum (22 queries)** | **3,061** | **3,215** | **3,384** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 76 | 151 | 65 |
| Q2 | 32 | 76 | 38 |
| Q3 | 56 | 139 | 85 |
| Q4 | 51 | 132 | 77 |
| Q5 | 63 | 131 | 91 |
| Q6 | 40 | 87 | 43 |
| Q7 | 63 | 140 | 93 |
| Q8 | 85 | 162 | 140 |
| Q9 | 97 | 183 | 147 |
| Q10 | 72 | 133 | 94 |
| Q11 | 29 | 66 | 26 |
| Q12 | 75 | 166 | 87 |
| Q13 | 31 | 60 | 36 |
| Q14 | 55 | 99 | 95 |
| Q15 | 47 | 85 | 53 |
| Q16 | 26 | 69 | 27 |
| Q17 | 45 | 90 | 91 |
| Q18 | 50 | 83 | 74 |
| Q19 | 80 | 177 | 111 |
| Q20 | 60 | 123 | 101 |
| Q21 | 55 | 105 | 76 |
| Q22 | 30 | 43 | 29 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 170 | 20 | 8.7 |
| Q2 | 6.9 | 11 | 6.8 |
| Q3 | 26 | 21 | 13 |
| Q4 | 84 | 17 | 15 |
| Q5 | 39 | 16 | 17 |
| Q6 | 12 | 4.7 | 4.6 |
| Q7 | 100 | 17 | 14 |
| Q8 | 12 | 15 | 13 |
| Q9 | 60 | 33 | 55 |
| Q10 | 36 | 23 | 22 |
| Q11 | 3.5 | 5.6 | 4.4 |
| Q12 | 71 | 14 | 22 |
| Q13 | 50 | 28 | 6.4 |
| Q14 | 10 | 11 | 9.0 |
| Q15 | 10 | 9.3 | 8.1 |
| Q16 | 13 | 25 | 23 |
| Q17 | 6.9 | 7.9 | 6.4 |
| Q18 | 105 | 91 | 109 |
| Q19 | 76 | 19 | 13 |
| Q20 | 17 | 12 | 10 |
| Q21 | 347 | 233 | 273 |
| Q22 | 9.5 | 15 | 16 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 78 | 115 | 111 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.13e-03 | max relative error 3.13e-03 |
| Q9 | polars | drift | 1.86e-06 | max relative error 1.86e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.0e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 1,885 WRONG | 138 | **65** | - |
| Q2 min-cost supplier | 50 | 28 | **22** | - |
| Q3 3-join+top10 | 263 | **87** | 159 | - |
| Q4 EXISTS semi-join | 1,164 | **114** | 121 | - |
| Q5 4-join+groupby | 451 | **88** | 152 | - |
| Q6 filter+global sum | 81 | 32 | **31** | - |
| Q7 volume shipping | 1,162 †4e-6 | **90** | 116 | - |
| Q8 market share | 102 | **61** | 104 | - |
| Q9 6-join profit | 641 †6e-6 | **236** | 534 | - |
| Q10 returned items top20 | 281 | **157** | 173 | - |
| Q11 important stock | 28 | **15** | 26 | - |
| Q12 shipping modes | 645 | **84** | 204 | - |
| Q13 customer distribution | 692 | 231 | **94** | PyArrow/NumPy |
| Q14 promo revenue | 68 | 61 | **55** | - |
| Q15 top supplier | 80 | 59 | **58** | - |
| Q16 part/supplier | 105 | **99** | 103 | - |
| Q17 small-qty revenue | **29** | 29 | 66 | - |
| Q18 large volume top100 | 1,688 | **1,114** | 1,306 | - |
| Q19 discounted revenue | 719 | 85 | **68** | - |
| Q20 potential promotion | 134 | **52** | 72 | - |
| Q21 suppliers waiting | 4,660 | **2,877** | 4,975 | - |
| Q22 global sales opportunity | 283 | **39** | 133 | - |
| **sum over the 21 queries valid for every engine** | **13,327** | **5,640** | **8,572** |  |

Fastest verified engine per query: Polars 1, DuckDB 14, mx-cpu 7.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 18/22** (21 comparable); **mx-cpu beats DuckDB on 7/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 1,958 WRONG | 248 | 174 |
| Q2 | 122 | 140 | 125 |
| Q3 | 337 | 196 | 260 |
| Q4 | 1,239 | 226 | 222 |
| Q5 | 524 | 197 | 255 |
| Q6 | 154 | 144 | 135 |
| Q7 | 1,233 †4e-6 | 199 | 220 |
| Q8 | 177 | 170 | 208 |
| Q9 | 713 †6e-6 | 358 | 635 |
| Q10 | 356 | 264 | 278 |
| Q11 | 102 | 123 | 129 |
| Q12 | 721 | 194 | 310 |
| Q13 | 767 | 343 | 198 |
| Q14 | 140 | 172 | 168 |
| Q15 | 155 | 209 | 168 |
| Q16 | 181 | 208 | 221 |
| Q17 | 100 | 144 | 207 |
| Q18 | 1,761 | 1,223 | 1,409 |
| Q19 | 792 | 192 | 168 |
| Q20 | 206 | 162 | 175 |
| Q21 | 4,734 | 2,987 | 5,078 |
| Q22 | 366 | 149 | 263 |
| **sum (21 queries)** | **14,881** | **7,999** | **10,832** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 562 | 645 | 535 |
| Q2 | 80 | 241 | 110 |
| Q3 | 326 | 507 | 678 |
| Q4 | 264 | 368 | 611 |
| Q5 | 364 | 526 | 700 |
| Q6 | 203 | 339 | 274 |
| Q7 | 396 | 578 | 731 |
| Q8 | 471 | 765 | 1,033 |
| Q9 | 529 | 916 | 1,230 |
| Q10 | 454 | 504 | 740 |
| Q11 | 55 | 93 | 76 |
| Q12 | 443 | 510 | 707 |
| Q13 | 79 | 102 | 210 |
| Q14 | 272 | 469 | 790 |
| Q15 | 253 | 413 | 435 |
| Q16 | 73 | 179 | 103 |
| Q17 | 244 | 424 | 787 |
| Q18 | 204 | 289 | 606 |
| Q19 | 581 | 798 | 886 |
| Q20 | 358 | 580 | 950 |
| Q21 | 341 | 472 | 648 |
| Q22 | 86 | 147 | 272 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 1,613 WRONG | 140 | 64 |
| Q2 | 48 | 25 | 20 |
| Q3 | 218 | 89 | 150 |
| Q4 | 1,090 | 107 | 119 |
| Q5 | 441 | 89 | 141 |
| Q6 | 77 | 33 | 29 |
| Q7 | 1,079 | 88 | 109 |
| Q8 | 73 | 62 | 97 |
| Q9 | 625 | 231 | 491 |
| Q10 | 264 | 126 | 182 |
| Q11 | 26 | 15 | 23 |
| Q12 | 640 | 82 | 201 |
| Q13 | 618 | 230 | 85 |
| Q14 | 67 | 62 | 51 |
| Q15 | 81 | 64 | 58 |
| Q16 | 112 | 107 | 92 |
| Q17 | 26 | 32 | 60 |
| Q18 | 1,673 | 1,084 | 1,268 |
| Q19 | 717 | 89 | 67 |
| Q20 | 132 | 54 | 69 |
| Q21 | 4,440 | 2,813 | 4,852 |
| Q22 | 269 | 48 | 109 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 73 | 110 | 104 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 3.69e-06 | max relative error 3.69e-06 |
| Q9 | polars | drift | 5.94e-06 | max relative error 5.94e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.0e-14, Polars 5.5e-02.


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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 8.3, max 16.2 over 132 runs; load average at report time: 5.0 4.1 3.9), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `dc88dca`. Last invocation took 16 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

