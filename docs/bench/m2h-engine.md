# Engine-only TPC-H scoreboard - 2026-10-13

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | mx-cpu | outside mx |
|---|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 262 †3e-3 | **10** | - |
| Q2 min-cost supplier | **7.6** | 13 | - |
| Q3 3-join+top10 | 32 | **23** | - |
| Q4 EXISTS semi-join | 86 | **38** | - |
| Q5 4-join+groupby | 50 | **18** | - |
| Q6 filter+global sum | 6.3 | **4.9** | - |
| Q7 volume shipping | 102 | **21** | - |
| Q8 market share | **14** | 21 | - |
| Q9 6-join profit | **71 †2e-6** | 73 | - |
| Q10 returned items top20 | 41 | **28** | - |
| Q11 important stock | **4.5** | 14 | - |
| Q12 shipping modes | **18** | 21 | - |
| Q13 customer distribution | 63 | **15** | PyArrow/NumPy |
| Q14 promo revenue | **10** | 15 | - |
| Q15 top supplier | **8.4** | 11 | - |
| Q16 part/supplier | **17** | 30 | - |
| Q17 small-qty revenue | **7.4** | 13 | - |
| Q18 large volume top100 | **134** | 171 | - |
| Q19 discounted revenue | 38 | **13** | - |
| Q20 potential promotion | **14** | 16 | - |
| Q21 suppliers waiting | **366** | 385 | - |
| Q22 global sales opportunity | **14** | 33 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **1,367** | **986** |  |

Fastest verified engine per query: Polars 13, mx-cpu 9.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 9/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 336 †3e-3 | 119 |
| Q2 | 81 | 142 |
| Q3 | 107 | 150 |
| Q4 | 159 | 146 |
| Q5 | 122 | 120 |
| Q6 | 79 | 111 |
| Q7 | 175 | 125 |
| Q8 | 85 | 124 |
| Q9 | 145 †2e-6 | 175 |
| Q10 | 125 | 130 |
| Q11 | 76 | 117 |
| Q12 | 90 | 153 |
| Q13 | 155 | 204 |
| Q14 | 84 | 150 |
| Q15 | 82 | 125 |
| Q16 | 92 | 175 |
| Q17 | 85 | 126 |
| Q18 | 207 | 274 |
| Q19 | 114 | 124 |
| Q20 | 89 | 125 |
| Q21 | 440 | 487 |
| Q22 | 91 | 167 |
| **sum (22 queries)** | **3,019** | **3,568** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 73 | 60 |
| Q2 | 32 | 36 |
| Q3 | 56 | 88 |
| Q4 | 49 | 78 |
| Q5 | 62 | 91 |
| Q6 | 35 | 37 |
| Q7 | 64 | 90 |
| Q8 | 77 | 133 |
| Q9 | 90 | 137 |
| Q10 | 76 | 84 |
| Q11 | 25 | 17 |
| Q12 | 65 | 78 |
| Q13 | 37 | 39 |
| Q14 | 45 | 90 |
| Q15 | 42 | 52 |
| Q16 | 30 | 34 |
| Q17 | 43 | 91 |
| Q18 | 43 | 79 |
| Q19 | 74 | 103 |
| Q20 | 56 | 107 |
| Q21 | 58 | 74 |
| Q22 | 28 | 25 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 220 | 9.5 |
| Q2 | 7.4 | 10 |
| Q3 | 26 | 13 |
| Q4 | 61 | 30 |
| Q5 | 42 | 12 |
| Q6 | 6.0 | 4.4 |
| Q7 | 86 | 14 |
| Q8 | 12 | 13 |
| Q9 | 58 | 57 |
| Q10 | 35 | 22 |
| Q11 | 4.5 | 6.3 |
| Q12 | 19 | 15 |
| Q13 | 43 | 12 |
| Q14 | 8.2 | 7.9 |
| Q15 | 8.5 | 10 |
| Q16 | 17 | 24 |
| Q17 | 5.3 | 8.5 |
| Q18 | 120 | 142 |
| Q19 | 42 | 9.7 |
| Q20 | 15 | 11 |
| Q21 | 341 | 334 |
| Q22 | 15 | 13 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | mx-cpu |
|---|---:|---:|
| import | 74 | 109 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.6e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | mx-cpu | outside mx |
|---|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,865 WRONG | **62** | - |
| Q2 min-cost supplier | 55 | **20** | - |
| Q3 3-join+top10 | 303 | **137** | - |
| Q4 EXISTS semi-join | 994 | **291** | - |
| Q5 4-join+groupby | 526 | **132** | - |
| Q6 filter+global sum | 49 | **31** | - |
| Q7 volume shipping | 1,146 †3e-6 | **115** | - |
| Q8 market share | 144 | **99** | - |
| Q9 6-join profit | 930 †7e-6 | **652** | - |
| Q10 returned items top20 | 352 | **237** | - |
| Q11 important stock | **32** | 33 | - |
| Q12 shipping modes | 144 | **125** | - |
| Q13 customer distribution | 646 | **150** | PyArrow/NumPy |
| Q14 promo revenue | 87 | **63** | - |
| Q15 top supplier | 70 | **67** | - |
| Q16 part/supplier | **113** | 142 | - |
| Q17 small-qty revenue | 64 | **48** | - |
| Q18 large volume top100 | 1,599 | **744** | - |
| Q19 discounted revenue | 442 | **54** | - |
| Q20 potential promotion | 122 | **71** | - |
| Q21 suppliers waiting | 4,196 | **3,934** | - |
| Q22 global sales opportunity | 205 | **135** | PyArrow/NumPy |
| **sum over the 21 queries valid for every engine** | **12,219** | **7,281** |  |

Fastest verified engine per query: Polars 2, mx-cpu 20.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 19/22** (21 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 2,942 WRONG | 167 |
| Q2 | 133 | 129 |
| Q3 | 378 | 244 |
| Q4 | 1,070 | 394 |
| Q5 | 599 | 240 |
| Q6 | 120 | 136 |
| Q7 | 1,218 †3e-6 | 218 |
| Q8 | 220 | 207 |
| Q9 | 1,003 †7e-6 | 757 |
| Q10 | 427 | 359 |
| Q11 | 103 | 135 |
| Q12 | 217 | 226 |
| Q13 | 721 | 253 |
| Q14 | 170 | 192 |
| Q15 | 146 | 176 |
| Q16 | 193 | 249 |
| Q17 | 139 | 150 |
| Q18 | 1,680 | 878 |
| Q19 | 513 | 156 |
| Q20 | 198 | 203 |
| Q21 | 4,268 | 4,036 |
| Q22 | 279 | 264 |
| **sum (21 queries)** | **13,795** | **9,604** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 550 | 495 |
| Q2 | 90 | 115 |
| Q3 | 334 | 674 |
| Q4 | 274 | 618 |
| Q5 | 373 | 709 |
| Q6 | 212 | 281 |
| Q7 | 393 | 727 |
| Q8 | 482 | 1,059 |
| Q9 | 552 | 1,233 |
| Q10 | 440 | 712 |
| Q11 | 53 | 79 |
| Q12 | 439 | 692 |
| Q13 | 76 | 206 |
| Q14 | 298 | 785 |
| Q15 | 262 | 408 |
| Q16 | 75 | 109 |
| Q17 | 258 | 780 |
| Q18 | 226 | 611 |
| Q19 | 581 | 897 |
| Q20 | 372 | 966 |
| Q21 | 341 | 675 |
| Q22 | 77 | 247 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu |
|---|---:|---:|
| Q1 | 2,441 WRONG | 70 |
| Q2 | 47 | 16 |
| Q3 | 252 | 115 |
| Q4 | 909 | 226 |
| Q5 | 490 | 104 |
| Q6 | 43 | 31 |
| Q7 | 967 | 105 |
| Q8 | 96 | 88 |
| Q9 | 838 | 611 |
| Q10 | 325 | 221 |
| Q11 | 25 | 23 |
| Q12 | 132 | 118 |
| Q13 | 577 | 142 |
| Q14 | 77 | 61 |
| Q15 | 55 | 60 |
| Q16 | 96 | 122 |
| Q17 | 36 | 46 |
| Q18 | 1,510 | 610 |
| Q19 | 399 | 49 |
| Q20 | 106 | 72 |
| Q21 | 4,110 | 3,731 |
| Q22 | 192 | 123 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | mx-cpu |
|---|---:|---:|
| import | 75 | 106 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 9.6e-15, Polars 5.5e-02.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 9.7, max 16.6 over 88 runs; load average at report time: 16.6 11.2 8.0), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `6de51e6`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

