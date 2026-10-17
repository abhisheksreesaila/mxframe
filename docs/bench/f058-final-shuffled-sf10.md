# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-17

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 1,905 WRONG | 139 | **63** | 69 | - |
| Q2 min-cost supplier | 51 | 27 | **24** | 25 | - |
| Q3 3-join+top10 | 256 | **83** | 113 | 122 | - |
| Q4 EXISTS semi-join | 1,136 | 113 | 107 | **107** | - |
| Q5 4-join+groupby | 412 | 86 | **71** | 75 | - |
| Q6 filter+global sum | 82 | 32 | **31** | 34 | - |
| Q7 volume shipping | 1,165 †4e-6 | 86 | **83** | 88 | - |
| Q8 market share | 72 | 68 | **66** | 69 | - |
| Q9 6-join profit | 635 †6e-6 | **225** | 323 | 322 | - |
| Q10 returned items top20 | 283 | **127** | 160 | 161 | - |
| Q11 important stock | 26 | **15** | 33 | 37 | - |
| Q12 shipping modes | 622 | **80** | 119 | 122 | - |
| Q13 customer distribution | 682 | 221 | **88** | 89 | PyArrow/NumPy |
| Q14 promo revenue | 68 | 58 | **28** | 30 | - |
| Q15 top supplier | 76 | **57** | 59 | 64 | - |
| Q16 part/supplier | 95 | **94** | 101 | 100 | - |
| Q17 small-qty revenue | **27** | 27 | 65 | 69 | - |
| Q18 large volume top100 | 1,552 | **1,044** | 1,259 | 1,255 | - |
| Q19 discounted revenue | 726 | 87 | **64** | 67 | - |
| Q20 potential promotion | 131 | **49** | 64 | 71 | - |
| Q21 suppliers waiting | 4,533 | 2,701 | 1,298 | **1,288** | - |
| Q22 global sales opportunity | 171 | **41** | 87 | 94 | - |
| **sum over the 21 queries valid for every engine** | **12,801** | **5,320** | **4,243** | **4,290** |  |

Fastest verified engine per query: Polars 1, DuckDB 10, mx-cpu 9, mx-auto 2.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 18/22** (21 comparable); **mx-cpu beats DuckDB on 11/22** (22 comparable)
- **mx-auto beats Polars on 18/22** (21 comparable); **mx-auto beats DuckDB on 8/22** (22 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q17 small-qty revenue | mx-auto | 69 | 27 | 2.6x |
| Q11 important stock | mx-auto | 37 | 26 | 1.4x |
| Q16 part/supplier | mx-auto | 100 | 95 | 1.1x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q17 small-qty revenue | mx-cpu | 65 | 27 | 2.4x |
| Q11 important stock | mx-cpu | 33 | 26 | 1.3x |
| Q16 part/supplier | mx-auto | 100 | 95 | 1.1x |

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 1,977 WRONG | 250 | 169 | 174 |
| Q2 | 127 | 140 | 130 | 129 |
| Q3 | 328 | 194 | 215 | 227 |
| Q4 | 1,210 | 223 | 211 | 210 |
| Q5 | 485 | 198 | 177 | 178 |
| Q6 | 155 | 141 | 136 | 141 |
| Q7 | 1,238 †4e-6 | 198 | 186 | 193 |
| Q8 | 146 | 180 | 201 | 177 |
| Q9 | 707 †6e-6 | 336 | 428 | 427 |
| Q10 | 359 | 242 | 266 | 266 |
| Q11 | 101 | 126 | 140 | 139 |
| Q12 | 695 | 191 | 224 | 225 |
| Q13 | 756 | 330 | 191 | 193 |
| Q14 | 141 | 169 | 128 | 134 |
| Q15 | 150 | 166 | 173 | 167 |
| Q16 | 170 | 205 | 204 | 205 |
| Q17 | 103 | 137 | 168 | 172 |
| Q18 | 1,624 | 1,156 | 1,361 | 1,361 |
| Q19 | 800 | 196 | 173 | 170 |
| Q20 | 204 | 161 | 174 | 176 |
| Q21 | 4,606 | 2,810 | 1,404 | 1,396 |
| Q22 | 245 | 152 | 189 | 201 |
| **sum (21 queries)** | **14,349** | **7,651** | **6,480** | **6,488** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 530 | 620 | 490 | 501 |
| Q2 | 82 | 248 | 117 | 115 |
| Q3 | 323 | 494 | 673 | 670 |
| Q4 | 256 | 341 | 605 | 600 |
| Q5 | 356 | 499 | 681 | 678 |
| Q6 | 201 | 323 | 265 | 268 |
| Q7 | 387 | 576 | 715 | 720 |
| Q8 | 462 | 731 | 1,038 | 1,035 |
| Q9 | 519 | 804 | 1,177 | 1,183 |
| Q10 | 420 | 493 | 680 | 682 |
| Q11 | 58 | 97 | 83 | 78 |
| Q12 | 433 | 481 | 692 | 694 |
| Q13 | 76 | 99 | 207 | 209 |
| Q14 | 262 | 449 | 764 | 772 |
| Q15 | 243 | 392 | 378 | 389 |
| Q16 | 65 | 176 | 99 | 101 |
| Q17 | 239 | 396 | 746 | 749 |
| Q18 | 202 | 247 | 596 | 594 |
| Q19 | 567 | 759 | 893 | 888 |
| Q20 | 353 | 559 | 912 | 911 |
| Q21 | 332 | 428 | 652 | 655 |
| Q22 | 80 | 135 | 245 | 239 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 1,501 WRONG | 143 | 62 | 64 |
| Q2 | 47 | 24 | 21 | 22 |
| Q3 | 199 | 84 | 98 | 101 |
| Q4 | 1,038 | 100 | 89 | 88 |
| Q5 | 414 | 85 | 61 | 63 |
| Q6 | 76 | 32 | 29 | 31 |
| Q7 | 1,064 | 85 | 73 | 75 |
| Q8 | 75 | 61 | 55 | 56 |
| Q9 | 603 | 222 | 319 | 318 |
| Q10 | 261 | 130 | 127 | 135 |
| Q11 | 27 | 15 | 25 | 27 |
| Q12 | 615 | 77 | 109 | 111 |
| Q13 | 601 | 221 | 88 | 88 |
| Q14 | 65 | 61 | 25 | 25 |
| Q15 | 72 | 57 | 57 | 59 |
| Q16 | 91 | 107 | 88 | 93 |
| Q17 | 26 | 28 | 65 | 63 |
| Q18 | 1,378 | 1,020 | 1,225 | 1,223 |
| Q19 | 721 | 84 | 62 | 63 |
| Q20 | 130 | 50 | 59 | - |
| Q21 | 4,209 | 2,701 | 1,085 | 1,084 |
| Q22 | 183 | 38 | 84 | 84 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| import | 73 | 111 | 105 | 105 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 3.69e-06 | max relative error 3.69e-06 |
| Q9 | polars | drift | 5.94e-06 | max relative error 5.94e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 1.7e-14, mx-auto 1.9e-14.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.2, max 11.2 over 88 runs; load average at report time: 5.3 4.1 5.2), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `8e001f3`. Last invocation took 17 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

