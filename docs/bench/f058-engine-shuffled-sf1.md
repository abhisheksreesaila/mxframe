# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-17

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 173 †3e-3 | 19 | **13** | 321 | 15 | - |
| Q2 min-cost supplier | **7.2** | 11 | 12 | 288 | 15 | - |
| Q3 3-join+top10 | 25 | **14** | 24 | 366 | 31 | - |
| Q4 EXISTS semi-join | 77 | **15** | 20 | 397 | 22 | - |
| Q5 4-join+groupby | 40 | **14** | 17 | 325 | 19 | - |
| Q6 filter+global sum | 9.9 | **4.8** | 7.9 | 303 | 11 | - |
| Q7 volume shipping | 110 | 18 | **18** | 568 | 23 | - |
| Q8 market share | **12** | 15 | 20 | 313 | 24 | - |
| Q9 6-join profit | 68 †2e-6 | **41** | 53 | 427 | 57 | - |
| Q10 returned items top20 | 43 | **28** | 31 | 358 | 34 | - |
| Q11 important stock | **4.2** | 7.0 | 12 | 291 | 18 | - |
| Q12 shipping modes | 71 | **17** | 23 | 348 | 28 | - |
| Q13 customer distribution | 61 | 31 | **12** | 276 | 18 | PyArrow/NumPy |
| Q14 promo revenue | **11** | 13 | 11 | 315 | 12 | - |
| Q15 top supplier | 11 | **9.6** | 17 | 310 | 21 | - |
| Q16 part/supplier | **18** | 26 | 38 | 323 | 40 | - |
| Q17 small-qty revenue | **8.7** | 9.8 | 12 | 291 | 16 | - |
| Q18 large volume top100 | 151 | **102** | 132 | 480 | 148 | - |
| Q19 discounted revenue | 93 | 23 | **17** | 343 | 19 | - |
| Q20 potential promotion | 22 | **14** | 20 | 349 | 25 | - |
| Q21 suppliers waiting | 444 | 275 | **186** | 1,131 | 190 | - |
| Q22 global sales opportunity | **15** | 15 | 20 | 286 | 24 | - |
| **sum over the 22 queries valid for every engine** | **1,476** | **724** | **715** | **8,407** | **812** |  |

Fastest verified engine per query: Polars 7, DuckDB 10, mx-cpu 5, mx-gpu 0, mx-auto 0.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 14/22** (22 comparable); **mx-cpu beats DuckDB on 6/22** (22 comparable)
- **mx-gpu beats Polars on 0/22** (22 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 11/22** (22 comparable); **mx-auto beats DuckDB on 5/22** (22 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-auto | 18 | 4.2 | 4.3x |
| Q16 part/supplier | mx-auto | 40 | 18 | 2.1x |
| Q2 min-cost supplier | mx-auto | 15 | 7.2 | 2.0x |
| Q8 market share | mx-auto | 24 | 12 | 2.0x |
| Q15 top supplier | mx-auto | 21 | 11 | 1.9x |
| Q17 small-qty revenue | mx-auto | 16 | 8.7 | 1.9x |
| Q22 global sales opportunity | mx-auto | 24 | 15 | 1.6x |
| Q3 3-join+top10 | mx-auto | 31 | 25 | 1.2x |
| Q20 potential promotion | mx-auto | 25 | 22 | 1.2x |
| Q6 filter+global sum | mx-auto | 11 | 9.9 | 1.1x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-cpu | 12 | 4.2 | 2.8x |
| Q16 part/supplier | mx-cpu | 38 | 18 | 2.1x |
| Q8 market share | mx-cpu | 20 | 12 | 1.7x |
| Q2 min-cost supplier | mx-cpu | 12 | 7.2 | 1.6x |
| Q15 top supplier | mx-cpu | 17 | 11 | 1.5x |
| Q17 small-qty revenue | mx-cpu | 12 | 8.7 | 1.4x |
| Q22 global sales opportunity | mx-cpu | 20 | 15 | 1.3x |
| Q14 promo revenue | mx-cpu | 11 | 11 | 1.0x |

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 245 †3e-3 | 131 | 116 | 422 | 129 |
| Q2 | 78 | 118 | 113 | 389 | 117 |
| Q3 | 97 | 125 | 129 | 489 | 133 |
| Q4 | 150 | 122 | 121 | 498 | 125 |
| Q5 | 114 | 123 | 118 | 427 | 123 |
| Q6 | 82 | 112 | 127 | 409 | 114 |
| Q7 | 184 | 130 | 125 | 669 | 145 |
| Q8 | 91 | 149 | 131 | 426 | 138 |
| Q9 | 147 †2e-6 | 150 | 165 | 542 | 174 |
| Q10 | 122 | 143 | 145 | 484 | 150 |
| Q11 | 87 | 149 | 127 | 404 | 132 |
| Q12 | 151 | 136 | 138 | 461 | 135 |
| Q13 | 140 | 149 | 120 | 391 | 129 |
| Q14 | 90 | 136 | 127 | 424 | 127 |
| Q15 | 91 | 133 | 130 | 423 | 135 |
| Q16 | 97 | 153 | 153 | 438 | 151 |
| Q17 | 90 | 130 | 121 | 406 | 122 |
| Q18 | 227 | 231 | 242 | 595 | 260 |
| Q19 | 167 | 146 | 125 | 461 | 127 |
| Q20 | 100 | 134 | 138 | 474 | 133 |
| Q21 | 522 | 392 | 301 | 1,239 | 311 |
| Q22 | 97 | 131 | 127 | 406 | 138 |
| **sum (22 queries)** | **3,171** | **3,324** | **3,137** | **10,878** | **3,247** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 70 | 148 | 55 | 48 | 54 |
| Q2 | 32 | 82 | 29 | 29 | 30 |
| Q3 | 55 | 134 | 88 | 88 | 80 |
| Q4 | 49 | 117 | 71 | 75 | 71 |
| Q5 | 62 | 121 | 87 | 90 | 93 |
| Q6 | 33 | 83 | 39 | 38 | 39 |
| Q7 | 66 | 124 | 96 | 94 | 110 |
| Q8 | 90 | 160 | 130 | 150 | 139 |
| Q9 | 103 | 188 | 172 | 162 | 167 |
| Q10 | 79 | 139 | 105 | 101 | 101 |
| Q11 | 31 | 55 | 28 | 19 | 18 |
| Q12 | 76 | 167 | 94 | 87 | 90 |
| Q13 | 37 | 51 | 40 | 41 | 42 |
| Q14 | 53 | 98 | 106 | 105 | 101 |
| Q15 | 50 | 91 | 62 | 60 | 65 |
| Q16 | 31 | 67 | 31 | 25 | 26 |
| Q17 | 50 | 91 | 98 | 99 | 87 |
| Q18 | 54 | 85 | 88 | 86 | 85 |
| Q19 | 92 | 189 | 103 | 117 | 114 |
| Q20 | 73 | 139 | 121 | 124 | 105 |
| Q21 | 64 | 123 | 95 | 101 | 93 |
| Q22 | 30 | 44 | 25 | 28 | 26 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 145 | 19 | 8.6 | 30 | 9.6 |
| Q2 | 6.3 | 8.2 | 7.0 | 9.4 | 8.1 |
| Q3 | 23 | 16 | 16 | 78 | 155 |
| Q4 | 68 | 15 | 12 | 103 | 176 |
| Q5 | 34 | 14 | 10 | 34 | 118 |
| Q6 | 8.8 | 5.2 | 4.0 | 13 | 4.4 |
| Q7 | 96 | 17 | 13 | 71 | 178 |
| Q8 | 15 | 16 | 14 | 36 | 112 |
| Q9 | 64 | 37 | 43 | 125 | 236 |
| Q10 | 38 | 25 | 17 | 74 | 176 |
| Q11 | 3.8 | 7.2 | 7.0 | 11 | 8.6 |
| Q12 | 73 | 17 | 19 | 54 | 145 |
| Q13 | 49 | 25 | 6.2 | 15 | 8.9 |
| Q14 | 10 | 15 | 5.5 | 43 | 125 |
| Q15 | 11 | 10 | 11 | 34 | 12 |
| Q16 | 15 | 24 | 28 | 42 | 28 |
| Q17 | 5.5 | 8.8 | 7.5 | 16 | 111 |
| Q18 | 132 | 104 | 111 | 175 | 281 |
| Q19 | 74 | 22 | 13 | 74 | 167 |
| Q20 | 20 | 13 | 13 | 56 | 160 |
| Q21 | 433 | 285 | 112 | 1,102 | 1,080 |
| Q22 | 13 | 16 | 15 | 17 | 16 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| import | 78 | 117 | 111 | 113 | 112 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.13e-03 | max relative error 3.13e-03 |
| Q9 | polars | drift | 1.86e-06 | max relative error 1.86e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 3.1e-03, mx-cpu 2.5e-15, mx-gpu 2.3e-15, mx-auto 2.4e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.8, max 7.7 over 110 runs; load average at report time: 5.3 3.3 1.5), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

