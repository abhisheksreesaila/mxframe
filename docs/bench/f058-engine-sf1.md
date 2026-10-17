# Engine-only TPC-H scoreboard - 2026-10-17

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 242 †3e-3 | 19 | **13** | 324 | 40 | - |
| Q2 min-cost supplier | **8.0** | 13 | 12 | 412 | 16 | - |
| Q3 3-join+top10 | 32 | **17** | 26 | 341 | 53 | - |
| Q4 EXISTS semi-join | 77 | **16** | 20 | 362 | 43 | - |
| Q5 4-join+groupby | 47 | **15** | 18 | 314 | 40 | - |
| Q6 filter+global sum | 6.4 | **5.6** | 15 | 284 | 12 | - |
| Q7 volume shipping | 107 | **18** | 19 | 345 | 43 | - |
| Q8 market share | **13** | 16 | 20 | 444 | 42 | - |
| Q9 6-join profit | 70 †2e-6 | 62 | **38** | 392 | 60 | - |
| Q10 returned items top20 | 36 | 23 | **23** | 330 | 48 | - |
| Q11 important stock | **4.4** | 6.3 | 18 | 299 | 42 | - |
| Q12 shipping modes | 20 | **14** | 17 | 346 | 39 | - |
| Q13 customer distribution | 51 | 32 | **10** | 415 | 35 | PyArrow/NumPy |
| Q14 promo revenue | 10 | 15 | **10.0** | 317 | 13 | - |
| Q15 top supplier | **8.2** | 8.7 | 13 | 418 | 37 | - |
| Q16 part/supplier | **14** | 26 | 31 | 336 | 34 | - |
| Q17 small-qty revenue | **7.4** | 8.9 | 13 | 305 | 16 | - |
| Q18 large volume top100 | 142 | **34** | 57 | 404 | 82 | - |
| Q19 discounted revenue | 36 | 20 | **18** | 374 | 20 | - |
| Q20 potential promotion | 15 | **13** | 18 | 318 | 44 | - |
| Q21 suppliers waiting | 350 | 225 | **82** | 808 | 109 | - |
| Q22 global sales opportunity | 16 | **15** | 21 | 299 | 42 | - |
| **sum over the 22 queries valid for every engine** | **1,311** | **622** | **514** | **8,188** | **911** |  |

Fastest verified engine per query: Polars 6, DuckDB 9, mx-cpu 7, mx-gpu 0, mx-auto 0.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 13/22** (22 comparable); **mx-cpu beats DuckDB on 8/22** (22 comparable)
- **mx-gpu beats Polars on 0/22** (22 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 9/22** (22 comparable); **mx-auto beats DuckDB on 3/22** (22 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-auto | 42 | 4.4 | 9.5x |
| Q15 top supplier | mx-auto | 37 | 8.2 | 4.6x |
| Q8 market share | mx-auto | 42 | 13 | 3.4x |
| Q20 potential promotion | mx-auto | 44 | 15 | 2.9x |
| Q22 global sales opportunity | mx-auto | 42 | 16 | 2.6x |
| Q16 part/supplier | mx-auto | 34 | 14 | 2.3x |
| Q17 small-qty revenue | mx-auto | 16 | 7.4 | 2.2x |
| Q2 min-cost supplier | mx-auto | 16 | 8.0 | 2.0x |
| Q12 shipping modes | mx-auto | 39 | 20 | 2.0x |
| Q6 filter+global sum | mx-auto | 12 | 6.4 | 1.9x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-cpu | 18 | 4.4 | 4.1x |
| Q16 part/supplier | mx-cpu | 31 | 14 | 2.1x |
| Q6 filter+global sum | mx-auto | 12 | 6.4 | 1.9x |
| Q17 small-qty revenue | mx-cpu | 13 | 7.4 | 1.8x |
| Q15 top supplier | mx-cpu | 13 | 8.2 | 1.6x |
| Q8 market share | mx-cpu | 20 | 13 | 1.6x |
| Q2 min-cost supplier | mx-cpu | 12 | 8.0 | 1.5x |
| Q22 global sales opportunity | mx-cpu | 21 | 16 | 1.3x |
| Q20 potential promotion | mx-cpu | 18 | 15 | 1.2x |

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 314 †3e-3 | 130 | 118 | 433 | 142 |
| Q2 | 80 | 121 | 117 | 549 | 119 |
| Q3 | 104 | 127 | 139 | 460 | 156 |
| Q4 | 148 | 123 | 122 | 463 | 146 |
| Q5 | 120 | 124 | 119 | 417 | 142 |
| Q6 | 82 | 117 | 117 | 386 | 117 |
| Q7 | 182 | 128 | 127 | 450 | 150 |
| Q8 | 84 | 123 | 122 | 545 | 148 |
| Q9 | 142 †2e-6 | 169 | 140 | 497 | 163 |
| Q10 | 108 | 131 | 127 | 434 | 182 |
| Q11 | 77 | 116 | 120 | 403 | 144 |
| Q12 | 92 | 124 | 118 | 453 | 144 |
| Q13 | 124 | 142 | 112 | 517 | 137 |
| Q14 | 86 | 125 | 119 | 423 | 117 |
| Q15 | 81 | 116 | 130 | 520 | 139 |
| Q16 | 87 | 135 | 133 | 439 | 154 |
| Q17 | 82 | 120 | 117 | 407 | 122 |
| Q18 | 214 | 145 | 158 | 508 | 186 |
| Q19 | 108 | 132 | 120 | 477 | 135 |
| Q20 | 89 | 123 | 122 | 422 | 146 |
| Q21 | 422 | 338 | 184 | 909 | 211 |
| Q22 | 88 | 124 | 123 | 401 | 147 |
| **sum (22 queries)** | **2,914** | **3,033** | **2,803** | **10,514** | **3,248** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 71 | 152 | 58 | 51 | 55 |
| Q2 | 31 | 75 | 35 | 35 | 30 |
| Q3 | 57 | 126 | 85 | 89 | 85 |
| Q4 | 51 | 117 | 67 | 69 | 72 |
| Q5 | 62 | 121 | 87 | 92 | 90 |
| Q6 | 37 | 84 | 31 | 30 | 38 |
| Q7 | 62 | 117 | 94 | 90 | 90 |
| Q8 | 76 | 153 | 129 | 124 | 128 |
| Q9 | 87 | 184 | 140 | 140 | 142 |
| Q10 | 69 | 132 | 86 | 88 | 88 |
| Q11 | 25 | 64 | 19 | 19 | 20 |
| Q12 | 67 | 150 | 76 | 77 | 80 |
| Q13 | 29 | 56 | 31 | 30 | 30 |
| Q14 | 42 | 96 | 89 | 91 | 94 |
| Q15 | 44 | 82 | 51 | 47 | 45 |
| Q16 | 30 | 61 | 27 | 25 | 25 |
| Q17 | 42 | 88 | 82 | 91 | 88 |
| Q18 | 47 | 85 | 69 | 70 | 68 |
| Q19 | 78 | 181 | 96 | 95 | 105 |
| Q20 | 64 | 130 | 105 | 104 | 107 |
| Q21 | 58 | 116 | 76 | 76 | 77 |
| Q22 | 24 | 54 | 27 | 29 | 24 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 197 | 17 | 10 | 27 | 9.6 |
| Q2 | 7.4 | 9.8 | 7.0 | 9.8 | 7.8 |
| Q3 | 26 | 17 | 17 | 58 | 20 |
| Q4 | 66 | 14 | 12 | 70 | 13 |
| Q5 | 39 | 14 | 10 | 30 | 12 |
| Q6 | 5.9 | 5.3 | 4.4 | 13 | 4.9 |
| Q7 | 84 | 19 | 14 | 63 | 14 |
| Q8 | 13 | 15 | 12 | 31 | 12 |
| Q9 | 60 | 56 | 29 | 95 | 31 |
| Q10 | 32 | 24 | 16 | 64 | 18 |
| Q11 | 5.6 | 5.3 | 7.2 | 10 | 9.1 |
| Q12 | 16 | 13 | 12 | 61 | 12 |
| Q13 | 42 | 25 | 5.7 | 12 | 8.9 |
| Q14 | 8.2 | 13 | 5.0 | 30 | 5.4 |
| Q15 | 8.5 | 9.0 | 8.1 | 26 | 11 |
| Q16 | 17 | 23 | 23 | 37 | 24 |
| Q17 | 6.1 | 8.7 | 7.1 | 18 | 7.1 |
| Q18 | 115 | 32 | 42 | 108 | 47 |
| Q19 | 38 | 20 | 13 | 86 | 13 |
| Q20 | 14 | 11 | 12 | 46 | 18 |
| Q21 | 325 | 234 | 61 | 527 | 64 |
| Q22 | 17 | 14 | 16 | 18 | 16 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| import | 72 | 110 | 102 | 103 | 104 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 3.1e-03, mx-cpu 3.3e-15, mx-gpu 3.0e-15, mx-auto 3.7e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.6, max 8.0 over 110 runs; load average at report time: 7.5 2.2 0.9), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

