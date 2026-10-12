# Engine-only TPC-H scoreboard - 2026-10-12

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 669 | 255 †3e-3 | 97 | **22** | 306 | 23 | - |
| Q2 min-cost supplier | 40 | **7.7** | 83 | 19 | 376 | 25 | - |
| Q3 3-join+top10 | 119 | **31** | 90 | 91 | 340 | 100 | - |
| Q4 EXISTS semi-join | 158 | **86** | 93 | 122 | 390 | 129 | - |
| Q5 4-join+groupby | 166 | **54** | 93 | 55 | 322 | 68 | - |
| Q6 filter+global sum | 29 | **6.4** | 80 | 6.6 | 260 | 10.0 | - |
| Q7 volume shipping | 258 | 112 | **89** | 105 | 321 | 117 | - |
| Q8 market share | 47 | **13** | 88 | 68 | 372 | 71 | - |
| Q9 6-join profit | 157 | **71 †2e-6** | 128 | 148 | 326 | 163 | - |
| Q10 returned items top20 | 125 | **39** | 112 | 121 | 334 | 118 | - |
| Q11 important stock | 12 | **5.1** | 80 | 11 | 354 | 15 | - |
| Q12 shipping modes | 121 | **19** | 84 | 170 | 400 | 190 | - |
| Q13 customer distribution | 100 | **53** | 114 | 137 | 366 | 162 | PyArrow/NumPy |
| Q14 promo revenue | 29 | **10** | 89 | 36 | 303 | 39 | - |
| Q15 top supplier | 35 | **9.1** | 89 | 31 | 415 | 35 | - |
| Q16 part/supplier | 44 | **18** | 102 | 50 | 336 | 58 | - |
| Q17 small-qty revenue | 22 | **7.7** | 83 | 35 | 278 | 45 | - |
| Q18 large volume top100 | 233 | 141 | **107** | 302 | 413 | 312 | - |
| Q19 discounted revenue | 677 | **44** | 98 | 216 | 475 | 226 | - |
| Q20 potential promotion | 84 | **16** | 116 | 56 | 303 | 62 | - |
| Q21 suppliers waiting | 1,353 | 378 | **297** | 709 | 903 | 791 | - |
| Q22 global sales opportunity | 21 | **14** | 88 | 14 | 259 | 17 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **4,499** | **1,391** | **2,299** | **2,524** | **8,151** | **2,777** |  |

Fastest verified engine per query: pandas 0, Polars 18, DuckDB 3, mx-cpu 1, mx-gpu 0, mx-auto 0.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 2/22** (22 comparable); **mx-cpu beats pandas on 15/22** (22 comparable); **mx-cpu beats DuckDB on 12/22** (22 comparable)
- **mx-gpu beats Polars on 0/22** (22 comparable); **mx-gpu beats pandas on 3/22** (22 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 1/22** (22 comparable); **mx-auto beats pandas on 12/22** (22 comparable); **mx-auto beats DuckDB on 12/22** (22 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q12 shipping modes | mx-auto | 190 | 19 | 9.9x |
| Q17 small-qty revenue | mx-auto | 45 | 7.7 | 5.8x |
| Q8 market share | mx-auto | 71 | 13 | 5.2x |
| Q19 discounted revenue | mx-auto | 226 | 44 | 5.1x |
| Q20 potential promotion | mx-auto | 62 | 16 | 3.9x |
| Q15 top supplier | mx-auto | 35 | 9.1 | 3.9x |
| Q14 promo revenue | mx-auto | 39 | 10 | 3.8x |
| Q16 part/supplier | mx-auto | 58 | 18 | 3.3x |
| Q2 min-cost supplier | mx-auto | 25 | 7.7 | 3.2x |
| Q3 3-join+top10 | mx-auto | 100 | 31 | 3.2x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q12 shipping modes | mx-cpu | 170 | 19 | 8.8x |
| Q8 market share | mx-cpu | 68 | 13 | 5.0x |
| Q19 discounted revenue | mx-cpu | 216 | 44 | 4.9x |
| Q17 small-qty revenue | mx-cpu | 35 | 7.7 | 4.6x |
| Q14 promo revenue | mx-cpu | 36 | 10 | 3.5x |
| Q20 potential promotion | mx-cpu | 56 | 16 | 3.5x |
| Q15 top supplier | mx-cpu | 31 | 9.1 | 3.4x |
| Q10 returned items top20 | mx-auto | 118 | 39 | 3.0x |
| Q3 3-join+top10 | mx-cpu | 91 | 31 | 2.9x |
| Q16 part/supplier | mx-cpu | 50 | 18 | 2.9x |

### SF1: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 960 | 326 †3e-3 | 140 | 162 | 445 | 152 |
| Q2 | 328 | 82 | 123 | 144 | 536 | 148 |
| Q3 | 411 | 111 | 129 | 220 | 500 | 262 |
| Q4 | 443 | 167 | 144 | 286 | 540 | 275 |
| Q5 | 457 | 130 | 136 | 222 | 455 | 219 |
| Q6 | 281 | 81 | 123 | 171 | 418 | 164 |
| Q7 | 505 | 189 | 131 | 262 | 460 | 280 |
| Q8 | 316 | 95 | 127 | 234 | 518 | 220 |
| Q9 | 393 | 144 †2e-6 | 168 | 273 | 499 | 342 |
| Q10 | 391 | 118 | 152 | 272 | 496 | 265 |
| Q11 | 297 | 78 | 120 | 173 | 496 | 144 |
| Q12 | 401 | 96 | 122 | 302 | 561 | 349 |
| Q13 | 380 | 131 | 159 | 296 | 527 | 332 |
| Q14 | 314 | 94 | 127 | 189 | 471 | 179 |
| Q15 | 294 | 87 | 136 | 191 | 576 | 191 |
| Q16 | 341 | 92 | 143 | 212 | 494 | 205 |
| Q17 | 300 | 85 | 125 | 203 | 442 | 205 |
| Q18 | 547 | 222 | 149 | 469 | 556 | 469 |
| Q19 | 943 | 121 | 136 | 360 | 624 | 388 |
| Q20 | 349 | 91 | 157 | 181 | 432 | 187 |
| Q21 | 1,593 | 452 | 336 | 840 | 1,072 | 914 |
| Q22 | 292 | 86 | 127 | 144 | 384 | 164 |
| **sum (22 queries)** | **10,536** | **3,080** | **3,209** | **5,807** | **11,502** | **6,052** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 181 | 65 | 134 | 64 | 64 | 60 |
| Q2 | 43 | 30 | 65 | 43 | 43 | 38 |
| Q3 | 121 | 59 | 116 | 112 | 105 | 108 |
| Q4 | 114 | 55 | 110 | 78 | 80 | 80 |
| Q5 | 119 | 68 | 113 | 109 | 101 | 115 |
| Q6 | 64 | 36 | 69 | 46 | 44 | 43 |
| Q7 | 124 | 71 | 119 | 107 | 113 | 115 |
| Q8 | 153 | 78 | 140 | 151 | 137 | 147 |
| Q9 | 161 | 81 | 166 | 149 | 153 | 160 |
| Q10 | 156 | 72 | 136 | 106 | 103 | 102 |
| Q11 | 30 | 27 | 48 | 38 | 37 | 35 |
| Q12 | 189 | 71 | 140 | 85 | 103 | 83 |
| Q13 | 39 | 30 | 45 | 46 | 50 | 55 |
| Q14 | 102 | 49 | 99 | 103 | 104 | 111 |
| Q15 | 83 | 43 | 73 | 61 | 70 | 62 |
| Q16 | 35 | 31 | 50 | 43 | 41 | 46 |
| Q17 | 96 | 48 | 75 | 106 | 109 | 103 |
| Q18 | 85 | 45 | 71 | 101 | 94 | 95 |
| Q19 | 223 | 82 | 161 | 120 | 127 | 121 |
| Q20 | 128 | 58 | 109 | 118 | 116 | 112 |
| Q21 | 121 | 58 | 96 | 99 | 99 | 97 |
| Q22 | 33 | 28 | 35 | 38 | 36 | 35 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 686 | 205 | 21 | 17 | 32 | 20 |
| Q2 | 40 | 6.5 | 10 | 20 | 19 | 19 |
| Q3 | 98 | 26 | 17 | 84 | 67 | 80 |
| Q4 | 165 | 68 | 18 | 109 | 104 | 111 |
| Q5 | 142 | 44 | 15 | 47 | 32 | 56 |
| Q6 | 27 | 6.4 | 5.3 | 5.6 | 15 | 5.2 |
| Q7 | 248 | 99 | 18 | 82 | 66 | 81 |
| Q8 | 41 | 15 | 15 | 59 | 33 | 60 |
| Q9 | 158 | 56 | 56 | 139 | 97 | 155 |
| Q10 | 105 | 37 | 24 | 107 | 96 | 113 |
| Q11 | 11 | 4.4 | 5.3 | 8.8 | 8.0 | 11 |
| Q12 | 117 | 16 | 17 | 166 | 160 | 197 |
| Q13 | 87 | 43 | 29 | 18 | 17 | 24 |
| Q14 | 29 | 8.3 | 15 | 32 | 32 | 38 |
| Q15 | 29 | 8.9 | 9.5 | 25 | 39 | 28 |
| Q16 | 42 | 18 | 24 | 43 | 45 | 48 |
| Q17 | 22 | 7.7 | 9.4 | 33 | 21 | 34 |
| Q18 | 259 | 131 | 38 | 295 | 120 | 283 |
| Q19 | 686 | 44 | 21 | 209 | 205 | 219 |
| Q20 | 78 | 14 | 13 | 53 | 48 | 61 |
| Q21 | 1,320 | 357 | 256 | 748 | 629 | 726 |
| Q22 | 20 | 15 | 14 | 12 | 13 | 13 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| import | 278 | 77 | 40 | 155 | 158 | 150 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.1e-07, Polars 3.1e-03, mx-cpu 3.3e-15, mx-auto 3.0e-15, mx-gpu 2.5e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.0, max 4.6 over 132 runs; load average at report time: 4.1 3.3 3.5), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `302cc70`. Last invocation took 2 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

