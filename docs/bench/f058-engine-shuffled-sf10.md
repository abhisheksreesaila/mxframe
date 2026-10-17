# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-17

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,011 WRONG | 138 | 70 | 549 | **69** | - |
| Q2 min-cost supplier | 49 | **25** | 27 | 322 | 31 | - |
| Q3 3-join+top10 | 271 | **82** | 150 | 1,153 | 155 | - |
| Q4 EXISTS semi-join | 1,468 | **131** | 139 | 1,816 | 145 | - |
| Q5 4-join+groupby | 559 | 112 | **72** | 597 | 616 | - |
| Q6 filter+global sum | 80 | **31** | 33 | 406 | 35 | - |
| Q7 volume shipping | 1,123 †4e-6 | **85** | 90 | 930 | 89 | - |
| Q8 market share | 72 | **61** | 67 | 578 | 571 | - |
| Q9 6-join profit | 644 †6e-6 | **229** | 323 | 1,291 | 1,282 | - |
| Q10 returned items top20 | 280 | **128** | 162 | 944 | 165 | - |
| Q11 important stock | 27 | **16** | 35 | 372 | 40 | - |
| Q12 shipping modes | 626 | **81** | 121 | 805 | 124 | - |
| Q13 customer distribution | 681 | 221 | **93** | 357 | 99 | PyArrow/NumPy |
| Q14 promo revenue | 72 | 56 | **30** | 566 | 33 | - |
| Q15 top supplier | 78 | **57** | 61 | 557 | 68 | - |
| Q16 part/supplier | **94** | 99 | 103 | 652 | 112 | - |
| Q17 small-qty revenue | **26** | 27 | 69 | 423 | 71 | - |
| Q18 large volume top100 | 1,461 | **1,040** | 1,247 | 2,246 | 2,238 | - |
| Q19 discounted revenue | 719 | 82 | **67** | 1,021 | 70 | - |
| Q20 potential promotion | 131 | **51** | 67 | 730 | 71 | - |
| Q21 suppliers waiting | 4,408 | 2,715 | **1,361** | 11,223 | 11,032 | - |
| Q22 global sales opportunity | 200 | **40** | 92 | 380 | 93 | - |
| **sum over the 21 queries valid for every engine** | **13,069** | **5,369** | **4,407** | **27,367** | **17,140** |  |

Fastest verified engine per query: Polars 2, DuckDB 14, mx-cpu 5, mx-gpu 0, mx-auto 1.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 18/22** (21 comparable); **mx-cpu beats DuckDB on 6/22** (22 comparable)
- **mx-gpu beats Polars on 2/22** (21 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 13/22** (21 comparable); **mx-auto beats DuckDB on 4/22** (22 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q8 market share | mx-auto | 571 | 72 | 7.9x |
| Q17 small-qty revenue | mx-auto | 71 | 26 | 2.8x |
| Q21 suppliers waiting | mx-auto | 11,032 | 4,408 | 2.5x |
| Q9 6-join profit | mx-auto | 1,282 | 644 | 2.0x |
| Q18 large volume top100 | mx-auto | 2,238 | 1,461 | 1.5x |
| Q11 important stock | mx-auto | 40 | 27 | 1.5x |
| Q16 part/supplier | mx-auto | 112 | 94 | 1.2x |
| Q5 4-join+groupby | mx-auto | 616 | 559 | 1.1x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q17 small-qty revenue | mx-cpu | 69 | 26 | 2.7x |
| Q11 important stock | mx-cpu | 35 | 27 | 1.3x |
| Q16 part/supplier | mx-cpu | 103 | 94 | 1.1x |

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 2,084 WRONG | 251 | 173 | 650 | 172 |
| Q2 | 122 | 134 | 134 | 426 | 133 |
| Q3 | 344 | 192 | 263 | 1,269 | 269 |
| Q4 | 1,548 | 250 | 253 | 1,928 | 262 |
| Q5 | 636 | 234 | 203 | 702 | 733 |
| Q6 | 155 | 143 | 139 | 509 | 139 |
| Q7 | 1,199 †4e-6 | 195 | 192 | 1,033 | 192 |
| Q8 | 144 | 170 | 173 | 682 | 676 |
| Q9 | 718 †6e-6 | 338 | 425 | 1,400 | 1,384 |
| Q10 | 353 | 237 | 266 | 1,052 | 269 |
| Q11 | 99 | 127 | 139 | 476 | 143 |
| Q12 | 698 | 192 | 225 | 909 | 258 |
| Q13 | 757 | 331 | 196 | 460 | 204 |
| Q14 | 143 | 164 | 132 | 671 | 139 |
| Q15 | 150 | 171 | 164 | 660 | 170 |
| Q16 | 166 | 212 | 205 | 755 | 216 |
| Q17 | 99 | 138 | 192 | 526 | 176 |
| Q18 | 1,533 | 1,151 | 1,350 | 2,352 | 2,342 |
| Q19 | 792 | 191 | 170 | 1,128 | 172 |
| Q20 | 203 | 165 | 172 | 833 | 173 |
| Q21 | 4,481 | 2,825 | 1,465 | 11,333 | 11,141 |
| Q22 | 276 | 153 | 195 | 482 | 197 |
| **sum (21 queries)** | **14,618** | **7,716** | **6,653** | **29,586** | **19,387** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 526 | 620 | 497 | 495 | 488 |
| Q2 | 83 | 230 | 119 | 114 | 108 |
| Q3 | 323 | 469 | 778 | 769 | 785 |
| Q4 | 308 | 429 | 689 | 665 | 684 |
| Q5 | 416 | 694 | 826 | 680 | 677 |
| Q6 | 202 | 322 | 267 | 267 | 267 |
| Q7 | 386 | 554 | 716 | 710 | 718 |
| Q8 | 459 | 724 | 1,034 | 1,022 | 1,032 |
| Q9 | 523 | 803 | 1,174 | 1,172 | 1,170 |
| Q10 | 416 | 477 | 678 | 670 | 673 |
| Q11 | 54 | 92 | 76 | 79 | 78 |
| Q12 | 435 | 477 | 681 | 694 | 694 |
| Q13 | 75 | 97 | 213 | 202 | 205 |
| Q14 | 265 | 448 | 766 | 772 | 771 |
| Q15 | 242 | 393 | 384 | 382 | 374 |
| Q16 | 70 | 174 | 100 | 98 | 97 |
| Q17 | 241 | 403 | 748 | 750 | 737 |
| Q18 | 204 | 249 | 595 | 597 | 601 |
| Q19 | 568 | 755 | 891 | 922 | 879 |
| Q20 | 358 | 626 | 910 | 914 | 915 |
| Q21 | 327 | 429 | 658 | 649 | 653 |
| Q22 | 78 | 151 | 244 | 238 | 239 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 1,596 WRONG | 137 | 63 | 260 | 63 |
| Q2 | 48 | 28 | 22 | 36 | 120 |
| Q3 | 296 | 83 | 130 | 823 | 935 |
| Q4 | 1,405 | 139 | 88 | 1,543 | 1,652 |
| Q5 | 557 | 109 | 61 | 308 | 306 |
| Q6 | 77 | 32 | 29 | 113 | 30 |
| Q7 | 1,050 | 88 | 76 | 640 | 733 |
| Q8 | 73 | 61 | 55 | 279 | 278 |
| Q9 | 609 | 228 | 320 | 977 | 986 |
| Q10 | 259 | 128 | 128 | 645 | 743 |
| Q11 | 24 | 15 | 27 | 77 | 176 |
| Q12 | 790 | 84 | 108 | 498 | 588 |
| Q13 | 609 | 224 | 82 | 64 | 87 |
| Q14 | 63 | 57 | 24 | 265 | 357 |
| Q15 | 72 | 57 | 57 | 256 | 150 |
| Q16 | 92 | 106 | 89 | 339 | 425 |
| Q17 | 26 | 28 | 64 | 143 | 222 |
| Q18 | 1,403 | 1,038 | 1,221 | 1,900 | 1,904 |
| Q19 | 725 | 84 | 63 | 705 | 850 |
| Q20 | 132 | 51 | 64 | 450 | 545 |
| Q21 | 4,470 | 2,726 | 1,101 | 10,950 | 10,986 |
| Q22 | 156 | 41 | 84 | 97 | 172 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| import | 73 | 111 | 104 | 104 | 104 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 3.69e-06 | max relative error 3.69e-06 |
| Q9 | polars | drift | 5.94e-06 | max relative error 5.94e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 1.9e-14, mx-gpu 1.6e-14, mx-auto 2.2e-14.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 2.4, max 7.0 over 110 runs; load average at report time: 3.3 2.9 2.7), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 22 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

