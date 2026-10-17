# Engine-only TPC-H scoreboard - 2026-10-17

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 3,148 WRONG | 154 | **87** | 612 | 110 | - |
| Q2 min-cost supplier | 65 | 35 | **28** | 448 | 32 | - |
| Q3 3-join+top10 | 356 | **106** | 132 | 1,109 | 159 | - |
| Q4 EXISTS semi-join | 1,025 | **96** | 113 | 1,149 | 118 | - |
| Q5 4-join+groupby | 484 | 71 | **65** | 601 | 600 | - |
| Q6 filter+global sum | 46 | **32** | 34 | 408 | 37 | - |
| Q7 volume shipping | 1,070 †3e-6 | 87 | **83** | 891 | 105 | - |
| Q8 market share | 132 | **62** | 67 | 709 | 595 | - |
| Q9 6-join profit | 853 †7e-6 | 438 | **226** | 1,239 | 1,242 | - |
| Q10 returned items top20 | 348 | **113** | 129 | 1,116 | 153 | - |
| Q11 important stock | 32 | **16** | 40 | 360 | 62 | - |
| Q12 shipping modes | 166 | **82** | 123 | 909 | 148 | - |
| Q13 customer distribution | 671 | 241 | **97** | 509 | 122 | PyArrow/NumPy |
| Q14 promo revenue | 88 | 63 | **30** | 567 | 30 | - |
| Q15 top supplier | 72 | **51** | 54 | 671 | 77 | - |
| Q16 part/supplier | 96 | **96** | 101 | 633 | 134 | - |
| Q17 small-qty revenue | 66 | **29** | 69 | 439 | 73 | - |
| Q18 large volume top100 | 1,520 | **316** | 386 | 1,641 | 1,625 | - |
| Q19 discounted revenue | 442 | 85 | **70** | 1,191 | 76 | - |
| Q20 potential promotion | 119 | **49** | 66 | 758 | 92 | - |
| Q21 suppliers waiting | 3,925 | 2,804 | **799** | 6,176 | 6,092 | - |
| Q22 global sales opportunity | 191 | **41** | 92 | 380 | 116 | - |
| **sum over the 21 queries valid for every engine** | **11,767** | **4,914** | **2,802** | **21,904** | **11,689** |  |

Fastest verified engine per query: Polars 0, DuckDB 13, mx-cpu 9, mx-gpu 0, mx-auto 0.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 18/22** (21 comparable); **mx-cpu beats DuckDB on 9/22** (22 comparable)
- **mx-gpu beats Polars on 2/22** (21 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 12/22** (21 comparable); **mx-auto beats DuckDB on 5/22** (22 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q8 market share | mx-auto | 595 | 132 | 4.5x |
| Q11 important stock | mx-auto | 62 | 32 | 2.0x |
| Q21 suppliers waiting | mx-auto | 6,092 | 3,925 | 1.6x |
| Q9 6-join profit | mx-auto | 1,242 | 853 | 1.5x |
| Q16 part/supplier | mx-auto | 134 | 96 | 1.4x |
| Q5 4-join+groupby | mx-auto | 600 | 484 | 1.2x |
| Q17 small-qty revenue | mx-auto | 73 | 66 | 1.1x |
| Q15 top supplier | mx-auto | 77 | 72 | 1.1x |
| Q18 large volume top100 | mx-auto | 1,625 | 1,520 | 1.1x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-cpu | 40 | 32 | 1.2x |
| Q16 part/supplier | mx-cpu | 101 | 96 | 1.0x |
| Q17 small-qty revenue | mx-cpu | 69 | 66 | 1.0x |

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 3,225 WRONG | 267 | 207 | 725 | 225 |
| Q2 | 147 | 162 | 143 | 556 | 137 |
| Q3 | 433 | 221 | 244 | 1,221 | 269 |
| Q4 | 1,103 | 216 | 257 | 1,255 | 264 |
| Q5 | 556 | 183 | 167 | 704 | 702 |
| Q6 | 120 | 142 | 137 | 511 | 139 |
| Q7 | 1,142 †3e-6 | 199 | 187 | 993 | 207 |
| Q8 | 207 | 170 | 170 | 812 | 697 |
| Q9 | 925 †7e-6 | 548 | 334 | 1,343 | 1,344 |
| Q10 | 423 | 225 | 235 | 1,221 | 274 |
| Q11 | 104 | 129 | 145 | 463 | 167 |
| Q12 | 239 | 192 | 228 | 1,011 | 250 |
| Q13 | 746 | 354 | 201 | 651 | 223 |
| Q14 | 163 | 173 | 139 | 673 | 135 |
| Q15 | 145 | 171 | 162 | 774 | 180 |
| Q16 | 169 | 207 | 213 | 739 | 241 |
| Q17 | 140 | 140 | 174 | 553 | 199 |
| Q18 | 1,596 | 428 | 491 | 1,743 | 1,729 |
| Q19 | 516 | 194 | 173 | 1,295 | 181 |
| Q20 | 192 | 159 | 171 | 860 | 198 |
| Q21 | 4,003 | 2,921 | 934 | 6,280 | 6,208 |
| Q22 | 264 | 152 | 195 | 483 | 222 |
| **sum (21 queries)** | **13,332** | **7,286** | **5,101** | **24,137** | **13,966** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 626 | 746 | 642 | 612 | 629 |
| Q2 | 93 | 259 | 129 | 140 | 123 |
| Q3 | 371 | 645 | 753 | 783 | 766 |
| Q4 | 269 | 426 | 650 | 605 | 600 |
| Q5 | 358 | 505 | 671 | 696 | 672 |
| Q6 | 200 | 314 | 273 | 262 | 263 |
| Q7 | 386 | 535 | 713 | 717 | 718 |
| Q8 | 457 | 717 | 1,023 | 1,027 | 1,024 |
| Q9 | 520 | 800 | 1,174 | 1,174 | 1,176 |
| Q10 | 421 | 483 | 678 | 667 | 679 |
| Q11 | 57 | 94 | 81 | 80 | 79 |
| Q12 | 435 | 484 | 694 | 691 | 700 |
| Q13 | 76 | 101 | 209 | 213 | 204 |
| Q14 | 281 | 448 | 807 | 775 | 788 |
| Q15 | 249 | 384 | 382 | 378 | 381 |
| Q16 | 68 | 174 | 104 | 100 | 103 |
| Q17 | 241 | 417 | 750 | 763 | 752 |
| Q18 | 206 | 252 | 598 | 605 | 597 |
| Q19 | 587 | 800 | 882 | 891 | 914 |
| Q20 | 358 | 539 | 920 | 910 | 920 |
| Q21 | 333 | 432 | 658 | 647 | 684 |
| Q22 | 77 | 144 | 237 | 237 | 241 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| Q1 | 2,894 WRONG | 154 | 82 | 319 | 86 |
| Q2 | 54 | 39 | 20 | 44 | 24 |
| Q3 | 296 | 109 | 113 | 735 | 907 |
| Q4 | 930 | 97 | 80 | 841 | 942 |
| Q5 | 419 | 67 | 51 | 297 | 297 |
| Q6 | 41 | 31 | 28 | 112 | 31 |
| Q7 | 969 | 84 | 69 | 605 | 686 |
| Q8 | 96 | 64 | 54 | 283 | 282 |
| Q9 | 760 | 440 | 218 | 924 | 928 |
| Q10 | 297 | 109 | 99 | 704 | 876 |
| Q11 | 25 | 15 | 25 | 69 | 27 |
| Q12 | 147 | 79 | 113 | 630 | 697 |
| Q13 | 568 | 236 | 86 | 78 | 91 |
| Q14 | 66 | 63 | 23 | 264 | 399 |
| Q15 | 55 | 48 | 46 | 258 | 250 |
| Q16 | 85 | 113 | 89 | 346 | 96 |
| Q17 | 33 | 28 | 65 | 148 | 228 |
| Q18 | 1,440 | 320 | 355 | 1,286 | 1,284 |
| Q19 | 400 | 87 | 65 | 918 | 945 |
| Q20 | 98 | 49 | 61 | 431 | 534 |
| Q21 | 4,071 | 2,831 | 712 | 5,776 | 6,953 |
| Q22 | 179 | 40 | 84 | 94 | 87 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|
| import | 74 | 112 | 105 | 104 | 105 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 1.1e-14, mx-gpu 1.2e-14, mx-auto 1.2e-14.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 7.0, max 11.0 over 110 runs; load average at report time: 7.5 6.0 3.1), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 4 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

