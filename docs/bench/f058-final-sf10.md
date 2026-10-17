# Engine-only TPC-H scoreboard - 2026-10-17

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,606 WRONG | 141 | **64** | 77 | - |
| Q2 min-cost supplier | 54 | 30 | **20** | 23 | - |
| Q3 3-join+top10 | 291 | **84** | 95 | 107 | - |
| Q4 EXISTS semi-join | 962 | 100 | **96** | 97 | - |
| Q5 4-join+groupby | 476 | 74 | **64** | 67 | - |
| Q6 filter+global sum | 40 | **31** | 34 | 38 | - |
| Q7 volume shipping | 1,135 †3e-6 | 94 | **79** | 82 | - |
| Q8 market share | 130 | **63** | 64 | 66 | - |
| Q9 6-join profit | 829 †7e-6 | 434 | **224** | 228 | - |
| Q10 returned items top20 | 336 | **111** | 122 | 127 | - |
| Q11 important stock | 29 | **16** | 34 | 36 | - |
| Q12 shipping modes | 145 | **77** | 121 | 126 | - |
| Q13 customer distribution | 649 | 238 | **89** | 92 | PyArrow/NumPy |
| Q14 promo revenue | 86 | 63 | **25** | 28 | - |
| Q15 top supplier | 68 | 52 | **49** | 54 | - |
| Q16 part/supplier | 98 | **93** | 100 | 100 | - |
| Q17 small-qty revenue | 60 | **30** | 66 | 70 | - |
| Q18 large volume top100 | 1,460 | **316** | 374 | 381 | - |
| Q19 discounted revenue | 508 | 83 | **66** | 72 | - |
| Q20 potential promotion | 116 | **49** | 61 | 67 | - |
| Q21 suppliers waiting | 3,924 | 2,749 | 800 | **796** | - |
| Q22 global sales opportunity | 184 | **44** | 90 | 94 | - |
| **sum over the 21 queries valid for every engine** | **11,580** | **4,830** | **2,674** | **2,751** |  |

Fastest verified engine per query: Polars 0, DuckDB 11, mx-cpu 10, mx-auto 1.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 18/22** (21 comparable); **mx-cpu beats DuckDB on 11/22** (22 comparable)
- **mx-auto beats Polars on 18/22** (21 comparable); **mx-auto beats DuckDB on 10/22** (22 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-auto | 36 | 29 | 1.2x |
| Q17 small-qty revenue | mx-auto | 70 | 60 | 1.2x |
| Q16 part/supplier | mx-auto | 100 | 98 | 1.0x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-cpu | 34 | 29 | 1.2x |
| Q17 small-qty revenue | mx-cpu | 66 | 60 | 1.1x |
| Q16 part/supplier | mx-auto | 100 | 98 | 1.0x |

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 2,681 WRONG | 251 | 184 | 182 |
| Q2 | 129 | 138 | 129 | 129 |
| Q3 | 367 | 193 | 197 | 227 |
| Q4 | 1,039 | 209 | 201 | 200 |
| Q5 | 550 | 185 | 170 | 172 |
| Q6 | 113 | 144 | 177 | 183 |
| Q7 | 1,207 †3e-6 | 214 | 183 | 183 |
| Q8 | 203 | 173 | 167 | 171 |
| Q9 | 901 †7e-6 | 549 | 327 | 333 |
| Q10 | 408 | 222 | 225 | 234 |
| Q11 | 103 | 127 | 140 | 146 |
| Q12 | 221 | 188 | 228 | 232 |
| Q13 | 722 | 351 | 194 | 194 |
| Q14 | 162 | 173 | 128 | 131 |
| Q15 | 142 | 164 | 153 | 158 |
| Q16 | 174 | 204 | 205 | 203 |
| Q17 | 133 | 142 | 175 | 176 |
| Q18 | 1,535 | 430 | 479 | 484 |
| Q19 | 583 | 194 | 171 | 174 |
| Q20 | 189 | 158 | 165 | 174 |
| Q21 | 3,998 | 2,859 | 906 | 901 |
| Q22 | 258 | 158 | 193 | 214 |
| **sum (21 queries)** | **13,139** | **7,174** | **4,911** | **5,020** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 535 | 621 | 499 | 490 |
| Q2 | 78 | 239 | 110 | 118 |
| Q3 | 315 | 494 | 655 | 666 |
| Q4 | 263 | 356 | 603 | 598 |
| Q5 | 352 | 537 | 678 | 676 |
| Q6 | 200 | 322 | 295 | 283 |
| Q7 | 387 | 623 | 726 | 717 |
| Q8 | 458 | 736 | 1,036 | 1,034 |
| Q9 | 516 | 805 | 1,183 | 1,175 |
| Q10 | 422 | 485 | 678 | 684 |
| Q11 | 56 | 96 | 82 | 82 |
| Q12 | 429 | 486 | 696 | 703 |
| Q13 | 76 | 102 | 207 | 208 |
| Q14 | 264 | 444 | 768 | 766 |
| Q15 | 243 | 386 | 378 | 378 |
| Q16 | 69 | 175 | 100 | 103 |
| Q17 | 236 | 402 | 741 | 741 |
| Q18 | 202 | 257 | 598 | 601 |
| Q19 | 564 | 780 | 866 | 873 |
| Q20 | 350 | 556 | 916 | 918 |
| Q21 | 330 | 423 | 656 | 644 |
| Q22 | 76 | 142 | 242 | 246 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 2,049 WRONG | 141 | 64 | 66 |
| Q2 | 44 | 29 | 17 | 18 |
| Q3 | 245 | 82 | 87 | 90 |
| Q4 | 894 | 99 | 81 | 81 |
| Q5 | 423 | 71 | 52 | 53 |
| Q6 | 40 | 31 | 32 | 33 |
| Q7 | 918 | 83 | 69 | 70 |
| Q8 | 94 | 61 | 55 | 55 |
| Q9 | 753 | 433 | 219 | 220 |
| Q10 | 297 | 113 | 98 | 106 |
| Q11 | 24 | 16 | 25 | 24 |
| Q12 | 156 | 79 | 113 | 114 |
| Q13 | 574 | 227 | 82 | 95 |
| Q14 | 65 | 60 | 22 | 24 |
| Q15 | 56 | 49 | 45 | 49 |
| Q16 | 87 | 109 | 87 | 87 |
| Q17 | 32 | 28 | 64 | 66 |
| Q18 | 1,508 | 314 | 349 | 359 |
| Q19 | 403 | 83 | 65 | 68 |
| Q20 | 100 | 47 | 59 | - |
| Q21 | 3,620 | 2,665 | 701 | 701 |
| Q22 | 169 | 41 | 85 | 87 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| import | 74 | 111 | 105 | 105 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 1.1e-14, mx-auto 8.8e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 8.3, max 13.0 over 88 runs; load average at report time: 11.2 7.4 8.8), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `8e001f3`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

