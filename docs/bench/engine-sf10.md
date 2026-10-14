# Engine-only TPC-H scoreboard - 2026-10-12

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 6,648 | 2,879 WRONG | 223 | **145** | 498 | 145 | - |
| Q2 min-cost supplier | 250 | **54** | 107 | 144 | 474 | 143 | - |
| Q3 3-join+top10 | 834 | 311 | **171** | 868 | 983 | 911 | - |
| Q4 EXISTS semi-join | 5,062 | 962 | **171** | 1,202 | 1,270 | 1,209 | - |
| Q5 4-join+groupby | 1,281 | 482 | **142** | 503 | 496 | 562 | - |
| Q6 filter+global sum | 266 | 41 | 100 | **31** | 331 | 33 | - |
| Q7 volume shipping | 2,839 | 1,105 †3e-6 | **156** | 841 | 899 | 936 | - |
| Q8 market share | 402 | **120** | 148 | 530 | 668 | 602 | - |
| Q9 6-join profit | 2,897 | 905 †7e-6 | **544** | 1,574 | 1,407 | 1,415 | - |
| Q10 returned items top20 | 1,163 | 368 | **192** | 1,284 | 1,429 | 1,307 | - |
| Q11 important stock | 102 | **32** | 87 | 97 | 447 | 97 | - |
| Q12 shipping modes | 1,160 | 173 | **167** | 1,827 | 1,929 | 1,943 | - |
| Q13 customer distribution | 1,072 | 728 | 339 | **271** | 424 | 297 | PyArrow/NumPy |
| Q14 promo revenue | 311 | **71** | 137 | 349 | 566 | 361 | - |
| Q15 top supplier | 255 | **71** | 135 | 265 | 703 | 277 | - |
| Q16 part/supplier | 337 | **125** | 178 | 451 | 707 | 448 | - |
| Q17 small-qty revenue | 178 | **56** | 109 | 226 | 390 | 241 | - |
| Q18 large volume top100 | 5,398 | 1,427 | **390** | 2,006 | 1,656 | 1,724 | - |
| Q19 discounted revenue | 12,199 | 487 | **171** | 2,271 | 2,497 | 2,233 | - |
| Q20 potential promotion | 911 | **137** | 160 | 497 | 693 | 510 | - |
| Q21 suppliers waiting | 28,781 | 4,237 | **2,994** | 7,158 | 6,902 | 7,136 | - |
| Q22 global sales opportunity | 502 | 253 | **116** | 117 | 356 | 125 | PyArrow/NumPy |
| **sum over the 21 queries valid for every engine** | **66,199** | **12,146** | **6,716** | **22,512** | **25,226** | **22,511** |  |

Fastest verified engine per query: pandas 0, Polars 8, DuckDB 11, mx-cpu 3, mx-gpu 0, mx-auto 0.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 4/22** (21 comparable); **mx-cpu beats pandas on 14/22** (22 comparable); **mx-cpu beats DuckDB on 3/22** (22 comparable)
- **mx-gpu beats Polars on 2/22** (21 comparable); **mx-gpu beats pandas on 11/22** (22 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 4/22** (21 comparable); **mx-auto beats pandas on 14/22** (22 comparable); **mx-auto beats DuckDB on 3/22** (22 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q12 shipping modes | mx-auto | 1,943 | 173 | 11.2x |
| Q14 promo revenue | mx-auto | 361 | 71 | 5.1x |
| Q8 market share | mx-auto | 602 | 120 | 5.0x |
| Q19 discounted revenue | mx-auto | 2,233 | 487 | 4.6x |
| Q17 small-qty revenue | mx-auto | 241 | 56 | 4.3x |
| Q15 top supplier | mx-auto | 277 | 71 | 3.9x |
| Q20 potential promotion | mx-auto | 510 | 137 | 3.7x |
| Q16 part/supplier | mx-auto | 448 | 125 | 3.6x |
| Q10 returned items top20 | mx-auto | 1,307 | 368 | 3.6x |
| Q11 important stock | mx-auto | 97 | 32 | 3.0x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q12 shipping modes | mx-cpu | 1,827 | 173 | 10.6x |
| Q14 promo revenue | mx-cpu | 349 | 71 | 4.9x |
| Q19 discounted revenue | mx-auto | 2,233 | 487 | 4.6x |
| Q8 market share | mx-cpu | 530 | 120 | 4.4x |
| Q17 small-qty revenue | mx-cpu | 226 | 56 | 4.1x |
| Q15 top supplier | mx-cpu | 265 | 71 | 3.7x |
| Q20 potential promotion | mx-cpu | 497 | 137 | 3.6x |
| Q16 part/supplier | mx-auto | 448 | 125 | 3.6x |
| Q10 returned items top20 | mx-cpu | 1,284 | 368 | 3.5x |
| Q11 important stock | mx-cpu | 97 | 32 | 3.0x |

### SF10: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 6,916 | 2,956 WRONG | 263 | 304 | 642 | 283 |
| Q2 | 496 | 130 | 145 | 307 | 609 | 270 |
| Q3 | 1,117 | 389 | 213 | 1,028 | 1,123 | 1,041 |
| Q4 | 5,344 | 1,036 | 209 | 1,330 | 1,396 | 1,343 |
| Q5 | 1,515 | 557 | 182 | 628 | 622 | 690 |
| Q6 | 500 | 115 | 140 | 158 | 456 | 160 |
| Q7 | 3,110 | 1,177 †3e-6 | 195 | 1,007 | 1,029 | 1,094 |
| Q8 | 690 | 195 | 192 | 692 | 829 | 744 |
| Q9 | 3,204 | 992 †7e-6 | 587 | 1,737 | 1,537 | 1,572 |
| Q10 | 1,436 | 440 | 230 | 1,443 | 1,592 | 1,442 |
| Q11 | 349 | 118 | 133 | 243 | 604 | 233 |
| Q12 | 1,453 | 247 | 211 | 1,983 | 2,076 | 2,085 |
| Q13 | 1,314 | 804 | 380 | 431 | 587 | 452 |
| Q14 | 597 | 144 | 178 | 527 | 701 | 492 |
| Q15 | 519 | 150 | 175 | 404 | 850 | 423 |
| Q16 | 594 | 214 | 219 | 577 | 843 | 609 |
| Q17 | 443 | 132 | 149 | 365 | 524 | 375 |
| Q18 | 5,675 | 1,501 | 429 | 2,136 | 1,807 | 1,872 |
| Q19 | 12,482 | 562 | 214 | 2,404 | 2,623 | 2,360 |
| Q20 | 1,173 | 212 | 203 | 656 | 856 | 679 |
| Q21 | 29,067 | 4,314 | 3,036 | 7,332 | 7,037 | 7,274 |
| Q22 | 762 | 331 | 158 | 243 | 516 | 269 |
| **sum (21 queries)** | **71,840** | **13,759** | **7,580** | **25,629** | **28,217** | **25,477** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 1,679 | 565 | 672 | 556 | 543 | 520 |
| Q2 | 180 | 93 | 224 | 132 | 116 | 138 |
| Q3 | 852 | 343 | 540 | 699 | 667 | 723 |
| Q4 | 764 | 259 | 337 | 595 | 598 | 601 |
| Q5 | 844 | 357 | 503 | 678 | 682 | 680 |
| Q6 | 428 | 199 | 307 | 262 | 260 | 256 |
| Q7 | 996 | 379 | 543 | 805 | 775 | 723 |
| Q8 | 1,248 | 467 | 813 | 1,144 | 1,102 | 1,103 |
| Q9 | 1,424 | 544 | 877 | 1,260 | 1,216 | 1,220 |
| Q10 | 1,399 | 425 | 474 | 710 | 737 | 737 |
| Q11 | 117 | 62 | 87 | 110 | 95 | 112 |
| Q12 | 1,618 | 440 | 516 | 750 | 735 | 725 |
| Q13 | 203 | 77 | 94 | 218 | 214 | 214 |
| Q14 | 828 | 265 | 493 | 818 | 778 | 819 |
| Q15 | 620 | 270 | 418 | 440 | 401 | 431 |
| Q16 | 168 | 84 | 172 | 98 | 116 | 103 |
| Q17 | 732 | 258 | 449 | 759 | 789 | 776 |
| Q18 | 582 | 198 | 243 | 588 | 638 | 666 |
| Q19 | 2,002 | 590 | 878 | 938 | 928 | 903 |
| Q20 | 982 | 382 | 643 | 966 | 957 | 981 |
| Q21 | 931 | 360 | 443 | 671 | 707 | 666 |
| Q22 | 219 | 80 | 122 | 251 | 246 | 248 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 6,625 | 2,233 WRONG | 147 | 152 | 278 | 167 |
| Q2 | 261 | 52 | 34 | 140 | 128 | 139 |
| Q3 | 846 | 261 | 98 | 887 | 655 | 878 |
| Q4 | 5,213 | 929 | 97 | 1,200 | 1,049 | 1,241 |
| Q5 | 1,322 | 410 | 69 | 474 | 273 | 276 |
| Q6 | 265 | 39 | 30 | 29 | 112 | 30 |
| Q7 | 2,741 | 977 | 84 | 800 | 586 | 817 |
| Q8 | 344 | 95 | 69 | 489 | 310 | 305 |
| Q9 | 2,663 | 908 | 485 | 1,553 | 1,085 | 1,168 |
| Q10 | 1,221 | 318 | 112 | 1,294 | 1,153 | 1,289 |
| Q11 | 105 | 29 | 16 | 77 | 61 | 95 |
| Q12 | 1,188 | 144 | 80 | 1,925 | 1,709 | 2,057 |
| Q13 | 1,122 | 567 | 253 | 169 | 94 | 169 |
| Q14 | 320 | 62 | 59 | 325 | 344 | 329 |
| Q15 | 256 | 65 | 56 | 253 | 353 | 272 |
| Q16 | 304 | 98 | 118 | 441 | 474 | 445 |
| Q17 | 171 | 33 | 31 | 244 | 179 | 262 |
| Q18 | 5,399 | 1,465 | 317 | 2,060 | 1,434 | 1,404 |
| Q19 | 12,745 | 487 | 87 | 2,263 | 2,578 | 2,413 |
| Q20 | 890 | 106 | 51 | 512 | 458 | 502 |
| Q21 | 28,031 | 4,221 | 2,929 | 7,080 | 6,751 | 7,221 |
| Q22 | 404 | 184 | 41 | 117 | 115 | 114 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| import | 269 | 75 | 41 | 157 | 138 | 138 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 9.6e-15, mx-auto 9.2e-15, mx-gpu 9.5e-15, pandas 1.1e-07.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 6.7, max 9.6 over 132 runs; load average at report time: 6.7 6.2 5.1), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `302cc70`. Last invocation took 9 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

