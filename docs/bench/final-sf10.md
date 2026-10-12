# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 7,787 | 2,164 WRONG | **335** | 655 | 976 | 667 | - |
| Q2 min-cost supplier | 424 | **98** | 163 | 238 | 570 | 256 | - |
| Q3 3-join+top10 | 1,622 | 490 | **357** | 1,328 | 1,460 | 1,318 | - |
| Q4 EXISTS semi-join | 5,843 | 1,062 | **260** | 1,770 | 1,863 | 1,779 | - |
| Q5 4-join+groupby | 2,137 | 778 | **377** | 1,202 | 1,191 | 1,244 | - |
| Q6 filter+global sum | 714 | 205 | **193** | 296 | 503 | 284 | - |
| Q7 volume shipping | 3,196 | 1,330 †3e-6 | **408** | 1,353 | 1,263 | 1,272 | - |
| Q8 market share | 1,516 | 524 | **430** | 1,530 | 1,677 | 1,630 | - |
| Q9 6-join profit | 3,831 | 1,312 †7e-6 | **879** | 2,628 | 2,453 | 2,503 | - |
| Q10 returned items top20 | 2,365 | **352** | 368 | 1,198 | 1,359 | 1,235 | - |
| Q11 important stock | 203 | **81** | 136 | 155 | 489 | 166 | - |
| Q12 shipping modes | 2,644 | 505 | **263** | 954 | 1,055 | 960 | - |
| Q13 customer distribution | 1,158 | 713 | **435** | 479 | 630 | 493 | PyArrow/NumPy |
| Q14 promo revenue | 1,095 | **168** | 362 | 412 | 601 | 415 | - |
| Q15 top supplier | 803 | **117** | 285 | 315 | 632 | 319 | - |
| Q16 part/supplier | 468 | **147** | 259 | 538 | 740 | 550 | - |
| Q17 small-qty revenue | 917 | 285 | **244** | 976 | 1,169 | 973 | - |
| Q18 large volume top100 | 5,364 | 1,691 | **550** | 3,009 | 2,563 | 2,640 | - |
| Q19 discounted revenue | 11,883 | 446 | **285** | 654 | 863 | 650 | - |
| Q20 potential promotion | 1,786 | **228** | 327 | 867 | 1,046 | 877 | - |
| Q21 suppliers waiting | 25,742 | 4,130 | **3,250** | 8,284 | 8,015 | 8,276 | - |
| Q22 global sales opportunity | 586 | 250 | **230** | 371 | 589 | 384 | PyArrow/NumPy |
| **sum over the 21 queries valid for every engine** | **74,295** | **14,910** | **10,060** | **28,555** | **30,732** | **28,225** |  |

Fastest verified engine per query: pandas 0, Polars 7, DuckDB 15, mx-cpu 0, mx-gpu 0, mx-auto 0.

### Wins at SF10 (headline, both results verified)

- **mx-cpu beats Polars on 1/22** (21 comparable); **mx-cpu beats pandas on 19/22** (22 comparable); **mx-cpu beats DuckDB on 0/22** (22 comparable)
- **mx-gpu beats Polars on 2/22** (21 comparable); **mx-gpu beats pandas on 16/22** (22 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 2/22** (21 comparable); **mx-auto beats pandas on 19/22** (22 comparable); **mx-auto beats DuckDB on 0/22** (22 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q20 potential promotion | mx-auto | 877 | 228 | 3.9x |
| Q16 part/supplier | mx-auto | 550 | 147 | 3.7x |
| Q10 returned items top20 | mx-auto | 1,235 | 352 | 3.5x |
| Q17 small-qty revenue | mx-auto | 973 | 285 | 3.4x |
| Q8 market share | mx-auto | 1,630 | 524 | 3.1x |
| Q15 top supplier | mx-auto | 319 | 117 | 2.7x |
| Q3 3-join+top10 | mx-auto | 1,318 | 490 | 2.7x |
| Q2 min-cost supplier | mx-auto | 256 | 98 | 2.6x |
| Q14 promo revenue | mx-auto | 415 | 168 | 2.5x |
| Q11 important stock | mx-auto | 166 | 81 | 2.1x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q20 potential promotion | mx-cpu | 867 | 228 | 3.8x |
| Q16 part/supplier | mx-cpu | 538 | 147 | 3.7x |
| Q17 small-qty revenue | mx-auto | 973 | 285 | 3.4x |
| Q10 returned items top20 | mx-cpu | 1,198 | 352 | 3.4x |
| Q8 market share | mx-cpu | 1,530 | 524 | 2.9x |
| Q15 top supplier | mx-cpu | 315 | 117 | 2.7x |
| Q3 3-join+top10 | mx-auto | 1,318 | 490 | 2.7x |
| Q14 promo revenue | mx-cpu | 412 | 168 | 2.4x |
| Q2 min-cost supplier | mx-cpu | 238 | 98 | 2.4x |
| Q21 suppliers waiting | mx-gpu | 8,015 | 4,130 | 1.9x |

### SF10: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 8,022 | 2,241 WRONG | 372 | 781 | 1,103 | 792 |
| Q2 | 657 | 169 | 203 | 364 | 696 | 382 |
| Q3 | 1,852 | 562 | 395 | 1,454 | 1,587 | 1,447 |
| Q4 | 6,073 | 1,133 | 300 | 1,894 | 1,992 | 1,906 |
| Q5 | 2,367 | 849 | 416 | 1,330 | 1,316 | 1,370 |
| Q6 | 947 | 278 | 231 | 424 | 627 | 410 |
| Q7 | 3,431 | 1,410 †3e-6 | 448 | 1,520 | 1,389 | 1,399 |
| Q8 | 1,774 | 594 | 468 | 1,669 | 1,817 | 1,753 |
| Q9 | 4,065 | 1,383 †7e-6 | 917 | 2,752 | 2,579 | 2,627 |
| Q10 | 2,600 | 423 | 407 | 1,325 | 1,485 | 1,363 |
| Q11 | 436 | 151 | 175 | 283 | 615 | 294 |
| Q12 | 2,878 | 577 | 301 | 1,080 | 1,179 | 1,088 |
| Q13 | 1,390 | 785 | 476 | 604 | 758 | 618 |
| Q14 | 1,328 | 241 | 400 | 538 | 727 | 543 |
| Q15 | 1,039 | 189 | 324 | 441 | 775 | 445 |
| Q16 | 700 | 218 | 297 | 663 | 864 | 678 |
| Q17 | 1,146 | 358 | 284 | 1,103 | 1,295 | 1,097 |
| Q18 | 5,596 | 1,762 | 589 | 3,134 | 2,688 | 2,765 |
| Q19 | 12,117 | 516 | 323 | 780 | 990 | 777 |
| Q20 | 2,020 | 299 | 365 | 993 | 1,173 | 1,006 |
| Q21 | 25,970 | 4,200 | 3,287 | 8,410 | 8,139 | 8,402 |
| Q22 | 821 | 319 | 269 | 497 | 714 | 516 |
| **sum (21 queries)** | **79,208** | **16,416** | **10,873** | **31,257** | **33,408** | **30,886** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 8,123 | 1,842 WRONG | 245 | 616 | 761 | 700 |
| Q2 | 381 | 76 | 80 | 213 | 209 | 217 |
| Q3 | 1,589 | 368 | 270 | 1,229 | 1,051 | 1,249 |
| Q4 | 5,772 | 972 | 174 | 1,705 | 1,593 | 1,706 |
| Q5 | 2,151 | 651 | 280 | 1,158 | 934 | 930 |
| Q6 | 650 | 171 | 104 | 235 | 248 | 258 |
| Q7 | 3,207 | 1,069 | 292 | 1,189 | 982 | 1,182 |
| Q8 | 1,451 | 390 | 337 | 1,464 | 1,305 | 1,318 |
| Q9 | 3,810 | 1,103 | 750 | 2,565 | 2,199 | 2,158 |
| Q10 | 2,297 | 317 | 286 | 1,130 | 987 | 1,133 |
| Q11 | 185 | 59 | 48 | 143 | 121 | 153 |
| Q12 | 2,893 | 428 | 170 | 851 | 759 | 884 |
| Q13 | 1,174 | 610 | 340 | 362 | 289 | 358 |
| Q14 | 1,049 | 138 | 277 | 351 | 338 | 360 |
| Q15 | 725 | 98 | 193 | 257 | 260 | 263 |
| Q16 | 407 | 118 | 169 | 481 | 505 | 493 |
| Q17 | 868 | 190 | 156 | 976 | 888 | 960 |
| Q18 | 5,338 | 1,596 | 451 | 2,922 | 2,301 | 2,299 |
| Q19 | 11,270 | 402 | 194 | 613 | 609 | 603 |
| Q20 | 1,751 | 196 | 227 | 795 | 762 | 797 |
| Q21 | 25,880 | 3,879 | 3,063 | 8,246 | 7,746 | 8,250 |
| Q22 | 533 | 213 | 147 | 338 | 336 | 339 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF10: import time, ms (median over queries; not in the headline)

|  | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| import | 234 | 71 | 39 | 126 | 126 | 126 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 5.5e-02, mx-cpu 8.2e-15, mx-auto 9.0e-15, mx-gpu 9.5e-15, pandas 1.1e-07.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.3, max 11.3 over 132 runs; load average at report time: 4.2 5.0 4.2), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `5a2d331`. Last invocation took 7 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

