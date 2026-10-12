# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 831 | 234 †3e-3 | 147 | **84** | 356 | 84 | - |
| Q2 min-cost supplier | 76 | **34** | 119 | 62 | 424 | 62 | - |
| Q3 3-join+top10 | 198 | **73** | 139 | 161 | 426 | 190 | - |
| Q4 EXISTS semi-join | 237 | **104** | 126 | 208 | 426 | 212 | - |
| Q5 4-join+groupby | 244 | **93** | 136 | 158 | 398 | 158 | - |
| Q6 filter+global sum | 85 | **38** | 105 | 51 | 273 | 56 | - |
| Q7 volume shipping | 335 | **141** | 146 | 194 | 421 | 194 | - |
| Q8 market share | 186 | **69** | 146 | 197 | 520 | 203 | - |
| Q9 6-join profit | 325 | **135 †2e-6** | 227 | 291 | 477 | 300 | - |
| Q10 returned items top20 | 254 | **54** | 136 | 230 | 423 | 234 | - |
| Q11 important stock | 36 | **28** | 102 | 43 | 366 | 47 | - |
| Q12 shipping modes | 294 | **81** | 133 | 278 | 463 | 288 | - |
| Q13 customer distribution | 125 | **79** | 181 | 166 | 369 | 185 | PyArrow/NumPy |
| Q14 promo revenue | 128 | **35** | 123 | 138 | 389 | 142 | - |
| Q15 top supplier | 105 | **31** | 109 | 78 | 455 | 94 | - |
| Q16 part/supplier | 76 | **32** | 139 | 89 | 330 | 92 | - |
| Q17 small-qty revenue | 116 | **47** | 112 | 116 | 364 | 119 | - |
| Q18 large volume top100 | 309 | **171** | 183 | 401 | 500 | 411 | - |
| Q19 discounted revenue | 830 | **56** | 129 | 272 | 473 | 275 | - |
| Q20 potential promotion | 197 | **48** | 148 | 186 | 403 | 195 | - |
| Q21 suppliers waiting | 1,355 | **396** | 423 | 899 | 983 | 882 | - |
| Q22 global sales opportunity | 53 | **34** | 126 | 57 | 272 | 57 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **6,394** | **2,014** | **3,336** | **4,357** | **9,512** | **4,479** |  |

Fastest verified engine per query: pandas 0, Polars 21, DuckDB 0, mx-cpu 1, mx-gpu 0, mx-auto 0.

### Wins at SF1 (headline, both results verified)

- **mx-cpu beats Polars on 1/22** (22 comparable); **mx-cpu beats pandas on 14/22** (22 comparable); **mx-cpu beats DuckDB on 8/22** (22 comparable)
- **mx-gpu beats Polars on 0/22** (22 comparable); **mx-gpu beats pandas on 3/22** (22 comparable); **mx-gpu beats DuckDB on 0/22** (22 comparable)
- **mx-auto beats Polars on 1/22** (22 comparable); **mx-auto beats pandas on 14/22** (22 comparable); **mx-auto beats DuckDB on 7/22** (22 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q19 discounted revenue | mx-auto | 275 | 56 | 4.9x |
| Q10 returned items top20 | mx-auto | 234 | 54 | 4.3x |
| Q20 potential promotion | mx-auto | 195 | 48 | 4.1x |
| Q14 promo revenue | mx-auto | 142 | 35 | 4.0x |
| Q12 shipping modes | mx-auto | 288 | 81 | 3.6x |
| Q15 top supplier | mx-auto | 94 | 31 | 3.0x |
| Q8 market share | mx-auto | 203 | 69 | 2.9x |
| Q16 part/supplier | mx-auto | 92 | 32 | 2.8x |
| Q3 3-join+top10 | mx-auto | 190 | 73 | 2.6x |
| Q17 small-qty revenue | mx-auto | 119 | 47 | 2.6x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q19 discounted revenue | mx-cpu | 272 | 56 | 4.8x |
| Q10 returned items top20 | mx-cpu | 230 | 54 | 4.3x |
| Q14 promo revenue | mx-cpu | 138 | 35 | 3.9x |
| Q20 potential promotion | mx-cpu | 186 | 48 | 3.9x |
| Q12 shipping modes | mx-cpu | 278 | 81 | 3.4x |
| Q8 market share | mx-cpu | 197 | 69 | 2.9x |
| Q16 part/supplier | mx-cpu | 89 | 32 | 2.7x |
| Q17 small-qty revenue | mx-cpu | 116 | 47 | 2.5x |
| Q15 top supplier | mx-cpu | 78 | 31 | 2.5x |
| Q18 large volume top100 | mx-cpu | 401 | 171 | 2.3x |

### SF1: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 1,070 | 309 †3e-3 | 188 | 211 | 481 | 213 |
| Q2 | 307 | 105 | 156 | 188 | 566 | 190 |
| Q3 | 443 | 142 | 177 | 285 | 556 | 316 |
| Q4 | 467 | 173 | 164 | 334 | 552 | 338 |
| Q5 | 478 | 164 | 175 | 287 | 527 | 283 |
| Q6 | 336 | 108 | 145 | 176 | 398 | 185 |
| Q7 | 569 | 213 | 184 | 320 | 546 | 320 |
| Q8 | 417 | 144 | 183 | 324 | 652 | 330 |
| Q9 | 560 | 205 †2e-6 | 266 | 414 | 601 | 444 |
| Q10 | 487 | 127 | 175 | 354 | 550 | 360 |
| Q11 | 271 | 101 | 141 | 166 | 493 | 171 |
| Q12 | 525 | 153 | 171 | 405 | 590 | 413 |
| Q13 | 359 | 150 | 222 | 289 | 495 | 323 |
| Q14 | 360 | 106 | 164 | 262 | 516 | 267 |
| Q15 | 335 | 103 | 146 | 204 | 584 | 220 |
| Q16 | 310 | 102 | 178 | 213 | 456 | 218 |
| Q17 | 351 | 117 | 150 | 240 | 490 | 252 |
| Q18 | 546 | 241 | 221 | 527 | 624 | 536 |
| Q19 | 1,061 | 126 | 166 | 397 | 602 | 401 |
| Q20 | 429 | 121 | 189 | 310 | 527 | 319 |
| Q21 | 1,590 | 469 | 461 | 1,024 | 1,108 | 1,011 |
| Q22 | 283 | 107 | 165 | 187 | 397 | 183 |
| **sum (22 queries)** | **11,552** | **3,587** | **4,187** | **7,118** | **12,311** | **7,291** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 765 | 187 | 62 | 58 | 71 | 61 |
| Q2 | 55 | 23 | 41 | 28 | 28 | 30 |
| Q3 | 159 | 57 | 63 | 137 | 123 | 139 |
| Q4 | 208 | 79 | 51 | 153 | 142 | 153 |
| Q5 | 215 | 70 | 60 | 113 | 95 | 113 |
| Q6 | 66 | 23 | 25 | 30 | 37 | 31 |
| Q7 | 287 | 115 | 66 | 148 | 132 | 148 |
| Q8 | 149 | 48 | 71 | 155 | 129 | 157 |
| Q9 | 288 | 107 | 133 | 254 | 206 | 247 |
| Q10 | 205 | 47 | 52 | 192 | 177 | 192 |
| Q11 | 20 | 21 | 29 | 16 | 15 | 18 |
| Q12 | 244 | 64 | 51 | 242 | 233 | 243 |
| Q13 | 104 | 58 | 102 | 38 | 36 | 43 |
| Q14 | 106 | 27 | 48 | 105 | 106 | 108 |
| Q15 | 80 | 23 | 33 | 55 | 69 | 58 |
| Q16 | 54 | 29 | 57 | 55 | 49 | 55 |
| Q17 | 91 | 29 | 37 | 107 | 88 | 106 |
| Q18 | 279 | 143 | 102 | 352 | 211 | 366 |
| Q19 | 791 | 44 | 55 | 242 | 231 | 243 |
| Q20 | 169 | 44 | 60 | 145 | 143 | 150 |
| Q21 | 1,318 | 350 | 327 | 872 | 762 | 847 |
| Q22 | 36 | 30 | 38 | 29 | 27 | 26 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| import | 234 | 71 | 38 | 125 | 126 | 126 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.1e-07, Polars 3.1e-03, mx-cpu 6.5e-15, mx-auto 6.7e-15, mx-gpu 6.8e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.0, max 4.3 over 132 runs; load average at report time: 2.4 2.4 3.1), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `5a2d331`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

