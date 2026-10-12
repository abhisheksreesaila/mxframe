# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 26 | **20 †3e-6** | 90 | 31 | - |
| Q2 min-cost supplier | 26 | **19** | 91 | 34 | - |
| Q3 3-join+top10 | 22 | **21** | 88 | 38 | - |
| Q4 EXISTS semi-join | 40 | **23** | 86 | 37 | - |
| Q5 4-join+groupby | 56 | **18** | 89 | 41 | - |
| Q6 filter+global sum | **17** | 19 | 93 | 39 | - |
| Q7 volume shipping | 31 | **21** | 104 | 162 | - |
| Q8 market share | 28 | **20** | 89 | 40 | - |
| Q9 6-join profit | 28 | **19** | 95 | 40 | - |
| Q10 returned items top20 | 23 | **18** | 87 | 33 | - |
| Q11 important stock | **19** | 20 | 82 | ERR | - |
| Q12 shipping modes | 22 | **19** | 91 | 38 | - |
| Q13 customer distribution | 18 | **16** | 86 | 141 | PyArrow/NumPy |
| Q14 promo revenue | 18 | **15** | 84 | 36 | - |
| Q15 top supplier | **20** | 24 | 87 | 36 | - |
| Q16 part/supplier | 22 | **20** | 94 | 149 | - |
| Q17 small-qty revenue | 20 | **18** | 92 | 161 WRONG | - |
| Q18 large volume top100 | **21** | 22 | 92 | 36 | - |
| Q19 discounted revenue | 23 WRONG | 16 WRONG | **91** | 150 | - |
| Q20 potential promotion | 23 | **21** | 94 | 36 | - |
| Q21 suppliers waiting | **21** | 21 | 110 | 154 | - |
| Q22 global sales opportunity | **23** | 24 | 90 | 34 | PyArrow/NumPy |
| **sum over the 19 queries valid for every engine** | **484** | **382** | **1,739** | **1,156** |  |

Fastest verified engine per query: pandas 6, Polars 15, DuckDB 1, mx-cpu 0.

### Wins at SF0.002 (headline, both results verified)

- **mx-cpu beats Polars on 0/22** (19 comparable); **mx-cpu beats pandas on 2/22** (19 comparable); **mx-cpu beats DuckDB on 15/22** (20 comparable)

### SF0.002: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 269 | 94 †3e-6 | 130 | 158 |
| Q2 | 277 | 90 | 129 | 163 |
| Q3 | 253 | 94 | 126 | 165 |
| Q4 | 528 | 108 | 125 | 172 |
| Q5 | 329 | 99 | 130 | 208 |
| Q6 | 312 | 99 | 135 | 184 |
| Q7 | 339 | 95 | 161 | 287 |
| Q8 | 277 | 92 | 127 | 168 |
| Q9 | 268 | 95 | 135 | 221 |
| Q10 | 257 | 90 | 125 | 160 |
| Q11 | 259 | 96 | 121 | ERR |
| Q12 | 262 | 93 | 131 | 165 |
| Q13 | 273 | 93 | 124 | 266 |
| Q14 | 254 | 88 | 122 | 163 |
| Q15 | 284 | 96 | 125 | 189 |
| Q16 | 295 | 92 | 133 | 274 |
| Q17 | 264 | 90 | 130 | 308 WRONG |
| Q18 | 261 | 95 | 132 | 164 |
| Q19 | 293 WRONG | 91 WRONG | 133 | 282 |
| Q20 | 258 | 97 | 133 | 163 |
| Q21 | 290 | 93 | 148 | 280 |
| Q22 | 279 | 99 | 128 | 164 |
| **sum (19 queries)** | **5,564** | **1,802** | **2,500** | **3,713** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 9.8 | 15 | 12 | 2.5 |
| Q2 | 10 | 15 | 17 | 3.3 |
| Q3 | 7.5 | 13 | 15 | 2.8 |
| Q4 | 6.7 | 4.0 | 15 | 2.4 |
| Q5 | 12 | 6.4 | 17 | 4.1 |
| Q6 | 2.6 | 12 | 9.1 | 3.7 |
| Q7 | 15 | 18 | 15 | 5.0 |
| Q8 | 12 | 15 | 18 | 4.8 |
| Q9 | 12 | 10 | 17 | 5.1 |
| Q10 | 8.7 | 17 | 15 | 3.6 |
| Q11 | 4.1 | 15 | 15 | - |
| Q12 | 6.4 | 5.9 | 14 | 2.7 |
| Q13 | 5.3 | 14 | 13 | 2.4 |
| Q14 | 4.0 | 13 | 12 | 2.4 |
| Q15 | 5.0 | 8.8 | 13 | 2.4 |
| Q16 | 7.5 | 9.8 | 16 | 3.3 |
| Q17 | 5.2 | 10.0 | 10 | 2.7 WRONG |
| Q18 | 6.0 | 16 | 14 | 3.3 |
| Q19 | 7.5 WRONG | 13 WRONG | 13 | 4.3 |
| Q20 | 8.0 | 13 | 14 | 3.8 |
| Q21 | 6.7 | 16 | 20 | 6.0 |
| Q22 | 7.3 | 14 | 14 | 2.1 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 250 | 74 | 39 | 128 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |
| Q11 | mx-cpu | error | - | TypeError: unsupported operand type(s) for *: 'NoneType' and 'float' |
| Q17 | mx-cpu | wrong | inf | max relative error inf / repeat run WRONG |
| Q19 | pandas | wrong | inf | max relative error inf / repeat run WRONG |
| Q19 | polars | wrong | inf | max relative error inf / repeat run WRONG |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 41 | **20 †2e-4** | 92 | 38 | - |
| Q2 min-cost supplier | 29 | **19** | 90 | 40 | - |
| Q3 3-join+top10 | 26 | **21** | 100 | 36 | - |
| Q4 EXISTS semi-join | 22 | **20** | 90 | 37 | - |
| Q5 4-join+groupby | 30 | **21** | 93 | 39 | - |
| Q6 filter+global sum | **19** | 19 | 87 | 33 | - |
| Q7 volume shipping | 33 | **23** | 94 | 39 | - |
| Q8 market share | 30 | **22** | 93 | 41 | - |
| Q9 6-join profit | 32 | **21** | 102 | 44 | - |
| Q10 returned items top20 | 29 | **19** | 97 | 39 | - |
| Q11 important stock | **18** | 19 | 88 | 152 | - |
| Q12 shipping modes | 25 | **22** | 91 | 42 | - |
| Q13 customer distribution | 22 | **20** | 91 | 165 | PyArrow/NumPy |
| Q14 promo revenue | 20 | **19** | 91 | 39 | - |
| Q15 top supplier | 22 | **20** | 92 | 35 | - |
| Q16 part/supplier | 24 | **20** | 97 | 166 | - |
| Q17 small-qty revenue | 22 | **21** | 100 | 38 | - |
| Q18 large volume top100 | 26 | **23** | 93 | 40 | - |
| Q19 discounted revenue | 38 | **17** | 96 | 41 | - |
| Q20 potential promotion | 26 | **23** | 92 | 38 | - |
| Q21 suppliers waiting | 37 | **30** | 122 | 61 | - |
| Q22 global sales opportunity | 23 | **19** | 90 | 38 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **593** | **459** | **2,082** | **1,240** |  |

Fastest verified engine per query: pandas 2, Polars 20, DuckDB 0, mx-cpu 0.

### Wins at SF0.017 (headline, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 1/22** (22 comparable); **mx-cpu beats DuckDB on 19/22** (22 comparable)

### SF0.017: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 290 | 92 †2e-4 | 132 | 167 |
| Q2 | 272 | 93 | 128 | 172 |
| Q3 | 262 | 93 | 143 | 168 |
| Q4 | 258 | 95 | 128 | 168 |
| Q5 | 276 | 96 | 133 | 169 |
| Q6 | 278 | 96 | 126 | 167 |
| Q7 | 275 | 97 | 135 | 171 |
| Q8 | 278 | 95 | 133 | 169 |
| Q9 | 273 | 95 | 148 | 186 |
| Q10 | 276 | 91 | 138 | 166 |
| Q11 | 257 | 92 | 127 | 283 |
| Q12 | 269 | 96 | 130 | 171 |
| Q13 | 282 | 95 | 131 | 298 |
| Q14 | 262 | 92 | 130 | 206 |
| Q15 | 264 | 95 | 132 | 166 |
| Q16 | 284 | 93 | 136 | 346 |
| Q17 | 364 | 96 | 167 | 167 |
| Q18 | 266 | 96 | 132 | 168 |
| Q19 | 276 | 89 | 136 | 176 |
| Q20 | 275 | 99 | 132 | 188 |
| Q21 | 299 | 119 | 165 | 230 |
| Q22 | 305 | 91 | 131 | 172 |
| **sum (22 queries)** | **6,141** | **2,094** | **2,994** | **4,275** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 22 | 17 | 17 | 4.5 |
| Q2 | 12 | 4.4 | 18 | 4.7 |
| Q3 | 11 | 4.5 | 18 | 4.9 |
| Q4 | 6.9 | 12 | 17 | 6.2 |
| Q5 | 14 | 17 | 19 | 6.0 |
| Q6 | 3.1 | 4.7 | 11 | 1.9 |
| Q7 | 18 | 17 | 19 | 6.7 |
| Q8 | 12 | 14 | 21 | 8.7 |
| Q9 | 16 | 16 | 24 | 8.7 |
| Q10 | 12 | 15 | 18 | 6.6 |
| Q11 | 3.9 | 13 | 15 | 2.8 |
| Q12 | 12 | 13 | 16 | 7.2 |
| Q13 | 6.6 | 16 | 16 | 3.1 |
| Q14 | 4.9 | 9.9 | 15 | 4.3 |
| Q15 | 5.9 | 9.3 | 13 | 3.9 |
| Q16 | 8.6 | 14 | 15 | 4.4 |
| Q17 | 7.0 | 12 | 12 | 3.9 |
| Q18 | 10 | 17 | 20 | 9.3 |
| Q19 | 21 | 13 | 16 | 7.4 |
| Q20 | 10 | 4.3 | 17 | 5.6 |
| Q21 | 21 | 12 | 26 | 26 |
| Q22 | 8.0 | 11 | 15 | 2.9 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 245 | 73 | 40 | 132 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.3e-15.


## Scale factor 0.17: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 171 | 60 †2e-3 | 128 | **45** | - |
| Q2 min-cost supplier | 37 | **20** | 99 | 45 | - |
| Q3 3-join+top10 | 71 | **31** | 115 | 68 | - |
| Q4 EXISTS semi-join | 67 | **35** | 113 | 63 | - |
| Q5 4-join+groupby | 73 | **34** | 110 | 63 | - |
| Q6 filter+global sum | 31 | **19** | 93 | 35 | - |
| Q7 volume shipping | 94 | **42** | 116 | 69 | - |
| Q8 market share | 57 | **29** | 114 | 61 | - |
| Q9 6-join profit | 84 | **40** | 131 | 76 | - |
| Q10 returned items top20 | 75 | **28** | 163 | 74 | - |
| Q11 important stock | 22 | **20** | 91 | 37 | - |
| Q12 shipping modes | 75 | **42** | 107 | 81 | - |
| Q13 customer distribution | 41 | **34** | 101 | 151 | PyArrow/NumPy |
| Q14 promo revenue | 34 | **23** | 103 | 43 | - |
| Q15 top supplier | 40 | **20** | 100 | 43 | - |
| Q16 part/supplier | 34 | **21** | 103 | 49 | - |
| Q17 small-qty revenue | 34 | **29** | 99 | 44 | - |
| Q18 large volume top100 | 67 | **45** | 130 | 100 | - |
| Q19 discounted revenue | 182 | **27** | 112 | 58 | - |
| Q20 potential promotion | 55 | **28** | 104 | 59 | - |
| Q21 suppliers waiting | 155 | **85** | 193 | 239 | - |
| Q22 global sales opportunity | 30 | **23** | 97 | 37 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **1,528** | **735** | **2,524** | **1,538** |  |

Fastest verified engine per query: pandas 0, Polars 21, DuckDB 0, mx-cpu 1.

### Wins at SF0.17 (headline, both results verified)

- **mx-cpu beats Polars on 1/22** (22 comparable); **mx-cpu beats pandas on 8/22** (22 comparable); **mx-cpu beats DuckDB on 20/22** (22 comparable)

### SF0.17: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 414 | 136 †2e-3 | 166 | 170 |
| Q2 | 280 | 105 | 140 | 173 |
| Q3 | 319 | 119 | 156 | 228 |
| Q4 | 317 | 107 | 154 | 189 |
| Q5 | 307 | 105 | 149 | 198 |
| Q6 | 275 | 94 | 132 | 161 |
| Q7 | 336 | 117 | 157 | 195 |
| Q8 | 297 | 102 | 157 | 190 |
| Q9 | 320 | 111 | 172 | 202 |
| Q10 | 341 | 98 | 205 | 202 |
| Q11 | 256 | 91 | 129 | 175 |
| Q12 | 318 | 113 | 145 | 219 |
| Q13 | 280 | 108 | 139 | 293 |
| Q14 | 270 | 100 | 142 | 171 |
| Q15 | 281 | 93 | 138 | 177 |
| Q16 | 270 | 97 | 143 | 217 |
| Q17 | 285 | 104 | 137 | 170 |
| Q18 | 302 | 117 | 171 | 231 |
| Q19 | 424 | 99 | 151 | 184 |
| Q20 | 293 | 102 | 142 | 185 |
| Q21 | 418 | 158 | 233 | 374 |
| Q22 | 308 | 97 | 136 | 163 |
| **sum (22 queries)** | **6,913** | **2,374** | **3,393** | **4,467** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.17: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 153 | 32 | 60 | 13 |
| Q2 | 20 | 14 | 25 | 11 |
| Q3 | 46 | 15 | 37 | 26 |
| Q4 | 41 | 31 | 33 | 33 |
| Q5 | 52 | 28 | 37 | 25 |
| Q6 | 15 | 6.3 | 20 | 7.3 |
| Q7 | 75 | 28 | 38 | 30 |
| Q8 | 38 | 20 | 38 | 26 |
| Q9 | 58 | 28 | 54 | 43 |
| Q10 | 51 | 23 | 33 | 37 |
| Q11 | 7.8 | 16 | 17 | 5.6 |
| Q12 | 54 | 31 | 33 | 46 |
| Q13 | 27 | 19 | 29 | 8.2 |
| Q14 | 17 | 15 | 24 | 10 |
| Q15 | 16 | 15 | 20 | 13 |
| Q16 | 16 | 15 | 23 | 13 |
| Q17 | 18 | 17 | 22 | 16 |
| Q18 | 47 | 36 | 55 | 66 |
| Q19 | 150 | 20 | 39 | 41 |
| Q20 | 34 | 17 | 30 | 23 |
| Q21 | 140 | 78 | 116 | 168 |
| Q22 | 13 | 16 | 20 | 7.3 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.17: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 242 | 73 | 39 | 128 |

### SF0.17: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 1.57e-03 | max relative error 1.57e-03 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.3e-07, Polars 1.6e-03, mx-cpu 1.4e-14.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.1, max 7.2 over 264 runs; load average at report time: 5.1 2.7 2.2), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `b4e986e`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

