# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 25 | **20 †3e-6** | 83 | 33 | - |
| Q2 min-cost supplier | 26 | **17** | 90 | 35 | - |
| Q3 3-join+top10 | 23 | **21** | 91 | 36 | - |
| Q4 EXISTS semi-join | 27 | **20** | 93 | 37 | - |
| Q5 4-join+groupby | 24 | **21** | 109 | 37 | - |
| Q6 filter+global sum | **16** | 18 | 88 | 34 | - |
| Q7 volume shipping | 29 | **20** | 87 | 38 | - |
| Q8 market share | 26 | **18** | 91 | 37 | - |
| Q9 6-join profit | 26 | **20** | 91 | 37 | - |
| Q10 returned items top20 | 24 | **17** | 88 | 35 | - |
| Q11 important stock | 17 | **14** | 85 | 37 | - |
| Q12 shipping modes | 21 | **16** | 87 | 35 | - |
| Q13 customer distribution | 31 | **19** | 86 | 34 | PyArrow/NumPy |
| Q14 promo revenue | **18** | 19 | 84 | 35 | - |
| Q15 top supplier | 20 | **19** | 89 | 36 | - |
| Q16 part/supplier | 22 | **19** | 93 | 33 | - |
| Q17 small-qty revenue | 20 | **19** | 83 | 39 | - |
| Q18 large volume top100 | **20** | 21 | 93 | 34 | - |
| Q19 discounted revenue | 23 | **18** | 85 | 39 | - |
| Q20 potential promotion | 22 | **21** | 88 | 39 | - |
| Q21 suppliers waiting | **21** | 23 | 104 | 40 | - |
| Q22 global sales opportunity | 21 | **19** | 89 | 30 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **501** | **420** | **1,976** | **789** |  |

Fastest verified engine per query: pandas 4, Polars 18, DuckDB 0, mx-cpu 0.

### Wins at SF0.002 (headline, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 0/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.002: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 261 | 92 †3e-6 | 121 | 163 |
| Q2 | 276 | 90 | 132 | 166 |
| Q3 | 291 | 94 | 132 | 192 |
| Q4 | 298 | 96 | 133 | 193 |
| Q5 | 274 | 94 | 149 | 199 |
| Q6 | 250 | 88 | 130 | 166 |
| Q7 | 267 | 92 | 126 | 164 |
| Q8 | 266 | 89 | 131 | 162 |
| Q9 | 260 | 93 | 130 | 165 |
| Q10 | 261 | 91 | 128 | 161 |
| Q11 | 252 | 86 | 125 | 165 |
| Q12 | 256 | 89 | 125 | 179 |
| Q13 | 293 | 94 | 127 | 160 |
| Q14 | 252 | 93 | 124 | 164 |
| Q15 | 259 | 91 | 129 | 178 |
| Q16 | 259 | 92 | 132 | 158 |
| Q17 | 253 | 94 | 122 | 164 |
| Q18 | 277 | 93 | 133 | 159 |
| Q19 | 258 | 91 | 123 | 164 |
| Q20 | 257 | 95 | 125 | 166 |
| Q21 | 283 | 95 | 145 | 203 |
| Q22 | 256 | 95 | 128 | 157 |
| **sum (22 queries)** | **5,857** | **2,026** | **2,848** | **3,751** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 9.4 | 12 | 15 | 2.5 |
| Q2 | 10 | 8.2 | 17 | 3.8 |
| Q3 | 7.4 | 13 | 13 | 3.5 |
| Q4 | 9.1 | 4.9 | 13 | 2.4 |
| Q5 | 9.7 | 3.9 | 14 | 3.9 |
| Q6 | 2.0 | 9.1 | 10.0 | 1.2 |
| Q7 | 12 | 14 | 17 | 4.8 |
| Q8 | 11 | 8.3 | 18 | 4.6 |
| Q9 | 13 | 14 | 19 | 4.9 |
| Q10 | 8.1 | 10.0 | 16 | 3.4 |
| Q11 | 3.9 | 16 | 13 | 1.8 |
| Q12 | 6.4 | 15 | 12 | 2.6 |
| Q13 | 4.3 | 15 | 13 | 2.0 |
| Q14 | 4.2 | 9.7 | 13 | 2.3 |
| Q15 | 4.6 | 7.0 | 13 | 2.6 |
| Q16 | 7.8 | 13 | 16 | 3.1 |
| Q17 | 5.1 | 9.3 | 13 | 2.3 |
| Q18 | 5.6 | 17 | 15 | 3.1 |
| Q19 | 7.6 | 9.2 | 14 | 4.1 |
| Q20 | 8.2 | 14 | 15 | 4.3 |
| Q21 | 6.1 | 16 | 17 | 6.2 |
| Q22 | 6.4 | 5.2 | 15 | 2.0 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 237 | 73 | 40 | 128 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 39 | **23 †2e-4** | 89 | 37 | - |
| Q2 min-cost supplier | 27 | **18** | 90 | 35 | - |
| Q3 3-join+top10 | **26** | 30 | 91 | 42 | - |
| Q4 EXISTS semi-join | 23 | **19** | 91 | 36 | - |
| Q5 4-join+groupby | 29 | **21** | 92 | 36 | - |
| Q6 filter+global sum | **17** | 19 | 83 | 32 | - |
| Q7 volume shipping | 33 | **22** | 91 | 40 | - |
| Q8 market share | 29 | **20** | 92 | 37 | - |
| Q9 6-join profit | 33 | **22** | 100 | 43 | - |
| Q10 returned items top20 | 28 | **21** | 89 | 38 | - |
| Q11 important stock | **18** | 21 | 85 | 34 | - |
| Q12 shipping modes | 108 | 74 | 93 | **48** | - |
| Q13 customer distribution | **25** | 49 | 170 | 33 | PyArrow/NumPy |
| Q14 promo revenue | **21** | 35 | 148 | 38 | - |
| Q15 top supplier | **22** | 43 | 87 | 40 | - |
| Q16 part/supplier | **33** | 33 | 138 | 40 | - |
| Q17 small-qty revenue | 24 | **21** | 125 | 36 | - |
| Q18 large volume top100 | 29 | **24** | 98 | 44 | - |
| Q19 discounted revenue | 42 | **19** | 93 | 43 | - |
| Q20 potential promotion | **33** | 60 | 93 | 43 | - |
| Q21 suppliers waiting | 35 | **30** | 119 | 67 | - |
| Q22 global sales opportunity | **32** | 56 | 145 | 34 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **704** | **679** | **2,306** | **876** |  |

Fastest verified engine per query: pandas 9, Polars 12, DuckDB 0, mx-cpu 1.

### Wins at SF0.017 (headline, both results verified)

- **mx-cpu beats Polars on 5/22** (22 comparable); **mx-cpu beats pandas on 2/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 270 | 96 †2e-4 | 128 | 162 |
| Q2 | 265 | 91 | 131 | 162 |
| Q3 | 260 | 105 | 129 | 182 |
| Q4 | 286 | 93 | 131 | 167 |
| Q5 | 265 | 94 | 134 | 162 |
| Q6 | 253 | 92 | 122 | 157 |
| Q7 | 270 | 94 | 130 | 169 |
| Q8 | 261 | 92 | 131 | 163 |
| Q9 | 268 | 95 | 139 | 170 |
| Q10 | 261 | 93 | 126 | 165 |
| Q11 | 255 | 92 | 124 | 170 |
| Q12 | 551 | 173 | 140 | 290 |
| Q13 | 411 | 128 | 222 | 169 |
| Q14 | 285 | 121 | 190 | 193 |
| Q15 | 271 | 118 | 137 | 209 |
| Q16 | 313 | 117 | 182 | 171 |
| Q17 | 286 | 107 | 168 | 180 |
| Q18 | 293 | 112 | 138 | 193 |
| Q19 | 309 | 103 | 142 | 203 |
| Q20 | 320 | 135 | 143 | 187 |
| Q21 | 309 | 120 | 166 | 206 |
| Q22 | 318 | 130 | 185 | 187 |
| **sum (22 queries)** | **6,583** | **2,403** | **3,238** | **4,019** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 23 | 17 | 18 | 3.8 |
| Q2 | 11 | 8.0 | 17 | 4.0 |
| Q3 | 10 | 4.4 | 18 | 5.6 |
| Q4 | 6.5 | 5.1 | 16 | 4.8 |
| Q5 | 13 | 18 | 19 | 5.2 |
| Q6 | 2.7 | 7.7 | 13 | 1.9 |
| Q7 | 18 | 17 | 18 | 6.3 |
| Q8 | 12 | 13 | 21 | 6.4 |
| Q9 | 17 | 15 | 22 | 8.1 |
| Q10 | 12 | 12 | 18 | 6.6 |
| Q11 | 3.9 | 2.8 | 13 | 3.3 |
| Q12 | 13 | 16 | 14 | 8.4 |
| Q13 | 8.7 | 4.6 | 27 | 2.8 |
| Q14 | 5.3 | 13 | 12 | 3.3 |
| Q15 | 5.8 | 9.3 | 13 | 4.6 |
| Q16 | 8.8 | 12 | 15 | 4.2 |
| Q17 | 6.5 | 14 | 29 | 4.3 |
| Q18 | 12 | 13 | 28 | 9.1 |
| Q19 | 22 | 14 | 14 | 8.3 |
| Q20 | 12 | 13 | 19 | 5.8 |
| Q21 | 18 | 18 | 25 | 49 |
| Q22 | 8.5 | 13 | 15 | 2.8 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 256 | 75 | 41 | 136 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.3e-15.


## Scale factor 0.17: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 202 | 68 †2e-3 | 135 | **48** | - |
| Q2 min-cost supplier | 42 | **22** | 102 | 45 | - |
| Q3 3-join+top10 | 62 | **29** | 118 | 63 | - |
| Q4 EXISTS semi-join | 64 | **37** | 113 | 65 | - |
| Q5 4-join+groupby | 73 | **33** | 110 | 58 | - |
| Q6 filter+global sum | 28 | **23** | 94 | 38 | - |
| Q7 volume shipping | 99 | **46** | 118 | 66 | - |
| Q8 market share | 59 | **31** | 112 | 66 | - |
| Q9 6-join profit | 89 | **42** | 130 | 84 | - |
| Q10 returned items top20 | 74 | **27** | 111 | 68 | - |
| Q11 important stock | 23 | **20** | 90 | 36 | - |
| Q12 shipping modes | 79 | **39** | 109 | 82 | - |
| Q13 customer distribution | 43 | **31** | 104 | 37 | PyArrow/NumPy |
| Q14 promo revenue | 35 | **20** | 100 | 44 | - |
| Q15 top supplier | 37 | **21** | 97 | 46 | - |
| Q16 part/supplier | 34 | **22** | 109 | 49 | - |
| Q17 small-qty revenue | 36 | **28** | 99 | 44 | - |
| Q18 large volume top100 | 68 | **41** | 130 | 105 | - |
| Q19 discounted revenue | 176 | **27** | 117 | 78 | - |
| Q20 potential promotion | 56 | **26** | 106 | 60 | - |
| Q21 suppliers waiting | 155 | **85** | 196 | 214 | - |
| Q22 global sales opportunity | 28 | **24** | 96 | 37 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **1,563** | **743** | **2,496** | **1,434** |  |

Fastest verified engine per query: pandas 0, Polars 21, DuckDB 0, mx-cpu 1.

### Wins at SF0.17 (headline, both results verified)

- **mx-cpu beats Polars on 1/22** (22 comparable); **mx-cpu beats pandas on 7/22** (22 comparable); **mx-cpu beats DuckDB on 21/22** (22 comparable)

### SF0.17: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 479 | 152 †2e-3 | 177 | 184 |
| Q2 | 289 | 101 | 141 | 179 |
| Q3 | 317 | 101 | 161 | 190 |
| Q4 | 298 | 110 | 152 | 190 |
| Q5 | 327 | 104 | 150 | 184 |
| Q6 | 273 | 102 | 133 | 168 |
| Q7 | 336 | 118 | 158 | 196 |
| Q8 | 300 | 105 | 153 | 201 |
| Q9 | 328 | 115 | 169 | 209 |
| Q10 | 334 | 101 | 150 | 191 |
| Q11 | 258 | 96 | 128 | 163 |
| Q12 | 315 | 111 | 148 | 206 |
| Q13 | 278 | 105 | 143 | 167 |
| Q14 | 274 | 91 | 140 | 170 |
| Q15 | 276 | 93 | 135 | 174 |
| Q16 | 274 | 96 | 148 | 177 |
| Q17 | 274 | 101 | 139 | 171 |
| Q18 | 309 | 113 | 170 | 268 |
| Q19 | 414 | 101 | 158 | 203 |
| Q20 | 293 | 98 | 145 | 186 |
| Q21 | 389 | 156 | 237 | 340 |
| Q22 | 263 | 96 | 134 | 166 |
| **sum (22 queries)** | **6,898** | **2,367** | **3,369** | **4,282** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.17: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 162 | 38 | 58 | 14 |
| Q2 | 20 | 11 | 25 | 10 |
| Q3 | 42 | 21 | 40 | 26 |
| Q4 | 43 | 23 | 33 | 30 |
| Q5 | 55 | 23 | 38 | 24 |
| Q6 | 11 | 14 | 23 | 7.8 |
| Q7 | 88 | 35 | 40 | 30 |
| Q8 | 40 | 24 | 38 | 28 |
| Q9 | 61 | 32 | 57 | 48 |
| Q10 | 54 | 24 | 35 | 40 |
| Q11 | 7.9 | 17 | 17 | 5.9 |
| Q12 | 56 | 27 | 32 | 44 |
| Q13 | 27 | 26 | 29 | 8.1 |
| Q14 | 16 | 15 | 26 | 12 |
| Q15 | 16 | 16 | 22 | 13 |
| Q16 | 17 | 17 | 22 | 13 |
| Q17 | 19 | 17 | 22 | 14 |
| Q18 | 47 | 35 | 54 | 66 |
| Q19 | 148 | 20 | 40 | 44 |
| Q20 | 38 | 20 | 29 | 26 |
| Q21 | 131 | 73 | 129 | 162 |
| Q22 | 13 | 16 | 19 | 6.9 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.17: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 239 | 73 | 39 | 127 |

### SF0.17: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 1.57e-03 | max relative error 1.57e-03 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.3e-07, Polars 1.6e-03, mx-cpu 1.5e-14.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; the comparison counts the two as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 4.0, max 6.0 over 264 runs; load average at report time: 5.3 3.5 3.0), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `79617e2`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

