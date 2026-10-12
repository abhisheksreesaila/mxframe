# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 25 | **19 †3e-6** | 86 | 33 | - |
| Q2 min-cost supplier | 25 | **16** | 88 | 35 | - |
| Q3 3-join+top10 | 22 | **21** | 88 | 34 | - |
| Q4 EXISTS semi-join | **18** | 20 | 87 | 35 | - |
| Q5 4-join+groupby | 24 | **19** | 87 | 34 | - |
| Q6 filter+global sum | **15** | 19 | 82 | 29 | - |
| Q7 volume shipping | 28 | **19** | 88 | 34 | - |
| Q8 market share | 26 | **19** | 88 | 36 | - |
| Q9 6-join profit | 27 | **20** | 94 | 37 | - |
| Q10 returned items top20 | 23 | **19** | 90 | 34 | - |
| Q11 important stock | 17 | **15** | 83 | 34 | - |
| Q12 shipping modes | 21 | **20** | 87 | 35 | - |
| Q13 customer distribution | 18 | **14** | 83 | 32 | PyArrow/NumPy |
| Q14 promo revenue | **18** | 18 | 83 | 35 | - |
| Q15 top supplier | 21 | **18** | 92 | 34 | - |
| Q16 part/supplier | 22 | **17** | 89 | 36 | - |
| Q17 small-qty revenue | 19 | **16** | 89 | 38 | - |
| Q18 large volume top100 | **20** | 22 | 90 | 35 | - |
| Q19 discounted revenue | 23 | **17** | 88 | 39 | - |
| Q20 potential promotion | 22 | **21** | 89 | 37 | - |
| Q21 suppliers waiting | 21 | **21** | 95 | 38 | - |
| Q22 global sales opportunity | 21 | **19** | 87 | 32 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **474** | **409** | **1,933** | **765** |  |

Fastest verified engine per query: pandas 4, Polars 18, DuckDB 0, mx-cpu 0.

### Wins at SF0.002 (headline, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 0/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.002: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 263 | 92 †3e-6 | 125 | 158 |
| Q2 | 259 | 89 | 126 | 161 |
| Q3 | 253 | 92 | 127 | 160 |
| Q4 | 251 | 94 | 125 | 160 |
| Q5 | 258 | 91 | 127 | 161 |
| Q6 | 248 | 91 | 121 | 164 |
| Q7 | 279 | 91 | 129 | 159 |
| Q8 | 262 | 91 | 126 | 160 |
| Q9 | 258 | 91 | 133 | 162 |
| Q10 | 255 | 91 | 128 | 161 |
| Q11 | 249 | 94 | 121 | 168 |
| Q12 | 255 | 93 | 125 | 159 |
| Q13 | 253 | 87 | 122 | 159 |
| Q14 | 251 | 92 | 121 | 161 |
| Q15 | 259 | 90 | 131 | 162 |
| Q16 | 257 | 89 | 129 | 161 |
| Q17 | 256 | 90 | 127 | 182 |
| Q18 | 254 | 94 | 128 | 161 |
| Q19 | 259 | 89 | 127 | 166 |
| Q20 | 258 | 95 | 128 | 163 |
| Q21 | 255 | 94 | 136 | 166 |
| Q22 | 257 | 93 | 128 | 159 |
| **sum (22 queries)** | **5,650** | **2,014** | **2,788** | **3,572** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 9.6 | 13 | 13 | 2.8 |
| Q2 | 10 | 13 | 15 | 3.4 |
| Q3 | 7.1 | 18 | 14 | 2.8 |
| Q4 | 3.7 | 14 | 14 | 2.1 |
| Q5 | 7.8 | 15 | 17 | 3.3 |
| Q6 | 2.1 | 7.0 | 12 | 1.2 |
| Q7 | 12 | 15 | 16 | 4.9 |
| Q8 | 11 | 15 | 16 | 3.9 |
| Q9 | 12 | 9.4 | 17 | 4.9 |
| Q10 | 8.3 | 16 | 15 | 3.5 |
| Q11 | 3.3 | 13 | 12 | 1.7 |
| Q12 | 6.1 | 12 | 14 | 2.7 |
| Q13 | 4.4 | 7.8 | 14 | 2.0 |
| Q14 | 3.8 | 11 | 13 | 2.0 |
| Q15 | 4.8 | 13 | 13 | 2.4 |
| Q16 | 7.7 | 16 | 16 | 3.0 |
| Q17 | 4.7 | 11 | 11 | 2.3 |
| Q18 | 5.8 | 18 | 15 | 3.0 |
| Q19 | 7.2 | 8.8 | 12 | 4.0 |
| Q20 | 7.7 | 3.8 | 16 | 3.8 |
| Q21 | 6.1 | 17 | 19 | 5.6 |
| Q22 | 6.3 | 15 | 15 | 2.1 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 234 | 73 | 39 | 126 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 40 | **23 †2e-4** | 88 | 35 | - |
| Q2 min-cost supplier | 27 | **18** | 90 | 37 | - |
| Q3 3-join+top10 | 26 | **20** | 90 | 36 | - |
| Q4 EXISTS semi-join | 22 | **20** | 91 | 37 | - |
| Q5 4-join+groupby | 28 | **20** | 92 | 38 | - |
| Q6 filter+global sum | 16 | **16** | 83 | 32 | - |
| Q7 volume shipping | 34 | **20** | 90 | 40 | - |
| Q8 market share | 29 | **21** | 97 | 38 | - |
| Q9 6-join profit | 32 | **22** | 95 | 41 | - |
| Q10 returned items top20 | 27 | **19** | 92 | 39 | - |
| Q11 important stock | **18** | 18 | 84 | 33 | - |
| Q12 shipping modes | 25 | **18** | 89 | 36 | - |
| Q13 customer distribution | 20 | **17** | 88 | 33 | PyArrow/NumPy |
| Q14 promo revenue | **20** | 20 | 84 | 35 | - |
| Q15 top supplier | 21 | **20** | 87 | 37 | - |
| Q16 part/supplier | 23 | **19** | 92 | 38 | - |
| Q17 small-qty revenue | 21 | **17** | 88 | 37 | - |
| Q18 large volume top100 | 26 | **23** | 92 | 41 | - |
| Q19 discounted revenue | 38 | **21** | 89 | 38 | - |
| Q20 potential promotion | 25 | **22** | 92 | 39 | - |
| Q21 suppliers waiting | 34 | **28** | 104 | 56 | - |
| Q22 global sales opportunity | 22 | **19** | 87 | 38 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **574** | **441** | **1,986** | **835** |  |

Fastest verified engine per query: pandas 2, Polars 20, DuckDB 0, mx-cpu 0.

### Wins at SF0.017 (headline, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 1/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 273 | 94 †2e-4 | 127 | 160 |
| Q2 | 261 | 93 | 128 | 163 |
| Q3 | 262 | 92 | 130 | 164 |
| Q4 | 256 | 91 | 131 | 163 |
| Q5 | 262 | 95 | 132 | 164 |
| Q6 | 250 | 90 | 123 | 159 |
| Q7 | 269 | 92 | 131 | 168 |
| Q8 | 265 | 96 | 137 | 164 |
| Q9 | 268 | 93 | 135 | 168 |
| Q10 | 257 | 91 | 131 | 166 |
| Q11 | 252 | 90 | 123 | 159 |
| Q12 | 281 | 89 | 128 | 162 |
| Q13 | 253 | 88 | 127 | 161 |
| Q14 | 252 | 94 | 123 | 159 |
| Q15 | 254 | 93 | 127 | 164 |
| Q16 | 257 | 93 | 133 | 164 |
| Q17 | 254 | 88 | 128 | 163 |
| Q18 | 258 | 96 | 132 | 167 |
| Q19 | 273 | 93 | 128 | 163 |
| Q20 | 259 | 95 | 131 | 163 |
| Q21 | 270 | 99 | 144 | 181 |
| Q22 | 256 | 91 | 128 | 165 |
| **sum (22 queries)** | **5,742** | **2,036** | **2,859** | **3,611** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 23 | 16 | 18 | 4.2 |
| Q2 | 11 | 9.2 | 18 | 4.2 |
| Q3 | 9.7 | 15 | 17 | 4.9 |
| Q4 | 6.5 | 14 | 17 | 5.4 |
| Q5 | 13 | 14 | 18 | 5.4 |
| Q6 | 2.7 | 6.6 | 12 | 1.8 |
| Q7 | 18 | 16 | 20 | 6.5 |
| Q8 | 13 | 16 | 20 | 6.6 |
| Q9 | 15 | 17 | 22 | 8.3 |
| Q10 | 12 | 15 | 18 | 6.4 |
| Q11 | 3.8 | 7.8 | 14 | 2.4 |
| Q12 | 10 | 13 | 15 | 6.7 |
| Q13 | 6.3 | 17 | 14 | 2.7 |
| Q14 | 4.6 | 14 | 16 | 3.0 |
| Q15 | 5.3 | 12 | 15 | 4.2 |
| Q16 | 8.1 | 15 | 17 | 4.0 |
| Q17 | 5.9 | 12 | 14 | 3.9 |
| Q18 | 10 | 17 | 19 | 9.6 |
| Q19 | 21 | 14 | 16 | 9.5 |
| Q20 | 9.3 | 17 | 16 | 4.9 |
| Q21 | 17 | 22 | 26 | 23 |
| Q22 | 6.9 | 13 | 16 | 2.5 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 234 | 72 | 40 | 126 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.3e-15.


## Scale factor 0.17: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 170 | 61 †2e-3 | 131 | **44** | - |
| Q2 min-cost supplier | 37 | **20** | 97 | 40 | - |
| Q3 3-join+top10 | 62 | **28** | 110 | 60 | - |
| Q4 EXISTS semi-join | 65 | **35** | 110 | 68 | - |
| Q5 4-join+groupby | 74 | **35** | 110 | 57 | - |
| Q6 filter+global sum | 29 | **21** | 95 | 36 | - |
| Q7 volume shipping | 93 | **41** | 113 | 65 | - |
| Q8 market share | 57 | **32** | 108 | 61 | - |
| Q9 6-join profit | 82 | **40** | 130 | 78 | - |
| Q10 returned items top20 | 74 | **28** | 110 | 70 | - |
| Q11 important stock | 23 | **22** | 87 | 39 | - |
| Q12 shipping modes | 75 | **41** | 112 | 81 | - |
| Q13 customer distribution | 42 | **28** | 104 | 38 | PyArrow/NumPy |
| Q14 promo revenue | 37 | **23** | 99 | 42 | - |
| Q15 top supplier | 35 | **23** | 95 | 42 | - |
| Q16 part/supplier | 33 | **21** | 107 | 49 | - |
| Q17 small-qty revenue | 38 | **33** | 94 | 43 | - |
| Q18 large volume top100 | 65 | **40** | 129 | 101 | - |
| Q19 discounted revenue | 169 | **25** | 108 | 75 | - |
| Q20 potential promotion | 57 | **26** | 105 | 57 | - |
| Q21 suppliers waiting | 154 | **82** | 222 | 215 | - |
| Q22 global sales opportunity | 28 | **24** | 93 | 37 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **1,496** | **729** | **2,469** | **1,400** |  |

Fastest verified engine per query: pandas 0, Polars 21, DuckDB 0, mx-cpu 1.

### Wins at SF0.17 (headline, both results verified)

- **mx-cpu beats Polars on 1/22** (22 comparable); **mx-cpu beats pandas on 8/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.17: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 407 | 134 †2e-3 | 172 | 170 |
| Q2 | 273 | 94 | 137 | 169 |
| Q3 | 292 | 101 | 149 | 186 |
| Q4 | 298 | 106 | 149 | 195 |
| Q5 | 308 | 107 | 150 | 185 |
| Q6 | 266 | 95 | 133 | 160 |
| Q7 | 324 | 112 | 152 | 191 |
| Q8 | 292 | 103 | 146 | 186 |
| Q9 | 311 | 113 | 169 | 205 |
| Q10 | 306 | 100 | 148 | 193 |
| Q11 | 255 | 97 | 128 | 165 |
| Q12 | 309 | 112 | 151 | 209 |
| Q13 | 277 | 101 | 143 | 163 |
| Q14 | 270 | 97 | 138 | 169 |
| Q15 | 265 | 96 | 135 | 174 |
| Q16 | 265 | 95 | 145 | 176 |
| Q17 | 274 | 105 | 132 | 169 |
| Q18 | 299 | 112 | 170 | 247 |
| Q19 | 401 | 97 | 146 | 200 |
| Q20 | 292 | 97 | 144 | 183 |
| Q21 | 388 | 153 | 262 | 342 |
| Q22 | 262 | 94 | 131 | 163 |
| **sum (22 queries)** | **6,635** | **2,323** | **3,326** | **4,199** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.17: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 139 | 43 | 58 | 13 |
| Q2 | 19 | 17 | 25 | 9.6 |
| Q3 | 38 | 23 | 37 | 28 |
| Q4 | 42 | 29 | 34 | 32 |
| Q5 | 53 | 27 | 37 | 24 |
| Q6 | 12 | 15 | 22 | 7.8 |
| Q7 | 77 | 34 | 36 | 27 |
| Q8 | 37 | 26 | 40 | 27 |
| Q9 | 58 | 27 | 51 | 46 |
| Q10 | 52 | 22 | 37 | 37 |
| Q11 | 7.7 | 17 | 18 | 5.7 |
| Q12 | 53 | 32 | 35 | 47 |
| Q13 | 26 | 22 | 29 | 8.2 |
| Q14 | 16 | 18 | 25 | 11 |
| Q15 | 17 | 15 | 22 | 11 |
| Q16 | 16 | 17 | 24 | 12 |
| Q17 | 19 | 17 | 24 | 15 |
| Q18 | 48 | 38 | 55 | 66 |
| Q19 | 146 | 18 | 39 | 43 |
| Q20 | 35 | 19 | 32 | 23 |
| Q21 | 130 | 70 | 123 | 156 |
| Q22 | 12 | 17 | 20 | 7.0 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.17: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 234 | 72 | 39 | 126 |

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
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 1.9, max 3.7 over 264 runs; load average at report time: 3.2 2.7 3.3), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `24ff273`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

