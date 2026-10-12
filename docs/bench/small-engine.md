# Engine-only TPC-H scoreboard - 2026-10-12

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 9.9 | **2.7 †3e-6** | 80 | 3.4 | - |
| Q2 min-cost supplier | 8.0 | **2.6** | 82 | 5.4 | - |
| Q3 3-join+top10 | 5.5 | **2.3** | 75 | 3.9 | - |
| Q4 EXISTS semi-join | 2.8 | **2.0** | 78 | 3.9 | - |
| Q5 4-join+groupby | 5.4 | **2.6** | 76 | 4.1 | - |
| Q6 filter+global sum | 1.4 | **1.4** | 71 | 25 | - |
| Q7 volume shipping | 11 | **2.9** | 76 | 123 | - |
| Q8 market share | 7.8 | **2.9** | 77 | 4.6 | - |
| Q9 6-join profit | 7.7 | **3.7** | 80 | 5.1 | - |
| Q10 returned items top20 | 5.9 | **2.4** | 75 | 4.3 | - |
| Q11 important stock | 2.4 | **1.4** | 74 | ERR | - |
| Q12 shipping modes | 4.8 | **2.6** | 77 | 3.9 | - |
| Q13 customer distribution | 4.6 | **1.6** | 75 | 116 | PyArrow/NumPy |
| Q14 promo revenue | 2.9 | **1.6** | 73 | 3.6 | - |
| Q15 top supplier | 3.7 | **2.4** | 83 | 5.9 | - |
| Q16 part/supplier | 6.8 | **3.2** | 84 | 131 | - |
| Q17 small-qty revenue | 6.2 | **1.9** | 78 | 140 WRONG | - |
| Q18 large volume top100 | 4.9 | **3.0** | 85 | 4.5 | - |
| Q19 discounted revenue | 6.1 WRONG | 2.2 WRONG | **73** | 117 | - |
| Q20 potential promotion | 5.4 | **2.8** | 77 | 4.7 | - |
| Q21 suppliers waiting | 4.5 | **3.0** | 80 | 115 | - |
| Q22 global sales opportunity | 6.0 | **2.3** | 76 | 4.2 | PyArrow/NumPy |
| **sum over the 19 queries valid for every engine** | **109** | **48** | **1,479** | **571** |  |

Fastest verified engine per query: pandas 0, Polars 21, DuckDB 1, mx-cpu 0.

### Wins at SF0.002 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (19 comparable); **mx-cpu beats pandas on 11/22** (19 comparable); **mx-cpu beats DuckDB on 15/22** (20 comparable)

### SF0.002: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 295 | 83 †3e-6 | 122 | 148 |
| Q2 | 304 | 77 | 129 | 135 |
| Q3 | 242 | 79 | 114 | 136 |
| Q4 | 254 | 74 | 118 | 130 |
| Q5 | 250 | 79 | 115 | 141 |
| Q6 | 270 | 74 | 110 | 151 |
| Q7 | 299 | 84 | 117 | 291 |
| Q8 | 315 | 85 | 119 | 172 |
| Q9 | 258 | 78 | 125 | 133 |
| Q10 | 242 | 75 | 116 | 145 |
| Q11 | 236 | 75 | 114 | ERR |
| Q12 | 272 | 76 | 118 | 149 |
| Q13 | 265 | 81 | 116 | 280 |
| Q14 | 279 | 76 | 113 | 137 |
| Q15 | 283 | 81 | 126 | 144 |
| Q16 | 285 | 87 | 128 | 269 |
| Q17 | 307 | 94 | 118 | 300 WRONG |
| Q18 | 279 | 79 | 126 | 134 |
| Q19 | 249 WRONG | 77 WRONG | 112 | 249 |
| Q20 | 263 | 77 | 117 | 177 |
| Q21 | 240 | 74 | 119 | 244 |
| Q22 | 242 | 76 | 119 | 186 |
| **sum (19 queries)** | **5,137** | **1,496** | **2,267** | **3,301** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.002: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 18 | 19 | 12 | 36 |
| Q2 | 17 | 19 | 12 | 33 |
| Q3 | 16 | 18 | 9.9 | 31 |
| Q4 | 17 | 18 | 9.9 | 31 |
| Q5 | 18 | 22 | 12 | 32 |
| Q6 | 14 | 11 | 9.0 | 6.0 |
| Q7 | 21 | 21 | 12 | 32 |
| Q8 | 21 | 26 | 13 | 40 |
| Q9 | 22 | 21 | 14 | 32 |
| Q10 | 17 | 20 | 11 | 31 |
| Q11 | 15 | 19 | 8.1 | 28 |
| Q12 | 17 | 19 | 10 | 32 |
| Q13 | 15 | 17 | 8.4 | 30 |
| Q14 | 18 | 19 | 9.7 | 31 |
| Q15 | 17 | 18 | 9.9 | 28 |
| Q16 | 19 | 21 | 9.6 | 32 |
| Q17 | 21 | 20 | 9.4 | 8.4 |
| Q18 | 17 | 19 | 9.8 | 30 |
| Q19 | 17 | 20 | 11 | 29 |
| Q20 | 18 | 19 | 11 | 33 |
| Q21 | 16 | 19 | 9.7 | 30 |
| Q22 | 15 | 17 | 8.7 | 32 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 8.5 | 2.2 | 2.3 | 2.0 |
| Q2 | 6.4 | 2.0 | 5.4 | 2.9 |
| Q3 | 4.6 | 1.8 | 3.8 | 2.0 |
| Q4 | 2.3 | 1.5 | 3.9 | 1.7 |
| Q5 | 4.5 | 1.9 | 3.5 | 2.1 |
| Q6 | 0.9 | 0.9 | 0.8 | 0.8 |
| Q7 | 9.4 | 2.3 | 3.4 | 3.8 |
| Q8 | 6.5 | 1.9 | 4.9 | 2.8 |
| Q9 | 7.0 | 2.7 | 5.3 | 3.3 |
| Q10 | 4.5 | 2.2 | 4.4 | 2.0 |
| Q11 | 1.7 | 0.8 | 1.9 | - |
| Q12 | 3.8 | 1.7 | 2.7 | 2.0 |
| Q13 | 3.5 | 1.1 | 2.8 | 1.7 |
| Q14 | 1.8 | 1.0 | 1.3 | 1.6 |
| Q15 | 2.8 | 1.3 | 2.2 | 1.8 |
| Q16 | 5.7 | 2.5 | 5.1 | 2.8 |
| Q17 | 3.4 | 1.6 | 1.7 | 1.9 WRONG |
| Q18 | 3.6 | 2.7 | 4.4 | 1.9 |
| Q19 | 4.9 WRONG | 2.2 WRONG | 2.0 | 3.4 |
| Q20 | 4.6 | 1.9 | 3.3 | 2.7 |
| Q21 | 3.4 | 2.8 | 7.9 | 4.5 |
| Q22 | 5.0 | 1.7 | 4.1 | 2.0 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 264 | 75 | 40 | 138 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |
| Q11 | mx-cpu | error | - | TypeError: unsupported operand type(s) for *: 'NoneType' and 'float' |
| Q17 | mx-cpu | wrong | inf | max relative error inf / repeat run WRONG |
| Q19 | pandas | wrong | inf | max relative error inf / repeat run WRONG |
| Q19 | polars | wrong | inf | max relative error inf / repeat run WRONG |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 24 | **5.4 †2e-4** | 75 | 27 | - |
| Q2 min-cost supplier | 8.8 | **2.8** | 79 | 4.7 | - |
| Q3 3-join+top10 | 7.9 | **2.7** | 80 | 5.4 | - |
| Q4 EXISTS semi-join | 4.6 | **2.9** | 86 | 6.0 | - |
| Q5 4-join+groupby | 8.7 | **3.2** | 77 | 5.0 | - |
| Q6 filter+global sum | 1.8 | **1.0** | 73 | 25 | - |
| Q7 volume shipping | 14 | **4.1** | 76 | 6.2 | - |
| Q8 market share | 7.6 | **2.5** | 81 | 5.9 | - |
| Q9 6-join profit | 9.8 | **4.5** | 78 | 7.5 | - |
| Q10 returned items top20 | 7.1 | **3.4** | 79 | 5.3 | - |
| Q11 important stock | 2.8 | **1.9** | 73 | 113 | - |
| Q12 shipping modes | 6.0 | **2.6** | 74 | 6.4 | - |
| Q13 customer distribution | 5.7 | **2.6** | 78 | 113 | PyArrow/NumPy |
| Q14 promo revenue | 2.9 | **1.8** | 71 | 3.8 | - |
| Q15 top supplier | 3.9 | **2.0** | 73 | 6.2 | - |
| Q16 part/supplier | 6.8 | **3.0** | 78 | 111 | - |
| Q17 small-qty revenue | 4.4 | **1.9** | 74 | 5.2 | - |
| Q18 large volume top100 | 8.8 | **3.6** | 80 | 10 | - |
| Q19 discounted revenue | 16 | **2.8** | 75 | 8.0 | - |
| Q20 potential promotion | 6.2 | **3.0** | 80 | 5.4 | - |
| Q21 suppliers waiting | 16 | **9.2** | 85 | 24 | - |
| Q22 global sales opportunity | 6.5 | **2.3** | 80 | 4.1 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **180** | **69** | **1,705** | **507** |  |

Fastest verified engine per query: pandas 0, Polars 22, DuckDB 0, mx-cpu 0.

### Wins at SF0.017 (engine-only, both results verified)

- **mx-cpu beats Polars on 0/22** (22 comparable); **mx-cpu beats pandas on 10/22** (22 comparable); **mx-cpu beats DuckDB on 19/22** (22 comparable)

### SF0.017: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 281 | 86 †2e-4 | 121 | 155 |
| Q2 | 287 | 76 | 120 | 134 |
| Q3 | 284 | 78 | 119 | 150 |
| Q4 | 257 | 76 | 125 | 132 |
| Q5 | 242 | 75 | 117 | 129 |
| Q6 | 237 | 72 | 111 | 167 |
| Q7 | 277 | 76 | 114 | 131 |
| Q8 | 243 | 74 | 119 | 132 |
| Q9 | 245 | 78 | 118 | 172 |
| Q10 | 241 | 75 | 118 | 137 |
| Q11 | 253 | 77 | 111 | 240 |
| Q12 | 249 | 76 | 114 | 134 |
| Q13 | 257 | 75 | 116 | 252 |
| Q14 | 249 | 79 | 110 | 134 |
| Q15 | 242 | 74 | 113 | 162 |
| Q16 | 243 | 77 | 116 | 238 |
| Q17 | 242 | 74 | 113 | 170 |
| Q18 | 245 | 78 | 119 | 138 |
| Q19 | 252 | 77 | 115 | 135 |
| Q20 | 260 | 77 | 120 | 135 |
| Q21 | 252 | 82 | 126 | 152 |
| Q22 | 264 | 76 | 125 | 150 |
| **sum (22 queries)** | **5,600** | **1,686** | **2,582** | **3,480** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.017: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 21 | 21 | 29 | 8.2 |
| Q2 | 19 | 20 | 11 | 32 |
| Q3 | 19 | 21 | 20 | 32 |
| Q4 | 18 | 19 | 17 | 30 |
| Q5 | 20 | 22 | 20 | 31 |
| Q6 | 16 | 14 | 16 | 7.0 |
| Q7 | 21 | 21 | 21 | 33 |
| Q8 | 21 | 23 | 22 | 32 |
| Q9 | 22 | 23 | 25 | 35 |
| Q10 | 20 | 21 | 24 | 33 |
| Q11 | 16 | 18 | 9.2 | 31 |
| Q12 | 19 | 20 | 23 | 32 |
| Q13 | 16 | 18 | 9.2 | 29 |
| Q14 | 18 | 18 | 16 | 38 |
| Q15 | 17 | 19 | 16 | 31 |
| Q16 | 17 | 19 | 9.9 | 28 |
| Q17 | 17 | 20 | 17 | 28 |
| Q18 | 17 | 21 | 15 | 28 |
| Q19 | 21 | 20 | 32 | 31 |
| Q20 | 19 | 20 | 19 | 33 |
| Q21 | 18 | 20 | 18 | 33 |
| Q22 | 16 | 17 | 9.9 | 29 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 21 | 4.5 | 5.4 | 3.3 |
| Q2 | 7.5 | 2.1 | 5.8 | 2.7 |
| Q3 | 6.3 | 2.4 | 5.3 | 3.4 |
| Q4 | 4.0 | 2.6 | 4.9 | 4.1 |
| Q5 | 7.6 | 2.7 | 5.1 | 3.0 |
| Q6 | 1.1 | 1.5 | 1.3 | 0.8 |
| Q7 | 12 | 3.6 | 5.6 | 4.0 |
| Q8 | 6.4 | 1.8 | 5.7 | 3.6 |
| Q9 | 8.8 | 3.8 | 6.3 | 5.4 |
| Q10 | 5.8 | 2.9 | 7.2 | 3.4 |
| Q11 | 1.9 | 2.0 | 2.1 | 2.1 |
| Q12 | 5.1 | 2.1 | 4.2 | 4.3 |
| Q13 | 4.8 | 2.2 | 3.6 | 2.3 |
| Q14 | 2.0 | 1.2 | 2.0 | 2.0 |
| Q15 | 2.9 | 1.6 | 2.5 | 3.1 |
| Q16 | 5.7 | 4.3 | 5.4 | 3.3 |
| Q17 | 3.5 | 3.7 | 2.7 | 3.3 |
| Q18 | 7.1 | 3.3 | 5.8 | 7.0 |
| Q19 | 15 | 2.2 | 5.0 | 6.6 |
| Q20 | 5.1 | 2.2 | 4.1 | 2.9 |
| Q21 | 13 | 14 | 12 | 21 |
| Q22 | 5.2 | 1.6 | 4.9 | 1.7 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 241 | 73 | 39 | 129 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.3e-15.


## Scale factor 0.17: engine-only (query on loaded data, first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 113 | 38 †2e-3 | 83 | **9.3** | - |
| Q2 min-cost supplier | 14 | **3.0** | 83 | 7.8 | - |
| Q3 3-join+top10 | 23 | **6.5** | 82 | 18 | - |
| Q4 EXISTS semi-join | 33 | **14** | 81 | 30 | - |
| Q5 4-join+groupby | 34 | **9.3** | 79 | 15 | - |
| Q6 filter+global sum | 5.6 | **2.0** | 74 | 3.4 | - |
| Q7 volume shipping | 52 | 21 | 77 | **19** | - |
| Q8 market share | 15 | **4.2** | 76 | 16 | - |
| Q9 6-join profit | 36 | **11** | 97 | 30 | - |
| Q10 returned items top20 | 28 | **7.4** | 85 | 27 | - |
| Q11 important stock | 4.5 | **2.8** | 75 | 6.1 | - |
| Q12 shipping modes | 24 | **6.1** | 78 | 32 | - |
| Q13 customer distribution | 22 | **13** | 85 | 111 | PyArrow/NumPy |
| Q14 promo revenue | 6.5 | **2.5** | 75 | 7.7 | - |
| Q15 top supplier | 8.8 | **3.2** | 75 | 9.9 | - |
| Q16 part/supplier | 12 | **5.3** | 91 | 13 | - |
| Q17 small-qty revenue | 9.3 | **3.0** | 76 | 12 | - |
| Q18 large volume top100 | 34 | **20** | 82 | 59 | - |
| Q19 discounted revenue | 114 | **7.9** | 83 | 40 | - |
| Q20 potential promotion | 23 | **5.3** | 80 | 15 | - |
| Q21 suppliers waiting | 127 | **59** | 123 | 162 | - |
| Q22 global sales opportunity | 9.2 | **4.4** | 79 | 7.8 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **751** | **249** | **1,818** | **652** |  |

Fastest verified engine per query: pandas 0, Polars 20, DuckDB 0, mx-cpu 2.

### Wins at SF0.17 (engine-only, both results verified)

- **mx-cpu beats Polars on 2/22** (22 comparable); **mx-cpu beats pandas on 12/22** (22 comparable); **mx-cpu beats DuckDB on 20/22** (22 comparable)

### SF0.17: import + first run, ms (engine-only + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 351 | 114 †2e-3 | 123 | 135 |
| Q2 | 253 | 77 | 124 | 135 |
| Q3 | 256 | 79 | 120 | 176 |
| Q4 | 301 | 87 | 121 | 157 |
| Q5 | 291 | 82 | 117 | 142 |
| Q6 | 245 | 75 | 112 | 134 |
| Q7 | 288 | 92 | 116 | 146 |
| Q8 | 258 | 76 | 115 | 142 |
| Q9 | 287 | 85 | 134 | 157 |
| Q10 | 315 | 86 | 126 | 200 |
| Q11 | 274 | 76 | 112 | 134 |
| Q12 | 269 | 83 | 116 | 163 |
| Q13 | 260 | 92 | 129 | 240 |
| Q14 | 245 | 76 | 114 | 150 |
| Q15 | 260 | 76 | 118 | 140 |
| Q16 | 249 | 77 | 133 | 140 |
| Q17 | 286 | 77 | 118 | 140 |
| Q18 | 272 | 98 | 121 | 201 |
| Q19 | 388 | 83 | 127 | 171 |
| Q20 | 304 | 80 | 122 | 171 |
| Q21 | 425 | 133 | 164 | 294 |
| Q22 | 248 | 79 | 117 | 140 |
| **sum (22 queries)** | **6,326** | **1,884** | **2,697** | **3,606** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF0.17: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 50 | 32 | 118 | 29 |
| Q2 | 23 | 22 | 26 | 35 |
| Q3 | 38 | 31 | 69 | 41 |
| Q4 | 35 | 30 | 51 | 36 |
| Q5 | 38 | 34 | 69 | 43 |
| Q6 | 23 | 21 | 57 | 27 |
| Q7 | 41 | 34 | 74 | 44 |
| Q8 | 41 | 37 | 83 | 46 |
| Q9 | 48 | 41 | 102 | 51 |
| Q10 | 56 | 35 | 84 | 43 |
| Q11 | 18 | 20 | 16 | 30 |
| Q12 | 58 | 35 | 84 | 48 |
| Q13 | 20 | 24 | 20 | 31 |
| Q14 | 29 | 24 | 58 | 40 |
| Q15 | 29 | 24 | 56 | 38 |
| Q16 | 21 | 22 | 19 | 29 |
| Q17 | 27 | 23 | 51 | 33 |
| Q18 | 31 | 27 | 42 | 35 |
| Q19 | 59 | 31 | 137 | 41 |
| Q20 | 38 | 29 | 69 | 41 |
| Q21 | 38 | 30 | 56 | 40 |
| Q22 | 18 | 19 | 15 | 28 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF0.17: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 106 | 33 | 7.3 | 5.7 |
| Q2 | 12 | 2.9 | 7.0 | 6.5 |
| Q3 | 20 | 6.0 | 8.6 | 14 |
| Q4 | 25 | 13 | 8.5 | 22 |
| Q5 | 35 | 8.4 | 9.6 | 11 |
| Q6 | 3.5 | 1.3 | 2.3 | 1.4 |
| Q7 | 50 | 15 | 8.1 | 15 |
| Q8 | 13 | 3.5 | 6.7 | 13 |
| Q9 | 31 | 11 | 22 | 27 |
| Q10 | 22 | 7.1 | 12 | 21 |
| Q11 | 3.6 | 2.2 | 3.7 | 3.5 |
| Q12 | 20 | 5.2 | 6.3 | 32 |
| Q13 | 19 | 10 | 12 | 4.6 |
| Q14 | 5.1 | 2.0 | 4.5 | 7.0 |
| Q15 | 5.8 | 2.5 | 2.9 | 7.8 |
| Q16 | 11 | 4.9 | 11 | 10 |
| Q17 | 8.2 | 2.9 | 4.8 | 9.7 |
| Q18 | 30 | 14 | 9.4 | 51 |
| Q19 | 116 | 7.5 | 7.5 | 35 |
| Q20 | 18 | 4.4 | 6.8 | 11 |
| Q21 | 117 | 60 | 54 | 145 |
| Q22 | 8.0 | 5.5 | 5.8 | 4.1 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF0.17: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 244 | 73 | 39 | 129 |

### SF0.17: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 1.57e-03 | max relative error 1.57e-03 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.3e-07, Polars 1.6e-03, mx-cpu 2.0e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.4, max 6.5 over 264 runs; load average at report time: 4.3 3.2 2.4), so timings are indicative.**
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

