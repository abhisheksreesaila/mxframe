# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-17

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 138 | **67** | 68 | - |
| Q2 min-cost supplier | **23** | 27 | 31 | - |
| Q3 3-join+top10 | **84** | 117 | 126 | - |
| Q4 EXISTS semi-join | 108 | **105** | 108 | - |
| Q5 4-join+groupby | 85 | **74** | 78 | - |
| Q6 filter+global sum | **32** | 33 | 35 | - |
| Q7 volume shipping | 88 | **86** | 89 | - |
| Q8 market share | **66** | 68 | 72 | - |
| Q9 6-join profit | **229** | 323 | 325 | - |
| Q10 returned items top20 | **134** | 165 | 168 | - |
| Q11 important stock | **15** | 36 | 44 | - |
| Q12 shipping modes | **81** | 120 | 122 | - |
| Q13 customer distribution | 219 | **94** | 102 | PyArrow/NumPy |
| Q14 promo revenue | 78 | **37** | 41 | - |
| Q15 top supplier | **73** | 82 | 87 | - |
| Q16 part/supplier | **128** | 130 | 167 | - |
| Q17 small-qty revenue | **36** | 72 | 73 | - |
| Q18 large volume top100 | **1,099** | 1,678 | 1,266 | - |
| Q19 discounted revenue | 84 | **67** | 70 | - |
| Q20 potential promotion | **49** | 65 | 72 | - |
| Q21 suppliers waiting | 2,624 | **1,288** | 1,299 | - |
| Q22 global sales opportunity | **40** | 91 | 93 | - |
| **sum over the 22 queries valid for every engine** | **5,513** | **4,825** | **4,538** |  |

Fastest verified engine per query: DuckDB 14, mx-cpu 8, mx-auto 0.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats DuckDB on 8/22** (22 comparable)
- **mx-auto beats DuckDB on 6/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 254 | 173 | 174 |
| Q2 | 134 | 130 | 135 |
| Q3 | 196 | 223 | 234 |
| Q4 | 219 | 208 | 211 |
| Q5 | 194 | 177 | 185 |
| Q6 | 141 | 155 | 143 |
| Q7 | 198 | 188 | 195 |
| Q8 | 176 | 170 | 177 |
| Q9 | 338 | 426 | 430 |
| Q10 | 243 | 269 | 272 |
| Q11 | 124 | 142 | 148 |
| Q12 | 190 | 223 | 228 |
| Q13 | 327 | 202 | 205 |
| Q14 | 191 | 150 | 170 |
| Q15 | 190 | 198 | 194 |
| Q16 | 248 | 239 | 280 |
| Q17 | 156 | 182 | 178 |
| Q18 | 1,219 | 1,790 | 1,381 |
| Q19 | 199 | 170 | 173 |
| Q20 | 162 | 168 | 177 |
| Q21 | 2,735 | 1,391 | 1,402 |
| Q22 | 153 | 198 | 199 |
| **sum (22 queries)** | **7,989** | **7,171** | **6,890** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 624 | 488 | 475 |
| Q2 | 248 | 106 | 109 |
| Q3 | 484 | 669 | 668 |
| Q4 | 349 | 603 | 600 |
| Q5 | 507 | 684 | 674 |
| Q6 | 316 | 267 | 268 |
| Q7 | 543 | 718 | 712 |
| Q8 | 731 | 1,029 | 1,032 |
| Q9 | 811 | 1,175 | 1,173 |
| Q10 | 472 | 675 | 683 |
| Q11 | 106 | 84 | 80 |
| Q12 | 488 | 701 | 703 |
| Q13 | 98 | 205 | 207 |
| Q14 | 451 | 865 | 878 |
| Q15 | 512 | 474 | 380 |
| Q16 | 196 | 113 | 119 |
| Q17 | 518 | 817 | 792 |
| Q18 | 329 | 666 | 673 |
| Q19 | 765 | 885 | 892 |
| Q20 | 553 | 909 | 910 |
| Q21 | 428 | 642 | 648 |
| Q22 | 135 | 242 | 239 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 139 | 63 | 63 |
| Q2 | 24 | 21 | 22 |
| Q3 | 88 | 99 | 102 |
| Q4 | 108 | 89 | 89 |
| Q5 | 85 | 61 | 63 |
| Q6 | 32 | 29 | 29 |
| Q7 | 85 | 75 | 77 |
| Q8 | 63 | 54 | 57 |
| Q9 | 229 | 317 | 319 |
| Q10 | 123 | 128 | 132 |
| Q11 | 15 | 24 | 27 |
| Q12 | 80 | 112 | 110 |
| Q13 | 220 | 82 | 86 |
| Q14 | 73 | 31 | 33 |
| Q15 | 75 | 76 | 81 |
| Q16 | 140 | 104 | 150 |
| Q17 | 35 | 64 | 68 |
| Q18 | 1,249 | 1,596 | 1,224 |
| Q19 | 86 | 62 | 63 |
| Q20 | 50 | 59 | - |
| Q21 | 2,715 | 1,082 | 1,079 |
| Q22 | 40 | 83 | 85 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 111 | 105 | 105 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.9e-14, mx-auto 2.3e-14.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.6, max 7.1 over 66 runs; load average at report time: 3.6 3.7 3.6), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 13 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

