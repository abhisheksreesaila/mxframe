# Engine-only TPC-H scoreboard - 2026-10-12

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. This isolates the query engine from file reading; the user-felt scoreboard (clock around read + query) stays the headline for what a user waits for. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | mx-cpu | outside mx |
|---|---:|---:|
| Q1 filter+groupby+8 aggs | **40** | - |
| Q2 min-cost supplier | **41** | - |
| Q3 3-join+top10 | **105** | - |
| Q4 EXISTS semi-join | **128** | - |
| Q5 4-join+groupby | **77** | - |
| Q6 filter+global sum | **5.6** | - |
| Q7 volume shipping | **112** | - |
| Q8 market share | **82** | - |
| Q9 6-join profit | **165** | - |
| Q10 returned items top20 | **139** | - |
| Q11 important stock | **35** | - |
| Q12 shipping modes | **197** | - |
| Q13 customer distribution | **49** | PyArrow/NumPy |
| Q14 promo revenue | **54** | - |
| Q15 top supplier | **45** | - |
| Q16 part/supplier | **68** | - |
| Q17 small-qty revenue | **58** | - |
| Q18 large volume top100 | **284** | - |
| Q19 discounted revenue | **241** | - |
| Q20 potential promotion | **75** | - |
| Q21 suppliers waiting | **696** | - |
| Q22 global sales opportunity | **37** | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **2,734** |  |

Fastest verified engine per query: mx-cpu 22.

### Wins at SF1 (engine-only, both results verified)

- 

### SF1: import + first run, ms (engine-only + engine import time)

| query | mx-cpu |
|---|---:|
| Q1 | 155 |
| Q2 | 143 |
| Q3 | 207 |
| Q4 | 251 |
| Q5 | 184 |
| Q6 | 109 |
| Q7 | 215 |
| Q8 | 185 |
| Q9 | 268 |
| Q10 | 247 |
| Q11 | 139 |
| Q12 | 299 |
| Q13 | 186 |
| Q14 | 156 |
| Q15 | 149 |
| Q16 | 170 |
| Q17 | 162 |
| Q18 | 388 |
| Q19 | 343 |
| Q20 | 178 |
| Q21 | 798 |
| Q22 | 140 |
| **sum (22 queries)** | **5,072** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | mx-cpu |
|---|---:|
| Q1 | 58 |
| Q2 | 25 |
| Q3 | 79 |
| Q4 | 70 |
| Q5 | 88 |
| Q6 | 32 |
| Q7 | 91 |
| Q8 | 124 |
| Q9 | 137 |
| Q10 | 87 |
| Q11 | 17 |
| Q12 | 77 |
| Q13 | 37 |
| Q14 | 83 |
| Q15 | 45 |
| Q16 | 24 |
| Q17 | 83 |
| Q18 | 74 |
| Q19 | 106 |
| Q20 | 100 |
| Q21 | 75 |
| Q22 | 24 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | mx-cpu |
|---|---:|
| Q1 | 16 |
| Q2 | 20 |
| Q3 | 74 |
| Q4 | 95 |
| Q5 | 43 |
| Q6 | 4.3 |
| Q7 | 74 |
| Q8 | 52 |
| Q9 | 134 |
| Q10 | 104 |
| Q11 | 9.1 |
| Q12 | 164 |
| Q13 | 14 |
| Q14 | 32 |
| Q15 | 15 |
| Q16 | 45 |
| Q17 | 32 |
| Q18 | 240 |
| Q19 | 203 |
| Q20 | 50 |
| Q21 | 645 |
| Q22 | 12 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | mx-cpu |
|---|---:|
| import | 103 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.4e-15.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.7, max 4.0 over 22 runs; load average at report time: 3.4 4.5 5.2), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `8de50e7`. Last invocation took 0 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

