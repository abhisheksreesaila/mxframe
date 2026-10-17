# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-17

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 19 | **12** | 16 | - |
| Q2 min-cost supplier | **9.6** | 12 | 14 | - |
| Q3 3-join+top10 | **15** | 26 | 31 | - |
| Q4 EXISTS semi-join | **15** | 21 | 23 | - |
| Q5 4-join+groupby | **14** | 15 | 19 | - |
| Q6 filter+global sum | **5.0** | 8.5 | 11 | - |
| Q7 volume shipping | **17** | 19 | 23 | - |
| Q8 market share | **16** | 18 | 21 | - |
| Q9 6-join profit | **36** | 44 | 45 | - |
| Q10 returned items top20 | **21** | 25 | 30 | - |
| Q11 important stock | **5.9** | 12 | 18 | - |
| Q12 shipping modes | **13** | 20 | 24 | - |
| Q13 customer distribution | 24 | **10** | 15 | PyArrow/NumPy |
| Q14 promo revenue | 11 | **10** | 13 | - |
| Q15 top supplier | **9.2** | 13 | 19 | - |
| Q16 part/supplier | **23** | 31 | 35 | - |
| Q17 small-qty revenue | **8.6** | 12 | 16 | - |
| Q18 large volume top100 | **78** | 98 | 109 | - |
| Q19 discounted revenue | 19 | **16** | 19 | - |
| Q20 potential promotion | **12** | 18 | 23 | - |
| Q21 suppliers waiting | 219 | **143** | 144 | - |
| Q22 global sales opportunity | **15** | 20 | 23 | - |
| **sum over the 22 queries valid for every engine** | **607** | **604** | **690** |  |

Fastest verified engine per query: DuckDB 17, mx-cpu 5, mx-auto 0.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats DuckDB on 5/22** (22 comparable)
- **mx-auto beats DuckDB on 3/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 131 | 132 | 119 |
| Q2 | 118 | 119 | 119 |
| Q3 | 124 | 129 | 137 |
| Q4 | 125 | 148 | 137 |
| Q5 | 128 | 118 | 121 |
| Q6 | 116 | 147 | 146 |
| Q7 | 127 | 123 | 127 |
| Q8 | 127 | 152 | 156 |
| Q9 | 145 | 147 | 152 |
| Q10 | 135 | 136 | 149 |
| Q11 | 123 | 129 | 156 |
| Q12 | 129 | 127 | 129 |
| Q13 | 133 | 116 | 121 |
| Q14 | 123 | 114 | 123 |
| Q15 | 119 | 146 | 156 |
| Q16 | 137 | 139 | 140 |
| Q17 | 121 | 117 | 140 |
| Q18 | 193 | 203 | 218 |
| Q19 | 134 | 148 | 153 |
| Q20 | 123 | 165 | 160 |
| Q21 | 329 | 273 | 259 |
| Q22 | 130 | 156 | 132 |
| **sum (22 queries)** | **3,069** | **3,182** | **3,251** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 150 | 58 | 55 |
| Q2 | 83 | 35 | 28 |
| Q3 | 125 | 88 | 90 |
| Q4 | 112 | 72 | 73 |
| Q5 | 143 | 90 | 91 |
| Q6 | 83 | 39 | 38 |
| Q7 | 119 | 94 | 92 |
| Q8 | 147 | 130 | 127 |
| Q9 | 179 | 144 | 144 |
| Q10 | 145 | 86 | 85 |
| Q11 | 61 | 28 | 23 |
| Q12 | 166 | 78 | 77 |
| Q13 | 53 | 35 | 36 |
| Q14 | 99 | 94 | 88 |
| Q15 | 84 | 51 | 50 |
| Q16 | 67 | 27 | 26 |
| Q17 | 86 | 90 | 92 |
| Q18 | 89 | 80 | 77 |
| Q19 | 180 | 104 | 97 |
| Q20 | 131 | 107 | 108 |
| Q21 | 115 | 87 | 88 |
| Q22 | 53 | 24 | 26 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 19 | 8.7 | 10 |
| Q2 | 9.0 | 7.7 | 7.8 |
| Q3 | 18 | 17 | 19 |
| Q4 | 15 | 12 | 13 |
| Q5 | 14 | 11 | 11 |
| Q6 | 5.9 | 4.6 | 4.7 |
| Q7 | 17 | 13 | 14 |
| Q8 | 14 | 12 | 13 |
| Q9 | 32 | 33 | 34 |
| Q10 | 20 | 18 | 18 |
| Q11 | 6.6 | 6.5 | 8.7 |
| Q12 | 14 | 16 | 14 |
| Q13 | 24 | 5.5 | 7.3 |
| Q14 | 13 | 15 | 5.6 |
| Q15 | 9.3 | 8.8 | 12 |
| Q16 | 21 | 25 | 27 |
| Q17 | 8.5 | 7.3 | 7.3 |
| Q18 | 77 | 84 | 97 |
| Q19 | 22 | 12 | 12 |
| Q20 | 12 | 12 | 14 |
| Q21 | 227 | 89 | 90 |
| Q22 | 14 | 15 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 111 | 109 | 110 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 2.6e-15, mx-auto 2.2e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.2, max 4.2 over 66 runs; load average at report time: 2.3 3.3 3.1), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

