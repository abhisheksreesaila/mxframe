# User-felt TPC-H scoreboard - 2026-09-30

> **Note (F-018 review):** Q2's harness query differs between the before and after runs: after, `scripts/tpch_user/q_mx.py` Q2 selects the columns the rest of the query reads. Harness narrowing alone takes SF10 mx-cpu Q2 from about 1,727 to about 600 ms; the join engine takes it from about 600 to 255 ms.

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | **36** | 338 | 317 | - |
| Q3 3-join+top10 | **69** | 308 | 337 | - |
| Q5 4-join+groupby | **93** | 235 | 236 | - |
| Q7 volume shipping | **156** | 732 | 776 | - |
| Q8 market share | **75** | 250 | 431 | - |
| Q9 6-join profit | **182 †2e-6** | 1,428 | 1,556 | - |
| Q10 returned items top20 | **71** | 278 | 263 | - |
| Q14 promo revenue | **35** | 121 | 144 | - |
| Q20 potential promotion | **52** | 191 | 220 | - |
| Q21 suppliers waiting | **393** | 2,200 WRONG | 2,201 WRONG | - |
| **sum over the 9 queries valid for every engine** | **768** | **3,882** | **4,280** |  |

Fastest verified engine per query: Polars 10, mx-cpu 0, mx-auto 0.

### Wins at SF1 (headline, both results verified)

- **mx-cpu beats Polars on 0/10** (9 comparable)
- **mx-auto beats Polars on 0/10** (9 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | mx-auto | 317 | 36 | 8.9x |
| Q9 6-join profit | mx-auto | 1,556 | 182 | 8.6x |
| Q8 market share | mx-auto | 431 | 75 | 5.7x |
| Q7 volume shipping | mx-auto | 776 | 156 | 5.0x |
| Q3 3-join+top10 | mx-auto | 337 | 69 | 4.9x |
| Q20 potential promotion | mx-auto | 220 | 52 | 4.2x |
| Q14 promo revenue | mx-auto | 144 | 35 | 4.2x |
| Q10 returned items top20 | mx-auto | 263 | 71 | 3.7x |
| Q5 4-join+groupby | mx-auto | 236 | 93 | 2.5x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | mx-auto | 317 | 36 | 8.9x |
| Q9 6-join profit | mx-cpu | 1,428 | 182 | 7.9x |
| Q7 volume shipping | mx-cpu | 732 | 156 | 4.7x |
| Q3 3-join+top10 | mx-cpu | 308 | 69 | 4.5x |
| Q10 returned items top20 | mx-auto | 263 | 71 | 3.7x |
| Q20 potential promotion | mx-cpu | 191 | 52 | 3.7x |
| Q14 promo revenue | mx-cpu | 121 | 35 | 3.5x |
| Q8 market share | mx-cpu | 250 | 75 | 3.3x |
| Q5 4-join+groupby | mx-cpu | 235 | 93 | 2.5x |

### SF1: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 155 | 473 | 471 |
| Q3 | 140 | 435 | 469 |
| Q5 | 163 | 362 | 363 |
| Q7 | 235 | 957 | 942 |
| Q8 | 155 | 442 | 598 |
| Q9 | 274 †2e-6 | 1,593 | 1,708 |
| Q10 | 225 | 413 | 389 |
| Q14 | 108 | 248 | 270 |
| Q20 | 124 | 317 | 345 |
| Q21 | 464 | 2,328 WRONG | 2,328 WRONG |
| **sum (9 queries)** | **1,578** | **5,240** | **5,556** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 22 | 152 | 148 |
| Q3 | 53 | 150 | 176 |
| Q5 | 71 | 176 | 186 |
| Q7 | 116 | 663 | 601 |
| Q8 | 56 | 304 | 269 |
| Q9 | 100 | 1,290 | 1,344 |
| Q10 | 42 | 207 | 210 |
| Q14 | 28 | 108 | 110 |
| Q20 | 41 | 151 | 178 |
| Q21 | 357 | 2,094 WRONG | 2,140 WRONG |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 76 | 131 | 130 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |
| Q21 | mx-auto | wrong | 7.69e-02 | top-N metric differs, max relative error 7.69e-02 / repeat run WRONG |
| Q21 | mx-cpu | wrong | 7.69e-02 | top-N metric differs, max relative error 7.69e-02 / repeat run WRONG |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 1.9e-06, mx-cpu 7.7e-02, mx-auto 7.7e-02.


## Scale factor 10: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | **99** | 1,727 | 1,865 | - |
| Q3 3-join+top10 | **492** | 2,100 | 2,268 | - |
| Q5 4-join+groupby | **1,082** | 2,761 | 3,550 | - |
| Q7 volume shipping | **1,445 †3e-6** | 7,419 | 7,417 | - |
| Q8 market share | **601** | 2,969 | 3,896 | - |
| Q9 6-join profit | **1,770 †7e-6** | 13,994 | 24,183 | - |
| Q10 returned items top20 | **402** | 3,132 | 3,124 | - |
| Q14 promo revenue | **192** | 1,454 | 1,976 | - |
| Q20 potential promotion | **274** | 2,157 | 2,215 | - |
| Q21 suppliers waiting | **5,574** | 40,218 WRONG | 36,246 WRONG | - |
| **sum over the 9 queries valid for every engine** | **6,358** | **37,713** | **50,494** |  |

Fastest verified engine per query: Polars 10, mx-cpu 0, mx-auto 0.

### Wins at SF10 (headline, both results verified)

- **mx-cpu beats Polars on 0/10** (9 comparable)
- **mx-auto beats Polars on 0/10** (9 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | mx-auto | 1,865 | 99 | 18.9x |
| Q9 6-join profit | mx-auto | 24,183 | 1,770 | 13.7x |
| Q14 promo revenue | mx-auto | 1,976 | 192 | 10.3x |
| Q20 potential promotion | mx-auto | 2,215 | 274 | 8.1x |
| Q10 returned items top20 | mx-auto | 3,124 | 402 | 7.8x |
| Q8 market share | mx-auto | 3,896 | 601 | 6.5x |
| Q7 volume shipping | mx-auto | 7,417 | 1,445 | 5.1x |
| Q3 3-join+top10 | mx-auto | 2,268 | 492 | 4.6x |
| Q5 4-join+groupby | mx-auto | 3,550 | 1,082 | 3.3x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | mx-cpu | 1,727 | 99 | 17.5x |
| Q9 6-join profit | mx-cpu | 13,994 | 1,770 | 7.9x |
| Q20 potential promotion | mx-cpu | 2,157 | 274 | 7.9x |
| Q10 returned items top20 | mx-auto | 3,124 | 402 | 7.8x |
| Q14 promo revenue | mx-cpu | 1,454 | 192 | 7.6x |
| Q7 volume shipping | mx-auto | 7,417 | 1,445 | 5.1x |
| Q8 market share | mx-cpu | 2,969 | 601 | 4.9x |
| Q3 3-join+top10 | mx-cpu | 2,100 | 492 | 4.3x |
| Q5 4-join+groupby | mx-cpu | 2,761 | 1,082 | 2.6x |

### SF10: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 172 | 1,854 | 2,025 |
| Q3 | 566 | 2,266 | 2,437 |
| Q5 | 1,181 | 2,943 | 3,703 |
| Q7 | 1,559 †3e-6 | 7,569 | 7,585 |
| Q8 | 693 | 3,121 | 4,041 |
| Q9 | 1,862 †7e-6 | 14,180 | 24,355 |
| Q10 | 487 | 3,328 | 3,286 |
| Q14 | 298 | 1,609 | 2,135 |
| Q20 | 405 | 2,289 | 2,401 |
| Q21 | 5,686 | 40,486 WRONG | 36,433 WRONG |
| **sum (9 queries)** | **7,224** | **39,161** | **51,967** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 81 | 1,589 | 1,581 |
| Q3 | 369 | 1,904 | 2,149 |
| Q5 | 1,038 | 2,577 | 2,861 |
| Q7 | 1,175 | 8,302 | 8,408 |
| Q8 | 443 | 2,806 | 3,601 |
| Q9 | 1,511 | 16,694 | 19,130 |
| Q10 | 351 | 2,651 | 2,782 |
| Q14 | 153 | 1,256 | 1,567 |
| Q20 | 249 | 2,026 | 2,289 |
| Q21 | 5,035 | 36,988 WRONG | 38,818 WRONG |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF10: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 96 | 161 | 165 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |
| Q21 | mx-auto | wrong | 6.25e-02 | top-N metric differs, max relative error 6.25e-02 / repeat run WRONG |
| Q21 | mx-cpu | wrong | 6.25e-02 | top-N metric differs, max relative error 6.25e-02 / repeat run WRONG |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 6.9e-06, mx-cpu 6.2e-02, mx-auto 6.2e-02.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.4, max 18.7 over 60 runs; load average at report time: 12.2 10.3 7.6), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `2c714ae`. Last invocation took 7 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

