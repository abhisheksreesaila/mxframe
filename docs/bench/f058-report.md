# F-058 core-gate scoreboard: 2026-10-17

Ryzen 9 5900X (24 threads), 64 GB, RTX 3090 (GPU 0). Engine-only first runs: a fresh process per (engine, query), the data loaded into each engine's own in-memory form outside the clock, no warm-up. Every result is checked against DuckDB. The final pass started at load 0.21 after 3 quiet minutes (median load during the run 4, from the engines' own threads). Code: main after F-128, F-129 and F-130.

## Headline

| | DuckDB | Polars | mx `cpu` | mx `auto` |
|---|---:|---:|---:|---:|
| TPC-H SF10, sum of 22 (21 for Polars: Q1 is wrong) | 4.83 s | 11.58 s | **2.67 s** | 2.75 s |
| TPC-H SF10 shuffled rows | 5.32 s | 12.80 s | **4.24 s** | 4.29 s |
| TPC-H SF1 | 618 ms | 1,307 ms | **445 ms** | 482 ms |
| TPC-H SF1 shuffled rows | 615 ms | 1,282 ms | **540 ms** | 585 ms |

| random queries, `auto` (geo. mean DuckDB / MXFrame; > 1 = MXFrame faster) | MXFrame faster than DuckDB | ratio |
|---|---:|---:|
| SF1, 100 queries | 49/100 | 0.99x (`cpu`: 55/100, 1.02x) |
| SF1 shuffled, 100 | 78/100 | 1.35x |
| SF10, 50 | 33/50 | 1.38x |

Correctness: every MXFrame result equals DuckDB (88 TPC-H, 300 random). Polars is wrong on SF10 Q1 (5.5%) and random q69.

Files: `f058-final-*.md/.csv`, `random-*-f058-final.*`, `random-sf1-f058b-*` (SF1 cpu and auto after F-130).

## The gate is not met

The gate is "each device beats DuckDB on at least 90% of queries". `auto` beats DuckDB on 10/22 TPC-H queries at SF10 (8/22 shuffled, 8/22 at SF1) and on 49-78% of random queries. MXFrame is faster in total at every scale, but DuckDB still wins many individual queries.

Where `auto` loses at SF10 (MXFrame / DuckDB, ms):

| query | ratio | DuckDB | mx `auto` | what it is |
|---|---:|---:|---:|---|
| Q17 | 2.34x | 30 | 70 | correlated avg per part (two stages) |
| Q11 | 2.27x | 16 | 36 | small partsupp aggregate + HAVING over a subquery |
| Q22 | 2.14x | 44 | 94 | substring of phone + NOT EXISTS, two computes |
| Q12 | 1.62x | 77 | 126 | lineitem x orders, CASE sums |
| Q20 | 1.38x | 49 | 67 | nested IN subqueries |
| Q3 | 1.28x | 84 | 107 | 3-join + top 10 |
| Q6 | 1.24x | 31 | 38 | filter + global sum |
| Q18 | 1.20x | 316 | 381 | group by orderkey (15M groups) + IN |
| Q10, Q16, Q8, Q15 | 1.03-1.15x | | | close |

The losses are multi-stage queries (Q17, Q11, Q22, Q20: several `compute()` calls with Python between them) and the lineitem x orders join family (Q12, Q3, Q18). They are the next work list.

## `auto` (F-128, F-130)

The first run of this scoreboard found `auto` at 11.7 s for SF10 against `cpu`'s 2.8 s. It priced the CPU from the old single-threaded kernels and sent five join queries to the GPU, which ran them 5-9x slower. With the native executor's own prices, `auto` is now within 3% of `cpu` in total at SF10 (2.75 vs 2.67 s), 8% at SF1. `mx-gpu` never beat `mx-cpu` on host data in these runs (0/22 at SF1 and SF10, cold process); where the GPU pays is data already resident on it (F-057), not measured here.

## Data bigger than memory (F-056, F-129)

SF10 with `MXFRAME_HOST_MEM_BUDGET=512M`, `cpu`, `scripts/bench_spill.py` (`f058-spill.jsonl` before F-129, `f058-spill-after.jsonl` after):

| query | in memory | streamed, 512 MB (before -> after F-129) | DuckDB, memory_limit 512 MB | peak anonymous RSS |
|---|---:|---:|---:|---:|
| J1 lineitem x orders, 8 groups | 1.3 s | 13.8 -> 6.6 s | 2.2 s | 0.75 GB |
| J2 lineitem x orders x customer | 0.8 s | 13.1 -> 4.7 s | 1.7 s | 0.5 GB |
| J3 15M groups, top 20 | 0.8 s | 16.2 -> 8.0 s | out of memory | 1.4 GB |
| J4 count distinct per key | 0.7 s | 28.3 -> 23.1 s | 6.0 s | 1.5 GB |
| J5 54M-row join to Parquet | 8.0 s | 15.1 -> 10.6 s | 3.1 s | 0.65 GB |

Every streamed result equals DuckDB's, on `cpu`, `gpu` and `auto` (Grace joins of 16-32 partition pairs). MXFrame finishes J3 where DuckDB runs out of memory at the same limit. It is 2-4x slower than DuckDB under the limit: the partition pass is single-threaded NumPy passes (read, filter, hash, gather, write, ~1 s each at SF10); overlapping them or a native partition kernel is the next step. The process alone holds ~400 MB, so peaks are 1.5-3x the budget.

## Regressions found and fixed on the way (F-130)

An A/B against the 0.6.0 release-run code found two small-query regressions that no test timed: MIN / MAX of a float expression over a join materialized the join twice (F-108; 26 -> 48 ms), and every first `compute()` imported the streaming and spill modules (F-056; +1.5-2 ms). Both are fixed and tested; random SF1 went from 0.60x to 0.99-1.02x of DuckDB.
