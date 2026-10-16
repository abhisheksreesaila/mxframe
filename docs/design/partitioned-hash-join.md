# Design: partitioned parallel hash join, join + aggregate fusion (F-046a)

Status: **proposed, 2026-10-15** · Ticket: F-046a and F-046's "remaining" line · Builds on F-032 (M2d star
scan, M2e semi partners), F-043 (runtime join filter, shared-output group-by), F-046b (radix-partitioned
count distinct), F-046d (dictionary codes through joins)

General fixes only: every choice below is made from row counts, key ranges, cache sizes and the consumer
of the join, never from a query's identity.

## 1. How joins run today

### The paths

| path | where | when | build | probe | output |
|---|---|---|---|---|---|
| star scan (M2d) | `join_engine._execute_star` -> `native_exec.try_filter` -> `mx_filter` v2 (`executor_cpu.mojo`) | a tree of inner joins with a Filter*(Scan) leaf, `device="cpu"`, native on | each partner (every leaf but the largest, a subtree joined first, recursively as a nested star) indexed by `_build_map`: 1-2 integer key columns normalized to one int64; **direct array** (row+1 per key) when range <= 2^28 and <= max(8n, 2^20), with a presence bitmap when keys are sparse (n*8 < range); else a **hash table in 64 partitions** (top 6 hash bits), linear probing, load <= 0.5, all built in parallel | the fact table's morsels (64k rows), predicate on the bytecode interpreter per 1024-row block, then each lookup column-wise over the block's candidates (most selective partner first); one random access per candidate per partner, no prefetch | pass 1 records kept rows + partner rows per morsel, pass 2 writes every needed column of every joined row into one arena (strings decoded, or int32 codes of a unified dictionary, F-046d) |
| semi partner (M2e) | same call, `Lookup(semi=True)` | partner is `Distinct(Project(keys)(Filter*(Scan)))` | presence only, repeats allowed; runtime filter by the fact's keys first when the partner has > 2x the fact's rows (F-043) | as above | no partner columns |
| leaf-by-leaf | `_join_indices` -> `_join_pairs` -> `encode_keys` + `_pairs_prefiltered` -> `inner_pairs` -> `CustomOpsCompiler._hash_join_mojo_cpu` -> `kernels_aot.mojo` `join_count` + `join_scatter` | star declined (`E_DUP`, a partner key that is not 1-2 integer columns, a left join below, non-CPU device, `MXFRAME_STAR=0`) | keys made **dense int32** first: small int ranges as they are, otherwise **Arrow `dictionary_encode` over both sides** (single-threaded hash), composites packed or `np.unique(axis=0)`; then a direct CSR index (`_build_index`) **built twice**, once in `join_count`, once in `join_scatter`; offsets by NumPy `cumsum` (single-threaded); optional bitmap pre-filter (`joinsemi_cpu.mojo`) | parallel morsels over the direct index | int32 (probe, build) index pairs; index vectors composed with NumPy `_take`; residual conditions in Arrow; final gather `mx_gather` |
| left join | `_execute_left` | `how="left"` (a barrier: both sides materialized) | `encode_keys` + `join_count_left` / `join_scatter_left` | as above | Arrow `take` with null indices |
| semi / anti | `_execute_filtering` | `how="semi" / "anti"` | `encode_keys` + `has_partner`: NumPy `mark[rk] = True` (single-threaded random writes over the whole right side) or `np.isin` | NumPy | `mx_gather` of the kept left rows |

Single-threaded today: Arrow `dictionary_encode` of wide / string / composite keys, NumPy `cumsum` of match
counts, `has_partner` / `np.isin`, Arrow residual predicates and `_take_rows` after the star, index
composition below 4M rows, and every Python step between native calls.

Materialized today: a star scan writes every joined row of every needed column before the aggregate
reads it again (`materialize` replaces the Join by a `Scan` of its result, then `try_execute` runs
`mx_execute` on that table). Nested partners (Q3's orders x customer, Q8's orders -> customer -> nation)
are written too, then indexed again.

### Which query takes which path (SF1 trace, `MXFRAME_JOIN_TRACE=1`, machine under load ~21, so paths only)

| query | path | detail |
|---|---|---|
| Q3 | nested star, then star | orders(1.5M) x customer(30k) -> 147k rows written; lineitem(6M) x that 147k partner (hashed: orderkey range 6M > 8 x 147k) -> 30.5k rows; then `mx_execute` |
| Q5 | three stars | customer x nation; orders x that; lineitem(6M) x orders-partner(46k, hashed) x supplier(10k, direct); `c_nationkey = s_nationkey` on the joined rows in Arrow |
| Q8 | three stars | lineitem x part(1.4k, direct + bitmap) x orders-partner(91k, hashed) x supplier |
| Q11 | star | partsupp(800k) x filtered supplier(396), then three `compute` calls |
| Q12 | star | lineitem(6M) x **all of orders (1.5M; direct array over 6M keys, 24 MB; 240 MB at SF10)** -> 31k rows |
| Q17 | two stars | lineitem x part(204, bitmap); the 6k-row result x its own per-part average |
| Q22 | Python semi/anti | `encode_keys` + NumPy `has_partner` over all of orders' `o_custkey` |
| random suite | star | all 16 SF10 and 30 SF1 join-shape queries: `filter[,filter...],_call`; none leaf-by-leaf |

So the partitioned hash join this ticket asks for is not, by itself, what TPC-H or the random suite run
through today; the losses are in the star scan's probes and builds and in writing joined rows. The design
therefore adds the new hash table under **both** the star scan and the leaf-by-leaf fallback, and adds
the aggregate sink.

## 2. Why the listed queries lose to DuckDB (hypotheses until slice 0 measures them)

SF10 engine-only, MXFrame / DuckDB ms (`docs/bench/m2s-engine*.md`): Q3 124/86 (shuffled 169/84), Q5 113/75
(157/89), Q8 105/61, Q11 24/16, Q12 118/82 (201/79), Q17 65/28, Q22 94/42; Q9 shuffled 551/232. Random
SF10 join shape 0.96x geo. mean, worst q23 0.57x, q33 0.65x; SF1 q3 0.32x, q23 0.58x, q62 0.68x.

1. **Joined rows are written, then read again (all random joins; Q5, Q8, Q11, Q12, Q3).** Random q3 (SF1:
   6M lineitem rows x 2 partners, no filter, GROUP BY s_nationkey, p_size) writes ~6M rows x 4 columns
   (~100 MB) and aggregates them in a second call: 44 ms vs DuckDB 14. DuckDB pipelines the probe into its
   aggregate. Confirm: per-call ms (`native_exec.last_timing()`, the trace).
2. **Probes into large hashed partner maps miss the cache.** A hashed map scrambles key order; there is no
   presence bitmap on hashed maps (only on direct ones), so a miss walks to an empty slot. Q5 probes every
   one of 60M lineitem rows into the orders-partner map (~460k keys, ~12 MB); Q3's ~1.47M-key orders partner
   (~35 MB) exceeds one CCD's 32 MB L3. Shuffled rows make direct maps random too (Q12's 240 MB array:
   201 vs 118 ms stored). Confirm: star-call ms stored vs shuffled; `perf stat -e cache-misses`.
3. **Builds larger than the probe (Q12).** Q12 builds a direct map over all 15M orders (calloc of 240 MB, a
   put pass, a duplicate-check pass) for ~310k surviving lineitem rows. Confirm: time `_build_map` alone.
4. **Cross-partner conditions in Arrow (Q5, Q7).** `post_res` runs `comp._eval_predicate` plus
   `_take_rows` on the joined rows.
5. **Python semi / anti (Q22).** `has_partner` writes a 1.5M-entry bool mark with 15M random NumPy stores on
   one thread, after an int32 cast of 15M keys.
6. **Many native calls with Python between them (Q11, Q17, nested stars).** Q17's 65 vs 28 ms is
   **unknown, profile first.**
7. **SELECT DISTINCT (0.57-0.99x).** Filter writes the surviving rows, `mx_distinct` hashes every row, then a
   gather; multi-chunk columns are combined (copied) first.

## 3. Design

### 3.1 One hash table for every join (`kernels_aot/hashjoin_cpu.mojo`, linked into `libmxexec_cpu.so`)

**Keys.** A build key is normalized once per row:

| key | normalized form | equality |
|---|---|---|
| 1 int column (int32 / int64 / date32), or 2 whose packed range < 2^62 | int64 `(v1 - lo1) * card2 + (v2 - lo2)` (as `_build_map` does) | the int64 |
| dictionary strings | int32 code of one unified dictionary (F-046d; across the two sides a translation map built once per dictionary) | the code |
| 3+ columns, packed overflow, plain utf8 | 64-bit hash of the tuple + build row id | hash tag, then column-wise compare (bytes for utf8) |
| float keys | declined | — |

Nulls never match: dropped by inner / semi, unmatched for left / anti. int64 keys past 2^53 are exact.

**Small ranges stay direct.** Today's range rule; direct CSR `start[range+1]`, `rows[n_build]` (count, scan,
fill), presence bitmap when n*8 < range. Everything else uses the partitioned table.

**Partition count from the cache.** L2 / L3 read once from `/sys/devices/system/cpu/cpu0/cache/index{2,3}/size`
(defaults 512 KB / 32 MB), passed in the plan:

    table_bytes = 2 * n_build * S + 4 * n_build (CSR rows) + payload bytes carried
    P = clamp(pow2_ceil(table_bytes / (L2 / 2)), 1, 2^14)
    radix passes = 1 if P <= 256 else 2

**Build.** (1) histogram per (block, partition); (2) exclusive prefix sum over (partition, block): stable,
no atomics; (3) scatter `(key, build row)` (+ payload when the probe is partitioned); (4) per partition:
open addressing, linear probing; count, scan to CSR starts, fill row ids in build order (HeavyDB's
one-to-many layout, deterministic, same on CPU and GPU); all counts 1 -> unique flag, head holds the row;
(5) optional presence bitmap over the normalized range when range / 8 <= L3 / 4.

**Probe modes.**
- *Streaming* (default): per 1024-row block compute keys and hashes, prefetch slots (group prefetching), resolve.
  Probe order kept.
- *Radix-partitioned* (table > L3 / 2, probe rows >= 4 x build rows, order-insensitive consumer, memory budget
  allows): partition the probe side's survivors with the same hash bits, probe each L2-resident partition.
  Skewed partitions (> 8x mean) shared by several workers.
- *Mark* (semi / anti / build-side outer when the build is the small side): byte flag per build row.

**Output size first.** Pass 1 counts matches per morsel in int64; too big -> `E_OUTPUT_TOO_BIG` -> MemoryError.

**Sinks** (comptime-parameterized): index pairs, the `mx_filter` arena, the aggregate sink (3.2), mark flags,
the distinct sink (3.4).

### 3.2 Join + aggregate fusion (`mx_execute` version 3)

Shape: `Aggregate([Filter | Project]* (inner-join tree))` that `_execute_star` would take and whose aggregate
`try_execute` would take on the joined columns.

- **Python** (`join_engine.try_fused_aggregate`, called from `CustomOpsCompiler` before
  `native_exec.try_execute`): build partners as `_execute_star` does, but compile `prog_pre` (the fact leaf's
  predicates) and `prog_agg` over a **virtual joined schema** (fact + partner columns; cross-partner residuals
  and filters above the join as its predicate; the aggregates and derived keys as its body). Any decline ->
  today's path.
- **Mojo**: per 1024-row block: `prog_pre` -> candidates; lookups -> partner rows; copy the columns `prog_agg`
  reads into an L1-sized block scratch; run `prog_agg`; encode group keys; accumulate into per-worker
  partials; merge as `pipeline_keyed` does. **No joined row is written.**
- **Group keys**: partner key ranges come from the (small) partner. New key kind `K_PROW`: group keys all
  columns of one partner including its unique join key -> group id = partner row. Partials over 128 MB use
  F-043's shared-output accumulation; hashed composite keys over joined rows decline in v1.

### 3.3 Where it plugs in

    custom_ops (Aggregate over joins) --try--> join_engine.try_fused_aggregate --> mx_execute v3
            | declined
            v
    join_engine.materialize -> execute_join
        inner:  _execute_star --> mx_filter v3 (repeats allowed, bitmap on hashed maps, prefetch,
            |                                  runtime join filter for oversized partners)
            | declined
            v
        _join_indices -> native_exec.try_hash_join(..., "inner") --> mx_hash_join (pairs sink)
        left / semi / anti: native_exec.try_hash_join(..., how) --> mx_hash_join (pairs or mark sink)
            | declined
            v
        today's encode_keys + join_count / join_scatter / NumPy path

Overrides for A/B: `MXFRAME_RADIX_JOIN=0|1|auto`, `MXFRAME_FUSED_JOIN_AGG=0`, `MXFRAME_STAR=0`.
Runtime join filter generalized from F-043: a partner whose base rows exceed 2x the fact's filtered rows is
filtered by a bitmap of the fact's surviving keys before its map is built (skipped when the fact keeps over
half its rows). Build side = fewer rows after filters.

### 3.4 SELECT DISTINCT on the same machinery

`DISTINCT k1..kn` = `GROUP BY k1..kn` with a hidden `MIN(row id)` (new `OP_ROWID`), then sort groups by it
(first-appearance order). Directly addressed keys fused with the filter; over a star, the 3.2 sink with no
user aggregates; wide keys keep `mx_distinct`.

### 3.5 GPU (later)

The count / scan / fill CSR build and the partition function carry over unchanged; P from shared memory.
F-056 step 3 is the same radix partitioning with P from device memory.

## 4. Slices (measurements on a quiet machine: load < 2 for 5 min, ps samples logged)

- **Slice 0: profile.** Trace lines for `_build_map`, pass 1 / pass 2, residuals, `_execute_filtering`; SF10
  stored + shuffled Q3, Q5, Q8, Q11, Q12, Q17, Q22 and random q3, q23, q33, q62; confirm or drop each
  hypothesis in section 2 (Q17 especially).
- **Slice 1: `mx_hash_join`, the native fallback** (pairs and mark sinks; `_join_pairs`, `_execute_left`,
  `_execute_filtering`). Tests vs DuckDB, asserting the native path ran: inner / left / semi / anti x key kinds
  (int32, int64 past 2^53, date32, 2-col, 3-col, dictionary with different dictionaries per side, utf8) x
  duplicates x null keys x empty sides x a hot key x multi-chunk; the F-016 many-to-many case raises
  MemoryError. Measure with `MXFRAME_STAR=0`, Q22, a synthetic suite.
- **Slice 2: star lookups on the new table** (repeats allowed, bitmap on hashed maps, group prefetch,
  generalized runtime join filter). Measure Q3, Q5, Q8, Q12 SF10 stored + shuffled.
- **Slice 3: join + aggregate fusion** (3.2, residuals in `prog_agg`, `K_PROW`). Tests: grouped / global /
  HAVING / ORDER BY+LIMIT over 1-3 partners, partner group keys, a cross-partner residual, CASE WHEN on a
  partner dictionary column, derived keys, declines equal; assert one native call and no `mx_filter`.
  Measure the random suite join shape and Q5, Q8, Q11, Q12.
- **Slice 4: radix-partitioned probe** behind `MXFRAME_RADIX_JOIN=auto`; A/B on shuffled SF10 Q3, Q9, Q12, Q18
  and a 100M x 10M synthetic join; `auto` never loses to the better of the two beyond noise.
- **Slice 5: DISTINCT through the keyed group-by** (3.4).

Every slice: `pixi run test`, `pixi run test-all`, the workflow's smoke scripts; no SF1 / SF10 TPC-H query
slower beyond noise.

## 5. Risks

- Memory: carried tuples double the probe's footprint; a budget check declines to streaming.
- Order: the radix path reorders output; used only below order-insensitive consumers.
- Skew and blow-up: work-sharing inside a partition; output counted before allocation.
- Silent fallback: every new test asserts the native path ran; a decline reason is logged per call.
- Bytecode over a virtual schema: dictionary literals on partner columns and `MAX_REGS` may decline more
  plans than expected; log decline counts on the 22 queries and the random suite first.
- Mojo: closure / `parallelize` ergonomics, prefetch intrinsics on the current Mojo version.
- Choosing badly: thresholds from cache sizes and counts, env overrides allow A/B; recheck on other caches.
