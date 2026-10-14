# 📊 MXFrame — TPC-H Benchmarks

> **Current headline (2026-10-13):** engine-only first runs, in [bench/m2p-engine.md](bench/m2p-engine.md)
> (shuffled rows: [bench/m2p-engine-shuffled.md](bench/m2p-engine-shuffled.md); random queries:
> [SF1](bench/random-sf1.md), [SF1 shuffled](bench/random-sf1-shuffled.md), [SF10](bench/random-sf10.md)) and [optimizations.md](optimizations.md):
> SF10 MXFrame 3.27 s, Polars 14.94 s, DuckDB 5.31 s (DuckDB is faster on 13 of the 22 queries and
> on shuffled data). Loading data is staging, outside the clock. Everything below is earlier and
> kept for history; DuckDB times before 2026-10-13 include its pyarrow import (about 70 ms).

---

## Method of the oldest runs (the 1M / 10M-row tables further down; history only)

> **Hardware:** NVIDIA RTX 3090 (sm_86) · AMD 12-core CPU · Mojo 26.4 (1.0.0b2) AOT kernels
> **Baselines:** Polars 1.29+ · Pandas 3.0 · MXFrame CPU path · MXFrame GPU path
> **Data:** TPC-H schema synthetic data (numpy RNG, fixed seed)
> **Methodology:** 1 warmup run + **median of 3 timed runs**, per engine. Join-cache cleared between runs for join-heavy queries (Q5/Q7/Q8/Q10/Q13/Q19) so each run includes the full join cost.

---

## Final user-felt scoreboard (2026-10-12, F-009)

**What is measured:** what a user feels. Every (engine, query) runs in a fresh Python process with the data starting as Parquet on disk (exact DECIMAL TPC-H data). The clock covers Parquet read + query + materialized result. There is no warm-up. Import time is reported separately. Every result is checked against DuckDB, and DuckDB's own answers are checked against the published TPC-H answers.
Machine: Ryzen 9 5900X (12 cores / 24 threads), 64 GB RAM, RTX 3090 (GPU 0; GPU 1 drives the display). The main run started on a quiet machine: load1 under 2 for 3 minutes, 0.13 at the start. The load average then rose to 3–4 at SF1 and 2–11 at SF10. I believe this came from the engines' own threads: in a check after an identical earlier run, the CPU was 99.5% idle straight afterwards. The generated reports still carry the script's automatic note "The machine was shared with other jobs during this run", which it adds whenever load1 is high; the data cannot tell the two apart. The three extra SF1 runs started straight after the SF10 run, with load1 still 2.8–4.4 from it and peaks of 5.1–7.2. That breaks the ticket's load < 2 rule for those three runs. The run used commit 5a2d331, which has all F-022 and F-009 changes.
Full tables: [SF1](bench/final-sf1.md) · [SF10](bench/final-sf10.md) (CSV next to each). The older tables further down are historical warm-run measurements on synthetic data. They are not comparable.

| sum of the 22 first runs, ms | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| SF1 | 6,394 | **2,014** | 3,336 | 4,357 | 9,512 | 4,479 |
| SF10 | 82,081 | 17,075 | **10,394** | 29,210 | 31,709 | 28,892 |
| import (median) | 234 | 71 | 38 | 125 | 126 | 126 |

Wins are counted on comparable results. A Polars result within float drift of DuckDB still counts (SF1 Q1/Q9, SF10 Q7/Q9). Polars' SF10 Q1 is off by 5.5%, so it does not count.

| mx-auto beats | pandas | Polars | DuckDB |
|---|---:|---:|---:|
| SF1 | 14/22 | 1/22 (Q1) | 7/22 |
| SF10 | 19/22 | 2/21 (Q7, Q13) | 0/22 |

**Correctness:** all 132 mx results (22 queries × cpu/gpu/auto × SF1/SF10) match DuckDB.

**`device="auto"`:** it lands within 10% of the better of cpu and gpu on **22/22 queries at SF10**. At SF1 the queries take 50–400 ms and one fresh-process run varies by about ±10%, so the check is judged on the median of 4 runs (this run + [3 more](bench/final-sf1-extra1.md)). That gives **21/22**: the one miss is Q22, 60 vs 54 ms. Auto is 0.98–1.09x the better device on the other 21, and 2.2% slower than CPU summed over the 22. Single runs gave 18, 21, 20 and 17/22, so only 2 of the 4 single runs reach 20. Which queries miss changes from run to run (Q3, Q4, Q6, Q12–Q15, Q22); only Q22 misses on the median. The GPU path is faster than the CPU path from a cold process on SF10 Q5, Q7, Q9, Q18 and Q21. Auto keeps pace with the GPU on Q9 (2,503 vs 2,453 ms) and Q18 (2,640 vs 2,563 ms); on Q21 it stays on the CPU. At SF1 the GPU is never faster: CUDA start-up and copies cost more than the query. Before F-009, auto missed 10 of 22 at SF1, on fixed per-query overhead of 10–40 ms. Three changes removed it: a cheaper group-cardinality sample, lazy imports in `calibration`, and starting background GPU work only after a 100 ms idle gap.

**Co-processing:** this needs data already resident on the GPU, so a cold first run cannot show it. In one session it runs SF10 Q1's fused stage in 25 ms split across CPU+GPU, against 57 ms CPU-only and 39 ms GPU-resident. Repeat queries on resident data run ~1.35x faster than CPU-only ([how it works](auto-mode.md)).

### Honest losses, and why

- **Parquet decode.** PyArrow's decoder is the floor for every mx query. Three SF10 lineitem columns take ~225 ms, against Polars' ~77 ms scan+filter. A native Mojo reader is ticket **F-021**. This section used to say decode was most of the loss on Q3, Q10, Q12, Q14 and Q19. The [engine-only run](#engine-only-scoreboard-2026-10-12-f-024) shows that holds only in part. At SF10, mx-cpu is still 2.8x–4.9x behind Polars on Q3, Q10 and Q14 with the read out of the clock, so most of their gap is the engine. At SF1 the read is most of the gap on Q10 and Q14. Q12 and Q19 run slower engine-only than user-felt, so that run can't split their gap.
- **Joins and distinct at SF10.** DuckDB wins every SF10 query; mx-auto is 1.1x–6.8x slower (median 2.6x). The worst cases are Q4 (6.8x), Q18 (4.8x), Q17 (4.0x), Q8 (3.8x), and Q12 and Q3 (3.7x). Join probe placement on the GPU is **F-007**.
- **Q21: 8.3 s against Polars' 4.1 s (SF10).** F-022 cut the (orderkey, suppkey) distincts from 1.3–1.8 s to 0.33 s, but the query still materializes a 60M-row distinct table. Planning `distinct → groupby → count` as one n_unique is **F-023**.
- **Small queries (SF1).** Polars is faster than mx-auto on 21 of 22 queries. mx pays ~126 ms of import and per-query Python planning that Polars does not.
- **GPU first run.** `device="gpu"` pays CUDA start-up (~250 ms) in every fresh process, so it never wins at SF1. `auto` avoids this by keeping small or cold work on the CPU.

### Reproduce

```bash
pixi run python scripts/bench_user_felt.py --sf 1 --engines pandas,polars,duckdb,mx-cpu,mx-auto,mx-gpu --timeout 120 --gpu 0 --name final-sf1
pixi run python scripts/bench_user_felt.py --sf 10 --engines polars,duckdb,mx-cpu,mx-auto,mx-gpu,pandas --timeout 300 --gpu 0 --name final-sf10
```

Check `uptime` (load < 2) and `nvidia-smi` (no other GPU users) first.

## Engine-only scoreboard (2026-10-12, F-024)

**What is measured:** the query engine alone. Same harness, data and machine as the user-felt run above, with one change: each engine first loads the columns its query uses into its own in-memory form, outside the clock. pandas and Polars get DataFrames, DuckDB gets in-memory tables, and MXFrame gets Arrow tables under a `LazyFrame`. The clock then covers query + materialized result only. It is still a fresh process per (engine, query) with no warm-up run. This run separates engine speed from file reading. (Since 2026-10-13 engine-only is the headline measure and user-felt is secondary; the current numbers are in [bench/m2l-engine.md](bench/m2l-engine.md).) One thing favours MXFrame and one works against it; the reports say so. The CPU warm-up thread that `import mxframe` starts finishes during the untimed load. Against that, MXFrame still combines Arrow chunks, and on the GPU uploads them, inside the clock.
Run started 09:15 on a quiet machine (load1 under 2 for 3 minutes). As in the user-felt run, the engines' own threads raised load1 to a median of 3.0 at SF1 and 6.7 at SF10. The run used commit 302cc70; the engine code is the same as the user-felt run's.
Full tables: [SF1](bench/engine-sf1.md) · [SF10](bench/engine-sf10.md).

| sum of first runs, ms | pandas | Polars | DuckDB | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|---:|---:|---:|
| SF1 engine-only (22 queries) | 4,499 | **1,391** | 2,299 | 2,524 | 8,151 | 2,777 |
| SF1 user-felt (same 22) | 6,394 | **2,014** | 3,336 | 4,357 | 9,512 | 4,479 |
| SF10 engine-only (21 queries) | 66,199 | 12,146 | **6,716** | 22,512 | 25,226 | 22,511 |
| SF10 user-felt (same 21) | 74,295 | 14,910 | **10,060** | 28,555 | 30,732 | 28,225 |
| SF10 load, outside the clock (22) | 18,813 | 6,696 | 9,848 | 13,448 | 13,299 | 13,345 |

SF10 is summed over the 21 queries with a valid result from every engine (Polars' Q1 is wrong in both runs).

| mx-auto beats | pandas | Polars | DuckDB |
|---|---:|---:|---:|
| SF1 engine-only | 12/22 | 1/22 (Q1) | 12/22 |
| SF10 engine-only | 14/22 | 4/21 (Q6, Q7, Q13, Q22) | 3/22 (Q1, Q6, Q13) |

**Correctness:** all 132 mx results match DuckDB.

**What it says:**

- **About half of mx-auto's gap to Polars is the engine, and half is the Parquet read.** Q12 and Q19 are left out here. At SF10 they run slower engine-only than user-felt (next bullets), which would inflate the engine side. At SF1 they don't, but they are left out at both scales for consistency. Over the other queries, mx-auto against Polars, summed, is 2.09x user-felt and 1.78x engine-only at SF1 (20 queries), and 1.91x against 1.60x at SF10 (19 queries). The engine is about 51% of the gap in milliseconds at SF1 and 54% at SF10. So a faster reader (F-021) is worth about as much as engine work. With Q12 and Q19 included, the sums read 2.22x/2.00x at SF1 (engine about 56% of the gap) and 1.89x/1.85x at SF10.
- **The read does cost MXFrame more than Polars.** Loading the same columns takes 13.4 s for mx at SF10 against 6.7 s for Polars (2.0x) and 9.8 s for DuckDB. That is PyArrow decoding against Polars' own reader, with no filter pushdown on either side.
- **Q12 and Q19 are slower on in-memory tables than through `scan_parquet`.** SF10 mx-cpu Q12 takes 1,827 ms engine-only against 954 ms user-felt, read included; Q19 takes 2,271 ms against 654 ms. The repeat runs are just as slow, so this is not first-run cost. The cause is not profiled yet. One candidate: in a Parquet scan MXFrame pushes Q12's `l_receiptdate` range and Q19's conditions into the reader, so the engine sees far fewer rows, while engine-only it gets all 60M rows, and mx's Q12 plan joins before it filters. This is ticket **F-025**. It matters for anyone who hands MXFrame an Arrow table rather than a Parquet path.
- **Where the engine is already fast.** Q1's fused aggregate runs in 145 ms at SF10, against 2,879 ms for Polars (which also accumulates in float32 and is 5.5% off) and 223 ms for DuckDB. mx also beats Polars on Q6, Q7, Q13 and Q22 at SF10.
- **The joins are still the big engine loss.** At SF10 engine-only, mx-auto stays 2.6x–5.0x behind Polars on Q2, Q3, Q8, Q10, Q16, Q17 and Q20. That is **F-007**.

```bash
pixi run python scripts/bench_user_felt.py --mode engine --sf 1 --engines pandas,polars,duckdb,mx-cpu,mx-auto,mx-gpu --timeout 120 --gpu 0 --name engine-sf1
pixi run python scripts/bench_user_felt.py --mode engine --sf 10 --engines polars,duckdb,mx-cpu,mx-auto,mx-gpu,pandas --timeout 300 --gpu 0 --name engine-sf10
```

---

## How the kernels dispatch

| Device | Path | Coverage |
|---|---|---|
| **CPU** | 100% ctypes → `libmxkernels_aot.so` | All 22 queries — group aggs, masked aggs, inner + left joins, gather, filter, sort, unique |
| **GPU** | ctypes → `libmxkernels_aot_gpu.so` | All grouped aggs + masked global aggs + hash joins + sort/top-k — every hot operator is AOT, zero session overhead |

No per-query JIT on CPU or GPU. All GPU operators dispatch directly to AOT ctypes.

---

## Mojo Kernel Catalogue

### Grouped aggregation (`kernels/`)

| Kernel | What it does |
|---|---|
| `group_sum.mojo` | Per-group sum — shared-memory privatisation up to 8 192 groups, global-atomic fallback |
| `group_min.mojo` | Per-group minimum — same shared-mem / atomic-fallback design |
| `group_max.mojo` | Per-group maximum |
| `group_count.mojo` | Per-group row count (output `float32` for consistent tensor shapes) |
| `group_composite.mojo` | Fused multi-key composite ID: `out[i] = k0[i]·s0 + k1[i]·s1 + …` in one memory pass |

### Global (ungrouped) aggregation

| Kernel | What it does |
|---|---|
| `masked_global_agg.mojo` | Fused masked reductions: `sum`, `sum(a×b)`, `min`, `max` — single GPU pass over only rows where `mask[i]=1` |

### Join kernels

| Kernel | What it does |
|---|---|
| `join_count.mojo` | Phase 1 inner hash join — count matching right rows per left key |
| `join_scatter.mojo` | Phase 2 inner hash join — emit `(left_idx, right_idx)` index pairs |
| `join_count_left.mojo` | Phase 1 left-outer join — count matches, record 1 for unmatched left rows |
| `join_scatter_left.mojo` | Phase 2 left-outer join — emit pairs; unmatched left rows get `right_idx = -1` |

### Sort, gather, and utility kernels

| Kernel | What it does |
|---|---|
| `sort_indices.mojo` | Bitonic sort — returns a permutation array for `ORDER BY` |
| `unique_mask.mojo` | Mark `out[i]=1` at the first row of each run in a sorted array; used by `DISTINCT` |
| `filter_gather.mojo` | Prefix-sum + scatter — compact a column by a boolean mask in two passes |

### AOT-only kernels (`kernels_aot/`)

Called via ctypes — no MAX Graph session, cold start ≈ 0 ms.

| Function | What it does |
|---|---|
| `group_encode_i32_gpu` | GPU open-addressing hash table: assigns dense `int32` group IDs to an integer key column |
| `group_count_f32_gpu` | Per-group row count via shared-memory privatisation (up to 8 192 groups) + global-atomic fallback |
| `masked_global_{min,max}_f32_gpu` | GPU block-reduction min/max for masked global aggregations |
| `gather_{f32,i32,i64}_gpu` | Coalesced GPU gather — reorders columns on-device after join/sort |
| `join_count_i32_gpu` | Phase 1 AOT GPU inner hash join — counts matching right rows per left key (v0.2.3) |
| `join_scatter_i32_gpu` | Phase 2 AOT GPU inner hash join — emits index pairs into pre-allocated output buffers (v0.2.3) |
| `sort_topk_i32_gpu` | Bitonic sort with top-k truncation — `ORDER BY … LIMIT k` fuses sort+limit, downloads only `k×4` bytes (v0.2.3) |
| `utf8_startswith_mask_gpu` | One thread per Arrow UTF-8 row compares its byte range with a literal prefix and emits an `int32` mask (v0.5.0) |
| `utf8_contains_mask_gpu` | One thread per Arrow UTF-8 row scans for a literal byte substring and emits an `int32` mask (v0.5.0) |

---

## 1 M rows — hot median of 3 runs *(v0.2.3, July 10 2026)*

> ⚠️ **Not a fair comparison; kept only as history.** MXFrame's result caches (joins, group encodings, input conversions) were cleared between queries but not between the warm-up and the timed runs, so the timed runs reused work done in the warm-up (hence Q9 in 0.6 ms); Polars and pandas recomputed every run. The data came from our own synthetic generator, with some tables smaller than the label. Use the user-felt and engine-only scoreboards above.

All times in **milliseconds · lower is better**. 🟢 = faster than Polars baseline; **bold** = MXFrame wins.

Q1, Q3, Q5–Q8, Q10, Q13, Q19 re-measured July 10 2026 (v0.2.3, join-cache cleared per run);
Q2/Q4/Q9/Q11/Q12/Q14–Q18/Q20–Q22 from July 7 2026 (v0.2.0) — same hardware and methodology.

> **v0.2.3 improvements:** Phase 4 AOT GPU join (cold-start 300 ms → 70 ms), Phase 5 GPU sort crash fixed,
> `group_count_f32_gpu` shared-memory path (Q1 GPU: 87 ms → 42 ms, **2× speedup**),
> `@always_inline` removed from Atomic struct (Q13 GPU cold compile: 2 min → 1.4 s).

| Query | Description | MX CPU | MX GPU | Polars | Pandas | CPU vs Polars | GPU vs Polars |
|---|---|---:|---:|---:|---:|---:|---:|
| Q1  | Filter + 8 aggregations       | 🟢 **10.7** | 42.1        | 33.3  | 97.4   | **3.1×**   | 0.8× |
| Q2  | Min-cost supplier             | 🟢 **6.6**  | 🟢 **15.6** | 16.3  | 9.5    | **2.5×**   | **1.0×** |
| Q3  | 3-table join + agg            | 🟢 **2.6**  | 🟢 **8.7**  | 23.3  | 17.7   | **9.0×**   | **2.7×** |
| Q4  | Order priority                | 15.0        | 192.5       | 15.0  | 29.9   | 1.0×       | 0.1× |
| Q5  | Multi-join + groupby          | 🟢 **17**   | 28          | 26    | 21.0   | **1.5×**   | 0.9× |
| Q6  | Masked global agg             | 🟢 **6.1**  | 9.2         | 9.3   | 6.7    | **1.5×**   | 1.0× |
| Q7  | Shipping volume               | 🟢 **13**   | 51          | 34    | 21.5   | **2.5×**   | 0.7× |
| Q8  | Market share                  | 🟢 **7**    | —           | 23    | 11     | **3.4×**   | — |
| Q9  | Product profit (6-table join) | 🟢 **0.6**  | 🟢 **6.6**  | 39.9  | 17.8   | **66.5×**  | **6.0×** |
| Q10 | Customer revenue              | 🟢 **14**   | 🟢 **21**   | 37.5  | 19     | **2.7×**   | **1.8×** |
| Q11 | Important stock               | 🟢 **0.5**  | 🟢 **2.7**  | 7.4   | 3.0    | **14.8×**  | **2.7×** |
| Q12 | 2-table join + agg            | 🟢 **0.5**  | 🟢 **3.4**  | 23.2  | 581.4  | **46.4×**  | **6.8×** |
| Q13 | Customer distribution         | 33          | 30.2        | 27.5  | 32     | 0.8×       | 0.9× |
| Q14 | Promo revenue                 | 🟢 **3.5**  | 9.0         | 8.1   | 241.8  | **2.3×**   | 0.9× |
| Q15 | Top-supplier revenue          | 🟢 **1.3**  | 11.8        | 9.9   | 6.4    | **7.6×**   | 0.8× |
| Q16 | Part/supplier relationships   | 🟢 **2.1**  | 🟢 **6.3**  | 16.9  | 6.8    | **8.0×**   | **2.7×** |
| Q17 | Small-qty order               | 🟢 **0.3**  | 🟢 **2.7**  | 7.9   | 4.7    | **26.3×**  | **2.9×** |
| Q18 | Large-volume customers        | 🟢 **4.2**  | 🟢 **22.3** | 33.4  | 16.8   | **7.9×**   | **1.5×** |
| Q19 | Discounted revenue            | 🟢 **18**   | —           | 19.6  | 20     | **1.1×**   | — |
| Q20 | Potential part promo          | 🟢 **4.3**  | 🟢 **5.6**  | 30.0  | 9.6    | **7.0×**   | **5.4×** |
| Q21 | Suppliers who kept (EXISTS)   | 🟢 **26.0** | 64.1        | 31.3  | 28.6   | **1.2×**   | 0.5× |
| Q22 | Global sales opportunity      | 🟢 **7.6**  | 🟢 **16.5** | 25.4  | 56.7   | **3.3×**   | **1.5×** |

**1 M summary (v0.2.3):** MX CPU beats Polars on **20/22** queries; MX GPU beats Polars on **12/22** queries.
Headline CPU wins: **Q9 67×**, **Q12 46×**, **Q17 26×**, **Q3 9×**, **Q8 3.4×**.
Headline GPU wins: **Q9 6.0×**, **Q12 6.8×**, **Q20 5.4×**, **Q3 2.7×**.
*(Q5/Q7/Q8/Q10/Q13/Q19 use fresh-join measurements; previous v0.2.0 numbers were warm join-cache hits.)*

---

## 10 M rows — hot median of 3 runs *(April 20 2026)*

> ⚠️ **Not a fair comparison; kept only as history.** MXFrame's result caches (joins, group encodings, input conversions) were cleared between queries but not between the warm-up and the timed runs, so the timed runs reused work done in the warm-up (hence Q9 in 0.6 ms); Polars and pandas recomputed every run. The data came from our own synthetic generator, with some tables smaller than the label. Use the user-felt and engine-only scoreboards above.

All times in **milliseconds · lower is better**. At 10 M rows the join-heavy queries show the biggest wins.

| Query | Description | MX CPU | MX GPU | Polars | Pandas | CPU vs Polars | GPU vs Polars |
|---|---|---:|---:|---:|---:|---:|---:|
| Q1  | Filter + 8 aggregations       | 🟢 **361.0** | 1190.7      | 946.5  | 1771.7  | **2.6×**   | 0.8× |
| Q2  | Min-cost supplier             | 🟢 **5.7**   | 🟢 **11.9** | 15.4   | 7.6     | **2.7×**   | **1.3×** |
| Q3  | 3-table join + agg            | 🟢 **57.7**  | 🟢 **67.2** | 72.2   | 581.8   | **1.3×**   | **1.1×** |
| Q4  | Order priority                | 301.5        | 492.2       | 113.8  | 807.6   | 0.4×       | 0.2× |
| Q5  | Multi-join + groupby          | 🟢 **2.8**   | 🟢 **13.8** | 60.7   | 332.9   | **21.7×**  | **4.4×** |
| Q6  | Masked global agg             | 399.6        | 523.2       | 92.0   | 246.4   | 0.2×       | 0.2× |
| Q7  | Shipping volume               | 🟢 **1.8**   | 🟢 **16.4** | 76.4   | 392.5   | **42.4×**  | **4.7×** |
| Q8  | Market share                  | 🟢 **1.3**   | 🟢 **3.8**  | 40.9   | 55.1    | **31.5×**  | **10.8×** |
| Q9  | Product profit (6-table join) | 🟢 **0.7**   | 🟢 **7.4**  | 89.7   | 431.3   | **128.1×** | **12.1×** |
| Q10 | Customer revenue              | 🟢 **39.2**  | 🟢 **45.0** | 131.1  | 216.2   | **3.3×**   | **2.9×** |
| Q11 | Important stock               | 🟢 **0.4**   | 🟢 **3.1**  | 6.6    | 2.5     | **16.5×**  | **2.1×** |
| Q12 | 2-table join + agg            | 🟢 **1.3**   | 🟢 **4.4**  | 116.4  | 6853.6  | **89.5×**  | **26.5×** |
| Q13 | Customer distribution         | 385.1        | 396.1       | 285.8  | 463.9   | 0.7×       | 0.7× |
| Q14 | Promo revenue                 | 🟢 **14.4**  | 🟢 **16.9** | 29.7   | 2719.2  | **2.1×**   | **1.8×** |
| Q15 | Top-supplier revenue          | 🟢 **2.7**   | 57.4        | 16.1   | 30.0    | **6.0×**   | 0.3× |
| Q16 | Part/supplier relationships   | 🟢 **2.7**   | 🟢 **6.5**  | 16.3   | 6.8     | **6.0×**   | **2.5×** |
| Q17 | Small-qty order               | 🟢 **0.6**   | 🟢 **1.5**  | 14.6   | 17.0    | **24.3×**  | **9.7×** |
| Q18 | Large-volume customers        | 🟢 **46.8**  | 69.4        | 63.4   | 242.8   | **1.4×**   | 0.9× |
| Q19 | Discounted revenue            | 🟢 **100.4** | 🟢 **97.9** | 112.9  | 234.8   | **1.1×**   | **1.2×** |
| Q20 | Potential part promo          | 🟢 **32.3**  | 🟢 **37.1** | 39.9   | 52.9    | **1.2×**   | **1.1×** |
| Q21 | Suppliers who kept            | 756.0        | 705.3       | 85.9   | 216.6   | 0.1×       | 0.1× |
| Q22 | Global sales opportunity      | 🟢 **54.1**  | 🟢 **59.2** | 132.8  | 1292.1  | **2.5×**   | **2.2×** |

**10 M summary:** MX CPU beats Polars on **18/22** queries; MX GPU beats Polars on **15/22**.
Join-heavy queries scale dramatically: **Q9 128×**, **Q12 89×**, **Q7 42×**, **Q8 31×**, **Q17 24×**, **Q5 22×** CPU vs Polars.
GPU join wins: **Q12 26.5×**, **Q9 12×**, **Q8 10.8×**, **Q17 9.7×**.
At 10 M, Q4/Q6/Q13/Q21 show Polars' strength on pure CPU-vectorised operations — Phase 6 targets.

---

## What the numbers mean

- **Correctness ✅** — all 22 queries return results that match Polars and Pandas output.
- **Coverage ✅** — every TPC-H query has a CPU AOT path; all group aggs, masked aggs, joins, and sort/top-k have GPU AOT paths; no MAX Graph session required for any hot-path operation.
- **No JIT tax** — CPU has no JIT at all. GPU dispatches directly to AOT ctypes for every hot operator.
- **Why GPU doesn't always win** — GPU wins scale with data size. At 1 M rows, PCIe upload cost can exceed GPU kernel savings. At 10 M rows, GPU crushes Polars on join-heavy queries (Q8/Q9/Q12). Where GPU loses at any scale: ops still routing through PyArrow fallback (Q4, Q13, Q21) — Phase 6 targets.
- **Where MXFrame loses** — Q4, Q6, Q13, Q21 fall back to PyArrow compute or do extra passes. Focus of next milestone.

### Why the benchmarks are honest

Methodology: **1 warmup + median of 3 timed runs**. Join-cache cleared per run for join-heavy queries.
The warmup primes three caches:

- **`_GROUP_ENCODE_CACHE`** — group-key dictionary encoding (PyArrow SIMD, once per unique `(table, keys)`)
- **`_INPUT_PREP_CACHE`** — column array preparation and filter masking
- **`_JOIN_RESULT_CACHE`** — join output `pa.Table` — cleared per-run for Q5/Q7/Q8/Q10/Q13/Q19

The 3 timed runs measure pure kernel dispatch + GPU execution — steady-state production cost.

---

## Current Limitations

MXFrame is **not 100% Mojo end-to-end** — a deliberate, documented trade-off.
The computational hot path (aggregation, joins, sort, gather) is fully Mojo-compiled.
Certain orchestration steps still use PyArrow or NumPy:

| Operation | Library used | Reason |
|---|---|---|
| **String predicate masks** — `startswith`, literal `contains` | Mojo AOT GPU (GPU sessions) | Kernels consume Arrow offsets and UTF-8 bytes directly and emit one mask value per row. CPU sessions retain PyArrow fallback. |
| **Remaining string predicates** — `isin`, `==` on string cols | PyArrow compute (SIMD C++) | Native GPU multi-pattern/equality kernels are the next string-pipeline step. |
| **String join-key encoding** | PyArrow shared `dictionary_encode` | Both join sides are encoded together into dense `int32` IDs, then dispatched to Mojo AOT CPU/GPU integer join kernels. Correct and cross-device, but dictionary construction still runs on CPU. |
| **String group-key encoding** — e.g. `l_returnflag`, `n_name` | PyArrow `dictionary_encode` | Phase 3 added GPU hash-table encoding for *integer* keys. String hashing on GPU requires a variable-length string type not yet in the kernel set. Result is cached after first call. |
| **Table assembly after joins** — string and date columns | `pa.Table.take()` | Numeric columns are gathered on GPU. String/date columns have no GPU representation yet — gathered on CPU via Arrow's `.take()`. |
| **Window functions** — `rank`, `dense_rank`, `lag`, `lead`, `cum_sum` | NumPy | Not required by TPC-H. Exist in API for completeness; GPU ports are Phase 7+. |
| **DISTINCT** | `pa.Table.group_by().aggregate([])` | Always applied post-aggregation (typically < 1 000 rows). Negligible cost. |
| **Plan-level orchestration** | Pure Python | Metadata only — no rows are touched here. |

---

## Roadmap

| Phase | What | Queries affected | Status |
|---|---|---|---|
| **Phase 4** | **Device-resident join pipeline** — AOT `join_count_i32_gpu` + `join_scatter_i32_gpu`; `gather_table` with single index upload | Q3, Q5, Q7, Q8, Q9, Q10, Q18, Q21 | ✅ **v0.2.3** |
| **Phase 5** | **GPU sort + top-k** — `sort_topk_i32_gpu` bitonic sort; `ORDER BY … LIMIT k` fuses sort+limit | Q3, Q5, Q8, Q9, Q20, Q21 | ✅ **v0.2.3** |
| **Phase 5.5** | **Shared string/composite join encoding** — jointly dictionary-encode both sides to dense `int32`; preserve null semantics; dispatch through existing AOT integer joins | String and mixed composite joins | ✅ **v0.4.0** |
| **Phase 3** | **GPU key encoding** — `group_encode_i32_gpu` for integer group keys; vectorised rank mapping | Q1–Q22 | ✅ **Partial** — string/float encoding is Phase 6 |
| **Phase 6A** | **Native GPU UTF-8 predicates** — Arrow offsets/data buffers feed AOT `startswith` and literal `contains` masks | Q9, Q13, Q14, Q16, Q20 | ✅ **v0.5.0** |
| **Phase 6B** | **Native GPU string/category encoding** — move shared dictionary construction for joins/group-by from Arrow CPU to a GPU UTF-8 hash table | Q1, Q4, Q5, Q7, Q8, Q9, Q12 | 🔭 Future |
| **Phase 7** | **Complete GPU string predicates** — multi-pattern `isin`, equality, and predicate/reduction fusion | Q4, Q12, Q16, Q19 | 🔭 Future |
| **Phase 8** | **GPU window functions** — `rank`, `dense_rank`, `lag`, `lead`, `cum_sum` | Non-TPC-H workloads | 🔭 Future |

### What Phases 4–5 delivered (v0.2.3)

**Phase 4 — AOT GPU join pipeline:**
- GPU inner joins bypass MAX Graph JIT entirely. Cold-start: ~300 ms → ~70 ms.
- `AOTKernelsGPU.hash_join()` chains `join_count_i32_gpu` + `join_scatter_i32_gpu` with `_cached_upload`.
- `AOTKernelsGPU.gather_table()` gathers all numeric columns with a single index H2D upload.

**Phase 5 — AOT GPU sort + top-k:**
- Fixes a crash: the previous GPU sort called `gather_f32` as a MAX Graph custom op that was never registered.
- `sort_topk_i32_gpu` AOT bitonic sort fuses `ORDER BY … LIMIT k` into one kernel. Downloads only `k × 4` bytes.
- `_apply_post_ops_custom` peek-ahead: a `Sort` immediately followed by a `Limit` passes `topk=k` to the sort kernel.

**Phase 3 vectorised rank (v0.2.3):**
- `_encode_sort_key` replaced a Python for-loop with a NumPy scatter: `rank_of[sorted_order] = np.arange(n_unique)`. ~1.5× speedup.

---

## Reproducing the Benchmark

## Current v0.5 status

The v0.2.3 tables above are retained as historical measurements. Since then, Q4, Q6, Q13, and Q21 received dedicated compact GPU kernels, and GPU UTF-8 support expanded to `startswith`, literal `contains`, equality/inequality, and packed non-null literal `isin`.

The memory-bounded July 11 run produced complete CPU, Polars, and Pandas matrices at both scales. MX CPU beat Polars on **22/22 at 1M** and **18/22 at 10M**. Among completed GPU paths, MX GPU beat Polars on **15/17 at 1M** and **14/17 at 10M**. Q8 GPU remains `N/A:JIT`; Q15, Q17, Q18, and Q20 GPU workers exceeded the bounded timeout. cuDF was unavailable, so RAPIDS remains `N/A` and no RAPIDS parity claim is made.

Two GPU string-pipeline gaps remain and are not claimed complete:

- shared UTF-8 dictionary construction for string join/group keys still runs through Arrow on CPU;
- string/date column gather after joins still uses Arrow `take`.

Run the synthetic 22-query benchmark in five-query groups. Every query and engine executes in its own process, so Arrow, Pandas, and CUDA allocations are released before the next engine starts, and one timeout cannot erase the other engines' results:

```bash
pixi run bench-all
pixi run bench-all --only-10m --rapids
```

`--batch-size` controls progress grouping and `--query-timeout` bounds each engine worker. The parent merges results into `scripts/bench_results_1M.csv` and `scripts/bench_results_10M.csv`. `--rapids` executes the existing Pandas query implementations through the official `cudf.pandas` accelerator in separate synchronized workers. When cuDF is unavailable, the final RAPIDS column is explicitly `N/A`.

```sh
# Step 1 — generate TPC-H data (requires: pip install duckdb)
#   SF=1  →  ~6M lineitem rows, ~200 MB Parquet
pixi run python3 scripts/gen_tpch_parquet.py --sf 1

# Step 2 — run all 22 queries against real data
pixi run python3 scripts/bench_real_tpch.py --data-dir data/tpch_sf1 --runs 3
```

### Scale factor guide

| `--sf` | lineitem rows | approx size | use case |
|---|---|---|---|
| 0.01 | ~60K | 2 MB | smoke test / CI |
| 0.1 | ~600K | 20 MB | local dev |
| 1 | ~6M | 200 MB | standard benchmark |
| 10 | ~60M | 2 GB | stress test |

### Legal note

Data is generated by **DuckDB's TPC-H extension** — a faithful port of the official TPC-H `dbgen` v3.0.1.
**TPC-H® is a trademark of the Transaction Processing Performance Council.**
These results are an *independent, non-certified* benchmark and may *not* be reported as "TPC-H results" without formal TPC certification.
Reference: <https://www.tpc.org/tpch/>