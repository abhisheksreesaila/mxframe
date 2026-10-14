# MXFrame Auto Mode — Research-Backed Spec

Sep 26, 2026 · @Abhishek Sreesaila

## Executive summary

MXFrame will add `device="auto"`: the engine picks CPU, GPU, or both for each query and each operator, instead of the user choosing. The design is built from published database and LLM-inference research, not invented from scratch.

**Why now.** MXFrame's own TPC-H runs show the problem. At 1M–10M rows the CPU path beats Polars on 21/22 and 18/22 queries; the GPU path on only 16/22 and 15/22. Copy and launch overhead often loses at small sizes, so a fixed "GPU mode" leaves speed on the table.

**The design in three layers.**

| Layer | Runs | Decides | Main source |
| --- | --- | --- | --- |
| Calibration | Once per machine, refined after each run | How fast this CPU, GPU, and CPU↔GPU link really are | HyPE/Ocelot (Breß et al.) |
| Planner + cost model | Every query | Is the GPU worth it? Which operators may run where? Where does data cross devices? | Li et al. 2025 (PVLDB), Crystal (SIGMOD 2020), Mordred (PVLDB 2022) |
| Morsel scheduler | During execution | Which worker (CPU thread or GPU) takes each small chunk of rows | Morsel-driven parallelism (SIGMOD 2014), HetExchange (PVLDB 2019), PowerInfer (SOSP 2024) |

Two helpers sit around the layers: a **residency tracker** (knows which columns already live on the GPU, so they aren't copied again) and **fragment fallback** (a GPU out-of-memory reruns that piece on the CPU instead of failing).

**Core rule the research supports.** Filter on the CPU, ship only surviving rows over PCIe, run joins, large group-bys and sorts on the GPU, bring back only small results.

**Plan.**

1. **Phase 1 — Safe auto (the 1.0 must-have):** calibration, whole-query CPU-vs-GPU choice, fragment fallback, `explain()`.
2. **Phase 2 — Per-operator placement:** device crossings, residency tracking, bloom-filter pushdown, memory pre-check, and copy/compute overlap so the CPU→GPU handoff is pipelined from day one.
3. **Phase 3 — Co-processing:** one morsel scheduler; the planner splits CPU threads into GPU feeders and full-pipeline workers.
4. **Phase 4 — Out-of-core:** data larger than GPU memory via streaming joins and spill (1.x, after 1.0).
5. **Phase 5 — 1.0 hardening:** stable API, published cross-vendor benchmarks, `auto` as default.

**What makes MXFrame different.** Every GPU engine we reviewed either picks one device per query, falls back to CPU only on failure, or is NVIDIA-only. MXFrame's Mojo kernels already run on NVIDIA, AMD, and Apple GPUs; adding true CPU+GPU co-processing on top is, to our knowledge, not offered by any open-source dataframe library today.

## Research foundation: what we borrow and from whom

Each idea in auto mode traces to a paper or system below. Rows are grouped by the layer they shape.

### Database systems

| Paper / system | Venue | What it showed | What MXFrame borrows |
| --- | --- | --- | --- |
| [Morsel-Driven Parallelism](https://dl.acm.org/doi/10.1145/2588555.2610507) — Leis, Boncz, Kemper, Neumann (HyPer) | SIGMOD 2014 | Split input into small chunks ("morsels"); workers pull them and run a whole pipeline until the next pipeline breaker. Parallelism is decided at runtime, not baked into the plan. | The scheduler model: CPU threads and GPU workers pull morsels from one queue, so faster workers naturally take more work. |
| [HetExchange](https://infoscience.epfl.ch/record/262531) — Chrysogelos et al. | PVLDB 12(5), 2019 | Wraps CPU/GPU heterogeneity in special "device-crossing" and "router" operators, so normal operators need no rewrite to run on both devices. | `DeviceCross` and `Router` nodes in the physical plan. |
| [Scaling GPU-Accelerated Databases beyond GPU Memory Size](https://www.vldb.org/pvldb/vol18/p4518-li.pdf) — Li et al. (Microsoft) | PVLDB 18(11), 2025 | PCIe is \~80× slower than GPU memory. CPU scans beat GPU scans when data must cross PCIe; GPU joins win even with the transfer. Bitvector filters cut join input sharply; compressed transfer was essential. | The core placement rule: filter on CPU, ship survivors (compressed), join/aggregate on GPU. Bloom-filter pushdown. Streaming and partitioned joins for out-of-core. |
| [Crystal](https://arxiv.org/abs/2003.01178) — Shanbhag, Madden, Yu | SIGMOD 2020 | GPU speedup for memory-bound operators ≈ the GPU/CPU memory-bandwidth ratio; full queries beat it through kernel fusion. Keeping the working set on the GPU beats "coprocessor" designs. | Expected-speedup priors for the cost model; fuse each pipeline into one kernel; keep hot data resident. |
| [Ocelot / HyPE](https://www.vldb.org/pvldb/vol7/p1609-bress.pdf) — Breß et al. | PVLDB 7, 2014 | Learned per-operator cost models from simple regressions over observed runtimes; vendor-portable via OpenCL; tracks where data lives to avoid copies. | Calibration + online refinement of per-operator costs; hardware-oblivious design. |
| Mordred — Yogatama, Gong, Yu | PVLDB 15(11), 2022 | A simple "run it where the data already is" heuristic beat a learned optimizer; adds semantic-aware GPU caching. | Residency is a first-class cost term; the residency tracker. |
| [Maximus](https://dl.acm.org/doi/10.1145/3725324) — ETH Zürich | SIGMOD 2025 | Per-operator device assignment with overlap of communication and compute (NVIDIA-only GPU side). | Confirms per-operator split execution is practical. |
| [Theseus](https://arxiv.org/abs/2508.05029) — Voltron Data | arXiv 2025 | Separate async executors for compute, pre-loading, and spilling; pooled fixed-size pinned host buffers. | Pinned staging-buffer pool; prefetch-ahead spill design. |
| [Sirius](https://vldb.org/cidrdb/papers/2026/p12-yogatama.pdf) — UW–Madison + NVIDIA | CIDR 2026 | GPU-native DuckDB front end; GPU as primary engine with CPU fallback. | Fallback pattern; benchmark target. |

### LLM inference systems

| System | Venue | Idea | What MXFrame borrows |
| --- | --- | --- | --- |
| [PowerInfer](https://arxiv.org/abs/2312.12456) — Song et al. | SOSP 2024 | "Hot" neurons live on GPU, "cold" ones computed on CPU, because moving cold data costs more than computing it; CPU and GPU executors pull from one operator queue. | Hot/cold column residency; shared ready-queue with per-device executors. |
| [FlexGen](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) — Sheng et al. | ICML 2023 | Chooses GPU/CPU/disk placement by solving a small linear program over a cost model; overlaps load, store, and compute. | LP-based sizing for out-of-core plans; copy/compute overlap pattern. |
| [llama.cpp offload](https://github.com/ggml-org/llama.cpp/discussions/18049) | Open source | User-level placement overrides (GPU layers, tensor split, regex overrides) and an automatic "fit" mode. | The hint API: placement overrides, `gpu_fraction`, fit-to-memory. |
| [Mirage MPK](https://arxiv.org/abs/2512.22219) | arXiv 2025 | Fuse a whole workload into one persistent "megakernel" to kill launch overhead. | Later-phase idea for sub-millisecond queries. |

### Hardware behaviour

| Study | What it showed | What MXFrame borrows |
| --- | --- | --- |
| [Data movement on Grace Hopper](https://arxiv.org/abs/2408.11556) | Coherent CPU–GPU memory is fast but not free; placement still matters. | Keep a transfer term in the cost model even on unified-memory machines. |

**Note on provenance.** The morsel-driven model was introduced from prior knowledge during design discussion and verified against the SIGMOD 2014 paper afterwards. The other rows come from the research pass behind this spec.

## Landscape: how others place work, and the gap

No engine we reviewed combines all three of: a dataframe API, CPU+GPU working on one query at once, and GPUs from more than one vendor.

| Engine | How it picks a device | CPU + GPU at the same time? | When GPU memory runs out | GPU vendors |
| --- | --- | --- | --- | --- |
| HeavyDB | Per query or session (`\cpu`, `\gpu`) | No — other queries can use the CPU meanwhile | Retries the query on CPU; no pre-flight memory estimate | NVIDIA (CUDA) |
| cudf.pandas | Each call tried on GPU, retried on CPU on failure | No | Managed memory with prefetch | NVIDIA |
| Polars GPU engine | Whole query on GPU; any unsupported op sends the whole query to CPU | No (hybrid listed as not yet built) | Streaming executor spills to host (since 26.06) | NVIDIA |
| Sirius (DuckDB on GPU) | GPU primary, CPU fallback | No | Spill listed as future work | NVIDIA |
| Theseus (Voltron) | GPU-native, distributed | Moves data while computing | Spills GPU → host → storage | NVIDIA |
| SQL Server / Fabric (TQP → CoddSpeed) | CPU scans, GPU joins/aggregates | Yes | Streaming + partitioning, per-partition CPU fallback | Research / product-internal |
| Maximus (ETH) | Per operator | Yes | Format conversion between Arrow and cuDF | NVIDIA |
| **MXFrame (target)** | **Per operator, cost-based, calibrated** | **Yes (Phase 3)** | **Fragment fallback, then spill (Phase 4)** | **NVIDIA, AMD, Apple** |

**Claim for the README (defensible version).** "To our knowledge, MXFrame is the first open-source dataframe library that plans CPU and GPU work together per operator and runs on NVIDIA, AMD, and Apple GPUs from one kernel source." Keep "to our knowledge": Maximus and CoddSpeed already do CPU+GPU co-processing, just not as a portable dataframe library.

Not verified against primary sources in this pass: BlazingSQL, Spark RAPIDS, PG-Strom, Kinetica, SQream, BrytlytDB. Check before citing them.

## Glossary

Use these words in code, tickets, and commit messages so humans and agents mean the same thing.

| Term | Meaning |
| --- | --- |
| Morsel | A fixed-size chunk of input rows (start at 1–4M rows or a few MB; tune per device). The unit of scheduling. |
| Pipeline | A run of operators that stream rows without materializing, e.g. scan → filter → project → partial aggregate. |
| Pipeline breaker | An operator that must see all input before emitting: hash-join build, final aggregate, sort. Pipelines end here. |
| Device | `cpu`, `gpu:<n>`, or `pyarrow` (the fallback path, treated as a CPU-only device). |
| Placement | The device (or set of eligible devices) the planner assigns to an operator. |
| Device crossing | A plan node where data moves between devices (`cpu→gpu`, `gpu→cpu`). The only place copies happen. |
| Router | A plan node that sends morsels of one pipeline to more than one device, then merges partial results. |
| Residency | Where a column chunk currently lives: GPU, pinned host, pageable host, or disk. |
| Cost model | Formula estimating an operator's time on each device, including the copy cost. |
| Calibration | Measured constants for this machine: link bandwidth, launch overhead, per-row costs per kernel. |
| Fragment fallback | Rerunning one failed fragment (e.g. GPU out-of-memory) on CPU instead of failing the query. |
| Cold run / hot run | Cold = data starts in host memory or on disk, transfers counted. Hot = data already resident on the GPU. |

## Architecture

The planner runs on every query and decides where operators may run; the scheduler decides, chunk by chunk, who actually runs them.

&#91;embedded content: auto-mode architecture · 3 layers + 2 helpers\]

The logical plan and optimizer already exist in MXFrame (`lazy_frame.py`, `optimizer.py`, `compiler.py`). Auto mode adds everything from the planner down.

### 1. Calibration store

- `mx.calibrate()` runs a microbenchmark grid once per machine: host→device and device→host bandwidth (pinned and pageable), kernel launch overhead, and per-row cost of each AOT kernel across row counts 10⁴–10⁸, key cardinalities, and selectivities.
- Results are saved per `(device name, driver version, MXFrame version)` in a small local file (e.g. `~/.mxframe/calibration.json`).
- After each run, observed operator times refine the stored constants (simple regression, as in HyPE). Once a shape has enough samples, observed times beat the model.
- No calibration file yet → use conservative built-in defaults and prefer CPU below a size threshold.

### 2. Planner + cost model

For each operator *o* on device *d*:

```latex
T_d(o) = \alpha_d(o) + n \cdot c_d(o) + \frac{B_{in}^{nonresident}}{BW_{h2d}} + \frac{B_{out}^{needed}}{BW_{d2h}}
```

- α = fixed overhead (launch, graph lookup); n = rows; c = measured per-row cost for this dtype, cardinality, and selectivity.
- The copy terms are zero for data already on that device (residency). When copies overlap compute (Phase 3), use max(copy, compute) instead of the sum.
- On unified-memory machines, the link bandwidth is the measured coherent bandwidth, not infinity. On Apple Silicon, Mojo does not yet offer zero-copy Metal buffers, so a copy cost still applies.

Planner outputs, in order:

1. **Go/no-go for the GPU.** If the best CPU-only plan beats the best GPU-using plan, stop: run on CPU.
2. **Eligible devices per operator.** PyArrow-fallback operators and ops with no GPU kernel are CPU-only.
3. **Device crossings.** Insert `DeviceCross` where data changes device, as late as possible after filters so fewer bytes cross.
4. **Routers.** For large decomposable aggregations (sum, count, min, max, mean as sum/count), insert a `Router` so both devices can take morsels (Phase 3).

Plans for repeated query shapes are cached, keyed by plan fingerprint plus coarse table-size buckets.

### 3. Morsel scheduler

- One ready-queue of morsels per pipeline. CPU threads and each GPU worker pull from it.
- GPU workers take larger morsels to amortize launches; CPU workers take smaller ones. Work-stealing corrects estimate errors automatically.
- Each GPU worker has one copy stream and one compute stream plus 2–3 pinned staging buffers, synchronized with events. While the GPU computes morsel k, morsel k+1 copies in and k−1 copies out.
- Bounded channels with backpressure between stages keep GPU memory use predictable.

CPU threads take one of two roles from the same queue: **feeders** hand filtered morsels to the GPU, and **full-pipeline workers** run the whole pipeline on the CPU. The planner sets the split per query (Phase 3); the one-at-a-time CPU→GPU handoff is never shipped.

### 4. Residency tracker

- Records where every column chunk lives (GPU, pinned host, pageable host, disk), with LRU eviction plus user pins.
- Feeds the cost model: resident data costs no copy.

### 5. Fragment fallback

- On GPU out-of-memory or an unsupported-kernel error, rerun only that fragment on CPU, log it, and mark the shape in the calibration store so the next plan avoids it.
- `strict=True` raises instead of falling back.

## Public API

`device="cpu"` and `device="gpu"` keep working exactly as today; `"auto"` is additive. It becomes the default only at 1.0.

```python
import mxframe as mx

# one-time, per machine (also runs lazily on first auto query if missing)
mx.calibrate()                     # quick (~seconds)
mx.calibrate(full=True)            # full grid (~minutes)

# run
lf.compute(device="auto")
lf.compute(device="auto", prefer="gpu")                           # bias, not force
lf.compute(device="auto", placement={"join": "gpu", "filter": "cpu"})  # per-operator override
lf.compute(device="auto", gpu_fraction=0.7)                       # fixed split for Router nodes (Phase 3)
lf.compute(device="auto", strict=True)                            # raise instead of fallback

# residency hints
lf.cache(device="gpu", columns=["l_orderkey", "l_extendedprice"])  # pin hot columns

# global config
mx.config(gpu_mem_fraction=0.8, pinned_pool="2GB", spill_dir="/nvme/mx", morsel_rows=2_000_000)

# explain
print(lf.explain(device="auto", costs=True))
res = lf.compute(device="auto", profile=True)
res.profile   # predicted vs actual per operator, bytes moved, fallbacks
```

**`explain()` output (sketch).** One line per operator: chosen device, estimated time per device, bytes that cross, and why.

```
Aggregate[sum(price) by region]   gpu   est cpu=41ms gpu=6ms   reason: high-cardinality groupby
  DeviceCross cpu→gpu              —     bytes=38MB (after filter, was 820MB)
  Filter[date >= 1995-01-01]       cpu   est cpu=9ms gpu=22ms(incl copy)  reason: data not resident
    Scan[lineitem]                 cpu
```

The exact return types and names should follow MXFrame's existing API style; the implementer should check `docs/api.md`.

## Phase 1 — Safe auto

**Goal:** `device="auto"` is never meaningfully slower than the better of `cpu` and `gpu` for the same query, and never fails where `cpu` would succeed.

**Scope — build:**

1. **Calibration module** (`calibration.py` or similar)
   - Measure H2D/D2H bandwidth with pinned and pageable host buffers, at sizes 1MB–1GB.
   - Measure empty-kernel launch overhead and MAX Graph cache-hit overhead.
   - For each AOT kernel (group sum/min/max, join build/probe, filter\_gather, unique\_mask, sort), time CPU and GPU at 4–6 row counts; fit `time = α + c·n` per kernel per device.
   - Persist to JSON keyed by device name, driver, MXFrame version. Version the schema.
2. **Cost estimator**
   - Walk the physical plan bottom-up, estimate rows per operator from input sizes and simple selectivity defaults (filters 0.3, joins by key uniqueness if known).
   - Compute total CPU-only time and total GPU time (GPU time includes every non-resident input copy and the final result copy back).
3. **Whole-query device choice**
   - Pick the lower estimate. Record the decision and both estimates.
   - Any operator that only runs on PyArrow → count its cost on CPU plus copies around it if the rest is on GPU.
4. **Fragment fallback**
   - Wrap GPU execution; on out-of-memory or kernel error, rerun on CPU from the last materialized input.
   - Log a warning once per query shape; remember the shape so the next call plans CPU.
5. **`explain(device="auto")`** printing the decision, both estimates, and bytes to be copied.
6. **`profile=True`** capturing actual per-operator times to feed calibration refinement.

**Out of scope:** per-operator placement, concurrent CPU+GPU, spill.

**Acceptance criteria:**

- [ ] On the existing TPC-H suite at 1M and 10M rows, `auto` is within 10% of `min(cpu, gpu)` on at least 20 of 22 queries, measured cold (data starts in host memory).
- [ ] With no GPU present, `auto` runs everything on CPU with zero errors.
- [ ] A forced GPU OOM (tiny `gpu_mem_fraction`) completes via fallback with a warning and correct results.
- [ ] Results under `auto` match `cpu` results exactly (or within float tolerance for sums) on the full test suite.
- [ ] Calibration quick mode finishes in under 30 seconds on the RTX 3090 reference box.
- [ ] `explain` shows the decision and estimates for every TPC-H query.

## Phase 2 — Per-operator placement

**Goal:** one query can use CPU for some operators and GPU for others, with copies only at planned device crossings.

**Scope — build:**

1. **Placement pass** on the physical plan: assign each operator a device using the cost model, then insert `DeviceCross` nodes. Start with these priors (from Li et al. 2025 and Crystal), then let calibration override:

   | Operator | Default | Why |
   | --- | --- | --- |
   | Scan, filter, project on host data | CPU | CPU memory bandwidth > PCIe; the GPU would wait on the copy |
   | Hash join (large probe side) | GPU | GPU joins win even including the transfer |
   | Group-by, high cardinality | GPU | Random-access hash tables favour GPU bandwidth |
   | Group-by, low cardinality | Either (cheapest) | Both are fast; residency decides |
   | Sort, top-k after GPU sort | GPU | Near bandwidth-ratio speedup |
   | String ops (`contains`, `startswith`) | CPU until GPU string kernels exist | No kernel yet |
   | PyArrow fallback ops | CPU only | Not a GPU path |
   | Tiny inputs (below calibrated threshold) | CPU | Launch + copy overhead dominates |
2. **Residency tracker:** a registry mapping column chunk → location, with LRU eviction under `gpu_mem_fraction`; `lf.cache(device="gpu")` pins. Resident columns cost zero copy in the model.
3. **Crossing minimization:** push `DeviceCross cpu→gpu` above filters and projections, so only surviving rows and needed columns move.
4. **Bloom-filter pushdown:** for a GPU join, build a bloom filter from the build-side keys, apply it in the CPU scan of the probe side. Only enable when the cost model says the filter pays for itself.
5. **Pre-flight memory check:** estimate peak GPU memory per fragment (inputs + hash tables + outputs). Over budget → plan it on CPU now; Phase 4 turns this into streaming instead.
6. **Storage layer investigation:** measure the copy cost of PyArrow → Mojo buffers. If significant, spike a Mojo-native Arrow layer (e.g. the community `marrow` project) so CPU kernels and device copies share one buffer type.

**Also in Phase 2: pipelined handoff.** Whenever a plan has a `DeviceCross cpu→gpu`, the GPU worker uses one copy stream and one compute stream with 2–3 pinned staging buffers, so morsel k+1 copies while morsel k computes and the CPU filters k+2. Without this, the CPU-filter → GPU-join split runs as a strict one-at-a-time handoff and can lose to plain `gpu` mode. The one-at-a-time version is never shipped.

**Acceptance criteria:**

- [ ] On TPC-H at 10M rows and at a scale where data > 25% of GPU memory, `auto` beats both `cpu` and `gpu` on at least 5 queries (mixed placement pays off somewhere).
- [ ] `explain` shows per-operator devices and bytes crossing each `DeviceCross`.
- [ ] Repeat queries on pinned columns show zero host→device bytes in `profile`.
- [ ] Pre-flight check prevents every OOM in the test suite that Phase 1 caught by fallback.

## Phase 3 — Co-processing (CPU and GPU at once)

**Goal:** one morsel scheduler keeps the GPU saturated and puts every spare CPU thread to work. Feeding the GPU and running whole pipelines on the CPU are not separate modes; the planner decides, per query, how many threads do each.

**Scope — build:**

1. **Morsel executor** (after Leis et al. 2014): split each pipeline's input into morsels; one shared ready-queue per pipeline; CPU worker threads and one worker per GPU pull from it.
2. **Two CPU worker roles, one queue:**
   - **Feeder:** runs the CPU part of the pipeline (scan, filter, project, bloom filter) and hands the survivors to the GPU worker.
   - **Full-pipeline:** takes raw morsels and runs filter, join and partial aggregate entirely on the CPU, using its own copy of any build-side hash table. Its morsels never cross PCIe.
3. **Thread split rule** (planner, per query, from calibration):

   ```
   feeders   = min(T, ceil(cpu_feed_time_per_morsel / gpu_time_per_morsel))
   full_pipe = T - feeders          # T = CPU threads available to the query
   ```
   - No GPU, GPU not worth it, or ops with no GPU kernel → `feeders = 0` (pure CPU).
   - CPU feed stage slower than the GPU even with all threads → `full_pipe = 0` (all feeders).
   - Full-pipeline workers only run when the pipeline's aggregate is decomposable (sum, count, min, max, mean as sum/count); otherwise leftovers become feeders.
4. **Runtime correction:** work-stealing. If filters are more selective than estimated, feeders finish early and grab more morsels; a feeder may switch to full-pipeline when the GPU's input backlog exceeds its buffer budget, and back when the backlog drains.
5. **Router + merge** (after HetExchange): every worker writes partial aggregates per group; the merge runs at the end on whichever device holds the larger share of partials (typically a few hundred to a few thousand rows).
6. **Morsel sizing:** GPU-bound morsels large enough to amortize launch (start at several MB, calibrate); full-pipeline CPU morsels smaller for balance.
7. **Pipeline fusion:** where possible, fuse filter → project → partial aggregate into one AOT kernel per pipeline (Crystal). An optimization, not a blocker.
8. **Backpressure:** bounded queues between stages so GPU memory in flight stays under budget.

**Acceptance criteria:**

- [ ] On a single-table scan + group-by at 100M+ rows (cold), `auto` beats both `cpu` and `gpu` by at least 15%.
- [ ] `profile` shows both device types busy during Router pipelines, and copy time overlapped with compute.
- [ ] Results are identical to `cpu` mode, including with 1, 2, and all CPU threads.
- [ ] Works on at least one NVIDIA, one AMD, and one Apple machine, or documents which backend lacks streams/events.

## Phase 4 — Out-of-core, and Phase 5 — 1.0

### Phase 4: data larger than GPU memory

**Goal:** queries whose working set exceeds GPU memory still use the GPU, instead of falling back to CPU wholesale.

1. **Streaming join** (Li et al. 2025): keep the build side's hash table resident on GPU; stream the probe side through in morsels.
2. **Partitioned join:** when the build side doesn't fit, hash-partition both sides and process partition pairs one at a time.
3. **Tiered spill:** GPU → pinned host pool → NVMe (`spill_dir`), with prefetch-ahead driven by the plan (Theseus).
4. **Compressed transfer:** send dictionary- or bit-packed columns over the link and decode on the GPU; Li et al. found this essential at scale.
5. **LP sizing (optional):** for very large jobs, choose chunk size and what stays resident by solving a small linear program over the cost model (FlexGen).

**Acceptance:** TPC-H at a scale factor 2× GPU memory completes on GPU-assisted plans, faster than `cpu` on the join-heavy queries.

### Phase 5: 1.0 hardening

- [ ] Hint API frozen and documented.
- [ ] GPU string kernels for `contains`, `startswith`, `isin` on dictionary-encoded columns.
- [ ] All 22 TPC-H queries run without PyArrow fallback (today Q4, Q6, Q13, Q21 fall back).
- [ ] Published cross-vendor benchmark report (NVIDIA, AMD, Apple), cold and hot runs.
- [ ] `device="auto"` becomes the default; `cpu` and `gpu` stay available.

## Roadmap: what 1.0 means, and what comes after

1.0 is a reliability release: anyone can `pip install` MXFrame, run it on NVIDIA, AMD, or Apple hardware (or CPU only), and `device="auto"` puts both CPU and GPU to work without tuning, the way llama.cpp does for models. Scale-up, scale-out, and the analytics layer come after.

&#91;embedded content: release roadmap · 4 stages, 1 gate\]

### 1.0 definition of done

- [ ] **Install:** `pip install mxframe` works with no extra steps on Linux and macOS (Apple Silicon), including a fresh Google Colab notebook. Confirm which Colab GPUs (e.g. T4) MAX supports; unsupported GPUs fall back to CPU cleanly.
- [ ] **Hardware:** CI runs the full test suite on CPU-only, NVIDIA, AMD, and Apple Silicon.
- [ ] **Auto mode:** Phases 1–3 shipped: calibration, per-operator placement with pipelined handoff, and the morsel scheduler with feeders and full-pipeline threads.
- [ ] **Coverage:** roughly 80–90% of everyday dataframe work: everything TPC-H and the H2O db-benchmark need, plus the most common Polars/pandas operations. Anything unsupported falls back or raises a clear error; it never crashes or returns wrong results.
- [ ] **Correctness:** results match Polars and DuckDB on the full suite; fuzz tests on random data and plans; fallback paths tested by forced failures.
- [ ] **Performance honesty:** published cold and hot benchmarks per vendor. Speed is documented per hardware class, not promised.
- [ ] **Phase 5 items:** frozen hint API, basic GPU string kernels, all 22 TPC-H queries without PyArrow fallback.

### After 1.0

| Release | Theme | What it adds | Notes |
| --- | --- | --- | --- |
| 1.x | Out-of-core | Phase 4: streaming and partitioned joins, tiered spill to host and NVMe, compressed transfer | Matters most on small-memory GPUs like Colab's |
| 2.x | Scale up | NUMA-aware queues (one per socket, steal across when idle); feeders pinned to the GPU's socket; multi-GPU with replicated build-side hash tables; then hash-partitioned multi-GPU joins over NVLink / xGMI; topology discovery in calibration | The morsel design was built for multi-socket machines (Leis et al. 2014), so the scheduler extends rather than changes |
| Later | Scale out | MXFrame as the per-node engine under Spark, Ray, or Dask, the way Velox (via Apache Gluten) and DataFusion (via Comet) plug into Spark | Compete on the node, not on the cluster framework |
| Later | Analytics layer | GPU "visual aggregation" operators (M4 downsampling, density binning), Arrow IPC to the browser, Flint chart specs compiled to ECharts, AI question → query + chart | Early test: an MXFrame backend for Microsoft's Data Formulator |

### Answering "why not just use Spark?"

Most analytical work fits on one node today: servers with 192+ cores, terabytes of RAM, and up to 8 GPUs. Two references to cite:

- McSherry, Isard, Murray. *Scalability! But at what COST?* HotOS 2015. A single-threaded laptop beat large clusters on graph workloads because clusters pay heavily for coordination and shuffle.
- Tigani. *Big Data is Dead.* MotherDuck, 2023. Most queries touch far less data than people assume.

Spark still wins when data exceeds one node, when jobs run for hours and need fault tolerance, or when data already lives across a cluster. That is why scale-out is "plug in as the node engine", not "replace Spark".

### Name decision (before 1.0)

Renaming is cheap now and expensive after 1.0. PyPI checked live on 2026-09-26:

| Candidate | Theme | PyPI | Conflict to weigh |
| --- | --- | --- | --- |
| RocketFrame (`rocketframe`) | Space, rocket logo | Free | Generic; Rust's Rocket framework shares the word |
| MachFrame (`machframe`) | Speed | Free | Keeps the "M" of MXFrame |
| Pronghorn (`pronghorn`) | Animal | Free | Small unrelated repos use the name |
| CheetahFrame / CheetahDF | Animal | Free | `cheetah` itself is taken (template engine) |
| LeopardFrame | Animal | Free | `leopard` taken; Mac OS X Leopard association |
| CondorFrame | Animal | Free | `condor` taken; HTCondor is a known distributed-computing system |
| Apogee | Space | Taken | — |
| Liftoff | Space | Taken | — |
| MXFrame (keep) | — | Yours | Already published and announced |

Run a trademark search on the shortlist before choosing.

## Benchmark and test plan

Every performance claim is reported **cold** (data starts in host memory or Parquet, transfers counted) and **hot** (data resident). Warm-only numbers are what make GPU engines look better than they are.

**Workloads**

- TPC-H / PDS-H at scale factors 1, 10, 100, and one above GPU memory (switch from row counts to scale factors so results compare with other engines).
- ClickBench (string-heavy, single wide table).
- H2O db-benchmark group-by and join at low and high key cardinality.
- Break-even sweep: rows (10⁴–10⁸) × selectivity × key cardinality, per operator, to locate the CPU/GPU crossover on each machine.

**Engines to compare:** pandas, Polars (in-memory and streaming), DuckDB, DataFusion, cudf.pandas, Polars GPU (in-memory and streaming), Sirius, HeavyDB (CPU and GPU modes), and MXFrame in `cpu`, `gpu`, and `auto`.

**Hardware:** RTX 3090 (existing, PCIe 4), one PCIe 5 datacenter GPU, one AMD GPU (ROCm), one Apple M-series, and a Grace Hopper (GH200) if rentable.

**Metrics per run:** wall time, bytes host→device and device→host, per-device busy time, fallback count, and **placement accuracy** — how often `auto` matched the fastest forced mode.

**Correctness tests (every phase):**

- `auto` results equal `cpu` results on the full test suite (exact for integers and keys, tolerance for float sums).
- Fallback path tested by forcing OOM and by injecting kernel errors.
- Determinism: same query, same data, same result regardless of morsel size or thread count.
- Calibration file missing, corrupt, or from an older version → safe defaults, no crash.

## Risks, open questions, and notes for the implementer

**Risks**

| Risk | Mitigation |
| --- | --- |
| Cost model is wrong and `auto` is slower than a forced mode | Always ship `cpu`/`gpu`; `explain` + `profile` expose bad picks; online refinement; placement-accuracy metric in CI |
| Merge step of split aggregations dominates at very high cardinality | Planner only inserts `Router` when estimated partial size is small; otherwise single device |
| PyArrow storage forces extra copies into Mojo buffers | Measure in Phase 2; spike a Mojo-native Arrow layer if the copy is large |
| Apple Silicon: no zero-copy host→Metal buffers in Mojo yet | Keep a copy cost for Apple in the model; revisit when Modular ships it |
| Streams/events or pinned memory behave differently per backend | Per-backend capability flags in calibration; degrade to sequential copy+compute |
| Scope creep delays 1.0 | 1.0 = Phases 1–3 + Phase 5; out-of-core and the post-1.0 roadmap ship as 1.x and later |

**Open questions**

- Default morsel size per device — calibrate or fix? Start fixed (2M rows CPU, larger GPU), measure.
- Should `auto` become the default in 1.0, or stay opt-in until Phase 2 lands?
- Where to store calibration on shared servers or containers (home directory vs. env var path)?
- Is HostBuffer page-locked on AMD and Apple backends?

**Notes for the implementer agent**

- Read `docs/api.md`, `lazy_frame.py`, `optimizer.py`, `compiler.py`, and `custom_ops.py` first; auto mode plugs in between the optimizer and kernel dispatch.
- Do not change `cpu` or `gpu` behaviour. Every change must be behind `device="auto"` until 1.0.
- Build Phase 1 in thin vertical slices: calibration → estimator → decision → fallback → explain, each with tests.
- Use the glossary terms in names (`Morsel`, `DeviceCross`, `Router`, `ResidencyTracker`, `Calibration`).
- Every perf PR posts cold and hot numbers from the benchmark script, not warm-only.

## README snippet and references

Paste-ready section for the MXFrame README:

```markdown
## Coming next: `device="auto"`

MXFrame already runs the same Mojo kernels on CPU and on NVIDIA, AMD, and Apple GPUs.
Auto mode lets the engine decide per query, and eventually per operator, where work runs,
and later run CPU and GPU on the same query at once.

The design builds on published research rather than ad-hoc heuristics:

- **Morsel-driven scheduling** — Leis et al., SIGMOD 2014. Work is split into small chunks
  that CPU threads and the GPU pull from one queue, so the faster device takes more.
- **Device-crossing operators** — HetExchange, Chrysogelos et al., PVLDB 2019.
- **Filter on CPU, join on GPU** — Li et al., "Scaling GPU-Accelerated Databases beyond
  GPU Memory Size", PVLDB 2025: PCIe is the bottleneck, so ship only filtered rows.
- **Bandwidth-based cost priors and kernel fusion** — Crystal, Shanbhag et al., SIGMOD 2020.
- **Calibrated, learned per-operator costs** — Ocelot/HyPE, Breß et al., PVLDB 2014.
- **Run where the data lives** — Mordred, Yogatama et al., PVLDB 2022.
- **Hot/cold placement and CPU+GPU executors** — PowerInfer, SOSP 2024 (LLM inference).
- **Overlapping copies with compute; LP-based offload sizing** — FlexGen, ICML 2023.

Roadmap: calibrated per-query choice → per-operator placement →
CPU+GPU co-processing → out-of-core → 1.0 with `auto` as default.
```

**References**

1. Leis, Boncz, Kemper, Neumann. [Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age](https://dl.acm.org/doi/10.1145/2588555.2610507). SIGMOD 2014.
2. Chrysogelos et al. [HetExchange: Encapsulating Heterogeneous CPU–GPU Parallelism in JIT Compiled Engines](https://infoscience.epfl.ch/record/262531). PVLDB 12(5), 2019.
3. Li et al. [Scaling GPU-Accelerated Databases beyond GPU Memory Size](https://www.vldb.org/pvldb/vol18/p4518-li.pdf). PVLDB 18(11), 2025.
4. Shanbhag, Madden, Yu. [A Study of the Fundamental Performance Characteristics of GPUs and CPUs for Database Analytics (Crystal)](https://arxiv.org/abs/2003.01178). SIGMOD 2020.
5. Breß et al. [Ocelot/HyPE: Optimized Data Processing on Heterogeneous Hardware](https://www.vldb.org/pvldb/vol7/p1609-bress.pdf). PVLDB 7, 2014.
6. Yogatama, Gong, Yu. Mordred: orchestrating data placement and query execution in heterogeneous CPU–GPU DBMS. PVLDB 15(11), 2022. (Link not verified in this pass.)
7. [Maximus: A Modular Accelerated Query Engine for Data Analytics on Heterogeneous Systems](https://dl.acm.org/doi/10.1145/3725324). SIGMOD 2025.
8. Aramburu et al. [Theseus: A Distributed and Scalable GPU-Accelerated Query Processing Platform](https://arxiv.org/abs/2508.05029). arXiv 2025.
9. [Rethinking Analytical Processing in the GPU Era (Sirius)](https://vldb.org/cidrdb/papers/2026/p12-yogatama.pdf). CIDR 2026.
10. Song et al. [PowerInfer: Fast LLM Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456). SOSP 2024.
11. Sheng et al. [FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf). ICML 2023.
12. [MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://arxiv.org/abs/2512.22219). arXiv 2025.
13. [Understanding Data Movement in Tightly Coupled Heterogeneous Systems: Grace Hopper](https://arxiv.org/abs/2408.11556). arXiv 2024.
14. [llama.cpp: automation for GPU layers, tensor split, and overrides](https://github.com/ggml-org/llama.cpp/discussions/18049). GitHub discussion.
15. [MXFrame on the Modular forum](https://forum.modular.com/t/mxframe-cross-vendor-gpu-dataframes-with-mojo-aot-kernels/3358) and [MXFrame API docs](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/api.md) — current architecture and benchmark figures.
