# When the working set is bigger than the GPU: how others avoid failing

Research for F-056 ("data bigger than GPU memory never fails"). Written 2026-10-15.
Sources are primary (docs, source code, papers) unless marked. Numbers are the sources' own; none were re-measured here.

## The box this is for

- 2x RTX 3090, 24 GB each. Each GPU runs on PCIe 4.0 **x8** (`nvidia-smi`: max width 16, current width 8). F-006 measured pinned host-to-device at 13.4 GB/s.
- Host: Ryzen 9 5900X, 24 threads, dual-channel DDR4 at ~40–50 GB/s, **62 GB RAM**.
- **New finding:** `nvidia-smi topo -m` shows the two GPUs joined by an **NVLink bridge (NV4)**: 4 links at 14.06 GB/s each, so about 56 GB/s per direction on paper. Nobody has measured it on this box, and we don't know yet whether Mojo/MAX exposes peer access. It matters for the multi-GPU part of the recommendation.
- **Honest flag on the F-056 quote** ("96 GB on a 48 GB machine"): 48 GB is the two GPUs together. A 96 GB working set is also bigger than host RAM (62 GB). So that case needs CPU-side out-of-core too (stream Parquet row groups, spill to disk), not only GPU tricks. This file covers the GPU side. The CPU side is a separate gap.

## Summary table

| System | Mechanism | Decides when | OOM in the middle of a query | CPU fallback? |
| --- | --- | --- | --- | --- |
| cuDF (Python) | Spillable buffers moved to host on demand | At allocation time (RMM OOM handler), optional soft limit | The handler spills, then the allocation retries | No (cudf.pandas: yes, per call) |
| cudf.pandas | Managed (UVM) memory pool plus prefetch | Driver page migration | Mostly avoided through oversubscription | Yes, per operation |
| cudf-polars streaming | Chunked partitions plus RapidsMPF spill to host | Plan time (partition size); spills at 80% of the device | Docs say: shrink partitions and spill earlier | Only for unsupported ops, at plan time |
| dask-cudf / dask-cuda | Partitions as tasks; worker spills at 80% | Spills per task output, or per buffer with cuDF spill | Task fails | No |
| Spark RAPIDS | Spillable batches plus retry, split and retry | At runtime | Roll back, spill, halve the batch, retry | Not on OOM (only unsupported ops) |
| BlazingSQL (archived 2021) | CacheMachines move data GPU→host→disk; memory monitor; tasks admitted against an estimate | Before every task, and every 50 ms | Task re-queued with the same inputs | No (GPU only) |
| HeavyDB | Pre-flight input check per GPU; retry ladder | Plan time (90% of GPU memory) plus runtime | Multi-fragment kernel → kernel per fragment → CPU | Yes, per query step |
| Sirius | Tiered GPU/host/disk memory, partitioning (README); CIDR paper says in-memory only | Not documented in detail | Not documented | Yes, for unsupported ops |
| Theseus | Async executors for compute, preload, spill; pinned fixed-size host pool | Runtime | Spill across tiers | No (GPU-native) |
| Li et al. 2025 (SQL Server + TQP) | CPU filter + bitvector, streaming probe, sequential partition pairs | Plan time | Prototype catches OOM and runs on CPU | Yes |

## 1. cuDF spilling (RMM)

**Mechanism.** With spilling on, cuDF wraps device memory in `SpillableBuffer`. A global spill manager tracks every such buffer and moves it to host memory when needed. Reading a buffer's pointer moves it back ("unspill") ([cuDF library design](https://docs.nvidia.com/cudf/26.10/cudf/developer_guide/library_design/)).

**When it decides.** `spill_on_demand` registers an RMM out-of-memory handler. A failed allocation triggers a spill, then the allocation retries. An optional `spill_device_limit` sets a soft byte limit. It costs "a modest overhead" and is off by default ([same page](https://docs.nvidia.com/cudf/26.10/cudf/developer_guide/library_design/)).

**Mid-query OOM.** A buffer whose raw pointer was handed out is marked "exposed" and can never be spilled again. So the limit is soft, and OOM is still possible when too many buffers are exposed. A [user report](https://forums.developer.nvidia.com/t/cudf-spill-on-demand-couldnt-find-memory-device-to-spill-spillmanager-device-memory-limit-spilled-unspilled/332484) shows that failure.

**Cost.** The docs give no throughput numbers. Each spill or unspill is a PCIe copy, and the operation that needs the data waits for it.

**cudf.pandas** takes a different route. By default it uses a managed (UVM) memory pool with prefetching, so data larger than GPU memory runs "in many cases, without CPU (pandas) fallback". Unsupported or failed operations still fall back to pandas ([how it works](https://docs.rapids.ai/api/cudf/latest/cudf_pandas/how-it-works); [NVIDIA UVM blog](https://developer.nvidia.com/blog/unified-virtual-memory-supercharges-pandas-with-rapids-cudf)).

**Measured UVM cost** ([Polars UVM post](https://pola.rs/posts/uvm-larger-than-ram-gpu), H200 141 GB, PDS-H SF10 and SF300):

- Plain UVM ran 1.5–2.4x slower than plain CUDA allocation.
- UVM with a pool and prefetch ran 1.2–1.3x slower.
- UVM finished every SF300 query (300 GB on a 141 GB GPU). Plain CUDA with a pool failed on the first query.
- These numbers come from an H200 on a much faster link than x8. Expect more on this box. Not measured.

## 2. cudf-polars streaming executor

**Mechanism.** The streaming engines split inputs into partitions and push them through the query graph in chunks ([engines](https://docs.nvidia.com/cudf/latest/cudf_polars/engines/)). The chunk target is `target_partition_size`: the smaller of 1.5 GB and 2.5% of the smallest GPU. It is "a target, not a hard ceiling" ([memory guide](https://docs.nvidia.com/cudf/latest/cudf_polars/memory_errors/)).

**Spill.** RapidsMPF spills chunks that aren't in use to host memory once the GPU passes `RAPIDSMPF_SPILL_DEVICE_LIMIT`, 80% by default. Spill memory is pageable by default. Pinned memory is opt-in because it is costly to register and can't be swapped ([memory guide](https://docs.nvidia.com/cudf/latest/cudf_polars/memory_errors/)).

**What still causes OOM** ([memory guide](https://docs.nvidia.com/cudf/latest/cudf_polars/memory_errors/)):

- shuffles: sort, high-cardinality group-by, join
- skewed keys
- outputs with several consumers
- CSV inputs, which can't be read in part

The documented fix is to lower the partition size and the spill limit. There is no automatic retry.

**CPU fallback.** Only at plan time, for operations the GPU doesn't support. `fallback_mode` is warn, raise or silent. `raise_on_fail` turns the fallback off ([options](https://docs.nvidia.com/cudf/latest/cudf_polars/options/); [Polars GPU docs](https://docs.pola.rs/user-guide/gpu-support/)). The Polars docs admit some unsupported queries "fail at runtime" instead of falling back. Neither doc describes falling back to CPU after a GPU OOM.

**Cost.** No spill-cost numbers are published. NVIDIA says the multi-GPU streaming engine ran PDS-H at 3 TB, but gives no timings ([NVIDIA blog](https://developer.nvidia.com/blog/how-to-work-with-data-exceeding-vram-in-the-polars-gpu-engine)).

## 3. dask-cudf / dask-cuda

**Mechanism.**

- dask-cudf splits a frame into partitions, and each partition is a task on one GPU worker.
- Joins and group-bys shuffle partitions between workers.
- A worker spills to host once it passes `device_memory_limit`, 80% of the GPU by default.
- Default spilling moves whole task outputs, so data created inside a running task can't be spilled.
- With `enable_cudf_spill=True`, single buffers can be spilled while a task runs. Buffers handed to other libraries become unspillable.

Source: [dask-cuda spilling docs](https://github.com/rapidsai/dask-cuda/blob/main/docs/source/spilling.rst).

**Mid-query OOM.** The task fails. The docs say to tune the limit or use smaller partitions. There is no CPU fallback.

## 4. Spark RAPIDS (not in the original list, but the most relevant OOM design)

Since 23.04, "many out of memory errors result in parts of the query being rolled back and retried instead of a task failure". The retry raises `RetryOOM`, or `SplitAndRetryOOM`, which halves the input batch and tries again ([tuning guide](https://docs.nvidia.com/cudf-spark/latest/tuning-guide.html)). Spilled batches go to a host buffer first, then to disk. This fits a morsel engine directly: a morsel that OOMs gets split and retried.

## 5. BlazingSQL (archived; last push 2022)

Source: the repo's own design docs, [memory_management.rst](https://github.com/BlazingDB/blazingsql/blob/branch-21.08/docsrc/source/engine/memory_management.rst) and [executor.rst](https://github.com/BlazingDB/blazingsql/blob/branch-21.08/docsrc/source/engine/executor.rst).

**Mechanism: spill to host and disk.**

- Data between kernels lives in "CacheMachines".
- When data enters a cache, the engine checks GPU, host and disk use, and stores it as GPU, host or disk "CacheData".
- A `MemoryMonitor` wakes every 50 ms. If GPU use is too high, it walks the plan from the output back to the scans and moves cached data to host or disk.

**Deciding before each task (admission control).**

- Before each task, the executor estimates its memory need: inputs to bring back, plus output size, plus algorithm overhead. Every kernel must provide these estimates.
- It does not start a task that would push GPU use past `BLAZING_PROCESSING_DEVICE_MEM_CONSUMPTION_THRESHOLD` (default 0.9). The exception is when nothing else is running.
- Other defaults: 0.6 for device caching and 0.75 for host ([context.py](https://github.com/BlazingDB/blazingsql/blob/branch-21.08/pyblazing/pyblazing/apiv2/context.py)).

**Mid-query OOM.**

- The task is rebuilt from its inputs and re-queued.
- If the task changed its inputs in place, it can't be retried, and the query fails.
- There is no CPU fallback, since BlazingSQL is GPU-only.
- Mordred reports BlazingSQL hit OOM on SSB Q4.x even when the data fit ([Mordred §6.5](https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf)).

**Lesson for us:** only retry fragments whose inputs are left untouched.

## 6. HeavyDB

Read from the source ([heavydb master, Oct 2026](https://github.com/heavyai/heavydb)). There is more in [heavydb.md](heavydb.md).

**Pre-flight check, per GPU.**

- Each fragment has a home GPU.
- `QueryFragmentDescriptor::checkDeviceMemoryUsage` adds up the input bytes per GPU. If they pass `gpu-input-mem-limit` × available memory (default **0.9**), it throws `QueryMustRunOnCpu`.
- Memory is counted per device, not as one pool.

**Retry ladder on OOM** (`RelAlgExecutor::handleOutOfMemoryRetry`):

1. The multi-fragment GPU kernel ran out of memory, so retry with **one kernel per fragment**, which needs less input at once.
2. If that also fails, rerun the step on **CPU**.
3. On CPU, if the group-by output buffer overflows, double the guess, up to twice.

The step-level CPU retry (`allow-query-step-cpu-retry`, default on) reruns only the failed step, not the whole query. `allow-cpu-retry` (default on) allows a CPU retry even with the watchdog on.

**Executor Resource Manager** (default since 7.0). When memory is short, it runs the query one chunk at a time on GPU. If the GPU is still too small, it goes chunk by chunk on CPU ([docs](https://docs.nvidia.com/heavyai/installation-and-configuration/executor-resource-manager)).

**Measured cost.** Li et al. ran HeavyDB on TPC-H SF1000 with an 80 GB A100. 9 of 22 queries ran on GPU, and 6 of those 9 were "significantly slower than the CPU-only approach", likely because of PCIe ([Li et al. §6.4](https://www.vldb.org/pvldb/vol18/p4518-li.pdf)). Mordred measured HeavyDB's "coprocessor mode" (streaming data from host on demand) about 11x slower than Mordred on SSB at 4x the GPU cache size ([Mordred §6.5.2](https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf)).

## 7. Sirius (GPU DuckDB)

- **The CIDR 2026 paper** says Sirius "operates under the assumption that the input data fits in the caching region and the intermediate results fit in the processing region". Spill to pinned memory and disk, and batched execution, are listed as future work. CPU fallback to DuckDB covers errors and missing features ([CIDR'26 §3.2.2, §3.4](https://vldb.org/cidrdb/papers/2026/p12-yogatama.pdf)).
- **Memory split in the paper:** 50% of GPU memory caches data and 50% holds intermediate results (an RMM pool).
- **The current README** claims "out-of-core with tiered memory management (GPU/host/disk)", "automatic data partitioning, and spilling". Defaults are 95% of GPU memory and 8 GiB pinned host memory per NUMA node. Unsupported operators "fall back silently to CPU" ([sirius-db/sirius](https://github.com/sirius-db/sirius)).
- **Not verified:** how the README's out-of-core mode decides, or what it costs. No paper describes it yet.

## 8. Theseus (Voltron Data)

The paper uses separate asynchronous executors for compute, network, pre-loading and spilling across memories and storage, plus fixed-size pinned host allocations to cut fragmentation ([arXiv 2508.05029](https://arxiv.org/abs/2508.05029)). It runs TPC-H and TPC-DS at SF100k on two DGX A100 nodes. The abstract gives no per-mechanism costs.

## 9. Out-of-GPU-memory join research

**Li et al. 2025, "Scaling GPU-Accelerated Databases beyond GPU Memory Size" (Microsoft, [PVLDB 18(11)](https://www.vldb.org/pvldb/vol18/p4518-li.pdf)). Real; read in full.**

- **Setup:** SQL Server for CPU work, TQP on the GPU, one 80 GB A100 on PCIe 4.0 (24 GB/s, 3.3x below CPU memory bandwidth).
- **Data filtering on CPU:** CPU scans filter rows and compact them while still compressed. Only the surviving rows cross PCIe, and the GPU decompresses.
- **Bitvector filters:** built from the small join side and applied to the big side's scan on CPU. Chosen by cost: only where saving transfer outweighs the CPU work.
- **Streaming:** the build hash table stays on the GPU and the probe side streams in chunks. Streaming "do[es] not reduce PCIe transfer volume and therefore do[es] not alleviate the PCIe bottleneck".
- **Partitioning:** both join sides are split by **repeated CPU scans with range predicates** on the join key, for example `l_partkey BETWEEN x AND y`. Partition pairs are processed **one after another**, each fitting the GPU.
- **On OOM:** "Our prototype handles such cases by catching the exceptions and falling back to CPU execution."
- **Results at TPC-H SF1000 (1 TB on an 80 GB GPU):**
  - The GPU-only plan ran 4 of 22 queries.
  - Adding streaming made 10 more runnable, but 7 of those 10 were slower or only marginally faster than CPU.
  - Predicate filtering made 1 more runnable and sped up 7.
  - Bitvector filtering made 3 more runnable.
  - Partitioning made the last 4 runnable (Q9, Q13, Q18, Q21), at 3.3–9.0x over CPU.
  - Overall: 22/22 queries, 3.5x over CPU, 0.8–9.1x per query, faster than CPU on 21 of 22.
  - At SF100 it moved Q1 and Q6 fully to the CPU, because scan-only queries run faster there.

**"Pump Up the Volume" (Lutz et al., [SIGMOD 2020](https://users.cs.utah.edu/~pandey/courses/cs6530/fall24/papers/GPU/3318464.3389705.pdf)).** A no-partitioning hash join whose hash table spills to CPU memory over NVLink 2.0. It was up to 18x faster than over PCIe 3.0 and 7.3x faster than an optimized CPU join. A "hybrid hash table" keeps as much as fits in GPU memory and puts the rest in CPU memory. Working inside GPU memory was still 6.5x faster than over NVLink. The authors note that earlier work, built for slow links, either can't grow past GPU memory or is "restricted to partitioning-based joins". **On PCIe this design does not pay off;** it depends on a fast coherent link.

**Triton join (Lutz et al., [SIGMOD 2022](https://www-live.dfki.de/fileadmin/user_upload/import/12723_Triton.pdf)).**

- A two-pass radix-partitioned join. Pass one partitions over the interconnect and spills to CPU memory what doesn't fit. Pass two partitions and joins inside the GPU.
- More than 100x faster than a no-partitioning GPU join, and up to 2.5x faster than a CPU radix join.
- Built for NVLink 2.0. The authors say earlier work "assume[s] that PCI-e causes a transfer bottleneck".
- On PCIe x8 the first pass would run at link speed, slower than a CPU radix partition from DRAM. Not measured.

**Hardware-conscious hash joins on GPUs (Sioulas et al., [ICDE 2019](https://infoscience.epfl.ch/record/262530)).** Radix-partitioned GPU joins beat non-partitioned ones and reach about 1 billion tuples/s even with no data resident on the GPU. They are designed around limited GPU memory and the PCIe link.

**Vortex (Yuan et al., [VLDB 2025](https://arxiv.org/abs/2502.09541)).**

- Uses **the PCIe links of all GPUs** in a box to feed one GPU: data is relayed through the other GPUs over peer links in 20 MB packets.
- On 4x MI100 over PCIe 4.0, it reached up to 140 GB/s against about 28 GB/s for one link.
- Out-of-core sort and a Triton-style radix join run on top.
- 3.4x faster than DuckDB on SSB SF1000.
- **Relevant here** because this box has two x8 links plus an NVLink bridge. Not tested; the CPU's root complex may cap the combined bandwidth.

## Recommendation for mxframe

Constraints: morsel engine, host-resident inputs, PCIe x8 at 13 GB/s (slower than DRAM), two separate 24 GB pools, results must equal CPU.

**What not to build first.**

- **UVM or managed memory:** costs 1.2–2.4x even on an H200, and we don't know if Mojo exposes it.
- **Generic GPU→host spill:** our inputs already live on the host, so spilling them gains nothing. What needs care is GPU-side intermediate state (hash tables, group tables).
- **Triton or Pump Up joins:** they need NVLink-to-host bandwidth.

**Order to build:**

1. **Never fail. This is the first, simplest safe step.**
   - **Memory pools.** Track a free-memory budget per GPU (`gpu:0` and `gpu:1` separately; HeavyDB does the same). Add a test-only override such as `MXFRAME_GPU_MEM_BUDGET` so tests can pretend a GPU is small.
   - **Up front.** Before a fragment runs, estimate inputs + hash or group tables + outputs (BlazingSQL's three terms). If the total is over 0.9 × budget (HeavyDB's default), plan that fragment on CPU and say why in `explain`.
   - **On OOM, step one.** Split the morsel in half and retry once on the same GPU (Spark RAPIDS split-and-retry; HeavyDB's kernel per fragment).
   - **On OOM, step two.** Rerun that fragment on CPU (HeavyDB step retry; Li et al.). Record the shape in calibration so the next plan goes straight to CPU.
   - **Retry safety.** Fragments must never change their inputs, and a failed fragment's GPU output is thrown away (BlazingSQL's lesson). That makes the CPU rerun produce the same answer.
   - **Why this is cheap.** It reuses the `coproc.py` GPU-failure fallback, the Placer's explain lines and calibration. It meets every F-056 acceptance check except "uses the GPU anyway".
2. **Streaming with resident state, for joins and high-cardinality group-bys only.**
   - When the build side or group table fits one GPU's budget, keep it resident and stream the probe side's morsels through the existing `DeviceStager`.
   - Filter on CPU first and apply a bitvector from the build keys (Li et al.), so fewer bytes cross x8.
   - Only plan this where calibration says the CPU join is slower than the link. F-006 and Li et al. agree that streaming scans or simple aggregates loses to CPU on this link.
3. **Partitioned execution across both GPUs.**
   - When the state doesn't fit one GPU, split both sides on the join or group key into P parts so each pair fits one GPU's budget.
   - Start with Li et al.'s simplest form: repeated CPU scans with key-range predicates. Each scan reads DRAM at 40–50 GB/s, about 3x faster than the link. Hash partitioning on CPU is the next step if keys are skewed.
   - Hand pairs to `gpu:0` and `gpu:1` as two independent workers, each with its own budget. Results are the union, or merged partials for aggregates.
   - The NVLink bridge could later move partitions or replicate small build tables between GPUs. Measure it first, and check that Mojo can enable peer access.

Out of scope here but needed for "96 GB": the CPU path itself must stream Parquet row groups and spill to disk, because 96 GB is bigger than host RAM (62 GB).

**Not known or not measured:**

- the NVLink bandwidth on this box
- the combined H2D bandwidth of both x8 links used at once
- whether Mojo exposes peer access or managed memory
- spill costs for cudf-polars
- how Sirius's current out-of-core mode works
