# HeavyDB (ex-MapD / OmniSciDB): what mxframe can borrow

Written 2026-10-15. HeavyDB is an open-source, GPU-native SQL database with commercial apps on top (Immerse and others). It is the closest open-source competitor in spirit.

Read from the source at [github.com/heavyai/heavydb](https://github.com/heavyai/heavydb) (master, last commit 2026-10-07) and the [HeavyDB docs](https://docs.nvidia.com/heavyai/). File paths below are relative to that repo. Defaults are read from `QueryEngine/Execute.cpp`, `ThriftHandler/CommandLineOptions.cpp` and `Shared/SystemParameters.h`.

How others measured it, for context:

- On SSB, Mordred was about 9x faster when the data fit in GPU memory and about 11x faster when it didn't ([Mordred §6.5](https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf)).
- On TPC-H SF100 with an 80 GB A100, TQP was 1.5x faster than HeavyDB with data already on the GPU. At SF1000, HeavyDB ran 9 of 22 queries on GPU, and 6 of those were slower than CPU-only SQL Server ([Li et al. 2025 §6.3–6.4](https://www.vldb.org/pvldb/vol18/p4518-li.pdf)).
- Its strengths are maturity and SQL coverage, not raw speed against the best research engines.

## 1. Query compilation: one LLVM JIT kernel per query step

**How it works.**

- A query becomes a sequence of relational-algebra **steps**. A step ends at a pipeline breaker, and each step's output is a materialized `ResultSet`.
- For each step, HeavyDB builds LLVM IR for a **row function**: filters, projected expressions, join hash probes, group key computation and aggregate updates, all inlined for one row.
- A fixed **query template** (`QueryEngine/QueryTemplateGenerator.cpp`) wraps it in the row loop. `pos_start`/`pos_step` give a grid-stride loop on GPU and a plain loop on CPU.
- So every operator inside a step is **fused into one kernel**. Joins become hash-table lookups inside the row function; the hash table itself is built before the step runs.
- Code generation runs at query time:
  - The same IR targets the CPU (LLVM JIT) or the GPU (NVPTX → PTX → cubin, linked with libdevice when needed; `NativeCodegen.cpp`).
  - Runtime helpers are precompiled to bitcode and linked in.
  - Compiled code is cached, keyed by the serialized IR (`CodeCacheKey`), so a repeated query shape skips compilation.
  - Literals are "hoisted" into a parameter buffer, so queries that differ only in constants can share code.

**Borrow.**

- **Fuse per step, not per operator.** One kernel per pipeline is what F-008's bytecode interpreter already does, and what Crystal recommends. Keep pushing more operators into one pass, especially join probes inside the aggregate pipeline.
- **Hoist literals:** treat constants as runtime inputs, so plan caching works across re-parameterized queries. This fits the "general, not benchmark-tuned" rule.
- **Cache key = the compiled program's content.** For our bytecode, the same idea is a fingerprint of the program bytes.

**Don't copy.**

- **A JIT per query shape.** F-008 chose AOT kernels plus a bytecode interpreter because a first run must not pay seconds of compilation. That trade-off still holds for our cold-run benchmarks.
- **Materializing a `ResultSet` at every step boundary.** It costs memory and forces round trips. Our morsel pipelines should stream across operators wherever there is no breaker.

## 2. GPU group-by and hash joins

### Group-by (`GroupByAndAggregate.cpp`, `GroupByRuntime.cpp`)

**Two layouts.**

- **Perfect hash:** the group key's value range (from column metadata) maps directly to a slot. Used when the product of key ranges is at most `g_baseline_groupby_threshold` = **1,000,000** entries.
- **Baseline hash:** open addressing with **linear probing** on the composite key. On GPU, keys are claimed with `atomicCAS`. Used for wide ranges, non-integer keys, or many keys.
- If filters exist, HeavyDB may run a cheap **cardinality-estimator pre-pass** (`CardinalityEstimator.cpp`). It sets bits in a bitmap and applies linear counting, `-m·ln(unset/m)`.
  - It picks baseline over perfect hash when the filtered group count is below a quarter of the range (`cardinality_estimate_less_than_column_range`).
  - The estimate also sizes the baseline buffer.

**Sizing and fallback.**

- The default group-buffer guess is **16,384** entries.
- If the baseline buffer fills (probing wraps around and returns NULL), the kernel reports "out of slots".
- The executor then retries with the guess **doubled, at most twice**, after which it fails.
- A **shared-memory** group-by variant is used when the group buffer fits in GPU shared memory (`gpu-shared-mem-threshold`).

**Borrow.**

- **Use min/max from metadata to pick a dense (perfect-hash) layout.** It costs no hashing and is safe to use for both CPU and GPU, and our TPC-H keys are often dense. Pick the threshold by calibration rather than copying 1M.
- **A linear-counting estimator pre-pass** to size hash tables when filters make metadata useless. It is cheap: one bitmap per morsel, merged with OR. It is better than a fixed guess and fits our morsel design.
- **Fall back explicitly when a table fills.** Grow and retry, or hand the work to the partitioned path from `gpu-out-of-core.md`. Never crash.

**Don't copy.** "Fail after two doublings." Our rule is to never fail: route to partitioned or CPU execution instead.

### Hash joins (`QueryEngine/JoinHashTable/`)

**Perfect join hash** (`PerfectJoinHashTable.cpp`).

- A direct-indexed array over the build key's value range, possibly bucketized.
- **One-to-one** layout when keys are unique.
- Otherwise **one-to-many**, a CSR-style layout built in three GPU passes: count matches per slot, inclusive scan to get offsets, fill row ids (`Runtime/HashJoinRuntimeGpu.cu`).

**Fallback to the baseline join hash** (`HashJoin::getInstance`). It switches from perfect to baseline when:

- the range needs more entries than memory can hold (`TooManyHashEntries`), or
- the range is more than **100x** the number of build rows (`g_ratio_num_hash_entry_to_num_tuple_switch_to_baseline`).

The baseline join hash handles composite and non-integer keys.

**Loop joins:** a nested loop is allowed when the inner side is tiny (`g_trivial_loop_join_threshold` = 1000 rows) or when `allow-loop-joins` is set.

**Hash tables are cached and reused** across queries.

**Borrow.**

- **The size rule:** use a dense layout when range ≤ ~100 × rows, otherwise a real hash.
- **The count / scan / fill CSR build for one-to-many joins.** It is deterministic, has no atomics on payloads, and gives the same result on CPU and GPU. That helps our "equal to CPU" rule.
- **Cache build-side hash tables.** For repeat queries with the same build side, the cached table is a resident structure, which feeds F-006's "resident data wins".

**Don't copy.** Silently allowing loop joins on larger tables.

## 3. Buffer manager, memory tiers and running out of memory

### Tiers (`DataMgr/`)

- Three levels: DISK, CPU and GPU (`MemoryLevel`). Tables are split into **fragments** (row groups, `DEFAULT_FRAGMENT_ROWS` = 32,000,000 in `Fragmenter/FragmentDefaultValues.h`), and each column of a fragment is a **chunk**.
- Each level has a `BufferMgr` that allocates fixed-size pages inside large **slabs**.
- If a slab allocation fails, it halves the slab size and tries again, down to a minimum ("allocations capped").
- **Eviction:** look for the run of unpinned chunks with the oldest "last touched" epoch, and pick the cheapest candidate to evict, scored like golf (`BufferMgr.cpp`). Pinned chunks, in use by a running query, are never evicted.
- The GPU tier is a cache of the CPU tier, which caches disk.

### When data is bigger than GPU memory or a query OOMs

- **Pre-flight:** input bytes per GPU are summed. Over `gpu-input-mem-limit` = **0.9** of that GPU's available memory, the step throws `QueryMustRunOnCpu` (`Descriptors/QueryFragmentDescriptor.cpp`).
- **Retry ladder at runtime** (`RelAlgExecutor::handleOutOfMemoryRetry`):
  1. multi-fragment GPU kernel
  2. one kernel per fragment
  3. **CPU**
- **Scope:** only the failed **step** is retried on CPU (`allow-query-step-cpu-retry`, default on). `allow-cpu-retry` (default on) allows this even with the watchdog on.
- **Watchdog:** off by default in source. When on, it rejects plans likely to blow up, such as projections over **128M rows per device**, huge IN lists, or loop joins.
- **Dynamic watchdog:** a time limit enforced inside kernels.
- **Executor Resource Manager** (default since 7.0) tracks CPU cores, GPUs, buffer memory and result memory, and queues requests in FIFO order. Under memory pressure it runs a query one chunk at a time, on GPU or, if that still doesn't fit, on CPU ([docs](https://docs.nvidia.com/heavyai/installation-and-configuration/executor-resource-manager)).

**Borrow (F-056).**

- The **0.9 per-GPU pre-flight**, the **per-step CPU retry**, and the **ladder of smaller pieces before CPU**. These are the "never fail" step 1 in `gpu-out-of-core.md`.
- **Pin chunks in use, evict the least recently touched.** This matches our `ResidencyTracker` (LRU plus pins). Add slab or pool allocation so we don't fragment and don't pay a `cudaMalloc` on every query.
- An **admission queue** (Executor Resource Manager) once mxframe serves several queries at once. Not needed for 1.0.

**Don't copy.**

- **Silent CPU fallback.** We should log and `explain` every fallback and record it in calibration.
- **"Stream the data on demand over PCIe" coprocessor mode as the out-of-core answer.** Both Mordred and Li et al. measured it as slow.

## 4. Multi-GPU execution

**How it works.**

- Each fragment has a fixed home GPU, recorded in its metadata (`fragment.deviceIds`).
- A step launches kernels per GPU over that GPU's fragments.
- Join hash tables are **built per device**, or built once on CPU and copied to each GPU. There is also **sharding**: with tables sharded on the join key, each GPU joins only its own shard.
- Per-GPU results are **reduced on the host** (`Executor::reduceMultiDeviceResultSets`).
- The memory check is **per GPU**. There is no pooled 48 GB.
- `num-gpus` and `start-gpu` choose which GPUs are used.

**Borrow.**

- **A home GPU per data chunk:** each resident column chunk lives on exactly one GPU, and a GPU's kernels see only its own chunks. Partials are merged on the host, which matches the Router merge.
- **Small build tables:** replicate them.
- **Large build tables:** co-partition both sides on the key across GPUs. This is the same plan as the partitioned path in `gpu-out-of-core.md`.

**Don't copy.** Splitting fragments round-robin regardless of calibration. Our two GPUs are identical, but give the CPU its share too (`cpu-gpu-coprocessing.md`).

**Note:** this box's GPUs have an NVLink bridge (`nvidia-smi topo -m`: NV4). We found no use of peer-to-peer GPU memory in HeavyDB's reduction path; reduction goes through the host.

## 5. Choosing CPU or GPU, and hybrid execution

**How it works.**

- The execution device is chosen **per session or query**: GPU by default, CPU via `cpu-only`, session settings, or the `cpu_mode` query hint.
- A step goes to CPU when:
  - an operator has no GPU support (MODE, APPROX_MEDIAN/PERCENTILE)
  - the step is an update or delete
  - inputs exceed GPU memory
  - the step OOMs
- **Concurrency is not co-processing.** `allow-cpu-gpu-kernel-concurrency` lets CPU queries run *while* a GPU query runs, but **one step never runs on both** ([docs](https://docs.nvidia.com/heavyai/installation-and-configuration/executor-resource-manager)).
- `enable-cpu-sub-tasks` (off by default; sub-task size 500k rows) splits a fragment across CPU threads, which is morsel-like, on CPU only.
- There is no cost model comparing CPU and GPU time. Device choice is rules plus fallback.

**Borrow.**

- **Per-step device choice with step-level retry.** It is simple, robust and easy to explain.
- **Query hints** (`cpu_mode`) as an escape hatch. We have `device=` and `gpu_fraction=` already.

**Where mxframe is ahead.** A calibrated cost model plus a CPU+GPU split on resident data (F-006). HeavyDB has neither, so keep it as a differentiator.

**Don't copy.** "GPU by default if present." On PCIe x8 with host data, that loses to CPU on scan-heavy queries; Li et al. measured HeavyDB slower than CPU on 6 of 9 queries it ran at SF1000.

## 6. Strings: dictionary encoding

**How it works.**

- `TEXT ENCODING DICT(8/16/32)` stores each string as an integer id.
- The dictionary lives on the host (`StringDictionary/`). The GPU only sees ids.
- **LIKE / REGEXP / compare on a dictionary column** are evaluated **once over the dictionary** (`StringDictionary::getLike`, `getRegexpLike`, `getCompare`, with caching). The result is a set of matching ids, which the kernel tests as an integer IN-set or bitmap.
- **Joins and comparisons across dictionaries** use translation maps from one dictionary's ids to another's (`StringDictionaryTranslationMgr.cpp`).
- **String functions** that produce new strings create "transient" ids (negative ids) in a proxy, so the stored dictionary isn't mutated.
- **None-encoded strings** have limited GPU support. The planner treats casts to them carefully (`RelAlgExecutor.cpp`).

**Borrow (this is the Phase 5 item "GPU string kernels on dictionary-encoded columns").**

- **Evaluate string predicates on the dictionary, not on rows.** `contains`, `startswith` or a regex runs on the CPU over the distinct values (thousands, not millions). The result becomes the existing **IN_SET on dictionary codes** in the F-008 bytecode. It needs no GPU string kernel at all, and gives the same result on CPU and GPU.
- **Translation maps** for joins and unions over two Arrow dictionaries. Arrow dictionary arrays per chunk can differ, so remap the codes once per chunk.
- **Transient ids** for string expressions computed inside a query.

**Don't copy.** A database-owned global dictionary. mxframe reads Arrow and Parquet, where each chunk has its own dictionary. Unify or translate per query instead.

## Summary: the top five ideas to take

1. Per-GPU 0.9 pre-flight plus the ladder: smaller pieces, then step-level CPU retry (F-056).
2. Dense (perfect-hash) layouts from min/max metadata, with a cardinality pre-pass to size hash tables. Grow or partition when full; never fail.
3. CSR one-to-many join build (count, scan, fill). It is deterministic, so CPU and GPU give equal results.
4. String predicates evaluated over the dictionary, then an integer IN-set in the kernel.
5. A home GPU per resident chunk, a replicated small build table, host-side merge of partials.

**Not verified:**

- JIT compile times: we have no primary measurement.
- How HeavyDB assigns fragments to GPUs inside the storage layer (we read only the executor side).
