# When does CPU + GPU on one query beat the faster device alone?

Research for F-057 (morsel co-processing that wins). Written 2026-10-15.
Numbers are the sources' own unless they are from F-006 (measured on this box).

## What F-006 measured on this box

Hardware: 2x RTX 3090, PCIe 4.0 x8 (13.4 GB/s pinned), DDR4 at ~40–50 GB/s, 24 threads.

SF10 Q1 fused stage, 60M rows:

| Plan | Time |
| --- | --- |
| CPU | 57 ms |
| GPU, data on host | 353 ms |
| GPU, data resident | 39 ms |
| **Static split, CPU 40% + GPU 60%, resident** | **25 ms** |
| Router streaming host data to GPU | 67–160 ms |
| Dynamic morsel queue, resident | 42 ms |

Each GPU call has about **4.5 ms fixed cost**.

So: on this box, splitting **resident** data with one GPU call won. Streaming **host** data to the GPU lost to the CPU alone.

## What the literature says

### Morsel-driven parallelism (Leis et al., [SIGMOD 2014](https://dl.acm.org/doi/10.1145/2588555.2610507))

- Input is cut into small morsels, about 100k tuples in the paper.
- Worker threads pull morsels from a dispatcher and run a whole pipeline on each, up to the next pipeline breaker.
- Load balance and elasticity come from pulling small units, not from a degree of parallelism fixed in the plan. Idle threads steal work.
- **Built for homogeneous CPU cores.** It assumes per-morsel cost is tiny. That doesn't hold for a GPU with a 4.5 ms fixed cost per call.

### HetExchange (Chrysogelos et al., [PVLDB 12(5) 2019](https://www.vldb.org/pvldb/vol12/p544-chrysogelos.pdf))

**Design.**

- Device-crossing operators (`cpu2gpu`, `gpu2cpu`, mem-move) plus a **router** that sends block handles to CPU or GPU consumers. The policy can be hash, round-robin or load-aware.
- Affinity and degree of parallelism are set **statically at query time**. Dynamic decisions are listed as future work.

**Hardware.** 2x 12-core Xeon and 2x GTX 1080 (8 GB), each on PCIe 3.0 x16 (about 12 GB/s measured). DRAM about 90 GB/s.

**SSB SF1000 results (600 GB on host, which forces streaming):**

- Hybrid beat CPU-only Proteus by up to 5.6x and GPU-only by up to 3.9x.
- Hybrid throughput averaged **88.5% of CPU throughput + GPU throughput added together**.
- The biggest wins came "when Proteus CPU and Proteus GPU have similar performance". When one device is much faster, hybrid ≈ the faster one.
- For scan-heavy, selective queries, CPU-only beat GPU-only, because the CPU scans 38–72 GB/s against 24 GB/s of PCIe.

**Microbenchmarks.**

- *Sum over a 23 GB column* (memory-bound): the CPU reached 89.7 GB/s. Adding 2 GPUs added 19 GB/s, but that gain "slowly diminishes as the number of CPU cores increases, due to exhausting the input memory bandwidth". With all cores, peak was the same with or without GPUs.
- *1:N join* (random-access-bound): adding devices scaled. Adding one CPU core to a GPU-only plan *slowed it down*, because the GPUs waited for the CPU build.

### Pump Up the Volume (Lutz et al., [SIGMOD 2020](https://users.cs.utah.edu/~pandey/courses/cs6530/fall24/papers/GPU/3318464.3389705.pdf))

- **Scheduling:** morsel dispatch adapted to the GPU by handing it **batches of morsels**, so each kernel launch covers more data. The batch size was tuned by hand.
- **Join strategies:**
  - *Het:* one shared hash table in CPU memory.
  - *GPU + Het:* build on the GPU, copy the table to the CPU, then probe on both.
- **Results on IBM POWER9 + V100 with NVLink 2.0, a fast coherent link:**
  - Adding the GPU always helped compared with CPU-only.
  - But "the GPU-only strategy achieves the best throughput for most of our workloads".
  - The co-processing gain over GPU alone was **+16%** on one workload.
  - In the build phase, two processors were slower than one.
- If co-processing mostly loses to GPU-alone on NVLink 2.0, it will lose more often on PCIe x8 with host data.

### CPU-GPU co-processing for joins on a coupled chip (He et al., [PVLDB 6(10) 2013](https://arxiv.org/pdf/1307.1955))

- On an AMD APU (CPU and GPU on one chip, shared memory, no PCIe), fine-grained co-processing beat CPU-only by up to 53%, GPU-only by 35%, and discrete-style co-processing by 28%.
- For discrete GPUs, PCIe is the limit. Overlapping transfer with compute reduces it but doesn't remove it.

### Robust query processing in co-processor databases (Breß et al., [SIGMOD 2016](https://dbis.cs.tu-dortmund.de/publikationen/2016/robust-query-processing/))

- Running operators on the co-processor without care caused "cache thrashing", up to 24x slower: the GPU cache was smaller than the working set.
- It also caused "heap contention", up to 6x slower: parallel GPU operators ran out of device memory.
- The fix was **data-driven placement** (run where the inputs already are) plus **query chopping** (cap concurrent GPU work).
- **Mordred** later showed that data-driven placement beat a cost-based optimizer while being simpler ([Mordred §4.1.1](https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf)).

### Mordred (Yogatama et al., [PVLDB 15(11) 2022](https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf)) and Lancelot ([PVLDB 17 2024](https://www.vldb.org/pvldb/vol17/p4709-yogatama.pdf))

These are the closest match to F-006's winning design.

**Mordred: segment-level co-processing.**

- An operator's work on segments cached on the GPU runs on the GPU. The rest runs on the CPU. Partials are merged.
- Joins need the build side fully cached on the GPU, and the probe is split.
- **Segment grouping** launches **one kernel per group of segments** instead of one per segment: up to 3x faster. Grouping plus merging was 0.3–4% of runtime.
- Hardware: V100 16 GB on PCIe 3 (12.8 GB/s).
- On SSB at 4x the GPU cache size, Mordred was about 11x faster than HeavyDB's streaming coprocessor mode.

**Lancelot: extends this to several GPUs.**

- Data is cached across GPUs, and replication is cost-based ("cache-aware replication").
- Up to 2.5x from replication. At least 2x faster than other GPU DBMSes on SSB and 12x on TPC-H.

### Li et al. 2025 ([PVLDB 18(11)](https://www.vldb.org/pvldb/vol18/p4518-li.pdf))

- On PCIe 4.0 x16 (24 GB/s), CPU scans reach close to DRAM bandwidth (80 GB/s), but CPU joins run "significantly slower than the PCIe speed".
- **The split is by operator, not by morsel:** filter on CPU, join and aggregate on GPU.
- Scan-only queries (Q1, Q6) ran **entirely on CPU**.
- On TPC-H SF100 cold it was 3.5x faster than SQL Server, and 45% slower than TQP with data already on the GPU.
- This is placement (F-007), not two devices running the same operator.

### Survey and others

- **Rosenfeld, Breß, Markl, "Query Processing on Heterogeneous CPU/GPU Systems"** ([ACM CSUR 55(2), 2023](https://dl.acm.org/doi/10.1145/3485126)) groups the field's work into three problems: dividing work, getting past the transfer bottleneck, and portable operators. It is useful as a map. We did not read it in full in this pass.
- **Paul, He, He: GPL** ([SIGMOD 2016](https://www.comp.nus.edu.sg/~hebs/pub/GPL.pdf)): pipelined GPU execution with concurrent kernels and channels between them, tuned by a cost model. Up to 48% over one kernel at a time (as summarized; not re-read).
- **Karnagel et al.** ([PVLDB 10(7) 2017](https://mail.vldb.org/pvldb/volumes/10/paper/Adaptive%20Work%20Placement%20for%20Query%20Processing%20on%20Heterogeneous%20Computing%20Resources)): cut the plan into "execution islands" and place each one at runtime, once its input cardinality is known. Avoids relying on bad estimates.
- **Sort.** Gowanlock and Karsin: GPU sorts chunks and CPU merges, up to 3.47x over a parallel CPU sort on PCIe machines (from the abstract of their 2019 *Parallel Computing* paper; [DaMoN'19 companion](https://par.nsf.gov/servlets/purl/10126755)). Not read in full.
- **Maximus** (Kabić, Chandran, Alonso, [SIGMOD 2025](https://dl.acm.org/doi/10.1145/3725324)): per-operator CPU/GPU placement that overlaps transfer and compute. We have not read the full text, so we can't report its numbers.
- **2023–2026 work.** Beyond Lancelot, Li et al. and Vortex ([VLDB 2025](https://arxiv.org/abs/2502.09541); several GPU links feed one GPU), we found **no new paper on morsel-level CPU+GPU splitting of one operator**. The trend runs the other way. Sirius argues for GPU-native design, with "fine-tuned CPU–GPU co-execution" increasingly matched by the GPU alone on new hardware ([CIDR 2026](https://vldb.org/cidrdb/papers/2026/p12-yogatama.pdf)).

### PCIe compared with NVLink and Grace Hopper

- **PCIe 3.0 or 4.0:** the link is 3–7x below DRAM. GPUs win on host data only for random-access-heavy operators (joins, high-cardinality group-by). For scans, the CPU wins (HetExchange, Li et al.).
- **NVLink 2.0 (POWER9):** a GPU reads CPU memory at about CPU speed. Out-of-core joins and co-processing become viable, but GPU-alone is still usually best (Pump Up).
- **Grace Hopper NVLink-C2C (900 GB/s bidirectional):** Sirius on GH200 treats the GPU as primary and gives the CPU fallback only. A [data-movement study](https://arxiv.org/abs/2408.11556) says placement still matters on GH200.
- **This box is the worst case:** PCIe 4.0 at **x8**, so 13 GB/s against 40–50 GB/s of DRAM, a 3–4x gap.

## The decision rule

Let `Tc` be the CPU-only time and `Tg` the GPU-only time, transfers included (zero if the data is resident). Let `r = max(Tc,Tg)/min(Tc,Tg)`.

- **Best-case split time** (perfect balance, no overhead): `Tc·Tg/(Tc+Tg)`. It saves `1/(1+r)` of the faster device's time:
  - r = 1 saves 50%
  - r = 2 saves 33%
  - r = 4 saves 20%
- **Things that eat into that saving:**
  - the GPU call's fixed cost (4.5 ms here)
  - merging the partials
  - **DRAM contention:** both devices read host memory, and GPU feeder threads take bandwidth from CPU workers (HetExchange's sum test)

**Rule:**

1. **Data residency first.**
   - If the data is on the host and the pipeline is memory-bound (scan, filter, simple aggregate): **no split, CPU only.** Here `Tg ≥ bytes/13 GB/s` and `Tc ≈ bytes/40 GB/s`, so r ≥ 3 *and* the devices compete for the same DRAM (F-006; Li et al. Q1/Q6; HetExchange).
   - If the data is resident on the GPU: a split is a candidate.
2. **Operator type.** Host data can still be split, or placed on the GPU, when the CPU's throughput on the operator is *below* the link's. That means join probe, high-cardinality group-by, or sort. Even then, Li et al. and Mordred split by **operator** (CPU filters, GPU joins) or by **where segments live**. They don't split the same morsel stream across devices.
3. **Gate the split on overhead.** Split only if the predicted `Tc·Tg/(Tc+Tg) + fixed_gpu + merge < 0.9 · min(Tc, Tg)`. That needs **r ≤ ~3** and a faster-device time of at least about 10x the GPU fixed cost (≥ ~45 ms on this box), or a split that uses one GPU call.
4. **Decomposable aggregates only:** sum, count, min, max, mean as sum/count. Merge cost must be small: groups ≪ rows.

## Split algorithm: static, dynamic, or both

- **Static fraction from calibration.** Set `f_gpu = Tc/(Tc+Tg)` and make **one GPU call** for that share.
  - Best when queries are short compared with the fixed cost. F-006 measured 25 ms static against 42 ms dynamic.
  - Mordred's segment grouping (one kernel per group) is the same idea.
  - It depends on calibration being right; refine it online (F-006's `observe`).
- **Dynamic morsels or work stealing** (Leis; Pump Up batches morsels to the GPU).
  - Fixes estimation errors, but each GPU morsel pays the fixed cost.
  - For fixed cost ≤ 10% of a morsel, a GPU morsel must run at least 45 ms. At F-006's resident rate (60M rows in 39 ms, about 1.5 G rows/s), that is **about 70M rows per GPU morsel**, bigger than all of SF10 lineitem.
  - **So on this box, dynamic GPU morsels only make sense for very large inputs.**
- **Both (recommended).** Hand out the static split, then let **CPU threads steal the tail** of the GPU's share in CPU-sized morsels if the GPU runs late. The GPU never takes extra small morsels. This keeps one GPU call and still fixes a bad estimate.
- **CPU morsels:** keep them small, at the scale of Leis's 100k tuples, so stealing stays fine-grained. Cost per CPU morsel is negligible.

## Which operators benefit

- **High-volume decomposable aggregation on resident data:** yes. This is F-006's win.
- **High-cardinality group-by:** the CPU is bound by random access, so the GPU is strong. A split only helps if r ≤ ~3. Merging grows with group count; merge on the device holding most partials (spec, Phase 3).
- **Join probe:** split the probe once the build is replicated to each device. Pump Up and HetExchange broadcast or copy small build tables. Building on two devices at once was slower in Pump Up, so **build on one device and copy**.
- **Sort:** the literature does GPU chunk sort plus CPU merge. Gains are reported, but we haven't verified them in detail. Lower priority.
- **Scan, filter, simple aggregate on host data:** never on this box.

## Recommendation for mxframe's auto planner

1. **Keep F-006's design as the rule.** Host data goes to CPU, plus background upload of hot columns. Resident data gets a static split with one GPU call.
   - The research agrees: Mordred is the published version of this design, and Li et al. run scan-only queries on CPU.
   - Close the open F-006 item "streaming host data beats CPU" as **not achievable on x8**. Revisit on PCIe 5 x16, NVLink-C2C, or unified memory.
2. **Add the split gate above.** Compute r from calibration, check the fixed-cost and merge terms, require a 10% predicted gain, and put the numbers in `explain` ("split: CPU 57 ms, GPU 39 ms, r = 1.5, predicted 25 ms"). This meets F-057's "chosen only where calibration predicts it beats both" check.
3. **Static split plus CPU tail stealing**, instead of a dynamic GPU morsel queue. Re-run F-006's comparison on a quiet machine after the native CPU executor takes the CPU share (F-057, item 2).
4. **Use the second GPU as a separate worker.**
   - With two GPUs, resident data can be split three ways: CPU, `gpu:0`, `gpu:1`. Each GPU holds its own resident share (two pools; see `gpu-out-of-core.md`).
   - Lancelot is the reference for caching across GPUs.
   - The NVLink bridge on this box (about 56 GB/s per direction on paper, unmeasured) could cheaply copy build tables or partials between GPUs. Measure it first.
5. **Joins on host data:** do placement (F-007), not same-operator splitting. Filter on CPU, add a bitvector, then join on GPU (Li et al.). Split the probe only when both the build and the probe are resident.
6. **Don't build** unbounded dynamic GPU morsel queues, CPU+GPU builds of a shared hash table, or co-processing for scan-only shapes.

**Not known or not measured:**

- the split benefit for joins and high-cardinality group-bys on this box
- whether CPU tail stealing beats the static split in practice
- the two-GPU three-way split
- NVLink bandwidth
- Maximus' and Rosenfeld's detailed findings (full texts not read)
