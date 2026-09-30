# How `device="auto"` works

MXFrame can run a query on the CPU, on the GPU, or on both at once. With
`compute(device="auto")` the engine decides, per query, from numbers it measured on
your machine. This page explains the algorithm we built during the auto-mode
work and the measurements that shaped it.

**Visual version:** [docs/auto-mode.html](https://htmlpreview.github.io/?https://github.com/abhisheksreesaila/mxframe/blob/main/docs/auto-mode.html)
(the same content with drawn diagrams; the link renders the file from the repository).

All numbers below come from the reference machine: AMD Ryzen 9 5900X (24 threads,
dual-channel DDR4) and an NVIDIA RTX 3090 on PCIe 4.0 x8, TPC-H data at scale factor 10.

## The short version

- **Not what it does:** stream every query's rows through one shared queue to the CPU
  and the GPU at once. That scheduler exists and is tested (`coproc.run_router`), but on
  this hardware it lost: moving rows from host memory over the PCIe link is slower
  than all CPU cores reading them in place.
- **What it does:** picks the cheaper device for each query from measured costs, runs a
  table's first scan on the CPU, copies that table's columns to the GPU in the
  background, and on later queries splits the rows: the GPU takes the slice it already
  holds while every CPU core takes the rest, at the same time.

## 1. Every query: plan, price, place

```mermaid
flowchart LR
    Q[your query] --> O[optimizer<br/>filters below joins<br/>only needed columns<br/>join order by size]
    O --> C[cost model<br/>T = α + c·rows<br/>+ bytes ÷ link GB/s<br/>+ GPU start-up left]
    C -->|most queries| CPU[CPU plan<br/>fused kernel, 24 threads<br/>may co-process]
    C -->|GPU wins by ≥ 10%| GPU[GPU plan<br/>AOT kernels, no JIT]
    GPU -.->|GPU error| CPU
```

- Costs come from `mx.calibrate()`: a fit of time against rows for every kernel on
  each device, the link bandwidth, and the GPU's start-up time. Every run refines the fits.
- Inputs under 200K rows go straight to the CPU; the GPU's fixed costs can't pay off there.
- A GPU error reruns the query on the CPU. A CUDA fault switches the GPU off for the rest
  of the process. A GPU path that hasn't been verified against DuckDB is never chosen.

## 2. One session: how the GPU gets involved

The GPU pays off once the data already lives there. So:

1. **First query on a table:** runs on the CPU. After it returns, the table's columns
   are copied to the GPU in the background. The copy pauses while any other query runs.
2. **Second query:** splits the rows between the devices, paying the GPU's one-time
   warm-up.
3. **Later queries:** split the rows with no copy over the link at all.

| Run (SF10 Q1, one session) | 1 | 2 | 3 and later |
|---|---:|---:|---:|
| CPU only | 177 ms | 58 ms | 58–65 ms |
| `auto` | 185 ms | 71–73 ms | **43–49 ms** |

## 3. The split: one slice each, then merge

```mermaid
flowchart LR
    R[60M rows<br/>columns resident on the GPU] --> G[GPU: rows 0–36M<br/>one kernel call]
    R --> P[CPU: rows 36M–60M<br/>128K-row morsels on 24 threads]
    G --> M[merge<br/>add per-group sums, counts, min/max]
    P --> M
```

An aggregation reads each row once and keeps a running total per group, so any slice
of rows can run on either device and the totals simply add up. Auto mode picks the
GPU's share so both devices finish together, starting from the calibration and
refining it from each run's measured times.

## 4. Why this design, in numbers

The fused aggregation stage of SF10 Q1 (60M rows):

| Strategy | Time |
|---|---:|
| CPU only, 24 threads | 57 ms |
| GPU only, data in host memory | 353 ms |
| Morsel queue streaming host → GPU | 67–160 ms |
| GPU only, data resident on the GPU | 39 ms |
| Morsel queue, data resident | 42 ms |
| **Static split, data resident** | **25 ms** |

- **The link is the bottleneck.** Host memory feeds the CPU cores at roughly 40–50 GB/s;
  the GPU's PCIe 4.0 x8 link carries about 10–13 GB/s.
- **Once columns are resident, splitting beats both devices:** 25 ms against 57 on the
  CPU and 39 on the GPU.
- **One call per device beats many small GPU morsels.** Each GPU call costs about 4.5 ms
  of fixed overhead, so one static slice wins over a dynamic queue.

## What's not in auto mode yet

- **Joins on the GPU.** Joins run on the CPU through the join engine; placing large joins on
  the GPU is planned.
- **Streaming co-processing for first scans.** Not worth it on PCIe x8. It becomes
  interesting on PCIe 5 or unified-memory machines (Grace Hopper, Apple silicon), where
  the scheduler already exists.
- **A native Parquet reader.** Reading uses PyArrow for now; on raw scans that is the
  largest remaining gap to Polars.

The full design, with the research it draws on, is in
[docs/spec-auto-mode.md](spec-auto-mode.md).
