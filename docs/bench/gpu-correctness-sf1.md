# GPU correctness sweep, TPC-H SF1 (F-055)

Results only: no timings, no speed claims. Every MXFrame result on `device="gpu"` is compared with
DuckDB (the reference), which is itself checked against the official TPC-H SF1 answers. Data:
`data/tpch_sf1_exact` (exact DECIMALs; the legacy float32 files hide errors, see lessons.md), engine-only
mode (each engine loads its columns first), a fresh process per (query, engine), one RTX 3090 (GPU 0),
2026-10-16. Each query runs twice in its process, and both the first and the repeat result are compared.
"Shuffled" puts every table's rows in a seeded random order (no help from TPC-H's stored order).

Reproduce:

```
pixi run python scripts/bench_user_felt.py --sf 1 --engines duckdb,mx-gpu --mode engine --gpu 0 --name gpu-correctness-sf1-stored
pixi run python scripts/bench_user_felt.py --sf 1 --engines duckdb,mx-gpu --mode engine --shuffle --gpu 0 --name gpu-correctness-sf1-shuffled
pixi run python scripts/bench_random/run.py --sf 1 --n 100 --device gpu --gpu <GPU UUID> --engines duckdb,mx --name gpu-random-sf1 [--shuffle]
```

The scoreboard CSV has a `paths` column: the `path@device` of every `compute()` the query made, from
`LazyFrame.last_compile_provenance` (the random suite's JSON has the same in `mx.paths`).

## TPC-H, 22 queries: 22/22 equal to DuckDB, stored and shuffled

| query | paths taken (one entry per compute call) | GPU kernel? | = DuckDB, stored | = DuckDB, shuffled |
|---:|---|---|:-:|:-:|
| Q1 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q2 | join_mojo_shortcut x2 | no GPU kernel (CPU only) | yes | yes |
| Q3 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q4 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q5 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q6 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q7 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q8 | fused_pipeline_gpu x1, pyarrow_shortcut x1 | GPU kernel ran | yes | yes |
| Q9 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q10 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q11 | fused_pipeline_gpu x2, pyarrow_shortcut x1 | GPU kernel ran | yes | yes |
| Q12 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q13 | fused_pipeline_gpu x3, pyarrow_shortcut x1 | GPU kernel ran | yes | yes |
| Q14 | fused_pipeline_gpu x1, pyarrow_shortcut x1 | GPU kernel ran | yes | yes |
| Q15 | fused_pipeline_gpu x2, join_mojo_shortcut x1 | GPU kernel ran | yes | yes |
| Q16 | pyarrow_shortcut x1, join_mojo_shortcut x1, fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q17 | join_mojo_shortcut x1, fused_pipeline_gpu x1, pyarrow_shortcut x1 | GPU kernel ran | yes | yes |
| Q18 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q19 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q20 | join_mojo_shortcut x1 | no GPU kernel (CPU only) | yes | yes |
| Q21 | fused_pipeline_gpu x1 | GPU kernel ran | yes | yes |
| Q22 | fused_pipeline_gpu x2 | GPU kernel ran | yes | yes |

(Every path was labelled `@gpu`, the session's device; that label does not mean the work ran there.)

How to read the paths before trusting a "yes" as GPU proof:

- `fused_pipeline_gpu`: the fused filter-and-aggregate stage ran in a GPU kernel. It is the only entry that
  proves GPU work. Joins, the filters under them and result gathers are not part of it: joins run on the
  CPU (`join_mojo_shortcut` is the join-index step in Mojo on the CPU), so a "GPU" TPC-H query is a CPU
  join feeding a GPU aggregate.
- `pyarrow_shortcut`, `join_mojo_shortcut`: no aggregate kernel; the plan ran in PyArrow or the CPU join
  engine. (A GPU session may still run an ORDER BY post-op on the GPU with `sort_topk`.)
- Q2 and Q20 ran no GPU kernel at all: their "GPU" result is a CPU result. They match DuckDB, which says
  nothing about the GPU.
- No query reported `gpu_declined` (the F-022 fallback that moves a declined aggregate to the CPU kernels),
  and none crashed or timed out.

## Random queries: 100/100 equal to DuckDB, stored and shuffled

`scripts/bench_random/` seeds 0..99, SF1, `device="gpu"`, the same seeds stored and shuffled. Verdict `ok`
for all 200 runs (rows equal within 1e-3 relative, ordered queries in ORDER BY order; none `drift`, none
wrong, no errors). The path counts are identical stored and shuffled.

| shape | queries | paths taken (queries) |
|---|---:|---|
| agg | 23 | fused_pipeline_gpu 23 |
| distinct (count distinct) | 13 | fused_pipeline_gpu 12, cpu_aot 1 (CPU kernels, the GPU declined) |
| global | 8 | fused_pipeline_gpu 8 |
| join | 35 | fused_pipeline_gpu 35 |
| select_distinct | 11 | pyarrow_shortcut 6, join_mojo_shortcut 5 (no GPU aggregate) |
| topn | 10 | pyarrow_shortcut 8, join_mojo_shortcut 2 (no GPU aggregate) |

So 78 of the 100 ran a GPU aggregate kernel; 22 ran on the CPU or PyArrow whatever the device.

## The flaky GPU sort test (F-014): memory pressure, and one real hazard behind it

`test_sort_end_to_end_single_and_multi_key_with_and_without_limit` did not fail in 14 runs here (6 as the
bare scenario, 8 through pytest, with the GPU shared). To test the memory-pressure hypothesis a side
process held all but N MB of GPU 1 while the scenario ran (3 runs each):

| free GPU memory | result |
|---:|---|
| 2.5 GB, 1.0 GB | passes |
| about 0.5 GB, before the fix | fails 3/3: `AssertionError (2000000, ['k','j'], True, 10)`, `oom log` on stderr, no exception |
| about 0.5 GB, after the fix | the wrong order is detected and redone on the CPU; the run then stops at `RuntimeError: cuMemAlloc failed: 2` from the next allocation, an honest out-of-memory |

Both suspects in the ticket were real, in one chain. The flake is memory pressure from other processes on
the shared GPU (the `cuMemAlloc 2`). And when a sort kernel's scratch allocation fails inside Mojo, it
prints `oom log` and returns garbage indices with no exception (the lessons.md entry about `sort_topk`).
That silent wrong answer was the engine bug, now fixed: the top-k result is checked in O(n)
(`sort_check.is_sorted_topk`) and redone with `np.argsort` when it is wrong
(`tests/test_gpu_sort_verified.py`). The remaining `cuMemAlloc` failure under pressure is not a bug
(`device="auto"` falls back to the CPU, `device="gpu"` raises), so the test stays strict: it needs about
1 GB of free GPU memory. Other GPU kernels may swallow an out-of-memory error the same way (scan,
group-by scratch); they were not probed.

## Shared-machine notes

- GPU 1 refused to create a device once (`CUDA_ERROR_OUT_OF_MEMORY` with only 1.6 GB in use per
  `nvidia-smi`; another process's pool probably held the rest at that moment). The sweep ran on GPU 0, and
  GPU 1 worked again minutes later. Not treated as a bug.
- Load average 16-19 during the runs. Nothing here is timed.
