# GPU correctness sweep, TPC-H SF1 (F-055)

Results only: no timings, no speed claims. Every MXFrame result on `device="gpu"` is compared with
DuckDB (the reference), which is itself checked against the official TPC-H SF1 answers. Data:
`data/tpch_sf1_exact` (exact DECIMALs; the legacy float32 files hide errors, see lessons.md), engine-only
mode (each engine loads its columns first), a fresh process per (query, engine), one RTX 3090 (GPU 0),
2026-10-16, run on the final code (every GPU kernel raises on failure, see the last section). Each query
runs twice in its process and both the first and the repeat result are compared (the repeat took the same
paths as the first in all 44 runs). "Shuffled" puts every table's rows in a seeded random order.

Reproduce:

```
pixi run python scripts/bench_user_felt.py --sf 1 --engines duckdb,mx-gpu --mode engine --gpu 0 --name gpu-correctness-sf1-stored
pixi run python scripts/bench_user_felt.py --sf 1 --engines duckdb,mx-gpu --mode engine --shuffle --gpu 0 --name gpu-correctness-sf1-shuffled
pixi run python scripts/bench_random/run.py --sf 1 --n 100 --device gpu --gpu <GPU UUID> --engines mx,duckdb --name gpu-random-sf1 [--shuffle]
```

The scoreboard CSV has `paths` and `repeat_paths` columns (the random suite's JSON has `mx.paths`): every
`compute()` the query made as `path@device xN`, then the GPU kernels it called (`gpu:<exported function>`),
plus `gpu_sort_fallback` if the backstop below had to redo a GPU sort (it never did in these runs).
Recorded by `scripts/tpch_user/provenance.py` from `LazyFrame.last_compile_provenance` and a call log in
`aot_kernels._bind_gpu`.

## TPC-H, 22 queries: 22/22 equal to DuckDB, stored and shuffled

| query | plan paths (compute calls) | work on the GPU | = DuckDB, stored | = DuckDB, shuffled |
|---:|---|---|:-:|:-:|
| Q1 | fused_pipeline_gpu x1 | aggregate | yes | yes |
| Q2 | join_mojo_shortcut x1 | join, sort, gather | yes | yes |
| Q3 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q4 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q5 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q6 | fused_pipeline_gpu x1 | aggregate | yes | yes |
| Q7 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q8 | fused_pipeline_gpu x1, pyarrow_shortcut x1 | aggregate, join, sort, gather | yes | yes |
| Q9 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q10 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q11 | fused_pipeline_gpu x1, pyarrow_shortcut x1 | aggregate, join, sort, gather | yes | yes |
| Q12 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q13 | fused_pipeline_gpu x1, pyarrow_shortcut x1 | aggregate, sort, gather | yes | yes |
| Q14 | fused_pipeline_gpu x1, pyarrow_shortcut x1 | aggregate, join, sort | yes | yes |
| Q15 | fused_pipeline_gpu x1, join_mojo_shortcut x1 | aggregate, join, sort, gather | yes | yes |
| Q16 | pyarrow_shortcut x1, join_mojo_shortcut x1, fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q17 | join_mojo_shortcut x1, fused_pipeline_gpu x1, pyarrow_shortcut x1 | aggregate, join, sort | yes | yes |
| Q18 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q19 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q20 | join_mojo_shortcut x1 | join, sort | yes | yes |
| Q21 | fused_pipeline_gpu x1 | aggregate, join, sort | yes | yes |
| Q22 | fused_pipeline_gpu x1 | aggregate | yes | yes |

The paths read like this:

- `fused_pipeline_gpu`: the fused filter-and-aggregate stage ran in a GPU kernel ("aggregate").
- `join_mojo_shortcut`, `pyarrow_shortcut`: the plan without a fused aggregate (a join, select or sort
  plan). In a GPU session every inner join goes to `AOTKernelsGPU.hash_join` (`join_engine.inner_pairs`,
  `custom_ops._execute_hash_join`): the `join_count`, `join_scatter`, `scan`, `max` and `fill` kernels build
  the pairs and `sort_topk` argsorts the right keys. So the labels say how the plan ended, not that the CPU
  did the work. ("sort" in the table is that argsort or an ORDER BY; "gather" is `gather_*` on the result.)
- `@gpu` on every path is the session's device. `cpu_aot` (a declined aggregate moved to the CPU kernels,
  F-022) was the only CPU-labelled path and appears in the random suite once, not in TPC-H.
- All 22 queries ran at least one GPU kernel. Q1, Q6, Q13 and Q22 call no join kernel; Q2 and Q20 have no
  fused aggregate and run on the join, sort and gather kernels alone. Result columns that are not 4 or 8
  byte numbers (strings, dictionaries) are gathered on the CPU, and the group-key encoding of the aggregate
  is not part of the "aggregate" entry's proof: the table says a GPU kernel ran, not that all of the query did.
- Nothing reported `gpu_declined` or `gpu_sort_fallback`; no crash, no timeout.

## Random queries: 100/100 equal to DuckDB, stored and shuffled

`scripts/bench_random/` seeds 0..99, SF1, `device="gpu"`, the same seeds stored and shuffled. Verdict `ok`
for all 200 runs (rows equal within 1e-3 relative, ordered queries in ORDER BY order; none `drift`, none
wrong, no errors). The paths are identical stored and shuffled.

| shape | queries | GPU work (queries) |
|---|---:|---|
| agg | 23 | aggregate 20, aggregate + sort + gather 3 |
| distinct (count distinct) | 13 | aggregate 6, aggregate + join + sort 5, aggregate + sort + gather 1, none 1 (`cpu_aot`: the GPU declined) |
| global | 8 | aggregate 8 |
| join | 35 | aggregate + join + sort 31, aggregate 4 |
| select_distinct | 11 | none 5 (PyArrow only), join + sort 3, join + sort + gather 2, sort + gather 1 |
| topn | 10 | sort + gather 8, join + sort + gather 2 |

94 of the 100 ran at least one GPU kernel; 6 (5 select-distinct plans in PyArrow, 1 declined count-distinct)
ran on the CPU whatever the device.

## The flaky GPU sort test (F-014): memory pressure, and the engine bug behind it

`test_sort_end_to_end_single_and_multi_key_with_and_without_limit` did not fail in 14 runs here. To test
the memory-pressure hypothesis a side process held all but N MB of GPU 1 while the scenario ran (3 runs each):

| free GPU memory | result |
|---:|---|
| 2.5 GB, 1.0 GB | passes |
| about 0.5 GB, original code | fails 3/3: `AssertionError (2000000, ['k','j'], True, 10)`, `oom log` on stderr, no exception |
| about 0.5 GB, with the status checks | raises `MemoryError: CUDA_ERROR_OUT_OF_MEMORY: sort_topk_i32_gpu ran out of GPU memory` |

Both suspects in the ticket were real, in one chain. The flake is memory pressure from other processes on
the shared GPU (the `cuMemAlloc 2`). When a kernel's scratch allocation fails inside Mojo it raised, and every
exported function in `kernels_aot_gpu.mojo` ended in `except: pass`, so the failure became garbage output
with no error: a sort returned wrong indices, a join (whose `order` buffer is a `sort_topk`) wrong rows.

The fix, for all 31 exported GPU functions (the pattern `pipeline_gpu.mojo` already had):

- each returns an Int status: 0 ok, -2 out of GPU memory, -1 any other error (`_status` in the Mojo file
  reads the exception text);
- `aot_kernels._bind_gpu` checks it for every function and raises `MemoryError` (out of memory) or
  `RuntimeError("CUDA_ERROR: ...")`; `_CUDADriver.h2d`, `d2h` and `sync` check their `CUresult`;
- `device="auto"` already reran a failed GPU plan on the CPU (F-005): a `MemoryError` falls back without
  disabling the GPU, any other `CUDA_ERROR` also disables it for the process; `device="gpu"` raises;
- `sort_check.is_sorted_topk` stays as a backstop on the GPU sort (O(n); redoes it with `np.argsort`,
  counts `gpu_sort_fallbacks`, warns once per kind of failure).

Tests: `tests/test_gpu_kernel_errors.py` injects a failing status at the library boundary for sort, masked
global sum, scan, group sum and hash join (OOM and other error: each raises), checks the hash join raises
when only its `order` sort fails, and checks `device="auto"` returns the CPU's exact result for a failed
sort (top-k and full) and a failed join while `device="gpu"` raises. `test_real_out_of_memory_raises` is the
real thing and opt-in (`MXFRAME_TEST_GPU_HOG=<MB to leave free>`, it holds most of one GPU for a few
seconds): the sort is exactly right or raises, never wrong. The suite: 2732 passed, 70 skipped, 2 xfailed.

## Shared-machine notes

- GPU 1 refused to create a device once (`CUDA_ERROR_OUT_OF_MEMORY` with only 1.6 GB in use per
  `nvidia-smi`; another process's pool probably held the rest at that moment). The sweep ran on GPU 0, and
  GPU 1 worked again minutes later. Not treated as a bug.
- Load average 16-19 during the runs. Nothing here is timed.
