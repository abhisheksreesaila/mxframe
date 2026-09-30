# Lessons

<!-- One line per lesson, added when you correct Claude or a mistake gets caught:
- YYYY-MM-DD · the rule that prevents it, and why
Claude reads this at the start of each session. -->
- 2026-09-29 · Mojo `parallelize` called from a `.so` loaded by ctypes segfaults (no async runtime in a host process): call `KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice` first, as in `kernels_aot/pipeline_cpu.mojo`.
- 2026-09-29 · The stdlib `Atomic` is sequentially consistent, so NVPTX fences around every atomic (float64 shared-memory accumulation ran ~100x slower than it should): use `llvm.atomicrmw` with monotonic ordering for accumulation (`_atomic_add_f64` in `kernels_aot/pipeline_gpu.mojo`). Separately, float64 compares run at 1/64 rate on GeForce: compare columns in their own type (CMPC), not widened to float64.
- 2026-09-29 · A Bool captured by a `@parameter` GPU kernel closure was read wrong (all rows went to group 0); compute such flags inside the kernel from Int arguments, and write GPU kernels as plain `def`s with explicit arguments.
- 2026-09-29 · The existing GPU sort/gather post-op rounds every float64 column to float32 (sums off by 1 part in 1e8); sort small aggregate results on the CPU.
- 2026-09-30 · GPU 1 is shared with other agents (fixed launch cost jumps 0.1 ms to 6.7 ms when busy): compare kernel variants back to back in one script and quote `_repeat` steady-state numbers, never single runs.
- 2026-09-30 · Comparing float32 data with a decimal literal in float64 drops every row equal to the literal (TPC-H Q6 lost 54% of its revenue on CPU): a float32 column takes the literal as float32, like DuckDB, Polars and NumPy 2.
- 2026-09-30 · A cache keyed by `id(obj)` must hold a reference to `obj` and check identity on lookup; ids are reused after garbage collection, which returned another table's results (found in F-010 review).
- 2026-09-30 · Judge correctness against DuckDB, never against our own CPU path: GPU Q6 (8.865e7) looked wrong next to CPU (4.07e7), but DuckDB gives 8.865e7 — the CPU path dropped boundary rows comparing float32 columns to float64 literals. GPU paths not yet verified against DuckDB go in `auto_mode.KNOWN_BROKEN_GPU`.
