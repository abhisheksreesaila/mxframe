# MXFrame API notes

The quick reference is the [API reference tables in the README](../README.md#api-reference). Every example there is run by `tests/test_readme_api.py`. This page covers what the tables leave out: arguments, inspection, SQL limits, devices and the source layout.

## Building a query

```python
import mxframe as mx
from mxframe import col, lit, when

lf = mx.scan_parquet("lineitem.parquet")   # lazy; a file or a directory of Parquet files
lf = mx.LazyFrame(arrow_table)             # or wrap a pyarrow.Table
lf = mx.sql("SELECT ... FROM t", t=table)  # or SQL: returns a LazyFrame (needs the `sql` extra)
```

Nothing runs until `.compute()`, `.to_pandas()`, `.to_polars()` or `.to_arrow()`.

`scan_parquet(path, *, dictionary="auto", max_dictionary_bytes=...)` keeps low-cardinality string columns dictionary-encoded, so filters, group-bys and joins work on integer codes and decode only the rows that reach the output.

## Joins

`lf.join(other, *, on=None, left_on=None, right_on=None, how="inner")`: the key arguments are keyword-only.

| `how` | Returns |
|---|---|
| `"inner"` | Matching pairs |
| `"left"` | Every left row; right columns null without a partner |
| `"semi"` | Left rows that have a partner, each once, left columns only (SQL `EXISTS`) |
| `"anti"` | Left rows with no partner, left columns only (SQL `NOT EXISTS`) |

Null keys never match.

## Running: `.compute()`

`lf.compute(device="auto", verbose=False, optimize=True, validate=True, *, prefer=None, placement=None, gpu_fraction=None, strict=False, profile=False)` returns a `pyarrow.Table`.

| Argument | Meaning |
|---|---|
| `device` | `"auto"` (default): estimate the query on each device from this machine's calibration and run it on the cheaper one. `"cpu"`: always the CPU. `"gpu"`: the GPU, raising `RuntimeError` if there is none. |
| `verbose` | Print optimizer passes and compile/execute timing |
| `optimize`, `validate` | Run the plan optimizer and the plan checks first (both on by default) |
| `prefer` | Auto only: `"cpu"` or `"gpu"` biases the choice without forcing it |
| `placement` | Auto only: e.g. `{"join": "gpu"}`; today any `"gpu"` value places the whole query on the GPU |
| `gpu_fraction` | Auto only: a fixed GPU share of rows when a fused aggregate is split over CPU and GPU |
| `strict` | Auto only: raise GPU errors instead of rerunning on the CPU |
| `profile` | Auto only: record the actual time next to the estimate in `last_auto_decision` |

**In 0.6.0, pass `device="cpu"`.** The native Mojo executor, which every 0.6.0 benchmark number measures, runs only under `device="cpu"`; `"auto"` runs the older fused pipeline when it picks the CPU (ticket F-053 in `docs/plan.md`).

`mx.calibrate(full=False)` measures this machine for `"auto"`. `mx.set_cpu_threads(n)` / `mx.get_cpu_threads()` (or env `MXFRAME_CPU_THREADS`) set the CPU worker budget.

## Inspecting a query

| Call | What it gives |
|---|---|
| `lf.explain(optimized=True, validate=False, include_runtime=False, device=None, costs=False)` | The plan as text; `device="auto", costs=True` adds the device decision and per-operator estimates without running it |
| `lf.last_compile_provenance` | After `compute()`: which path ran (`"native"`, `"fused_pipeline_cpu"`, `"cpu_aot"`, ...) and timings |
| `lf.last_auto_decision` | After an auto `compute()`: the chosen device, the estimates and the reason |
| `from mxframe import native_exec; native_exec.last_decline()` | Why the native executor declined the last plan, if it did |
| `lf.schema` | The output schema without computing the result (a property) |
| `lf.describe()` | count, mean, std, min, quartiles and max of the numeric columns |

The [behind-the-scenes page](https://abhisheksreesaila.github.io/mxframe/internals.html) explains each path and the environment switches.

## SQL

`mx.sql(query, **tables)` parses with sqlglot into the same plan the DataFrame API builds. Table names in the query are the keyword names.

Supported: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN … ON`, `CASE WHEN`, aggregates and arithmetic.

Known limits in 0.6.0:
- `COUNT(DISTINCT …)` raises `NotImplementedError`; use `lf.select(a, b).distinct().groupby(a).agg(col(b).count())`.
- `HAVING` is dropped without an error (F-042); filter the aggregated frame instead.

## Device selection and the GPU

- `device="gpu"` needs the MAX runtime (`pip install "mxframe[runtime]"`, or `pixi` with Modular's conda channel). The GPU path is validated on NVIDIA.
- `lf.to_gpu()` uploads the numeric columns once and returns a `GPUFrame`, so repeated queries on the same data skip the copy.
- On import, MXFrame sets `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=25` unless you set it yourself, so the Mojo runtime doesn't reserve most of the GPU's memory. The setting is process-wide.

## Tests

```bash
pixi run test        # pytest over tests/ (works from any checkout or worktree)
pixi run test-all    # the older scripts/_test_*.py suites, as CI runs them
```

## Source layout

| Path | What it holds |
|---|---|
| `__init__.py` | Public API: `LazyFrame`, `col`, `lit`, `when`, `scan_parquet`, `sql`, `calibrate`, thread settings |
| `lazy_frame.py`, `lazy_expr.py` | The plan tree (`Scan`, `Filter`, `Aggregate`, `Join`, `Sort`, ...), `LazyFrame` and `Expr` |
| `sql_frontend.py` | SQL → plan, via sqlglot |
| `io_parquet.py`, `arrow_dict.py` | Lazy Parquet sources; dictionary-encoded column helpers |
| `plan_validation.py`, `optimizer.py` | Plan checks; rewrites (filter merge and pushdown, column pruning, limits, sort flags) |
| `native_exec.py` | Front end of the native Mojo executor: packs the plan, one ctypes call into `libmxexec_cpu.so` |
| `pipeline_exec.py`, `pipeline_program.py`, `pipeline_kernels.py` | The fused filter + group + aggregate pipeline and its bytecode |
| `custom_ops.py`, `aot_kernels.py`, `join_engine.py`, `group_keys.py`, `sort_check.py` | The per-operator path: dispatch, AOT kernel bindings, joins (with join reordering), group keys, verified sort |
| `compiler.py`, `_max_lazy.py` | The MAX Graph compiler path |
| `auto_mode.py`, `calibration.py`, `coproc.py`, `coproc_fused.py`, `residency.py`, `gpu_stream.py` | `device="auto"`: the device choice, calibration, CPU+GPU co-processing, GPU copies |
| `kernels_aot/*.mojo` | Mojo sources: `executor_cpu.mojo` (native executor), `pipeline_cpu.mojo` / `pipeline_gpu.mojo`, `groupkeys_cpu.mojo`, `joinsemi_cpu.mojo`, `kernels_aot.mojo` / `kernels_aot_gpu.mojo` |
| `kernels_aot/*.so` | The compiled libraries, shipped in the wheel (`scripts/build_pipeline.sh`, `scripts/release.sh` rebuild them) |
| `scripts/` | Benchmarks (`bench_user_felt.py`, `bench_random/`), release and build scripts, older test suites |
| `docs/site/` | The docs site, deployed to GitHub Pages |
