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

## Scalar subqueries

`lf.scalar(agg)` is SQL's `(SELECT agg FROM lf)`: an expression for another plan's `filter`, `with_columns` or HAVING. `agg` is one `sum`, `min`, `max`, `mean` or `count`.

```python
total = base.scalar((col("cost") * col("qty")).sum())
big = (base.groupby("part").agg((col("cost") * col("qty")).sum().alias("value"))
       .filter(col("value") > total * lit(0.0001)))       # HAVING value > (SELECT sum(..) * 0.0001 ...)
```

**NaN and infinity (F-070).** Float comparisons order values as DuckDB does: NaN is above every number, `inf` included, and NaN = NaN. So `x > 1`, `x >= 1` and `x <> 1` keep a NaN row, `x = float("nan")` matches it, `x <> x` is false for it, and `BETWEEN`, `IN` (`-0.0` equals `0.0` too), `NOT`, `HAVING` and a filter over an aggregate result follow; `MIN` skips NaN (NaN only for a group of nothing else), `MAX` is NaN when the group holds one, and `ORDER BY` puts NaN last ascending and first descending. NULL stays NULL: a comparison with it keeps no row. The same on every path (native executor, fused pipeline, GPU, Arrow). SQL writes the constants as `CAST('nan' AS DOUBLE)`, `CAST('inf' AS DOUBLE)`, `CAST('-inf' AS FLOAT)`; the DataFrame API takes Python `float("nan")` / `math.inf`. Integer comparisons are untouched, and against a constant that is not NaN only `x > c`, `x >= c` (and `c < x`, `c <= x`) run the NaN-aware instruction.

**NULL tests.** `col("x").is_null()` (and `~col("x").is_null()`) is `IS [NOT] NULL`; a float NaN is a value, not NULL. SQL: `IS [NOT] NULL`, `IS [NOT] TRUE / FALSE`, `IS [NOT] DISTINCT FROM`, also on the right key of a `LEFT JOIN` (the anti-join pattern). The native filter has no validity test yet, so these run on the Arrow path.

**String patterns.** `col("s").like(pattern, escape=None, ignore_case=False)`, `.ilike(pattern)`, `.startswith(p)`, `.endswith(p)`, `.contains(p)`, `.isin([...])` and `== "x"` / `!= "x"` run natively on plain utf8 columns (Mojo over the Arrow offsets and bytes, in the filter program) and on dictionary columns (the pattern is tested once per dictionary entry, then the codes). As in DuckDB there is no escape character unless `escape` names one, `%` is any run of characters and `_` one UTF-8 character; a NULL string matches neither the pattern nor its negation. `ILIKE` is native on ASCII dictionaries only (case folding of other text is DuckDB's own) and otherwise answered by Arrow; `isin` over a plain column is native up to 32 values. SQL: `LIKE`, `NOT LIKE`, `ILIKE`, `ESCAPE`. An aggregate that reads a string column holding NULLs still takes the null-aware Arrow path first.

`compute()` runs each subquery first as a native plan of its own and puts its value in the expression as a literal, so the main plan is a plain filter or HAVING and needs no row-level Python in between. A subquery over no rows is NULL: a comparison with it keeps no rows (as in SQL, also in an `AND` / `OR`, and under a `NOT` of a plain comparison); a `NOT` over an `AND` / `OR` that contains the NULL raises `ValueError`, since it would need three-valued logic over rows. When the subquery aggregates an aggregate plan that the outer plan also contains (TPC-H Q15 with an uncomputed view: `view.filter(col("tot") == view.scalar(col("tot").max()))`), that plan is computed once and both sides read the same table, so the equality is exact whatever the float sums' last bits. Two separately written but equal plans are matched by structure and by the identity of their in-memory tables.

## Running: `.compute()`

`lf.compute(device="auto", verbose=False, optimize=True, validate=True, *, prefer=None, placement=None, gpu_fraction=None, strict=False, profile=False, reproducible=None)` returns a `pyarrow.Table`.

| Argument | Meaning |
|---|---|
| `device` | `"auto"` (default): estimate the query on each device from this machine's calibration and run it on the cheaper one. `"cpu"`: always the CPU. `"gpu"`: the GPU, raising `RuntimeError` if there is none. A GPU piece (fused aggregate, join, sort, gather, string filter, masked global aggregate) that does not fit the GPU's free memory, or runs out of it, reruns on the CPU with a `RuntimeWarning` (see `strict` and below). |
| `verbose` | Print optimizer passes and compile/execute timing |
| `optimize`, `validate` | Run the plan optimizer and the plan checks first (both on by default) |
| `prefer` | Auto only: `"cpu"` or `"gpu"` biases the choice without forcing it |
| `placement` | Auto only: e.g. `{"join": "gpu"}`; today any `"gpu"` value places the whole query on the GPU |
| `gpu_fraction` | Auto only: a fixed GPU share of rows when a fused aggregate is split over CPU and GPU |
| `strict` | Raise GPU errors instead of rerunning on the CPU; for a GPU piece that does not fit or runs out of memory this is a `MemoryError` that names the piece, its estimate and the free budget (`device="gpu"` and `"auto"`) |
| `profile` | Auto only: record the actual time next to the estimate in `last_auto_decision` |
| `reproducible` | `True` (or env `MXFRAME_REPRODUCIBLE=1`; `False` overrides the variable) makes float `SUM` / `MEAN` on the native CPU paths bit-identical from run to run whatever the number of groups; see below |

**CASE and `when()` (F-105).** `when(cond, then, otherwise)` and SQL `CASE WHEN` give NULL when no branch matches and there is no ELSE: `when(p, 1)` is `CASE WHEN p THEN 1 END`, not `ELSE 0` (older versions used 0; pass `0` explicitly for that). A NULL condition is not true. The result has the common type of the branches as DuckDB types it: equal types stay, integers widen to int64, `float32` next to an integer or a DECIMAL literal stays `float32` (only a DOUBLE, an exponent literal or a float64 column, widens it), any other number mix is float64, `string` and `large_string` are one family (a dictionary column is read as its values), dates stay dates, a NULL branch takes the other branch's type, all-NULL is int32; text mixed with numbers raises. `sum(case when .. then x else 0 end)` stays on the native executor; a CASE that can be NULL is answered by the Arrow NULL-aware path.

**Reproducible float sums.** Float addition is not associative, so the order in which a group's rows are added decides its last bits. The native CPU executor defines one order: for each 65,536-row morsel (counted from the first row) a group's rows are added in row order into a sub-total, and the sub-totals are added in morsel order. That order does not depend on the number of threads, on which worker ran a morsel, or on the other aggregates in the query. **By default** the defined order runs only where it is as fast as before or faster: a global aggregate (no group keys) and a fused join + aggregate with 16,384 or more groups (up to 2x faster than the shared atomic output it replaces). Every other aggregate keeps the previous fastest code (a partial table per worker, or a shared output with atomic adds), whose float sums can differ in the last bits between two runs; a table per morsel measured 10-20% slower for them. **For a guarantee on every aggregate, pass `reproducible=True`.** `reproducible=True` forces the defined order on every native CPU path, including the older pipeline when the native executor is off (`MXFRAME_NATIVE=0`) and the CPU share of a CPU+GPU split (only for an explicit `gpu_fraction`; on the GPU: see below): 0-20% slower on keyed group-bys (about +0.5 ms on 6M rows at 1k-8k groups, up to +20% on many groups), no cost for global aggregates; a table per morsel is used up to 131,072 groups (a wave of partial tables inside 128 MB, folded into the output in morsel order, at any number of rows; about 50 or more sums at 8,192 groups outgrow that memory bound and are staged instead), more groups are staged per morsel. Guaranteed in that mode: the same input table (same chunking and filter) gives the same bits. Not covered: equality with another engine's or a sequential sum (up to 1,024 groups each group keeps 4 accumulator lanes, a global aggregate one per SIMD slot), the GPU kernel's order (it is fixed run to run, but is not the CPU's order: the same query on `cpu` and on `gpu` can differ in the last bits), and the per-call CPU kernels across different thread counts. **On the GPU** (`device="gpu"`, and the GPU share of a CPU+GPU split) `reproducible=True` adds without atomics: each block keeps its group table in shared memory and the thread owning a group adds that group's rows in row order, then the blocks' tables are added in block order; a split merges the GPU share (first rows) and the CPU share in row order. The ordered table holds (slots + 1) x groups <= 4,096 (about 1,300 groups with two sums); with more groups the GPU declines and the CPU's defined order runs, so the bits stay fixed (the provenance path says `fused_pipeline_cpu`). The default mode keeps the faster atomics for these shapes, whose last bits can differ between runs. With `device="gpu"` a GPU memory shortfall (the aggregate is halved into smaller chunks, or falls back to the CPU: `_gpu_aggregate`) changes how rows are grouped into partials, so the bits can change with the free GPU memory. `device="auto"` with `reproducible=True` runs the CPU's defined order (no GPU, no CPU+GPU split: the split fraction is refined by every run and would change the bits); the GPU kernel applies when you ask for `device="gpu"`, and a split only with an explicit `gpu_fraction`. The ordered kernel's cost on large inputs is not measured yet. The flag travels with the call (a per-thread context variable and a plan word), so threads computing at once in different modes do not affect each other; `MXFRAME_REPRODUCIBLE` is only read as the default. A scalar subquery over an aggregate plan the outer plan also contains (`view.filter(col("t") == view.scalar(col("t").max()))`) shares one materialized result, so such a self-comparison is exact in every mode. Only plans that end in an aggregate (one row per group, possibly filtered, projected, sorted or limited) are shared, so nothing large is materialized early, and the shared plan is computed in full (the optimizer's column pruning and filter pushdown do not reach into it). A CTE used twice in a `FROM` (a self-join) is still computed twice: in the default mode a join of the CTE with itself on a float sum can miss rows; with `reproducible=True` both copies have the same bits.

**Data bigger than GPU memory.** Each GPU piece estimates its device memory first; over 90% of the GPU's free memory it runs on the CPU. A piece that runs out of memory anyway is retried once with half the rows per chunk (aggregates), then rerun on the CPU, and the shape is remembered until clearly more (1.5x) memory is free. `lf.last_compile_provenance["gpu_pieces"]` and `lf.explain(include_runtime=True)` (`gpu_pieces=...`) say which piece left the GPU and why; under `"auto"`, `lf.last_auto_decision.explain()` lists them as `gpu piece:` lines. `MXFRAME_GPU_MEM_BUDGET` (for example `512M`, or `0:512M,1:2G` per GPU) fakes a smaller GPU for tests. A join whose output passes 2^31 rows raises `MemoryError` on every device (a size limit, not GPU memory).

**Data bigger than RAM.** A `scan_parquet` query whose columns (after row-group pruning) are bigger than the host budget runs in batches of row groups and merges per-batch partial aggregates, so memory is one batch plus the group state, not the file. The budget is `MXFRAME_HOST_MEM_BUDGET` (for example `64M`, `2G`), by default 70% of `MemAvailable` (50% of `MemTotal` with `reproducible=True`, so the batches do not move with other processes). Each batch is a normal `compute(device)`: GPU where the batch fits (`MXFRAME_GPU_MEM_BUDGET` fakes a smaller GPU), otherwise the CPU native executor; small dimension tables joined to the big file are read once and stay resident. Supported: `sum`, `mean`, `count`, `min`, `max` over a Filter / Project / Join pipeline, with Sort / Limit / Filter above the aggregate. `lf.last_compile_provenance["path"] == "streamed"`, with `["streamed"]` giving `batches`, `row_groups`, `row_groups_read`, `devices`, `group_state_rows` and the budgets. Not yet (F-056 step 3, the plan warns and runs in memory): plans that return rows without an aggregate, a Sort / Limit / Distinct below the aggregate, count distinct, the other tables of the aggregate bigger than a third of the budget (big-to-big joins); group state over half the budget raises `MemoryError`. `MXFRAME_STREAM=0` turns it off.

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

Supported: `SELECT`, `FROM` (tables, `WITH` names, derived tables), `WHERE`, `GROUP BY` (also `ALL`), `HAVING`, `ORDER BY`, `LIMIT`, `JOIN … ON` / `USING` (inner, left, semi, anti), comma joins (`FROM a, b WHERE a.k = b.k`: the key equalities in `WHERE`, also one every branch of an `OR` has), `CASE WHEN`, aggregates (also `COUNT(DISTINCT x)`) and arithmetic, `DATE '…'` with `± INTERVAL 'n' DAY | MONTH | YEAR` (a day number: dates are int32 days since 1970-01-01, as in the TPC-H Parquet files), `EXTRACT(YEAR FROM x)`, `SUBSTRING(x FROM a FOR n)`, `LIKE 'prefix%'` and `LIKE '%substring%'`.

Subqueries (F-065):
- `(SELECT agg FROM …)` in `WHERE`, `HAVING` or the `SELECT` list is `lf.scalar(agg)`: run first as a native plan, used as a literal. It must be one aggregate over an ungrouped query.
- `x [NOT] IN (SELECT …)` and `[NOT] EXISTS (SELECT …)` as conditions joined by `AND` at the top of `WHERE` become semi / anti joins. `NOT IN` follows SQL's NULL rules (a NULL in the subquery, or a NULL `x` against a non-empty subquery, keeps no row; an empty subquery keeps every row).
- A correlated comparison in `WHERE` (`v > (SELECT AVG(v) FROM u WHERE u.k = t.k)`) is a join with the per-key aggregate. The correlation must be `inner.col = outer.col` equalities; `COUNT` is refused (an outer row without partners must see 0).
- `WITH name AS (…)` (not `RECURSIVE`, no column list).

`JOIN … ON` (F-109, F-096): each side of an `=` belongs to the tables joined so far or to the table being joined by its table qualifier (`x.v = t.k` and `t.k = x.v` are the same; a bare name must be in exactly one side, else `ValueError` "ambiguous"). A side may be an expression over one side (`ON t.a + 1 = u.b`): it is computed into a key column first. Any other AND-ed condition is a residual: in an INNER join it filters after the join; in a LEFT join it decides which right rows match, so a left row whose partners all fail it comes back once with NULLs (a test on the right table alone filters the right side, one on the left table alone is one more join key, one on both runs the join, then keeps the satisfying rows and one blanked row for each left row with none). A join with no `=` between the tables (`ON t.x > u.y`, `CROSS JOIN`, `FROM a, b WHERE a.x > b.y`) is a cross product filtered by the condition, refused (`NotImplementedError`, "no equality") unless both inputs are known to hold at most 4,000,000 row pairs together. Not supported: a subquery in an `ON`, a residual in a `SEMI` / `ANTI JOIN`, `RIGHT` / `FULL JOIN`.

Everything else raises `NotImplementedError` or `ValueError` naming the form: `IN` / `EXISTS` under `OR`, `NOT (…)`, `CASE` or in the `SELECT` list, a correlated scalar subquery anywhere but a `WHERE` comparison, a correlated `<>` or `<`, a correlated `NOT IN`, `UNION`, `ANY` / `ALL`, `RIGHT` / `FULL JOIN`, `LIKE '%suffix'` and patterns with an inner `%` or `_`, `EXTRACT` of anything but the year, other `INTERVAL` units.

Known limits: `EXTRACT(YEAR …)` on an integer day-number column works in a projection or group key, not in `WHERE` (the same `year_of_days()` limit as the DataFrame API). Integer literals in arithmetic are floats (`y + 1` is a float64), so `IN (SELECT y + 1 …)` against an integer column fails with a join key type error. A CTE used twice is computed twice (TPC-H Q15 compares two float sums: F-064). `scripts/tpch_sql/run.py` runs the specification's text of the 22 TPC-H queries through `mx.sql` and compares them with DuckDB.

## Window functions

`expr.over(*partition_by, order_by=None, descending=False)` follows DuckDB on NULL, NaN and ties
(F-084): min / max aggregate the valid values only (NULL when none; a NaN follows the F-070 rule),
rank / dense_rank tie NaNs, -0.0 with 0.0 and NULLs with each other, and in an ORDER BY NaN is the
largest value (last ascending, first descending) while NULL is last in both directions. The result
is always float64: DuckDB returns int64 for rank / row_number, the input type for integer min / max
and decimal for integer sum. Sum and mean skip NULLs (NULL when none are valid), a running sum is NULL until its first valid
value, and a NaN among the valid values propagates. `count` is not a window function here.

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
