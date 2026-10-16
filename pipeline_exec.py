"""Dispatch of aggregate plans onto the fused pipeline kernels (F-008).

``try_fused_aggregate`` is the single entry point used by
``CustomOpsCompiler.compile_and_run``. For a plan of the shape
``Aggregate(Filter*(Scan))`` whose predicate and aggregate expressions compile
to bytecode (see :mod:`mxframe.pipeline_program`) it runs filter, arithmetic
and aggregation in one vectorized pass over the raw columns on the compiler's
current device, and assembles the same ``pa.Table`` the existing paths build.
Anything else returns ``None`` and the caller carries on with its existing
path, so an unsupported expression can never produce a wrong result.

Set ``MXFRAME_FUSED_PIPELINE=0`` to switch the fused path off (A/B timing).
"""

from __future__ import annotations

import os
import time as _time
from typing import Any, List, Optional

import numpy as np
import pyarrow as pa
from .np_arrow import array as _npa
from ._lazy_pc import pc

from . import gpu_budget
from .lazy_frame import Aggregate, Filter, Scan
from .pipeline_program import (
    ChunkedColumns, PipelineProgram, Unsupported, compile_pipeline, finalize, merge_partials,
)

# Rows per GPU call: bounds device memory for the resident columns of one call.
GPU_CHUNK_ROWS = 1 << 25
# Post-ops (sort/limit) on results up to this many rows run on the CPU.
SMALL_RESULT_ROWS = 1 << 16


def enabled() -> bool:
    return os.environ.get("MXFRAME_FUSED_PIPELINE", "1") not in ("0", "false", "off")


def _shape(plan):
    """``(predicates, aggregate, scan)`` for ``Aggregate(Filter*(Scan))``, else None."""
    if not isinstance(plan, Aggregate):
        return None
    preds: List[Any] = []
    node = plan.input
    while isinstance(node, Filter):
        preds.append(node.predicate)
        node = node.input
    if not isinstance(node, Scan):
        return None
    return preds, plan, node


def _run_gpu(pk, prog: PipelineProgram, cc: ChunkedColumns, gids, n_groups: int, n_rows: int,
             chunk_rows: int = GPU_CHUNK_ROWS):
    parts = []
    for s in range(0, n_rows, chunk_rows):
        parts.append(pk.run_pipeline_gpu(prog, cc, gids, n_groups, s, min(s + chunk_rows, n_rows)))
    return merge_partials(prog, parts)


def _gpu_aggregate(pk, prog: PipelineProgram, cc: ChunkedColumns, gids, n_groups: int, n_rows: int):
    """The GPU piece of a fused aggregate (F-056): a chunk that does not fit the GPU's free memory, or that
    runs out of it, is halved once, then the whole aggregate runs on the CPU. Returns (partials, device)."""
    chunk = max(min(GPU_CHUNK_ROWS, n_rows), 1)
    row_bytes = sum(b.itemsize for b in cc.chunks[0]) + (4 if n_groups > 1 else 0) if cc.chunks else 4
    on_cpu = []

    def cpu():
        on_cpu.append(True)
        return pk.run_pipeline_cpu(prog, cc, gids, n_groups, 0, n_rows)

    partials = gpu_budget.run_piece(
        "aggregate", gpu_budget.aggregate_bytes(row_bytes, chunk, n_groups, prog.n_slots),
        lambda: _run_gpu(pk, prog, cc, gids, n_groups, n_rows, chunk), cpu,
        shape=(n_rows, n_groups, prog.n_slots, tuple(prog.columns), row_bytes), retry=lambda: _run_gpu(pk, prog, cc, gids, n_groups, n_rows, (chunk + 1) // 2))
    return partials, "cpu" if on_cpu else "gpu"


def _decline(compiler, reason: str):
    """Remember why a plan was not served (``compiler.last_fused_decline``) and let the caller continue."""
    compiler.last_fused_decline = reason
    return None


def _table_of(plan):
    """``(predicates, aggregate, table)`` of an ``Aggregate(Filter*(Scan))`` with rows and aggregates, else why not."""
    shape = _shape(plan)
    if shape is None:
        return (f"plan is not Aggregate(Filter*(Scan)): {type(plan).__name__}"
                f" over {type(getattr(plan, 'input', None)).__name__}")
    predicates, agg_node, scan = shape
    table: pa.Table = scan.table
    if table.num_rows == 0 or not agg_node.aggs:
        return "empty input or no aggregates"
    return predicates, agg_node, table


def _bind(predicates, agg_node, table):
    """``(program, chunked columns)`` the fused pipeline runs on this table, else why it declines: the one
    place that knows which tables and expressions it takes (``split_check`` asks it too)."""
    n_rows = table.num_rows
    prog = compile_pipeline(predicates, list(agg_node.aggs), table)
    if prog is None:
        return "expression not supported by the bytecode compiler"
    if _int64_past_2_53(table, prog.columns):
        return "an int64 column holds values past 2^53 (the pipeline computes in float64)"
    try:
        cc = prog.bind_chunks(table)
    except Unsupported:
        return "a column has nulls or an unsupported type"
    if cc.n_cols and cc.n_rows != n_rows:
        return "column lengths differ"
    return prog, cc


def try_fused_aggregate(compiler, plan, having_pred, post_ops, optimizer_trace=None,
                        t0: Optional[float] = None) -> Optional[pa.Table]:
    if not enabled():
        return _decline(compiler, "disabled (MXFRAME_FUSED_PIPELINE=0)")
    got = _table_of(plan)
    if isinstance(got, str):
        return _decline(compiler, got)
    predicates, agg_node, table = got
    n_rows = table.num_rows
    try:
        from . import pipeline_kernels as pk
    except Exception:
        return _decline(compiler, "pipeline kernels unavailable")

    if getattr(compiler, "_device_arg", None) == "auto":
        compiler._maybe_switch_device(n_rows)
    device = compiler._session_device
    if device == "gpu" and not pk.gpu_available():
        return _decline(compiler, "libmxpipeline_gpu.so not built")
    if not pk.cpu_available():
        return _decline(compiler, "libmxpipeline_cpu.so not built")

    bound = _bind(predicates, agg_node, table)
    if isinstance(bound, str):
        return _decline(compiler, bound)
    prog, cc = bound

    t_start = t0 if t0 is not None else _time.perf_counter()
    group_keys: List[str] = []
    unique_key_cols = {}
    gids = None
    n_groups = 1
    grouped = bool(agg_node.group_by)
    if grouped:
        gk_table = table
        derived = []
        for i, e in enumerate(agg_node.group_by):
            if e.op == "col" and e._alias in (None, e.args[0]):
                group_keys.append(e.args[0])
            else:  # derived key, e.g. year(col), or a renamed column: same treatment as the existing path
                name = e._alias or f"__gk_{len(group_keys)}__"
                # evaluated against the source table; a key that reuses a source column's name
                # (``with_columns((k + 10).alias("k"))`` inlined into the aggregate) replaces it
                derived.append((name, table.column(e.args[0]) if e.op == "col"
                                else compiler._eval_expr_arrow(e, table)))
                group_keys.append(name)
        for name, arr in derived:
            if name in gk_table.column_names:
                gk_table = gk_table.set_column(gk_table.schema.get_field_index(name), name, arr)
            else:
                gk_table = gk_table.append_column(name, arr)
        # On a GPU session the existing GPU key-encoding kernels are avoided: the single-int-key
        # one hangs (even at 1000 rows) and the multi-key one corrupts the heap (a later
        # pq.read_table segfaults). The group ids must come back to the host for the fused
        # kernel anyway, so the CPU encoder costs nothing extra.
        gids, n_groups, unique_key_cols = compiler._build_group_ids_cached(
            gk_table, group_keys, gpu_compiler=None if device == "gpu" else compiler)
        if n_groups <= 0:
            return _decline(compiler, "no groups")
        gids = np.ascontiguousarray(gids, dtype=np.int32)
        if len(gids) != n_rows:
            return _decline(compiler, "group ids do not cover the rows")

    used = "cpu"
    partials = None
    coproc_info = None
    cpu_range = None
    if device == "cpu":
        from . import coproc_fused
        if coproc_fused.current_policy() is not None:
            # Under the co-processing policy the native executor declined the whole plan (it must not run
            # beside the GPU's share), so it takes the CPU rows here: its range of a split, or all of them.
            from . import native_exec
            repro = native_exec.reproducible()      # read here: the router's worker threads do not inherit it

            def cpu_range(a, b):
                return _cpu_rows(compiler, pk, prog, cc, table, predicates, agg_node, gids, n_groups, a, b, repro)
        stable = not grouped or coproc_fused.stable_group_keys(agg_node.group_by)
        split = coproc_fused.try_split(
            prog, cc, gids, n_groups, n_rows, table, stable_gids=stable, cpu_rows=cpu_range,
            gid_fp=(_gid_fingerprint(table, unique_key_cols, n_groups)
                    if grouped and coproc_fused.current_policy() is not None else None),
            gkeys=tuple(group_keys))
        if split is not None:
            partials, used, coproc_info = split
    if device == "gpu":
        try:
            partials, used = _gpu_aggregate(pk, prog, cc, gids, n_groups, n_rows)
        except pk.GpuUnsupported:
            partials = None            # this program does not fit the GPU kernel: CPU kernel
        # any other GPU error propagates, so auto_mode sees it (is_sticky_gpu_error / disable_gpu)
    if partials is None:
        partials = (cpu_range or (lambda a, b: pk.run_pipeline_cpu(prog, cc, gids, n_groups, a, b)))(0, n_rows)

    present, cols = finalize(prog, partials, grouped)
    if grouped:
        keep = None if bool(present.all()) else _npa(present)
        key_arrays = []
        for k in group_keys:
            v = unique_key_cols[k]
            key_arrays.append(v if keep is None else pc.filter(v, keep))
        # counts stay exact int64 (float32/float64 would saturate at 2**24 / lose exactness)
        agg_arrays = [_npa(cols[name].astype(np.int64 if op == "count" else np.float64))
                      for name, op, _ in prog.outputs]
        result = pa.Table.from_arrays(key_arrays + agg_arrays,
                                      names=group_keys + [name for name, _, _ in prog.outputs])
    else:
        arrays = {}
        for name, op, _ in prog.outputs:
            typ = pa.int64() if op == "count" else pa.float64()
            arrays[name] = _npa(cols[name], type=typ)
        result = pa.table(arrays)

    if having_pred is not None:
        result = pc.filter(result, compiler._eval_predicate(having_pred, result))
    if post_ops:
        # Aggregate results are small: sort/limit them on the CPU. The GPU sort path
        # gathers every column through float32, which would round the float64 sums.
        session_device = compiler._session_device
        if result.num_rows <= SMALL_RESULT_ROWS:
            compiler._session_device = "cpu"
        try:
            result = compiler._apply_post_ops_custom(result, post_ops)
        finally:
            compiler._session_device = session_device
    compiler.last_compile_provenance = {
        "device": {"split": "cpu+gpu", "gpu_resident": "gpu"}.get(used, used),
        "rows": int(result.num_rows),
        "grouped": grouped,
        "n_groups": int(n_groups) if grouped else 0,
        "cache_hit": False,
        "compile_ms": 0.0,
        "execute_ms": round((_time.perf_counter() - t_start) * 1000, 1),
        "gpu_input_path": "fused_pipeline",
        "optimizer_trace": list(optimizer_trace or []),
        "path": f"fused_pipeline_{used}",
    }
    if coproc_info is not None:
        compiler.last_compile_provenance["coproc"] = coproc_info
    return result


def _cpu_rows(compiler, pk, prog, cc, table, predicates, agg_node, gids, n_groups, start, end, reproducible):
    """CPU partials of rows ``[start, end)``: the native executor's (``native_exec.range_partials``), or the
    older pipeline kernel's when the executor declines the range. Runs on a router worker thread, so the
    caller's ``reproducible`` mode (F-064) is set here for the executor and passed to the kernel."""
    from . import native_exec
    tok = native_exec._reproducible.set(reproducible)
    try:
        got = native_exec.range_partials(compiler, prog, table, predicates, agg_node, gids, n_groups, start, end)
    except Exception as exc:
        got = native_exec.swallowed(exc, "split share")
    finally:
        native_exec._reproducible.reset(tok)
    if got is not None:
        return got
    return pk.run_pipeline_cpu(prog, cc, gids, n_groups, start, end, reproducible=reproducible)


def _gid_fingerprint(table, unique_key_cols, n_groups):
    """What the group ids of a table read from a file mean (see ``coproc_fused.gid_fingerprint``), or None
    for a table that is not a file read (its residency is keyed by buffer address, which fixes the ids)."""
    from .io_parquet import table_source
    if table_source(table) is None:
        return None
    from . import coproc_fused
    return coproc_fused.gid_fingerprint(unique_key_cols, n_groups)


class SplitInputs:
    """What ``split_check`` found for a plan the CPU+GPU split can take."""

    def __init__(self, prog, cc, table, agg_node, source, gkeys):
        self.prog, self.cc, self.table, self.agg_node = prog, cc, table, agg_node
        self.source, self.gkeys = source, gkeys
        self.columns = frozenset(prog.columns)


def split_check(plan, full: bool = True):
    """``SplitInputs`` when the CPU+GPU split can take ``plan``, else the reason it cannot. This is the one
    predicate for the split: it asks the same ``_shape`` / ``_bind`` the fused pipeline runs, then adds what
    only the split needs (stable group ids, columns whose buffers a device copy can be keyed by). Joins,
    a Project or Distinct between the aggregate and the scan, nulls, int64 past 2^53 and shapes the
    bytecode compiler declines all come out of those, so a split is never predicted for a plan the pipeline
    then declines (F-053 review).

    ``full=False`` stops after the program is compiled (shape, stable keys, the bytecode compiler): enough to
    name the columns and ask whether they are resident, and it reads no column. The rest (nulls, int64 past
    2^53, integer sums, binding the buffers) passes over the data, so it runs only for the background
    upload and for the run itself; a query whose columns are resident passed it when they were uploaded."""
    from . import coproc_fused
    from .custom_ops import CustomOpsCompiler
    from .io_parquet import table_source

    node = plan
    while not isinstance(node, Aggregate):
        if getattr(node, "input", None) is None:
            return "no aggregate to split"
        node = node.input
    node = CustomOpsCompiler._inline_project_into_aggregate(node)     # as compile_and_run does first
    if not enabled():
        return "disabled (MXFRAME_FUSED_PIPELINE=0)"
    got = _table_of(node)
    if isinstance(got, str):
        return got
    predicates, agg_node, table = got
    if agg_node.group_by and not coproc_fused.stable_group_keys(agg_node.group_by):
        return "derived or renamed group keys get fresh ids on every query"
    source = table_source(table)
    gkeys = tuple(e.args[0] for e in agg_node.group_by)
    if not full:
        try:
            prog = compile_pipeline(predicates, list(agg_node.aggs), table)
        except Exception as exc:
            return f"compiling raised {type(exc).__name__}"
        if prog is None:
            return "expression not supported by the bytecode compiler"
        return SplitInputs(prog, None, table, agg_node, source or ("memory", id(table)), gkeys)
    why, _ = CustomOpsCompiler._arrow_aggregate_reason(agg_node, predicates, table)   # answered before the pipelines
    if why:
        return f"Arrow's hash aggregate answers it: {why}"
    try:
        bound = _bind(predicates, agg_node, table)
    except Exception as exc:                                         # the run would raise the same and fall back
        return f"binding raised {type(exc).__name__}"
    if isinstance(bound, str):
        return bound
    prog, cc = bound
    if source is None and not coproc_fused.stable_columns(cc):
        return "a column is converted on every query (dictionary codes, bool, narrow ints), so it has no stable buffer to keep on the GPU"
    return SplitInputs(prog, cc, table, agg_node, source or ("memory", id(table)), gkeys)


def queue_resident_upload(plan) -> None:
    """Start the background upload of what a split of ``plan`` reads (its columns and group ids), without running
    the query: the CPU decision that predicted a split has not found them resident yet. A background thread
    calls this once the query has returned."""
    from . import coproc_fused
    got = split_check(plan, full=False)
    if isinstance(got, str) or not coproc_fused.begin_prepare(got.source, got.columns, got.gkeys):
        return
    got = split_check(plan)                  # the passes over the data, once per source, on this background thread
    if isinstance(got, str):
        return
    from .custom_ops import CustomOpsCompiler
    gids, n_groups, fp = None, 1, None
    if got.gkeys:
        gids, n_groups, uniq = CustomOpsCompiler._build_group_ids_cached(got.table, list(got.gkeys))
        gids = np.ascontiguousarray(gids, dtype=np.int32)
        if n_groups <= 0 or len(gids) != got.table.num_rows:
            return
        fp = _gid_fingerprint(got.table, uniq, n_groups)
    coproc_fused.request_upload(got.prog, got.cc, gids, n_groups, got.table.num_rows, got.table,
                                gid_fp=fp, gkeys=got.gkeys)


_I64_RANGE: dict = {}          # id(table) -> (weakref to the table, {column: values past 2^53?})


def _int64_past_2_53(table, names) -> bool:
    """The pipeline computes in float64, exact for integers only within +-2^53 (F-032 review). Remembered
    per table and column: the split's decision and the run both ask, and a min / max pass over the rows
    costs as much as a native query."""
    import weakref
    got = _I64_RANGE.get(id(table))
    if got is None or got[0]() is not table:
        key = id(table)
        try:
            got = (weakref.ref(table, lambda _r, k=key: _I64_RANGE.pop(k, None)), {})
        except TypeError:                      # not weak-referenceable: nothing to remember it by, compute every time
            return any(_column_past_2_53(table.column(n)) for n in names)
        _I64_RANGE[id(table)] = got
    seen = got[1]
    return any(seen[n] if n in seen else seen.setdefault(n, _column_past_2_53(table.column(n))) for n in names)


def _column_past_2_53(col) -> bool:
    import numpy as np
    lim = 1 << 53
    if col.type != pa.int64():
        return False
    for ch in col.chunks:
        if len(ch) and ch.null_count == 0:
            a = np.frombuffer(ch.buffers()[1], dtype=np.int64, count=len(ch) + ch.offset)[ch.offset:]
            if a.max() > lim or a.min() < -lim:
                return True
        elif len(ch):
            return True              # nulls: bind_chunks declines anyway
    return False
