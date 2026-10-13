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


def _run_gpu(pk, prog: PipelineProgram, cc: ChunkedColumns, gids, n_groups: int, n_rows: int):
    parts = []
    for s in range(0, n_rows, GPU_CHUNK_ROWS):
        parts.append(pk.run_pipeline_gpu(prog, cc, gids, n_groups, s, min(s + GPU_CHUNK_ROWS, n_rows)))
    return merge_partials(prog, parts)


def _decline(compiler, reason: str):
    """Remember why a plan was not served (``compiler.last_fused_decline``) and let the caller continue."""
    compiler.last_fused_decline = reason
    return None


def try_fused_aggregate(compiler, plan, having_pred, post_ops, optimizer_trace=None,
                        t0: Optional[float] = None) -> Optional[pa.Table]:
    if not enabled():
        return _decline(compiler, "disabled (MXFRAME_FUSED_PIPELINE=0)")
    shape = _shape(plan)
    if shape is None:
        return _decline(compiler, f"plan is not Aggregate(Filter*(Scan)): {type(plan).__name__}"
                                  f" over {type(getattr(plan, 'input', None)).__name__}")
    predicates, agg_node, scan = shape
    table: pa.Table = scan.table
    n_rows = table.num_rows
    if n_rows == 0 or not agg_node.aggs:
        return _decline(compiler, "empty input or no aggregates")
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

    prog = compile_pipeline(predicates, list(agg_node.aggs), table)
    if prog is None:
        return _decline(compiler, "expression not supported by the bytecode compiler")
    if _int64_past_2_53(table, prog.columns):
        return _decline(compiler, "an int64 column holds values past 2^53 (the pipeline computes in float64)")
    try:
        cc = prog.bind_chunks(table)
    except Unsupported:
        return _decline(compiler, "a column has nulls or an unsupported type")
    if cc.n_cols and cc.n_rows != n_rows:
        return _decline(compiler, "column lengths differ")

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
            if e.op == "col":
                group_keys.append(e.args[0])
            else:  # derived key, e.g. year(col): same treatment as the existing path
                name = e._alias or f"__gk_{len(group_keys)}__"
                # evaluated against the source table; a key that reuses a source column's name
                # (``with_columns((k + 10).alias("k"))`` inlined into the aggregate) replaces it
                derived.append((name, compiler._eval_expr_arrow(e, table)))
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
    if device == "cpu":
        from . import coproc_fused
        stable = not grouped or all(e.op == "col" for e in agg_node.group_by)   # derived keys: fresh ids per query
        split = coproc_fused.try_split(prog, cc, gids, n_groups, n_rows, table, stable_gids=stable)
        if split is not None:
            partials, used, coproc_info = split
    if device == "gpu":
        try:
            partials = _run_gpu(pk, prog, cc, gids, n_groups, n_rows)
            used = "gpu"
        except pk.GpuUnsupported:
            partials = None            # this program does not fit the GPU kernel: CPU kernel
        except MemoryError:
            # Out of device memory: an explicit device="gpu" is served by the CPU kernel
            # (provenance says cpu); under auto the caller's own fallback decides.
            if getattr(compiler, "_device_arg", None) != "gpu":
                raise
            partials = None
        # any other GPU error propagates, so auto_mode sees it (is_sticky_gpu_error / disable_gpu)
    if partials is None:
        partials = pk.run_pipeline_cpu(prog, cc, gids, n_groups, 0, n_rows)

    present, cols = finalize(prog, partials, grouped)
    if grouped:
        keep = None if bool(present.all()) else _npa(present)
        key_arrays = []
        for k in group_keys:
            v = unique_key_cols[k]
            key_arrays.append(v if keep is None else v.filter(keep))
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
        result = result.filter(compiler._eval_predicate(having_pred, result))
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


def _int64_past_2_53(table, names) -> bool:
    """The pipeline computes in float64, exact for integers only within +-2^53 (F-032 review)."""
    import numpy as np
    lim = 1 << 53
    for n in names:
        col = table.column(n)
        if col.type != pa.int64():
            continue
        for ch in col.chunks:
            if len(ch) and ch.null_count == 0:
                a = np.frombuffer(ch.buffers()[1], dtype=np.int64, count=len(ch) + ch.offset)[ch.offset:]
                if a.max() > lim or a.min() < -lim:
                    return True
            elif len(ch):
                return True              # nulls: bind_chunks declines anyway
    return False
