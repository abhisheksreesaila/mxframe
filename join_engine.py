"""Join execution (F-018): projection pushdown, late materialization, join reordering.

A tree of inner joins is flattened into *leaves* (the non-join inputs, filtered and cut down to
the columns the query needs) and *edges* (equality keys between two leaves).  The leaves are
joined in an order chosen from their real (post-filter) row counts, and the intermediate result
is only a set of row-index vectors, one per leaf that is still needed: a join gathers the key
columns it compares and composes the index vectors, and every output column is gathered once, at
the end, from its own leaf.  A join therefore never copies a payload column it does not return,
and never copies one twice.

Left joins are barriers: their two sides are executed on their own and the result is a leaf of
whatever joins above it.

The output columns and their names are exactly those of the tree written by the user
(``join_output_names`` applied join by join), whatever order the leaves are joined in.
"""
from __future__ import annotations

import functools
import os
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
from . import gpu_budget
from ._lazy_pc import pc

from .lazy_frame import (join_output_names, Scan, Filter, Project, Aggregate, Sort, Limit,
                         Distinct, Tail, Join)
from .np_arrow import array as _npa, to_numpy as _np_of, scalar as _npscalar, take_rows as _take_rows, combine as _combine, canonical_floats as _canonical_floats
from .optimizer import _output_columns as plan_columns


# A one-shot gather is cheaper on the CPU below this many rows: the GPU one uploads the source columns
# and downloads the result (measured: TPC-H Q9 SF10, 3.3M rows x 6 columns, 0.12 s on the CPU, 0.5-0.7 s on the GPU).
GPU_GATHER_MIN_ROWS = int(os.environ.get("MXFRAME_GPU_GATHER_MIN_ROWS", 1 << 62))

CACHE_MAX_CELLS = 1 << 25        # join results larger than this (rows x columns) are not kept for the next query

_TRACE = bool(os.environ.get("MXFRAME_JOIN_TRACE"))      # print one line per join step to stderr


def _trace(msg: str) -> None:
    import sys
    print(f"[join] {msg}", file=sys.stderr, flush=True)


# ── which columns does the plan above a join need? ──────────────────────────────────────────

def _cols(expr) -> Set[str]:
    from .io_parquet import _expr_cols
    return set(_expr_cols(expr))


def _union(needed: Optional[Set[str]], extra: Set[str]) -> Optional[Set[str]]:
    return None if needed is None else (needed | extra)


def _out_name(e, i: int) -> str:
    return e._alias or (e.args[0] if e.op == "col" else f"col_{i}")


def materialize(comp, plan, needed: Optional[Set[str]] = None, dict_ok: bool = True):
    """Replace every Join in ``plan`` by a ``Scan`` of its result.

    ``needed`` is the set of column names the caller reads from ``plan``'s output (None: all).
    It is narrowed on the way down so each join only produces the columns read above it.
    ``dict_ok`` says every consumer up to the top (filters, projections, group keys, aggregates) accepts
    dictionary-typed string columns; it is False below a sort, limit or distinct and below a join that
    could use such a column as a key.
    """
    if isinstance(plan, Scan):
        return plan
    if isinstance(plan, Join):
        return Scan(execute_join(comp, plan, needed, dict_ok))
    if isinstance(plan, Filter):
        return Filter(materialize(comp, plan.input, _union(needed, _cols(plan.predicate)), dict_ok), plan.predicate)
    if isinstance(plan, Project):
        exprs = list(plan.exprs)
        named = all(e._alias or e.op == "col" for e in exprs)      # unnamed expressions are named by position
        produced = {_out_name(e, i) for i, e in enumerate(exprs)}
        if needed is not None and named:
            # `with_columns` after a join passes every column through: keep only what the parent reads
            keep = [e for i, e in enumerate(exprs) if _out_name(e, i) in needed]
            exprs = keep or exprs[:1]
        used: Set[str] = set()
        for e in exprs:
            used |= _cols(e)
        # a column the parent reads that this Project does not produce passes through from below
        # (a Project of new columns only, with_columns after a join, is completed by the aggregate
        # that inlines it)
        below = used if needed is None else (used | (needed - produced))
        return Project(materialize(comp, plan.input, below, dict_ok), exprs)
    if isinstance(plan, Aggregate):
        used = set()
        for e in list(plan.group_by) + list(plan.aggs):
            used |= _cols(e)
        return Aggregate(materialize(comp, plan.input, used, dict_ok), plan.group_by, plan.aggs)
    if isinstance(plan, Sort):
        used = set()
        for e in plan.by:
            used |= _cols(e) if hasattr(e, "op") else {e}
        return Sort(materialize(comp, plan.input, _union(needed, used), False), plan.by, plan.descending)
    if isinstance(plan, Limit):
        return Limit(materialize(comp, plan.input, needed, False), plan.n)
    if isinstance(plan, Tail):
        return Tail(materialize(comp, plan.input, needed, False), plan.n)
    if isinstance(plan, Distinct):
        below = None if (needed is None or not plan.subset) else (needed | set(plan.subset))
        return Distinct(materialize(comp, plan.input, below, False), plan.subset)
    if hasattr(plan, "input"):
        plan.input = materialize(comp, plan.input, None, dict_ok)
    return plan


# ── leaf tables ─────────────────────────────────────────────────────────────────────────────

def _select(table: pa.Table, needed: Optional[Set[str]]) -> pa.Table:
    if needed is None:
        return table
    names = [c for c in table.column_names if c in needed]
    if not names:
        names = table.column_names[:1]          # keep the row count
    return table if len(names) == table.num_columns else table.select(names)


def _plain_columns(plan) -> Optional[List[str]]:
    """The columns of a Project that only selects columns (no expression, no rename), else None."""
    if not isinstance(plan, Project) or not plan.exprs:
        return None
    if any(e.op != "col" or e._alias not in (None, e.args[0]) for e in plan.exprs):
        return None
    return [e.args[0] for e in plan.exprs]


def _scan_under_filters(plan):
    """(Scan, predicates) when ``plan`` is Filter*(Scan), else (None, []). A Project that only
    selects columns on top (``.filter(...).select(key)``) is seen through: callers ask for the
    columns they need, which are among the selected ones."""
    if _plain_columns(plan) is not None:
        plan = plan.input
    preds, node = [], plan
    while isinstance(node, Filter):
        preds.append(node.predicate)
        node = node.input
    return (node, preds) if isinstance(node, Scan) else (None, [])


def _decode_deferred(table: pa.Table, cols) -> pa.Table:
    """Decode the dictionary columns io_parquet.resolve_plan left encoded until after the leaf's
    filter (so only surviving rows are decoded)."""
    from .arrow_dict import decode
    for i, (n, t) in enumerate(zip(table.column_names, table.schema.types)):
        if n in cols and pa.types.is_dictionary(t):
            table = table.set_column(i, n, decode(table.column(n)))
    return table


def leaf_table(comp, plan, needed: Optional[Set[str]], keep: Optional[Set[str]] = None) -> pa.Table:
    """``plan`` evaluated to a table holding (at least) the ``needed`` columns.

    A Filter*(Scan) leaf runs as one native call when the executor can (F-032: predicate per morsel
    in Mojo, only the surviving rows of the needed columns written, dictionary strings decoded on
    the way out). Otherwise the predicates are evaluated here, then the dictionary columns
    resolve_plan deferred are decoded on the surviving rows, except those in ``keep``: the caller
    decodes them itself, after the join has gathered only the joined rows."""
    scan, preds = _scan_under_filters(plan)
    if scan is not None:
        sel = _plain_columns(plan)
        if sel is not None:
            needed = set(sel) if needed is None else set(needed) & set(sel)
        if preds:
            from . import native_exec
            out = native_exec.try_filter(comp, scan.table, preds, needed, keep_dict=keep or ())   # (codes the caller decodes)
            if out is not None:
                return out
        later = set(getattr(scan, "_mx_decode_after", None) or ()) - set(keep or ())
        out = _leaf_table_py(comp, plan, needed)
        return _decode_deferred(out, later) if later else out
    return _leaf_table_py(comp, plan, needed)


def _leaf_table_py(comp, plan, needed: Optional[Set[str]]) -> pa.Table:
    if isinstance(plan, Scan):
        return _select(plan.table, needed)
    if isinstance(plan, Filter):
        inner = _leaf_table_py(comp, plan.input, _union(needed, _cols(plan.predicate)))
        if not isinstance(plan.input, Scan):          # a computed table (F-034, Q22's anti join side): native filter
            got = comp._native_filter_table(inner, [plan.predicate], needed)
            if got is not None:
                return got
        mask = comp._eval_predicate(plan.predicate, inner)
        return pc.filter(_select(inner, needed), mask)     # the mask is done: drop the predicate-only columns first
    if isinstance(plan, Join):
        return execute_join(comp, plan, needed, False)
    return comp._plan_to_table(materialize(comp, plan, needed, False))


# ── flattening a tree of inner joins ────────────────────────────────────────────────────────

class _Flat:
    """Leaves, key edges and output naming of a group of inner joins."""
    __slots__ = ("leaves", "edges", "names", "src", "residuals")

    def __init__(self, leaves, edges, names, src, residuals=()):
        self.leaves = leaves          # leaf plans
        self.edges = edges            # (leaf a, leaf b, a columns, b columns)
        self.names = names            # output column names, in order
        self.src = src                # output name -> (leaf, column)
        self.residuals = list(residuals)   # (predicate, {column name: (leaf, column)}): a Filter between joins


def _flatten(node) -> Optional[_Flat]:
    if isinstance(node, Filter) and isinstance(node.input, Join) and node.input.how == "inner":
        # a condition across leaves between two joins (TPC-H Q5: c_nationkey = s_nationkey): applied as soon
        # as the leaves it reads are joined, whatever order that is
        inner = _flatten(node.input)
        cols = _cols(node.predicate)
        if inner is not None and cols and all(c in inner.src for c in cols):
            inner.residuals.append((node.predicate, {c: inner.src[c] for c in cols}))
            return inner
    if isinstance(node, Join) and node.how == "inner":
        left, right = _flatten(node.left), _flatten(node.right)
        if left is None or right is None:
            return None
        try:
            lsrc = [left.src[k] for k in node.left_on]
            rsrc = [right.src[k] for k in node.right_on]
        except KeyError:
            return None
        if len({s[0] for s in lsrc}) == 1 and len({s[0] for s in rsrc}) == 1:
            off = len(left.leaves)
            mapping = join_output_names(left.names, right.names, node.right_on)
            src = dict(left.src)
            for orig, out in mapping:
                leaf, c = right.src[orig]
                src[out] = (leaf + off, c)
            edges = list(left.edges) + [(a + off, b + off, ac, bc) for a, b, ac, bc in right.edges]
            edges.append((lsrc[0][0], rsrc[0][0] + off, [s[1] for s in lsrc], [s[1] for s in rsrc]))
            residuals = list(left.residuals) + [(p, {n: (l + off, c) for n, (l, c) in cm.items()})
                                                for p, cm in right.residuals]
            return _Flat(left.leaves + right.leaves, edges, left.names + [o for _, o in mapping], src, residuals)
        # keys drawn from several leaves of one side: join the two sides as units
        lnames, rnames = plan_columns(node.left), plan_columns(node.right)
        if lnames is None or rnames is None:
            return None
        mapping = join_output_names(lnames, rnames, node.right_on)
        src = {n: (0, n) for n in lnames}
        src.update({out: (1, orig) for orig, out in mapping})
        return _Flat([node.left, node.right], [(0, 1, list(node.left_on), list(node.right_on))],
                     lnames + [o for _, o in mapping], src)
    names = plan_columns(node)
    if names is None:
        return None
    return _Flat([node], [], list(names), {n: (0, n) for n in names})


# ── join order ──────────────────────────────────────────────────────────────────────────────

def _estimate(r_rows: float, r_frac: float, rows, base, r_leaf: int, new: int) -> Tuple[float, float]:
    """(estimated rows, fraction of the new leaf present) after joining leaf ``new`` to a relation
    of ``r_rows`` rows that reaches ``r_leaf`` through the key.  The side whose base table is
    smaller is taken to be the one holding the unique key (TPC-H style foreign key to primary key):
    joining it filters the relation by the share of its rows that survived, joining the larger
    one multiplies the relation by that leaf's rows per key."""
    bn, br = max(1, base[new]), max(1, base[r_leaf])
    if bn <= br:
        sel = min(1.0, rows[new] / bn)
        return r_rows * sel, sel
    out = rows[new] * min(1.0, r_frac)
    return out, min(1.0, out / bn)


def _simulate(start: int, rows, base, edges) -> Tuple[float, List[Tuple[int, int, int]]]:
    n = len(rows)
    in_r = {start}
    r_rows = float(rows[start])
    frac = {start: min(1.0, rows[start] / max(1, base[start]))}
    used: Set[int] = set()
    steps: List[Tuple[int, int, int]] = []          # (new leaf, edge index, leaf it joins in the relation)
    cost = 0.0
    while len(in_r) < n:
        best = None
        for ei, (a, b, _, _) in enumerate(edges):
            if ei in used:
                continue
            if a in in_r and b not in in_r:
                r_leaf, new = a, b
            elif b in in_r and a not in in_r:
                r_leaf, new = b, a
            else:
                continue
            out, f_new = _estimate(r_rows, frac[r_leaf], rows, base, r_leaf, new)
            cand = (out, rows[new], ei, new, r_leaf, f_new)
            if best is None or cand[:3] < best[:3]:
                best = cand
        if best is None:                             # not connected: cannot happen for a join tree
            return float("inf"), steps
        out, _, ei, new, r_leaf, f_new = best
        cost += r_rows + rows[new] + out
        if base[new] <= base[r_leaf]:
            frac[r_leaf] *= f_new
        r_rows = out
        frac[new] = f_new
        in_r.add(new)
        used.add(ei)
        steps.append((new, ei, r_leaf))
    return cost, steps


def plan_order(rows: Sequence[int], base: Sequence[int], edges) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Start leaf and ``[(leaf to add, edge, leaf of the relation it joins), ...]`` with the least
    estimated work.  ``rows`` are the leaves' row counts after their filters, ``base`` the rows of
    the tables they filter.  A start within 10% of the cheapest is preferred if it is further left, i.e. closer to the order the user wrote."""
    runs = [(_simulate(start, rows, base, edges), start) for start in range(len(rows))]
    least = min(cost for (cost, _), _ in runs)
    for (cost, steps), start in runs:          # within 10% of the best: keep the written order
        if cost <= least * 1.1:
            return start, steps
    raise AssertionError("unreachable")


def _base_rows(plan, materialized: int) -> int:
    while isinstance(plan, (Filter, Project)):
        plan = plan.input
    return plan.table.num_rows if isinstance(plan, Scan) else materialized


# ── join keys ───────────────────────────────────────────────────────────────────────────────

def _is_plain_int(x) -> bool:
    return isinstance(x, np.ndarray) and x.dtype.kind in "iu"


def _as_pa(x) -> pa.Array:
    if isinstance(x, np.ndarray):
        return _npa(x)
    if isinstance(x, pa.ChunkedArray):
        return _combine(x) if x.num_chunks != 1 else x.chunk(0)
    return x


def _as_np_int(x):
    """A NumPy integer array for a null-free integer key column, else None."""
    if isinstance(x, np.ndarray):
        return x if x.dtype.kind in "iu" else None
    if pa.types.is_integer(x.type) and x.null_count == 0:
        a = _np_of(x, zero_copy_only=False)
        return a if a.dtype.kind in "iu" else None
    return None


PAR_MIN_ROWS = 1 << 22           # NumPy passes over fewer rows run on one thread
_PAR_PIECE = 1 << 20


def _pieces(n: int):
    parts = max(1, min(os.cpu_count() or 1, 16, n // _PAR_PIECE))
    return [((i * n) // parts, ((i + 1) * n) // parts) for i in range(parts)]


def _parallel(fn, n: int) -> list:
    """``fn(lo, hi)`` over pieces of ``range(n)`` on several threads (NumPy releases the GIL)."""
    from .io_parquet import _parallel_map
    pieces = _pieces(n)
    return _parallel_map(lambda b: fn(*b), pieces, len(pieces))


def _minmax(a: np.ndarray) -> Tuple[int, int]:
    n = len(a)
    if n == 0:
        return 0, 0
    if n < PAR_MIN_ROWS:
        return int(a.min()), int(a.max())
    parts = _parallel(lambda lo, hi: (a[lo:hi].min(), a[lo:hi].max()), n)
    return int(min(p[0] for p in parts)), int(max(p[1] for p in parts))


def _as_i32(a: np.ndarray) -> np.ndarray:
    """``a`` as a contiguous int32 array (values known to fit)."""
    if a.dtype == np.int32 and a.flags["C_CONTIGUOUS"]:
        return a
    if len(a) < PAR_MIN_ROWS:
        return np.ascontiguousarray(a, dtype=np.int32)
    out = np.empty(len(a), dtype=np.int32)
    _parallel(lambda lo, hi: np.copyto(out[lo:hi], a[lo:hi], casting="unsafe"), len(a))
    return out


def _take(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``a[idx]``, on several threads for long ``idx``."""
    n = len(idx)
    if n < PAR_MIN_ROWS:
        return a[idx]
    out = np.empty(n, dtype=a.dtype)
    _parallel(lambda lo, hi: np.take(a, idx[lo:hi], out=out[lo:hi], mode="clip"), n)
    return out


def _pack(lparts: List[np.ndarray], rparts: List[np.ndarray]):
    """Combine several integer key columns into one int64 key (mixed radix); None if it would overflow."""
    los, strides, total = [], [], 1
    for l, r in zip(lparts, rparts):
        (lo1, hi1), (lo2, hi2) = _minmax(l), _minmax(r)
        lo, hi = min(lo1, lo2), max(hi1, hi2)
        if hi > np.iinfo(np.int64).max or lo < np.iinfo(np.int64).min:
            return None            # uint64 keys above 2^63 don't fit the int64 packing; general path joins them
        los.append(lo)
        strides.append(total)
        total *= (hi - lo + 1)
        if total >= (1 << 62):
            return None
    lk = np.zeros(len(lparts[0]), dtype=np.int64)
    rk = np.zeros(len(rparts[0]), dtype=np.int64)
    for l, r, lo, st in zip(lparts, rparts, los, strides):
        lk += (l.astype(np.int64, copy=False) - lo) * st
        rk += (r.astype(np.int64, copy=False) - lo) * st
    return lk, rk


def dense_limit(n_rows: int) -> int:
    """Largest key a direct-address join takes as it is.  The kernel's table costs ~3 ns per possible key
    and hashing the keys into dense codes ~100 ns per row, so a key range up to ~30x the rows is cheaper
    as a table (TPC-H order keys are 4x sparse, filtered ones far more)."""
    return min(max(1024, 32 * n_rows), (1 << 28) - 1)


def _dense(lk: np.ndarray, rk: np.ndarray):
    """Both sides as int32 keys usable by a direct-address join, if the key range is small."""
    n = len(lk) + len(rk)
    if n == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    (l0, l1), (r0, r1) = _minmax(lk), _minmax(rk)
    if len(lk) == 0:
        lo, hi = r0, r1
    elif len(rk) == 0:
        lo, hi = l0, l1
    else:
        lo, hi = min(l0, r0), max(l1, r1)
    if lo >= 0 and hi <= dense_limit(n):
        return _as_i32(lk), _as_i32(rk)
    return None


def encode_keys(left_cols: Sequence, right_cols: Sequence, common_key_type):
    """Encode both join sides into one shared dense int32 key space.

    Columns are NumPy arrays or Arrow arrays.  A single null-free integer key with a small range
    is used as it is; several integer keys are packed into one; everything else (strings, floats,
    keys with nulls, wide ranges) goes through one dictionary over both sides, so equal values get
    equal codes.  Nulls get side-specific codes and never match.
    """
    if len(left_cols) != len(right_cols) or not left_cols:
        raise ValueError("join key column lists must be non-empty and equal length")
    n_left = len(left_cols[0])

    lints = [_as_np_int(c) for c in left_cols]
    rints = [_as_np_int(c) for c in right_cols]
    if all(a is not None for a in lints) and all(a is not None for a in rints):
        same_kind = all((a.dtype.kind == "u") == (b.dtype.kind == "u") or
                        max(a.dtype.itemsize, b.dtype.itemsize) < 8
                        for a, b in zip(lints, rints))
        if same_kind:
            if len(lints) == 1:
                lk, rk = lints[0], rints[0]
            else:
                packed = _pack(lints, rints)
                lk, rk = packed if packed is not None else (None, None)
            if lk is not None:
                dense = _dense(lk, rk)
                if dense is not None:
                    return dense
                enc = pc.dictionary_encode(pa.concat_arrays([_npa(np.ascontiguousarray(lk, dtype=np.int64)),
                                        _npa(np.ascontiguousarray(rk, dtype=np.int64))]))
                codes = _np_of(enc.indices, zero_copy_only=False)
                codes = codes.astype(np.int32, copy=False)
                return np.ascontiguousarray(codes[:n_left]), np.ascontiguousarray(codes[n_left:])

    paired = []
    for lc, rc in zip(left_cols, right_cols):
        la, ra = _as_pa(lc), _as_pa(rc)
        if la.type != ra.type:
            common = common_key_type(la.type, ra.type)
            if common is None:
                raise TypeError(f"join key types differ: {la.type} and {ra.type}")
            la, ra = pc.cast(la, common), pc.cast(ra, common)
        la, ra = _canonical_floats(la), _canonical_floats(ra)     # -0.0 = 0.0, one NaN: SQL key equality (F-079)
        enc = pc.dictionary_encode(pa.concat_arrays([la, ra]))
        codes = _np_of(pc.fill_null(enc.indices, _npscalar(-1, type=enc.indices.type)),
                       zero_copy_only=False).astype(np.int64)
        nulls = codes < 0
        if nulls.any():
            nxt = len(enc.dictionary)
            codes[:n_left][nulls[:n_left]] = nxt
            codes[n_left:][nulls[n_left:]] = nxt + 1
        paired.append(codes)
    if len(paired) == 1:
        dense = paired[0]
    else:
        packed = _pack([c[:n_left] for c in paired], [c[n_left:] for c in paired])
        if packed is not None:
            both = np.concatenate(packed)
            enc = pc.dictionary_encode(_npa(both))
            dense = _np_of(enc.indices, zero_copy_only=False)
        else:
            _, dense = np.unique(np.column_stack(paired), axis=0, return_inverse=True)
    dense = dense.astype(np.int32, copy=False).reshape(-1)
    return np.ascontiguousarray(dense[:n_left]), np.ascontiguousarray(dense[n_left:])


def _gather_key(col, idx: Optional[np.ndarray]):
    """A key column at the relation's rows: NumPy for null-free integers, else Arrow."""
    if idx is None:
        return col
    a = _as_np_int(col)
    if a is not None:
        return _take(a, idx)
    return _as_pa(_take_rows(col, idx))


# ── kernels ─────────────────────────────────────────────────────────────────────────────────

_semi = {"lib": None, "tried": False}
SEMI_MIN_PROBE = 1 << 20         # probe rows below which the prefilter is not worth a pass
SEMI_MIN_RANGE = 1 << 20         # key range below which the join table stays in cache anyway
SEMI_MAX_SHARE = 0.6             # the prefilter is kept only if it drops at least this much of the probe side


def _semi_lib():
    """The semi-join prefilter library (``kernels_aot/libmxjoin_cpu.so``), or None if it is missing."""
    if not _semi["tried"]:
        _semi["tried"] = True
        import ctypes
        from pathlib import Path
        here = Path(__file__).parent
        env = os.environ.get("MXFRAME_JOIN_LIB")
        for cand in ([Path(env)] if env else []) + [here / "kernels_aot" / "libmxjoin_cpu.so", here / "libmxjoin_cpu.so"]:
            if cand.exists():
                try:
                    lib = ctypes.CDLL(str(cand))
                    i64 = ctypes.c_int64
                    lib.mx_join_abi_version.restype = i64
                    if lib.mx_join_abi_version() != 1:
                        continue
                    lib.mx_join_semi_select.restype = i64
                    lib.mx_join_semi_select.argtypes = [i64] * 7
                    lib.mx_join_init_runtime.restype = i64
                    lib.mx_join_init_runtime()
                    _semi["lib"] = lib
                    break
                except (OSError, AttributeError):
                    continue
    return _semi["lib"]


def semi_select(probe: np.ndarray, build: np.ndarray, max_key: int) -> Optional[np.ndarray]:
    """Row numbers (ascending, int32) of the ``probe`` keys that occur in ``build``; both are contiguous
    int32 arrays of keys in ``[0, max_key]``.  None when the kernel is not available."""
    lib = _semi_lib()
    if lib is None:
        return None
    from .pipeline_kernels import default_threads
    out = np.empty(len(probe), dtype=np.int32)
    n = lib.mx_join_semi_select(int(probe.ctypes.data), len(probe), int(build.ctypes.data), len(build),
                                int(max_key), int(out.ctypes.data), default_threads())
    if n < 0:
        return None
    return out[:n].copy()


def inner_pairs(comp, probe_keys: np.ndarray, build_keys: np.ndarray):
    """(probe_idx, build_idx) int32 pairs of the equal keys on the session device."""
    if comp._session_device == "gpu" and comp._aot_gpu is not None:
        n_p, n_b = len(probe_keys), len(build_keys)
        size = max(int(probe_keys.max(initial=0)), int(build_keys.max(initial=0))) + 1 if n_p and n_b else 0
        return gpu_budget.run_piece(          # F-056: over the GPU's free memory, or out of it: the CPU join
            "join", gpu_budget.join_bytes(n_p, n_b, size), lambda: comp._aot_gpu.hash_join(probe_keys, build_keys),
            lambda: comp._hash_join_mojo_cpu(probe_keys, build_keys), shape=(n_p, n_b, size))
    if comp._session_device == "gpu" and comp._aot is None:
        return comp._hash_join_mojo_gpu(probe_keys, build_keys)
    return comp._hash_join_mojo_cpu(probe_keys, build_keys)


def _dictionary_gather(col, idx: np.ndarray):
    """A string column gathered as a dictionary array (codes gathered, values shared) when it has far
    fewer rows than the gather: n_name of 25 nations over 18M joined rows moves 4-byte codes, not
    strings, and the group-by above uses the codes as they are.  None when that does not apply."""
    if not (pa.types.is_string(col.type) or pa.types.is_large_string(col.type)):
        return None
    if len(col) * 8 > len(idx) or len(idx) < (1 << 16):
        return None
    enc = pc.dictionary_encode(_as_pa(col))
    if enc.indices.null_count:
        return None
    codes = _np_of(enc.indices, zero_copy_only=False)[idx]
    return pa.DictionaryArray.from_arrays(_npa(codes), enc.dictionary)


def _gather_table(comp, table: pa.Table, idx: np.ndarray, decode_dicts: bool = False) -> pa.Table:
    """``table`` rows ``idx``. On the CPU one native call (F-032) gathers every column; dictionary
    columns only when ``decode_dicts`` (they come back decoded). Else Arrow ``take``."""
    if table.num_columns == 0:
        return table
    if comp._session_device == "gpu" and comp._aot_gpu is not None and len(idx) >= GPU_GATHER_MIN_ROWS:
        return gpu_budget.run_piece(
            "gather", gpu_budget.gather_bytes(table, len(idx)), lambda: comp._aot_gpu.gather_table(table, idx),
            lambda: _take_rows(table, idx), shape=(table.num_rows, len(idx), str(table.schema.types)))
    from . import native_exec
    out = native_exec.try_gather(comp, table, idx, decode_dicts)
    if out is not None:
        return out
    return _take_rows(table, idx)


# ── executing a join ────────────────────────────────────────────────────────────────────────

def _stable_key(comp, plan):
    if isinstance(plan, Join):
        l, r = _stable_key(comp, plan.left), _stable_key(comp, plan.right)
        if l is None or r is None:
            return None
        return ("J", l, r, tuple(plan.left_on), tuple(plan.right_on), plan.how)
    return type(comp)._plan_stable_key(plan)


def _keep_sig(plan) -> tuple:
    """The dictionary columns the leaves under ``plan`` keep coded (F-046d): part of the join cache
    key, so a result with coded columns never serves a plan that wants them decoded."""
    out = []

    def walk(p):
        k = getattr(p, "_mx_keep_dict", None)
        if k:
            out.append(tuple(sorted(k)))
        for child in (getattr(p, "input", None), getattr(p, "left", None), getattr(p, "right", None)):
            if child is not None:
                walk(child)

    walk(plan)
    return tuple(out)


def execute_join(comp, node, needed: Optional[Set[str]], dict_ok: bool = False) -> pa.Table:
    from . import custom_ops as _co
    cache = _co._JOIN_RESULT_CACHE
    key = _stable_key(comp, node)
    if key is not None:
        key = (key, None if needed is None else frozenset(needed), dict_ok, _keep_sig(node),
               comp._session_device)
        hit = cache.get(key)
        if hit is not None:
            return hit
    out = _execute_join(comp, node, needed, dict_ok)
    if key is not None and out.num_rows * max(1, out.num_columns) <= CACHE_MAX_CELLS:
        cache[key] = out
        if len(cache) > 24:
            for old in list(cache.keys())[:12]:
                cache.pop(old, None)
    return out


def _legacy(comp, node):
    left = comp._plan_to_table(materialize(comp, node.left, None))
    right = comp._plan_to_table(materialize(comp, node.right, None))
    return comp._execute_hash_join(left, right, node.left_on, node.right_on, how=node.how)


def has_partner(lk: np.ndarray, rk: np.ndarray) -> np.ndarray:
    """For each left key code, whether some right key code equals it (codes from ``encode_keys``:
    non-negative, nulls on side-specific codes that never match)."""
    if len(lk) == 0 or len(rk) == 0:
        return np.zeros(len(lk), dtype=bool)
    hi = int(max(lk.max(), rk.max())) + 1
    if hi <= 4 * (len(lk) + len(rk)) + 1024:
        mark = np.zeros(hi, dtype=bool)
        mark[rk] = True
        return mark[lk]
    return np.isin(lk, rk)


def _execute_filtering(comp, node, needed):
    """A semi join (left rows with a partner) or anti join (left rows without one): the left
    columns, in the left table's row order."""
    lcols = plan_columns(node.left)
    if lcols is None:
        return None
    want = lcols if needed is None else [n for n in lcols if n in needed] or lcols[:1]
    lt = leaf_table(comp, node.left, set(want) | set(node.left_on))
    rt = leaf_table(comp, node.right, set(node.right_on))
    if lt.num_rows and rt.num_rows:
        lk, rk = encode_keys([lt.column(c) for c in node.left_on], [rt.column(c) for c in node.right_on],
                             comp._common_key_type)
        hit = has_partner(lk, rk)
    else:
        hit = np.zeros(lt.num_rows, dtype=bool)
    keep = ~hit if node.how == "anti" else hit
    sub = lt.select(want)
    if keep.all():
        return sub
    return _gather_table(comp, sub, np.flatnonzero(keep).astype(np.int64))


def _execute_join(comp, node, needed, dict_ok=False):
    if node.how in ("semi", "anti"):
        out = _execute_filtering(comp, node, needed)
    elif node.how == "left":
        out = _execute_left(comp, node, needed)
    elif node.how == "inner":
        out = _execute_inner(comp, node, needed, dict_ok)
    else:
        out = None
    return out if out is not None else _legacy(comp, node)


def _index_with_nulls(idx: np.ndarray, valid: np.ndarray) -> pa.Array:
    """An int64 Arrow index array that is null where ``valid`` is False (built from buffers: ``pa.array``
    with a mask, or a null scalar in a compute call, would import pandas)."""
    n = len(idx)
    data = np.where(valid, idx, 0).astype(np.int64)
    bits = np.packbits(valid, bitorder="little") if n else np.zeros(0, np.uint8)
    return pa.Array.from_buffers(pa.int64(), n, [pa.py_buffer(bits), pa.py_buffer(data)])


def _execute_left(comp, node, needed):
    lcols, rcols = plan_columns(node.left), plan_columns(node.right)
    if lcols is None or rcols is None:
        return None
    mapping = join_output_names(lcols, rcols, node.right_on)
    names = list(lcols) + [o for _, o in mapping]
    want = names if needed is None else [n for n in names if n in needed] or names[:1]
    lneed = {n for n in want if n in set(lcols)} | set(node.left_on)
    rneed = {orig for orig, out in mapping if out in set(want)} | set(node.right_on)
    lt = leaf_table(comp, node.left, lneed)
    rt = leaf_table(comp, node.right, rneed)
    lk, rk = encode_keys([lt.column(c) for c in node.left_on], [rt.column(c) for c in node.right_on],
                         comp._common_key_type)
    if comp._session_device == "gpu" and comp._aot is None:
        li, ri = comp._hash_join_left_mojo_gpu(lk, rk)
    else:
        li, ri = comp._hash_join_left_mojo_cpu(lk, rk)
    matched = ri >= 0
    all_matched = bool(matched.all()) if len(ri) else True
    lwant = [n for n in want if n in set(lcols)]
    rwant = [(orig, out) for orig, out in mapping if out in set(want)]
    ltake = _take_rows(lt.select(lwant), li) if lwant else None
    arrays = {}
    for n in lwant:
        arrays[n] = ltake.column(n)
    if rwant:
        if rt.num_rows == 0:
            for orig, out in rwant:
                arrays[out] = pa.nulls(len(li), type=rt.schema.field(orig).type)
        else:
            sub = rt.select([o for o, _ in rwant])
            if all_matched:
                rtake = _take_rows(sub, ri)
            else:
                # unmatched rows carry a null index: Arrow's take turns those into null values
                rtake = pc.take(sub, _index_with_nulls(ri, matched))
            for orig, out in rwant:
                arrays[out] = rtake.column(orig)
    return pa.Table.from_arrays([arrays[n] for n in want], names=want)


MAX_RESIDUAL_EXPANSION = 8.0     # a composite key joined on one column may emit this many candidate pairs per row


def _pairs_prefiltered(comp, probe: np.ndarray, build: np.ndarray):
    """``inner_pairs``, after dropping the probe rows that have no partner when the build side is a sparse
    subset of a wide key range: the join table is then large, every probe a cache miss and nearly all of them
    misses of the key too (a 456K-order relation against 60M lineitem rows)."""
    n_p, n_b = len(probe), len(build)
    if (n_p >= SEMI_MIN_PROBE and n_b > 0 and not (comp._session_device == "gpu")):
        max_key = max(_minmax(probe)[1], _minmax(build)[1])
        if max_key + 1 >= SEMI_MIN_RANGE and 4 * n_b <= max_key + 1:
            sel = semi_select(probe, build, max_key)
            if sel is not None and len(sel) <= SEMI_MAX_SHARE * n_p:
                if len(sel) == 0:
                    e = np.empty(0, dtype=np.int32)
                    return e, e
                pi, bi = inner_pairs(comp, _take(probe, sel), build)
                return _take(sel, pi), bi
    return inner_pairs(comp, probe, build)


def _counts(a: np.ndarray, size: int) -> np.ndarray:
    """Occurrences of each value 0..size-1 of a non-negative integer array, as int64."""
    if a.dtype == np.uint64:
        a = a.astype(np.int64)
    return np.bincount(a, minlength=size).astype(np.int64)


def _residual_split(rel_keys, new_keys, n_rel: int, n_new: int):
    """For a composite integer key, the column to join on directly (largest key range, small enough
    for a direct-address join) with the others checked on the matching pairs, which is much cheaper
    than dictionary-encoding the pair of columns.  None when that does not apply."""
    if len(rel_keys) < 2:
        return None
    ints_r = [_as_np_int(c) for c in rel_keys]
    ints_n = [_as_np_int(c) for c in new_keys]
    if any(a is None for a in ints_r) or any(a is None for a in ints_n):
        return None
    best = None
    for j, (a, b) in enumerate(zip(ints_r, ints_n)):
        if (a.dtype.kind == "u") != (b.dtype.kind == "u") and max(a.dtype.itemsize, b.dtype.itemsize) == 8:
            return None
        (a0, a1), (b0, b1) = _minmax(a), _minmax(b)
        lo, hi = min(a0, b0), max(a1, b1)
        if lo < 0 or hi > dense_limit(n_rel + n_new):
            continue
        if best is None or hi > best[1]:
            best = (j, hi + 1)
    if best is None or best[1] > 8 * (n_rel + n_new):
        return None
    # candidate pairs the one column produces, counted exactly: a skewed column (one value on most rows)
    # would blow up here where hashing the whole pair would not
    a, b = ints_r[best[0]], ints_n[best[0]]
    pairs = int(np.dot(_counts(a, best[1]), _counts(b, best[1])))
    if pairs > MAX_RESIDUAL_EXPANSION * max(n_rel, n_new):
        return None
    return best[0], ints_r, ints_n


def _join_pairs(comp, rel_keys, new_keys, n_rel: int, n_new: int):
    """(relation row, new-leaf row) pairs with equal keys, and the time the key encoding took."""
    t0 = time.perf_counter()
    split = _residual_split(rel_keys, new_keys, n_rel, n_new)
    if split is not None:
        j, ints_r, ints_n = split
        rk, nk = _dense(ints_r[j], ints_n[j])
    else:
        rk, nk = encode_keys(rel_keys, new_keys, comp._common_key_type)
    t1 = time.perf_counter()
    if n_rel >= n_new:                                       # probe with the larger side, build on the smaller
        ri, ni = _pairs_prefiltered(comp, rk, nk)
    else:
        ni, ri = _pairs_prefiltered(comp, nk, rk)
    if split is not None and len(ri):
        keep = None
        for k, (a, b) in enumerate(zip(ints_r, ints_n)):
            if k == j:
                continue
            eq = a[ri] == b[ni]
            keep = eq if keep is None else (keep & eq)
        if not keep.all():
            ri, ni = ri[keep], ni[keep]
    return ri, ni, t1


def _apply_residuals(comp, tables, idx, n_rel: int, pending):
    """Filter the relation by every pending cross-leaf condition whose leaves are all in it (removes them
    from ``pending``); returns the new index vectors and row count."""
    for item in list(pending):
        pred, cm, leaves = item
        if not leaves <= idx.keys():
            continue
        pending.remove(item)
        if n_rel == 0:
            continue
        arrays, names = [], []
        for name, (leaf, c) in cm.items():
            col = tables[leaf].column(c)
            arrays.append(col if idx[leaf] is None else _take_rows(col, idx[leaf]))
            names.append(name)
        mask = comp._eval_predicate(pred, pa.Table.from_arrays(arrays, names=names))
        keep = _np_of(pc.fill_null(mask, _npscalar(False, type=pa.bool_())), zero_copy_only=False)
        sel = np.flatnonzero(keep).astype(np.int32)
        if len(sel) != n_rel:
            idx = {x: (sel if v is None else _take(v, sel)) for x, v in idx.items()}
            n_rel = len(sel)
    return idx, n_rel


# ── star joins fused into the fact table's scan (F-032, M2d) ─────────────────────────────────

def _star_fact(comp, fl) -> Optional[int]:
    """The leaf to scan with its partners looked up in the same pass: the largest Filter*(Scan)
    leaf, when the native executor runs this plan. None: join leaf by leaf."""
    if os.environ.get("MXFRAME_STAR", "1") == "0" or len(fl.leaves) < 2 or getattr(comp, "_device_arg", None) != "cpu":
        return None
    from . import native_exec
    if not (native_exec.enabled() and native_exec.available()):
        return None
    best, best_rows = None, -1
    for i, leaf in enumerate(fl.leaves):
        scan, _ = _scan_under_filters(leaf)
        if scan is not None and scan.table.num_rows > best_rows:
            best, best_rows = i, scan.table.num_rows
    return best


def _star_name(leaf: int, c: str) -> str:
    return f"{leaf}:{c}"


def _semi_keys(plan) -> Optional[Tuple[Scan, list, List[str]]]:
    """(Scan, predicates, key columns) when ``plan`` is Distinct(Project(columns)(Filter*(Scan))):
    joining it on exactly those columns is a semi join (a row of the other side either has a
    partner or not), so the distinct never has to run."""
    if not (isinstance(plan, Distinct) and not plan.subset and isinstance(plan.input, Project)):
        return None
    exprs = plan.input.exprs
    if not exprs or any(e.op != "col" or (e._alias not in (None, e.args[0])) for e in exprs):
        return None
    scan, preds = _scan_under_filters(plan.input.input)
    if scan is None:
        return None
    return scan, preds, [e.args[0] for e in exprs]


def _rows_of(plan) -> int:
    """Rows of the table under a semi partner's filters (0 when it is not that shape)."""
    sk = _semi_keys(plan)
    return sk[0].table.num_rows if sk is not None else 0


def _semi_partner(comp, plan, bkeys, reduce=None) -> Optional[pa.Table]:
    """The key rows of a semi-join partner (repeats and all), filtered natively; None: not one.
    ``reduce``: a callable giving (the other side's filtered key columns, their names), or None.
    When given, the partner's filter also keeps only rows whose keys occur there (a runtime join
    filter: TPC-H Q4's 38M late lineitems shrink to those of the 570k orders in range)."""
    sk = _semi_keys(plan)
    if sk is None or set(sk[2]) != set(bkeys) or len(sk[2]) != len(bkeys):
        return None
    scan, preds, _ = sk
    from . import native_exec
    keep = reduce() if reduce is not None else None
    if keep is not None:
        ktab, knames = keep
        out = native_exec.try_filter(comp, scan.table, preds, set(bkeys),
                                     [native_exec.Lookup(list(bkeys), ktab, list(knames), [], [], semi=True)])
        if out is not None:
            return out.select(list(bkeys))
    if not preds:
        return scan.table.select(list(bkeys))
    out = native_exec.try_filter(comp, scan.table, preds, set(bkeys))
    return None if out is None else out.select(list(bkeys))


_MAP_BYTES_PER_ROW = 16            # a partner's map: a direct array over a key range of ~4x its rows, or a hash table


@functools.lru_cache(maxsize=1)
def _l3_bytes() -> int:
    """Size of the CPU's last-level cache (32 MB when unknown)."""
    try:
        txt = open("/sys/devices/system/cpu/cpu0/cache/index3/size").read().strip().upper()
        return int(txt[:-1]) * {"K": 1 << 10, "M": 1 << 20}[txt[-1]]
    except (OSError, ValueError, KeyError, IndexError):
        return 32 << 20


def _runtime_reduced(comp, t: pa.Table, bkeys, probe, preds, fact_keys, keep_dict) -> pa.Table:
    """The runtime join filter (F-043, generalized): a partner of more than twice the rows the fact's own
    filter keeps is cut to the rows whose keys occur among the fact's surviving keys before its map is built
    (TPC-H Q12's orders: 1.5M rows indexed for 31k lineitems). It costs a second pass over the fact's filter
    columns, so it is taken only when the partner's map would not fit the last-level cache (rows x 16 bytes
    over L3: SF1's Q12 stays whole, SF10's 15M orders at 240 MB do not) and skipped when the fact keeps over
    half of its rows or the native filter cannot run (``t`` as it was). ``MXFRAME_RUNTIME_JOIN_FILTER`` =
    0 (never) | 1 (whenever the row counts allow) | auto (default)."""
    mode = os.environ.get("MXFRAME_RUNTIME_JOIN_FILTER", "auto")
    if mode == "0" or not preds or (mode != "1" and t.num_rows * _MAP_BYTES_PER_ROW <= _l3_bytes()):
        return t
    keys = fact_keys(probe)
    if keys is None or t.num_rows <= 2 * keys[0].num_rows:
        return t
    from . import native_exec
    ktab, knames = keys
    out = native_exec.try_filter(comp, t, [], set(t.column_names),
                                 [native_exec.Lookup(list(bkeys), ktab, list(knames), [], [], semi=True)],
                                 keep_dict=keep_dict)
    if _TRACE:
        _trace(f"runtime join filter: partner {t.num_rows} -> {None if out is None else out.num_rows} rows "
               f"by {ktab.num_rows} fact keys")
    return t if out is None else out


class _Star:
    """A star join ready to run (``_star_setup``): the fact leaf's scan and predicates, one native
    ``Lookup`` per partner (most selective first) with the neighbour leaf it stands for, which partner
    every leaf belongs to, the columns each leaf delivers, and the residual conditions across partners."""
    __slots__ = ("scan", "preds", "lookups", "roots", "group", "deliver", "via_probe", "post_res", "fact")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def name_of(self, leaf: int, c: str) -> str:
        """The column of the joined rows (the star scan's output) holding column ``c`` of leaf ``leaf``."""
        if (leaf, c) in self.via_probe:
            return self.via_probe[(leaf, c)]
        return c if leaf == self.fact else _star_name(leaf, c)


def _execute_star(comp, fl, fact: int, get, want, leaf_need, coded=None, star=None) -> Optional[pa.Table]:
    """The join as one scan of leaf ``fact``: every other leaf belongs to the subtree of one of the
    fact's neighbours (the edges form a tree); each subtree is joined on its own (small tables) and
    becomes a partner looked up by key while the fact's filter runs (``native_exec.try_filter``).
    Residual conditions across subtrees run on the joined rows. None: declined (non-unique partner
    keys, unsupported types), the caller joins leaf by leaf."""
    from . import native_exec
    st = _star_setup(comp, fl, fact, get, want, leaf_need, coded) if star is None else star
    if not st:                       # declined (False: the fused attempt already found out)
        return None
    scan, preds, lookups, roots = st.scan, st.preds, st.lookups, st.roots
    deliver, post_res, name_of = st.deliver, st.post_res, st.name_of
    t0 = time.perf_counter()
    out = native_exec.try_filter(comp, scan.table, preds, set(deliver[fact]), lookups,
                                 keep_dict=coded[fact] if coded is not None else ())
    if _TRACE:
        _trace(f"star: scan leaf {fact} ({scan.table.num_rows} rows) with partners "
               f"{[(c, lk.build.num_rows) for (c, _), lk in zip(roots, lookups)]} -> "
               f"{None if out is None else out.num_rows} rows, {1e3*(time.perf_counter()-t0):.0f} ms "
               f"{native_exec.last_filter_maps() if out is not None else native_exec.last_filter()[1]}")
    if out is None:
        return None
    for pred, cm in post_res:
        if out.num_rows == 0:
            break
        t = pa.Table.from_arrays([out.column(name_of(leaf, c)) for leaf, c in cm.values()], names=list(cm))
        mask = comp._eval_predicate(pred, t)
        keep = _np_of(pc.fill_null(mask, _npscalar(False, type=pa.bool_())), zero_copy_only=False)
        if not keep.all():
            out = _take_rows(out, np.flatnonzero(keep).astype(np.int32))
    return pa.Table.from_arrays([out.column(name_of(*fl.src[nm])) for nm in want], names=want)


def _star_setup(comp, fl, fact: int, get, want, leaf_need, coded=None) -> Optional[_Star]:
    """Build the partners of the star join of leaf ``fact`` (see ``_execute_star``); None: declined."""
    from . import native_exec
    n = len(fl.leaves)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for ei, (a, b, _, _) in enumerate(fl.edges):
        adj[a].append(ei)
        adj[b].append(ei)
    group = {fact: -1}
    roots = []
    for ei in adj[fact]:
        a, b, _, _ = fl.edges[ei]
        c = b if a == fact else a
        if c == fact or c in group:
            return None
        group[c] = c
        roots.append((c, ei))
    for c, _ in roots:
        stack = [c]
        while stack:
            x = stack.pop()
            for ei in adj[x]:
                a, b, _, _ = fl.edges[ei]
                y = b if a == x else a
                if y not in group:
                    group[y] = c
                    stack.append(y)
    if len(group) != n:
        return None

    scan, preds = _scan_under_filters(fl.leaves[fact])
    if scan is None:
        return None
    members: Dict[int, List[int]] = {c: sorted(x for x, g in group.items() if g == c) for c, _ in roots}
    inner_res: Dict[int, list] = {c: [] for c, _ in roots}
    post_res = []
    for pred, cm in fl.residuals:
        gs = {group[leaf] for leaf, _ in cm.values()}
        if len(gs) == 1 and -1 not in gs and len(members[next(iter(gs))]) > 1:
            inner_res[gs.pop()].append((pred, cm))       # joined with its subtree
        else:
            post_res.append((pred, cm))                  # on the joined rows (a one-leaf partner too)
    # columns each leaf must deliver to the joined rows: outputs and cross-subtree residual inputs
    deliver: Dict[int, List[str]] = {i: [] for i in range(n)}
    for nm in want:
        leaf, c = fl.src[nm]
        if c not in deliver[leaf]:
            deliver[leaf].append(c)
    for _, cm in post_res:
        for leaf, c in cm.values():
            if c not in deliver[leaf]:
                deliver[leaf].append(c)

    # the fact's surviving key columns (its own filter, run natively once per probe columns), or None: the
    # runtime join filter's input (F-043; generalized to every single-leaf partner, F-046a slice 2)
    fact_keys_memo: Dict[tuple, Optional[tuple]] = {}

    def fact_keys(probe):
        key = tuple(probe)
        if key not in fact_keys_memo:
            kt = native_exec.try_filter(comp, scan.table, preds, set(probe))
            if kt is None or 2 * kt.num_rows > scan.table.num_rows:
                kt = None                     # a weak fact filter: the lookup would cost more than it saves
            fact_keys_memo[key] = None if kt is None else (kt.select(list(probe)), list(probe))
        return fact_keys_memo[key]

    lookups = []
    # semi partners' key columns, served from the fact's equal probe columns. Defensive: a join drops
    # its right key, so no output or residual can name one today
    via_probe: Dict[Tuple[int, str], str] = {}
    for c, ei in roots:
        a, b, ac, bc = fl.edges[ei]
        probe, bkeys = (ac, bc) if a == fact else (bc, ac)
        S = members[c]
        if len(S) == 1 and set(deliver[c]) <= set(bkeys):
            reduce = None
            if preds and _rows_of(fl.leaves[c]) > 2 * scan.table.num_rows:
                # a partner much larger than the (filtered) fact: filter it by the fact's keys first
                def reduce(probe=probe):
                    return fact_keys(probe)
            keys = _semi_partner(comp, fl.leaves[c], bkeys, reduce)
            if keys is not None:
                for pk, bk in zip(probe, bkeys):
                    via_probe[(c, bk)] = pk
                    if bk in deliver[c] and pk not in deliver[fact]:
                        deliver[fact].append(pk)
                lookups.append(native_exec.Lookup(probe, keys, list(bkeys), [], [], semi=True))
                continue
        bnames = []
        cols_of: Dict[int, List[str]] = {}
        for x in S:
            cols = list(deliver[x])
            if x == c:
                cols += [k for k in bkeys if k not in cols]
            cols_of[x] = cols
        if len(S) == 1:
            t = get(c).select(cols_of[c])
            t = _runtime_reduced(comp, t, bkeys, probe, preds, fact_keys,
                                 coded[c] if coded is not None else ())
            B = pa.Table.from_arrays([t.column(k) for k in cols_of[c]],
                                     names=[_star_name(c, k) for k in cols_of[c]])
        else:
            loc = {x: i for i, x in enumerate(S)}
            ledges = [(loc[a2], loc[b2], ac2, bc2) for a2, b2, ac2, bc2 in fl.edges
                      if a2 in loc and b2 in loc]
            lres = [(p, {k: (loc[leaf], col_) for k, (leaf, col_) in cm.items()}) for p, cm in inner_res[c]]
            # the subtree as a star of its own (Q8's orders -> customer -> nation: one scan of orders
            # instead of filtering it and joining leaf by leaf); else the leaf-by-leaf join
            bn = [_star_name(x, k) for x in S for k in cols_of[x]]
            sub = _Flat([fl.leaves[x] for x in S], ledges, bn,
                        {_star_name(x, k): (loc[x], k) for x in S for k in cols_of[x]}, lres)
            sub_fact = _star_fact(comp, sub)
            B = None
            sub_coded = None if coded is None else [coded[x] for x in S]    # (the subtree's leaves in its own numbering)
            if sub_fact is not None:
                B = _execute_star(comp, sub, sub_fact, lambda i: get(S[i]), bn, None, sub_coded)
            nested_codes = B is not None and sub_coded is not None
            if B is None:
                lt = [get(x) for x in S]
                rows = [t.num_rows for t in lt]
                base = [_base_rows(fl.leaves[x], rows[loc[x]]) for x in S]
                idx, _ = _join_indices(comp, lt, rows, base, ledges, lres, {loc[x] for x in S if cols_of[x]})
                arrays, names = [], []
                for x in S:
                    if not cols_of[x]:
                        continue
                    t = lt[loc[x]].select(cols_of[x])
                    ix = idx.get(loc[x])
                    g = t if ix is None else _gather_table(comp, t, ix, decode_dicts=True)
                    arrays += [g.column(k) for k in cols_of[x]]
                    names += [_star_name(x, k) for k in cols_of[x]]
                B = pa.Table.from_arrays(arrays, names=names)
        outs = [nm for nm in B.column_names if nm.split(":", 1)[0] != str(c) or
                nm.split(":", 1)[1] in deliver[c]]
        keep_b = {_star_name(c, k) for k in (coded[c] if coded is not None and len(S) == 1 else ())}
        if len(S) > 1 and nested_codes:           # a nested star's dictionary columns stay codes up to the lookup
            keep_b = {_star_name(x, k) for x in S for k in coded[x]}
        lookups.append(native_exec.Lookup(probe, B, [_star_name(c, k) for k in bkeys], outs, outs, keep=keep_b))

    # probe the most selective partner first: a row stops at its first miss (Q8: the filtered part
    # table before supplier and the orders subtree)
    def kept_share(c, lk):
        sk = _semi_keys(fl.leaves[c])
        sc = sk[0] if sk is not None else _scan_under_filters(fl.leaves[c])[0]
        return lk.build.num_rows / max(1, sc.table.num_rows) if sc is not None else 1.0

    order = sorted(range(len(lookups)), key=lambda i: kept_share(roots[i][0], lookups[i]))
    roots, lookups = [roots[i] for i in order], [lookups[i] for i in order]
    return _Star(scan=scan, preds=preds, lookups=lookups, roots=roots, group=group, deliver=deliver,
                 via_probe=via_probe, post_res=post_res, fact=fact)


def fused_join_aggregate(comp, plan, having_pred, post_ops, distinct: bool = False) -> Optional[pa.Table]:
    """``[Sort Limit] [HAVING] Aggregate([Filter]* (inner-join tree))`` as ONE native call (F-046a slice 3):
    the fact leaf's filter, the partner lookups, the conditions across partners (and filters above the join)
    and the aggregate run per block, and no joined row is written. The partners are built as the star scan
    builds them (``_star_setup``); ``native_exec.try_execute_join`` compiles the aggregate over the joined
    schema. None: the plan does not fit (or ``MXFRAME_FUSED_JOIN_AGG=0``), the caller materializes the join."""
    from . import native_exec
    comp._star_stash = None
    if native_exec.fused_join_blocked(comp) is not None:         # before any partner is built
        return None
    plan = comp._inline_project_into_aggregate(plan)
    if not isinstance(plan, Aggregate) or not plan.aggs:
        return None
    above, node = [], plan.input
    while isinstance(node, Filter):
        above.append(node.predicate)
        node = node.input
    if not (isinstance(node, Join) and node.how == "inner"):
        return None
    needed: Set[str] = set()
    for e in list(plan.group_by) + list(plan.aggs) + above:
        needed |= _cols(e)
    if _join_cached(comp, node, needed):                         # the old path would answer from its cache
        native_exec._decline("join result cached")
        return None
    setup = _inner_setup(comp, node, needed)
    if setup is None:
        return None
    fl, want, leaf_need, later, coded, tables, get = setup
    fact = _star_fact(comp, fl)
    if fact is None:
        return None
    # The aggregate reads a partner's deferred dictionary column as codes whatever it does with it (CASE,
    # group key, predicate): the bytecode compiler takes dictionaries. If the fused call declines, the old
    # path needs strings: it gets `setup`'s own `coded`, and partners built with these codes are dropped.
    fcoded = [set(c) if i == fact else set(c) | later[i] for i, c in enumerate(coded)]
    widened = fcoded != coded
    st = _star_setup(comp, fl, fact, get, want, leaf_need, fcoded)
    # whatever the fused call does, the old path (should it run) must not build the partners again
    comp._star_stash = (node, frozenset(needed), setup, st if st is not None else False)
    if st is None:
        return None
    lookup_of = {c: j for j, (c, _) in enumerate(st.roots)}
    vcols = {}                      # a column of the joined schema -> where the executor reads it
    for name in set(want) | {n for _, cm in st.post_res for n in cm}:
        leaf, c = fl.src[name]
        if (leaf, c) in st.via_probe or leaf == fact:
            vcols[name] = ("fact", st.via_probe.get((leaf, c), c))
        else:
            vcols[name] = ("partner", lookup_of[st.group[leaf]], _star_name(leaf, c))
    fj = native_exec.FusedJoin(st.scan, st.preds, st.lookups, vcols, above + [p for p, _ in st.post_res])
    if _TRACE:
        _trace(f"fused join + aggregate: scan leaf {fact} ({st.scan.table.num_rows} rows) with partners "
               f"{[(c, lk.build.num_rows) for (c, _), lk in zip(st.roots, st.lookups)]}")
    out = native_exec.try_execute_join(comp, plan, fj, having_pred, post_ops, distinct)
    if out is not None:
        comp._star_stash = None
    elif widened and st:                           # partners holding codes the old path cannot read
        comp._star_stash = (node, frozenset(needed), setup, None)
    return out


def _join_cached(comp, node, needed) -> bool:
    """True when ``execute_join`` holds this join's result already (a repeated query)."""
    from . import custom_ops as _co
    key = _stable_key(comp, node)
    if key is None:
        return False
    return any((key, frozenset(needed), ok, _keep_sig(node), comp._session_device) in _co._JOIN_RESULT_CACHE
               for ok in (True, False))


def _join_indices(comp, tables, rows, base, edges, residuals, out_leaves):
    """Row-index vectors (leaf -> rows of that leaf, None: all) of the inner join of ``tables`` over
    ``edges``, in the order ``plan_order`` picks; residual conditions applied as soon as their leaves
    are joined. Only leaves in ``out_leaves`` (and those still needed to join) keep an index."""
    start, steps = plan_order(rows, base, edges)
    if _TRACE:
        _trace(f"leaves rows={rows} base={base} start={start} steps={[(n, r) for n, _, r in steps]}")
    idx: Dict[int, Optional[np.ndarray]] = {start: None}     # leaf -> row index of the relation (None: identity)
    n_rel = rows[start]
    consumed: Set[int] = set()
    pending = [(p, cm, {leaf for leaf, _ in cm.values()}) for p, cm in residuals]
    if pending:
        idx, n_rel = _apply_residuals(comp, tables, idx, n_rel, pending)
    for new, ei, r_leaf in steps:
        a, b, ac, bc = edges[ei]
        if r_leaf == a:
            rcols_r, rcols_new = ac, bc
        else:
            rcols_r, rcols_new = bc, ac
        consumed.add(ei)
        if n_rel == 0 or rows[new] == 0:
            n_rel = 0
            idx = {x: np.empty(0, dtype=np.int32) for x in list(idx) + [new]}
            continue
        t0 = time.perf_counter()
        rel_keys = [_gather_key(tables[r_leaf].column(c), idx[r_leaf]) for c in rcols_r]
        new_keys = [tables[new].column(c) for c in rcols_new]
        t1 = time.perf_counter()
        ri, ni, t2 = _join_pairs(comp, rel_keys, new_keys, n_rel, rows[new])
        t3 = time.perf_counter()
        pending_leaves = set().union(*[lv for _, _, lv in pending]) if pending else set()
        alive = {x for x in idx if x in out_leaves or x in pending_leaves or any(
            e not in consumed and (edges[e][0] == x or edges[e][1] == x) for e in range(len(edges)))}
        idx = {x: (ri if idx[x] is None else _take(idx[x], ri)) for x in alive}
        idx[new] = ni
        if _TRACE:
            t4 = time.perf_counter()
            _trace(f"add leaf {new} via {r_leaf}: rel {n_rel} x {rows[new]} -> {len(ri)}  "
                   f"keys {1e3*(t1-t0):.0f} ms, encode {1e3*(t2-t1):.0f} ms, join {1e3*(t3-t2):.0f} ms, compose {1e3*(t4-t3):.0f} ms")
        n_rel = len(ri)
        if pending:
            idx, n_rel = _apply_residuals(comp, tables, idx, n_rel, pending)

    return idx, n_rel


def _inner_setup(comp, node, needed):
    """What running a tree of inner joins needs: the flattened tree, the output columns wanted, what each
    leaf must deliver, the deferred dictionary columns and a leaf evaluator. None: not flattenable."""
    fl = _flatten(node)
    if fl is None:
        return None
    names = fl.names
    want = names if needed is None else ([n for n in names if n in needed] or names[:1])
    n_leaves = len(fl.leaves)
    leaf_need: List[Set[str]] = [set() for _ in range(n_leaves)]
    for n in want:
        leaf, c = fl.src[n]
        leaf_need[leaf].add(c)
    for a, b, ac, bc in fl.edges:
        leaf_need[a].update(ac)
        leaf_need[b].update(bc)
    for _, cm in fl.residuals:
        for leaf, c in cm.values():
            leaf_need[leaf].add(c)
    # dictionary columns resolve_plan deferred (decoded only because the plan has a join): decoded
    # after the final gather, on the joined rows only, unless they are join keys or residual inputs
    later: List[Set[str]] = []
    coded: List[Set[str]] = []           # deferred ones the plan reads as dictionaries: never decoded
    for i in range(n_leaves):
        sc, _ = _scan_under_filters(fl.leaves[i])
        later.append(set(getattr(sc, "_mx_decode_after", None) or ()) if sc is not None else set())
        coded.append(set(getattr(sc, "_mx_keep_dict", None) or ()) if sc is not None else set())
    for a, b, ac, bc in fl.edges:
        later[a] -= set(ac)
        later[b] -= set(bc)
    for _, cm in fl.residuals:
        for leaf, c in cm.values():
            later[leaf].discard(c)
    for i in range(n_leaves):
        coded[i] &= later[i]
    tables: List[Optional[pa.Table]] = [None] * n_leaves

    def get(i: int) -> pa.Table:             # a leaf evaluated once, when first needed
        if tables[i] is None:
            tables[i] = leaf_table(comp, fl.leaves[i], leaf_need[i], keep=later[i])
        return tables[i]

    return fl, want, leaf_need, later, coded, tables, get


def _execute_inner(comp, node, needed, dict_ok=False):
    stash, comp._star_stash = getattr(comp, "_star_stash", None), None       # from a declined fused attempt
    star = None
    if stash is not None and stash[0] is node and stash[1] == (None if needed is None else frozenset(needed)):
        setup, star = stash[2], stash[3]
    else:
        setup = _inner_setup(comp, node, needed)
    if setup is None:
        return None
    fl, want, leaf_need, later, coded, tables, get = setup
    n_leaves = len(fl.leaves)
    fact = _star_fact(comp, fl)
    if fact is not None:
        out = _execute_star(comp, fl, fact, get, want, leaf_need, coded,
                            star if star is not None and (star is False or star.fact == fact) else None)
        if out is not None:
            return out
    for i in range(n_leaves):
        get(i)
    rows = [t.num_rows for t in tables]
    base = [_base_rows(fl.leaves[i], rows[i]) for i in range(n_leaves)]
    out_leaves = {fl.src[n][0] for n in want}
    idx, n_rel = _join_indices(comp, tables, rows, base, fl.edges, fl.residuals, out_leaves)

    t_g = time.perf_counter()
    by_leaf: Dict[int, List[Tuple[str, str]]] = {}
    for n in want:
        leaf, c = fl.src[n]
        by_leaf.setdefault(leaf, []).append((n, c))
    arrays = {}
    for leaf, cols in by_leaf.items():
        names_c = list(dict.fromkeys(c for _, c in cols))
        sub = tables[leaf].select(names_c)
        if idx[leaf] is None:
            got = {c: sub.column(c) for c in names_c}
        else:
            got = {}
            rest = names_c
            if needed is not None and dict_ok:          # a consumer above reads these: dictionary columns are fine
                for c in names_c:
                    d = _dictionary_gather(sub.column(c), idx[leaf])
                    if d is not None:
                        got[c] = d
                rest = [c for c in names_c if c not in got]
            for c in [c for c in rest if c in coded[leaf] and pa.types.is_dictionary(sub.schema.field(c).type)]:
                got[c] = pc.take(sub.column(c), _npa(idx[leaf]))     # codes gathered, never decoded
            rest = [c for c in rest if c not in got]
            if rest:
                # deferred dictionary columns are decoded right after anyway: let the gather do it
                deferred_only = all(c in later[leaf] for c in rest
                                    if pa.types.is_dictionary(sub.schema.field(c).type))
                taken = _gather_table(comp, sub.select(rest), idx[leaf], decode_dicts=deferred_only)
                got.update({c: taken.column(c) for c in rest})
        if later[leaf]:
            from .arrow_dict import decode
            for c in names_c:
                if c in later[leaf] and c not in coded[leaf] and pa.types.is_dictionary(got[c].type):
                    got[c] = decode(got[c])
        for n, c in cols:
            arrays[n] = got[c]
    if _TRACE:
        _trace(f"gather {len(want)} columns of {n_rel} rows: {1e3*(time.perf_counter()-t_g):.0f} ms")
    return pa.Table.from_arrays([arrays[n] for n in want], names=want)
