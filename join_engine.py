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

import os
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .lazy_frame import (join_output_names, Scan, Filter, Project, Aggregate, Sort, Limit,
                         Distinct, Tail, Join)
from .np_arrow import array as _npa, to_numpy as _np_of, scalar as _npscalar, take_rows as _take_rows
from .optimizer import _output_columns as plan_columns


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


def materialize(comp, plan, needed: Optional[Set[str]] = None):
    """Replace every Join in ``plan`` by a ``Scan`` of its result.

    ``needed`` is the set of column names the caller reads from ``plan``'s output (None: all).
    It is narrowed on the way down so each join only produces the columns read above it.
    """
    if isinstance(plan, Scan):
        return plan
    if isinstance(plan, Join):
        return Scan(execute_join(comp, plan, needed))
    if isinstance(plan, Filter):
        return Filter(materialize(comp, plan.input, _union(needed, _cols(plan.predicate))), plan.predicate)
    if isinstance(plan, Project):
        used: Set[str] = set()
        produced = set()
        for i, e in enumerate(plan.exprs):
            used |= _cols(e)
            produced.add(_out_name(e, i))
        # a column the parent reads that this Project does not produce passes through from below
        # (a Project of new columns only, with_columns after a join, is completed by the aggregate
        # that inlines it)
        below = used if needed is None else (used | (needed - produced))
        return Project(materialize(comp, plan.input, below), plan.exprs)
    if isinstance(plan, Aggregate):
        used = set()
        for e in list(plan.group_by) + list(plan.aggs):
            used |= _cols(e)
        return Aggregate(materialize(comp, plan.input, used), plan.group_by, plan.aggs)
    if isinstance(plan, Sort):
        used = set()
        for e in plan.by:
            used |= _cols(e) if hasattr(e, "op") else {e}
        return Sort(materialize(comp, plan.input, _union(needed, used)), plan.by, plan.descending)
    if isinstance(plan, Limit):
        return Limit(materialize(comp, plan.input, needed), plan.n)
    if isinstance(plan, Tail):
        return Tail(materialize(comp, plan.input, needed), plan.n)
    if isinstance(plan, Distinct):
        below = None if (needed is None or not plan.subset) else (needed | set(plan.subset))
        return Distinct(materialize(comp, plan.input, below), plan.subset)
    if hasattr(plan, "input"):
        plan.input = materialize(comp, plan.input, None)
    return plan


# ── leaf tables ─────────────────────────────────────────────────────────────────────────────

def _select(table: pa.Table, needed: Optional[Set[str]]) -> pa.Table:
    if needed is None:
        return table
    names = [c for c in table.column_names if c in needed]
    if not names:
        names = table.column_names[:1]          # keep the row count
    return table if len(names) == table.num_columns else table.select(names)


def leaf_table(comp, plan, needed: Optional[Set[str]]) -> pa.Table:
    """``plan`` evaluated to a table holding (at least) the ``needed`` columns."""
    if isinstance(plan, Scan):
        return _select(plan.table, needed)
    if isinstance(plan, Filter):
        inner = leaf_table(comp, plan.input, _union(needed, _cols(plan.predicate)))
        mask = comp._eval_predicate(plan.predicate, inner)
        return _select(inner, needed).filter(mask)      # the mask is done: drop the predicate-only columns first
    if isinstance(plan, Join):
        return execute_join(comp, plan, needed)
    return comp._plan_to_table(materialize(comp, plan, needed))


# ── flattening a tree of inner joins ────────────────────────────────────────────────────────

class _Flat:
    """Leaves, key edges and output naming of a group of inner joins."""
    __slots__ = ("leaves", "edges", "names", "src")

    def __init__(self, leaves, edges, names, src):
        self.leaves = leaves          # leaf plans
        self.edges = edges            # (leaf a, leaf b, a columns, b columns)
        self.names = names            # output column names, in order
        self.src = src                # output name -> (leaf, column)


def _flatten(node) -> Optional[_Flat]:
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
            return _Flat(left.leaves + right.leaves, edges, left.names + [o for _, o in mapping], src)
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
        return x.combine_chunks() if x.num_chunks != 1 else x.chunk(0)
    return x


def _as_np_int(x):
    """A NumPy integer array for a null-free integer key column, else None."""
    if isinstance(x, np.ndarray):
        return x if x.dtype.kind in "iu" else None
    if pa.types.is_integer(x.type) and x.null_count == 0:
        a = _np_of(x, zero_copy_only=False)
        return a if a.dtype.kind in "iu" else None
    return None


def _minmax(a: np.ndarray) -> Tuple[int, int]:
    return (int(a.min()), int(a.max())) if len(a) else (0, 0)


def _pack(lparts: List[np.ndarray], rparts: List[np.ndarray]):
    """Combine several integer key columns into one int64 key (mixed radix); None if it would overflow."""
    los, strides, total = [], [], 1
    for l, r in zip(lparts, rparts):
        (lo1, hi1), (lo2, hi2) = _minmax(l), _minmax(r)
        lo, hi = min(lo1, lo2), max(hi1, hi2)
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
    if lo >= 0 and hi <= max(1024, 4 * n):
        return (np.ascontiguousarray(lk, dtype=np.int32), np.ascontiguousarray(rk, dtype=np.int32))
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
                enc = pa.concat_arrays([_npa(np.ascontiguousarray(lk, dtype=np.int64)),
                                        _npa(np.ascontiguousarray(rk, dtype=np.int64))]).dictionary_encode()
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
            la, ra = la.cast(common), ra.cast(common)
        enc = pa.concat_arrays([la, ra]).dictionary_encode()
        codes = _np_of(enc.indices.fill_null(_npscalar(-1, type=enc.indices.type)),
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
            enc = _npa(both).dictionary_encode()
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
        return a[idx]
    return _as_pa(_take_rows(col, idx))


# ── kernels ─────────────────────────────────────────────────────────────────────────────────

def inner_pairs(comp, probe_keys: np.ndarray, build_keys: np.ndarray):
    """(probe_idx, build_idx) int32 pairs of the equal keys on the session device."""
    if comp._session_device == "gpu" and comp._aot_gpu is not None:
        return comp._aot_gpu.hash_join(probe_keys, build_keys)
    if comp._session_device == "gpu" and comp._aot is None:
        return comp._hash_join_mojo_gpu(probe_keys, build_keys)
    return comp._hash_join_mojo_cpu(probe_keys, build_keys)


def _gather_table(comp, table: pa.Table, idx: np.ndarray) -> pa.Table:
    if table.num_columns == 0:
        return table
    if comp._session_device == "gpu" and comp._aot_gpu is not None and len(idx) > 0:
        return comp._aot_gpu.gather_table(table, idx)
    return _take_rows(table, idx)


# ── executing a join ────────────────────────────────────────────────────────────────────────

def _stable_key(comp, plan):
    if isinstance(plan, Join):
        l, r = _stable_key(comp, plan.left), _stable_key(comp, plan.right)
        if l is None or r is None:
            return None
        return ("J", l, r, tuple(plan.left_on), tuple(plan.right_on), plan.how)
    return type(comp)._plan_stable_key(plan)


def execute_join(comp, node, needed: Optional[Set[str]]) -> pa.Table:
    from . import custom_ops as _co
    cache = _co._JOIN_RESULT_CACHE
    key = _stable_key(comp, node)
    if key is not None:
        key = (key, None if needed is None else frozenset(needed), comp._session_device)
        hit = cache.get(key)
        if hit is not None:
            return hit
    out = _execute_join(comp, node, needed)
    if key is not None:
        cache[key] = out
        if len(cache) > 24:
            for old in list(cache.keys())[:12]:
                cache.pop(old, None)
    return out


def _legacy(comp, node):
    left = comp._plan_to_table(materialize(comp, node.left, None))
    right = comp._plan_to_table(materialize(comp, node.right, None))
    return comp._execute_hash_join(left, right, node.left_on, node.right_on, how=node.how)


def _execute_join(comp, node, needed):
    if node.how == "left":
        out = _execute_left(comp, node, needed)
    elif node.how == "inner":
        out = _execute_inner(comp, node, needed)
    else:
        out = None
    return out if out is not None else _legacy(comp, node)


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
            safe = np.where(matched, ri, 0).astype(np.int64, copy=False) if not all_matched else ri
            rtake = _take_rows(rt.select([o for o, _ in rwant]), safe)
            valid = None if all_matched else _npa(matched)
            for orig, out in rwant:
                col = rtake.column(orig)
                arrays[out] = col if valid is None else pc.if_else(valid, col, None)
    return pa.Table.from_arrays([arrays[n] for n in want], names=want)


MAX_RESIDUAL_EXPANSION = 8.0     # a composite key joined on one column may emit this many candidate pairs per row


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
        if lo < 0 or hi > max(1024, 4 * (n_rel + n_new)):
            continue
        if best is None or hi > best[1]:
            best = (j, hi + 1)
    if best is None or min(n_rel, n_new) / best[1] > MAX_RESIDUAL_EXPANSION:
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
        ri, ni = inner_pairs(comp, rk, nk)
    else:
        ni, ri = inner_pairs(comp, nk, rk)
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


def _execute_inner(comp, node, needed):
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
    tables = [leaf_table(comp, fl.leaves[i], leaf_need[i]) for i in range(n_leaves)]
    rows = [t.num_rows for t in tables]
    base = [_base_rows(fl.leaves[i], rows[i]) for i in range(n_leaves)]

    start, steps = plan_order(rows, base, fl.edges)
    if _TRACE:
        _trace(f"leaves rows={rows} base={base} start={start} steps={[(n, r) for n, _, r in steps]}")
    out_leaves = {fl.src[n][0] for n in want}

    idx: Dict[int, Optional[np.ndarray]] = {start: None}     # leaf -> row index of the relation (None: identity)
    n_rel = rows[start]
    consumed: Set[int] = set()
    for new, ei, r_leaf in steps:
        a, b, ac, bc = fl.edges[ei]
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
        alive = {x for x in idx if x in out_leaves or any(
            e not in consumed and (fl.edges[e][0] == x or fl.edges[e][1] == x) for e in range(len(fl.edges)))}
        idx = {x: (ri if idx[x] is None else idx[x][ri]) for x in alive}
        idx[new] = ni
        if _TRACE:
            t4 = time.perf_counter()
            _trace(f"add leaf {new} via {r_leaf}: rel {n_rel} x {rows[new]} -> {len(ri)}  "
                   f"keys {1e3*(t1-t0):.0f} ms, encode {1e3*(t2-t1):.0f} ms, join {1e3*(t3-t2):.0f} ms, compose {1e3*(t4-t3):.0f} ms")
        n_rel = len(ri)

    t_g = time.perf_counter()
    by_leaf: Dict[int, List[Tuple[str, str]]] = {}
    for n in want:
        leaf, c = fl.src[n]
        by_leaf.setdefault(leaf, []).append((n, c))
    arrays = {}
    for leaf, cols in by_leaf.items():
        sub = tables[leaf].select(list(dict.fromkeys(c for _, c in cols)))
        got = sub if idx[leaf] is None else _gather_table(comp, sub, idx[leaf])
        for n, c in cols:
            arrays[n] = got.column(c)
    if _TRACE:
        _trace(f"gather {len(want)} columns of {n_rel} rows: {1e3*(time.perf_counter()-t_g):.0f} ms")
    return pa.Table.from_arrays([arrays[n] for n in want], names=want)
