"""Helpers for Arrow dictionary-encoded (categorical) columns.

Parquet stores low-cardinality strings dictionary-encoded, and ``scan_parquet``
can hand those columns to the engine as ``pa.dictionary`` arrays.  The engine
consumes them natively in two places (group-key encoding and string filters);
everywhere else they are decoded back to plain values at the plan boundary
(see ``io_parquet.resolve_plan``), so results are identical to the plain path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa
from .np_arrow import array as _npa, to_numpy as _np_of, scalar as _npscalar, combine as _combine, canonical_floats as _canonical_floats
from ._lazy_pc import pc

__all__ = ["is_dictionary", "decode", "combine_dictionary", "encode_key", "dict_predicate"]


def is_dictionary(arr_or_type) -> bool:
    typ = getattr(arr_or_type, "type", arr_or_type)
    return pa.types.is_dictionary(typ)


def decode(arr):
    """Return ``arr`` with dictionary encoding removed (no-op for plain arrays)."""
    if not is_dictionary(arr):
        return arr
    return pc.cast(arr, arr.type.value_type)   # not arr.cast: it imports pyarrow.compute (_lazy_pc)


def combine_dictionary(col) -> pa.DictionaryArray:
    """One ``DictionaryArray`` for a (chunked) dictionary column.

    Chunks read from different row groups carry different dictionaries; they are
    unified first so a single set of indices addresses a single dictionary.
    """
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 0:
            return _npa([], type=col.type)
        if col.num_chunks == 1:
            return col.chunk(0)
        return _combine(col.unify_dictionaries())
    return col


def _encode_int(arr):
    """``arr.dictionary_encode()`` for an integer column, by direct addressing (F-020), or None.

    Same groups, same numbering (first appearance) as Arrow's hash; the kernel reads the values in
    place and gives up (None) on nulls, wide value ranges and small inputs.
    """
    t = arr.type
    if not pa.types.is_integer(t) or t == pa.uint64() or arr.null_count or len(arr) < 50_000:
        return None
    from . import group_keys
    r = group_keys.encode_int(_np_of(arr, zero_copy_only=False))
    if r is None:
        return None
    ids, uniq, _ = r
    return pa.DictionaryArray.from_arrays(_npa(ids), _npa(uniq.astype(_NP_OF_ARROW[t])), safe=False)


_NP_OF_ARROW = {pa.int8(): np.int8, pa.int16(): np.int16, pa.int32(): np.int32, pa.int64(): np.int64,
                pa.uint8(): np.uint8, pa.uint16(): np.uint16, pa.uint32(): np.uint32}


def _encode_float(arr):
    """``dictionary_encode`` for a float key with SQL equality (F-077): -0.0 and 0.0 are one group
    and so are all NaNs. Each group still reports the value of its first row, like DuckDB."""
    enc = pc.dictionary_encode(_canonical_floats(arr))
    if enc.indices.null_count or not len(arr):
        return enc
    idx = _np_of(enc.indices, zero_copy_only=False)
    run = np.maximum.accumulate(idx)                      # codes are numbered by first appearance
    first = np.flatnonzero(np.concatenate(([True], run[1:] != run[:-1])))
    return pa.DictionaryArray.from_arrays(enc.indices, pc.take(arr, _npa(first.astype(np.int64))))


def encode_key(col):
    """Group-key encoding: an object with ``.indices`` and ``.dictionary``.

    Dictionary columns reuse their own indices (no hashing pass over the rows);
    dictionary entries no row refers to are dropped so every dictionary entry is
    a real group.
    Plain columns go through ``dictionary_encode()``.
    """
    if not is_dictionary(col):
        arr = _combine(col) if isinstance(col, pa.ChunkedArray) else col
        fast = _encode_int(arr)
        if fast is not None:
            return fast
        if arr.type in (pa.float32(), pa.float64()):
            return _encode_float(arr)
        return pc.dictionary_encode(arr)
    d = combine_dictionary(col)
    if d.null_count:
        # Keep the historical null handling of dictionary_encode().
        return pc.dictionary_encode(decode(d))
    n = len(d.dictionary)
    idx = _np_of(d.indices, zero_copy_only=False)
    used = np.zeros(n, dtype=bool)
    used[idx] = True
    if used.all():
        return d
    remap = (np.cumsum(used) - 1).astype(np.int32)
    return pa.DictionaryArray.from_arrays(
        _npa(remap[idx]), pc.filter(d.dictionary, _npa(used)))


_CMP = {   # names: pyarrow.compute loads on first use (_lazy_pc)
    "eq": "equal", "ne": "not_equal", "gt": "greater",
    "ge": "greater_equal", "lt": "less", "le": "less_equal",
}


def dict_predicate(expr, table) -> Optional[pa.Array]:
    """Evaluate ``col <cmp> literal`` / isin / startswith / contains natively.

    Works when the column is dictionary typed: the predicate is evaluated once per
    dictionary entry and expanded through the indices.  Returns ``None`` when the
    expression is not of that shape (the caller then decodes and evaluates normally).
    """
    op, args = expr.op, expr.args
    if op in _CMP:
        if not (args[1].op == "lit"):
            return None
        fn = lambda v: getattr(pc, _CMP[op])(v, _npscalar(args[1].args[0]))   # noqa: E731
    elif op == "isin":
        fn = lambda v: pc.is_in(v, value_set=_npa(args[1], type=v.type))   # noqa: E731
    elif op == "startswith":
        fn = lambda v: pc.starts_with(v, pattern=args[1])          # noqa: E731
    elif op == "contains":
        fn = lambda v: pc.match_substring(v, pattern=args[1])      # noqa: E731
    else:
        return None
    target = args[0]
    if target.op != "col" or target.args[0] not in table.column_names:
        return None
    col = table.column(target.args[0])
    if not is_dictionary(col):
        return None
    d = combine_dictionary(col)
    per_entry = fn(d.dictionary)
    if op == "isin":
        # is_in is never null.  Null rows (null index) match only a None in the value set.
        out = pc.take(per_entry, d.indices)
        if d.null_count:
            out = pc.if_else(pc.is_null(d.indices), _npscalar(None in args[1]), out)
        return out
    return pc.take(per_entry, d.indices)
