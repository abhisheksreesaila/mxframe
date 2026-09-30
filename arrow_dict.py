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
from .np_arrow import array as _npa, to_numpy as _np_of, scalar as _npscalar
import pyarrow.compute as pc

__all__ = ["is_dictionary", "decode", "combine_dictionary", "encode_key", "dict_predicate"]


def is_dictionary(arr_or_type) -> bool:
    typ = getattr(arr_or_type, "type", arr_or_type)
    return pa.types.is_dictionary(typ)


def decode(arr):
    """Return ``arr`` with dictionary encoding removed (no-op for plain arrays)."""
    if not is_dictionary(arr):
        return arr
    return arr.cast(arr.type.value_type)


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
        return col.unify_dictionaries().combine_chunks()
    return col


def encode_key(col):
    """Group-key encoding: an object with ``.indices`` and ``.dictionary``.

    Dictionary columns reuse their own indices (no hashing pass over the rows);
    dictionary entries no row refers to are dropped so every dictionary entry is
    a real group.
    Plain columns go through ``dictionary_encode()``.
    """
    if not is_dictionary(col):
        arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        return arr.dictionary_encode()
    d = combine_dictionary(col)
    if d.null_count:
        # Keep the historical null handling of dictionary_encode().
        return decode(d).dictionary_encode()
    n = len(d.dictionary)
    if len(pc.unique(d.indices)) == n:
        return d
    idx = _np_of(d.indices, zero_copy_only=False)
    used = np.zeros(n, dtype=bool)
    used[idx] = True
    remap = (np.cumsum(used) - 1).astype(np.int32)
    return pa.DictionaryArray.from_arrays(
        _npa(remap[idx]), d.dictionary.filter(_npa(used)))


_CMP = {
    "eq": pc.equal, "ne": pc.not_equal, "gt": pc.greater,
    "ge": pc.greater_equal, "lt": pc.less, "le": pc.less_equal,
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
        fn = lambda v: _CMP[op](v, _npscalar(args[1].args[0]))   # noqa: E731
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
            out = pc.if_else(d.indices.is_null(), _npscalar(None in args[1]), out)
        return out
    return pc.take(per_entry, d.indices)
