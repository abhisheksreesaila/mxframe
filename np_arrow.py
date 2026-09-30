"""Arrow arrays from NumPy without importing pandas.

``pa.array`` asks PyArrow's lazy pandas shim whether its input is a pandas
object, which imports pandas (~150 ms, ~300 modules) the first time it runs in a
process, even for plain NumPy input. The engine turns NumPy results into Arrow
on every query, so a user who never touches pandas paid that on their first
query. ``array`` builds the same Arrow array straight from the NumPy buffer
(zero-copy for contiguous numeric input) and defers to ``pa.array`` for
anything else.
"""

from __future__ import annotations

import sys

import numpy as np
import pyarrow as pa

_NUMERIC = {
    np.dtype("int8"): pa.int8(), np.dtype("int16"): pa.int16(),
    np.dtype("int32"): pa.int32(), np.dtype("int64"): pa.int64(),
    np.dtype("uint8"): pa.uint8(), np.dtype("uint16"): pa.uint16(),
    np.dtype("uint32"): pa.uint32(), np.dtype("uint64"): pa.uint64(),
    np.dtype("float32"): pa.float32(), np.dtype("float64"): pa.float64(),
}


def _is_masked(obj) -> bool:
    ma = sys.modules.get("numpy.ma")          # touching np.ma would import it (~9 ms)
    return ma is not None and isinstance(obj, ma.MaskedArray)


def array(obj, type=None, **kwargs) -> pa.Array:
    """``pa.array(obj, type)`` for 1-D numeric/bool NumPy input, without pandas."""
    if kwargs or not isinstance(obj, np.ndarray) or obj.ndim != 1 or _is_masked(obj):
        return pa.array(obj, type=type, **kwargs)
    a = obj
    if a.dtype == np.bool_:
        n = len(a)
        bits = np.packbits(a, bitorder="little") if n else np.zeros(0, np.uint8)
        out = pa.Array.from_buffers(pa.bool_(), n, [None, pa.py_buffer(bits)])
    else:
        at = _NUMERIC.get(a.dtype.newbyteorder("=") if a.dtype.byteorder not in "=|" else a.dtype)
        if at is None or a.dtype.byteorder == ">":
            return pa.array(obj, type=type)
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        out = pa.Array.from_buffers(at, len(a), [None, pa.py_buffer(a)])
    if type is not None and out.type != type:
        out = out.cast(type, safe=True)
    return out


_NP_OF = {pa.int8(): np.int8, pa.int16(): np.int16, pa.int32(): np.int32, pa.int64(): np.int64,
          pa.uint8(): np.uint8, pa.uint16(): np.uint16, pa.uint32(): np.uint32, pa.uint64(): np.uint64,
          pa.float32(): np.float32, pa.float64(): np.float64,
          pa.date32(): np.int32}


def to_numpy(obj, zero_copy_only: bool = False, **kwargs):
    """``obj.to_numpy(...)`` for Arrow arrays without importing pandas.

    Null-free fixed-width numeric (and bool) arrays become NumPy views of the Arrow
    buffer, exactly what PyArrow returns for them. Everything else - chunked arrays
    with several chunks, nulls, strings, dictionaries, non-Arrow objects such as
    MAX tensors - goes through the object's own ``to_numpy``.
    """
    if isinstance(obj, pa.ChunkedArray):
        if obj.num_chunks == 1:
            return to_numpy(obj.chunk(0), zero_copy_only=zero_copy_only, **kwargs)
        if obj.num_chunks == 0:
            dt = _NP_OF.get(obj.type)
            if dt is not None and not kwargs:
                return np.empty(0, dtype=dt)
        if not kwargs and _NP_OF.get(obj.type) is not None and obj.null_count == 0:
            return np.concatenate([to_numpy(c) for c in obj.chunks])
    if not isinstance(obj, pa.Array) or kwargs or obj.null_count:
        return obj.to_numpy(zero_copy_only=zero_copy_only, **kwargs) if isinstance(obj, (pa.Array, pa.ChunkedArray)) \
            else obj.to_numpy(**kwargs)
    t = obj.type
    if t == pa.bool_():
        if zero_copy_only:
            return obj.to_numpy(zero_copy_only=True)       # raises, as PyArrow does
        n = len(obj)
        if n == 0:
            return np.zeros(0, dtype=np.bool_)
        bits = np.frombuffer(obj.buffers()[1], dtype=np.uint8)
        return np.unpackbits(bits, bitorder="little", count=obj.offset + n)[obj.offset:].astype(np.bool_)
    dt = _NP_OF.get(t)
    if dt is None or t == pa.date32():
        return obj.to_numpy(zero_copy_only=zero_copy_only)
    buf = obj.buffers()[1]
    if buf is None:
        return np.empty(0, dtype=dt)
    return np.frombuffer(buf, dtype=dt, count=len(obj), offset=obj.offset * np.dtype(dt).itemsize)


# --- gathering rows -------------------------------------------------------------------------

TAKE_CHUNK_ROWS = 1 << 22        # 4M rows per slice for variable-width columns


def _has_32bit_offsets(t) -> bool:
    return pa.types.is_string(t) or pa.types.is_binary(t) or pa.types.is_list(t)


def take_rows(obj, idx):
    """``obj.take(idx)`` for a table, chunked or plain array and a NumPy row-index array.

    A column with 32-bit offsets (string, binary, list) gathered into more than 2 GiB of
    values comes back as a corrupt array (``Negative offsets in binary array``): joining
    60M TPC-H lineitem rows to their orders gathers 3 GB of ``o_comment``.  Such columns are
    gathered in slices of ``TAKE_CHUNK_ROWS`` rows and returned as a chunked array; every
    other column is gathered in one call, as before."""
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    n = len(idx)

    def one(col, sliced: bool):
        if not sliced or n <= TAKE_CHUNK_ROWS:
            return col.take(array(idx))
        parts = [col.take(array(idx[i:i + TAKE_CHUNK_ROWS])) for i in range(0, n, TAKE_CHUNK_ROWS)]
        chunks = [c for p in parts for c in (p.chunks if isinstance(p, pa.ChunkedArray) else [p])]
        return pa.chunked_array(chunks, type=col.type)

    if isinstance(obj, pa.Table):
        if n <= TAKE_CHUNK_ROWS or not any(_has_32bit_offsets(t) for t in obj.schema.types):
            return obj.take(array(idx))
        cols = [one(obj.column(i), _has_32bit_offsets(obj.schema.types[i])) for i in range(obj.num_columns)]
        return pa.Table.from_arrays(cols, schema=obj.schema)
    return one(obj, _has_32bit_offsets(obj.type))
