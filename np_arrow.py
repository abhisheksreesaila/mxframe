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


def _strings(values) -> pa.Array:
    data = [v.encode("utf-8") for v in values]
    offsets = np.zeros(len(data) + 1, dtype=np.int32)
    np.cumsum([len(b) for b in data], out=offsets[1:])
    return pa.Array.from_buffers(pa.string(), len(data),
                                 [None, pa.py_buffer(offsets), pa.py_buffer(b"".join(data))])


def _same_kind(kind, type) -> bool:
    """True when a Python value of ``kind`` converts to ``type`` exactly as PyArrow would."""
    if type is None:
        return True
    if kind is bool:
        return pa.types.is_boolean(type)
    if kind is int:
        return pa.types.is_integer(type) or pa.types.is_floating(type)
    if kind is float:
        return pa.types.is_floating(type)
    if kind is str:
        return pa.types.is_string(type) or pa.types.is_large_string(type)
    return False


def _from_pylist(values, type):
    """Homogeneous list of bool / int / float / str (no None) without pandas, else None.

    Only takes the fast path when PyArrow would produce the same result; anything
    else (typed conversions like int->decimal, str->date, numpy scalars of other
    widths) returns None so the caller defers to PyArrow.
    """
    if not values:
        return None
    if all(isinstance(v, np.generic) for v in values):
        a = np.asarray(values)
        if a.dtype in _NUMERIC or a.dtype == np.bool_:
            out = array(a)
            return out if type is None or out.type == type else None
        return None
    if any(isinstance(v, np.generic) for v in values):
        return None
    kinds = {bool if isinstance(v, bool) else int if isinstance(v, int) else
             float if isinstance(v, float) else str if isinstance(v, str) else None for v in values}
    if kinds == {int, float}:
        k = float
    elif len(kinds) == 1:
        k = next(iter(kinds))
    else:
        return None
    if k is None or not _same_kind(k, type):
        return None
    if k is str:
        out = _strings(values)
    elif k is bool:
        out = array(np.asarray(values, dtype=np.bool_))
    elif k is int:
        try:
            out = array(np.asarray(values, dtype=np.int64))
        except OverflowError:
            return None
    else:
        out = array(np.asarray(values, dtype=np.float64))
    if type is not None and out.type != type:
        try:
            out = out.cast(type, safe=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return None          # let PyArrow raise its own error
    return out


def scalar(value, type=None):
    """``pa.scalar(value, type)`` without importing pandas (bool/int/float/str; else PyArrow)."""
    if isinstance(value, pa.Scalar):
        return value.cast(type) if type is not None and value.type != type else value
    import datetime as _dt
    if type is None and isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        days = (value - _dt.date(1970, 1, 1)).days
        return array(np.array([days], dtype=np.int32)).cast(pa.date32())[0]
    if type is None and isinstance(value, _dt.datetime) and value.tzinfo is None:
        us = (value - _dt.datetime(1970, 1, 1)) // _dt.timedelta(microseconds=1)
        return array(np.array([us], dtype=np.int64)).cast(pa.timestamp("us"))[0]
    arr = _from_pylist([value], type) if value is not None else None
    if arr is None:
        return pa.scalar(value, type=type)
    return arr[0]


def array(obj, type=None, **kwargs) -> pa.Array:
    """``pa.array(obj, type)`` for 1-D numeric/bool NumPy input, without pandas."""
    if not kwargs and isinstance(obj, (list, tuple)):
        out = _from_pylist(list(obj), type)
        if out is not None:
            return out
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
        if obj.num_chunks == 0 and obj.type != pa.date32():
            dt = _NP_OF.get(obj.type)
            if dt is not None and not kwargs:
                return np.empty(0, dtype=dt)
        if not kwargs and obj.num_chunks > 1 and _NP_OF.get(obj.type) is not None \
                and obj.type != pa.date32() and obj.null_count == 0:
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
    out = np.frombuffer(buf, dtype=dt, count=len(obj), offset=obj.offset * np.dtype(dt).itemsize)
    out.flags.writeable = False          # a view of immutable Arrow memory, as PyArrow returns
    return out
