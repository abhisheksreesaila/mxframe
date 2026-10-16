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

from ._lazy_pc import pc

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
    if k is float and type is not None and pa.types.is_integer(type) \
            and all(float(v).is_integer() and abs(v) < 2 ** 53 for v in values):
        k = int                  # whole floats into an integer type: PyArrow gives the same ints
        values = [int(v) for v in values]
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
            out = pc.cast(out, type, safe=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return None          # let PyArrow raise its own error
    return out


def scalar(value, type=None):
    """``pa.scalar(value, type)`` without importing pandas (bool/int/float/str; else PyArrow)."""
    if isinstance(value, pa.Scalar):
        return pc.cast(value, type) if type is not None and value.type != type else value
    import datetime as _dt
    if type is None and isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        days = (value - _dt.date(1970, 1, 1)).days
        return pc.cast(array(np.array([days], dtype=np.int32)), pa.date32())[0]
    if type is None and isinstance(value, _dt.datetime) and value.tzinfo is None:
        us = (value - _dt.datetime(1970, 1, 1)) // _dt.timedelta(microseconds=1)
        return pc.cast(array(np.array([us], dtype=np.int64)), pa.timestamp("us"))[0]
    arr = _from_pylist([value], type) if value is not None else None
    if arr is None:
        return pa.scalar(value, type=type)
    return arr[0]


def combine(arr) -> pa.Array:
    """``ChunkedArray.combine_chunks()`` without pandas: PyArrow builds the zero-chunk case with
    ``pa.array([])``, which imports pandas, and an empty intermediate (a filter that keeps no
    rows) is common on small data. One chunk is returned as is (unlike ``combine_chunks``, which
    copies it, the result can keep a non-zero offset); an Array passes through."""
    if not isinstance(arr, pa.ChunkedArray):
        return arr
    if arr.num_chunks == 0:
        return pa.nulls(0, arr.type)
    if arr.num_chunks == 1:
        return arr.chunk(0)
    return arr.combine_chunks()


def value_set(values, like) -> pa.Array:
    """``value_set`` for ``pc.is_in`` against a column of type ``like``. An empty list has no type
    to infer, and ``pa.array([])`` imports pandas: build it empty with the column's value type."""
    if len(values) == 0:
        return pa.nulls(0, like.value_type if pa.types.is_dictionary(like) else like)
    return array(values)


def is_in(col, values):
    """SQL ``col IN (values)`` as a boolean array: NULL where ``col`` is NULL (``pc.is_in`` answers False there,
    which made ``NOT IN`` keep the NULL rows)."""
    values = list(values)
    has_null = any(v is None for v in values)          # x IN (1, NULL) is true or NULL, never false
    values = [v for v in values if v is not None]
    if pa.types.is_floating(col.type):
        # DuckDB: -0.0 = 0.0 and NaN = NaN; Arrow's hash lookup tells -0.0 from 0.0 (F-070). Adding 0.0 turns
        # -0.0 into 0.0, and NaN is looked up with a mask of its own.
        has_nan = any(v != v for v in values)
        zero = scalar(0.0, type=col.type)
        r = pc.is_in(pc.add(col, zero), value_set=value_set([v + 0.0 for v in values if v == v], col.type))
        if has_nan:
            r = pc.or_kleene(r, pc.is_nan(col))
    else:
        r = pc.is_in(col, value_set=value_set(values, col.type))
    if has_null:
        r = pc.if_else(r, scalar(True, type=pa.bool_()), pa.nulls(1, pa.bool_())[0])
    if getattr(col, "null_count", 0):
        r = pc.if_else(pc.is_valid(col), r, pa.nulls(1, pa.bool_())[0])
    return r


def _masked(obj, type):
    """1-D masked numeric/bool array: values plus a validity bitmap, None if not supported."""
    data = np.ma.getdata(obj)
    if data.ndim != 1 or not (data.dtype in _NUMERIC or data.dtype == np.bool_):
        return None
    mask = np.ma.getmaskarray(obj)
    base = array(np.ascontiguousarray(data))
    if not mask.any():
        out = base
    else:
        valid = np.packbits(~mask, bitorder="little")
        out = pa.Array.from_buffers(base.type, len(base), [pa.py_buffer(valid), base.buffers()[1]],
                                    null_count=int(mask.sum()))
    return pc.cast(out, type, safe=True) if type is not None and out.type != type else out


def array(obj, type=None, **kwargs) -> pa.Array:
    """``pa.array(obj, type)`` for 1-D numeric/bool NumPy input, without pandas."""
    if not kwargs and isinstance(obj, (list, tuple)):
        if type is not None and all(v is None for v in obj):
            return pa.nulls(len(obj), type)          # empty, or all NULL (an aggregate over no rows)
        out = _from_pylist(list(obj), type)
        if out is not None:
            return out
    if not kwargs and _is_masked(obj):
        out = _masked(obj, type)
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
        out = pc.cast(out, type, safe=True)
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


# --- gathering rows -------------------------------------------------------------------------

TAKE_CHUNK_ROWS = 1 << 22        # rows per slice for columns whose values cannot be measured
TAKE_CHUNK_BYTES = 1 << 30       # value bytes per slice of a string / binary column
TAKE_SLICE_MIN_ROWS = 1 << 17    # gathers of fewer rows are never sliced


def _has_32bit_offsets(t) -> bool:
    if pa.types.is_struct(t):
        return any(_has_32bit_offsets(f.type) for f in t)
    return pa.types.is_string(t) or pa.types.is_binary(t) or pa.types.is_list(t)


def _byte_slices(col, idx):
    """Slice bounds over ``idx`` such that each slice gathers at most TAKE_CHUNK_BYTES of
    string / binary values (one row alone may exceed it), or None for other types."""
    if not (pa.types.is_string(col.type) or pa.types.is_binary(col.type)):
        return None
    lens = pc.binary_length(col)
    lens = to_numpy(pc.fill_null(lens, scalar(0, type=lens.type))).astype(np.int64, copy=False)   # not a Python 0: that imports pandas
    cum = np.cumsum(lens[idx])
    n, bounds, start, base = len(idx), [], 0, 0
    while start < n:
        end = int(np.searchsorted(cum, base + TAKE_CHUNK_BYTES, side="right"))
        end = min(max(end, start + 1), n)
        bounds.append((start, end))
        base, start = int(cum[end - 1]), end
    return bounds


def take_rows(obj, idx):
    """``obj.take(idx)`` for a table, chunked or plain array and a NumPy row-index array.

    A column with 32-bit offsets (string, binary, list) gathered into more than 2 GiB of
    values comes back as a corrupt array (``Negative offsets in binary array``, visible only
    to ``validate(full=True)``): joining 60M TPC-H lineitem rows to their orders gathers 3 GB
    of ``o_comment``.  String and binary columns are gathered in slices of at most
    ``TAKE_CHUNK_BYTES`` of values (measured from the value lengths); lists and structs
    holding strings in slices of rows sized from the average value size, each checked with
    ``validate`` (an ``ArrowInvalid`` is raised rather than a corrupt column returned).
    The result is a chunked array.  Every other column is gathered in one call."""
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    n = len(idx)

    def one(col):
        if n <= TAKE_SLICE_MIN_ROWS or not _has_32bit_offsets(col.type):
            return pc.take(col, array(idx))
        bounds = _byte_slices(col, idx)
        checked = bounds is None
        if bounds is None:
            per_row = max(1.0, col.nbytes / max(1, len(col)))          # average value bytes per source row
            step = int(max(1, min(TAKE_CHUNK_ROWS, TAKE_CHUNK_BYTES // per_row)))
            bounds = [(i, min(i + step, n)) for i in range(0, n, step)]
        parts = [pc.take(col, array(idx[a:b])) for a, b in bounds]
        chunks = [c for p in parts for c in (p.chunks if isinstance(p, pa.ChunkedArray) else [p])]
        if checked:
            for c in chunks:
                c.validate(full=True)
        return pa.chunked_array(chunks, type=col.type)

    if isinstance(obj, pa.Table):
        if n <= TAKE_SLICE_MIN_ROWS or not any(_has_32bit_offsets(t) for t in obj.schema.types):
            return pc.take(obj, array(idx))
        return pa.Table.from_arrays([one(obj.column(i)) for i in range(obj.num_columns)], schema=obj.schema)
    return one(obj)


# ---- pandas-free replacements for Table.group_by -----------------------------------------------
# PyArrow's Table.group_by / Table.join import pandas (through pyarrow.dataset) on every call, even
# on non-empty data. These cover the engine's fallback uses: DISTINCT and grouped aggregates over
# columns with nulls. Groups come out in first-appearance order, like Table.group_by.

def _codes(col) -> np.ndarray:
    """Dense int64 codes per row, nulls encoded as their own value."""
    a = combine(col)
    if pa.types.is_dictionary(a.type):
        a = a.dictionary_decode()
    a = canonical_floats(a)
    return to_numpy(pc.dictionary_encode(a, null_encoding="encode").indices).astype(np.int64)


def canonical_floats(a):
    """A float array grouped like SQL: -0.0 equals 0.0 and every NaN (any sign or payload) equals
    every other NaN; NULLs stay NULL. Arrow's dictionary encoding compares floats bit for bit and
    gave each of those its own group (F-077). Used for the codes only: a group still reports the
    key value of its first row, which is what DuckDB does too."""
    if a.type not in (pa.float32(), pa.float64()) or len(a) == 0:
        return a
    dt = np.dtype(np.float32 if a.type == pa.float32() else np.float64)
    x = np.frombuffer(a.buffers()[1], dtype=dt, count=a.offset + len(a))
    with np.errstate(invalid="ignore"):                     # a signalling NaN raises "invalid" here
        out = x + dt.type(0)                                # -0.0 + 0.0 is 0.0 (a new array)
    nan = np.isnan(x)
    if nan.any():
        out[nan] = dt.type("nan")
    return pa.Array.from_buffers(a.type, len(a), [a.buffers()[0], pa.py_buffer(out)],
                                 null_count=a.null_count, offset=a.offset)


def _group_ids(table: pa.Table, cols):
    """(group id per row, first row of each group), groups numbered by first appearance."""
    codes = np.stack([_codes(table.column(c)) for c in cols], axis=1)
    _, first, inv = np.unique(codes, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv.reshape(-1)], first[order]


def _float_values(a) -> np.ndarray:
    """float64 values of a numeric Array, null slots unspecified (callers mask them). Reads the
    data buffer: ``to_numpy`` on an array with nulls goes through pandas."""
    if a.type != pa.float64():
        a = pc.cast(a, pa.float64())
    if len(a) == 0:
        return np.zeros(0, np.float64)
    return np.frombuffer(a.buffers()[1], dtype=np.float64, count=a.offset + len(a))[a.offset:]


_INT64_MIN, _INT64_MAX = -(1 << 63), (1 << 63) - 1


def _sum_overflow():
    return OverflowError("Overflow in sum of INT64: the total does not fit in a 64-bit integer "
                         "(DuckDB widens it to a 128-bit HUGEINT)")


def _int_group_sum(a, valid, g, n_groups) -> np.ndarray:
    """Exact int64 per-group sum (np.bincount adds in float64, which loses digits past 2^53): the values
    are cut into three 21-bit limbs, each limb's sums are exact in float64, then recombined in int64.
    A group whose total does not fit int64 raises OverflowError instead of wrapping."""
    if a.type != pa.int64():
        a = pc.cast(a, pa.int64())
    v = np.frombuffer(a.buffers()[1], dtype=np.int64, count=a.offset + len(a))[a.offset:][valid]
    mask = (1 << 21) - 1
    s0 = np.bincount(g, weights=(v & mask), minlength=n_groups)
    s1 = np.bincount(g, weights=((v >> 21) & mask), minlength=n_groups)
    s2 = np.bincount(g, weights=(v >> 42), minlength=n_groups)
    # float64 estimate of the total: only groups near or past the int64 limits are rechecked exactly
    for i in np.flatnonzero(np.abs(s2 * 2.0 ** 42 + s1 * 2.0 ** 21 + s0) > 2.0 ** 62):
        if not _INT64_MIN <= (int(s2[i]) << 42) + (int(s1[i]) << 21) + int(s0[i]) <= _INT64_MAX:
            raise _sum_overflow()
    return (s2.astype(np.int64) << 42) + (s1.astype(np.int64) << 21) + s0.astype(np.int64)


def int_sum(a):
    """``pc.sum`` of an integer Array as a Python int (None when no value is valid); OverflowError
    when the total does not fit int64 (``pc.sum`` wraps). Only a column whose rows * largest
    magnitude could pass 2^63 pays for the exact limb sum."""
    n = len(a) - a.null_count
    if n == 0:
        return None
    mm = pc.min_max(a)
    if max(abs(mm["min"].as_py()), abs(mm["max"].as_py())) * n <= _INT64_MAX:
        return pc.sum(a).as_py()
    valid = to_numpy(pc.is_valid(a)).astype(bool) if a.null_count else np.ones(len(a), bool)
    return int(_int_group_sum(a, valid, np.zeros(int(valid.sum()), np.intp), 1)[0])


_ARITH = {"add": ("addition", "add_checked"), "sub": ("subtraction", "subtract_checked"),
          "mul": ("multiplication", "multiply_checked")}


def checked_arith(op, lhs, rhs):
    """``add`` / ``sub`` / ``mul`` of two operands, raising OverflowError ("Overflow in
    multiplication of INT64", DuckDB's message) where Arrow's plain kernels would wrap. Float
    operands behave as with the plain kernels."""
    word, name = _ARITH[op]
    try:
        return getattr(pc, name)(lhs, rhs)
    except pa.ArrowInvalid as e:
        if "overflow" not in str(e).lower():
            raise
        t = lhs.type if hasattr(lhs, "type") else rhs.type
        raise OverflowError(f"Overflow in {word} of {str(t).upper()}") from e


def distinct_rows(table: pa.Table) -> pa.Table:
    """Distinct rows of ``table`` (nulls equal to each other), first-appearance order."""
    if table.num_rows == 0:
        return table
    _, first = _group_ids(table, table.column_names)
    return pc.take(table, array(first.astype(np.int64)))


def float_extreme(a, op: str):
    """``min`` / ``max`` of an Arrow array as a Python number (None: no valid value), ordering NaN above every
    number like DuckDB (F-070): min skips NaN (NaN if that is all there is), max is NaN if there is one."""
    v = pc.min_max(a)[op].as_py()
    if pa.types.is_floating(a.type):
        n_nan = pc.sum(pc.is_nan(a)).as_py() or 0
        if n_nan and (op == "max" or n_nan == len(a) - a.null_count):
            return float("nan")
    return v


def group_aggregate(table: pa.Table, keys, spec) -> pa.Table:
    """``table.group_by(keys).aggregate(spec)`` for spec ops sum / min / max / mean / count (nulls
    skipped; a group with no valid value gives NULL, count 0). Columns: keys, then ``<col>_<op>``."""
    names = list(keys) + [f"{c}_{op}" for c, op in spec]
    if table.num_rows == 0:
        cols = [pa.nulls(0, table.schema.field(k).type) for k in keys]
        for c, op in spec:
            t = table.schema.field(c).type
            cols.append(pa.nulls(0, pa.int64() if op == "count" else
                                 t if op in ("min", "max") else
                                 pa.int64() if op == "sum" and pa.types.is_integer(t) else pa.float64()))
        return pa.Table.from_arrays(cols, names=names)
    gid, first = _group_ids(table, keys)
    n_groups = len(first)
    take = array(first.astype(np.int64))
    out = [pc.take(combine(table.column(k)), take) for k in keys]
    for c, op in spec:
        a = combine(table.column(c))
        valid = to_numpy(pc.is_valid(a)).astype(bool) if a.null_count else np.ones(len(a), bool)
        g = gid[valid]
        cnt = np.bincount(g, minlength=n_groups)
        if op == "count":
            out.append(array(cnt.astype(np.int64)))
            continue
        if op == "sum" and pa.types.is_integer(a.type):
            out.append(array(np.ma.array(_int_group_sum(a, valid, g, n_groups), mask=cnt == 0), type=pa.int64()))
            continue
        x = _float_values(a)[valid]
        if op in ("sum", "mean"):
            res = np.bincount(g, weights=x, minlength=n_groups)
            if op == "mean":
                with np.errstate(invalid="ignore", divide="ignore"):
                    res = res / cnt
        else:
            # DuckDB orders NaN above every number (F-070): min skips NaN (a group of only NaN gives NaN), max is
            # NaN when the group holds one (Table.group_by skips NaN in both and gives +-inf for only-NaN)
            num = ~np.isnan(x)
            res = np.full(n_groups, np.inf if op == "min" else -np.inf)
            (np.minimum if op == "min" else np.maximum).at(res, g[num], x[num])
            res[np.bincount(g[num], minlength=n_groups) == 0] = np.nan
            if op == "max":
                res[np.bincount(g[~num], minlength=n_groups) > 0] = np.nan
        t = a.type
        typ = (t if op in ("min", "max") else
               pa.int64() if op == "sum" and pa.types.is_integer(t) else pa.float64())
        out.append(array(np.ma.array(res, mask=cnt == 0), type=typ))
    return pa.Table.from_arrays(out, names=names)
