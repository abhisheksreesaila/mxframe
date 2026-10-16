"""Lazy Parquet source: ``scan_parquet`` and the source-resolution pass.

``scan_parquet(path)`` reads only the file footer.  When the query is computed,
``resolve_plan`` walks the logical plan, works out which columns the query
touches, and reads just those (with simple comparisons above the scan pushed
into the reader so whole row groups can be skipped).

Low-cardinality string columns are read as Arrow dictionaries.  The engine uses
those natively as group keys and in string filters; in every other position they
are decoded back to plain values before execution, so results match the
``pq.read_table`` path exactly.

``pyarrow.parquet`` is imported lazily to keep ``import mxframe`` fast.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa

from ._lazy_pc import pc

from .lazy_expr import Expr
from .lazy_frame import (
    FILTERING_JOINS, Aggregate, Distinct, Filter, Join, LazyFrame, Limit, LogicalPlan,
    ParquetScan, Project, Scan, Sort, Tail,
)

def _decode_dict(col):
    from .arrow_dict import decode       # pyarrow.compute (~25 ms) loads on first use, not at import
    return decode(col)


def _is_dict(arr_or_type) -> bool:
    # Same as arrow_dict.is_dictionary; duplicated so importing io_parquet doesn't load pyarrow.compute.
    return pa.types.is_dictionary(getattr(arr_or_type, "type", arr_or_type))


__all__ = ["scan_parquet", "read_parquet_table", "dictionary_columns", "resolve_plan"]

DEFAULT_MAX_DICTIONARY_BYTES = 64 * 1024
# Filtering while decoding pays for its thread hand-offs only on big reads: at 6M rows (SF1
# lineitem) ParquetFile.read's own threads finish in ~25 ms and the engine filters the rest in ~15.
MIN_ROWS_FOR_ROW_FILTER = 16_000_000


# ── public API ──────────────────────────────────────────────────────────────

def scan_parquet(path, *, dictionary: Any = "auto",
                 max_dictionary_bytes: int = DEFAULT_MAX_DICTIONARY_BYTES) -> LazyFrame:
    """Lazily scan a Parquet file (or directory) as a ``LazyFrame``.

    Nothing but the footer is read here.  At ``compute()`` only the columns the
    query references are read, and simple ``col <op> literal`` / ``isin``
    filters directly above the scan are pushed into the reader.

    Args:
        path: Parquet file or dataset directory.
        dictionary: how string columns are read.  ``"auto"`` (default) reads
            columns that are dictionary-encoded in the file with a small
            dictionary (<= ``max_dictionary_bytes`` per row group) as Arrow
            dictionaries; ``True`` does so for every string column; ``False``
            never; a list of names does so for those columns.
        max_dictionary_bytes: dictionary-page size limit for ``"auto"``.
    """
    import pyarrow.parquet as pq
    path = os.fspath(path)
    if os.path.isfile(path):
        # footer parsed once per process: the same metadata serves the read later
        schema = pq.ParquetFile(path, metadata=_metadata(path)).schema_arrow
    else:
        schema = pq.read_schema(path)
    return LazyFrame(ParquetScan(path, schema, dictionary, int(max_dictionary_bytes)))


def read_parquet_table(path, columns: Optional[Sequence[str]] = None, filters=None,
                       dictionary: Any = "auto",
                       max_dictionary_bytes: int = DEFAULT_MAX_DICTIONARY_BYTES) -> pa.Table:
    """Eagerly read a Parquet file into a ``pa.Table`` (dictionary-aware).

    ``filters`` is a ``pyarrow.dataset`` expression or ``None``.  Strings selected
    by ``dictionary`` (see :func:`scan_parquet`) come back as ``pa.dictionary``
    columns, which the engine accepts anywhere a string column is accepted.
    """
    import pyarrow.parquet as pq
    path = os.fspath(path)
    schema = pq.read_schema(path)
    names = list(columns) if columns is not None else list(schema.names)
    dict_cols = dictionary_columns(path, schema, dictionary, max_dictionary_bytes, names)
    if filters is None and os.path.isfile(path):
        token = source_token(path, names, ())
        return tag_source(_read_file(path, names, dict_cols), token)
    return pq.read_table(path, columns=names, filters=filters,
                         read_dictionary=dict_cols or None)


def _read_file(path, columns, dict_cols, pushed=(), mask_dict=(), only=None) -> pa.Table:
    """Read one Parquet file without ``pyarrow.dataset``.

    ``pq.read_table`` goes through ``pyarrow.dataset``, whose import pulls in
    pandas (~150 ms of a first query). ``ParquetFile`` reads the same data
    without it. ``pushed`` simple filters skip row groups whose statistics rule
    them out and, for integer comparisons, drop non-matching rows right after
    each row group is decoded (see ``_read_groups``); rows are still filtered
    exactly by the plan's Filter node.  String conjuncts are evaluated on dictionary codes: on the
    ``dict_cols`` the caller wants back as dictionaries and on ``mask_dict`` columns, which are read
    as dictionaries only to build the row mask and returned decoded.

    ``only`` (F-056 step 2, streamed execution) restricts the read to those row groups: a batch.
    """
    import pyarrow.parquet as pq
    md = _metadata(path)
    pf = pq.ParquetFile(path, read_dictionary=dict_cols or None, metadata=md)
    groups = _row_groups_for(pf, pushed) if pushed else None
    if only is not None:
        groups = [g for g in (groups if groups is not None else range(md.num_row_groups)) if g in only]
    if groups is not None and not groups:
        return pf.schema_arrow.empty_table().select(columns) if columns else pf.schema_arrow.empty_table()
    if groups is None:
        groups = list(range(md.num_row_groups))
    rowwise = [p for p in pushed if _row_filterable(p, pf.schema_arrow, tuple(dict_cols or ()) + tuple(mask_dict))]
    if (len(groups) >= 2 and rowwise and md.num_columns == len(pf.schema_arrow.names)
            and sum(md.row_group(g).num_rows for g in groups) >= MIN_ROWS_FOR_ROW_FILTER):
        need = _mask_needs(md, pf.schema_arrow.names, groups, rowwise)
        if any(need.values()) and _estimated_pass(md, pf.schema_arrow.names, need) < 0.6:
            return _read_groups(path, md, pf.schema_arrow, columns, dict_cols, groups, need, mask_dict)
    if len(groups) == md.num_row_groups:
        return pf.read(columns=columns, use_threads=True)
    return pf.read_row_groups(groups, columns=columns, use_threads=True)


_ROW_OPS = {"eq": "__eq__", "ne": "__ne__", "lt": "__lt__", "le": "__le__", "gt": "__gt__", "ge": "__ge__"}


def _row_filterable(pushed_one, schema, dict_cols=()) -> bool:
    """Integer comparisons/isin, and string ones on columns read as dictionaries (F-020)."""
    name, op, _ = pushed_one
    if op not in _ROW_OPS and op != "isin":
        return False
    kind = _kind(schema, name)
    return kind == "int" or (kind == "str" and name in (dict_cols or ()))


def _mask_needs(md, names, groups, rowwise) -> dict:
    """Per row group, the row-level conjuncts its statistics do not already guarantee.

    A group whose min/max prove every row passes a conjunct (a 98%-selective ``shipdate <= x``
    over sorted-ish data, say) needs no mask for it, so it is read exactly like an unfiltered file.
    """
    out = {}
    for g in groups:
        rg = md.row_group(g)
        todo = []
        for c in rowwise:
            name, op, value = c
            j = names.index(name) if name in names else -1
            st = rg.column(j).statistics if j >= 0 else None
            if st is not None and st.has_min_max and _covers(st.min, st.max, op, value):
                continue
            todo.append(c)
        out[g] = tuple(todo)
    return out


def _estimated_pass(md, names, need) -> float:
    """Average fraction of rows expected to survive the pushed conjuncts, from min/max alone.

    Values are assumed uniform between a group's min and max (conjuncts independent).  Only used
    to decide whether filtering while decoding is worth it: a 98%-selective ``shipdate <= x`` is
    cheaper to read as one plain table and let the engine's Filter node drop the few rows.
    """
    total = 0.0
    for g, todo in need.items():
        rg = md.row_group(g)
        f = 1.0
        for name, op, value in todo:
            j = names.index(name)
            st = rg.column(j).statistics
            if st is None or not st.has_min_max:
                continue
            lo, hi = st.min, st.max
            if isinstance(value, str) or (op == "isin" and isinstance(next(iter(value), 0), str)):
                # no value spread for strings: an equality keeps a modest share of a handful of values
                if op == "eq":
                    f *= 0.3
                elif op == "isin":
                    f *= min(1.0, 0.3 * len(value))
                elif op == "ne":
                    f *= 0.7                         # 1 - 1/distinct for a handful of values
                elif op in ("lt", "le", "gt", "ge"):
                    try:
                        pos = _str_fraction(value, lo, hi)
                    except (TypeError, ValueError):
                        continue
                    f *= pos if op in ("lt", "le") else 1.0 - pos
                continue
            span = max(hi - lo + 1, 1)
            try:
                if op == "eq":
                    f *= (1.0 / span) if lo <= value <= hi else 0.0
                elif op == "isin":
                    f *= sum(1 for v in value if lo <= v <= hi) / span
                elif op == "ne":
                    pass
                elif op in ("lt", "le"):
                    f *= min(max((value - lo + (op == "le")) / span, 0.0), 1.0)
                else:
                    f *= min(max((hi - value + (op == "ge")) / span, 0.0), 1.0)
            except TypeError:
                pass
        total += f
    return total / max(len(need), 1)


def _str_fraction(value, lo, hi) -> float:
    """Where ``value`` falls between strings ``lo`` and ``hi`` (0..1), by their first 6 bytes."""
    def num(x):
        b = x if isinstance(x, (bytes, bytearray)) else str(x).encode("utf-8")
        return int.from_bytes(bytes(b[:6]).ljust(6, b"\0"), "big")
    try:
        a, z, v = num(lo), num(hi), num(value)
    except Exception:             # an estimate only: never let odd statistics fail a query
        return 0.5
    return min(max((v - a) / (z - a), 0.0), 1.0) if z > a else 0.5


def _covers(lo, hi, op, value) -> bool:
    """True when every value in [lo, hi] satisfies ``op value``."""
    try:
        if op == "eq":
            return lo == hi == value
        if op == "ne":
            return value < lo or value > hi
        if op == "lt":
            return hi < value
        if op == "le":
            return hi <= value
        if op == "gt":
            return lo > value
        if op == "ge":
            return lo >= value
        if op == "isin":
            return lo == hi and lo in value
    except TypeError:
        pass
    return False


def _row_mask(table, rowwise):
    """Boolean NumPy mask of the rows satisfying every integer conjunct, or None (keep all)."""
    from .np_arrow import to_numpy
    import numpy as np
    mask = None
    for name, op, value in rowwise:
        c = table.column(name)
        if c.null_count:
            continue                       # nulls: leave the row-level decision to the Filter node
        if _is_dict(c.type):
            m = _dictionary_mask(c, op, value)
            if m is not None:
                mask = m if mask is None else (mask & m)
            continue
        a = to_numpy(c)
        try:
            if op == "isin":
                m = np.isin(a, np.asarray(value, dtype=np.int64))
            else:
                m = getattr(a, _ROW_OPS[op])(value)
        except (OverflowError, TypeError):
            continue                       # literal outside the column's range: the Filter node decides
        mask = m if mask is None else (mask & m)
    return mask


def _dictionary_mask(col, op, value):
    """Boolean NumPy mask of ``col <op> value`` on a dictionary column: compare each dictionary entry
    once, then look the codes up (F-020).  None when the column is not a plain string dictionary."""
    from .np_arrow import to_numpy, combine as _combine
    import numpy as np
    arr = _combine(col) if isinstance(col, pa.ChunkedArray) else col
    if not isinstance(arr, pa.DictionaryArray) or arr.null_count or arr.dictionary.null_count:
        return None
    entries = arr.dictionary.to_pylist()
    if op == "isin":
        vals = set(value)
        keep = [e in vals for e in entries]
    else:
        keep = [getattr(e, _ROW_OPS[op])(value) for e in entries]
    if NotImplemented in keep:
        return None
    return np.asarray(keep, dtype=np.bool_)[to_numpy(arr.indices)]


def _read_groups(path, md, schema, columns, dict_cols, groups, need, mask_dict=()) -> pa.Table:
    """Decode (row group, column) pieces on a thread pool, dropping rows as they are decoded.

    Stage 1 decodes the columns the pushed integer comparisons need, per row group, and turns
    them into NumPy masks.  Stage 2 decodes every other column piece and keeps only the rows
    the mask selects, while the piece is still cache-hot, so filtered-out rows never reach the
    engine.  Pieces are single-threaded reads of a shared memory map; the (group, column) grain
    keeps all threads busy even on files with few row groups.
    """
    import pyarrow.parquet as pq
    cols = list(columns) if columns else list(schema.names)
    rd = dict_cols or None
    kept = set(dict_cols or ())
    masked_as_dict = set(mask_dict) - kept
    src = pa.memory_map(path)              # one mapping shared by all readers: no per-group open/copy

    def piece(g, name, as_dict=False):
        rdict = (list(rd or []) + [name]) if as_dict and name in masked_as_dict else rd
        t = pq.ParquetFile(src, read_dictionary=rdict, metadata=md).read_row_group(g, columns=[name], use_threads=False)
        arrays = t.column(0).chunks
        return arrays[0] if len(arrays) == 1 else pa.concat_arrays(arrays)

    workers = max(1, min(len(groups) * len(cols), _read_threads()))
    masked = [g for g in groups if need[g]]
    have = {}

    def stage1(g):
        names = list(dict.fromkeys(n for n, _, _ in need[g]))
        tbl = pa.table([pa.chunked_array([piece(g, n, True)]) for n in names], names=names)
        for n in names:
            have[(g, n)] = tbl.column(n).chunk(0)
        m = _row_mask(tbl, need[g])
        # dropping under ~1/4 of a group is not worth a gather; the Filter node drops them
        return m if m is not None and _count(m) * 4 < len(m) * 3 else None

    masks = dict(zip(masked, _parallel_map(stage1, masked, workers)))

    def stage2(task):
        g, name = task
        arr = have.get((g, name))
        if arr is None:
            arr = piece(g, name)
        m = masks.get(g)
        out = arr if m is None else _take_array(arr, m)
        if name in masked_as_dict and _is_dict(out.type):
            out = pc.cast(out, out.type.value_type)          # read as a dictionary only for the mask
        return out

    tasks = [(g, n) for n in cols for g in groups]
    pieces = _parallel_map(stage2, tasks, workers)
    out = []
    for k, name in enumerate(cols):
        arrs = pieces[k * len(groups):(k + 1) * len(groups)]
        arrs = [a for a in arrs if len(a)] or arrs[:1]
        typ = arrs[0].type
        if len(arrs) == 1 or pa.types.is_dictionary(typ):
            out.append(pa.chunked_array(arrs, typ))
        else:
            out.append(("concat", arrs, typ))
    # one chunk per plain column (what ParquetFile.read gives), copied column-parallel;
    # dictionary columns keep one chunk per group, each with its own dictionary
    todo = [i for i, o in enumerate(out) if isinstance(o, tuple)]
    done = _parallel_map(lambda i: pa.chunked_array([pa.concat_arrays(out[i][1])], out[i][2]), todo, min(workers, len(todo)) or 1)
    for i, d in zip(todo, done):
        out[i] = d
    return pa.table(out, names=cols)


def _parallel_map(fn, items, workers) -> list:
    """``[fn(x) for x in items]`` on ``workers`` threads (results in order; first error re-raised).

    Plain ``threading``: ``concurrent.futures`` costs ~10-40 ms to import in a fresh process.
    """
    import threading
    items = list(items)
    out = [None] * len(items)
    errors = []
    nxt = iter(range(len(items)))
    lock = threading.Lock()

    def work():
        while not errors:
            with lock:
                i = next(nxt, None)
            if i is None:
                return
            try:
                out[i] = fn(items[i])
            except BaseException as e:      # noqa: BLE001 - re-raised in the caller
                errors.append(e)
                return

    threads = [threading.Thread(target=work, daemon=True) for _ in range(max(1, workers) - 1)]
    for t in threads:
        t.start()
    work()                                  # the calling thread works too
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return out


def _count(mask) -> int:
    import numpy as np
    return int(np.count_nonzero(mask))


def _read_threads() -> int:
    """Decode threads.  More than ~8 slow Arrow's decoder down (allocator and page-fault contention)."""
    try:
        n = int(os.environ.get("MXFRAME_READ_THREADS", 0))
    except ValueError:
        n = 0
    return n if n > 0 else min(8, max(2, (os.cpu_count() or 2) // 2))


def _take_array(arr: pa.Array, mask) -> pa.Array:
    """``arr`` rows where the NumPy boolean ``mask`` is set.

    Plain numeric arrays go through NumPy (no ``pyarrow.compute`` import, ~25 ms of a first
    query); anything else (strings, dictionaries, nulls) uses Arrow.
    """
    from .np_arrow import array as _arr, to_numpy
    t = arr.type
    if (pa.types.is_integer(t) or pa.types.is_floating(t)) and not arr.null_count:
        return _arr(to_numpy(arr)[mask], type=t)
    return pc.filter(arr, _arr(mask))


def _row_groups_for(pf, pushed) -> List[int]:
    """Row groups whose min/max statistics may satisfy every pushed conjunct."""
    md = pf.metadata
    names = pf.schema_arrow.names
    if md.num_columns != len(names):
        return list(range(md.num_row_groups))
    keep = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        ok = True
        for name, op, value in pushed:
            j = names.index(name) if name in names else -1
            st = rg.column(j).statistics if j >= 0 else None
            if st is None or not st.has_min_max:
                continue
            lo, hi = st.min, st.max
            if isinstance(lo, bytes):
                lo, hi = lo.decode("utf-8", "replace"), hi.decode("utf-8", "replace")
            try:
                if op == "eq":
                    ok = lo <= value <= hi
                elif op == "ne":
                    ok = not (lo == hi == value)
                elif op == "lt":
                    ok = lo < value
                elif op == "le":
                    ok = lo <= value
                elif op == "gt":
                    ok = hi > value
                elif op == "ge":
                    ok = hi >= value
                elif op == "isin":
                    ok = any(lo <= v <= hi for v in value)
            except TypeError:
                ok = True
            if not ok:
                break
        if ok:
            keep.append(g)
    return keep


def dictionary_columns(path, schema: pa.Schema, dictionary: Any = "auto",
                       max_dictionary_bytes: int = DEFAULT_MAX_DICTIONARY_BYTES,
                       among: Optional[Iterable[str]] = None) -> List[str]:
    """Names of string columns to read as Arrow dictionaries (see ``scan_parquet``)."""
    allowed = set(among) if among is not None else set(schema.names)
    strings = [f.name for f in schema
               if f.name in allowed and (pa.types.is_string(f.type) or pa.types.is_large_string(f.type))]
    if dictionary is False or dictionary is None:
        return []
    if dictionary is True:
        return strings
    if dictionary != "auto":
        wanted = [dictionary] if isinstance(dictionary, str) else list(dictionary)
        return [n for n in strings if n in set(wanted)]
    return [n for n in strings if _chunks_are_small_dictionaries(path, schema, n, max_dictionary_bytes)]


def _chunks_are_small_dictionaries(path, schema, name, max_bytes) -> bool:
    """True if every row group stores ``name`` dictionary-encoded with a small dictionary."""
    import pyarrow.parquet as pq
    if not os.path.isfile(path):
        return False
    md = _metadata(path)
    if md.num_columns != len(schema.names):      # nested columns: leaf index != field index
        return False
    j = schema.get_field_index(name)
    if md.num_row_groups == 0:
        return False
    for g in range(md.num_row_groups):
        c = md.row_group(g).column(j)
        if not c.has_dictionary_page:
            return False
        if not any("DICTIONARY" in e for e in c.encodings):
            return False
        d_off = c.dictionary_page_offset
        if d_off is None or d_off <= 0 or c.data_page_offset - d_off > max_bytes:
            return False
    return True


@functools.lru_cache(maxsize=32)
def _metadata_cached(path, dev, ino, mtime_ns, size):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata


# ── source identity (F-057) ────────────────────────────────────────────────
# Every compute of a `scan_parquet` frame reads the file again into new buffers, so the GPU
# residency cache cannot key them by address. A table read from a file carries a token naming
# what it holds: the file's identity (path, device, inode, mtime, size), the columns read and the
# filters pushed into the reader. Two reads with the same token hold identical rows.
_SOURCES: dict = {}                      # id(table) -> (weakref to the table, token)


def source_token(path, columns, pushed, groups=None) -> Optional[tuple]:
    """Token of a read of ``columns`` of file ``path`` with ``pushed`` filters, or None for a directory.
    ``groups`` (a streamed batch) names the row groups read: another batch is another source."""
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return ("parquet", path, st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size, tuple(columns),
            repr(tuple(pushed or ())) + (f"|groups={tuple(groups)}" if groups is not None else ""))


def tag_source(table: pa.Table, token: Optional[tuple]) -> pa.Table:
    """Record ``token`` for ``table``, with its final column types (a string read as a dictionary or decoded
    gives different codes, so those reads are different sources)."""
    if token is not None:
        import weakref
        key = id(table)
        token = token + (tuple(str(t) for t in table.schema.types),)
        _SOURCES[key] = (weakref.ref(table, lambda _r, k=key: _SOURCES.pop(k, None)), token)
    return table


def table_source(table: pa.Table) -> Optional[tuple]:
    """The token of a table read by `resolve_plan`, or None (an in-memory table, or one derived from a read)."""
    got = _SOURCES.get(id(table))
    return got[1] if got is not None and got[0]() is table else None


def _metadata(path):
    st = os.stat(path)
    # the cached footer now drives data reads: key on file identity as well as mtime/size
    return _metadata_cached(os.path.abspath(os.fspath(path)), st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)


# ── streamed execution (F-056 step 2) ──────────────────────────────────────
# A query over a file bigger than memory runs once per batch of row groups (`streamed.py`). While it runs,
# the streamed file reads only the current batch's groups and every other file is read once for all batches.
class StreamContext:
    def __init__(self, path):
        self.path = os.path.abspath(os.fspath(path))
        self.groups: Optional[tuple] = None          # the current batch's row groups
        self._once: dict = {}

    def is_streamed(self, path) -> bool:
        return os.path.abspath(os.fspath(path)) == self.path

    def once(self, key, read):
        if key not in self._once:
            self._once[key] = read()
        return self._once[key]


_stream: contextvars.ContextVar = contextvars.ContextVar("mxframe_stream", default=None)


@contextlib.contextmanager
def streaming(path):
    st = StreamContext(path)
    token = _stream.set(st)
    try:
        yield st
    finally:
        _stream.reset(token)


def _fixed_width(t) -> Optional[int]:
    try:
        return max(t.bit_width // 8, 1)
    except ValueError:                                   # strings, binary, nested
        return None


def kept_groups(path, schema, pushed) -> List[int]:
    """The row groups of ``path`` the pushed filters leave (all of them without filters)."""
    import pyarrow.parquet as pq
    md = _metadata(path)
    if not pushed:
        return list(range(md.num_row_groups))
    return _row_groups_for(pq.ParquetFile(path, metadata=md), pushed)


def group_bytes(path, schema, columns, groups) -> List[int]:
    """Decoded size in memory of ``columns`` for each of ``groups``: rows x width for fixed-width columns,
    the uncompressed bytes plus a 4-byte offset per row for strings (an estimate from the footer)."""
    md = _metadata(path)
    names = list(schema.names)
    flat = md.num_columns == len(names)
    cols = [(n, names.index(n), _fixed_width(schema.field(n).type)) for n in columns if n in names]
    out = []
    for g in groups:
        rg = md.row_group(g)
        if not flat:                                     # nested columns: the group's whole uncompressed size
            out.append(rg.total_byte_size)
            continue
        n = rg.num_rows
        out.append(sum(n * w if w is not None else rg.column(j).total_uncompressed_size + 4 * n
                       for _, j, w in cols))
    return out


def decoded_upper_bound(path, schema) -> int:
    """An upper bound on what reading any columns of ``path`` decodes to, without visiting a column chunk
    (one number per file version, remembered: this runs on every compute of a lazy Parquet source)."""
    st = os.stat(path)
    return _upper_bound(os.path.abspath(os.fspath(path)), st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size, schema)


@functools.lru_cache(maxsize=64)
def _upper_bound(path, dev, ino, mtime_ns, size, schema) -> int:
    md = _metadata(path)
    fixed = sum(w for w in (_fixed_width(f.type) for f in schema) if w is not None)
    return md.num_rows * fixed + sum(md.row_group(g).total_byte_size for g in range(md.num_row_groups))


# ── plan analysis ───────────────────────────────────────────────────────────

_CMP_OPS = ("eq", "ne", "gt", "ge", "lt", "le")
_FLIP = {"eq": "eq", "ne": "ne", "gt": "lt", "lt": "gt", "ge": "le", "le": "ge"}
# Predicate forms the engine evaluates natively on dictionary columns.
_NATIVE_DICT_PRED = ("eq", "ne", "gt", "ge", "lt", "le", "isin", "startswith", "contains", "like")


class _Info:
    """What a plan references, gathered in one walk."""

    def __init__(self):
        self.refs: set = set()          # every column name referenced anywhere
        self.bad: set = set()           # referenced in a way that needs plain values
        self.bad_expr: set = set()      # ... inside an expression (not a bare column projected out)
        self.has_join = False
        self.passthrough: dict = {}     # id(leaf) -> scan columns flow to the plan output

    def use_plain(self, expr: Expr, bare_ok: bool = False):
        """``bare_ok``: a bare column projected out of the plan is plain by then when the join
        engine decodes it after its gather (deferral), so it does not count in ``bad_expr``."""
        for name in _expr_cols(expr):
            self.refs.add(name)
            self.bad.add(name)
            if not (bare_ok and expr.op == "col"):
                self.bad_expr.add(name)

    def predicate(self, expr: Expr):
        """A Filter predicate: `col <cmp> literal`, isin, startswith, contains may see dictionaries."""
        op, args = expr.op, expr.args
        if op in ("and", "or", "__and__", "__or__"):
            for a in args:
                self.predicate(a)
            return
        if op == "not":
            self.predicate(args[0])
            return
        if (op in _NATIVE_DICT_PRED and isinstance(args[0], Expr) and args[0].op == "col"
                and (op not in _CMP_OPS or (isinstance(args[1], Expr) and args[1].op == "lit"))):
            self.refs.add(args[0].args[0])
            return
        self.use_plain(expr)


def _expr_cols(expr) -> set:
    out: set = set()

    def visit(e):
        if not isinstance(e, Expr):
            return
        if e.op == "col":
            out.add(e.args[0])
            return
        if e.op == "window":
            spec = e.args[1]
            out.update(spec[0])
            out.update(spec[1])
        for a in e.args:
            visit(a)

    visit(expr)
    return out


def count_distinct_shape(plan) -> bool:
    """``Aggregate(Distinct(input), group_by=[a], aggs=[count(b) or count(literal)])``: one bare key
    and one count over a whole-row distinct (``select(a, b).distinct().groupby(a)``)."""
    if not isinstance(plan, Aggregate) or not isinstance(plan.input, Distinct) or plan.input.subset:
        return False
    if len(plan.group_by) != 1 or len(plan.aggs) != 1:
        return False
    k, agg = plan.group_by[0], plan.aggs[0]
    if not isinstance(k, Expr) or k.op != "col" or k._alias not in (None, k.args[0]):
        return False
    if not isinstance(agg, Expr) or agg.op != "count" or not agg._alias:
        return False
    arg = agg.args[0]
    return isinstance(arg, Expr) and arg.op in ("col", "lit") and not (arg.op == "col" and arg.args[0] == k.args[0])


def _bare_pair(project, agg) -> bool:
    """``project`` selects exactly the group key and the counted column, bare and not renamed (or
    is an in-memory scan of just those two columns: the optimizer drops a select of every column)."""
    if isinstance(project, Scan):
        names = project.table.column_names
        key, arg = agg.group_by[0].args[0], agg.aggs[0].args[0]
        return len(names) == 2 and key in names and (arg.op == "lit" or arg.args[0] in names)
    if not isinstance(project, Project) or len(project.exprs) != 2:
        return False
    names = []
    for e in project.exprs:
        if not isinstance(e, Expr) or e.op != "col" or e._alias not in (None, e.args[0]):
            return False
        names.append(e.args[0])
    key, arg = agg.group_by[0].args[0], agg.aggs[0].args[0]
    return key in names and len(set(names)) == 2 and (arg.op == "lit" or arg.args[0] in names)


def _analyze(plan: LogicalPlan, count_distinct_codes: bool = False) -> _Info:
    """Walk the plan once.

    ``walk`` returns whether the subtree contains a Project/Aggregate.  Expressions
    of a node above such a cut refer to derived (already plain) columns, not to
    the scan's, so they only count as references and never force decoding.
    """
    info = _Info()

    def walk(node, passthru: bool, native_keys=frozenset()) -> bool:
        """``native_keys``: names the parent Aggregate reads only as bare group keys (dictionary-native):
        a Project right below passes them through without forcing a decode (F-034, Q9's with_columns)."""
        if isinstance(node, (Scan, ParquetScan)):
            info.passthrough[id(node)] = info.passthrough.get(id(node), False) or passthru
            return False
        if isinstance(node, Filter):
            cut = walk(node.input, passthru)
            if cut:
                info.refs.update(_expr_cols(node.predicate))
            else:
                info.predicate(node.predicate)
            return cut
        if isinstance(node, Project):
            walk(node.input, False)
            for e in node.exprs:
                if (native_keys and e.op == "col" and e._alias in (None, e.args[0]) and e.args[0] in native_keys
                        and sum(e.args[0] in _expr_cols(o) for o in node.exprs) == 1):
                    info.refs.add(e.args[0])            # a bare pass-through nothing else in the Project reads
                else:
                    info.use_plain(e, bare_ok=True)
            return True
        if isinstance(node, Aggregate):
            src = node.input.input if isinstance(node.input, Distinct) else None
            if count_distinct_codes and count_distinct_shape(node) and _bare_pair(src, node):
                # COUNT(DISTINCT b) per a: distinct (a, b) pairs are the same on codes of one
                # dictionary as on values, so both columns may stay encoded (F-046h)
                if isinstance(src, Project):
                    walk(src.input, False)
                    info.refs.update(_expr_cols(src.exprs[0]) | _expr_cols(src.exprs[1]))
                else:                                # the scan is the pair (select of every column)
                    walk(src, False)
                    info.refs.update(src.table.column_names)
                return True
            used = set()
            for e in list(node.aggs) + [g for g in node.group_by if g.op != "col"]:
                used |= _expr_cols(e)
            walk(node.input, False, frozenset(g.args[0] for g in node.group_by
                                              if g.op == "col" and g._alias in (None, g.args[0])
                                              and g.args[0] not in used))
            for e in node.group_by:
                if e.op == "col":
                    info.refs.add(e.args[0])          # bare group key: dictionary-native
                else:
                    info.use_plain(e)
            for e in node.aggs:
                info.use_plain(e)
            return True
        if isinstance(node, Sort):
            cut = walk(node.input, passthru)
            for e in node.by:
                if cut:
                    info.refs.update(_expr_cols(e))
                else:
                    info.use_plain(e)
            return cut
        if isinstance(node, (Limit, Tail)):
            return walk(node.input, passthru)
        if isinstance(node, Distinct):
            cut = walk(node.input, passthru or not node.subset)
            if node.subset:
                info.refs.update(node.subset)
                if not cut:
                    info.bad.update(node.subset)
                    info.bad_expr.update(node.subset)
            return cut
        if isinstance(node, Join):
            info.has_join = True
            info.refs.update(node.left_on)
            info.refs.update(node.right_on)
            info.bad.update(node.left_on)
            info.bad.update(node.right_on)
            info.bad_expr.update(node.left_on)       # join keys and distinct subsets stay decoded
            info.bad_expr.update(node.right_on)      # up front, as before (only bare projections defer)
            lcut = walk(node.left, passthru)
            # a semi / anti join's right columns never reach the output: only its keys are read
            rcut = walk(node.right, passthru and node.how not in FILTERING_JOINS)
            return lcut or rcut
        info.has_join = True                       # unknown node: be conservative
        if hasattr(node, "input"):
            return walk(node.input, True)
        return False

    walk(plan, True)
    return info


def _renamed_ref(name: str, refs) -> bool:
    """True when a join could have renamed column ``name`` and the query refers to the new name
    (``x`` -> ``x_right``, ``x_right2``, ...; see ``lazy_frame.join_output_names``)."""
    stem = name + "_right"
    return any(r.startswith(stem) and (len(r) == len(stem) or r[len(stem):].isdigit()) for r in refs)


def _join_leaf_filters(plan: LogicalPlan) -> dict:
    """id(leaf) -> columns read by its Filter chain, for every leaf that is a Filter*(Scan) input
    of a Join (the join engine's leaf_table handles exactly these). Other leaves are absent."""
    out: dict = {}

    def walk(node, above, under_join):
        if isinstance(node, (Scan, ParquetScan)):
            if under_join:
                out[id(node)] = out.get(id(node), set()) | above
        elif isinstance(node, Filter):
            walk(node.input, above | set(_expr_cols(node.predicate)), under_join)
        elif isinstance(node, Join):
            walk(node.left, set(), True)
            walk(node.right, set(), True)
        elif under_join and not above and _selects_columns(node) and not isinstance(node.input, Project):
            walk(node.input, set(), True)        # .filter(...).select(key) under a join: still a leaf
                                                 # (one Project, as join_engine._scan_under_filters)
        elif hasattr(node, "input"):
            walk(node.input, set(), False)
    walk(plan, set(), False)
    return out


def _selects_columns(node) -> bool:
    """A Project that only selects columns (join_engine sees through it to a Filter*(Scan) leaf)."""
    return (isinstance(node, Project) and bool(node.exprs)
            and all(e.op == "col" and e._alias in (None, e.args[0]) for e in node.exprs))


def _leaves(plan: LogicalPlan):
    if isinstance(plan, (Scan, ParquetScan)):
        yield plan
    elif isinstance(plan, Join):
        yield from _leaves(plan.left)
        yield from _leaves(plan.right)
    elif hasattr(plan, "input"):
        yield from _leaves(plan.input)


# ── predicate pushdown ──────────────────────────────────────────────────────

def _conjuncts(expr: Expr):
    if expr.op in ("and", "__and__"):
        for a in expr.args:
            yield from _conjuncts(a)
    else:
        yield expr


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _pushable(expr: Expr, schema: pa.Schema) -> Optional[Tuple]:
    """(column, op, value) for `col <cmp> literal` / `col.isin([...])`, else None.

    Only integer and string columns with a literal of the same kind are pushed:
    the reader then compares exactly as the engine does.  Float columns are left
    to the engine (float32 data vs a Python float literal is a precision hazard).
    """
    op, args = expr.op, expr.args
    if op in _CMP_OPS:
        a, b = args
        if a.op == "lit" and b.op == "col":
            a, b, op = b, a, _FLIP[op]
        if not (a.op == "col" and b.op == "lit"):
            return None
        name, value = a.args[0], b.args[0]
        kind = _kind(schema, name)
        if (kind == "int" and _is_int(value)) or (kind == "str" and isinstance(value, str)):
            return (name, op, value)
        return None
    if op == "isin":
        target, values = args
        if target.op != "col" or not values:
            return None
        name = target.args[0]
        kind = _kind(schema, name)
        if (kind == "int" and all(_is_int(v) for v in values)) or \
           (kind == "str" and all(isinstance(v, str) for v in values)):
            return (name, "isin", tuple(values))
    return None


def _kind(schema: pa.Schema, name: str) -> Optional[str]:
    i = schema.get_field_index(name)
    if i < 0:
        return None
    t = schema.field(i).type
    if pa.types.is_dictionary(t):            # a file opened with read_dictionary reports dictionary types
        t = t.value_type
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "str"
    return None


def _filters_expression(filters):
    import pyarrow.dataset as ds
    out = None
    for name, op, value in filters:
        f = ds.field(name)
        e = {"eq": lambda: f == value, "ne": lambda: f != value, "gt": lambda: f > value,
             "ge": lambda: f >= value, "lt": lambda: f < value, "le": lambda: f <= value,
             "isin": lambda: f.isin(list(value))}[op]()
        out = e if out is None else (out & e)
    return out


# ── resolution ──────────────────────────────────────────────────────────────

def _single_table_shape(plan) -> bool:
    """``[Sort / Limit / Distinct(all columns)]* [Project(plain columns)] Filter+ (Scan | ParquetScan)``:
    one table filtered and selected, with nothing that reads its columns as values on the way. A DISTINCT
    needs no filter (F-033: it runs on the codes as a group-by)."""
    node = plan
    distinct = False
    while isinstance(node, (Sort, Limit, Tail)) or (isinstance(node, Distinct) and not node.subset):
        distinct = distinct or isinstance(node, Distinct)
        node = node.input
    if isinstance(node, Project):
        if not all(isinstance(e, Expr) and e.op == "col" for e in node.exprs):
            return False
        node = node.input
    if not isinstance(node, Filter):
        return distinct and isinstance(node, (Scan, ParquetScan))
    while isinstance(node, Filter):
        node = node.input
    return isinstance(node, (Scan, ParquetScan))


def _fold_renames(node: LogicalPlan) -> LogicalPlan:
    """A select that renames columns of an in-memory table (``nation.select(col("n_name").alias("supp_nation"))``,
    TPC-H Q7) becomes a Scan of the same buffers under the new names (F-034): the dictionary analysis and the
    join engine then see a plain leaf, so its dictionary columns stay codes through the join like any other."""
    if isinstance(node, Project) and isinstance(node.input, Scan) and node.exprs \
            and all(e.op == "col" for e in node.exprs) \
            and any(e._alias not in (None, e.args[0]) for e in node.exprs):
        inner = [e.args[0] for e in node.exprs]
        outer = [e._alias or e.args[0] for e in node.exprs]
        if len(set(outer)) == len(outer) and all(n in node.input.table.column_names for n in inner):
            return Scan(node.input.table.select(inner).rename_columns(outer))
    if isinstance(node, Join):
        return dataclasses.replace(node, left=_fold_renames(node.left), right=_fold_renames(node.right))
    if hasattr(node, "input") and node.input is not None:
        return dataclasses.replace(node, input=_fold_renames(node.input))
    return node


def resolve_plan(plan: LogicalPlan, read: bool = True, count_distinct_codes: bool = False) -> LogicalPlan:
    """Replace every ``ParquetScan`` by a ``Scan`` of just the needed columns.

    Also makes dictionary columns safe for the engine: a dictionary column stays a
    dictionary only when the query uses it exclusively as a bare group key or in
    ``col <cmp> literal`` / isin / startswith / contains filters (and the scan's
    columns are not passed through to the output, and there is no join); every
    other dictionary column is decoded to its plain type.

    ``count_distinct_codes`` (device="cpu", where the native count reads codes): both columns of a
    COUNT(DISTINCT b) per a stay dictionaries (F-046h); other devices get them decoded as before.

    With ``read=False`` nothing is read: each ``ParquetScan`` just records the
    ``columns`` and ``filters`` it would read (for ``explain``).
    """
    leaves = list(_leaves(plan))
    has_parquet = any(isinstance(l, ParquetScan) for l in leaves)
    has_dict = any(isinstance(l, Scan) and any(_is_dict(t) for t in l.table.schema.types)
                   for l in leaves)
    if not has_parquet and not has_dict:
        return plan
    if has_dict:
        plan = _fold_renames(plan)
    info = _analyze(plan, count_distinct_codes)
    single = _single_table_shape(plan)

    def keepable(name: str, leaf) -> bool:
        return (name in info.refs and name not in info.bad
                and not info.has_join and not info.passthrough.get(id(leaf), True))

    leaf_preds = _join_leaf_filters(plan) if info.has_join else {}

    def hit(name: str, names) -> bool:
        return name in names or (info.has_join and _renamed_ref(name, names))

    def deferrable(name: str, leaf) -> bool:
        """Under a join, leave a dictionary column encoded: the join engine decodes it after the
        leaf's filter, or after the final gather (joined rows only), so every consumer above the
        join still sees plain strings (F-032). A use the Python path can't take on dictionaries
        ("bad": CASE WHEN, arithmetic, ...) only matters in the leaf's own filter, where the
        column is decoded first as before; a bare column projected out above the join is not
        such a use (`bad_expr`): it is decoded after the gather."""
        if single:
            # one table, filtered then selected (F-046f): the native filter decodes only the rows
            # that pass, or keeps codes for a DISTINCT above; unless a filter reads the column in a
            # way the engine can't evaluate on codes (CASE WHEN, arithmetic, ...)
            return name not in info.bad_expr
        # a join renames a clashing right column (`n_name` -> `n_name_right`): the plan reads that name
        if not info.has_join or not hit(name, info.refs) or info.passthrough.get(id(leaf), True):
            return False
        if id(leaf) not in leaf_preds:              # not a Filter*(Scan) input of a join
            return False
        return not hit(name, info.bad_expr) or name not in leaf_preds[id(leaf)]

    def _scan(table, defer):
        s = Scan(table)
        if defer:
            s._mx_decode_after = frozenset(defer)
            # read above the join only in dictionary-native ways (a bare group key, a predicate the
            # engine evaluates on codes): the join passes them on as dictionaries, never decoded
            keep = frozenset(n for n in defer if not hit(n, info.bad))
            if keep:
                s._mx_keep_dict = keep
        return s

    def resolve_leaf(leaf: ParquetScan, pushed: tuple):
        names = leaf.schema.names
        if info.passthrough.get(id(leaf), True):
            cols = list(names)
        else:
            cols = [n for n in names if n in info.refs or (info.has_join and _renamed_ref(n, info.refs))] or names[:1]
        if not read:
            return dataclasses.replace(leaf, columns=tuple(cols), filters=pushed or None)
        candidates = dictionary_columns(leaf.path, leaf.schema, leaf.dictionary,
                                        leaf.max_dictionary_bytes, cols)
        dict_cols = [n for n in candidates if keepable(n, leaf) or deferrable(n, leaf)]
        mask_dict = [n for n in candidates if n not in dict_cols]
        import pyarrow.parquet as pq
        st = _stream.get()
        batch = st.groups if st is not None and st.is_streamed(leaf.path) else None
        token = source_token(leaf.path, cols, pushed, batch)  # before the read: a file replaced meanwhile never matches
        if os.path.isfile(leaf.path):
            if batch is not None:                              # the streamed file: this batch's row groups only
                table = _read_file(leaf.path, cols, dict_cols, pushed or (), mask_dict, only=batch)
            elif st is not None:                               # any other file is read once for all batches
                table = st.once((token, tuple(dict_cols), tuple(mask_dict)),
                                lambda: _read_file(leaf.path, cols, dict_cols, pushed or (), mask_dict))
            else:
                table = _read_file(leaf.path, cols, dict_cols, pushed or (), mask_dict)
        else:
            table = pq.read_table(leaf.path, columns=cols,
                                  filters=_filters_expression(pushed) if pushed else None,
                                  read_dictionary=dict_cols or None)
        table, defer = _decode_unkept(table, leaf)
        return _scan(tag_source(table, token), defer)

    def _decode_unkept(table, leaf):
        defer = []
        for i, (n, t) in enumerate(zip(table.column_names, table.schema.types)):
            if _is_dict(t) and not keepable(n, leaf):
                if deferrable(n, leaf):
                    defer.append(n)
                else:
                    table = table.set_column(i, n, _decode_dict(table.column(n)))
        return table, defer

    def resolve_scan(leaf: Scan):
        table = leaf.table
        drop = [n for n, t in zip(table.column_names, table.schema.types)
                if _is_dict(t) and not keepable(n, leaf)]
        if not drop:
            return leaf
        defer = [n for n in drop if deferrable(n, leaf)]
        for n in drop:
            if n in defer:
                continue
            i = table.schema.get_field_index(n)
            table = table.set_column(i, n, _decode_dict(table.column(n)))
        return _scan(table, defer)

    def rebuild(node):
        if isinstance(node, ParquetScan):
            return resolve_leaf(node, ())
        if isinstance(node, Scan):
            return resolve_scan(node)
        if isinstance(node, Filter):
            chain, n = [], node
            while isinstance(n, Filter):
                chain.append(n)
                n = n.input
            if isinstance(n, ParquetScan):
                pushed = []
                for f in chain:
                    for c in _conjuncts(f.predicate):
                        p = _pushable(c, n.schema)
                        if p is not None and p not in pushed:
                            pushed.append(p)
                out = resolve_leaf(n, tuple(pushed))
                for f in reversed(chain):
                    out = Filter(out, f.predicate)
                return out
            return Filter(rebuild(node.input), node.predicate)
        if isinstance(node, Join):
            return dataclasses.replace(node, left=rebuild(node.left), right=rebuild(node.right))
        if hasattr(node, "input"):
            return dataclasses.replace(node, input=rebuild(node.input))
        return node

    return rebuild(plan)
