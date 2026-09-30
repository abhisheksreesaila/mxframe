"""Group ids for dictionary-encoded keys, straight from the Arrow code buffers (F-015).

``build(table, keys)`` returns ``(dense_ids, n_groups, unique_key_cols)`` exactly as
``CustomOpsCompiler._build_group_ids`` does, or ``None`` when the shape is not handled
(the caller then uses the existing path): a key that is not dictionary typed, nulls, an
unsupported index type or a composite space too large for a dense presence table.

Cost model: one parallel pass over the code buffers (any of int8/int16/int32/uint8/uint16,
read in place, no widening copy) writes int32 composite ids and a presence table; a second
parallel pass rewrites them to dense ids when some combinations never occur. Chunks that
carry different dictionaries (one per Parquet row group) are unified on the tiny
dictionaries only, never on the rows, and chunks that share a dictionary skip that too.
Only combinations that occur become groups, so unused dictionary entries need no check.
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
from .np_arrow import array as _npa, to_numpy as _np_of

__all__ = ["available", "build", "MAX_SPACE"]

MAX_SPACE = 1 << 18            # dense presence/remap table limit (composite id space)
_MAX_DICT_TOTAL = 1 << 20      # total chunk-dictionary entries unified in Python
_HERE = Path(__file__).parent
_lock = threading.Lock()
_lib = None
_I64 = ctypes.c_int64

_WIDTH = {pa.int8(): (0, 1), pa.int16(): (1, 2), pa.int32(): (2, 4), pa.uint8(): (3, 1), pa.uint16(): (4, 2)}


def _find() -> Optional[Path]:
    env = os.environ.get("MXFRAME_GROUPKEYS_LIB")
    if env and Path(env).exists():
        return Path(env)
    for cand in (_HERE / "kernels_aot" / "libmxgroupkeys_cpu.so", _HERE / "libmxgroupkeys_cpu.so"):
        if cand.exists():
            return cand
    return None


def available() -> bool:
    return _find() is not None


def _load():
    global _lib
    with _lock:
        if _lib is None:
            path = _find()
            if path is None:
                raise FileNotFoundError("libmxgroupkeys_cpu.so not found")
            lib = ctypes.CDLL(str(path))
            lib.mx_groupkeys_composite.restype = _I64
            lib.mx_groupkeys_composite.argtypes = [_I64] * 9
            lib.mx_groupkeys_remap.restype = _I64
            lib.mx_groupkeys_remap.argtypes = [_I64] * 4
            lib.mx_gk_init_runtime.restype = _I64
            lib.mx_gk_touch.restype = _I64
            lib.mx_gk_touch.argtypes = [_I64] * 3
            lib.mx_gk_init_runtime()
            _lib = lib
    return _lib


def _threads() -> int:
    from .pipeline_kernels import default_threads
    return default_threads()


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _key_chunks(col, n_rows):
    """The non-empty dictionary chunks of ``col`` or None if the column is not handled."""
    if isinstance(col, pa.ChunkedArray):
        chunks = [c for c in col.chunks if len(c)]
    else:
        chunks = [col] if len(col) else []
    if not chunks or not pa.types.is_dictionary(chunks[0].type):
        return None
    if sum(len(c) for c in chunks) != n_rows:
        return None
    for c in chunks:
        if c.null_count or c.dictionary.null_count or c.type.index_type not in _WIDTH:
            return None
    return chunks


def _unify(chunks):
    """(unified dictionary, per-chunk int32 remap arrays or None when all chunks share one dictionary)."""
    d0 = chunks[0].dictionary
    if all(c.dictionary is d0 or c.dictionary.equals(d0) for c in chunks[1:]):
        return d0, None
    dicts = [c.dictionary for c in chunks]
    if sum(len(d) for d in dicts) > _MAX_DICT_TOTAL:
        return None
    enc = pa.concat_arrays(dicts).dictionary_encode()
    ids = _np_of(enc.indices, zero_copy_only=False).astype(np.int32, copy=False)
    remaps, off = [], 0
    for d in dicts:
        remaps.append(np.ascontiguousarray(ids[off:off + len(d)]))
        off += len(d)
    return enc.dictionary, remaps


def build(table, keys):
    if not available():
        return None
    n_rows = table.num_rows
    if n_rows == 0 or not keys or len(keys) > 8:
        return None
    dicts, sizes, rows, keep = [], [], [], []
    for k in keys:
        chunks = _key_chunks(table.column(k), n_rows)
        if chunks is None or any(len(c.dictionary) == 0 for c in chunks):
            return None      # non-empty chunk with an empty dictionary: invalid codes, old path decides
        u = _unify(chunks)
        if u is None:
            return None
        d, remaps = u
        dicts.append(d)
        sizes.append(len(d))
        keep.append((chunks, remaps))
    strides = []
    for i in range(len(sizes)):
        s = 1
        for j in range(i + 1, len(sizes)):
            s *= sizes[j]
        strides.append(s)
    space = strides[0] * sizes[0]
    if space > MAX_SPACE or space < 1:
        return None

    tbl_rows, key_off = [], [0]
    for chunks, remaps in keep:
        start = 0
        for j, c in enumerate(chunks):
            idx = c.indices
            wcode, isz = _WIDTH[idx.type]
            base = idx.buffers()[1].address + idx.offset * isz - start * isz
            tbl_rows.append((start, start + len(c), base, _addr(remaps[j]) if remaps else 0, wcode,
                             len(remaps[j]) if remaps else len(c.dictionary)))
            start += len(c)
        key_off.append(len(tbl_rows))
    tbl = np.array(tbl_rows, dtype=np.int64)
    key_off = np.array(key_off, dtype=np.int64)
    strides_a = np.array(strides, dtype=np.int64)
    ids = np.empty(n_rows, dtype=np.int32)
    present = np.empty(space, dtype=np.uint8)

    lib = _load()
    th = _threads()
    lib.mx_gk_touch(_addr(ids), ids.nbytes, th)     # fault the pages in with all cores first
    rc = lib.mx_groupkeys_composite(_addr(tbl), _addr(key_off), len(keys), _addr(strides_a), n_rows,
                                    _addr(ids), _addr(present), space, th)
    if rc != 0:
        return None
    pres = np.flatnonzero(present)
    n_groups = len(pres)
    if n_groups == 0:
        return None
    if n_groups != space:
        table_a = np.zeros(space, dtype=np.int32)
        table_a[pres] = np.arange(n_groups, dtype=np.int32)
        if lib.mx_groupkeys_remap(_addr(ids), n_rows, _addr(table_a), th) != 0:
            return None
    unique_key_cols = {}
    for i, k in enumerate(keys):
        key_idx = ((pres // strides[i]) % sizes[i]).astype(np.int64)
        unique_key_cols[k] = dicts[i].take(_npa(key_idx))
    return ids, n_groups, unique_key_cols
