"""Verified radix sort: the native argsort's result is checked in O(n) before anything trusts it.

One radix-sort order failure was seen once in a loaded full-suite run (F-014, not reproducible); a bad
permutation would silently split groups, and an out-of-range index would be written through by kernels
that consume it, so every caller goes through ``checked_sort_indices``.
"""
from __future__ import annotations

import warnings

import numpy as np

_WARNED = False


def is_sorted_permutation(keys, idx) -> bool:
    """keys[idx] ascending and idx a valid index vector (cheap O(n) checks)."""
    if len(idx) != len(keys) or (len(idx) and (idx.min() < 0 or idx.max() >= len(keys))):
        return False
    k = keys[idx]
    if k.dtype.kind == "f":
        k = np.where(np.isnan(k), np.inf, k)
    return bool(np.all(k[1:] >= k[:-1])) if len(k) > 1 else True


def warn_fallback(what="radix sort returned an unsorted order") -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        warnings.warn(f"mxframe: {what}; using np.argsort (please report)",
                      RuntimeWarning, stacklevel=3)


def checked_sort_indices(aot, keys):
    """``aot.sort_indices(keys)``, or a stable ``np.argsort`` (with one warning) when its order is bad."""
    idx = aot.sort_indices(keys)
    if not is_sorted_permutation(keys, idx):
        warn_fallback()
        return np.argsort(keys, kind="stable").astype(np.int32)
    return idx


def is_sorted_topk(keys, idx, k) -> bool:
    """`idx` is exactly ``np.argsort(keys, kind="stable")[:k]`` (checked in O(n), no sort): valid
    rows in strictly increasing (key, row) order, with no row outside it that sorts before the last."""
    k = min(int(k), len(keys))
    if len(idx) != k:
        return False
    if k == 0:
        return True
    if idx.min() < 0 or idx.max() >= len(keys):
        return False
    sel = keys[idx]
    if k > 1:
        step = sel[1:] > sel[:-1]
        if not np.all(step | ((sel[1:] == sel[:-1]) & (idx[1:] > idx[:-1]))):
            return False
    last_key, last_row = sel[-1], int(idx[-1])
    return int((keys < last_key).sum()) + int((keys[:last_row + 1] == last_key).sum()) == k
