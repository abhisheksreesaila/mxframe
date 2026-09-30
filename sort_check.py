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


def warn_fallback() -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        warnings.warn("mxframe: radix sort returned an unsorted order; using np.argsort (please report)",
                      RuntimeWarning, stacklevel=3)


def checked_sort_indices(aot, keys):
    """``aot.sort_indices(keys)``, or a stable ``np.argsort`` (with one warning) when its order is bad."""
    idx = aot.sort_indices(keys)
    if not is_sorted_permutation(keys, idx):
        warn_fallback()
        return np.argsort(keys, kind="stable").astype(np.int32)
    return idx
