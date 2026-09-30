"""ResidencyTracker: remembers which host column buffers already have a GPU copy.

Arrow buffers are immutable, so (address, byte length, dtype) identifies a
column's contents for as long as the buffer is alive; the tracker keeps a
reference to the host array so the address cannot be recycled while its GPU
copy is cached. Least-recently-used entries are evicted to stay within the
budget (`gpu_mem_fraction` of device memory); pinned entries are never evicted.

The cost model treats resident input as free to move (Mordred, PVLDB 2022:
"run it where the data already is"); `lf.cache(device="gpu")` pins columns.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class DeviceLike(Protocol):
    def alloc_and_upload(self, arr: np.ndarray) -> int: ...
    def free(self, ptr: int) -> None: ...


@dataclass
class ResidencyStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_h2d: int = 0


@dataclass
class _Entry:
    ptr: int
    nbytes: int
    owner: np.ndarray
    pinned: bool


def _key(arr: np.ndarray):
    return (arr.ctypes.data, arr.nbytes, arr.dtype.str)


class ResidencyTracker:
    def __init__(self, budget_bytes: int, device: DeviceLike):
        self.budget = int(budget_bytes)
        self.device = device
        self._entries: "OrderedDict[tuple, _Entry]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.stats = ResidencyStats()

    def is_resident(self, arr: np.ndarray) -> bool:
        with self._lock:
            return _key(arr) in self._entries

    def resident_bytes(self) -> int:
        return self._bytes

    def get_or_upload(self, arr: np.ndarray, pin: bool = False) -> int:
        """Device pointer for `arr`, uploading it on a miss.

        An array larger than the whole budget is uploaded but not kept; call
        `release_transient(ptr)` when done with it.
        """
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("residency needs C-contiguous arrays")
        k = _key(arr)
        with self._lock:
            e = self._entries.get(k)
            if e is not None:
                self._entries.move_to_end(k)
                e.pinned = e.pinned or pin
                self.stats.hits += 1
                return e.ptr
            self.stats.misses += 1
        ptr = self.device.alloc_and_upload(arr)
        with self._lock:
            self.stats.bytes_h2d += arr.nbytes
            if arr.nbytes > self.budget:
                return ptr
            self._make_room(arr.nbytes)
            self._entries[k] = _Entry(ptr, arr.nbytes, arr, pin)
            self._bytes += arr.nbytes
        return ptr

    def release_transient(self, ptr: int) -> None:
        with self._lock:
            if any(e.ptr == ptr for e in self._entries.values()):
                return
        self.device.free(ptr)

    def _make_room(self, need: int) -> None:
        for k in list(self._entries):
            if self._bytes + need <= self.budget:
                return
            e = self._entries[k]
            if e.pinned:
                continue
            del self._entries[k]
            self._bytes -= e.nbytes
            self.device.free(e.ptr)
            self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            for e in self._entries.values():
                self.device.free(e.ptr)
            self._entries.clear()
            self._bytes = 0


GPU_MEM_FRACTION = 0.8
_tracker: Optional[ResidencyTracker] = None
_tracker_lock = threading.Lock()


def get_residency(gpu_mem_fraction: float = GPU_MEM_FRACTION) -> ResidencyTracker:
    """Process-wide tracker for GPU 0; budget = fraction of free memory at first use."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            from . import gpu_stream
            free, _ = gpu_stream.mem_info()
            _tracker = ResidencyTracker(int(free * gpu_mem_fraction), gpu_stream.CudaDevice())
        return _tracker
