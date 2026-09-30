"""Pinned staging buffers and async host->device copies (CUDA driver API via ctypes).

The Router's GPU worker uploads each morsel through a `DeviceStager` slot while
the previous morsel's kernel runs. Within one upload, column i is copied into
pinned memory while column i-1 is already crossing the link (the async copy
from pinned memory returns immediately), so the CPU memcpy and the DMA overlap.

The CUDA context is current per thread; every entry point binds the device's
primary context (the one the Mojo kernels' DeviceContext uses) first, so the
stager works from the Router's helper threads.
"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_ALIGN = 256
_CU_STREAM_NON_BLOCKING = 1
_CU_MEMHOSTALLOC_PORTABLE = 1

_lock = threading.Lock()
_drv: Optional["_Driver"] = None


class _Driver:
    def __init__(self):
        lib = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("libcuda not found")
        u64, vp, sz = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t
        P = ctypes.POINTER
        sigs = {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGetCount": [P(ctypes.c_int)],
            "cuDeviceGet": [P(ctypes.c_int), ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [P(u64), ctypes.c_int],
            "cuCtxSetCurrent": [u64],
            "cuMemHostAlloc": [P(vp), sz, ctypes.c_uint],
            "cuMemFreeHost": [vp],
            "cuMemAlloc_v2": [P(u64), sz],
            "cuMemFree_v2": [u64],
            "cuStreamCreate": [P(u64), ctypes.c_uint],
            "cuStreamDestroy_v2": [u64],
            "cuStreamSynchronize": [u64],
            "cuMemcpyHtoDAsync_v2": [u64, vp, sz, u64],
            "cuMemcpyHtoD_v2": [u64, vp, sz],
            "cuMemcpyDtoH_v2": [vp, u64, sz],
        }
        for fn, args in sigs.items():
            f = getattr(lib, fn)
            f.argtypes = args
            f.restype = ctypes.c_int
        self.lib = lib
        self._check(lib.cuInit(0), "cuInit")
        n = ctypes.c_int(0)
        self._check(lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
        if n.value == 0:
            raise RuntimeError("no CUDA device")
        dev = ctypes.c_int(0)
        self._check(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        ctx = ctypes.c_uint64(0)
        self._check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self.ctx = ctx.value

    @staticmethod
    def _check(err: int, what: str) -> None:
        if err != 0:
            raise RuntimeError(f"{what} failed: CUDA error {err}")

    def bind(self) -> None:
        self._check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")


def _driver() -> _Driver:
    global _drv
    with _lock:
        if _drv is None:
            _drv = _Driver()
    _drv.bind()
    return _drv


def cuda_available() -> bool:
    try:
        _driver()
        return True
    except Exception:
        return False


def d2h(host: np.ndarray, dev_ptr: int) -> None:
    d = _driver()
    d._check(d.lib.cuMemcpyDtoH_v2(host.ctypes.data, dev_ptr, host.nbytes), "cuMemcpyDtoH")


@dataclass
class Slot:
    index: int
    device_ptrs: List[int]
    nbytes: List[int]


class _SlotBuffers:
    def __init__(self, d: _Driver, index: int, nbytes: int):
        self.index = index
        self.nbytes = nbytes
        host = ctypes.c_void_p(0)
        d._check(d.lib.cuMemHostAlloc(ctypes.byref(host), nbytes, _CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc")
        self.host = host.value
        dev = ctypes.c_uint64(0)
        d._check(d.lib.cuMemAlloc_v2(ctypes.byref(dev), nbytes), "cuMemAlloc")
        self.dev = dev.value
        stream = ctypes.c_uint64(0)
        d._check(d.lib.cuStreamCreate(ctypes.byref(stream), _CU_STREAM_NON_BLOCKING), "cuStreamCreate")
        self.stream = stream.value
        self.host_view = np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(self.host))

    def free(self, d: _Driver) -> None:
        d.lib.cuStreamDestroy_v2(self.stream)
        d.lib.cuMemFree_v2(self.dev)
        d.lib.cuMemFreeHost(self.host)


class DeviceStager:
    """A small pool of (pinned host buffer, device buffer, stream) slots.

    `upload(arrays)` blocks until a slot is free, lays the arrays out in it and
    returns device pointers; `release(slot)` makes the slot reusable.
    """

    def __init__(self, slot_bytes: int, n_slots: int = 3):
        d = _driver()
        self.slot_bytes = int(slot_bytes)
        self._slots = [_SlotBuffers(d, i, self.slot_bytes) for i in range(n_slots)]
        self._free: "queue.Queue[_SlotBuffers]" = queue.Queue()
        for s in self._slots:
            self._free.put(s)
        self._closed = False

    def upload(self, arrays: List[np.ndarray]) -> Slot:
        offsets, total = [], 0
        for a in arrays:
            offsets.append(total)
            total += (a.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
        if total > self.slot_bytes:
            raise ValueError(f"morsel needs {total} bytes, slot holds {self.slot_bytes}")
        d = _driver()
        s = self._free.get()
        try:
            for a, off in zip(arrays, offsets):
                dst = s.host_view[off:off + a.nbytes].view(a.dtype)
                np.copyto(dst, a, casting="no")
                d._check(d.lib.cuMemcpyHtoDAsync_v2(s.dev + off, s.host + off, a.nbytes, s.stream),
                         "cuMemcpyHtoDAsync")
            d._check(d.lib.cuStreamSynchronize(s.stream), "cuStreamSynchronize")
        except BaseException:
            self._free.put(s)
            raise
        slot = Slot(index=s.index, device_ptrs=[s.dev + off for off in offsets],
                    nbytes=[a.nbytes for a in arrays])
        return slot

    def release(self, slot: Slot) -> None:
        self._free.put(self._slots[slot.index])

    def close(self) -> None:
        if self._closed:
            return
        d = _driver()
        for s in self._slots:
            s.free(d)
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def measure_h2d_gbps(nbytes: int = 256 << 20, pinned: bool = True, reps: int = 3) -> float:
    """Best-of-`reps` host->device bandwidth in GB/s."""
    d = _driver()
    dev = ctypes.c_uint64(0)
    d._check(d.lib.cuMemAlloc_v2(ctypes.byref(dev), nbytes), "cuMemAlloc")
    host_ptr = None
    try:
        if pinned:
            hp = ctypes.c_void_p(0)
            d._check(d.lib.cuMemHostAlloc(ctypes.byref(hp), nbytes, _CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc")
            host_ptr = hp.value
            src = host_ptr
        else:
            buf = np.ones(nbytes, dtype=np.uint8)
            src = buf.ctypes.data
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            d._check(d.lib.cuMemcpyHtoD_v2(dev.value, src, nbytes), "cuMemcpyHtoD")
            best = min(best, time.perf_counter() - t)
        return nbytes / best / 1e9
    finally:
        d.lib.cuMemFree_v2(dev.value)
        if host_ptr:
            d.lib.cuMemFreeHost(host_ptr)
