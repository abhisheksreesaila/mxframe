"""ResidencyTracker: which host columns already have a copy on the GPU."""
import numpy as np
import pytest

from mxframe.residency import ResidencyTracker


class FakeDevice:
    """Records allocations instead of touching a GPU."""

    def __init__(self):
        self.next_ptr = 0x1000
        self.live = {}
        self.uploads = 0

    def alloc_and_upload(self, arr):
        ptr = self.next_ptr
        self.next_ptr += max(arr.nbytes, 1) + 256
        self.live[ptr] = arr.copy()
        self.uploads += 1
        return ptr

    def free(self, ptr):
        del self.live[ptr]


def make(budget):
    dev = FakeDevice()
    return ResidencyTracker(budget_bytes=budget, device=dev), dev


def test_second_request_for_same_buffer_is_a_hit():
    rt, dev = make(1 << 20)
    a = np.arange(1000, dtype=np.float32)
    p1 = rt.get_or_upload(a)
    p2 = rt.get_or_upload(a)
    assert p1 == p2 and dev.uploads == 1
    assert rt.stats.hits == 1 and rt.stats.bytes_h2d == a.nbytes


def test_view_of_same_memory_with_different_extent_is_a_different_entry():
    rt, dev = make(1 << 20)
    a = np.arange(1000, dtype=np.float32)
    rt.get_or_upload(a)
    rt.get_or_upload(a[:500])
    assert dev.uploads == 2


def test_is_resident_and_resident_bytes():
    rt, _ = make(1 << 20)
    a = np.zeros(100, dtype=np.int32)
    assert not rt.is_resident(a)
    rt.get_or_upload(a)
    assert rt.is_resident(a)
    assert rt.resident_bytes() == a.nbytes


def test_lru_eviction_under_budget():
    rt, dev = make(1000)
    a, b, c = (np.zeros(100, dtype=np.float32) for _ in range(3))  # 400 B each
    rt.get_or_upload(a)
    rt.get_or_upload(b)
    rt.get_or_upload(a)          # a is now most recently used
    rt.get_or_upload(c)          # evicts b
    assert rt.is_resident(a) and rt.is_resident(c) and not rt.is_resident(b)
    assert len(dev.live) == 2 and rt.stats.evictions == 1


def test_pinned_entries_are_not_evicted():
    rt, _ = make(1000)
    a, b, c = (np.zeros(100, dtype=np.float32) for _ in range(3))
    rt.get_or_upload(a, pin=True)
    rt.get_or_upload(b)
    rt.get_or_upload(c)          # must evict b, not pinned a
    assert rt.is_resident(a) and not rt.is_resident(b)


def test_too_big_for_budget_is_uploaded_but_not_kept():
    rt, dev = make(100)
    big = np.zeros(1000, dtype=np.float32)
    ptr = rt.get_or_upload(big)
    assert ptr is not None and not rt.is_resident(big)
    rt.release_transient(ptr)
    assert dev.live == {}


def test_keeps_owner_alive_so_addresses_are_not_recycled():
    rt, _ = make(1 << 20)
    a = np.arange(10, dtype=np.float64)
    rt.get_or_upload(a)
    addr = a.ctypes.data
    del a
    b = np.arange(10, dtype=np.float64) + 5
    # a's buffer is still referenced by the tracker, so b cannot reuse its address
    assert b.ctypes.data != addr


def test_clear_frees_everything():
    rt, dev = make(1 << 20)
    rt.get_or_upload(np.zeros(10))
    rt.get_or_upload(np.ones(10), pin=True)
    rt.clear()
    assert dev.live == {} and rt.resident_bytes() == 0


def test_non_contiguous_input_rejected():
    rt, _ = make(1 << 20)
    with pytest.raises(ValueError):
        rt.get_or_upload(np.zeros((10, 10))[:, 0])


def test_real_gpu_roundtrip():
    from mxframe import gpu_stream
    if not gpu_stream.cuda_available():
        pytest.skip("no CUDA GPU")
    rt = ResidencyTracker(1 << 26, gpu_stream.CudaDevice())
    a = np.arange(12345, dtype=np.float32)
    ptr = rt.get_or_upload(a)
    back = np.empty_like(a)
    gpu_stream.d2h(back, ptr)
    np.testing.assert_array_equal(back, a)
    assert rt.get_or_upload(a) == ptr
    rt.clear()
