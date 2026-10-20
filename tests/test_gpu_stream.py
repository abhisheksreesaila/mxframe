"""Pinned staging + async host->device copies for the Router's GPU worker."""
import threading

import numpy as np
import pytest

from omxcore import gpu_stream

pytestmark = pytest.mark.skipif(not gpu_stream.cuda_available(), reason="no CUDA GPU")


def _roundtrip(stager, slot, arrays):
    out = []
    for dev_ptr, a in zip(slot.device_ptrs, arrays):
        host = np.empty_like(a)
        gpu_stream.d2h(host, dev_ptr)
        out.append(host)
    return out


def test_upload_roundtrip_multiple_dtypes():
    st = gpu_stream.DeviceStager(slot_bytes=1 << 20, n_slots=2)
    arrays = [np.arange(1000, dtype=np.int32), np.linspace(0, 1, 1000, dtype=np.float32),
              np.arange(1000, dtype=np.float64)]
    slot = st.upload(arrays)
    for got, want in zip(_roundtrip(st, slot, arrays), arrays):
        np.testing.assert_array_equal(got, want)
    st.release(slot)
    st.close()


def test_non_contiguous_slices_are_uploaded_correctly():
    st = gpu_stream.DeviceStager(slot_bytes=1 << 20, n_slots=1)
    base = np.arange(10_000, dtype=np.float32)
    view = base[100:5100]
    slot = st.upload([view])
    np.testing.assert_array_equal(_roundtrip(st, slot, [view])[0], view)
    st.release(slot)
    st.close()


def test_upload_larger_than_slot_raises():
    st = gpu_stream.DeviceStager(slot_bytes=1024, n_slots=1)
    with pytest.raises(ValueError):
        st.upload([np.zeros(1024, dtype=np.float64)])
    st.close()


def test_slots_block_until_released_and_work_from_other_threads():
    st = gpu_stream.DeviceStager(slot_bytes=1 << 16, n_slots=1)
    a = np.ones(100, dtype=np.float32)
    s1 = st.upload([a])
    got = []

    def other():
        s2 = st.upload([a * 2])  # blocks until s1 is released; runs on a new thread
        got.append(_roundtrip(st, s2, [a])[0])
        st.release(s2)

    t = threading.Thread(target=other)
    t.start()
    t.join(0.2)
    assert t.is_alive()  # waiting for a free slot
    st.release(s1)
    t.join(5)
    assert not t.is_alive()
    np.testing.assert_array_equal(got[0], a * 2)
    st.close()


def test_measure_link_bandwidth_is_positive():
    gbps = gpu_stream.measure_h2d_gbps(nbytes=64 << 20, pinned=True)
    assert gbps > 0.5
