"""F-153: the hash join kernel through its C ABI (libmxhashjoin_cpu.so), the way a non-Mojo caller uses it."""
import ctypes
from array import array
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parent.parent / "kernels_aot" / "libmxhashjoin_cpu.so"


@pytest.fixture(scope="module")
def lib():
    L = ctypes.CDLL(str(LIB))
    for name in ("mx_hj_build", "mx_hj_info", "mx_hj_probe_unique", "mx_hj_probe_pairs", "mx_hj_abi_version", "mx_hj_init_runtime"):
        getattr(L, name).restype = ctypes.c_long
    L.mx_hj_init_runtime()
    return L


def addr(a):
    return a.buffer_info()[0] if len(a) else 0


def build(lib, keys, ok=None):
    k = array("q", keys)
    o = array("B", ok) if ok is not None else None
    h = lib.mx_hj_build(ctypes.c_long(addr(k)), ctypes.c_long(addr(o) if o else 0), ctypes.c_long(len(k)), ctypes.c_long(2))
    assert h > 0
    return h


def probe_pairs(lib, h, keys, ok=None):
    k = array("q", keys)
    o = array("B", ok) if ok is not None else None
    out = array("q", [0, 0, 0])
    rc = lib.mx_hj_probe_pairs(ctypes.c_long(h), ctypes.c_long(addr(k)), ctypes.c_long(addr(o) if o else 0), ctypes.c_long(len(k)),
                               ctypes.c_long(2), ctypes.c_long(10**9), ctypes.c_long(addr(out)))
    assert rc == 0
    n = out[0]
    if n == 0:
        assert out[1] == 0 and out[2] == 0          # no pairs: no arrays to release
    pr = (ctypes.c_int32 * n).from_address(out[1]) if n else []
    br = (ctypes.c_int32 * n).from_address(out[2]) if n else []
    pairs = list(zip(pr, br))
    if n:
        lib.mx_hj_release_array(ctypes.c_long(out[1]))
        lib.mx_hj_release_array(ctypes.c_long(out[2]))
    return pairs


def test_abi_version(lib):
    assert lib.mx_hj_abi_version() == 1


def test_unique_keys_probe_gives_the_build_row_or_minus_one(lib):
    h = build(lib, [10, 20, 30, 40])
    info = array("q", [0] * 4)
    lib.mx_hj_info(ctypes.c_long(h), ctypes.c_long(addr(info)))
    assert list(info)[:3] == [1, 4, 1]      # direct array, 4 usable rows, unique
    k = array("q", [30, 5, 10, 41, 40])
    out = array("i", [0] * 5)
    rc = lib.mx_hj_probe_unique(ctypes.c_long(h), ctypes.c_long(addr(k)), ctypes.c_long(0), ctypes.c_long(5), ctypes.c_long(addr(out)), ctypes.c_long(2))
    assert rc == 0 and list(out) == [2, -1, 0, -1, 3]
    lib.mx_hj_free(ctypes.c_long(h))


def test_repeated_keys_and_null_keys_make_every_pair_in_probe_order(lib):
    h = build(lib, [7, 7, 9, 7], ok=[1, 1, 1, 0])        # build row 3 is a NULL key: never matches
    pairs = probe_pairs(lib, h, [7, 8, 9, 7], ok=[1, 1, 1, 0])   # probe row 3 is NULL too
    assert pairs == [(0, 0), (0, 1), (2, 2)]
    lib.mx_hj_free(ctypes.c_long(h))


def test_wide_sparse_keys_use_the_hash_layout_and_keep_int64_exact(lib):
    keys = [(1 << 60) + i * 1000003 for i in range(20000)]
    h = build(lib, keys)
    info = array("q", [0] * 4)
    lib.mx_hj_info(ctypes.c_long(h), ctypes.c_long(addr(info)))
    assert info[0] == 2
    pairs = probe_pairs(lib, h, [keys[5], keys[5] + 1, keys[19999]])
    assert pairs == [(0, 5), (2, 19999)]
    lib.mx_hj_free(ctypes.c_long(h))


def test_pairs_past_the_budget_are_refused_with_minus_32(lib):
    h = build(lib, [1] * 100)
    k = array("q", [1] * 100)
    out = array("q", [0, 0, 0])
    rc = lib.mx_hj_probe_pairs(ctypes.c_long(h), ctypes.c_long(addr(k)), ctypes.c_long(0), ctypes.c_long(100), ctypes.c_long(2),
                               ctypes.c_long(1000), ctypes.c_long(addr(out)))
    assert rc == -32 and out[0] == 10000
    lib.mx_hj_free(ctypes.c_long(h))


def test_a_handle_that_is_not_a_map_is_refused_or_ignored_not_a_crash(lib):
    lib.mx_hj_free(ctypes.c_long(0))                         # like free(NULL)
    lib.mx_hj_free(ctypes.c_long(-31))
    info = array("q", [0] * 4)
    assert lib.mx_hj_info(ctypes.c_long(-31), ctypes.c_long(addr(info))) == -33
    assert lib.mx_hj_info(ctypes.c_long(0), ctypes.c_long(addr(info))) == -33
    k = array("q", [1])
    out = array("q", [0, 0, 0])
    assert lib.mx_hj_probe_pairs(ctypes.c_long(0), ctypes.c_long(addr(k)), ctypes.c_long(0), ctypes.c_long(1), ctypes.c_long(1),
                                 ctypes.c_long(10), ctypes.c_long(addr(out))) == -33
    o = array("i", [0])
    assert lib.mx_hj_probe_unique(ctypes.c_long(-1), ctypes.c_long(addr(k)), ctypes.c_long(0), ctypes.c_long(1), ctypes.c_long(addr(o)),
                                  ctypes.c_long(1)) == -33


def test_a_build_of_2_pow_31_rows_is_refused_with_minus_31(lib):
    assert lib.mx_hj_build(ctypes.c_long(0), ctypes.c_long(0), ctypes.c_long(1 << 31), ctypes.c_long(1)) == -31


def test_no_matching_pair_returns_no_arrays(lib):
    h = build(lib, [1, 2, 3])
    assert probe_pairs(lib, h, [10, 20]) == []
    lib.mx_hj_free(ctypes.c_long(h))
