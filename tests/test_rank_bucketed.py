"""F-022: the counting-sort path of `group_keys.dense_sorted` (composites led by a small code, an
order key say) returns exactly what the radix-sort path and np.unique return."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.group_keys as gk
from mxframe import LazyFrame, col
from mxframe.lazy_frame import Scan

pytestmark = pytest.mark.skipif(not gk.available(), reason="libmxgroupkeys_cpu.so not built")


def _has_kernel():
    return hasattr(gk._load(), "mx_ik_rank_bucketed")


def _pairs(n, n_lead, n_second, seed):
    r = np.random.default_rng(seed)
    lead = np.sort(r.integers(0, n_lead, n)).astype(np.int64)     # lineitem-like: clustered by order
    r.shuffle(lead[: n // 3])                                       # and partly shuffled
    second = r.integers(0, n_second, n).astype(np.int64)
    return lead, second


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_lead", [60_000, 250_000])
def test_matches_np_unique_and_the_sort_path(seed, n_lead):
    if not _has_kernel():
        pytest.skip("kernel not built")
    n, n_second = 1_000_000, 1000
    lead, second = _pairs(n, n_lead, n_second, seed)
    comp = lead * n_second + second
    got = gk.dense_sorted(comp, lead=(n_second, n_lead))
    sorted_path = gk.dense_sorted(comp)            # no lead: radix sort
    uniq, first, inv = np.unique(comp, return_index=True, return_inverse=True)
    for ids, u, f in (got, sorted_path):
        np.testing.assert_array_equal(ids, inv.astype(np.int32))
        np.testing.assert_array_equal(u, uniq)
        np.testing.assert_array_equal(f, first)


def test_long_buckets_and_bad_keys_fall_back():
    if not _has_kernel():
        pytest.skip("kernel not built")
    n = 200_000
    r = np.random.default_rng(1)
    comp = r.integers(0, 1 << 40, n).astype(np.int64)
    lib, th = gk._load(), gk._threads()
    # few codes, so buckets are long: declined before the kernel runs
    assert gk._rank_bucketed(lib, comp, 1 << 37, 8, th) is None
    # a key past n_codes * stride: the kernel reports it
    assert gk._rank_bucketed(lib, comp, 1 << 20, 1000, th) is None
    # a bucket with more than IK_BUCKET_MAX rows
    skew = np.arange(n, dtype=np.int64)
    skew[:1000] = 5
    assert gk._rank_bucketed(lib, skew, 1, n, th) is None
    # and dense_sorted still answers through the sort
    ids, u, _ = gk.dense_sorted(comp, lead=(1 << 20, 1000))
    np.testing.assert_array_equal(u, np.unique(comp))


def test_two_key_distinct_through_the_engine():
    n = 300_000
    lead, second = _pairs(n, 75_000, 997, 7)
    t = pa.table({"o": pa.array(lead * 4 + 1), "s": pa.array(second.astype(np.int32))})
    out = LazyFrame(Scan(t)).select("o", "s").distinct().compute(device="cpu")
    want = {(int(a), int(b)) for a, b in zip(lead * 4 + 1, second)}
    got = set(zip(out.column("o").to_pylist(), out.column("s").to_pylist()))
    assert len(got) == out.num_rows == len(want)
    assert got == want
