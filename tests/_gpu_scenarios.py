"""GPU kernel scenarios for tests/test_gpu_kernels.py.

A kernel that faults (CUDA_ERROR_ILLEGAL_ADDRESS) or hangs poisons or blocks the
whole process, so each scenario runs in its own subprocess:

    python tests/_gpu_scenarios.py <scenario> [args...]

A scenario asserts internally and exits non-zero on failure.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np

import mxframe.aot_kernels as ak

INT_MIN = np.int32(-2147483648)
F32_MAX = np.float32(3.4028235e38)


def gpu():
    return ak.AOTKernelsGPU()


def _labels(rng, n, ng):
    return rng.integers(0, ng, n).astype(np.int32)


def group_aggs(ng: str, n: str = "0"):
    """sum/count/min/max at `ng` groups vs NumPy (n defaults to max(4*ng, 2M))."""
    ng = int(ng)
    n = int(n) or max(4 * ng, 2_000_000)
    g = gpu()
    rng = np.random.default_rng(ng)
    lab = _labels(rng, n, ng)
    val = (rng.random(n, dtype=np.float32) * 100 - 50).astype(np.float32)
    ref_sum = np.bincount(lab, val.astype(np.float64), ng)
    ref_cnt = np.bincount(lab, minlength=ng).astype(np.float32)
    ref_min = np.full(ng, F32_MAX, np.float32)
    ref_max = np.full(ng, -F32_MAX, np.float32)
    np.minimum.at(ref_min, lab, val)
    np.maximum.at(ref_max, lab, val)
    tol = dict(rtol=1e-4, atol=1e-3 * max(1.0, n / ng / 1000))
    np.testing.assert_allclose(g.group_sum_f32(val, lab, ng), ref_sum, **tol)
    np.testing.assert_array_equal(g.group_count_f32(lab, ng), ref_cnt)
    np.testing.assert_array_equal(g.group_min_f32(val, lab, ng), ref_min)
    np.testing.assert_array_equal(g.group_max_f32(val, lab, ng), ref_max)
    np.testing.assert_allclose(
        g.group_mean_f32(val, lab, ng),
        np.where(ref_cnt > 0, ref_sum / np.maximum(ref_cnt, 1), 0), rtol=1e-3, atol=1e-2)


def group_edge_cases():
    """Empty groups keep their identity, labels outside [0, ng) are ignored, n=0."""
    g = gpu()
    val = np.array([5, -2, 7, 1, 9], np.float32)
    lab = np.array([0, 3, 3, -1, 99], np.int32)          # groups 1, 2 empty; -1 and 99 ignored
    for ng in (4, 20_000):                              # shared and global paths
        np.testing.assert_array_equal(g.group_count_f32(lab, ng)[:4], [1, 0, 0, 2])
        np.testing.assert_array_equal(g.group_sum_f32(val, lab, ng)[:4], [5, 0, 0, 5])
        mn = g.group_min_f32(val, lab, ng)
        mx = g.group_max_f32(val, lab, ng)
        np.testing.assert_array_equal(mn[:4], [5, F32_MAX, F32_MAX, -2])
        np.testing.assert_array_equal(mx[:4], [5, -F32_MAX, -F32_MAX, 7])
    empty = np.array([], np.float32)
    assert list(g.group_sum_f32(empty, np.array([], np.int32), 3)) == [0, 0, 0]
    assert list(g.group_count_f32(np.array([], np.int32), 3)) == [0, 0, 0]


def group_encode():
    """Terminates and is correct: duplicates, negative keys, INT_MIN, all-equal, all-distinct."""
    g = gpu()
    rng = np.random.default_rng(1)
    cases = {
        "single": np.array([5], np.int32),
        "all dup": np.array([3, 3, 3, 3], np.int32),
        "negatives": np.array([1, 2, 3, 1, 2, 3, -1, -5, -1], np.int32),
        "int_min": np.array([INT_MIN, 0, INT_MIN, 7, INT_MIN, -1, INT_MIN], np.int32),
        "all int_min": np.full(5000, INT_MIN, np.int32),
        "random small range": rng.integers(-1000, 1000, 100_000).astype(np.int32),
        "random full range": rng.integers(-2**31, 2**31 - 1, 200_000, dtype=np.int64).astype(np.int32),
        "all distinct": np.arange(300_000, dtype=np.int32),
        "8 groups 1M": (np.arange(1_000_000) % 8).astype(np.int32),
        "50k groups 2M": rng.integers(0, 50_000, 2_000_000).astype(np.int32),
        "colliding low bits": ((np.arange(50_000, dtype=np.int64) << 16) - 2**30).astype(np.int32),
    }
    for name, keys in cases.items():
        n = len(keys)
        cap = 1
        while cap < 2 * n:
            cap *= 2
        ids, ng = g.group_encode_i32(keys, cap)
        uniq = np.unique(keys)
        assert ng == len(uniq), (name, ng, len(uniq))
        assert ids.min() >= 0 and ids.max() == ng - 1, name
        order = np.argsort(keys, kind="stable")
        sk, si = keys[order], ids[order]
        assert np.array_equal(sk[1:] == sk[:-1], si[1:] == si[:-1]), name   # same key <=> same id


def sort_topk(n: str = "0"):
    """Stable sort, ascending and descending, for any n (not just powers of two)."""
    g = gpu()
    rng = np.random.default_rng(7)
    sizes = [int(n)] if int(n) else [1, 2, 3, 10, 1000, 1024, 1025, 4097, 100_000, 1_000_003]
    for size in sizes:
        for keys in (rng.integers(-5, 5, size).astype(np.int32),                      # heavy ties
                     rng.integers(-2**31, 2**31 - 1, size, dtype=np.int64).astype(np.int32),
                     np.full(size, INT_MIN, np.int32), np.full(size, 2**31 - 1, np.int32)):
            want = np.argsort(keys, kind="stable").astype(np.int32)
            np.testing.assert_array_equal(g.sort_topk(keys, size), want)
            # descending, ties keep row order (stable): sort the negated widened keys
            want_d = np.argsort(-keys.astype(np.int64), kind="stable").astype(np.int32)
            np.testing.assert_array_equal(g.sort_topk(keys, size, descending=True), want_d)
            k = min(size, 7)
            np.testing.assert_array_equal(g.sort_topk(keys, k), want[:k])
    assert len(g.sort_topk(np.array([], np.int32), 5)) == 0


def hash_join(nl: str = "0", nr: str = "0", keyspace: str = "0"):
    """Inner join pairs equal np-based reference (left-major, right rows in order)."""
    g = gpu()
    rng = np.random.default_rng(11)
    configs = ([(int(nl), int(nr), int(keyspace))] if int(nl) else
               [(1, 1, 1), (10, 7, 5), (1000, 50, 30), (5000, 5000, 100_000), (100_000, 1000, 1000),
                (50_000, 200_000, 60_000)])
    for l, r, ks in configs:
        lk = rng.integers(0, ks, l).astype(np.int32)
        rk = rng.integers(0, ks, r).astype(np.int32)
        li, ri = g.hash_join(lk, rk)
        # reference: for each left row (in order) the matching right rows in row order
        order = np.argsort(rk, kind="stable")
        srt = rk[order]
        lo = np.searchsorted(srt, lk, "left")
        hi = np.searchsorted(srt, lk, "right")
        cnt = hi - lo
        want_l = np.repeat(np.arange(l, dtype=np.int32), cnt)
        want_r = np.concatenate([order[a:b] for a, b in zip(lo, hi)]).astype(np.int32) if cnt.sum() else np.empty(0, np.int32)
        np.testing.assert_array_equal(li, want_l)
        np.testing.assert_array_equal(ri, want_r)
    for a, b in (([], [1]), ([1], []), ([1, 2], [3, 4])):                  # empty / no match
        li, ri = g.hash_join(np.array(a, np.int32), np.array(b, np.int32))
        assert len(li) == 0 and len(ri) == 0 and li.dtype == np.int32


def device_chaining():
    """Device-resident inputs and device_out: operators chain with no host round trip."""
    g = gpu()
    D = ak.DeviceArray
    rng = np.random.default_rng(5)
    n, ng = 300_000, 500
    val = rng.random(n, dtype=np.float32)
    lab = rng.integers(0, ng, n).astype(np.int32)
    ref_sum = np.bincount(lab, val.astype(np.float64), ng)

    # NumPy in, device out, then that output feeds the next kernel
    dsum = g.group_sum_f32(val, lab, ng, device_out=True)
    assert isinstance(dsum, D) and dsum.dtype == np.float32 and len(dsum) == ng and dsum.owns
    np.testing.assert_allclose(dsum.to_numpy(), ref_sum, rtol=1e-4, atol=1e-2)
    idx = np.arange(ng, dtype=np.int32)[::-1].copy()
    gathered = g.gather_f32(dsum, idx)                       # device array as gather source
    np.testing.assert_allclose(gathered, dsum.to_numpy()[::-1])
    dsum.free()
    assert dsum.ptr == 0

    # borrowed pointers (residency tracker / stager slot) are used in place and never freed
    vp = g._cu.malloc(val.nbytes); g._cu.h2d(vp, val)
    lp = g._cu.malloc(lab.nbytes); g._cu.h2d(lp, lab)
    dv, dl = D.borrow(vp, np.float32, n), D.borrow(lp, np.int32, n)
    np.testing.assert_allclose(g.group_sum_f32(dv, dl, ng), ref_sum, rtol=1e-4, atol=1e-2)
    np.testing.assert_array_equal(g.group_count_f32(dl, ng), np.bincount(lab, minlength=ng))
    mask = (val > 0.5).astype(np.int32)
    mp = g._cu.malloc(mask.nbytes); g._cu.h2d(mp, mask)
    dm = D.borrow(mp, np.int32, n)
    assert abs(g.masked_global_sum_f32(dv, dm) - float(val[mask == 1].astype(np.float64).sum())) < 1.0
    ms = g.masked_global_sum_f32(dv, dm, device_out=True)
    assert isinstance(ms, D) and len(ms) == 1
    ms.free()
    dv.free(); dl.free()                                     # borrowed: must not free the buffers
    np.testing.assert_allclose(g.group_sum_f32(dv, dl, ng), ref_sum, rtol=1e-4, atol=1e-2)   # still valid
    g._cu.free(vp); g._cu.free(lp); g._cu.free(mp)

    # group_encode -> group_sum on the ids, all on device
    keys = rng.integers(-50, 50, n).astype(np.int32)
    ids, k = g.group_encode_i32(keys, device_out=True)
    dvals = g.group_sum_f32(val, ids, k)
    assert k == len(np.unique(keys)) and len(dvals) == k
    assert abs(dvals.sum() - val.sum()) < 1.0

    # join keys on device, pairs on device, then gather through them
    lk = rng.integers(0, 1000, 20_000).astype(np.int32)
    rk = rng.integers(0, 1000, 3_000).astype(np.int32)
    li, ri = g.hash_join(lk, rk, device_out=True)
    payload = rng.random(3_000, dtype=np.float32)
    got = g.gather_f32(payload, ri)
    hl, hr = g.hash_join(lk, rk)
    np.testing.assert_array_equal(got, payload[hr])
    np.testing.assert_array_equal(li.to_numpy(), hl)
    li.free(); ri.free()

    # sort on device, permutation stays on device, gather through it
    srt = g.sort_topk(keys, n, device_out=True)
    np.testing.assert_array_equal(g.gather_i32(keys, srt), np.sort(keys, kind="stable"))
    srt.free()

    # filter_gather / unique_mask with device arrays
    m = (rng.random(n) > 0.7).astype(np.int32)
    offs = (np.cumsum(m) - m).astype(np.int32)
    out = g.filter_gather_f32(val, m, offs, int(m.sum()), device_out=True)
    np.testing.assert_array_equal(out.to_numpy(), val[m == 1])
    out.free()
    sk = np.sort(keys)
    um = g.unique_mask(sk, device_out=True)
    np.testing.assert_array_equal(um.to_numpy(), np.r_[1, (sk[1:] != sk[:-1])].astype(np.int32))
    um.free()

    # type/length mismatches are errors, not silent reinterpretation
    try:
        g.group_sum_f32(D.borrow(1, np.int32, 4), lab[:4], 2)
        raise AssertionError("dtype mismatch accepted")
    except TypeError:
        pass
    try:
        g.group_sum_f32(val[:10], lab[:11], 2)
        raise AssertionError("length mismatch accepted")
    except ValueError:
        pass
    try:
        g.gather_f32(val[:10], np.array([0, 10], np.int32))
        raise AssertionError("out-of-range gather accepted")
    except IndexError:
        pass


SCENARIOS = {f.__name__: f for f in (group_aggs, group_edge_cases, group_encode, sort_topk, hash_join, device_chaining)}


def register(fn):
    SCENARIOS[fn.__name__] = fn
    return fn


if __name__ == "__main__":
    SCENARIOS[sys.argv[1]](*sys.argv[2:])
    print("OK")
