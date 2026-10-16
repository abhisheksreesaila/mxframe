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
    # negative keys (null markers) never match, on either side
    lk = np.array([3, -1, 0, 2, -5, 3, 1], np.int32)
    rk = np.array([-1, 3, 2, -1, 3, 0, -7, 1], np.int32)
    li, ri = g.hash_join(lk, rk)
    want = [(i, j) for i, a in enumerate(lk) if a >= 0 for j, b in enumerate(rk) if b == a]
    assert list(zip(li.tolist(), ri.tolist())) == want
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


def sort_multikey():
    """LazyFrame.sort("k", "j") on the GPU (and CPU): correct order, ties in row order,
    including key columns whose combined radix overflows int32."""
    import pyarrow as pa
    from mxframe import LazyFrame, Scan
    rng = np.random.default_rng(3)
    for n, kmax, jmax in [(10, 3, 3), (1000, 5, 7), (1024, 5, 7), (100_000, 100, 100),
                          (100_000, 100_000, 100_000)]:
        t = pa.table({"k": rng.integers(0, kmax, n).astype(np.int32),
                      "j": rng.integers(0, jmax, n).astype(np.int32),
                      "v": np.arange(n, dtype=np.int32)})
        df = t.to_pandas()
        for desc in (False, True):
            want = df.sort_values(["k", "j"], ascending=not desc, kind="stable")["v"].to_numpy()
            for dev in ("cpu", "gpu"):
                got = LazyFrame(Scan(t)).sort("k", "j", descending=desc).compute(device=dev)
                assert got.column("v").to_numpy().tolist() == want.tolist(), (n, kmax, jmax, desc, dev)


def late_suppliers():
    """Q21 kernel (CAS-published hash table): per-target-supplier counts vs NumPy."""
    g = gpu()
    rng = np.random.default_rng(2)
    n, n_orders, n_sup = 400_000, 60_000, 500
    order = rng.integers(0, n_orders, n).astype(np.int32)
    sup = rng.integers(0, n_sup, n).astype(np.int32)
    commit = rng.integers(0, 100, n).astype(np.int32)
    receipt = (commit + rng.integers(-30, 40, n)).astype(np.int32)
    finished = (rng.random(n_orders) > 0.3).astype(np.int32)
    target = (rng.random(n_sup) > 0.5).astype(np.int32)
    late = receipt > commit
    pairs = np.unique(np.stack([order[late], sup[late]], 1), axis=0)
    per_order = np.bincount(pairs[:, 0], minlength=n_orders)
    sel = late & (per_order[order] == 1) & (finished[order] != 0) & (target[sup] != 0)
    want = np.bincount(sup[sel], minlength=n_sup).astype(np.int32)
    for _ in range(3):
        got = g.single_late_supplier_counts_i32(order, sup, commit, receipt, finished, target)
        np.testing.assert_array_equal(got, want)


def sort_end_to_end():
    """LazyFrame sort on the GPU vs a stable pandas sort: single and multi key, ascending and
    descending, with and without LIMIT (top-k), n not a power of two and spanning many blocks,
    and after a group-by (the reviewer's 2M rows / 1000 groups case)."""
    import pyarrow as pa
    from mxframe import LazyFrame, Scan, col
    rng = np.random.default_rng(9)
    for n in (5, 1000, 4097, 100_003, 2_000_000):
        t = pa.table({"k": rng.integers(0, 1000, n).astype(np.int32),
                      "j": rng.integers(0, 7, n).astype(np.int32),
                      "v": np.arange(n, dtype=np.int32)})
        df = t.to_pandas()
        for by in (["k"], ["k", "j"], ["j", "k"]):
            for desc in (False, True):
                base = LazyFrame(Scan(t)).sort(*by, descending=desc)
                want = df.sort_values(by, ascending=not desc, kind="stable")["v"].to_numpy()
                got = base.compute(device="gpu").column("v").to_numpy()
                assert got.tolist() == want.tolist(), (n, by, desc, "full")
                for lim in (1, 10, 1000):
                    if lim < n:
                        got = base.limit(lim).compute(device="gpu").column("v").to_numpy()
                        assert got.tolist() == want[:lim].tolist(), (n, by, desc, lim)
    n, ng = 2_000_000, 1000
    t = pa.table({"k": rng.integers(0, ng, n).astype(np.int32), "v": rng.random(n, dtype=np.float32)})
    for desc in (False, True):
        out = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).sort("k", descending=desc).compute(device="gpu")
        keys = out.column("k").to_numpy()
        assert len(keys) == ng and (np.diff(keys) < 0 if desc else np.diff(keys) > 0).all(), desc
        ref = np.bincount(t.column("k").to_numpy(), t.column("v").to_numpy().astype(np.float64), ng)
        np.testing.assert_allclose(out.column("s").to_numpy(), ref[keys], rtol=1e-3, atol=1e-2)


def other_thread():
    """The co-processing Router runs the GPU worker on its own thread: every wrapper, including
    DeviceArray.to_numpy()/free(), must work from a thread that never touched CUDA."""
    import threading
    g = gpu()                                   # created on the main thread
    rng = np.random.default_rng(4)
    val = rng.random(50_000, dtype=np.float32)
    lab = rng.integers(0, 50, 50_000).astype(np.int32)
    lk = rng.integers(0, 500, 5_000).astype(np.int32)
    rk = rng.integers(0, 500, 800).astype(np.int32)
    errors, results = [], {}

    def worker():
        try:
            results["sum"] = g.group_sum_f32(val, lab, 50)
            dev = g.group_sum_f32(val, lab, 50, device_out=True)
            results["dev"] = dev.to_numpy()
            dev.free()
            results["join"] = g.hash_join(lk, rk)
            results["sort"] = g.sort_topk(lab, 10)
            results["enc"] = g.group_encode_i32(lab)
        except BaseException as e:            # noqa: BLE001
            errors.append(repr(e))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not errors, errors
    ref = np.bincount(lab, val.astype(np.float64), 50)
    np.testing.assert_allclose(results["sum"], ref, rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(results["dev"], ref, rtol=1e-4, atol=1e-2)
    li, ri = results["join"]
    assert len(li) == int(sum((rk == a).sum() for a in lk))
    np.testing.assert_array_equal(results["sort"], np.argsort(lab, kind="stable")[:10])
    assert results["enc"][1] == len(np.unique(lab))


def group_skew(kind: str):
    """Skewed high-cardinality group-bys (global-atomic path): correct, and no slower than a small
    multiple of the uniform case.  Float atomicAdd used to lower to a compare-exchange loop, so a
    hot group made this super-linear (10M rows / 100k groups, one key at 25%: 40 s)."""
    import time
    n, ng = 10_000_000, 100_000
    g = gpu()
    rng = np.random.default_rng(0)
    val = rng.random(n, dtype=np.float32)

    def timed(lab):
        g.group_sum_f32(val, lab, ng)                       # warm up + upload
        t = time.perf_counter()
        out = g.group_sum_f32(val, lab, ng)
        cnt = g.group_count_f32(lab, ng)
        return time.perf_counter() - t, out, cnt

    uni = rng.integers(0, ng, n).astype(np.int32)
    t_uni, _, _ = timed(uni)
    if kind == "zipf":
        lab = np.minimum(rng.zipf(1.3, n) - 1, ng - 1).astype(np.int32)
    else:
        lab = rng.integers(0, ng, n).astype(np.int32)
        lab[rng.random(n) < float(kind)] = 7
    dt, out, cnt = timed(lab)
    np.testing.assert_allclose(out, np.bincount(lab, val.astype(np.float64), ng), rtol=1e-3, atol=1e-2)
    np.testing.assert_array_equal(cnt, np.bincount(lab, minlength=ng).astype(np.float32))
    assert dt < max(6 * t_uni, 0.1), f"skew {kind}: {dt:.3f}s vs uniform {t_uni:.3f}s"


SKIP = 77        # exit status: the scenario cannot run here (test_gpu_kernels turns it into a skip)


def _own_gpu_bytes() -> int:
    """GPU memory held by this process (the driver's per-process accounting), so other
    processes allocating or freeing on the same GPU do not move it.  Exits with SKIP when the
    driver cannot say (no nvidia-smi, memory reported as [N/A], our pid not listed: a container
    with its own pid namespace)."""
    import subprocess

    def skip(why):
        print(f"SKIP: per-process GPU accounting unavailable: {why}")
        sys.exit(SKIP)

    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        skip(f"nvidia-smi failed ({e})")
    mine = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        pid, mem = (x.strip() for x in ln.split(","))
        if pid == str(os.getpid()):
            if not mem.isdigit():
                skip(f"used_memory is {mem!r}")
            mine.append(int(mem))
    if not mine:
        skip(f"pid {os.getpid()} is not listed")
    return sum(mine) << 20


def join_failure_frees_scratch():
    """A hash_join that dies mid-way (table for a huge key does not fit) leaks nothing."""
    g = gpu()
    # warm up; 2M rows also make the recursive scan create its first kernel-side buffer, which is
    # when the kernel library's memory pool (a quarter of the GPU) is reserved: not a leak
    g.hash_join(np.arange(2_000_000, dtype=np.int32), np.array([2], np.int32))
    used0 = _own_gpu_bytes()
    for _ in range(5):
        try:
            g.hash_join(np.array([1, 2**31 - 2], np.int32), np.array([2**31 - 2], np.int32))
        except (RuntimeError, MemoryError, OverflowError):
            pass
    leaked = _own_gpu_bytes() - used0
    assert leaked < 64 << 20, leaked


def masked_aggs(n: str):
    """masked sum / sum_product / min / max on the GPU vs NumPy float64 (CPU-kernel semantics)."""
    n = int(n)
    g = gpu()
    rng = np.random.default_rng(n + 7)
    a = (rng.random(n, dtype=np.float32) * 200 - 100).astype(np.float32)
    b = (rng.random(n, dtype=np.float32) * 4 - 2).astype(np.float32)
    F32MAX = float(F32_MAX)

    def check(a, b, mask, tag):
        live = mask != 0
        for name, got, want in (
            ("sum", g.masked_global_sum_f32(a, mask), float(a[live].astype(np.float64).sum())),
            ("prod", g.masked_global_sum_product_f32(a, b, mask),
             float((a[live].astype(np.float64) * b[live].astype(np.float64)).sum())),
        ):
            tol = 1e-6 * max(1.0, float(np.abs(a[live].astype(np.float64)).sum()) * 2)
            assert abs(got - want) <= tol, (tag, name, got, want)
        wmin = float(np.nanmin(a[live])) if live.any() and not np.isnan(a[live]).all() else F32MAX
        wmax = float(np.nanmax(a[live])) if live.any() and not np.isnan(a[live]).all() else -F32MAX
        assert g.masked_global_min_f32(a, mask) == wmin, (tag, "min")
        assert g.masked_global_max_f32(a, mask) == wmax, (tag, "max")
        # deterministic: same bits every run
        assert g.masked_global_sum_f32(a, mask) == g.masked_global_sum_f32(a, mask)

    rand = (rng.random(n) > 0.4).astype(np.int32)
    check(a, b, rand, "random")
    check(a, b, np.zeros(n, np.int32), "none")
    check(a, b, np.ones(n, np.int32), "all")
    # NaN in masked-out rows is ignored
    an = a.copy(); mk = rand.copy()
    an[::7] = np.nan; mk[::7] = 0
    check(an, b, mk, "nan-masked-out")
    if n:
        # NaN in a live row: sum poisons (like the CPU kernel), min/max skip it
        ln = a.copy(); ln[n // 2] = np.nan
        mall = np.ones(n, np.int32)
        assert np.isnan(g.masked_global_sum_f32(ln, mall))
        okv = ln[~np.isnan(ln)]
        assert g.masked_global_min_f32(ln, mall) == (float(okv.min()) if len(okv) else F32MAX)
        assert g.masked_global_max_f32(ln, mall) == (float(okv.max()) if len(okv) else -F32MAX)
    # range5 fused predicate + product
    if n:
        ic = rng.integers(0, 100, n).astype(np.int32)
        fc = rng.random(n, dtype=np.float32)
        uc = (rng.random(n, dtype=np.float32) * 50).astype(np.float32)
        sel = (ic >= 10) & (ic < 60) & (fc >= np.float32(0.2)) & (fc <= np.float32(0.7)) & (uc < np.float32(24))
        want = float((a[sel].astype(np.float64) * b[sel].astype(np.float64)).sum())
        got = g.range5_sum_product_f32(a, b, ic, fc, uc, 10, 60, 0.2, 0.7, 24.0)
        assert abs(got - want) <= 1e-6 * max(1.0, float(np.abs(a).astype(np.float64).sum()) * 2), (got, want)


def masked_f64_matches_cpu():
    """GPU masked outputs are float64: at ~1e9 magnitude they agree with the CPU kernels to 1e-12."""
    g = gpu()
    cpu = ak.AOTKernels()
    n = 4_000_000
    rng = np.random.default_rng(3)
    a = (rng.random(n, dtype=np.float32) * 500 + 500).astype(np.float32)    # ~1e9 total
    b = (rng.random(n, dtype=np.float32) + 0.5).astype(np.float32)
    m = (rng.random(n) > 0.2).astype(np.int32)
    for name, got, want in (
        ("sum", g.masked_global_sum_f32(a, m), cpu.masked_global_sum(a, m)),
        ("prod", g.masked_global_sum_product_f32(a, b, m), cpu.masked_global_sum_product(a, b, m)),
    ):
        assert want > 1e8 and abs(got - want) <= 1e-12 * abs(want), (name, got, want)
    assert g.masked_global_min_f32(a, m) == cpu.masked_global_min(a, m)
    assert g.masked_global_max_f32(a, m) == cpu.masked_global_max(a, m)
    ic = rng.integers(0, 100, n).astype(np.int32)
    fc = rng.random(n, dtype=np.float32)
    uc = (rng.random(n, dtype=np.float32) * 50).astype(np.float32)
    sel = (ic >= 10) & (ic < 60) & (fc >= np.float32(0.2)) & (fc <= np.float32(0.7)) & (uc < np.float32(24))
    want = float((a[sel].astype(np.float64) * b[sel].astype(np.float64)).sum())
    got = g.range5_sum_product_f32(a, b, ic, fc, uc, 10, 60, 0.2, 0.7, 24.0)
    assert abs(got - want) <= 1e-12 * abs(want), (got, want)
    assert g.masked_global_sum_f32(a, m, device_out=True).dtype == np.float64


SCENARIOS = {f.__name__: f for f in (masked_f64_matches_cpu, masked_aggs, join_failure_frees_scratch, other_thread, group_skew, sort_end_to_end, late_suppliers, sort_multikey, group_aggs, group_edge_cases, group_encode, sort_topk, hash_join, device_chaining)}


def register(fn):
    SCENARIOS[fn.__name__] = fn
    return fn


if __name__ == "__main__":
    SCENARIOS[sys.argv[1]](*sys.argv[2:])
    print("OK")
