"""
MXFrame group-key kernel for dictionary-encoded keys, CPU side (F-015).

Turns the dictionary-code buffers of Arrow ``DictionaryArray`` chunks (read in place,
any integer width) into int32 composite group ids in one parallel pass, and records which
composite ids occur in a presence table. A second small pass rewrites the composite ids
to dense ids (only combinations that occur become groups).

Chunk table (int64, one row of 6 per chunk, all keys back to back):
    row_start, row_end, base_addr, remap_addr, width, size
  base_addr is the address of the chunk's code buffer minus row_start * itemsize, so
  ``base_addr + row * itemsize`` is valid for every row of the chunk. remap_addr is 0 or
  an int32 table mapping the chunk's codes to the unified dictionary. width: 0 int8,
  1 int16, 2 int32, 3 uint8, 4 uint16.
key_off int64[n_keys + 1]: first chunk row of each key in the table.
strides int64[n_keys]: composite = sum(code_k * stride_k), must stay below `space`.

Build:
    pixi run mojo build kernels_aot/groupkeys_cpu.mojo -I kernels_aot --target-cpu x86-64-v3 \
        --emit shared-lib -o kernels_aot/libmxgroupkeys_cpu.so
"""

from std.memory import UnsafePointer, alloc
from std.algorithm import parallelize
from std.atomic import Atomic
from std.ffi import external_call

comptime MORSEL = 16384

comptime IP = UnsafePointer[Int32, MutAnyOrigin]
comptime LP = UnsafePointer[Int64, MutAnyOrigin]
comptime BP = UnsafePointer[UInt8, MutAnyOrigin]


@always_inline
def _ip(addr: Int) -> IP:
    return IP(unsafe_from_address=addr)

@always_inline
def _lp(addr: Int) -> LP:
    return LP(unsafe_from_address=addr)

@always_inline
def _bp(addr: Int) -> BP:
    return BP(unsafe_from_address=addr)


@always_inline
def _pass[T: DType, REMAP: Bool, FIRST: Bool](
    dst: IP, addr: Int, remap: IP, stride: Int32, size: Int32, err: IP, a: Int, b: Int,
):
    var src = UnsafePointer[Scalar[T], MutAnyOrigin](unsafe_from_address=addr)
    var mn = Int32(2147483647)
    var mx = Int32(-2147483648)
    for i in range(a, b):
        var v = Int32(src[i])
        mn = min(mn, v)
        mx = max(mx, v)
        # An invalid code is reported (rc -5) but clamped so nothing is read or written out of bounds.
        v = min(max(v, Int32(0)), size - 1)
        comptime if REMAP:
            v = remap[Int(v)]
        comptime if FIRST:
            dst[i] = v * stride
        else:
            dst[i] += v * stride
    if mn < 0 or mx >= size:
        err[0] = 1


@always_inline
def _dispatch[FIRST: Bool](dst: IP, width: Int, addr: Int, remap_a: Int, stride: Int32, size: Int32, err: IP, a: Int, b: Int):
    var remap = _ip(remap_a)
    if remap_a != 0:
        if width == 0:
            _pass[DType.int8, True, FIRST](dst, addr, remap, stride, size, err, a, b)
        elif width == 1:
            _pass[DType.int16, True, FIRST](dst, addr, remap, stride, size, err, a, b)
        elif width == 2:
            _pass[DType.int32, True, FIRST](dst, addr, remap, stride, size, err, a, b)
        elif width == 3:
            _pass[DType.uint8, True, FIRST](dst, addr, remap, stride, size, err, a, b)
        else:
            _pass[DType.uint16, True, FIRST](dst, addr, remap, stride, size, err, a, b)
    else:
        if width == 0:
            _pass[DType.int8, False, FIRST](dst, addr, remap, stride, size, err, a, b)
        elif width == 1:
            _pass[DType.int16, False, FIRST](dst, addr, remap, stride, size, err, a, b)
        elif width == 2:
            _pass[DType.int32, False, FIRST](dst, addr, remap, stride, size, err, a, b)
        elif width == 3:
            _pass[DType.uint8, False, FIRST](dst, addr, remap, stride, size, err, a, b)
        else:
            _pass[DType.uint16, False, FIRST](dst, addr, remap, stride, size, err, a, b)


@export
def mx_groupkeys_composite(
    tbl_a: Int, key_off_a: Int, n_keys: Int, strides_a: Int, n_rows: Int,
    out_a: Int, present_a: Int, space: Int, threads: Int,
) abi("C") -> Int:
    """Composite int32 ids into out (n_rows) and presence flags into present (space bytes)."""
    if n_keys < 1 or space < 1 or space > 1073741824:
        return -1
    var tbl = _lp(tbl_a)
    var key_off = _lp(key_off_a)
    var strides = _lp(strides_a)
    var dst_p = _ip(out_a)
    var present = _bp(present_a)
    for s in range(space):
        present[s] = 0
    if n_rows <= 0:
        return 0
    var morsels = (n_rows + MORSEL - 1) // MORSEL
    var nthreads = max(1, min(threads, morsels))
    # Per-worker presence tables, padded so two workers never share a cache line.
    var pstride = ((space + 63) // 64) * 64 + 128
    var local = alloc[UInt8](nthreads * pstride)
    var counter = alloc[Int64](1)
    counter[0] = 0
    var err = alloc[Int32](1)
    err[0] = 0

    @parameter
    def worker(wid: Int):
        var pl = local + wid * pstride
        for s in range(space):
            pl[s] = 0
        while True:
            var m = Int(Atomic[DType.int64].fetch_add(counter, 1))
            var lo = m * MORSEL
            if lo >= n_rows:
                break
            var hi = min(lo + MORSEL, n_rows)
            for k in range(n_keys):
                var stride = Int32(strides[k])
                var c0 = Int(key_off[k])
                var c1 = Int(key_off[k + 1])
                for c in range(c0, c1):
                    var e = tbl + c * 6
                    var cs = Int(e[0])
                    var ce = Int(e[1])
                    if ce <= lo or cs >= hi:
                        continue
                    var a = max(lo, cs)
                    var b = min(hi, ce)
                    var w = Int(e[4])
                    var addr = Int(e[2])
                    if k == 0:
                        _dispatch[True](dst_p, w, addr, Int(e[3]), stride, Int32(e[5]), err, a, b)
                    else:
                        _dispatch[False](dst_p, w, addr, Int(e[3]), stride, Int32(e[5]), err, a, b)
            for i in range(lo, hi):
                pl[Int(dst_p[i])] = 1

    if nthreads == 1:
        worker(0)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[worker](nthreads)
    for t in range(nthreads):
        var pl = local + t * pstride
        for s in range(space):
            if pl[s] != 0:
                present[s] = 1
    local.free()
    counter.free()
    var bad = err[0]
    err.free()
    if bad != 0:
        return -5
    return 0


@export
def mx_groupkeys_remap(ids_a: Int, n: Int, table_a: Int, threads: Int) abi("C") -> Int:
    """Set ids[i] = table[ids[i]] in place, in parallel."""
    var ids = _ip(ids_a)
    var table = _ip(table_a)
    if n <= 0:
        return 0
    var morsels = (n + MORSEL - 1) // MORSEL
    var nthreads = max(1, min(threads, morsels))
    var counter = alloc[Int64](1)
    counter[0] = 0

    @parameter
    def worker(wid: Int):
        while True:
            var m = Int(Atomic[DType.int64].fetch_add(counter, 1))
            var lo = m * MORSEL * 4
            if lo >= n:
                break
            var hi = min(lo + MORSEL * 4, n)
            for i in range(lo, hi):
                ids[i] = table[Int(ids[i])]

    if nthreads == 1:
        worker(0)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[worker](nthreads)
    counter.free()
    return 0


@export
def mx_gk_init_runtime() abi("C") -> Int:
    """Pin the Mojo async runtime (worker pool) for the life of the process (see kernels_aot.mojo)."""
    return external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


@export
def mx_gk_touch(addr: Int, nbytes: Int, threads: Int) abi("C") -> Int:
    """Write zeros over a fresh buffer in parallel, so its pages are faulted in before the hot pass."""
    var CH = 4 * 1024 * 1024
    var chunks = (nbytes + CH - 1) // CH
    if chunks <= 0:
        return 0
    var p = _bp(addr)
    var nthreads = max(1, min(threads, chunks))

    @parameter
    def worker(c: Int):
        var lo = c * CH
        var hi = min(lo + CH, nbytes)
        for i in range(lo, hi):
            p[i] = 0

    if nthreads == 1:
        for c in range(chunks):
            worker(c)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[worker](chunks, nthreads)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# F-020: dense group ids for integer keys (no hashing)
#
# Direct-address encoding for an integer column whose value range is small compared with its
# row count (TPC-H keys are dense): code = value - lo. Group ids are numbered in order of first
# appearance, exactly like Arrow's ``dictionary_encode``, so results keep today's order:
#   mx_ik_range   parallel min/max of the column
#   mx_ik_first   tab[code] = smallest row index holding it (n when absent): per-worker tables
#                 for tiny ranges, one shared table with atomic min otherwise
#   mx_ik_rank    mark first-appearance rows in scr (n int32), exclusive prefix sum -> rank of
#                 each first row; returns the number of groups
#   mx_ik_finish  tab[code] = group id (-1 absent), uniq[g] = value, first[g] = its first row
#   mx_ik_apply   ids[i] = tab[value_i - lo]
# Element widths: 0 int8, 1 int16, 2 int32, 3 int64, 4 uint8, 5 uint16, 6 uint32.
# ─────────────────────────────────────────────────────────────────────────────
comptime IK_LOCAL_MAX = 65536
comptime IK_CHUNK = 262144


@always_inline
def _load(addr: Int, w: Int, i: Int) -> Int:
    if w == 3:
        return Int(UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=addr)[i])
    if w == 2:
        return Int(UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)[i])
    if w == 0:
        return Int(UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=addr)[i])
    if w == 1:
        return Int(UnsafePointer[Int16, MutAnyOrigin](unsafe_from_address=addr)[i])
    if w == 4:
        return Int(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)[i])
    if w == 5:
        return Int(UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=addr)[i])
    return Int(UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=addr)[i])


def _workers(n: Int, threads: Int) -> Int:
    return max(1, min(threads, (n + IK_CHUNK - 1) // IK_CHUNK))


@export
def mx_ik_range(src_a: Int, w: Int, n: Int, out_a: Int, threads: Int) abi("C") -> Int:
    """out[0] = min, out[1] = max of the n values."""
    var out = _lp(out_a)
    if n <= 0:
        return -1
    var nth = _workers(n, threads)
    var part = alloc[Int64](nth * 16)     # padded: two workers never share a cache line

    @parameter
    def worker(wid: Int):
        var lo = (n * wid) // nth
        var hi = (n * (wid + 1)) // nth
        var mn = _load(src_a, w, lo)
        var mx = mn
        for i in range(lo, hi):
            var v = _load(src_a, w, i)
            mn = min(mn, v)
            mx = max(mx, v)
        part[wid * 16] = Int64(mn)
        part[wid * 16 + 1] = Int64(mx)

    if nth == 1:
        worker(0)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[worker](nth)
    var mn = part[0]
    var mx = part[1]
    for t in range(1, nth):
        mn = min(mn, part[t * 16])
        mx = max(mx, part[t * 16 + 1])
    out[0] = mn
    out[1] = mx
    part.free()
    return 0


@export
def mx_ik_first(src_a: Int, w: Int, n: Int, lo: Int, r: Int, tab_a: Int, threads: Int) abi("C") -> Int:
    """tab[code] = first row holding `code` (n when none). rc -5: a value outside [lo, lo + r)."""
    if n <= 0 or r <= 0 or n >= 2147483647 or r >= 2147483647:
        return -1
    var tab = _ip(tab_a)
    var nth = _workers(n, threads)
    var nn = Int32(n)
    var err = alloc[Int32](1)
    err[0] = 0
    var counter = alloc[Int64](1)
    counter[0] = 0
    var chunks = (n + IK_CHUNK - 1) // IK_CHUNK
    if nth > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()

    if r <= IK_LOCAL_MAX and nth > 1:
        var pstride = ((r + 15) // 16) * 16 + 16
        var local = alloc[Int32](nth * pstride)

        @parameter
        def wl(wid: Int):
            var t = local + wid * pstride
            for s in range(r):
                t[s] = nn
            var bad = False
            while True:
                var m = Int(Atomic[DType.int64].fetch_add(counter, 1))
                if m >= chunks:
                    break
                var a = m * IK_CHUNK
                var b = min(a + IK_CHUNK, n)
                for i in range(a, b):
                    var c = _load(src_a, w, i) - lo
                    if c < 0 or c >= r:
                        bad = True
                        continue
                    if t[c] == nn:
                        t[c] = Int32(i)
            if bad:
                err[0] = 1

        parallelize[wl](nth)
        for s in range(r):
            var best = nn
            for t in range(nth):
                best = min(best, local[t * pstride + s])
            tab[s] = best
        local.free()
    else:
        # shared table: initialise in parallel, then atomic min per row
        var ich = (r + IK_CHUNK - 1) // IK_CHUNK

        @parameter
        def wi(c: Int):
            var a = c * IK_CHUNK
            var b = min(a + IK_CHUNK, r)
            for s in range(a, b):
                tab[s] = nn

        if nth == 1:
            for c in range(ich):
                wi(c)
        else:
            parallelize[wi](ich, nth)

        @parameter
        def ws(wid: Int):
            var bad = False
            while True:
                var m = Int(Atomic[DType.int64].fetch_add(counter, 1))
                if m >= chunks:
                    break
                var a = m * IK_CHUNK
                var b = min(a + IK_CHUNK, n)
                for i in range(a, b):
                    var c = _load(src_a, w, i) - lo
                    if c < 0 or c >= r:
                        bad = True
                        continue
                    # plain read racing other workers' atomic mins is benign: the entry only ever
                    # decreases, a stale (larger) value just makes one redundant atomic min
                    if Int32(i) < tab[c]:
                        _ = Atomic[DType.int32].min(tab + c, Int32(i))
            if bad:
                err[0] = 1

        if nth == 1:
            ws(0)
        else:
            parallelize[ws](nth)
    var bad_rc = err[0]
    err.free()
    counter.free()
    if bad_rc != 0:
        return -5
    return 0


@export
def mx_ik_rank(tab_a: Int, r: Int, n: Int, scr_a: Int, threads: Int) abi("C") -> Int:
    """Number of groups; scr[first_row] = group rank (0-based) for every first-appearance row."""
    var tab = _ip(tab_a)
    var scr = _ip(scr_a)
    var nn = Int32(n)
    var nth = _workers(max(n, r), threads)
    var chunks_n = (n + IK_CHUNK - 1) // IK_CHUNK
    var chunks_r = (r + IK_CHUNK - 1) // IK_CHUNK
    if nth > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()

    @parameter
    def zero(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        for i in range(a, b):
            scr[i] = 0

    @parameter
    def mark(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, r)
        for s in range(a, b):
            var p = tab[s]
            if p < nn:
                scr[Int(p)] = 1

    var sums = alloc[Int64](chunks_n + 1)

    @parameter
    def count(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        var s = Int64(0)
        for i in range(a, b):
            s += Int64(scr[i])
        sums[c] = s

    @parameter
    def scan(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        var run = Int32(sums[c])
        for i in range(a, b):
            var v = scr[i]
            scr[i] = run
            run += v

    if nth == 1:
        for c in range(chunks_n):
            zero(c)
        for c in range(chunks_r):
            mark(c)
        for c in range(chunks_n):
            count(c)
    else:
        parallelize[zero](chunks_n, nth)
        parallelize[mark](chunks_r, nth)
        parallelize[count](chunks_n, nth)
    var total = Int64(0)
    for c in range(chunks_n):
        var s = sums[c]
        sums[c] = total
        total += s
    if nth == 1:
        for c in range(chunks_n):
            scan(c)
    else:
        parallelize[scan](chunks_n, nth)
    sums.free()
    return Int(total)


@export
def mx_ik_finish(tab_a: Int, r: Int, n: Int, lo: Int, scr_a: Int, uniq_a: Int, first_a: Int,
                 threads: Int) abi("C") -> Int:
    """tab[code] = group id (or -1); uniq[g] = value (int64); first[g] = first row (int32)."""
    var tab = _ip(tab_a)
    var scr = _ip(scr_a)
    var uniq = _lp(uniq_a)
    var first = _ip(first_a)
    var nn = Int32(n)
    var nth = _workers(r, threads)
    var chunks_r = (r + IK_CHUNK - 1) // IK_CHUNK

    @parameter
    def fin(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, r)
        for s in range(a, b):
            var p = tab[s]
            if p < nn:
                var g = Int(scr[Int(p)])
                uniq[g] = Int64(s + lo)
                first[g] = p
                tab[s] = Int32(g)
            else:
                tab[s] = -1

    if nth == 1:
        for c in range(chunks_r):
            fin(c)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[fin](chunks_r, nth)
    return 0


@export
def mx_ik_apply(src_a: Int, w: Int, n: Int, lo: Int, tab_a: Int, ids_a: Int, threads: Int) abi("C") -> Int:
    """ids[i] = tab[value_i - lo]."""
    var tab = _ip(tab_a)
    var ids = _ip(ids_a)
    var nth = _workers(n, threads)
    var chunks = (n + IK_CHUNK - 1) // IK_CHUNK

    @parameter
    def work(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        for i in range(a, b):
            ids[i] = tab[_load(src_a, w, i) - lo]

    if nth == 1:
        for c in range(chunks):
            work(c)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[work](chunks, nth)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Composite keys and SORTED dense ranks (what np.unique(composite, return_inverse=True) gives):
#   mx_ik_composite  out[i] = sum_k code_k[i] * stride_k over per-key code buffers (any width above)
#   mx_ik_present    pres[value - lo] = 1
#   mx_ik_prefix     tab = exclusive prefix sum of pres (rank of each present value); returns groups
#   mx_ik_uniq       uniq[tab[c]] = c + lo for present c
#   mx_ik_rank_sorted  ranks from a stable argsort of int64 keys (sparse composites)
# ─────────────────────────────────────────────────────────────────────────────
@export
def mx_ik_composite(n_keys: Int, addrs_a: Int, widths_a: Int, strides_a: Int, n: Int, out_a: Int,
                    threads: Int) abi("C") -> Int:
    """Composite int64 = sum of code * stride over up to 8 keys."""
    if n_keys < 1 or n_keys > 8:
        return -1
    var addrs = _lp(addrs_a)
    var widths = _lp(widths_a)
    var strides = _lp(strides_a)
    var out = _lp(out_a)
    var nth = _workers(n, threads)
    var chunks = (n + IK_CHUNK - 1) // IK_CHUNK

    @parameter
    def work(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        for i in range(a, b):
            out[i] = 0
        for k in range(n_keys):
            var ad = Int(addrs[k])
            var w = Int(widths[k])
            var s = Int(strides[k])
            for i in range(a, b):
                out[i] += Int64(_load(ad, w, i) * s)

    if nth == 1:
        for c in range(chunks):
            work(c)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[work](chunks, nth)
    return 0


@export
def mx_ik_present(src_a: Int, w: Int, n: Int, lo: Int, r: Int, pres_a: Int, threads: Int) abi("C") -> Int:
    """pres[value - lo] = 1 (pres is zeroed by the caller). rc -5: a value outside [lo, lo + r)."""
    var pres = _bp(pres_a)
    var nth = _workers(n, threads)
    var chunks = (n + IK_CHUNK - 1) // IK_CHUNK
    var err = alloc[Int32](1)
    err[0] = 0

    @parameter
    def work(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        var bad = False
        for i in range(a, b):
            var v = _load(src_a, w, i) - lo
            if v < 0 or v >= r:
                bad = True
                continue
            pres[v] = 1
        if bad:
            err[0] = 1

    if nth == 1:
        for c in range(chunks):
            work(c)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[work](chunks, nth)
    var bad_rc = err[0]
    err.free()
    if bad_rc != 0:
        return -5
    return 0


@export
def mx_ik_prefix(pres_a: Int, r: Int, tab_a: Int, threads: Int) abi("C") -> Int:
    """tab[c] = number of present values below c; returns the number of present values."""
    var pres = _bp(pres_a)
    var tab = _ip(tab_a)
    var nth = _workers(r, threads)
    var chunks = (r + IK_CHUNK - 1) // IK_CHUNK
    var sums = alloc[Int64](chunks + 1)

    @parameter
    def count(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, r)
        var s = Int64(0)
        for i in range(a, b):
            s += Int64(pres[i])
        sums[c] = s

    @parameter
    def scan(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, r)
        var run = Int32(sums[c])
        for i in range(a, b):
            tab[i] = run
            run += Int32(pres[i])

    if nth > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[count](chunks, nth)
    else:
        for c in range(chunks):
            count(c)
    var total = Int64(0)
    for c in range(chunks):
        var s = sums[c]
        sums[c] = total
        total += s
    if nth > 1:
        parallelize[scan](chunks, nth)
    else:
        for c in range(chunks):
            scan(c)
    sums.free()
    return Int(total)


@export
def mx_ik_uniq(pres_a: Int, tab_a: Int, r: Int, lo: Int, uniq_a: Int, threads: Int) abi("C") -> Int:
    """uniq[tab[c]] = c + lo for every present c."""
    var pres = _bp(pres_a)
    var tab = _ip(tab_a)
    var uniq = _lp(uniq_a)
    var nth = _workers(r, threads)
    var chunks = (r + IK_CHUNK - 1) // IK_CHUNK

    @parameter
    def work(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, r)
        for i in range(a, b):
            if pres[i] != 0:
                uniq[Int(tab[i])] = Int64(i + lo)

    if nth > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[work](chunks, nth)
    else:
        for c in range(chunks):
            work(c)
    return 0


@export
def mx_ik_rank_sorted(keys_a: Int, perm_a: Int, n: Int, ids_a: Int, first_a: Int, threads: Int) abi("C") -> Int:
    """Group ids from a stable argsort `perm` of int64 keys; ids in sorted-key order, first[g] = smallest row."""
    var keys = _lp(keys_a)
    var perm = _ip(perm_a)
    var ids = _ip(ids_a)
    var first = _ip(first_a)
    var nth = _workers(n, threads)
    var chunks = (n + IK_CHUNK - 1) // IK_CHUNK
    var sums = alloc[Int64](chunks + 1)

    @parameter
    def count(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        var s = Int64(0)
        for j in range(a, b):
            if j == 0 or keys[Int(perm[j])] != keys[Int(perm[j - 1])]:
                s += 1
        sums[c] = s

    @parameter
    def emit(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        var g = Int(sums[c]) - 1
        for j in range(a, b):
            var p = Int(perm[j])
            if j == 0 or keys[p] != keys[Int(perm[j - 1])]:
                g += 1
                first[g] = Int32(p)
            ids[p] = Int32(g)

    if nth > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[count](chunks, nth)
    else:
        for c in range(chunks):
            count(c)
    var total = Int64(0)
    for c in range(chunks):
        var s = sums[c]
        sums[c] = total
        total += s
    if nth > 1:
        parallelize[emit](chunks, nth)
    else:
        for c in range(chunks):
            emit(c)
    sums.free()
    return Int(total)


comptime IK_BUCKET_MAX = 256


@export
def mx_ik_rank_bucketed(keys_a: Int, n: Int, stride: Int, nb: Int, ids_a: Int, first_a: Int,
                        cnt_a: Int, rows_a: Int, threads: Int) abi("C") -> Int:
    """``mx_ik_rank_sorted`` without the sort, for keys whose leading part is a small code (F-022).

    Rows are bucketed by ``keys[i] // stride`` (in [0, nb)) with a counting sort, then each bucket
    is sorted by (key, row). Ids are numbered in ascending key order and first[g] is the group's
    smallest row, exactly as the radix-sort path gives them. Scratch: cnt int32[nb + 1],
    rows int32[n]. Returns the number of groups, -5 for a key outside [0, nb * stride), -6 when a
    bucket holds more than IK_BUCKET_MAX rows (the caller then sorts)."""
    var keys = _lp(keys_a)
    var ids = _ip(ids_a)
    var first = _ip(first_a)
    var cnt = _ip(cnt_a)
    var rows = _ip(rows_a)
    var st = Int64(stride)
    var nth = _workers(max(n, nb), threads)
    var chunks_n = (n + IK_CHUNK - 1) // IK_CHUNK
    var chunks_b = (nb + IK_CHUNK - 1) // IK_CHUNK
    var err = alloc[Int32](1)
    err[0] = 0
    var sums = alloc[Int64](chunks_b + 1)
    if nth > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()

    @parameter
    def zero(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, nb + 1)
        for j in range(a, b):
            cnt[j] = 0

    @parameter
    def hist(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        var bad = False
        for i in range(a, b):
            var k = keys[i]
            if k < 0:
                bad = True
                continue
            var bk = Int(k // st)
            if bk >= nb:
                bad = True
                continue
            _ = Atomic[DType.int32].fetch_add(cnt + bk, 1)
        if bad:
            err[0] = 1

    # bucket offsets: cnt[b] becomes the start of bucket b (exclusive prefix), cnt[nb] = n
    @parameter
    def bsum(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, nb)
        var s = Int64(0)
        for j in range(a, b):
            s += Int64(cnt[j])
        sums[c] = s

    @parameter
    def bscan(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, nb)
        var run = Int32(sums[c])
        for j in range(a, b):
            var v = cnt[j]
            cnt[j] = run
            run += v

    # scatter rows; cnt[b] advances to the end of bucket b (= start of b + 1)
    @parameter
    def scatter(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, n)
        for i in range(a, b):
            var bk = Int(keys[i] // st)
            var p = Atomic[DType.int32].fetch_add(cnt + bk, 1)
            rows[Int(p)] = Int32(i)

    # sort each bucket by (key, row); count its distinct keys into sums
    @parameter
    def sort_count(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, nb)
        var s = Int64(0)
        for j in range(a, b):
            var lo = 0 if j == 0 else Int(cnt[j - 1])
            var hi = Int(cnt[j])
            if hi - lo > IK_BUCKET_MAX:
                err[0] = 2
                continue
            for x in range(lo + 1, hi):
                var r = rows[x]
                var kr = keys[Int(r)]
                var y = x - 1
                while y >= lo:
                    var ry = rows[y]
                    var ky = keys[Int(ry)]
                    if ky < kr or (ky == kr and ry < r):
                        break
                    rows[y + 1] = ry
                    y -= 1
                rows[y + 1] = r
            for x in range(lo, hi):
                if x == lo or keys[Int(rows[x])] != keys[Int(rows[x - 1])]:
                    s += 1
        sums[c] = s

    @parameter
    def emit(c: Int):
        var a = c * IK_CHUNK
        var b = min(a + IK_CHUNK, nb)
        var g = Int(sums[c]) - 1
        for j in range(a, b):
            var lo = 0 if j == 0 else Int(cnt[j - 1])
            var hi = Int(cnt[j])
            for x in range(lo, hi):
                var p = Int(rows[x])
                if x == lo or keys[p] != keys[Int(rows[x - 1])]:
                    g += 1
                    first[g] = Int32(p)
                ids[p] = Int32(g)

    def run_n[f: def(Int) capturing -> None](chunks: Int):
        if nth > 1:
            parallelize[f](chunks, nth)
        else:
            for c in range(chunks):
                f(c)

    run_n[zero]((nb + 1 + IK_CHUNK - 1) // IK_CHUNK)
    run_n[hist](chunks_n)
    if err[0] != 0:
        err.free()
        sums.free()
        return -5
    run_n[bsum](chunks_b)
    var total = Int64(0)
    for c in range(chunks_b):
        var s = sums[c]
        sums[c] = total
        total += s
    run_n[bscan](chunks_b)
    cnt[nb] = Int32(n)
    run_n[scatter](chunks_n)
    run_n[sort_count](chunks_b)
    if err[0] != 0:
        err.free()
        sums.free()
        return -6
    total = 0
    for c in range(chunks_b):
        var s = sums[c]
        sums[c] = total
        total += s
    run_n[emit](chunks_b)
    err.free()
    sums.free()
    return Int(total)
