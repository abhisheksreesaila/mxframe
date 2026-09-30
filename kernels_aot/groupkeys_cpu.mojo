"""
MXFrame group-key kernel for dictionary-encoded keys, CPU side (F-015).

Turns the dictionary-code buffers of Arrow ``DictionaryArray`` chunks (read in place,
any integer width) into int32 composite group ids in one parallel pass, and records which
composite ids occur in a presence table. A second small pass rewrites the composite ids
to dense ids (only combinations that occur become groups).

Chunk table (int64, one row of 5 per chunk, all keys back to back):
    row_start, row_end, base_addr, remap_addr, width
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
    dst: IP, addr: Int, remap: IP, stride: Int32, a: Int, b: Int,
):
    var src = UnsafePointer[Scalar[T], MutAnyOrigin](unsafe_from_address=addr)
    for i in range(a, b):
        var v = Int32(src[i])
        comptime if REMAP:
            v = remap[Int(v)]
        comptime if FIRST:
            dst[i] = v * stride
        else:
            dst[i] += v * stride


@always_inline
def _dispatch[FIRST: Bool](dst: IP, width: Int, addr: Int, remap_a: Int, stride: Int32, a: Int, b: Int):
    var remap = _ip(remap_a)
    if remap_a != 0:
        if width == 0:
            _pass[DType.int8, True, FIRST](dst, addr, remap, stride, a, b)
        elif width == 1:
            _pass[DType.int16, True, FIRST](dst, addr, remap, stride, a, b)
        elif width == 2:
            _pass[DType.int32, True, FIRST](dst, addr, remap, stride, a, b)
        elif width == 3:
            _pass[DType.uint8, True, FIRST](dst, addr, remap, stride, a, b)
        else:
            _pass[DType.uint16, True, FIRST](dst, addr, remap, stride, a, b)
    else:
        if width == 0:
            _pass[DType.int8, False, FIRST](dst, addr, remap, stride, a, b)
        elif width == 1:
            _pass[DType.int16, False, FIRST](dst, addr, remap, stride, a, b)
        elif width == 2:
            _pass[DType.int32, False, FIRST](dst, addr, remap, stride, a, b)
        elif width == 3:
            _pass[DType.uint8, False, FIRST](dst, addr, remap, stride, a, b)
        else:
            _pass[DType.uint16, False, FIRST](dst, addr, remap, stride, a, b)


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
                    var e = tbl + c * 5
                    var cs = Int(e[0])
                    var ce = Int(e[1])
                    if ce <= lo or cs >= hi:
                        continue
                    var a = max(lo, cs)
                    var b = min(hi, ce)
                    var w = Int(e[4])
                    var addr = Int(e[2])
                    if k == 0:
                        _dispatch[True](dst_p, w, addr, Int(e[3]), stride, a, b)
                    else:
                        _dispatch[False](dst_p, w, addr, Int(e[3]), stride, a, b)
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
