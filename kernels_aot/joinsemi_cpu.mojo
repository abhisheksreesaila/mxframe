"""
MXFrame semi-join prefilter kernel, CPU side (F-018).

A direct-address hash join (kernels_aot.mojo join_count / join_scatter) sizes its table by the
largest key.  When the build side is small and the keys are sparse (TPC-H: 456K order keys among
60M possible ones, probed by 60M lineitem rows), the table is a few hundred MB and every probe is a
cache miss, although almost every probe row has no partner.  This kernel makes one cheap pass first:

    mx_join_semi_select(probe, n_probe, build, n_build, max_key, out, threads) -> count

builds a bitmap of the build keys (max_key + 1 bits: 60M keys are 7.5 MB, which stays in cache) and writes
the row numbers of the probe keys that are in it, ascending, into `out` (capacity n_probe).  The caller then
joins only those rows.  Keys outside [0, max_key] never match.  Returns the count, or a negative error.

Both passes are parallel: the bitmap is set with atomic ORs by workers that claim morsels of the build
keys; the probe workers claim morsels of the probe keys, each writes its hits at the start of its own
morsel-sized slot of `out`, and the slots are then compacted in order.

Build:
    pixi run mojo build kernels_aot/joinsemi_cpu.mojo -I kernels_aot --target-cpu x86-64-v3 \
        --emit shared-lib -o kernels_aot/libmxjoin_cpu.so
"""

from std.memory import UnsafePointer, alloc, memset_zero, memcpy
from std.algorithm import parallelize
from std.atomic import Atomic
from std.ffi import external_call

comptime MORSEL = 1 << 16
comptime ABI_VERSION = 1

comptime IP = UnsafePointer[Int32, MutAnyOrigin]
comptime UP = UnsafePointer[UInt64, MutAnyOrigin]


@always_inline
def _ip(addr: Int) -> IP:
    return IP(unsafe_from_address=addr)


@always_inline
def _up(addr: Int) -> UP:
    return UP(unsafe_from_address=addr)


@always_inline
def _fetch_or_u64(ptr: UP, val: UInt64):
    """Atomic OR (monotonic) via llvm.atomicrmw (bin_op 5 = or); same MLIR pattern as kernels_aot.mojo."""
    var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr.address)
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i64](val)
    _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`5 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i64](p, v)


@export
def mx_join_abi_version() abi("C") -> Int:
    return ABI_VERSION


@export
def mx_join_init_runtime() abi("C") -> Int:
    """Pin the async runtime for the life of the process (see kernels_aot.mojo mx_init_runtime)."""
    return external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


@export
def mx_join_semi_select(
    probe_a: Int, n_probe: Int, build_a: Int, n_build: Int, max_key: Int, out_a: Int, threads: Int,
) abi("C") -> Int:
    if n_probe < 0 or n_build < 0 or max_key < 0 or max_key > 4294967295:
        return -1
    if n_probe == 0 or n_build == 0:
        return 0
    var probe = _ip(probe_a)
    var build = _ip(build_a)
    var out = _ip(out_a)
    var T = max_key + 1
    var n_words = (T + 63) // 64
    var bits = alloc[UInt64](n_words)
    memset_zero(bits, n_words)

    var nb_m = (n_build + MORSEL - 1) // MORSEL
    var np_m = (n_probe + MORSEL - 1) // MORSEL
    var w_build = max(1, min(threads, nb_m))
    var w_probe = max(1, min(threads, np_m))
    var ctr = alloc[Int64](2)
    ctr[0] = 0
    ctr[1] = 0
    var counts = alloc[Int32](np_m)

    @parameter
    def set_bits(w: Int):
        while True:
            var m = Int(Atomic[DType.int64].fetch_add(ctr, 1))
            if m >= nb_m:
                break
            var lo = m * MORSEL
            var hi = min(lo + MORSEL, n_build)
            for i in range(lo, hi):
                var k = Int(build[i])
                if k >= 0 and k < T:
                    _fetch_or_u64(bits + (k >> 6), UInt64(1) << UInt64(k & 63))

    @parameter
    def probe_bits(w: Int):
        while True:
            var m = Int(Atomic[DType.int64].fetch_add(ctr + 1, 1))
            if m >= np_m:
                break
            var lo = m * MORSEL
            var hi = min(lo + MORSEL, n_probe)
            var c = 0
            var dst = out + lo
            for i in range(lo, hi):
                var k = Int(probe[i])
                if k >= 0 and k < T:
                    var word = bits[k >> 6]
                    if (word >> UInt64(k & 63)) & UInt64(1) != 0:
                        dst[c] = Int32(i)
                        c += 1
            counts[m] = Int32(c)

    if w_build == 1:
        set_bits(0)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[set_bits](w_build)
    if w_probe == 1:
        probe_bits(0)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize[probe_bits](w_probe)

    # compact the per-morsel slots (each starts at m * MORSEL) into one ascending run
    var total = 0
    for m in range(np_m):
        var c = Int(counts[m])
        if c > 0 and total != m * MORSEL:
            memcpy(dest=out + total, src=out + m * MORSEL, count=c)
        total += c
    bits.free()
    ctr.free()
    counts.free()
    return total
