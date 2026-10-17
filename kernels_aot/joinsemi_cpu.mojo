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

F-074 adds a second entry point, `mx_join_key_codes`: dense int32 codes for the key tuples of both sides of a
join (strings compared by their bytes, integers by value), so that a string key joins on codes without
Arrow's `dictionary_encode`.  See its comment below.

Build:
    pixi run mojo build kernels_aot/joinsemi_cpu.mojo -I kernels_aot --target-cpu x86-64-v3 \
        --emit shared-lib -o kernels_aot/libmxjoin_cpu.so
    bash scripts/set_runpath.sh kernels_aot/libmxjoin_cpu.so
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from _fpenv import par, par_n
from std.atomic import Atomic
from std.ffi import external_call

comptime MORSEL = 1 << 16
comptime ABI_VERSION = 1

comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime UP = Pointer[UInt64, MutUntrackedOrigin]
comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime BP = Pointer[UInt8, MutUntrackedOrigin]


@always_inline
def _ip(addr: Int) -> IP:
    return IP(unsafe_from_address=addr)


@always_inline
def _up(addr: Int) -> UP:
    return UP(unsafe_from_address=addr)


@always_inline
def _lp(addr: Int) -> LP:
    return LP(unsafe_from_address=addr)


@always_inline
def _bp(addr: Int) -> BP:
    return BP(unsafe_from_address=addr)


@always_inline
def _fetch_or_u64(ptr: UP, val: UInt64):
    """Atomic OR (monotonic) via llvm.atomicrmw (bin_op 5 = or); same MLIR pattern as kernels_aot.mojo."""
    var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
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
    var bits = unsafe_alloc[UInt64](n_words)
    unsafe_memset_zero(bits, n_words)

    var nb_m = (n_build + MORSEL - 1) // MORSEL
    var np_m = (n_probe + MORSEL - 1) // MORSEL
    var w_build = max(1, min(threads, nb_m))
    var w_probe = max(1, min(threads, np_m))
    var ctr = unsafe_alloc[Int64](2)
    ctr[0] = 0
    ctr[1] = 0
    var counts = unsafe_alloc[Int32](np_m)

    def set_bits(w: Int) {imm}:
        while True:
            var m = Int(Atomic[Scalar[DType.int64]].fetch_add(ctr, 1))
            if m >= nb_m:
                break
            var lo = m * MORSEL
            var hi = min(lo + MORSEL, n_build)
            for i in range(lo, hi):
                var k = Int(build[i])
                if k >= 0 and k < T:
                    _fetch_or_u64(bits + (k >> 6), UInt64(1) << UInt64(k & 63))

    def probe_bits(w: Int) {imm}:
        while True:
            var m = Int(Atomic[Scalar[DType.int64]].fetch_add(ctr + 1, 1))
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
        par(set_bits, w_build)
    if w_probe == 1:
        probe_bits(0)
    else:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        par(probe_bits, w_probe)

    # compact the per-morsel slots (each starts at m * MORSEL) into one ascending run
    var total = 0
    for m in range(np_m):
        var c = Int(counts[m])
        if c > 0 and total != m * MORSEL:
            for i in range(c):                      # dest <= src and the ranges may overlap: copy forward
                out[total + i] = out[m * MORSEL + i]
        total += c
    bits.free()
    ctr.free()
    counts.free()
    return total


# ── F-074: dense codes for join keys (strings, integers, composites) ─────────────────────────────
#
#     mx_join_key_codes(plan, out) -> number of distinct non-null key tuples, or a negative error
#
# Plan (int64 words): 0 KMAGIC  1 n_keys  2 n_left  3 n_right  4 threads  5 descs*  6 codes* (int32[n_left + n_right],
# written by this call).  descs holds, for key j and side s (0 left, 1 right), 8 words at (2 j + s) * 8:
#   0 kind: 0 int32, 1 int64, 2 utf8 (int32 offsets), 3 large utf8 (int64 offsets), 4 dictionary of utf8
#           (int32 indices, dictionary values with int32 offsets and no NULL)
#   1 values / offsets / dictionary indices address     2 string bytes (kind 2, 3) or dictionary offsets (4)
#   3 dictionary bytes (4)    4 the array's offset (elements)    5 validity bitmap address, 0: no NULLs
#   6 the dictionary's own offset (4)    7 unused
# Both sides' columns of one key are of the same class (integer, or string of any kind). Row g < n_left is left
# row g, any other is right row g - n_left.  Rows with a NULL in any key get code -1 (the caller gives them
# side-specific codes that never match); equal tuples (an integer by value, a string by its bytes) get equal
# codes in 0 .. result - 1.  Rows are hashed in parallel, scattered to 64 partitions by the hash's top bits in
# row order, and each partition builds its own open-addressing table in parallel; partition code ranges are
# stacked by a prefix sum.

comptime KMAGIC = 0x4D584B43
comptime E_BAD_PLAN = -2
comptime HP = 64
comptime HP_SHIFT = 58


@always_inline
def _mix(h: UInt64) -> UInt64:
    """splitmix64 finalizer."""
    var z = h
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def _hash_bytes(a: Int, n: Int) -> UInt64:
    """Hash of n bytes at a: 8 bytes per step (unaligned loads), the tail and the length last."""
    var h = UInt64(0xCBF29CE484222325)
    var k = 0
    while k + 8 <= n:
        h = _mix(h ^ UInt64(Int(_lp(a + k).unsafe_load[alignment=1](0))))
        k += 8
    var tail = UInt64(0)
    var p = _bp(a)
    for i in range(k, n):
        tail = (tail << 8) | UInt64(p[i])
    return _mix(h ^ tail ^ (UInt64(n) << 56))


@always_inline
def _bytes_eq(a: Int, b: Int, n: Int) -> Bool:
    """n bytes at a equal n bytes at b (8 at a time)."""
    var k = 0
    while k + 8 <= n:
        if _lp(a + k).unsafe_load[alignment=1](0) != _lp(b + k).unsafe_load[alignment=1](0):
            return False
        k += 8
    var pa = _bp(a)
    var pb = _bp(b)
    for i in range(k, n):
        if pa[i] != pb[i]:
            return False
    return True


@always_inline
def _is_valid(d: LP, i: Int) -> Bool:
    var v = Int(d[5])
    if v == 0:
        return True
    var b = Int(d[4]) + i
    return ((Int(_bp(v)[b >> 3]) >> (b & 7)) & 1) == 1


@always_inline
def _int_at(d: LP, i: Int) -> Int:
    var off = Int(d[4]) + i
    if Int(d[0]) == 0:
        return Int(_ip(Int(d[1]))[off])
    return Int(_lp(Int(d[1]))[off])


@always_inline
def _s_start(d: LP, i: Int) -> Int:
    var kind = Int(d[0])
    var off = Int(d[4]) + i
    if kind == 2:
        return Int(d[2]) + Int(_ip(Int(d[1]))[off])
    if kind == 3:
        return Int(d[2]) + Int(_lp(Int(d[1]))[off])
    var c = Int(_ip(Int(d[1]))[off])
    return Int(d[3]) + Int(_ip(Int(d[2]))[Int(d[6]) + c])


@always_inline
def _s_len(d: LP, i: Int) -> Int:
    var kind = Int(d[0])
    var off = Int(d[4]) + i
    if kind == 2:
        var o = _ip(Int(d[1]))
        return Int(o[off + 1]) - Int(o[off])
    if kind == 3:
        var o = _lp(Int(d[1]))
        return Int(o[off + 1]) - Int(o[off])
    var c = Int(_ip(Int(d[1]))[off])
    var o = _ip(Int(d[2])) + Int(d[6])
    return Int(o[c + 1]) - Int(o[c])


@always_inline
def _part_hash(d: LP, i: Int) -> UInt64:
    if Int(d[0]) <= 1:
        return _mix(UInt64(_int_at(d, i)))
    return _hash_bytes(_s_start(d, i), _s_len(d, i))


def _tuple_eq(descs: LP, nk: Int, nl: Int, a: Int, b: Int) -> Bool:
    var sa = 0 if a < nl else 1
    var ia = a if a < nl else a - nl
    var sb = 0 if b < nl else 1
    var ib = b if b < nl else b - nl
    for j in range(nk):
        var da = descs + (2 * j + sa) * 8
        var db = descs + (2 * j + sb) * 8
        if Int(da[0]) <= 1:
            if _int_at(da, ia) != _int_at(db, ib):
                return False
        else:
            var la = _s_len(da, ia)
            if la != _s_len(db, ib) or not _bytes_eq(_s_start(da, ia), _s_start(db, ib), la):
                return False
    return True


def _par_k[F: def(Int) -> None](f: F, nt: Int):
    """f(0) .. f(nt - 1) on nt workers (each pulls blocks from an atomic counter)."""
    if nt > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        par(f, nt)
    else:
        f(0)


@export
def mx_join_key_codes(plan_a: Int, out_a: Int) abi("C") -> Int:
    var P = _lp(plan_a)
    if Int(P[0]) != KMAGIC:
        return E_BAD_PLAN
    var nk = Int(P[1])
    var nl = Int(P[2])
    var n = nl + Int(P[3])
    var threads = max(1, Int(P[4]))
    var descs = _lp(Int(P[5]))
    var out = _ip(Int(P[6]))
    if nk < 1 or nl < 0 or n < 1 or n >= 2147483647:
        return E_BAD_PLAN
    var nb = max(1, min(threads * 4, (n + 65535) // 65536))
    var nt = max(1, min(threads, nb))
    var hs = unsafe_alloc[UInt64](n)
    var rows = unsafe_alloc[Int32](n)
    var cnt = unsafe_alloc[Int64](nb * HP)
    var poff = unsafe_alloc[Int64](HP + 1)
    var ucount = unsafe_alloc[Int64](HP)
    var counter = unsafe_alloc[Int64](1)
    counter[0] = 0

    # 1. hash every row (NULL rows are marked -1 and left out), count rows per (block, partition)
    def hash_rows(wid: Int) {imm}:
        while True:
            var b = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            if b >= nb:
                break
            for p in range(HP):
                cnt[b * HP + p] = 0
            for g in range(b * n // nb, (b + 1) * n // nb):
                var side = 0 if g < nl else 1
                var i = g if g < nl else g - nl
                var h = UInt64(0x9E3779B97F4A7C15)
                var ok = True
                for j in range(nk):
                    var d = descs + (2 * j + side) * 8
                    if not _is_valid(d, i):
                        ok = False
                        break
                    h = _mix(h ^ _part_hash(d, i))
                if ok:
                    hs[g] = h
                    out[g] = 0
                    cnt[b * HP + Int(h >> HP_SHIFT)] += 1
                else:
                    out[g] = -1

    _par_k(hash_rows, nt)
    # 2. partition-major offsets, block order inside a partition (so row order is kept), then scatter
    var run = 0
    for p in range(HP):
        poff[p] = Int64(run)
        for b in range(nb):
            var k = Int(cnt[b * HP + p])
            cnt[b * HP + p] = Int64(run)
            run += k
    poff[HP] = Int64(run)
    counter[0] = 0

    def scatter(wid: Int) {imm}:
        while True:
            var b = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            if b >= nb:
                break
            var at = unsafe_alloc[Int64](HP)
            for p in range(HP):
                at[p] = cnt[b * HP + p]
            for g in range(b * n // nb, (b + 1) * n // nb):
                if out[g] < 0:
                    continue
                var p = Int(hs[g] >> HP_SHIFT)
                rows[Int(at[p])] = Int32(g)
                at[p] += 1
            at.free()

    _par_k(scatter, nt)
    # 3. per partition, in row order: an equal tuple seen before gives its code, else the next local one
    counter[0] = 0

    def build(wid: Int) {imm}:
        while True:
            var p = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            if p >= HP:
                break
            var lo = Int(poff[p])
            var hi = Int(poff[p + 1])
            ucount[p] = 0
            if hi == lo:
                continue
            var cap = 16
            while cap < 2 * (hi - lo) + 2:
                cap *= 2
            var tab = unsafe_alloc[Int32](cap)
            for s in range(cap):
                tab[s] = -1
            var u = 0
            for k in range(lo, hi):
                var r = Int(rows[k])
                var h = hs[r]
                var s = Int(h & UInt64(cap - 1))
                while True:
                    var t = Int(tab[s])
                    if t < 0:
                        tab[s] = Int32(r)
                        out[r] = Int32(u)
                        u += 1
                        break
                    if hs[t] == h and _tuple_eq(descs, nk, nl, t, r):
                        out[r] = out[t]
                        break
                    s = (s + 1) & (cap - 1)
            ucount[p] = Int64(u)
            tab.free()

    _par_k(build, min(threads, HP))
    # 4. stack the partitions' code ranges
    var total = 0
    for p in range(HP):
        var u = Int(ucount[p])
        ucount[p] = Int64(total)
        total += u
    counter[0] = 0

    def rebase(wid: Int) {imm}:
        while True:
            var p = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            if p >= HP:
                break
            var base = Int32(ucount[p])
            for k in range(Int(poff[p]), Int(poff[p + 1])):
                var r = Int(rows[k])
                out[r] = out[r] + base

    _par_k(rebase, min(threads, HP))
    hs.free()
    rows.free()
    cnt.free()
    poff.free()
    ucount.free()
    counter.free()
    return total
