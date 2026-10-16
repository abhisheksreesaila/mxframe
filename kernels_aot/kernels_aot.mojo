"""
MXFrame AOT CPU kernels — compiled once to libmxkernels_aot.so at build time.

All kernels accept raw Int addresses (pointer-as-int) and cast to
Pointer[T, MutUntrackedOrigin] inside. This satisfies @export's C-ABI requirement
while giving fully typed pointer access. N is always a runtime parameter so one
compiled binary handles any row count — no MAX Graph JIT cold starts.

Multi-core (F-002)
------------------
Every hot kernel takes a trailing `threads` argument: the maximum number of
worker tasks it may use.  The Python wrapper always passes >= 1 (default: every
logical core the process may use); a value <= 0 falls back to the physical core count.  Work is split into
morsels of MORSEL rows that workers claim from one atomic counter (morsel-driven,
so a worker that loses its core to another process simply takes fewer morsels) and
run with `sync_parallelize`; small inputs (< MIN_ROWS per worker) run on one worker.  No global state:
several Python threads (ctypes releases the GIL) can call kernels at once, each
with its own `threads` budget, so a co-processing scheduler can give some cores
to feeding the GPU and the rest to CPU work.

Accumulation is float64 for sum/mean (values may be float32 or float64), so a
60M-row float32 sum no longer drifts.  Which worker sums which morsel varies from
run to run, so float64 sums/means can differ in the last ulps between runs; counts,
min/max, integer sums and all join/gather/composite outputs are exact and do not
depend on `threads`.

Grouped aggregates pick one of two strategies:
  * per-worker partials — every worker aggregates its rows into a private
                          (padded) array, merged in parallel by group range.
                          Workers are capped so workers * n_groups <= PARTIAL_ELEMS
  * serial              — one worker, tiny input, or n_groups > PARTIAL_ELEMS / 2
Measured on a 12-core Ryzen 5900X (60M rows, 1.5M random groups) radix
partitioning, SIMD owner-scan, per-worker partials and f64 atomics were all
no faster than the serial loop (28 GB/s DRAM ceiling; the 12 MB accumulator
array is latency-bound), so that regime stays serial rather than pay for
extra memory traffic.

Build:
    mojo build -I kernels_aot --emit shared-lib kernels_aot/kernels_aot.mojo -o kernels_aot/libmxkernels_aot.so
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer, unsafe_memset_zero
from _fpenv import sync_par
from std.sys.info import num_physical_cores
from std.ffi import external_call

comptime F32 = Float32
comptime F64 = Float64
comptime I32 = Int32
comptime I64 = Int64

# ABI version of the exported symbol set; aot_kernels.py refuses a stale .so.
comptime ABI_VERSION = 3

comptime MIN_ROWS = 1 << 16          # rows per worker below which we stay serial
comptime MIN_ROWS_STREAM = 1 << 17   # same, for pure streaming reductions
comptime MAX_WORKERS = 256
comptime SW = 8                      # SIMD lanes (AVX2-friendly)
comptime PARTIAL_ELEMS = 3 << 20     # cap on workers*groups for private partials (~25 MB of f64)

comptime OP_SUM = 0
comptime OP_MIN = 1
comptime OP_MAX = 2
comptime OP_COUNT = 3
comptime OP_MEAN = 4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: typed pointers from raw Int addresses
# ─────────────────────────────────────────────────────────────────────────────

@always_inline
def _f32(addr: Int) -> Pointer[F32, MutUntrackedOrigin]:
    return Pointer[F32, MutUntrackedOrigin](unsafe_from_address=addr)

@always_inline
def _f64(addr: Int) -> Pointer[F64, MutUntrackedOrigin]:
    return Pointer[F64, MutUntrackedOrigin](unsafe_from_address=addr)

@always_inline
def _i32(addr: Int) -> Pointer[I32, MutUntrackedOrigin]:
    return Pointer[I32, MutUntrackedOrigin](unsafe_from_address=addr)

@always_inline
def _i64(addr: Int) -> Pointer[I64, MutUntrackedOrigin]:
    return Pointer[I64, MutUntrackedOrigin](unsafe_from_address=addr)

@always_inline
def _sp[T: DType](addr: Int) -> Pointer[Scalar[T], MutUntrackedOrigin]:
    return Pointer[Scalar[T], MutUntrackedOrigin](unsafe_from_address=addr)


comptime MORSEL = 1 << 17            # rows per dynamically claimed morsel


@always_inline
def _fetch_add_i32(ptr: Pointer[I32, MutUntrackedOrigin], val: I32) -> I32:
    """Atomic add (monotonic) via llvm.atomicrmw; same MLIR pattern as _atomic.mojo."""
    var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](val)
    var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`1 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i32](p, v)
    return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[I32]](r)


@always_inline
def _claim(ctr: Pointer[I32, MutUntrackedOrigin]) -> Int:
    """Next unclaimed morsel index (work-queue pop)."""
    return Int(_fetch_add_i32(ctr, I32(1)))


def _new_queue() -> Pointer[I32, MutUntrackedOrigin]:
    var ctr = unsafe_alloc[I32](1)
    ctr[0] = I32(0)
    return ctr


def _n_morsels(n: Int) -> Int:
    return (n + MORSEL - 1) // MORSEL


def _pick_workers(n: Int, threads: Int, min_rows: Int) -> Int:
    """Number of parallel tasks for n rows: <= threads, >= 1, >= min_rows each."""
    var t = threads
    if t <= 0:
        t = num_physical_cores()
    if t > MAX_WORKERS:
        t = MAX_WORKERS
    var cap = n // min_rows
    if cap < 1:
        cap = 1
    if t > cap:
        t = cap
    return t


@export
def mx_init_runtime() abi("C") -> Int:
    """Pin the Mojo async runtime (worker pool) for the life of the process.

    Without a live reference, every sync_parallelize call builds and tears down
    the whole pool (~1.7 ms/call); with one, a launch costs ~20 us.  Idempotent:
    the reference is taken once and never released."""
    return external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


@export
def mx_abi_version() abi("C") -> Int:
    return ABI_VERSION


@export
def mx_num_physical_cores() abi("C") -> Int:
    return num_physical_cores()


# ─────────────────────────────────────────────────────────────────────────────
# Grouped aggregates:  sum / min / max / mean / count
#   out[labels[i]] (op)= values[i]      labels outside [0, n_groups) are skipped
# Accumulator type AT is float64 for float inputs (int64 for the int64 sum).
# ─────────────────────────────────────────────────────────────────────────────

@always_inline
def _ident[OP: Int, VT: DType, AT: DType]() -> Scalar[AT]:
    """Initial accumulator value. min/max keep the historical float32 sentinel."""
    comptime if OP == OP_MIN:
        comptime if AT == DType.int64:
            return Scalar[AT].MAX
        elif VT == DType.float32:
            return Scalar[AT](3.4028234663852886e+38)
        else:
            return Scalar[AT](1.7976931348623157e+308)
    elif OP == OP_MAX:
        comptime if AT == DType.int64:
            return Scalar[AT].MIN
        elif VT == DType.float32:
            return Scalar[AT](-3.4028234663852886e+38)
        else:
            return Scalar[AT](-1.7976931348623157e+308)
    else:
        return Scalar[AT](0)


@always_inline
def _val[OP: Int, VT: DType](values: Pointer[Scalar[VT], MutUntrackedOrigin], i: Int) -> Scalar[VT]:
    comptime if OP == OP_COUNT:
        return Scalar[VT](0)
    else:
        return values[i]


@always_inline
def _step[OP: Int, VT: DType, AT: DType](
    acc: Pointer[Scalar[AT], MutUntrackedOrigin],
    g: Int,
    v: Scalar[VT],
):
    """One row.  Mean keeps (sum, count) interleaved at acc[2g], acc[2g+1] so a
    row touches a single cache line."""
    comptime if OP == OP_SUM:
        acc[g] += v.cast[AT]()
    elif OP == OP_MIN:
        var x = v.cast[AT]()
        if x < acc[g]:
            acc[g] = x
    elif OP == OP_MAX:
        var x = v.cast[AT]()
        if x > acc[g]:
            acc[g] = x
    elif OP == OP_COUNT:
        acc[g] += Scalar[AT](1)
    else:  # mean
        acc[2 * g] += v.cast[AT]()
        acc[2 * g + 1] += Scalar[AT](1)


@always_inline
def _combine[OP: Int, AT: DType](a: Scalar[AT], b: Scalar[AT]) -> Scalar[AT]:
    comptime if OP == OP_MIN:
        return b if b < a else a
    elif OP == OP_MAX:
        return b if b > a else a
    else:
        return a + b


def _accumulate[OP: Int, VT: DType, AT: DType](
    acc: Pointer[Scalar[AT], MutUntrackedOrigin],
    values: Pointer[Scalar[VT], MutUntrackedOrigin],
    labels: Pointer[I32, MutUntrackedOrigin],
    lo: Int, hi: Int, G: Int,
):
    for i in range(lo, hi):
        var g = Int(labels[i])
        if g >= 0 and g < G:
            _step[OP, VT, AT](acc, g, _val[OP, VT](values, i))


def _group_agg[OP: Int, VT: DType, AT: DType](
    dst: Pointer[Scalar[AT], MutUntrackedOrigin],
    values: Pointer[Scalar[VT], MutUntrackedOrigin],
    labels: Pointer[I32, MutUntrackedOrigin],
    n_rows: Int,
    n_groups: Int,
    threads: Int,
):
    if n_groups <= 0:
        return
    var out = dst
    var G = n_groups
    comptime K = 2 if OP == OP_MEAN else 1      # accumulator slots per group
    var init = _ident[OP, VT, AT]()
    var W = _pick_workers(n_rows, threads, MIN_ROWS)
    if n_rows < 6 * G:
        W = 1                        # init + merge of W x G partials would cost as much as the scan
    elif K * G * W > PARTIAL_ELEMS:
        W = PARTIAL_ELEMS // (K * G)     # fewer workers so private partials stay cache-sized

    if W <= 1 or n_rows == 0:
        # ---- serial: accumulate straight into `out` (mean: into a pair array) ----
        comptime if OP == OP_MEAN:
            var tmp = unsafe_alloc[Scalar[AT]](2 * G)
            unsafe_memset_zero(tmp, 2 * G)
            _accumulate[OP, VT, AT](tmp, values, labels, 0, n_rows, G)
            for g in range(G):
                var c = tmp[2 * g + 1]
                out[g] = tmp[2 * g] / c if c > 0 else tmp[2 * g]
            tmp.free()
        else:
            for g in range(G):
                out[g] = init
            _accumulate[OP, VT, AT](out, values, labels, 0, n_rows, G)
    else:
        # ---- per-worker private partial arrays, merged by group range ----
        var stride = ((K * G + 15) // 16) * 16 + 16          # pad: no false sharing
        var part = unsafe_alloc[Scalar[AT]](W * stride)
        var n_m = _n_morsels(n_rows)

        def scan(w: Int) {imm}:
            var p = part + w * stride
            comptime if OP == OP_MEAN:
                unsafe_memset_zero(p, 2 * G)
            else:
                for g in range(G):
                    p[g] = init
            var m = w                                     # worker w: morsels w, w + W, ... (fixed, so float
            while m < n_m:                                # sums are bit-identical from run to run, F-064)
                var lo = m * MORSEL
                var hi = lo + MORSEL
                if hi > n_rows:
                    hi = n_rows
                _accumulate[OP, VT, AT](p, values, labels, lo, hi, G)
                m += W

        sync_par(scan, W)

        var MT = G // 4096
        if MT > W:
            MT = W
        if MT < 1:
            MT = 1

        def merge(t: Int) {imm}:
            var g0 = (t * G) // MT
            var g1 = ((t + 1) * G) // MT
            for g in range(g0, g1):
                comptime if OP == OP_MEAN:
                    var s = part[2 * g]
                    var c = part[2 * g + 1]
                    for w in range(1, W):
                        s += part[w * stride + 2 * g]
                        c += part[w * stride + 2 * g + 1]
                    out[g] = s / c if c > 0 else s
                else:
                    var a = part[g]
                    for w in range(1, W):
                        a = _combine[OP, AT](a, part[w * stride + g])
                    out[g] = a

        sync_par(merge, MT)
        part.free()


# ---- exports: out is float64 for sum/min/max/mean/count (int64 for sum_i64) ----

@export
def group_sum_f32(out_addr: Int, values_addr: Int, labels_addr: Int,
                  n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_SUM, DType.float32, DType.float64](
        _f64(out_addr), _f32(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_sum_f64(out_addr: Int, values_addr: Int, labels_addr: Int,
                  n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_SUM, DType.float64, DType.float64](
        _f64(out_addr), _f64(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_sum_i64(out_addr: Int, values_addr: Int, labels_addr: Int,
                  n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_SUM, DType.int64, DType.int64](
        _i64(out_addr), _i64(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_min_f32(out_addr: Int, values_addr: Int, labels_addr: Int,
                  n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_MIN, DType.float32, DType.float64](
        _f64(out_addr), _f32(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_min_f64(out_addr: Int, values_addr: Int, labels_addr: Int,
                  n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_MIN, DType.float64, DType.float64](
        _f64(out_addr), _f64(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_max_f32(out_addr: Int, values_addr: Int, labels_addr: Int,
                  n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_MAX, DType.float32, DType.float64](
        _f64(out_addr), _f32(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_max_f64(out_addr: Int, values_addr: Int, labels_addr: Int,
                  n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_MAX, DType.float64, DType.float64](
        _f64(out_addr), _f64(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_mean_f32(out_addr: Int, values_addr: Int, labels_addr: Int,
                   n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_MEAN, DType.float32, DType.float64](
        _f64(out_addr), _f32(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_mean_f64(out_addr: Int, values_addr: Int, labels_addr: Int,
                   n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    _group_agg[OP_MEAN, DType.float64, DType.float64](
        _f64(out_addr), _f64(values_addr), _i32(labels_addr), n_rows, n_groups, threads)

@export
def group_count(out_addr: Int, labels_addr: Int,
                n_rows: Int, n_groups: Int, threads: Int) abi("C"):
    """Float64 out[g] = number of rows with labels == g."""
    _group_agg[OP_COUNT, DType.float32, DType.float64](
        _f64(out_addr), _f32(0), _i32(labels_addr), n_rows, n_groups, threads)


# ─────────────────────────────────────────────────────────────────────────────
# group_composite
# out[i] = k0[i]*s0 + k1[i]*s1 + k2[i]*s2 + k3[i]*s3  (int64)
# ─────────────────────────────────────────────────────────────────────────────
@export
def group_composite(
    out_addr:     Int,
    k0_addr:      Int,
    k1_addr:      Int,
    k2_addr:      Int,
    k3_addr:      Int,
    strides_addr: Int,   # int64[4]: s0,s1,s2,s3
    n_rows:       Int,
    threads:      Int,
) abi("C"):
    var out     = _i64(out_addr)
    var k0      = _i32(k0_addr)
    var k1      = _i32(k1_addr)
    var k2      = _i32(k2_addr)
    var k3      = _i32(k3_addr)
    var strides = _i64(strides_addr)
    var s0 = strides[0]
    var s1 = strides[1]
    var s2 = strides[2]
    var s3 = strides[3]
    var W = _pick_workers(n_rows, threads, MIN_ROWS_STREAM)
    var ctr = _new_queue()
    var n_m = _n_morsels(n_rows)

    def work(w: Int) {imm}:
        while True:
            var m = _claim(ctr)
            if m >= n_m:
                break
            var lo = m * MORSEL
            var hi = lo + MORSEL
            if hi > n_rows:
                hi = n_rows
            for i in range(lo, hi):
                out[i] = I64(k0[i])*s0 + I64(k1[i])*s1 + I64(k2[i])*s2 + I64(k3[i])*s3

    sync_par(work, W)
    ctr.free()


# ─────────────────────────────────────────────────────────────────────────────
# Masked global aggregates (pure streaming reductions, SIMD + multi-core)
#   sum / sum_product accumulate in float64; min / max are exact.
#   `out` is a single float64.  mask is int32, non-zero = row passes.
# ─────────────────────────────────────────────────────────────────────────────



@always_inline
def _big[VT: DType]() -> Scalar[VT]:
    comptime if VT == DType.float32:
        return Scalar[VT](3.4028234663852886e+38)
    else:
        return Scalar[VT](1.7976931348623157e+308)


def _msum_range[VT: DType](
    values: Pointer[Scalar[VT], MutUntrackedOrigin],
    mask: Pointer[I32, MutUntrackedOrigin], lo: Int, hi: Int,
) -> F64:
    var zero = SIMD[DType.float64, SW](0)
    var a0 = zero
    var a1 = zero
    var i = lo
    while i + 2 * SW <= hi:
        var m0 = mask.load[width=SW](i)
        var m1 = mask.load[width=SW](i + SW)
        var v0 = values.load[width=SW](i).cast[DType.float64]()
        var v1 = values.load[width=SW](i + SW).cast[DType.float64]()
        a0 += m0.ne(0).select(v0, zero)
        a1 += m1.ne(0).select(v1, zero)
        i += 2 * SW
    var acc = (a0 + a1).reduce_add()
    while i < hi:
        if mask[i] != I32(0):
            acc += values[i].cast[DType.float64]()
        i += 1
    return acc


def _mprod_range[VT: DType](
    a: Pointer[Scalar[VT], MutUntrackedOrigin],
    b: Pointer[Scalar[VT], MutUntrackedOrigin],
    mask: Pointer[I32, MutUntrackedOrigin], lo: Int, hi: Int,
) -> F64:
    var zero = SIMD[DType.float64, SW](0)
    var a0 = zero
    var a1 = zero
    var i = lo
    while i + 2 * SW <= hi:
        var m0 = mask.load[width=SW](i)
        var m1 = mask.load[width=SW](i + SW)
        var p0 = a.load[width=SW](i).cast[DType.float64]() * b.load[width=SW](i).cast[DType.float64]()
        var p1 = a.load[width=SW](i + SW).cast[DType.float64]() * b.load[width=SW](i + SW).cast[DType.float64]()
        a0 += m0.ne(0).select(p0, zero)
        a1 += m1.ne(0).select(p1, zero)
        i += 2 * SW
    var acc = (a0 + a1).reduce_add()
    while i < hi:
        if mask[i] != I32(0):
            acc += a[i].cast[DType.float64]() * b[i].cast[DType.float64]()
        i += 1
    return acc


def _mext_range[VT: DType, IS_MIN: Bool](
    values: Pointer[Scalar[VT], MutUntrackedOrigin],
    mask: Pointer[I32, MutUntrackedOrigin], lo: Int, hi: Int,
) -> Scalar[VT]:
    """Masked min (IS_MIN) or max; NaN values are ignored like the scalar loop."""
    var ident: Scalar[VT]
    comptime if IS_MIN:
        ident = _big[VT]()
    else:
        ident = -_big[VT]()
    var acc = SIMD[VT, SW](ident)
    var i = lo
    while i + SW <= hi:
        var m = mask.load[width=SW](i).ne(0)
        var v = m.select(values.load[width=SW](i), SIMD[VT, SW](ident))
        comptime if IS_MIN:
            acc = v.lt(acc).select(v, acc)
        else:
            acc = v.gt(acc).select(v, acc)
        i += SW
    var r = ident
    for k in range(SW):
        var x = acc[k]
        comptime if IS_MIN:
            if x < r:
                r = x
        else:
            if x > r:
                r = x
    while i < hi:
        if mask[i] != I32(0):
            var x = values[i]
            comptime if IS_MIN:
                if x < r:
                    r = x
            else:
                if x > r:
                    r = x
        i += 1
    return r


def _masked_sum[VT: DType](dst: Pointer[F64, MutUntrackedOrigin],
                           values: Pointer[Scalar[VT], MutUntrackedOrigin],
                           mask: Pointer[I32, MutUntrackedOrigin], n: Int, threads: Int):
    var out = dst
    var W = _pick_workers(n, threads, MIN_ROWS_STREAM)
    var parts = unsafe_alloc[F64](W)
    var n_m = _n_morsels(n)

    def work(w: Int) {imm}:
        var acc: F64 = 0.0
        var m = w                       # worker w: morsels w, w + W, ... (fixed order, F-064)
        while m < n_m:
            var lo = m * MORSEL
            acc += _msum_range[VT](values, mask, lo, lo + MORSEL if lo + MORSEL < n else n)
            m += W
        parts[w] = acc

    sync_par(work, W)
    var total: F64 = 0.0
    for w in range(W):
        total += parts[w]
    out[0] = total
    parts.free()


def _masked_sum_product[VT: DType](dst: Pointer[F64, MutUntrackedOrigin],
                                   a: Pointer[Scalar[VT], MutUntrackedOrigin],
                                   b: Pointer[Scalar[VT], MutUntrackedOrigin],
                                   mask: Pointer[I32, MutUntrackedOrigin], n: Int, threads: Int):
    var out = dst
    var W = _pick_workers(n, threads, MIN_ROWS_STREAM)
    var parts = unsafe_alloc[F64](W)
    var n_m = _n_morsels(n)

    def work(w: Int) {imm}:
        var acc: F64 = 0.0
        var m = w                       # worker w: morsels w, w + W, ... (fixed order, F-064)
        while m < n_m:
            var lo = m * MORSEL
            acc += _mprod_range[VT](a, b, mask, lo, lo + MORSEL if lo + MORSEL < n else n)
            m += W
        parts[w] = acc

    sync_par(work, W)
    var total: F64 = 0.0
    for w in range(W):
        total += parts[w]
    out[0] = total
    parts.free()


def _masked_ext[VT: DType, IS_MIN: Bool](dst: Pointer[F64, MutUntrackedOrigin],
                                         values: Pointer[Scalar[VT], MutUntrackedOrigin],
                                         mask: Pointer[I32, MutUntrackedOrigin], n: Int, threads: Int):
    var out = dst
    var W = _pick_workers(n, threads, MIN_ROWS_STREAM)
    var parts = unsafe_alloc[Scalar[VT]](W)
    var ctr = _new_queue()
    var n_m = _n_morsels(n)
    var ident: Scalar[VT]
    comptime if IS_MIN:
        ident = _big[VT]()
    else:
        ident = -_big[VT]()

    def work(w: Int) {imm}:
        var best = ident
        while True:
            var m = _claim(ctr)
            if m >= n_m:
                break
            var lo = m * MORSEL
            var x = _mext_range[VT, IS_MIN](values, mask, lo, lo + MORSEL if lo + MORSEL < n else n)
            comptime if IS_MIN:
                if x < best:
                    best = x
            else:
                if x > best:
                    best = x
        parts[w] = best

    sync_par(work, W)
    ctr.free()
    var r = parts[0]
    for w in range(1, W):
        comptime if IS_MIN:
            if parts[w] < r:
                r = parts[w]
        else:
            if parts[w] > r:
                r = parts[w]
    out[0] = r.cast[DType.float64]()
    parts.free()


@export
def masked_global_sum_f32(out_addr: Int, values_addr: Int, mask_addr: Int,
                          n: Int, threads: Int) abi("C"):
    _masked_sum[DType.float32](_f64(out_addr), _f32(values_addr), _i32(mask_addr), n, threads)

@export
def masked_global_sum_f64(out_addr: Int, values_addr: Int, mask_addr: Int,
                          n: Int, threads: Int) abi("C"):
    _masked_sum[DType.float64](_f64(out_addr), _f64(values_addr), _i32(mask_addr), n, threads)

@export
def masked_global_min_f32(out_addr: Int, values_addr: Int, mask_addr: Int,
                          n: Int, threads: Int) abi("C"):
    _masked_ext[DType.float32, True](_f64(out_addr), _f32(values_addr), _i32(mask_addr), n, threads)

@export
def masked_global_min_f64(out_addr: Int, values_addr: Int, mask_addr: Int,
                          n: Int, threads: Int) abi("C"):
    _masked_ext[DType.float64, True](_f64(out_addr), _f64(values_addr), _i32(mask_addr), n, threads)

@export
def masked_global_max_f32(out_addr: Int, values_addr: Int, mask_addr: Int,
                          n: Int, threads: Int) abi("C"):
    _masked_ext[DType.float32, False](_f64(out_addr), _f32(values_addr), _i32(mask_addr), n, threads)

@export
def masked_global_max_f64(out_addr: Int, values_addr: Int, mask_addr: Int,
                          n: Int, threads: Int) abi("C"):
    _masked_ext[DType.float64, False](_f64(out_addr), _f64(values_addr), _i32(mask_addr), n, threads)

@export
def masked_global_sum_product_f32(out_addr: Int, a_addr: Int, b_addr: Int, mask_addr: Int,
                                  n: Int, threads: Int) abi("C"):
    _masked_sum_product[DType.float32](_f64(out_addr), _f32(a_addr), _f32(b_addr), _i32(mask_addr), n, threads)

@export
def masked_global_sum_product_f64(out_addr: Int, a_addr: Int, b_addr: Int, mask_addr: Int,
                                  n: Int, threads: Int) abi("C"):
    _masked_sum_product[DType.float64](_f64(out_addr), _f64(a_addr), _f64(b_addr), _i32(mask_addr), n, threads)


# ─────────────────────────────────────────────────────────────────────────────
# gather_{f32,i32,i64}   output[i] = values[indices[i]]   (parallel over output)
# ─────────────────────────────────────────────────────────────────────────────

def _gather[T: DType](out_addr: Int, values_addr: Int, indices_addr: Int, m: Int, threads: Int):
    var out = _sp[T](out_addr)
    var values = _sp[T](values_addr)
    var indices = _i32(indices_addr)
    var W = _pick_workers(m, threads, MIN_ROWS)
    var ctr = _new_queue()
    var n_m = _n_morsels(m)

    def work(w: Int) {imm}:
        while True:
            var q = _claim(ctr)
            if q >= n_m:
                break
            var lo = q * MORSEL
            var hi = lo + MORSEL
            if hi > m:
                hi = m
            for i in range(lo, hi):
                out[i] = values[Int(indices[i])]

    sync_par(work, W)
    ctr.free()


@export
def gather_f32(out_addr: Int, values_addr: Int, indices_addr: Int, m: Int, threads: Int) abi("C"):
    _gather[DType.float32](out_addr, values_addr, indices_addr, m, threads)

@export
def gather_i32(out_addr: Int, values_addr: Int, indices_addr: Int, m: Int, threads: Int) abi("C"):
    _gather[DType.int32](out_addr, values_addr, indices_addr, m, threads)

@export
def gather_i64(out_addr: Int, values_addr: Int, indices_addr: Int, m: Int, threads: Int) abi("C"):
    _gather[DType.int64](out_addr, values_addr, indices_addr, m, threads)


# ─────────────────────────────────────────────────────────────────────────────
# sort_indices{,_i64,_f32,_f64}
# out[i] = original row index at sorted position i.  STABLE (equal keys keep row
# order, ascending and descending), O(n) per radix pass, multi-core.
#
# Parallel LSD radix sort on 8-bit digits over (key, row) pairs:
#   1. keys are mapped to unsigned integers that sort like the values (sign bit
#      flipped for ints, the usual flip for floats; -0.0 == +0.0 and NaN sorts
#      last in both directions); descending complements the mapped key, which
#      still leaves equal keys in row order
#   2. per pass every worker histograms its contiguous chunk of the current order,
#      a tiny serial scan turns (digit, worker) counts into start offsets, and
#      every worker scatters its chunk in order, which is what keeps the sort
#      stable; a pass in which one digit holds every row is skipped, so keys
#      with a small range cost fewer passes
# The previous version was an in-place rotate merge sort: O(n^2), it hung at 1e4
# random rows.
# ─────────────────────────────────────────────────────────────────────────────
comptime RADIX = 256

comptime KIND_I32 = 0
comptime KIND_I64 = 1
comptime KIND_F32 = 2
comptime KIND_F64 = 3


@always_inline
def _to_key[KIND: Int, UT: DType](raw: Scalar[UT], desc: Bool) -> Scalar[UT]:
    """Order-preserving unsigned key for the bits of one value."""
    comptime if UT == DType.uint32:
        comptime SIGN = Scalar[UT](0x80000000)
        comptime ALL = ~Scalar[UT](0)
        comptime if KIND == KIND_I32:
            var k = raw ^ SIGN
            return ~k if desc else k
        else:
            if (raw & Scalar[UT](0x7FFFFFFF)) > Scalar[UT](0x7F800000):
                return ALL                                  # NaN last, either direction
            var b = raw
            if b == SIGN:
                b = Scalar[UT](0)                            # -0.0 == +0.0
            var k = ~b if (b & SIGN) != Scalar[UT](0) else (b | SIGN)
            return ~k if desc else k
    else:
        comptime SIGN = Scalar[UT](1) << Scalar[UT](63)
        comptime ALL = ~Scalar[UT](0)
        comptime if KIND == KIND_I64:
            var k = raw ^ SIGN
            return ~k if desc else k
        else:
            if (raw & ~SIGN) > (Scalar[UT](0x7FF) << Scalar[UT](52)):
                return ALL
            var b = raw
            if b == SIGN:
                b = Scalar[UT](0)
            var k = ~b if (b & SIGN) != Scalar[UT](0) else (b | SIGN)
            return ~k if desc else k


def _radix_pass[UT: DType](
    src_k: Pointer[Scalar[UT], MutUntrackedOrigin],
    dst_k: Pointer[Scalar[UT], MutUntrackedOrigin],
    src_i: Pointer[I32, MutUntrackedOrigin],
    dst_i: Pointer[I32, MutUntrackedOrigin],
    hist: Pointer[I32, MutUntrackedOrigin],
    n: Int, W: Int, shift: Int,
) -> Bool:
    """One stable counting-sort pass on byte `shift`; False if it was skipped."""
    def count(w: Int) {imm}:
        var lo = (n * w) // W
        var hi = (n * (w + 1)) // W
        var h = hist + w * RADIX
        for d in range(RADIX):
            h[d] = I32(0)
        for i in range(lo, hi):
            var d = Int((src_k[i] >> Scalar[UT](shift)) & Scalar[UT](0xFF))
            h[d] += I32(1)

    sync_par(count, W)

    # digit totals: a digit holding every row makes the pass a no-op
    var run = 0
    for d in range(RADIX):
        var total = 0
        for w in range(W):
            total += Int(hist[w * RADIX + d])
        if total == n:
            return False
        for w in range(W):                      # (digit, worker) -> start offset
            var c = Int(hist[w * RADIX + d])
            hist[w * RADIX + d] = I32(run)
            run += c

    def scatter(w: Int) {imm}:
        var lo = (n * w) // W
        var hi = (n * (w + 1)) // W
        var h = hist + w * RADIX
        for i in range(lo, hi):
            var k = src_k[i]
            var d = Int((k >> Scalar[UT](shift)) & Scalar[UT](0xFF))
            var p = Int(h[d])
            h[d] = I32(p + 1)
            dst_k[p] = k
            dst_i[p] = src_i[i]

    sync_par(scatter, W)
    return True


def _radix_sort[UT: DType](
    out_idx: Pointer[I32, MutUntrackedOrigin],
    keys: Pointer[Scalar[UT], MutUntrackedOrigin],
    n: Int, threads: Int,
):
    """Stable sort of the mapped `keys` (destroyed); out_idx gets the row order."""
    comptime NBYTES = 4 if UT == DType.uint32 else 8
    var W = _pick_workers(n, threads, MIN_ROWS)
    var k_a: Pointer[Scalar[UT], MutUntrackedOrigin] = keys
    var k_b: Pointer[Scalar[UT], MutUntrackedOrigin] = unsafe_alloc[Scalar[UT]](n)
    var i_a: Pointer[I32, MutUntrackedOrigin] = unsafe_alloc[I32](n)
    var i_b: Pointer[I32, MutUntrackedOrigin] = out_idx
    var hist: Pointer[I32, MutUntrackedOrigin] = unsafe_alloc[I32](W * RADIX)

    def init(w: Int) {imm}:
        for i in range((n * w) // W, (n * (w + 1)) // W):
            i_a[i] = I32(i)

    sync_par(init, W)

    for p in range(NBYTES):
        if _radix_pass[UT](k_a, k_b, i_a, i_b, hist, n, W, p * 8):
            var tk: Pointer[Scalar[UT], MutUntrackedOrigin] = k_a
            k_a = k_b
            k_b = tk
            var ti: Pointer[I32, MutUntrackedOrigin] = i_a
            i_a = i_b
            i_b = ti

    if i_a != out_idx:                          # result ended in the scratch buffer
        def copy_out(w: Int) {imm}:
            for i in range((n * w) // W, (n * (w + 1)) // W):
                out_idx[i] = i_a[i]

        sync_par(copy_out, W)
    # free the two scratch index/key buffers (never `keys`' owner or out_idx)
    if i_a != out_idx:
        i_a.free()
    else:
        i_b.free()
    if k_a != keys:
        k_a.free()
    else:
        k_b.free()
    hist.free()


def _sort_impl[KIND: Int, UT: DType](
    out_addr: Int, keys_addr: Int, n: Int, descending: Int, threads: Int
):
    if n <= 0:
        return
    var raw = _sp[UT](keys_addr)
    var mapped = unsafe_alloc[Scalar[UT]](n)
    var desc = descending > 0
    var W = _pick_workers(n, threads, MIN_ROWS_STREAM)

    def map_keys(w: Int) {imm}:
        for i in range((n * w) // W, (n * (w + 1)) // W):
            mapped[i] = _to_key[KIND, UT](raw[i], desc)

    sync_par(map_keys, W)
    _radix_sort[UT](_i32(out_addr), mapped, n, threads)
    mapped.free()


@export
def sort_indices(out_addr: Int, keys_addr: Int, n: Int, descending: Int, threads: Int) abi("C"):
    _sort_impl[KIND_I32, DType.uint32](out_addr, keys_addr, n, descending, threads)

@export
def sort_indices_i64(out_addr: Int, keys_addr: Int, n: Int, descending: Int, threads: Int) abi("C"):
    _sort_impl[KIND_I64, DType.uint64](out_addr, keys_addr, n, descending, threads)

@export
def sort_indices_f32(out_addr: Int, keys_addr: Int, n: Int, descending: Int, threads: Int) abi("C"):
    _sort_impl[KIND_F32, DType.uint32](out_addr, keys_addr, n, descending, threads)

@export
def sort_indices_f64(out_addr: Int, keys_addr: Int, n: Int, descending: Int, threads: Int) abi("C"):
    _sort_impl[KIND_F64, DType.uint64](out_addr, keys_addr, n, descending, threads)


# ─────────────────────────────────────────────────────────────────────────────
# unique_mask   out[i] = 1 if sorted_keys[i] != sorted_keys[i-1] (or i == 0)
# ─────────────────────────────────────────────────────────────────────────────
@export
def unique_mask(out_addr: Int, keys_addr: Int, n: Int, threads: Int) abi("C"):
    if n <= 0:
        return
    var out = _i32(out_addr)
    var keys = _i32(keys_addr)
    var W = _pick_workers(n, threads, MIN_ROWS_STREAM)

    def work(w: Int) {imm}:
        var lo = (n * w) // W
        var hi = (n * (w + 1)) // W
        if lo == 0:
            out[0] = I32(1)
            lo = 1
        for i in range(lo, hi):
            out[i] = I32(1) if keys[i] != keys[i - 1] else I32(0)

    sync_par(work, W)


# ─────────────────────────────────────────────────────────────────────────────
# prefix_sum_count   offsets[0..n] = exclusive prefix sum of mask, offsets[n] = total
# Two parallel sweeps over contiguous chunks with a tiny serial scan of the
# chunk totals in between.
# ─────────────────────────────────────────────────────────────────────────────
@export
def prefix_sum_count(offsets_addr: Int, mask_addr: Int, n: Int, threads: Int) abi("C"):
    var offsets = _i32(offsets_addr)
    var mask = _i32(mask_addr)
    if n <= 0:
        offsets[0] = I32(0)
        return
    var W = _pick_workers(n, threads, MIN_ROWS_STREAM)
    var sums = unsafe_alloc[I32](W)

    def total(w: Int) {imm}:
        var s = I32(0)
        for i in range((n * w) // W, (n * (w + 1)) // W):
            s += mask[i]
        sums[w] = s

    sync_par(total, W)
    var run = I32(0)
    for w in range(W):
        var s = sums[w]
        sums[w] = run
        run += s
    offsets[n] = run

    def fill(w: Int) {imm}:
        var acc = sums[w]
        for i in range((n * w) // W, (n * (w + 1)) // W):
            offsets[i] = acc
            acc += mask[i]

    sync_par(fill, W)
    sums.free()


# ─────────────────────────────────────────────────────────────────────────────
# filter_gather_{f32,i32,i64}   out[offsets[i]] = values[i]  where mask[i] == 1
# Every selected row has its own output slot, so chunks are independent.
# ─────────────────────────────────────────────────────────────────────────────
def _filter_gather[T: DType](
    out_addr: Int, values_addr: Int, mask_addr: Int, offsets_addr: Int, n: Int, threads: Int
):
    if n <= 0:
        return
    var out = _sp[T](out_addr)
    var values = _sp[T](values_addr)
    var mask = _i32(mask_addr)
    var offsets = _i32(offsets_addr)
    var W = _pick_workers(n, threads, MIN_ROWS_STREAM)

    def work(w: Int) {imm}:
        for i in range((n * w) // W, (n * (w + 1)) // W):
            if mask[i] == I32(1):
                out[Int(offsets[i])] = values[i]

    sync_par(work, W)


@export
def filter_gather_f32(out_addr: Int, values_addr: Int, mask_addr: Int, offsets_addr: Int,
                      n: Int, threads: Int) abi("C"):
    _filter_gather[DType.float32](out_addr, values_addr, mask_addr, offsets_addr, n, threads)

@export
def filter_gather_i32(out_addr: Int, values_addr: Int, mask_addr: Int, offsets_addr: Int,
                      n: Int, threads: Int) abi("C"):
    _filter_gather[DType.int32](out_addr, values_addr, mask_addr, offsets_addr, n, threads)

@export
def filter_gather_i64(out_addr: Int, values_addr: Int, mask_addr: Int, offsets_addr: Int,
                      n: Int, threads: Int) abi("C"):
    _filter_gather[DType.int64](out_addr, values_addr, mask_addr, offsets_addr, n, threads)


# ─────────────────────────────────────────────────────────────────────────────
# Joins (inner / left) on dense int32 keys.
#
# Build side (right): a direct-address "CSR" index over keys 0..max_key,
#   A[k + 1] = end offset of key k's run, positions[] = right row ids grouped by
#   key (ascending row id within a key, so output order matches the serial
#   kernel exactly).  Built in parallel by KEY-RANGE OWNERSHIP: worker w owns a
#   contiguous key range and sweeps all right keys, touching only its own
#   counters — no atomics, no partial tables.  Probe (left) is embarrassingly
#   parallel over left rows.
#
#   count-only layout : start(k) = A[k],     end(k) = A[k + 1]
#   after fill layout : start(k) = A[k + 1], end(k) = A[k + 2]
# ─────────────────────────────────────────────────────────────────────────────

comptime MIN_KEYS_PER_OWNER = 4096


def _build_index(
    A: Pointer[I32, MutUntrackedOrigin],          # (table_size + 2) entries
    positions: Pointer[I32, MutUntrackedOrigin],  # n_right entries (fill only)
    right_keys: Pointer[I32, MutUntrackedOrigin],
    n_right: Int,
    table_size: Int,
    fill: Bool,
    threads: Int,
):
    var T = table_size
    var Wk = _pick_workers(n_right, threads, 1 << 18)
    var cap = T // MIN_KEYS_PER_OWNER
    if cap < 1:
        cap = 1
    if Wk > cap:
        Wk = cap
    var Wz = _pick_workers(T + 2, threads, 1 << 18)
    var tot = unsafe_alloc[Int](Wk + 1)

    def zero(w: Int) {imm}:
        var lo = (w * (T + 2)) // Wz
        var hi = ((w + 1) * (T + 2)) // Wz
        unsafe_memset_zero(A + lo, hi - lo)

    def count(w: Int) {imm}:
        var kl = (w * T) // Wk
        var kh = ((w + 1) * T) // Wk
        for i in range(n_right):
            var k = Int(right_keys[i])
            if k >= kl and k < kh:
                A[k + 1] += 1

    def scan1(w: Int) {imm}:
        var kl = (w * T) // Wk
        var kh = ((w + 1) * T) // Wk
        var run = 0
        for k in range(kl, kh):
            run += Int(A[k + 1])
            A[k + 1] = I32(run)
        tot[w] = run

    def scan2(w: Int) {imm}:
        var kl = (w * T) // Wk
        var kh = ((w + 1) * T) // Wk
        var base = I32(tot[w])          # exclusive offset of this range (set below)
        if base != 0:
            for k in range(kl, kh):
                A[k + 1] += base

    def fill_pass(w: Int) {imm}:
        var kl = (w * T) // Wk
        var kh = ((w + 1) * T) // Wk
        var i = n_right - 1
        while i >= 0:
            var k = Int(right_keys[i])
            if k >= kl and k < kh:
                var p = A[k + 1] - 1
                A[k + 1] = p
                positions[Int(p)] = I32(i)
            i -= 1

    sync_par(zero, Wz)
    sync_par(count, Wk)
    sync_par(scan1, Wk)
    # turn per-range totals into exclusive bases, in place
    var run = 0
    for w in range(Wk):
        var t = tot[w]
        tot[w] = run
        run += t
    A[T + 1] = I32(run)                 # sentinel: total valid right rows
    sync_par(scan2, Wk)
    if fill:
        sync_par(fill_pass, Wk)
    tot.free()


@always_inline
def _run_bounds(A: Pointer[I32, MutUntrackedOrigin], k: Int, sh: Int) -> Int:
    """Number of right rows matching key k (0 for out-of-range keys is handled
    by the caller)."""
    return Int(A[k + sh + 1]) - Int(A[k + sh])


@export
def join_count(
    match_counts_addr: Int,   # int32[n_left]
    left_keys_addr:    Int,   # int32[n_left]
    right_keys_addr:   Int,   # int32[n_right]
    n_left:            Int,
    n_right:           Int,
    max_key:           Int,   # caller provides max(max(left), max(right))
    threads:           Int,
) abi("C"):
    var match_counts = _i32(match_counts_addr)
    var left_keys    = _i32(left_keys_addr)
    var right_keys   = _i32(right_keys_addr)
    var T = max_key + 1
    if T < 1:
        T = 1
    var A = unsafe_alloc[I32](T + 2)
    var pos = unsafe_alloc[I32](1)
    _build_index(A, pos, right_keys, n_right, T, False, threads)
    var W = _pick_workers(n_left, threads, MIN_ROWS)

    var ctr = _new_queue()
    var n_m = _n_morsels(n_left)

    def probe(w: Int) {imm}:
        while True:
            var q = _claim(ctr)
            if q >= n_m:
                break
            var lo = q * MORSEL
            var hi = lo + MORSEL
            if hi > n_left:
                hi = n_left
            for i in range(lo, hi):
                var k = Int(left_keys[i])
                if k >= 0 and k < T:
                    match_counts[i] = I32(_run_bounds(A, k, 0))
                else:
                    match_counts[i] = I32(0)

    sync_par(probe, W)
    ctr.free()
    A.free()
    pos.free()


@export
def join_count_left(
    match_counts_addr: Int,
    left_keys_addr:    Int,
    right_keys_addr:   Int,
    n_left:            Int,
    n_right:           Int,
    max_key:           Int,
    threads:           Int,
) abi("C"):
    """Left-join counts: max(1, right matches), so unmatched left rows get 1."""
    var match_counts = _i32(match_counts_addr)
    var left_keys    = _i32(left_keys_addr)
    var right_keys   = _i32(right_keys_addr)
    var T = max_key + 1
    if T < 1:
        T = 1
    var A = unsafe_alloc[I32](T + 2)
    var pos = unsafe_alloc[I32](1)
    _build_index(A, pos, right_keys, n_right, T, False, threads)
    var W = _pick_workers(n_left, threads, MIN_ROWS)

    var ctr = _new_queue()
    var n_m = _n_morsels(n_left)

    def probe(w: Int) {imm}:
        while True:
            var q = _claim(ctr)
            if q >= n_m:
                break
            var lo = q * MORSEL
            var hi = lo + MORSEL
            if hi > n_left:
                hi = n_left
            for i in range(lo, hi):
                var k = Int(left_keys[i])
                var cnt = 0
                if k >= 0 and k < T:
                    cnt = _run_bounds(A, k, 0)
                match_counts[i] = I32(cnt if cnt > 0 else 1)

    sync_par(probe, W)
    ctr.free()
    A.free()
    pos.free()


def _join_scatter_impl[LEFT: Bool](
    left_out_addr:   Int,
    right_out_addr:  Int,
    left_keys_addr:  Int,
    right_keys_addr: Int,
    offsets_addr:    Int,
    n_left:          Int,
    n_right:         Int,
    max_key:         Int,
    threads:         Int,
):
    var left_out   = _i32(left_out_addr)
    var right_out  = _i32(right_out_addr)
    var left_keys  = _i32(left_keys_addr)
    var right_keys = _i32(right_keys_addr)
    var offsets    = _i32(offsets_addr)
    var T = max_key + 1
    if T < 1:
        T = 1
    var A = unsafe_alloc[I32](T + 2)
    var pos = unsafe_alloc[I32](n_right if n_right > 0 else 1)
    _build_index(A, pos, right_keys, n_right, T, True, threads)
    var W = _pick_workers(n_left, threads, MIN_ROWS)

    var ctr = _new_queue()
    var n_m = _n_morsels(n_left)

    def probe(w: Int) {imm}:
        while True:
            var q = _claim(ctr)
            if q >= n_m:
                break
            var lo = q * MORSEL
            var hi = lo + MORSEL
            if hi > n_left:
                hi = n_left
            for i in range(lo, hi):
                var k = Int(left_keys[i])
                var s = 0
                var rc = 0
                if k >= 0 and k < T:
                    s = Int(A[k + 1])
                    rc = Int(A[k + 2]) - s
                var base = Int(offsets[i])
                if rc == 0:
                    comptime if LEFT:
                        left_out[base]  = I32(i)
                        right_out[base] = I32(-1)
                else:
                    for j in range(rc):
                        left_out[base + j]  = I32(i)
                        right_out[base + j] = pos[s + j]

    sync_par(probe, W)
    ctr.free()
    A.free()
    pos.free()


# join_scatter (inner join): emit (left_index, right_index) pairs given
# exclusive prefix-sum offsets of match_counts.
@export
def join_scatter(
    left_out_addr:   Int,   # int32[total_matches]
    right_out_addr:  Int,   # int32[total_matches]
    left_keys_addr:  Int,   # int32[n_left]
    right_keys_addr: Int,   # int32[n_right]
    offsets_addr:    Int,   # int32[n_left] exclusive prefix sum of match_counts
    n_left:          Int,
    n_right:         Int,
    max_key:         Int,
    threads:         Int,
) abi("C"):
    _join_scatter_impl[False](left_out_addr, right_out_addr, left_keys_addr,
        right_keys_addr, offsets_addr, n_left, n_right, max_key, threads)


# join_scatter_left (left join): like join_scatter but unmatched left rows
# emit right_index = -1.
@export
def join_scatter_left(
    left_out_addr:   Int,
    right_out_addr:  Int,
    left_keys_addr:  Int,
    right_keys_addr: Int,
    offsets_addr:    Int,
    n_left:          Int,
    n_right:         Int,
    max_key:         Int,
    threads:         Int,
) abi("C"):
    _join_scatter_impl[True](left_out_addr, right_out_addr, left_keys_addr,
        right_keys_addr, offsets_addr, n_left, n_right, max_key, threads)
