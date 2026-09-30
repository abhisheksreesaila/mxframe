"""
MXFrame fused pipeline kernel, GPU side (F-008).

Interprets the same bytecode program as pipeline_cpu.mojo, one row per thread
in a grid-stride loop. Every thread of a warp executes the same instruction, so
interpretation does not diverge; only the predicate can retire a row early.

Aggregation tiers (chosen from the program's slot and group counts):
  * small  ((n_slots + 1) * n_groups <= SMALL): every thread keeps its own
    partial per (slot, group) in local memory, then the warp reduces with
    shuffles and lane 0 issues one float64 atomic per slot. No per-row atomics.
  * large: per-row float64 atomics (add / min / max) straight into the output.

All arithmetic is float64 (columns are widened on load). Device pointers are
passed as integers; the Python side owns device memory (CUDA driver API on the
primary context, like AOTKernelsGPU), so already-resident morsels are accepted
without another copy.

Build:
    pixi run mojo build kernels_aot/pipeline_gpu.mojo \
        --emit shared-lib -o kernels_aot/libmxpipeline_gpu.so
"""

from std.gpu.host import DeviceContext
from std.math import ceildiv, floor
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from std.gpu.primitives.warp import shuffle_down
from std.memory import UnsafePointer, stack_allocation, bitcast
from std.atomic import Atomic

comptime F64 = DType.float64
comptime BLOCK: Int = 256
comptime MAX_REGS: Int = 48
comptime SMALL: Int = 64
comptime HDR: Int = 16

comptime OP_LOAD_COL = 1
comptime OP_LOAD_CONST = 2
comptime OP_ADD = 10
comptime OP_SUB = 11
comptime OP_MUL = 12
comptime OP_DIV = 13
comptime OP_LT = 20
comptime OP_LE = 21
comptime OP_GT = 22
comptime OP_GE = 23
comptime OP_EQ = 24
comptime OP_NE = 25
comptime OP_AND = 30
comptime OP_OR = 31
comptime OP_NOT = 32
comptime OP_FLOOR = 33
comptime OP_SELECT = 35
comptime OP_BETWEEN = 36
comptime OP_IN_SET = 37
comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime OP_AGG_MAX = 52

comptime FP = UnsafePointer[Float64, MutAnyOrigin]
comptime IP = UnsafePointer[Int32, MutAnyOrigin]
comptime LP = UnsafePointer[Int64, MutAnyOrigin]
comptime BP = UnsafePointer[UInt8, MutAnyOrigin]


@always_inline
def _fp(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)

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
def _load(addr: Int, dtype: Int, row: Int) -> Float64:
    if dtype == 0:
        return _ip(addr)[row].cast[F64]()
    elif dtype == 1:
        return _lp(addr)[row].cast[F64]()
    elif dtype == 2:
        return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)[row].cast[F64]()
    elif dtype == 3:
        return _fp(addr)[row]
    else:
        return _bp(addr)[row].cast[F64]()


@always_inline
def _b(x: Bool) -> Float64:
    return 1.0 if x else 0.0


@always_inline
def _shfl_down(v: Float64, off: Int) -> Float64:
    """Float64 warp shuffle as two 32-bit shuffles (the stdlib has no f64 shuffle)."""
    var bits = bitcast[DType.uint32, 2](SIMD[F64, 1](v))
    var lo = shuffle_down(SIMD[DType.uint32, 1](bits[0]), UInt32(off))
    var hi = shuffle_down(SIMD[DType.uint32, 1](bits[1]), UInt32(off))
    return bitcast[F64, 1](SIMD[DType.uint32, 2](lo[0], hi[0]))[0]

@always_inline
def _warp_reduce_add(v: Float64) -> Float64:
    var x = v
    var off = 16
    while off > 0:
        x += _shfl_down(x, off)
        off >>= 1
    return x

@always_inline
def _warp_reduce_min(v: Float64) -> Float64:
    var x = v
    var off = 16
    while off > 0:
        x = min(x, _shfl_down(x, off))
        off >>= 1
    return x

@always_inline
def _warp_reduce_max(v: Float64) -> Float64:
    var x = v
    var off = 16
    while off > 0:
        x = max(x, _shfl_down(x, off))
        off >>= 1
    return x


@always_inline
def _run(code: IP, base: Int, count: Int, regs: UnsafePointer[Float64, MutAnyOrigin],
         consts: FP, cols: LP, dts: IP, set_bytes: BP, set_meta: LP, row: Int):
    for k in range(count):
        var ins = code + (HDR + (base + k) * 5)
        var op = Int(ins[0])
        var d = Int(ins[1])
        if op == OP_LOAD_COL:
            var c = Int(ins[2])
            regs[d] = _load(Int(cols[c]), Int(dts[c]), row)
        elif op == OP_LOAD_CONST:
            regs[d] = consts[Int(ins[2])]
        elif op == OP_NOT:
            regs[d] = _b(regs[Int(ins[2])] == 0.0)
        elif op == OP_FLOOR:
            regs[d] = floor(regs[Int(ins[2])])
        elif op == OP_SELECT:
            regs[d] = regs[Int(ins[3])] if regs[Int(ins[2])] != 0.0 else regs[Int(ins[4])]
        elif op == OP_BETWEEN:
            var x = regs[Int(ins[2])]
            regs[d] = _b(x >= regs[Int(ins[3])] and x <= regs[Int(ins[4])])
        elif op == OP_IN_SET:
            var s = Int(ins[3])
            var x = regs[Int(ins[2])] - Float64(Int(set_meta[s * 3 + 2]))
            var hit = 0.0
            if x >= 0.0 and x < Float64(Int(set_meta[s * 3 + 1])):
                if set_bytes[Int(set_meta[s * 3]) + Int(x)] != 0:
                    hit = 1.0
            regs[d] = hit
        else:
            var a = regs[Int(ins[2])]
            var b = regs[Int(ins[3])]
            var r = 0.0
            if op == OP_ADD:
                r = a + b
            elif op == OP_SUB:
                r = a - b
            elif op == OP_MUL:
                r = a * b
            elif op == OP_DIV:
                r = a / b
            elif op == OP_LT:
                r = _b(a < b)
            elif op == OP_LE:
                r = _b(a <= b)
            elif op == OP_GT:
                r = _b(a > b)
            elif op == OP_GE:
                r = _b(a >= b)
            elif op == OP_EQ:
                r = _b(a == b)
            elif op == OP_NE:
                r = _b(a != b)
            elif op == OP_AND:
                r = _b(a != 0.0 and b != 0.0)
            else:
                r = _b(a != 0.0 or b != 0.0)
            regs[d] = r


@export
def mx_pipeline_gpu(
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    cols_a: Int, dtypes_a: Int, gids_a: Int, n_groups: Int, n_slots: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, ident_a: Int,
    n_blocks: Int,
) abi("C") -> Int:
    if n_slots < 0 or n_groups < 1:
        return -2
    var G = n_groups
    var A = n_slots
    var has_gids = gids_a != 0
    if (not has_gids) and G != 1:
        return -2

    @parameter
    def init_out(total: Int, g_: Int, cnt_total: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < total:
            _fp(out_vals_a)[t] = _fp(ident_a)[t // g_]
        if t < cnt_total:
            _lp(out_cnt_a)[t] = 0

    @parameter
    def kernel(n_rows_end: Int, row0: Int):
        var code = _ip(code_a)
        var n_pro = Int(code[2])
        var n_pred = Int(code[3])
        var n_body = Int(code[4])
        var pred_reg = Int(code[6])
        var consts = _fp(consts_a)
        var cols = _lp(cols_a)
        var dts = _ip(dtypes_a)
        var set_bytes = _bp(set_bytes_a)
        var set_meta = _lp(set_meta_a)
        var out_vals = _fp(out_vals_a)
        var out_cnt = _lp(out_cnt_a)
        var gids = _ip(gids_a)
        var agg_base = n_pro + n_pred + n_body
        var has_gids = gids_a != 0
        var small = (n_slots + 1) * n_groups <= SMALL
        var A = n_slots
        var G = n_groups

        var regs = stack_allocation[MAX_REGS, Scalar[F64]]()
        var acc = stack_allocation[SMALL, Scalar[F64]]()
        # constants once per thread
        _run(code, 0, n_pro, regs, consts, cols, dts, set_bytes, set_meta, 0)
        if small:
            for a in range(A):
                var kind = Int(code[HDR + (agg_base + a) * 5])
                var init = 0.0
                if kind == OP_AGG_MIN:
                    init = Float64.MAX
                elif kind == OP_AGG_MAX:
                    init = -Float64.MAX
                for g in range(G):
                    acc[a * G + g] = init
            for g in range(G):
                acc[A * G + g] = 0.0

        var stride = Int(grid_dim.x) * BLOCK
        var row = row0 + Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        while row < n_rows_end:
            var keep = True
            if pred_reg >= 0:
                _run(code, n_pro, n_pred, regs, consts, cols, dts, set_bytes, set_meta, row)
                keep = regs[pred_reg] != 0.0
            if keep:
                var g = 0
                if has_gids:
                    g = Int(gids[row])
                if g >= 0 and g < G:
                    _run(code, n_pro + n_pred, n_body, regs, consts, cols, dts,
                         set_bytes, set_meta, row)
                    if small:
                        acc[A * G + g] += 1.0
                        for a in range(A):
                            var ins = code + (HDR + (agg_base + a) * 5)
                            var kind = Int(ins[0])
                            var v = regs[Int(ins[2])]
                            var q = a * G + g
                            if kind == OP_AGG_SUM:
                                acc[q] += v
                            elif kind == OP_AGG_MIN:
                                acc[q] = min(acc[q], v)
                            else:
                                acc[q] = max(acc[q], v)
                    else:
                        _ = Atomic[DType.int64].fetch_add(out_cnt + g, Int64(1))
                        for a in range(A):
                            var ins = code + (HDR + (agg_base + a) * 5)
                            var kind = Int(ins[0])
                            var v = regs[Int(ins[2])]
                            var p = out_vals + (a * G + g)
                            if kind == OP_AGG_SUM:
                                _ = Atomic[F64].fetch_add(p, v)
                            elif kind == OP_AGG_MIN:
                                Atomic[F64].min(p, v)
                            else:
                                Atomic[F64].max(p, v)
            row += stride

        if small:
            # warp-level reduction, then one atomic per slot per warp
            var lane = Int(thread_idx.x) & 31
            for a in range(A):
                var kind = Int(code[HDR + (agg_base + a) * 5])
                for g in range(G):
                    var q = a * G + g
                    if kind == OP_AGG_SUM:
                        var s = _warp_reduce_add(acc[q])
                        if lane == 0 and s != 0.0:
                            _ = Atomic[F64].fetch_add(out_vals + q, s)
                    elif kind == OP_AGG_MIN:
                        var s = _warp_reduce_min(acc[q])
                        if lane == 0 and s != Float64.MAX:
                            Atomic[F64].min(out_vals + q, s)
                    else:
                        var s = _warp_reduce_max(acc[q])
                        if lane == 0 and s != -Float64.MAX:
                            Atomic[F64].max(out_vals + q, s)
            for g in range(G):
                var c = _warp_reduce_add(acc[A * G + g])
                if lane == 0 and c != 0.0:
                    _ = Atomic[DType.int64].fetch_add(out_cnt + g, Int64(c))

    try:
        var ctx = DeviceContext()
        var total = A * G
        var cnt_total = G
        var init_n = max(total, cnt_total)
        ctx.enqueue_function_experimental[init_out](
            total, G, cnt_total, grid_dim=ceildiv(init_n, BLOCK), block_dim=BLOCK)
        if end > start:
            var blocks = min(ceildiv(end - start, BLOCK), max(n_blocks, 1))
            ctx.enqueue_function_experimental[kernel](
                end, start, grid_dim=blocks, block_dim=BLOCK)
        ctx.synchronize()
        return 0
    except:
        return -1
