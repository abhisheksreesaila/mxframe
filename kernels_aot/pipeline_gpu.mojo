"""
MXFrame fused pipeline kernel, GPU side (F-008).

Interprets the same bytecode program as pipeline_cpu.mojo. Each thread owns
RPT rows of a tile (TILE = BLOCK * RPT rows per block); the interpreter's
"registers" are a thread-private register file kept in dynamic shared memory
(`regs[reg][tile_row]`, conflict-free), so nothing spills to local memory and
no barriers are needed between instructions. Every thread of a warp executes
the same instruction (uniform control flow); the predicate only masks rows.
The program itself is staged in shared memory once per block.

Aggregation modes (chosen from the slot and group counts):
  * 0, few groups ((n_slots + 1) * n_groups <= SMALL, including the global
    aggregate with one group): every thread accumulates its own partial per
    (slot, group); at the end the block reduces them with warp shuffles and
    writes one row of `scratch`; a second kernel merges the per-block rows in
    block order. No atomics, and results are deterministic.
  * 2, many groups: per-row float64 atomics (add / min / max) into the output.
  (Shared-memory float64 atomics were tried for a middle tier and are ~100x
  slower on sm_86: they lower to compare-and-swap loops under warp conflicts.)

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
from std.gpu import thread_idx, block_idx, grid_dim, barrier
from std.gpu.primitives.warp import shuffle_down
from std.gpu.memory import AddressSpace, external_memory
from std.memory import UnsafePointer, stack_allocation, bitcast
from std.atomic import Atomic

comptime F64 = DType.float64
comptime BLOCK: Int = 256
comptime SMALL: Int = 48            # per-thread accumulators (mode 0)
comptime HDR: Int = 16
comptime MAX_CONSTS: Int = 64
comptime WARPS: Int = BLOCK // 32

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

comptime SH = AddressSpace.SHARED
comptime SFP = UnsafePointer[Float64, MutAnyOrigin, address_space=SH]
comptime SIP = UnsafePointer[Int32, MutAnyOrigin, address_space=SH]
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
    if dtype == 2:
        return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)[row].cast[F64]()
    elif dtype == 0:
        return _ip(addr)[row].cast[F64]()
    elif dtype == 3:
        return _fp(addr)[row]
    elif dtype == 1:
        return _lp(addr)[row].cast[F64]()
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
def _combine(kind: Int, a: Float64, b: Float64) -> Float64:
    if kind == OP_AGG_MIN:
        return min(a, b)
    elif kind == OP_AGG_MAX:
        return max(a, b)
    return a + b


@always_inline
def _warp_reduce(kind: Int, v: Float64) -> Float64:
    var x = v
    var off = 16
    while off > 0:
        x = _combine(kind, x, _shfl_down(x, off))
        off >>= 1
    return x


@always_inline
def _ident(kind: Int) -> Float64:
    if kind == OP_AGG_MIN:
        return Float64.MAX
    elif kind == OP_AGG_MAX:
        return -Float64.MAX
    return 0.0


def _run[RPT: Int](
    code: SIP, base: Int, count: Int, regs: SFP, consts: SFP, cols: LP, dts: IP,
    set_bytes: BP, set_meta: LP, tid: Int, row_base: Int, mask: Int,
):
    """Run `count` instructions for this thread's RPT rows; bit k of `mask` = row k is live."""
    comptime TILE = BLOCK * RPT
    for n in range(count):
        var ins = code + (HDR + (base + n) * 5)
        var op = Int(ins[0])
        var d = regs + Int(ins[1]) * TILE + tid
        if op == OP_LOAD_COL:
            var c = Int(ins[2])
            var addr = Int(cols[c])
            var dt = Int(dts[c])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * BLOCK] = _load(addr, dt, row_base + tid + k * BLOCK)
        elif op == OP_LOAD_CONST:
            var v = consts[Int(ins[2])]
            comptime for k in range(RPT):
                d[k * BLOCK] = v
        elif op < OP_LT:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * BLOCK]
                    var y = b[k * BLOCK]
                    var r = x + y
                    if op == OP_MUL:
                        r = x * y
                    elif op == OP_SUB:
                        r = x - y
                    elif op == OP_DIV:
                        r = x / y
                    d[k * BLOCK] = r
        elif op < OP_AND:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * BLOCK]
                    var y = b[k * BLOCK]
                    var r = _b(x < y)
                    if op == OP_LE:
                        r = _b(x <= y)
                    elif op == OP_GT:
                        r = _b(x > y)
                    elif op == OP_GE:
                        r = _b(x >= y)
                    elif op == OP_EQ:
                        r = _b(x == y)
                    elif op == OP_NE:
                        r = _b(x != y)
                    d[k * BLOCK] = r
        elif op < OP_NOT:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * BLOCK] != 0.0
                    var y = b[k * BLOCK] != 0.0
                    d[k * BLOCK] = _b(x and y) if op == OP_AND else _b(x or y)
        elif op == OP_NOT:
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * BLOCK] = _b(a[k * BLOCK] == 0.0)
        elif op == OP_FLOOR:
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * BLOCK] = floor(a[k * BLOCK])
        elif op == OP_SELECT:
            var c = regs + Int(ins[2]) * TILE + tid
            var a = regs + Int(ins[3]) * TILE + tid
            var b = regs + Int(ins[4]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * BLOCK] = a[k * BLOCK] if c[k * BLOCK] != 0.0 else b[k * BLOCK]
        elif op == OP_BETWEEN:
            var x = regs + Int(ins[2]) * TILE + tid
            var lo = regs + Int(ins[3]) * TILE + tid
            var hi = regs + Int(ins[4]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var v = x[k * BLOCK]
                    d[k * BLOCK] = _b(v >= lo[k * BLOCK] and v <= hi[k * BLOCK])
        elif op == OP_IN_SET:
            var s = Int(ins[3])
            var off = Int(set_meta[s * 3])
            var length = Float64(Int(set_meta[s * 3 + 1]))
            var lo = Float64(Int(set_meta[s * 3 + 2]))
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * BLOCK] - lo
                    var hit = 0.0
                    if x >= 0.0 and x < length:
                        if set_bytes[off + Int(x)] != 0:
                            hit = 1.0
                    d[k * BLOCK] = hit


def pipeline_kernel[RPT: Int](
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    cols_a: Int, dtypes_a: Int, gids_a: Int, n_groups: Int, n_slots: Int,
    row0: Int, row_end: Int, out_vals_a: Int, out_cnt_a: Int, scratch_a: Int,
    n_regs: Int, code_ints: Int, mode: Int,
):
    comptime TILE = BLOCK * RPT
    var tid = Int(thread_idx.x)
    var G = n_groups
    var A = n_slots

    # dynamic shared memory: [register file | constants | warp partials | program]
    var smem = external_memory[Scalar[F64], address_space=SH, alignment=16]()
    var regs = smem
    var consts = smem + n_regs * TILE
    var wpart = consts + MAX_CONSTS
    var code = (wpart + WARPS * SMALL).bitcast[Int32]()
    var i = tid
    while i < code_ints:
        code[i] = _ip(code_a)[i]
        i += BLOCK
    i = tid
    while i < MAX_CONSTS:
        consts[i] = _fp(consts_a)[i]
        i += BLOCK
    barrier()

    var n_pro = Int(code[2])
    var n_pred = Int(code[3])
    var n_body = Int(code[4])
    var pred_reg = Int(code[6])
    var agg_base = n_pro + n_pred + n_body
    var cols = _lp(cols_a)
    var dts = _ip(dtypes_a)
    var set_bytes = _bp(set_bytes_a)
    var set_meta = _lp(set_meta_a)
    var gids = _ip(gids_a)
    var has_gids = gids_a != 0
    var out_vals = _fp(out_vals_a)
    var out_cnt = _lp(out_cnt_a)

    var all_rows = (1 << RPT) - 1
    _run[RPT](code, 0, n_pro, regs, consts, cols, dts, set_bytes, set_meta, tid, 0, all_rows)

    var acc = stack_allocation[SMALL, Scalar[F64]]()
    var ns = (A + 1) * G
    if mode == 0:
        for a in range(A):
            var init = _ident(Int(code[HDR + (agg_base + a) * 5]))
            for g in range(G):
                acc[a * G + g] = init
        for g in range(G):
            acc[A * G + g] = 0.0

    var n_tiles = ceildiv(row_end - row0, TILE)
    var tile = Int(block_idx.x)
    var step = Int(grid_dim.x)
    while tile < n_tiles:
        var row_base = row0 + tile * TILE
        var valid = 0
        comptime for k in range(RPT):
            if row_base + tid + k * BLOCK < row_end:
                valid |= 1 << k
        var active = valid
        if pred_reg >= 0:
            _run[RPT](code, n_pro, n_pred, regs, consts, cols, dts, set_bytes, set_meta,
                      tid, row_base, valid)
            active = 0
            comptime for k in range(RPT):
                if (valid >> k) & 1 != 0 and regs[pred_reg * TILE + tid + k * BLOCK] != 0.0:
                    active |= 1 << k
        if active != 0:
            _run[RPT](code, n_pro + n_pred, n_body, regs, consts, cols, dts, set_bytes,
                      set_meta, tid, row_base, active)
            comptime for k in range(RPT):
                if (active >> k) & 1 != 0:
                    var g = 0
                    if has_gids:
                        g = Int(gids[row_base + tid + k * BLOCK])
                    if g >= 0 and g < G:
                        if mode == 0:
                            acc[A * G + g] += 1.0
                            for a in range(A):
                                var ins = code + (HDR + (agg_base + a) * 5)
                                var kind = Int(ins[0])
                                var v = regs[Int(ins[2]) * TILE + tid + k * BLOCK]
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
                                var v = regs[Int(ins[2]) * TILE + tid + k * BLOCK]
                                var p = out_vals + (a * G + g)
                                if kind == OP_AGG_SUM:
                                    _ = Atomic[F64].fetch_add(p, v)
                                elif kind == OP_AGG_MIN:
                                    Atomic[F64].min(p, v)
                                else:
                                    Atomic[F64].max(p, v)
        tile += step

    if mode == 0:
        # reduce the block's per-thread partials: warp shuffles, then across warps
        var lane = tid & 31
        var warp = tid >> 5
        for s_ in range(ns):
            var kind = OP_AGG_SUM
            if s_ < A * G:
                kind = Int(code[HDR + (agg_base + s_ // G) * 5])
            var w = _warp_reduce(kind, acc[s_])
            if lane == 0:
                wpart[warp * SMALL + s_] = w
        barrier()
        if tid < ns:
            var kind = OP_AGG_SUM
            if tid < A * G:
                kind = Int(code[HDR + (agg_base + tid // G) * 5])
            var r = wpart[tid]
            for w in range(1, WARPS):
                r = _combine(kind, r, wpart[w * SMALL + tid])
            _fp(scratch_a)[Int(block_idx.x) * ns + tid] = r


def init_kernel(out_vals_a: Int, out_cnt_a: Int, ident_a: Int, n_groups: Int, total: Int):
    var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if t < total:
        _fp(out_vals_a)[t] = _fp(ident_a)[t // n_groups]
    if t < n_groups:
        _lp(out_cnt_a)[t] = 0


def merge_kernel(
    code_a: Int, scratch_a: Int, out_vals_a: Int, out_cnt_a: Int, n_blocks: Int,
    n_groups: Int, n_slots: Int, agg_base: Int, ident_a: Int,
):
    """Merge the per-block partials in block order (deterministic)."""
    var s = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var A = n_slots
    var G = n_groups
    var ns = (A + 1) * G
    if s >= ns:
        return
    var kind = OP_AGG_SUM
    var r = 0.0
    if s < A * G:
        kind = Int(_ip(code_a)[HDR + (agg_base + s // G) * 5])
        r = _fp(ident_a)[s // G]
    for b in range(n_blocks):
        r = _combine(kind, r, _fp(scratch_a)[b * ns + s])
    if s < A * G:
        _fp(out_vals_a)[s] = r
    else:
        _lp(out_cnt_a)[s - A * G] = Int64(r)


@export
def mx_pipeline_gpu(
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    cols_a: Int, dtypes_a: Int, gids_a: Int, n_groups: Int, n_slots: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, ident_a: Int,
    scratch_a: Int, n_regs: Int, code_ints: Int, agg_base: Int,
    n_blocks: Int, repeat: Int,
) abi("C") -> Int:
    if n_slots < 0 or n_groups < 1:
        return -2
    var G = n_groups
    var A = n_slots
    if gids_a == 0 and G != 1:
        return -2
    var ns = (A + 1) * G
    var mode = 0 if ns <= SMALL else 2
    var small = 1 if mode == 0 else 0
    var rpt = 2 if n_regs <= 6 else 1
    var tile = BLOCK * rpt
    var smem = n_regs * tile * 8 + MAX_CONSTS * 8 + WARPS * SMALL * 8 + code_ints * 4
    if smem > 48 * 1024:
        return -5
    try:
        var ctx = DeviceContext()
        var total = A * G
        if small == 0 or end <= start:
            ctx.enqueue_function[init_kernel](
                out_vals_a, out_cnt_a, ident_a, G, total,
                grid_dim=ceildiv(max(total, G), BLOCK), block_dim=BLOCK)
        if end > start:
            var blocks = min(ceildiv(end - start, tile), max(n_blocks, 1))
            for _ in range(max(repeat, 1)):
                if rpt == 2:
                    ctx.enqueue_function[pipeline_kernel[2]](
                        code_a, consts_a, set_bytes_a, set_meta_a, cols_a, dtypes_a, gids_a,
                        G, A, start, end, out_vals_a, out_cnt_a, scratch_a, n_regs, code_ints,
                        mode, grid_dim=blocks, block_dim=BLOCK, shared_mem_bytes=smem)
                else:
                    ctx.enqueue_function[pipeline_kernel[1]](
                        code_a, consts_a, set_bytes_a, set_meta_a, cols_a, dtypes_a, gids_a,
                        G, A, start, end, out_vals_a, out_cnt_a, scratch_a, n_regs, code_ints,
                        mode, grid_dim=blocks, block_dim=BLOCK, shared_mem_bytes=smem)
            if small != 0:
                ctx.enqueue_function[merge_kernel](
                    code_a, scratch_a, out_vals_a, out_cnt_a, blocks, G, A, agg_base, ident_a,
                    grid_dim=ceildiv((A + 1) * G, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
        return 0
    except:
        return -1
