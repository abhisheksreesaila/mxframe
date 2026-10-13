"""
MXFrame fused pipeline kernel, CPU side (F-008).

A vectorized interpreter in the style of MonetDB/X100: Python compiles a filter
predicate plus a list of aggregates into a small bytecode program
(`pipeline_program.py`); this kernel runs that program over a row range
[start, end) of raw column pointers, without JIT and without materialising
filtered columns.

Per worker thread, per vector of VEC (1024) rows:
  1. PRED ops evaluate the predicate over all rows of the vector into a
     register (1.0 / 0.0) and a selection vector is built from it.
  2. BODY ops evaluate the aggregate input expressions on the selected rows
     only (LOAD_COL gathers through the selection vector; dense when every row
     survived). Registers are cache-resident float64 vectors (SIMD).
  3. AGG ops accumulate into worker-local float64 partials per group.
Workers pull morsels from an atomic counter (dynamic balancing), and their
partials are merged at the end. All arithmetic is float64 (integers and
float32 columns are widened on load), so sums accumulate in float64.

Build (mirrors scripts/release.sh):
    pixi run mojo build kernels_aot/pipeline_cpu.mojo \
        --emit shared-lib -o kernels_aot/libmxpipeline_cpu.so

Program/ABI contract (keep in sync with pipeline_program.py):
  code   int32[]  : header (16 ints) then 5-int instructions [op,dst,a,b,c]
                    in the order prologue, pred, body, aggs.
  consts float64[], set_bytes uint8[], set_meta int64[] (offset, length, lo)*
  column dtypes: 0 i32, 1 i64, 2 f32, 3 f64, 4 u8, 5 | (slot << 8) = int32 dictionary codes plus the
  integer in chunk-table column `slot` (the native executor's string predicates, F-031)
  chunk table int64[n_chunks, 2 + n_cols]: row_start, row_end, then per column the
                    chunk's base address minus row_start * itemsize, so that
                    `addr + row * itemsize` is valid for every row of the chunk
                    (columns arrive as Arrow chunks; nothing is concatenated).
                    A single contiguous column set is one chunk [0, n).
  dtypes int32[] (0 i32,1 i64,2 f32,3 f64,4 u8)
  gids   int32[]  : group id per row (0 address = global aggregate, n_groups==1)
Outputs: out_vals float64[n_aggs * n_groups] (sum/min/max per agg and group),
         out_cnt int64[n_groups] (rows that passed the predicate per group).
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from max.algorithm import parallelize
from std.sys import simd_width_of
from std.atomic import Atomic
from std.math import floor
from std.ffi import external_call

comptime F64 = DType.float64
comptime W = simd_width_of[F64]()
comptime VEC = 1024
comptime MORSEL = 65536
comptime MAX_REGS = 64
comptime HDR = 16
comptime BUDGET_BYTES = 768 * 1024 * 1024
comptime MERGE_GROUPS = 16384

comptime FP = Pointer[Float64, MutUntrackedOrigin]
comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime BP = Pointer[UInt8, MutUntrackedOrigin]

# ── opcodes (mirror pipeline_program.py) ───────────────────────────────────
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
comptime OP_CMPC = 38
comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime OP_AGG_MAX = 52

comptime INF = Float64(1.7976931348623157e308) * 10.0


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


# ── column loads (widen to float64) ────────────────────────────────────────

@always_inline
def _load_dense[T: DType](dst: FP, addr: Int, i0: Int, n: Int):
    var src = Pointer[Scalar[T], MutUntrackedOrigin](unsafe_from_address=addr) + i0
    var i = 0
    while i + W <= n:
        dst.store(i, src.load[width=W](i).cast[F64]())
        i += W
    while i < n:
        dst[i] = src[i].cast[F64]()
        i += 1

@always_inline
def _load_gather[T: DType](dst: FP, addr: Int, i0: Int, sel: IP, n: Int):
    var src = Pointer[Scalar[T], MutUntrackedOrigin](unsafe_from_address=addr) + i0
    for j in range(n):
        dst[j] = src[Int(sel[j])].cast[F64]()

def _load_col(dst: FP, addr: Int, dtype: Int, i0: Int, sel: IP, use_sel: Bool, n: Int):
    if use_sel:
        if dtype == 0:
            _load_gather[DType.int32](dst, addr, i0, sel, n)
        elif dtype == 1:
            _load_gather[DType.int64](dst, addr, i0, sel, n)
        elif dtype == 2:
            _load_gather[DType.float32](dst, addr, i0, sel, n)
        elif dtype == 3:
            _load_gather[DType.float64](dst, addr, i0, sel, n)
        else:
            _load_gather[DType.uint8](dst, addr, i0, sel, n)
    else:
        if dtype == 0:
            _load_dense[DType.int32](dst, addr, i0, n)
        elif dtype == 1:
            _load_dense[DType.int64](dst, addr, i0, n)
        elif dtype == 2:
            _load_dense[DType.float32](dst, addr, i0, n)
        elif dtype == 3:
            _load_dense[DType.float64](dst, addr, i0, n)
        else:
            _load_dense[DType.uint8](dst, addr, i0, n)


# ── vector ops: n is padded up to a multiple of W by the caller ────────────

@always_inline
def _bin[OP: Int](d: FP, a: FP, b: FP, n: Int):
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        var x = a.load[width=W](i)
        var y = b.load[width=W](i)
        var r: SIMD[F64, W]
        comptime if OP == OP_ADD:
            r = x + y
        elif OP == OP_SUB:
            r = x - y
        elif OP == OP_MUL:
            r = x * y
        elif OP == OP_DIV:
            r = x / y
        elif OP == OP_LT:
            r = x.lt(y).select(one, zero)
        elif OP == OP_LE:
            r = x.le(y).select(one, zero)
        elif OP == OP_GT:
            r = x.gt(y).select(one, zero)
        elif OP == OP_GE:
            r = x.ge(y).select(one, zero)
        elif OP == OP_EQ:
            r = x.eq(y).select(one, zero)
        elif OP == OP_NE:
            r = x.eq(y).select(zero, one)
        elif OP == OP_AND:
            r = (x.ne(zero) & y.ne(zero)).select(one, zero)
        else:
            r = (x.ne(zero) | y.ne(zero)).select(one, zero)
        d.store(i, r)
        i += W

def _binop(op: Int, d: FP, a: FP, b: FP, n: Int):
    if op == OP_ADD:
        _bin[OP_ADD](d, a, b, n)
    elif op == OP_SUB:
        _bin[OP_SUB](d, a, b, n)
    elif op == OP_MUL:
        _bin[OP_MUL](d, a, b, n)
    elif op == OP_DIV:
        _bin[OP_DIV](d, a, b, n)
    elif op == OP_LT:
        _bin[OP_LT](d, a, b, n)
    elif op == OP_LE:
        _bin[OP_LE](d, a, b, n)
    elif op == OP_GT:
        _bin[OP_GT](d, a, b, n)
    elif op == OP_GE:
        _bin[OP_GE](d, a, b, n)
    elif op == OP_EQ:
        _bin[OP_EQ](d, a, b, n)
    elif op == OP_NE:
        _bin[OP_NE](d, a, b, n)
    elif op == OP_AND:
        _bin[OP_AND](d, a, b, n)
    else:
        _bin[OP_OR](d, a, b, n)

@always_inline
def _cmp_thr[OP: Int](d: FP, thr: Float64, n: Int):
    """In place: d = 1.0 if d <OP> thr else 0.0."""
    var t = SIMD[F64, W](thr)
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        var x = d.load[width=W](i)
        var r: SIMD[F64, W]
        comptime if OP == OP_LT:
            r = x.lt(t).select(one, zero)
        elif OP == OP_LE:
            r = x.le(t).select(one, zero)
        elif OP == OP_GT:
            r = x.gt(t).select(one, zero)
        elif OP == OP_GE:
            r = x.ge(t).select(one, zero)
        elif OP == OP_EQ:
            r = x.eq(t).select(one, zero)
        else:
            r = x.eq(t).select(zero, one)
        d.store(i, r)
        i += W

def _cmp_const(op: Int, d: FP, thr: Float64, n: Int):
    if op == OP_LT:
        _cmp_thr[OP_LT](d, thr, n)
    elif op == OP_LE:
        _cmp_thr[OP_LE](d, thr, n)
    elif op == OP_GT:
        _cmp_thr[OP_GT](d, thr, n)
    elif op == OP_GE:
        _cmp_thr[OP_GE](d, thr, n)
    elif op == OP_EQ:
        _cmp_thr[OP_EQ](d, thr, n)
    else:
        _cmp_thr[OP_NE](d, thr, n)

def _not(d: FP, a: FP, n: Int):
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        d.store(i, a.load[width=W](i).eq(zero).select(one, zero))
        i += W

def _floor(d: FP, a: FP, n: Int):
    var i = 0
    while i < n:
        d.store(i, floor(a.load[width=W](i)))
        i += W

def _select(d: FP, c: FP, a: FP, b: FP, n: Int):
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        d.store(i, c.load[width=W](i).ne(zero).select(a.load[width=W](i), b.load[width=W](i)))
        i += W

def _between(d: FP, x: FP, lo: FP, hi: FP, n: Int):
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        var v = x.load[width=W](i)
        d.store(i, (v.ge(lo.load[width=W](i)) & v.le(hi.load[width=W](i))).select(one, zero))
        i += W

def _in_set(d: FP, a: FP, bytes: BP, off: Int, length: Int, lo: Int, n: Int):
    for i in range(n):
        var x = a[i] - Float64(lo)
        var hit = 0.0
        if x >= 0.0 and x < Float64(length):
            if bytes[off + Int(x)] != 0:
                hit = 1.0
        d[i] = hit


# ── per-run immutable state ────────────────────────────────────────────────

struct Prog(Copyable, Movable):
    var code: IP
    var consts: FP
    var set_bytes: BP
    var set_meta: LP
    var tbl: LP
    var n_chunks: Int
    var tbl_stride: Int
    var dtypes: IP
    var gids: IP
    var n_groups: Int
    var has_gids: Bool
    var n_regs: Int
    var n_pro: Int
    var n_pred: Int
    var n_body: Int
    var n_aggs: Int
    var pred_reg: Int
    var nl: Int      # partial lanes per (agg, group)

    def __init__(out self, code: IP, consts: FP, set_bytes: BP, set_meta: LP,
                 tbl: LP, n_chunks: Int, dtypes: IP, gids: IP, n_groups: Int, has_gids: Bool):
        self.code = code
        self.consts = consts
        self.set_bytes = set_bytes
        self.set_meta = set_meta
        self.tbl = tbl
        self.n_chunks = n_chunks
        self.tbl_stride = 2 + Int(code[7])
        self.dtypes = dtypes
        self.gids = gids
        self.n_groups = n_groups
        self.has_gids = has_gids
        self.n_regs = Int(code[1])
        self.n_pro = Int(code[2])
        self.n_pred = Int(code[3])
        self.n_body = Int(code[4])
        self.n_aggs = Int(code[5])
        self.pred_reg = Int(code[6])
        if not has_gids:
            self.nl = W
        elif n_groups * (Int(code[5]) + 1) <= 2048:
            self.nl = 4
        else:
            self.nl = 1

    @always_inline
    def instr(self, base: Int, k: Int) -> IP:
        return self.code + (HDR + (base + k) * 5)


def _run_ops(p: Prog, regs: FP, cols: LP, base: Int, count: Int, i0: Int, sel: IP,
             use_sel: Bool, n: Int):
    var npad = ((n + W - 1) // W) * W
    for k in range(count):
        var ins = p.instr(base, k)
        var op = Int(ins[0])
        var d = regs + Int(ins[1]) * VEC
        if op == OP_LOAD_COL:
            var c = Int(ins[2])
            var dt = Int(p.dtypes[c])
            if (dt & 255) == 5:
                # dictionary codes of this chunk, shifted to its dictionary's place in the
                # concatenation of all chunks' dictionaries (the offset sits in tbl slot dt >> 8)
                _load_col(d, Int(cols[c]), 0, i0, sel, use_sel, n)
                var off = Float64(Int(cols[dt >> 8]))
                for i in range(n):
                    d[i] += off
            else:
                _load_col(d, Int(cols[c]), dt, i0, sel, use_sel, n)
        elif op == OP_LOAD_CONST:
            var v = SIMD[F64, W](p.consts[Int(ins[2])])
            var i = 0
            while i < VEC:
                d.store(i, v)
                i += W
        elif op == OP_CMPC:
            # column <cmp> constant: widen exactly, compare against the pre-rounded threshold
            var c = Int(ins[2])
            _load_col(d, Int(cols[c]), Int(p.dtypes[c]), i0, sel, use_sel, n)
            _cmp_const(Int(ins[4]), d, p.consts[Int(ins[3])], npad)
        elif op < OP_NOT:
            _binop(op, d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC, npad)
        elif op == OP_NOT:
            _not(d, regs + Int(ins[2]) * VEC, npad)
        elif op == OP_FLOOR:
            _floor(d, regs + Int(ins[2]) * VEC, npad)
        elif op == OP_SELECT:
            _select(d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC,
                    regs + Int(ins[4]) * VEC, npad)
        elif op == OP_BETWEEN:
            _between(d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC,
                     regs + Int(ins[4]) * VEC, npad)
        elif op == OP_IN_SET:
            var s = Int(ins[3])
            _in_set(d, regs + Int(ins[2]) * VEC, p.set_bytes, Int(p.set_meta[s * 3]),
                    Int(p.set_meta[s * 3 + 1]), Int(p.set_meta[s * 3 + 2]), n)


def _accumulate(p: Prog, regs: FP, gp: IP, n: Int, pv: FP, pc: LP):
    """Accumulate n compacted rows into the worker partials.

    Layout: pv[(a * G + g) * NL + lane], pc[g * NL + lane]; lane = j & (NL-1),
    which breaks the dependency chain when consecutive rows hit one group.
    """
    var G = p.n_groups
    var NL = p.nl
    var A = p.n_aggs
    if not p.has_gids:
        # Global aggregate: NL == W, accumulate straight into SIMD lanes.
        pc[0] += Int64(n)
        for a in range(A):
            var ins = p.instr(p.n_pro + p.n_pred + p.n_body, a)
            var kind = Int(ins[0])
            var r = regs + Int(ins[2]) * VEC
            var acc = pv + a * NL
            var accv = acc.load[width=W]()
            var j = 0
            while j + W <= n:
                var x = r.load[width=W](j)
                if kind == OP_AGG_SUM:
                    accv = accv + x
                elif kind == OP_AGG_MIN:
                    accv = min(accv, x)
                else:
                    accv = max(accv, x)
                j += W
            acc.store(accv)
            while j < n:
                var x = r[j]
                if kind == OP_AGG_SUM:
                    acc[0] += x
                elif kind == OP_AGG_MIN:
                    acc[0] = min(acc[0], x)
                else:
                    acc[0] = max(acc[0], x)
                j += 1
        return
    var lm = NL - 1
    for j in range(n):
        var g = Int(gp[j])
        if g >= 0 and g < G:
            pc[g * NL + (j & lm)] += 1
    for a in range(A):
        var ins = p.instr(p.n_pro + p.n_pred + p.n_body, a)
        var kind = Int(ins[0])
        var r = regs + Int(ins[2]) * VEC
        var base = pv + a * G * NL
        if kind == OP_AGG_SUM:
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    base[g * NL + (j & lm)] += r[j]
        elif kind == OP_AGG_MIN:
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    var q = base + (g * NL + (j & lm))
                    q[0] = min(q[0], r[j])
        else:
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    var q = base + (g * NL + (j & lm))
                    q[0] = max(q[0], r[j])


def _init_partials(p: Prog, pv: FP, pc: LP):
    var G = p.n_groups
    var NL = p.nl
    for a in range(p.n_aggs):
        var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
        var init = 0.0
        if kind == OP_AGG_MIN:
            init = INF
        elif kind == OP_AGG_MAX:
            init = -INF
        var base = pv + a * G * NL
        for i in range(G * NL):
            base[i] = init
    for i in range(G * NL):
        pc[i] = 0


def _find_chunk(p: Prog, row: Int) -> Int:
    """Index of the chunk holding `row` (binary search on row_start)."""
    var lo = 0
    var hi = p.n_chunks - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(p.tbl[mid * p.tbl_stride]) <= row:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _process_range(p: Prog, regs: FP, sel: IP, gscratch: IP, pv: FP, pc: LP,
                   lo: Int, hi: Int):
    var pos = lo
    while pos < hi:
        var k = _find_chunk(p, pos)
        var row = p.tbl + k * p.tbl_stride
        var cols = row + 2
        var stop = min(hi, Int(row[1]))
        var i0 = pos
        while i0 < stop:
            var n = min(VEC, stop - i0)
            var nsel = n
            var use_sel = False
            if p.pred_reg >= 0:
                _run_ops(p, regs, cols, p.n_pro, p.n_pred, i0, sel, False, n)
                var pr = regs + p.pred_reg * VEC
                var kk = 0
                for i in range(n):
                    sel[kk] = Int32(i)
                    kk += 1 if pr[i] != 0.0 else 0
                nsel = kk
                use_sel = nsel < n
            if nsel > 0:
                _run_ops(p, regs, cols, p.n_pro + p.n_pred, p.n_body, i0, sel, use_sel, nsel)
                var gp = gscratch
                if p.has_gids:
                    var gsrc = p.gids + i0
                    if use_sel:
                        for j in range(nsel):
                            gscratch[j] = gsrc[Int(sel[j])]
                    else:
                        gp = gsrc
                _accumulate(p, regs, gp, nsel, pv, pc)
            i0 += n
        pos = stop


@export
def mx_pipeline_cpu(
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    tbl_a: Int, n_chunks: Int, dtypes_a: Int, gids_a: Int, n_groups: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, threads: Int,
) abi("C") -> Int:
    var code = _ip(code_a)
    if Int(code[0]) != 0x4d585031:
        return -1
    var has_gids = gids_a != 0
    if (not has_gids) and n_groups != 1:
        return -2
    var p = Prog(code, _fp(consts_a), _bp(set_bytes_a), _lp(set_meta_a), _lp(tbl_a),
                 n_chunks, _ip(dtypes_a), _ip(gids_a), n_groups, has_gids)
    if p.n_regs > MAX_REGS:
        return -3
    if n_chunks < 1 and end > start:
        return -4
    var A = p.n_aggs
    var G = n_groups
    var NL = p.nl
    var out_vals = _fp(out_vals_a)
    var out_cnt = _lp(out_cnt_a)

    # identities for the output
    for a in range(A):
        var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
        var init = 0.0
        if kind == OP_AGG_MIN:
            init = INF
        elif kind == OP_AGG_MAX:
            init = -INF
        for g in range(G):
            out_vals[a * G + g] = init
    for g in range(G):
        out_cnt[g] = 0
    var total = end - start
    if total <= 0:
        return 0

    var per_worker = (A + 1) * G * NL * 8
    var nthreads = threads if threads > 0 else 1
    var morsels = (total + MORSEL - 1) // MORSEL
    nthreads = min(nthreads, morsels)
    if per_worker > 0:
        nthreads = min(nthreads, max(1, BUDGET_BYTES // per_worker))
    nthreads = max(nthreads, 1)

    var pv_all = unsafe_alloc[Float64](nthreads * max(A * G * NL, 1))
    var pc_all = unsafe_alloc[Int64](nthreads * max(G * NL, 1))
    var counter = unsafe_alloc[Int64](1)
    counter[0] = 0
    var pv_stride = max(A * G * NL, 1)
    var pc_stride = max(G * NL, 1)

    def worker(wid: Int) {imm}:
        var regs = unsafe_alloc[Float64](p.n_regs * VEC + W, alignment=64)
        for i in range(p.n_regs * VEC + W):
            regs[i] = 0.0
        var sel = unsafe_alloc[Int32](VEC)
        var gscratch = unsafe_alloc[Int32](VEC)
        var pv = pv_all + wid * pv_stride
        var pc = pc_all + wid * pc_stride
        _init_partials(p, pv, pc)
        # prologue (constants) once per worker
        _run_ops(p, regs, p.tbl, 0, p.n_pro, 0, sel, False, VEC)
        while True:
            var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            var lo = start + m * MORSEL
            if lo >= end:
                break
            _process_range(p, regs, sel, gscratch, pv, pc, lo, min(lo + MORSEL, end))
        regs.free()
        sel.free()
        gscratch.free()

    if nthreads == 1:
        worker(0)
    else:
        # A host process (Python via ctypes) has no Mojo async runtime yet:
        # create the global CPU device (thread pool) once, then parallelize.
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize(worker, nthreads)

    # merge worker partials into the output (parallel over group ranges when there are many groups)
    def merge(chunk: Int) {imm}:
        var g0 = chunk * MERGE_GROUPS
        var g1 = min(g0 + MERGE_GROUPS, G)
        for a in range(A):
            var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
            for g in range(g0, g1):
                var acc = out_vals[a * G + g]
                for t in range(nthreads):
                    var q = pv_all + t * pv_stride + (a * G + g) * NL
                    for l in range(NL):
                        if kind == OP_AGG_SUM:
                            acc += q[l]
                        elif kind == OP_AGG_MIN:
                            acc = min(acc, q[l])
                        else:
                            acc = max(acc, q[l])
                out_vals[a * G + g] = acc
        for g in range(g0, g1):
            var c = Int64(0)
            for t in range(nthreads):
                var q = pc_all + t * pc_stride + g * NL
                for l in range(NL):
                    c += q[l]
            out_cnt[g] = c

    var merge_chunks = (G + MERGE_GROUPS - 1) // MERGE_GROUPS
    if nthreads > 1 and merge_chunks > 1:
        parallelize(merge, merge_chunks, nthreads)
    else:
        for chunk in range(merge_chunks):
            merge(chunk)

    pv_all.free()
    pc_all.free()
    counter.free()
    return 0
