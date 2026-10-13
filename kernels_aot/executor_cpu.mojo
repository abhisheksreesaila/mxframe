"""
MXFrame native executor, CPU (F-030 / F-031, milestones M0-M1 of docs/design/mojo-executor.md).

ONE call runs a whole plan: Python compiles it once into a small int64 plan buffer and calls
`mx_execute`; everything that touches rows happens here. M0 covers the pipeline

    Scan (Arrow C Data Interface) -> Filter* -> GroupBy(keys) + Aggregate -> Having? -> Sort? -> Limit?

  1. columns arrive as ArrowArray structs (one per chunk per column); nothing is copied or
     converted on the way in;
  2. group keys (dictionary-encoded strings, int32, int64) become dense group ids, in parallel;
  3. filter + aggregate expressions run on the existing bytecode interpreter (pipeline_cpu.mojo);
  4. means / counts are finished; HAVING (aggregate <op> constant, AND-ed) drops groups; the groups
     are ordered (key order by default; ORDER BY any mix of keys and aggregates, stable, top-k
     when a LIMIT follows; strings by their bytes), cut to the LIMIT, and the output columns are
     written, string keys included, into one arena that Python wraps zero-copy and frees with
     `mx_exec_free`.

Key ranges up to `max dense groups` are addressed directly (fused per morsel up to 4096 groups);
up to `max compact range` the composite codes are compacted through a presence table and a prefix
sum (only the combinations that occur become groups); sparser key tuples go through a hash table
(groups numbered in order of first appearance). Strings (dictionary entries of every chunk, and
plain utf8 keys) are merged by hashing their bytes.

Anything the plan cannot express returns a negative code and Python runs its existing path.

Plan buffer (int64 words):
   0 magic 0x4d584531 ("MXE1")   1 version (2)
   2 n_chunks   3 n_cols (bytecode value columns)   4 n_keys   5 n_outputs   6 n_sort
   7 code*  8 consts*  9 set_bytes*  10 set_meta*  11 dtypes* (int32 per value column)
  12 arrays* : int64[n_chunks][n_cols + n_keys] = ArrowArray* (value columns, then keys)
  13 key_kinds* : int64[n_keys]  0 = dictionary<int32, utf8>, 1 = int32, 2 = int64, 3 = utf8
  14 outputs* : int64[n_outputs][3] = (op, slot, NaN-count slot or -1)  op 0 sum 1 min 2 max 3 mean 4 count
  15 sort* : int64[n_sort][3] = (what, index, descending)  what 0 = group key, 1 = output
  16 threads   17 max dense groups
  18 limit (-1: none)   19 n_having   20 having* : int64[n_having][3] = (output, cmp, 0)
     cmp 0 <  1 <=  2 >  3 >=  4 ==  5 !=
  21 max compact range (composite key range compacted through a presence table)
  22 having constants* : float64[n_having]
  23 n_dcols (dictionary-coded string columns in predicates)   24 doffs* : int64[n_chunks][n_dcols]
     (offset of the chunk's dictionary in the concatenation Python built; read through dtype 5)

Result (int64 words, caller-allocated, 3 + 2 * n_keys + n_outputs):
   0 n_rows   1 arena* (free with mx_exec_free)   2 flags (bit 0: global aggregate over no rows)
   then per key (A, B): dictionary key -> (int32 offsets[n+1]*, utf8 data*); int32 / int64 key -> (values*, 0)
   then per output: float64[n]* (int64[n]* for count)

Return: 0 ok; < 0 declined or failed (the caller falls back).
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer, unsafe_memset_zero
from max.algorithm import parallelize
from std.atomic import Atomic
from std.ffi import external_call

from std.utils.numerics import nan

from pipeline_cpu import mx_pipeline_cpu, Prog, _run_ops, _find_chunk, VEC, MAX_REGS

comptime FP = Pointer[Float64, MutUntrackedOrigin]
comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime BP = Pointer[UInt8, MutUntrackedOrigin]

comptime MAGIC = 0x4D584531
comptime MORSEL = 65536

comptime K_DICT = 0
comptime K_I32 = 1
comptime K_I64 = 2
comptime K_STR = 3     # utf8 (int32 offsets), not dictionary-encoded: hashed to entry ids in Mojo

comptime O_SUM = 0
comptime O_MIN = 1
comptime O_MAX = 2
comptime O_MEAN = 3
comptime O_COUNT = 4

comptime E_BAD_PLAN = -1
comptime E_TOO_MANY_GROUPS = -10
comptime E_BAD_KEY = -11
comptime E_PIPELINE = -12
comptime E_DICT_TOO_BIG = -13
comptime E_OUTPUT_TOO_BIG = -14   # string keys past int32 offsets
comptime E_NAN_SORT = -15         # NaN among ORDER BY values
comptime FUSE_GROUPS = 4096   # up to this many groups: keys encoded and aggregated per morsel
comptime HDR = 16
comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime INF = Float64(1.7976931348623157e308) * 10.0


@always_inline
def _lp(a: Int) -> LP:
    return LP(unsafe_from_address=a)

@always_inline
def _ip(a: Int) -> IP:
    return IP(unsafe_from_address=a)

@always_inline
def _bp(a: Int) -> BP:
    return BP(unsafe_from_address=a)


# ── Arrow C Data Interface: struct ArrowArray is ten 8-byte words ───────────────────────────────
# length, null_count, offset, n_buffers, n_children, buffers**, children**, dictionary*, release, private

@always_inline
def _arr_len(a: Int) -> Int:
    return Int(_lp(a)[0])

@always_inline
def _arr_off(a: Int) -> Int:
    return Int(_lp(a)[2])

@always_inline
def _arr_buf(a: Int, i: Int) -> Int:
    return Int(_lp(Int(_lp(a)[5]))[i])

@always_inline
def _arr_dict(a: Int) -> Int:
    return Int(_lp(a)[7])


def _itemsize(dtype: Int) -> Int:
    # pipeline dtypes: 0 i32, 1 i64, 2 f32, 3 f64, 4 u8, 5 dictionary codes (int32)
    if dtype == 1 or dtype == 3:
        return 8
    if dtype == 4:
        return 1
    return 4


struct Keys(Copyable, Movable):
    """Per-key description: kind, value range (base, card) and dense stride. Dictionary keys
    also get a global dictionary (entry -> chunk, code) and per-chunk remaps (code -> entry):
    each Parquet row group carries its own dictionary, in its own order."""
    var n: Int
    var n_chunks: Int
    var kinds: LP
    var base: LP
    var card: LP
    var stride: LP
    var gchunk: LP      # per key: address of int64[card] (source chunk of each global entry)
    var gcode: LP       # per key: address of int64[card] (code in that chunk's dictionary)
    var remap: LP       # per key: address of int64[n_chunks] of addresses of int32[dict len]
    var rowcode: LP     # per utf8 key: address of int32[n_rows] (entry id of every row)

    def __init__(out self, n: Int, n_chunks: Int):
        self.n = n
        self.n_chunks = n_chunks
        self.kinds = unsafe_alloc[Int64](max(n, 1))
        self.base = unsafe_alloc[Int64](max(n, 1))
        self.card = unsafe_alloc[Int64](max(n, 1))
        self.stride = unsafe_alloc[Int64](max(n, 1))
        self.gchunk = unsafe_alloc[Int64](max(n, 1))
        self.gcode = unsafe_alloc[Int64](max(n, 1))
        self.remap = unsafe_alloc[Int64](max(n, 1))
        self.rowcode = unsafe_alloc[Int64](max(n, 1))
        for k in range(max(n, 1)):
            self.gchunk[k] = 0
            self.gcode[k] = 0
            self.remap[k] = 0
            self.rowcode[k] = 0

    def free(self):
        for k in range(self.n):
            if Int(self.gchunk[k]) != 0:
                _lp(Int(self.gchunk[k])).free()
                _lp(Int(self.gcode[k])).free()
            if Int(self.remap[k]) != 0:
                var rm = _lp(Int(self.remap[k]))
                var prev = 0
                for c in range(self.n_chunks):
                    var a = Int(rm[c])
                    if a != 0 and a != prev:      # chunks sharing a dictionary share a remap
                        _ip(a).free()
                    prev = a
                rm.free()
            if Int(self.rowcode[k]) != 0:
                _ip(Int(self.rowcode[k])).free()
        self.rowcode.free()
        self.kinds.free()
        self.base.free()
        self.card.free()
        self.stride.free()
        self.gchunk.free()
        self.gcode.free()
        self.remap.free()


@always_inline
def _key_value(kind: Int, arr: Int, i: Int) -> Int:
    """Value of row i (chunk-local) of a key array: dictionary code or integer."""
    var off = _arr_off(arr)
    var data = _arr_buf(arr, 1)
    if kind == K_I64:
        return Int(_lp(data)[off + i])
    return Int(_ip(data)[off + i])


def _int_range(arrays: LP, stride_a: Int, n_chunks: Int, col: Int, kind: Int,
               mut lo: Int, mut hi: Int):
    lo = Int(Int64.MAX)
    hi = Int(Int64.MIN)
    for c in range(n_chunks):
        var arr = Int(arrays[c * stride_a + col])
        var n = _arr_len(arr)
        for i in range(n):
            var v = _key_value(kind, arr, i)
            lo = min(lo, v)
            hi = max(hi, v)


@always_inline
def _dstart(d: Int, code: Int) -> Int:
    var offs = _ip(_arr_buf(d, 1)) + _arr_off(d)
    return _arr_buf(d, 2) + Int(offs[code])

@always_inline
def _dlen(d: Int, code: Int) -> Int:
    var offs = _ip(_arr_buf(d, 1)) + _arr_off(d)
    return Int(offs[code + 1]) - Int(offs[code])


def _bytes_cmp(a: Int, la: Int, b: Int, lb: Int) -> Int:
    var pa = _bp(a)
    var pb = _bp(b)
    for i in range(min(la, lb)):
        if pa[i] != pb[i]:
            return -1 if pa[i] < pb[i] else 1
    if la == lb:
        return 0
    return -1 if la < lb else 1


@always_inline
def _same_dict(d1: Int, d2: Int) -> Bool:
    return (_arr_len(d1) == _arr_len(d2) and _arr_off(d1) == _arr_off(d2)
            and _arr_buf(d1, 1) == _arr_buf(d2, 1) and _arr_buf(d1, 2) == _arr_buf(d2, 2))


@always_inline
def _mix(h: UInt64) -> UInt64:
    """splitmix64 finalizer."""
    var z = h
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def _hash_bytes(a: Int, n: Int) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)          # FNV-1a, then mixed
    var p = _bp(a)
    for i in range(n):
        h = (h ^ UInt64(p[i])) * 0x100000001B3
    return _mix(h ^ UInt64(n))


def _pow2_at_least(n: Int) -> Int:
    var t = 1024
    while t < n:
        t *= 2
    return t


@always_inline
def _entry_src(kind: Int, arrays: LP, stride_a: Int, col: Int, chunk: Int) -> Int:
    """The ArrowArray whose (offsets, bytes) hold a string key's entries in `chunk`: the chunk's
    dictionary for dictionary keys, the chunk itself for utf8 keys (same buffer layout)."""
    var arr = Int(arrays[chunk * stride_a + col])
    return _arr_dict(arr) if kind == K_DICT else arr


def _dict_entries(arrays: LP, stride_a: Int, n_chunks: Int, col: Int) -> Int:
    """Total dictionary entries over the chunks (chunks sharing a dictionary counted once)."""
    var total = 0
    var prev_d = 0
    for c in range(n_chunks):
        var d = _arr_dict(Int(arrays[c * stride_a + col]))
        if c > 0 and _same_dict(d, prev_d):
            continue
        prev_d = d
        total += _arr_len(d)
    return total


def _build_global_dict(arrays: LP, stride_a: Int, n_chunks: Int, col: Int, cap: Int,
                       gchunk: LP, gcode: LP, remaps: LP) -> Int:
    """Merge the chunks' dictionaries by hashing their strings: returns the number of distinct
    strings (or -1 past `cap`). remaps[c] = address of int32[len(dict of chunk c)] mapping chunk
    codes to global entries (first-appearance order)."""
    var tsize = _pow2_at_least(2 * cap + 2)
    var table = unsafe_alloc[Int32](tsize)
    for i in range(tsize):
        table[i] = -1
    var ng = 0
    var prev_d = 0
    for c in range(n_chunks):
        var d = _arr_dict(Int(arrays[c * stride_a + col]))
        if c > 0 and _same_dict(d, prev_d):
            remaps[c] = remaps[c - 1]
            continue
        prev_d = d
        var nd = _arr_len(d)
        var rm = unsafe_alloc[Int32](max(nd, 1))
        remaps[c] = Int64(Int(rm))
        for code in range(nd):
            var s0 = _dstart(d, code)
            var l0 = _dlen(d, code)
            var pos = Int(_hash_bytes(s0, l0) & UInt64(tsize - 1))
            var e = -1
            while True:
                e = Int(table[pos])
                if e < 0:
                    if ng >= cap:
                        table.free()
                        return -1
                    gchunk[ng] = Int64(c)
                    gcode[ng] = Int64(code)
                    table[pos] = Int32(ng)
                    e = ng
                    ng += 1
                    break
                var dg = _arr_dict(Int(arrays[Int(gchunk[e]) * stride_a + col]))
                if _bytes_cmp(s0, l0, _dstart(dg, Int(gcode[e])), _dlen(dg, Int(gcode[e]))) == 0:
                    break
                pos = (pos + 1) & (tsize - 1)
            rm[code] = Int32(e)
    table.free()
    return ng


def _build_row_strings(arrays: LP, stride_a: Int, n_chunks: Int, col: Int, tbl: LP, tstride: Int,
                       n_rows: Int, gchunk: LP, gcode: LP, rowcode: IP) -> Int:
    """Distinct strings of a utf8 key column by hashing: entries (chunk, row) in first-appearance
    order; rowcode[i] = entry of row i. Returns the number of entries."""
    var tsize = _pow2_at_least(2 * n_rows + 2)
    var table = unsafe_alloc[Int32](tsize)
    for i in range(tsize):
        table[i] = -1
    var ng = 0
    for c in range(n_chunks):
        var arr = Int(arrays[c * stride_a + col])
        var c0 = Int(tbl[c * tstride])
        var n = _arr_len(arr)
        for r in range(n):
            var s0 = _dstart(arr, r)
            var l0 = _dlen(arr, r)
            var pos = Int(_hash_bytes(s0, l0) & UInt64(tsize - 1))
            var e = -1
            while True:
                e = Int(table[pos])
                if e < 0:
                    gchunk[ng] = Int64(c)
                    gcode[ng] = Int64(r)
                    table[pos] = Int32(ng)
                    e = ng
                    ng += 1
                    break
                var ag = Int(arrays[Int(gchunk[e]) * stride_a + col])
                if _bytes_cmp(s0, l0, _dstart(ag, Int(gcode[e])), _dlen(ag, Int(gcode[e]))) == 0:
                    break
                pos = (pos + 1) & (tsize - 1)
            rowcode[c0 + r] = Int32(e)
    table.free()
    return ng


def _global_ranks(kind: Int, arrays: LP, stride_a: Int, col: Int, n: Int, gchunk: LP, gcode: LP,
                  ranks: LP):
    """ranks[entry] = position of string entry `entry` in byte order (merge sort)."""
    var order = unsafe_alloc[Int64](max(n, 1))
    var tmp = unsafe_alloc[Int64](max(n, 1))
    for i in range(n):
        order[i] = Int64(i)
    var src = order
    var dst = tmp
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                var ei = Int(src[i])
                var ej = Int(src[j])
                var di = _entry_src(kind, arrays, stride_a, col, Int(gchunk[ei]))
                var dj = _entry_src(kind, arrays, stride_a, col, Int(gchunk[ej]))
                if _bytes_cmp(_dstart(dj, Int(gcode[ej])), _dlen(dj, Int(gcode[ej])),
                              _dstart(di, Int(gcode[ei])), _dlen(di, Int(gcode[ei]))) < 0:
                    dst[k] = src[j]
                    j += 1
                else:
                    dst[k] = src[i]
                    i += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
            lo += 2 * width
        var t = src
        src = dst
        dst = t
        width *= 2
    for r in range(n):
        ranks[Int(src[r])] = Int64(r)
    order.free()
    tmp.free()


def _encode_keys(K: Keys, arrays: LP, stride_a: Int, n_cols: Int, n_keys: Int, tbl: LP,
                tstride: Int, lo: Int, hi: Int, gids: IP) -> Bool:
    """gids[i] for rows i in [lo, hi): dense group id, one tight loop per key column. False when a
    dictionary code is outside its chunk's dictionary (invalid Arrow input): nothing is read past
    the remap, and the caller declines."""
    var c = 0
    while Int(tbl[c * tstride + 1]) <= lo:
        c += 1
    var r = lo
    while r < hi:
        var c0 = Int(tbl[c * tstride])
        var c1 = min(hi, Int(tbl[c * tstride + 1]))
        for k in range(n_keys):
            var arr = Int(arrays[c * stride_a + n_cols + k])
            var kind = Int(K.kinds[k])
            var sk = Int32(Int(K.stride[k]))
            var first = k == 0
            if kind == K_I64:
                var src = _lp(_arr_buf(arr, 1) + (_arr_off(arr) - c0) * 8)
                var b64 = Int64(Int(K.base[k]))
                for i in range(r, c1):
                    var t = Int32(src[i] - b64) * sk
                    gids[i] = t if first else gids[i] + t
            elif kind == K_STR:
                var rcode = _ip(Int(K.rowcode[k]))
                for i in range(r, c1):
                    var t = rcode[i] * sk
                    gids[i] = t if first else gids[i] + t
            elif kind == K_DICT:
                var src = _ip(_arr_buf(arr, 1) + (_arr_off(arr) - c0) * 4)
                var m = _ip(Int(_lp(Int(K.remap[k]))[c]))
                var nd = Int32(_arr_len(_arr_dict(arr)))
                var bad = False
                for i in range(r, c1):
                    bad = bad or src[i] < 0 or src[i] >= nd
                if bad:
                    return False
                for i in range(r, c1):
                    var t = m[Int(src[i])] * sk
                    gids[i] = t if first else gids[i] + t
            else:
                var src = _ip(_arr_buf(arr, 1) + (_arr_off(arr) - c0) * 4)
                var bk = Int32(Int(K.base[k]))
                for i in range(r, c1):
                    var t = (src[i] - bk) * sk
                    gids[i] = t if first else gids[i] + t
        r = c1
        c += 1
    return True


@export
def mx_exec_abi_version() abi("C") -> Int:
    return 1


@export
def mx_exec_free(arena: Int) abi("C") -> Int:
    if arena != 0:
        _bp(arena).free()
    return 0


@export
def mx_execute(plan_a: Int, out_a: Int) abi("C") -> Int:
    var P = _lp(plan_a)
    if Int(P[0]) != MAGIC or Int(P[1]) != 2:
        return E_BAD_PLAN
    var n_chunks = Int(P[2])
    var n_cols = Int(P[3])
    var n_keys = Int(P[4])
    var n_out = Int(P[5])
    var n_sort = Int(P[6])
    var code = Int(P[7])
    var consts = Int(P[8])
    var set_bytes = Int(P[9])
    var set_meta = Int(P[10])
    var dtypes = _ip(Int(P[11]))
    var arrays = _lp(Int(P[12]))
    var key_kinds = _lp(Int(P[13]))
    var outs = _lp(Int(P[14]))
    var sorts = _lp(Int(P[15]))
    var threads = max(1, Int(P[16]))
    var max_groups = Int(P[17])
    var limit = Int(P[18])
    var n_having = Int(P[19])
    var having = _lp(Int(P[20]))
    var max_compact = Int(P[21])
    var hconst = FP(unsafe_from_address=Int(P[22]))
    var n_dcols = Int(P[23])
    var doffs = _lp(Int(P[24]))
    var O = _lp(out_a)
    var stride_a = n_cols + n_keys
    if n_chunks < 1:
        return E_BAD_PLAN

    # ── 1. chunk table for the interpreter: row_start, row_end, per column base address ──────
    var tstride = 2 + n_cols + n_dcols
    var tbl = unsafe_alloc[Int64](n_chunks * tstride)
    var row = 0
    for c in range(n_chunks):
        var first = Int(arrays[c * stride_a])
        var n = _arr_len(first)
        tbl[c * tstride] = Int64(row)
        tbl[c * tstride + 1] = Int64(row + n)
        for j in range(n_cols):
            var arr = Int(arrays[c * stride_a + j])
            var isz = _itemsize(Int(dtypes[j]) & 255)
            tbl[c * tstride + 2 + j] = Int64(_arr_buf(arr, 1) + (_arr_off(arr) - row) * isz)
        for i in range(n_dcols):
            tbl[c * tstride + 2 + n_cols + i] = doffs[c * n_dcols + i]
        row += n
    var n_rows = row

    # ── 2. group keys -> dense group ids ─────────────────────────────────────────────────────
    var K = Keys(n_keys, n_chunks)
    var D = 1
    var use_hash = False
    for k in range(n_keys):
        var kind = Int(key_kinds[k])
        K.kinds[k] = Int64(kind)
        var col = n_cols + k
        if kind == K_DICT:
            var cap = max(_dict_entries(arrays, stride_a, n_chunks, col), 1)
            var gch = unsafe_alloc[Int64](cap)
            var gco = unsafe_alloc[Int64](cap)
            var rms = unsafe_alloc[Int64](n_chunks)
            for c in range(n_chunks):
                rms[c] = 0
            K.gchunk[k] = Int64(Int(gch))
            K.gcode[k] = Int64(Int(gco))
            K.remap[k] = Int64(Int(rms))
            var ng = _build_global_dict(arrays, stride_a, n_chunks, col, cap, gch, gco, rms)
            if ng < 0:
                K.free()
                tbl.free()
                return E_DICT_TOO_BIG
            K.base[k] = 0
            K.card[k] = Int64(max(ng, 1))
        elif kind == K_STR:
            var gch = unsafe_alloc[Int64](max(n_rows, 1))
            var gco = unsafe_alloc[Int64](max(n_rows, 1))
            var rcode = unsafe_alloc[Int32](max(n_rows, 1))
            K.gchunk[k] = Int64(Int(gch))
            K.gcode[k] = Int64(Int(gco))
            K.rowcode[k] = Int64(Int(rcode))
            var ng = _build_row_strings(arrays, stride_a, n_chunks, col, tbl, tstride, n_rows, gch, gco, rcode)
            K.base[k] = 0
            K.card[k] = Int64(max(ng, 1))
        elif kind == K_I32 or kind == K_I64:
            var lo = 0
            var hi = -1
            _int_range(arrays, stride_a, n_chunks, col, kind, lo, hi)
            if hi < lo:          # no rows
                lo = 0
                hi = 0
            K.base[k] = Int64(lo)
            K.card[k] = Int64(hi - lo + 1)
        else:
            K.free()
            tbl.free()
            return E_BAD_KEY
        if Int(K.card[k]) <= 0:
            K.free()
            tbl.free()
            return E_BAD_KEY
        if use_hash or D > max(max_groups, max_compact) // Int(K.card[k]):
            use_hash = True                       # too sparse for direct addressing: hash the tuples
        else:
            D *= Int(K.card[k])
    var s = 1
    for k in range(n_keys - 1, -1, -1):
        K.stride[k] = Int64(s)
        s *= Int(K.card[k]) if not use_hash else 1

    var grouped = n_keys > 0
    var compact = grouped and not use_hash and D > max_groups   # composite codes compacted to the groups present
    var G = D if (grouped and not use_hash) else 1
    var code_p = _ip(code)
    var n_slots = Int(code_p[5])
    var vals = unsafe_alloc[Float64](1 if (compact or use_hash) else max(n_slots * G, 1))
    var cnt = unsafe_alloc[Int64](1 if (compact or use_hash) else G)
    var comp = unsafe_alloc[Int64](1)             # group id -> composite key code (set below)
    var kv = unsafe_alloc[Int64](max(n_keys, 1))   # per key: address of int64[G] key codes per group
    for k in range(max(n_keys, 1)):
        kv[k] = 0
    var rc = 0
    var morsels = (n_rows + MORSEL - 1) // MORSEL
    var nt = max(1, min(threads, morsels))

    if not grouped:
        rc = mx_pipeline_cpu(code, consts, set_bytes, set_meta, Int(tbl), n_chunks, Int(dtypes),
                             0, 1, 0, n_rows, Int(vals), Int(cnt), threads)
    elif not compact and not use_hash and G <= FUSE_GROUPS:
        # ── fused: per 64k-row morsel, encode keys into a small reused buffer and aggregate it
        #    right away (no n_rows-sized group-id array: no page faults, cache-resident)
        var agg_base = HDR + (Int(code_p[2]) + Int(code_p[3]) + Int(code_p[4])) * 5
        var acc_v = unsafe_alloc[Float64](nt * max(n_slots * G, 1))
        var acc_c = unsafe_alloc[Int64](nt * G)
        var vstride = max(n_slots * G, 1)
        var failed = unsafe_alloc[Int64](1)
        failed[0] = 0
        var counter = unsafe_alloc[Int64](1)
        counter[0] = 0

        def fused_worker(wid: Int) {imm}:
            var gbuf = unsafe_alloc[Int32](MORSEL)
            var tv = unsafe_alloc[Float64](vstride)
            var tc = unsafe_alloc[Int64](G)
            var av = acc_v + wid * vstride
            var ac = acc_c + wid * G
            for a in range(n_slots):
                var kind = Int(code_p[agg_base + a * 5])
                var init = 0.0 if kind == OP_AGG_SUM else (INF if kind == OP_AGG_MIN else -INF)
                for g in range(G):
                    av[a * G + g] = init
            for g in range(G):
                ac[g] = 0
            while True:
                var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
                var lo = m * MORSEL
                if lo >= n_rows:
                    break
                var hi = min(lo + MORSEL, n_rows)
                var gbase = _ip(Int(gbuf) - lo * 4)          # gbase[i] valid for i in [lo, hi)
                if not _encode_keys(K, arrays, stride_a, n_cols, n_keys, tbl, tstride, lo, hi, gbase):
                    failed[0] = 2
                    break
                var r = mx_pipeline_cpu(code, consts, set_bytes, set_meta, Int(tbl), n_chunks,
                                        Int(dtypes), Int(gbase), G, lo, hi, Int(tv), Int(tc), 1)
                if r < 0:
                    failed[0] = 1
                    break
                for a in range(n_slots):
                    var kind = Int(code_p[agg_base + a * 5])
                    for g in range(G):
                        var x = tv[a * G + g]
                        var i = a * G + g
                        if kind == OP_AGG_SUM:
                            av[i] += x
                        elif kind == OP_AGG_MIN:
                            av[i] = min(av[i], x)
                        else:
                            av[i] = max(av[i], x)
                for g in range(G):
                    ac[g] += tc[g]
            gbuf.free()
            tv.free()
            tc.free()

        if nt > 1:
            _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
            parallelize(fused_worker, nt)
        else:
            fused_worker(0)
        for a in range(n_slots):
            var kind = Int(code_p[agg_base + a * 5])
            for g in range(G):
                var acc = acc_v[a * G + g]
                for w in range(1, nt):
                    var x = acc_v[w * vstride + a * G + g]
                    if kind == OP_AGG_SUM:
                        acc += x
                    elif kind == OP_AGG_MIN:
                        acc = min(acc, x)
                    else:
                        acc = max(acc, x)
                vals[a * G + g] = acc
        for g in range(G):
            var c = Int64(0)
            for w in range(nt):
                c += acc_c[w * G + g]
            cnt[g] = c
        rc = 0 if Int(failed[0]) == 0 else (E_BAD_KEY if Int(failed[0]) == 2 else E_PIPELINE)
        acc_v.free()
        acc_c.free()
        failed.free()
        counter.free()
    else:
        # ── many groups: one group-id array, then one interpreter pass (M1 replaces this with a
        #    hash-table aggregate)
        var gids = unsafe_alloc[Int32](max(n_rows, 1))
        var counter = unsafe_alloc[Int64](1)
        counter[0] = 0
        var bad_key = unsafe_alloc[Int64](1)
        bad_key[0] = 0

        def keys_worker(wid: Int) {imm}:
            while True:
                var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
                var lo = m * MORSEL
                if lo >= n_rows:
                    break
                if not _encode_keys(K, arrays, stride_a, n_cols, n_keys, tbl, tstride, lo, min(lo + MORSEL, n_rows), gids):
                    bad_key[0] = 1

        if use_hash:
            # sparse key tuples: a hash table numbers the groups in order of first appearance
            for k in range(n_keys):
                kv[k] = Int64(Int(unsafe_alloc[Int64](max(n_rows, 1))))
            var ng = _hash_groups(K, arrays, stride_a, n_cols, n_keys, tbl, tstride, n_chunks, gids, kv, n_rows)
            if ng < 0:
                bad_key[0] = 1
            else:
                G = max(ng, 1)
                vals.free()
                cnt.free()
                vals = unsafe_alloc[Float64](max(n_slots * G, 1))
                cnt = unsafe_alloc[Int64](G)
                if nt > 1:
                    _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        elif nt > 1:
            _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
            parallelize(keys_worker, nt)
        else:
            keys_worker(0)
        counter.free()
        if Int(bad_key[0]) == 0 and compact:
            # only the combinations that occur become groups, numbered in composite (key) order
            var pres = unsafe_alloc[UInt8](D)
            unsafe_memset_zero(pres, D)
            for i in range(n_rows):
                pres[Int(gids[i])] = 1
            var dense = unsafe_alloc[Int32](D)
            var ng = 0
            for c in range(D):
                if pres[c] != 0:
                    dense[c] = Int32(ng)
                    ng += 1
            comp.free()
            comp = unsafe_alloc[Int64](max(ng, 1))
            for c in range(D):
                if pres[c] != 0:
                    comp[Int(dense[c])] = Int64(c)
            pres.free()
            var counter2 = unsafe_alloc[Int64](1)
            counter2[0] = 0

            def remap_worker(wid: Int) {imm}:
                while True:
                    var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter2, 1))
                    var lo = m * MORSEL
                    if lo >= n_rows:
                        break
                    for i in range(lo, min(lo + MORSEL, n_rows)):
                        gids[i] = dense[Int(gids[i])]

            if nt > 1:
                parallelize(remap_worker, nt)
            else:
                remap_worker(0)
            counter2.free()
            dense.free()
            G = max(ng, 1)
            vals.free()
            cnt.free()
            vals = unsafe_alloc[Float64](max(n_slots * G, 1))
            cnt = unsafe_alloc[Int64](G)
        if Int(bad_key[0]) != 0:
            rc = E_BAD_KEY
        else:
            rc = mx_pipeline_cpu(code, consts, set_bytes, set_meta, Int(tbl), n_chunks, Int(dtypes),
                                 Int(gids), G, 0, n_rows, Int(vals), Int(cnt), threads)
        bad_key.free()
        gids.free()
    tbl.free()
    if rc < 0:
        vals.free()
        cnt.free()
        comp.free()
        _free_kv(kv, n_keys)
        K.free()
        return rc if rc == E_BAD_KEY else E_PIPELINE
    if grouped and not use_hash:
        # per-key codes of every group, from its composite code (dense: the group id itself)
        for k in range(n_keys):
            var a = unsafe_alloc[Int64](G)
            var st = Int(K.stride[k])
            var cd = Int(K.card[k])
            for g in range(G):
                var c = Int(comp[g]) if compact else g
                a[g] = Int64((c // st) % cd)
            kv[k] = Int64(Int(a))
    comp.free()

    # ── 4. which groups survive, in output order ─────────────────────────────────────────────
    var order = unsafe_alloc[Int64](max(G, 1))     # output position -> group id
    var n_res = 0
    if not grouped:
        order[0] = 0
        n_res = 1
    else:
        for g in range(G):                         # key order (hash path: first appearance)
            if cnt[g] > 0:
                order[n_res] = Int64(g)
                n_res += 1
    var has_str_key = False
    for k in range(n_keys):
        var kd = Int(K.kinds[k])
        has_str_key = has_str_key or kd == K_DICT or kd == K_STR
    if (use_hash or has_str_key) and n_res > 1:
        # key order (pandas' groupby default), which is also the ORDER BY tie-break: the hash table
        # numbers groups by first appearance, and string codes are entry ids (first appearance in
        # the dictionaries), so those groups are reordered by the keys' values (strings by bytes);
        # integer keys on the dense / compacted paths are already in order
        var kvals = unsafe_alloc[Float64](n_keys * n_res)
        for k in range(n_keys):
            _key_sort_values(K, k, _lp(Int(kv[k])), order, n_res, arrays, stride_a, n_cols,
                             kvals + k * n_res, 1.0)
        var kperm = unsafe_alloc[Int64](n_res)
        for i in range(n_res):
            kperm[i] = Int64(i)
        _merge_sort(kperm, n_res, kvals, n_keys, n_res)
        var korder = unsafe_alloc[Int64](max(G, 1))
        for i in range(n_res):
            korder[i] = order[Int(kperm[i])]
        order.free()
        order = korder
        kperm.free()
        kvals.free()
    var empty_global = not grouped and cnt[0] == 0
    if n_having > 0:
        var kept = 0
        for i in range(n_res):
            var g = Int(order[i])
            var ok = True
            for h in range(n_having):
                var j = Int(having[3 * h])
                var op = Int(outs[3 * j])
                if empty_global and op != O_COUNT:    # NULL compares false
                    ok = False
                    break
                if not _cmp(_out_value(op, Int(outs[3 * j + 1]), Int(outs[3 * j + 2]), g, G, vals, cnt),
                            Int(having[3 * h + 1]), hconst[h]):
                    ok = False
                    break
            if ok:
                order[kept] = Int64(g)
                kept += 1
        n_res = kept

    if n_sort > 0 and n_res > 1:
        # one float64 sort value per (sort column, row): key ranks or aggregate values, negated
        # when descending; ties fall back to key order, so the order is total and deterministic
        var sv = unsafe_alloc[Float64](n_sort * n_res)
        var bad = False
        for j in range(n_sort):
            var what = Int(sorts[3 * j])
            var idx = Int(sorts[3 * j + 1])
            var sign = -1.0 if Int(sorts[3 * j + 2]) != 0 else 1.0
            if what == 0:
                _key_sort_values(K, idx, _lp(Int(kv[idx])), order, n_res, arrays, stride_a, n_cols,
                                 sv + j * n_res, sign)
            else:
                var op = Int(outs[3 * idx])
                var slot = Int(outs[3 * idx + 1])
                var nslot = Int(outs[3 * idx + 2])
                for i in range(n_res):
                    var x = _out_value(op, slot, nslot, Int(order[i]), G, vals, cnt)
                    if x != x:
                        bad = True
                    sv[j * n_res + i] = sign * x
        if bad:                                    # NaN: the caller's sort decides where it goes
            sv.free()
            order.free()
            vals.free()
            cnt.free()
            _free_kv(kv, n_keys)
            K.free()
            return E_NAN_SORT
        var perm = unsafe_alloc[Int64](n_res)
        var n_keep = n_res
        if limit >= 0 and limit * 8 < n_res:
            n_keep = _top_k(perm, n_res, limit, sv, n_sort)
        else:
            for i in range(n_res):
                perm[i] = Int64(i)
        _merge_sort(perm, n_keep, sv, n_sort, n_res)
        var sorted = unsafe_alloc[Int64](max(n_keep, 1))
        for i in range(n_keep):
            sorted[i] = order[Int(perm[i])]
        order.free()
        order = sorted
        n_res = n_keep
        perm.free()
        sv.free()
    if limit >= 0 and limit < n_res:
        n_res = limit

    # ── 5. output arena ─────────────────────────────────────────────────────────────────────
    var key_bytes = unsafe_alloc[Int64](max(n_keys, 1))
    var total = 0
    for k in range(n_keys):
        var kind = Int(K.kinds[k])
        if kind == K_DICT or kind == K_STR:
            var gch = _lp(Int(K.gchunk[k]))
            var gco = _lp(Int(K.gcode[k]))
            var kvk = _lp(Int(kv[k]))
            var nb = 0
            for i in range(n_res):
                var e = Int(kvk[Int(order[i])])
                nb += _dlen(_entry_src(kind, arrays, stride_a, n_cols + k, Int(gch[e])), Int(gco[e]))
            if nb > 2147483647:
                key_bytes.free()
                order.free()
                vals.free()
                cnt.free()
                _free_kv(kv, n_keys)
                K.free()
                return E_OUTPUT_TOO_BIG
            key_bytes[k] = Int64(nb)
            total += _align8((n_res + 1) * 4) + _align8(max(nb, 1))
        elif kind == K_I32:
            total += _align8(n_res * 4)
        else:
            total += n_res * 8
    total += n_out * n_res * 8
    var arena = unsafe_alloc[UInt8](max(total, 8), alignment=64)
    var at = Int(arena)

    O[0] = Int64(n_res)
    O[1] = Int64(Int(arena))
    O[2] = Int64(1 if (empty_global and n_res == 1) else 0)
    for k in range(n_keys):
        var kind = Int(K.kinds[k])
        var kvk = _lp(Int(kv[k]))
        if kind == K_DICT or kind == K_STR:
            var gch = _lp(Int(K.gchunk[k]))
            var gco = _lp(Int(K.gcode[k]))
            var offs = _ip(at)
            at += _align8((n_res + 1) * 4)
            var dst = _bp(at)
            at += _align8(max(Int(key_bytes[k]), 1))
            var p = 0
            offs[0] = 0
            for i in range(n_res):
                var e = Int(kvk[Int(order[i])])
                var d = _entry_src(kind, arrays, stride_a, n_cols + k, Int(gch[e]))
                var src = _bp(_dstart(d, Int(gco[e])))
                var ln = _dlen(d, Int(gco[e]))
                for b in range(ln):
                    dst[p] = src[b]
                    p += 1
                offs[i + 1] = Int32(p)
            O[3 + 2 * k] = Int64(Int(offs))
            O[4 + 2 * k] = Int64(Int(dst))
        elif kind == K_I32:
            var dst = _ip(at)
            at += _align8(n_res * 4)
            for i in range(n_res):
                dst[i] = Int32(Int(kvk[Int(order[i])]) + Int(K.base[k]))
            O[3 + 2 * k] = Int64(Int(dst))
            O[4 + 2 * k] = 0
        else:
            var dst = _lp(at)
            at += n_res * 8
            for i in range(n_res):
                dst[i] = Int64(Int(kvk[Int(order[i])]) + Int(K.base[k]))
            O[3 + 2 * k] = Int64(Int(dst))
            O[4 + 2 * k] = 0
    var obase = 3 + 2 * n_keys
    for j in range(n_out):
        var op = Int(outs[3 * j])
        var slot = Int(outs[3 * j + 1])
        var nslot = Int(outs[3 * j + 2])
        if op == O_COUNT:
            var dst = _lp(at)
            for i in range(n_res):
                dst[i] = cnt[Int(order[i])]
            O[obase + j] = Int64(Int(dst))
        else:
            var dst = FP(unsafe_from_address=at)
            for i in range(n_res):
                dst[i] = _out_value(op, slot, nslot, Int(order[i]), G, vals, cnt)
            O[obase + j] = Int64(Int(dst))
        at += n_res * 8

    key_bytes.free()
    order.free()
    vals.free()
    cnt.free()
    _free_kv(kv, n_keys)
    K.free()
    return 0


@always_inline
def _align8(n: Int) -> Int:
    return (n + 7) // 8 * 8


@always_inline
def _out_value(op: Int, slot: Int, nslot: Int, g: Int, G: Int, vals: FP, cnt: LP) -> Float64:
    """Finished value of output (op, slot) for group g: mean = sum / count, count as a float;
    min / max with a NaN-count slot are NaN when the group had a NaN (max) or only NaNs (min)."""
    if op == O_COUNT:
        return Float64(Int(cnt[g]))
    var v = vals[slot * G + g]
    if op == O_MEAN:
        var c = Int(cnt[g])
        v = v / Float64(c) if c > 0 else 0.0
    elif nslot >= 0:
        var n_nan = vals[nslot * G + g]
        var c = Float64(Int(cnt[g]))
        var hit = n_nan > 0.0 if op == O_MAX else (n_nan >= c and c > 0.0)
        if hit:
            v = nan[DType.float64]()
    return v


@always_inline
def _cmp(x: Float64, op: Int, c: Float64) -> Bool:
    if op == 0:
        return x < c
    if op == 1:
        return x <= c
    if op == 2:
        return x > c
    if op == 3:
        return x >= c
    if op == 4:
        return x == c
    return x != c


@always_inline
def _less(a: Int, b: Int, sv: FP, ns: Int, n: Int) -> Bool:
    """Row a before row b: lexicographic on the sort values, then the original (key) order."""
    for j in range(ns):
        var x = sv[j * n + a]
        var y = sv[j * n + b]
        if x < y:
            return True
        if x > y:
            return False
    return a < b


def _merge_sort(perm: LP, m: Int, sv: FP, ns: Int, n: Int):
    """Sort perm[0..m) (row numbers) by _less; bottom-up merge sort."""
    if m < 2:
        return
    var tmp = unsafe_alloc[Int64](m)
    var src = perm
    var dst = tmp
    var width = 1
    while width < m:
        var lo = 0
        while lo < m:
            var mid = min(lo + width, m)
            var hi = min(lo + 2 * width, m)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if _less(Int(src[j]), Int(src[i]), sv, ns, n):
                    dst[k] = src[j]
                    j += 1
                else:
                    dst[k] = src[i]
                    i += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
            lo += 2 * width
        var t = src
        src = dst
        dst = t
        width *= 2
    if Int(src) != Int(perm):
        for i in range(m):
            perm[i] = src[i]
    tmp.free()


def _top_k(perm: LP, n: Int, k: Int, sv: FP, ns: Int) -> Int:
    """The k first rows by _less into perm[0..k) (unordered): a max-heap whose root is the worst kept."""
    if k <= 0:
        return 0
    var size = 0
    for r in range(n):
        if size < k:
            var c = size
            perm[c] = Int64(r)
            size += 1
            while c > 0:                            # sift up
                var p = (c - 1) // 2
                if _less(Int(perm[p]), Int(perm[c]), sv, ns, n):
                    var t = perm[p]
                    perm[p] = perm[c]
                    perm[c] = t
                    c = p
                else:
                    break
        elif _less(r, Int(perm[0]), sv, ns, n):
            perm[0] = Int64(r)
            var c = 0
            while True:                             # sift down
                var l = 2 * c + 1
                if l >= size:
                    break
                var w = l
                if l + 1 < size and _less(Int(perm[l]), Int(perm[l + 1]), sv, ns, n):
                    w = l + 1
                if _less(Int(perm[c]), Int(perm[w]), sv, ns, n):
                    var t = perm[c]
                    perm[c] = perm[w]
                    perm[w] = t
                    c = w
                else:
                    break
    return size


def _free_kv(kv: LP, n_keys: Int):
    for k in range(n_keys):
        if Int(kv[k]) != 0:
            _lp(Int(kv[k])).free()
    kv.free()


def _hash_groups(K: Keys, arrays: LP, stride_a: Int, n_cols: Int, n_keys: Int, tbl: LP, tstride: Int,
                 n_chunks: Int, gids: IP, kv: LP, n_rows: Int) -> Int:
    """Group ids for sparse key tuples: open addressing on the tuple of per-key codes (integer
    value - base, or string entry id); groups are numbered in order of first appearance and their
    codes stored in kv[k][g]. Returns the number of groups, or -1 on an invalid dictionary code."""
    var tsize = _pow2_at_least(2 * n_rows + 2)
    var table = unsafe_alloc[Int32](tsize)
    for i in range(tsize):
        table[i] = -1
    var v = unsafe_alloc[Int64](max(n_keys, 1))
    var ng = 0
    for c in range(n_chunks):
        var c0 = Int(tbl[c * tstride])
        var c1 = Int(tbl[c * tstride + 1])
        for i in range(c0, c1):
            var h = UInt64(0x9E3779B97F4A7C15)
            for k in range(n_keys):
                var kind = Int(K.kinds[k])
                var arr = Int(arrays[c * stride_a + n_cols + k])
                var x = 0
                if kind == K_STR:
                    x = Int(_ip(Int(K.rowcode[k]))[i])
                elif kind == K_DICT:
                    var code = Int(_ip(_arr_buf(arr, 1))[_arr_off(arr) + i - c0])
                    if code < 0 or code >= _arr_len(_arr_dict(arr)):
                        table.free()
                        v.free()
                        return -1
                    x = Int(_ip(Int(_lp(Int(K.remap[k]))[c]))[code])
                else:
                    x = _key_value(kind, arr, i - c0) - Int(K.base[k])
                v[k] = Int64(x)
                h = _mix(h ^ UInt64(x))
            var pos = Int(h & UInt64(tsize - 1))
            while True:
                var g = Int(table[pos])
                if g < 0:
                    g = ng
                    ng += 1
                    table[pos] = Int32(g)
                    for k in range(n_keys):
                        _lp(Int(kv[k]))[g] = v[k]
                    gids[i] = Int32(g)
                    break
                var same = True
                for k in range(n_keys):
                    if _lp(Int(kv[k]))[g] != v[k]:
                        same = False
                        break
                if same:
                    gids[i] = Int32(g)
                    break
                pos = (pos + 1) & (tsize - 1)
    table.free()
    v.free()
    return ng


def _key_sort_values(K: Keys, k: Int, kvi: LP, order: LP, n_res: Int, arrays: LP, stride_a: Int,
                     n_cols: Int, dst: FP, sign: Float64):
    """dst[i] = sign * (exact rank of row i's value of key k): strings by their bytes; integers by
    their code when it fits in float64 exactly (dense and compacted codes are below 2^27), else by
    their rank among the rows (hash-path codes, value - base, can reach 2^63)."""
    var kind = Int(K.kinds[k])
    if kind == K_DICT or kind == K_STR:
        var card = Int(K.card[k])
        var rk = unsafe_alloc[Int64](max(card, 1))
        _global_ranks(kind, arrays, stride_a, n_cols + k, card, _lp(Int(K.gchunk[k])),
                      _lp(Int(K.gcode[k])), rk)
        for i in range(n_res):
            dst[i] = sign * Float64(Int(rk[Int(kvi[Int(order[i])])]))
        rk.free()
        return
    var hi = 0
    for i in range(n_res):
        hi = max(hi, Int(kvi[Int(order[i])]))
    if hi < (1 << 53):
        for i in range(n_res):
            dst[i] = sign * Float64(Int(kvi[Int(order[i])]))
        return
    var perm = unsafe_alloc[Int64](n_res)
    var tmp = unsafe_alloc[Int64](n_res)
    for i in range(n_res):
        perm[i] = Int64(i)
    # merge sort of row numbers by their int64 code (ties by row number)
    var src = perm
    var dstp = tmp
    var width = 1
    while width < n_res:
        var lo = 0
        while lo < n_res:
            var mid = min(lo + width, n_res)
            var hi2 = min(lo + 2 * width, n_res)
            var a = lo
            var b = mid
            var t = lo
            while a < mid and b < hi2:
                var ca = kvi[Int(order[Int(src[a])])]
                var cb = kvi[Int(order[Int(src[b])])]
                if cb < ca or (cb == ca and src[b] < src[a]):
                    dstp[t] = src[b]
                    b += 1
                else:
                    dstp[t] = src[a]
                    a += 1
                t += 1
            while a < mid:
                dstp[t] = src[a]
                a += 1
                t += 1
            while b < hi2:
                dstp[t] = src[b]
                b += 1
                t += 1
            lo += 2 * width
        var sw = src
        src = dstp
        dstp = sw
        width *= 2
    var r = 0
    for t in range(n_res):
        if t > 0 and kvi[Int(order[Int(src[t])])] != kvi[Int(order[Int(src[t - 1])])]:
            r += 1
        dst[Int(src[t])] = sign * Float64(r)
    perm.free()
    tmp.free()
# ── M2: native leaf filter ─────────────────────────────────────────────────────────────────
#
# Filter*(Scan) leaves under a join: the predicate runs on the bytecode interpreter per morsel (in
# parallel), and the surviving rows of the needed columns are written into one arena. Dictionary
# string columns are read as codes (predicates: dtype 5) and written out decoded, so only the rows
# that survive are ever decoded.
#
# Plan (int64 words):
#   0 magic 0x4d584631 ("MXF1")   1 version (1)   2 n_chunks   3 n_cols (bytecode value columns)
#   4 n_out (output columns)   5 code*   6 consts*   7 set_bytes*   8 set_meta*   9 dtypes* (int32)
#  10 arrays* : int64[n_chunks][n_cols + n_out] = ArrowArray*
#  11 out_kinds* : int64[n_out]  0 fixed width, 1 utf8 (int32 offsets), 2 dictionary<int32, utf8>
#                                (written as utf8)
#  12 out_size* : int64[n_out]  bytes per value of fixed-width columns
#  13 threads   14 n_dcols   15 doffs* : int64[n_chunks][n_dcols]
# Result: 0 n_rows   1 arena*   then per output column (A, B): fixed (values*, 0); strings
#         (int32 offsets[n+1]*, bytes*)

comptime FMAGIC = 0x4D584631
comptime HUGE = 1 << 21


def _big_alloc(n: Int) -> BP:
    """A buffer written once, in parallel, right after allocation: 2 MiB-aligned and marked for
    transparent huge pages above 2 MiB, so first-touch faults are 512x fewer (24 threads faulting
    4 KiB pages serialize on the kernel's mm lock: ~25 ms for 50 MB at SF1)."""
    if n < HUGE:
        return unsafe_alloc[UInt8](n, alignment=64)
    var size = (n + HUGE - 1) // HUGE * HUGE
    var b = unsafe_alloc[UInt8](size, alignment=HUGE)
    _ = external_call["madvise", Int32](Int(b), size, Int32(14))     # MADV_HUGEPAGE
    return b
comptime OK_FIXED = 0
comptime OK_UTF8 = 1
comptime OK_DICT = 2


@always_inline
def _out_str(kind: Int, arr: Int, local: Int, mut start: Int, mut ln: Int):
    """Bytes of row `local` of a string output column (utf8, or a dictionary entry)."""
    if kind == OK_DICT:
        var code = Int(_ip(_arr_buf(arr, 1))[_arr_off(arr) + local])
        var d = _arr_dict(arr)
        start = _dstart(d, code)
        ln = _dlen(d, code)
    else:
        start = _dstart(arr, local)
        ln = _dlen(arr, local)


@export
def mx_filter(plan_a: Int, out_a: Int) abi("C") -> Int:
    var P = _lp(plan_a)
    if Int(P[0]) != FMAGIC or Int(P[1]) != 1:
        return E_BAD_PLAN
    var n_chunks = Int(P[2])
    var n_cols = Int(P[3])
    var n_out = Int(P[4])
    var code = _ip(Int(P[5]))
    var dtypes = _ip(Int(P[9]))
    var arrays = _lp(Int(P[10]))
    var kinds = _lp(Int(P[11]))
    var sizes = _lp(Int(P[12]))
    var threads = max(1, Int(P[13]))
    var n_dcols = Int(P[14])
    var doffs = _lp(Int(P[15]))
    var O = _lp(out_a)
    var stride_a = n_cols + n_out
    if n_chunks < 1 or Int(code[0]) != 0x4D585031:
        return E_BAD_PLAN

    # chunk table: row_start, row_end, per value column its base address, then dictionary offsets
    var tstride = 2 + n_cols + n_dcols
    var tbl = unsafe_alloc[Int64](n_chunks * tstride)
    var row = 0
    for c in range(n_chunks):
        var first = Int(arrays[c * stride_a])
        var n = _arr_len(first)
        tbl[c * tstride] = Int64(row)
        tbl[c * tstride + 1] = Int64(row + n)
        for j in range(n_cols):
            var arr = Int(arrays[c * stride_a + j])
            var isz = _itemsize(Int(dtypes[j]) & 255)
            tbl[c * tstride + 2 + j] = Int64(_arr_buf(arr, 1) + (_arr_off(arr) - row) * isz)
        for i in range(n_dcols):
            tbl[c * tstride + 2 + n_cols + i] = doffs[c * n_dcols + i]
        row += n
    var n_rows = row
    var no_gids = unsafe_alloc[Int32](1)          # no group ids (Mojo has no null pointer)
    var p = Prog(code, FP(unsafe_from_address=Int(P[6])), _bp(Int(P[7])), _lp(Int(P[8])), tbl,
                 n_chunks, dtypes, no_gids, 1, False)
    if p.n_regs > MAX_REGS:
        no_gids.free()
        tbl.free()
        return E_PIPELINE

    var morsels = max((n_rows + MORSEL - 1) // MORSEL, 1)
    var nt = max(1, min(threads, morsels))
    var mask = _big_alloc(max(n_rows, 1))
    var mcnt = unsafe_alloc[Int64](morsels)                    # surviving rows per morsel
    var mbytes = unsafe_alloc[Int64](morsels * max(n_out, 1))  # string bytes per morsel per column
    for i in range(morsels * max(n_out, 1)):
        mbytes[i] = 0
    var counter = unsafe_alloc[Int64](1)
    counter[0] = 0

    # ── pass 1: predicate -> mask, counts and string bytes per morsel ───────────────────────
    def pass1(wid: Int) {imm}:
        var regs = unsafe_alloc[Float64](max(p.n_regs, 1) * VEC)
        var sel = unsafe_alloc[Int32](VEC)
        _run_ops(p, regs, p.tbl, 0, p.n_pro, 0, sel, False, VEC)     # constants, once
        while True:
            var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            var lo = m * MORSEL
            if lo >= n_rows:
                break
            var hi = min(lo + MORSEL, n_rows)
            var pos = lo
            var kept = 0
            while pos < hi:
                var k = _find_chunk(p, pos)
                var rowp = p.tbl + k * p.tbl_stride
                var cols = rowp + 2
                var stop = min(hi, Int(rowp[1]))
                var c0 = Int(rowp[0])
                var i0 = pos
                while i0 < stop:
                    var n = min(VEC, stop - i0)
                    if p.pred_reg >= 0:
                        _run_ops(p, regs, cols, p.n_pro, p.n_pred, i0, sel, False, n)
                        var pr = regs + p.pred_reg * VEC
                        for i in range(n):
                            mask[i0 + i] = 1 if pr[i] != 0.0 else 0
                    else:
                        for i in range(n):
                            mask[i0 + i] = 1
                    for i in range(n):
                        if mask[i0 + i] != 0:
                            kept += 1
                            for j in range(n_out):
                                var kd = Int(kinds[j])
                                if kd != OK_FIXED:
                                    var st = 0
                                    var ln = 0
                                    _out_str(kd, Int(arrays[k * stride_a + n_cols + j]), i0 + i - c0, st, ln)
                                    mbytes[m * n_out + j] += Int64(ln)
                    i0 += n
                pos = stop
            mcnt[m] = Int64(kept)
        regs.free()
        sel.free()

    if nt > 1:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
        parallelize(pass1, nt)
    else:
        pass1(0)

    # ── prefix sums: each morsel's first output row and first string byte ──────────────────
    var mrow = unsafe_alloc[Int64](morsels)
    var total = 0
    for m in range(morsels):
        mrow[m] = Int64(total)
        total += Int(mcnt[m])
    var n_sel = total
    var col_bytes = unsafe_alloc[Int64](max(n_out, 1))
    for j in range(n_out):
        var b = 0
        for m in range(morsels):
            var x = Int(mbytes[m * n_out + j])
            mbytes[m * n_out + j] = Int64(b)
            b += x
        col_bytes[j] = Int64(b)
        if Int(kinds[j]) != OK_FIXED and b > 2147483647:
            mask.free()
            mcnt.free()
            mbytes.free()
            counter.free()
            mrow.free()
            col_bytes.free()
            no_gids.free()
            tbl.free()
            return E_OUTPUT_TOO_BIG

    # ── arena ────────────────────────────────────────────────────────────────────────────
    var asize = 0
    for j in range(n_out):
        if Int(kinds[j]) == OK_FIXED:
            asize += _align8(n_sel * Int(sizes[j]))
        else:
            asize += _align8((n_sel + 1) * 4) + _align8(max(Int(col_bytes[j]), 1))
    var arena = _big_alloc(max(asize, 8))
    var dst_a = unsafe_alloc[Int64](max(n_out, 1))    # values / offsets
    var dst_b = unsafe_alloc[Int64](max(n_out, 1))    # string bytes
    var at = Int(arena)
    for j in range(n_out):
        if Int(kinds[j]) == OK_FIXED:
            dst_a[j] = Int64(at)
            dst_b[j] = 0
            at += _align8(n_sel * Int(sizes[j]))
        else:
            dst_a[j] = Int64(at)
            at += _align8((n_sel + 1) * 4)
            dst_b[j] = Int64(at)
            at += _align8(max(Int(col_bytes[j]), 1))
            _ip(Int(dst_a[j]))[n_sel] = Int32(Int(col_bytes[j]))

    # ── pass 2: write the surviving rows ─────────────────────────────────────────────────
    counter[0] = 0

    def pass2(wid: Int) {imm}:
        while True:
            var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            var lo = m * MORSEL
            if lo >= n_rows:
                break
            var hi = min(lo + MORSEL, n_rows)
            var r = Int(mrow[m])
            var pos = lo
            while pos < hi:
                var k = _find_chunk(p, pos)
                var rowp = p.tbl + k * p.tbl_stride
                var stop = min(hi, Int(rowp[1]))
                var c0 = Int(rowp[0])
                for j in range(n_out):
                    var arr = Int(arrays[k * stride_a + n_cols + j])
                    var kd = Int(kinds[j])
                    var rr = r
                    if kd == OK_FIXED:
                        var sz = Int(sizes[j])
                        var base = _arr_buf(arr, 1) + _arr_off(arr) * sz
                        if sz == 8 or sz == 16:             # int64 / float64 / dates64 / decimal128
                            var w = sz // 8
                            var src = _lp(base)
                            var dst = _lp(Int(dst_a[j]))
                            for i in range(pos, stop):
                                if mask[i] != 0:
                                    for b in range(w):
                                        dst[rr * w + b] = src[(i - c0) * w + b]
                                    rr += 1
                        elif sz == 4:                         # int32 / float32 / date32
                            var src = _ip(base)
                            var dst = _ip(Int(dst_a[j]))
                            for i in range(pos, stop):
                                if mask[i] != 0:
                                    dst[rr] = src[i - c0]
                                    rr += 1
                        else:
                            var src = _bp(base)
                            var dst = _bp(Int(dst_a[j]))
                            for i in range(pos, stop):
                                if mask[i] != 0:
                                    var s0 = (i - c0) * sz
                                    var d0 = rr * sz
                                    for b in range(sz):
                                        dst[d0 + b] = src[s0 + b]
                                    rr += 1
                    else:
                        var offs = _ip(Int(dst_a[j]))
                        var data = _bp(Int(dst_b[j]))
                        var bp = Int(mbytes[m * n_out + j])
                        # rows of this morsel before `pos` (earlier chunks) already advanced bp:
                        # recompute it from the rows written so far in this morsel
                        for i in range(pos, stop):
                            if mask[i] != 0:
                                var st = 0
                                var ln = 0
                                _out_str(kd, arr, i - c0, st, ln)
                                offs[rr] = Int32(bp)
                                var sp = _bp(st)
                                for b in range(ln):
                                    data[bp + b] = sp[b]
                                bp += ln
                                rr += 1
                        mbytes[m * n_out + j] = Int64(bp)      # next chunk of this morsel continues here
                var adv = 0
                for i in range(pos, stop):
                    adv += Int(mask[i])
                r += adv
                pos = stop

    if nt > 1:
        parallelize(pass2, nt)
    else:
        pass2(0)

    O[0] = Int64(n_sel)
    O[1] = Int64(Int(arena))
    for j in range(n_out):
        O[2 + 2 * j] = dst_a[j]
        O[3 + 2 * j] = dst_b[j]
    mask.free()
    mcnt.free()
    mbytes.free()
    counter.free()
    mrow.free()
    col_bytes.free()
    dst_a.free()
    dst_b.free()
    no_gids.free()
    tbl.free()
    return 0
