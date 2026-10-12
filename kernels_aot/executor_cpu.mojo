"""
MXFrame native executor, CPU (F-030, milestone M0 of docs/design/mojo-executor.md).

ONE call runs a whole plan: Python compiles it once into a small int64 plan buffer and calls
`mx_execute`; everything that touches rows happens here. M0 covers the pipeline

    Scan (Arrow C Data Interface) -> Filter* -> GroupBy(keys) + Aggregate -> Sort(keys)

  1. columns arrive as ArrowArray structs (one per chunk per column); nothing is copied or
     converted on the way in;
  2. group keys (dictionary-encoded strings, int32, int64) become dense group ids, in parallel;
  3. filter + aggregate expressions run on the existing bytecode interpreter (pipeline_cpu.mojo);
  4. means / counts are finished, the groups are ordered by their key values (strings by their
     bytes, through the dictionary), and the output columns are written, string keys included,
     into one arena that Python wraps zero-copy and frees with `mx_exec_free`.

Anything the plan cannot express returns a negative code and Python runs its existing path.

Plan buffer (int64 words):
   0 magic 0x4d584531 ("MXE1")   1 version (1)
   2 n_chunks   3 n_cols (bytecode value columns)   4 n_keys   5 n_outputs   6 n_sort
   7 code*  8 consts*  9 set_bytes*  10 set_meta*  11 dtypes* (int32 per value column)
  12 arrays* : int64[n_chunks][n_cols + n_keys] = ArrowArray* (value columns, then keys)
  13 key_kinds* : int64[n_keys]  0 = dictionary<int32, utf8>, 1 = int32, 2 = int64
  14 outputs* : int64[n_outputs][2] = (op, slot)  op 0 sum 1 min 2 max 3 mean 4 count
  15 sort* : int64[n_sort][2] = (key index, descending)
  16 threads   17 max dense groups

Result (int64 words, caller-allocated, 3 + 2 * n_keys + n_outputs):
   0 n_rows   1 arena* (free with mx_exec_free)   2 flags (bit 0: global aggregate over no rows)
   then per key (A, B): dictionary key -> (int32 offsets[n+1]*, utf8 data*); int32 / int64 key -> (values*, 0)
   then per output: float64[n]* (int64[n]* for count)

Return: 0 ok; < 0 declined or failed (the caller falls back).
"""

from std.memory import UnsafePointer, alloc
from std.algorithm import parallelize
from std.atomic import Atomic
from std.ffi import external_call

from pipeline_cpu import mx_pipeline_cpu

comptime FP = UnsafePointer[Float64, MutAnyOrigin]
comptime IP = UnsafePointer[Int32, MutAnyOrigin]
comptime LP = UnsafePointer[Int64, MutAnyOrigin]
comptime BP = UnsafePointer[UInt8, MutAnyOrigin]

comptime MAGIC = 0x4D584531
comptime MORSEL = 65536

comptime K_DICT = 0
comptime K_I32 = 1
comptime K_I64 = 2

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
comptime FUSE_GROUPS = 4096   # up to this many groups: keys encoded and aggregated per morsel
comptime HDR = 16
comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime INF = Float64(1.7976931348623157e308) * 10.0
comptime DICT_CAP = 4096      # M0: merged dictionary size (linear merge); M1 brings a hash table


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
    # pipeline dtypes: 0 i32, 1 i64, 2 f32, 3 f64, 4 u8
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

    def __init__(out self, n: Int, n_chunks: Int):
        self.n = n
        self.n_chunks = n_chunks
        self.kinds = alloc[Int64](max(n, 1))
        self.base = alloc[Int64](max(n, 1))
        self.card = alloc[Int64](max(n, 1))
        self.stride = alloc[Int64](max(n, 1))
        self.gchunk = alloc[Int64](max(n, 1))
        self.gcode = alloc[Int64](max(n, 1))
        self.remap = alloc[Int64](max(n, 1))
        for k in range(max(n, 1)):
            self.gchunk[k] = 0
            self.gcode[k] = 0
            self.remap[k] = 0

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


def _build_global_dict(arrays: LP, stride_a: Int, n_chunks: Int, col: Int, cap: Int,
                       gchunk: LP, gcode: LP, remaps: LP) -> Int:
    """Merge the chunks' dictionaries: returns the number of distinct strings (or -1 past `cap`).
    remaps[c] = address of int32[len(dict of chunk c)] mapping chunk codes to global entries."""
    var ng = 0
    var prev_d = 0
    for c in range(n_chunks):
        var d = _arr_dict(Int(arrays[c * stride_a + col]))
        if c > 0 and _same_dict(d, prev_d):
            remaps[c] = remaps[c - 1]
            continue
        prev_d = d
        var nd = _arr_len(d)
        var rm = alloc[Int32](max(nd, 1))
        remaps[c] = Int64(Int(rm))
        for code in range(nd):
            var s0 = _dstart(d, code)
            var l0 = _dlen(d, code)
            var found = -1
            for g in range(ng):
                var dg = _arr_dict(Int(arrays[Int(gchunk[g]) * stride_a + col]))
                if _bytes_cmp(s0, l0, _dstart(dg, Int(gcode[g])), _dlen(dg, Int(gcode[g]))) == 0:
                    found = g
                    break
            if found < 0:
                if ng >= cap:
                    return -1
                gchunk[ng] = Int64(c)
                gcode[ng] = Int64(code)
                found = ng
                ng += 1
            rm[code] = Int32(found)
    return ng


def _global_ranks(arrays: LP, stride_a: Int, col: Int, n: Int, gchunk: LP, gcode: LP, ranks: LP):
    """ranks[entry] = position of global dictionary entry `entry` in byte order."""
    var order = alloc[Int64](max(n, 1))
    for i in range(n):
        order[i] = Int64(i)
    for i in range(1, n):
        var cur = Int(order[i])
        var dc = _arr_dict(Int(arrays[Int(gchunk[cur]) * stride_a + col]))
        var j = i - 1
        while j >= 0:
            var o = Int(order[j])
            var do = _arr_dict(Int(arrays[Int(gchunk[o]) * stride_a + col]))
            if _bytes_cmp(_dstart(dc, Int(gcode[cur])), _dlen(dc, Int(gcode[cur])),
                          _dstart(do, Int(gcode[o])), _dlen(do, Int(gcode[o]))) >= 0:
                break
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = Int64(cur)
    for r in range(n):
        ranks[Int(order[r])] = Int64(r)
    order.free()


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
    if Int(P[0]) != MAGIC or Int(P[1]) != 1:
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
    var O = _lp(out_a)
    var stride_a = n_cols + n_keys
    if n_chunks < 1:
        return E_BAD_PLAN

    # ── 1. chunk table for the interpreter: row_start, row_end, per column base address ──────
    var tstride = 2 + n_cols
    var tbl = alloc[Int64](n_chunks * tstride)
    var row = 0
    for c in range(n_chunks):
        var first = Int(arrays[c * stride_a])
        var n = _arr_len(first)
        tbl[c * tstride] = Int64(row)
        tbl[c * tstride + 1] = Int64(row + n)
        for j in range(n_cols):
            var arr = Int(arrays[c * stride_a + j])
            var isz = _itemsize(Int(dtypes[j]))
            tbl[c * tstride + 2 + j] = Int64(_arr_buf(arr, 1) + (_arr_off(arr) - row) * isz)
        row += n
    var n_rows = row

    # ── 2. group keys -> dense group ids ─────────────────────────────────────────────────────
    var K = Keys(n_keys, n_chunks)
    var D = 1
    for k in range(n_keys):
        var kind = Int(key_kinds[k])
        K.kinds[k] = Int64(kind)
        var col = n_cols + k
        if kind == K_DICT:
            var gch = alloc[Int64](DICT_CAP)
            var gco = alloc[Int64](DICT_CAP)
            var rms = alloc[Int64](n_chunks)
            for c in range(n_chunks):
                rms[c] = 0
            K.gchunk[k] = Int64(Int(gch))
            K.gcode[k] = Int64(Int(gco))
            K.remap[k] = Int64(Int(rms))
            var ng = _build_global_dict(arrays, stride_a, n_chunks, col, DICT_CAP, gch, gco, rms)
            if ng < 0:
                K.free()
                tbl.free()
                return E_DICT_TOO_BIG
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
        if Int(K.card[k]) <= 0 or D > max_groups // max(Int(K.card[k]), 1):
            K.free()
            tbl.free()
            return E_TOO_MANY_GROUPS
        D *= Int(K.card[k])
    var s = 1
    for k in range(n_keys - 1, -1, -1):
        K.stride[k] = Int64(s)
        s *= Int(K.card[k])

    var grouped = n_keys > 0
    var G = D if grouped else 1
    var code_p = _ip(code)
    var n_slots = Int(code_p[5])
    var vals = alloc[Float64](max(n_slots * G, 1))
    var cnt = alloc[Int64](G)
    var rc = 0
    var morsels = (n_rows + MORSEL - 1) // MORSEL
    var nt = max(1, min(threads, morsels))

    if not grouped:
        rc = mx_pipeline_cpu(code, consts, set_bytes, set_meta, Int(tbl), n_chunks, Int(dtypes),
                             0, 1, 0, n_rows, Int(vals), Int(cnt), threads)
    elif G <= FUSE_GROUPS:
        # ── fused: per 64k-row morsel, encode keys into a small reused buffer and aggregate it
        #    right away (no n_rows-sized group-id array: no page faults, cache-resident)
        var agg_base = HDR + (Int(code_p[2]) + Int(code_p[3]) + Int(code_p[4])) * 5
        var acc_v = alloc[Float64](nt * max(n_slots * G, 1))
        var acc_c = alloc[Int64](nt * G)
        var vstride = max(n_slots * G, 1)
        var failed = alloc[Int64](1)
        failed[0] = 0
        var counter = alloc[Int64](1)
        counter[0] = 0

        @parameter
        def fused_worker(wid: Int):
            var gbuf = alloc[Int32](MORSEL)
            var tv = alloc[Float64](vstride)
            var tc = alloc[Int64](G)
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
                var m = Int(Atomic[DType.int64].fetch_add(counter, 1))
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
            parallelize[fused_worker](nt)
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
        var gids = alloc[Int32](max(n_rows, 1))
        var counter = alloc[Int64](1)
        counter[0] = 0
        var bad_key = alloc[Int64](1)
        bad_key[0] = 0

        @parameter
        def keys_worker(wid: Int):
            while True:
                var m = Int(Atomic[DType.int64].fetch_add(counter, 1))
                var lo = m * MORSEL
                if lo >= n_rows:
                    break
                if not _encode_keys(K, arrays, stride_a, n_cols, n_keys, tbl, tstride, lo, min(lo + MORSEL, n_rows), gids):
                    bad_key[0] = 1

        if nt > 1:
            _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
            parallelize[keys_worker](nt)
        else:
            keys_worker(0)
        counter.free()
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
        K.free()
        return rc if rc == E_BAD_KEY else E_PIPELINE

    # ── 4. which groups exist, in output order ───────────────────────────────────────────────
    var order = alloc[Int64](G)      # output position -> dense group id
    var n_res = 0
    if not grouped:
        order[0] = 0
        n_res = 1
    elif n_sort == 0:
        for g in range(G):
            if cnt[g] > 0:
                order[n_res] = Int64(g)
                n_res += 1
    else:
        # sort composite: sort keys first (ranks, flipped when descending), dense order otherwise
        var ranks = alloc[LP](n_keys)
        for k in range(n_keys):
            var card = Int(K.card[k])
            ranks[k] = alloc[Int64](card)
            if Int(K.kinds[k]) == K_DICT:
                _global_ranks(arrays, stride_a, n_cols + k, card, _lp(Int(K.gchunk[k])),
                              _lp(Int(K.gcode[k])), ranks[k])
            else:
                for v in range(card):
                    ranks[k][v] = Int64(v)
        var slot_of = alloc[Int64](G)
        for g in range(G):
            var comp = 0
            for j in range(n_sort):
                var k = Int(sorts[2 * j])
                var desc = Int(sorts[2 * j + 1]) != 0
                var card = Int(K.card[k])
                var r = Int(ranks[k][(g // Int(K.stride[k])) % card])
                if desc:
                    r = card - 1 - r
                comp = comp * card + r
            slot_of[g] = Int64(comp)
        var pos = alloc[Int64](G)
        for i in range(G):
            pos[i] = -1
        for g in range(G):
            pos[Int(slot_of[g])] = Int64(g)
        for i in range(G):
            var g = Int(pos[i])
            if g >= 0 and cnt[g] > 0:
                order[n_res] = Int64(g)
                n_res += 1
        pos.free()
        slot_of.free()
        for k in range(n_keys):
            ranks[k].free()
        ranks.free()

    # ── 5. output arena ─────────────────────────────────────────────────────────────────────
    var key_bytes = alloc[Int64](max(n_keys, 1))
    var total = 0
    for k in range(n_keys):
        var kind = Int(K.kinds[k])
        if kind == K_DICT:
            var gch = _lp(Int(K.gchunk[k]))
            var gco = _lp(Int(K.gcode[k]))
            var nb = 0
            for i in range(n_res):
                var e = (Int(order[i]) // Int(K.stride[k])) % Int(K.card[k])
                nb += _dlen(_arr_dict(Int(arrays[Int(gch[e]) * stride_a + n_cols + k])), Int(gco[e]))
            if nb > 2147483647:
                key_bytes.free()
                order.free()
                vals.free()
                cnt.free()
                K.free()
                return E_OUTPUT_TOO_BIG
            key_bytes[k] = Int64(nb)
            total += _align8((n_res + 1) * 4) + _align8(max(nb, 1))
        elif kind == K_I32:
            total += _align8(n_res * 4)
        else:
            total += n_res * 8
    total += n_out * n_res * 8
    var arena = alloc[UInt8](max(total, 8), alignment=64)
    var at = Int(arena)

    O[0] = Int64(n_res)
    O[1] = Int64(Int(arena))
    O[2] = Int64(1 if (not grouped and cnt[0] == 0) else 0)
    for k in range(n_keys):
        var kind = Int(K.kinds[k])
        var stride_k = Int(K.stride[k])
        var card = Int(K.card[k])
        if kind == K_DICT:
            var gch = _lp(Int(K.gchunk[k]))
            var gco = _lp(Int(K.gcode[k]))
            var offs = _ip(at)
            at += _align8((n_res + 1) * 4)
            var dst = _bp(at)
            at += _align8(max(Int(key_bytes[k]), 1))
            var p = 0
            offs[0] = 0
            for i in range(n_res):
                var e = (Int(order[i]) // stride_k) % card
                var d = _arr_dict(Int(arrays[Int(gch[e]) * stride_a + n_cols + k]))
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
                dst[i] = Int32((Int(order[i]) // stride_k) % card + Int(K.base[k]))
            O[3 + 2 * k] = Int64(Int(dst))
            O[4 + 2 * k] = 0
        else:
            var dst = _lp(at)
            at += n_res * 8
            for i in range(n_res):
                dst[i] = Int64((Int(order[i]) // stride_k) % card + Int(K.base[k]))
            O[3 + 2 * k] = Int64(Int(dst))
            O[4 + 2 * k] = 0
    var obase = 3 + 2 * n_keys
    for j in range(n_out):
        var op = Int(outs[2 * j])
        var slot = Int(outs[2 * j + 1])
        if op == O_COUNT:
            var dst = _lp(at)
            for i in range(n_res):
                dst[i] = cnt[Int(order[i])]
            O[obase + j] = Int64(Int(dst))
        else:
            var dst = FP(unsafe_from_address=at)
            for i in range(n_res):
                var g = Int(order[i])
                var v = vals[slot * G + g]
                if op == O_MEAN:
                    var c = Int(cnt[g])
                    v = v / Float64(c) if c > 0 else 0.0
                dst[i] = v
            O[obase + j] = Int64(Int(dst))
        at += n_res * 8

    key_bytes.free()
    order.free()
    vals.free()
    cnt.free()
    K.free()
    return 0


@always_inline
def _align8(n: Int) -> Int:
    return (n + 7) // 8 * 8
