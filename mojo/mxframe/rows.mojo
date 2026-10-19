"""Row queries (F-152): the materialize sink and the top-k sink of the executor.

A query with no aggregate returns rows: `SELECT a, b + 1 FROM t WHERE c > 3 ORDER BY a LIMIT 10`. The executor reads the source
in chunks (`exec.mojo`) and for each chunk runs one kernel program (`sqlprog.lower_rows`) that evaluates the WHERE and computes
the select list's expressions and the sort keys into registers. The kernel's block hook (`_rows_hook`, the pattern of
query.mojo's `_mat_hook`) then writes, for each block of 1,024 rows, the numbers of the rows that passed and the values of
the computed registers at the block's own place in output arrays, so no two workers write the same entry; a prefix sum over
the blocks' counts compacts them. A plain column of the select list is not carried through registers: it is gathered from the
chunk by the selected rows' numbers (so a wide `SELECT *` costs no registers). TEXT keeps the source's codes and the result's
dictionary is built from the words actually present (`_to_table`).

After the last chunk the rows are ordered (a stable radix sort per key, from the last key to the first, `_order_rows`, with the
decided text order and NULLS FIRST / LAST), cut by OFFSET and LIMIT, and made into a Table. With a LIMIT and no ORDER BY the
chunks stop being read once enough rows are in hand (`early_stop`). With ORDER BY and a small LIMIT the top-k sink keeps the
best rows in per-worker heaps inside the hook and never materializes the rest.
"""
from std.time import perf_counter_ns
from std.atomic import Atomic
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .frame import text_ranks, _radix_order
from .query import Program, MAGIC, SRC_WORDS, CTL_REG, OVERFLOW_MESSAGE
from .kernels._fpenv import core_count
from .kernels.pipeline_cpu import IP, LP, FP, BP, VEC, E_OVERFLOW, pipeline_keyed
from .sql import sql_error
from .bind import SqlSource, BoundQuery, SortKey, B_COL, B_LIT, SRC_FRAME, pow10_checked
from .planner import PhysicalPlan, PlannerOptions, commas
from .sqlprog import SqlProgram, DictAccess, lower_rows, bkey, collect_columns
from .sqleval import ValCol
from .holistic import float_key
from .exec import (_table_addresses, _alias_lists, _Chunks, _open_chunks, _load_chunk, _check_footer, _Inputs, _inputs, _gather,
                   _order_rows, _to_table)


# ── the hook ─────────────────────────────────────────────────────────────────────────────────────────

def _rows_hook(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    """Writes a block's selected rows: their numbers in the chunk and the values of the computed registers, packed at the block's
    own place (row0 - the first row of the run), and their count at the block's place in the counts. Control block (Int64 words,
    its address in register CTL_REG): [0] computed expressions, [1] first row of the run, [2] row numbers (Int32), [3] counts
    (Int32), then per expression 4 words: [values (Int64), NULL flags (UInt8) or 0, value register, flag register or -1]."""
    for j in range(nsel):
        dst[j] = 0                                              # one group
    var ctl = LP(unsafe_from_address=Int(LP(unsafe_from_address=Int(regs + CTL_REG * VEC))[0]))
    var base = row0 - Int(ctl[1])
    var ids = IP(unsafe_from_address=Int(ctl[2])) + base
    if use_sel:
        for j in range(nsel):
            ids[j] = Int32(row0) + sel[j]
    else:
        for j in range(nsel):
            ids[j] = Int32(row0 + j)
    for c in range(Int(ctl[0])):
        var w = ctl + (4 + 4 * c)
        var src = LP(unsafe_from_address=Int(regs + Int(w[2]) * VEC))
        var vp = LP(unsafe_from_address=Int(w[0])) + base
        for j in range(nsel):
            vp[j] = src[j]
        if w[3] >= 0:
            var nsrc = LP(unsafe_from_address=Int(regs + Int(w[3]) * VEC))
            var np = BP(unsafe_from_address=Int(w[1])) + base
            for j in range(nsel):
                np[j] = UInt8(nsrc[j])
    IP(unsafe_from_address=Int(ctl[3]))[base // VEC] = Int32(nsel)
    return True


struct RowChunk(Movable):
    """The rows one kernel run selected, compacted: their numbers in the chunk and the computed expressions' values."""
    var ids: List[Int32]
    var vals: List[List[Int64]]
    var nuls: List[List[UInt8]]      # empty list for an expression that cannot be NULL
    var n: Int

    def __init__(out self):
        self.ids = List[Int32]()
        self.vals = List[List[Int64]]()
        self.nuls = List[List[UInt8]]()
        self.n = 0


def _run_rows(t: SharedTable, sp: SqlProgram, mask_addr: Int, start: Int, stop: Int, threads: Int) raises -> RowChunk:
    """The kernel over rows [start, stop) of one chunk; the selected rows and the computed values, in row order."""
    ref p = sp.p
    var total = stop - start
    var out = RowChunk()
    if total <= 0:
        for _ in range(len(sp.row_vregs)):
            out.vals.append(List[Int64]())
            out.nuls.append(List[UInt8]())
        return out^
    var nblk = (total + VEC - 1) // VEC
    var ncomp = len(sp.row_vregs)
    var ids = List[Int32](unsafe_uninit_length=total)
    var cnt = List[Int32](length=nblk, fill=0)
    var vals = List[List[Int64]]()
    var nuls = List[List[UInt8]]()
    var ctl = List[Int64]()
    ctl.append(Int64(ncomp))
    ctl.append(Int64(start))
    ctl.append(Int64(Int(ids.unsafe_ptr())))
    ctl.append(Int64(Int(cnt.unsafe_ptr())))
    for c in range(ncomp):
        vals.append(List[Int64](unsafe_uninit_length=total))
        nuls.append(List[UInt8](length=total, fill=0) if sp.row_nregs[c] >= 0 else List[UInt8]())
        ctl.append(Int64(Int(vals[c].unsafe_ptr())))
        ctl.append(Int64(Int(nuls[c].unsafe_ptr())) if sp.row_nregs[c] >= 0 else Int64(0))
        ctl.append(Int64(sp.row_vregs[c]))
        ctl.append(Int64(sp.row_nregs[c]))
    var code = List[Int32]()
    var header: List[Int] = [MAGIC, p.n_regs, 0, len(p.pred) // 5, len(p.body) // 5, len(p.aggs) // 5,
                             p.pred_reg, len(p.cols), 0, 0, 0, 0, 0, 0, 0, 0]
    for h in header:
        code.append(Int32(h))
    for w in p.pred:
        code.append(w)
    for w in p.body:
        code.append(w)
    for w in p.aggs:
        code.append(w)
    var consts = List[Float64]()
    for v in p.consts:
        consts.append(Float64(from_bits=UInt64(v)))
    consts[p.ctl_const] = Float64(from_bits=UInt64(Int(ctl.unsafe_ptr())))
    var set_bytes = sp.set_bytes.copy()
    if len(set_bytes) == 0:
        set_bytes.append(0)
    var set_meta = sp.set_meta.copy()
    if len(set_meta) == 0:
        set_meta = [0, 0, 0]
    var tbl = _table_addresses(t, p, mask_addr)
    var dtypes = p.dtypes.copy()
    if len(dtypes) == 0:
        dtypes.append(1)
    var out_vals = List[Float64](length=max(p.n_slots, 1), fill=0.0)
    var out_cnt = List[Int64](length=1, fill=0)
    var rc = pipeline_keyed[type_of(_rows_hook)](Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                                                 Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()), 1,
                                                 start, stop, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()),
                                                 threads if threads > 0 else core_count(), _rows_hook, False)
    _ = code^
    _ = consts^
    _ = set_bytes^
    _ = set_meta^
    _ = tbl^
    _ = dtypes^
    _ = out_vals^
    _ = out_cnt^
    _ = ctl^
    if rc == E_OVERFLOW:
        raise Error(OVERFLOW_MESSAGE)
    if rc != 0:
        raise Error("the CPU kernel failed with code " + String(rc))
    # compact: the blocks' entries move down over the gaps the filter left (a block never moves up)
    var at = 0
    var idp = ids.unsafe_ptr()
    var cp = cnt.unsafe_ptr()
    for b in range(nblk):
        var m = Int(cp[b])
        var from_ = b * VEC
        if m == 0:
            continue
        if from_ != at:
            for j in range(m):
                idp[at + j] = idp[from_ + j]
            for c in range(ncomp):
                var vp = vals[c].unsafe_ptr()
                for j in range(m):
                    vp[at + j] = vp[from_ + j]
                if len(nuls[c]) > 0:
                    var np = nuls[c].unsafe_ptr()
                    for j in range(m):
                        np[at + j] = np[from_ + j]
        at += m
    out.n = at
    out.ids = ids^
    out.vals = vals^
    out.nuls = nuls^
    return out^


# ── collecting the rows ──────────────────────────────────────────────────────────────────────────────

def _append_run(mut dst: ValCol, p: LP, nul: BP, has_nul: Bool, m: Int):
    """Appends m values (and NULL flags, when `nul` is not null) to a column being built."""
    var at = len(dst.v)
    dst.v.resize(at + m, 0)
    var d = dst.v.unsafe_ptr() + at
    if dst.kind == FLOAT.value:                                  # every NaN is the one NaN (DuckDB's: equal to itself, past every number)
        for j in range(m):
            var x = Float64(from_bits=UInt64(p[j]))
            d[j] = p[j] if x == x else Int64(0x7FF8000000000000)
    else:
        for j in range(m):
            d[j] = p[j]
    if has_nul:
        if len(dst.nul) == 0:
            dst.nul = List[UInt8](length=at, fill=0)
        dst.nul.resize(at + m, 0)
        var dn = dst.nul.unsafe_ptr() + at
        for j in range(m):
            dn[j] = nul[j]
    elif len(dst.nul) > 0:
        dst.nul.resize(at + m, 0)


def _append_gathered(mut dst: ValCol, chunk: SharedTable, pos: Int, ids: IP, m: Int):
    """Appends the values of chunk column `pos` at the rows `ids` (the plain columns of the select list)."""
    var at = len(dst.v)
    dst.v.resize(at + m, 0)
    var d = dst.v.unsafe_ptr() + at
    var src = chunk.values(pos).unsafe_ptr()
    for j in range(m):
        d[j] = src[Int(ids[j])]
    var valid = chunk._table[].cols[pos].validity_address()
    if valid != 0:
        var bits = BP(unsafe_from_address=valid)
        var any = False
        if len(dst.nul) == 0:
            dst.nul = List[UInt8](length=at, fill=0)
        dst.nul.resize(at + m, 0)
        var dn = dst.nul.unsafe_ptr() + at
        for j in range(m):
            var r = Int(ids[j])
            if (bits[r >> 3] >> UInt8(r & 7)) & 1 == 0:
                dn[j] = 1
                any = True
    elif len(dst.nul) > 0:
        dst.nul.resize(at + m, 0)


struct _Item(Copyable, Movable):
    """One column of the collected rows: a select item or a sort key. `computed >= 0`: the program computes it (its index among
    the computed expressions); otherwise it is a plain column gathered from the chunk."""
    var expr: Int
    var key: String
    var col: Int                     # the source column of a plain item
    var computed: Int

    def __init__(out self, expr: Int, key: String, col: Int, computed: Int):
        self.expr = expr
        self.key = key
        self.col = col
        self.computed = computed


def _item_of(items: List[_Item], q: BoundQuery, e: Int) -> Int:
    var k = bkey(q.rexprs, e)
    for i in range(len(items)):
        if items[i].key == k:
            return i
    return -1


def build_items(q: BoundQuery, keys_computed: Bool = False) -> Tuple[List[_Item], List[Int]]:
    """The columns the rows are collected into (the select list's, then the sort keys' that are not in it) and, of those, the
    expressions the program computes (everything that is not a plain column). `keys_computed` (top-k): the sort keys go through
    registers even when they are plain columns, because the heaps compare them in the hook."""
    var items = List[_Item]()
    var computed = List[Int]()
    var exprs = List[Int]()
    if keys_computed:
        for o in q.rorder:
            exprs.append(o.expr)
    for e in q.routs:
        exprs.append(e)
    for o in q.rorder:
        exprs.append(o.expr)
    for e in exprs:
        if _item_of(items, q, e) >= 0:
            continue
        ref x = q.rexprs[e]
        var force = False
        if keys_computed:
            for o in q.rorder:
                if bkey(q.rexprs, o.expr) == bkey(q.rexprs, e):
                    force = True
        if x.op == B_COL and not force:
            items.append(_Item(e, bkey(q.rexprs, e), Int(x.value), -1))
        else:
            computed.append(e)
            items.append(_Item(e, bkey(q.rexprs, e), -1, len(computed) - 1))
    return (items^, computed^)


# ── top-k: per-worker heaps in the hook ──────────────────────────────────────────────────────────────


@always_inline
def _before(ctl: LP, nk: Int, av: LP, an: BP, aid: Int, bv: LP, bn: BP, bid: Int) -> Bool:
    """Row a sorts before row b: by each key (NULLS FIRST / LAST, descending, TEXT through its ranks), ties by row number."""
    for k in range(nk):
        var w = ctl + (10 + 4 * k)
        var slot = Int(w[0])
        var x_null = an[slot] != 0
        var y_null = bn[slot] != 0
        if x_null or y_null:
            if x_null and y_null:
                continue
            return x_null == (w[2] != 0)
        var x = av[slot]
        var y = bv[slot]
        if w[3] == -1:                                           # a FLOAT: its bits in the order of the numbers, every NaN last
            x = float_key(x)
            y = float_key(y)
        elif w[3] != 0:
            var rk = LP(unsafe_from_address=Int(w[3]))
            x = rk[Int(x)]
            y = rk[Int(y)]
        if x != y:
            return (x > y) if w[1] != 0 else (x < y)
    return aid < bid


@always_inline
def _copy_entry(ns: Int, dv: LP, dn: BP, sv: LP, sn: BP):
    for s in range(ns):
        dv[s] = sv[s]
        dn[s] = sn[s]


@always_inline
def _swap_entries(ns: Int, vals: LP, nuls: BP, ids: IP, a: Int, b: Int):
    for t in range(ns):
        var x = vals[a * ns + t]
        vals[a * ns + t] = vals[b * ns + t]
        vals[b * ns + t] = x
        var y = nuls[a * ns + t]
        nuls[a * ns + t] = nuls[b * ns + t]
        nuls[b * ns + t] = y
    var tid = ids[a]
    ids[a] = ids[b]
    ids[b] = tid


def _topk_worker(ctl: LP, regs: FP) -> Int:
    """The calling worker's number: its register file's address is its identity, entered in the control block's table the first time."""
    var table = LP(unsafe_from_address=Int(ctl[4]))
    var counter = LP(unsafe_from_address=Int(ctl[3]))
    var mine = Int64(Int(regs))
    for i in range(min(Int(counter[0]), Int(ctl[5]))):
        if table[i] == mine:
            return i
    var i = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
    if i >= Int(ctl[5]):
        return -1
    table[i] = mine
    return i


def _topk_hook(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    """Offers each selected row to the calling worker's heap of the best K rows so far. Control block (Int64 words): [0] value slots,
    [1] sort keys, [2] K, [3] worker counter, [4] worker table, [5] its size, [6] values (Int64), [7] NULL flags (UInt8), [8] row numbers
    (Int32), [9] heap sizes (Int64) -- the arrays hold K + 1 entries per worker, the last one a scratch for the row on offer -- then
    per key [slot, descending, nulls first, rank table address or 0] from word 10, then per slot [value register, flag register or -1]."""
    for j in range(nsel):
        dst[j] = 0
    var ctl = LP(unsafe_from_address=Int(LP(unsafe_from_address=Int(regs + CTL_REG * VEC))[0]))
    var me = _topk_worker(ctl, regs)
    if me < 0:
        return False
    var ns = Int(ctl[0])
    var nk = Int(ctl[1])
    var K = Int(ctl[2])
    if K == 0:
        return True
    var slots = ctl + (10 + 4 * nk)
    var vals = LP(unsafe_from_address=Int(ctl[6])) + me * (K + 1) * ns
    var nuls = BP(unsafe_from_address=Int(ctl[7])) + me * (K + 1) * ns
    var ids = IP(unsafe_from_address=Int(ctl[8])) + me * (K + 1)
    var size_p = LP(unsafe_from_address=Int(ctl[9])) + me
    var sv = vals + K * ns
    var sn = nuls + K * ns
    # the first key decides almost every row once the heap is full: read it straight from its register and drop a row that is
    # strictly worse than the heap's worst before anything is copied
    var k0 = ctl + 10
    var k0_slot = Int(k0[0])
    var k0_v = LP(unsafe_from_address=Int(regs + Int(slots[2 * k0_slot]) * VEC))
    var k0_has_null = slots[2 * k0_slot + 1] >= 0
    var k0_n = LP(unsafe_from_address=Int(regs + Int(slots[2 * k0_slot + 1]) * VEC)) if k0_has_null else k0_v
    var k0_rank = LP(unsafe_from_address=Int(k0[3])) if k0[3] > 0 else k0_v
    for j in range(nsel):
        var row = row0 + (Int(sel[j]) if use_sel else j)
        if Int(size_p[0]) == K and not (k0_has_null and k0_n[j] != 0) and nuls[k0_slot] == 0:
            var x = k0_v[j]
            var y = vals[k0_slot]
            if k0[3] == -1:
                x = float_key(x)
                y = float_key(y)
            elif k0[3] != 0:
                x = k0_rank[Int(x)]
                y = k0_rank[Int(y)]
            if x != y and ((x < y) if k0[1] != 0 else (x > y)):
                continue
        for s in range(ns):
            var w = slots + 2 * s
            sv[s] = LP(unsafe_from_address=Int(regs + Int(w[0]) * VEC))[j]
            sn[s] = UInt8(LP(unsafe_from_address=Int(regs + Int(w[1]) * VEC))[j]) if w[1] >= 0 else UInt8(0)
        var size = Int(size_p[0])
        if size < K:
            _copy_entry(ns, vals + size * ns, nuls + size * ns, sv, sn)
            ids[size] = Int32(row)
            var i = size
            size += 1
            size_p[0] = Int64(size)
            while i > 0:                                         # sift up: the worst of the kept rows stays on top
                var parent = (i - 1) // 2
                if _before(ctl, nk, vals + parent * ns, nuls + parent * ns, Int(ids[parent]), vals + i * ns, nuls + i * ns, Int(ids[i])):
                    _swap_entries(ns, vals, nuls, ids, parent, i)
                    i = parent
                else:
                    break
        elif _before(ctl, nk, sv, sn, row, vals, nuls, Int(ids[0])):
            _copy_entry(ns, vals, nuls, sv, sn)
            ids[0] = Int32(row)
            var i = 0
            while True:
                var l = 2 * i + 1
                var r = l + 1
                var worst = i
                if l < size and _before(ctl, nk, vals + worst * ns, nuls + worst * ns, Int(ids[worst]), vals + l * ns, nuls + l * ns, Int(ids[l])):
                    worst = l
                if r < size and _before(ctl, nk, vals + worst * ns, nuls + worst * ns, Int(ids[worst]), vals + r * ns, nuls + r * ns, Int(ids[r])):
                    worst = r
                if worst == i:
                    break
                _swap_entries(ns, vals, nuls, ids, worst, i)
                i = worst
    return True


def _run_topk(t: SharedTable, sp: SqlProgram, mask_addr: Int, K: Int, key_slot: List[Int], key_desc: List[Bool], key_nf: List[Bool],
              key_rank: List[Int], threads: Int) raises -> RowChunk:
    """The kernel over the whole chunk with each worker keeping its best K rows; the union of the heaps (at most workers x K rows,
    unordered), as the row numbers and the values of every slot."""
    ref p = sp.p
    var ns = len(sp.row_vregs)
    var nk = len(key_slot)
    var maxw = (threads if threads > 0 else core_count()) + 4
    var vals = List[Int64](length=maxw * (K + 1) * ns + 1, fill=0)
    var nuls = List[UInt8](length=maxw * (K + 1) * ns + 1, fill=0)
    var ids = List[Int32](length=maxw * (K + 1) + 1, fill=0)
    var sizes = List[Int64](length=maxw, fill=0)
    var table = List[Int64](length=maxw, fill=0)
    var counter = List[Int64](length=1, fill=0)
    var ctl = List[Int64]()
    ctl.append(Int64(ns))
    ctl.append(Int64(nk))
    ctl.append(Int64(K))
    ctl.append(Int64(Int(counter.unsafe_ptr())))
    ctl.append(Int64(Int(table.unsafe_ptr())))
    ctl.append(Int64(maxw))
    ctl.append(Int64(Int(vals.unsafe_ptr())))
    ctl.append(Int64(Int(nuls.unsafe_ptr())))
    ctl.append(Int64(Int(ids.unsafe_ptr())))
    ctl.append(Int64(Int(sizes.unsafe_ptr())))
    for k in range(nk):
        ctl.append(Int64(key_slot[k]))
        ctl.append(Int64(1 if key_desc[k] else 0))
        ctl.append(Int64(1 if key_nf[k] else 0))
        ctl.append(Int64(key_rank[k]))
    for s in range(ns):
        ctl.append(Int64(sp.row_vregs[s]))
        ctl.append(Int64(sp.row_nregs[s]))
    var code = List[Int32]()
    var header: List[Int] = [MAGIC, p.n_regs, 0, len(p.pred) // 5, len(p.body) // 5, len(p.aggs) // 5,
                             p.pred_reg, len(p.cols), 0, 0, 0, 0, 0, 0, 0, 0]
    for h in header:
        code.append(Int32(h))
    for w in p.pred:
        code.append(w)
    for w in p.body:
        code.append(w)
    for w in p.aggs:
        code.append(w)
    var consts = List[Float64]()
    for v in p.consts:
        consts.append(Float64(from_bits=UInt64(v)))
    consts[p.ctl_const] = Float64(from_bits=UInt64(Int(ctl.unsafe_ptr())))
    var set_bytes = sp.set_bytes.copy()
    if len(set_bytes) == 0:
        set_bytes.append(0)
    var set_meta = sp.set_meta.copy()
    if len(set_meta) == 0:
        set_meta = [0, 0, 0]
    var tbl = _table_addresses(t, p, mask_addr)
    var dtypes = p.dtypes.copy()
    if len(dtypes) == 0:
        dtypes.append(1)
    var out_vals = List[Float64](length=max(p.n_slots, 1), fill=0.0)
    var out_cnt = List[Int64](length=1, fill=0)
    var rc = pipeline_keyed[type_of(_topk_hook)](Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                                                 Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()), 1,
                                                 0, t.rows(), Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()),
                                                 threads if threads > 0 else core_count(), _topk_hook, False)
    _ = code^
    _ = consts^
    _ = set_bytes^
    _ = set_meta^
    _ = tbl^
    _ = dtypes^
    _ = out_vals^
    _ = out_cnt^
    _ = ctl^
    _ = table^                                                # (the hook reads these through addresses: keep them alive past the call)
    if rc == E_OVERFLOW:
        raise Error(OVERFLOW_MESSAGE)
    if rc != 0:
        raise Error("the CPU kernel failed with code " + String(rc))
    var out = RowChunk()
    for s in range(ns):
        out.vals.append(List[Int64]())
        out.nuls.append(List[UInt8]())
    for w in range(min(Int(counter[0]), maxw)):
        for e in range(Int(sizes[w])):
            var at = w * (K + 1) + e
            out.ids.append(ids[at])
            for s in range(ns):
                out.vals[s].append(vals[at * ns + s])
                out.nuls[s].append(nuls[at * ns + s])
    out.n = len(out.ids)
    return out^


def run_rows(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, opts: PlannerOptions, dicts: DictAccess, mut notes: List[String],
             inp: _Inputs, sp: SqlProgram, items: List[_Item], trust: Bool) raises -> Table:
    """Reads the chunks, collects the selected rows (ordered, cut) and makes the result table."""
    var used = inp.used.copy()
    var ch = _open_chunks(src, used)
    var cols = List[ValCol]()
    for it in items:
        ref x = q.rexprs[it.expr]
        var vc = ValCol(x.kind, x.scale, 0)
        vc.src = x.src
        cols.append(vc^)
    var limit = q.limit
    var offset = q.offset
    var need = offset + limit
    var rows_in = 0
    var chunks_read = 0
    var t_run = perf_counter_ns()
    var ns_load = 0
    var stop_early = False
    for i in range(ch.n()):
        if plan.early_stop and rows_in >= need:
            stop_early = True
            break
        var t_load = perf_counter_ns()
        var chunk = _load_chunk(src, ch, i)
        if trust:
            _check_footer(src, ch, i, chunk)
        ns_load += perf_counter_ns() - t_load
        chunks_read += 1
        var n_rows = chunk.rows()
        var step = n_rows
        if plan.early_stop:                                     # read the chunk in slices, so a LIMIT stops it early
            step = max(VEC * 16, ((need * 4) // VEC + 1) * VEC)
        var at = 0
        while at < n_rows:
            if plan.early_stop and rows_in >= need:
                stop_early = True
                break
            var end = min(n_rows, at + step)
            var rc = _run_rows(chunk, sp, ch.mask_addr, at, end, opts.threads)
            for k in range(len(items)):
                ref it = items[k]
                if it.computed >= 0:
                    var vp = LP(unsafe_from_address=Int(rc.vals[it.computed].unsafe_ptr()))
                    var has_nul = len(rc.nuls[it.computed]) > 0
                    var np = BP(unsafe_from_address=Int(rc.nuls[it.computed].unsafe_ptr()) if has_nul else Int(vp))
                    _append_run(cols[k], vp, np, has_nul, rc.n)
                else:
                    _append_gathered(cols[k], chunk, inp.pos[it.col], IP(unsafe_from_address=Int(rc.ids.unsafe_ptr())), rc.n)
            rows_in += rc.n
            at = end
    var ms_run = Float64(perf_counter_ns() - t_run) / 1e6
    if src.kind == SRC_FRAME:
        notes.append("ran: frame " + src.name + ": " + commas(src.table.rows()) + " shared rows (the changed ones masked out) + " +
                     commas(len(src.overlay.delta)) + " changed or appended rows")
    elif inp.scan_mode:
        notes.append("ran: scan " + src.name + ": " + String(chunks_read) + " of " + String(src.scan.value().row_groups()) + " row groups read (" +
                     String(len(used)) + (" column" if len(used) == 1 else " columns") + "), one chunk each" +
                     ((", + " + commas(len(src.overlay.delta)) + " changed or appended rows") if len(src.overlay.delta) > 0 else ""))
    else:
        notes.append("ran: table " + src.name + ": " + String(chunks_read) + " chunk of " + commas(src.table.rows()) + " rows")
    notes.append("rows: " + commas(rows_in) + " rows selected" + (" (stopped early: the LIMIT was met)" if stop_early else ""))
    # order, offset, limit
    var t_ord = perf_counter_ns()
    var fin = List[Int]()
    if len(q.rorder) > 0:
        var kc = List[ValCol]()
        for o in q.rorder:
            kc.append(cols[_item_of(items, q, o.expr)].copy())
        fin = _order_rows(kc, q.rorder, dicts, rows_in)
        notes.append("order: sorted " + commas(rows_in) + " rows by " + String(len(q.rorder)) + (" key" if len(q.rorder) == 1 else " keys") + " (stable radix passes)")
    else:
        fin = List[Int](length=rows_in, fill=0)
        for i in range(rows_in):
            fin[i] = i
    var lo = min(offset, len(fin))
    var hi = len(fin) if limit < 0 else min(len(fin), lo + limit)
    var pick = List[Int]()
    for i in range(lo, hi):
        pick.append(fin[i])
    var outs = List[ValCol]()
    for e in q.routs:
        outs.append(_gather(cols[_item_of(items, q, e)], pick))
    var table = _to_table(q, outs, dicts)
    notes.append("time: " + ("decode " + _ms(Float64(ns_load) / 1e6) + " ms, " if inp.scan_mode else "") + "kernel " + _ms(ms_run - Float64(ns_load) / 1e6) +
                 " ms, order and output " + _ms(Float64(perf_counter_ns() - t_ord) / 1e6) + " ms")
    return table^


def _ms(x: Float64) -> String:
    return String(Float64(Int(x * 100.0)) / 100.0)


def _concat(a: ValCol, b: ValCol) -> ValCol:
    var out = ValCol(a.kind, a.scale, 0)
    out.src = a.src
    out.v = a.v.copy()
    out.v.extend(b.v.copy())
    if len(a.nul) > 0 or len(b.nul) > 0:
        out.nul = List[UInt8](length=len(a.v), fill=0)
        for i in range(len(a.nul)):
            out.nul[i] = a.nul[i]
        for i in range(len(b.v)):
            out.nul.append(b.nul[i] if len(b.nul) > 0 else UInt8(0))
    return out^


def run_topk_rows(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, opts: PlannerOptions, dicts: DictAccess, mut notes: List[String],
                  inp: _Inputs, sp: SqlProgram, items: List[_Item], trust: Bool) raises -> Table:
    """ORDER BY with a small LIMIT: each chunk's kernel run keeps the best K = limit + offset rows in per-worker heaps; the candidates
    (at most workers x K) are merged with the best so far and cut to K again, so the rest of the rows are never materialized."""
    var used = inp.used.copy()
    var ch = _open_chunks(src, used)
    var K = plan.top_k
    var nk = len(q.rorder)
    var key_slot = List[Int]()
    var key_desc = List[Bool]()
    var key_nf = List[Bool]()
    var key_rank = List[Int]()
    var ranks = List[List[Int64]]()                                 # alive until the kernel has run
    for o in q.rorder:
        key_slot.append(items[_item_of(items, q, o.expr)].computed)
        key_desc.append(o.desc)
        key_nf.append(o.nulls_first)
        ref x = q.rexprs[o.expr]
        if x.kind == TEXT.value:
            var words = List[String]()
            for c in range(dicts.count(x.src)):
                words.append(dicts.word(x.src, c))
            ranks.append(text_ranks(words))
            key_rank.append(Int(ranks[len(ranks) - 1].unsafe_ptr()))
        elif x.kind == FLOAT.value:
            key_rank.append(-1)
        else:
            key_rank.append(0)
    var best = List[ValCol]()
    for it in items:
        ref x = q.rexprs[it.expr]
        var vc = ValCol(x.kind, x.scale, 0)
        vc.src = x.src
        best.append(vc^)
    var scanned = 0
    var chunks_read = 0
    var t_run = perf_counter_ns()
    var ns_load = 0
    for i in range(ch.n()):
        var t_load = perf_counter_ns()
        var chunk = _load_chunk(src, ch, i)
        if trust:
            _check_footer(src, ch, i, chunk)
        ns_load += perf_counter_ns() - t_load
        chunks_read += 1
        scanned += chunk.rows()
        if chunk.rows() == 0:
            continue
        var rc = _run_topk(chunk, sp, ch.mask_addr, K, key_slot, key_desc, key_nf, key_rank, opts.threads)
        if rc.n == 0:
            continue
        # the candidates, in the order of their rows in the chunk (ties between equal keys keep the source order)
        var by_row = List[Int64](length=rc.n, fill=0)
        for j in range(rc.n):
            by_row[j] = Int64(rc.ids[j])
        var ident = List[Int](length=rc.n, fill=0)
        for j in range(rc.n):
            ident[j] = j
        var perm = _radix_order(by_row, ident, False)
        var comb = List[ValCol]()
        for k in range(len(items)):
            ref it = items[k]
            ref x = q.rexprs[it.expr]
            var cand = ValCol(x.kind, x.scale, 0)
            cand.src = x.src
            if it.computed >= 0:
                var vp = LP(unsafe_from_address=Int(rc.vals[it.computed].unsafe_ptr()))
                var has_nul = sp.row_nregs[it.computed] >= 0
                var np = BP(unsafe_from_address=Int(rc.nuls[it.computed].unsafe_ptr()) if has_nul else Int(vp))
                _append_run(cand, vp, np, has_nul, rc.n)
            else:
                _append_gathered(cand, chunk, inp.pos[it.col], IP(unsafe_from_address=Int(rc.ids.unsafe_ptr())), rc.n)
            comb.append(_concat(best[k], _gather(cand, perm)))
        var n = len(comb[0].v)
        var kc = List[ValCol]()
        for o in q.rorder:
            kc.append(comb[_item_of(items, q, o.expr)].copy())
        var ord = _order_rows(kc, q.rorder, dicts, n)
        var keep = List[Int]()
        for j in range(min(K, n)):
            keep.append(ord[j])
        for k in range(len(items)):
            best[k] = _gather(comb[k], keep)
    var ms_run = Float64(perf_counter_ns() - t_run) / 1e6
    _ = ranks^
    if src.kind == SRC_FRAME:
        notes.append("ran: frame " + src.name + ": " + commas(src.table.rows()) + " shared rows (the changed ones masked out) + " +
                     commas(len(src.overlay.delta)) + " changed or appended rows")
    elif inp.scan_mode:
        notes.append("ran: scan " + src.name + ": " + String(chunks_read) + " of " + String(src.scan.value().row_groups()) + " row groups read (" +
                     String(len(used)) + (" column" if len(used) == 1 else " columns") + "), one chunk each" +
                     ((", + " + commas(len(src.overlay.delta)) + " changed or appended rows") if len(src.overlay.delta) > 0 else ""))
    else:
        notes.append("ran: table " + src.name + ": " + String(chunks_read) + " chunk of " + commas(src.table.rows()) + " rows")
    var kept = len(best[0].v) if len(best) > 0 else 0
    notes.append("rows: the best " + commas(kept) + " of " + commas(scanned) + " rows kept in per-worker heaps, " + String(nk) + (" key" if nk == 1 else " keys"))
    var lo = min(q.offset, kept)
    var hi = kept if q.limit < 0 else min(kept, lo + q.limit)
    var pick = List[Int]()
    for i in range(lo, hi):
        pick.append(i)
    var outs = List[ValCol]()
    for e in q.routs:
        outs.append(_gather(best[_item_of(items, q, e)], pick))
    var t_out = perf_counter_ns()
    var table = _to_table(q, outs, dicts)
    notes.append("time: " + ("decode " + _ms(Float64(ns_load) / 1e6) + " ms, " if inp.scan_mode else "") + "kernel and merge " + _ms(ms_run - Float64(ns_load) / 1e6) +
                 " ms, output " + _ms(Float64(perf_counter_ns() - t_out) / 1e6) + " ms")
    return table^


struct RowStage(Movable):
    """A row query ready to run: its kernel program and the columns it collects."""
    var sp: SqlProgram
    var items: List[_Item]

    def __init__(out self, var sp: SqlProgram, var items: List[_Item]):
        self.sp = sp^
        self.items = items^


def prepare_rows(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, dicts: DictAccess, inp: _Inputs, mut notes: List[String],
                 report: Bool) raises -> RowStage:
    """Lowers a row query (the program computes what is not a plain column; a top-k also computes its sort keys) and, when `report`,
    says in the notes which columns it reads and what the kernel does."""
    var built = build_items(q, plan.top_k > 0)
    var sp: SqlProgram
    try:
        sp = lower_rows(q, built[1], inp.nullable.copy(), inp.pos.copy(), inp.chunk_cols, DictAccess(src.table, src.scan, dicts.extra),
                        _alias_lists(src), inp.masked)
    except e:
        var msg = String(e)
        if msg.startswith("the query is too big for one program"):
            raise sql_error(q.line, q.col, msg + ": it computes too many different expressions (about 20 in the select list and ORDER BY); split it in two")
        if msg.startswith("the query has more than"):
            raise sql_error(q.line, q.col, msg + " (a program holds 64 different numbers); split the query in two")
        raise e^
    if report:
        var names = String()
        var read = collect_columns(q)
        for k in range(len(read)):
            names += (", " if k > 0 else "") + src.table.name(read[k])
        notes.append("columns read: " + names + " (" + String(len(read)) + " of " + String(src.table.columns()) + ")")
        notes.append("kernel: " + String(sp.instr) + " instructions a row, " + String(len(built[1])) + " computed " + ("value" if len(built[1]) == 1 else "values") +
                     ", " + String(len(built[0]) - len(built[1])) + " plain " + ("column" if len(built[0]) - len(built[1]) == 1 else "columns") + " gathered")
        notes.append("device: CPU (SQL runs on the CPU)")
    return RowStage(sp^, built[0].copy())


def execute_rows(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, opts: PlannerOptions, dicts: DictAccess, mut notes: List[String],
                 inp: _Inputs, var stage: RowStage, trust: Bool) raises -> Table:
    if plan.top_k > 0:
        return run_topk_rows(src, q, plan, opts, dicts, notes, inp, stage.sp, stage.items, trust)
    return run_rows(src, q, plan, opts, dicts, notes, inp, stage.sp, stage.items, trust)
