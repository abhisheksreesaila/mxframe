"""Row queries (F-152): the materialize sink and the top-k sink of the executor.

A query with no aggregate returns rows: `SELECT a, b + 1 FROM t WHERE c > 3 ORDER BY a LIMIT 10`. The executor reads the source
in chunks (`exec.mojo`) and for each chunk runs one kernel program (`sqlprog.lower_rows`) that evaluates the WHERE and computes
the select list's expressions and the sort keys into registers. The kernel's block hook (`_rows_hook`, the pattern of
query.mojo's `_mat_hook`) then writes, for each block of 4,096 rows, the numbers of the rows that passed and the values of
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
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .frame import text_ranks
from .query import Program, MAGIC, SRC_WORDS, CTL_REG
from .kernels._fpenv import core_count
from .kernels.pipeline_cpu import IP, LP, FP, BP, VEC, E_OVERFLOW, pipeline_keyed
from .sql import sql_error
from .bind import SqlSource, BoundQuery, SortKey, B_COL, B_LIT, SRC_FRAME, pow10_checked
from .planner import PhysicalPlan, PlannerOptions, commas
from .sqlprog import SqlProgram, DictAccess, lower_rows, bkey
from .sqleval import ValCol
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


def build_items(q: BoundQuery) -> Tuple[List[_Item], List[Int]]:
    """The columns the rows are collected into (the select list's, then the sort keys' that are not in it) and, of those, the
    expressions the program computes (everything that is not a plain column)."""
    var items = List[_Item]()
    var computed = List[Int]()
    var exprs = q.routs.copy()
    for o in q.rorder:
        exprs.append(o.expr)
    for e in exprs:
        if _item_of(items, q, e) >= 0:
            continue
        ref x = q.rexprs[e]
        if x.op == B_COL:
            items.append(_Item(e, bkey(q.rexprs, e), Int(x.value), -1))
        else:
            computed.append(e)
            items.append(_Item(e, bkey(q.rexprs, e), -1, len(computed) - 1))
    return (items^, computed^)


def run_rows(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, opts: PlannerOptions, dicts: DictAccess, mut notes: List[String],
             inp: _Inputs, var sp: SqlProgram, items: List[_Item], trust: Bool) raises -> Table:
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
