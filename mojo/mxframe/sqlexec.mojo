"""Running SQL (F-150): `sql(catalog, text)` -> `SqlResult` holding a Table.

    var cat = Catalog()
    cat.add_table("orders", orders)              # or add_frame(name, frame), add_scan(name, scan)
    var r = sql(cat, "SELECT region, sum(revenue) FROM orders WHERE qty > 2 GROUP BY region ORDER BY 2 DESC LIMIT 5")
    r.table; r.notes; r.explain; r.ms

The steps: parse and bind (`bind.mojo`), plan from statistics (`planner.mojo`: dense or hashed groups, with the reason),
lower to one kernel program (`sqlprog.mojo`), run it over the source's chunks and merge their partial results,
then finish in the SQL way (`sqleval.mojo` for HAVING / ORDER BY / select expressions), producing an owned `Table`.

Chunks: a table is one chunk; a frame is its shared table with the changed rows masked out plus a small table of
the changed and appended rows; a scan is one chunk per row group. Each chunk's partial result (sums and counts add,
min / max combine; hashed groups merge by key) folds into one.
"""
from std.time import perf_counter_ns
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, F32_COLUMN
from .frame import NULL_EDIT, text_ranks, _radix_order
from .query import (Program, QueryResult, _run_cpu, _run_mat, _key_from_reg0, _make_result, _merge_partial, _identity_partial, _ScanMask, _group_mask,
                    OVERFLOW_MESSAGE, overflow_message, CPU, MAGIC, A_MIN, A_MAX, A_FSUM, MORSEL_ROWS, WINDOW_BYTES, SRC_WORDS, fold_float, slot_identity)
from .kernels._fpenv import core_count, par_n
from .kernels.pipeline_cpu import IP, LP, FP, BP, VEC, E_OVERFLOW, _no_front, _no_keys, pipeline_keyed
from .kernels.hashagg_cpu import hash_aggregate
from .holistic import HolStates, HolSpec, HolResult, A_DISTINCT, A_PCT, A_MEDIAN, window_blocks, float_key
from .sql import sql_error, EXPLAIN_NONE, EXPLAIN_ANALYZE
from .bind import (copy_query, Catalog, SqlSource, BoundQuery, SortKey, bind_sql, SRC_TABLE, SRC_FRAME, SRC_SCAN, B_COL, F_COUNT_STAR,
                   F_COUNT, F_SUM, F_MIN, F_MAX, F_AVG, F_COUNT_DISTINCT, F_MEDIAN, pow10_checked)
from .plan import LogicalPlan, build_plan, executable
from .post import run_post
from .join import needed_columns, side_query, joined_query, virtual_schema, all_needed, needed_after, table_of_column, filter_query, final_query
from .joincore import JoinKeys, JoinSide, normalize_keys, unmatchable_not_null, join_pairs, select_rows, gather_join, JoinPairs, JOIN_BUDGET_BYTES
from .kernels.hashjoin_cpu import hj_build, PM_SEMI, PM_ANTI, PM_ANTI_NA, PM_NONE
from .kargs import KernelRun
from .jfront import FusedJoin, front_spec, jf_front
from std.memory import ArcPointer
from .rows import RowStage, prepare_rows, execute_rows, _Item
from .sqlprog import lower_rows
from .planner import (plan_sinks, Planner, RulePlanner, PlannerOptions, PhysicalPlan, Stats, ColStat, G_NONE, G_DENSE, G_HASHED, commas, JoinStats, JoinPlan,
                      BUILD_AUTO, BUILD_LEFT, BUILD_RIGHT)
from .sqlprog import SqlProgram, DictAccess, lower_sql, collect_columns, PHASE_ALL, PHASE_PLAIN, PHASE_HOL
from .sqleval import ValCol, float_bits, ld_div
from .exec import (_table_addresses, _column_range, _is_range_kind, _key_columns, prepare_ranges, _alias_lists, _delta_nulls, make_stats,
                   delta_table, _gather, _sort_keys, _order_rows, _SortCols, _top_rows, _fill_column, _unique_name, _to_table, _ones_mask,
                   _Chunks, _open_chunks, _load_chunk, LIE, _check_footer, _all, _Inputs, _inputs)


struct SqlResult(Movable):
    var table: Table
    var explain: String              # the plan as text (always filled; EXPLAIN returns it as the table too)
    var ms: Float64
    var device: Int                  # CPU (the SQL path runs on the CPU in F-150)
    var fallback_reason: String
    var notes: List[String]          # what the planner decided and what ran, one line each: tests assert them
    var actuals: List[String]        # F-155: what each operator really did (rows, milliseconds), one line each: EXPLAIN ANALYZE prints them

    def __init__(out self):
        self.actuals = List[String]()
        self.table = Table.empty()
        self.explain = String()
        self.ms = 0.0
        self.device = CPU
        self.fallback_reason = String()
        self.notes = List[String]()

    def take_table(mut self) -> Table:
        """The result table, moved out (the result keeps an empty one)."""
        var t = self.table^
        self.table = Table.empty()
        return t^


# ── partial results ──────────────────────────────────────────────────────────────────────────────────

struct Partial(Movable):
    var slots: List[List[Int]]       # [slot][group]
    var counts: List[Int]            # rows per group
    var kw: List[List[Int64]]        # hashed: [key word][group]

    def __init__(out self):
        self.slots = List[List[Int]]()
        self.counts = List[Int]()
        self.kw = List[List[Int64]]()

    def groups(self) -> Int:
        return len(self.counts)


def _checked_add(a: Int, b: Int) raises -> Int:
    var x = Int64(a)
    var y = Int64(b)
    var s = x + y
    if ((x ^ s) & (y ^ s)) < 0:
        raise Error(OVERFLOW_MESSAGE)
    return Int(s)


def _mix(h: UInt64) -> UInt64:
    var x = h ^ (h >> 33)
    x *= 0xFF51AFD7ED558CCD
    x ^= x >> 33
    x *= 0xC4CEB9FE1A85EC53
    x ^= x >> 33
    return x


def _hash_group(kw: List[List[Int64]], g: Int) -> UInt64:
    var h = UInt64(0x9E3779B97F4A7C15)
    for k in range(len(kw)):
        h = _mix(h ^ UInt64(kw[k][g]))
    return h


struct _KeyIndex(Movable):
    """The hash table from key words to group number over an accumulator's groups, kept across the merges of a scan's row
    groups: it grows when the groups do, and only the groups added since it last looked are inserted."""
    var table: List[Int32]
    var mask: UInt64
    var indexed: Int

    def __init__(out self):
        self.table = List[Int32](length=16, fill=-1)
        self.mask = 15
        self.indexed = 0

    def insert(mut self, acc: Partial, g: Int):
        var pos = Int(_hash_group(acc.kw, g) & self.mask)
        while self.table[pos] >= 0:
            pos = (pos + 1) & Int(self.mask)
        self.table[pos] = Int32(g)

    def sync(mut self, acc: Partial, room: Int):
        """Makes room for `room` more groups and indexes the groups acc gained since."""
        var cap = len(self.table)
        if 2 * (acc.groups() + room) > cap:
            while cap < 2 * (acc.groups() + room):
                cap *= 2
            self.table = List[Int32](length=cap, fill=-1)
            self.mask = UInt64(cap - 1)
            self.indexed = 0
        for g in range(self.indexed, acc.groups()):
            self.insert(acc, g)
        self.indexed = acc.groups()


def _merge_hashed(mut acc: Partial, other: Partial, ops: List[Int], mut ix: _KeyIndex) raises:
    """Folds `other`'s groups into `acc` by key."""
    var og = other.groups()
    if og == 0:
        return
    if acc.groups() == 0:
        acc.slots = other.slots.copy()
        acc.counts = other.counts.copy()
        acc.kw = other.kw.copy()
        return
    ix.sync(acc, og)
    for o in range(og):
        var pos = Int(_hash_group(other.kw, o) & ix.mask)
        var found = -1
        while ix.table[pos] >= 0:
            var g = Int(ix.table[pos])
            var same = True
            for k in range(len(acc.kw)):
                if acc.kw[k][g] != other.kw[k][o]:
                    same = False
                    break
            if same:
                found = g
                break
            pos = (pos + 1) & Int(ix.mask)
        if found < 0:
            found = acc.groups()
            for s in range(len(acc.slots)):
                acc.slots[s].append(other.slots[s][o])
            acc.counts.append(other.counts[o])
            for k in range(len(acc.kw)):
                acc.kw[k].append(other.kw[k][o])
            ix.table[pos] = Int32(found)
            ix.indexed = acc.groups()
            continue
        for s in range(len(acc.slots)):
            var a = acc.slots[s][found]
            var b = other.slots[s][o]
            if ops[s] >= A_FSUM:
                acc.slots[s][found] = fold_float(ops[s], a, b)
            elif ops[s] == A_MIN:
                acc.slots[s][found] = min(a, b)
            elif ops[s] == A_MAX:
                acc.slots[s][found] = max(a, b)
            else:
                acc.slots[s][found] = _checked_add(a, b)
        acc.counts[found] += other.counts[o]


def _hashed_result(run: KernelRun, res_r: Int, n_slots: Int, nkw: Int) -> Partial:
    """The groups hash_aggregate found, from the addresses it left in the run's result region (the run is a parameter: alive here)."""
    var res = run.i64(res_r)
    var out = Partial()
    var g = Int(res[0])
    var stride = max(g, 1)
    var vals = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[1]))
    var cnt = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[2]))
    for s in range(n_slots):
        var col = List[Int]()
        col.reserve(g)
        for x in range(g):
            col.append(Int(vals[s * stride + x]))
        out.slots.append(col^)
    for x in range(g):
        out.counts.append(Int(cnt[x]))
    for k in range(nkw):
        var kp = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[3 + k]))
        var col = List[Int64]()
        col.reserve(g)
        for x in range(g):
            col.append(kp[x])
        out.kw.append(col^)
        kp.free()
    vals.free()
    cnt.free()
    return out^


def _run_hashed(t: SharedTable, sp: SqlProgram, mask_addr: Int, nkw: Int, threads: Int, fj: Optional[ArcPointer[FusedJoin]]) raises -> Partial:
    """The kernel with hashed groups (hashagg_cpu.hash_aggregate): each worker numbers the key tuples of its rows in a local
    table, the partitions are merged in parallel, and the result is the groups with their key words, slots and counts."""
    var run = KernelRun(sp.p, sp.set_bytes, sp.set_meta, t, mask_addr)
    var res_r = run.region(3 + max(nkw, 1))

    def keyw(c: Int, i0: Int, sl: IP, nsel: Int, use_sel: Bool, dst: LP, rg: FP, wsp: BP) {imm} -> Bool:
        for k in range(nkw):
            var src = LP(unsafe_from_address=Int(rg) + k * VEC * 8)
            var o = dst + k * VEC
            for j in range(nsel):
                o[j] = src[j]
        return True

    var rc: Int
    if fj:
        var fs = front_spec(run, fj.value()[], sp.p)

        def front(k: Int, i0: Int, n: Int, sel: IP, ws: BP) {imm} -> Int:
            return jf_front(fs, i0, n, sel, ws)

        rc = run.hashed_fused(keyw, front, fs.ws_bytes, nkw, t.rows(), threads, res_r)
    else:
        rc = run.hashed(keyw, nkw, t.rows(), threads, res_r)
    if rc == -7 or rc == E_OVERFLOW:
        raise Error(overflow_message(sp.p.has_float))
    if rc != 0:
        raise Error("the hashed group-by kernel failed with code " + String(rc))
    return _hashed_result(run, res_r, sp.p.n_slots, nkw)


# ── text for EXPLAIN ─────────────────────────────────────────────────────────────────────────────────

def _explain_text(lp: LogicalPlan, notes: List[String]) -> String:
    var out = String("Logical plan\n")
    var lines = lp.tree()
    for i in range(len(lines)):
        out += "  " * (1 + lines[i][0]) + lines[i][1] + "\n"
    out += "Physical plan\n"
    for n in notes:
        out += "  " + n + "\n"
    return out^


def _finish(q: BoundQuery, plan: PhysicalPlan, sp: SqlProgram, acc: Partial, hres: HolResult, hol_of: List[Int],
            dicts: DictAccess, opts: PlannerOptions, mut notes: List[String]) raises -> Table:
    # the groups that exist: only non-empty ones, except the one row of a query with no GROUP BY
    var t_a = perf_counter_ns()
    var gids = List[Int]()
    if plan.group == G_NONE:
        gids.append(0)
    elif plan.group == G_DENSE:
        for g in range(plan.domain):
            if acc.counts[g] > 0:
                gids.append(g)
    else:
        for g in range(acc.groups()):
            gids.append(g)
    var n = len(gids)
    var t_b = perf_counter_ns()
    # group keys
    var keys = List[ValCol]()
    var kwat = 0
    for k in range(len(plan.keys)):
        ref lay = plan.keys[k]
        var vc = ValCol(lay.kind, lay.scale, n)
        vc.src = lay.src
        for i in range(n):
            if plan.group == G_DENSE:
                var part = (gids[i] // lay.stride) % lay.radix
                if lay.nullable:
                    if part == 0:
                        vc.set_null(i)
                        continue
                    part -= 1
                vc.v[i] = Int64(part) + (0 if (lay.kind == TEXT.value or lay.kind == BOOL.value) else lay.lo)
            else:
                if lay.nullable and acc.kw[kwat + 1][gids[i]] != 0:
                    vc.set_null(i)
                    continue
                vc.v[i] = acc.kw[kwat][gids[i]]
        kwat += lay.words
        keys.append(vc^)
    # aggregates
    var aggs = List[ValCol]()
    for a in range(len(q.aggs)):
        ref sp_a = q.aggs[a]
        var vc = ValCol(sp_a.kind, sp_a.scale, n)
        var vs = sp.agg_val_slot[a]
        var ns = sp.agg_nul_slot[a]
        var h = hol_of[a]
        for i in range(n):
            var g = gids[i]
            var rows = acc.counts[g]
            var nonnull = rows - (acc.slots[ns][g] if ns >= 0 else 0)
            var f = sp_a.func
            var arg_float = sp_a.arg >= 0 and q.rexprs[sp_a.arg].kind == FLOAT.value
            if sp_a.hol:                                          # the holistic states: count(DISTINCT), median, quantile_disc, min / max of TEXT
                var present = hres.counts[h][g] > 0
                if f == F_COUNT_DISTINCT:
                    vc.v[i] = Int64(hres.vals[h][g])
                elif not present:
                    vc.set_null(i)
                elif f == F_MEDIAN and sp_a.kind == FLOAT.value and not arg_float:
                    vc.v[i] = float_bits(Float64(hres.vals[h][g]) / 10.0)
                elif f == F_MEDIAN and arg_float and q.rexprs[sp_a.arg].scale == F32_COLUMN:       # median of REAL is a REAL: the interpolation rounded to 32 bits
                    vc.v[i] = float_bits(Float64(Float32(Float64(from_bits=UInt64(Int64(hres.vals[h][g]))))))
                else:
                    vc.v[i] = Int64(hres.vals[h][g])
            elif f == F_COUNT_STAR:
                vc.v[i] = Int64(rows)
            elif f == F_COUNT:
                vc.v[i] = Int64(nonnull)
            elif f == F_SUM or f == F_MIN or f == F_MAX:
                if nonnull == 0:
                    vc.set_null(i)
                elif arg_float and f != F_SUM:                       # NaN is above every number: MAX is NaN when there is one, MIN only when there is nothing else
                    var nans = acc.slots[sp.agg_nan_slot[a]][g]
                    if (f == F_MAX and nans > 0) or (f == F_MIN and nonnull - nans == 0):
                        vc.v[i] = Int64(0x7FF8000000000000)
                    else:
                        vc.v[i] = Int64(acc.slots[vs][g])
                else:
                    vc.v[i] = Int64(acc.slots[vs][g])
            elif f == F_AVG:
                if nonnull == 0:
                    vc.set_null(i)
                elif arg_float:                                      # a sum of doubles over the count, in doubles
                    vc.v[i] = float_bits(Float64(from_bits=UInt64(Int64(acc.slots[vs][g]))) / Float64(nonnull))
                else:
                    var den = UInt128(nonnull) * UInt128(Int(pow10_checked(q.rexprs[sp_a.arg].scale)))
                    vc.v[i] = float_bits(ld_div(Int64(acc.slots[vs][g]), den))
        if sp_a.kind == TEXT.value:
            vc.src = q.rexprs[sp_a.arg].src
        aggs.append(vc^)
    var t_c = perf_counter_ns()
    # HAVING, the select list, ORDER BY, OFFSET and LIMIT: a row query over the groups (post.mojo)
    var table = run_post(q, keys, aggs, dicts, opts, n, notes)
    notes.append("finish: groups " + _ms(Float64(t_b - t_a) / 1e6) + " ms, columns " + _ms(Float64(t_c - t_b) / 1e6) + " ms, then HAVING, ORDER BY and the select list over the groups " +
                 _ms(Float64(perf_counter_ns() - t_c) / 1e6) + " ms")
    return table^


# ── running ──────────────────────────────────────────────────────────────────────────────────────────

def _holistic_states(q: BoundQuery, groups: Int, sp: SqlProgram, dicts: DictAccess, rows: Int) raises -> HolStates:
    var specs = List[HolSpec]()
    var ranks = List[List[Int64]]()
    for _ in range(sp.n_src):
        ranks.append(List[Int64]())
    for a in range(len(q.aggs)):
        if sp.agg_src[a] < 0:
            continue
        ref sa = q.aggs[a]
        var kind = A_DISTINCT if sa.func == F_COUNT_DISTINCT else A_MEDIAN if sa.func == F_MEDIAN else A_PCT
        specs.append(HolSpec(a, kind, sa.p, 0, sp.agg_src[a]))
        if sa.func != F_COUNT_DISTINCT and q.rexprs[sa.arg].kind == TEXT.value and len(ranks[sp.agg_src[a]]) == 0:
            var c = Int(q.rexprs[sa.arg].value)
            var words = List[String]()
            for k in range(dicts.count(c)):
                words.append(dicts.word(c, k))
            ranks[sp.agg_src[a]] = text_ranks(words)
    return HolStates(groups, specs^, ranks^, rows, sp.p.src_float.copy())


def _run_dense_fused(t: SharedTable, sp: SqlProgram, mask_addr: Int, threads: Int, fj: ArcPointer[FusedJoin]) raises -> QueryResult:
    """The dense / global aggregate over the rows the join's front stage lets through (the kernel call of query.mojo's `_run_cpu`, fused)."""
    ref p = sp.p
    var run = KernelRun(p, sp.set_bytes, sp.set_meta, t, mask_addr, p.groups)
    var fs = front_spec(run, fj[], p)

    def front(k: Int, i0: Int, n: Int, sel: IP, ws: BP) {imm} -> Int:
        return jf_front(fs, i0, n, sel, ws)

    var rc: Int
    if p.keyed:
        rc = run.fused(_key_from_reg0, front, fs.ws_bytes, True, p.groups, 0, t.rows(), threads)
    else:
        rc = run.fused(_no_keys, front, fs.ws_bytes, False, 1, 0, t.rows(), threads)
    if rc == E_OVERFLOW:
        raise Error(overflow_message(p.has_float))
    if rc != 0:
        raise Error("the CPU kernel failed with code " + String(rc))
    return _make_result(p, run.out_vals, run.out_cnt)


def _fold(sp: SqlProgram, plan: PhysicalPlan, chunk: SharedTable, mask_addr: Int, mut hs: HolStates, ops: List[Int], threads: Int,
          mut acc: Partial, mut dense_acc: QueryResult, mut ix: _KeyIndex, fj: Optional[ArcPointer[FusedJoin]]) raises:
    """Runs the program over one chunk and folds its partial result into the accumulator."""
    if plan.group == G_HASHED:
        _merge_hashed(acc, _run_hashed(chunk, sp, mask_addr, plan.key_words(), threads, fj), ops, ix)
        return
    var r: QueryResult
    if fj:
        r = _run_dense_fused(chunk, sp, mask_addr, threads, fj.value())
    elif sp.p.mat:
        r = _run_mat(chunk, sp.p, mask_addr, hs, 0)
    else:
        r = _run_cpu(chunk, sp.p, mask_addr, 0, -1, 0, threads)
    _merge_partial(dense_acc, r, ops)


# ── hashed groups with holistic aggregates: a second pass maps each row to its group ─────────────────

def _hmat_hook(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    """The group id of each selected row, found by its key words in the table of the groups pass 1 found (register 0 holds
    the address of the control block, registers 1 .. nkw the key words), then the pairs of every holistic source, as
    query.mojo's `_mat_hook` writes them."""
    var ctl = LP(unsafe_from_address=Int(LP(unsafe_from_address=Int(regs))[0]))
    var nsrc = Int(ctl[0])
    var tail = 6 + SRC_WORDS * nsrc
    var nkw = Int(ctl[tail])
    var table = IP(unsafe_from_address=Int(ctl[tail + 1]))
    var mask = UInt64(ctl[tail + 2])
    var kwp = LP(unsafe_from_address=Int(ctl[tail + 3]))
    for j in range(nsel):
        var h = UInt64(0x9E3779B97F4A7C15)
        for k in range(nkw):
            h = _mix(h ^ UInt64(LP(unsafe_from_address=Int(regs + (1 + k) * VEC))[j]))
        var pos = Int(h & mask)
        var gid = -1
        while True:
            var g = Int(table[pos])
            if g < 0:
                break
            var same = True
            for k in range(nkw):
                if LP(unsafe_from_address=Int(kwp[k]))[g] != LP(unsafe_from_address=Int(regs + (1 + k) * VEC))[j]:
                    same = False
                    break
            if same:
                gid = g
                break
            pos = (pos + 1) & Int(mask)
        dst[j] = Int32(gid)
    var groups = ctl[2]
    var base = row0 - Int(ctl[1])
    for s in range(nsrc):
        var c = ctl + (6 + SRC_WORDS * s)
        var vals = LP(unsafe_from_address=Int(regs + Int(c[0]) * VEC))
        var gp = IP(unsafe_from_address=Int(c[2])) + base
        var vp = LP(unsafe_from_address=Int(c[3])) + base
        var o = 0
        if c[1] < 0:
            for j in range(nsel):
                var g = Int64(dst[j])
                gp[o] = Int32(g)
                vp[o] = vals[j]
                o += Int((g >= 0) & (g < groups))
        else:
            var valid = LP(unsafe_from_address=Int(regs + Int(c[1]) * VEC))
            for j in range(nsel):
                var g = Int64(dst[j])
                gp[o] = Int32(g)
                vp[o] = vals[j]
                o += Int((g >= 0) & (g < groups) & (valid[j] != 0))
        if c[10] != 0:
            for q in range(o):
                vp[q] = float_key(vp[q])
        IP(unsafe_from_address=Int(c[4]))[base // VEC] = Int32(o)
    return True


struct _GroupLookup(Movable):
    """The hash table from key words to group number, over the groups of pass 1."""
    var table: List[Int32]
    var kwp: List[Int]
    var mask: Int
    var nkw: Int
    var groups: Int

    def __init__(out self, acc: Partial):
        self.groups = acc.groups()
        self.nkw = len(acc.kw)
        var cap = 16
        while cap < 2 * max(self.groups, 1):
            cap *= 2
        self.table = List[Int32](length=cap, fill=-1)
        self.mask = cap - 1
        self.kwp = List[Int]()
        for k in range(self.nkw):
            self.kwp.append(Int(acc.kw[k].unsafe_ptr()))
        for g in range(self.groups):
            var pos = Int(_hash_group(acc.kw, g) & UInt64(self.mask))
            while self.table[pos] >= 0:
                pos = (pos + 1) & self.mask
            self.table[pos] = Int32(g)


def _absorb_window(run: KernelRun, mut hs: HolStates, s: Int, g_r: Int, v_r: Int, c_r: Int, nblk: Int) raises:
    """The window's pairs of source s into the states (the run is a parameter: the regions stay alive while the states read them)."""
    hs.absorb(s, window_blocks(Int(run.i32(g_r)), Int(run.i64(v_r)), Int(run.i32(c_r)), nblk, VEC))


def _run_hol_hashed(t: SharedTable, sp: SqlProgram, look: _GroupLookup, mask_addr: Int, mut hs: HolStates, threads: Int) raises:
    """Pass 2 over one chunk: the kernel runs the predicate and the key words, `_hmat_hook` finds each row's group and
    writes the holistic sources' (group, value) pairs, which the states take in windows (as query.mojo's `_run_mat`)."""
    ref p = sp.p
    var n = t.rows()
    var rows = max(MORSEL_ROWS, (WINDOW_BYTES // (12 * max(p.n_src, 1))) // MORSEL_ROWS * MORSEL_ROWS)
    var done = 0
    var first = True
    var G = max(look.groups, 1)
    while first or done < n:
        var end = min(n, done + rows)
        var nblk = (end - done + VEC - 1) // VEC
        var run = KernelRun(p, sp.set_bytes, sp.set_meta, t, mask_addr, G)
        var ctl = run.region(6 + SRC_WORDS * p.n_src + 4)
        var cw = run.i64(ctl)
        cw[0] = Int64(p.n_src)
        cw[1] = Int64(done)
        cw[2] = Int64(look.groups)
        var g_r = List[Int]()
        var v_r = List[Int]()
        var c_r = List[Int]()
        for s in range(p.n_src):
            g_r.append(run.bytes_region(4 * max(end - done, 1), False))
            v_r.append(run.region(max(end - done, 1), False))
            c_r.append(run.bytes_region(4 * max(nblk, 1)))
            var w = cw + (6 + SRC_WORDS * s)
            w[0] = Int64(p.src_vreg[s])
            w[1] = Int64(p.src_valid[s])
            w[2] = Int64(Int(run.i32(g_r[s])))
            w[3] = Int64(Int(run.i64(v_r[s])))
            w[4] = Int64(Int(run.i32(c_r[s])))
            w[10] = Int64(1 if p.src_float[s] else 0)
        var tail = cw + (6 + SRC_WORDS * p.n_src)
        tail[0] = Int64(look.nkw)
        tail[1] = Int64(Int(look.table.unsafe_ptr()))
        tail[2] = Int64(look.mask)
        tail[3] = Int64(Int(look.kwp.unsafe_ptr()))
        run.control(ctl)
        var rc = run.keyed(_hmat_hook, G, done, end, threads)
        if rc == E_OVERFLOW:
            raise Error(overflow_message(p.has_float))
        if rc != 0:
            raise Error("the CPU kernel failed with code " + String(rc))
        for s in range(p.n_src):
            _absorb_window(run, hs, s, g_r[s], v_r[s], c_r[s], nblk)
        first = False
        done = end


def _lower(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, dicts: DictAccess, inp: _Inputs, phase: Int) raises -> SqlProgram:
    try:
        return lower_sql(q, plan, inp.nullable.copy(), inp.pos.copy(), inp.chunk_cols, DictAccess(src.table, src.scan, dicts.extra),
                         _alias_lists(src), inp.masked, phase)
    except e:
        var msg = String(e)
        if msg.startswith("the query is too big for one program"):
            raise sql_error(q.line, q.col, msg + ": it holds too many aggregates over different expressions (about 20); split the select list into two queries")
        if msg.startswith("the query has more than"):
            raise sql_error(q.line, q.col, msg + " (a program holds 64 different numbers); split the query in two")
        raise e^


def _kernel_notes(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, dicts: DictAccess, inp: _Inputs, sp: SqlProgram, mut notes: List[String]):
    var names = String()
    var k = 0
    var read = collect_columns(q)
    for c in read:
        names += (", " if k > 0 else "") + src.table.name(c)
        k += 1
    notes.append("columns read: " + (names if k > 0 else "none") + " (" + String(len(read)) + " of " + String(src.table.columns()) + ")")
    notes.append("kernel: " + String(sp.instr) + " instructions a row" + (", " + String(len(sp.set_meta) // 3) + " sets" if len(sp.set_meta) > 0 else "") +
                 " (" + String(sp.p.n_slots) + (" slot" if sp.p.n_slots == 1 else " slots") + ")")
    if plan.group == G_HASHED and plan.holistic:
        notes.append("holistic: a second pass maps each row to its group for count(DISTINCT), median, quantile_disc and min / max of TEXT")
    notes.append("device: CPU (SQL runs on the CPU)")


def _execute(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, opts: PlannerOptions, dicts: DictAccess,
             mut notes: List[String], inp: _Inputs, var sp: SqlProgram, trust: Bool = True, fj: Optional[ArcPointer[FusedJoin]] = None) raises -> Table:
    var used = inp.used.copy()
    var scan_mode = inp.scan_mode
    var two_pass = plan.group == G_HASHED and plan.holistic
    var ops = sp.slot_ops.copy()
    var threads = opts.threads
    var rows_total = -1 if src.kind != SRC_TABLE else src.rows()
    var hs = _holistic_states(q, sp.p.groups, sp, dicts, rows_total) if (sp.p.mat and not two_pass) else \
             HolStates(1, List[HolSpec](), List[List[Int64]]())
    var domain = sp.p.groups                                      # (a dense plan with a footer-stated range has one extra, for rows outside it)
    var acc = Partial()
    var ix = _KeyIndex()
    var dense_acc = _identity_partial(sp.p.n_slots, domain, ops)
    var ch = _open_chunks(src, used)
    if fj:
        ch.override = fj.value()[].probe
    var delta_rows = len(src.overlay.delta)
    var t_run = perf_counter_ns()
    var ns_load = 0
    for i in range(ch.n()):
        var t_load = perf_counter_ns()
        var chunk = _load_chunk(src, ch, i)
        if trust:
            _check_footer(src, ch, i, chunk)
        ns_load += perf_counter_ns() - t_load
        _fold(sp, plan, chunk, ch.mask_addr, hs, ops, threads, acc, dense_acc, ix, fj)
    if sp.guard_slot >= 0 and dense_acc.counts[sp.guard_slot] > 0:
        raise Error(LIE + String(dense_acc.counts[sp.guard_slot]) + (" row lies" if dense_acc.counts[sp.guard_slot] == 1 else " rows lie") +
                    " outside the range of a group key that the footer gave")
    var ms_run = Float64(perf_counter_ns() - t_run) / 1e6
    var ms_load = Float64(ns_load) / 1e6
    if src.kind == SRC_FRAME:
        notes.append("ran: frame " + src.name + ": " + commas(src.table.rows()) + " shared rows (the changed ones masked out) + " +
                     commas(delta_rows) + " changed or appended rows")
    elif scan_mode:
        notes.append("ran: scan " + src.name + ": " + String(len(ch.groups)) + " of " + String(src.scan.value().row_groups()) +
                     " row groups read (" + String(len(used)) + (" column" if len(used) == 1 else " columns") + "), one chunk each" +
                     ((", + " + commas(delta_rows) + " changed or appended rows") if delta_rows > 0 else ""))
    else:
        notes.append("ran: table " + src.name + ": 1 chunk of " + commas(src.table.rows()) + " rows")
    if plan.group != G_HASHED:
        acc.slots = dense_acc.slots.copy()
        acc.counts = dense_acc.counts.copy()
    var hres = HolResult()
    var hol_of = List[Int](length=len(q.aggs), fill=-1)
    var t_hol = perf_counter_ns()
    if two_pass:
        # pass 2: the groups are known, find each row's group again and feed the holistic states
        var sp2 = _lower(src, q, plan, dicts, inp, PHASE_HOL)
        var look = _GroupLookup(acc)
        var hs2 = _holistic_states(q, max(acc.groups(), 1), sp2, dicts, rows_total)
        var ch2 = _open_chunks(src, used)
        for i in range(ch2.n()):
            var chunk = _load_chunk(src, ch2, i)
            _run_hol_hashed(chunk, sp2, look, ch2.mask_addr, hs2, threads)
        notes.append("ran: holistic second pass over the " + String(ch2.n()) + (" chunk" if ch2.n() == 1 else " chunks") + " mapped each row to one of " +
                     commas(acc.groups()) + " groups")
        hres = hs2.finish()
        for h in range(len(hs2.specs)):
            hol_of[hs2.specs[h].agg] = h
    elif sp.p.mat:
        hres = hs.finish()
        for h in range(len(hs.specs)):
            hol_of[hs.specs[h].agg] = h
    var t_fin = perf_counter_ns()
    var table = _finish(q, plan, sp, acc, hres, hol_of, dicts, opts, notes)
    notes.append("time: " + ("decode " + _ms(ms_load) + " ms, " if scan_mode else "") + "kernel " + _ms(ms_run - ms_load) + " ms, holistic states " + _ms(Float64(t_fin - t_hol) / 1e6) + " ms, finish " + _ms(Float64(perf_counter_ns() - t_fin) / 1e6) + " ms")
    return table^


def _ms(x: Float64) -> String:
    return String(Float64(Int(x * 100.0)) / 100.0)


def _fuse_inputs(mut inp: _Inputs, fj: FusedJoin):
    """The program of a fused join reads the probe table's own columns where the joined column is one, and placeholder columns after them
    (the front stage points those at the gathered build-side values)."""
    var np = fj.probe.columns()
    for v in range(fj.virtual_columns()):
        inp.pos[v] = fj.vreal[v] if fj.vprobe[v] else np + fj.vpartner[v]
    inp.chunk_cols = np + len(fj.partner)


def _prefill_ranges(mut src: SqlSource, jq: BoundQuery, pt: SharedTable, bt: SharedTable, vprobe: List[Bool], vreal: List[Int]):
    """The value range of each integer-like joined column a group key reads, from the real tables (the schema holds no rows to scan)."""
    var cols = List[Int]()
    for k in jq.keys:
        _key_columns(jq, k.expr, cols)
    for c in cols:
        var r = _column_range(pt if vprobe[c] else bt, vreal[c])
        src.range_cols.append(c)
        src.range_lo.append(r[1])
        src.range_hi.append(r[2])
        src.range_has.append(r[0])


def _side_names(q: BoundQuery, t: Int) -> String:
    return q.tab_alias[t] if q.tab_alias[t].byte_length() > 0 else q.tab_names[t]


def _kind_name(k: Int) -> String:
    if k == 1:
        return "left"
    if k == 2:
        return "semi"
    if k == 3:
        return "anti"
    if k == 4:
        return "anti (null-aware)"
    return "inner"


def _join_side[P: Planner](mut catalog: Catalog, q: BoundQuery, t: Int, needed: List[Int], planner: P, opts: PlannerOptions, trust: Bool) raises -> JoinSide:
    """Side `t` of the join: the source table itself when nothing filters it and it is a plain table, else the result of a row query."""
    ref src = catalog.sources[q.sources[t]]
    var start = q.tab_start[t]
    var conj = q.conj_l.copy() if t == 0 else q.conj_r.copy()
    var keycols = q.jk_l.copy() if t == 0 else q.jk_r.copy()
    if len(conj) == 0 and src.kind == SRC_TABLE and len(src.overlay.mask) == 0 and len(src.overlay.delta) == 0:
        var cols = List[Int]()
        for g in needed:
            cols.append(g - start)
        var kc = List[Int]()
        for g in keycols:
            kc.append(g - start)
        return JoinSide(src.table, cols^, kc^, True)
    var sq = side_query(catalog, q, t, needed)
    var r = _run_sql_once(catalog, sq, planner, opts, trust, String())
    var cols = List[Int]()
    for i in range(len(needed)):
        cols.append(i)
    var kc = List[Int]()
    for g in keycols:
        for i in range(len(needed)):
            if needed[i] == g:
                kc.append(i)
    return JoinSide(r.take_table().share(), cols^, kc^, False)


def _run_join_query[P: Planner](mut catalog: Catalog, q: BoundQuery, lp: LogicalPlan, planner: P, opts: PlannerOptions, trust: Bool, lie: String,
                                t0: Int) raises -> SqlResult:
    """A query over two tables: each side is reduced to the rows and columns the query needs (a row query over its source, with the filter
    that reads only it), the smaller side is the build side of the hash join (`kernels/hashjoin_cpu.mojo`), the matching pairs are gathered
    into one table, and the rest of the query (the filter that reads both sides, aggregates, ORDER BY, LIMIT) runs over that table."""
    var res = SqlResult()
    if q.explain != EXPLAIN_NONE:
        res.notes.append("join: " + _kind_name(q.join_kind) + ", on " + ", ".join(q.jk_l_names) + " = " + ", ".join(q.jk_r_names) +
                         (" (the build side is the side with fewer rows after its filter: decided when the query runs)" if q.join_kind == 0 else " (the build side is the right table)"))
        res.explain = _explain_text(lp, res.notes)
        var t = Table.empty()
        var c = t.add_column("plan", TEXT)
        for line in res.explain.split("\n"):
            if line.byte_length() > 0:
                t.append_text(c, String(line))
        res.table = t^
        res.ms = Float64(perf_counter_ns() - t0) / 1e6
        return res^
    var kind = q.join_kind
    var semi_anti = kind >= 2
    var lneed = needed_columns(q, 0)
    var rneed = needed_columns(q, 1)                                 # a semi / anti join: the key columns only (the right table has no columns after it)
    var t_sides = perf_counter_ns()
    var left = _join_side[P](catalog, q, 0, lneed, planner, opts, trust)
    var right = _join_side[P](catalog, q, 1, rneed, planner, opts, trust)
    var ms_sides = Float64(perf_counter_ns() - t_sides) / 1e6
    var lrows = left.table.rows()
    var rrows = right.table.rows()
    var left_builds = kind == 0 and lrows < rrows                    # left, semi and anti joins build on the right table
    if kind == 0 and opts.build_side == BUILD_LEFT:
        left_builds = True
    elif kind == 0 and opts.build_side == BUILD_RIGHT:
        left_builds = False
    var t_keys = perf_counter_ns()
    var keys = JoinKeys()
    if left_builds:
        normalize_keys(left.table, right.table, left.key_cols, right.key_cols, keys, opts.threads)
    else:
        normalize_keys(right.table, left.table, right.key_cols, left.key_cols, keys, opts.threads)
    var nb = lrows if left_builds else rrows
    var np = rrows if left_builds else lrows
    var fmode = 0                                                    # the front stage's mode: 0 inner, 1 left, else a presence filter
    var na_note = String()
    if kind == 1:
        fmode = 1
    elif semi_anti:
        fmode = PM_SEMI if kind == 2 else PM_ANTI
        if kind == 4:                                                # NOT IN: an empty list keeps every row; a NULL in the list keeps none; else a NULL key is dropped
            if rrows == 0:
                na_note = " (the list is empty: every row is kept, a NULL key too)"
            elif right.table.null_count(right.key_cols[0]) > 0:
                fmode = PM_NONE
                na_note = " (the list holds a NULL: no row passes)"
            else:
                fmode = PM_ANTI_NA
                na_note = " (a NULL key is dropped)"
    if semi_anti:
        right.cols = List[Int]()                                     # nothing of the right table is carried
        rneed = List[Int]()
    var t_build = perf_counter_ns()
    var m = hj_build(keys.bk_addr, keys.bo_addr, nb, opts.threads, semi_anti)
    if fmode == PM_ANTI_NA:
        unmatchable_not_null(keys, m.lo, m.hi, left.table, left.key_cols[0], lrows)
    var t_probe = perf_counter_ns()
    var ncols = len(lneed) + len(rneed)
    var budget = (opts.join_budget if opts.join_budget > 0 else JOIN_BUDGET_BYTES) // (8 * max(ncols, 1))
    res.notes.append("join: " + _kind_name(kind) + " on " + ", ".join(q.jk_l_names) + " = " + ", ".join(q.jk_r_names) + ": " + commas(lrows) + " rows of " + _side_names(q, 0) +
                     " and " + commas(rrows) + " rows of " + _side_names(q, 1) + " after their filters" +
                     (" (" + _side_names(q, 0) + " read in place)" if left.zero_copy else "") + (" (" + _side_names(q, 1) + " read in place)" if right.zero_copy else "") + na_note)
    res.notes.append("join build side: " + (_side_names(q, 0) if left_builds else _side_names(q, 1)) + " (" + commas(nb) + " rows, " +
                     ("the smaller side" if (kind == 0 and opts.build_side == BUILD_AUTO) else "forced by the options" if kind == 0 else "the right side: a " + _kind_name(kind) + " join builds on it") + "): " + m.kind())
    res.notes.append("join keys: " + keys.layout)
    res.notes.append("join time: sides " + _ms(ms_sides) + " ms, keys " + _ms(Float64(t_build - t_keys) / 1e6) + " ms, build " + _ms(Float64(t_probe - t_build) / 1e6) + " ms")
    var any_holistic = False
    for a in q.aggs:
        if a.hol:
            any_holistic = True
    var jq = joined_query(q, lneed, rneed)
    var cat2 = Catalog()
    var inner: SqlResult
    var fusable = opts.fuse and not keys.verify and not any_holistic and (m.unique or semi_anti)
    if fusable:
        # fused: no joined row is written; the pipeline probes block by block (keeps or drops the rows by the join's rule) and gathers the build side's columns into scratch
        var vprobe = List[Bool]()
        var vreal = List[Int]()
        for c in left.cols:
            vprobe.append(not left_builds)
            vreal.append(c)
        for c in right.cols:
            vprobe.append(left_builds)
            vreal.append(c)
        var vt = virtual_schema(left, right, kind == 1)
        cat2.add_table("join", vt^.share())
        cat2.sources[0].fixed_rows = np
        var pt = right.table if left_builds else left.table
        var bt = left.table if left_builds else right.table
        _prefill_ranges(cat2.sources[0], jq, pt, bt, vprobe, vreal)
        var fj = ArcPointer(FusedJoin(pt, bt, vprobe^, vreal^, m^, keys^, fmode))
        if semi_anti:
            res.notes.append("join path: fused: " + commas(np) + " probe rows are " +
                             ("kept when their key is present in the build side" if kind == 2 else "dropped when their key is present in the build side" if kind == 3 else "kept when their key is absent from the list") +
                             ", in blocks of 1,024 (" + _kind_name(kind) + " join: no build-side column is gathered, no joined row is written)")
        else:
            res.notes.append("join path: fused: " + commas(np) + " probe rows look their partner up in blocks of 1,024 and " + String(len(fj[].partner)) +
                             " build-side " + ("column is" if len(fj[].partner) == 1 else "columns are") + " gathered into per-worker scratch (no joined row is written)" +
                             (" (a row without a partner stays: its build-side columns are NULL)" if kind == 1 else ""))
        inner = _run_sql_once(cat2, jq, planner, opts, trust, lie, fj)
    else:
        var t_gather = perf_counter_ns()
        var jt: Table
        var n_out: Int
        if semi_anti:
            var pick = select_rows(m, keys, np, fmode, opts.threads)
            n_out = pick.n
            jt = gather_join(left, right, pick, False, opts.threads)
        else:
            var pairs = join_pairs(m, keys, np, budget, opts.threads, kind == 1)
            n_out = pairs.n
            jt = gather_join(left, right, pairs, left_builds, opts.threads, kind == 1)
        var ms_gather = Float64(perf_counter_ns() - t_gather) / 1e6
        var why = String()
        if not opts.fuse:
            why = " (fusing is off in the options)"
        elif any_holistic:
            why = " (a holistic aggregate needs every joined row)"
        elif keys.verify:
            why = " (the keys are hashed tuples, verified per pair)"
        elif not m.unique and not semi_anti:
            why = " (the build side's keys repeat)"
        if semi_anti:
            res.notes.append("join path: materialized: " + commas(n_out) + " of " + commas(np) + " rows pass the " + _kind_name(kind) + " join and are gathered into one table of " + String(ncols) + " columns" + why)
        else:
            res.notes.append("join path: materialized: " + commas(n_out) + (" pairs (unmatched rows included)" if kind == 1 else " matching pairs") + " gathered into one table of " + String(ncols) + " columns" + why)
        res.notes.append("join time: gather " + _ms(ms_gather) + " ms")
        cat2.add_table("join", jt^.share())
        inner = _run_sql_once(cat2, jq, planner, opts, trust, lie)
    for n in inner.notes:
        res.notes.append(n)
    res.actuals.append("join (" + _kind_name(kind) + "): " + commas(lrows) + " rows of " + _side_names(q, 0) + " and " + commas(rrows) + " rows of " + _side_names(q, 1) +
                       " after their filters: sides " + _ms(ms_sides) + " ms, keys " + _ms(Float64(t_build - t_keys) / 1e6) + " ms, build on " + commas(nb) + " rows " +
                       _ms(Float64(t_probe - t_build) / 1e6) + " ms")
    for a in inner.actuals:
        res.actuals.append(a)
    res.table = inner.take_table()
    res.explain = _explain_text(lp, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


struct _Leaf(Movable):
    """One table of a multi-way join after its own filters: the table (the source itself when nothing filters it, else a row query's result), the column of it
    that each needed global column is, and the global number of each of its columns."""
    var table: SharedTable
    var cols: List[Int]              # the table's column of each needed global column (ascending)
    var gcols: List[Int]             # ... those global columns
    var gtable: List[Int]            # the global column of each of the table's columns (-1: not one of the query's)
    var rows: Int
    var base_rows: Int

    def __init__(out self, table: SharedTable, var cols: List[Int], var gcols: List[Int], var gtable: List[Int], rows: Int, base_rows: Int):
        self.table = table
        self.cols = cols^
        self.gcols = gcols^
        self.gtable = gtable^
        self.rows = rows
        self.base_rows = base_rows


def _position(g: List[Int], col: Int) -> Int:
    for i in range(len(g)):
        if g[i] == col:
            return i
    return -1


def _label(q: BoundQuery, t: Int) -> String:
    return q.tab_alias[t] if q.tab_alias[t].byte_length() > 0 else q.tab_names[t]


def _run_multiway[P: Planner](mut catalog: Catalog, q: BoundQuery, lp: LogicalPlan, planner: P, opts: PlannerOptions, trust: Bool, lie: String,
                              t0: Int) raises -> SqlResult:
    """Three or more tables joined by equalities (inner joins). Each table runs its own filter first (a row query over its source), so the planner works
    from exact row counts; the planner (`Planner.join_order`) picks the order and the build side of each join; every join but the last materializes its pairs
    (each keeps only the columns later steps read, and the conditions across tables are checked as soon as the tables they read are joined); the last is the
    two-table join of `_run_join_query` with the rest of the query (aggregates, ORDER BY, LIMIT), fused when the build side's keys are unique."""
    var res = SqlResult()
    var n = len(q.sources)
    var threads = opts.threads
    var budget_bytes = opts.join_budget if opts.join_budget > 0 else JOIN_BUDGET_BYTES
    var needed = all_needed(q)
    var leaves = List[_Leaf]()
    var js = JoinStats()
    for t in range(n):
        var t_leaf = perf_counter_ns()
        var start = q.tab_start[t]
        var gcols = List[Int]()
        for g in needed:
            if table_of_column(q, g) == t:
                gcols.append(g)
        ref src = catalog.sources[q.sources[t]]
        var base_rows = src.table.rows()
        var table: SharedTable
        var cols = List[Int]()
        var gtable = List[Int]()
        if len(q.leaf_conj[t]) == 0 and src.kind == SRC_TABLE and len(src.overlay.mask) == 0 and len(src.overlay.delta) == 0:
            table = src.table
            for g in gcols:
                cols.append(g - start)
            for c in range(src.table.columns()):
                gtable.append(start + c)
        else:
            var sq = side_query(catalog, q, t, gcols)
            var r = _run_sql_once(catalog, sq, planner, opts, trust, String())
            table = r.take_table().share()
            for i in range(len(gcols)):
                cols.append(i)
                gtable.append(gcols[i])
        var rows = table.rows()
        js.names.append(_label(q, t))
        js.base_rows.append(base_rows)
        js.rows.append(rows)
        res.actuals.append("scan " + _label(q, t) + ": " + commas(base_rows) + " rows, " + commas(rows) + " after " + ("its filter" if len(q.leaf_conj[t]) > 0 else "no filter") +
                           " (" + String(len(gcols)) + " columns kept), " + _ms(Float64(perf_counter_ns() - t_leaf) / 1e6) + " ms")
        leaves.append(_Leaf(table, cols^, gcols^, gtable^, rows, base_rows))
    for k in range(len(q.edge_a)):
        js.edge_a.append(table_of_column(q, q.edge_a[k]))
        js.edge_b.append(table_of_column(q, q.edge_b[k]))
    var jp = planner.join_order(js, opts)
    for nt in jp.notes:
        res.notes.append(nt)
    if q.explain != EXPLAIN_NONE and q.explain != EXPLAIN_ANALYZE:
        res.explain = _explain_text(lp, res.notes)
        var tb = Table.empty()
        var c = tb.add_column("plan", TEXT)
        for line in res.explain.split("\n"):
            if line.byte_length() > 0:
                tb.append_text(c, String(line))
        res.table = tb^
        res.ms = Float64(perf_counter_ns() - t0) / 1e6
        return res^
    # the steps
    var done = List[Bool](length=len(q.resid), fill=False)
    var joined = 1 << jp.start
    var cur_table = leaves[jp.start].table
    var cur_g = leaves[jp.start].gcols.copy()
    var cur_pos = leaves[jp.start].cols.copy()
    for si in range(len(jp.steps) - 1):
        var t_step = perf_counter_ns()
        ref step = jp.steps[si]
        var lf = step.leaf
        var after = joined | (1 << lf)
        var keep = needed_after(q, after, done)
        var lkeep = List[Int]()                                      # the table columns each side carries on
        var lkeep_g = List[Int]()
        for i in range(len(cur_g)):
            if _position(keep, cur_g[i]) >= 0:
                lkeep.append(cur_pos[i])
                lkeep_g.append(cur_g[i])
        var rkeep = List[Int]()
        var rkeep_g = List[Int]()
        for i in range(len(leaves[lf].gcols)):
            if _position(keep, leaves[lf].gcols[i]) >= 0:
                rkeep.append(leaves[lf].cols[i])
                rkeep_g.append(leaves[lf].gcols[i])
        var lkey = List[Int]()
        var rkey = List[Int]()
        for k in range(len(q.edge_a)):
            var ga = q.edge_a[k]
            var gb = q.edge_b[k]
            var ta = table_of_column(q, ga)
            var tb = table_of_column(q, gb)
            if ta == lf and (joined >> tb) & 1 != 0:
                lkey.append(cur_pos[_position(cur_g, gb)])
                rkey.append(leaves[lf].cols[_position(leaves[lf].gcols, ga)])
            elif tb == lf and (joined >> ta) & 1 != 0:
                lkey.append(cur_pos[_position(cur_g, ga)])
                rkey.append(leaves[lf].cols[_position(leaves[lf].gcols, gb)])
        var left = JoinSide(cur_table, lkeep^, lkey^, False)
        var right = JoinSide(leaves[lf].table, rkeep^, rkey^, False)
        var left_builds = not step.build_new
        var keys = JoinKeys()
        if left_builds:
            normalize_keys(left.table, right.table, left.key_cols, right.key_cols, keys, threads)
        else:
            normalize_keys(right.table, left.table, right.key_cols, left.key_cols, keys, threads)
        var nb = left.table.rows() if left_builds else right.table.rows()
        var np = right.table.rows() if left_builds else left.table.rows()
        var m = hj_build(keys.bk_addr, keys.bo_addr, nb, threads)
        var build_kind = m.kind()
        var pairs = join_pairs(m, keys, np, budget_bytes // (8 * max(len(lkeep_g) + len(rkeep_g), 1)), threads)
        var jt = gather_join(left, right, pairs, left_builds, threads)
        var out_rows = jt.rows()
        res.notes.append("join step " + String(si + 1) + ": " + commas(left.table.rows()) + " rows with " + commas(right.table.rows()) + " rows of " + _label(q, lf) + " on " + String(len(left.key_cols)) +
                         (" key (" if len(left.key_cols) == 1 else " keys (") + keys.layout + "), hash table on " + commas(nb) + " rows: " + build_kind + ", " + commas(out_rows) + " rows out (estimated " + commas(step.est_rows) + ")")
        res.actuals.append("join step " + String(si + 1) + " (" + _label(q, lf) + "): " + commas(left.table.rows()) + " x " + commas(right.table.rows()) + " -> " + commas(out_rows) + " rows (estimated " +
                           commas(step.est_rows) + "), build " + commas(nb) + " rows, " + _ms(Float64(perf_counter_ns() - t_step) / 1e6) + " ms")
        cur_g = lkeep_g^
        for g in rkeep_g:
            cur_g.append(g)
        cur_table = jt^.share()
        cur_pos = List[Int]()
        for i in range(len(cur_g)):
            cur_pos.append(i)
        joined = after
        # the conditions across tables whose tables are all joined now
        var conj = List[Int]()
        var ctext = List[String]()
        for k in range(len(q.resid)):
            if not done[k] and (q.resid_mask[k] & ~joined) == 0:
                conj.append(q.resid[k])
                ctext.append(q.resid_text[k])
                done[k] = True
        if len(conj) > 0:
            var t_f = perf_counter_ns()
            var fq = filter_query(q, conj, cur_g, cur_table)
            var tmp = Catalog()
            tmp.add_table("rows", cur_table)
            var fr = _run_sql_once(tmp, fq, planner, opts, trust, String())
            var before = cur_table.rows()
            cur_table = fr.take_table().share()
            res.notes.append("conditions across tables checked after join step " + String(si + 1) + ": " + " AND ".join(ctext) + ": " + commas(before) + " rows to " + commas(cur_table.rows()))
            res.actuals.append("filter across tables (" + " AND ".join(ctext) + "): " + commas(before) + " -> " + commas(cur_table.rows()) + " rows, " + _ms(Float64(perf_counter_ns() - t_f) / 1e6) + " ms")
    # the last join, with the rest of the query
    ref last = jp.steps[len(jp.steps) - 1]
    var lf = last.leaf
    var key_l = List[Int]()
    var key_r = List[Int]()
    for k in range(len(q.edge_a)):
        var ga = q.edge_a[k]
        var gb = q.edge_b[k]
        var ta = table_of_column(q, ga)
        var tb = table_of_column(q, gb)
        if ta == lf and (joined >> tb) & 1 != 0:
            key_l.append(gb)
            key_r.append(ga)
        elif tb == lf and (joined >> ta) & 1 != 0:
            key_l.append(ga)
            key_r.append(gb)
    var rest = List[Int]()
    for k in range(len(q.resid)):
        if not done[k]:
            rest.append(q.resid[k])
    var fq = final_query(q, cur_g, leaves[lf].gtable, cur_table, leaves[lf].table, key_l, key_r, rest)
    var cat3 = Catalog()
    cat3.add_table("rows joined so far", cur_table)
    cat3.add_table(_label(q, lf), leaves[lf].table)
    var o2 = opts.copy()
    o2.build_side = BUILD_RIGHT if last.build_new else BUILD_LEFT
    var lp2 = build_plan(fq, "rows joined so far")
    var inner = _run_join_query(cat3, fq, lp2, planner, o2, trust, lie, perf_counter_ns())
    res.notes.append("last join: the rows joined so far (" + commas(cur_table.rows()) + ") with " + _label(q, lf) + " (" + commas(leaves[lf].table.rows()) + " rows), estimated " + commas(last.est_rows) + " rows out")
    for nt in inner.notes:
        res.notes.append(nt)
    for a in inner.actuals:
        res.actuals.append(a)
    res.table = inner.take_table()
    res.explain = _explain_text(lp, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


def _run_sql_once[P: Planner](mut catalog: Catalog, q0: BoundQuery, planner: P, opts: PlannerOptions, trust: Bool, lie: String,
                              fj: Optional[ArcPointer[FusedJoin]] = None) raises -> SqlResult:
    var t0 = perf_counter_ns()
    var lp = build_plan(q0, catalog.sources[q0.source].name)         # the plan; the executor lowers its query from the nodes
    if q0.joined and len(q0.sources) > 2:
        return _run_multiway(catalog, q0, lp, planner, opts, trust, lie, t0)
    var q = executable(lp, q0)
    if q.joined:
        return _run_join_query(catalog, q, lp, planner, opts, trust, lie, t0)
    prepare_ranges(catalog, q)
    ref src = catalog.sources[q.source]
    var scan = src.scan.copy()
    if scan:
        for e in q.rexprs:
            if e.op == B_COL and e.kind == TEXT.value:
                scan.value().ensure_words(Int(e.value))
    var extra = src.overlay.words.copy()
    var dicts = DictAccess(src.table, scan, extra)
    var st = make_stats(src, q, dicts, trust)
    var plan = planner.plan(q, st, opts)
    plan_sinks(lp, q, st, opts, plan)
    var res = SqlResult()
    res.notes = plan.notes.copy()
    if lie.byte_length() > 0:
        res.notes.append("statistics: the footer of `" + src.name + "` cannot be believed (" + lie + "); ran again without them: hashed groups, every column nullable")
    var inp = _inputs(src, q, trust)
    var partner = List[Bool]()
    if fj:                                                           # the fused join: the program's table columns are the probe table's, then placeholders
        for v in fj.value()[].vprobe:
            partner.append(not v)
        _fuse_inputs(inp, fj.value()[])
    var sp = SqlProgram(Program(0))
    var stage = RowStage(SqlProgram(Program(0)), List[_Item]())
    if q.rows:
        stage = prepare_rows(src, q, plan, dicts, inp, res.notes, True, partner)
    else:
        sp = _lower(src, q, plan, dicts, inp, PHASE_PLAIN if (plan.group == G_HASHED and plan.holistic) else PHASE_ALL)
        _kernel_notes(src, q, plan, dicts, inp, sp, res.notes)
    if q.explain != EXPLAIN_NONE:
        res.explain = _explain_text(lp, res.notes)
        var t = Table.empty()
        var c = t.add_column("plan", TEXT)
        for line in res.explain.split("\n"):
            if line.byte_length() > 0:
                t.append_text(c, String(line))
        res.table = t^
        res.ms = Float64(perf_counter_ns() - t0) / 1e6
        return res^
    var notes = res.notes.copy()
    if q.rows:
        res.table = execute_rows(src, q, plan, opts, dicts, notes, inp, stage^, trust, fj)
    else:
        res.table = _execute(src, q, plan, opts, dicts, notes, inp, sp^, trust, fj)
    res.notes = notes^
    res.explain = _explain_text(lp, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


def run_sql[P: Planner](mut catalog: Catalog, q: BoundQuery, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """Runs the query. A Parquet footer's min / max and null counts are a hint that lets the planner use dense groups and skip NULL
    checks; they are verified as the row groups are read (a null count that is wrong, a row outside a stated range), and when one
    is wrong the query runs again without the statistics, which the notes say."""
    if q.explain == EXPLAIN_ANALYZE:
        return _explain_analyze(catalog, q, planner, opts)
    try:
        return _run_sql_once(catalog, q, planner, opts, True, String())
    except e:
        var msg = String(e)
        if not msg.startswith(LIE):
            raise e^
        return _run_sql_once(catalog, q, planner, opts, False, String(msg[byte=LIE.byte_length():]))


def _explain_analyze[P: Planner](mut catalog: Catalog, q0: BoundQuery, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """EXPLAIN ANALYZE: the query runs, and the plan comes back with what each operator really did (rows, milliseconds) after the planner's decisions."""
    var q = copy_query(q0)
    q.explain = EXPLAIN_NONE
    var r: SqlResult
    try:
        r = _run_sql_once(catalog, q, planner, opts, True, String())
    except e:
        var msg = String(e)
        if not msg.startswith(LIE):
            raise e^
        r = _run_sql_once(catalog, q, planner, opts, False, String(msg[byte=LIE.byte_length():]))
    var text = r.explain.rstrip("\n") + "\nActual\n"
    for a in r.actuals:
        text += "  " + a + "\n"
    for nt in r.notes:
        if nt.startswith("time:") or nt.startswith("join time:") or nt.startswith("ran:"):
            text += "  " + nt + "\n"
    text += "  result: " + commas(r.table.rows()) + " rows in " + _ms(r.ms) + " ms\n"
    var out = SqlResult()
    out.notes = r.notes.copy()
    out.actuals = r.actuals.copy()
    out.explain = text
    out.ms = r.ms
    var t = Table.empty()
    var c = t.add_column("plan", TEXT)
    for line in text.split("\n"):
        if line.byte_length() > 0:
            t.append_text(c, String(line))
    out.table = t^
    return out^


def sql_with[P: Planner](mut catalog: Catalog, text: String, planner: P, options: PlannerOptions = PlannerOptions()) raises -> SqlResult:
    """`sql` with the physical planner of your choice (anything that implements `Planner`)."""
    var q = bind_sql(catalog, text)
    return run_sql(catalog, q, planner, options)


def sql(mut catalog: Catalog, text: String, options: PlannerOptions = PlannerOptions()) raises -> SqlResult:
    """Runs one SELECT (or EXPLAIN SELECT) over the catalog's sources with the rule-based planner. The result owns its table."""
    return sql_with(catalog, text, RulePlanner(), options)
