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
from std.memory import Pointer
from std.ffi import external_call
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .frame import NULL_EDIT, Overlay, text_ranks, _radix_order, _text_less
from .parquet import ParquetScan
from .query import (Program, QueryResult, _run_cpu, _run_mat, _merge_partial, _identity_partial, _ScanMask, _group_mask,
                    OVERFLOW_MESSAGE, CPU, MAGIC, A_SUM, A_MIN, A_MAX)
from .kernels._fpenv import core_count
from .kernels.pipeline_cpu import IP, LP, FP, BP, VEC, E_OVERFLOW, _no_front
from .kernels.hashagg_cpu import hash_aggregate
from .holistic import HolStates, HolSpec, HolResult, A_DISTINCT, A_PCT, A_MEDIAN
from .sql import sql_error, EXPLAIN_NONE
from .bind import (Catalog, SqlSource, BoundQuery, BExpr, AggSpec, SortKey, bind_sql, SRC_TABLE, SRC_FRAME, SRC_SCAN, B_COL, F_COUNT_STAR,
                   F_COUNT, F_SUM, F_MIN, F_MAX, F_AVG, F_COUNT_DISTINCT, F_MEDIAN, F_QUANTILE, K_NULL, kind_name, pow10_checked)
from .planner import (Planner, RulePlanner, PlannerOptions, PhysicalPlan, KeyLayout, Stats, ColStat, G_NONE, G_DENSE, G_HASHED, commas)
from .sqlprog import SqlProgram, DictAccess, lower_sql, collect_columns, bkey
from .sqleval import ValCol, eval_post, float_bits, ld_div


struct SqlResult(Movable):
    var table: Table
    var explain: String              # the plan as text (always filled; EXPLAIN returns it as the table too)
    var ms: Float64
    var device: Int                  # CPU (the SQL path runs on the CPU in F-150)
    var fallback_reason: String
    var notes: List[String]          # what the planner decided and what ran, one line each: tests assert them

    def __init__(out self):
        self.table = Table.empty()
        self.explain = String()
        self.ms = 0.0
        self.device = CPU
        self.fallback_reason = String()
        self.notes = List[String]()


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


def _merge_hashed(mut acc: Partial, other: Partial, ops: List[Int]) raises:
    """Folds `other`'s groups into `acc` by key."""
    var og = other.groups()
    if og == 0:
        return
    if acc.groups() == 0:
        acc.slots = other.slots.copy()
        acc.counts = other.counts.copy()
        acc.kw = other.kw.copy()
        return
    var cap = 16
    while cap < 2 * (acc.groups() + og):
        cap *= 2
    var table = List[Int32](length=cap, fill=-1)
    var mask = UInt64(cap - 1)
    for g in range(acc.groups()):
        var pos = Int(_hash_group(acc.kw, g) & mask)
        while table[pos] >= 0:
            pos = (pos + 1) & Int(mask)
        table[pos] = Int32(g)
    for o in range(og):
        var pos = Int(_hash_group(other.kw, o) & mask)
        var found = -1
        while table[pos] >= 0:
            var g = Int(table[pos])
            var same = True
            for k in range(len(acc.kw)):
                if acc.kw[k][g] != other.kw[k][o]:
                    same = False
                    break
            if same:
                found = g
                break
            pos = (pos + 1) & Int(mask)
        if found < 0:
            found = acc.groups()
            for s in range(len(acc.slots)):
                acc.slots[s].append(other.slots[s][o])
            acc.counts.append(other.counts[o])
            for k in range(len(acc.kw)):
                acc.kw[k].append(other.kw[k][o])
            table[pos] = Int32(found)
            continue
        for s in range(len(acc.slots)):
            var a = acc.slots[s][found]
            var b = other.slots[s][o]
            if ops[s] == A_MIN:
                acc.slots[s][found] = min(a, b)
            elif ops[s] == A_MAX:
                acc.slots[s][found] = max(a, b)
            else:
                acc.slots[s][found] = _checked_add(a, b)
        acc.counts[found] += other.counts[o]


def _table_addresses(t: SharedTable, p: Program, mask_addr: Int) -> List[Int64]:
    var tbl: List[Int64] = [0, Int64(t.rows())]
    for c in p.cols:
        if c >= 0:
            tbl.append(Int64(Int(t.values(c).unsafe_ptr())))
        elif c == -1_000_000 or c == -5_000_000:
            tbl.append(0)
        elif c == -4_000_000:
            tbl.append(Int64(mask_addr))
        else:
            tbl.append(Int64(t._table[].cols[-1 - c].validity_address()))
    return tbl^


def _run_hashed(t: SharedTable, sp: SqlProgram, mask_addr: Int, nkw: Int, threads: Int) raises -> Partial:
    """The kernel with hashed groups (hashagg_cpu.hash_aggregate): each worker numbers the key tuples of its rows in a local
    table, the partitions are merged in parallel, and the result is the groups with their key words, slots and counts."""
    ref p = sp.p
    var n = t.rows()
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
    if len(consts) == 0:
        consts.append(0.0)
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
    var res = List[Int64](length=3 + max(nkw, 1), fill=0)

    def keyw(c: Int, i0: Int, sl: IP, nsel: Int, use_sel: Bool, dst: LP, rg: FP, wsp: BP) {imm} -> Bool:
        for k in range(nkw):
            var src = LP(unsafe_from_address=Int(rg) + k * VEC * 8)
            var o = dst + k * VEC
            for j in range(nsel):
                o[j] = src[j]
        return True

    var rc = hash_aggregate[fused=False](Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                                         Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()), nkw, n,
                                         threads if threads > 0 else core_count(), False, keyw, _no_front, 0,
                                         LP(unsafe_from_address=Int(res.unsafe_ptr())), 0)
    _ = code^
    _ = consts^
    _ = set_bytes^
    _ = set_meta^
    _ = tbl^
    _ = dtypes^
    if rc == -7 or rc == E_OVERFLOW:
        raise Error(OVERFLOW_MESSAGE)
    if rc != 0:
        raise Error("the hashed group-by kernel failed with code " + String(rc))
    var out = Partial()
    var g = Int(res[0])
    var stride = max(g, 1)
    var vals = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[1]))
    var cnt = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[2]))
    for s in range(p.n_slots):
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


# ── sources: statistics, the delta table, the program's inputs ───────────────────────────────────────

def _column_range(t: SharedTable, c: Int) -> Tuple[Bool, Int64, Int64]:
    var n = t.rows()
    var vals = t.values(c)
    var has_nulls = t.null_count(c) > 0
    var lo = Int64.MAX
    var hi = Int64.MIN
    var any = False
    if not has_nulls:
        for i in range(n):
            var v = vals[i]
            lo = min(lo, v)
            hi = max(hi, v)
        any = n > 0
    else:
        for i in range(n):
            if not t.is_null(c, i):
                var v = vals[i]
                lo = min(lo, v)
                hi = max(hi, v)
                any = True
    return (any, lo, hi)


def _is_range_kind(k: Int) -> Bool:
    return k == INT.value or k == DECIMAL.value or k == DATE.value


def prepare_ranges(mut catalog: Catalog, q: BoundQuery) raises:
    """Caches the value range of each integer-like column the query groups by (a table or frame source; a scan has none)."""
    ref src = catalog.sources[q.source]
    if src.kind == SRC_SCAN:
        return
    for k in q.keys:
        var e = q.rexprs[k.expr].copy()
        if e.op != B_COL or not _is_range_kind(e.kind):
            continue
        var c = Int(e.value)
        var cached = False
        for x in src.range_cols:
            if x == c:
                cached = True
        if cached:
            continue
        var r = _column_range(src.table, c)
        var has = r[0]
        var lo = r[1]
        var hi = r[2]
        for row in src.overlay.delta:                      # a frame's changed and appended rows
            if row[c] != NULL_EDIT:
                lo = min(lo, row[c])
                hi = max(hi, row[c])
                has = True
        src.range_cols.append(c)
        src.range_lo.append(lo)
        src.range_hi.append(hi)
        src.range_has.append(has)


def _alias_lists(src: SqlSource) -> List[List[Int]]:
    var out = List[List[Int]]()
    for _ in range(src.table.columns()):
        out.append(List[Int]())
    for item in src.overlay.aliases.items():
        var c = Int(item.key >> 32)
        if c < len(out):
            out[c].append(Int(item.key & 0xFFFFFFFF))
            out[c].append(Int(item.value))
    return out^


def _delta_nulls(src: SqlSource, c: Int) -> Bool:
    for row in src.overlay.delta:
        if row[c] == NULL_EDIT:
            return True
    return False


def make_stats(src: SqlSource, q: BoundQuery, dicts: DictAccess) raises -> Stats:
    var st = Stats(src.name, src.kind, src.rows() + (len(src.overlay.delta) if src.kind != SRC_SCAN else 0))
    st.edits = len(src.overlay.delta)
    if src.scan:
        st.row_groups = src.scan.value().row_groups()
    for c in range(src.table.columns()):
        var cs = ColStat(src.table.kind(c).value, src.table.scale(c))
        if src.kind == SRC_SCAN:
            cs.nullable = True
        else:
            cs.nullable = src.table.null_count(c) > 0 or _delta_nulls(src, c)
        if cs.kind == TEXT.value:
            cs.words = dicts.count(c)
        for i in range(len(src.range_cols)):
            if src.range_cols[i] == c and src.range_has[i]:
                cs.has_range = True
                cs.lo = src.range_lo[i]
                cs.hi = src.range_hi[i]
        st.cols.append(cs^)
    return st^


def delta_table(src: SqlSource, cols: List[Int]) raises -> SharedTable:
    """The frame's changed and appended rows as a small table of `cols` (merged TEXT codes applied)."""
    var t = Table.empty()
    for c in cols:
        _ = t.add_column(src.table.name(c), src.table.kind(c), scale=src.table.scale(c))
    for row in src.overlay.delta:
        for k in range(len(cols)):
            var raw = row[cols[k]]
            if raw == NULL_EDIT:
                t.cols[k].push(0, False)
            else:
                if src.table.kind(cols[k]) == TEXT:
                    var hit = src.overlay.aliases.get((cols[k] << 32) | Int(raw))
                    if hit:
                        raw = hit.value()
                t.cols[k].push(raw, True)
    return t^.share()


# ── text for EXPLAIN ─────────────────────────────────────────────────────────────────────────────────

def _logical_text(q: BoundQuery, src: SqlSource) -> List[String]:
    var lines = List[String]()
    if q.limit >= 0 or q.offset > 0:
        lines.append("Limit " + (String(q.limit) if q.limit >= 0 else "all") + (" offset " + String(q.offset) if q.offset > 0 else ""))
    if len(q.order) > 0:
        var s = String("Sort ")
        for k in range(len(q.order_text)):
            s += (", " if k > 0 else "") + q.order_text[k]
        lines.append(s)
    var proj = String("Project ")
    for k in range(len(q.out_text)):
        proj += (", " if k > 0 else "") + q.out_text[k]
    lines.append(proj)
    if q.having >= 0:
        lines.append("Filter (HAVING) " + q.having_text)
    var agg = String("Aggregate")
    if len(q.keys) > 0:
        agg += " keys=["
        for k in range(len(q.keys)):
            agg += (", " if k > 0 else "") + q.keys[k].name
        agg += "]"
    agg += " aggs=["
    for k in range(len(q.aggs)):
        agg += (", " if k > 0 else "") + q.aggs[k].name
    lines.append(agg + "]")
    if q.where >= 0:
        lines.append("Filter (WHERE) " + q.where_text)
    lines.append("Scan " + src.name)
    return lines^


def _explain_text(q: BoundQuery, src: SqlSource, notes: List[String]) -> String:
    var out = String("Logical plan\n")
    var lines = _logical_text(q, src)
    for i in range(len(lines)):
        out += "  " * (1 + i) + lines[i] + "\n"
    out += "Physical plan\n"
    for n in notes:
        out += "  " + n + "\n"
    return out^


# ── finishing: groups -> HAVING -> ORDER BY -> LIMIT -> the result table ─────────────────────────────

def _gather(vc: ValCol, idx: List[Int]) -> ValCol:
    var out = ValCol(vc.kind, vc.scale, len(idx))
    out.src = vc.src
    for i in range(len(idx)):
        out.v[i] = vc.v[idx[i]]
    if len(vc.nul) > 0:
        out.nul = List[UInt8](length=len(idx), fill=0)
        for i in range(len(idx)):
            out.nul[i] = vc.nul[idx[i]]
    return out^


def _sort_keys(vc: ValCol, dicts: DictAccess) raises -> List[Int64]:
    """Int64 keys whose signed order is the column's order: ints as they are, floats by their bits, TEXT by the place of
    the word among the words present (alphabetical, ignoring ASCII case, bytes as the tie-break)."""
    var n = vc.n()
    var keys = List[Int64](length=n, fill=0)
    if vc.kind == FLOAT.value:
        for i in range(n):
            var b = vc.v[i]
            keys[i] = b ^ ((b >> 63) & Int64(0x7FFFFFFFFFFFFFFF))
        return keys^
    if vc.kind == TEXT.value:
        var seen = List[Int](length=0, fill=0)
        var present = List[Int]()
        var index = List[Int]()
        var mx = 0
        for i in range(n):
            if not vc.is_null(i):
                mx = max(mx, Int(vc.v[i]))
        var at = List[Int32](length=mx + 1, fill=-1)
        for i in range(n):
            if vc.is_null(i):
                continue
            var c = Int(vc.v[i])
            if at[c] < 0:
                at[c] = Int32(len(present))
                present.append(c)
        var words = List[String]()
        for c in present:
            words.append(dicts.word(vc.src, c))
        var ranks = text_ranks(words)
        for i in range(n):
            if not vc.is_null(i):
                keys[i] = ranks[Int(at[Int(vc.v[i])])]
        return keys^
    for i in range(n):
        keys[i] = vc.v[i]
    return keys^


def _order_rows(cols: List[ValCol], order: List[SortKey], dicts: DictAccess, n: Int) raises -> List[Int]:
    """The rows 0 .. n-1 sorted by the ORDER BY keys: stable LSD passes from the last key to the first (and, within a
    key with NULLs, the values first and then the NULL flag, so NULLS LAST / FIRST holds whatever the direction)."""
    var perm = List[Int](length=n, fill=0)
    for i in range(n):
        perm[i] = i
    var k = len(order) - 1
    while k >= 0:
        var vc = cols[k].copy()
        var keys = _sort_keys(vc, dicts)
        var aligned = List[Int64](length=n, fill=0)
        for i in range(n):
            aligned[i] = keys[perm[i]]
        perm = _radix_order(aligned, perm, order[k].desc)
        if vc.has_nulls():
            var flags = List[Int64](length=n, fill=0)
            for i in range(n):
                var isn = vc.is_null(perm[i])
                flags[i] = Int64((0 if isn else 1) if order[k].nulls_first else (1 if isn else 0))
            perm = _radix_order(flags, perm, False)
        k -= 1
    return perm^


def _fill_column(mut col: Column, vals: List[Int64], nul: List[UInt8]):
    col.values = vals.copy()
    if len(nul) > 0:
        var n = len(vals)
        col.validity = List[UInt8](length=(n + 7) // 8, fill=0xFF)
        var nulls = 0
        for i in range(n):
            if nul[i] != 0:
                col.validity[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
                nulls += 1
        col.nulls = nulls
        if nulls == 0:
            col.validity = List[UInt8]()


def _unique_name(t: Table, name: String) -> String:
    if t.index(name) < 0:
        return name
    var i = 1
    while t.index(name + "_" + String(i)) >= 0:
        i += 1
    return name + "_" + String(i)


def _to_table(q: BoundQuery, outs: List[ValCol], dicts: DictAccess) raises -> Table:
    var t = Table.empty()
    for i in range(len(outs)):
        ref vc = outs[i]
        var kind = ColumnKind(vc.kind)
        var c = t.add_column(_unique_name(t, q.out_names[i]), kind, scale=vc.scale)
        if vc.kind == TEXT.value:
            var n = vc.n()
            var mx = 0
            for r in range(n):
                if not vc.is_null(r):
                    mx = max(mx, Int(vc.v[r]))
            var remap = List[Int32](length=mx + 1, fill=-1)
            var codes = List[Int64](length=n, fill=0)
            for r in range(n):
                if vc.is_null(r):
                    continue
                var old = Int(vc.v[r])
                if remap[old] < 0:
                    remap[old] = Int32(t.add_word(c, dicts.word(vc.src, old)))
                codes[r] = Int64(remap[old])
            _fill_column(t.cols[c], codes, vc.nul)
        else:
            _fill_column(t.cols[c], vc.v, vc.nul)
    return t^


def _finish(q: BoundQuery, plan: PhysicalPlan, sp: SqlProgram, acc: Partial, hres: HolResult, hol_of: List[Int],
            dicts: DictAccess) raises -> Table:
    # the groups that exist: only non-empty ones, except the one row of a query with no GROUP BY
    var gids = List[Int]()
    if plan.group == G_NONE:
        gids.append(0)
    elif plan.group == G_DENSE:
        for g in range(len(acc.counts)):
            if acc.counts[g] > 0:
                gids.append(g)
    else:
        for g in range(acc.groups()):
            gids.append(g)
    var n = len(gids)
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
            if f == F_COUNT_STAR:
                vc.v[i] = Int64(rows)
            elif f == F_COUNT:
                vc.v[i] = Int64(nonnull)
            elif f == F_SUM or f == F_MIN or f == F_MAX:
                if nonnull == 0:
                    vc.set_null(i)
                else:
                    vc.v[i] = Int64(acc.slots[vs][g])
            elif f == F_AVG:
                if nonnull == 0:
                    vc.set_null(i)
                else:
                    var den = UInt128(nonnull) * UInt128(Int(pow10_checked(q.rexprs[sp_a.arg].scale)))
                    vc.v[i] = float_bits(ld_div(Int64(acc.slots[vs][g]), den))
            else:                                                 # count(DISTINCT), median, quantile_disc: the holistic states
                var present = hres.counts[h][g] > 0
                if f == F_COUNT_DISTINCT:
                    vc.v[i] = Int64(hres.vals[h][g])
                elif not present:
                    vc.set_null(i)
                elif f == F_MEDIAN and sp_a.kind == FLOAT.value:
                    vc.v[i] = float_bits(Float64(hres.vals[h][g]) / 10.0)
                else:
                    vc.v[i] = Int64(hres.vals[h][g])
        if sp_a.kind == TEXT.value:
            vc.src = q.rexprs[sp_a.arg].src
        aggs.append(vc^)
    # HAVING
    var sel = List[Int]()
    if q.having >= 0:
        var hv = eval_post(q, q.having, keys, aggs, dicts, n)
        for i in range(n):
            if not hv.is_null(i) and hv.v[i] != 0:
                sel.append(i)
    else:
        for i in range(n):
            sel.append(i)
    var skeys = List[ValCol]()
    for k in keys:
        skeys.append(_gather(k, sel))
    var saggs = List[ValCol]()
    for a in aggs:
        saggs.append(_gather(a, sel))
    var m = len(sel)
    # ORDER BY, OFFSET, LIMIT
    var fin = List[Int]()
    if len(q.order) > 0:
        var ocols = List[ValCol]()
        for o in q.order:
            ocols.append(eval_post(q, o.expr, skeys, saggs, dicts, m))
        fin = _order_rows(ocols, q.order, dicts, m)
    else:
        for i in range(m):
            fin.append(i)
    var lo = min(q.offset, len(fin))
    var hi = len(fin) if q.limit < 0 else min(len(fin), lo + q.limit)
    var rows = List[Int]()
    for i in range(lo, hi):
        rows.append(fin[i])
    var fkeys = List[ValCol]()
    for k in skeys:
        fkeys.append(_gather(k, rows))
    var faggs = List[ValCol]()
    for a in saggs:
        faggs.append(_gather(a, rows))
    var outs = List[ValCol]()
    for e in q.outputs:
        var vc = eval_post(q, e, fkeys, faggs, dicts, len(rows))
        outs.append(vc^)
    return _to_table(q, outs, dicts)


# ── running ──────────────────────────────────────────────────────────────────────────────────────────

def _holistic_states(q: BoundQuery, plan: PhysicalPlan, sp: SqlProgram, dicts: DictAccess, rows: Int) raises -> HolStates:
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
        if sa.func == F_QUANTILE and q.rexprs[sa.arg].kind == TEXT.value and len(ranks[sp.agg_src[a]]) == 0:
            var c = Int(q.rexprs[sa.arg].value)
            var words = List[String]()
            for k in range(dicts.count(c)):
                words.append(dicts.word(c, k))
            ranks[sp.agg_src[a]] = text_ranks(words)
    return HolStates(plan.domain if plan.group == G_DENSE else 1, specs^, ranks^, rows)


def _ones_mask(rows: Int) -> List[UInt8]:
    return List[UInt8](length=(rows + 7) // 8 + 8, fill=0xFF)


def _execute(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, opts: PlannerOptions, dicts: DictAccess,
             mut notes: List[String]) raises -> Table:
    var n_cols = src.table.columns()
    var used = collect_columns(q)
    var scan_mode = src.kind == SRC_SCAN
    # how the program sees the chunks: a table or frame as it is, a scan's row groups holding only the columns used
    var pos = List[Int](length=n_cols, fill=-1)
    var chunk_cols = n_cols
    if scan_mode:
        if len(used) == 0 and n_cols > 0:
            used.append(0)
        for k in range(len(used)):
            pos[used[k]] = k
        chunk_cols = len(used)
    else:
        for c in range(n_cols):
            pos[c] = c
    var nullable = List[Bool]()
    for c in range(n_cols):
        nullable.append(scan_mode or src.table.null_count(c) > 0 or _delta_nulls(src, c))
    var masked = len(src.overlay.mask) > 0
    var t_lower = perf_counter_ns()
    var sp = lower_sql(q, plan, nullable^, pos.copy(), chunk_cols, DictAccess(src.table, src.scan, dicts.extra), _alias_lists(src), masked)
    notes.append("kernel: " + String(sp.instr) + " instructions a row" + (", " + String(len(sp.set_meta) // 3) + " sets" if len(sp.set_meta) > 0 else "") +
                 " (" + String(sp.p.n_slots) + (" slot" if sp.p.n_slots == 1 else " slots") + ")")
    var ops = sp.slot_ops.copy()
    var threads = opts.threads
    var hs = _holistic_states(q, plan, sp, dicts, -1 if src.kind != SRC_TABLE else src.rows()) if sp.p.mat else HolStates(1, List[HolSpec](), List[List[Int64]]())
    var domain = plan.domain if plan.group == G_DENSE else 1
    var acc = Partial()
    var dense_acc = _identity_partial(sp.p.n_slots, domain, ops)
    var nchunks = 0
    var delta_rows = len(src.overlay.delta)

    if not scan_mode:
        var mask_addr = Int(src.overlay.mask.unsafe_ptr()) if masked else 0
        if src.table.rows() > 0:
            _fold(sp, plan, src.table, mask_addr, hs, ops, threads, acc, dense_acc)
            nchunks += 1
        if delta_rows > 0:
            var dcols = _all(n_cols)
            var dt = delta_table(src, dcols)
            var ones = _ones_mask(dt.rows())
            _fold(sp, plan, dt, Int(ones.unsafe_ptr()) if masked else 0, hs, ops, threads, acc, dense_acc)
            _ = ones^
            nchunks += 1
        if src.kind == SRC_FRAME:
            notes.append("ran: frame " + src.name + ": " + commas(src.table.rows()) + " shared rows (the changed ones masked out) + " +
                         commas(delta_rows) + " changed or appended rows")
        else:
            notes.append("ran: table " + src.name + ": 1 chunk of " + commas(src.table.rows()) + " rows")
    else:
        ref scan = src.scan.value()
        for c in used:
            if scan.kind(c) == TEXT:
                scan.ensure_words(c)
        var sm = _ScanMask()
        if masked:
            sm.addr = Int(src.overlay.mask.unsafe_ptr())
            var at = 0
            for g in range(scan.row_groups()):
                sm.starts.append(at)
                at += scan.group_rows(g)
        var big_ones = List[UInt8]()
        if masked:
            var mx = 0
            for g in range(scan.row_groups()):
                mx = max(mx, scan.group_rows(g))
            big_ones = _ones_mask(max(mx, delta_rows))
        for g in range(scan.row_groups()):
            if scan.group_rows(g) == 0:
                continue
            var chunk = scan.read_group(g, used)^.share()
            var gm = _group_mask(sm, g, scan.group_rows(g)) if masked else List[UInt8]()
            var mask_addr = Int(gm.unsafe_ptr()) if len(gm) > 0 else (Int(big_ones.unsafe_ptr()) if masked else 0)
            _fold(sp, plan, chunk, mask_addr, hs, ops, threads, acc, dense_acc)
            _ = gm^
            nchunks += 1
        if delta_rows > 0:
            var dt = delta_table(src, used)
            _fold(sp, plan, dt, Int(big_ones.unsafe_ptr()) if masked else 0, hs, ops, threads, acc, dense_acc)
            nchunks += 1
        notes.append("ran: scan " + src.name + ": " + String(nchunks - (1 if delta_rows > 0 else 0)) + " of " + String(scan.row_groups()) +
                     " row groups read (" + String(len(used)) + (" column" if len(used) == 1 else " columns") + "), one chunk each" +
                     ((", + " + commas(delta_rows) + " changed or appended rows") if delta_rows > 0 else ""))
        _ = big_ones^
    if plan.group != G_HASHED:
        acc.slots = dense_acc.slots.copy()
        acc.counts = dense_acc.counts.copy()
    var hres = HolResult()
    var hol_of = List[Int](length=len(q.aggs), fill=-1)
    if sp.p.mat:
        hres = hs.finish()
        for h in range(len(hs.specs)):
            hol_of[hs.specs[h].agg] = h
    return _finish(q, plan, sp, acc, hres, hol_of, dicts)


def _all(n: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append(i)
    return out^


def _fold(sp: SqlProgram, plan: PhysicalPlan, chunk: SharedTable, mask_addr: Int, mut hs: HolStates, ops: List[Int], threads: Int,
          mut acc: Partial, mut dense_acc: QueryResult) raises:
    """Runs the program over one chunk and folds its partial result into the accumulator."""
    if plan.group == G_HASHED:
        _merge_hashed(acc, _run_hashed(chunk, sp, mask_addr, plan.key_words(), threads), ops)
        return
    var r: QueryResult
    if sp.p.mat:
        r = _run_mat(chunk, sp.p, mask_addr, hs, 0)
    else:
        r = _run_cpu(chunk, sp.p, mask_addr, 0, -1, 0, threads)
    _merge_partial(dense_acc, r, ops)


def run_sql(mut catalog: Catalog, q: BoundQuery, opts: PlannerOptions) raises -> SqlResult:
    var t0 = perf_counter_ns()
    prepare_ranges(catalog, q)
    ref src = catalog.sources[q.source]
    var scan = src.scan.copy()
    if scan:
        for e in q.rexprs:
            if e.op == B_COL and e.kind == TEXT.value:
                scan.value().ensure_words(Int(e.value))
    var extra = src.overlay.words.copy()
    var dicts = DictAccess(src.table, scan, extra)
    var st = make_stats(src, q, dicts)
    var planner = RulePlanner()
    var plan = planner.plan(q, st, opts)
    var res = SqlResult()
    res.notes = plan.notes.copy()
    if q.explain != EXPLAIN_NONE:
        res.explain = _explain_text(q, src, res.notes)
        var t = Table.empty()
        var c = t.add_column("plan", TEXT)
        for line in res.explain.split("\n"):
            if line.byte_length() > 0:
                t.append_text(c, String(line))
        res.table = t^
        res.ms = Float64(perf_counter_ns() - t0) / 1e6
        return res^
    var notes = res.notes.copy()
    res.table = _execute(src, q, plan, opts, dicts, notes)
    res.notes = notes^
    res.explain = _explain_text(q, src, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


def sql(mut catalog: Catalog, text: String, options: PlannerOptions = PlannerOptions()) raises -> SqlResult:
    """Runs one SELECT (or EXPLAIN SELECT) over the catalog's sources. The result owns its table."""
    var q = bind_sql(catalog, text)
    return run_sql(catalog, q, options)
