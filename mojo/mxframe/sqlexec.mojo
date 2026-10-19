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
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .frame import NULL_EDIT, text_ranks, _radix_order
from .query import (Program, QueryResult, _run_cpu, _run_mat, _merge_partial, _identity_partial, _ScanMask, _group_mask,
                    OVERFLOW_MESSAGE, overflow_message, CPU, MAGIC, A_MIN, A_MAX, A_FSUM, MORSEL_ROWS, WINDOW_BYTES, SRC_WORDS, fold_float, slot_identity)
from .kernels._fpenv import core_count, par_n
from .kernels.pipeline_cpu import IP, LP, FP, BP, VEC, E_OVERFLOW, _no_front, pipeline_keyed
from .kernels.hashagg_cpu import hash_aggregate
from .holistic import HolStates, HolSpec, HolResult, A_DISTINCT, A_PCT, A_MEDIAN, window_blocks, float_key
from .sql import sql_error, EXPLAIN_NONE
from .bind import (Catalog, SqlSource, BoundQuery, SortKey, bind_sql, SRC_TABLE, SRC_FRAME, SRC_SCAN, B_COL, F_COUNT_STAR,
                   F_COUNT, F_SUM, F_MIN, F_MAX, F_AVG, F_COUNT_DISTINCT, F_MEDIAN, pow10_checked)
from .planner import Planner, RulePlanner, PlannerOptions, PhysicalPlan, Stats, ColStat, G_NONE, G_DENSE, G_HASHED, commas
from .sqlprog import SqlProgram, DictAccess, lower_sql, collect_columns, PHASE_ALL, PHASE_PLAIN, PHASE_HOL
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
                                         threads if threads > 0 else core_count(), sp.p.repro, keyw, _no_front, 0,
                                         LP(unsafe_from_address=Int(res.unsafe_ptr())), 0)
    _ = code^
    _ = consts^
    _ = set_bytes^
    _ = set_meta^
    _ = tbl^
    _ = dtypes^
    if rc == -7 or rc == E_OVERFLOW:
        raise Error(overflow_message(sp.p.has_float))
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
    """The lowest and highest non-null value of a column (and whether there is one): in parallel over slices of the rows."""
    var n = t.rows()
    var tasks = max(1, min(core_count(), n // 65536))
    var los = List[Int64](length=tasks, fill=Int64.MAX)
    var his = List[Int64](length=tasks, fill=Int64.MIN)
    var seen = List[Int64](length=tasks, fill=0)
    var vp = Int(t.values(c).unsafe_ptr())
    var valid = t._table[].cols[c].validity_address()
    var lp = Int(los.unsafe_ptr())
    var hp = Int(his.unsafe_ptr())
    var sp = Int(seen.unsafe_ptr())

    def work(i: Int) {imm}:
        var vals = LP(unsafe_from_address=vp)
        var bits = BP(unsafe_from_address=valid)
        var lo = Int64.MAX
        var hi = Int64.MIN
        var any = Int64(0)
        var a = n * i // tasks
        var b = n * (i + 1) // tasks
        if valid == 0:
            for r in range(a, b):
                var v = vals[r]
                lo = min(lo, v)
                hi = max(hi, v)
            any = Int64(1) if b > a else Int64(0)
        else:
            for r in range(a, b):
                if (bits[r >> 3] >> UInt8(r & 7)) & 1 != 0:
                    var v = vals[r]
                    lo = min(lo, v)
                    hi = max(hi, v)
                    any = 1
        LP(unsafe_from_address=lp)[i] = lo
        LP(unsafe_from_address=hp)[i] = hi
        LP(unsafe_from_address=sp)[i] = any

    if tasks == 1:
        work(0)
    else:
        par_n(work, tasks, tasks)
    var lo = Int64.MAX
    var hi = Int64.MIN
    var any = False
    for i in range(tasks):
        if seen[i] != 0:
            lo = min(lo, los[i])
            hi = max(hi, his[i])
            any = True
    return (any, lo, hi)


def _is_range_kind(k: Int) -> Bool:
    return k == INT.value or k == DECIMAL.value or k == DATE.value


def _key_columns(q: BoundQuery, e: Int, mut out: List[Int]):
    """The integer-like source columns a row-level expression reads."""
    ref x = q.rexprs[e]
    if x.op == B_COL and _is_range_kind(x.kind):
        var c = Int(x.value)
        for k in out:
            if k == c:
                return
        out.append(c)
    if x.a >= 0:
        _key_columns(q, x.a, out)
    if x.b >= 0:
        _key_columns(q, x.b, out)


def prepare_ranges(mut catalog: Catalog, q: BoundQuery) raises:
    """Caches the value range of each integer-like column a group key reads: one pass over a table or frame's column (the frame's
    changed rows included), the row groups' statistics for a scan (none when a row group states none, then it is hashed)."""
    ref src = catalog.sources[q.source]
    var cols = List[Int]()
    for k in q.keys:
        _key_columns(q, k.expr, cols)
    for c in cols:
        var cached = False
        for x in src.range_cols:
            if x == c:
                cached = True
        if cached:
            continue
        var r = _column_range(src.table, c) if src.kind != SRC_SCAN else src.scan.value().column_range(c)
        if src.kind == SRC_SCAN and not r[0]:
            continue                                       # no statistics: the planner will say so
        var has = r[0] and r[1] <= r[2] if src.kind == SRC_SCAN else r[0]
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


def make_stats(src: SqlSource, q: BoundQuery, dicts: DictAccess, trust: Bool = True) raises -> Stats:
    """`trust` False: a file's footer statistics are not used (every column may hold NULLs, no range is known)."""
    var st = Stats(src.name, src.kind, src.rows() + (len(src.overlay.delta) if src.kind != SRC_SCAN else 0))
    st.edits = len(src.overlay.delta)
    st.version = src.version
    if src.scan:
        st.row_groups = src.scan.value().row_groups()
    for c in range(src.table.columns()):
        var cs = ColStat(src.table.kind(c).value, src.table.scale(c))
        if src.kind == SRC_SCAN:
            cs.nullable = not trust or src.scan.value().null_count(c) != 0 or _delta_nulls(src, c)
        else:
            cs.nullable = src.table.null_count(c) > 0 or _delta_nulls(src, c)
        if cs.kind == TEXT.value:
            cs.words = dicts.count(c)
        for i in range(len(src.range_cols)):
            if src.range_cols[i] == c and src.range_has[i] and (trust or src.kind != SRC_SCAN):
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
            keys[i] = float_key(vc.v[i])                  # -0.0 ties with 0.0 and every NaN with every NaN, above +inf (DuckDB's order)
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


struct _SortCols(Movable):
    """The ORDER BY keys of a set of rows as comparable numbers, for `_top_rows`."""
    var keys: List[List[Int64]]
    var nulls: List[List[UInt8]]
    var desc: List[Bool]
    var nulls_first: List[Bool]

    def __init__(out self):
        self.keys = List[List[Int64]]()
        self.nulls = List[List[UInt8]]()
        self.desc = List[Bool]()
        self.nulls_first = List[Bool]()

    def before(self, a: Int, b: Int) -> Bool:
        """Row a sorts before row b (ties: the lower row number, as the stable radix order)."""
        for k in range(len(self.keys)):
            var an = len(self.nulls[k]) > 0 and self.nulls[k][a] != 0
            var bn = len(self.nulls[k]) > 0 and self.nulls[k][b] != 0
            if an or bn:
                if an and bn:
                    continue
                return an == self.nulls_first[k]
            var x = self.keys[k][a]
            var y = self.keys[k][b]
            if x != y:
                return (x > y) if self.desc[k] else (x < y)
        return a < b


def _top_rows(cols: List[ValCol], order: List[SortKey], dicts: DictAccess, n: Int, want: Int) raises -> List[Int]:
    """The first `want` rows of 0 .. n-1 in ORDER BY order, by a bounded heap: one pass, no full sort (a big result with a
    small LIMIT)."""
    var sc = _SortCols()
    for k in range(len(order)):
        sc.keys.append(_sort_keys(cols[k], dicts))
        sc.nulls.append(cols[k].nul.copy())
        sc.desc.append(order[k].desc)
        sc.nulls_first.append(order[k].nulls_first)
    var heap = List[Int]()                  # a max-heap: the worst of the best `want` rows on top
    if want <= 0:
        return heap^
    for r in range(n):
        if len(heap) < want:
            heap.append(r)
            var i = len(heap) - 1
            while i > 0:
                var parent = (i - 1) // 2
                if sc.before(heap[parent], heap[i]):
                    var t = heap[parent]
                    heap[parent] = heap[i]
                    heap[i] = t
                    i = parent
                else:
                    break
        elif sc.before(r, heap[0]):
            heap[0] = r
            var i = 0
            while True:
                var l = 2 * i + 1
                var rr = l + 1
                var worst = i
                if l < len(heap) and sc.before(heap[worst], heap[l]):
                    worst = l
                if rr < len(heap) and sc.before(heap[worst], heap[rr]):
                    worst = rr
                if worst == i:
                    break
                var t = heap[worst]
                heap[worst] = heap[i]
                heap[i] = t
                i = worst
    # sort the survivors (insertion sort: `want` is small)
    for i in range(1, len(heap)):
        var j = i
        while j > 0 and sc.before(heap[j], heap[j - 1]):
            var t = heap[j]
            heap[j] = heap[j - 1]
            heap[j - 1] = t
            j -= 1
    return heap^


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
            dicts: DictAccess, mut notes: List[String]) raises -> Table:
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
    # HAVING
    var skeys = List[ValCol]()
    var saggs = List[ValCol]()
    if q.having >= 0:
        var hv = eval_post(q, q.having, keys, aggs, dicts, n)
        var sel = List[Int]()
        for i in range(n):
            if not hv.is_null(i) and hv.v[i] != 0:
                sel.append(i)
        for k in keys:
            skeys.append(_gather(k, sel))
        for a in aggs:
            saggs.append(_gather(a, sel))
    else:                                                       # nothing to filter: the columns move on as they are
        skeys = keys^
        saggs = aggs^
    var m = n if q.having < 0 else (skeys[0].n() if len(skeys) > 0 else saggs[0].n() if len(saggs) > 0 else 0)
    var t_d = perf_counter_ns()
    # ORDER BY, OFFSET, LIMIT
    var fin = List[Int]()
    var notes_sort = String()
    if len(q.order) > 0:
        var ocols = List[ValCol]()
        for o in q.order:
            ocols.append(eval_post(q, o.expr, skeys, saggs, dicts, m))
        var want = (q.offset + q.limit) if q.limit >= 0 else -1
        if want >= 0 and want <= 4096 and m > 8 * want:
            fin = _top_rows(ocols, q.order, dicts, m, want)
            notes_sort = "top-" + String(want) + " by a bounded heap over " + commas(m) + " groups"
        else:
            fin = _order_rows(ocols, q.order, dicts, m)
            notes_sort = "sorted " + commas(m) + " groups by " + String(len(q.order)) + (" key" if len(q.order) == 1 else " keys") + " (stable radix passes)"
    else:
        for i in range(m):
            fin.append(i)
    if notes_sort.byte_length() > 0:
        notes.append("order: " + notes_sort)
    if q.having >= 0:
        notes.append("having: " + commas(m) + " of " + commas(n) + " groups kept")
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
    var t_e = perf_counter_ns()
    var outs = List[ValCol]()
    for e in q.outputs:
        var vc = eval_post(q, e, fkeys, faggs, dicts, len(rows))
        outs.append(vc^)
    var table = _to_table(q, outs, dicts)
    notes.append("finish: groups " + _ms(Float64(t_b - t_a) / 1e6) + " ms, columns " + _ms(Float64(t_c - t_b) / 1e6) + " ms, having " +
                 _ms(Float64(t_d - t_c) / 1e6) + " ms, order " + _ms(Float64(t_e - t_d) / 1e6) + " ms, output " + _ms(Float64(perf_counter_ns() - t_e) / 1e6) + " ms")
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


def _ones_mask(rows: Int) -> List[UInt8]:
    return List[UInt8](length=(rows + 7) // 8 + 8, fill=0xFF)


struct _Chunks(Movable):
    """The chunks a source is read in, one at a time: a table (its rows, then the frame's changed rows), or a scan's row
    groups (then the changed rows). `load(i)` gives chunk i and sets `mask_addr` for it."""
    var groups: List[Int]            # a scan: the row groups with rows
    var base: Bool                   # a table or frame: the shared table has rows
    var has_delta: Bool
    var scan_mode: Bool
    var used: List[Int]              # the source columns the chunks hold (all of them for a table)
    var sm: _ScanMask
    var ones: List[UInt8]
    var gm: List[UInt8]
    var mask_addr: Int
    var masked: Bool

    def __init__(out self):
        self.groups = List[Int]()
        self.base = False
        self.has_delta = False
        self.scan_mode = False
        self.used = List[Int]()
        self.sm = _ScanMask()
        self.ones = List[UInt8]()
        self.gm = List[UInt8]()
        self.mask_addr = 0
        self.masked = False

    def n(self) -> Int:
        return (len(self.groups) if self.scan_mode else (1 if self.base else 0)) + (1 if self.has_delta else 0)


def _open_chunks(src: SqlSource, used: List[Int]) raises -> _Chunks:
    var ch = _Chunks()
    ch.scan_mode = src.kind == SRC_SCAN
    ch.used = used.copy()
    ch.masked = len(src.overlay.mask) > 0
    ch.has_delta = len(src.overlay.delta) > 0
    if ch.scan_mode:
        ref scan = src.scan.value()
        for g in range(scan.row_groups()):
            if scan.group_rows(g) > 0:
                ch.groups.append(g)
        if ch.masked:
            ch.sm.addr = Int(src.overlay.mask.unsafe_ptr())
            var at = 0
            for g in range(scan.row_groups()):
                ch.sm.starts.append(at)
                at += scan.group_rows(g)
            var mx = len(src.overlay.delta)
            for g in range(scan.row_groups()):
                mx = max(mx, scan.group_rows(g))
            ch.ones = _ones_mask(mx)
    else:
        ch.base = src.table.rows() > 0
        if ch.masked:
            ch.ones = _ones_mask(len(src.overlay.delta))
    return ch^


def _load_chunk(src: SqlSource, mut ch: _Chunks, i: Int) raises -> SharedTable:
    var nb = len(ch.groups) if ch.scan_mode else (1 if ch.base else 0)
    if i >= nb:                                                   # the frame's changed and appended rows
        ch.mask_addr = Int(ch.ones.unsafe_ptr()) if ch.masked else 0
        return delta_table(src, ch.used)
    if not ch.scan_mode:
        ch.mask_addr = Int(src.overlay.mask.unsafe_ptr()) if ch.masked else 0
        return src.table
    var g = ch.groups[i]
    ref scan = src.scan.value()
    var chunk = scan.read_group(g, ch.used)^.share()
    ch.gm = _group_mask(ch.sm, g, scan.group_rows(g)) if ch.masked else List[UInt8]()
    ch.mask_addr = Int(ch.gm.unsafe_ptr()) if len(ch.gm) > 0 else (Int(ch.ones.unsafe_ptr()) if ch.masked else 0)
    return chunk^


comptime LIE = "footer statistics wrong: "


def _check_footer(src: SqlSource, ch: _Chunks, i: Int, chunk: SharedTable) raises:
    """A trusted footer states each column chunk's null count: a decoded row group that holds another number shows the footer
    cannot be believed (a NULL read as a value would otherwise be counted as 0)."""
    if not ch.scan_mode or i >= len(ch.groups):
        return
    ref scan = src.scan.value()
    for k in range(len(ch.used)):
        var stated = scan.group_null_count(ch.groups[i], ch.used[k])
        var real = chunk.null_count(k)
        if stated >= 0 and stated != real:
            raise Error(LIE + "row group " + String(ch.groups[i]) + " states " + String(stated) + (" null" if stated == 1 else " nulls") +
                        " in column `" + scan.name(ch.used[k]) + "` and holds " + String(real))



def _all(n: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append(i)
    return out^


def _fold(sp: SqlProgram, plan: PhysicalPlan, chunk: SharedTable, mask_addr: Int, mut hs: HolStates, ops: List[Int], threads: Int,
          mut acc: Partial, mut dense_acc: QueryResult, mut ix: _KeyIndex) raises:
    """Runs the program over one chunk and folds its partial result into the accumulator."""
    if plan.group == G_HASHED:
        _merge_hashed(acc, _run_hashed(chunk, sp, mask_addr, plan.key_words(), threads), ops, ix)
        return
    var r: QueryResult
    if sp.p.mat:
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


def _run_hol_hashed(t: SharedTable, sp: SqlProgram, look: _GroupLookup, mask_addr: Int, mut hs: HolStates, threads: Int) raises:
    """Pass 2 over one chunk: the kernel runs the predicate and the key words, `_hmat_hook` finds each row's group and
    writes the holistic sources' (group, value) pairs, which the states take in windows (as query.mojo's `_run_mat`)."""
    ref p = sp.p
    var n = t.rows()
    var rows = max(MORSEL_ROWS, (WINDOW_BYTES // (12 * max(p.n_src, 1))) // MORSEL_ROWS * MORSEL_ROWS)
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
    var done = 0
    var first = True
    var G = max(look.groups, 1)
    while first or done < n:
        var end = min(n, done + rows)
        var nblk = (end - done + VEC - 1) // VEC
        var gids = List[List[Int32]]()
        var vals = List[List[Int64]]()
        var cnts = List[List[Int32]]()
        var ctl = List[Int64]()
        ctl.append(Int64(p.n_src))
        ctl.append(Int64(done))
        ctl.append(Int64(look.groups))
        ctl.append(0)
        ctl.append(0)
        ctl.append(0)
        for s in range(p.n_src):
            gids.append(List[Int32](unsafe_uninit_length=max(end - done, 1)))
            vals.append(List[Int64](unsafe_uninit_length=max(end - done, 1)))
            cnts.append(List[Int32](length=max(nblk, 1), fill=0))
            ctl.append(Int64(p.src_vreg[s]))
            ctl.append(Int64(p.src_valid[s]))
            ctl.append(Int64(Int(gids[s].unsafe_ptr())))
            ctl.append(Int64(Int(vals[s].unsafe_ptr())))
            ctl.append(Int64(Int(cnts[s].unsafe_ptr())))
            ctl.append(0)
            ctl.append(0)
            ctl.append(0)
            ctl.append(0)
            ctl.append(0)
            ctl.append(Int64(1 if p.src_float[s] else 0))
        ctl.append(Int64(look.nkw))
        ctl.append(Int64(Int(look.table.unsafe_ptr())))
        ctl.append(Int64(look.mask))
        ctl.append(Int64(Int(look.kwp.unsafe_ptr())))
        var consts = List[Float64]()
        for v in p.consts:
            consts.append(Float64(from_bits=UInt64(v)))
        consts[p.ctl_const] = Float64(from_bits=UInt64(Int(ctl.unsafe_ptr())))
        var out_vals = List[Float64](length=max(p.n_slots, 1) * G, fill=0.0)
        var out_cnt = List[Int64](length=G, fill=0)
        var rc = pipeline_keyed[type_of(_hmat_hook)](Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                                                     Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()), G,
                                                     done, end, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()),
                                                     threads if threads > 0 else core_count(), _hmat_hook, False)
        if rc == E_OVERFLOW:
            raise Error(OVERFLOW_MESSAGE)
        if rc != 0:
            raise Error("the CPU kernel failed with code " + String(rc))
        for s in range(p.n_src):
            hs.absorb(s, window_blocks(Int(gids[s].unsafe_ptr()), Int(vals[s].unsafe_ptr()), Int(cnts[s].unsafe_ptr()), nblk, VEC))
        _ = ctl^
        _ = gids^
        _ = vals^
        _ = cnts^
        _ = consts^
        _ = out_vals^
        _ = out_cnt^
        first = False
        done = end
    _ = code^
    _ = set_bytes^
    _ = set_meta^
    _ = tbl^
    _ = dtypes^


struct _Inputs(Movable):
    """How the program sees the source: which columns the chunks hold and where, which can be NULL, whether a mask applies."""
    var used: List[Int]
    var pos: List[Int]
    var chunk_cols: Int
    var nullable: List[Bool]
    var masked: Bool
    var scan_mode: Bool

    def __init__(out self):
        self.used = List[Int]()
        self.pos = List[Int]()
        self.chunk_cols = 0
        self.nullable = List[Bool]()
        self.masked = False
        self.scan_mode = False


def _inputs(src: SqlSource, q: BoundQuery, trust: Bool = True) raises -> _Inputs:
    var inp = _Inputs()
    var n_cols = src.table.columns()
    inp.used = collect_columns(q)
    inp.scan_mode = src.kind == SRC_SCAN
    # a table or frame is seen as it is; a scan's row groups hold only the columns used
    inp.pos = List[Int](length=n_cols, fill=-1)
    inp.chunk_cols = n_cols
    if inp.scan_mode:
        if len(inp.used) == 0 and n_cols > 0:
            inp.used.append(0)
        for k in range(len(inp.used)):
            inp.pos[inp.used[k]] = k
        inp.chunk_cols = len(inp.used)
        for c in inp.used:
            if src.table.kind(c) == TEXT:
                src.scan.value().ensure_words(c)
    else:
        inp.used = _all(n_cols)
        for c in range(n_cols):
            inp.pos[c] = c
    for c in range(n_cols):
        inp.nullable.append((not trust or src.scan.value().null_count(c) != 0 if inp.scan_mode else src.table.null_count(c) > 0) or _delta_nulls(src, c))
    inp.masked = len(src.overlay.mask) > 0
    return inp^


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
             mut notes: List[String], inp: _Inputs, var sp: SqlProgram, trust: Bool = True) raises -> Table:
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
    var delta_rows = len(src.overlay.delta)
    var t_run = perf_counter_ns()
    var ns_load = 0
    for i in range(ch.n()):
        var t_load = perf_counter_ns()
        var chunk = _load_chunk(src, ch, i)
        if trust:
            _check_footer(src, ch, i, chunk)
        ns_load += perf_counter_ns() - t_load
        _fold(sp, plan, chunk, ch.mask_addr, hs, ops, threads, acc, dense_acc, ix)
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
    var table = _finish(q, plan, sp, acc, hres, hol_of, dicts, notes)
    notes.append("time: " + ("decode " + _ms(ms_load) + " ms, " if scan_mode else "") + "kernel " + _ms(ms_run - ms_load) + " ms, holistic states " + _ms(Float64(t_fin - t_hol) / 1e6) + " ms, finish " + _ms(Float64(perf_counter_ns() - t_fin) / 1e6) + " ms")
    return table^


def _ms(x: Float64) -> String:
    return String(Float64(Int(x * 100.0)) / 100.0)


def _run_sql_once[P: Planner](mut catalog: Catalog, q: BoundQuery, planner: P, opts: PlannerOptions, trust: Bool, lie: String) raises -> SqlResult:
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
    var st = make_stats(src, q, dicts, trust)
    var plan = planner.plan(q, st, opts)
    var res = SqlResult()
    res.notes = plan.notes.copy()
    if lie.byte_length() > 0:
        res.notes.append("statistics: the footer of `" + src.name + "` cannot be believed (" + lie + "); ran again without them: hashed groups, every column nullable")
    var inp = _inputs(src, q, trust)
    var sp = _lower(src, q, plan, dicts, inp, PHASE_PLAIN if (plan.group == G_HASHED and plan.holistic) else PHASE_ALL)
    _kernel_notes(src, q, plan, dicts, inp, sp, res.notes)
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
    res.table = _execute(src, q, plan, opts, dicts, notes, inp, sp^, trust)
    res.notes = notes^
    res.explain = _explain_text(q, src, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


def run_sql[P: Planner](mut catalog: Catalog, q: BoundQuery, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """Runs the query. A Parquet footer's min / max and null counts are a hint that lets the planner use dense groups and skip NULL
    checks; they are verified as the row groups are read (a null count that is wrong, a row outside a stated range), and when one
    is wrong the query runs again without the statistics, which the notes say."""
    try:
        return _run_sql_once(catalog, q, planner, opts, True, String())
    except e:
        var msg = String(e)
        if not msg.startswith(LIE):
            raise e^
        return _run_sql_once(catalog, q, planner, opts, False, String(msg[byte=LIE.byte_length():]))


def sql_with[P: Planner](mut catalog: Catalog, text: String, planner: P, options: PlannerOptions = PlannerOptions()) raises -> SqlResult:
    """`sql` with the physical planner of your choice (anything that implements `Planner`)."""
    var q = bind_sql(catalog, text)
    return run_sql(catalog, q, planner, options)


def sql(mut catalog: Catalog, text: String, options: PlannerOptions = PlannerOptions()) raises -> SqlResult:
    """Runs one SELECT (or EXPLAIN SELECT) over the catalog's sources with the rule-based planner. The result owns its table."""
    return sql_with(catalog, text, RulePlanner(), options)
