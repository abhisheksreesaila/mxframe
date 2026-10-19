"""The inner equi-join of two tables (F-153): what the SQL layer does around the hash join kernel (`kernels/hashjoin_cpu.mojo`).

The join runs on two in-memory sides. Each side is first reduced to what the query needs of it: the executor runs it as a row query over
its source (so a Frame's edits, a Parquet scan's row groups and the filter that reads only that side are all applied, and a TEXT column
gets a compact dictionary), and the rows and columns that survive are the side's table. This module then

- decides the build side (the side with fewer rows after its filter),
- turns each key column into one int64 per row (`normalize`): integers as they are, DECIMAL aligned to the larger scale (INT is scale 0;
  a value that overflows when aligned cannot equal anything on the other side, so it never matches), DATE and BOOL as stored, TEXT
  through a code translation from the probe side's dictionary to the build side's (byte-exact: join equality is byte equality, only
  ordering ignores case; a word the build side lacks never matches), FLOAT by `float_key` (-0.0 equals 0.0 and every NaN equals every
  NaN, DuckDB's rule), NULL never matches; several keys are packed into one int64 when their ranges multiply to fewer than 2^62,
  otherwise hashed to one and the candidate pairs are verified against the key columns,
- gathers the joined rows' columns (`gather_join`) into one Table, which the rest of the query runs over as an ordinary single table.

Nothing here knows an application: the sides are `SharedTable`s and the output is a Table.
"""
from .table import Table, SharedTable, Column, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .bind import (BoundQuery, BExpr, Catalog, KeySpec, AggSpec, SortKey, copy_query, B_COL, B_LIT, B_AND, pow10_checked)
from .holistic import float_key
from .kernels._fpenv import par_n, core_count
from .kernels.hashjoin_cpu import JoinMap, hj_build, hj_probe_pairs, hj_probe_unique, hj_release, E_BUDGET
from .sqlprog import DictAccess
from .sqleval import ValCol
from .exec import _put_column, _unique_name
from .planner import commas

comptime JOIN_BUDGET_BYTES = 3 << 30        # the joined rows a join may materialize (8 bytes a value); past it a readable error (spill is later)


# ── what each side must deliver ──────────────────────────────────────────────────────────────────────

def _mark(q: BoundQuery, e: Int, mut cols: List[Bool], mut seen: List[Bool]):
    if e < 0 or seen[e]:
        return
    seen[e] = True
    ref x = q.rexprs[e]
    if x.op == B_COL:
        cols[Int(x.value)] = True
    _mark(q, x.a, cols, seen)
    _mark(q, x.b, cols, seen)
    for it in x.items:
        _mark(q, it, cols, seen)


def _mark_rows(q: BoundQuery, e: Int, mut seen: List[Bool]):
    if e < 0 or seen[e]:
        return
    seen[e] = True
    _mark_rows(q, q.rexprs[e].a, seen)
    _mark_rows(q, q.rexprs[e].b, seen)
    for it in q.rexprs[e].items:
        _mark_rows(q, it, seen)


def live_roots(q: BoundQuery) -> List[Int]:
    """The row-level expressions that still run on the joined rows: the filter the sides leave, the group keys, the aggregates'
    arguments, the select list and ORDER BY of a row query."""
    var roots = List[Int]()
    roots.append(q.where)
    for k in q.keys:
        roots.append(k.expr)
    for a in q.aggs:
        roots.append(a.arg)
    for e in q.routs:
        roots.append(e)
    for o in q.rorder:
        roots.append(o.expr)
    return roots^


def needed_columns(q: BoundQuery, t: Int) -> List[Int]:
    """The global columns of table `t` the joined rows must carry: what the live expressions read, and the join keys."""
    var total = len(q.rexprs)
    var cols = List[Bool](length=q.tab_start[1] + q.tab_cols[1], fill=False)
    var seen = List[Bool](length=total, fill=False)
    for r in live_roots(q):
        _mark(q, r, cols, seen)
    for c in q.jk_l:
        cols[c] = True
    for c in q.jk_r:
        cols[c] = True
    var out = List[Int]()
    for c in range(q.tab_start[t], q.tab_start[t] + q.tab_cols[t]):
        if cols[c]:
            out.append(c)
    return out^


def side_query(catalog: Catalog, q: BoundQuery, t: Int, needed: List[Int]) -> BoundQuery:
    """The row query that delivers side `t`: its needed columns, filtered by the conjuncts that read only that side."""
    var sq = copy_query(q)
    var start = q.tab_start[t]
    sq.joined = False
    sq.sources = List[Int]()
    sq.sources.append(q.sources[t])
    sq.source = q.sources[t]
    sq.rows = True
    sq.distinct = False
    sq.keys = List[KeySpec]()
    sq.aggs = List[AggSpec]()
    sq.outputs = List[Int]()
    sq.order = List[SortKey]()
    sq.rorder = List[SortKey]()
    sq.having = -1
    sq.limit = -1
    sq.offset = 0
    sq.jk_l = List[Int]()
    sq.jk_r = List[Int]()
    sq.conj_l = List[Int]()
    sq.conj_r = List[Int]()
    sq.conj_x = List[Int]()
    for i in range(len(sq.rexprs)):                                 # this side's columns by their own numbers
        if sq.rexprs[i].op == B_COL:
            var g = Int(sq.rexprs[i].value)
            if g >= start and g < start + q.tab_cols[t]:
                sq.rexprs[i].value = Int64(g - start)
                if sq.rexprs[i].kind == TEXT.value:
                    sq.rexprs[i].src = g - start
    var conj = q.conj_l.copy() if t == 0 else q.conj_r.copy()
    sq.where = -1
    if len(conj) > 0:
        var acc = conj[0]
        for k in range(1, len(conj)):
            sq.rexprs.append(BExpr(B_AND, BOOL.value, 0, acc, conj[k], 0, "", 0, 0))
            acc = len(sq.rexprs) - 1
        sq.where = acc
    ref src = catalog.sources[q.sources[t]]
    sq.routs = List[Int]()
    sq.out_names = List[String]()
    sq.out_text = List[String]()
    for g in needed:
        var c = g - start
        sq.rexprs.append(BExpr(B_COL, src.table.kind(c).value, src.table.scale(c), -1, -1, Int64(c), src.table.name(c), 0, 0))
        var at = len(sq.rexprs) - 1
        if src.table.kind(c).value == TEXT.value:
            sq.rexprs[at].src = c
        sq.routs.append(at)
        sq.out_names.append(src.table.name(c))
        sq.out_text.append(src.table.name(c))
    sq.where_text = "the filter that reads only this table"
    sq.order_text = List[String]()
    # the other table's columns (and everything else this query does not read) are not this query's: nothing of them is left
    var seen = List[Bool](length=len(sq.rexprs), fill=False)
    var cols = List[Bool](length=q.tab_start[1] + q.tab_cols[1], fill=False)
    _mark_rows(sq, sq.where, seen)
    for e in sq.routs:
        _mark_rows(sq, e, seen)
    for i in range(len(sq.rexprs)):
        if not seen[i]:
            sq.rexprs[i].op = B_LIT
            sq.rexprs[i].kind = INT.value
            sq.rexprs[i].scale = 0
            sq.rexprs[i].value = 0
            sq.rexprs[i].a = -1
            sq.rexprs[i].b = -1
            sq.rexprs[i].items = List[Int]()
    return sq^


def joined_query(q: BoundQuery, lneeded: List[Int], rneeded: List[Int]) -> BoundQuery:
    """The query that runs over the joined table (a single source whose columns are the left side's needed columns, then the right's)."""
    var jq = copy_query(q)
    var total = q.tab_start[1] + q.tab_cols[1]
    var jmap = List[Int](length=total, fill=-1)
    for i in range(len(lneeded)):
        jmap[lneeded[i]] = i
    for i in range(len(rneeded)):
        jmap[rneeded[i]] = len(lneeded) + i
    var seen = List[Bool](length=len(jq.rexprs), fill=False)
    var cols = List[Bool](length=total, fill=False)
    for r in live_roots(q):
        _mark(q, r, cols, seen)
    for i in range(len(jq.rexprs)):
        if not seen[i]:                                             # read only before the join (a side's filter): nothing is left of it
            jq.rexprs[i].op = B_LIT
            jq.rexprs[i].kind = INT.value
            jq.rexprs[i].scale = 0
            jq.rexprs[i].value = 0
            jq.rexprs[i].a = -1
            jq.rexprs[i].b = -1
            jq.rexprs[i].items = List[Int]()
            continue
        if jq.rexprs[i].op == B_COL:
            var g = Int(jq.rexprs[i].value)
            jq.rexprs[i].value = Int64(jmap[g])
            if jq.rexprs[i].kind == TEXT.value:
                jq.rexprs[i].src = jmap[g]
    jq.joined = False
    jq.sources = List[Int]()
    jq.sources.append(0)
    jq.source = 0
    jq.jk_l = List[Int]()
    jq.jk_r = List[Int]()
    jq.conj_l = List[Int]()
    jq.conj_r = List[Int]()
    jq.conj_x = List[Int]()
    return jq^


# ── keys ─────────────────────────────────────────────────────────────────────────────────────────────

struct JoinKeys(Movable):
    var bkeys: List[Int64]           # the build side's key of each row, the probe side's ...
    var bok: List[UInt8]             # ... and whether the row can match at all
    var pkeys: List[Int64]
    var pok: List[UInt8]
    var verify: Bool                 # the keys are hashes of tuples: candidate pairs are checked against the columns
    var bcomp: List[List[Int64]]     # verify: each key column's normalized values
    var pcomp: List[List[Int64]]
    var layout: String               # for the notes: how several keys became one
    var bk_addr: Int                 # the arrays the kernel reads: the lists above, or a column's own values when no key had to change
    var bo_addr: Int                 # (a `ok` address of 0: every row may match)
    var pk_addr: Int
    var po_addr: Int

    def __init__(out self):
        self.bk_addr = 0
        self.bo_addr = 0
        self.pk_addr = 0
        self.po_addr = 0
        self.bkeys = List[Int64]()
        self.bok = List[UInt8]()
        self.pkeys = List[Int64]()
        self.pok = List[UInt8]()
        self.verify = False
        self.bcomp = List[List[Int64]]()
        self.pcomp = List[List[Int64]]()
        self.layout = String()


def _mix(h: UInt64) -> UInt64:
    var z = h
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _valid_bits(t: SharedTable, c: Int) -> Int:
    return t._table[].cols[c].validity_address()


def _column_keys(t: SharedTable, c: Int, kind: Int, mul: Int64, translate: List[Int32], translating: Bool, mut vals: List[Int64], mut ok: List[UInt8]):
    """One key column as int64 values and `ok` flags (see the module doc)."""
    var n = t.rows()
    vals = List[Int64](length=n, fill=0)
    ok = List[UInt8](length=n, fill=1)
    var src = t.values(c).unsafe_ptr()
    var valid = _valid_bits(t, c)
    var vp = vals.unsafe_ptr()
    var op = ok.unsafe_ptr()
    for i in range(n):
        if valid != 0 and (Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)[i >> 3] >> UInt8(i & 7)) & 1 == 0:
            op[i] = 0
            continue
        var v = src[i]
        if kind == FLOAT.value:
            vp[i] = float_key(v)
        elif kind == TEXT.value:
            if translating:
                var tr = Int(translate[Int(v)])
                if tr < 0:
                    op[i] = 0
                    continue
                vp[i] = Int64(tr)
            else:
                vp[i] = v
        elif mul != 1:
            var r = v * mul
            if (v != 0 and r // v != mul) or (v == Int64.MIN):
                op[i] = 0
                continue
            vp[i] = r
        else:
            vp[i] = v


def _ok_bytes(t: SharedTable, c: Int, mut ok: List[UInt8]) -> Int:
    """The `ok` bytes of a column's validity (0 when it has no NULL: every row may match)."""
    if t.null_count(c) == 0:
        return 0
    var n = t.rows()
    ok = List[UInt8](unsafe_uninit_length=n)
    var valid = _valid_bits(t, c)
    var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
    for i in range(n):
        ok[i] = (bits[i >> 3] >> UInt8(i & 7)) & 1
    return Int(ok.unsafe_ptr())


def normalize_keys(bt: SharedTable, pt: SharedTable, bcols: List[Int], pcols: List[Int], mut keys: JoinKeys) raises:
    """The keys of the build side `bt` and the probe side `pt`, columns `bcols[j]` and `pcols[j]` being compared pairwise."""
    var nk = len(bcols)
    if nk == 1:
        var kind = bt.kind(bcols[0]).value
        var bs = bt.scale(bcols[0])
        var ps = pt.scale(pcols[0])
        var numeric = kind == INT.value or kind == DECIMAL.value
        if (numeric and bs == ps) or kind == DATE.value or kind == BOOL.value or (kind == TEXT.value and False):
            # the key is the column itself: no pass over the rows, only the NULL flags (when there are any)
            keys.bk_addr = Int(bt.values(bcols[0]).unsafe_ptr())
            keys.pk_addr = Int(pt.values(pcols[0]).unsafe_ptr())
            keys.bo_addr = _ok_bytes(bt, bcols[0], keys.bok)
            keys.po_addr = _ok_bytes(pt, pcols[0], keys.pok)
            keys.layout = "one key (the column itself)"
            return
    var bvals = List[List[Int64]]()
    var boks = List[List[UInt8]]()
    var pvals = List[List[Int64]]()
    var poks = List[List[UInt8]]()
    for j in range(nk):
        var kind = bt.kind(bcols[j]).value
        var bs = bt.scale(bcols[j])
        var ps = pt.scale(pcols[j])
        var bmul = Int64(1)
        var pmul = Int64(1)
        var numeric = kind == INT.value or kind == DECIMAL.value
        if numeric and bs != ps:
            if bs < ps:
                bmul = pow10_checked(ps - bs)
            else:
                pmul = pow10_checked(bs - ps)
        var translate = List[Int32]()
        if kind == TEXT.value:                                       # the probe side's code -> the build side's, once per word
            var words = pt.dictionary(pcols[j])
            for w in range(len(words)):
                translate.append(Int32(bt.code_of(bcols[j], String(words[w]))))
        var bv = List[Int64]()
        var bo = List[UInt8]()
        var pv = List[Int64]()
        var po = List[UInt8]()
        _column_keys(bt, bcols[j], kind, bmul, List[Int32](), False, bv, bo)
        _column_keys(pt, pcols[j], kind, pmul, translate, kind == TEXT.value, pv, po)
        bvals.append(bv^)
        boks.append(bo^)
        pvals.append(pv^)
        poks.append(po^)
    var nb = bt.rows()
    var np = pt.rows()
    keys.bok = List[UInt8](length=nb, fill=1)
    keys.pok = List[UInt8](length=np, fill=1)
    for j in range(nk):
        for i in range(nb):
            if boks[j][i] == 0:
                keys.bok[i] = 0
        for i in range(np):
            if poks[j][i] == 0:
                keys.pok[i] = 0
    if nk == 1:
        keys.bkeys = bvals[0].copy()
        keys.pkeys = pvals[0].copy()
        keys.layout = "one key"
        keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
        keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
        keys.bo_addr = Int(keys.bok.unsafe_ptr())
        keys.po_addr = Int(keys.pok.unsafe_ptr())
        return
    # several keys: packed into one int64 when the build side's ranges multiply to fewer than 2^62 values
    var los = List[Int64]()
    var spans = List[UInt64]()
    var fits = True
    var product = UInt64(1)
    for j in range(nk):
        var lo = Int64.MAX
        var hi = Int64.MIN
        for i in range(nb):
            if keys.bok[i] != 0:
                lo = min(lo, bvals[j][i])
                hi = max(hi, bvals[j][i])
        if lo > hi:
            lo = 0
            hi = 0
        var span = UInt64(hi) - UInt64(lo) + 1
        los.append(lo)
        spans.append(span)
        if span == 0 or product > (UInt64(1) << 62) // span:
            fits = False
        else:
            product *= span
    keys.bkeys = List[Int64](length=nb, fill=0)
    keys.pkeys = List[Int64](length=np, fill=0)
    if fits:
        keys.layout = String(nk) + " keys packed into one int64 (" + String(product) + " combinations)"
        var stride = UInt64(1)
        for j in range(nk):
            for i in range(nb):
                keys.bkeys[i] = Int64(UInt64(keys.bkeys[i]) + UInt64(bvals[j][i] - los[j]) * stride)
            for i in range(np):
                if keys.pok[i] != 0 and (pvals[j][i] < los[j] or UInt64(pvals[j][i] - los[j]) >= spans[j]):
                    keys.pok[i] = 0                                  # outside the build side's range: it cannot match
                else:
                    keys.pkeys[i] = Int64(UInt64(keys.pkeys[i]) + UInt64(pvals[j][i] - los[j]) * stride)
            stride *= spans[j]
        keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
        keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
        keys.bo_addr = Int(keys.bok.unsafe_ptr())
        keys.po_addr = Int(keys.pok.unsafe_ptr())
        return
    keys.verify = True
    keys.layout = String(nk) + " keys hashed to one int64 (their ranges are too wide to pack), pairs verified against the columns"
    for i in range(nb):
        var h = UInt64(0x9E3779B97F4A7C15)
        for j in range(nk):
            h = _mix(h ^ UInt64(bvals[j][i]))
        keys.bkeys[i] = Int64(h)
    for i in range(np):
        var h = UInt64(0x9E3779B97F4A7C15)
        for j in range(nk):
            h = _mix(h ^ UInt64(pvals[j][i]))
        keys.pkeys[i] = Int64(h)
    keys.bcomp = bvals^
    keys.pcomp = pvals^
    keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
    keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
    keys.bo_addr = Int(keys.bok.unsafe_ptr())
    keys.po_addr = Int(keys.pok.unsafe_ptr())


# ── the pairs ────────────────────────────────────────────────────────────────────────────────────────

struct JoinPairs(Movable):
    """The matching (probe row, build row) pairs, in probe-row order: two Int32 arrays the kernel allocated (freed with the pairs)."""
    var n: Int
    var probe_a: Int
    var build_a: Int

    def __init__(out self):
        self.n = 0
        self.probe_a = 0
        self.build_a = 0

    def __del__(deinit self):
        hj_release(self.probe_a)
        hj_release(self.build_a)


def join_pairs(m: JoinMap, keys: JoinKeys, n_probe: Int, budget_rows: Int, threads: Int) raises -> JoinPairs:
    var out = JoinPairs()
    if n_probe == 0:
        return out^
    var info = List[Int64](length=4, fill=0)
    var rc = hj_probe_pairs(m, keys.pk_addr, keys.po_addr, n_probe, threads, budget_rows, Int(info.unsafe_ptr()))
    if rc == E_BUDGET:
        raise Error("the join produces " + commas(Int(info[0])) + " rows, past the budget of " + commas(budget_rows) + " (the joined columns would take over " +
                    String(JOIN_BUDGET_BYTES >> 30) + " GiB): filter either side first, or join on a unique key (a join that spills to disk is not built yet)")
    if rc != 0:
        raise Error("the hash join failed with code " + String(rc))
    var n = Int(info[0])
    var pr = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(info[1]))
    var br = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(info[2]))
    if keys.verify:                                                  # hashed tuples: keep the pairs whose key columns are really equal
        var kept = 0
        for i in range(n):
            var same = True
            for j in range(len(keys.bcomp)):
                if keys.bcomp[j][Int(br[i])] != keys.pcomp[j][Int(pr[i])]:
                    same = False
                    break
            if same:
                pr[kept] = pr[i]
                br[kept] = br[i]
                kept += 1
        n = kept
    out.n = n
    out.probe_a = Int(info[1])
    out.build_a = Int(info[2])
    return out^


struct JoinSide(Movable):
    """One side of the join: its table, the table's columns the joined rows carry (the needed ones, in order) and the table's column of each
    join key. A side with no filter over a plain table is the source table itself (nothing is copied); any other side is the result of a row
    query over its source."""
    var table: SharedTable
    var cols: List[Int]
    var key_cols: List[Int]
    var zero_copy: Bool

    def __init__(out self, table: SharedTable, var cols: List[Int], var key_cols: List[Int], zero_copy: Bool):
        self.table = table
        self.cols = cols^
        self.key_cols = key_cols^
        self.zero_copy = zero_copy


def _gather_column(src: Int, valid: Int, rows_a: Int, n: Int, dst_a: Int, nul_a: Int, threads: Int):
    """dst[i] = src[rows[i]] (and the NULL byte from the validity bits), over slices of the rows in parallel."""
    var tasks = max(1, min(threads if threads > 0 else core_count(), n // 32768 + 1))

    def work(t: Int) {imm}:
        var s = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=src)
        var rows = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=rows_a)
        var dst = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst_a)
        var a = n * t // tasks
        var b = n * (t + 1) // tasks
        for i in range(a, b):
            dst[i] = s[Int(rows[i])]
        if nul_a != 0 and valid != 0:
            var nd = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=nul_a)
            var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
            for i in range(a, b):
                var r = Int(rows[i])
                nd[i] = 1 - ((bits[r >> 3] >> UInt8(r & 7)) & 1)

    par_n(work, tasks, tasks)


def gather_join(left: JoinSide, right: JoinSide, pairs: JoinPairs, left_builds: Bool, threads: Int) raises -> Table:
    """The joined table: the carried columns of the left side at its rows of the pairs, then the right side's (the build side's rows are the
    pairs' build rows, the probe side's the pairs' probe rows)."""
    var n = pairs.n
    var t = Table.empty()
    for side in range(2):
        ref sd = left if side == 0 else right
        var rows_a = (pairs.build_a if left_builds else pairs.probe_a) if side == 0 else (pairs.probe_a if left_builds else pairs.build_a)
        for c in sd.cols:
            var kind = sd.table.kind(c)
            var col = t.add_column(_unique_name(t, sd.table.name(c)), kind, scale=sd.table.scale(c))
            var vals = List[Int64](unsafe_uninit_length=n)
            var has_nulls = sd.table.null_count(c) > 0
            var nul = List[UInt8](length=n if has_nulls else 0, fill=0)
            _gather_column(Int(sd.table.values(c).unsafe_ptr()), _valid_bits(sd.table, c), rows_a, n, Int(vals.unsafe_ptr()),
                           Int(nul.unsafe_ptr()) if has_nulls else 0, threads)
            if kind == TEXT:                                          # the side's whole dictionary, codes unchanged
                var words = sd.table.dictionary(c)
                for w in range(len(words)):
                    _ = t.add_word(col, String(words[w]))
            _fill_column_owned(t.cols[col], vals^, nul^)
    return t^


def _fill_column_owned(mut col: Column, var vals: List[Int64], var nul: List[UInt8]):
    col.values = vals^
    if len(nul) > 0:
        var n = len(col.values)
        col.validity = List[UInt8](length=(n + 7) // 8, fill=0xFF)
        var nulls = 0
        for i in range(n):
            if nul[i] != 0:
                col.validity[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
                nulls += 1
        col.nulls = nulls
        if nulls == 0:
            col.validity = List[UInt8]()


def virtual_schema(left: JoinSide, right: JoinSide) raises -> Table:
    """A table with no rows that has the joined columns' names, kinds, scales, dictionaries and NULL counts: what the planner and the
    program's lowering read of the joined source when the join is fused (the rows come from the probe table and the build side's gather)."""
    var t = Table.empty()
    for side in range(2):
        ref sd = left if side == 0 else right
        for c in sd.cols:
            var kind = sd.table.kind(c)
            var col = t.add_column(_unique_name(t, sd.table.name(c)), kind, scale=sd.table.scale(c))
            if kind == TEXT:
                var words = sd.table.dictionary(c)
                for w in range(len(words)):
                    _ = t.add_word(col, String(words[w]))
            t.cols[col].nulls = sd.table.null_count(c)
    return t^
