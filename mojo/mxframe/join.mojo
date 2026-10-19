"""The SQL side of the inner equi-join (F-153): what a bound query needs of each table, and the queries that run before and after the join.

The join runs on two in-memory sides (`joincore.mojo`). Each side is first reduced to what the query needs of it: the executor runs it as a row
query over its source (so a Frame's edits, a Parquet scan's row groups and the filter that reads only that side are all applied, and a TEXT column
gets a compact dictionary), or uses the source table itself when nothing filters it. The rest of the query (the filter that reads both sides,
aggregates, ORDER BY, LIMIT) runs over the joined rows: a table made from the pairs, or, when the build side's keys are unique, the probe table with
the build side's columns gathered per block inside the pipeline (`jfront.mojo`).
"""
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .bind import (BoundQuery, BExpr, Catalog, KeySpec, AggSpec, SortKey, copy_query, B_COL, B_LIT, B_AND)
from .joincore import JoinSide, _unique_name


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
