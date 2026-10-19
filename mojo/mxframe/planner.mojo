"""The physical planner (F-150): how a bound query runs, decided from statistics alone.

`Planner` is a trait: `plan(query, stats, options)` returns a `PhysicalPlan` and the reasons for each decision in
`notes`, which EXPLAIN prints and the tests assert. `RulePlanner` is the first implementation; a cost-based one
(F-155) replaces it without touching the executor. `PlannerOptions` forces each choice, so a test can reach every path
and check which one ran.

What it decides today: the group strategy.
- no GROUP BY: one group, the plain kernel;
- dense: the keys are columns whose values lie in a known range (a TEXT column's dictionary, BOOL, the min / max of an
  integer, decimal or date column), laid out mixed-radix with a NULL slot per nullable key, and the kernel addresses
  the group directly: no hashing, the fastest path while the domain is not much bigger than the rows;
- hashed: key words go through per-worker hash tables and a partitioned merge (`hashagg_cpu`): any keys, any range.
"""
from .table import ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL
from .bind import (BoundQuery, BExpr, SqlSource, SRC_TABLE, SRC_FRAME, SRC_SCAN, B_COL, B_NULL, B_IDIV, B_IMOD, kind_name)

comptime GROUP_AUTO = 0
comptime GROUP_DENSE = 1
comptime GROUP_HASHED = 2
comptime G_NONE = 0          # a plan's strategy: one group
comptime G_DENSE = 1
comptime G_HASHED = 2

comptime DEFAULT_DENSE_LIMIT = 4_194_304     # the most group slots a dense layout may take
comptime SPARSE_FACTOR = 4                   # past this many slots a row, the domain is too sparse for dense (unless small)
comptime SMALL_DOMAIN = 65_536               # a domain this small is always dense


struct PlannerOptions(Copyable, Movable):
    var group: Int                           # GROUP_AUTO, GROUP_DENSE or GROUP_HASHED
    var dense_limit: Int
    var threads: Int                         # 0: every core

    def __init__(out self):
        self.group = GROUP_AUTO
        self.dense_limit = DEFAULT_DENSE_LIMIT
        self.threads = 0


struct ColStat(Copyable, Movable):
    var kind: Int
    var scale: Int
    var nullable: Bool           # may hold a NULL (unknown counts as yes)
    var has_range: Bool
    var lo: Int64
    var hi: Int64
    var words: Int               # a TEXT column's dictionary size

    def __init__(out self, kind: Int, scale: Int):
        self.kind = kind
        self.scale = scale
        self.nullable = True
        self.has_range = False
        self.lo = 0
        self.hi = 0
        self.words = 0


struct Stats(Movable):
    var source: String
    var kind: Int                # SRC_TABLE, SRC_FRAME or SRC_SCAN
    var rows: Int
    var cols: List[ColStat]
    var row_groups: Int
    var edits: Int               # a frame: the changed and appended rows that go through the delta
    var version: Int             # a frame: its edit counter when it was registered

    def __init__(out self, source: String, kind: Int, rows: Int):
        self.source = source
        self.kind = kind
        self.rows = rows
        self.cols = List[ColStat]()
        self.row_groups = 0
        self.edits = 0
        self.version = 0


@fieldwise_init
struct KeyLayout(Copyable, Movable):
    var expr: Int                # the key's row-level expression
    var kind: Int
    var scale: Int
    var nullable: Bool
    var lo: Int64                # dense: the smallest value (a TEXT code or BOOL starts at 0)
    var radix: Int               # dense: the slots this key takes, its NULL slot included
    var stride: Int              # dense: the slot step of this key
    var src: Int                 # the source column of a plain-column key, else -1
    var words: Int               # hashed: key words (the value, and a NULL flag when nullable)


struct PhysicalPlan(Movable):
    var group: Int
    var keys: List[KeyLayout]
    var domain: Int
    var holistic: Bool
    var source_kind: Int
    var notes: List[String]

    def __init__(out self):
        self.group = G_NONE
        self.keys = List[KeyLayout]()
        self.domain = 1
        self.holistic = False
        self.source_kind = SRC_TABLE
        self.notes = List[String]()

    def key_words(self) -> Int:
        var n = 0
        for k in self.keys:
            n += k.words
        return n


trait Planner:
    def plan(self, q: BoundQuery, st: Stats, opts: PlannerOptions) raises -> PhysicalPlan:
        ...


def expr_nullable(q: BoundQuery, e: Int, st: Stats) -> Bool:
    """Can the row-level expression be NULL? (a nullable column, a NULL literal, a division that may meet zero)"""
    ref x = q.rexprs[e]
    if x.op == B_COL:
        return st.cols[Int(x.value)].nullable
    if x.op == B_NULL or x.op == B_IDIV or x.op == B_IMOD:
        return True
    var r = False
    if x.a >= 0:
        r = r or expr_nullable(q, x.a, st)
    if x.b >= 0:
        r = r or expr_nullable(q, x.b, st)
    return r


def commas(n: Int) -> String:
    var s = String(n if n >= 0 else -n)
    var out = String()
    var k = s.byte_length()
    for i in range(k):
        if i > 0 and (k - i) % 3 == 0:
            out += ","
        out += String(s[byte=i:i + 1])
    return ("-" if n < 0 else "") + out


struct RulePlanner(Planner, Copyable, Movable):
    def __init__(out self):
        pass

    def plan(self, q: BoundQuery, st: Stats, opts: PlannerOptions) raises -> PhysicalPlan:
        var plan = PhysicalPlan()
        plan.source_kind = st.kind
        for a in q.aggs:
            if a.hol:
                plan.holistic = True
        var what = "table" if st.kind == SRC_TABLE else "frame at edit " + String(st.version) + ": the edits as a mask over the shared rows plus " + \
                   commas(st.edits) + " changed or appended rows" if st.kind == SRC_FRAME else "Parquet scan, " + commas(st.row_groups) + " row groups"
        plan.notes.append("source: " + st.source + " (" + what + "), " + commas(st.rows) + " rows")
        if len(q.keys) == 0:
            plan.group = G_NONE
            plan.notes.append("group: none (one result row)")
            return plan^
        # per key: can it be addressed directly, and how many slots does it take?
        var dense_ok = True
        var why_not = String()
        var domain = 1
        var parts = String()
        for k in range(len(q.keys)):
            var e = q.rexprs[q.keys[k].expr].copy()
            var kind = e.kind
            var layout = KeyLayout(q.keys[k].expr, kind, e.scale, True, 0, 1, 1, -1, 1)
            if e.op == B_COL:
                var c = Int(e.value)
                layout.src = c
                ref cs = st.cols[c]
                layout.nullable = cs.nullable
                var radix = 0
                if kind == TEXT.value:
                    radix = cs.words
                    layout.lo = 0
                elif kind == BOOL.value:
                    radix = 2
                    layout.lo = 0
                elif cs.has_range:
                    layout.lo = cs.lo
                    var span = Int(cs.hi) - Int(cs.lo)
                    radix = span + 1 if span >= 0 and span < (1 << 40) else -1
                else:
                    radix = -1
                if radix < 0:
                    dense_ok = False
                    if why_not.byte_length() == 0:
                        why_not = "the range of `" + q.keys[k].name + "` is " + ("too wide" if cs.has_range else "unknown (the source has no statistics on it)")
                    radix = 1
                if layout.nullable:
                    radix += 1
                layout.radix = radix
            else:
                dense_ok = False
                if why_not.byte_length() == 0:
                    why_not = "`" + q.keys[k].name + "` is an expression, so its range is unknown"
                layout.nullable = expr_nullable(q, q.keys[k].expr, st)
            layout.words = 1 + (1 if layout.nullable else 0)
            if dense_ok:
                if domain > (1 << 40) // max(layout.radix, 1):
                    dense_ok = False
                    why_not = "the domain passes 2^40 slots"
                else:
                    domain *= layout.radix
            parts += (" x " if k > 0 else "") + q.keys[k].name + " " + String(layout.radix) + (" (with a NULL slot)" if layout.nullable else "")
            plan.keys.append(layout^)
        var use_dense = False
        var reason = String()
        var est_slots = 1 + 2 * len(q.aggs)                         # the dense kernel holds a word per slot per group
        if dense_ok and domain > (1 << 27) // est_slots:
            dense_ok = False
            why_not = "the domain of " + commas(domain) + " slots times " + String(est_slots) + " accumulators passes the memory bound"
        if opts.group == GROUP_DENSE:
            if not dense_ok:
                raise Error("cannot force dense groups: " + why_not)
            if domain > opts.dense_limit:
                raise Error("cannot force dense groups: the domain of " + commas(domain) + " slots passes the limit of " + commas(opts.dense_limit))
            use_dense = True
            reason = "forced by the options"
        elif opts.group == GROUP_HASHED:
            reason = "forced by the options"
        elif dense_ok and domain <= opts.dense_limit and (domain <= SMALL_DOMAIN or domain <= SPARSE_FACTOR * max(st.rows, 1)):
            use_dense = True
            reason = "the domain of " + commas(domain) + " slots is " + ("small" if domain <= SMALL_DOMAIN else "no more than " + String(SPARSE_FACTOR) + "x the rows")
        elif not dense_ok:
            reason = why_not
        elif domain > opts.dense_limit:
            reason = "the domain of " + commas(domain) + " slots passes the dense limit of " + commas(opts.dense_limit)
        else:
            reason = "the domain of " + commas(domain) + " slots is sparse for " + commas(st.rows) + " rows"
        if use_dense:
            plan.group = G_DENSE
            plan.domain = domain
            var stride = 1
            var k = len(plan.keys) - 1
            while k >= 0:
                plan.keys[k].stride = stride
                stride *= plan.keys[k].radix
                k -= 1
            plan.notes.append("group: dense, " + String(len(plan.keys)) + (" key" if len(plan.keys) == 1 else " keys") + " " + parts +
                              " = " + commas(domain) + " slots: " + reason)
        else:
            plan.group = G_HASHED
            plan.domain = 0
            plan.notes.append("group: hashed on " + String(plan.key_words()) + " key " + ("word" if plan.key_words() == 1 else "words") + ": " + reason)
        return plan^
