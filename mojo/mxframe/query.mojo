"""Queries (docs/mxframe-mojo-api.md section 5), on a small plan representation lowered to the native kernel.

    var r = Query(table).where(col("region") == 2)
                        .group_by(col("carrier"), groups=7)       # a key expression with a known group count
                        .agg(sum(col("revenue")), count())
                        .compute()
    r.sums; r.counts; r.values(i); r.means(i); r.milliseconds

The plan is a list of nodes (Scan, Filter, Aggregate today; Project, Sort, Limit and Join have their kinds
reserved), so the smart planner can later rewrite and extend it without changing this API. `compute` lowers the
plan to one program for the CPU kernel (`kernels/pipeline_cpu.mojo`): filters ANDed into the predicate, the group
key computed per block, every value an exact int64.

Semantics, as the app's stand-in (mojoui frame_gpu.group_sum), plus nulls:
- groups are dense 0 .. groups-1; a row whose key is outside that range, or null, is left out;
- a filter that reads a null cell is false for that row (the whole filter, not only the comparison);
- sum / mean / min / max skip a row whose value reads a null cell; count() counts rows, count(e) rows where e
  reads no null; `present(i, g)`: aggregate i has a value in group g (its min / max / mean are meaningless otherwise);
- values are the stored integers (a DECIMAL(2) literal 5 is 0.05); + - and comparisons refuse columns of different
  scales (decimal-aware formulas: F-136);
- the null rule for `|`: a filter reading a null cell is false as a whole, so `(a > 0) | (b == 1)` drops a row whose
  a is null even when b == 1 (SQL would keep it);
- a total that passes int64 raises (the stand-in wrapped silently).
"""
from std.time import perf_counter_ns
from std.sys.info import num_physical_cores
from std.memory import Pointer
from .table import SharedTable
from .expr import (Expr, Node, N_COL, N_LIT, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG, N_LEAST, N_GREATEST,
                   N_LT, N_LE, N_GT, N_GE, N_EQ, N_NE, N_AND, N_OR, N_NOT)
from .kernels.pipeline_cpu import pipeline_keyed, pipeline_plain, E_OVERFLOW, IP, FP, LP

# plan node kinds
comptime P_SCAN = 0
comptime P_FILTER = 1
comptime P_PROJECT = 2       # reserved
comptime P_AGGREGATE = 3
comptime P_SORT = 4          # reserved
comptime P_LIMIT = 5         # reserved
comptime P_JOIN = 6          # reserved

# aggregate kinds
comptime A_SUM = 0
comptime A_COUNT = 1         # count() with no argument: rows
comptime A_COUNT_OF = 2      # count(e): rows where e reads no null
comptime A_MIN = 3
comptime A_MAX = 4
comptime A_MEAN = 5

# devices
comptime CPU = 0
comptime GPU = 1
comptime AUTO = 2

# kernel opcodes (kernels_aot/pipeline_cpu.mojo, pipeline_program.py)
comptime MAGIC = 0x4D585031
comptime OP_LOAD_CONST = 2
comptime OP_LOADI = 3
comptime OP_IADD = 15
comptime OP_ISUB = 16
comptime OP_IMUL = 17
comptime OP_F2I = 18
comptime OP_AND = 30
comptime OP_OR = 31
comptime OP_NOT = 32
comptime OP_ISNULL = 41
comptime OP_IDIV = 42
comptime OP_IMOD = 43
comptime OP_IMIN = 44
comptime OP_IMAX = 45
comptime OP_ICMP = 46
comptime OP_AGG_ISUM = 53
comptime OP_AGG_IMIN = 54
comptime OP_AGG_IMAX = 55
comptime MAX_REGS = 48
comptime MAX_CONSTS = 64
comptime DT_I64 = 1
comptime DT_NULLMASK = 7


@fieldwise_init
struct Agg(Copyable, Movable):
    var kind: Int
    var arg: Expr


def sum(e: Expr) -> Agg:
    return Agg(A_SUM, e.copy())


def count() -> Agg:
    return Agg(A_COUNT, Expr(0))


def count(e: Expr) -> Agg:
    return Agg(A_COUNT_OF, e.copy())


def mean(e: Expr) -> Agg:
    return Agg(A_MEAN, e.copy())


def min_of(e: Expr) -> Agg:
    return Agg(A_MIN, e.copy())


def max_of(e: Expr) -> Agg:
    return Agg(A_MAX, e.copy())


@fieldwise_init
struct PlanNode(Copyable, Movable):
    var kind: Int
    var input: Int                # the node below (-1 for a scan)
    var exprs: List[Expr]         # Filter: the predicate; Aggregate: the group key (none for one group)
    var aggs: List[Agg]
    var groups: Int


struct Query(Movable):
    """A plan over one shared table, built by chaining; `compute` runs it."""
    var table: SharedTable
    var nodes: List[PlanNode]

    def __init__(out self, table: SharedTable):
        self.table = table
        self.nodes = [PlanNode(P_SCAN, -1, List[Expr](), List[Agg](), 0)]

    def _top(self) -> Int:
        return len(self.nodes) - 1

    def where(var self, predicate: Expr) -> Self:
        """Keep rows where `predicate` holds (several `where`s are ANDed)."""
        self.nodes.append(PlanNode(P_FILTER, self._top(), [predicate.copy()], List[Agg](), 0))
        return self^

    def _aggregate(self) -> Int:
        for i in range(len(self.nodes)):
            if self.nodes[i].kind == P_AGGREGATE:
                return i
        return -1

    def group_by(var self, key: Expr, groups: Int) raises -> Self:
        """Group by an integer key expression with values 0 .. groups-1 (dense: missing groups are zeros).
        `where` filters rows whether it comes before or after (there is no HAVING yet)."""
        if self._aggregate() >= 0:
            raise Error("a query has one group_by (and its aggregates); this one has a second")
        self.nodes.append(PlanNode(P_AGGREGATE, self._top(), [key.copy()], List[Agg](), groups))
        return self^

    def agg(var self, *aggs: Agg) -> Self:
        var at = self._aggregate()
        if at < 0:
            self.nodes.append(PlanNode(P_AGGREGATE, self._top(), List[Expr](), List[Agg](), 1))
            at = self._top()
        for a in aggs:
            self.nodes[at].aggs.append(a.copy())
        return self^

    def compute(self, device: Int = AUTO) raises -> QueryResult:
        """Runs the plan. `device`: CPU, GPU or AUTO (the GPU arrives in M6: GPU raises, AUTO runs on the CPU)."""
        if device == GPU:
            raise Error("device GPU: the GPU path of the Mojo API is not built yet (F-137); use CPU or AUTO")
        var t0 = perf_counter_ns()
        var prog = _lower(self)
        var r = _run_cpu(self.table, prog)
        r.milliseconds = Float64(perf_counter_ns() - t0) / 1e6
        return r^


struct QueryResult(Movable):
    var kinds: List[Int]
    var slot_of: List[Int]          # per aggregate: its kernel slot, -1 when none (count(), mean's count)
    var count_slot_of: List[Int]    # per aggregate: the slot counting its non-null rows (count(e), mean), -1 when none
    var slots: List[List[Int]]      # per kernel slot, per group
    var counts: List[Int]           # rows per group (count(*))
    var sums: List[Int]             # the first sum's values (the spec's `r.sums`), else empty
    var milliseconds: Float64

    def __init__(out self):
        self.kinds = List[Int]()
        self.slot_of = List[Int]()
        self.count_slot_of = List[Int]()
        self.slots = List[List[Int]]()
        self.counts = List[Int]()
        self.sums = List[Int]()
        self.milliseconds = 0.0

    def values(self, i: Int) -> List[Int]:
        """Aggregate i's value per group (a mean: use `means`)."""
        if self.kinds[i] == A_COUNT:
            return self.counts.copy()
        if self.kinds[i] == A_COUNT_OF:
            return self.slots[self.count_slot_of[i]].copy() if self.count_slot_of[i] >= 0 else self.counts.copy()
        return self.slots[self.slot_of[i]].copy()

    def means(self, i: Int) -> List[Float64]:
        """Aggregate i's mean per group: its exact sum over its count (NaN for a group with none)."""
        var s = self.slots[self.slot_of[i]].copy()
        var n = self.slots[self.count_slot_of[i]].copy() if self.count_slot_of[i] >= 0 else self.counts.copy()
        var out = List[Float64]()
        for g in range(len(s)):
            out.append(Float64(s[g]) / Float64(n[g]) if n[g] > 0 else Float64.MAX * 0.0 / 0.0)
        return out^

    def is_empty(self, g: Int) -> Bool:
        """Group g has no rows."""
        return self.counts[g] == 0

    def present(self, i: Int, g: Int) -> Bool:
        """Aggregate i has a value in group g: a row whose value is not null (a min / max / mean over only nulls
        has none, and its number is meaningless)."""
        if self.count_slot_of[i] >= 0:
            return self.slots[self.count_slot_of[i]][g] > 0
        return self.counts[g] > 0


# ── lowering ─────────────────────────────────────────────────────────────────────────────────────────

struct Program(Movable):
    var pred: List[Int32]
    var body: List[Int32]
    var aggs: List[Int32]
    var consts: List[Int64]          # int64 bits, loaded exactly by LOAD_CONST
    var cols: List[Int]              # table column of each program column (data first, then validity triplets)
    var dtypes: List[Int32]
    var null_of: List[Int]           # table column -> program column whose dtype is the null mask, or -1
    var n_regs: Int
    var pred_reg: Int
    var keyed: Bool
    var groups: Int
    var kinds: List[Int]
    var slot_of: List[Int]
    var count_slot_of: List[Int]
    var n_slots: Int
    var section: Int                 # where `emit` writes: 0 predicate, 1 body, 2 aggregates
    var next_keep: Int               # registers 1 .. next_keep-1 hold values that must live on (the predicate's
    var temp_top: Int                # running AND, each aggregate's value); temporaries come down from the top

    def __init__(out self, n_table_cols: Int):
        self.pred = List[Int32]()
        self.body = List[Int32]()
        self.aggs = List[Int32]()
        self.consts = List[Int64]()
        self.cols = List[Int]()
        self.dtypes = List[Int32]()
        self.null_of = List[Int](length=n_table_cols, fill=-1)
        self.n_regs = MAX_REGS           # register 0: the group key (int), read by `_key_from_reg0`
        self.next_keep = 1
        self.temp_top = MAX_REGS - 1
        self.pred_reg = -1
        self.keyed = False
        self.groups = 1
        self.kinds = List[Int]()
        self.slot_of = List[Int]()
        self.count_slot_of = List[Int]()
        self.n_slots = 0
        self.section = 0

    def emit(mut self, op: Int, d: Int, a: Int = 0, b: Int = 0, c: Int = 0):
        var ins: List[Int32] = [Int32(op), Int32(d), Int32(a), Int32(b), Int32(c)]
        if self.section == 0:
            self.pred.extend(ins^)
        elif self.section == 1:
            self.body.extend(ins^)
        else:
            self.aggs.extend(ins^)

    def reg(mut self) raises -> Int:
        """A temporary register, free again after `reset_temps`."""
        if self.temp_top < self.next_keep:
            raise Error("the query is too big for one program (more than " + String(MAX_REGS) + " registers)")
        self.temp_top -= 1
        return self.temp_top + 1

    def keep(mut self, r: Int, boolean: Bool) raises -> Int:
        """Copies register r into one that lives to the end of the program."""
        if self.next_keep > self.temp_top:
            raise Error("the query is too big for one program (more than " + String(MAX_REGS) + " registers)")
        var k = self.next_keep
        self.next_keep += 1
        self.emit(OP_AND if boolean else OP_IMAX, k, r, r)       # a & a, max(a, a): a copy
        return k

    def reset_temps(mut self):
        self.temp_top = MAX_REGS - 1

    def const(mut self, v: Int64) raises -> Int:
        for k in range(len(self.consts)):
            if self.consts[k] == v:
                return k
        if len(self.consts) >= MAX_CONSTS:
            raise Error("the query has more than " + String(MAX_CONSTS) + " distinct constants")
        self.consts.append(v)
        return len(self.consts) - 1

    def column(mut self, table_col: Int) -> Int:
        for k in range(len(self.cols)):
            if self.cols[k] == table_col and Int(self.dtypes[k]) == DT_I64:
                return k
        self.cols.append(table_col)
        self.dtypes.append(DT_I64)
        return len(self.cols) - 1


def _op_name(op: Int) -> String:
    var names: List[String] = ["+", "-", "*", "//", "%", "-", "least", "greatest", "", "", "<", "<=", ">", ">=",
                               "==", "!=", "", "", "", "", "&", "|", "~"]
    return names[op - N_ADD] if N_ADD <= op <= N_NOT else "operator " + String(op)


@fieldwise_init
struct _Slot(Copyable, Movable):
    var reg: Int
    var boolean: Bool       # a comparison's 1.0 / 0.0 (float) rather than an int64 value
    var scale: Int          # a column's decimal places; -1 for a literal or a mix that has none


def _compile_expr(mut p: Program, e: Expr, t: SharedTable) raises -> Int:
    """Emits e; returns its register: an int register for a value, a float (1.0 / 0.0) for a boolean. Raises when an
    operator gets the wrong kind (a number where a comparison is due, or the reverse) or when + - and comparisons
    meet columns of different decimal scales (decimal-aware formulas are F-136: values are the stored integers)."""
    var stack = List[_Slot]()
    for n in e.nodes:
        if n.op == N_COL:
            var c = t.column(n.name)
            var r = p.reg()
            p.emit(OP_LOADI, r, p.column(c))
            stack.append(_Slot(r, False, t.scale(c)))
        elif n.op == N_LIT:
            var r = p.reg()
            p.emit(OP_LOAD_CONST, r, p.const(n.value))
            stack.append(_Slot(r, False, -1))
        elif n.op == N_NEG or n.op == N_NOT:
            var a = stack.pop()
            if a.boolean != (n.op == N_NOT):
                raise Error("`" + _op_name(n.op) + "` needs " + ("a comparison" if n.op == N_NOT else "a number") +
                            ", got " + ("a number" if n.op == N_NOT else "a comparison"))
            var r = p.reg()
            if n.op == N_NOT:
                p.emit(OP_NOT, r, a.reg)
            else:
                var z = p.reg()
                p.emit(OP_LOAD_CONST, z, p.const(0))
                p.emit(OP_ISUB, r, z, a.reg)
            stack.append(_Slot(r, n.op == N_NOT, a.scale))
        else:
            var b = stack.pop()
            var a = stack.pop()
            var logical = n.op == N_AND or n.op == N_OR
            if a.boolean != logical or b.boolean != logical:
                raise Error("`" + _op_name(n.op) + "` needs " + ("comparisons" if logical else "numbers") + " on both sides")
            var compare = n.op >= N_LT and n.op <= N_NE
            if (n.op == N_ADD or n.op == N_SUB or n.op == N_LEAST or n.op == N_GREATEST or compare) and \
               a.scale >= 0 and b.scale >= 0 and a.scale != b.scale:
                raise Error("`" + _op_name(n.op) + "` meets values with " + String(a.scale) + " and " + String(b.scale) +
                            " decimal places: the stored integers would not line up (decimal-aware formulas: F-136)")
            var r = p.reg()
            var scale = a.scale if a.scale >= 0 else b.scale
            if n.op == N_ADD:
                p.emit(OP_IADD, r, a.reg, b.reg)
            elif n.op == N_SUB:
                p.emit(OP_ISUB, r, a.reg, b.reg)
            elif n.op == N_MUL:
                p.emit(OP_IMUL, r, a.reg, b.reg)
                scale = a.scale + b.scale if a.scale >= 0 and b.scale >= 0 else scale
            elif n.op == N_FLOORDIV:
                p.emit(OP_IDIV, r, a.reg, b.reg)
                scale = -1
            elif n.op == N_MOD:
                p.emit(OP_IMOD, r, a.reg, b.reg)
            elif n.op == N_LEAST:
                p.emit(OP_IMIN, r, a.reg, b.reg)
            elif n.op == N_GREATEST:
                p.emit(OP_IMAX, r, a.reg, b.reg)
            elif compare:
                p.emit(OP_ICMP, r, a.reg, b.reg, n.op)      # N_LT .. N_NE are the kernel's OP_LT .. OP_NE
            elif n.op == N_AND:
                p.emit(OP_AND, r, a.reg, b.reg)
            elif n.op == N_OR:
                p.emit(OP_OR, r, a.reg, b.reg)
            else:
                raise Error("unknown expression operator " + String(n.op))
            stack.append(_Slot(r, compare or logical, -1 if compare or logical else scale))
    if len(stack) != 1:
        raise Error("malformed expression")
    return stack[0].reg


def _null_mask(mut p: Program, table_col: Int) -> Int:
    """The program column whose dtype makes OP_ISNULL read `table_col`'s validity bitmap (two more entries hold the
    bitmap's address and its first bit)."""
    if p.null_of[table_col] >= 0:
        return p.null_of[table_col]
    var at = len(p.cols)
    p.cols.append(table_col)                       # the mask column (its own entry is never read)
    p.dtypes.append(Int32(DT_NULLMASK | ((at + 1) << 8)))
    p.cols.append(-1 - table_col)                  # -1 - c: the validity bitmap of column c
    p.dtypes.append(DT_I64)
    p.cols.append(-1_000_000)                      # the bit base (0: one chunk starting at row 0)
    p.dtypes.append(DT_I64)
    p.null_of[table_col] = at
    return at


def _valid_of(mut p: Program, e: Expr, t: SharedTable) raises -> Int:
    """An int register: 1 where every column e reads is non-null, else 0. -1 when e reads no nullable column."""
    var out = -1
    for name in e.columns():
        var c = t.column(name)
        if t.null_count(c) == 0:
            continue
        var isnull = p.reg()
        p.emit(OP_ISNULL, isnull, _null_mask(p, c))
        var i = p.reg()
        p.emit(OP_F2I, i, isnull)
        var one = p.reg()
        p.emit(OP_LOAD_CONST, one, p.const(1))
        var v = p.reg()
        p.emit(OP_ISUB, v, one, i)
        if out < 0:
            out = v
        else:
            var both = p.reg()
            p.emit(OP_IMUL, both, out, v)
            out = both
    return out


def _lower(q: Query) raises -> Program:
    var t = q.table
    var p = Program(t.columns())
    var filters = List[Expr]()
    var agg_node = -1
    for i in range(len(q.nodes)):
        ref node = q.nodes[i]
        if node.kind == P_FILTER:
            filters.append(node.exprs[0].copy())
        elif node.kind == P_AGGREGATE:
            agg_node = i
        elif node.kind != P_SCAN:
            raise Error("plan node kind " + String(node.kind) + " is not supported yet")
    if agg_node < 0 or len(q.nodes[agg_node].aggs) == 0:
        raise Error("a query needs at least one aggregate (.agg(...))")
    ref an = q.nodes[agg_node]
    p.groups = an.groups
    if p.groups < 1:
        raise Error("groups must be at least 1, got " + String(p.groups))

    # predicate: every filter, each false where it reads a null; the key in range and not null. Each part is ANDed
    # into one kept register as it completes, so its temporaries are free for the next.
    p.section = 0
    var acc = -1
    for f in filters:
        if not f.is_bool():
            raise Error("a filter must be a comparison (==, <, between, &, |, ~), not a value")
        var r = _compile_expr(p, f, t)
        var v = _valid_of(p, f, t)
        if v >= 0:
            var vf = p.reg()
            p.emit(OP_ICMP, vf, v, _const_reg(p, 1), 24)            # valid == 1, as a boolean
            var both = p.reg()
            p.emit(OP_AND, both, r, vf)
            r = both
        acc = _and_into(p, acc, r)
    if len(an.exprs) > 0:
        p.keyed = True
        ref key = an.exprs[0]
        if key.is_bool():
            raise Error("a group key must be a value, not a comparison")
        var kv = _valid_of(p, key, t)                      # a null key: the row is left out
        if kv >= 0:
            var vf = p.reg()
            p.emit(OP_ICMP, vf, kv, _const_reg(p, 1), 24)
            acc = _and_into(p, acc, vf)
        # (the key's range is checked per selected row by `_key_from_reg0`: a key formula is never evaluated on a
        # row the filters left out, so it cannot raise there)
        # the body: the key into register 0, where `_key_from_reg0` reads it per block
        p.section = 1
        var kb = _compile_expr(p, key, t)
        p.emit(OP_IMAX, 0, kb, kb)
        p.reset_temps()
        p.section = 0
    elif p.groups != 1:
        raise Error("groups=" + String(p.groups) + " needs a group_by key")
    p.pred_reg = acc
    p.reset_temps()

    # aggregates: one kernel slot per sum / min / max, one more per count(e) / mean over nullable columns
    for a in an.aggs:
        p.kinds.append(a.kind)
        p.slot_of.append(-1)
        p.count_slot_of.append(-1)
        var i = len(p.kinds) - 1
        if a.kind == A_COUNT:
            continue
        p.section = 1
        var valid = _valid_of(p, a.arg, t)
        if a.kind == A_COUNT_OF or a.kind == A_MEAN or a.kind == A_MIN or a.kind == A_MAX:
            if valid >= 0:                               # the rows this aggregate counts (non-null values)
                var kept_valid = p.keep(valid, False)
                p.section = 2
                p.emit(OP_AGG_ISUM, p.n_slots, kept_valid)
                p.section = 1
                p.count_slot_of[i] = p.n_slots
                p.n_slots += 1
            if a.kind == A_COUNT_OF:
                p.reset_temps()
                continue
        var v = _compile_expr(p, a.arg, t)
        if a.arg.is_bool():
            raise Error("an aggregate needs a value, not a comparison")
        if valid >= 0:                                   # skip null rows: value * valid (+ the identity * isnull)
            var kept = p.reg()
            p.emit(OP_IMUL, kept, v, valid)
            v = kept
            if a.kind == A_MIN or a.kind == A_MAX:
                var one = _const_reg(p, 1)
                var isnull = p.reg()
                p.emit(OP_ISUB, isnull, one, valid)
                var ident = _const_reg(p, Int64.MAX if a.kind == A_MIN else Int64.MIN)
                var fill = p.reg()
                p.emit(OP_IMUL, fill, ident, isnull)
                var both = p.reg()
                p.emit(OP_IADD, both, v, fill)
                v = both
        var op = OP_AGG_IMIN if a.kind == A_MIN else OP_AGG_IMAX if a.kind == A_MAX else OP_AGG_ISUM
        var kept = p.keep(v, False)
        p.reset_temps()
        p.section = 2
        p.emit(op, p.n_slots, kept)
        p.slot_of[i] = p.n_slots
        p.n_slots += 1
    if p.n_slots == 0:                                   # only count(): one dummy sum keeps the kernel happy
        p.section = 1
        var z = p.keep(_const_reg(p, 0), False)
        p.section = 2
        p.emit(OP_AGG_ISUM, 0, z)
        p.n_slots = 1
    return p^


def _and_into(mut p: Program, acc: Int, r: Int) raises -> Int:
    """acc AND r in the predicate's kept register (made from r the first time); temporaries are free after."""
    if acc < 0:
        var k = p.keep(r, True)
        p.reset_temps()
        return k
    p.emit(OP_AND, acc, acc, r)
    p.reset_temps()
    return acc


def _const_reg(mut p: Program, v: Int64) raises -> Int:
    var r = p.reg()
    p.emit(OP_LOAD_CONST, r, p.const(v))
    return r


# ── running ──────────────────────────────────────────────────────────────────────────────────────────

def _key_from_reg0(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    """The group ids of a block's selected rows from register 0 (int64 bits). A key outside 0 .. 2^31-1 becomes -1;
    the kernel's accumulators skip any id outside 0 .. groups-1, so out-of-range keys are left out (the app's rule)."""
    var k = LP(unsafe_from_address=Int(regs))
    for j in range(nsel):
        var v = k[j]
        dst[j] = Int32(v) if v >= 0 and v <= Int64(Int32.MAX) else -1
    return True


def _run_cpu(t: SharedTable, p: Program) raises -> QueryResult:
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
    for v in p.consts:                       # int64 bits in the float64 pool: LOAD_CONST copies them exactly
        consts.append(Float64(from_bits=UInt64(v)))
    if len(consts) == 0:
        consts.append(0.0)
    var set_bytes: List[UInt8] = [0]
    var set_meta: List[Int64] = [0, 0, 0]
    var tbl: List[Int64] = [0, Int64(n)]
    for c in p.cols:
        if c >= 0:
            tbl.append(Int64(Int(t.values(c).unsafe_ptr())))
        elif c == -1_000_000:
            tbl.append(0)
        else:
            tbl.append(Int64(t._table[].cols[-1 - c].validity_address()))
    var dtypes = p.dtypes.copy()
    if len(dtypes) == 0:
        dtypes.append(DT_I64)
    var out_vals = List[Float64](length=p.n_slots * p.groups, fill=0.0)
    var out_cnt = List[Int64](length=p.groups, fill=0)
    var threads = num_physical_cores()
    var rc: Int
    if p.keyed:
        rc = pipeline_keyed[type_of(_key_from_reg0)](Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                                            Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()),
                                            p.groups, 0, n, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()),
                                            threads, _key_from_reg0, False)
    else:
        rc = pipeline_plain(Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                            Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()),
                            0, 1, 0, n, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()), threads, False)
    _ = code^, consts^, set_bytes^, set_meta^, tbl^, dtypes^      # the kernel read them: keep them past the call
    if rc == E_OVERFLOW:
        raise Error("a sum or an expression passed the int64 range (about 9.2e18)")
    if rc != 0:
        raise Error("the CPU kernel failed with code " + String(rc))
    var r = QueryResult()
    r.kinds = p.kinds.copy()
    r.slot_of = p.slot_of.copy()
    r.count_slot_of = p.count_slot_of.copy()
    var bits = out_vals.unsafe_ptr().bitcast[Int64]()
    for s in range(p.n_slots):
        var vals = List[Int]()
        for g in range(p.groups):
            vals.append(Int(bits[s * p.groups + g]))
        r.slots.append(vals^)
    for g in range(p.groups):
        r.counts.append(Int(out_cnt[g]))
    for i in range(len(p.kinds)):
        if p.kinds[i] == A_SUM:
            r.sums = r.slots[p.slot_of[i]].copy()
            break
    _ = out_vals^
    return r^
