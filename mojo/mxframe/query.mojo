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
from .frame import Frame, Overlay, _eval_row
from .expr import (Expr, Node, N_COL, N_LIT, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG, N_LEAST, N_GREATEST,
                   N_LT, N_LE, N_GT, N_GE, N_EQ, N_NE, N_AND, N_OR, N_NOT, N_BOOL2INT)
from .kernels.pipeline_cpu import pipeline_keyed, pipeline_plain, E_OVERFLOW, IP, FP, LP
from .kernels.pipeline_gpu import run_pipeline_gpu
from .gpu import DeviceTable

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
# AUTO uses the GPU when a DeviceTable is attached and the table has at least this many rows: below it the CPU
# kernel (all cores, no launch or copy-back) answers in about the time a GPU launch plus the result copy take.
comptime AUTO_GPU_MIN_ROWS = 1_000_000

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
    """A plan over one shared table, or over a session's view of it (`Query(frame)`: its edits, appended and removed
    rows, merges and formula columns, copied when the query is built), built by chaining; `compute` runs it."""
    var table: SharedTable
    var nodes: List[PlanNode]
    var framed: Bool
    var overlay: Overlay
    var device: Optional[DeviceTable]

    def __init__(out self, table: SharedTable):
        self.table = table
        self.device = None
        self.nodes = [PlanNode(P_SCAN, -1, List[Expr](), List[Agg](), 0)]
        self.framed = False
        self.overlay = Overlay(table)

    def __init__(out self, frame: Frame) raises:
        self.table = frame.base
        self.device = None
        self.nodes = [PlanNode(P_SCAN, -1, List[Expr](), List[Agg](), 0)]
        self.framed = True
        self.overlay = frame.overlay()

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
        var at = self._aggregate()
        if at >= 0 and len(self.nodes[at].exprs) == 0:
            raise Error("group_by must come before agg")
        if at >= 0:
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

    def on_device(var self, device_table: DeviceTable) raises -> Self:
        """Attaches the device copy of this query's table, so `compute(GPU)` (and AUTO, on a big table) can use it.
        Explicit rather than a process-wide registry: Mojo has no safe mutable globals, the app owns the one
        DeviceTable next to its SharedTable and hands it to each session's queries, and a handle that was made
        from another table is refused here, not silently answered with the wrong data."""
        if not device_table.mirrors(self.table):
            raise Error("on_device: this DeviceTable was not made from the query's table")
        self.device = device_table
        return self^

    def compute(self, device: Int = AUTO) raises -> QueryResult:
        """Runs the plan. `device`: CPU, GPU (raises without a live attached DeviceTable) or AUTO (the GPU when a
        live DeviceTable is attached and the table has `AUTO_GPU_MIN_ROWS` rows or more, else the CPU). The result is
        the same either way, edits of a frame included."""
        var use_gpu = False
        if device == GPU:
            if not self.device:
                raise Error("device GPU: no DeviceTable is attached (Query.on_device(DeviceTable(table)))")
            if not self.device.value().is_live():
                raise Error("device GPU: the DeviceTable was released")
            use_gpu = True
        elif device == AUTO and self.device and self.device.value().is_live() and self.table.rows() >= AUTO_GPU_MIN_ROWS:
            use_gpu = True
        var t0 = perf_counter_ns()
        var nodes = self.nodes.copy()
        if self.framed:                                  # formulas inlined, merges remapped
            for i in range(len(nodes)):
                for k in range(len(nodes[i].exprs)):
                    nodes[i].exprs[k] = self.overlay.expand(nodes[i].exprs[k])
                for k in range(len(nodes[i].aggs)):
                    nodes[i].aggs[k].arg = self.overlay.expand(nodes[i].aggs[k].arg)
        var prog = _lower(self.table, nodes, self.framed, len(self.overlay.mask) > 0)
        var r = QueryResult()
        var done = False
        if use_gpu and self.table.rows() > 0:
            var g = _run_gpu(self.device.value(), self.table, prog, self.overlay.mask, device == GPU)
            if g:
                r = g.take()
                r.device = GPU
                done = True
        if not done:
            r = _run_cpu(self.table, prog, Int(self.overlay.mask.unsafe_ptr()) if len(self.overlay.mask) > 0 else 0)
        if self.framed and len(self.overlay.delta) > 0:
            _add_delta(r, self.table, nodes, self.overlay)
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
    var device: Int                 # CPU or GPU: where the kernel ran

    def __init__(out self):
        self.device = CPU
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
        elif n.op == N_BOOL2INT:                         # a comparison as the number 1 / 0 (merges)
            var a = stack.pop()
            if not a.boolean:
                raise Error("internal: int() of a number")
            var r = p.reg()
            p.emit(OP_F2I, r, a.reg)
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


def _lower(t: SharedTable, nodes: List[PlanNode], force_counts: Bool, masked: Bool) raises -> Program:
    """The plan as one kernel program. `force_counts` (a frame's query): every non-count aggregate carries its non-null
    count (an edit can null a cell of a column the shared data has no nulls in). `masked`: the predicate also skips
    the shared rows the session changed (the frame mask, filled in by `_run_cpu`)."""
    var p = Program(t.columns())
    var filters = List[Expr]()
    var agg_node = -1
    for i in range(len(nodes)):
        ref node = nodes[i]
        if node.kind == P_FILTER:
            filters.append(node.exprs[0].copy())
        elif node.kind == P_AGGREGATE:
            agg_node = i
        elif node.kind != P_SCAN:
            raise Error("plan node kind " + String(node.kind) + " is not supported yet")
    if agg_node < 0 or len(nodes[agg_node].aggs) == 0:
        raise Error("a query needs at least one aggregate (.agg(...))")
    ref an = nodes[agg_node]
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
    if masked:                                           # the frame's changed shared rows: the delta adds them
        var isnull = p.reg()
        p.emit(OP_ISNULL, isnull, _frame_mask(p))
        var keep = p.reg()
        p.emit(OP_NOT, keep, isnull)
        acc = _and_into(p, acc, keep)
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
        if valid < 0 and force_counts:
            valid = _const_reg(p, 1)
        if a.kind != A_COUNT:
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


def _frame_mask(mut p: Program) -> Int:
    """The program column whose OP_ISNULL reads the frame mask (bit cleared: a shared row the session changed)."""
    var at = len(p.cols)
    p.cols.append(-5_000_000)                      # the mask column (never read)
    p.dtypes.append(Int32(DT_NULLMASK | ((at + 1) << 8)))
    p.cols.append(-4_000_000)                      # the mask bitmap's address (`_run_cpu`)
    p.dtypes.append(DT_I64)
    p.cols.append(-1_000_000)                      # bit base 0
    p.dtypes.append(DT_I64)
    return at


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


def _run_cpu(t: SharedTable, p: Program, mask_addr: Int) raises -> QueryResult:
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
        elif c == -1_000_000 or c == -5_000_000:
            tbl.append(0)
        elif c == -4_000_000:
            tbl.append(Int64(mask_addr))
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
    var r = _make_result(p, out_vals, out_cnt)
    _ = out_vals^
    return r^


def _make_result(p: Program, out_vals: List[Float64], out_cnt: List[Int64]) -> QueryResult:
    """The kernel's output (slot words are int64 bits, per slot then group; counts per group) as a QueryResult."""
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
    return r^


# ── running on the GPU ───────────────────────────────────────────────────────────────────────────────

def _reg_fields(op: Int) -> Int:
    """Which of an instruction's operands d, a, b name registers: bit 0 = d, bit 1 = a, bit 2 = b."""
    if op == OP_LOAD_CONST or op == OP_LOADI or op == OP_ISNULL:
        return 1
    if op == OP_F2I or op == OP_NOT:
        return 3
    if op >= OP_AGG_ISUM:
        return 2                                          # an aggregate: d is the slot, a the value register
    return 7


def _compact_registers(pred_reg: Int, mut code: List[Int32]) -> Tuple[Int, Int]:
    """The program uses registers 0 .. 47 (kept ones from the bottom, temporaries from the top). A GPU thread block
    holds its registers in shared memory (256 rows x 8 bytes each), so renumber the registers actually used densely
    in `code` (all instructions, predicate then body then aggregates). Returns (registers used, new predicate register)."""
    var used = List[Bool](length=MAX_REGS, fill=False)
    used[0] = True
    if pred_reg >= 0:
        used[pred_reg] = True
    for i in range(len(code) // 5):
        var f = _reg_fields(Int(code[i * 5]))
        for k in range(3):
            if (f >> k) & 1 != 0:
                used[Int(code[i * 5 + 1 + k])] = True
    var map = List[Int32](length=MAX_REGS, fill=0)
    var n = 0
    for r in range(MAX_REGS):
        if used[r]:
            map[r] = Int32(n)
            n += 1
    for i in range(len(code) // 5):
        var f = _reg_fields(Int(code[i * 5]))
        for k in range(3):
            if (f >> k) & 1 != 0:
                code[i * 5 + 1 + k] = map[Int(code[i * 5 + 1 + k])]
    return (n, Int(map[pred_reg]) if pred_reg >= 0 else -1)


def _run_gpu(dt: DeviceTable, t: SharedTable, p: Program, mask: List[UInt8], strict: Bool) raises -> Optional[QueryResult]:
    """The lowered program on the device table (kernels/pipeline_gpu.mojo, keyed mode: the group id is register 0).
    Same answers as `_run_cpu`: exact int64 totals, the overflow flag raises the same error. `strict` (device=GPU):
    a program the kernel cannot take raises; otherwise (AUTO) it returns None and the CPU answers."""
    var ctx = dt.context()
    var ins = p.pred.copy()
    ins.extend(p.body.copy())
    ins.extend(p.aggs.copy())
    var compact = _compact_registers(p.pred_reg, ins)
    var n_regs = compact[0]
    var code = List[Int32]()
    var header: List[Int] = [MAGIC, n_regs, 0, len(p.pred) // 5, len(p.body) // 5, len(p.aggs) // 5,
                             compact[1], len(p.cols), 0, 0, 0, 0, 0, 0, 0, 0]
    for h in header:
        code.append(Int32(h))
    code.extend(ins^)
    var consts = List[Float64](length=MAX_CONSTS, fill=0.0)
    for k in range(len(p.consts)):                       # int64 bits in the float64 pool, copied exactly
        consts[k] = Float64(from_bits=UInt64(p.consts[k]))
    var dmask = ctx.enqueue_create_buffer[DType.uint8](max(len(mask), 1))
    if len(mask) > 0:
        ctx.enqueue_copy(dst_buf=dmask, src_ptr=mask.unsafe_ptr())
    var cols = List[Int64]()
    for c in p.cols:
        if c >= 0:
            cols.append(Int64(dt.data_address(c)))
        elif c == -1_000_000 or c == -5_000_000:
            cols.append(0)
        elif c == -4_000_000:
            cols.append(Int64(Int(dmask.unsafe_ptr())))
        else:
            cols.append(Int64(dt.validity_address(-1 - c)))
    if len(cols) == 0:
        cols.append(0)
    var dtypes = p.dtypes.copy()
    if len(dtypes) == 0:
        dtypes.append(DT_I64)
    var A = p.n_slots
    var G = p.groups
    var ident = List[Float64](length=A, fill=0.0)
    for a in range(len(p.aggs) // 5):
        var op = Int(p.aggs[a * 5])
        var slot = Int(p.aggs[a * 5 + 1])
        if op == OP_AGG_IMIN:
            ident[slot] = Float64(from_bits=UInt64(Int64.MAX))
        elif op == OP_AGG_IMAX:
            ident[slot] = Float64(from_bits=UInt64(Int64.MIN))
    var n_blocks = ctx.default_device_info.sm_count
    var scratch_words = n_blocks * 8 * (A + 1) * G       # at most sm_count x 8 blocks, one row of (slots + 1) x groups each
    var d_code = ctx.enqueue_create_buffer[DType.int32](len(code))
    var d_consts = ctx.enqueue_create_buffer[DType.float64](MAX_CONSTS)
    var d_cols = ctx.enqueue_create_buffer[DType.int64](len(cols))
    var d_dt = ctx.enqueue_create_buffer[DType.int32](len(dtypes))
    var d_ident = ctx.enqueue_create_buffer[DType.float64](A)
    var d_set = ctx.enqueue_create_buffer[DType.int64](4)        # no string sets: a dummy
    var d_vals = ctx.enqueue_create_buffer[DType.float64](A * G)
    var d_cnt = ctx.enqueue_create_buffer[DType.int64](G)
    var d_scratch = ctx.enqueue_create_buffer[DType.float64](scratch_words)
    var d_ovf = ctx.enqueue_create_buffer[DType.int64](1)
    ctx.enqueue_copy(dst_buf=d_code, src_ptr=code.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_consts, src_ptr=consts.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_cols, src_ptr=cols.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_dt, src_ptr=dtypes.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ident, src_ptr=ident.unsafe_ptr())
    ctx.enqueue_memset(d_set, 0)
    ctx.enqueue_memset(d_ovf, 0)
    var agg_base = (len(p.pred) + len(p.body)) // 5
    var rc = run_pipeline_gpu(ctx, Int(d_code.unsafe_ptr()), Int(d_consts.unsafe_ptr()), Int(d_set.unsafe_ptr()),
                              Int(d_set.unsafe_ptr()), Int(d_cols.unsafe_ptr()), Int(d_dt.unsafe_ptr()),
                              -1 if p.keyed else 0, G, A, 0, t.rows(), Int(d_vals.unsafe_ptr()),
                              Int(d_cnt.unsafe_ptr()), Int(d_ident.unsafe_ptr()), Int(d_scratch.unsafe_ptr()),
                              n_regs, len(code), agg_base, n_blocks, 1, 0, Int(d_ovf.unsafe_ptr()), True)
    if rc != 0:
        if not strict and (rc == -5 or rc == -6):
            return None                                   # more shared memory than a block has: the CPU answers
        raise Error("the GPU kernel failed with code " + String(rc) +
                    (" (the query needs more GPU shared memory than a block has: " + String(n_regs) + " registers, " + String(A) + " slots, " + String(G) + " groups)"
                     if rc == -5 else ""))
    var out_vals = List[Float64](length=A * G, fill=0.0)
    var out_cnt = List[Int64](length=G, fill=0)
    var ovf = List[Int64](length=1, fill=0)
    ctx.enqueue_copy(dst_ptr=out_vals.unsafe_ptr(), src_buf=d_vals)
    ctx.enqueue_copy(dst_ptr=out_cnt.unsafe_ptr(), src_buf=d_cnt)
    ctx.enqueue_copy(dst_ptr=ovf.unsafe_ptr(), src_buf=d_ovf)
    ctx.synchronize()
    _ = code^, consts^, cols^, dtypes^, ident^
    if ovf[0] != 0:
        raise Error("a sum or an expression passed the int64 range (about 9.2e18)")
    return _make_result(p, out_vals, out_cnt)


# ── a frame's changed and appended rows, on the CPU ──────────────────────────────────────────────────

def _add_delta(mut r: QueryResult, t: SharedTable, nodes: List[PlanNode], o: Overlay) raises:
    """Adds the session's changed (live) and appended rows to the kernel's result over the unchanged shared rows,
    with the kernel's rules: a filter reading a null is false, a null or out-of-range key leaves the row out, an
    aggregate skips a null value, totals are exact (an overflow raises)."""
    var filters = List[Expr]()
    var keyed = False
    var key = Expr(0)
    var groups = 1
    var aggs = List[Agg]()
    for node in nodes:
        if node.kind == P_FILTER:
            filters.append(node.exprs[0].copy())
        elif node.kind == P_AGGREGATE:
            groups = node.groups
            aggs = node.aggs.copy()
            if len(node.exprs) > 0:
                keyed = True
                key = node.exprs[0].copy()
    for raw in o.delta:
        var keep = True
        for f in filters:
            var v = _eval_row(f, t, raw)
            if v[1] or v[0] == 0:
                keep = False
                break
        if not keep:
            continue
        var g = 0
        if keyed:
            var k = _eval_row(key, t, raw)
            if k[1] or k[0] < 0 or k[0] >= Int64(groups):
                continue
            g = Int(k[0])
        r.counts[g] += 1
        for i in range(len(aggs)):
            if aggs[i].kind == A_COUNT:
                continue
            var v = _eval_row(aggs[i].arg, t, raw)
            if v[1]:
                continue
            if r.count_slot_of[i] >= 0:
                r.slots[r.count_slot_of[i]][g] += 1
            if aggs[i].kind == A_COUNT_OF:
                continue
            var s = r.slot_of[i]
            var x = Int64(r.slots[s][g])
            if aggs[i].kind == A_MIN:
                r.slots[s][g] = Int(min(x, v[0]))
            elif aggs[i].kind == A_MAX:
                r.slots[s][g] = Int(max(x, v[0]))
            else:
                var y = x + v[0]
                if ((x ^ y) & (v[0] ^ y)) < 0:
                    raise Error("a sum or an expression passed the int64 range (about 9.2e18)")
                r.slots[s][g] = Int(y)
    for i in range(len(r.kinds)):
        if r.kinds[i] == A_SUM:
            r.sums = r.slots[r.slot_of[i]].copy()
            break
