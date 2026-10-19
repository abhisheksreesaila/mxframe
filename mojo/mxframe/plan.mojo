"""The logical plan (F-152): a query as a tree of operators, between the binder and the planner.

    SQL text --parse--> SqlStatement --bind--> BoundQuery (typed expressions) --build_plan--> LogicalPlan --plan--> PhysicalPlan --run

The binder says what the names and types are; this module says what the query *does*, as a list of `PlanNode`s from the
scan up to the last operator (a node's `input` is the node below it):

    Scan -> Filter (WHERE) -> [Aggregate] -> [Filter (HAVING)] -> Project -> [Sort] -> [Limit]

Everything after the Scan reads the node below. `SELECT DISTINCT` is an Aggregate with no aggregates (the rewrite is made by
the binder, which turns the select list into group keys; the node remembers `distinct` so EXPLAIN says so). Rewrite rules
work on the list: `rewrite` marks a Sort under a Limit as a top-k, and a Limit alone as one that can stop the scan early.
The planner (`planner.mojo`) reads the plan and the statistics and chooses the physical strategy; the executor
(`sqlexec.mojo`) runs what the physical plan says. A second table (F-153) is a second Scan and a Join node.

Expressions are not copied into the nodes: a node holds the index of its expression in the BoundQuery (`rexprs` for row-level
ones, `pexprs` after aggregation) and the text EXPLAIN shows.
"""
from .bind import BoundQuery, SortKey

comptime P_SCAN = 0
comptime P_FILTER = 1
comptime P_AGGREGATE = 2
comptime P_PROJECT = 3
comptime P_SORT = 4
comptime P_LIMIT = 5


struct PlanNode(Copyable, Movable):
    var kind: Int
    var input: Int                   # the node below (-1 for a scan)
    var source: Int                  # Scan: the catalog's source
    var expr: Int                    # Filter: the predicate; `post` says which pool it is in
    var post: Bool                   # Filter: HAVING (an expression after aggregation) rather than WHERE
    var keys: List[Int]              # Aggregate: indices into the query's group keys; Project: the output expressions
    var names: List[String]          # Project: the output column names
    var sort: List[SortKey]          # Sort: the keys (expressions in the pool of the query's mode)
    var limit: Int                   # Limit: rows to return (-1 all); Sort: the rows wanted from it (-1 all) once a Limit sits above
    var offset: Int
    var distinct: Bool               # Aggregate: made from SELECT DISTINCT
    var groups: Int                  # estimated groups (0 = unknown)
    var text: String                 # the node as EXPLAIN shows it

    def __init__(out self, kind: Int, input: Int):
        self.kind = kind
        self.input = input
        self.source = -1
        self.expr = -1
        self.post = False
        self.keys = List[Int]()
        self.names = List[String]()
        self.sort = List[SortKey]()
        self.limit = -1
        self.offset = 0
        self.distinct = False
        self.groups = 0
        self.text = String()


struct LogicalPlan(Movable):
    var nodes: List[PlanNode]
    var rows: Bool                   # a row query: no Aggregate, the Project reads the source's columns (rexprs)
    var top_k: Bool                  # rewrite: a Sort under a Limit keeps only the first limit + offset rows
    var early_stop: Bool             # rewrite: a Limit with no Sort or Aggregate below it can stop the scan once it has its rows

    def __init__(out self):
        self.nodes = List[PlanNode]()
        self.rows = False
        self.top_k = False
        self.early_stop = False

    def add(mut self, var n: PlanNode) -> Int:
        self.nodes.append(n^)
        return len(self.nodes) - 1

    def root(self) -> Int:
        return len(self.nodes) - 1

    def find(self, kind: Int) -> Int:
        """The index of the node of this kind (the highest one), -1 when there is none."""
        var i = len(self.nodes) - 1
        while i >= 0:
            if self.nodes[i].kind == kind:
                return i
            i -= 1
        return -1

    def find_filter(self, post: Bool) -> Int:
        for i in range(len(self.nodes)):
            if self.nodes[i].kind == P_FILTER and self.nodes[i].post == post:
                return i
        return -1

    def has(self, kind: Int) -> Bool:
        return self.find(kind) >= 0

    def lines(self) -> List[String]:
        """The plan as EXPLAIN prints it: the last operator first, the scan last, one line each."""
        var out = List[String]()
        var i = self.root()
        while i >= 0:
            out.append(self.nodes[i].text)
            i = self.nodes[i].input
        return out^


def build_plan(q: BoundQuery, source_name: String) -> LogicalPlan:
    """The logical plan of a bound query, rewritten."""
    var lp = LogicalPlan()
    lp.rows = q.rows
    var scan = PlanNode(P_SCAN, -1)
    scan.source = q.source
    scan.text = "Scan " + source_name
    var at = lp.add(scan^)
    if q.where >= 0:
        var f = PlanNode(P_FILTER, at)
        f.expr = q.where
        f.text = "Filter (WHERE) " + q.where_text
        at = lp.add(f^)
    if not q.rows:
        var agg = PlanNode(P_AGGREGATE, at)
        for k in range(len(q.keys)):
            agg.keys.append(k)
        agg.distinct = q.distinct
        var s = String("Distinct" if q.distinct else "Aggregate")
        if len(q.keys) > 0:
            s += " keys=["
            for k in range(len(q.keys)):
                s += (", " if k > 0 else "") + q.keys[k].name
            s += "]"
        if not q.distinct:
            s += " aggs=["
            for k in range(len(q.aggs)):
                s += (", " if k > 0 else "") + q.aggs[k].name
            s += "]"
        agg.text = s
        at = lp.add(agg^)
        if q.having >= 0:
            var h = PlanNode(P_FILTER, at)
            h.expr = q.having
            h.post = True
            h.text = "Filter (HAVING) " + q.having_text
            at = lp.add(h^)
    var proj = PlanNode(P_PROJECT, at)
    var ptext = String("Project ")
    for k in range(len(q.out_text)):
        ptext += (", " if k > 0 else "") + q.out_text[k]
    proj.text = ptext
    proj.names = q.out_names.copy()
    proj.keys = (q.routs.copy() if q.rows else q.outputs.copy())
    at = lp.add(proj^)
    if len(q.order) > 0 or len(q.rorder) > 0:
        var srt = PlanNode(P_SORT, at)
        srt.sort = (q.rorder.copy() if q.rows else q.order.copy())
        var s = String("Sort ")
        for k in range(len(q.order_text)):
            s += (", " if k > 0 else "") + q.order_text[k]
        srt.text = s
        at = lp.add(srt^)
    if q.limit >= 0 or q.offset > 0:
        var lim = PlanNode(P_LIMIT, at)
        lim.limit = q.limit
        lim.offset = q.offset
        lim.text = "Limit " + (String(q.limit) if q.limit >= 0 else "all") + (" offset " + String(q.offset) if q.offset > 0 else "")
        at = lp.add(lim^)
    rewrite(lp)
    return lp^


def rewrite(mut lp: LogicalPlan):
    """The rules that look at the shape of the plan.
    - top-k: a Sort directly under a Limit only needs its first limit + offset rows (no full sort);
    - early stop: a Limit directly over a Project of a row query (no Sort, no Aggregate) can stop reading once it has its rows."""
    var lim = lp.find(P_LIMIT)
    if lim < 0 or lp.nodes[lim].limit < 0:
        return
    var below = lp.nodes[lim].input
    var want = lp.nodes[lim].limit + lp.nodes[lim].offset
    if lp.nodes[below].kind == P_SORT:
        lp.nodes[below].limit = want
        lp.top_k = True
    elif lp.nodes[below].kind == P_PROJECT and lp.rows:
        lp.early_stop = True
