"""The logical plan (F-152): a query as a tree of operators, between the binder and the planner.

    SQL text --parse--> SqlStatement --bind--> BoundQuery (typed expressions) --build_plan--> LogicalPlan --plan--> PhysicalPlan --run

The binder says what the names and types are; this module says what the query *does*, as a list of `PlanNode`s from the
scan up to the last operator (a node's `input` is the node below it):

    Scan -> Filter (WHERE) -> [Aggregate] -> [Filter (HAVING)] -> Project -> [Sort] -> [Limit]

and, for two tables (F-153), a tree: each table is its own Scan (with the filter that reads only it), a Join takes the two
(`input` is the first table's subtree, `right` the second's), and what the join leaves for both tables (WHERE conjuncts that read
columns of each) is a Filter above it:

    Scan a -> [Filter] --\
                          Join -> [Filter] -> [Aggregate] -> ...
    Scan b -> [Filter] --/

Everything after the Scan reads the node below. `SELECT DISTINCT` is an Aggregate with no aggregates (the rewrite is made by
the binder, which turns the select list into group keys; the node remembers `distinct` so EXPLAIN says so). Rewrite rules
work on the list: `rewrite` marks a Sort under a Limit as a top-k, and a Limit alone as one that can stop the scan early.
The planner (`planner.mojo`) reads the plan and the statistics and chooses the physical strategy; the executor
(`sqlexec.mojo`) runs what the physical plan says. A second table (F-153) is a second Scan and a Join node.

Expressions are not copied into the nodes: a node holds the index of its expression in the BoundQuery (`rexprs` for row-level
ones, `pexprs` after aggregation) and the text EXPLAIN shows.
"""
from .bind import BoundQuery, SortKey, KeySpec, AggSpec, copy_query

comptime P_SCAN = 0
comptime P_FILTER = 1
comptime P_AGGREGATE = 2
comptime P_PROJECT = 3
comptime P_SORT = 4
comptime P_LIMIT = 5
comptime P_JOIN = 6


struct PlanNode(Copyable, Movable):
    var kind: Int
    var input: Int                   # the node below (-1 for a scan); a Join: its first table's subtree
    var right: Int                   # a Join: its second table's subtree, else -1
    var source: Int                  # Scan: the catalog's source
    var expr: Int                    # Filter: the predicate; `post` says which pool it is in
    var post: Bool                   # Filter: HAVING (an expression after aggregation) rather than WHERE
    var keys: List[Int]              # Aggregate: indices into the query's group keys; Project: the output expressions; Join: the first table's key columns
    var keys2: List[Int]             # Join: the second table's key columns (equal to `keys`, pairwise)
    var aggs: List[Int]              # Aggregate: indices into the query's aggregates
    var conj: List[Int]              # a Filter that reads one table of a join: its conjuncts (row-level expressions)
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
        self.right = -1
        self.keys2 = List[Int]()
        self.aggs = List[Int]()
        self.conj = List[Int]()
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

    def tree(self) -> List[Tuple[Int, String]]:
        """The plan as EXPLAIN prints it: (depth, text) from the last operator down, a Join's two subtrees one after the other."""
        var out = List[Tuple[Int, String]]()
        self._walk(self.root(), 0, out)
        return out^

    def _walk(self, i: Int, depth: Int, mut out: List[Tuple[Int, String]]):
        if i < 0:
            return
        out.append((depth, self.nodes[i].text))
        self._walk(self.nodes[i].input, depth + 1, out)
        if self.nodes[i].right >= 0:
            self._walk(self.nodes[i].right, depth + 1, out)

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
    var at: Int
    if q.joined:
        var sides = List[Int]()
        for t in range(2):
            var sc = PlanNode(P_SCAN, -1)
            sc.source = q.sources[t]
            sc.text = "Scan " + q.tab_names[t] + (" AS " + q.tab_alias[t] if q.tab_alias[t].byte_length() > 0 else "")
            var node = lp.add(sc^)
            var conj = q.conj_l.copy() if t == 0 else q.conj_r.copy()
            var texts = q.conj_l_text.copy() if t == 0 else q.conj_r_text.copy()
            if len(conj) > 0:
                var f = PlanNode(P_FILTER, node)
                f.conj = conj^
                var ft = String("Filter ")
                for k in range(len(texts)):
                    ft += (" AND " if k > 0 else "") + texts[k]
                f.text = ft
                node = lp.add(f^)
            sides.append(node)
        var jn = PlanNode(P_JOIN, sides[0])
        jn.right = sides[1]
        jn.keys = q.jk_l.copy()
        jn.keys2 = q.jk_r.copy()
        var jt = String("Join (inner) on ")
        var lab0 = q.tab_alias[0] if q.tab_alias[0].byte_length() > 0 else q.tab_names[0]
        var lab1 = q.tab_alias[1] if q.tab_alias[1].byte_length() > 0 else q.tab_names[1]
        for k in range(len(q.jk_l)):
            jt += (" AND " if k > 0 else "") + lab0 + "." + q.jk_l_names[k] + " = " + lab1 + "." + q.jk_r_names[k]
        jn.text = jt
        at = lp.add(jn^)
    else:
        var scan = PlanNode(P_SCAN, -1)
        scan.source = q.source
        scan.text = "Scan " + source_name
        at = lp.add(scan^)
    if q.where >= 0:
        var f = PlanNode(P_FILTER, at)
        f.expr = q.where
        f.text = "Filter (WHERE) " + q.where_text if not q.joined else "Filter " + q.where_text
        at = lp.add(f^)
    if not q.rows:
        var agg = PlanNode(P_AGGREGATE, at)
        for k in range(len(q.keys)):
            agg.keys.append(k)
        for k in range(len(q.aggs)):
            agg.aggs.append(k)
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


def executable(lp: LogicalPlan, q: BoundQuery) -> BoundQuery:
    """The query the executor runs, lowered by walking the plan from its last node down: the Limit gives LIMIT / OFFSET, the Sort the
    ORDER BY keys, the Project the select list and its names, the HAVING Filter the HAVING, the Aggregate its keys and aggregates, a WHERE
    Filter the WHERE, the Scan the source, and a Join its two Scans (with the filter that reads only each), its keys and the filter it
    leaves above it. The binder's own copies of those are not read: the nodes are the truth the rewrite rules may change. The expression
    pools (`rexprs`, `pexprs`) stay the binder's: nodes hold indices into them."""
    var e = copy_query(q)
    e.where = -1
    e.having = -1
    e.limit = -1
    e.offset = 0
    e.keys = List[KeySpec]()
    e.aggs = List[AggSpec]()
    e.order = List[SortKey]()
    e.rorder = List[SortKey]()
    e.outputs = List[Int]()
    e.routs = List[Int]()
    e.conj_l = List[Int]()
    e.conj_r = List[Int]()
    e.jk_l = List[Int]()
    e.jk_r = List[Int]()
    e.sources = List[Int]()
    var i = lp.root()
    while i >= 0:
        ref n = lp.nodes[i]
        if n.kind == P_LIMIT:
            e.limit = n.limit
            e.offset = n.offset
        elif n.kind == P_SORT:
            if lp.rows:
                e.rorder = n.sort.copy()
            else:
                e.order = n.sort.copy()
        elif n.kind == P_PROJECT:
            if lp.rows:
                e.routs = n.keys.copy()
            else:
                e.outputs = n.keys.copy()
            e.out_names = n.names.copy()
        elif n.kind == P_FILTER and n.post:
            e.having = n.expr
        elif n.kind == P_AGGREGATE:
            for k in n.keys:
                e.keys.append(q.keys[k].copy())
            for a in n.aggs:
                e.aggs.append(q.aggs[a].copy())
        elif n.kind == P_FILTER and n.expr >= 0:
            e.where = n.expr
        elif n.kind == P_JOIN:
            e.jk_l = n.keys.copy()
            e.jk_r = n.keys2.copy()
            var l = n.input
            var r = n.right
            if lp.nodes[l].kind == P_FILTER:
                e.conj_l = lp.nodes[l].conj.copy()
                l = lp.nodes[l].input
            if lp.nodes[r].kind == P_FILTER:
                e.conj_r = lp.nodes[r].conj.copy()
                r = lp.nodes[r].input
            e.sources.append(lp.nodes[l].source)
            e.sources.append(lp.nodes[r].source)
            e.source = e.sources[0]
            break
        elif n.kind == P_SCAN:
            e.sources.append(n.source)
            e.source = n.source
        i = n.input
    return e^
