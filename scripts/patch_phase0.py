"""Phase 0 notebook patcher — run once then delete."""
import json

# ════════════════════════════════════════════════════════════════════
# 03_compiler.ipynb
# ════════════════════════════════════════════════════════════════════

NEW_COMPILER_CLASS = (
    "#| export\n"
    "# ── dtype helpers ────────────────────────────────────────────────────────\n"
    "\n"
    "_NP_TO_MAX = {\n"
    '    np.dtype("int32"):   DType.int32,\n'
    '    np.dtype("int64"):   DType.int64,\n'
    '    np.dtype("float32"): DType.float32,\n'
    '    np.dtype("float64"): DType.float64,\n'
    '    np.dtype("bool"):    DType.bool,\n'
    "}\n"
    "\n"
    "def _max_dtype(arr: np.ndarray) -> DType:\n"
    '    """Map a numpy dtype to the corresponding MAX DType."""\n'
    "    return _NP_TO_MAX.get(arr.dtype, DType.float32)\n"
    "\n"
    "# ── PyArrow compute ops used in predicate evaluation ─────────────────────\n"
    "_PC_CMP_OPS = {\n"
    '    "gt":  pc.greater,\n'
    '    "ge":  pc.greater_equal,\n'
    '    "lt":  pc.less,\n'
    '    "le":  pc.less_equal,\n'
    '    "eq":  pc.equal,\n'
    '    "and": pc.and_,\n'
    '    "or":  pc.or_,\n'
    "}\n"
    "\n"
    "# ── MAX Graph comparison ops ──────────────────────────────────────────────\n"
    "_CMP_OPS = {\n"
    '    "gt": ops.greater,\n'
    '    "ge": ops.greater_equal,\n'
    '    "lt": lambda a, b: ops.greater(b, a),\n'
    '    "le": lambda a, b: ops.greater_equal(b, a),\n'
    '    "eq": ops.equal,\n'
    "}\n"
    "\n"
    "\n"
    "class GraphCompiler:\n"
    '    """Compiles a LogicalPlan into a MAX Graph.\n'
    "\n"
    "    Key design:\n"
    "    - All Filter nodes are evaluated eagerly in PyArrow *before* the graph is built\n"
    "      (_strip_filters). This correctly removes rows and makes count() trivially correct.\n"
    "    - Supports: Scan, Project, Filter (pre-applied), Aggregate (global sum/min/max/mean/count).\n"
    '    """\n'
    "\n"
    "    def __init__(self):\n"
    "        self.session = engine.InferenceSession(devices=[driver.CPU()])\n"
    "\n"
    "    # ── public API ───────────────────────────────────────────────────────\n"
    "\n"
    "    def compile_and_run(self, plan: LogicalPlan) -> pa.Table:\n"
    '        """Apply filters eagerly in PyArrow, then compile the rest to a MAX Graph."""\n'
    "        plan = self._strip_filters(plan)\n"
    "\n"
    "        scan = self._find_scan(plan)\n"
    "        col_names = scan.table.column_names\n"
    "        col_arrays = {name: np.array(scan.table[name]) for name in col_names}\n"
    "\n"
    "        input_types = [\n"
    "            TensorType(_max_dtype(col_arrays[n]), list(col_arrays[n].shape), DeviceRef.CPU())\n"
    "            for n in col_names\n"
    "        ]\n"
    "\n"
    '        graph = Graph(name="mxframe_query", input_types=input_types)\n'
    "        with graph:\n"
    "            input_nodes = {name: graph.inputs[i] for i, name in enumerate(col_names)}\n"
    "            result_nodes = self._visit_plan(plan, input_nodes)\n"
    "            graph.output(*result_nodes.values())\n"
    "\n"
    "        model = self.session.load(graph)\n"
    "        output_vals = model.execute(*[col_arrays[n] for n in col_names])\n"
    "        result_names = list(result_nodes.keys())\n"
    "        arrays = [pa.array(t.to_numpy()) for t in output_vals]\n"
    "        return pa.Table.from_arrays(arrays, names=result_names)\n"
    "\n"
    "    # ── filter pre-processing (PyArrow, before graph build) ──────────────\n"
    "\n"
    "    @staticmethod\n"
    "    def _eval_predicate(expr: Expr, table: pa.Table):\n"
    '        """Evaluate a filter predicate against a PyArrow table.\n'
    "        Returns a PyArrow BooleanArray. Supports: col, lit, gt, ge, lt, le, eq, and, or.\n"
    '        """\n'
    "        op, args = expr.op, expr.args\n"
    '        if op == "col":\n'
    "            arr = table.column(args[0])\n"
    "            return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr\n"
    '        if op == "lit":\n'
    "            return args[0]\n"
    "        if op in _PC_CMP_OPS:\n"
    "            lhs = GraphCompiler._eval_predicate(args[0], table)\n"
    "            rhs = GraphCompiler._eval_predicate(args[1], table)\n"
    "            return _PC_CMP_OPS[op](lhs, rhs)\n"
    "        raise NotImplementedError(\n"
    "            f\"Cannot evaluate predicate op '{op}' in PyArrow. \"\n"
    "            f\"Add it to _PC_CMP_OPS or handle it here.\"\n"
    "        )\n"
    "\n"
    "    @classmethod\n"
    "    def _strip_filters(cls, plan: LogicalPlan) -> LogicalPlan:\n"
    '        """Recursively remove all Filter nodes, applying them eagerly to the Scan."""\n'
    "        if isinstance(plan, Scan):\n"
    "            return plan\n"
    "        if isinstance(plan, Filter):\n"
    "            clean_inner = cls._strip_filters(plan.input)\n"
    "            scan = cls._find_scan_static(clean_inner)\n"
    "            mask = cls._eval_predicate(plan.predicate, scan.table)\n"
    "            filtered_scan = Scan(scan.table.filter(mask))\n"
    "            return cls._replace_scan(clean_inner, filtered_scan)\n"
    "        if isinstance(plan, Project):\n"
    "            return Project(cls._strip_filters(plan.input), plan.exprs)\n"
    "        if isinstance(plan, Aggregate):\n"
    "            return Aggregate(cls._strip_filters(plan.input), plan.group_by, plan.aggs)\n"
    "        if hasattr(plan, 'input'):\n"
    "            plan.input = cls._strip_filters(plan.input)  # type: ignore\n"
    "        return plan\n"
    "\n"
    "    @classmethod\n"
    "    def _replace_scan(cls, plan: LogicalPlan, new_scan: 'Scan') -> LogicalPlan:\n"
    '        """Replace the Scan leaf in a plan tree."""\n'
    "        if isinstance(plan, Scan):\n"
    "            return new_scan\n"
    "        if isinstance(plan, Project):\n"
    "            return Project(cls._replace_scan(plan.input, new_scan), plan.exprs)\n"
    "        if isinstance(plan, Aggregate):\n"
    "            return Aggregate(cls._replace_scan(plan.input, new_scan), plan.group_by, plan.aggs)\n"
    "        if hasattr(plan, 'input'):\n"
    "            plan.input = cls._replace_scan(plan.input, new_scan)  # type: ignore\n"
    "        return plan\n"
    "\n"
    "    @staticmethod\n"
    "    def _find_scan_static(plan: LogicalPlan) -> 'Scan':\n"
    "        if isinstance(plan, Scan): return plan\n"
    "        if hasattr(plan, 'input'): return GraphCompiler._find_scan_static(plan.input)\n"
    "        raise ValueError(f'No Scan found in plan: {type(plan)}')\n"
    "\n"
    "    # ── plan traversal ───────────────────────────────────────────────────\n"
    "\n"
    "    def _find_scan(self, plan: LogicalPlan) -> 'Scan':\n"
    "        return self._find_scan_static(plan)\n"
    "\n"
    "    def _visit_plan(self, plan: LogicalPlan, nodes: Dict[str, Any]) -> Dict[str, Any]:\n"
    "        if isinstance(plan, Scan):\n"
    "            return nodes\n"
    "        elif isinstance(plan, Project):\n"
    "            return self._visit_project(plan, nodes)\n"
    "        elif isinstance(plan, Filter):\n"
    "            return self._visit_filter(plan, nodes)\n"
    "        elif isinstance(plan, Aggregate):\n"
    "            return self._visit_aggregate(plan, nodes)\n"
    "        raise NotImplementedError(f'Unsupported plan node: {type(plan)}')\n"
    "\n"
    "    # ── plan node visitors ───────────────────────────────────────────────\n"
    "\n"
    "    def _visit_project(self, plan: 'Project', nodes: Dict[str, Any]) -> Dict[str, Any]:\n"
    "        upstream = self._visit_plan(plan.input, nodes)\n"
    "        out = {}\n"
    "        for i, expr in enumerate(plan.exprs):\n"
    "            name = expr._alias or f'col_{i}'\n"
    "            out[name] = self._visit_expr(expr, upstream)\n"
    "        return out\n"
    "\n"
    "    def _visit_filter(self, plan: 'Filter', nodes: Dict[str, Any]) -> Dict[str, Any]:\n"
    '        """Safety fallback only — _strip_filters should have removed all Filter nodes."""\n'
    "        upstream = self._visit_plan(plan.input, nodes)\n"
    "        mask = self._visit_expr(plan.predicate, upstream)\n"
    "        filtered = {}\n"
    "        for name, node in upstream.items():\n"
    "            filtered[name] = ops.mul(node, ops.cast(mask, node.type.dtype))\n"
    "        return filtered\n"
    "\n"
    "    def _visit_aggregate(self, plan: 'Aggregate', nodes: Dict[str, Any]) -> Dict[str, Any]:\n"
    '        """Global (non-grouped) aggregation via built-in MAX ops."""\n'
    "        upstream = self._visit_plan(plan.input, nodes)\n"
    "        out = {}\n"
    "        for i, expr in enumerate(plan.aggs):\n"
    "            name = expr._alias or f'agg_{i}'\n"
    "            out[name] = self._visit_expr(expr, upstream)\n"
    "        return out\n"
    "\n"
    "    # ── expression visitor ───────────────────────────────────────────────\n"
    "\n"
    "    def _visit_expr(self, expr: Expr, nodes: Dict[str, Any]) -> Any:\n"
    '        """Translate an Expr tree into MAX graph ops."""\n'
    "        op = expr.op\n"
    "        args = expr.args\n"
    "\n"
    '        if op == "col":\n'
    "            return nodes[args[0]]\n"
    '        if op == "lit":\n'
    "            val = args[0]\n"
    "            if isinstance(val, int):\n"
    "                return ops.constant(val, dtype=DType.int64, device=DeviceRef.CPU())\n"
    "            elif isinstance(val, float):\n"
    "                return ops.constant(val, dtype=DType.float64, device=DeviceRef.CPU())\n"
    "            return ops.constant(val, dtype=DType.float32, device=DeviceRef.CPU())\n"
    "\n"
    '        if op == "add":\n'
    "            return ops.add(self._visit_expr(args[0], nodes), self._visit_expr(args[1], nodes))\n"
    '        if op == "sub":\n'
    "            return ops.sub(self._visit_expr(args[0], nodes), self._visit_expr(args[1], nodes))\n"
    '        if op == "mul":\n'
    "            return ops.mul(self._visit_expr(args[0], nodes), self._visit_expr(args[1], nodes))\n"
    '        if op == "div":\n'
    "            return ops.div(self._visit_expr(args[0], nodes), self._visit_expr(args[1], nodes))\n"
    "\n"
    "        if op in _CMP_OPS:\n"
    "            return _CMP_OPS[op](self._visit_expr(args[0], nodes), self._visit_expr(args[1], nodes))\n"
    "\n"
    '        if op == "sum":\n'
    "            return ops.sum(self._visit_expr(args[0], nodes), axis=0)\n"
    '        if op == "min":\n'
    "            return ops.min(self._visit_expr(args[0], nodes), axis=0)\n"
    '        if op == "max":\n'
    "            return ops.max(self._visit_expr(args[0], nodes), axis=0)\n"
    '        if op == "mean":\n'
    "            return ops.mean(self._visit_expr(args[0], nodes), axis=0)\n"
    '        if op == "count":\n'
    "            # Shape is statically known after pre-filtering.\n"
    "            # Return shape [1] (not 0-D scalar) to match sum/min/max/mean output shape.\n"
    "            col_node = self._visit_expr(args[0], nodes)\n"
    "            n = col_node.type.shape[0]\n"
    "            return ops.constant(np.array([int(n)], dtype=np.int64), dtype=DType.int64, device=DeviceRef.CPU())\n"
    "\n"
    "        raise NotImplementedError(f\"Unsupported expression op: '{op}'\")\n"
)

NEW_COMPILER_TESTS = (
    "import pyarrow as pa\n"
    "from mxframe.lazy_expr import col, lit\n"
    "from mxframe.lazy_frame import LazyFrame, Scan\n"
    "\n"
    "# \u2500\u2500 Test 1: Simple projection \u2500\u2500\n"
    "table = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})\n"
    "lf = LazyFrame(Scan(table))\n"
    "result = GraphCompiler().compile_and_run(lf.select(col('a') + lit(10)).plan)\n"
    "assert result.column(0).to_pylist() == [11, 12, 13], f'Projection failed: {result}'\n"
    "print('\u2705 Test 1 passed: projection')\n"
    "\n"
    "# \u2500\u2500 Test 2: Global sum aggregation \u2500\u2500\n"
    "lf2 = LazyFrame(Scan(pa.table({'x': [1.0, 2.0, 3.0, 4.0]})))\n"
    "result2 = GraphCompiler().compile_and_run(\n"
    "    lf2.groupby().agg(col('x').sum().alias('total')).plan\n"
    ")\n"
    "assert result2.column('total').to_pylist()[0] == 10.0, f'Sum failed: {result2}'\n"
    "print('\u2705 Test 2 passed: global sum aggregation')\n"
    "\n"
    "# \u2500\u2500 Test 3: Filter REMOVES rows (not zeros them) \u2500\u2500\n"
    "table3 = pa.table({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0]})\n"
    "lf3 = LazyFrame(Scan(table3))\n"
    "result3 = GraphCompiler().compile_and_run(lf3.filter(col('a') > lit(2)).plan)\n"
    "assert result3.num_rows == 3, f'Filter should produce 3 rows, got {result3.num_rows}'\n"
    "assert result3.column('a').to_pylist() == [3, 4, 5], f'Wrong rows: {result3.column(\"a\").to_pylist()}'\n"
    "print('\u2705 Test 3 passed: filter removes rows (not masks them)')\n"
    "\n"
    "# \u2500\u2500 Test 4: mean() is correct after filter \u2500\u2500\n"
    "# Old bug: mask-multiply gave mean([0,0,30,40,50])/5 = 24.0\n"
    "# Correct:  mean([30.0, 40.0, 50.0]) = 40.0\n"
    "result4 = GraphCompiler().compile_and_run(\n"
    "    lf3.filter(col('a') > lit(2)).groupby().agg(col('b').mean().alias('avg_b')).plan\n"
    ")\n"
    "avg_b = result4.column('avg_b').to_pylist()[0]\n"
    "assert abs(avg_b - 40.0) < 1e-6, f'Mean after filter should be 40.0, got {avg_b} (old bug was 24.0)'\n"
    "print('\u2705 Test 4 passed: mean() correct after filter (old bug was 24.0)')\n"
    "\n"
    "# \u2500\u2500 Test 5: count() global \u2500\u2500\n"
    "table5 = pa.table({'v': [10, 20, 30, 40, 50]})\n"
    "lf5 = LazyFrame(Scan(table5))\n"
    "result5 = GraphCompiler().compile_and_run(\n"
    "    lf5.groupby().agg(col('v').count().alias('n')).plan\n"
    ")\n"
    "assert result5.column('n').to_pylist()[0] == 5, f'Count should be 5, got {result5.column(\"n\").to_pylist()[0]}'\n"
    "print('\u2705 Test 5 passed: count() global')\n"
    "\n"
    "# \u2500\u2500 Test 6: count() after filter \u2500\u2500\n"
    "result6 = GraphCompiler().compile_and_run(\n"
    "    lf5.filter(col('v') > lit(25)).groupby().agg(col('v').count().alias('n')).plan\n"
    ")\n"
    "assert result6.column('n').to_pylist()[0] == 3, f'Count after filter should be 3, got {result6.column(\"n\").to_pylist()[0]}'\n"
    "print('\u2705 Test 6 passed: count() after filter')\n"
    "\n"
    "print('\\nAll GraphCompiler tests passed! \u2705')\n"
)

with open('nbs/03_compiler.ipynb') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if '_NP_TO_MAX' in src and 'class GraphCompiler' in src:
        cell['source'] = NEW_COMPILER_CLASS.splitlines(keepends=True)
        patched += 1
        print(f"  ✓ patched GraphCompiler class (cell had {len(src)} chars)")
    elif 'Test 1' in src and 'All GraphCompiler' in src:
        cell['source'] = NEW_COMPILER_TESTS.splitlines(keepends=True)
        patched += 1
        print(f"  ✓ patched compiler tests")

with open('nbs/03_compiler.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Saved 03_compiler.ipynb ({patched} cells patched)")


# ════════════════════════════════════════════════════════════════════
# 04_custom_ops.ipynb
# ════════════════════════════════════════════════════════════════════

NEW_CUSTOM_OPS_CLASS = (
    "#| export\n"
    "# KERNELS_PATH resolves next to the installed module, or uses absolute path in notebooks.\n"
    "KERNELS_PATH = (\n"
    '    str(Path(__file__).parent / "kernels.mojopkg")\n'
    '    if "__file__" in dir()\n'
    '    else str(Path("/home/ablearn/mxdf/mxframe/kernels.mojopkg"))\n'
    ")\n"
    "\n"
    "WARP_SIZE = 32\n"
    "\n"
    "\n"
    "class CustomOpsCompiler(GraphCompiler):\n"
    '    """Compiles a LogicalPlan into a MAX Graph with custom Mojo kernels.\n'
    "\n"
    "    Extends GraphCompiler with:\n"
    "    1. Pre-filter   -- inherited _strip_filters applies Filter nodes in PyArrow.\n"
    "    2. Group ids    -- PyArrow dictionary-encodes group-by keys into int32 ids.\n"
    "    3. Kernel dispatch -- group_sum routed to the Mojo custom kernel.\n"
    "    4. Keys in result -- group-by key columns are prepended to the output table.\n"
    "    5. Guarded fallback -- grouped min/max/mean/count raise NotImplementedError\n"
    "       rather than silently returning wrong global results.\n"
    '    """\n'
    "\n"
    "    def __init__(self, kernels_path: str = None):\n"
    "        super().__init__()\n"
    "        self.kernels_path = kernels_path or KERNELS_PATH\n"
    "\n"
    "    # \u2500\u2500 public API \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "\n"
    "    def compile_and_run(self, plan: LogicalPlan) -> pa.Table:\n"
    '        """Apply filters, build group ids, run custom graph, prepend key columns."""\n'
    "        # Step 1: Pre-filter all Filter nodes in PyArrow (inherited from GraphCompiler)\n"
    "        plan = self._strip_filters(plan)\n"
    "        scan = self._find_scan(plan)\n"
    "\n"
    "        # Step 2: Grouped aggregation setup\n"
    "        extra_inputs: Dict[str, Any] = {}\n"
    "        agg_node = self._find_aggregate(plan)\n"
    "        group_keys: List[str] = []\n"
    "        n_groups = 0\n"
    "        unique_key_cols: Dict[str, pa.Array] = {}\n"
    "\n"
    "        if agg_node is not None and agg_node.group_by:\n"
    "            group_keys = [e.args[0] for e in agg_node.group_by]\n"
    "            group_ids_arr, n_groups, unique_key_cols = self._build_group_ids(\n"
    "                scan.table, group_keys\n"
    "            )\n"
    '            extra_inputs["__group_ids__"] = group_ids_arr.astype(np.int32)\n'
    "\n"
    "        # Step 3: Build col_arrays -- skip non-numeric columns (strings, etc.)\n"
    "        col_names = []\n"
    "        col_arrays: Dict[str, np.ndarray] = {}\n"
    "        for name in scan.table.column_names:\n"
    "            arr = np.array(scan.table[name])\n"
    '            if arr.dtype.kind in ("i", "u", "f", "b"):\n'
    "                col_names.append(name)\n"
    "                col_arrays[name] = arr\n"
    "\n"
    "        all_names = col_names + list(extra_inputs.keys())\n"
    "        all_arrays = {**col_arrays, **extra_inputs}\n"
    "\n"
    "        input_types = [\n"
    "            TensorType(_max_dtype(all_arrays[n]), list(all_arrays[n].shape), DeviceRef.CPU())\n"
    "            for n in all_names\n"
    "        ]\n"
    "\n"
    "        graph = Graph(\n"
    '            name="mxframe_custom",\n'
    "            input_types=input_types,\n"
    "            custom_extensions=[Path(self.kernels_path)],\n"
    "        )\n"
    "        with graph:\n"
    "            nodes = {n: graph.inputs[i] for i, n in enumerate(all_names)}\n"
    "            result_nodes = self._visit_plan_custom(plan, nodes, n_groups=n_groups)\n"
    "            graph.output(*result_nodes.values())\n"
    "\n"
    "        model = self.session.load(graph)\n"
    "        output_vals = model.execute(*[all_arrays[n] for n in all_names])\n"
    "\n"
    "        agg_names = list(result_nodes.keys())\n"
    "        agg_arrays = [pa.array(t.to_numpy()) for t in output_vals]\n"
    "\n"
    "        # Step 4: Prepend group-by key columns (keys first, then aggs -- Polars convention)\n"
    "        if unique_key_cols:\n"
    "            key_arrays = [unique_key_cols[k] for k in group_keys]\n"
    "            return pa.Table.from_arrays(key_arrays + agg_arrays, names=group_keys + agg_names)\n"
    "        return pa.Table.from_arrays(agg_arrays, names=agg_names)\n"
    "\n"
    "    # \u2500\u2500 plan traversal (custom-aware) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "\n"
    "    def _visit_plan_custom(self, plan, nodes, *, n_groups=0):\n"
    "        if isinstance(plan, Scan):\n"
    "            return nodes\n"
    "        elif isinstance(plan, Project):\n"
    "            upstream = self._visit_plan_custom(plan.input, nodes, n_groups=n_groups)\n"
    "            out = {}\n"
    "            for i, expr in enumerate(plan.exprs):\n"
    "                name = expr._alias or f'col_{i}'\n"
    "                out[name] = self._visit_expr(expr, upstream)\n"
    "            return out\n"
    "        elif isinstance(plan, Filter):\n"
    "            raise RuntimeError(\n"
    "                'Filter node reached _visit_plan_custom -- '\n"
    "                '_strip_filters should have removed all Filter nodes before graph construction.'\n"
    "            )\n"
    "        elif isinstance(plan, Aggregate):\n"
    "            return self._visit_aggregate_custom(plan, nodes, n_groups=n_groups)\n"
    "        raise NotImplementedError(f'Unsupported plan node: {type(plan)}')\n"
    "\n"
    "    def _visit_aggregate_custom(self, plan, nodes, *, n_groups):\n"
    "        upstream = self._visit_plan_custom(plan.input, nodes, n_groups=n_groups)\n"
    "        out = {}\n"
    "        grouped = bool(plan.group_by)\n"
    "        for i, expr in enumerate(plan.aggs):\n"
    "            name = expr._alias or f'agg_{i}'\n"
    "            if grouped and n_groups > 0:\n"
    '                if expr.op == "sum":\n'
    "                    # Route to the Mojo group_sum kernel (float32 values + int32 ids)\n"
    "                    val_node = ops.cast(self._visit_expr(expr.args[0], upstream), DType.float32)\n"
    '                    gid_node = ops.cast(upstream["__group_ids__"], DType.int32)\n'
    "                    out_type = TensorType(DType.float32, [n_groups], DeviceRef.CPU())\n"
    "                    results = ops.custom(\n"
    '                        name="group_sum",\n'
    "                        values=[val_node, gid_node],\n"
    "                        out_types=[out_type],\n"
    "                        device=DeviceRef.CPU(),\n"
    "                    )\n"
    "                    out[name] = results[0]\n"
    "                else:\n"
    "                    raise NotImplementedError(\n"
    "                        f\"Grouped '{expr.op}' is not yet wired. \"\n"
    "                        f\"Implement group_{expr.op}.mojo and register it in Phase 1.\"\n"
    "                    )\n"
    "            else:\n"
    "                # Global (non-grouped) aggregation -- fall through to built-in MAX ops\n"
    "                out[name] = self._visit_expr(expr, upstream)\n"
    "        return out\n"
    "\n"
    "    # \u2500\u2500 helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "\n"
    "    @staticmethod\n"
    "    def _find_aggregate(plan):\n"
    "        if isinstance(plan, Aggregate): return plan\n"
    "        if hasattr(plan, 'input'): return CustomOpsCompiler._find_aggregate(plan.input)\n"
    "        return None\n"
    "\n"
    "    @staticmethod\n"
    "    def _build_group_ids(\n"
    "        table: pa.Table, keys: List[str]\n"
    "    ):\n"
    '        """Dictionary-encode group keys into contiguous int32 group ids.\n'
    "\n"
    "        Returns:\n"
    "            ids            -- int32 array mapping each row to a group id.\n"
    "            n_groups       -- number of unique groups.\n"
    "            unique_key_cols -- dict mapping key name to unique values in group-id order.\n"
    '        """\n'
    "        if len(keys) == 1:\n"
    "            col_arr = table.column(keys[0])\n"
    "            if isinstance(col_arr, pa.ChunkedArray):\n"
    "                col_arr = col_arr.combine_chunks()\n"
    "            encoded = col_arr.dictionary_encode()\n"
    "            ids = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32)\n"
    "            unique_key_cols = {keys[0]: encoded.dictionary}\n"
    "            return ids, len(encoded.dictionary), unique_key_cols\n"
    "\n"
    "        # Multi-key: concatenate string representations, then dictionary-encode\n"
    "        parts = [pc.cast(table.column(k), pa.string()) for k in keys]\n"
    "        composite = parts[0]\n"
    "        for p in parts[1:]:\n"
    '            composite = pc.binary_join_element_wise(composite, p, "|||")\n'
    "        if isinstance(composite, pa.ChunkedArray):\n"
    "            composite = composite.combine_chunks()\n"
    "        encoded = composite.dictionary_encode()\n"
    "        ids = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32)\n"
    "\n"
    "        # Reconstruct per-column unique values from the composite dictionary\n"
    '        sep = "|||"\n'
    "        split_rows = [s.split(sep) for s in encoded.dictionary.to_pylist()]\n"
    "        unique_key_cols = {\n"
    "            keys[i]: pa.array([row[i] for row in split_rows])\n"
    "            for i in range(len(keys))\n"
    "        }\n"
    "        return ids, len(encoded.dictionary), unique_key_cols\n"
)

NEW_CUSTOM_OPS_TESTS = (
    "import pyarrow as pa\n"
    "from mxframe.lazy_expr import col, lit\n"
    "from mxframe.lazy_frame import LazyFrame, Scan\n"
    "\n"
    "kernels_path = (\n"
    '    str(Path(__file__).parent.parent / "mxframe" / "kernels.mojopkg")\n'
    '    if "__file__" in dir()\n'
    '    else str(Path("/home/ablearn/mxdf/mxframe/kernels.mojopkg"))\n'
    ")\n"
    "\n"
    "compiler = CustomOpsCompiler(kernels_path)\n"
    "\n"
    "# \u2500\u2500 Test 1: Projection falls through to built-in ops \u2500\u2500\n"
    "table = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})\n"
    "lf = LazyFrame(Scan(table))\n"
    "result = compiler.compile_and_run(lf.select(col('a') + lit(10)).plan)\n"
    "assert result.num_columns == 1\n"
    "assert result.column(0).to_pylist() == [11, 12, 13], f'Projection failed: {result}'\n"
    "print('\u2705 Test 1 passed: projection')\n"
    "\n"
    "# \u2500\u2500 Test 2: Global sum via built-in ops \u2500\u2500\n"
    "table2 = pa.table({'x': [1.0, 2.0, 3.0, 4.0]})\n"
    "lf2 = LazyFrame(Scan(table2))\n"
    "result2 = compiler.compile_and_run(\n"
    "    lf2.groupby().agg(col('x').sum().alias('total')).plan\n"
    ")\n"
    "assert abs(result2.column('total').to_pylist()[0] - 10.0) < 1e-6, f'Sum failed: {result2}'\n"
    "print('\u2705 Test 2 passed: global sum aggregation')\n"
    "\n"
    "# \u2500\u2500 Test 3: Grouped sum via Mojo group_sum kernel -- with group keys in result \u2500\u2500\n"
    "table3 = pa.table({\n"
    "    'group': ['a', 'b', 'a', 'b', 'a'],\n"
    "    'val':   [1.0, 2.0, 3.0, 4.0, 5.0],\n"
    "})\n"
    "lf3 = LazyFrame(Scan(table3))\n"
    "result3 = compiler.compile_and_run(\n"
    "    lf3.groupby('group').agg(col('val').sum().alias('total')).plan\n"
    ")\n"
    "assert 'group' in result3.column_names, f'Missing group key: {result3.column_names}'\n"
    "assert 'total' in result3.column_names, f'Missing total: {result3.column_names}'\n"
    "assert result3.column_names[0] == 'group', f'Key should be first: {result3.column_names}'\n"
    "groups = result3.column('group').to_pylist()\n"
    "totals = result3.column('total').to_pylist()\n"
    "result_dict = dict(zip(groups, totals))\n"
    "assert abs(result_dict['a'] - 9.0) < 1e-6, f'Sum for a should be 9.0: {result_dict}'\n"
    "assert abs(result_dict['b'] - 6.0) < 1e-6, f'Sum for b should be 6.0: {result_dict}'\n"
    "print('\u2705 Test 3 passed: grouped sum with key columns in result (a=9, b=6)')\n"
    "\n"
    "# \u2500\u2500 Test 4: Filter removes rows before grouped aggregation \u2500\u2500\n"
    "table4 = pa.table({\n"
    "    'group': ['a', 'b', 'a', 'b', 'a'],\n"
    "    'val':   [1.0, 2.0, 3.0, 4.0, 5.0],\n"
    "    'flag':  [1,   1,   0,   1,   1  ],\n"
    "})\n"
    "lf4 = LazyFrame(Scan(table4))\n"
    "result4 = compiler.compile_and_run(\n"
    "    lf4.filter(col('flag') > lit(0)).groupby('group').agg(col('val').sum().alias('total')).plan\n"
    ")\n"
    "groups4 = result4.column('group').to_pylist()\n"
    "totals4 = result4.column('total').to_pylist()\n"
    "result_dict4 = dict(zip(groups4, totals4))\n"
    "assert abs(result_dict4['a'] - 6.0) < 1e-6, f'Filtered sum for a should be 6.0: {result_dict4}'\n"
    "assert abs(result_dict4['b'] - 6.0) < 1e-6, f'Filtered sum for b should be 6.0: {result_dict4}'\n"
    "print('\u2705 Test 4 passed: filter removes rows before grouped aggregation')\n"
    "\n"
    "# \u2500\u2500 Test 5: Grouped min/max/mean raise NotImplementedError \u2500\u2500\n"
    "table5 = pa.table({'g': ['x', 'x', 'y'], 'v': [1.0, 2.0, 3.0]})\n"
    "lf5 = LazyFrame(Scan(table5))\n"
    "for bad_agg in [col('v').min(), col('v').max(), col('v').mean()]:\n"
    "    try:\n"
    "        compiler.compile_and_run(lf5.groupby('g').agg(bad_agg).plan)\n"
    "        assert False, f'Should have raised NotImplementedError for {bad_agg}'\n"
    "    except NotImplementedError as e:\n"
    "        assert 'Phase 1' in str(e), f'Error should mention Phase 1: {e}'\n"
    "print('\u2705 Test 5 passed: grouped min/max/mean raise NotImplementedError')\n"
    "\n"
    "print('\\nAll CustomOpsCompiler tests passed! \U0001f389')\n"
)

with open('nbs/04_custom_ops.ipynb') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'KERNELS_PATH' in src and 'class CustomOpsCompiler' in src:
        cell['source'] = NEW_CUSTOM_OPS_CLASS.splitlines(keepends=True)
        patched += 1
        print(f"  \u2713 patched CustomOpsCompiler class (cell had {len(src)} chars)")
    elif 'Test 1' in src and 'All CustomOpsCompiler' in src:
        cell['source'] = NEW_CUSTOM_OPS_TESTS.splitlines(keepends=True)
        patched += 1
        print("  \u2713 patched custom_ops tests")

with open('nbs/04_custom_ops.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Saved 04_custom_ops.ipynb ({patched} cells patched)")
