# MXFrame TODO

## v0.0.2 - GroupBy & Advanced Ops
- [ ] Simple `groupby().agg()` API using PyArrow's `group_by()`
- [ ] Multi-column groupby support
- [ ] `sort()` method

## v0.0.3 - GPU Dispatch
- [ ] Add `device='gpu'|'cpu'` parameter to operations
- [ ] Wire up existing MAX kernels (`masked_sum_simple`, `fused_group_agg`)
- [ ] Auto-dispatch based on data size threshold

## v0.0.4 - Lazy Execution
- [ ] `02_graph.ipynb` - Lazy graph building
- [ ] Deferred execution with `.collect()` or `.compute()`
- [ ] Operation fusion for GPU efficiency

## v0.0.5 - SQL Frontend
- [ ] `05_sql.ipynb` - SQLGlot integration
- [ ] `frame.sql("SELECT * FROM df WHERE qty > 10")`

## Future Ideas
- [ ] Joins: `inner_join()`, `left_join()`
- [ ] Window functions
- [ ] String operations
- [ ] PyPI publish workflow

## Notes
- Mask-based aggregation is the core GPU pattern
- Pre-encode strings before GPU ops
- Compile once, execute many - cache MAX graphs