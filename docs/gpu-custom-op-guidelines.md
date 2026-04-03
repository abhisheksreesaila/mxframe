# GPU Custom Op Guidelines (MXFrame)

Canonical playbook for writing and debugging GPU custom ops in MXFrame.

## Scope and Authority

- Scope: current `mxframe` kernels, grouped aggregations, and fused/groupby paths.
- Source of truth: notebooks first (`nbs/*`), then export to `mxframe/*.py`.
- Kernel sources: `mxframe/kernels/*.mojo`; package artifact: `mxframe/kernels.mojopkg`.

## Non-Negotiable Rules

1. Notebook-first implementation.
2. Explicit dtype/shape kernel contract.
3. Grouped GPU partial output contract: `[num_warps * n_groups]`.
4. Hard fail for `n_groups > 64` in grouped GPU path.
5. Device consistency across `TensorType`, `ops.custom`, and input tensors.
6. Mandatory bounds checks in kernels.

## One-Command Validation

Run the full GPU playbook checks with:

- `pixi run bash scripts/validate_gpu_playbook.sh`

This command rebuilds kernels and runs all required smoke/regression checks in the approved order.

## Debugging Ladder (Run in Order)

1. `pixi run bash scripts/build_kernels.sh`
2. `pixi run python3 scripts/_test_gpu_debug_write.py`
3. `pixi run python3 scripts/_test_gpu_custom_op.py`
4. `pixi run python3 scripts/_test_gpu_noop.py`
5. `pixi run python3 scripts/_test_gpu_regression.py`
6. `pixi run python3 scripts/test_fused_reduce.py`

## Failure Classification

- Packaging issue
- Graph contract mismatch
- Device mismatch
- Indexing/bounds issue
- Post-processing mismatch

## Pre-PR Checklist

- [ ] Notebook-first changes + export
- [ ] Kernel registered/re-exported
- [ ] Kernels package rebuilt
- [ ] GPU sanity tests pass
- [ ] Grouped regression passes
- [ ] Fused reduction test passes when applicable
- [ ] `pixi run bash scripts/validate_gpu_playbook.sh` passes

## References (Local)

- `archive/soure code/modular/max/examples/custom_ops/`
- `archive/soure code/modular/docs/max/develop/build-custom-ops.mdx`
- `llm-ctx/llms-mojo.txt`
- `llm-ctx/llms-python.txt`