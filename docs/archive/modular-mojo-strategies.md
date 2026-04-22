# Modular-Informed MXFrame Uplift

Status date: 2026-04-02

## Goal

Align MXFrame with proven Modular/Mojo architecture patterns while preserving MXFrame's DataFrame-first ergonomics and existing cache-aware execution model.

This document is the implementation baseline and handoff source for remaining phases.

## Evidence Snapshot

### Reference-side signals (Modular/Mojo style)

- Layered, package-oriented runtime boundaries in [llm-ctx/src](llm-ctx/src).
- Explicit vendor/runtime adapter separation in [llm-ctx/src/_cublas/cublas.mojo](llm-ctx/src/_cublas/cublas.mojo), [llm-ctx/src/_cudnn/backend.mojo](llm-ctx/src/_cudnn/backend.mojo), and [llm-ctx/src/_rocblas/rocblas.mojo](llm-ctx/src/_rocblas/rocblas.mojo).
- Build and packaging contracts per subsystem via [llm-ctx/src/_cublas/BUILD.bazel](llm-ctx/src/_cublas/BUILD.bazel) and neighboring package BUILD files.

### MXFrame current implementation anchors

- Rule-based planner boundary exists in [mxframe/optimizer.py](mxframe/optimizer.py).
- Plan validation boundary exists in [mxframe/plan_validation.py](mxframe/plan_validation.py#L120).
- Compute routes through optimize/validate/provenance in [mxframe/lazy_frame.py](mxframe/lazy_frame.py#L126).
- Explain/runtime provenance surface exists in [mxframe/lazy_frame.py](mxframe/lazy_frame.py#L175).
- Optimizer-aware cache key and execution provenance exist in [mxframe/custom_ops.py](mxframe/custom_ops.py#L209) and [mxframe/custom_ops.py](mxframe/custom_ops.py#L264).
- Kernel package build contract exists in [scripts/build_kernels.sh](scripts/build_kernels.sh).

## Current Alignment

### Already aligned (implemented)

- Clear layer split across expression, logical plan, compiler, and runtime execution.
- Rule-based optimization pass pipeline with rewrite trace.
- Plan validation hooks in compute path.
- Explainability surface for optimized plan and runtime provenance.
- Cache key versioning includes optimizer-pass signature.
- Explicit shortcut-vs-compiled execution path provenance.

### Partially aligned

- Planner boundary exists but pass contracts and hooks are still minimal.
- Python hot path has been reduced in places but still owns key preprocessing paths.
- Join materialization is practical but join key encoding safety at larger composite domains needs hardening.
- Kernel artifact build/runtime flow exists but is not yet fully unified/documented across packaging and runtime diagnostics.

### Not yet aligned

- Formal fallback policy matrix for grouped high-cardinality behavior across all grouped ops.
- Feature-flagged A/B rollout with explicit promotion gates.
- Full benchmark matrix tied to cold/warm compile-execute lifecycle.

## Gap Matrix

| Area | Current state | Gap | Priority |
|---|---|---|---|
| Planner boundary | Rule-based optimizer exists | Expand deterministic pass contracts and join planning hooks | High |
| Compute routing | optimize/validate integrated | Ensure no bypass path without invariants and trace coverage | High |
| Python hot path | Reduced in parts | Move remaining preprocessing from Python into graph/kernel paths where safe | High |
| Grouped high-cardinality | Sum path has resilient fallback | Align min/max/count/mean behavior and policy | High |
| Join key safety | Working joins | Reduce overflow/collision risk for wide/composite keys | High |
| Kernel artifact lifecycle | Build script and package exist | Unify build/runtime/package contract and diagnostics | Medium |
| Rollout controls | Ad hoc verification scripts exist | Add flags, A/B gates, and promotion criteria | Medium |

## Adopt Now / Later / Not Applicable

| Pattern | Decision | Why |
|---|---|---|
| Explicit planner pass manager | Adopt now | Already present; needs formalization and tests, not reinvention |
| Deterministic pass trace in user explainability | Adopt now | Already partially present and directly useful for debugging and cache correctness |
| Runtime provenance object across all execution paths | Adopt now | Needed for confidence, observability, and rollout gating |
| Cost-based optimizer and full join reordering | Adopt later | Requires statistics collection and calibration not yet present |
| Service-oriented model-serving runtime abstractions | Not applicable | MXFrame targets DataFrame execution, not serving stack orchestration |
| Vendor-specific runtime adapter sprawl | Adopt selectively | Keep minimal adapter surface needed for MXFrame kernels/runtime |

## Phased Execution Plan (Continuation)

### Phase 1: Baseline Strategy and Documentation Integrity

- Maintain this document as canonical strategy baseline.
- Synchronize architecture narrative with [docs/vision-and-architecture.md](docs/vision-and-architecture.md).
- Add contributor-facing implementation notes in [docs/dev-instructions.md](docs/dev-instructions.md).

### Phase 2: Planner Boundary Hardening

- Formalize optimizer pass ordering/contracts in [mxframe/optimizer.py](mxframe/optimizer.py).
- Ensure all compile entry paths preserve planner+validation guarantees across [mxframe/lazy_frame.py](mxframe/lazy_frame.py#L126) and [mxframe/custom_ops.py](mxframe/custom_ops.py#L264).
- Keep explain/provenance outputs consistent and deterministic in [mxframe/lazy_frame.py](mxframe/lazy_frame.py#L175).

### Phase 3: Python Hot-Path Reduction

- Inventory remaining Python preprocessing in [mxframe/custom_ops.py](mxframe/custom_ops.py).
- Shift safe sort/distinct/filter preparatory work into graph/kernel paths where it improves throughput without semantic risk.
- Keep explicit fallback policy controls and provenance path labeling.

### Phase 4: Kernel and Join Robustness

- Align grouped high-cardinality behavior across [mxframe/kernels_v261/group_sum.mojo](mxframe/kernels_v261/group_sum.mojo), [mxframe/kernels_v261/group_min.mojo](mxframe/kernels_v261/group_min.mojo), and [mxframe/kernels_v261/group_max.mojo](mxframe/kernels_v261/group_max.mojo).
- Harden join key encoding and collision/overflow safeguards in [mxframe/custom_ops.py](mxframe/custom_ops.py).
- Keep join kernel contracts explicit in [mxframe/kernels_v261/join_count.mojo](mxframe/kernels_v261/join_count.mojo) and [mxframe/kernels_v261/join_scatter.mojo](mxframe/kernels_v261/join_scatter.mojo).

### Phase 5: Build/Runtime Contract and Rollout

- Unify kernel artifact contract across [scripts/build_kernels.sh](scripts/build_kernels.sh), [MANIFEST.in](MANIFEST.in), and runtime loading in [mxframe/custom_ops.py](mxframe/custom_ops.py).
- Expand runtime diagnostics in [scripts/_check_gpu.py](scripts/_check_gpu.py) for kernel-resolution failures.
- Introduce feature flags and A/B gates before default path promotion.

## Measurable Success Criteria

1. Planner determinism:
Pass trace and optimized plan are deterministic for identical logical plans across repeated runs.

2. Semantic equivalence:
Optimizer-on and optimizer-off produce equivalent outputs for representative workloads within numeric tolerance.

3. Provenance completeness:
Runtime provenance includes execution path, optimizer trace, cache-hit status, and device backend for every compute path.

4. High-cardinality robustness:
Grouped aggregations pass stress tests on large group counts with documented fallback behavior.

5. Join safety:
Wide/composite-key joins pass correctness tests without overflow/collision regressions.

6. Artifact reliability:
Kernel build/package/runtime checks pass in CI and surface actionable diagnostics when environment prerequisites are missing.

7. Rollout gate:
Feature-flagged A/B mode demonstrates output parity and acceptable performance deltas before default promotion.

## Verification Checklist

- Optimizer rewrite tests for equivalence and trace stability.
- Plan validation tests for malformed plans and clear errors.
- Groupby high-cardinality stress tests.
- Join correctness tests including many-to-many and wide key domains.
- Cold/warm compile-execute benchmark matrix and regression thresholds.
- Environment diagnostics for MAX kernel resolution errors.

## Notes

- This continuation baseline intentionally preserves completed work and avoids re-planning already delivered architecture uplifts.
- Notebook-first parity must be maintained for [nbs/02_lazy_frame.ipynb](nbs/02_lazy_frame.ipynb) and [nbs/04_custom_ops.ipynb](nbs/04_custom_ops.ipynb) before any future exports.
