"""mxframe - GPU-accelerated DataFrames with MAX Engine"""

__version__ = "0.1.0"

# Lazy expression layer
from .lazy_expr import Expr, col, lit, when, row_number

# Lazy frame & logical plan
from .lazy_frame import (
    LogicalPlan, Scan, Filter, Project, Aggregate,
    Sort, Limit, Distinct, Join,
    LazyFrame, LazyGroupBy, GPUFrame,
    DeviceType,
)

# Compiler
from .compiler import GraphCompiler

# Optimizer
from .optimizer import PlanOptimizer, OptimizationResult, optimize_plan

# Plan validation
from .plan_validation import PlanValidationError, validate_plan, validate_plan_or_raise

# Custom ops compiler
from .custom_ops import CustomOpsCompiler, KERNELS_PATH, clear_cache

# SQL frontend
from .sql_frontend import sql


def warmup(device: str = "auto") -> float:
    """Pre-initialize MAX runtime and pre-compile a canonical graph for fast first-query.

    Call once at application startup (or at the top of a notebook) to absorb the
    one-time JIT / framework-bootstrap cost before running real workloads.

    Steps performed:
      1. Create and cache the InferenceSession  (GPU context init, kernel library load).
      2. Run a tiny synthetic groupby-sum computation (~1 K rows) through the full
         MAX Graph compile-and-execute pipeline.  This triggers MLIR / LLVM / CUDA JIT
         bootstrapping so the first real query experiences no cold-start penalty.

    Args:
        device: "auto" (default), "cpu", or "gpu".

    Returns:
        Wall-clock seconds consumed by warmup (diagnostic; can be ignored).
    """
    import time as _t
    import pyarrow as _pa
    import numpy as _np

    t0 = _t.perf_counter()

    # Step 1: session init (cached -- effectively free on repeat calls)
    CustomOpsCompiler(device=device)

    # Step 2: synthetic groupby-sum to trigger MAX JIT bootstrap.
    # 1 024-row table, 8 groups, one float32 value column.
    _N = 1024
    _rng = _np.random.default_rng(0)
    _tiny = _pa.table({
        "g": _pa.array((_rng.integers(0, 8, size=_N)).astype(_np.int32)),
        "v": _pa.array(_rng.uniform(0.0, 1.0, size=_N).astype(_np.float32)),
    })
    try:
        (LazyFrame(Scan(_tiny))
         .groupby("g")
         .agg(col("v").sum().alias("s"))
         .compute(device=device))
    except Exception:
        pass  # best-effort; never block user code on warmup failure

    return _t.perf_counter() - t0




# ── I/O & conversion helpers (Phase 7) ─────────────────────── #

import pyarrow.csv as _pa_csv
import pyarrow.parquet as _pq


def from_arrow(table) -> "LazyFrame":
    return LazyFrame(Scan(table))


def from_pandas(df) -> "LazyFrame":
    import pandas as _pd
    import pyarrow as _pa
    return LazyFrame(Scan(_pa.Table.from_pandas(df, preserve_index=False)))


def from_polars(df) -> "LazyFrame":
    return LazyFrame(Scan(df.to_arrow()))


def read_csv(path: str, **kwargs) -> "LazyFrame":
    return LazyFrame(Scan(_pa_csv.read_csv(path, **kwargs)))


def read_parquet(path: str, **kwargs) -> "LazyFrame":
    return LazyFrame(Scan(_pq.read_table(path, **kwargs)))

__all__ = [
    # lazy_expr
    "Expr", "col", "lit", "when", "row_number",
    # lazy_frame
    "LogicalPlan", "Scan", "Filter", "Project", "Aggregate",
    "Sort", "Limit", "Distinct", "Join",
    "LazyFrame", "LazyGroupBy", "GPUFrame", "DeviceType",
    # compiler
    "GraphCompiler",
    # optimizer
    "PlanOptimizer", "OptimizationResult", "optimize_plan",
    # plan validation
    "PlanValidationError", "validate_plan", "validate_plan_or_raise",
    # custom_ops
    "CustomOpsCompiler", "KERNELS_PATH",
    # caching
    "clear_cache", "warmup",
    # I/O
    "from_arrow", "from_pandas", "from_polars", "read_csv", "read_parquet",
    # SQL
    "sql",
]
