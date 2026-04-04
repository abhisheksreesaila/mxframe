"""mxframe - GPU-accelerated DataFrames with MAX Engine"""

__version__ = "0.0.1"

# Lazy expression layer
from .lazy_expr import Expr, col, lit, when

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


def warmup(device: str = "auto"):
    """Pre-initialize MAX runtime and InferenceSession for the given device.

    Call at application startup to move the one-time MAX runtime init cost
    out of the first query.  The session is cached and reused by all
    subsequent .compute() calls.

    Args:
        device: "auto" (default), "cpu", or "gpu".
    """
    CustomOpsCompiler(device=device)




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
    "Expr", "col", "lit", "when",
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
