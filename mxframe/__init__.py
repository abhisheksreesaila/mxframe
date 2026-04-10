"""mxframe - GPU-accelerated DataFrames with MAX Engine"""

__version__ = "0.0.1"

# Lazy expression layer
from .lazy_expr import Expr, col, lit

# Lazy frame & logical plan
from .lazy_frame import (
    LogicalPlan, Scan, Filter, Project, Aggregate,
    Sort, Limit, Distinct, Join,
    LazyFrame, LazyGroupBy,
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


def warmup(device: str = "cpu"):
    """Pre-initialize MAX runtime and InferenceSession for the given device.

    Call at application startup to move the one-time MAX runtime init cost
    out of the first query.  The session is cached and reused by all
    subsequent .compute() calls.

    Args:
        device: "cpu", "gpu", or "auto".
    """
    CustomOpsCompiler(device=device)


__all__ = [
    # lazy_expr
    "Expr", "col", "lit",
    # lazy_frame
    "LogicalPlan", "Scan", "Filter", "Project", "Aggregate",
    "Sort", "Limit", "Distinct", "Join",
    "LazyFrame", "LazyGroupBy", "DeviceType",
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
]
