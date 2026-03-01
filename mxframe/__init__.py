"""mxframe - GPU-accelerated DataFrames with MAX Engine"""

__version__ = "0.0.1"

# Lazy expression layer
from .lazy_expr import Expr, col, lit

# Lazy frame & logical plan
from .lazy_frame import (
    LogicalPlan, Scan, Filter, Project, Aggregate,
    LazyFrame, LazyGroupBy,
    DeviceType,
)

# Compiler
from .compiler import GraphCompiler

# Custom ops compiler
from .custom_ops import CustomOpsCompiler, KERNELS_PATH

__all__ = [
    # lazy_expr
    "Expr", "col", "lit",
    # lazy_frame
    "LogicalPlan", "Scan", "Filter", "Project", "Aggregate",
    "LazyFrame", "LazyGroupBy", "DeviceType",
    # compiler
    "GraphCompiler",
    # custom_ops
    "CustomOpsCompiler", "KERNELS_PATH",
]
