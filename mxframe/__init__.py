"""mxframe - GPU-accelerated DataFrames with MAX Engine"""

__version__ = "0.0.1"

# Core bridge - zero-copy PyArrow to MAX
from .core_bridge import (
    MXFrame,
    arrow_to_numpy_view,
    arrow_to_max_tensor,
    get_max_dtype,
    get_numpy_dtype,
    ARROW_TO_MAX_DTYPE,
    ARROW_TO_NUMPY_DTYPE,
    DeviceType,
)

__all__ = [
    "MXFrame",
    "arrow_to_numpy_view", 
    "arrow_to_max_tensor",
    "get_max_dtype",
    "get_numpy_dtype",
    "ARROW_TO_MAX_DTYPE",
    "ARROW_TO_NUMPY_DTYPE",
    "DeviceType",
]
