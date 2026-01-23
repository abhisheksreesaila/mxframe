import pandas as pd
import polars as pl
import pyarrow as pa
import numpy as np
import time
import matplotlib.pyplot as plt
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType


# --- 1. THE ZERO-COPY BRIDGE ---
def arrow_to_max_bridge(arrow_arr, device):
    """Bridges Arrow to MAX without moving a single byte."""
    np_view = arrow_arr.to_numpy(zero_copy_only=True)
    # Create tensor on CPU first, then copy to target device if needed
    cpu_tensor = driver.Tensor(np_view, driver.CPU())
    if isinstance(device, driver.Accelerator):
        return cpu_tensor.copy(device=device)
    return cpu_tensor



# --- 2. DATA GENERATION ---
print("Generating 10 Million rows of data...")
rows = 10_000_000
raw_data = np.random.randn(rows).astype(np.float32)
arrow_col = pa.array(raw_data)

# --- 3. PANDAS BENCHMARK ---
# Pandas must convert Arrow to its own internal block manager
start_pd_load = time.perf_counter()
df = pd.DataFrame({'val': arrow_col.to_numpy()})
pd_load_time = time.perf_counter() - start_pd_load

start_pd_math = time.perf_counter()
pd_result = (df['val'] * 5.0) + 10.0
pd_math_time = time.perf_counter() - start_pd_math

# Pandas filter benchmark
start_pd_filter = time.perf_counter()
pd_filtered = df['val'].where(df['val'] > 0.5, 0.0)
pd_filter_time = time.perf_counter() - start_pd_filter

# --- 3b. POLARS BENCHMARK ---
# Polars uses zero-copy from Arrow natively
start_pl_load = time.perf_counter()
pl_df = pl.from_arrow(pa.table({'val': arrow_col}))
pl_load_time = time.perf_counter() - start_pl_load

# Try GPU first, fall back to CPU if GPU not available
try:
    # Polars math benchmark - Run on GPU
    start_pl_math = time.perf_counter()
    pl_result = (pl_df.lazy().select((pl.col('val') * 5.0 + 10.0).alias('result'))).collect(engine="gpu")
    pl_math_time = time.perf_counter() - start_pl_math
    
    # Polars filter benchmark - Run on GPU
    start_pl_filter = time.perf_counter()
    pl_filtered = (
        pl_df.lazy()
        .select(pl.when(pl.col('val') > 0.5).then(pl.col('val')).otherwise(0.0).alias('filtered'))
        .collect(engine="gpu")
    )
    pl_filter_time = time.perf_counter() - start_pl_filter
    polars_engine = "GPU"
except Exception as e:
    print(f"Polars GPU not available, falling back to CPU")
    # Polars math benchmark - CPU
    start_pl_math = time.perf_counter()
    pl_result = (pl_df.lazy().select((pl.col('val') * 5.0 + 10.0).alias('result'))).collect()
    pl_math_time = time.perf_counter() - start_pl_math
    
    # Polars filter benchmark - CPU
    start_pl_filter = time.perf_counter()
    pl_filtered = (
        pl_df.lazy()
        .select(pl.when(pl.col('val') > 0.5).then(pl.col('val')).otherwise(0.0).alias('filtered'))
        .collect()
    )
    pl_filter_time = time.perf_counter() - start_pl_filter
    polars_engine = "CPU"

# --- 4. MXFRAME (MAX) BENCHMARK ---
# MAX uses the bridge (instant) and fused execution

# Select device: Use GPU if available, otherwise CPU
device = driver.Accelerator() if driver.accelerator_count() > 0 else driver.CPU()
device_ref = DeviceRef.GPU(0) if driver.accelerator_count() > 0 else DeviceRef.CPU()
print(f"Using device: {device}")

start_mx_load = time.perf_counter()
mx_tensor = arrow_to_max_bridge(arrow_col, device)
mx_load_time = time.perf_counter() - start_mx_load

# Define computation as a callable class
class Compute:
    def __call__(self, x):
        five = ops.constant(5.0, dtype=DType.float32, device=device_ref)
        ten = ops.constant(10.0, dtype=DType.float32, device=device_ref)
        multiplied = ops.mul(x, five)
        result = ops.add(multiplied, ten)
        return result

# Create graph with proper input types
compute_graph = Graph(
    "compute",
    Compute(),
    input_types=[TensorType(DType.float32, mx_tensor.shape, device_ref)]
)

# Load model with MAX Engine
session = engine.InferenceSession(devices=[device])
model = session.load(compute_graph)

# WARM UP: Run once to trigger compilation (don't time this)
_ = model.execute(mx_tensor)[0]

# REAL TIMING: Now the kernel is already compiled in memory
start_mx_math = time.perf_counter()
mx_result = model.execute(mx_tensor)[0]
mx_math_time = time.perf_counter() - start_mx_math

# --- 4b. FILTER BENCHMARK (MAX) ---
# Define filtering computation
class FilterCompute:
    def __call__(self, x):
        threshold = ops.constant(0.5, dtype=DType.float32, device=device_ref)
        zero = ops.constant(0.0, dtype=DType.float32, device=device_ref)
        # Generate boolean mask: which values are > 0.5?
        mask = ops.greater(x, threshold)
        # Apply filter: keep value if > 0.5, else set to 0.0
        filtered = ops.where(mask, x, zero)
        return filtered

# Create filter graph
filter_graph = Graph(
    "filter",
    FilterCompute(),
    input_types=[TensorType(DType.float32, mx_tensor.shape, device_ref)]
)

filter_model = session.load(filter_graph)

# WARM UP: Run once to trigger compilation
_ = filter_model.execute(mx_tensor)[0]

# REAL TIMING: Measure filter performance
start_mx_filter = time.perf_counter()
mx_filtered = filter_model.execute(mx_tensor)[0]
mx_filter_time = time.perf_counter() - start_mx_filter


# --- 5. RESULTS & VISUALIZATION ---
print(f"\n[Pandas]  Load: {pd_load_time:.4f}s | Math: {pd_math_time:.4f}s | Filter: {pd_filter_time:.4f}s")
print(f"[Polars {polars_engine}]  Load: {pl_load_time:.4f}s | Math: {pl_math_time:.4f}s | Filter: {pl_filter_time:.4f}s")
print(f"[MXFrame] Load: {mx_load_time:.4f}s | Math: {mx_math_time:.4f}s | Filter: {mx_filter_time:.4f}s")

def plot_benchmark(pd_t, pl_t, mx_t, title):
    labels = ['Pandas', f'Polars ({polars_engine})', 'MXFrame (GPU)']
    times = [pd_t, pl_t, mx_t]
    colors = ['#150458', '#CD792C', '#FF4B4B']
    plt.bar(labels, times, color=colors)
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

# Plotting the Math Performance
plot_benchmark(pd_math_time, pl_math_time, mx_math_time, "10M Row Math Execution (Lower is Better)")

# Plotting the Filter Performance
plot_benchmark(pd_filter_time, pl_filter_time, mx_filter_time, "10M Row Filter Execution (Lower is Better)")
