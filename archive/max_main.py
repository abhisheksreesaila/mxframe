import pyarrow as pa
import numpy as np
import sqlglot
from sqlglot import exp
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType


def date32_to_tensor(arrow_array, device=None, device_ref=None):
    """
    Convert PyArrow Date32 array to MAX Tensor of Int32 (zero-copy).
    
    Date32 stores dates as int32 (days since Unix epoch 1970-01-01).
    This is critical for TPC-H Q1 date filter: l_shipdate <= '1998-09-02' (epoch day 10471).
    
    Args:
        arrow_array: PyArrow Date32 array
        device: MAX device (CPU or Accelerator)
        device_ref: MAX DeviceRef for graph operations
        
    Returns:
        MAX Tensor of Int32 on the specified device
        
    Raises:
        TypeError: If array is not Date32 type
        ValueError: If array contains nulls (not yet supported)
    """
    # Validate type
    if not pa.types.is_date32(arrow_array.type):
        raise TypeError(f"Expected Date32 array, got {arrow_array.type}")
    
    # Reject nulls for now (TPC-H lineitem.l_shipdate has no nulls)
    if arrow_array.null_count > 0:
        raise ValueError(f"Date32 array contains {arrow_array.null_count} nulls. Nulls not yet supported.")
    
    # Zero-copy: Date32 is stored as int32 in buffer[1] (buffer[0] is validity bitmap)
    # buffers() returns [validity_bitmap, data_buffer]
    data_buffer = arrow_array.buffers()[1]
    
    # Create NumPy view over the raw buffer - no copy!
    np_view = np.frombuffer(data_buffer, dtype=np.int32)
    
    # Setup device if not provided
    if device is None:
        device = driver.Accelerator() if driver.accelerator_count() > 0 else driver.CPU()
    if device_ref is None:
        device_ref = DeviceRef.GPU(0) if isinstance(device, driver.Accelerator) else DeviceRef.CPU()
    
    # Create tensor: CPU first, then copy to device if needed
    cpu_tensor = driver.Tensor(np_view, driver.CPU())
    if isinstance(device, driver.Accelerator):
        return cpu_tensor.copy(device=device)
    return cpu_tensor


class MXFrame:
    """
    A DataFrame-like interface that bridges Arrow to MAX Engine with SQL support.
    """
    
    def __init__(self, arrow_data, device=None):
        """
        Initialize MXFrame from Arrow data.
        
        Args:
            arrow_data: PyArrow array or table
            device: MAX device (CPU or GPU). Auto-detects if None.
            
        Supports:
            - Float32/Float64 arrays (zero-copy)
            - Date32 arrays (zero-copy, converted to Int32 epoch days for TPC-H Q1)
        """
        if device is None:
            self.device = driver.Accelerator() if driver.accelerator_count() > 0 else driver.CPU()
        else:
            self.device = device
            
        self.device_ref = DeviceRef.GPU(0) if isinstance(self.device, driver.Accelerator) else DeviceRef.CPU()
        
        # Extract Arrow array from table if needed
        if isinstance(arrow_data, pa.Table):
            arrow_array = arrow_data.column(0).combine_chunks()
        elif isinstance(arrow_data, pa.ChunkedArray):
            arrow_array = arrow_data.combine_chunks()
        elif isinstance(arrow_data, pa.Array):
            arrow_array = arrow_data
        else:
            # Fallback: convert to numpy float32
            np_view = np.array(arrow_data, dtype=np.float32)
            self.dtype = DType.float32
            self.tensor = self._create_tensor(np_view)
            self.session = engine.InferenceSession(devices=[self.device])
            return
        
        # Type-specific conversion (zero-copy where possible)
        if pa.types.is_date32(arrow_array.type):
            # Date32 -> Int32 (zero-copy) for TPC-H Q1 date filtering
            self.tensor = date32_to_tensor(arrow_array, self.device, self.device_ref)
            self.dtype = DType.int32
        elif pa.types.is_floating(arrow_array.type) or pa.types.is_integer(arrow_array.type):
            # Numeric types: zero-copy
            np_view = arrow_array.to_numpy(zero_copy_only=True)
            self.dtype = DType.float32 if np_view.dtype == np.float32 else DType.float64
            if np_view.dtype == np.int32:
                self.dtype = DType.int32
            elif np_view.dtype == np.int64:
                self.dtype = DType.int64
            self.tensor = self._create_tensor(np_view)
        else:
            raise TypeError(f"Unsupported Arrow type: {arrow_array.type}")
        
        # Create inference session
        self.session = engine.InferenceSession(devices=[self.device])
    
    def _create_tensor(self, np_view):
        """Create MAX tensor from numpy array (CPU first, then copy to device)."""
        cpu_tensor = driver.Tensor(np_view, driver.CPU())
        if isinstance(self.device, driver.Accelerator):
            return cpu_tensor.copy(device=self.device)
        return cpu_tensor
    
    def _constant(self, value):
        """Helper to create a constant with correct device."""
        return ops.constant(value, dtype=DType.float32, device=self.device_ref)
    
    def sql(self, query: str):
        """
        Execute a SQL query on the data using MAX Graph operations.
        
        Supports:
        - WHERE clauses with comparison operators (>, <, >=, <=, =)
        - Simple SELECT with arithmetic operations
        
        Example:
            mxf.sql("SELECT * FROM data WHERE val > 0.5")
        """
        # Parse the SQL query
        parsed = sqlglot.parse_one(query)
        
        # Extract WHERE clause (filter)
        where_clause = parsed.find(exp.Where)
        if where_clause:
            condition = where_clause.this
            return self._build_filter_graph(condition)
        
        return self
    
    def _build_filter_graph(self, condition):
        """Build a MAX graph from SQL WHERE condition."""
        tensor_dtype = self.dtype  # Capture dtype for inner class
        
        # Define filtering computation based on SQL condition
        class SQLFilterCompute:
            def __init__(self, condition, device_ref, dtype):
                self.condition = condition
                self.device_ref = device_ref
                self.dtype = dtype
            
            def __call__(self, x):
                # Parse the condition and build the graph
                mask = self._parse_condition(x, self.condition)
                # Use correct zero value for dtype (int32 for dates, float for numeric)
                zero_val = 0 if self.dtype == DType.int32 else 0.0
                zero = ops.constant(zero_val, dtype=self.dtype, device=self.device_ref)
                # Apply filter: keep value if condition is true, else set to 0.0
                return ops.where(mask, x, zero)
            
            def _parse_threshold(self, value_str):
                """Parse threshold value with correct type."""
                if self.dtype == DType.int32:
                    return int(value_str)
                return float(value_str)
            
            def _parse_condition(self, x, cond):
                """Parse SQL condition into MAX ops."""
                if isinstance(cond, exp.GT):  # Greater than
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.greater(x, ops.constant(threshold, dtype=self.dtype, device=self.device_ref))
                
                elif isinstance(cond, exp.LT):  # Less than (use reverse greater)
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    # x < threshold is equivalent to threshold > x
                    return ops.greater(ops.constant(threshold, dtype=self.dtype, device=self.device_ref), x)
                
                elif isinstance(cond, exp.GTE):  # Greater than or equal
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.greater_equal(x, ops.constant(threshold, dtype=self.dtype, device=self.device_ref))
                
                elif isinstance(cond, exp.LTE):  # Less than or equal (use reverse greater_equal)
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    # x <= threshold is equivalent to threshold >= x
                    return ops.greater_equal(ops.constant(threshold, dtype=self.dtype, device=self.device_ref), x)
                
                elif isinstance(cond, exp.EQ):  # Equal
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.equal(x, ops.constant(threshold, dtype=self.dtype, device=self.device_ref))
                
                else:
                    raise NotImplementedError(f"Condition type {type(cond)} not yet supported")
        
        # Create and compile the filter graph
        filter_compute = SQLFilterCompute(condition, self.device_ref, tensor_dtype)
        filter_graph = Graph(
            "sql_filter",
            filter_compute,
            input_types=[TensorType(tensor_dtype, self.tensor.shape, self.device_ref)]
        )
        
        model = self.session.load(filter_graph)
        
        # Execute and return result
        result_tensor = model.execute(self.tensor)[0]
        
        # Return a new MXFrame with the result
        new_mxf = MXFrame.__new__(MXFrame)
        new_mxf.device = self.device
        new_mxf.device_ref = self.device_ref
        new_mxf.tensor = result_tensor
        new_mxf.dtype = self.dtype
        new_mxf.session = self.session
        return new_mxf
    
    def to_numpy(self):
        """Convert the tensor back to NumPy array."""
        if 'cuda' in str(self.tensor):
            return self.tensor.copy(device=driver.CPU()).to_numpy()
        return self.tensor.to_numpy()
    
    def __repr__(self):
        return f"MXFrame(shape={self.tensor.shape}, device={self.device})"
    
    # ========== TPC-H Q1 Optimized Filtering: Prefix Sum + Gather ==========
    # Instead of replacing non-matching values with zeros (wastes memory),
    # we compact the result to only include matching rows.
    
    def _build_mask_graph(self, condition):
        """
        Build a MAX graph that returns a boolean mask tensor.
        This is the first step for efficient filtering with compaction.
        
        For TPC-H Q1: l_shipdate <= 10471 returns [True, True, False, False, True]
        """
        tensor_dtype = self.dtype
        
        class MaskCompute:
            def __init__(self, condition, device_ref, dtype):
                self.condition = condition
                self.device_ref = device_ref
                self.dtype = dtype
            
            def __call__(self, x):
                return self._parse_condition(x, self.condition)
            
            def _parse_threshold(self, value_str):
                if self.dtype == DType.int32:
                    return int(value_str)
                return float(value_str)
            
            def _parse_condition(self, x, cond):
                if isinstance(cond, exp.GT):
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.greater(x, ops.constant(threshold, dtype=self.dtype, device=self.device_ref))
                elif isinstance(cond, exp.LT):
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.greater(ops.constant(threshold, dtype=self.dtype, device=self.device_ref), x)
                elif isinstance(cond, exp.GTE):
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.greater_equal(x, ops.constant(threshold, dtype=self.dtype, device=self.device_ref))
                elif isinstance(cond, exp.LTE):
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.greater_equal(ops.constant(threshold, dtype=self.dtype, device=self.device_ref), x)
                elif isinstance(cond, exp.EQ):
                    threshold = self._parse_threshold(cond.right.name if hasattr(cond.right, 'name') else str(cond.right))
                    return ops.equal(x, ops.constant(threshold, dtype=self.dtype, device=self.device_ref))
                else:
                    raise NotImplementedError(f"Condition type {type(cond)} not yet supported")
        
        mask_compute = MaskCompute(condition, self.device_ref, tensor_dtype)
        mask_graph = Graph(
            "mask_compute",
            mask_compute,
            input_types=[TensorType(tensor_dtype, self.tensor.shape, self.device_ref)]
        )
        
        model = self.session.load(mask_graph)
        return model.execute(self.tensor)[0]
    
    def _prefix_sum_and_gather(self, mask_tensor):
        """
        Task 1 & 2: Prefix Sum + Gather for efficient compaction.
        
        Given a boolean mask, compute:
        1. Count via MAX ops.sum (GPU-accelerated)
        2. Prefix sum via MAX ops.cumsum (GPU-accelerated) 
        3. Gather via MAX ops.gather (GPU-accelerated)
        
        For TPC-H Q1: This reduces a 6M row lineitem table to ~5.9M rows (96%)
        by removing rows where l_shipdate > '1998-09-02'.
        
        Returns: (compact_tensor, count) where count is number of matching rows
        """
        tensor_dtype = self.dtype
        device_ref = self.device_ref
        data_tensor = self.tensor
        
        # Step 1: Count matching elements using MAX ops (GPU-accelerated)
        # Cast bool to int32, then sum
        class CountCompute:
            def __init__(self, device_ref):
                self.device_ref = device_ref
            
            def __call__(self, mask):
                int_mask = ops.cast(mask, DType.int32)
                return ops.sum(int_mask)
        
        count_graph = Graph(
            "count_mask",
            CountCompute(device_ref),
            input_types=[TensorType(DType.bool, mask_tensor.shape, device_ref)]
        )
        count_model = self.session.load(count_graph)
        count_tensor = count_model.execute(mask_tensor)[0]
        
        # Transfer only the count (1 int32) to CPU - shape is (1,)
        if 'cuda' in str(count_tensor):
            count = int(count_tensor.copy(device=driver.CPU()).to_numpy()[0])
        else:
            count = int(count_tensor.to_numpy()[0])
        
        if count == 0:
            # No matches - return empty tensor
            empty_np = np.array([], dtype=np.int32 if tensor_dtype == DType.int32 else np.float32)
            return driver.Tensor(empty_np, driver.CPU()), 0
        
        input_size = data_tensor.shape[0]
        if count == input_size:
            # All match - return original tensor (no copy needed)
            return data_tensor, count
        
        # Step 2: Compute gather indices using prefix sum (GPU-accelerated)
        # For stream compaction: prefix_sum gives destination index for each True element
        # We then need to extract only the positions where mask is True
        #
        # mask =           [T, T, F, F, T]
        # int_mask =       [1, 1, 0, 0, 1]
        # prefix_exclusive = [0, 1, 2, 2, 2]  <- destination indices
        # 
        # To get source indices [0, 1, 4], we use the inverse:
        # Create range [0,1,2,3,4], filter by mask
        
        class IndexCompute:
            def __init__(self, device_ref, input_size, output_size):
                self.device_ref = device_ref
                self.input_size = input_size
                self.output_size = output_size
            
            def __call__(self, mask):
                # Cast mask to int32 for cumsum
                int_mask = ops.cast(mask, DType.int32)
                
                # Exclusive prefix sum: gives destination index for each element
                # [1,1,0,0,1] -> [0,1,2,2,2]
                prefix_sum = ops.cumsum(int_mask, axis=0, exclusive=True)
                
                # Create source indices [0, 1, 2, 3, 4, ...]
                indices = ops.range(0, self.input_size, 1, out_dim=self.input_size, dtype=DType.int32, device=self.device_ref)
                
                # We need to scatter indices to their prefix_sum positions where mask is True
                # But MAX doesn't have scatter, so we use a different approach:
                # Output indices at positions given by prefix_sum, only where mask is True
                
                # Alternative: Use where to mark invalid indices, then we'll filter on CPU
                # Actually, let's use a simpler approach for now that stays on GPU
                
                # mask_positions = where(mask, prefix_sum, -1)
                # This gives us [-1 at non-matching, dest_idx at matching]
                neg_one = ops.constant(-1, dtype=DType.int32, device=self.device_ref)
                dest_positions = ops.where(mask, prefix_sum, neg_one)
                
                # Return both: source indices and their destinations
                return indices, dest_positions
        
        index_graph = Graph(
            "compute_indices",
            IndexCompute(device_ref, input_size, count),
            input_types=[TensorType(DType.bool, mask_tensor.shape, device_ref)]
        )
        index_model = self.session.load(index_graph)
        source_indices, dest_positions = index_model.execute(mask_tensor)
        
        # Step 3: Build compacted indices on GPU
        # We have dest_positions = [0, 1, -1, -1, 2] (for mask [T,T,F,F,T])
        # We need to create inverse: [0, 1, 4] (source indices for each output position)
        #
        # For now, use a scatter simulation via argsort on dest_positions
        # Stable sort puts -1s at the end, valid positions at the start
        
        class GatherIndicesCompute:
            def __init__(self, device_ref, count):
                self.device_ref = device_ref
                self.count = count
            
            def __call__(self, source_idx, dest_pos):
                # Add input_size to -1 values so they sort to the end
                # Then argsort to get source indices in destination order
                large_val = ops.constant(2147483647, dtype=DType.int32, device=self.device_ref)
                neg_one = ops.constant(-1, dtype=DType.int32, device=self.device_ref)
                is_invalid = ops.equal(dest_pos, neg_one)
                sort_key = ops.where(is_invalid, large_val, dest_pos)
                
                # Argsort gives us source indices sorted by destination
                # Note: argsort returns int64, we need int32 for gather
                sorted_indices_i64 = ops.argsort(sort_key, ascending=True)
                sorted_indices = ops.cast(sorted_indices_i64, DType.int32)
                
                # Take only first 'count' indices (the valid ones)
                # Use slice/gather to extract [0:count]
                take_indices = ops.range(0, self.count, 1, out_dim=self.count, dtype=DType.int32, device=self.device_ref)
                gather_indices = ops.gather(sorted_indices, take_indices, axis=0)
                
                return gather_indices
        
        gather_idx_graph = Graph(
            "gather_indices",
            GatherIndicesCompute(device_ref, count),
            input_types=[
                TensorType(DType.int32, (input_size,), device_ref),
                TensorType(DType.int32, (input_size,), device_ref)
            ]
        )
        gather_idx_model = self.session.load(gather_idx_graph)
        final_indices = gather_idx_model.execute(source_indices, dest_positions)[0]
        
        # Step 4: Final gather - collect data at computed indices (GPU-accelerated)
        class GatherCompute:
            def __init__(self, dtype, device_ref):
                self.dtype = dtype
                self.device_ref = device_ref
            
            def __call__(self, data, indices):
                return ops.gather(data, indices, axis=0)
        
        gather_graph = Graph(
            "gather_compact",
            GatherCompute(tensor_dtype, device_ref),
            input_types=[
                TensorType(tensor_dtype, data_tensor.shape, device_ref),
                TensorType(DType.int32, (count,), device_ref)
            ]
        )
        
        gather_model = self.session.load(gather_graph)
        compact_tensor = gather_model.execute(data_tensor, final_indices)[0]
        
        return compact_tensor, count
    
    def where(self, query: str):
        """
        Task 3: Efficient filtering with compaction for TPC-H Q1.
        
        Unlike sql() which replaces non-matching values with zero,
        where() compacts the result to only include matching rows.
        
        Args:
            query: SQL WHERE condition, e.g., "l_shipdate <= 10471"
            
        Returns:
            Tuple of (new_mxframe, mask_tensor, count)
            - new_mxframe: MXFrame with only matching rows (compact)
            - mask_tensor: Boolean mask (for applying to other columns)
            - count: Number of matching rows
            
        Example:
            # TPC-H Q1 date filter
            compact_mxf, mask, count = mxf.where("l_shipdate <= 10471")
            print(f"Filtered {count} rows from {mxf.tensor.shape[0]}")
        """
        # Parse SQL to extract condition
        if not query.strip().upper().startswith("SELECT"):
            query = f"SELECT * FROM data WHERE {query}"
        
        parsed = sqlglot.parse_one(query)
        where_clause = parsed.find(exp.Where)
        
        if not where_clause:
            raise ValueError("No WHERE condition found in query")
        
        condition = where_clause.this
        
        # Step 1: Execute mask graph
        mask_tensor = self._build_mask_graph(condition)
        
        # Step 2 & 3: Prefix sum + gather for compaction
        compact_tensor, count = self._prefix_sum_and_gather(mask_tensor)
        
        # Step 4: Build result MXFrame
        new_mxf = MXFrame.__new__(MXFrame)
        new_mxf.device = self.device
        new_mxf.device_ref = self.device_ref
        new_mxf.tensor = compact_tensor
        new_mxf.dtype = self.dtype
        new_mxf.session = self.session
        
        return new_mxf, mask_tensor, count


def apply_mask(arrays, mask, device=None):
    """
    Apply a boolean mask to multiple arrays/tensors, compacting each.
    
    This is the key function for TPC-H Q1: after filtering by l_shipdate,
    we apply the same mask to all other columns (l_quantity, l_extendedprice, etc.)
    
    Args:
        arrays: Dict of {name: array} where array is numpy array, PyArrow array, or MAX tensor
        mask: Boolean mask tensor from MXFrame.where()
        device: MAX device (auto-detect if None)
        
    Returns:
        Dict of {name: compacted_numpy_array}
        
    Example:
        # Filter by date, then apply mask to all columns
        shipdate_mxf = MXFrame(lineitem['l_shipdate'])
        _, mask, count = shipdate_mxf.where("l_shipdate <= 10471")
        
        filtered = apply_mask({
            'l_quantity': lineitem['l_quantity'],
            'l_extendedprice': lineitem['l_extendedprice'],
            'l_discount': lineitem['l_discount'],
            'l_tax': lineitem['l_tax'],
            'l_returnflag': lineitem['l_returnflag'],
            'l_linestatus': lineitem['l_linestatus'],
        }, mask)
    """
    if device is None:
        device = driver.Accelerator() if driver.accelerator_count() > 0 else driver.CPU()
    device_ref = DeviceRef.GPU(0) if isinstance(device, driver.Accelerator) else DeviceRef.CPU()
    session = engine.InferenceSession(devices=[device])
    
    # Get mask as numpy for index computation
    if isinstance(mask, driver.Tensor):
        if 'cuda' in str(mask):
            mask_np = mask.copy(device=driver.CPU()).to_numpy()
        else:
            mask_np = mask.to_numpy()
    else:
        mask_np = np.asarray(mask, dtype=bool)
    
    # Compute gather indices once (reused for all columns)
    indices = np.nonzero(mask_np)[0].astype(np.int32)
    count = len(indices)
    
    if count == 0:
        # No matches - return empty arrays
        return {name: np.array([]) for name in arrays}
    
    if count == len(mask_np):
        # All match - return original arrays as numpy
        result = {}
        for name, arr in arrays.items():
            if isinstance(arr, driver.Tensor):
                if 'cuda' in str(arr):
                    result[name] = arr.copy(device=driver.CPU()).to_numpy()
                else:
                    result[name] = arr.to_numpy()
            elif isinstance(arr, pa.Array):
                result[name] = arr.to_numpy(zero_copy_only=False)
            else:
                result[name] = np.asarray(arr)
        return result
    
    # Create indices tensor
    indices_tensor = driver.Tensor(indices, driver.CPU())
    if isinstance(device, driver.Accelerator):
        indices_tensor = indices_tensor.copy(device=device)
    
    result = {}
    
    for name, arr in arrays.items():
        # Convert to numpy first
        if isinstance(arr, driver.Tensor):
            if 'cuda' in str(arr):
                np_arr = arr.copy(device=driver.CPU()).to_numpy()
            else:
                np_arr = arr.to_numpy()
        elif isinstance(arr, pa.Array):
            np_arr = arr.to_numpy(zero_copy_only=False)
        else:
            np_arr = np.asarray(arr)
        
        # Handle string arrays: use numpy indexing directly (CPU only)
        if np_arr.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
            result[name] = np_arr[indices]
            continue
        
        # Determine dtype for MAX
        if np_arr.dtype == np.float64:
            tensor_dtype = DType.float64
        elif np_arr.dtype == np.float32:
            tensor_dtype = DType.float32
        elif np_arr.dtype == np.int64:
            tensor_dtype = DType.int64
        elif np_arr.dtype == np.int32:
            tensor_dtype = DType.int32
        elif np_arr.dtype == np.int8:
            tensor_dtype = DType.int8
        else:
            # Fallback: convert to float32
            np_arr = np_arr.astype(np.float32)
            tensor_dtype = DType.float32
        
        # Create data tensor
        data_tensor = driver.Tensor(np_arr, driver.CPU())
        if isinstance(device, driver.Accelerator):
            data_tensor = data_tensor.copy(device=device)
        
        # Build gather graph
        class GatherColumn:
            def __init__(self, dtype, device_ref):
                self.dtype = dtype
                self.device_ref = device_ref
            
            def __call__(self, data, idx):
                return ops.gather(data, idx, axis=0)
        
        gather_graph = Graph(
            f"gather_{name}",
            GatherColumn(tensor_dtype, device_ref),
            input_types=[
                TensorType(tensor_dtype, (len(np_arr),), device_ref),
                TensorType(DType.int32, (count,), device_ref)
            ]
        )
        
        model = session.load(gather_graph)
        compact_tensor = model.execute(data_tensor, indices_tensor)[0]
        
        # Convert back to numpy
        if 'cuda' in str(compact_tensor):
            result[name] = compact_tensor.copy(device=driver.CPU()).to_numpy()
        else:
            result[name] = compact_tensor.to_numpy()
    
    return result


# ========== TPC-H Q1 Transposed Aggregator ==========
# Algorithm 2 from MojoFrame paper: 40x speedup via transposed group-by
#
# TPC-H Q1 groups by (l_returnflag, l_linestatus) - only 4 possible groups:
#   Group 0: (A, F) - returnflag=A, linestatus=F
#   Group 1: (N, F) - returnflag=N, linestatus=F
#   Group 2: (N, O) - returnflag=N, linestatus=O
#   Group 3: (R, F) - returnflag=R, linestatus=F
#
# Strategy: Encode groups as integers, use 4-mask broadcast aggregation.


def encode_returnflag(char_array):
    """
    Encode l_returnflag char column to int32: A=0, N=1, R=2.
    Optimized using numpy view on bytes.
    
    Args:
        char_array: PyArrow string array or numpy array of single chars ('A', 'N', 'R')
        
    Returns:
        NumPy int32 array
    """
    # Convert to numpy string array if needed
    if isinstance(char_array, pa.Array):
        str_arr = char_array.to_numpy(zero_copy_only=False)
    else:
        str_arr = np.asarray(char_array)
    
    # Get ASCII codes: A=65, N=78, R=82
    # Map: 65->0, 78->1, 82->2
    # Use lookup table indexed by (ascii - 65)
    lookup = np.zeros(26, dtype=np.int32)
    lookup[0] = 0   # A (65-65=0)
    lookup[13] = 1  # N (78-65=13)
    lookup[17] = 2  # R (82-65=17)
    
    # Get first byte of each string as uint8, then lookup
    if str_arr.dtype.kind == 'U':  # Unicode
        # Convert to bytes - each Unicode char is 4 bytes, we want first byte
        bytes_view = str_arr.view(np.uint32) 
        ascii_codes = (bytes_view & 0xFF).astype(np.int32)
    else:  # Already bytes/object
        ascii_codes = np.array([ord(s[0]) if s else 65 for s in str_arr], dtype=np.int32)
    
    return lookup[np.clip(ascii_codes - 65, 0, 25)]


def encode_linestatus(char_array):
    """
    Encode l_linestatus char column to int32: F=0, O=1.
    Optimized using numpy view on bytes.
    
    Args:
        char_array: PyArrow string array or numpy array of single chars ('F', 'O')
        
    Returns:
        NumPy int32 array
    """
    if isinstance(char_array, pa.Array):
        str_arr = char_array.to_numpy(zero_copy_only=False)
    else:
        str_arr = np.asarray(char_array)
    
    # Get ASCII codes: F=70, O=79
    # Map: 70->0, 79->1
    if str_arr.dtype.kind == 'U':  # Unicode
        bytes_view = str_arr.view(np.uint32)
        ascii_codes = (bytes_view & 0xFF).astype(np.int32)
    else:  # Already bytes/object
        ascii_codes = np.array([ord(s[0]) if s else 70 for s in str_arr], dtype=np.int32)
    
    # F=70->0, O=79->1: just check if it's 'O'
    return (ascii_codes == 79).astype(np.int32)


def compute_group_id(returnflag_encoded, linestatus_encoded):
    """
    Compute group ID from encoded returnflag and linestatus.
    
    Group encoding:
        (A=0, F=0) -> 0*2 + 0 = 0
        (N=1, F=0) -> 1*2 + 0 = 2
        (N=1, O=1) -> 1*2 + 1 = 3
        (R=2, F=0) -> 2*2 + 0 = 4
        
    But TPC-H Q1 only has 4 actual groups, so we use a direct mapping:
        (A, F) -> 0
        (N, F) -> 1
        (N, O) -> 2
        (R, F) -> 3
    
    Args:
        returnflag_encoded: int32 array (A=0, N=1, R=2)
        linestatus_encoded: int32 array (F=0, O=1)
        
    Returns:
        int32 array of group IDs (0-3)
    """
    # Skip astype if already int32
    if returnflag_encoded.dtype != np.int32:
        returnflag_encoded = returnflag_encoded.astype(np.int32)
    if linestatus_encoded.dtype != np.int32:
        linestatus_encoded = linestatus_encoded.astype(np.int32)
    
    # Direct computation: A(0)*2+F(0)=0, N(1)*2+F(0)=2->1, N(1)*2+O(1)=3->2, R(2)*2+F(0)=4->3
    raw_id = returnflag_encoded * 2 + linestatus_encoded
    
    # Use pre-computed lookup table (avoid allocation each time)
    # Index 5 = R(2)*2+O(1) shouldn't happen in valid data, but handle it
    remap = _GROUP_ID_REMAP
    return remap[np.clip(raw_id, 0, 5)]

# Pre-computed lookup table for group ID remapping
# Indices: 0=A+F, 1=A+O, 2=N+F, 3=N+O, 4=R+F, 5=R+O
_GROUP_ID_REMAP = np.array([0, 0, 1, 2, 3, 3], dtype=np.int32)


class Q1Accumulator:
    """
    TPC-H Q1 Accumulator: holds running totals for 4 groups.
    
    For each group, we track 8 aggregates:
        0: sum_qty        = SUM(l_quantity)
        1: sum_base_price = SUM(l_extendedprice)
        2: sum_disc_price = SUM(l_extendedprice * (1 - l_discount))
        3: sum_charge     = SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax))
        4: sum_disc       = SUM(l_discount)  (for avg_disc calculation)
        5: count_order    = COUNT(*)
        6: (reserved for avg calculations)
        7: (reserved for avg calculations)
        
    Shape: (4 groups, 6 values) stored as MAX tensors
    """
    
    # Group labels for output
    GROUP_LABELS = [
        ('A', 'F'),  # Group 0
        ('N', 'F'),  # Group 1
        ('N', 'O'),  # Group 2
        ('R', 'F'),  # Group 3
    ]
    
    def __init__(self, device=None):
        """Initialize accumulator with zeros."""
        if device is None:
            self.device = driver.Accelerator() if driver.accelerator_count() > 0 else driver.CPU()
        else:
            self.device = device
            
        self.device_ref = DeviceRef.GPU(0) if isinstance(self.device, driver.Accelerator) else DeviceRef.CPU()
        self.session = engine.InferenceSession(devices=[self.device])
        
        # Initialize result storage (computed after aggregation)
        self.results = None
    
    def aggregate(self, group_ids, l_quantity, l_extendedprice, l_discount, l_tax):
        """
        Perform TPC-H Q1 aggregation using 4-mask broadcast strategy.
        
        All inputs must be numpy arrays or MAX tensors of the same length.
        This runs 4 parallel mask-sum operations on GPU for each aggregate.
        
        Args:
            group_ids: int32 array of group IDs (0-3)
            l_quantity: float64 array
            l_extendedprice: float64 array
            l_discount: float64 array
            l_tax: float64 array
            
        Returns:
            Dict with results per group
        """
        # Convert inputs to tensors if needed
        def to_tensor(arr, dtype=DType.float32):
            if isinstance(arr, driver.Tensor):
                return arr
            np_arr = np.asarray(arr, dtype=np.float32 if dtype == DType.float32 else np.int32)
            t = driver.Tensor(np_arr, driver.CPU())
            if isinstance(self.device, driver.Accelerator):
                return t.copy(device=self.device)
            return t
        
        group_tensor = to_tensor(group_ids, DType.int32)
        qty_tensor = to_tensor(l_quantity)
        price_tensor = to_tensor(l_extendedprice)
        disc_tensor = to_tensor(l_discount)
        tax_tensor = to_tensor(l_tax)
        
        n_rows = group_ids.shape[0] if hasattr(group_ids, 'shape') else len(group_ids)
        
        # Precompute derived values on GPU
        # disc_price = l_extendedprice * (1 - l_discount)
        # charge = l_extendedprice * (1 - l_discount) * (1 + l_tax)
        
        class PrecomputeQ1:
            def __init__(self, device_ref):
                self.device_ref = device_ref
            
            def __call__(self, price, disc, tax):
                one = ops.constant(1.0, dtype=DType.float32, device=self.device_ref)
                one_minus_disc = ops.sub(one, disc)
                one_plus_tax = ops.add(one, tax)
                disc_price = ops.mul(price, one_minus_disc)
                charge = ops.mul(disc_price, one_plus_tax)
                return disc_price, charge
        
        precompute_graph = Graph(
            "q1_precompute",
            PrecomputeQ1(self.device_ref),
            input_types=[
                TensorType(DType.float32, (n_rows,), self.device_ref),
                TensorType(DType.float32, (n_rows,), self.device_ref),
                TensorType(DType.float32, (n_rows,), self.device_ref),
            ]
        )
        precompute_model = self.session.load(precompute_graph)
        disc_price_tensor, charge_tensor = precompute_model.execute(price_tensor, disc_tensor, tax_tensor)
        
        # 4-mask aggregation: for each group, mask and sum
        # This is O(4*N) but simple and runs on GPU
        results = {}
        
        for g in range(4):
            # Build mask and aggregation graph for this group
            class GroupAggregate:
                def __init__(self, group_id, device_ref):
                    self.group_id = group_id
                    self.device_ref = device_ref
                
                def __call__(self, groups, qty, price, disc, disc_price, charge):
                    # Create mask for this group
                    group_const = ops.constant(self.group_id, dtype=DType.int32, device=self.device_ref)
                    mask = ops.equal(groups, group_const)
                    
                    # Zero constant for masking
                    zero = ops.constant(0.0, dtype=DType.float32, device=self.device_ref)
                    
                    # Apply mask to each column and sum
                    masked_qty = ops.where(mask, qty, zero)
                    masked_price = ops.where(mask, price, zero)
                    masked_disc = ops.where(mask, disc, zero)
                    masked_disc_price = ops.where(mask, disc_price, zero)
                    masked_charge = ops.where(mask, charge, zero)
                    
                    # Cast mask to float for counting
                    mask_float = ops.cast(mask, DType.float32)
                    
                    # Sum all
                    sum_qty = ops.sum(masked_qty)
                    sum_price = ops.sum(masked_price)
                    sum_disc = ops.sum(masked_disc)
                    sum_disc_price = ops.sum(masked_disc_price)
                    sum_charge = ops.sum(masked_charge)
                    count = ops.sum(mask_float)
                    
                    return sum_qty, sum_price, sum_disc, sum_disc_price, sum_charge, count
            
            agg_graph = Graph(
                f"q1_group_{g}",
                GroupAggregate(g, self.device_ref),
                input_types=[
                    TensorType(DType.int32, (n_rows,), self.device_ref),
                    TensorType(DType.float32, (n_rows,), self.device_ref),
                    TensorType(DType.float32, (n_rows,), self.device_ref),
                    TensorType(DType.float32, (n_rows,), self.device_ref),
                    TensorType(DType.float32, (n_rows,), self.device_ref),
                    TensorType(DType.float32, (n_rows,), self.device_ref),
                ]
            )
            
            agg_model = self.session.load(agg_graph)
            sums = agg_model.execute(
                group_tensor, qty_tensor, price_tensor, disc_tensor,
                disc_price_tensor, charge_tensor
            )
            
            # Extract scalar values (shape is (1,))
            def get_scalar(t):
                if 'cuda' in str(t):
                    return float(t.copy(device=driver.CPU()).to_numpy()[0])
                return float(t.to_numpy()[0])
            
            rf, ls = self.GROUP_LABELS[g]
            count = get_scalar(sums[5])
            
            if count > 0:
                results[(rf, ls)] = {
                    'l_returnflag': rf,
                    'l_linestatus': ls,
                    'sum_qty': get_scalar(sums[0]),
                    'sum_base_price': get_scalar(sums[1]),
                    'sum_disc_price': get_scalar(sums[3]),
                    'sum_charge': get_scalar(sums[4]),
                    'avg_qty': get_scalar(sums[0]) / count,
                    'avg_price': get_scalar(sums[1]) / count,
                    'avg_disc': get_scalar(sums[2]) / count,
                    'count_order': int(count),
                }
        
        self.results = results
        return results
    
    def to_dataframe(self):
        """Convert results to a pandas-like dict for display."""
        if self.results is None:
            return None
        
        # Sort by (returnflag, linestatus) as per TPC-H Q1 spec
        rows = sorted(self.results.values(), key=lambda x: (x['l_returnflag'], x['l_linestatus']))
        return rows


# --- DEMO: SQL-Powered Filtering ---
if __name__ == "__main__":
    import time
    
    print("Generating 10 Million rows of data...")
    rows = 10_000_000
    raw_data = np.random.randn(rows).astype(np.float32)
    arrow_col = pa.array(raw_data)
    
    # Create MXFrame
    print("\nCreating MXFrame...")
    mxf = MXFrame(arrow_col)
    print(f"Device: {mxf.device}")
    
    # SQL Filter Benchmark
    print("\n--- SQL Filter Benchmark ---")
    sql_query = "SELECT * FROM data WHERE val > 0.5"
    print(f"Query: {sql_query}")
    
    # Warm up
    _ = mxf.sql(sql_query)
    
    # Timed execution
    start = time.perf_counter()
    result = mxf.sql(sql_query)
    elapsed = time.perf_counter() - start
    
    print(f"Execution time: {elapsed:.4f}s")
    print(f"Result: {result}")
    
    # Test different SQL conditions
    print("\n--- Testing Different SQL Conditions ---")
    
    queries = [
        "SELECT * FROM data WHERE val > 0.5",
        "SELECT * FROM data WHERE val < -0.5",
        "SELECT * FROM data WHERE val >= 1.0",
        "SELECT * FROM data WHERE val <= -1.0",
    ]
    
    for query in queries:
        start = time.perf_counter()
        result = mxf.sql(query)
        elapsed = time.perf_counter() - start
        print(f"{query}: {elapsed:.4f}s")


# ========== TPC-H Q1 End-to-End Pipeline ==========

def generate_tpch_lineitem(n_rows, seed=42):
    """
    Generate synthetic TPC-H lineitem data for Q1 benchmark.
    
    Columns generated (matching TPC-H spec):
        l_shipdate: Date32 (1992-01-01 to 1998-12-01)
        l_returnflag: char (A, N, R)
        l_linestatus: char (F, O) - F if shipped before 1995-06-17, else O
        l_quantity: float32 (1-50)
        l_extendedprice: float32 (quantity * part_price, ~100-10000)
        l_discount: float32 (0.00-0.10)
        l_tax: float32 (0.00-0.08)
    
    Args:
        n_rows: Number of rows to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dict of numpy arrays keyed by column name
    """
    np.random.seed(seed)
    
    # Date range: 1992-01-01 to 1998-12-01 (epoch days 8035 to 10561)
    start_epoch = 8035
    end_epoch = 10561
    l_shipdate = np.random.randint(start_epoch, end_epoch + 1, size=n_rows).astype(np.int32)
    
    # Return flag: A=0, N=1, R=2 (store as int32 directly for speed)
    # Distribution roughly: A=25%, N=50%, R=25%
    l_returnflag_enc = np.random.choice([0, 1, 2], n_rows, p=[0.25, 0.50, 0.25]).astype(np.int32)
    
    # Line status: F=0 if shipdate < 1995-06-17, else O=1
    # 1995-06-17 = epoch day 9299
    cutoff_epoch = 9299
    l_linestatus_enc = (l_shipdate >= cutoff_epoch).astype(np.int32)
    
    # Also store string versions for compatibility
    rf_map = {0: 'A', 1: 'N', 2: 'R'}
    ls_map = {0: 'F', 1: 'O'}
    l_returnflag = np.array([rf_map[x] for x in l_returnflag_enc], dtype='U1')
    l_linestatus = np.array([ls_map[x] for x in l_linestatus_enc], dtype='U1')
    
    # Quantity: 1-50 units
    l_quantity = np.random.uniform(1, 50, n_rows).astype(np.float32)
    
    # Extended price: quantity * random part price (roughly $2-200 per unit)
    part_price = np.random.uniform(2, 200, n_rows).astype(np.float32)
    l_extendedprice = (l_quantity * part_price).astype(np.float32)
    
    # Discount: 0-10%
    l_discount = np.random.uniform(0, 0.10, n_rows).astype(np.float32)
    
    # Tax: 0-8%
    l_tax = np.random.uniform(0, 0.08, n_rows).astype(np.float32)
    
    return {
        'l_shipdate': l_shipdate,
        'l_returnflag': l_returnflag,
        'l_linestatus': l_linestatus,
        'l_returnflag_enc': l_returnflag_enc,  # Pre-encoded for speed
        'l_linestatus_enc': l_linestatus_enc,  # Pre-encoded for speed
        'l_quantity': l_quantity,
        'l_extendedprice': l_extendedprice,
        'l_discount': l_discount,
        'l_tax': l_tax,
    }


def tpch_q1(lineitem, verbose=False):
    """
    Execute TPC-H Query 1 (Pricing Summary Report) using MXFrame.
    
    SQL equivalent:
        SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) as sum_qty,
            SUM(l_extendedprice) as sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) as sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
            AVG(l_quantity) as avg_qty,
            AVG(l_extendedprice) as avg_price,
            AVG(l_discount) as avg_disc,
            COUNT(*) as count_order
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    
    Args:
        lineitem: Dict of arrays with TPC-H lineitem columns
        verbose: Print timing for each step
        
    Returns:
        List of result dicts, one per group, sorted by (returnflag, linestatus)
    """
    import time
    timings = {}
    
    # Step 1: Filter by shipdate <= 1998-09-02 (epoch day 10471)
    # 1998-12-01 - 90 days = 1998-09-02
    t0 = time.perf_counter()
    
    l_shipdate = lineitem['l_shipdate']
    if isinstance(l_shipdate, np.ndarray) and l_shipdate.dtype == np.int32:
        # Already epoch days
        shipdate_arr = pa.array(l_shipdate, type=pa.date32())
    else:
        shipdate_arr = pa.array(l_shipdate, type=pa.date32())
    
    shipdate_mxf = MXFrame(shipdate_arr)
    _, mask, filter_count = shipdate_mxf.where("l_shipdate <= 10471")
    
    timings['filter'] = time.perf_counter() - t0
    if verbose:
        print(f"Step 1 - Filter: {timings['filter']*1000:.1f}ms ({filter_count:,} rows pass)")
    
    # Step 2: Apply mask to all columns
    t0 = time.perf_counter()
    
    filtered = apply_mask({
        'l_quantity': lineitem['l_quantity'],
        'l_extendedprice': lineitem['l_extendedprice'],
        'l_discount': lineitem['l_discount'],
        'l_tax': lineitem['l_tax'],
        'l_returnflag': lineitem['l_returnflag'],
        'l_linestatus': lineitem['l_linestatus'],
    }, mask)
    
    timings['apply_mask'] = time.perf_counter() - t0
    if verbose:
        print(f"Step 2 - Apply mask: {timings['apply_mask']*1000:.1f}ms")
    
    # Step 3: Encode groups
    t0 = time.perf_counter()
    
    rf_enc = encode_returnflag(filtered['l_returnflag'])
    ls_enc = encode_linestatus(filtered['l_linestatus'])
    group_ids = compute_group_id(rf_enc, ls_enc)
    
    timings['encode'] = time.perf_counter() - t0
    if verbose:
        print(f"Step 3 - Encode groups: {timings['encode']*1000:.1f}ms")
    
    # Step 4: Aggregate
    t0 = time.perf_counter()
    
    acc = Q1Accumulator()
    results = acc.aggregate(
        group_ids,
        filtered['l_quantity'],
        filtered['l_extendedprice'],
        filtered['l_discount'],
        filtered['l_tax']
    )
    
    timings['aggregate'] = time.perf_counter() - t0
    if verbose:
        print(f"Step 4 - Aggregate: {timings['aggregate']*1000:.1f}ms")
    
    # Sort by (returnflag, linestatus) per TPC-H spec
    result_rows = acc.to_dataframe()
    
    timings['total'] = sum(timings.values())
    if verbose:
        print(f"Total: {timings['total']*1000:.1f}ms")
    
    return result_rows, timings


class CompiledQ1:
    """
    Pre-compiled TPC-H Q1 pipeline for maximum performance.
    
    All MAX graphs are compiled once at construction time.
    Subsequent executions only run inference - no compilation overhead.
    
    Usage:
        # Compile once (takes ~30-60 seconds)
        q1 = CompiledQ1(n_rows=1_000_000)
        
        # Execute many times (fast!)
        results = q1.execute(lineitem)
        results = q1.execute(another_lineitem)  # Reuses compiled graphs
    """
    
    # Group labels for output
    GROUP_LABELS = [
        ('A', 'F'),  # Group 0
        ('N', 'F'),  # Group 1
        ('N', 'O'),  # Group 2
        ('R', 'F'),  # Group 3
    ]
    
    def __init__(self, n_rows, verbose=True):
        """
        Pre-compile all graphs for TPC-H Q1 with given input size.
        
        Args:
            n_rows: Expected number of input rows
            verbose: Print compilation progress
        """
        import time
        self.n_rows = n_rows
        self.verbose = verbose
        
        # Setup device
        self.device = driver.Accelerator() if driver.accelerator_count() > 0 else driver.CPU()
        self.device_ref = DeviceRef.GPU(0) if isinstance(self.device, driver.Accelerator) else DeviceRef.CPU()
        self.session = engine.InferenceSession(devices=[self.device])
        
        if verbose:
            print(f"Compiling TPC-H Q1 for {n_rows:,} rows...")
            t_start = time.perf_counter()
        
        # Estimate filtered row count (~96.5% pass the date filter typically)
        self.filtered_rows = int(n_rows * 0.965)
        
        self._compile_all_graphs()
        
        if verbose:
            elapsed = time.perf_counter() - t_start
            print(f"Compilation complete in {elapsed:.1f}s")
    
    def _compile_all_graphs(self):
        """Compile all graphs needed for Q1."""
        import time
        n = self.n_rows
        nf = self.filtered_rows
        
        # === Graph 1: Date mask ===
        if self.verbose:
            print("  Compiling: date_mask...", end=" ", flush=True)
            t0 = time.perf_counter()
        
        class DateMask:
            def __init__(self, device_ref, cutoff=10471):
                self.device_ref = device_ref
                self.cutoff = cutoff
            
            def __call__(self, shipdate):
                threshold = ops.constant(self.cutoff, dtype=DType.int32, device=self.device_ref)
                return ops.greater_equal(threshold, shipdate)
        
        mask_graph = Graph(
            "q1_date_mask",
            DateMask(self.device_ref),
            input_types=[TensorType(DType.int32, (n,), self.device_ref)]
        )
        self.mask_model = self.session.load(mask_graph)
        
        if self.verbose:
            print(f"{time.perf_counter()-t0:.1f}s")
        
        # === Graph 2: Count matching rows ===
        if self.verbose:
            print("  Compiling: count...", end=" ", flush=True)
            t0 = time.perf_counter()
        
        class CountMask:
            def __init__(self, device_ref):
                self.device_ref = device_ref
            
            def __call__(self, mask):
                int_mask = ops.cast(mask, DType.int32)
                return ops.sum(int_mask)
        
        count_graph = Graph(
            "q1_count",
            CountMask(self.device_ref),
            input_types=[TensorType(DType.bool, (n,), self.device_ref)]
        )
        self.count_model = self.session.load(count_graph)
        
        if self.verbose:
            print(f"{time.perf_counter()-t0:.1f}s")
        
        # === Graph 3: Gather float32 columns ===
        if self.verbose:
            print("  Compiling: gather_float32...", end=" ", flush=True)
            t0 = time.perf_counter()
        
        class GatherFloat32:
            def __call__(self, data, indices):
                return ops.gather(data, indices, axis=0)
        
        # We'll compile for multiple output sizes and pick at runtime
        # For now, use estimated filtered_rows
        gather_f32_graph = Graph(
            "q1_gather_f32",
            GatherFloat32(),
            input_types=[
                TensorType(DType.float32, (n,), self.device_ref),
                TensorType(DType.int32, (nf,), self.device_ref)
            ]
        )
        self.gather_f32_model = self.session.load(gather_f32_graph)
        
        if self.verbose:
            print(f"{time.perf_counter()-t0:.1f}s")
        
        # === Graph 4: Precompute disc_price and charge ===
        if self.verbose:
            print("  Compiling: precompute...", end=" ", flush=True)
            t0 = time.perf_counter()
        
        class Precompute:
            def __init__(self, device_ref):
                self.device_ref = device_ref
            
            def __call__(self, price, disc, tax):
                one = ops.constant(1.0, dtype=DType.float32, device=self.device_ref)
                one_minus_disc = ops.sub(one, disc)
                one_plus_tax = ops.add(one, tax)
                disc_price = ops.mul(price, one_minus_disc)
                charge = ops.mul(disc_price, one_plus_tax)
                return disc_price, charge
        
        precompute_graph = Graph(
            "q1_precompute",
            Precompute(self.device_ref),
            input_types=[
                TensorType(DType.float32, (nf,), self.device_ref),
                TensorType(DType.float32, (nf,), self.device_ref),
                TensorType(DType.float32, (nf,), self.device_ref),
            ]
        )
        self.precompute_model = self.session.load(precompute_graph)
        
        if self.verbose:
            print(f"{time.perf_counter()-t0:.1f}s")
        
        # === Graph 5: Fused 4-group aggregation ===
        if self.verbose:
            print("  Compiling: aggregation...", end=" ", flush=True)
            t0 = time.perf_counter()
        
        class FusedAggregate:
            """Compute all 4 group aggregations in a single graph."""
            def __init__(self, device_ref):
                self.device_ref = device_ref
            
            def __call__(self, groups, qty, price, disc, disc_price, charge):
                results = []
                zero = ops.constant(0.0, dtype=DType.float32, device=self.device_ref)
                
                for g in range(4):
                    group_const = ops.constant(g, dtype=DType.int32, device=self.device_ref)
                    mask = ops.equal(groups, group_const)
                    
                    masked_qty = ops.where(mask, qty, zero)
                    masked_price = ops.where(mask, price, zero)
                    masked_disc = ops.where(mask, disc, zero)
                    masked_disc_price = ops.where(mask, disc_price, zero)
                    masked_charge = ops.where(mask, charge, zero)
                    mask_float = ops.cast(mask, DType.float32)
                    
                    results.extend([
                        ops.sum(masked_qty),
                        ops.sum(masked_price),
                        ops.sum(masked_disc),
                        ops.sum(masked_disc_price),
                        ops.sum(masked_charge),
                        ops.sum(mask_float),
                    ])
                
                # Returns 24 values: 6 per group × 4 groups
                return tuple(results)
        
        agg_graph = Graph(
            "q1_aggregate",
            FusedAggregate(self.device_ref),
            input_types=[
                TensorType(DType.int32, (nf,), self.device_ref),
                TensorType(DType.float32, (nf,), self.device_ref),
                TensorType(DType.float32, (nf,), self.device_ref),
                TensorType(DType.float32, (nf,), self.device_ref),
                TensorType(DType.float32, (nf,), self.device_ref),
                TensorType(DType.float32, (nf,), self.device_ref),
            ]
        )
        self.agg_model = self.session.load(agg_graph)
        
        if self.verbose:
            print(f"{time.perf_counter()-t0:.1f}s")
    
    def execute(self, lineitem):
        """
        Execute pre-compiled TPC-H Q1 on lineitem data.
        
        Args:
            lineitem: Dict with columns l_shipdate, l_quantity, l_extendedprice,
                      l_discount, l_tax, l_returnflag, l_linestatus
                      
        Returns:
            Tuple of (results_list, timings_dict)
        """
        import time
        timings = {}
        
        # === Step 1: Create shipdate tensor ===
        t0 = time.perf_counter()
        
        shipdate = lineitem['l_shipdate']
        if not isinstance(shipdate, np.ndarray):
            shipdate = np.asarray(shipdate, dtype=np.int32)
        
        shipdate_tensor = driver.Tensor(shipdate.astype(np.int32), driver.CPU())
        if isinstance(self.device, driver.Accelerator):
            shipdate_tensor = shipdate_tensor.copy(device=self.device)
        
        timings['prep'] = time.perf_counter() - t0
        
        # === Step 2: Compute mask ===
        t0 = time.perf_counter()
        
        mask_tensor = self.mask_model.execute(shipdate_tensor)[0]
        
        timings['mask'] = time.perf_counter() - t0
        
        # === Step 3: Get mask as numpy for indexing ===
        t0 = time.perf_counter()
        
        if 'cuda' in str(mask_tensor):
            mask_np = mask_tensor.copy(device=driver.CPU()).to_numpy()
        else:
            mask_np = mask_tensor.to_numpy()
        
        indices = np.nonzero(mask_np)[0].astype(np.int32)
        actual_count = len(indices)
        
        timings['indices'] = time.perf_counter() - t0
        
        # === Step 4: Filter numeric columns ===
        t0 = time.perf_counter()
        
        # Prepare indices tensor
        indices_tensor = driver.Tensor(indices, driver.CPU())
        if isinstance(self.device, driver.Accelerator):
            indices_tensor = indices_tensor.copy(device=self.device)
        
        # Filter each numeric column using numpy (fast for CPU data)
        qty = lineitem['l_quantity'][indices].astype(np.float32)
        price = lineitem['l_extendedprice'][indices].astype(np.float32)
        disc = lineitem['l_discount'][indices].astype(np.float32)
        tax = lineitem['l_tax'][indices].astype(np.float32)
        
        timings['filter'] = time.perf_counter() - t0
        
        # === Step 5: Filter and encode group columns ===
        t0 = time.perf_counter()
        
        # Use pre-encoded columns if available (much faster)
        if 'l_returnflag_enc' in lineitem and 'l_linestatus_enc' in lineitem:
            rf_enc = lineitem['l_returnflag_enc'][indices]
            ls_enc = lineitem['l_linestatus_enc'][indices]
        else:
            rf_filtered = lineitem['l_returnflag'][indices]
            ls_filtered = lineitem['l_linestatus'][indices]
            rf_enc = encode_returnflag(rf_filtered)
            ls_enc = encode_linestatus(ls_filtered)
        
        group_ids = compute_group_id(rf_enc, ls_enc)
        
        group_tensor = driver.Tensor(group_ids, driver.CPU())
        if isinstance(self.device, driver.Accelerator):
            group_tensor = group_tensor.copy(device=self.device)
        
        timings['encode'] = time.perf_counter() - t0
        
        # === Step 6: Precompute derived values ===
        t0 = time.perf_counter()
        
        # Need to recompile if size doesn't match (use dynamic approach)
        disc_price_np = price * (1 - disc)
        charge_np = disc_price_np * (1 + tax)
        
        timings['precompute'] = time.perf_counter() - t0
        
        # === Step 7: Aggregate using bincount (vectorized) ===
        t0 = time.perf_counter()
        
        # Use bincount for fast group-by aggregation
        n_groups = 4
        
        # Count per group
        counts = np.bincount(group_ids, minlength=n_groups)
        
        # Sum per group using bincount with weights
        sum_qty = np.bincount(group_ids, weights=qty.astype(np.float64), minlength=n_groups)
        sum_price = np.bincount(group_ids, weights=price.astype(np.float64), minlength=n_groups)
        sum_disc = np.bincount(group_ids, weights=disc.astype(np.float64), minlength=n_groups)
        sum_disc_price = np.bincount(group_ids, weights=disc_price_np.astype(np.float64), minlength=n_groups)
        sum_charge = np.bincount(group_ids, weights=charge_np.astype(np.float64), minlength=n_groups)
        
        results = {}
        for g in range(n_groups):
            count = int(counts[g])
            if count > 0:
                rf, ls = self.GROUP_LABELS[g]
                results[(rf, ls)] = {
                    'l_returnflag': rf,
                    'l_linestatus': ls,
                    'sum_qty': float(sum_qty[g]),
                    'sum_base_price': float(sum_price[g]),
                    'sum_disc_price': float(sum_disc_price[g]),
                    'sum_charge': float(sum_charge[g]),
                    'avg_qty': float(sum_qty[g]) / count,
                    'avg_price': float(sum_price[g]) / count,
                    'avg_disc': float(sum_disc[g]) / count,
                    'count_order': count,
                }
        
        timings['aggregate'] = time.perf_counter() - t0
        
        # Sort by (returnflag, linestatus)
        result_rows = sorted(results.values(), key=lambda x: (x['l_returnflag'], x['l_linestatus']))
        
        timings['total'] = sum(timings.values())
        
        return result_rows, timings


class MaxNativeQ1:
    """
    Fully MAX-native TPC-H Q1 implementation.
    
    All operations run on MAX Engine with zero numpy fallbacks in the hot path.
    Uses mask-based aggregation on full arrays to avoid dynamic shape issues.
    
    Strategy:
    - Keep all data as full-size tensors (no compaction)
    - Use mask to zero out non-matching rows before aggregation
    - Combine date mask + group mask for each of 4 groups
    - Single fused graph does: mask → precompute → aggregate
    
    Usage:
        q1 = MaxNativeQ1(n_rows=1_000_000)
        results = q1.execute(lineitem)
    """
    
    GROUP_LABELS = [('A', 'F'), ('N', 'F'), ('N', 'O'), ('R', 'F')]
    
    def __init__(self, n_rows, verbose=True):
        """Pre-compile the fused Q1 graph."""
        import time
        self.n_rows = n_rows
        self.verbose = verbose
        
        # Setup device
        # NOTE: GPU execution fails with ops.sum in MAX nightly 26.1.0.dev2026011405
        # Falling back to CPU which is still very fast
        self.device = driver.CPU()
        self.device_ref = DeviceRef.CPU()
        self.device_name = "CPU"
        
        # Uncomment to try GPU when the nightly is fixed:
        # if driver.accelerator_count() > 0:
        #     self.device = driver.Accelerator()
        #     self.device_ref = DeviceRef.GPU(0)
        #     self.device_name = "GPU"
        
        self.session = engine.InferenceSession(devices=[self.device])
        
        if verbose:
            print(f"Compiling MAX-native TPC-H Q1 for {n_rows:,} rows on {self.device_name}...")
            t_start = time.perf_counter()
        
        self._compile_fused_graph()
        
        if verbose:
            print(f"Compilation complete in {time.perf_counter() - t_start:.1f}s")
    
    def _compile_fused_graph(self):
        """
        Compile a single fused graph that does everything:
        - Date filtering (mask)
        - Group encoding (via pre-encoded inputs)
        - Precompute disc_price and charge
        - 4-group masked aggregation
        """
        import time
        n = self.n_rows
        device_ref = self.device_ref
        
        class FusedQ1Graph:
            """
            Fused TPC-H Q1 computation graph.
            
            Inputs:
                shipdate: int32[n] - epoch days
                rf_enc: int32[n] - returnflag encoded (A=0, N=1, R=2)
                ls_enc: int32[n] - linestatus encoded (F=0, O=1)
                qty: float32[n]
                price: float32[n]
                disc: float32[n]
                tax: float32[n]
            
            Outputs:
                24 scalars: 6 aggregates × 4 groups
                [sum_qty, sum_price, sum_disc, sum_disc_price, sum_charge, count] × 4
            """
            def __init__(self, device_ref, date_cutoff=10471):
                self.device_ref = device_ref
                self.date_cutoff = date_cutoff  # 1998-09-02 as epoch days
            
            def __call__(self, shipdate, rf_enc, ls_enc, qty, price, disc, tax):
                # Constants
                zero_f = ops.constant(0.0, dtype=DType.float32, device=self.device_ref)
                one_f = ops.constant(1.0, dtype=DType.float32, device=self.device_ref)
                cutoff = ops.constant(self.date_cutoff, dtype=DType.int32, device=self.device_ref)
                
                # Date mask: shipdate <= cutoff (as float: 1.0 if true, 0.0 if false)
                date_cond = ops.greater_equal(cutoff, shipdate)
                date_mask = ops.cast(date_cond, DType.float32)
                
                # Precompute derived columns (full array, will be masked later)
                one_minus_disc = ops.sub(one_f, disc)
                one_plus_tax = ops.add(one_f, tax)
                disc_price = ops.mul(price, one_minus_disc)
                charge = ops.mul(disc_price, one_plus_tax)
                
                # Compute group IDs: rf * 2 + ls, then remap
                # Group 0: A(0)*2 + F(0) = 0
                # Group 1: N(1)*2 + F(0) = 2 -> remap to 1
                # Group 2: N(1)*2 + O(1) = 3 -> remap to 2
                # Group 3: R(2)*2 + F(0) = 4 -> remap to 3
                two = ops.constant(2, dtype=DType.int32, device=self.device_ref)
                raw_group = ops.add(ops.mul(rf_enc, two), ls_enc)
                
                results = []
                
                # For each of 4 output groups, compute masked aggregation
                # Using arithmetic instead of logical ops for GPU compatibility
                # raw=0 (A+F) -> group 0
                # raw=1 (A+O) -> group 0
                # raw=2 (N+F) -> group 1
                # raw=3 (N+O) -> group 2
                # raw=4 (R+F) -> group 3
                # raw=5 (R+O) -> group 3
                
                for g in range(4):
                    # Build mask for all raw values that map to this group
                    if g == 0:  # A+F or A+O (raw 0 or 1)
                        raw_vals = [0, 1]
                    elif g == 1:  # N+F (raw 2)
                        raw_vals = [2]
                    elif g == 2:  # N+O (raw 3)
                        raw_vals = [3]
                    else:  # R+F or R+O (raw 4 or 5)
                        raw_vals = [4, 5]
                    
                    # Build group mask using arithmetic (sum of equal checks)
                    # Each equal returns bool, cast to float, sum gives >= 1 if any match
                    group_mask = None
                    for raw_v in raw_vals:
                        val = ops.constant(raw_v, dtype=DType.int32, device=self.device_ref)
                        eq_cond = ops.equal(raw_group, val)
                        eq_float = ops.cast(eq_cond, DType.float32)
                        if group_mask is None:
                            group_mask = eq_float
                        else:
                            group_mask = ops.add(group_mask, eq_float)
                    
                    # Combined mask: date_mask * group_mask (both are float, 1.0 or 0.0)
                    combined_mask = ops.mul(date_mask, group_mask)
                    
                    # Apply mask to each column via multiplication (GPU-friendly)
                    masked_qty = ops.mul(combined_mask, qty)
                    masked_price = ops.mul(combined_mask, price)
                    masked_disc = ops.mul(combined_mask, disc)
                    masked_disc_price = ops.mul(combined_mask, disc_price)
                    masked_charge = ops.mul(combined_mask, charge)
                    
                    # Aggregate
                    results.extend([
                        ops.sum(masked_qty),
                        ops.sum(masked_price),
                        ops.sum(masked_disc),
                        ops.sum(masked_disc_price),
                        ops.sum(masked_charge),
                        ops.sum(combined_mask),  # count (sum of 1.0s)
                    ])
                
                return tuple(results)
        
        if self.verbose:
            print("  Compiling fused Q1 graph...", end=" ", flush=True)
            t0 = time.perf_counter()
        
        graph = Graph(
            "max_native_q1",
            FusedQ1Graph(device_ref),
            input_types=[
                TensorType(DType.int32, (n,), device_ref),    # shipdate
                TensorType(DType.int32, (n,), device_ref),    # rf_enc
                TensorType(DType.int32, (n,), device_ref),    # ls_enc
                TensorType(DType.float32, (n,), device_ref),  # qty
                TensorType(DType.float32, (n,), device_ref),  # price
                TensorType(DType.float32, (n,), device_ref),  # disc
                TensorType(DType.float32, (n,), device_ref),  # tax
            ]
        )
        self.model = self.session.load(graph)
        
        if self.verbose:
            print(f"{time.perf_counter() - t0:.1f}s")
    
    def execute(self, lineitem):
        """
        Execute MAX-native TPC-H Q1.
        
        Args:
            lineitem: Dict with l_shipdate, l_returnflag_enc, l_linestatus_enc,
                      l_quantity, l_extendedprice, l_discount, l_tax
        
        Returns:
            Tuple of (results_list, timings_dict)
        """
        import time
        timings = {}
        
        # === Step 1: Prepare input tensors ===
        t0 = time.perf_counter()
        
        # Get arrays (use pre-encoded if available)
        shipdate = np.asarray(lineitem['l_shipdate'], dtype=np.int32)
        
        if 'l_returnflag_enc' in lineitem:
            rf_enc = np.asarray(lineitem['l_returnflag_enc'], dtype=np.int32)
            ls_enc = np.asarray(lineitem['l_linestatus_enc'], dtype=np.int32)
        else:
            rf_enc = encode_returnflag(lineitem['l_returnflag'])
            ls_enc = encode_linestatus(lineitem['l_linestatus'])
        
        qty = np.asarray(lineitem['l_quantity'], dtype=np.float32)
        price = np.asarray(lineitem['l_extendedprice'], dtype=np.float32)
        disc = np.asarray(lineitem['l_discount'], dtype=np.float32)
        tax = np.asarray(lineitem['l_tax'], dtype=np.float32)
        
        # Create MAX tensors
        shipdate_t = driver.Tensor(shipdate, self.device)
        rf_t = driver.Tensor(rf_enc, self.device)
        ls_t = driver.Tensor(ls_enc, self.device)
        qty_t = driver.Tensor(qty, self.device)
        price_t = driver.Tensor(price, self.device)
        disc_t = driver.Tensor(disc, self.device)
        tax_t = driver.Tensor(tax, self.device)
        
        timings['prep'] = time.perf_counter() - t0
        
        # === Step 2: Execute fused graph ===
        t0 = time.perf_counter()
        
        outputs = self.model.execute(shipdate_t, rf_t, ls_t, qty_t, price_t, disc_t, tax_t)
        
        timings['execute'] = time.perf_counter() - t0
        
        # === Step 3: Extract results ===
        t0 = time.perf_counter()
        
        # Convert outputs to Python floats (copy to CPU if on GPU)
        values = []
        for out in outputs:
            if self.device_name == "GPU":
                cpu_tensor = out.copy(device=driver.CPU())
                values.append(float(cpu_tensor.to_numpy()[0]))
            else:
                values.append(float(out.to_numpy()[0]))
        
        results = {}
        for g in range(4):
            base = g * 6
            sum_qty = values[base + 0]
            sum_price = values[base + 1]
            sum_disc = values[base + 2]
            sum_disc_price = values[base + 3]
            sum_charge = values[base + 4]
            count = int(values[base + 5])
            
            if count > 0:
                rf, ls = self.GROUP_LABELS[g]
                results[(rf, ls)] = {
                    'l_returnflag': rf,
                    'l_linestatus': ls,
                    'sum_qty': sum_qty,
                    'sum_base_price': sum_price,
                    'sum_disc_price': sum_disc_price,
                    'sum_charge': sum_charge,
                    'avg_qty': sum_qty / count,
                    'avg_price': sum_price / count,
                    'avg_disc': sum_disc / count,
                    'count_order': count,
                }
        
        timings['extract'] = time.perf_counter() - t0
        
        result_rows = sorted(results.values(), key=lambda x: (x['l_returnflag'], x['l_linestatus']))
        timings['total'] = sum(timings.values())
        
        return result_rows, timings
