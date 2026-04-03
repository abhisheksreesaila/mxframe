import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime key_dtype = DType.int32
comptime out_dtype = DType.int32


# -- CPU: direct-address count-based join match counting --------------------

fn _join_count_cpu(
    match_counts: ManagedTensorSlice[mut=True, dtype=out_dtype, rank=1],
    left_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
):
    """Count how many right rows match each left row's key."""
    var n_left = left_keys.dim_size(0)
    var n_right = right_keys.dim_size(0)

    if n_left == 0:
        return

    # Find max key value across both sides
    var max_key = Int(left_keys[0])
    for i in range(n_left):
        var k = Int(left_keys[i])
        if k > max_key:
            max_key = k
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k > max_key:
            max_key = k

    # Allocate and zero right-side frequency table
    var table_size = max_key + 1
    var right_count = List[Int32](capacity=table_size)
    for i in range(table_size):
        right_count.append(Int32(0))

    # Count right-side key frequencies
    for i in range(n_right):
        var k = Int(right_keys[i])
        right_count[k] += 1

    # For each left row, look up how many right rows match
    for i in range(n_left):
        var k = Int(left_keys[i])
        match_counts[i] = right_count[k]


# -- GPU: direct-address count-based join match counting --------------------

fn _join_count_gpu(
    match_counts: ManagedTensorSlice[mut=True, dtype=out_dtype, rank=1],
    left_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    max_key_buf: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_count_buf: ManagedTensorSlice[mut=True, dtype=out_dtype, rank=1],
    ctx: DeviceContextPtr,
) raises:
    """GPU join count: uses pre-allocated right_count buffer."""
    comptime BLOCK_SIZE = 256
    var n_left = left_keys.dim_size(0)
    var n_right = right_keys.dim_size(0)
    # Derive table_size from the buffer dimension — NOT from max_key_buf data,
    # because max_key_buf lives in GPU memory and cannot be read from host code.
    var table_size = right_count_buf.dim_size(0)

    if n_left == 0:
        return

    # Phase 1: Zero the right_count buffer
    var buf_size = right_count_buf.dim_size(0)

    @parameter
    fn zero_kernel(buf_size: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(buf_size):
            right_count_buf[Int(tid)] = 0

    if buf_size > 0:
        var zero_blocks = ceildiv(buf_size, BLOCK_SIZE)
        ctx.get_device_context().enqueue_function_experimental[zero_kernel](
            buf_size,
            grid_dim=zero_blocks, block_dim=BLOCK_SIZE,
        )

    # Phase 2: Count right-side key frequencies (atomic add)
    @parameter
    fn count_right_kernel(n_right: Int, table_size: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(n_right):
            var k = Int(right_keys[Int(tid)])
            if k >= 0 and k < table_size:
                var ptr = right_count_buf.unsafe_ptr() + k
                _ = Atomic.fetch_add(ptr, Int32(1))

    if n_right > 0:
        var blocks_r = ceildiv(n_right, BLOCK_SIZE)
        ctx.get_device_context().enqueue_function_experimental[count_right_kernel](
            n_right, table_size,
            grid_dim=blocks_r, block_dim=BLOCK_SIZE,
        )

    # Phase 3: Probe -- each left row looks up its match count
    @parameter
    fn probe_kernel(n_left: Int, table_size: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(n_left):
            var k = Int(left_keys[Int(tid)])
            if k >= 0 and k < table_size:
                match_counts[Int(tid)] = right_count_buf[k]
            else:
                match_counts[Int(tid)] = 0

    var blocks_l = ceildiv(n_left, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[probe_kernel](
        n_left, table_size,
        grid_dim=blocks_l, block_dim=BLOCK_SIZE,
    )


@compiler.register("join_count_cpu")
struct JoinCountCPU:
    """CPU-only join count kernel (3 inputs, 1 output)."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        match_counts: OutputTensor[dtype=out_dtype, rank=1],
        left_keys: InputTensor[dtype=key_dtype, rank=1],
        right_keys: InputTensor[dtype=key_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _join_count_cpu(match_counts, left_keys, right_keys)
        else:
            raise Error("join_count_cpu: only CPU target supported")


@compiler.register("join_count_gpu")
struct JoinCountGPU:
    """GPU join count kernel with pre-allocated right_count buffer."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        match_counts: OutputTensor[dtype=out_dtype, rank=1],
        right_count_buf: OutputTensor[dtype=out_dtype, rank=1],
        left_keys: InputTensor[dtype=key_dtype, rank=1],
        right_keys: InputTensor[dtype=key_dtype, rank=1],
        max_key_buf: InputTensor[dtype=key_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _join_count_gpu(match_counts, left_keys, right_keys, max_key_buf, right_count_buf, ctx)
        else:
            raise Error("join_count_gpu: only GPU target supported")
