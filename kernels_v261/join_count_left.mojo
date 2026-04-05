import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime key_dtype = DType.int32
comptime out_dtype = DType.int32

fn _join_count_left_cpu(
    match_counts: ManagedTensorSlice[mut=True, dtype=out_dtype, rank=1],
    left_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
):
    var n_left = left_keys.dim_size(0)
    var n_right = right_keys.dim_size(0)
    if n_left == 0:
        return
    var max_key = Int(left_keys[0])
    for i in range(n_left):
        var k = Int(left_keys[i])
        if k > max_key:
            max_key = k
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k > max_key:
            max_key = k
    var table_size = max_key + 1
    var right_count = List[Int32](capacity=table_size)
    for i in range(table_size):
        right_count.append(Int32(0))
    for i in range(n_right):
        right_count[Int(right_keys[i])] += 1
    for i in range(n_left):
        var k = Int(left_keys[i])
        var cnt = Int(right_count[k])
        match_counts[i] = Int32(cnt if cnt > 0 else 1)

fn _join_count_left_gpu(
    match_counts: ManagedTensorSlice[mut=True, dtype=out_dtype, rank=1],
    left_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    max_key_buf: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_count_buf: ManagedTensorSlice[mut=True, dtype=out_dtype, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    var n_left = left_keys.dim_size(0)
    var n_right = right_keys.dim_size(0)
    var table_size = right_count_buf.dim_size(0)
    if n_left == 0:
        return
    var buf_size = right_count_buf.dim_size(0)
    @parameter
    fn zero_kernel(buf_size: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(buf_size):
            right_count_buf[Int(tid)] = 0
    if buf_size > 0:
        ctx.get_device_context().enqueue_function_experimental[zero_kernel](
            buf_size, grid_dim=ceildiv(buf_size, BLOCK_SIZE), block_dim=BLOCK_SIZE)
    @parameter
    fn count_right_kernel(n_right: Int, table_size: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(n_right):
            var k = Int(right_keys[Int(tid)])
            if k >= 0 and k < table_size:
                _ = Atomic.fetch_add(right_count_buf.unsafe_ptr() + k, Int32(1))
    if n_right > 0:
        ctx.get_device_context().enqueue_function_experimental[count_right_kernel](
            n_right, table_size,
            grid_dim=ceildiv(n_right, BLOCK_SIZE), block_dim=BLOCK_SIZE)
    @parameter
    fn probe_kernel(n_left: Int, table_size: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(n_left):
            var k = Int(left_keys[Int(tid)])
            var cnt = Int32(0)
            if k >= 0 and k < table_size:
                cnt = right_count_buf[k]
            match_counts[Int(tid)] = cnt if cnt > 0 else Int32(1)
    ctx.get_device_context().enqueue_function_experimental[probe_kernel](
        n_left, table_size,
        grid_dim=ceildiv(n_left, BLOCK_SIZE), block_dim=BLOCK_SIZE)

@compiler.register("join_count_left_cpu")
struct JoinCountLeftCPU:
    @staticmethod
    fn execute[target: StaticString](
        match_counts: OutputTensor[dtype=out_dtype, rank=1],
        left_keys: InputTensor[dtype=key_dtype, rank=1],
        right_keys: InputTensor[dtype=key_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _join_count_left_cpu(match_counts, left_keys, right_keys)
        else:
            raise Error("join_count_left_cpu: only CPU target supported")

@compiler.register("join_count_left_gpu")
struct JoinCountLeftGPU:
    @staticmethod
    fn execute[target: StaticString](
        match_counts: OutputTensor[dtype=out_dtype, rank=1],
        right_count_buf: OutputTensor[dtype=out_dtype, rank=1],
        left_keys: InputTensor[dtype=key_dtype, rank=1],
        right_keys: InputTensor[dtype=key_dtype, rank=1],
        max_key_buf: InputTensor[dtype=key_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _join_count_left_gpu(match_counts, left_keys, right_keys, max_key_buf, right_count_buf, ctx)
        else:
            raise Error("join_count_left_gpu: only GPU target supported")
