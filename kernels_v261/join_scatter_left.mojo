import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime key_dtype = DType.int32
comptime idx_dtype = DType.int32
alias LEFT_NO_MATCH = Int32(-1)

fn _join_scatter_left_cpu(
    left_out: ManagedTensorSlice[mut=True, dtype=idx_dtype, rank=1],
    right_out: ManagedTensorSlice[mut=True, dtype=idx_dtype, rank=1],
    left_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    offsets: ManagedTensorSlice[dtype=idx_dtype, rank=1],
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
    var right_start = List[Int32](capacity=table_size)
    right_start.append(Int32(0))
    for i in range(1, table_size):
        right_start.append(right_start[i-1] + right_count[i-1])
    var right_positions = List[Int32](capacity=n_right)
    for i in range(n_right):
        right_positions.append(Int32(0))
    var cursor = List[Int32](capacity=table_size)
    for i in range(table_size):
        cursor.append(right_start[i])
    for i in range(n_right):
        var k = Int(right_keys[i])
        right_positions[Int(cursor[k])] = Int32(i)
        cursor[k] += 1
    for i in range(n_left):
        var k = Int(left_keys[i])
        var rc = Int(right_count[k])
        var base_offset = Int(offsets[i])
        if rc == 0:
            left_out[base_offset] = Int32(i)
            right_out[base_offset] = LEFT_NO_MATCH
        else:
            var rs = Int(right_start[k])
            for j in range(rc):
                left_out[base_offset + j] = Int32(i)
                right_out[base_offset + j] = right_positions[rs + j]

fn _join_scatter_left_gpu(
    left_out: ManagedTensorSlice[mut=True, dtype=idx_dtype, rank=1],
    right_out: ManagedTensorSlice[mut=True, dtype=idx_dtype, rank=1],
    left_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    right_keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    offsets: ManagedTensorSlice[dtype=idx_dtype, rank=1],
    right_sorted_idx: ManagedTensorSlice[dtype=idx_dtype, rank=1],
    right_key_starts: ManagedTensorSlice[dtype=idx_dtype, rank=1],
    right_key_counts: ManagedTensorSlice[dtype=idx_dtype, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    var n_left = left_keys.dim_size(0)
    if n_left == 0:
        return
    @parameter
    fn scatter_kernel(n_left: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid >= UInt(n_left):
            return
        var i = Int(tid)
        var k = Int(left_keys[i])
        var rc = Int(right_key_counts[k])
        var base_offset = Int(offsets[i])
        if rc == 0:
            left_out[base_offset]  = Int32(i)
            right_out[base_offset] = LEFT_NO_MATCH
        else:
            var rs = Int(right_key_starts[k])
            for j in range(rc):
                left_out[base_offset + j]  = Int32(i)
                right_out[base_offset + j] = right_sorted_idx[rs + j]
    ctx.get_device_context().enqueue_function_experimental[scatter_kernel](
        n_left,
        grid_dim=ceildiv(n_left, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

@compiler.register("join_scatter_left_cpu")
struct JoinScatterLeftCPU:
    @staticmethod
    fn execute[target: StaticString](
        left_out: OutputTensor[dtype=idx_dtype, rank=1],
        right_out: OutputTensor[dtype=idx_dtype, rank=1],
        left_keys: InputTensor[dtype=key_dtype, rank=1],
        right_keys: InputTensor[dtype=key_dtype, rank=1],
        offsets: InputTensor[dtype=idx_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _join_scatter_left_cpu(left_out, right_out, left_keys, right_keys, offsets)
        else:
            raise Error("join_scatter_left_cpu: only CPU target supported")

@compiler.register("join_scatter_left_gpu")
struct JoinScatterLeftGPU:
    @staticmethod
    fn execute[target: StaticString](
        left_out: OutputTensor[dtype=idx_dtype, rank=1],
        right_out: OutputTensor[dtype=idx_dtype, rank=1],
        left_keys: InputTensor[dtype=key_dtype, rank=1],
        right_keys: InputTensor[dtype=key_dtype, rank=1],
        offsets: InputTensor[dtype=idx_dtype, rank=1],
        right_sorted_idx: InputTensor[dtype=idx_dtype, rank=1],
        right_key_starts: InputTensor[dtype=idx_dtype, rank=1],
        right_key_counts: InputTensor[dtype=idx_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _join_scatter_left_gpu(
                left_out, right_out, left_keys, right_keys,
                offsets, right_sorted_idx, right_key_starts, right_key_counts, ctx)
        else:
            raise Error("join_scatter_left_gpu: only GPU target supported")
