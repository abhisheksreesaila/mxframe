import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime dtype = DType.float32
comptime MAX_GROUPS = 64


fn _group_sum_cpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1],
    values: ManagedTensorSlice[dtype=dtype, rank=1],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1],
):
    var size = values.dim_size(0)
    var ng = output.dim_size(0)

    for g in range(ng):
        output[g] = 0.0

    for i in range(size):
        var gid = Int(group_ids[i])
        if gid >= 0 and gid < ng:
            output[gid] += values[i]


fn _group_sum_gpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1],
    values: ManagedTensorSlice[dtype=dtype, rank=1],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 128
    var size = values.dim_size(0)
    var out_len = output.dim_size(0)

    if size == 0:
        @parameter
        fn zero_kernel(out_len: Int):
            var tid = block_dim.x * block_idx.x + thread_idx.x
            if tid < UInt(out_len):
                output[Int(tid)] = 0.0

        if out_len > 0:
            var blocks = ceildiv(out_len, BLOCK_SIZE)
            ctx.get_device_context().enqueue_function_experimental[zero_kernel](
                out_len,
                grid_dim=blocks,
                block_dim=BLOCK_SIZE,
            )
        return

    var num_warps = ceildiv(size, WARP_SIZE)
    if num_warps <= 0:
        raise Error("group_sum: invalid num_warps (must be > 0)")
    if out_len % num_warps != 0:
        raise Error("group_sum: output length must be divisible by num_warps")

    var ng = out_len // num_warps
    if ng <= 0:
        raise Error("group_sum: inferred n_groups must be > 0")
    if ng > MAX_GROUPS:
        raise Error("group_sum: inferred n_groups exceeds MAX_GROUPS")

    @parameter
    fn sum_kernel(size: Int, out_len: Int, ng: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid >= UInt(out_len):
            return

        var out_i = Int(tid)
        var warp_idx = out_i // ng
        var gid = out_i % ng

        if warp_idx != 0:
            output[out_i] = 0.0
            return

        var total: Float32 = 0.0
        for i in range(size):
            if Int(group_ids[i]) == gid:
                total += values[i]
        output[out_i] = total

    var num_blocks = ceildiv(out_len, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[sum_kernel](
        size,
        out_len,
        ng,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("group_sum")
struct GroupSum:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        values: InputTensor[dtype=dtype, rank=1],
        group_ids: InputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _group_sum_cpu(output, values, group_ids)
        elif target == "gpu":
            _group_sum_gpu(output, values, group_ids, ctx)
        else:
            raise Error("No known target:", target)
