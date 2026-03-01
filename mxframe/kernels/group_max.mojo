import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime dtype = DType.float32
comptime FLOAT32_MIN: Scalar[dtype] = -3.4028234663852886e+38
comptime MAX_GROUPS = 64


fn _group_max_cpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1],
    values: ManagedTensorSlice[dtype=dtype, rank=1],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1],
):
    var size = values.dim_size(0)
    var ng = output.dim_size(0)

    for g in range(ng):
        output[g] = FLOAT32_MIN

    for i in range(size):
        var gid = Int(group_ids[i])
        var value_i = values[i]
        if gid >= 0 and gid < ng and value_i > output[gid]:
            output[gid] = value_i


fn _group_max_gpu(
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
        fn empty_kernel(out_len: Int):
            var tid = block_dim.x * block_idx.x + thread_idx.x
            if tid < UInt(out_len):
                output[Int(tid)] = FLOAT32_MIN

        if out_len > 0:
            var blocks = ceildiv(out_len, BLOCK_SIZE)
            ctx.get_device_context().enqueue_function_experimental[empty_kernel](
                out_len,
                grid_dim=blocks,
                block_dim=BLOCK_SIZE,
            )
        return

    var num_warps = ceildiv(size, WARP_SIZE)
    if num_warps <= 0:
        raise Error("group_max: invalid num_warps (must be > 0)")
    if out_len % num_warps != 0:
        raise Error("group_max: output length must be divisible by num_warps")

    var ng = out_len // num_warps
    if ng <= 0:
        raise Error("group_max: inferred n_groups must be > 0")
    if ng > MAX_GROUPS:
        raise Error("group_max: inferred n_groups exceeds MAX_GROUPS")

    @parameter
    fn max_kernel(size: Int, out_len: Int, ng: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid >= UInt(out_len):
            return

        var out_i = Int(tid)
        var warp_idx = out_i // ng
        var gid = out_i % ng

        if warp_idx != 0:
            output[out_i] = FLOAT32_MIN
            return

        var max_v: Float32 = FLOAT32_MIN
        for i in range(size):
            var value_i = values[i]
            if Int(group_ids[i]) == gid and value_i > max_v:
                max_v = value_i
        output[out_i] = max_v

    var num_blocks = ceildiv(out_len, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[max_kernel](
        size,
        out_len,
        ng,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("group_max")
struct GroupMax:
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
            _group_max_cpu(output, values, group_ids)
        elif target == "gpu":
            _group_max_gpu(output, values, group_ids, ctx)
        else:
            raise Error("No known target:", target)
