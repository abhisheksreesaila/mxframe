import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime dtype = DType.float32
comptime FLOAT32_MAX: Scalar[dtype] = 3.4028234663852886e+38
comptime MAX_GROUPS = 64


fn _group_min_cpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1],
    values: ManagedTensorSlice[dtype=dtype, rank=1],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1],
):
    var size = values.dim_size(0)
    var ng = output.dim_size(0)

    for g in range(ng):
        output[g] = FLOAT32_MAX

    for i in range(size):
        var gid = Int(group_ids[i])
        var value_i = values[i]
        if gid >= 0 and gid < ng and value_i < output[gid]:
            output[gid] = value_i


fn _group_min_gpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1],
    values: ManagedTensorSlice[dtype=dtype, rank=1],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    var size = values.dim_size(0)
    var out_len = output.dim_size(0)

    if size == 0:
        @parameter
        fn empty_kernel(out_len: Int):
            var tid = block_dim.x * block_idx.x + thread_idx.x
            if tid < UInt(out_len):
                output[Int(tid)] = FLOAT32_MAX

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
        raise Error("group_min: invalid num_warps (must be > 0)")
    if out_len % num_warps != 0:
        raise Error("group_min: output length must be divisible by num_warps")

    var ng = out_len // num_warps
    if ng <= 0:
        raise Error("group_min: inferred n_groups must be > 0")
    if ng > MAX_GROUPS:
        raise Error("group_min: inferred n_groups exceeds MAX_GROUPS")

    @parameter
    fn fill_max_kernel(out_len: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(out_len):
            output[Int(tid)] = FLOAT32_MAX

    if out_len > 0:
        var fill_blocks = ceildiv(out_len, BLOCK_SIZE)
        ctx.get_device_context().enqueue_function_experimental[fill_max_kernel](
            out_len,
            grid_dim=fill_blocks,
            block_dim=BLOCK_SIZE,
        )

    var total_threads = num_warps * WARP_SIZE

    @parameter
    fn min_kernel(size: Int, ng: Int, total_threads: Int):
        var tid_u = block_dim.x * block_idx.x + thread_idx.x
        var tid = Int(tid_u)
        if tid >= total_threads:
            return

        var warp_idx = tid // WARP_SIZE
        var base = warp_idx * ng

        var i = tid
        while i < size:
            var gid = Int(group_ids[i])
            if gid >= 0 and gid < ng:
                var out_ptr = output.unsafe_ptr() + (base + gid)
                _ = Atomic.min(out_ptr, values[i])
            i += total_threads

    var num_blocks = ceildiv(total_threads, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[min_kernel](
        size,
        ng,
        total_threads,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("group_min")
struct GroupMin:
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
            _group_min_cpu(output, values, group_ids)
        elif target == "gpu":
            _group_min_gpu(output, values, group_ids, ctx)
        else:
            raise Error("No known target:", target)
