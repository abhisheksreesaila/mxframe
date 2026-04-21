import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx, barrier
from gpu.memory import AddressSpace
from memory import stack_allocation
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime dtype = DType.float32
comptime FLOAT32_MIN: Scalar[dtype] = -3.4028234663852886e+38
comptime MAX_GROUPS = 8192


fn _group_max_cpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1, io_spec=_, static_spec=_],
    values: ManagedTensorSlice[dtype=dtype, rank=1, io_spec=_, static_spec=_],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1, io_spec=_, static_spec=_],
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
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1, io_spec=_, static_spec=_],
    values: ManagedTensorSlice[dtype=dtype, rank=1, io_spec=_, static_spec=_],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1, io_spec=_, static_spec=_],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    comptime COARSE_FACTOR = 4
    var size = values.dim_size(0)
    var ng = output.dim_size(0)

    if ng <= 0 or ng > MAX_GROUPS:
        raise Error("group_max: n_groups must be in [1, MAX_GROUPS]")

    # Fill global output with identity (FLOAT32_MIN)
    @parameter
    fn fill_kernel(ng: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(ng):
            output[Int(tid)] = FLOAT32_MIN

    var fill_blocks = ceildiv(ng, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[fill_kernel](
        ng,
        grid_dim=fill_blocks,
        block_dim=BLOCK_SIZE,
    )

    if size == 0:
        return

    var num_blocks = ceildiv(size, BLOCK_SIZE * COARSE_FACTOR)

    @parameter
    fn max_kernel(size: Int, ng: Int):
        # Shared memory private bins for this block
        var shared_bins = stack_allocation[
            MAX_GROUPS,
            Scalar[dtype],
            address_space=AddressSpace.SHARED,
        ]()

        # Initialize shared bins to identity (FLOAT32_MIN)
        var t = Int(thread_idx.x)
        while t < MAX_GROUPS:
            shared_bins[t] = FLOAT32_MIN
            t += BLOCK_SIZE
        barrier()

        # Each thread processes COARSE_FACTOR elements
        var block_start = Int(block_idx.x) * BLOCK_SIZE * COARSE_FACTOR
        var tid = block_start + Int(thread_idx.x)
        for c in range(COARSE_FACTOR):
            var i = tid + c * BLOCK_SIZE
            if i < size:
                var gid = Int(group_ids[i])
                if gid >= 0 and gid < ng:
                    _ = Atomic.max(shared_bins + gid, values[i])
        barrier()

        # Merge shared bins to global output
        t = Int(thread_idx.x)
        while t < ng:
            var sval = shared_bins[t]
            if sval > FLOAT32_MIN:
                _ = Atomic.max(output.unsafe_ptr() + t, sval)
            t += BLOCK_SIZE

    ctx.get_device_context().enqueue_function_experimental[max_kernel](
        size,
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
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        values: InputTensor[dtype=dtype, rank=1, static_spec=_],
        group_ids: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _group_max_cpu(output, values, group_ids)
        elif target == "gpu":
            _group_max_gpu(output, values, group_ids, ctx)
        else:
            raise Error("No known target:", target)
