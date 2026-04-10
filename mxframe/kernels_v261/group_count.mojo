import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx, barrier
from gpu.memory import AddressSpace
from memory import stack_allocation
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime dtype = DType.float32
comptime MAX_GROUPS = 8192


fn _group_count_cpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1, static_spec=...],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1, static_spec=...],
):
    var size = group_ids.dim_size(0)
    var ng = output.dim_size(0)

    for g in range(ng):
        output[g] = 0.0

    for i in range(size):
        var gid = Int(group_ids[i])
        if gid >= 0 and gid < ng:
            output[gid] += 1.0


fn _group_count_gpu(
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=1, static_spec=...],
    group_ids: ManagedTensorSlice[dtype=DType.int32, rank=1, static_spec=...],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    comptime COARSE_FACTOR = 4
    var size = group_ids.dim_size(0)
    var ng = output.dim_size(0)

    if ng <= 0:
        raise Error("group_count: n_groups must be > 0")

    # Zero global output
    @parameter
    fn zero_kernel(ng: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(ng):
            output[Int(tid)] = 0.0

    var zero_blocks = ceildiv(ng, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[zero_kernel](
        ng,
        grid_dim=zero_blocks,
        block_dim=BLOCK_SIZE,
    )

    if size == 0:
        return

    var num_blocks = ceildiv(size, BLOCK_SIZE * COARSE_FACTOR)

    if ng <= MAX_GROUPS:
        # Fast path: shared-memory privatization
        @parameter
        fn count_kernel_shared(size: Int, ng: Int):
            var shared_bins = stack_allocation[
                MAX_GROUPS,
                Scalar[dtype],
                address_space=AddressSpace.SHARED,
            ]()

            var t = Int(thread_idx.x)
            while t < MAX_GROUPS:
                shared_bins[t] = Scalar[dtype](0)
                t += BLOCK_SIZE
            barrier()

            var block_start = Int(block_idx.x) * BLOCK_SIZE * COARSE_FACTOR
            var tid = block_start + Int(thread_idx.x)
            for c in range(COARSE_FACTOR):
                var i = tid + c * BLOCK_SIZE
                if i < size:
                    var gid = Int(group_ids[i])
                    if gid >= 0 and gid < ng:
                        _ = Atomic.fetch_add(shared_bins + gid, Scalar[dtype](1))
            barrier()

            t = Int(thread_idx.x)
            while t < ng:
                var sval = shared_bins[t]
                if sval != Scalar[dtype](0):
                    _ = Atomic.fetch_add(output.unsafe_ptr() + t, sval)
                t += BLOCK_SIZE

        ctx.get_device_context().enqueue_function_experimental[count_kernel_shared](
            size,
            ng,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )
    else:
        # Fallback: global-memory atomics (no shared-memory limit)
        @parameter
        fn count_kernel_global(size: Int, ng: Int):
            var block_start = Int(block_idx.x) * BLOCK_SIZE * COARSE_FACTOR
            var tid = block_start + Int(thread_idx.x)
            for c in range(COARSE_FACTOR):
                var i = tid + c * BLOCK_SIZE
                if i < size:
                    var gid = Int(group_ids[i])
                    if gid >= 0 and gid < ng:
                        _ = Atomic.fetch_add(output.unsafe_ptr() + gid, Scalar[dtype](1))

        ctx.get_device_context().enqueue_function_experimental[count_kernel_global](
            size,
            ng,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )


@compiler.register("group_count")
struct GroupCount:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=...],
        group_ids: InputTensor[dtype=DType.int32, rank=1, static_spec=...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _group_count_cpu(output, group_ids)
        elif target == "gpu":
            _group_count_gpu(output, group_ids, ctx)
        else:
            raise Error("No known target:", target)
