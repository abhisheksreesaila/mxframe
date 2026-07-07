"""masked_global_agg: Fused single-pass masked global aggregation.

Instead of:
    filtered_table = table.filter(mask)   # allocates N_pass × n_cols bytes
    result = sum(filtered_table[col])

We do:
    masked_global_sum(col[N], mask[N])    # single pass, zero intermediate copy

Registered ops:
    masked_global_sum  → float32[1]: sum(values[i] where mask[i] != 0)
    masked_global_min  → float32[1]: min(values[i] where mask[i] != 0)  [CPU path]
    masked_global_max  → float32[1]: max(values[i] where mask[i] != 0)  [CPU path]
"""

import compiler
from math import ceildiv
from gpu import block_dim, block_idx, thread_idx, barrier
from gpu.memory import AddressSpace
from memory import stack_allocation
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime val_dtype = DType.float32
comptime mask_dtype = DType.int32
comptime BLOCK_SIZE = 256


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_sum  — sum(values[i] where mask[i] != 0)
# CPU: sequential conditional accumulate (Mojo auto-vectorizes the hot loop).
# GPU: per-block tree reduction, blocks atomically add partial sums to output[0].
# ─────────────────────────────────────────────────────────────────────────────

fn _masked_global_sum_cpu(
    output: ManagedTensorSlice[mut=True, dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    values: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    mask: ManagedTensorSlice[dtype=mask_dtype, rank=1, io_spec=_, static_spec=_],
):
    var n = values.dim_size(0)
    var acc: Scalar[val_dtype] = 0.0
    for i in range(n):
        if mask[i] != 0:
            acc += values[i]
    output[0] = acc


fn _masked_global_sum_gpu(
    output: ManagedTensorSlice[mut=True, dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    values: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    mask: ManagedTensorSlice[dtype=mask_dtype, rank=1, io_spec=_, static_spec=_],
    ctx: DeviceContextPtr,
) raises:
    var n = values.dim_size(0)

    # Zero the single-element output before block reductions add into it.
    # Follows the zero_kernel pattern from group_sum.mojo (takes a launch arg).
    @parameter
    fn zero_out(n_out: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(n_out):
            output[Int(tid)] = Scalar[val_dtype](0.0)

    ctx.get_device_context().enqueue_function_experimental[zero_out](
        1, grid_dim=1, block_dim=BLOCK_SIZE
    )

    if n == 0:
        return

    var num_blocks = ceildiv(n, BLOCK_SIZE)

    @parameter
    fn sum_kernel(n: Int):
        var shared = stack_allocation[
            BLOCK_SIZE, Scalar[val_dtype], address_space=AddressSpace.SHARED
        ]()
        var tid = Int(thread_idx.x)
        var gid = Int(block_idx.x) * BLOCK_SIZE + tid

        # Each thread loads its element (0 if out of bounds or masked out).
        var val: Scalar[val_dtype] = 0.0
        if gid < n and mask[gid] != 0:
            val = values[gid]
        shared[tid] = val
        barrier()

        # Tree reduction within the block.
        var s: Int = BLOCK_SIZE >> 1
        while s > 0:
            if tid < s:
                shared[tid] += shared[tid + s]
            barrier()
            s = s >> 1

        # Thread 0 atomically adds this block's partial sum to global output.
        if tid == 0:
            _ = Atomic.fetch_add(output.unsafe_ptr(), shared[0])

    ctx.get_device_context().enqueue_function_experimental[sum_kernel](
        n, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )


@compiler.register("masked_global_sum")
struct MaskedGlobalSum:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=val_dtype, rank=1, static_spec=_],
        values: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
        mask: InputTensor[dtype=mask_dtype, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _masked_global_sum_cpu(output, values, mask)
        elif target == "gpu":
            _masked_global_sum_gpu(output, values, mask, ctx)
        else:
            raise Error("No known target:", target)


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_min  — min(values[i] where mask[i] != 0)
# CPU-only fast path; GPU falls back to CPU sequential (correct but not parallel).
# For rare global-min queries this is acceptable; sum is the primary hot path.
# ─────────────────────────────────────────────────────────────────────────────

fn _masked_global_min_cpu(
    output: ManagedTensorSlice[mut=True, dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    values: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    mask: ManagedTensorSlice[dtype=mask_dtype, rank=1, io_spec=_, static_spec=_],
):
    var n = values.dim_size(0)
    # 3.4028235e+38 is the maximum finite float32 value.
    var cur: Scalar[val_dtype] = Scalar[val_dtype](3.4028235e38)
    for i in range(n):
        if mask[i] != 0 and values[i] < cur:
            cur = values[i]
    output[0] = cur


@compiler.register("masked_global_min")
struct MaskedGlobalMin:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=val_dtype, rank=1, static_spec=_],
        values: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
        mask: InputTensor[dtype=mask_dtype, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        # Both paths use the CPU sequential loop (correct on all targets).
        _masked_global_min_cpu(output, values, mask)


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_max  — max(values[i] where mask[i] != 0)
# CPU-only fast path; same rationale as masked_global_min.
# ─────────────────────────────────────────────────────────────────────────────

fn _masked_global_max_cpu(
    output: ManagedTensorSlice[mut=True, dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    values: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    mask: ManagedTensorSlice[dtype=mask_dtype, rank=1, io_spec=_, static_spec=_],
):
    var n = values.dim_size(0)
    # -3.4028235e+38 is the minimum finite float32 value.
    var cur: Scalar[val_dtype] = Scalar[val_dtype](-3.4028235e38)
    for i in range(n):
        if mask[i] != 0 and values[i] > cur:
            cur = values[i]
    output[0] = cur


@compiler.register("masked_global_max")
struct MaskedGlobalMax:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=val_dtype, rank=1, static_spec=_],
        values: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
        mask: InputTensor[dtype=mask_dtype, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        # Both paths use the CPU sequential loop (correct on all targets).
        _masked_global_max_cpu(output, values, mask)


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_sum_product  — sum(col_a[i] * col_b[i] where mask[i] != 0)
#
# TRUE single-pass fused kernel: reads two input columns and mask together,
# computes the product and accumulates only where mask != 0.
# Avoids BOTH the filter allocation AND the intermediate multiplication array
# that `masked_global_sum(pc.multiply(a, b), mask)` would require.
#
# Usage: for plans of the form  sum(col_a * col_b)  filtered by a WHERE clause.
# ─────────────────────────────────────────────────────────────────────────────

fn _masked_global_sum_product_cpu(
    output: ManagedTensorSlice[mut=True, dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    col_a: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    col_b: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    mask: ManagedTensorSlice[dtype=mask_dtype, rank=1, io_spec=_, static_spec=_],
):
    var n = col_a.dim_size(0)
    var acc: Scalar[val_dtype] = 0.0
    for i in range(n):
        if mask[i] != 0:
            acc += col_a[i] * col_b[i]
    output[0] = acc


fn _masked_global_sum_product_gpu(
    output: ManagedTensorSlice[mut=True, dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    col_a: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    col_b: ManagedTensorSlice[dtype=val_dtype, rank=1, io_spec=_, static_spec=_],
    mask: ManagedTensorSlice[dtype=mask_dtype, rank=1, io_spec=_, static_spec=_],
    ctx: DeviceContextPtr,
) raises:
    var n = col_a.dim_size(0)

    @parameter
    fn zero_out(n_out: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(n_out):
            output[Int(tid)] = Scalar[val_dtype](0.0)

    ctx.get_device_context().enqueue_function_experimental[zero_out](
        1, grid_dim=1, block_dim=BLOCK_SIZE
    )

    if n == 0:
        return

    var num_blocks = ceildiv(n, BLOCK_SIZE)

    @parameter
    fn sum_product_kernel(n: Int):
        var shared = stack_allocation[
            BLOCK_SIZE, Scalar[val_dtype], address_space=AddressSpace.SHARED
        ]()
        var tid = Int(thread_idx.x)
        var gid = Int(block_idx.x) * BLOCK_SIZE + tid

        var val: Scalar[val_dtype] = 0.0
        if gid < n and mask[gid] != 0:
            val = col_a[gid] * col_b[gid]
        shared[tid] = val
        barrier()

        var s: Int = BLOCK_SIZE >> 1
        while s > 0:
            if tid < s:
                shared[tid] += shared[tid + s]
            barrier()
            s = s >> 1

        if tid == 0:
            _ = Atomic.fetch_add(output.unsafe_ptr(), shared[0])

    ctx.get_device_context().enqueue_function_experimental[sum_product_kernel](
        n, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )


@compiler.register("masked_global_sum_product")
struct MaskedGlobalSumProduct:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=val_dtype, rank=1, static_spec=_],
        col_a: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
        col_b: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
        mask: InputTensor[dtype=mask_dtype, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _masked_global_sum_product_cpu(output, col_a, col_b, mask)
        elif target == "gpu":
            _masked_global_sum_product_gpu(output, col_a, col_b, mask, ctx)
        else:
            raise Error("No known target:", target)
