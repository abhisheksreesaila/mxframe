"""gather_rows: parallel index-based row gather for GPU sort and distinct.

Replaces the CPU Arrow.table.take(sorted_indices) call at the end of
_apply_sort_custom with a GPU kernel, eliminating the GPU→CPU index
readback and the Arrow take on CPU.

Two kernels are registered:

  gather_f32(output[M], values[N], indices[M])
    output[i] = values[indices[i]]   — for float32 columns

  gather_i32(output[M], values[N], indices[M])
    output[i] = values[indices[i]]   — for int32 columns (group_ids, sort keys)

Both CPU and GPU implementations are provided.  MAX Graph is compiled once
per unique (N, M, device) triple and cached in _POST_OP_MODEL_CACHE.
"""

import compiler
from math import ceildiv
from gpu import block_dim, block_idx, thread_idx
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor


# ── float32 gather ───────────────────────────────────────────────────────────

fn _gather_f32_cpu(
    output: ManagedTensorSlice[mut=True, dtype=DType.float32, rank=1],
    values: ManagedTensorSlice[dtype=DType.float32, rank=1],
    indices: ManagedTensorSlice[dtype=DType.int32, rank=1],
):
    var m = indices.dim_size(0)
    for i in range(m):
        output[i] = values[Int(indices[i])]


fn _gather_f32_gpu(
    output: ManagedTensorSlice[mut=True, dtype=DType.float32, rank=1],
    values: ManagedTensorSlice[dtype=DType.float32, rank=1],
    indices: ManagedTensorSlice[dtype=DType.int32, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    var m = indices.dim_size(0)
    var num_blocks = ceildiv(m, BLOCK_SIZE)

    @parameter
    fn gather_kernel_f32(m: Int):
        var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        if gid >= m:
            return
        output[gid] = values[Int(indices[gid])]

    ctx.get_device_context().enqueue_function_experimental[gather_kernel_f32](
        m,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("gather_f32")
struct GatherF32:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=1],
        values: InputTensor[dtype=DType.float32, rank=1],
        indices: InputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _gather_f32_cpu(output, values, indices)
        elif target == "gpu":
            _gather_f32_gpu(output, values, indices, ctx)
        else:
            raise Error("No known target:", target)


# ── int32 gather ─────────────────────────────────────────────────────────────

fn _gather_i32_cpu(
    output: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    values: ManagedTensorSlice[dtype=DType.int32, rank=1],
    indices: ManagedTensorSlice[dtype=DType.int32, rank=1],
):
    var m = indices.dim_size(0)
    for i in range(m):
        output[i] = values[Int(indices[i])]


fn _gather_i32_gpu(
    output: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    values: ManagedTensorSlice[dtype=DType.int32, rank=1],
    indices: ManagedTensorSlice[dtype=DType.int32, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    var m = indices.dim_size(0)
    var num_blocks = ceildiv(m, BLOCK_SIZE)

    @parameter
    fn gather_kernel_i32(m: Int):
        var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        if gid >= m:
            return
        output[gid] = values[Int(indices[gid])]

    ctx.get_device_context().enqueue_function_experimental[gather_kernel_i32](
        m,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("gather_i32")
struct GatherI32:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.int32, rank=1],
        values: InputTensor[dtype=DType.int32, rank=1],
        indices: InputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _gather_i32_cpu(output, values, indices)
        elif target == "gpu":
            _gather_i32_gpu(output, values, indices, ctx)
        else:
            raise Error("No known target:", target)
