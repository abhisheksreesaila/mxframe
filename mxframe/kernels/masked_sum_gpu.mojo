# Minimal GPU-only masked sum kernel (dot product approach)
# Computes sum(mask * values) = dot(mask, values)

import compiler
from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor

alias dtype = DType.float32


fn dot_product_warp[
    in_layout: Layout, out_layout: Layout, size: Int
](
    output: LayoutTensor[dtype, out_layout, MutableAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
):
    """GPU kernel: Each warp computes a partial dot product."""
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    var partial_product: Scalar[dtype] = 0
    if global_i < size:
        # Each thread computes one element's product
        partial_product = (a[global_i] * b[global_i]).reduce_add()

    # Warp-level sum
    var total = warp_sum(partial_product)

    # Only lane 0 writes the warp's partial sum
    if lane_id() == 0:
        output[global_i // WARP_SIZE] = total


@compiler.register("masked_sum_gpu_only")
struct MaskedSumGpuOnly:
    """GPU-only masked sum (dot product) using warp reduction."""

    @staticmethod
    fn execute[
        target: StaticString,
        in_dtype: DType = DType.float32,
        # SIZE must be known at compile time for the kernel
        SIZE: Int = 10,
    ](
        output: OutputTensor[dtype=in_dtype, rank=1],
        mask: InputTensor[dtype=in_dtype, rank=1],
        values: InputTensor[dtype=in_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            alias BLOCK_SIZE = 32  # One warp
            alias BLOCKS_PER_GRID = ceildiv(SIZE, BLOCK_SIZE)
            
            var out_tensor = output.to_layout_tensor()
            var mask_tensor = mask.to_layout_tensor().get_immutable()
            var values_tensor = values.to_layout_tensor().get_immutable()
            
            alias out_layout = out_tensor.layout
            alias in_layout = mask_tensor.layout
            
            # Get GPU context using rebind pattern
            var gpu_ctx = rebind[DeviceContext](ctx[])
            
            gpu_ctx.enqueue_function_checked[
                dot_product_warp[in_layout, out_layout, SIZE],
                dot_product_warp[in_layout, out_layout, SIZE],
            ](
                out_tensor,
                mask_tensor,
                values_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=BLOCK_SIZE,
            )
        else:
            raise Error("This kernel is GPU-only!")
