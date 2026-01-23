# Simple GPU masked sum kernel using warp reduction
# Computes sum(mask * values) in a single fused kernel
#
# Uses warp reduction for efficiency.

import compiler
from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor

alias dtype = DType.float32


fn masked_sum_warp_kernel[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutableAnyOrigin],
    mask: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    values: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    size: Int,
):
    """GPU kernel: Each warp computes a partial sum of mask*values."""
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    
    # Each thread computes one element's masked value
    var masked_val: Scalar[dtype] = 0
    if global_i < size:
        masked_val = (mask[global_i] * values[global_i]).reduce_add()
    
    # Warp-level reduction
    var warp_total = warp_sum(masked_val)
    
    # Lane 0 of each warp writes the partial sum
    if lane_id() == 0:
        var warp_idx = global_i // WARP_SIZE
        output[warp_idx] = warp_total


@compiler.register("masked_sum_simple")
struct MaskedSumSimple:
    """Simple GPU masked sum using warp reduction."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        mask: InputTensor[dtype=dtype, rank=1],
        values: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var size = Int(mask.spec().shape[0])
        
        @parameter
        if target == "gpu":
            alias BLOCK_SIZE = 256
            var num_blocks = ceildiv(size, BLOCK_SIZE)
            
            var out_tensor = output.to_layout_tensor()
            var mask_tensor = mask.to_layout_tensor().get_immutable()
            var values_tensor = values.to_layout_tensor().get_immutable()
            
            alias out_layout = out_tensor.layout
            alias in_layout = mask_tensor.layout
            
            var gpu_ctx = rebind[DeviceContext](ctx[])
            
            gpu_ctx.enqueue_function_checked[
                masked_sum_warp_kernel[in_layout, out_layout],
                masked_sum_warp_kernel[in_layout, out_layout],
            ](
                out_tensor,
                mask_tensor,
                values_tensor,
                size,
                grid_dim=num_blocks,
                block_dim=BLOCK_SIZE,
            )
        else:
            # CPU fallback - simple loop
            var total: Scalar[dtype] = 0
            var mask_tensor = mask.to_layout_tensor().get_immutable()
            var values_tensor = values.to_layout_tensor().get_immutable()
            for i in range(size):
                total += (mask_tensor[i] * values_tensor[i]).reduce_add()
            output.to_layout_tensor()[0] = total
