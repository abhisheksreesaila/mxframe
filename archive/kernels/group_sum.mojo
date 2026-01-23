# Generic Group Reduce Kernel
# Single-pass kernel that computes sum per group using group_ids array.
# 
# This is more general than fused_q1_full which hardcodes 4 groups.
# Supports arbitrary number of groups (up to MAX_GROUPS).

import compiler
from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor
from memory import UnsafePointer

alias dtype = DType.float32
alias MAX_GROUPS = 64  # Maximum number of groups supported
alias NUM_AGGS = 1  # Just sum for now (can extend to count, mean, etc.)


fn group_sum_kernel[
    val_layout: Layout, 
    gid_layout: Layout,
    out_layout: Layout,
](
    output: LayoutTensor[dtype, out_layout, MutableAnyOrigin],
    values: LayoutTensor[dtype, val_layout, ImmutableAnyOrigin],
    group_ids: LayoutTensor[DType.int32, gid_layout, ImmutableAnyOrigin],
    size: Int,
    n_groups: Int,
    num_warps: Int,
):
    """GPU kernel: Each warp computes partial sums per group, then reduces."""
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var warp_idx = global_i // WARP_SIZE
    
    # Thread-local accumulators for each group
    var local_sums = InlineArray[Scalar[dtype], MAX_GROUPS](0)
    
    # Each thread accumulates its element to the correct group
    if global_i < size:
        var gid = Int(group_ids[global_i].reduce_add())
        var val = values[global_i].reduce_add()
        if gid >= 0 and gid < n_groups:
            local_sums[gid] = val
    
    # Warp-level reduction for each group
    for g in range(n_groups):
        var sum_g = warp_sum(local_sums[g])
        
        # Lane 0 writes this warp's partial sum for group g
        if lane_id() == 0 and warp_idx < num_warps:
            # Output layout: [num_warps, n_groups]
            var idx = warp_idx * n_groups + g
            output[idx] = sum_g


@compiler.register("group_sum")
struct GroupSum:
    """Single-pass group sum kernel.
    
    Computes sum(values) for each group defined by group_ids.
    Output shape: [num_warps * n_groups] - partial sums to be reduced on CPU.
    """

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        values: InputTensor[dtype=dtype, rank=1],
        group_ids: InputTensor[dtype=DType.int32, rank=1],
        n_groups: InputTensor[dtype=DType.int32, rank=0],  # Scalar
        ctx: DeviceContextPtr,
    ) raises:
        var size = Int(values.spec().shape[0])
        var num_warps = ceildiv(size, WARP_SIZE)
        var ng = 6  # Default, will be overridden
        
        @parameter
        if target == "gpu":
            alias BLOCK_SIZE = 256
            var num_blocks = ceildiv(size, BLOCK_SIZE)
            
            var out_tensor = output.to_layout_tensor()
            var val_tensor = values.to_layout_tensor().get_immutable()
            var gid_tensor = group_ids.to_layout_tensor().get_immutable()
            
            var gpu_ctx = ctx.get_device_context()
            gpu_ctx.enqueue_function[group_sum_kernel](
                out_tensor,
                val_tensor,
                gid_tensor,
                size,
                ng,
                num_warps,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            # CPU fallback: Simple loop
            var out_ptr = output.unsafe_ptr()
            var val_ptr = values.unsafe_ptr()
            var gid_ptr = group_ids.unsafe_ptr()
            
            # Zero output
            for i in range(ng):
                out_ptr[i] = 0.0
            
            # Accumulate
            for i in range(size):
                var g = Int(gid_ptr[i])
                if g >= 0 and g < ng:
                    out_ptr[g] += val_ptr[i]
