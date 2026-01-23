# Level 1 Fused Kernel: All 6 aggregations for one group in a single launch
# Computes: sum_qty, sum_price, sum_disc, sum_disc_price, sum_charge, count
# 
# Each warp computes 6 partial sums, then we sum across warps on CPU.

import compiler
from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor

alias dtype = DType.float32
alias NUM_AGGS = 6  # qty, price, disc, disc_price, charge, count


fn fused_group_agg_kernel[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutableAnyOrigin],
    mask: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    qty: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    price: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    disc: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    disc_price: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    charge: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    size: Int,
    num_warps: Int,
):
    """GPU kernel: Each warp computes 6 partial sums for masked data."""
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var warp_idx = global_i // WARP_SIZE
    
    # Each thread loads its element's masked values
    var m: Scalar[dtype] = 0
    var v_qty: Scalar[dtype] = 0
    var v_price: Scalar[dtype] = 0
    var v_disc: Scalar[dtype] = 0
    var v_disc_price: Scalar[dtype] = 0
    var v_charge: Scalar[dtype] = 0
    
    if global_i < size:
        m = mask[global_i].reduce_add()
        v_qty = (m * qty[global_i]).reduce_add()
        v_price = (m * price[global_i]).reduce_add()
        v_disc = (m * disc[global_i]).reduce_add()
        v_disc_price = (m * disc_price[global_i]).reduce_add()
        v_charge = (m * charge[global_i]).reduce_add()
    
    # Warp-level reduction for all 6 values
    var sum_qty = warp_sum(v_qty)
    var sum_price = warp_sum(v_price)
    var sum_disc = warp_sum(v_disc)
    var sum_disc_price = warp_sum(v_disc_price)
    var sum_charge = warp_sum(v_charge)
    var count = warp_sum(m)
    
    # Lane 0 writes 6 values for this warp
    if lane_id() == 0 and warp_idx < num_warps:
        var base = warp_idx * NUM_AGGS
        output[base + 0] = sum_qty
        output[base + 1] = sum_price
        output[base + 2] = sum_disc
        output[base + 3] = sum_disc_price
        output[base + 4] = sum_charge
        output[base + 5] = count


@compiler.register("fused_group_agg")
struct FusedGroupAgg:
    """Level 1 fusion: All 6 aggregations for one group in one kernel."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        mask: InputTensor[dtype=dtype, rank=1],
        qty: InputTensor[dtype=dtype, rank=1],
        price: InputTensor[dtype=dtype, rank=1],
        disc: InputTensor[dtype=dtype, rank=1],
        disc_price: InputTensor[dtype=dtype, rank=1],
        charge: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var size = Int(mask.spec().shape[0])
        var num_warps = ceildiv(size, WARP_SIZE)
        
        @parameter
        if target == "gpu":
            alias BLOCK_SIZE = 256
            var num_blocks = ceildiv(size, BLOCK_SIZE)
            
            var out_tensor = output.to_layout_tensor()
            var mask_tensor = mask.to_layout_tensor().get_immutable()
            var qty_tensor = qty.to_layout_tensor().get_immutable()
            var price_tensor = price.to_layout_tensor().get_immutable()
            var disc_tensor = disc.to_layout_tensor().get_immutable()
            var disc_price_tensor = disc_price.to_layout_tensor().get_immutable()
            var charge_tensor = charge.to_layout_tensor().get_immutable()
            
            alias out_layout = out_tensor.layout
            alias in_layout = mask_tensor.layout
            
            var gpu_ctx = rebind[DeviceContext](ctx[])
            
            gpu_ctx.enqueue_function_checked[
                fused_group_agg_kernel[in_layout, out_layout],
                fused_group_agg_kernel[in_layout, out_layout],
            ](
                out_tensor,
                mask_tensor,
                qty_tensor,
                price_tensor,
                disc_tensor,
                disc_price_tensor,
                charge_tensor,
                size,
                num_warps,
                grid_dim=num_blocks,
                block_dim=BLOCK_SIZE,
            )
        else:
            # CPU fallback - simple loop
            var sum_qty: Scalar[dtype] = 0
            var sum_price: Scalar[dtype] = 0
            var sum_disc: Scalar[dtype] = 0
            var sum_disc_price: Scalar[dtype] = 0
            var sum_charge: Scalar[dtype] = 0
            var count: Scalar[dtype] = 0
            
            var mask_t = mask.to_layout_tensor().get_immutable()
            var qty_t = qty.to_layout_tensor().get_immutable()
            var price_t = price.to_layout_tensor().get_immutable()
            var disc_t = disc.to_layout_tensor().get_immutable()
            var disc_price_t = disc_price.to_layout_tensor().get_immutable()
            var charge_t = charge.to_layout_tensor().get_immutable()
            
            for i in range(size):
                var m = mask_t[i].reduce_add()
                sum_qty += (m * qty_t[i]).reduce_add()
                sum_price += (m * price_t[i]).reduce_add()
                sum_disc += (m * disc_t[i]).reduce_add()
                sum_disc_price += (m * disc_price_t[i]).reduce_add()
                sum_charge += (m * charge_t[i]).reduce_add()
                count += m
            
            var out = output.to_layout_tensor()
            out[0] = sum_qty
            out[1] = sum_price
            out[2] = sum_disc
            out[3] = sum_disc_price
            out[4] = sum_charge
            out[5] = count
