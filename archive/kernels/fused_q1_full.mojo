# Level 2 Fused Kernel: All 4 groups × 6 aggregations = 24 values in one kernel
# This is maximum fusion for TPC-H Q1 - single kernel launch for entire query
#
# Groups: A (rf=0), N+F (rf=1,ls=0), N+O (rf=1,ls=1), R (rf=2)
# Aggregations per group: sum_qty, sum_price, sum_disc, sum_disc_price, sum_charge, count

import compiler
from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor

alias dtype = DType.float32
alias NUM_GROUPS = 4
alias NUM_AGGS = 6
alias TOTAL_OUTPUTS = NUM_GROUPS * NUM_AGGS  # 24


fn fused_q1_kernel[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutableAnyOrigin],
    # Date filter
    shipdate: LayoutTensor[DType.int32, in_layout, ImmutableAnyOrigin],
    date_cutoff: Int,
    # Group encoding
    rf_enc: LayoutTensor[DType.int32, in_layout, ImmutableAnyOrigin],
    ls_enc: LayoutTensor[DType.int32, in_layout, ImmutableAnyOrigin],
    # Values to aggregate
    qty: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    price: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    disc: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    disc_price: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    charge: LayoutTensor[dtype, in_layout, ImmutableAnyOrigin],
    size: Int,
    num_warps: Int,
):
    """GPU kernel: Each warp computes 24 partial sums (4 groups × 6 aggs)."""
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var warp_idx = global_i // WARP_SIZE
    
    # Per-thread accumulators for each group (4 groups × 6 values = 24 total)
    # Group 0: A (rf=0)
    var g0_qty: Scalar[dtype] = 0
    var g0_price: Scalar[dtype] = 0
    var g0_disc: Scalar[dtype] = 0
    var g0_disc_price: Scalar[dtype] = 0
    var g0_charge: Scalar[dtype] = 0
    var g0_count: Scalar[dtype] = 0
    
    # Group 1: N+F (rf=1, ls=0)
    var g1_qty: Scalar[dtype] = 0
    var g1_price: Scalar[dtype] = 0
    var g1_disc: Scalar[dtype] = 0
    var g1_disc_price: Scalar[dtype] = 0
    var g1_charge: Scalar[dtype] = 0
    var g1_count: Scalar[dtype] = 0
    
    # Group 2: N+O (rf=1, ls=1)
    var g2_qty: Scalar[dtype] = 0
    var g2_price: Scalar[dtype] = 0
    var g2_disc: Scalar[dtype] = 0
    var g2_disc_price: Scalar[dtype] = 0
    var g2_charge: Scalar[dtype] = 0
    var g2_count: Scalar[dtype] = 0
    
    # Group 3: R (rf=2)
    var g3_qty: Scalar[dtype] = 0
    var g3_price: Scalar[dtype] = 0
    var g3_disc: Scalar[dtype] = 0
    var g3_disc_price: Scalar[dtype] = 0
    var g3_charge: Scalar[dtype] = 0
    var g3_count: Scalar[dtype] = 0
    
    if global_i < size:
        # Check date filter
        var ship = Int(shipdate[global_i].reduce_add())
        if ship <= date_cutoff:
            var rf = Int(rf_enc[global_i].reduce_add())
            var ls = Int(ls_enc[global_i].reduce_add())
            
            # Load values
            var v_qty = qty[global_i].reduce_add()
            var v_price = price[global_i].reduce_add()
            var v_disc = disc[global_i].reduce_add()
            var v_disc_price = disc_price[global_i].reduce_add()
            var v_charge = charge[global_i].reduce_add()
            
            # Assign to correct group based on (rf, ls) encoding
            # Group 0: A (rf=0, any ls)
            if rf == 0:
                g0_qty = v_qty
                g0_price = v_price
                g0_disc = v_disc
                g0_disc_price = v_disc_price
                g0_charge = v_charge
                g0_count = 1
            # Group 1: N+F (rf=1, ls=0)
            elif rf == 1 and ls == 0:
                g1_qty = v_qty
                g1_price = v_price
                g1_disc = v_disc
                g1_disc_price = v_disc_price
                g1_charge = v_charge
                g1_count = 1
            # Group 2: N+O (rf=1, ls=1)
            elif rf == 1 and ls == 1:
                g2_qty = v_qty
                g2_price = v_price
                g2_disc = v_disc
                g2_disc_price = v_disc_price
                g2_charge = v_charge
                g2_count = 1
            # Group 3: R (rf=2, any ls)
            elif rf == 2:
                g3_qty = v_qty
                g3_price = v_price
                g3_disc = v_disc
                g3_disc_price = v_disc_price
                g3_charge = v_charge
                g3_count = 1
    
    # Warp-level reduction for all 24 values
    var s0_qty = warp_sum(g0_qty)
    var s0_price = warp_sum(g0_price)
    var s0_disc = warp_sum(g0_disc)
    var s0_disc_price = warp_sum(g0_disc_price)
    var s0_charge = warp_sum(g0_charge)
    var s0_count = warp_sum(g0_count)
    
    var s1_qty = warp_sum(g1_qty)
    var s1_price = warp_sum(g1_price)
    var s1_disc = warp_sum(g1_disc)
    var s1_disc_price = warp_sum(g1_disc_price)
    var s1_charge = warp_sum(g1_charge)
    var s1_count = warp_sum(g1_count)
    
    var s2_qty = warp_sum(g2_qty)
    var s2_price = warp_sum(g2_price)
    var s2_disc = warp_sum(g2_disc)
    var s2_disc_price = warp_sum(g2_disc_price)
    var s2_charge = warp_sum(g2_charge)
    var s2_count = warp_sum(g2_count)
    
    var s3_qty = warp_sum(g3_qty)
    var s3_price = warp_sum(g3_price)
    var s3_disc = warp_sum(g3_disc)
    var s3_disc_price = warp_sum(g3_disc_price)
    var s3_charge = warp_sum(g3_charge)
    var s3_count = warp_sum(g3_count)
    
    # Lane 0 writes 24 values for this warp
    if lane_id() == 0 and warp_idx < num_warps:
        var base = warp_idx * TOTAL_OUTPUTS
        # Group 0
        output[base + 0] = s0_qty
        output[base + 1] = s0_price
        output[base + 2] = s0_disc
        output[base + 3] = s0_disc_price
        output[base + 4] = s0_charge
        output[base + 5] = s0_count
        # Group 1
        output[base + 6] = s1_qty
        output[base + 7] = s1_price
        output[base + 8] = s1_disc
        output[base + 9] = s1_disc_price
        output[base + 10] = s1_charge
        output[base + 11] = s1_count
        # Group 2
        output[base + 12] = s2_qty
        output[base + 13] = s2_price
        output[base + 14] = s2_disc
        output[base + 15] = s2_disc_price
        output[base + 16] = s2_charge
        output[base + 17] = s2_count
        # Group 3
        output[base + 18] = s3_qty
        output[base + 19] = s3_price
        output[base + 20] = s3_disc
        output[base + 21] = s3_disc_price
        output[base + 22] = s3_charge
        output[base + 23] = s3_count


@compiler.register("fused_q1_full")
struct FusedQ1Full:
    """Level 2 fusion: All 4 groups × 6 aggregations in one kernel."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        shipdate: InputTensor[dtype=DType.int32, rank=1],
        rf_enc: InputTensor[dtype=DType.int32, rank=1],
        ls_enc: InputTensor[dtype=DType.int32, rank=1],
        qty: InputTensor[dtype=dtype, rank=1],
        price: InputTensor[dtype=dtype, rank=1],
        disc: InputTensor[dtype=dtype, rank=1],
        disc_price: InputTensor[dtype=dtype, rank=1],
        charge: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var size = Int(shipdate.spec().shape[0])
        var num_warps = ceildiv(size, WARP_SIZE)
        alias DATE_CUTOFF = 10471  # 1998-09-02 as epoch days
        
        @parameter
        if target == "gpu":
            alias BLOCK_SIZE = 256
            var num_blocks = ceildiv(size, BLOCK_SIZE)
            
            var out_tensor = output.to_layout_tensor()
            var shipdate_tensor = shipdate.to_layout_tensor().get_immutable()
            var rf_tensor = rf_enc.to_layout_tensor().get_immutable()
            var ls_tensor = ls_enc.to_layout_tensor().get_immutable()
            var qty_tensor = qty.to_layout_tensor().get_immutable()
            var price_tensor = price.to_layout_tensor().get_immutable()
            var disc_tensor = disc.to_layout_tensor().get_immutable()
            var disc_price_tensor = disc_price.to_layout_tensor().get_immutable()
            var charge_tensor = charge.to_layout_tensor().get_immutable()
            
            alias out_layout = out_tensor.layout
            alias in_layout_f32 = qty_tensor.layout
            alias in_layout_i32 = shipdate_tensor.layout
            
            var gpu_ctx = rebind[DeviceContext](ctx[])
            
            gpu_ctx.enqueue_function_checked[
                fused_q1_kernel[in_layout_i32, out_layout],
                fused_q1_kernel[in_layout_i32, out_layout],
            ](
                out_tensor,
                shipdate_tensor,
                DATE_CUTOFF,
                rf_tensor,
                ls_tensor,
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
            # CPU fallback
            var sums = InlineArray[Scalar[dtype], TOTAL_OUTPUTS](0)
            
            var shipdate_t = shipdate.to_layout_tensor().get_immutable()
            var rf_t = rf_enc.to_layout_tensor().get_immutable()
            var ls_t = ls_enc.to_layout_tensor().get_immutable()
            var qty_t = qty.to_layout_tensor().get_immutable()
            var price_t = price.to_layout_tensor().get_immutable()
            var disc_t = disc.to_layout_tensor().get_immutable()
            var disc_price_t = disc_price.to_layout_tensor().get_immutable()
            var charge_t = charge.to_layout_tensor().get_immutable()
            
            for i in range(size):
                var ship = Int(shipdate_t[i].reduce_add())
                if ship <= DATE_CUTOFF:
                    var rf = Int(rf_t[i].reduce_add())
                    var ls = Int(ls_t[i].reduce_add())
                    
                    var v_qty = qty_t[i].reduce_add()
                    var v_price = price_t[i].reduce_add()
                    var v_disc = disc_t[i].reduce_add()
                    var v_disc_price = disc_price_t[i].reduce_add()
                    var v_charge = charge_t[i].reduce_add()
                    
                    var g: Int = 0
                    if rf == 0:
                        g = 0
                    elif rf == 1 and ls == 0:
                        g = 1
                    elif rf == 1 and ls == 1:
                        g = 2
                    elif rf == 2:
                        g = 3
                    
                    var base = g * NUM_AGGS
                    sums[base + 0] += v_qty
                    sums[base + 1] += v_price
                    sums[base + 2] += v_disc
                    sums[base + 3] += v_disc_price
                    sums[base + 4] += v_charge
                    sums[base + 5] += 1
            
            var out = output.to_layout_tensor()
            for j in range(TOTAL_OUTPUTS):
                out[j] = sums[j]
