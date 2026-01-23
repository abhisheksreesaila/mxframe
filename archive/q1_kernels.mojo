# TPC-H Q1 Custom Kernels for MAX Engine
# CPU and GPU implementations

import math
import compiler
from gpu import global_idx, WARP_SIZE, block_idx, block_dim, thread_idx, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import stack_allocation

# Block size for GPU kernels
comptime BLOCK_SIZE: Int = 256
comptime MAX_WARPS_PER_BLOCK: Int = BLOCK_SIZE // WARP_SIZE


# =============================================================================
# KERNEL 1: Masked Sum
# Computes sum(mask * values) in a single pass
# =============================================================================

@always_inline
fn masked_sum_cpu[
    dtype: DType
](
    output: LayoutTensor[mut=True, dtype, **_],
    mask: LayoutTensor[dtype, **_],
    values: LayoutTensor[dtype, **_],
):
    """CPU kernel: Compute sum of values where mask is non-zero."""
    var acc: Scalar[dtype] = 0
    for i in range(values.size()):
        acc += mask[i][0] * values[i][0]
    output[0] = acc


@compiler.register("masked_sum")
struct MaskedSumOp:
    """Fused mask * values + sum reduction."""
    
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1],  # 1-element vector output
        mask: InputTensor[dtype=dtype, rank=1],
        values: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        output_tensor = output.to_layout_tensor()
        mask_tensor = mask.to_layout_tensor().get_immutable()
        values_tensor = values.to_layout_tensor().get_immutable()

        @parameter
        if target == "cpu":
            masked_sum_cpu[dtype](output_tensor, mask_tensor, values_tensor)
        elif target == "gpu":
            # For GPU, use the CPU kernel for now
            # TODO: Implement proper GPU reduction
            masked_sum_cpu[dtype](output_tensor, mask_tensor, values_tensor)
        else:
            raise Error("Unsupported target device: " + target)


# =============================================================================
# KERNEL 2: Compute Derived Columns
# Computes disc_price = price * (1 - disc)
#           charge = disc_price * (1 + tax)
# =============================================================================

@always_inline
fn derived_columns_cpu[
    dtype: DType
](
    disc_price_out: LayoutTensor[mut=True, dtype, **_],
    charge_out: LayoutTensor[mut=True, dtype, **_],
    price: LayoutTensor[dtype, **_],
    discount: LayoutTensor[dtype, **_],
    tax: LayoutTensor[dtype, **_],
):
    """CPU kernel: Compute disc_price and charge in a single pass."""
    for i in range(price.size()):
        var p = price[i][0]
        var d = discount[i][0]
        var t = tax[i][0]
        var one: Scalar[dtype] = 1.0
        var dp = p * (one - d)
        disc_price_out[i] = dp
        charge_out[i] = dp * (one + t)


@always_inline
fn derived_columns_gpu[
    dtype: DType, layout: Layout
](
    disc_price_out: LayoutTensor[dtype, layout, MutAnyOrigin],
    charge_out: LayoutTensor[dtype, layout, MutAnyOrigin],
    price: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    discount: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    tax: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """GPU kernel: Compute disc_price and charge per element."""
    var i = Int(global_idx.x)
    if i < price.size():
        var p = price[i][0]
        var d = discount[i][0]
        var t = tax[i][0]
        var one: Scalar[dtype] = 1.0
        var dp = p * (one - d)
        disc_price_out[i] = dp
        charge_out[i] = dp * (one + t)


@compiler.register("compute_derived_columns")
struct ComputeDerivedColumnsOp:
    """Fused computation of disc_price and charge."""
    
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType = DType.float32,
    ](
        disc_price_out: OutputTensor[dtype=dtype, rank=1],
        charge_out: OutputTensor[dtype=dtype, rank=1],
        price: InputTensor[dtype=dtype, rank=1],
        discount: InputTensor[dtype=dtype, rank=1],
        tax: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        disc_price_tensor = disc_price_out.to_layout_tensor()
        charge_tensor = charge_out.to_layout_tensor()
        price_tensor = price.to_layout_tensor().get_immutable()
        discount_tensor = discount.to_layout_tensor().get_immutable()
        tax_tensor = tax.to_layout_tensor().get_immutable()

        @parameter
        if target == "cpu":
            derived_columns_cpu[dtype](
                disc_price_tensor, charge_tensor,
                price_tensor, discount_tensor, tax_tensor
            )
        elif target == "gpu":
            # For GPU, use the CPU kernel for now 
            # TODO: Implement proper GPU element-wise kernel
            derived_columns_cpu[dtype](
                disc_price_tensor, charge_tensor,
                price_tensor, discount_tensor, tax_tensor
            )
        else:
            raise Error("Unsupported target device: " + target)


# =============================================================================
# KERNEL 3: Build Combined Mask
# Creates mask = (shipdate <= cutoff) AND (group_id == target_group)
# =============================================================================

@always_inline
fn build_combined_mask_cpu[
    dtype: DType,
](
    mask_out: LayoutTensor[mut=True, dtype, **_],
    shipdate: LayoutTensor[DType.int32, **_],
    cutoff: Int32,
    group_id: LayoutTensor[DType.int32, **_],
    target_group: Int32,
):
    """CPU kernel: Build combined date + group mask."""
    var one: Scalar[dtype] = 1.0
    var zero: Scalar[dtype] = 0.0
    for i in range(shipdate.size()):
        var date_ok = one if Int32(shipdate[i][0]) <= cutoff else zero
        var group_ok = one if Int32(group_id[i][0]) == target_group else zero
        mask_out[i] = date_ok * group_ok


@compiler.register("build_combined_mask")
struct BuildCombinedMaskOp:
    """Build mask for date filter AND group matching."""
    
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType = DType.float32,
    ](
        mask_out: OutputTensor[dtype=dtype, rank=1],
        shipdate: InputTensor[dtype=DType.int32, rank=1],
        cutoff: InputTensor[dtype=DType.int32, rank=1],  # 1-element vector
        group_id: InputTensor[dtype=DType.int32, rank=1],
        target_group: InputTensor[dtype=DType.int32, rank=1],  # 1-element vector
        ctx: DeviceContextPtr,
    ) raises:
        mask_tensor = mask_out.to_layout_tensor()
        shipdate_tensor = shipdate.to_layout_tensor().get_immutable()
        group_id_tensor = group_id.to_layout_tensor().get_immutable()
        
        # Get scalar values from 1-element tensors
        var cutoff_val = Int32(cutoff.to_layout_tensor().get_immutable()[0][0])
        var target_group_val = Int32(target_group.to_layout_tensor().get_immutable()[0][0])

        @parameter
        if target == "cpu":
            build_combined_mask_cpu[dtype](
                mask_tensor, shipdate_tensor, cutoff_val,
                group_id_tensor, target_group_val
            )
        elif target == "gpu":
            # For GPU, use the CPU kernel for now (mask building is memory-bound)
            build_combined_mask_cpu[dtype](
                mask_tensor, shipdate_tensor, cutoff_val,
                group_id_tensor, target_group_val
            )
        else:
            raise Error("Unsupported target device: " + target)


# =============================================================================
# KERNEL 4: Compute Group ID
# Computes group_id = returnflag * 2 + linestatus
# =============================================================================

@always_inline
fn compute_group_id_cpu[
    dtype: DType
](
    group_id_out: LayoutTensor[mut=True, dtype, **_],
    returnflag: LayoutTensor[dtype, **_],
    linestatus: LayoutTensor[dtype, **_],
):
    """CPU kernel: Compute group_id = rf * 2 + ls."""
    for i in range(returnflag.size()):
        group_id_out[i] = returnflag[i][0] * 2 + linestatus[i][0]


@always_inline
fn compute_group_id_gpu[
    dtype: DType, layout: Layout
](
    group_id_out: LayoutTensor[dtype, layout, MutAnyOrigin],
    returnflag: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    linestatus: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """GPU kernel: Compute group_id per element."""
    var i = Int(global_idx.x)
    if i < returnflag.size():
        group_id_out[i] = returnflag[i][0] * 2 + linestatus[i][0]


@compiler.register("compute_group_id")
struct ComputeGroupIdOp:
    """Compute combined group ID from returnflag and linestatus."""
    
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType = DType.int32,
    ](
        group_id_out: OutputTensor[dtype=dtype, rank=1],
        returnflag: InputTensor[dtype=dtype, rank=1],
        linestatus: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        group_id_tensor = group_id_out.to_layout_tensor()
        rf_tensor = returnflag.to_layout_tensor().get_immutable()
        ls_tensor = linestatus.to_layout_tensor().get_immutable()

        @parameter
        if target == "cpu":
            compute_group_id_cpu[dtype](group_id_tensor, rf_tensor, ls_tensor)
        elif target == "gpu":
            # For GPU, use the CPU kernel for now
            # TODO: Implement proper GPU element-wise kernel
            compute_group_id_cpu[dtype](group_id_tensor, rf_tensor, ls_tensor)
        else:
            raise Error("Unsupported target device: " + target)
