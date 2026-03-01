# Generic Group Sum Kernel
# Single-pass scatter-sum indexed by a group_ids array.
#
# Computes sum(values[i]) for each group g where group_ids[i] == g.
# Supports up to MAX_GROUPS groups.
#
# Output layout:
#   GPU: [num_warps * n_groups]  — one partial sum per (warp, group); caller
#        reduces with reshape/transpose/sum in the MAX Graph.
#   CPU: [n_groups]              — direct final sums (no warp structure).
#
# n_groups is NOT passed as an input tensor — it is derived from the output
# shape baked into the compiled Graph:
#   GPU: out_size = ceil(N/32) * n_groups  → ng = out_size // num_warps
#   CPU: out_size = n_groups               → ng = out_size

import compiler
from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor

alias dtype = DType.float32
alias MAX_GROUPS = 64   # Maximum groups supported per kernel launch


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
    """GPU warp-reduction scatter-sum.

    Each thread contributes one element to one group bucket in its
    thread-local InlineArray, then warp_sum reduces across all 32 threads
    in a warp for each group index.  Lane 0 writes the partial sum for this
    warp to the output tensor.

    Output[warp_idx * n_groups + g] = partial sum of group g for this warp.
    """
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var warp_idx  = global_i // WARP_SIZE

    # Thread-local buckets — one slot per possible group
    var local_sums = InlineArray[Scalar[dtype], MAX_GROUPS](0)

    if global_i < size:
        var gid = Int(group_ids[global_i].reduce_add())
        var val = values[global_i].reduce_add()
        if gid >= 0 and gid < n_groups:
            local_sums[gid] = val

    # Warp-level reduction across all MAX_GROUPS buckets
    for g in range(n_groups):
        var sum_g = warp_sum(local_sums[g])
        if lane_id() == 0 and warp_idx < num_warps:
            output[warp_idx * n_groups + g] = sum_g


@compiler.register("group_sum")
struct GroupSum:
    """Generic single-pass scatter-sum kernel.

    Inputs:
        values    float32 [N]    values to accumulate
        group_ids int32   [N]    group index (0-based) for each row

    Output:
        GPU: float32 [ceil(N/32) * n_groups]  — warp partial sums
        CPU: float32 [n_groups]               — final sums

    n_groups is inferred from the output tensor shape baked at graph-compile
    time — no separate scalar input needed.
    """

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        values: InputTensor[dtype=dtype, rank=1],
        group_ids: InputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var size      = Int(values.spec().shape[0])
        var num_warps = ceildiv(size, WARP_SIZE)
        var out_len   = Int(output.spec().shape[0])

        @parameter
        if target == "gpu":
            # GPU: out_len = num_warps * n_groups
            var ng = out_len // num_warps

            alias BLOCK_SIZE = 256
            var num_blocks = ceildiv(size, BLOCK_SIZE)

            var out_tensor = output.to_layout_tensor()
            var val_tensor = values.to_layout_tensor().get_immutable()
            var gid_tensor = group_ids.to_layout_tensor().get_immutable()

            # Bind layout type parameters at compile time (required by MAX JIT elaboration)
            alias out_layout_t = out_tensor.layout
            alias val_layout_t = val_tensor.layout
            alias gid_layout_t = gid_tensor.layout

            var gpu_ctx = rebind[DeviceContext](ctx[])
            gpu_ctx.enqueue_function_checked[
                group_sum_kernel[val_layout_t, gid_layout_t, out_layout_t],
                group_sum_kernel[val_layout_t, gid_layout_t, out_layout_t],
            ](
                out_tensor,
                val_tensor,
                gid_tensor,
                size,
                ng,
                num_warps,
                grid_dim=num_blocks,
                block_dim=BLOCK_SIZE,
            )
        else:
            # CPU: out_len = n_groups — scalar scatter-add, no warp structure
            var ng      = out_len
            var out_ptr = output.unsafe_ptr()
            var val_ptr = values.unsafe_ptr()
            var gid_ptr = group_ids.unsafe_ptr()

            for i in range(ng):
                out_ptr[i] = 0.0

            for i in range(size):
                var g = Int(gid_ptr[i])
                if g >= 0 and g < ng:
                    out_ptr[g] += val_ptr[i]
