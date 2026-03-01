# Standalone test for group_sum_kernel — no MAX Graph, no Python, pure Mojo.
#
# Usage:
#   cd /home/ablearn/mxdf
#   pixi run mojo archive/test_group_sum.mojo
#
# Test data:
#   N = 8 values, 2 groups
#   values    = [1, 2, 3, 4, 5, 6, 7, 8]
#   group_ids = [0, 1, 0, 1, 0, 1, 0, 1]
#
# Expected results:
#   group 0 = 1 + 3 + 5 + 7 = 16.0
#   group 1 = 2 + 4 + 6 + 8 = 20.0
#
# GPU output layout (N=8 fits in 1 warp → num_warps=1):
#   out[0 * 2 + 0] = out[0] = 16.0   (warp 0, group 0)
#   out[0 * 2 + 1] = out[1] = 20.0   (warp 0, group 1)

from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.host import DeviceContext
from memory import UnsafePointer
from layout import Layout, LayoutTensor

alias dtype = DType.float32
alias MAX_GROUPS = 64

# ── compile-time test parameters ────────────────────────────────────────────
alias N         = 8          # number of rows (< WARP_SIZE → 1 warp)
alias NG        = 2          # number of groups
alias NW        = 1          # num_warps = ceil(8 / 32) = 1  (GPU output rows)
alias BLOCK_SZ  = 32         # threads per block (one warp for clarity)
alias N_BLOCKS  = 1          # ceil(N / BLOCK_SZ) = 1

# flat 1-D layouts baked at compile time
alias val_layout = Layout.row_major(N)
alias gid_layout = Layout.row_major(N)
alias out_layout_gpu = Layout.row_major(NW * NG)   # 1 warp × 2 groups = 2
alias out_layout_cpu = Layout.row_major(NG)         # 2 final sums


# ── kernel (identical copy from group_sum.mojo) ──────────────────────────────
fn group_sum_kernel[
    val_layout_t: Layout,
    gid_layout_t: Layout,
    out_layout_t: Layout,
](
    output:    LayoutTensor[dtype,         out_layout_t, MutableAnyOrigin],
    values:    LayoutTensor[dtype,         val_layout_t, ImmutableAnyOrigin],
    group_ids: LayoutTensor[DType.int32,   gid_layout_t, ImmutableAnyOrigin],
    size:      Int,
    n_groups:  Int,
    num_warps: Int,
):
    """GPU warp-reduction scatter-sum (same code as in group_sum.mojo)."""
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var warp_idx  = global_i // WARP_SIZE

    var local_sums = InlineArray[Scalar[dtype], MAX_GROUPS](0)

    if global_i < size:
        var gid = Int(group_ids[global_i].reduce_add())
        var val = values[global_i].reduce_add()
        if gid >= 0 and gid < n_groups:
            local_sums[gid] = val

    for g in range(n_groups):
        var sum_g = warp_sum(local_sums[g])
        if lane_id() == 0 and warp_idx < num_warps:
            output[warp_idx * n_groups + g] = sum_g


# ── SMOKE TEST: minimal GPU kernel (just writes thread ID into output) ────────
fn smoke_kernel(
    out: UnsafePointer[Float32],
    n:   Int,
):
    """Writes the global thread index into out[i]. If this crashes → basic
    GPU launch on this device is broken regardless of our kernel code."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        out[i] = Float32(i)


fn test_smoke_gpu() raises:
    print("─── Smoke test: minimal GPU launch ────────────────────")
    var ctx = DeviceContext()

    var d_out = ctx.alloc[DType.float32](N)
    ctx.enqueue_function[smoke_kernel](
        d_out.unsafe_ptr(), N,
        grid_dim  = N_BLOCKS,
        block_dim = BLOCK_SZ,
    )
    ctx.synchronize()

    # copy back
    var h_out = UnsafePointer[Float32].alloc(N)
    ctx.enqueue_copy(h_out, d_out.unsafe_ptr(), N)
    ctx.synchronize()

    print("  out[0..7] =", end=" ")
    for i in range(N):
        print(h_out[i], end=" ")
    print()

    # verify: out[i] should equal Float32(i)
    var ok = True
    for i in range(N):
        if h_out[i] != Float32(i):
            ok = False
            print("  MISMATCH at i=", i, " expected", Float32(i), " got", h_out[i])

    h_out.free()
    if ok:
        print("  PASSED ✓")
    else:
        print("  FAILED ✗")


# ── CPU test ─────────────────────────────────────────────────────────────────
fn test_cpu() raises:
    print("─── CPU group_sum test ─────────────────────────────────")

    # host buffers
    var h_vals = UnsafePointer[Float32].alloc(N)
    var h_gids = UnsafePointer[Int32].alloc(N)
    var h_out  = UnsafePointer[Float32].alloc(NG)

    for i in range(N):
        h_vals[i] = Float32(i + 1)         # [1,2,3,4,5,6,7,8]
        h_gids[i] = Int32(i % 2)           # [0,1,0,1,0,1,0,1]
    for g in range(NG):
        h_out[g] = 0.0

    # wrap in LayoutTensors for the CPU kernel call
    var val_tensor = LayoutTensor[dtype,       val_layout, ImmutableAnyOrigin](h_vals)
    var gid_tensor = LayoutTensor[DType.int32, gid_layout, ImmutableAnyOrigin](h_gids)
    var out_tensor = LayoutTensor[dtype,       out_layout_cpu, MutableAnyOrigin](h_out)

    # direct CPU path: call the kernel-level logic inline (no warp reduction)
    for i in range(NG):
        out_tensor[i] = Float32(0)
    for i in range(N):
        var g = Int(gid_tensor[i].reduce_add())
        if g >= 0 and g < NG:
            out_tensor[g] = out_tensor[g].reduce_add() + val_tensor[i].reduce_add()

    print("  group 0 sum =", h_out[0], " (expected 16.0)")
    print("  group 1 sum =", h_out[1], " (expected 20.0)")

    var ok = (h_out[0] == 16.0 and h_out[1] == 20.0)
    if ok:
        print("  PASSED ✓")
    else:
        print("  FAILED ✗")

    h_vals.free()
    h_gids.free()
    h_out.free()


# ── GPU test: calls group_sum_kernel directly ─────────────────────────────────
fn test_gpu() raises:
    print("─── GPU group_sum_kernel test ──────────────────────────")

    var ctx = DeviceContext()

    # ── host data
    var h_vals = UnsafePointer[Float32].alloc(N)
    var h_gids = UnsafePointer[Int32].alloc(N)
    for i in range(N):
        h_vals[i] = Float32(i + 1)   # [1,2,3,4,5,6,7,8]
        h_gids[i] = Int32(i % 2)     # [0,1,0,1,0,1,0,1]

    # ── device buffers
    var d_vals = ctx.alloc[DType.float32](N)
    var d_gids = ctx.alloc[DType.int32](N)
    var d_out  = ctx.alloc[DType.float32](NW * NG)

    # ── copy host → device
    ctx.enqueue_copy(d_vals.unsafe_ptr(), h_vals, N)
    ctx.enqueue_copy(d_gids.unsafe_ptr(), h_gids, N)
    ctx.synchronize()

    # ── wrap device pointers in LayoutTensors
    var val_tensor = LayoutTensor[dtype,       val_layout,     ImmutableAnyOrigin](d_vals.unsafe_ptr())
    var gid_tensor = LayoutTensor[DType.int32, gid_layout,     ImmutableAnyOrigin](d_gids.unsafe_ptr())
    var out_tensor = LayoutTensor[dtype,       out_layout_gpu,  MutableAnyOrigin](d_out.unsafe_ptr())

    # ── launch kernel
    ctx.enqueue_function[
        group_sum_kernel[val_layout, gid_layout, out_layout_gpu],
    ](
        out_tensor, val_tensor, gid_tensor,
        N, NG, NW,
        grid_dim  = N_BLOCKS,
        block_dim = BLOCK_SZ,
    )
    ctx.synchronize()

    # ── copy result back to host
    var h_out = UnsafePointer[Float32].alloc(NW * NG)
    ctx.enqueue_copy(h_out, d_out.unsafe_ptr(), NW * NG)
    ctx.synchronize()

    # GPU layout: out[warp_idx * NG + g]
    # warp_idx=0, group_0 → h_out[0], group_1 → h_out[1]
    print("  out[0] (warp0, group0) =", h_out[0], " (expected 16.0)")
    print("  out[1] (warp0, group1) =", h_out[1], " (expected 20.0)")

    var ok = (h_out[0] == 16.0 and h_out[1] == 20.0)
    if ok:
        print("  PASSED ✓")
    else:
        print("  FAILED ✗")

    h_vals.free()
    h_gids.free()
    h_out.free()


fn main() raises:
    # 1. cpu logic check (no GPU involved at all)
    test_cpu()
    print()

    # 2. minimal GPU smoke test — just writes thread IDs
    #    If THIS fails → basic CUDA launch is broken on this device
    test_smoke_gpu()
    print()

    # 3. full group_sum_kernel on GPU
    #    If (2) passed but (3) fails → the kernel implementation has an issue
    test_gpu()
