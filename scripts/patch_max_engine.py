"""
Patch nbs/03_max_engine.ipynb to fix:
1. gpu_masked_sum_kernel: fallback if not kernels_available() OR device != 'gpu'
2. gpu_fused_group_agg: same fallback AND replace ops.slice with
   ops.transpose + ops.sum + ops.reshape + ops.split
"""
import json

path = "/home/ablearn/mxdf/nbs/03_max_engine.ipynb"
nb = json.load(open(path))

# Print current code cells (code cells only)
code_cells = [(i,c) for i,c in enumerate(nb["cells"]) if c["cell_type"]=="code"]
print(f"Total code cells: {len(code_cells)}")
for idx, (i, c) in enumerate(code_cells):
    src = "".join(c["source"])
    print(f"  code[{idx}] nb[{i}] | {src[:60].replace(chr(10),' ')}")

# Find cell indices
def find_cell(nb, keyword):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"]=="code" and keyword in "".join(c["source"]):
            return i
    return None

idx_masked = find_cell(nb, "gpu_masked_sum_kernel")
idx_fused  = find_cell(nb, "gpu_fused_group_agg")
print(f"\ngpu_masked_sum_kernel at cell[{idx_masked}]")
print(f"gpu_fused_group_agg   at cell[{idx_fused}]")

NEW_MASKED_SUM = """\
#| export
def gpu_masked_sum_kernel(
    mask:   np.ndarray,    # Boolean/float32 mask (1.0 = include)
    values: np.ndarray,    # Values to sum where mask is true
    device: DeviceType = "gpu",
) -> float:
    \"\"\"
    Masked sum via custom Mojo warp-reduction kernel.

    One kernel launch: `mask * values` → warp partial sums → scalar.
    Compiled once per array shape and cached in `MAXSession`.

    Falls back to `max_masked_sum()` (pure Graph ops) if
    `kernels.mojopkg` is not found *or* if `device='cpu'`
    (the Mojo kernel only executes on GPU hardware).

    Args:
        mask:   float32 array, 1.0 = include row
        values: float32 array to sum
        device: `'cpu'` or `'gpu'`

    Returns:
        Scalar sum as float.
    \"\"\"
    if not kernels_available() or device != "gpu":
        return max_masked_sum(mask, values, device=device)

    sess = MAXSession.get(device)
    n    = len(mask)
    m_f  = np.ascontiguousarray(mask,   dtype=np.float32)
    v_f  = np.ascontiguousarray(values, dtype=np.float32)
    m_t  = sess.to_tensor(m_f)
    v_t  = sess.to_tensor(v_f)

    out_size = _partial_sum_size(n)
    key      = ("masked_sum_kernel", n, "float32")

    def build():
        in_type  = TensorType(DType.float32, (n,),       sess.device_ref)
        out_type = TensorType(DType.float32, (out_size,), sess.device_ref)
        with Graph(
            "masked_sum_kernel",
            input_types=[in_type, in_type],
            custom_extensions=[KERNELS_PATH],
        ) as g:
            m, v     = g.inputs
            partials = ops.custom(
                name      = "masked_sum_simple",
                device    = sess.device_ref,
                values    = [m, v],
                out_types = [out_type],
            )[0]
            g.output(ops.sum(partials))
        return g

    model  = sess._get_or_compile(key, build)
    result = model.execute(m_t, v_t)[0]
    return float(sess.from_tensor(result).flat[0])\
"""

NEW_FUSED = """\
#| export
# Canonical column order that fused_group_agg kernel expects / returns
_FUSED_AGG_COLS = ("sum_qty", "sum_base_price", "sum_disc", "sum_disc_price", "sum_charge", "count_order")
_FUSED_AGG_IDX  = {name: i for i, name in enumerate(_FUSED_AGG_COLS)}


def gpu_fused_group_agg(
    mask:       np.ndarray,   # float32, shape [N] — group membership mask
    qty:        np.ndarray,   # float32, shape [N]
    price:      np.ndarray,   # float32, shape [N]
    disc:       np.ndarray,   # float32, shape [N]
    disc_price: np.ndarray,   # float32, shape [N]  = price × (1−disc)
    charge:     np.ndarray,   # float32, shape [N]  = disc_price × (1+tax)
    device: DeviceType = "gpu",
) -> dict:
    \"\"\"
    6-column fused aggregation for one group via custom Mojo kernel.

    Returns a dict with keys matching `_FUSED_AGG_COLS`:
      `sum_qty`, `sum_base_price`, `sum_disc`,
      `sum_disc_price`, `sum_charge`, `count_order`

    Falls back to six separate `max_masked_sum()` calls if
    `kernels.mojopkg` is not found *or* if `device='cpu'`
    (the Mojo kernel only executes on GPU hardware).

    Args:
        mask:       float32 boolean mask (1.0 = row in this group)
        qty / price / disc / disc_price / charge: pre-computed columns
        device:     `'cpu'` or `'gpu'`

    Returns:
        dict[str, float] with the 6 aggregation values.
    \"\"\"
    if not kernels_available() or device != "gpu":
        ms = lambda v: max_masked_sum(mask, v, device=device)
        return {
            "sum_qty":        ms(qty),
            "sum_base_price": ms(price),
            "sum_disc":       ms(disc),
            "sum_disc_price": ms(disc_price),
            "sum_charge":     ms(charge),
            "count_order":    float(mask.astype(bool).sum()),
        }

    sess = MAXSession.get(device)
    n    = len(mask)

    def _t(arr): return sess.to_tensor(np.ascontiguousarray(arr, dtype=np.float32))
    m_t, q_t, p_t, d_t, dp_t, ch_t = map(_t, [mask, qty, price, disc, disc_price, charge])

    num_warps = _partial_sum_size(n)
    out_size  = num_warps * 6
    key       = ("fused_group_agg", n, "float32")

    def build():
        tt     = TensorType(DType.float32, (n,),       sess.device_ref)
        out_tt = TensorType(DType.float32, (out_size,), sess.device_ref)
        with Graph(
            "fused_group_agg",
            input_types=[tt, tt, tt, tt, tt, tt],
            custom_extensions=[KERNELS_PATH],
        ) as g:
            m, q, p, d, dp, ch = g.inputs
            partials = ops.custom(
                name      = "fused_group_agg",
                device    = sess.device_ref,
                values    = [m, q, p, d, dp, ch],
                out_types = [out_tt],
            )[0]
            # Kernel outputs [num_warps * 6] interleaved (warp-major, agg-minor).
            # Reduce: reshape → [num_warps, 6], transpose → [6, num_warps],
            #   sum rows → [6, 1], reshape → [6], split → 6 × [1] scalars.
            shaped   = ops.reshape(partials, [num_warps, 6])    # [NW, 6]
            transpos = ops.transpose(shaped, 0, 1)              # [6, NW]
            col_sums = ops.sum(transpos)                        # [6, 1]
            flat6    = ops.reshape(col_sums, [6])               # [6]
            cols     = ops.split(flat6, [1,1,1,1,1,1], 0)      # 6 × [1]
            g.output(*cols)
        return g

    model   = sess._get_or_compile(key, build)
    outputs = model.execute(m_t, q_t, p_t, d_t, dp_t, ch_t)
    return {
        name: float(sess.from_tensor(outputs[i]).flat[0])
        for i, name in enumerate(_FUSED_AGG_COLS)
    }\
"""

# Apply patches
if idx_masked is not None:
    nb["cells"][idx_masked]["source"] = [NEW_MASKED_SUM]
    print(f"Patched cell[{idx_masked}]: gpu_masked_sum_kernel")

if idx_fused is not None:
    nb["cells"][idx_fused]["source"] = [NEW_FUSED]
    print(f"Patched cell[{idx_fused}]: gpu_fused_group_agg")

json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
print("Saved.")
