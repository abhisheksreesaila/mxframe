# import %%mojo jupyter magic
import mojo.notebook

%%mojo package -o ops.mojopkg

import math
import compiler
from gpu import global_idx
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE


@always_inline
fn vector_sum_cpu[
    dtype: DType, layout: Layout
](
    output: LayoutTensor[mut=True, dtype, **_], # auto-parametrization with **_
    x: LayoutTensor[dtype, **_],
    y: LayoutTensor[dtype, **_],
):
    constrained[output.rank == x.rank == y.rank == 1]()
    for i in range(x.size()):
        output[i] = x[i][0] + y[i][0] # [0] for getting the SIMD value out

@always_inline
fn vector_sum_gpu[
    dtype: DType, layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    x: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    y: LayoutTensor[dtype, layout, ImmutAnyOrigin]
):
    i = global_idx.x
    if i < UInt(x.size()):
        output[i] = x[i] + y[i]

@compiler.register("vector_sum")
struct VectorSumCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        x: InputTensor[dtype=dtype, rank=1],
        y: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        comptime layout = type_of(output_tensor).layout
        output_tensor = output.to_layout_tensor()
        x_tensor = x.to_layout_tensor().get_immutable()
        y_tensor = y.to_layout_tensor().get_immutable()

        @parameter
        if target == "cpu":
            vector_sum_cpu[dtype, layout](output_tensor, x_tensor, y_tensor)
        elif target == "gpu":
            comptime kernel = vector_sum_gpu[dtype, layout]
            size = output_tensor.size()
            gpu_ctx = rebind[DeviceContext](ctx[])
            gpu_ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                x_tensor,
                y_tensor,
                grid_dim=1,
                block_dim=size,
            )
        else:
            raise Error("Unsupported target device")

def custom_vector_sum(x: Tensor, y: Tensor) -> Tensor:
    # TODO: expose custom kernel loading 
    from max.experimental import tensor
    from pathlib import Path
    tensor.GRAPH.graph._import_kernels([Path("ops.mojopkg")])
    return F.custom(name="vector_sum", device=x.device, values=[x, y], out_types=[x.type])