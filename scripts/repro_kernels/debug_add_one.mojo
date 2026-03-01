import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

alias dtype = DType.float32


@compiler.register("repro_debug_add_one")
struct ReproDebugAddOne:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        input: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn add_one[width: Int](idx: IndexList[input.rank]) -> SIMD[input.dtype, width]:
            return input.load[width](idx) + 1

        foreach[add_one, target=target](output, ctx)
