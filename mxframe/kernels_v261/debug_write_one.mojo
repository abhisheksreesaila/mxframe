import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

alias dtype = DType.float32


@compiler.register("debug_write_one")
struct DebugWriteOne:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=...],
        input: InputTensor[dtype=dtype, rank=1, static_spec=...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn elementwise_add_one[
            width: Int,
        ](idx: IndexList[input.rank]) -> SIMD[input.dtype, width]:
            return input.load[width](idx) + 1

        foreach[elementwise_add_one, target=target](output, ctx)
