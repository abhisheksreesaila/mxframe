"""
_atomic.mojo — Mojo 26.4 drop-in for `from os.atomic import Atomic`.
Uses MLIR llvm.atomicrmw / llvm.cmpxchg — works in both mojo package
and mojo build --emit shared-lib (including GPU shared memory via addrspacecast).

AtomicBinOp: xchg=0, add=1(int), or=5, fadd=11, fmax=13, fmin=14
AtomicOrdering: monotonic=2, acquire=4, release=5, seq_cst=7
add/min/max use monotonic (CUDA atomicAdd semantics); seq_cst makes NVPTX fence around every atomic.
"""

from max.gpu.memory import AddressSpace
from std.sys._assembly import inlined_assembly
from std.memory import Pointer


@always_inline
def _to_llvm_ptr[space: AddressSpace](ptr_addr: __mlir_type.`!kgen.pointer<scalar<f32>>`) -> __mlir_type.`!llvm.ptr`:
    """Convert kgen.pointer → !llvm.ptr, handling shared memory via addrspacecast."""
    comptime if space == AddressSpace.SHARED:
        var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr_addr)
        return __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
    else:
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr_addr)


struct Atomic:

    @staticmethod
    def fetch_add[space: AddressSpace = AddressSpace.GENERIC](
        ptr: Pointer[Float32, MutUntrackedOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic float add via MLIR llvm.atomicrmw (fadd=11)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr._mlir_value)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`11 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.f32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Float32]](r)

    @staticmethod
    def fetch_add[space: AddressSpace = AddressSpace.GENERIC](
        ptr: Pointer[Int32, MutUntrackedOrigin, address_space=space],
        val: Int32,
    ) -> Int32:
        """Atomic int32 add via MLIR llvm.atomicrmw (add=1)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr._mlir_value)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`1 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Int32]](r)

    @staticmethod
    def min[space: AddressSpace = AddressSpace.GENERIC](
        ptr: Pointer[Float32, MutUntrackedOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic float min via MLIR llvm.atomicrmw (fmin=14)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr._mlir_value)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`14 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.f32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Float32]](r)

    @staticmethod
    def max[space: AddressSpace = AddressSpace.GENERIC](
        ptr: Pointer[Float32, MutUntrackedOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic float max via MLIR llvm.atomicrmw (fmax=13)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr._mlir_value)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`13 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.f32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Float32]](r)

    @staticmethod
    def compare_exchange(
        ptr: Pointer[Int32, MutUntrackedOrigin],
        mut expected: Int32,
        desired: Int32,
    ) -> Bool:
        """True CAS on global int32 memory via MLIR llvm.cmpxchg (seq_cst; a failed CAS is an
        acquire load of the value it found, so a published slot is safe to read)."""
        var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var cmp = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](expected)
        var new = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](desired)
        var result = __mlir_op.`llvm.cmpxchg`[
            success_ordering = __mlir_attr.`7 : i64`,
            failure_ordering = __mlir_attr.`4 : i64`,
            _type = __mlir_type.`!llvm.struct<(i32, i1)>`,
        ](p, cmp, new)
        var old_i32 = __mlir_op.`llvm.extractvalue`[position = __mlir_attr.`array<i64: 0>`, _type = __mlir_type.i32](result)
        var suc_i1  = __mlir_op.`llvm.extractvalue`[position = __mlir_attr.`array<i64: 1>`, _type = __mlir_type.`i1`](result)
        var ok = Bool(__mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`i1`](suc_i1))
        if not ok:
            expected = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Int32]](old_i32)
        return ok

    @staticmethod
    def load(ptr: Pointer[Int32, MutUntrackedOrigin]) -> Int32:
        """Acquire load of global int32 memory that always reads L2, never a
        stale L1 line (atomicrmw or 0; bin_op or=5, ordering acquire=4).
        Plain loads in a spin-wait can be served from L1 forever."""
        var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](Int32(0))
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`5 : i64`, ordering = __mlir_attr.`4 : i64`, _type = __mlir_type.i32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Int32]](r)

    @staticmethod
    def store(ptr: Pointer[Int32, MutUntrackedOrigin], val: Int32):
        """Release store of global int32 memory (atomicrmw xchg=0, ordering
        release=5): earlier plain writes are visible to a thread that
        acquire-loads this value."""
        var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](val)
        _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`0 : i64`, ordering = __mlir_attr.`5 : i64`, _type = __mlir_type.i32](p, v)

    @staticmethod
    def max(ptr: Pointer[Int32, MutUntrackedOrigin], val: Int32) -> Int32:
        """Atomic signed int32 max via MLIR llvm.atomicrmw (max=7), global memory."""
        var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`7 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Int32]](r)

    @staticmethod
    def red_add_global(addr: Int, val: Float32):
        """Fire-and-forget native `red.global.add.f32` on a global address.
        `cvta.to.global` converts the generic address a caller holds to the global
        window first, as nvcc does; a `.global` instruction on a generic address is
        undefined in the PTX spec."""
        inlined_assembly["{ .reg .u64 %g; cvta.to.global.u64 %g, $0; "
                         "red.relaxed.gpu.global.add.f32 [%g], $1; }", NoneType,
                         constraints="l,f,~{memory}"](addr, val)
