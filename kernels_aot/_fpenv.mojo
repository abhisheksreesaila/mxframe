"""
_fpenv.mojo (F-082) -- parallelize / sync_parallelize that keep denormal floats.

Mojo's `max.algorithm.parallelize` and `sync_parallelize` run every task with the CPU's flush-to-zero (FTZ)
and denormals-are-zero (DAZ) bits set (MXCSR 0x9FE0 instead of the default 0x1FA0; measured with stmxcsr in a
task). A float64 below 2.2e-308 (or a float32 below 1.2e-38) then reads as 0 inside the kernel: `x > 0.0` on
5e-324 was false and `sum(x)` was 0, where DuckDB, NumPy and Arrow keep the value. Every kernel launches its
workers through `par` / `par_n` / `sync_par` below, which reset MXCSR to the IEEE default at the start of each task
(the runtime restores its own value when the task returns). Never call `parallelize` directly in this package.

F-138b: the serial build. An Apple app (the omxplorer Mac build) links no Mojo async runtime, so every task runs in
a plain loop on the calling thread and `core_count()` is 1. SERIAL is on for any target but Linux, and
`-D MXFRAME_SERIAL` turns it on for Linux too (to test it). A Linux build with the switch off is unchanged.
"""

from max.algorithm import parallelize, sync_parallelize
from std.sys import CompilationTarget, is_defined
from std.sys.info import num_physical_cores
from std.ffi import external_call
from std.sys._assembly import inlined_assembly
from std.memory import Pointer


comptime SERIAL = is_defined["MXFRAME_SERIAL"]() or not CompilationTarget.is_linux()


def core_count() -> Int:
    """The workers a kernel should start: the physical cores, or 1 on a serial build (no thread pool)."""
    comptime if SERIAL:
        return 1
    else:
        return num_physical_cores()


def start_pool():
    """A host process (Python via ctypes) has no Mojo async runtime yet: create the global CPU device (the thread
    pool) once before the first `par`. A serial build has no pool and links none."""
    comptime if not SERIAL:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


@always_inline
def ieee_denormals():
    """x86: MXCSR = 0x1F80 (all exceptions masked, round to nearest, FTZ and DAZ off), loaded from a local variable.
    Elsewhere a no-op: the ARM equivalent is the FPCR FZ bit, which is UNCHECKED here (no aarch64 build has been
    made or tested; check whether Mojo's parallelize sets FZ there before shipping an ARM kernel)."""
    comptime if CompilationTarget.is_x86():
        var csr = Int32(0x1F80)
        inlined_assembly[
            "ldmxcsr ($0)",
            NoneType,
            constraints="r,~{memory}",
            has_side_effect=True,
        ](Pointer(to=csr))


def par[F: def(Int) -> None](f: F, n: Int):
    """parallelize(f, n) with denormals kept."""

    comptime if SERIAL:
        for i in range(n):
            f(i)
    else:
        def task(i: Int) {imm f}:
            ieee_denormals()
            f(i)

        parallelize(task, n)


def par_n[F: def(Int) -> None](f: F, n: Int, nthreads: Int):
    """parallelize(f, n, nthreads) with denormals kept."""

    comptime if SERIAL:
        for i in range(n):
            f(i)
    else:
        def task(i: Int) {imm f}:
            ieee_denormals()
            f(i)

        parallelize(task, n, nthreads)


def sync_par[F: def(Int) -> None](f: F, n: Int):
    """sync_parallelize(f, n) with denormals kept."""

    comptime if SERIAL:
        for i in range(n):
            f(i)
    else:
        def task(i: Int) {imm f}:
            ieee_denormals()
            f(i)

        sync_parallelize(task, n)
