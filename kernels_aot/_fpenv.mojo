"""
_fpenv.mojo (F-082) -- parallelize / sync_parallelize that keep denormal floats.

Mojo's `max.algorithm.parallelize` and `sync_parallelize` run every task with the CPU's flush-to-zero (FTZ)
and denormals-are-zero (DAZ) bits set (MXCSR 0x9FE0 instead of the default 0x1FA0; measured with stmxcsr in a
task). A float64 below 2.2e-308 (or a float32 below 1.2e-38) then reads as 0 inside the kernel: `x > 0.0` on
5e-324 was false and `sum(x)` was 0, where DuckDB, NumPy and Arrow keep the value. Every kernel launches its
workers through `par` / `par_n` / `sync_par` below, which reset MXCSR to the IEEE default at the start of each task
(the runtime restores its own value when the task returns). Never call `parallelize` directly in this package.
"""

from max.algorithm import parallelize, sync_parallelize
from std.sys._assembly import inlined_assembly


@always_inline
def ieee_denormals():
    """MXCSR = 0x1F80: all exceptions masked, round to nearest, FTZ and DAZ off."""
    inlined_assembly[
        "pushq $$0x1F80; ldmxcsr (%rsp); addq $$8, %rsp",
        NoneType,
        constraints="~{memory},~{rsp},~{flags}",
        has_side_effect=True,
    ]()


def par[F: def(Int) -> None](f: F, n: Int):
    """parallelize(f, n) with denormals kept."""

    def task(i: Int) {imm f}:
        ieee_denormals()
        f(i)

    parallelize(task, n)


def par_n[F: def(Int) -> None](f: F, n: Int, nthreads: Int):
    """parallelize(f, n, nthreads) with denormals kept."""

    def task(i: Int) {imm f}:
        ieee_denormals()
        f(i)

    parallelize(task, n, nthreads)


def sync_par[F: def(Int) -> None](f: F, n: Int):
    """sync_parallelize(f, n) with denormals kept."""

    def task(i: Int) {imm f}:
        ieee_denormals()
        f(i)

    sync_parallelize(task, n)
