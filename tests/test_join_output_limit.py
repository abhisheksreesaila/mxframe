"""F-018 / F-016 fuzz: a many-to-many join whose output exceeds the int32 offsets of the join kernels
segfaulted in `join_scatter` (46,500 x 46,500 rows on one key = 2.16e9 pairs); it must raise MemoryError.

Run in a subprocess: a fault must fail one test, not take the session down.
"""
import os
import subprocess
import sys
import textwrap

from conftest import _shim

SCRIPT = textwrap.dedent("""
    import numpy as np
    from omxcore import aot_kernels
    from omxcore.custom_ops import CustomOpsCompiler

    assert aot_kernels.AOT_AVAILABLE
    aot = aot_kernels.AOTKernels()

    # counts that add up to more rows than int32 offsets address; keys are tiny, so a kernel call
    # would have written through wrapped offsets (and 26 GB of output arrays would have been allocated)
    keys = np.zeros(4, dtype=np.int32)
    counts = np.full(4, 1 << 30, dtype=np.int32)
    for fn in (aot.join_scatter, aot.join_scatter_left):
        try:
            fn(keys, keys, counts)
        except MemoryError as e:
            assert "rows" in str(e), e
        else:
            raise AssertionError("no MemoryError from " + fn.__name__)

    # the whole CPU join: 2^16 x 2^16 rows on one key is 4.3e9 pairs; counted, refused, never scattered
    comp = CustomOpsCompiler(device="cpu")
    one = np.zeros(1 << 16, dtype=np.int32)
    try:
        comp._hash_join_mojo_cpu(one, one)
    except MemoryError:
        pass
    else:
        raise AssertionError("no MemoryError from the CPU join")
    lo, ro = comp._hash_join_mojo_cpu(np.zeros(1000, dtype=np.int32), np.zeros(1000, dtype=np.int32))
    assert len(lo) == 1_000_000            # a big but addressable join still works
""")


def test_join_output_beyond_int32_raises_instead_of_faulting():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_shim) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-X", "faulthandler", "-c", SCRIPT], capture_output=True, text=True,
                       env=env, timeout=600)
    assert r.returncode == 0, (r.returncode, (r.stdout + r.stderr)[-3000:])
