"""F-020: the GPU hash_join sums match counts in int32; an output beyond int32 must raise MemoryError before
the scatter, like the CPU `check_join_output_size`.

Run in a subprocess: if the guard were missing the scatter kernel would write through wrapped offsets and
fault, which must fail one test, not take the session down.  The input is small (two 50,000-row key
columns); nothing near 2.5e9 pairs is ever allocated.
"""
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import _shim

SCRIPT = textwrap.dedent("""
    import numpy as np
    from omxcore.aot_kernels import GPU_AOT_AVAILABLE, get_aot_gpu
    if not GPU_AOT_AVAILABLE:
        print("NO_GPU")
        raise SystemExit(0)
    g = get_aot_gpu()
    if g is None:
        print("NO_GPU")
        raise SystemExit(0)

    # 50,000 x 50,000 on one key = 2.5e9 pairs (> 2**31 - 1)
    keys = np.zeros(50_000, dtype=np.int32)
    try:
        g.hash_join(keys, keys)
    except MemoryError as e:
        assert "rows" in str(e), e
    else:
        raise AssertionError("no MemoryError for a 2.5e9-row join")

    # the guard must not get in the way of joins that fit: the bound is exceeded (n_left * max matches)
    # but the exact total is small, and a many-to-one join is untouched
    left = np.arange(40_000, dtype=np.int32) % 1000
    right = np.concatenate([np.arange(1000, dtype=np.int32), np.zeros(3, dtype=np.int32)])
    li, ri = g.hash_join(left, right)
    assert len(li) == len(ri) == 40_000 + 3 * 40
    li, ri = g.hash_join(np.arange(100, dtype=np.int32), np.arange(50, dtype=np.int32))
    assert len(li) == 50
    print("OK")
""")


def test_gpu_hash_join_refuses_an_output_beyond_int32():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_shim) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    if "NO_GPU" in r.stdout:
        pytest.skip("no GPU AOT kernels")
    assert "OK" in r.stdout, (r.stdout + r.stderr)[-1500:]
