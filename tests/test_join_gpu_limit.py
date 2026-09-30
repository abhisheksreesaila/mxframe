"""F-020: the GPU hash_join sums match counts in int32 and does not guard the overflow."""
import pytest


@pytest.mark.skip(reason="F-020: GPU hash_join prefix sum is int32 and unguarded; running this would fault the process")
def test_gpu_hash_join_refuses_an_output_beyond_int32():
    from mxframe.aot_kernels import GPU_AOT_AVAILABLE, get_aot_gpu
    if not GPU_AOT_AVAILABLE:
        pytest.skip("no GPU AOT kernels")
    import numpy as np
    g = get_aot_gpu()
    keys = np.zeros(50_000, dtype=np.int32)          # 50,000 x 50,000 on one key = 2.5e9 pairs
    with pytest.raises(MemoryError):
        g.hash_join(keys, keys)
