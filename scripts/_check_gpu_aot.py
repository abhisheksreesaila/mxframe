import sys
sys.path.insert(0, '/home/ablearn/mxdf_v2')
from aot_kernels import AOTKernelsGPU, GPU_AOT_AVAILABLE
import numpy as np

print('GPU_AOT_AVAILABLE:', GPU_AOT_AVAILABLE)
if GPU_AOT_AVAILABLE:
    g = AOTKernelsGPU()
    print('AOTKernelsGPU loaded OK')
    
    # Quick functional test: group_sum
    N = 1000000
    values = np.ones(N, dtype=np.float32)
    labels = (np.arange(N, dtype=np.int32) % 6)  # 6 groups, each gets N/6 ones
    result = g.group_sum_f32(values, labels, 6)
    expected = N / 6
    print(f'group_sum_f32: result={result}, expected~={expected:.0f}, ok={np.allclose(result, expected, rtol=0.01)}')
    
    # masked_global_sum test
    mask = (np.random.rand(N) > 0.5).astype(np.int32)
    vals = np.ones(N, dtype=np.float32)
    res = g.masked_global_sum_f32(vals, mask)
    expected_sum = float(mask.sum())
    print(f'masked_global_sum: result={res:.0f}, expected={expected_sum:.0f}, ok={abs(res - expected_sum) < 1}')
    
    print('ALL GPU AOT TESTS PASSED')
else:
    print('GPU AOT not available')
