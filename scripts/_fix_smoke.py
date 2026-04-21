"""Fix stale mxdf_v2 imports in _test_aot_smoke.py."""
path = '/home/ablearn/mxdf_v2/scripts/_test_aot_smoke.py'
with open(path) as f:
    c = f.read()
c = c.replace("from mxdf_v2.aot_kernels", "from mxframe.aot_kernels")
c = c.replace("sys.path.insert(0, '/home/ablearn/mxdf_v2')\n", "")
c = c.replace("import traceback\n", "")
c = c.replace("np.array(got)", "np.array(got if hasattr(got,'__len__') else [got], dtype=float)")
c = c.replace("np.array(expected)", "np.array(expected if hasattr(expected,'__len__') else [expected], dtype=float)")
with open(path, 'w') as f:
    f.write(c)
print("Fixed smoke test imports")
