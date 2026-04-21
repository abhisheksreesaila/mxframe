from mxframe.custom_ops import KERNELS_PATH, CustomOpsCompiler
print("KERNELS_PATH =", KERNELS_PATH)
comp = CustomOpsCompiler(device="cpu")
print("comp.kernels_path =", comp.kernels_path)
print("_aot =", comp._aot)
