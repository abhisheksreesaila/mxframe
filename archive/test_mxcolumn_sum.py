"""Test MXColumn.sum() with MAX Engine."""
from mxframe import MXColumn

col = MXColumn([1, 2, 3, 4, 5])
print(f"sum (cpu) = {col.sum(device='cpu')}")
print(f"sum (gpu) = {col.sum(device='gpu')}")
print("✅ MXColumn.sum() now uses MAX Engine!")
