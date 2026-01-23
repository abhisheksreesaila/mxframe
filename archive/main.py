import pyarrow as pa
import ctypes

# 1. Create a simple Arrow Array (Int64)
# Note: 'None' creates a null value, triggering the 'Validity Bitmap'
data = [10, 20, None, 40, 50]
arr = pa.array(data, type=pa.int64())

print(f"Array: {arr}")
print(f"Length: {len(arr)}")
print(f"Number of buffers: {len(arr.buffers())}") 

# 2. Access the physical buffers
# Buffer 0: Validity Bitmap (tells us which indices are null)
# Buffer 1: Values Buffer (the actual numbers)
val_buf = arr.buffers()[1]

print(f"--- Physical Memory Info ---")
print(f"Values Buffer Address: {val_buf.address}")
print(f"Values Buffer Size (bytes): {val_buf.size}")

# 3. PROOF: Peek at the raw memory using ctypes
# We'll read the first 8 bytes (one int64) at that address
first_val = ctypes.c_int64.from_address(val_buf.address).value
print(f"Raw Value at address {val_buf.address}: {first_val}") # Should be 10