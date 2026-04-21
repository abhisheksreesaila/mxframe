import sys
path = '/home/ablearn/mxdf_v2/scripts/_test_aot_smoke.py'
with open(path) as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines[104:], start=105):
    print(f"{i}: {line}", end='')
