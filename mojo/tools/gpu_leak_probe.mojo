# F-146: runs N GPU scans that raise (an int64 overflow in the kernel) in one process, prints "failed N", and then waits
# 1.5 s before it exits so an outside observer can read the process's GPU memory (nvidia-smi) with the scans behind it.
# A scan that gave its device context back leaves nothing; one that kept it holds the CUDA context and its pool.
# Usage: gpu_leak_probe <path of a file whose sum overflows> <N>      (tests/test_mojo_parquet.py)
from std.sys import argv
from std.time import sleep
from mxframe import Query, col, sum, scan_parquet, GPU


def main() raises:
    var args = argv()
    var path = String(args[1])
    var n = Int(String(args[2]))
    var failed = 0
    for _ in range(n):
        try:
            _ = Query(scan_parquet(path)).agg(sum(col("v"))).compute(GPU)
        except:
            failed += 1
    print("failed", failed)
    sleep(1.5)
