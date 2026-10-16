"""F-107: concurrent device="auto" computes with background resident uploads and the GPU warm-up must not hang.

A child process (with a timeout, so a hang fails instead of freezing the suite) runs several threads that each
compute fresh tables again and again -- every compute queues an upload, every dropped table fires the
residency finalizer -- with the cyclic GC set to run at almost every allocation (the trigger of the F-107
deadlock).  A watchdog dumps every thread's stack with faulthandler and exits if no compute finishes for a while.
Short by default; ``MXFRAME_F107_STRESS=1`` runs a long version.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mxframe import pipeline_kernels as pk

HERE = Path(__file__).resolve().parent
LONG = os.environ.get("MXFRAME_F107_STRESS") == "1"

CHILD = """
import faulthandler, gc, sys, threading, time
sys.path.insert(0, ".")
import conftest  # noqa: F401
import numpy as np, pyarrow as pa
import mxframe as mx
from mxframe import LazyFrame, col, lit, coproc_fused as cf, auto_mode as am

THREADS, ITERS, STALL = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
from mxframe import calibration as cal
cf.MIN_UPLOAD_BYTES = 1
cf.IDLE_BEFORE_BACKGROUND_S = 0.0

class Calib:                              # says "a resident split wins", so every compute queues an upload
    has_gpu = True
    def estimate(self, op, device, rows, card=None, **k):
        if op != "pipeline":
            return 10.0 if device == "cpu" else 1e6
        return 100.0 if device == "cpu" else 30.0
    def estimate_gpu_join_total(self, *a, **k): return 1e6
    def transfer_ms(self, *a, **k): return 0.0
    def gpu_init_cost_ms(self): return 200.0
    def observe(self, *a, **k): pass
cal.get_calibration = lambda: Calib()
uploads = [0]
_real_start = cf._start_upload
def counted(*a, **k):
    uploads[0] += 1
    return _real_start(*a, **k)
cf._start_upload = counted
done = [0]
progress = threading.Lock()

def watchdog():
    last, seen = time.monotonic(), 0
    while True:
        time.sleep(1)
        if done[0] != seen:
            seen, last = done[0], time.monotonic()
        elif time.monotonic() - last > STALL:
            sys.stderr.write("WATCHDOG: no compute finished for %ds, stacks follow\\n" % STALL)
            faulthandler.dump_traceback(all_threads=True)
            sys.stderr.flush()
            import os; os._exit(97)
threading.Thread(target=watchdog, daemon=True).start()

def table(seed, n=400_000):
    rng = np.random.default_rng(seed)
    return pa.table({"k": pa.array(rng.integers(0, 17, n).astype(np.int32)),
                     "v": pa.array(rng.random(n)), "d": pa.array(rng.integers(0, 100, n).astype(np.int32))})

def work(tid):
    for i in range(ITERS):
        t = table(tid * 100_000 + i)
        want = t.group_by("k").aggregate([("v", "sum")]).sort_by("k")
        lf = LazyFrame(t).filter(col("d") > lit(20)).groupby("k").agg(col("v").sum().alias("s")).sort("k")
        got = lf.compute(device="auto")
        assert got.num_rows == 17 and got.column_names == ["k", "s"], got.schema
        del t, lf, got, want
        if i % 7 == 0:
            gc.collect()
        with progress:
            done[0] += 1

gc.set_threshold(2, 1, 1)               # the cyclic GC fires at nearly every allocation
am.get_gpu_warmup()                      # the warm-up thread runs alongside
errors = []
def run(tid):
    try:
        work(tid)
    except BaseException as e:
        errors.append(repr(e))
ts = [threading.Thread(target=run, args=(i,)) for i in range(THREADS)]
[t.start() for t in ts]; [t.join() for t in ts]
cf.wait_for_uploads(60)
assert not errors, errors
assert uploads[0] > 0, "no upload was queued: the stress did not exercise the residency threads"
print("OK", done[0])
sys.stderr.write("uploads started: %d\\n" % uploads[0])
"""


@pytest.mark.skipif(not pk.gpu_available(), reason="needs a GPU")
def test_concurrent_auto_computes_with_uploads_and_warmup_finish():
    threads, iters = (6, 1000) if LONG else (4, 12)
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD), str(threads), str(iters), "60"],
                       capture_output=True, text=True, cwd=str(HERE), timeout=3600 if LONG else 240)
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr[-6000:]}"
    assert r.stdout.strip() == f"OK {threads * iters}"
