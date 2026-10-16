"""F-063: a process that exits while the background GPU warm-up is inside CUDA must not abort.

Seen: TPC-H Q17 on device="auto" exited -6 (glibc `tpp.c` / `pthread_mutex_lock` assertion) or
-11 (SIGSEGV inside `cuDevicePrimaryCtxRetain`) about 1 run in 75.  The core dump showed the
main thread in libc `exit()` (static destructors: Arrow's thread pool, libcuda's teardown)
while the daemon warm-up thread was still in a ctypes call into libcuda.  A daemon thread
inside native code is not stopped by interpreter shutdown, so `exit()` tore the driver down
under it.  Fix: an atexit hook waits for in-flight background GPU work and starts no more.
"""
import os
import random
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _run(code, *args):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code), *args], capture_output=True,
                          text=True, cwd=str(HERE), timeout=120)


PREAMBLE = """
import sys
sys.path.insert(0, ".")
import conftest  # noqa: F401  (makes `import mxframe` resolve to this checkout)
from mxframe import auto_mode
"""


def test_exit_waits_for_an_in_flight_gpu_warmup(tmp_path):
    marker = tmp_path / "init_finished"
    code = PREAMBLE + """
import time
def slow_init():
    time.sleep(0.6)               # stands in for CUDA initialisation
    open(sys.argv[1], "w").write("finished")
    return object()
auto_mode._init_gpu = slow_init
auto_mode.get_gpu_warmup().start()
time.sleep(0.1)                   # the interpreter now exits with the init thread mid-call
"""
    r = _run(code, str(marker))
    assert r.returncode == 0, r.stderr
    assert marker.read_text() == "finished"


def test_exit_does_not_start_a_warmup_that_has_not_begun(tmp_path):
    marker = tmp_path / "init_started"
    code = PREAMBLE + """
import time
def init():
    open(sys.argv[1], "w").write("started")
    return object()
w = auto_mode.GpuWarmup(init, 10)
auto_mode._close_background_gpu_work()      # what the atexit hook runs
w.start()                                   # a late launcher thread: refused once exiting
time.sleep(0.2)
"""
    r = _run(code, str(marker))
    assert r.returncode == 0, r.stderr
    assert not marker.exists()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("MXFRAME_STRESS") != "1", reason="set MXFRAME_STRESS=1 (needs a GPU, ~1 min)")
def test_real_gpu_warmup_survives_exit_at_any_moment():
    """Before the fix 2 of 150 runs aborted (-6 / -11) with the exit landing 0.28-0.42 s into CUDA init."""
    code = PREAMBLE + """
import time
auto_mode.get_gpu_warmup().start()
time.sleep(float(sys.argv[1]))
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

    def one(_):
        d = random.uniform(0.28, 0.42)
        p = subprocess.run([sys.executable, "-c", textwrap.dedent(code), f"{d:.4f}"], capture_output=True,
                           text=True, cwd=str(HERE), env=env, timeout=120)
        return d, p.returncode, p.stderr[-300:]
    with ThreadPoolExecutor(3) as ex:
        bad = [r for r in ex.map(one, range(300)) if r[1] != 0]
    assert not bad, bad
