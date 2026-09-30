"""F-020: the CPU kernels of all four libraries hold up when several processes each call them from several
Python threads at once (rare crashes were seen under heavy load; see scripts/stress_cpu_kernels.py).

Slow: ``pytest -m slow tests/test_stress_kernels.py``.  Longer, heavier runs:
``pixi run python scripts/stress_cpu_kernels.py --procs 8 --threads 4 --seconds 240 --warm``.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "stress_cpu_kernels.py"


def _run(*args, timeout=600):
    return subprocess.run([sys.executable, "-X", "faulthandler", str(SCRIPT), *args], capture_output=True,
                          text=True, timeout=timeout, env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "1")})


def test_single_process_quick_check():
    """Not slow: one process, two threads, a few seconds: the harness itself works and the kernels agree with NumPy."""
    r = _run("--child", "--threads", "2", "--seconds", "4", timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert "CHILD OK" in r.stdout


@pytest.mark.slow
def test_concurrent_processes_and_threads_do_not_crash():
    r = _run("--procs", "4", "--threads", "4", "--seconds", "15", "--warm")
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert "all clean" in r.stdout


@pytest.mark.slow
def test_first_calls_racing_the_warmup_thread_in_fresh_processes():
    r = _run("--procs", "6", "--threads", "4", "--seconds", "2", "--rounds", "4", "--warm")
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert "all clean" in r.stdout
