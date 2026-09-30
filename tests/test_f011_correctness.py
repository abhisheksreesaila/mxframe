"""F-011 engine correctness: counts, nulls, limits, caches (see tests/_f011_scenarios.py).

``cpu`` scenarios run in-process; ``gpu`` and ``auto`` in a subprocess (a fault must not
take the session down).
"""
import subprocess
import sys
from pathlib import Path

import pytest

import _f011_scenarios as sc
from mxframe.aot_kernels import GPU_AOT_AVAILABLE

SCRIPT = Path(__file__).with_name("_f011_scenarios.py")


def _run(name, device):
    if device == "cpu":
        sc.SCENARIOS[name](device)
        return
    if not GPU_AOT_AVAILABLE:
        pytest.skip("no GPU AOT kernels")
    r = subprocess.run([sys.executable, str(SCRIPT), name, device], capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("name", [
    "count_exact_beyond_2p24",
    "count_exact_beyond_2p24_unfused",
    "count_types_all_paths",
])
def test_scenario(name, device):
    _run(name, device)
