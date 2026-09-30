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
    "nulls_filter_drops_null_comparisons",
    "nulls_group_key_is_its_own_group",
    "nulls_global_aggregates_skip_nulls",
    "string_ne_with_null_keys",
    "sort_with_null_keys",
    "limit_below_aggregate_is_applied",
    "multi_key_sort_order",
    "join_keys_of_different_integer_widths",
    "join_cache_is_keyed_by_device",
    "case_when_sums_are_float64",
    "min_max_identity_with_infinities",
])
def test_scenario(name, device):
    _run(name, device)
