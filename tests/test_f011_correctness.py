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
    "float32_column_keeps_boundary_rows",
    "gpu_scratch_does_not_reserve_the_device",
])
def test_scenario(name, device):
    _run(name, device)


# --- gathering strings ---------------------------------------------------------------------

def test_take_rows_slices_string_columns(monkeypatch):
    import numpy as np
    import pyarrow as pa
    from mxframe import np_arrow
    monkeypatch.setattr(np_arrow, "TAKE_CHUNK_ROWS", 100)
    t = pa.table({"s": pa.array([f"v{i}" for i in range(1000)]), "k": pa.array(np.arange(1000, dtype=np.int64))})
    idx = np.random.default_rng(1).integers(0, 1000, 950)
    got = np_arrow.take_rows(t, idx)
    assert got.equals(t.take(pa.array(idx)))
    assert got.column("s").num_chunks == 10 and got.column("k").num_chunks == 1     # only strings are sliced
    col = np_arrow.take_rows(t.column("s").combine_chunks(), idx)
    assert col.to_pylist() == [f"v{i}" for i in idx]


@pytest.mark.slow
def test_take_rows_survives_more_than_2gib_of_strings():
    """Arrow's take returns a corrupt column (`Negative offsets in binary array`) past 2 GiB of
    string data: a lineitem-to-orders join at SF10 gathers 3 GB of o_comment."""
    import numpy as np
    import pyarrow as pa
    from mxframe import np_arrow
    n = 1_000_000
    t = pa.table({"s": pa.array(["x" * 100 + str(i % 10) for i in range(n)])})       # 100 MB
    idx = np.tile(np.arange(n, dtype=np.int64), 24)                                    # 2.4 GB gathered
    out = np_arrow.take_rows(t, idx)
    out.validate(full=True)
    assert out.num_rows == 24 * n and out.column("s")[24 * n - 1].as_py().endswith("9")
