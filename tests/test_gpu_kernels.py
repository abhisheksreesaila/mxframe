"""F-012: GPU kernel health (crashes, hangs, wrong results).

Each scenario runs in a subprocess (tests/_gpu_scenarios.py): a faulting kernel
poisons the process and a hanging one never returns, so the timeout is the
assertion for "terminates".
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omxcore import gpu_stream

pytestmark = pytest.mark.skipif(not gpu_stream.cuda_available(), reason="no CUDA GPU")

SCRIPT = Path(__file__).with_name("_gpu_scenarios.py")


def run_scenario(*args, timeout=300):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               PYTHONPATH=os.pathsep.join([sys.path[0], os.environ.get("PYTHONPATH", "")]))
    try:
        r = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"scenario {args} did not terminate within {timeout}s (hang)")
    if r.returncode == 77:                  # _gpu_scenarios.SKIP
        pytest.skip(r.stdout.strip())
    assert r.returncode == 0, f"scenario {args} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"


# 8192 was the shared-memory limit of the old kernel, 10_000 faulted; 12_289 is the first
# count past the new shared-memory tier; 1_500_000 exercises the global-atomic path at scale.
@pytest.mark.parametrize("ng", [1, 8, 100, 1000, 4096, 8192, 10_000, 12_288, 12_289, 200_000, 1_500_000])
def test_group_aggregates_match_numpy_at_group_counts(ng):
    run_scenario("group_aggs", ng)


def test_group_aggregate_edge_cases():
    run_scenario("group_edge_cases")


def test_group_encode_terminates_and_is_correct():
    run_scenario("group_encode", timeout=120)


@pytest.mark.parametrize("n", [1, 3, 1000, 1025, 100_000, 1_000_003])
def test_sort_is_stable_and_correct_for_any_length(n):
    run_scenario("sort_topk", n)


def test_sort_handles_all_lengths_ties_and_extreme_keys():
    run_scenario("sort_topk")


def test_hash_join_matches_reference():
    run_scenario("hash_join")


def test_device_resident_inputs_and_outputs_chain_without_host_round_trips():
    run_scenario("device_chaining")


def test_multi_key_sort_order_on_gpu_and_cpu():
    run_scenario("sort_multikey")


def test_late_supplier_hash_table_terminates_and_matches_numpy():
    run_scenario("late_suppliers", timeout=120)


def test_sort_end_to_end_single_and_multi_key_with_and_without_limit():
    run_scenario("sort_end_to_end", timeout=600)


@pytest.mark.parametrize("kind", ["0.01", "0.05", "0.25", "zipf"])
def test_skewed_high_cardinality_group_by_is_not_super_linear(kind):
    run_scenario("group_skew", kind, timeout=120)


def test_wrappers_work_from_a_thread_that_never_touched_cuda():
    run_scenario("other_thread")


def test_failed_hash_join_frees_its_scratch_buffers():
    run_scenario("join_failure_frees_scratch")


# F-013: masked global aggregates are block-reduced (were 100x too slow with per-block float atomics).
# Sizes straddle the block (256), the unrolled step and the grid-stride; 0 and 1 are the edges.
@pytest.mark.parametrize("n", [0, 1, 255, 256, 257, 4099, 1_000_003, 6_000_001])
def test_masked_global_aggregates_match_numpy(n):
    run_scenario("masked_aggs", n, timeout=120)


def test_gpu_masked_outputs_are_float64_and_agree_with_cpu_at_1e9():
    run_scenario("masked_f64_matches_cpu", timeout=120)
