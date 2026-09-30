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

from mxframe import gpu_stream

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
