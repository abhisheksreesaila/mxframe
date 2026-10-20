"""F-018: the semi-join prefilter kernel (kernels_aot/libmxjoin_cpu.so) and its use in the join step."""
import numpy as np
import pytest

from omxcore import join_engine as je
from omxcore.custom_ops import CustomOpsCompiler

pytestmark = pytest.mark.skipif(je._semi_lib() is None, reason="libmxjoin_cpu.so not built")


def _pairs(li, ri):
    return sorted(zip(li.tolist(), ri.tolist()))


@pytest.mark.parametrize("n_probe,n_build,T", [(1, 1, 1), (10, 3, 50), (300_000, 1_000, 5_000_000),
                                              (200_000, 150_000, 400_000), (70_000, 5, 70_000)])
def test_semi_select_matches_isin(n_probe, n_build, T):
    rng = np.random.default_rng(n_probe + n_build)
    probe = rng.integers(0, T, n_probe).astype(np.int32)
    build = rng.integers(0, T, n_build).astype(np.int32)
    got = je.semi_select(probe, build, T - 1)
    want = np.flatnonzero(np.isin(probe, build)).astype(np.int32)
    assert got.dtype == np.int32
    assert (got == want).all() and len(got) == len(want)


def test_semi_select_ignores_keys_outside_the_range_and_handles_empty():
    probe = np.array([0, 5, 9, 12, -1, 3], dtype=np.int32)
    build = np.array([5, 9, 100, -7], dtype=np.int32)
    assert je.semi_select(probe, build, 10).tolist() == [1, 2]
    assert len(je.semi_select(np.empty(0, np.int32), build, 10)) == 0
    assert len(je.semi_select(probe, np.empty(0, np.int32), 10)) == 0


def test_prefiltered_join_gives_the_same_pairs_as_the_plain_join(monkeypatch):
    monkeypatch.setattr(je, "SEMI_MIN_PROBE", 1000)
    monkeypatch.setattr(je, "SEMI_MIN_RANGE", 1000)
    rng = np.random.default_rng(3)
    T = 2_000_000
    probe = rng.integers(0, T, 400_000).astype(np.int32)
    build = np.concatenate([rng.choice(T, 20_000, replace=False), probe[:500]]).astype(np.int32)   # some duplicates
    comp = CustomOpsCompiler(device="cpu")
    plain = je.inner_pairs(comp, probe, build)
    pre = je._pairs_prefiltered(comp, probe, build)
    assert len(pre[0]) == len(plain[0]) > 0
    assert _pairs(*pre) == _pairs(*plain)


def test_join_step_survives_a_missing_prefilter_library(monkeypatch):
    monkeypatch.setitem(je._semi, "lib", None)
    monkeypatch.setitem(je._semi, "tried", True)
    monkeypatch.setattr(je, "SEMI_MIN_PROBE", 10)
    monkeypatch.setattr(je, "SEMI_MIN_RANGE", 10)
    comp = CustomOpsCompiler(device="cpu")
    probe = np.arange(2000, dtype=np.int32)
    build = np.array([5, 70, 1999], dtype=np.int32)
    li, ri = je._pairs_prefiltered(comp, probe, build)
    assert _pairs(li, ri) == [(5, 0), (70, 1), (1999, 2)]
