"""Shared pytest setup: make `import omxcore` resolve to this checkout (works in git worktrees)."""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_shim = Path(tempfile.gettempdir()) / f"omxcore-shim-{hashlib.sha1(str(_ROOT).encode()).hexdigest()[:10]}"
_shim.mkdir(exist_ok=True)
for _name, _target in (("omxcore", _ROOT), ("mxframe", _ROOT / "compat" / "mxframe")):   # mxframe: the forwarding module
    if not (_shim / _name).exists():
        (_shim / _name).symlink_to(_target)
sys.path.insert(0, str(_shim))

TPCH_DATA = Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs TPC-H Parquet data on disk / runs long")

# Tests must never write observations into the user's real calibration file.
os.environ.setdefault("MXFRAME_CALIBRATION", str(Path(tempfile.gettempdir()) / f"mxframe-test-calibration-{os.getpid()}.json"))


import pytest  # noqa: E402


def _mx_env():
    return {k: v for k, v in os.environ.items() if k.startswith("MXFRAME_")}


@pytest.fixture(autouse=True)
def _no_env_leaks():
    """A test that leaves an MXFRAME_* variable changed fails here, and the variable is put back. Leaks broke later
    tests on the same xdist worker twice (`MXFRAME_HOST_MEM_BUDGET`: small in-memory queries streamed and failed
    with a 0-byte budget). Use monkeypatch.setenv, or restore in a finally. Declared before the other autouse
    fixtures, so it checks after their monkeypatch has undone its changes."""
    before = _mx_env()
    yield
    after = _mx_env()
    if after != before:
        for k in after.keys() - before.keys():
            del os.environ[k]
        os.environ.update(before)
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        pytest.fail(f"the test left environment variables changed: {changed} "
                    f"(was {[before.get(k) for k in changed]}, now {[after.get(k) for k in changed]})")


@pytest.fixture(autouse=True)
def _strict_native(monkeypatch):
    """An unexpected exception in a native hook is raised in the tests, not turned into a quiet fallback
    (a test that wants the fallback unsets it)."""
    monkeypatch.setenv("MXFRAME_STRICT_NATIVE", "1")
