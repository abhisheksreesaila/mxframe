"""Shared pytest setup: make `import mxframe` resolve to this checkout (works in git worktrees)."""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_shim = Path(tempfile.gettempdir()) / f"mxframe-shim-{hashlib.sha1(str(_ROOT).encode()).hexdigest()[:10]}"
_shim.mkdir(exist_ok=True)
_link = _shim / "mxframe"
if not _link.exists():
    _link.symlink_to(_ROOT)
sys.path.insert(0, str(_shim))

TPCH_DATA = Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs TPC-H Parquet data on disk / runs long")
