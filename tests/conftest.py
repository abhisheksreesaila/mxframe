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

# The repo root has an __init__.py (it *is* the package), so pytest builds a Package node for it and
# imports that __init__ at setup time.  In a git worktree the directory name (e.g. "agent-a55...")
# is not a valid module name and that import fails.  Skip the root Package import there; tests
# import `mxframe` through the shim above.
import _pytest.python as _pt_python  # noqa: E402

_orig_package_setup = _pt_python.Package.setup


def _package_setup(self):
    if self.path.name.isidentifier():
        return _orig_package_setup(self)


_pt_python.Package.setup = _package_setup

TPCH_DATA =Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data"))
