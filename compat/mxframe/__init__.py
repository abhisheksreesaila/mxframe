"""`mxframe` is the old import name of omxcore (F-165). `import mxframe` gives the very same module object as
`import omxcore`, and `mxframe.<submodule>` the same module as `omxcore.<submodule>` (`from mxframe.lazy_frame import X`,
`import mxframe.native_exec as nx`), so one process never holds two copies of the engine. No warning for now; new code
writes `import omxcore as mx`. This file is the whole package: delete compat/ and the `mxframe` line in pyproject.toml
when the forwarding ends.
"""
import importlib
import importlib.abc
import importlib.util
import sys

import omxcore as _omxcore


class _AliasLoader(importlib.abc.Loader):
    """Loads `mxframe.x` as the module `omxcore.x`."""

    def __init__(self, real_name):
        self.real_name = real_name
        self.real_spec = None

    def create_module(self, spec):
        real = importlib.import_module(self.real_name)
        self.real_spec = real.__spec__        # the import system stamps the alias spec on the module: put the real one back
        return real

    def exec_module(self, module):
        module.__spec__ = self.real_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith("mxframe."):
            return importlib.util.spec_from_loader(name, _AliasLoader("omxcore." + name[len("mxframe."):]))
        return None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
sys.modules[__name__] = _omxcore
