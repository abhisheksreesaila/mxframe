"""`pc`: pyarrow.compute, imported on first use.

`import pyarrow.compute` costs ~20-25 ms per process (it generates a Python wrapper for each of
its ~270 functions). The engine modules import it at module level, so every query paid it, even
when the plan runs natively and never calls a compute function. This proxy defers the import to
the first attribute access.
"""

import sys


class _LazyCompute:
    __slots__ = ()

    def __getattr__(self, name):
        mod = sys.modules.get("pyarrow.compute")
        if mod is None:
            import pyarrow.compute as mod
        return getattr(mod, name)

    def __dir__(self):
        import pyarrow.compute as mod
        return dir(mod)


pc = _LazyCompute()
