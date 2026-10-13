"""`pc`: pyarrow.compute, imported on first use.

`import pyarrow.compute` costs ~20-25 ms per process (it generates a Python wrapper for each of
its ~270 functions). The engine modules import it at module level, so every query paid it, even
when the plan runs natively and never calls a compute function. This proxy defers the import to
the first attribute access.
"""



class _LazyCompute:
    __slots__ = ()

    def __getattr__(self, name):
        # always through the import system: it waits on the module lock, so a thread never sees a
        # half-imported pyarrow.compute (a cached module costs a dict lookup)
        import pyarrow.compute as mod
        return getattr(mod, name)

    def __dir__(self):
        import pyarrow.compute as mod
        return dir(mod)


pc = _LazyCompute()
