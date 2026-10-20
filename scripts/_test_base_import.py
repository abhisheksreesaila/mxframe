"""Verify the base package imports without the optional Modular MAX runtime."""

import builtins


real_import = builtins.__import__


def blocked_import(name, *args, **kwargs):
    if name == "max" or name.startswith("max."):
        raise ImportError("MAX intentionally unavailable")
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

import omxcore

assert omxcore.col("x").op == "col"
# Since F-019 MAX is imported lazily (omxcore/_max_lazy.py): the runtime exports load without
# it and only need MAX when a MAX graph is actually built.
import sys
omxcore.GraphCompiler
assert not any(m == "max" or m.startswith("max.") for m in sys.modules), "MAX imported eagerly"

print("OK base package import without optional MAX runtime")
