"""Verify the base package imports without the optional Modular MAX runtime."""

import builtins


real_import = builtins.__import__


def blocked_import(name, *args, **kwargs):
    if name == "max" or name.startswith("max."):
        raise ImportError("MAX intentionally unavailable")
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

import mxframe

assert mxframe.col("x").op == "col"
try:
    mxframe.GraphCompiler
except ImportError as exc:
    assert "mxframe[runtime]" in str(exc)
else:
    raise AssertionError("runtime export unexpectedly loaded")

print("OK base package import without optional MAX runtime")
