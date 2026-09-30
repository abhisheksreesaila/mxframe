"""Lazy handles for the MAX Engine / Graph API (F-015).

``import max.engine`` + ``max.graph`` costs ~150 ms and the CPU paths (fused pipeline,
AOT kernels, PyArrow) never use them. ``custom_ops.py`` and ``compiler.py`` bind
``engine``, ``driver``, ``Graph``, ``TensorType``, ``ops``, ``DType`` and ``DeviceRef`` to
these proxies instead of importing them at module top; the real import happens on the first
attribute access or call, i.e. only when a MAX Graph fallback or the GPU session is reached.
"""
from __future__ import annotations

import importlib

__all__ = ["LazyRef", "engine", "driver", "Graph", "TensorType", "ops", "DType", "DeviceRef"]


class LazyRef:
    """Stands for ``module`` or ``module.attr`` and resolves on first use."""

    __slots__ = ("_module", "_attr", "_obj")

    def __init__(self, module: str, attr: str | None = None):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_attr", attr)
        object.__setattr__(self, "_obj", None)

    def _resolve(self):
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            obj = importlib.import_module(object.__getattribute__(self, "_module"))
            attr = object.__getattribute__(self, "_attr")
            if attr is not None:
                obj = getattr(obj, attr)
            object.__setattr__(self, "_obj", obj)
        return obj

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __repr__(self):
        return f"<lazy {object.__getattribute__(self, '_module')}" + \
            (f".{object.__getattribute__(self, '_attr')}" if object.__getattribute__(self, "_attr") else "") + ">"


engine = LazyRef("max.engine")
driver = LazyRef("max.driver")
ops = LazyRef("max.graph.ops")
Graph = LazyRef("max.graph", "Graph")
TensorType = LazyRef("max.graph", "TensorType")
DType = LazyRef("max.graph.type", "DType")
DeviceRef = LazyRef("max.graph.type", "DeviceRef")
