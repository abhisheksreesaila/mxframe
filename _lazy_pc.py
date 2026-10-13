"""`pc`: pyarrow.compute without importing pyarrow.compute.

`import pyarrow.compute` costs ~20-25 ms per process: it generates a Python wrapper, docstring and
signature for each of its ~270 functions. Almost all of that is the docstrings. `pc.<name>` builds
the same call for a generated function straight on `pyarrow._compute` (~6 ms to import, on first
use): the function's positional arguments, then its options class from the remaining arguments,
keyword arguments or `options=` (pyarrow.compute's `_make_generic_wrapper`). Of the functions
pyarrow.compute defines in Python, `cast`, `take` and `fill_null` are rebuilt here the same way; the
others, anything that is not a registered function, and every name once pyarrow.compute is loaded
anyway go to the real module.

Arrow's own methods (`Array.cast`, `Table.filter`, `ChunkedArray.dictionary_encode`, ...) import
pyarrow.compute: engine code calls `pc.cast(a, t)`, `pc.filter(t, mask)`, ... instead.
"""

import sys

# defined in Python by pyarrow.compute, with their own argument handling
_PYTHON_DEFINED = frozenset({"index", "top_k_unstable", "bottom_k_unstable", "random", "field", "scalar"})
_REWRITES = {"and_": "and", "or_": "or"}


def _real(name):
    # always through the import system: it waits on the module lock, so a thread never sees a
    # half-imported module (a cached module costs a dict lookup)
    import pyarrow.compute as mod
    return getattr(mod, name)


def _cast(arr, target_type=None, safe=None, options=None, memory_pool=None):
    import pyarrow as pa
    import pyarrow._compute as _c
    if (safe is not None or target_type is not None) and options is not None:
        raise ValueError("Must either pass values for 'target_type' and 'safe' or pass a value for 'options'")
    if options is None:
        target_type = pa.types.lib.ensure_type(target_type)
        options = _c.CastOptions.unsafe(target_type) if safe is False else _c.CastOptions.safe(target_type)
    return _c.call_function("cast", [arr], options, memory_pool)


def _take(data, indices, *, boundscheck=True, memory_pool=None):
    import pyarrow._compute as _c
    return _c.call_function("take", [data, indices], _c.TakeOptions(boundscheck=boundscheck), memory_pool)


def _fill_null(values, fill_value):
    import pyarrow as pa
    import pyarrow._compute as _c
    if not isinstance(fill_value, (pa.Array, pa.ChunkedArray, pa.Scalar)):
        fill_value = pa.scalar(fill_value, type=values.type)
    elif values.type != fill_value.type:
        fill_value = pa.scalar(fill_value.as_py(), type=values.type)
    return _c.call_function("coalesce", [values, fill_value])


_REBUILT = {"cast": _cast, "take": _take, "fill_null": _fill_null}


def _wrapper(name):
    """pyarrow.compute's generated wrapper for `name`, or None when it is not a plain generated one."""
    if name in _PYTHON_DEFINED or name.startswith("_") or "pyarrow.compute" in sys.modules:
        return None
    if name in _REBUILT:
        return _REBUILT[name]
    import pyarrow._compute as _c
    cpp = _REWRITES.get(name, name)
    try:
        func = _c.get_function(cpp)
    except Exception:
        return None
    if func.kind == "hash_aggregate" or (func.kind == "scalar_aggregate" and func.arity == 0):
        return None
    opt_name = func._doc.options_class
    options_class = getattr(_c, opt_name, None) if opt_name else None
    if opt_name and options_class is None:
        return None
    arity = func.arity
    call = func.call

    if options_class is None:
        def wrapper(*args, memory_pool=None):
            if arity is not Ellipsis and len(args) != arity:
                raise TypeError(f"{name} takes {arity} positional argument(s), but {len(args)} were given")
            if args and isinstance(args[0], _c.Expression):
                return _c.Expression._call(cpp, list(args))
            return call(args, None, memory_pool)
    else:
        def wrapper(*args, memory_pool=None, options=None, **kwargs):
            if arity is not Ellipsis:
                if len(args) < arity:
                    raise TypeError(f"{name} takes {arity} positional argument(s), but {len(args)} were given")
                option_args, args = args[arity:], args[:arity]
            else:
                option_args = ()
            if option_args or kwargs:
                if options is not None:
                    raise TypeError(f"Function {name!r} called with both an 'options' argument "
                                    f"and additional arguments")
                options = options_class(*option_args, **kwargs)
            elif isinstance(options, dict):
                options = options_class(**options)
            elif options is not None and not isinstance(options, options_class):
                raise TypeError(f"Function {name!r} expected a {options_class} parameter, got {type(options)}")
            if args and isinstance(args[0], _c.Expression):
                return _c.Expression._call(cpp, list(args), options)
            return call(args, options, memory_pool)
    wrapper.__name__ = name
    return wrapper


class _LazyCompute:
    __slots__ = ("_cache",)

    def __init__(self):
        self._cache = {}

    def __getattr__(self, name):
        if name == "_cache":
            raise AttributeError(name)
        f = self._cache.get(name)
        if f is None:
            f = _wrapper(name)
            if f is None:
                return _real(name)
            self._cache[name] = f
        return f

    def __dir__(self):
        import pyarrow.compute as mod
        return dir(mod)


pc = _LazyCompute()
