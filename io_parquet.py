"""Lazy Parquet source: ``scan_parquet`` and the source-resolution pass.

``scan_parquet(path)`` reads only the file footer.  When the query is computed,
``resolve_plan`` walks the logical plan, works out which columns the query
touches, and reads just those (with simple comparisons above the scan pushed
into the reader so whole row groups can be skipped).

Low-cardinality string columns are read as Arrow dictionaries.  The engine uses
those natively as group keys and in string filters; in every other position they
are decoded back to plain values before execution, so results match the
``pq.read_table`` path exactly.

``pyarrow.parquet`` is imported lazily to keep ``import mxframe`` fast.
"""
from __future__ import annotations

import dataclasses
import functools
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa

from .lazy_expr import Expr
from .lazy_frame import (
    Aggregate, Distinct, Filter, Join, LazyFrame, Limit, LogicalPlan,
    ParquetScan, Project, Scan, Sort, Tail,
)

def _decode_dict(col):
    from .arrow_dict import decode       # pyarrow.compute (~25 ms) loads on first use, not at import
    return decode(col)


def _is_dict(arr_or_type) -> bool:
    return pa.types.is_dictionary(getattr(arr_or_type, "type", arr_or_type))


__all__ = ["scan_parquet", "read_parquet_table", "dictionary_columns", "resolve_plan"]

DEFAULT_MAX_DICTIONARY_BYTES = 64 * 1024


# ── public API ──────────────────────────────────────────────────────────────

def scan_parquet(path, *, dictionary: Any = "auto",
                 max_dictionary_bytes: int = DEFAULT_MAX_DICTIONARY_BYTES) -> LazyFrame:
    """Lazily scan a Parquet file (or directory) as a ``LazyFrame``.

    Nothing but the footer is read here.  At ``compute()`` only the columns the
    query references are read, and simple ``col <op> literal`` / ``isin``
    filters directly above the scan are pushed into the reader.

    Args:
        path: Parquet file or dataset directory.
        dictionary: how string columns are read.  ``"auto"`` (default) reads
            columns that are dictionary-encoded in the file with a small
            dictionary (<= ``max_dictionary_bytes`` per row group) as Arrow
            dictionaries; ``True`` does so for every string column; ``False``
            never; a list of names does so for those columns.
        max_dictionary_bytes: dictionary-page size limit for ``"auto"``.
    """
    import pyarrow.parquet as pq
    path = os.fspath(path)
    if os.path.isfile(path):
        # footer parsed once per process: the same metadata serves the read later
        schema = pq.ParquetFile(path, metadata=_metadata(path)).schema_arrow
    else:
        schema = pq.read_schema(path)
    return LazyFrame(ParquetScan(path, schema, dictionary, int(max_dictionary_bytes)))


def read_parquet_table(path, columns: Optional[Sequence[str]] = None, filters=None,
                       dictionary: Any = "auto",
                       max_dictionary_bytes: int = DEFAULT_MAX_DICTIONARY_BYTES) -> pa.Table:
    """Eagerly read a Parquet file into a ``pa.Table`` (dictionary-aware).

    ``filters`` is a ``pyarrow.dataset`` expression or ``None``.  Strings selected
    by ``dictionary`` (see :func:`scan_parquet`) come back as ``pa.dictionary``
    columns, which the engine accepts anywhere a string column is accepted.
    """
    import pyarrow.parquet as pq
    path = os.fspath(path)
    schema = pq.read_schema(path)
    names = list(columns) if columns is not None else list(schema.names)
    dict_cols = dictionary_columns(path, schema, dictionary, max_dictionary_bytes, names)
    if filters is None and os.path.isfile(path):
        return _read_file(path, names, dict_cols)
    return pq.read_table(path, columns=names, filters=filters,
                         read_dictionary=dict_cols or None)


def _read_file(path, columns, dict_cols, pushed=()) -> pa.Table:
    """Read one Parquet file without ``pyarrow.dataset``.

    ``pq.read_table`` goes through ``pyarrow.dataset``, whose import pulls in
    pandas (~150 ms of a first query). ``ParquetFile`` reads the same data
    without it. ``pushed`` simple filters skip row groups whose statistics rule
    them out and, for integer comparisons, drop non-matching rows right after
    each row group is decoded (see ``_read_groups``); rows are still filtered
    exactly by the plan's Filter node.
    """
    import pyarrow.parquet as pq
    md = _metadata(path)
    pf = pq.ParquetFile(path, read_dictionary=dict_cols or None, metadata=md)
    groups = _row_groups_for(pf, pushed) if pushed else None
    if groups is not None and not groups:
        return pf.schema_arrow.empty_table().select(columns) if columns else pf.schema_arrow.empty_table()
    if groups is None:
        groups = list(range(md.num_row_groups))
    rowwise = [p for p in pushed if _row_filterable(p, pf.schema_arrow)]
    if len(groups) >= 2 and md.num_columns == len(pf.schema_arrow.names) and (rowwise or len(groups) > 2):
        return _read_groups(path, md, pf.schema_arrow, columns, dict_cols, groups, rowwise)
    if len(groups) == md.num_row_groups:
        return pf.read(columns=columns, use_threads=True)
    return pf.read_row_groups(groups, columns=columns, use_threads=True)


_ROW_OPS = {"eq": "__eq__", "ne": "__ne__", "lt": "__lt__", "le": "__le__", "gt": "__gt__", "ge": "__ge__"}


def _row_filterable(pushed_one, schema) -> bool:
    name, op, _ = pushed_one
    return _kind(schema, name) == "int" and (op in _ROW_OPS or op == "isin")


def _row_mask(table, rowwise):
    """Boolean NumPy mask of the rows satisfying every integer conjunct, or None (keep all)."""
    from .np_arrow import to_numpy
    import numpy as np
    mask = None
    for name, op, value in rowwise:
        c = table.column(name)
        if c.null_count:
            continue                       # nulls: leave the row-level decision to the Filter node
        a = to_numpy(c)
        try:
            if op == "isin":
                m = np.isin(a, np.asarray(value, dtype=np.int64))
            else:
                m = getattr(a, _ROW_OPS[op])(value)
        except (OverflowError, TypeError):
            continue                       # literal outside the column's range: the Filter node decides
        mask = m if mask is None else (mask & m)
    return mask


def _read_groups(path, md, schema, columns, dict_cols, groups, rowwise) -> pa.Table:
    """Decode row groups on a thread pool, filtering each right after it is decoded.

    Each worker reads one row group single-threaded (its own ``ParquetFile`` over
    the shared cached footer) and applies the pushed integer comparisons while the
    columns are still cache-hot, so only the surviving rows are kept and later
    concatenated.  This is what makes ``scan -> filter -> select`` on a big file
    cost about as much as decoding the columns once.
    """
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    cols = list(columns) if columns else list(schema.names)
    fnames = [n for n, _, _ in rowwise]
    read_cols = cols + [n for n in dict.fromkeys(fnames) if n not in cols]
    rd = dict_cols or None
    src = pa.memory_map(path)              # one mapping shared by all readers: no per-group open/copy

    def one(g):
        pf = pq.ParquetFile(src, read_dictionary=rd, metadata=md)
        t = pf.read_row_group(g, columns=read_cols, use_threads=False)
        if rowwise:
            m = _row_mask(t, rowwise)
            if m is not None:
                t = _take_rows(t, m)
        return t.select(cols) if len(read_cols) != len(cols) else t

    workers = max(1, min(len(groups), _read_threads()))
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(one, groups))
    kept = [t for t in parts if t.num_rows] or parts[:1]
    return pa.concat_tables(kept)


def _read_threads() -> int:
    """Decode threads.  More than ~8 slow Arrow's decoder down (allocator and page-fault contention)."""
    n = int(os.environ.get("MXFRAME_READ_THREADS", 0))
    return n if n > 0 else min(8, max(2, (os.cpu_count() or 2) // 2))


def _take_rows(table: pa.Table, mask) -> pa.Table:
    """``table`` rows where the NumPy boolean ``mask`` is set.

    Plain numeric columns go through NumPy (no ``pyarrow.compute`` import, ~25 ms
    of a first query); anything else (strings, dictionaries, nulls) uses Arrow.
    """
    import numpy as np
    from .np_arrow import array as _arr, to_numpy
    if mask.all():
        return table
    cols, fallback = [], None
    for c in table.columns:
        t = c.type
        if (pa.types.is_integer(t) or pa.types.is_floating(t)) and not c.null_count:
            cols.append(_arr(to_numpy(c)[mask], type=t))
        else:
            fallback = True
            break
    if fallback:
        return table.filter(_arr(mask))
    return pa.table(cols, names=table.column_names)


def _row_groups_for(pf, pushed) -> List[int]:
    """Row groups whose min/max statistics may satisfy every pushed conjunct."""
    md = pf.metadata
    names = pf.schema_arrow.names
    if md.num_columns != len(names):
        return list(range(md.num_row_groups))
    keep = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        ok = True
        for name, op, value in pushed:
            j = names.index(name) if name in names else -1
            st = rg.column(j).statistics if j >= 0 else None
            if st is None or not st.has_min_max:
                continue
            lo, hi = st.min, st.max
            if isinstance(lo, bytes):
                lo, hi = lo.decode("utf-8", "replace"), hi.decode("utf-8", "replace")
            try:
                if op == "eq":
                    ok = lo <= value <= hi
                elif op == "ne":
                    ok = not (lo == hi == value)
                elif op == "lt":
                    ok = lo < value
                elif op == "le":
                    ok = lo <= value
                elif op == "gt":
                    ok = hi > value
                elif op == "ge":
                    ok = hi >= value
                elif op == "isin":
                    ok = any(lo <= v <= hi for v in value)
            except TypeError:
                ok = True
            if not ok:
                break
        if ok:
            keep.append(g)
    return keep


def dictionary_columns(path, schema: pa.Schema, dictionary: Any = "auto",
                       max_dictionary_bytes: int = DEFAULT_MAX_DICTIONARY_BYTES,
                       among: Optional[Iterable[str]] = None) -> List[str]:
    """Names of string columns to read as Arrow dictionaries (see ``scan_parquet``)."""
    allowed = set(among) if among is not None else set(schema.names)
    strings = [f.name for f in schema
               if f.name in allowed and (pa.types.is_string(f.type) or pa.types.is_large_string(f.type))]
    if dictionary is False or dictionary is None:
        return []
    if dictionary is True:
        return strings
    if dictionary != "auto":
        wanted = [dictionary] if isinstance(dictionary, str) else list(dictionary)
        return [n for n in strings if n in set(wanted)]
    return [n for n in strings if _chunks_are_small_dictionaries(path, schema, n, max_dictionary_bytes)]


def _chunks_are_small_dictionaries(path, schema, name, max_bytes) -> bool:
    """True if every row group stores ``name`` dictionary-encoded with a small dictionary."""
    import pyarrow.parquet as pq
    if not os.path.isfile(path):
        return False
    md = _metadata(path)
    if md.num_columns != len(schema.names):      # nested columns: leaf index != field index
        return False
    j = schema.get_field_index(name)
    if md.num_row_groups == 0:
        return False
    for g in range(md.num_row_groups):
        c = md.row_group(g).column(j)
        if not c.has_dictionary_page:
            return False
        if not any("DICTIONARY" in e for e in c.encodings):
            return False
        d_off = c.dictionary_page_offset
        if d_off is None or d_off <= 0 or c.data_page_offset - d_off > max_bytes:
            return False
    return True


@functools.lru_cache(maxsize=32)
def _metadata_cached(path, mtime_ns, size):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata


def _metadata(path):
    st = os.stat(path)
    return _metadata_cached(os.fspath(path), st.st_mtime_ns, st.st_size)


# ── plan analysis ───────────────────────────────────────────────────────────

_CMP_OPS = ("eq", "ne", "gt", "ge", "lt", "le")
_FLIP = {"eq": "eq", "ne": "ne", "gt": "lt", "lt": "gt", "ge": "le", "le": "ge"}
# Predicate forms the engine evaluates natively on dictionary columns.
_NATIVE_DICT_PRED = ("eq", "ne", "gt", "ge", "lt", "le", "isin", "startswith", "contains")


class _Info:
    """What a plan references, gathered in one walk."""

    def __init__(self):
        self.refs: set = set()          # every column name referenced anywhere
        self.bad: set = set()           # referenced in a way that needs plain values
        self.has_join = False
        self.passthrough: dict = {}     # id(leaf) -> scan columns flow to the plan output

    def use_plain(self, expr: Expr):
        for name in _expr_cols(expr):
            self.refs.add(name)
            self.bad.add(name)

    def predicate(self, expr: Expr):
        """A Filter predicate: `col <cmp> literal`, isin, startswith, contains may see dictionaries."""
        op, args = expr.op, expr.args
        if op in ("and", "or", "__and__", "__or__"):
            for a in args:
                self.predicate(a)
            return
        if op == "not":
            self.predicate(args[0])
            return
        if (op in _NATIVE_DICT_PRED and isinstance(args[0], Expr) and args[0].op == "col"
                and (op not in _CMP_OPS or (isinstance(args[1], Expr) and args[1].op == "lit"))):
            self.refs.add(args[0].args[0])
            return
        self.use_plain(expr)


def _expr_cols(expr) -> set:
    out: set = set()

    def visit(e):
        if not isinstance(e, Expr):
            return
        if e.op == "col":
            out.add(e.args[0])
            return
        if e.op == "window":
            spec = e.args[1]
            out.update(spec[0])
            out.update(spec[1])
        for a in e.args:
            visit(a)

    visit(expr)
    return out


def _analyze(plan: LogicalPlan) -> _Info:
    """Walk the plan once.

    ``walk`` returns whether the subtree contains a Project/Aggregate.  Expressions
    of a node above such a cut refer to derived (already plain) columns, not to
    the scan's, so they only count as references and never force decoding.
    """
    info = _Info()

    def walk(node, passthru: bool) -> bool:
        if isinstance(node, (Scan, ParquetScan)):
            info.passthrough[id(node)] = info.passthrough.get(id(node), False) or passthru
            return False
        if isinstance(node, Filter):
            cut = walk(node.input, passthru)
            if cut:
                info.refs.update(_expr_cols(node.predicate))
            else:
                info.predicate(node.predicate)
            return cut
        if isinstance(node, Project):
            walk(node.input, False)
            for e in node.exprs:
                info.use_plain(e)
            return True
        if isinstance(node, Aggregate):
            walk(node.input, False)
            for e in node.group_by:
                if e.op == "col":
                    info.refs.add(e.args[0])          # bare group key: dictionary-native
                else:
                    info.use_plain(e)
            for e in node.aggs:
                info.use_plain(e)
            return True
        if isinstance(node, Sort):
            cut = walk(node.input, passthru)
            for e in node.by:
                if cut:
                    info.refs.update(_expr_cols(e))
                else:
                    info.use_plain(e)
            return cut
        if isinstance(node, (Limit, Tail)):
            return walk(node.input, passthru)
        if isinstance(node, Distinct):
            cut = walk(node.input, passthru or not node.subset)
            if node.subset:
                info.refs.update(node.subset)
                if not cut:
                    info.bad.update(node.subset)
            return cut
        if isinstance(node, Join):
            info.has_join = True
            info.refs.update(node.left_on)
            info.refs.update(node.right_on)
            info.bad.update(node.left_on)
            info.bad.update(node.right_on)
            lcut = walk(node.left, passthru)
            rcut = walk(node.right, passthru)
            return lcut or rcut
        info.has_join = True                       # unknown node: be conservative
        if hasattr(node, "input"):
            return walk(node.input, True)
        return False

    walk(plan, True)
    return info


def _leaves(plan: LogicalPlan):
    if isinstance(plan, (Scan, ParquetScan)):
        yield plan
    elif isinstance(plan, Join):
        yield from _leaves(plan.left)
        yield from _leaves(plan.right)
    elif hasattr(plan, "input"):
        yield from _leaves(plan.input)


# ── predicate pushdown ──────────────────────────────────────────────────────

def _conjuncts(expr: Expr):
    if expr.op in ("and", "__and__"):
        for a in expr.args:
            yield from _conjuncts(a)
    else:
        yield expr


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _pushable(expr: Expr, schema: pa.Schema) -> Optional[Tuple]:
    """(column, op, value) for `col <cmp> literal` / `col.isin([...])`, else None.

    Only integer and string columns with a literal of the same kind are pushed:
    the reader then compares exactly as the engine does.  Float columns are left
    to the engine (float32 data vs a Python float literal is a precision hazard).
    """
    op, args = expr.op, expr.args
    if op in _CMP_OPS:
        a, b = args
        if a.op == "lit" and b.op == "col":
            a, b, op = b, a, _FLIP[op]
        if not (a.op == "col" and b.op == "lit"):
            return None
        name, value = a.args[0], b.args[0]
        kind = _kind(schema, name)
        if (kind == "int" and _is_int(value)) or (kind == "str" and isinstance(value, str)):
            return (name, op, value)
        return None
    if op == "isin":
        target, values = args
        if target.op != "col" or not values:
            return None
        name = target.args[0]
        kind = _kind(schema, name)
        if (kind == "int" and all(_is_int(v) for v in values)) or \
           (kind == "str" and all(isinstance(v, str) for v in values)):
            return (name, "isin", tuple(values))
    return None


def _kind(schema: pa.Schema, name: str) -> Optional[str]:
    i = schema.get_field_index(name)
    if i < 0:
        return None
    t = schema.field(i).type
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "str"
    return None


def _filters_expression(filters):
    import pyarrow.dataset as ds
    out = None
    for name, op, value in filters:
        f = ds.field(name)
        e = {"eq": lambda: f == value, "ne": lambda: f != value, "gt": lambda: f > value,
             "ge": lambda: f >= value, "lt": lambda: f < value, "le": lambda: f <= value,
             "isin": lambda: f.isin(list(value))}[op]()
        out = e if out is None else (out & e)
    return out


# ── resolution ──────────────────────────────────────────────────────────────

def resolve_plan(plan: LogicalPlan, read: bool = True) -> LogicalPlan:
    """Replace every ``ParquetScan`` by a ``Scan`` of just the needed columns.

    Also makes dictionary columns safe for the engine: a dictionary column stays a
    dictionary only when the query uses it exclusively as a bare group key or in
    ``col <cmp> literal`` / isin / startswith / contains filters (and the scan's
    columns are not passed through to the output, and there is no join); every
    other dictionary column is decoded to its plain type.

    With ``read=False`` nothing is read: each ``ParquetScan`` just records the
    ``columns`` and ``filters`` it would read (for ``explain``).
    """
    leaves = list(_leaves(plan))
    has_parquet = any(isinstance(l, ParquetScan) for l in leaves)
    has_dict = any(isinstance(l, Scan) and any(_is_dict(t) for t in l.table.schema.types)
                   for l in leaves)
    if not has_parquet and not has_dict:
        return plan
    info = _analyze(plan)

    def keepable(name: str, leaf) -> bool:
        return (name in info.refs and name not in info.bad
                and not info.has_join and not info.passthrough.get(id(leaf), True))

    def resolve_leaf(leaf: ParquetScan, pushed: tuple):
        names = leaf.schema.names
        if info.passthrough.get(id(leaf), True):
            cols = list(names)
        else:
            cols = [n for n in names if n in info.refs] or names[:1]
        if not read:
            return dataclasses.replace(leaf, columns=tuple(cols), filters=pushed or None)
        dict_cols = [n for n in dictionary_columns(leaf.path, leaf.schema, leaf.dictionary,
                                                   leaf.max_dictionary_bytes, cols)
                     if keepable(n, leaf)]
        import pyarrow.parquet as pq
        if os.path.isfile(leaf.path):
            table = _read_file(leaf.path, cols, dict_cols, pushed or ())
        else:
            table = pq.read_table(leaf.path, columns=cols,
                                  filters=_filters_expression(pushed) if pushed else None,
                                  read_dictionary=dict_cols or None)
        return Scan(_decode_unkept(table, leaf))

    def _decode_unkept(table, leaf):
        for i, (n, t) in enumerate(zip(table.column_names, table.schema.types)):
            if _is_dict(t) and not keepable(n, leaf):
                table = table.set_column(i, n, _decode_dict(table.column(n)))
        return table

    def resolve_scan(leaf: Scan):
        table = leaf.table
        drop = [n for n, t in zip(table.column_names, table.schema.types)
                if _is_dict(t) and not keepable(n, leaf)]
        if not drop:
            return leaf
        for n in drop:
            i = table.schema.get_field_index(n)
            table = table.set_column(i, n, _decode_dict(table.column(n)))
        return Scan(table)

    def rebuild(node):
        if isinstance(node, ParquetScan):
            return resolve_leaf(node, ())
        if isinstance(node, Scan):
            return resolve_scan(node)
        if isinstance(node, Filter):
            chain, n = [], node
            while isinstance(n, Filter):
                chain.append(n)
                n = n.input
            if isinstance(n, ParquetScan):
                pushed = []
                for f in chain:
                    for c in _conjuncts(f.predicate):
                        p = _pushable(c, n.schema)
                        if p is not None and p not in pushed:
                            pushed.append(p)
                out = resolve_leaf(n, tuple(pushed))
                for f in reversed(chain):
                    out = Filter(out, f.predicate)
                return out
            return Filter(rebuild(node.input), node.predicate)
        if isinstance(node, Join):
            return dataclasses.replace(node, left=rebuild(node.left), right=rebuild(node.right))
        if hasattr(node, "input"):
            return dataclasses.replace(node, input=rebuild(node.input))
        return node

    return rebuild(plan)
