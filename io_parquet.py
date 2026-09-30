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
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa

from .arrow_dict import decode as _decode_dict, is_dictionary as _is_dict, mark_dense as _mark_dense
from .lazy_expr import Expr
from .lazy_frame import (
    Aggregate, Distinct, Filter, Join, LazyFrame, Limit, LogicalPlan,
    ParquetScan, Project, Scan, Sort, Tail,
)

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
    schema = pq.read_schema(path)          # footer only
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
    return pq.read_table(path, columns=names, filters=filters,
                         read_dictionary=dict_cols or None)


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


def _metadata(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata


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
    created = []

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
        table = pq.read_table(leaf.path, columns=cols,
                              filters=_filters_expression(pushed) if pushed else None,
                              read_dictionary=dict_cols or None)
        if dict_cols:
            # Row-group dictionaries hold only values present in the file, and any
            # pushed-down filter is repeated by a Filter node that drops empty groups.
            table = _mark_dense(table, dict_cols)
        created.append(table)
        return Scan(table)

    def resolve_scan(leaf: Scan):
        table = leaf.table
        drop = [n for n, t in zip(table.column_names, table.schema.types)
                if _is_dict(t) and not keepable(n, leaf)]
        if not drop:
            return leaf
        for n in drop:
            i = table.schema.get_field_index(n)
            table = table.set_column(i, n, _decode_dict(table.column(n)))
        created.append(table)
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

    out = rebuild(plan)
    if created:
        _forget_table_caches()
    return out


def _forget_table_caches():
    """Engine caches are keyed by ``id(table)``; freshly built tables must not alias old ids."""
    from . import custom_ops as co
    for cache in (co._INPUT_PREP_CACHE, co._GROUP_ENCODE_CACHE,
                  co._GPU_BUFFER_CACHE, co._JOIN_RESULT_CACHE):
        cache.clear()
