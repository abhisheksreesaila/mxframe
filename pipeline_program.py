"""Bytecode compiler for the fused pipeline kernel (F-008).

Compiles a filter predicate and a list of aggregates (sum / min / max / mean /
count of arithmetic expressions over numeric columns) into a compact bytecode
program that the ahead-of-time compiled kernels ``libmxpipeline_cpu.so`` and
``libmxpipeline_gpu.so`` interpret (MonetDB/X100-style vectorized
interpretation). Nothing here compiles at run time: no MAX Graph, no JIT.

Everything is evaluated in float64: integer and float32 columns are widened on
load, comparisons produce 1.0 / 0.0. String predicates are supported on
dictionary-encoded columns: literals are translated to dictionary codes here in
Python and the kernel only does an integer set-membership test (``IN_SET``).

Anything that cannot be expressed exactly returns ``None`` from
:func:`compile_pipeline` so the caller keeps its existing path: never a wrong
result.

Bytecode (5 int32 words per instruction: ``[op, dst, a, b, c]``; registers are
float64 vectors, ``k`` a constant-pool index, ``col`` a column index):

=========  ===============================================  ===================
op         meaning                                          operands
=========  ===============================================  ===================
LOAD_COL   dst = float64(col[rows])                         dst, col
LOAD_CONST dst = broadcast(consts[k])                       dst, k
ADD SUB    dst = a (+|-|*|/) b                              dst, a, b
MUL DIV
LT LE GT   dst = 1.0 if a (<|<=|>|>=|==|!=) b else 0.0      dst, a, b
GE EQ NE
CMPC       dst = column <cmp> constant, compared in the      dst, col, k, cmp
           column's own type: a float32 column compares
           against the constant converted to float32 (the
           literal adopts the column's type, as in DuckDB,
           Polars and NumPy 2), an integer column against the
           exact ceil / floor of the constant, float64 as is
AND OR     dst = logical and / or of a, b (non-zero = true) dst, a, b
NOT        dst = 1.0 if a == 0 else 0.0                     dst, a
FLOOR      dst = floor(a)                                   dst, a
SELECT     dst = a if cond != 0 else b                      dst, cond, a, b
BETWEEN    dst = lo <= x <= hi                              dst, x, lo, hi
IN_SET     dst = set[int(x) - lo]  (dictionary codes)       dst, x, set
AGG_SUM    slot += src (per group)                          slot, src
AGG_MIN    slot = min(slot, src)                            slot, src
AGG_MAX    slot = max(slot, src)                            slot, src
=========  ===============================================  ===================

Layout: 16-int header (magic, n_regs, n_prologue, n_pred, n_body, n_slots,
pred_reg, n_cols), then the prologue (constants, run once per worker), the
predicate section (dense rows), the body section (only rows that passed) and
the aggregate slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pyarrow as pa
from .np_arrow import array as _npa, to_numpy as _np_of, scalar as _npscalar, combine as _combine
from ._lazy_pc import pc

from .lazy_expr import Expr

MAGIC = 0x4D585031
HEADER_INTS = 16
INSTR_INTS = 5
MAX_REGS = 48          # kernels keep registers on the stack / in local memory
MAX_INSTR = 320        # GPU kernels stage the program in shared memory
MAX_CONSTS = 64
MAX_SET_SPAN = 1 << 22  # widest numeric IN_SET table we will build

OP_LOAD_COL, OP_LOAD_CONST = 1, 2
OP_ADD, OP_SUB, OP_MUL, OP_DIV = 10, 11, 12, 13
OP_LT, OP_LE, OP_GT, OP_GE, OP_EQ, OP_NE = 20, 21, 22, 23, 24, 25
OP_AND, OP_OR, OP_NOT, OP_FLOOR = 30, 31, 32, 33
OP_SELECT, OP_BETWEEN, OP_IN_SET, OP_CMPC = 35, 36, 37, 38
OP_AGG_SUM, OP_AGG_MIN, OP_AGG_MAX = 50, 51, 52

OP_NAMES = {
    OP_LOAD_COL: "LOAD_COL", OP_LOAD_CONST: "LOAD_CONST",
    OP_ADD: "ADD", OP_SUB: "SUB", OP_MUL: "MUL", OP_DIV: "DIV",
    OP_LT: "LT", OP_LE: "LE", OP_GT: "GT", OP_GE: "GE", OP_EQ: "EQ", OP_NE: "NE",
    OP_AND: "AND", OP_OR: "OR", OP_NOT: "NOT", OP_FLOOR: "FLOOR",
    OP_SELECT: "SELECT", OP_BETWEEN: "BETWEEN", OP_IN_SET: "IN_SET", OP_CMPC: "CMPC",
    OP_AGG_SUM: "AGG_SUM", OP_AGG_MIN: "AGG_MIN", OP_AGG_MAX: "AGG_MAX",
}

_ARITH = {"add": OP_ADD, "sub": OP_SUB, "mul": OP_MUL, "div": OP_DIV}
_CMP = {"lt": OP_LT, "le": OP_LE, "gt": OP_GT, "ge": OP_GE, "eq": OP_EQ, "ne": OP_NE}
_FOLD = {
    OP_ADD: lambda a, b: a + b, OP_SUB: lambda a, b: a - b, OP_MUL: lambda a, b: a * b,
    OP_LT: lambda a, b: float(a < b), OP_LE: lambda a, b: float(a <= b),
    OP_GT: lambda a, b: float(a > b), OP_GE: lambda a, b: float(a >= b),
    OP_EQ: lambda a, b: float(a == b), OP_NE: lambda a, b: float(a != b),
    OP_AND: lambda a, b: float(a != 0 and b != 0), OP_OR: lambda a, b: float(a != 0 or b != 0),
}

# dtype codes shared with the kernels
DT_I32, DT_I64, DT_F32, DT_F64, DT_U8 = 0, 1, 2, 3, 4
_DT_NP = {DT_I32: np.int32, DT_I64: np.int64, DT_F32: np.float32, DT_F64: np.float64, DT_U8: np.uint8}
_NP_DT = {np.dtype(v): k for k, v in _DT_NP.items()}
_DIRECT = {pa.int32(): (4, DT_I32), pa.int64(): (8, DT_I64), pa.float32(): (4, DT_F32), pa.float64(): (8, DT_F64)}


def dtype_code(arr: np.ndarray) -> int:
    return _NP_DT[arr.dtype]


class Unsupported(Exception):
    """Raised inside the compiler; compile_pipeline turns it into ``None``."""


@dataclass
class Partials:
    """Per-(aggregate slot, group) partial results of one or more row ranges."""

    vals: np.ndarray   # float64 [n_slots, n_groups]: sum / min / max per slot
    cnt: np.ndarray    # int64   [n_groups]: rows that passed the predicate

    @property
    def n_groups(self) -> int:
        return int(self.cnt.shape[0])


class Buf(NamedTuple):
    """One contiguous, null-free column chunk: address, row count and element type.

    ``owner`` keeps the memory alive (an Arrow array or a numpy array); no data is copied
    to describe a chunk, which is what makes binding a many-row-group table cheap.
    """

    addr: int
    n: int
    itemsize: int
    dt: int
    owner: Any

    def slice(self, lo: int, hi: int) -> "Buf":
        return Buf(self.addr + lo * self.itemsize, hi - lo, self.itemsize, self.dt, self.owner)


def buf_from_numpy(a: np.ndarray) -> Buf:
    return Buf(int(a.ctypes.data), len(a), a.itemsize, dtype_code(a), a)


def buf_numpy(b: Buf) -> np.ndarray:
    """A numpy view of a chunk (for host-side use and tests)."""
    if isinstance(b.owner, np.ndarray) and b.owner.ctypes.data == b.addr and len(b.owner) == b.n:
        return b.owner
    import ctypes
    raw = (ctypes.c_char * (b.n * b.itemsize)).from_address(b.addr)
    return np.frombuffer(raw, dtype=_DT_NP[b.dt], count=b.n)


@dataclass
class ChunkedColumns:
    """Column data as row-aligned chunks (nothing concatenated).

    ``bounds[k] = (start, end)`` is the row range of chunk ``k`` and
    ``chunks[k][c]`` the :class:`Buf` of column ``c`` in it. Arrow tables read from
    Parquet arrive in many chunks (one per row group); the kernels take them as
    they are instead of paying for a full copy of every column.
    """

    bounds: List[Tuple[int, int]]
    chunks: List[List[Buf]]
    n_cols: int

    @property
    def n_rows(self) -> int:
        return self.bounds[-1][1] if self.bounds else 0

    @staticmethod
    def from_arrays(arrays: List[np.ndarray]) -> "ChunkedColumns":
        n = min((len(a) for a in arrays), default=0)
        if not n:
            return ChunkedColumns([], [], len(arrays))
        return ChunkedColumns([(0, n)], [[buf_from_numpy(a[:n]) for a in arrays]], len(arrays))

    def column(self, c: int) -> np.ndarray:
        """Column ``c`` as one contiguous array (copies when it has several chunks)."""
        parts = [buf_numpy(ch[c]) for ch in self.chunks]
        if not parts:
            return np.empty(0)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _align(per_column: List[List[Tuple[int, Buf]]]) -> ChunkedColumns:
    """Cut every column at the union of all columns' chunk boundaries."""
    import bisect
    cuts = sorted({s for col in per_column for s, _ in col})
    n = min((col[-1][0] + col[-1][1].n if col else 0) for col in per_column) if per_column else 0
    cuts = [c for c in cuts if c < n] + [n]
    bounds, chunks = [], []
    starts = [[s for s, _ in col] for col in per_column]
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        row = []
        for col, st in zip(per_column, starts):
            i = bisect.bisect_right(st, a) - 1
            cs, buf = col[i]
            row.append(buf.slice(a - cs, b - cs))
        bounds.append((a, b))
        chunks.append(row)
    return ChunkedColumns(bounds, chunks, len(per_column))


@dataclass
class PipelineProgram:
    code: np.ndarray                  # int32
    consts: np.ndarray                # float64
    set_bytes: np.ndarray             # uint8 (membership tables, concatenated)
    set_meta: np.ndarray              # int64 [n_sets, 3] = offset, length, lo
    columns: List[str]                # source column per LOAD_COL index
    dict_arrays: Dict[str, pa.DictionaryArray]   # string columns, as codes
    slot_kinds: List[str]             # "sum" | "min" | "max" per slot
    outputs: List[Tuple[str, str, Optional[int]]]  # (name, op, slot) per user aggregate
    n_regs: int
    n_instr: int
    identity: np.ndarray = field(default=None)  # float64 [n_slots] initial value per slot
    nan_slots: Dict[str, int] = field(default_factory=dict)  # min / max output -> hidden NaN-count slot
    slot_int: List[bool] = field(default_factory=list)       # slot input is integer-valued (whole numbers; exact while within 2^53)
    dict_concat: Dict[str, Dict[tuple, int]] = field(default_factory=dict)   # concat_dicts mode: name -> {dictionary key: offset}

    @property
    def n_slots(self) -> int:
        return len(self.slot_kinds)

    def disassemble(self) -> List[str]:
        out = []
        c = self.code
        n_pro, n_pred, n_body, n_slots = (int(c[2]), int(c[3]), int(c[4]), int(c[5]))
        base = HEADER_INTS
        for section, count in (("prologue", n_pro), ("pred", n_pred), ("body", n_body), ("aggs", n_slots)):
            out.append(f"; {section}")
            for _ in range(count):
                ins = [int(x) for x in c[base:base + INSTR_INTS]]
                out.append(f"{OP_NAMES[ins[0]]:<10} {ins[1:]}")
                base += INSTR_INTS
        return out

    def op_names(self) -> List[str]:
        """Opcode mnemonics in program order (for tests / debugging)."""
        n = int(self.code[2]) + int(self.code[3]) + int(self.code[4]) + int(self.code[5])
        return [OP_NAMES[int(self.code[HEADER_INTS + i * INSTR_INTS])] for i in range(n)]

    # ── binding to data ────────────────────────────────────────────────────
    def bind_chunks(self, table: pa.Table) -> ChunkedColumns:
        """Like :meth:`bind` but keeps the table's chunks: no column is concatenated,
        and null-free int32 / int64 / float32 / float64 chunks are described by their
        Arrow buffer address without creating any array object."""
        per_column: List[List[Tuple[int, Buf]]] = []
        for name in self.columns:
            if name in self.dict_arrays:
                per_column.append([(0, buf_from_numpy(_dict_codes(self.dict_arrays[name])))])
                continue
            col = table.column(name)
            chunks = col.chunks if isinstance(col, pa.ChunkedArray) else [col]
            parts, off = [], 0
            for ch in chunks:
                n = len(ch)
                if n == 0:
                    continue
                spec = _DIRECT.get(ch.type)
                if spec is not None and ch.null_count == 0:
                    itemsize, dt = spec
                    buf = Buf(ch.buffers()[1].address + ch.offset * itemsize, n, itemsize, dt, ch)
                else:
                    arr = _column_numpy(ch)
                    if arr is None:
                        raise Unsupported(f"column {name!r} cannot be bound")
                    buf = buf_from_numpy(arr)
                parts.append((off, buf))
                off += n
            per_column.append(parts)
        return _align(per_column)

    def bind(self, table: pa.Table) -> List[np.ndarray]:
        """Contiguous numpy arrays for ``self.columns`` (zero-copy where possible)."""
        arrays = []
        for name in self.columns:
            if name in self.dict_arrays:
                arrays.append(_dict_codes(self.dict_arrays[name]))
                continue
            arr = _column_numpy(table.column(name))
            if arr is None:
                raise Unsupported(f"column {name!r} cannot be bound")
            arrays.append(arr)
        return arrays


# ── column access helpers ──────────────────────────────────────────────────

DICT_MEMBER_PY_MAX = 50_000          # above this a Python loop costs more than importing pyarrow.compute


def _dict_member(d: pa.Array, test, arrow) -> np.ndarray:
    """Membership of every dictionary value in a string predicate (dictionary-sized work, never
    row work). Plain Python below DICT_MEMBER_PY_MAX values: importing pyarrow.compute costs
    ~25 ms per process, more than the test itself on TPC-H's dictionaries; `arrow` above it."""
    if len(d) <= DICT_MEMBER_PY_MAX:
        return np.fromiter((v is not None and bool(test(v)) for v in d.to_pylist()), dtype=bool, count=len(d))
    return np.asarray(_np_of(pc.fill_null(arrow(), _npscalar(False)), zero_copy_only=False))


def _dict_codes(d: pa.DictionaryArray) -> np.ndarray:
    idx = d.indices
    if idx.type != pa.int32():
        idx = pc.cast(idx, pa.int32())
    return np.ascontiguousarray(_np_of(idx, zero_copy_only=False))


def _single_array(col) -> pa.Array:
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 1:
            return col.chunk(0)
        if col.num_chunks == 0:
            return _npa([], type=col.type)
        return _combine(col)
    return col


def _column_numpy(col) -> Optional[np.ndarray]:
    """Numeric Arrow column -> contiguous numpy in a kernel-supported dtype."""
    arr = _single_array(col)
    if isinstance(arr, pa.ChunkedArray):  # combine_chunks on some versions
        arr = arr.chunk(0) if arr.num_chunks else _npa([], type=arr.type)
    if arr.null_count:
        return None
    t = arr.type
    if pa.types.is_boolean(t):
        return np.ascontiguousarray(_np_of(pc.cast(arr, pa.uint8()), zero_copy_only=False))
    if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
        return None
    if pa.types.is_uint64(t) or pa.types.is_float16(t):
        return None
    np_arr = _np_of(arr, zero_copy_only=False)
    if np_arr.dtype in _NP_DT:
        return np.ascontiguousarray(np_arr)
    if np_arr.dtype.kind in "iu":
        return np.ascontiguousarray(np_arr.astype(np.int64 if np_arr.dtype.itemsize > 2 else np.int32))
    return None


def _is_string_type(t) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t) or (
        pa.types.is_dictionary(t) and (pa.types.is_string(t.value_type) or pa.types.is_large_string(t.value_type)))


def _numeric_kind(t) -> Optional[str]:
    """'int' | 'float' | 'bool' | None for a supported numeric Arrow type."""
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_floating(t) and not pa.types.is_float16(t):
        return "float"
    if pa.types.is_integer(t) and not pa.types.is_uint64(t):
        return "int"
    return None


# ── expression compiler ────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Const:
    value: float
    kind: str          # "int" | "float" | "bool"


@dataclass(frozen=True)
class _Val:
    vreg: int
    kind: str          # "int" | "float" | "bool"
    bound: int = 0     # int kind: every value is within +-bound (exact Python int)
    src: Optional[str] = None   # int kind: a bare column; its bound is read from the data when needed


EXACT = 1 << 53        # integers within +-2^53 are exact in float64, and so are their sums / products


class _Section:
    """Instruction list over virtual registers with value numbering (CSE)."""

    def __init__(self):
        self.instrs: List[List[int]] = []   # [op, dst_vreg, a, b, c] (a/b/c vregs unless noted)
        self.cse: Dict[tuple, int] = {}


def _dict_key(d: pa.Array) -> tuple:
    """Identity of a chunk's dictionary: its buffers (chunks sharing one dictionary share it)."""
    bufs = d.buffers()
    return tuple(b.address if b is not None else 0 for b in bufs) + (d.offset, len(d))


class _Builder:
    def __init__(self, table: pa.Table, concat_dicts: bool = False):
        self.table = table
        # native executor: a string column's dictionary is the concatenation of its chunks'
        # dictionaries (no unify, no row work); dict_concat[name] = {dictionary key: offset}
        self.concat_dicts = concat_dicts
        self.dict_concat: Dict[str, Dict[tuple, int]] = {}
        self.dict_values: Dict[str, pa.Array] = {}
        self.columns: List[str] = []
        self.col_kind: Dict[str, str] = {}
        self.col_bounds: Dict[str, int] = {}
        self.dict_arrays: Dict[str, pa.DictionaryArray] = {}
        self.consts: List[float] = []
        self.const_ix: Dict[float, int] = {}
        self.sets: List[Tuple[bytes, int]] = []      # (table bytes, lo)
        self.set_ix: Dict[Tuple[bytes, int], int] = {}
        self._nv = 0
        self.const_vregs: Dict[int, int] = {}        # const index -> vreg
        self.sec: _Section = _Section()

    # -- registers / constants
    def new_vreg(self) -> int:
        self._nv += 1
        return self._nv - 1

    def pool_index(self, value: float) -> int:
        v = float(value)
        key = v if v == v else "nan"
        if key not in self.const_ix:
            if len(self.consts) >= MAX_CONSTS:
                raise Unsupported("too many constants")
            self.const_ix[key] = len(self.consts)
            self.consts.append(v)
        return self.const_ix[key]

    def const_reg(self, value: float) -> int:
        k = self.pool_index(value)
        if k not in self.const_vregs:
            self.const_vregs[k] = self.new_vreg()
        return self.const_vregs[k]

    def as_reg(self, x) -> int:
        return self.const_reg(x.value) if isinstance(x, _Const) else x.vreg

    def emit(self, op: int, a: int = 0, b: int = 0, c: int = 0, key_extra: tuple = ()) -> int:
        key = (op, a, b, c) + key_extra
        hit = self.sec.cse.get(key)
        if hit is not None:
            return hit
        dst = self.new_vreg()
        self.sec.instrs.append([op, dst, a, b, c])
        self.sec.cse[key] = dst
        return dst

    # -- columns
    def _column(self, name: str) -> int:
        if name not in self.table.column_names:
            raise Unsupported(f"unknown column {name!r}")
        if name not in self.columns:
            self.columns.append(name)
        return self.columns.index(name)

    def column_kind(self, name: str) -> str:
        if name not in self.table.column_names:
            raise Unsupported(f"unknown column {name!r}")
        t = self.table.schema.field(name).type
        if _is_string_type(t):
            return "str"
        k = _numeric_kind(t)
        if k is None:
            raise Unsupported(f"column {name!r} has unsupported type {t}")
        return k

    def dictionary(self, name: str) -> pa.Array:
        """The dictionary (distinct values) of a string column, built once."""
        if self.concat_dicts:
            if name not in self.dict_values:
                col = self.table.column(name)
                if not pa.types.is_dictionary(col.type) or col.type.index_type != pa.int32():
                    raise Unsupported("string predicate on a column without int32 dictionary codes")
                if col.null_count:
                    raise Unsupported("string column has nulls")
                offs, parts, pos = {}, [], 0
                for ch in col.chunks:
                    d = ch.dictionary
                    if d.null_count:
                        raise Unsupported("string column has nulls")
                    k = _dict_key(d)
                    if k not in offs:
                        offs[k] = pos
                        parts.append(d)
                        pos += len(d)
                if not parts:
                    raise Unsupported("no rows")
                self.dict_concat[name] = offs
                self.dict_values[name] = pa.concat_arrays(parts) if len(parts) > 1 else parts[0]
            return self.dict_values[name]
        if name not in self.dict_arrays:
            col = self.table.column(name)
            arr = _single_array(col)
            if isinstance(col, pa.ChunkedArray) and col.num_chunks > 1 and pa.types.is_dictionary(col.type):
                arr = _combine(col.unify_dictionaries())
            if arr.null_count:
                raise Unsupported("string column has nulls")
            if not pa.types.is_dictionary(arr.type):
                arr = pc.dictionary_encode(arr)
            if arr.indices.null_count:
                raise Unsupported("string column has nulls")
            self.dict_arrays[name] = arr
        return self.dict_arrays[name].dictionary

    def load_col(self, name: str):
        kind = self.column_kind(name)
        if kind == "str":
            raise Unsupported("string column used as a number")
        ix = self._column(name)
        if kind != "int":
            return _Val(self.emit(OP_LOAD_COL, ix, key_extra=("col",)), kind)
        return _Val(self.emit(OP_LOAD_COL, ix, key_extra=("col",)), kind, src=name)

    def col_bound(self, name: str) -> int:
        """max |value| of an integer column, from the data's own min / max (one pass, only when integer
        arithmetic needs it; the type's range would decline int32 * int32)."""
        if name not in self.col_bounds:
            col = self.table.column(name)
            tb: Optional[int] = 0
            for ch in (col.chunks if isinstance(col, pa.ChunkedArray) else [col]):
                if not len(ch):
                    continue
                if ch.null_count:
                    tb = None
                    break
                a = _np_of(ch, zero_copy_only=False)
                tb = max(tb, abs(int(a.max())), abs(int(a.min())))
            self.col_bounds[name] = EXACT + 1 if tb is None else tb
        return self.col_bounds[name]

    def bound_of(self, x) -> int:
        if isinstance(x, _Const):
            return abs(int(x.value))
        return self.col_bound(x.src) if x.src is not None else x.bound

    def set_reg(self, src_col: str, member: np.ndarray, lo: int = 0):
        """IN_SET over a column's values / codes given a boolean membership table."""
        data = np.ascontiguousarray(member.astype(np.uint8)).tobytes()
        key = (data, lo)
        if key not in self.set_ix:
            self.set_ix[key] = len(self.sets)
            self.sets.append(key)
        return self.set_ix[key]

    # -- expression walk
    def compile(self, e: Expr):
        op, args = e.op, e.args
        if op == "col":
            return self.load_col(args[0])
        if op == "lit":
            v = args[0]
            if isinstance(v, bool):
                return _Const(float(v), "bool")
            if isinstance(v, (int, np.integer)):
                if abs(int(v)) > 2 ** 53:
                    raise Unsupported("integer literal beyond 2**53 is not exact in float64")
                return _Const(float(v), "int")
            if isinstance(v, (float, np.floating)):
                if v != v:
                    raise Unsupported("NaN literal")
                return _Const(float(v), "float")
            raise Unsupported(f"literal {type(v).__name__} outside a string predicate")
        if op in _ARITH:
            return self._arith(op, args)
        if op in _CMP:
            return self._compare(op, args)
        if op in ("and", "or"):
            return self._logic(op, args)
        if op == "not":
            x = self._need(args[0], "bool")
            if isinstance(x, _Const):
                return _Const(float(x.value == 0), "bool")
            return _Val(self.emit(OP_NOT, x.vreg), "bool")
        if op == "isin":
            return self._isin(args[0], args[1])
        if op in ("startswith", "contains"):
            return self._str_match(op, args[0], args[1])
        if op == "case_when":
            cond = self._need(args[0], "bool")
            then = self._need(args[1], "num")
            other = self._need(args[2], "num")
            kind = "int" if (then.kind == "int" and other.kind == "int") else "float"
            if isinstance(cond, _Const):
                return then if cond.value != 0 else other
            bound = max(self.bound_of(then), self.bound_of(other)) if kind == "int" else 0
            return _Val(self.emit(OP_SELECT, cond.vreg, self.as_reg(then), self.as_reg(other)), kind, bound)
        if op == "year":
            x = self.compile(args[0])
            if isinstance(x, _Const) or x.kind != "int":
                raise Unsupported("year() only on integer YYYYMMDD columns")
            d = self.const_reg(10000.0)
            q = self.emit(OP_DIV, x.vreg, d)
            return _Val(self.emit(OP_FLOOR, q), "int", self.bound_of(x) // 10000 + 1)
        raise Unsupported(f"op {op!r}")

    def _need(self, e: Expr, want: str):
        x = self.compile(e)
        if want == "bool" and x.kind != "bool":
            raise Unsupported("boolean expected")
        if want == "num" and x.kind == "bool":
            raise Unsupported("number expected")
        return x

    def _arith(self, op: str, args):
        a = self._need(args[0], "num")
        b = self._need(args[1], "num")
        code = _ARITH[op]
        if code == OP_DIV and a.kind == "int" and b.kind == "int":
            raise Unsupported("integer division has floor semantics in Arrow")
        kind = "float" if "float" in (a.kind, b.kind) or code == OP_DIV else "int"
        bound = 0
        if kind == "int":
            # the interpreter is float64: an integer intermediate is exact only within +-2^53, so bound
            # it from its inputs (columns' real min / max) and decline when it could pass (F-037)
            ba, bb = self.bound_of(a), self.bound_of(b)
            bound = ba * bb if code == OP_MUL else ba + bb
            if bound > EXACT:
                raise Unsupported("integer arithmetic could pass 2**53, not exact in float64")
        if isinstance(a, _Const) and isinstance(b, _Const):
            if code == OP_DIV:
                if b.value == 0:
                    raise Unsupported("division by constant zero")
                return _Const(a.value / b.value, kind)
            return _Const(_FOLD[code](a.value, b.value), kind)
        return _Val(self.emit(code, self.as_reg(a), self.as_reg(b)), kind, bound)

    def _compare(self, op: str, args):
        lhs, rhs = args
        # string column vs string literal -> dictionary code membership
        for col_e, lit_e, flip in ((lhs, rhs, False), (rhs, lhs, True)):
            if (col_e.op == "col" and lit_e.op == "lit" and isinstance(lit_e.args[0], str)):
                if self.column_kind(col_e.args[0]) != "str":
                    raise Unsupported("string literal against a numeric column")
                if op not in ("eq", "ne"):
                    raise Unsupported("ordered string comparison")
                d = self.dictionary(col_e.args[0])
                lit_s = lit_e.args[0]
                member = _dict_member(d, lambda v: v == lit_s,
                                      lambda: pc.equal(d, _npscalar(lit_s, type=d.type)))
                return self._dict_test(col_e.args[0], member, negate=(op == "ne"))
        cmpc = self._try_cmpc(op, lhs, rhs)
        if cmpc is not None:
            return cmpc
        a = self._need(lhs, "num")
        b = self._need(rhs, "num")
        code = _CMP[op]
        if isinstance(a, _Const) and isinstance(b, _Const):
            return _Const(_FOLD[code](a.value, b.value), "bool")
        return _Val(self.emit(code, self.as_reg(a), self.as_reg(b)), "bool")

    _FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}

    def _fold(self, e: Expr) -> Optional[float]:
        """Value of a constant numeric expression (literals and + - * /), else None."""
        if e.op == "lit":
            v = e.args[0]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                return None
            if isinstance(v, (int, np.integer)) and abs(int(v)) > 2 ** 53:
                raise Unsupported("integer literal beyond 2**53 is not exact in float64")
            return float(v)
        if e.op in _ARITH:
            a, b = self._fold(e.args[0]), self._fold(e.args[1])
            if a is None or b is None:
                return None
            if e.op == "div":
                return a / b if b != 0 else None
            if a.is_integer() and b.is_integer() and self._int_expr(e):
                r = _FOLD[_ARITH[e.op]](int(a), int(b))       # exact Python integers
                if abs(r) > EXACT:
                    raise Unsupported("integer arithmetic could pass 2**53, not exact in float64")
                return float(r)
            return _FOLD[_ARITH[e.op]](a, b)
        return None

    def _int_expr(self, e: Expr) -> bool:
        """A constant expression made of integer literals and + - * only."""
        if e.op == "lit":
            return isinstance(e.args[0], (int, np.integer)) and not isinstance(e.args[0], bool)
        return e.op in ("add", "sub", "mul") and all(self._int_expr(a) for a in e.args)

    def _try_cmpc(self, op: str, lhs: Expr, rhs: Expr):
        """`column <cmp> constant` in the column's own type, or None if not that shape."""
        for col_e, other, flip in ((lhs, rhs, False), (rhs, lhs, True)):
            if col_e.op != "col":
                continue
            c = self._fold(other)
            if c is None or c != c:
                continue
            name = col_e.args[0]
            if self.column_kind(name) not in ("int", "float", "bool"):
                continue
            eff = self._FLIP[op] if flip else op
            t = self.table.schema.field(name).type
            kind = "f32" if pa.types.is_float32(t) else "f64" if pa.types.is_floating(t) else "int"
            res = _threshold(kind, eff, c)
            if isinstance(res, bool):
                return _Const(float(res), "bool")
            thr, code = res
            ix = self._column(name)
            k = self.pool_index(thr)
            return _Val(self.emit(OP_CMPC, ix, k, code, key_extra=("cmpc",)), "bool")
        return None

    def _dict_test(self, name: str, member: np.ndarray, negate: bool = False):
        ix = self._column(name)
        self.dictionary(name)   # make sure it is registered
        codes = _Val(self.emit(OP_LOAD_COL, ix, key_extra=("col",)), "int")
        s = self.set_reg(name, member)
        r = self.emit(OP_IN_SET, codes.vreg, s, key_extra=("set",))
        if negate:
            r = self.emit(OP_NOT, r)
        return _Val(r, "bool")

    def _str_match(self, op: str, col_e: Expr, pattern):
        if col_e.op != "col" or not isinstance(pattern, str):
            raise Unsupported("string op on a non-column")
        name = col_e.args[0]
        if self.column_kind(name) != "str":
            raise Unsupported("string op on a numeric column")
        d = self.dictionary(name)
        if op == "startswith":
            member = _dict_member(d, lambda v: v.startswith(pattern), lambda: pc.starts_with(d, pattern=pattern))
        else:
            member = _dict_member(d, lambda v: pattern in v, lambda: pc.match_substring(d, pattern=pattern))
        return self._dict_test(name, member)

    def _isin(self, col_e: Expr, values):
        values = list(values)
        if col_e.op == "col" and self.column_kind(col_e.args[0]) == "str":
            if not all(isinstance(v, str) for v in values):
                raise Unsupported("non-string values against a string column")
            d = self.dictionary(col_e.args[0])
            vs = set(values)
            member = _dict_member(d, lambda v: v in vs, lambda: pc.is_in(d, value_set=_npa(values, type=d.type)))
            return self._dict_test(col_e.args[0], member)
        x = self._need(col_e, "num")
        if isinstance(x, _Const):
            raise Unsupported("isin on a constant")
        if not values:
            return _Const(0.0, "bool")
        if not all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in values):
            raise Unsupported("isin values must be numbers")
        if any(isinstance(v, (int, np.integer)) and abs(int(v)) > 2 ** 53 for v in values):
            raise Unsupported("integer literal beyond 2**53 is not exact in float64")
        if x.kind == "int" and all(float(v) == int(v) for v in values):
            ints = sorted({int(v) for v in values})
            lo, hi = ints[0], ints[-1]
            if hi - lo + 1 <= MAX_SET_SPAN:
                member = np.zeros(hi - lo + 1, dtype=np.uint8)
                member[np.array(ints, dtype=np.int64) - lo] = 1
                s = self.set_reg("", member, lo)
                return _Val(self.emit(OP_IN_SET, x.vreg, s, key_extra=("set",)), "bool")
        if len(values) > 16:
            raise Unsupported("wide float isin")
        acc = None
        for v in values:
            t = self._compare("eq", (col_e, Expr("lit", v))) if col_e.op == "col" else None
            if t is None:
                t = _Val(self.emit(OP_EQ, x.vreg, self.const_reg(float(v))), "bool")
            if isinstance(t, _Const):
                if t.value == 0:
                    continue
                return t
            acc = t if acc is None else _Val(self.emit(OP_OR, acc.vreg, t.vreg), "bool")
        return acc if acc is not None else _Const(0.0, "bool")

    def _logic(self, op: str, args):
        code = OP_AND if op == "and" else OP_OR
        # and(ge(x, lo), le(x, hi)) -> BETWEEN
        if op == "and":
            b = self._try_between(args[0], args[1])
            if b is not None:
                return b
        a = self._need(args[0], "bool")
        b = self._need(args[1], "bool")
        if isinstance(a, _Const) and isinstance(b, _Const):
            return _Const(_FOLD[code](a.value, b.value), "bool")
        if isinstance(a, _Const) or isinstance(b, _Const):
            c, v = (a, b) if isinstance(a, _Const) else (b, a)
            if code == OP_AND:
                return v if c.value != 0 else _Const(0.0, "bool")
            return _Const(1.0, "bool") if c.value != 0 else v
        return _Val(self.emit(code, a.vreg, b.vreg), "bool")

    def _try_between(self, l: Expr, r: Expr):
        if l.op == "ge" and r.op == "le" and l.args[0].signature() == r.args[0].signature():
            x_e, lo_e, hi_e = l.args[0], l.args[1], r.args[1]
        else:
            return None
        if x_e.op == "lit":
            return None
        if x_e.op == "col" and self._fold(lo_e) is not None and self._fold(hi_e) is not None:
            return None   # column between constants: two CMPCs, which follow the column-type rule
        try:
            x = self._need(x_e, "num")
            lo = self._need(lo_e, "num")
            hi = self._need(hi_e, "num")
        except Unsupported:
            return None
        if isinstance(x, _Const):
            return None
        return _Val(self.emit(OP_BETWEEN, x.vreg, self.as_reg(lo), self.as_reg(hi)), "bool")


def compile_pipeline(predicates: List[Expr], aggs: List[Expr], table: pa.Table,
                     concat_dicts: bool = False) -> Optional[PipelineProgram]:
    """Compile ``predicates`` (ANDed) and ``aggs`` against ``table``.

    Returns ``None`` when anything is unsupported (the caller falls back to its
    existing path). ``concat_dicts`` (native executor): string predicates test codes against the
    concatenation of the chunks' dictionaries (``PipelineProgram.dict_concat``) instead of a
    unified dictionary built over the rows.
    """
    try:
        return _compile(predicates, aggs, table, concat_dicts)
    except Unsupported:
        return None


def _compile(predicates: List[Expr], aggs: List[Expr], table: pa.Table, concat_dicts: bool = False) -> PipelineProgram:
    b = _Builder(table, concat_dicts)

    # ── predicate section (dense rows) ─────────────────────────────────────
    pred_sec = _Section()
    b.sec = pred_sec
    pred = None
    for p in predicates:
        v = b._need(p, "bool")
        if isinstance(v, _Const):
            if v.value != 0:
                continue          # constant true: no effect
            pred = _Const(0.0, "bool")
            break                 # constant false: nothing survives
        pred = v if pred is None else _Val(b.emit(OP_AND, pred.vreg, v.vreg), "bool")
    pred_reg_v = -1
    if pred is not None:
        pred_reg_v = b.as_reg(pred)

    # ── body section (selected rows only) ──────────────────────────────────
    body_sec = _Section()
    b.sec = body_sec
    slot_index: Dict[tuple, int] = {}
    slot_kinds: List[str] = []
    slot_src: List[int] = []
    slot_int: List[bool] = []
    outputs: List[Tuple[str, str, Optional[int]]] = []
    nan_slots: Dict[str, int] = {}
    for i, e in enumerate(aggs):
        name = e._alias or f"agg_{i}"
        if e.op == "count":
            outputs.append((name, "count", None))
            continue
        if e.op not in ("sum", "min", "max", "mean"):
            raise Unsupported(f"aggregate {e.op!r}")
        inner = b._need(e.args[0], "num")
        kind = "sum" if e.op == "mean" else e.op
        key = (kind, e.args[0].signature())
        if e.op in ("min", "max") and not isinstance(inner, _Const) and inner.kind == "float":
            # DuckDB orders NaN above every number: min skips NaN (NaN only if the group is all NaN),
            # max is NaN if any value is. NaN is masked out of the compare (so no min/max instruction
            # ever sees one) and counted in a hidden sum slot; finalize applies the rule.
            isnan = b.emit(OP_NE, inner.vreg, inner.vreg)
            sentinel = b.const_reg(float("inf") if e.op == "min" else float("-inf"))
            masked = b.emit(OP_SELECT, isnan, sentinel, inner.vreg)
            key = (kind, e.args[0].signature(), "nan-masked")
            nan_key = ("sum", e.args[0].signature(), "nan-count")
            if nan_key not in slot_index:
                slot_index[nan_key] = len(slot_kinds)
                slot_kinds.append("sum")
                slot_src.append(isnan)
                slot_int.append(False)
            nan_slots[name] = slot_index[nan_key]
            inner = _Val(masked, "float")
        if key not in slot_index:
            slot_index[key] = len(slot_kinds)
            slot_kinds.append(kind)
            slot_src.append(b.as_reg(inner))
            slot_int.append(inner.kind in ("int", "bool"))
        outputs.append((name, e.op, slot_index[key]))
    if not outputs and not concat_dicts:
        raise Unsupported("no aggregates")       # (a predicate-only program is the executor's leaf filter)

    # ── dead code, then keep only the columns still read ───────────────────
    pred_sec.instrs = _dce(pred_sec.instrs, [pred_reg_v] if pred_reg_v >= 0 else [])
    body_sec.instrs = _dce(body_sec.instrs, list(slot_src))
    used: List[int] = []
    for ins in pred_sec.instrs + body_sec.instrs:
        if ins[0] in (OP_LOAD_COL, OP_CMPC) and ins[2] not in used:
            used.append(ins[2])
    remap = {old: new for new, old in enumerate(used)}
    for ins in pred_sec.instrs + body_sec.instrs:
        if ins[0] in (OP_LOAD_COL, OP_CMPC):
            ins[2] = remap[ins[2]]
    b.columns = [b.columns[old] for old in used]
    b.dict_arrays = {n: d for n, d in b.dict_arrays.items() if n in b.columns}

    # ── register allocation ────────────────────────────────────────────────
    const_vregs = [b.const_vregs[k] for k in sorted(b.const_vregs)]
    agg_instrs = [[OP_AGG_SUM if k == "sum" else OP_AGG_MIN if k == "min" else OP_AGG_MAX,
                   s, src, 0, 0] for s, (k, src) in enumerate(zip(slot_kinds, slot_src))]
    body_all = body_sec.instrs + agg_instrs

    phys, n_regs = _allocate_sections(const_vregs, pred_sec.instrs, pred_reg_v, body_all)
    if n_regs > MAX_REGS:
        raise Unsupported("too many registers")

    def phys_of(v: int) -> int:
        return phys[v]

    def encode(ins: List[int]) -> List[int]:
        op = ins[0]
        if op == OP_LOAD_COL:
            return [op, phys_of(ins[1]), ins[2], 0, 0]
        if op == OP_CMPC:
            return [op, phys_of(ins[1]), ins[2], ins[3], ins[4]]
        if op == OP_IN_SET:
            return [op, phys_of(ins[1]), phys_of(ins[2]), ins[3], 0]
        if op in (OP_NOT, OP_FLOOR):
            return [op, phys_of(ins[1]), phys_of(ins[2]), 0, 0]
        if op in (OP_SELECT, OP_BETWEEN):
            return [op, phys_of(ins[1]), phys_of(ins[2]), phys_of(ins[3]), phys_of(ins[4])]
        return [op, phys_of(ins[1]), phys_of(ins[2]), phys_of(ins[3]), 0]

    prologue = [[OP_LOAD_CONST, phys[b.const_vregs[k]], k, 0, 0] for k in sorted(b.const_vregs)]
    pred_code = [encode(i) for i in pred_sec.instrs]
    body_code = [encode(i) for i in body_sec.instrs]
    agg_code = [[ins[0], ins[1], phys_of(ins[2]), 0, 0] for ins in agg_instrs]
    total_instr = len(prologue) + len(pred_code) + len(body_code) + len(agg_code)
    if total_instr > MAX_INSTR:
        raise Unsupported("program too long")

    header = [0] * HEADER_INTS
    header[0] = MAGIC
    header[1] = n_regs
    header[2] = len(prologue)
    header[3] = len(pred_code)
    header[4] = len(body_code)
    header[5] = len(agg_code)
    header[6] = phys_of(pred_reg_v) if pred_reg_v >= 0 else -1
    header[7] = len(b.columns)
    flat = header + [w for ins in prologue + pred_code + body_code + agg_code for w in ins]

    # sets: concatenate membership tables
    set_bytes = np.zeros(max(1, sum(len(d) for d, _ in b.sets)), dtype=np.uint8)
    set_meta = np.zeros((max(1, len(b.sets)), 3), dtype=np.int64)
    off = 0
    for i, (data, lo) in enumerate(b.sets):
        set_bytes[off:off + len(data)] = np.frombuffer(data, dtype=np.uint8)
        set_meta[i] = (off, len(data), lo)
        off += len(data)

    identity = np.array([0.0 if k == "sum" else np.inf if k == "min" else -np.inf for k in slot_kinds],
                        dtype=np.float64)
    return PipelineProgram(
        code=np.asarray(flat, dtype=np.int32),
        consts=np.asarray(b.consts if b.consts else [0.0], dtype=np.float64),
        set_bytes=set_bytes,
        set_meta=np.ascontiguousarray(set_meta),
        columns=list(b.columns),
        dict_arrays=dict(b.dict_arrays),
        dict_concat={n: o for n, o in b.dict_concat.items() if n in b.columns},
        slot_kinds=slot_kinds,
        slot_int=slot_int,
        outputs=outputs,
        nan_slots=nan_slots,
        n_regs=n_regs,
        n_instr=total_instr,
        identity=identity,
    )


def _threshold(kind: str, op: str, c: float):
    """Threshold and compare code for ``column <op> c``, compared in the column's own type.

    * float32 column: the literal is converted to float32 (nearest), as DuckDB, Polars and
      NumPy 2 do; this is what makes TPC-H Q6's ``l_discount <= 0.07`` include 0.07 in
      float32 data, where comparing in float64 would drop those rows.
    * float64 column: as is.
    * integer column: exact (``ceil`` / ``floor`` of the constant), never rounded.

    Returns ``bool`` when the result is the same for every value, else
    ``(threshold, compare_opcode)``. ``kind`` is "f32", "f64" or "int".
    """
    code = _CMP[op]
    if kind == "f64":
        return c, code
    if kind == "f32":
        with np.errstate(over="ignore"):
            return float(np.float32(c)), code
    # integer column (int8..int64, bool)
    import math
    if c >= 2.0 ** 63 or c == np.inf:
        return op in ("lt", "le", "ne")
    if c <= -(2.0 ** 63) or c == -np.inf:
        return op in ("gt", "ge", "ne")
    lo, hi = math.ceil(c), math.floor(c)
    if op == "ge":
        return float(lo), code
    if op == "gt":
        return float(hi), code
    if op == "le":
        return float(hi), code
    if op == "lt":
        return float(lo), code
    integral = lo == hi
    if op == "eq":
        return (float(lo), code) if integral else False
    return (float(lo), code) if integral else True


def _dce(instrs: List[List[int]], roots: List[int]) -> List[List[int]]:
    """Drop instructions whose result is never used."""
    live = set(roots)
    keep: List[List[int]] = []
    for ins in reversed(instrs):
        if ins[1] in live:
            keep.append(ins)
            live.update(_uses_ext(ins))
    keep.reverse()
    return keep


def _allocate_sections(const_vregs: List[int], pred: List[List[int]], pred_reg: int,
                       body: List[List[int]]) -> Tuple[Dict[int, int], int]:
    """Register allocation over the predicate and body sections.

    Constants live in dedicated registers for the whole program. Within a
    section a register is released after its last use and may be reused as the
    destination of the same instruction. The predicate register stays live
    until the end of the predicate section; nothing survives into the body.
    """
    phys: Dict[int, int] = {}
    for i, v in enumerate(const_vregs):
        phys[v] = i
    n_regs = len(const_vregs)
    cset = set(const_vregs)

    pool: List[int] = []   # registers free at the start of the next section
    for sec, keep in ((pred, pred_reg), (body, -1)):
        last: Dict[int, int] = {}
        for i, ins in enumerate(sec):
            for u in _uses_ext(ins):
                last[u] = i
        if keep >= 0 and keep not in cset:
            last[keep] = len(sec)
        free: List[int] = list(pool)
        held: Dict[int, int] = {}
        allocated: List[int] = []
        for i, ins in enumerate(sec):
            for u in set(_uses_ext(ins)):
                if u in cset:
                    continue
                if last.get(u) == i and u in held:
                    free.append(held.pop(u))
            if ins[0] < OP_AGG_SUM:
                dst = ins[1]
                if free:
                    p = free.pop()
                else:
                    p = n_regs
                    n_regs += 1
                phys[dst] = p
                held[dst] = p
                allocated.append(p)
        # nothing but constants survives a section boundary
        pool = sorted(set(pool) | set(allocated))
    return phys, n_regs


def _uses_ext(ins: List[int]) -> List[int]:
    op = ins[0]
    if op in (OP_LOAD_COL, OP_LOAD_CONST, OP_CMPC):
        return []
    if op in (OP_NOT, OP_FLOOR, OP_IN_SET):
        return [ins[2]]
    if op in (OP_SELECT, OP_BETWEEN):
        return [ins[2], ins[3], ins[4]]
    if op >= OP_AGG_SUM:
        return [ins[2]]
    return [ins[2], ins[3]]


# ── partial results: merge and finalize ────────────────────────────────────

def empty_partials(program: PipelineProgram, n_groups: int) -> Partials:
    vals = np.repeat(program.identity[:, None], n_groups, axis=1).copy()
    return Partials(vals=vals, cnt=np.zeros(n_groups, dtype=np.int64))


def merge_partials(program: PipelineProgram, parts: List[Partials]) -> Partials:
    """Combine partials of disjoint row ranges (any order)."""
    parts = list(parts)
    if not parts:
        raise ValueError("merge_partials needs at least one partial")
    out = Partials(vals=parts[0].vals.copy(), cnt=parts[0].cnt.copy())
    for p in parts[1:]:
        for s, kind in enumerate(program.slot_kinds):
            if kind == "sum":
                out.vals[s] += p.vals[s]
            elif kind == "min":
                np.minimum(out.vals[s], p.vals[s], out=out.vals[s])
            else:
                np.maximum(out.vals[s], p.vals[s], out=out.vals[s])
        out.cnt += p.cnt
    return out


def finalize(program: PipelineProgram, partials: Partials, grouped: bool):
    """Turn partials into ``(present, columns)``.

    ``present`` is a boolean mask of groups that saw at least one row (grouped
    queries drop the others, like the existing path); ``columns`` maps each
    output name to a numpy array over the present groups (all of the single
    group for a global aggregate). Sums and means are float64, min/max float64;
    counts are int64. A global aggregate over no rows yields NULL (a masked
    array) for sum / min / max / mean, and 0 for count, like SQL.
    """
    cnt = partials.cnt
    present = cnt > 0
    if not grouped:
        present = np.ones(1, dtype=bool)
    cols: Dict[str, np.ndarray] = {}
    for name, op, slot in program.outputs:
        if op == "count":
            cols[name] = cnt[present].astype(np.int64)
            continue
        v = partials.vals[slot][present].copy()
        if op == "mean":
            c = cnt[present]
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(c > 0, v / np.maximum(c, 1), 0.0)
        if name in program.nan_slots:
            n_nan = partials.vals[program.nan_slots[name]][present]
            c = cnt[present]
            hit = (n_nan > 0) if op == "max" else ((n_nan >= c) & (c > 0))
            v = np.where(hit, np.nan, v)
        if not grouped and cnt[0] == 0:
            v = np.ma.masked_all(v.shape, dtype=v.dtype)      # SQL: sum/min/max/mean over no rows is NULL
        cols[name] = v
    return present, cols
