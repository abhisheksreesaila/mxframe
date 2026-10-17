"""F-056 step 3a: hash-partitioned spill files for group state bigger than the memory budget.

A streamed aggregate whose merged partials no longer fit is written, and every later batch's partial with it, into
`PARTITIONS` Arrow IPC streams under `MXFRAME_SPILL_DIR` (default `~/.cache/mxframe/spill`, on disk): a row goes to partition `hash(group keys) % PARTITIONS`, so a
group lives in exactly one partition. Each partition is then merged on its own (by `streamed._State`, which
splits it again with another salt if even one partition is too big) and the partition outputs are concatenated.

The hash is the same for equal keys on every run and in every process (no Python `hash()`): it mixes the bits of
integers, floats (-0.0 / 0.0 and every NaN canonicalised, as the engine groups them, F-077) and the CRC of
anything else, NULL being its own value. That makes the partition of a group, and so the output order in
reproducible mode, a function of the data alone.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import warnings
import zlib
from typing import Iterator, List, Optional

import numpy as np
import pyarrow as pa

from . import np_arrow
from ._lazy_pc import pc

PARTITIONS = 16
MAX_DEPTH = 6                    # 16^6 partitions: past that the keys do not split (one group is bigger than memory)

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_NULL = np.uint64(0x9E3779B97F4A7C15)


def _mix(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finalizer: spreads every input bit over the 64 output bits (uint64 arithmetic wraps)."""
    x = x ^ (x >> np.uint64(30))
    x = x * _M1
    x = x ^ (x >> np.uint64(27))
    x = x * _M2
    return x ^ (x >> np.uint64(31))


def _value_hashes(values: pa.Array) -> np.ndarray:
    """One uint64 per (distinct) value of a dictionary."""
    t = values.type
    n = len(values)
    null = np_arrow.to_numpy(pc.is_null(values)) if values.null_count else np.zeros(n, dtype=np.bool_)
    if pa.types.is_integer(t) or pa.types.is_boolean(t) or pa.types.is_floating(t):
        filled = pc.fill_null(values, np_arrow.scalar(False if pa.types.is_boolean(t) else 0, t)) if values.null_count else values
        x = np_arrow.to_numpy(filled)
        if pa.types.is_floating(t):
            h = x.astype(np.float64).view(np.uint64)
        elif t == pa.uint64():
            h = x
        else:
            h = x.astype(np.int64).view(np.uint64)
    else:                                                            # strings, dates, decimals, ...
        h = np.fromiter((zlib.crc32(str(v).encode("utf-8", "surrogatepass")) if v is not None else 0
                         for v in values.to_pylist()), dtype=np.uint64, count=n)
    h = _mix(np.ascontiguousarray(h, dtype=np.uint64))
    h[null] = _NULL
    return h


def _column_hash(column) -> np.ndarray:
    a = np_arrow.combine(column)
    if pa.types.is_dictionary(a.type):
        a = a.dictionary_decode()
    a = np_arrow.canonical_floats(a)
    enc = pc.dictionary_encode(a, null_encoding="encode")
    return _value_hashes(enc.dictionary)[np_arrow.to_numpy(enc.indices).astype(np.int64)]


def hash_keys(table: pa.Table, names: List[str], salt: int = 0) -> np.ndarray:
    """A deterministic uint64 per row from the values of the key columns `names` (equal keys, equal hash)."""
    h = _mix(np.full(table.num_rows, salt + 1, dtype=np.uint64))
    for name in names:
        h = _mix(h * _M1 + _column_hash(table.column(name)))
    return h


class SpillError(OSError):
    """Spilling failed (no space, no write access): names the directory and what was written."""


def spill_base() -> str:
    """Where spill folders go: `MXFRAME_SPILL_DIR`, else `<cache dir>/mxframe/spill` on disk. Never `$TMPDIR` or
    /tmp by default: on many systems that is tmpfs, i.e. RAM, which is what spilling is meant to save."""
    d = os.environ.get("MXFRAME_SPILL_DIR", "").strip()
    if d:
        return os.path.abspath(os.path.expanduser(d))
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache, "mxframe", "spill")


def ram_backed(path: str) -> Optional[str]:
    """The file system type (`tmpfs` / `ramfs`) when `path` lives in RAM, per /proc/mounts, else None."""
    try:
        path = os.path.realpath(path)
        best, kind = "", None
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    mount = parts[1].replace("\\040", " ")
                    if (path == mount or path.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
                        best, kind = mount, parts[2]
        return kind if kind in ("tmpfs", "ramfs") else None
    except OSError:
        return None


_NAME = re.compile(r"mxframe-spill-(\d+)-")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True                    # exists, but not ours
    return True


def remove_stale(base: str) -> None:
    """Remove spill folders left by processes that no longer exist (killed before they could clean up)."""
    try:
        names = os.listdir(base)
    except OSError:
        return
    for n in names:
        m = _NAME.match(n)
        if m and int(m.group(1)) != os.getpid() and not _pid_alive(int(m.group(1))):
            shutil.rmtree(os.path.join(base, n), ignore_errors=True)


class SpillDir:
    """The temporary directory of one query's spill files; `close()` removes it (also after an error)."""

    def __init__(self):
        self.path: Optional[str] = None          # made when the first file is named: a query that never spills has none
        self.bytes = 0                 # on disk, counted when a partition is finished
        self.written = 0               # handed to the writers so far (approximate, in memory size)
        self.partitions = 0            # created, over every split level
        self.depth = 0                 # deepest split level used
        self._n = 0

    def name(self) -> str:
        if self.path is None:
            base = spill_base()
            try:
                os.makedirs(base, exist_ok=True)
                remove_stale(base)
                self.path = tempfile.mkdtemp(prefix=f"mxframe-spill-{os.getpid()}-", dir=base)
            except OSError as e:
                raise SpillError(f"cannot spill group state to disk: the spill directory {base} cannot be created "
                                 f"({e}); set MXFRAME_SPILL_DIR to a writable directory on disk") from e
            kind = ram_backed(self.path)
            if kind:
                warnings.warn(f"mxframe: the spill directory {base} is on {kind} (RAM), so spilling uses the memory it "
                              f"is meant to save; set MXFRAME_SPILL_DIR to a directory on disk",
                              RuntimeWarning, stacklevel=2)
        self._n += 1
        return os.path.join(self.path, f"s{self._n}")

    def check_space(self, need: int) -> None:
        free = shutil.disk_usage(self.path).free
        if free < need:
            raise SpillError(f"not enough disk space to spill group state: {self.path} has {free:,} bytes free, "
                             f"{need:,} are needed for the next write ({self.written:,} written so far); set "
                             f"MXFRAME_SPILL_DIR to a bigger disk")

    def close(self) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)


class Partitioned:
    """`PARTITIONS` Arrow IPC streams under a `SpillDir`: `write` splits a table by key hash, `read` yields the
    tables of one partition back in the order they were written."""

    def __init__(self, root: SpillDir, keys: List[str], salt: int, depth: int):
        self.root, self.keys, self.salt = root, list(keys), salt
        self._base = root.name()
        self._writers: dict = {}
        self._sinks: dict = {}
        self.rows = [0] * PARTITIONS
        root.partitions += PARTITIONS
        root.depth = max(root.depth, depth)

    def _path(self, i: int) -> str:
        return f"{self._base}_{i:02d}.arrows"

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self.root.check_space(int(table.nbytes * 1.2))
        try:
            self._write(table)
            self.root.written += table.nbytes
        except SpillError:
            raise
        except OSError as e:                       # a full disk surfaces as OSError / ArrowIOError
            raise SpillError(f"spilling group state to {self.root.path} failed after about "
                             f"{self.root.written:,} bytes were written: {e}") from e

    def _write(self, table: pa.Table) -> None:
        part = ((hash_keys(table, self.keys, self.salt) >> np.uint64(33)) % np.uint64(PARTITIONS)).astype(np.int64)
        order = np.argsort(part, kind="stable")                      # keeps the batch order inside a partition
        counts = np.bincount(part, minlength=PARTITIONS)
        taken = np_arrow.take_rows(table, order)
        at = 0
        for i, c in enumerate(counts.tolist()):
            if c:
                if i not in self._writers:
                    self._sinks[i] = pa.OSFile(self._path(i), "wb")
                    self._writers[i] = pa.ipc.new_stream(self._sinks[i], taken.schema)
                self._writers[i].write_table(taken.slice(at, c))
                self.rows[i] += c
                at += c

    def finish(self) -> None:
        for i, w in self._writers.items():
            w.close()
            self._sinks[i].close()
            self.root.bytes += os.path.getsize(self._path(i))
        self._writers, self._sinks = {}, {}

    def read(self, i: int) -> Iterator[pa.Table]:
        """The tables of partition `i` (none if nothing hashed there); the file is deleted once read."""
        path = self._path(i)
        if not os.path.exists(path):
            return
        try:
            with pa.OSFile(path, "rb") as f:
                for rb in pa.ipc.open_stream(f):
                    yield pa.Table.from_batches([rb])
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
