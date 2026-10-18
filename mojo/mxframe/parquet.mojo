"""Parquet files with no Python (docs/mxframe-mojo-api.md section 2): `write_parquet` and `read_parquet`.

The `.omx` document is one Parquet file whose key-value metadata holds one JSON string under the key "omx".

How MX frame's kinds are stored (any Parquet reader understands them):
  INT      INT64
  DECIMAL  INT64 with the DECIMAL(18, scale) logical type (the stored value is the scaled integer, as in memory)
  DATE     INT32 with the DATE logical type (days since 1970)
  BOOL     BOOLEAN
  TEXT     BYTE_ARRAY (STRING): a dictionary page with the column's words, data pages of dictionary indices
A column with nulls is OPTIONAL: its definition levels are the validity bitmap's bits (Parquet packs them the same
way, least significant first), written as bit-packed runs. Pages are compressed with zstd (default), snappy or not
at all; one data page per column per row group. `write_parquet` writes to `path + ".tmp"` and renames it into place,
so a crash or a refused table never destroys an existing file.

Reading takes what other writers produce: signed and unsigned integers of 8 to 64 bits (a UINT_64 past the int64 range
is refused, naming the column), TEXT only for string types (a BYTE_ARRAY with no string type is BINARY and is refused,
as are TIME columns, so invalid UTF-8 never reaches a table), and a timestamp becomes its UTC day: isAdjustedToUTC
or not, the stored instant is floored to a day since 1970 in UTC, never shifted to a local zone.

The file is untrusted input: every length, count, run and bit width is checked against what the page or footer can
hold before anything is allocated or read, and a bad file raises "(corrupt file)", never crashes.

F-138b, the no-codec build: zstd and snappy are C libraries. An Apple app links neither, so there (any target but
Linux, or `-D MXFRAME_NO_CODECS` on Linux to test it) `write_parquet` saves UNCOMPRESSED whatever `compression` says,
and `read_parquet` refuses a zstd or snappy column chunk with an error naming the column. No codec symbol is
referenced, so nothing links. On Linux both codecs work as before.
"""
from std.ffi import external_call
from std.sys import is_defined, CompilationTarget
from std.bit import pop_count, byte_swap
from .kernels._fpenv import par_n, core_count
from std.collections import Dict
from std.memory import ArcPointer
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL
from .thrift import ThriftWriter, ThriftReader, T_I32, T_I64, T_BINARY, T_LIST, T_STRUCT, T_TRUE, T_FALSE, T_STOP, _valid_utf8

# Parquet enums (parquet.thrift)
comptime PT_BOOLEAN = 0
comptime PT_INT32 = 1
comptime PT_INT64 = 2
comptime PT_INT96 = 3
comptime PT_FLOAT = 4
comptime PT_DOUBLE = 5
comptime PT_BYTE_ARRAY = 6
comptime PT_FIXED = 7
comptime REQUIRED = 0
comptime OPTIONAL = 1
comptime REPEATED = 2
comptime CT_UTF8 = 0
comptime CT_ENUM = 4
comptime CT_DECIMAL = 5
comptime CT_DATE = 6
comptime CT_TIME_MILLIS = 7
comptime CT_TIME_MICROS = 8
comptime CT_TIMESTAMP_MILLIS = 9
comptime CT_TIMESTAMP_MICROS = 10
comptime CT_UINT_8 = 11
comptime CT_UINT_64 = 14
comptime CT_JSON = 19
comptime ENC_PLAIN = 0
comptime ENC_PLAIN_DICTIONARY = 2
comptime ENC_RLE = 3
comptime ENC_BIT_PACKED = 4
comptime ENC_DELTA_BINARY_PACKED = 5
comptime ENC_RLE_DICTIONARY = 8
comptime PAGE_DATA = 0
comptime PAGE_DICTIONARY = 2
comptime PAGE_DATA_V2 = 3

# compression
comptime UNCOMPRESSED = 0
comptime SNAPPY = 1
comptime GZIP = 2
comptime ZSTD = 6

comptime NO_CODECS = is_defined["MXFRAME_NO_CODECS"]() or not CompilationTarget.is_linux()

comptime DEFAULT_ROW_GROUP = 1 << 20
comptime INT32_MAX = 2147483647
comptime MAX_DECIMAL = 1_000_000_000_000_000_000      # DECIMAL(18, s): the stored integer has at most 18 digits


# ── bytes ────────────────────────────────────────────────────────────────────────────────────────────

def _put32(mut b: List[UInt8], v: Int):
    for k in range(4):
        b.append(UInt8((v >> (8 * k)) & 0xFF))


def _put64(mut b: List[UInt8], v: Int64):
    for k in range(8):
        b.append(UInt8((v >> Int64(8 * k)) & 0xFF))


def _put_varint(mut b: List[UInt8], v: Int):
    var x = UInt64(v)
    while x >= 0x80:
        b.append(UInt8(x & 0x7F) | 0x80)
        x >>= 7
    b.append(UInt8(x))


def _bit_width(max_value: Int) -> Int:
    var w = 0
    while (1 << w) <= max_value:
        w += 1
    return w


def _compress(codec: Int, src: List[UInt8]) raises -> List[UInt8]:
    if codec == UNCOMPRESSED:
        return src.copy()
    comptime if NO_CODECS:
        raise Error("this build has no zstd or snappy: pages are saved UNCOMPRESSED")
    else:
        if codec == ZSTD:
            var cap = external_call["ZSTD_compressBound", Int](len(src))
            var out = List[UInt8](length=max(cap, 1), fill=0)
            var n = external_call["ZSTD_compress", Int](out.unsafe_ptr(), cap, src.unsafe_ptr(), len(src), Int32(3))
            if external_call["ZSTD_isError", UInt32](n) != 0:
                raise Error("zstd compression failed")
            out.resize(n, 0)
            return out^
        if codec == SNAPPY:
            var cap = external_call["snappy_max_compressed_length", Int](len(src))
            var out = List[UInt8](length=max(cap, 1), fill=0)
            var n = cap
            if external_call["snappy_compress", Int32](src.unsafe_ptr(), len(src), out.unsafe_ptr(), UnsafePointer(to=n)) != 0:
                raise Error("snappy compression failed")
            out.resize(n, 0)
            return out^
        raise Error("compression codec " + String(codec) + " is not supported (UNCOMPRESSED, SNAPPY, ZSTD)")


def _decompress(codec: Int, src: Int, n: Int, size: Int) raises -> List[UInt8]:
    """`n` compressed bytes at address `src` into `size` bytes. Both sizes come from the file: they are checked
    (and the codec's own stated size compared) before the output is allocated."""
    if n < 0 or size < 0 or size > INT32_MAX:
        raise Error("a Parquet page has an impossible size (corrupt file)")
    if codec == UNCOMPRESSED:
        if n != size:
            raise Error("an uncompressed Parquet page has two different sizes (corrupt file)")
        var out = List[UInt8](unsafe_uninit_length=max(size, 1))
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=src)
        for i in range(size):
            out[i] = p[i]
        out.resize(size, 0)
        return out^
    comptime if NO_CODECS:
        if codec == ZSTD or codec == SNAPPY:
            raise Error("this build has no zstd or snappy, so it cannot read this page")
        raise Error("compression codec " + String(codec) + " is not supported (this build reads UNCOMPRESSED only)")
    else:
        if codec == ZSTD:
            var stated = external_call["ZSTD_getFrameContentSize", UInt64](src, n)
            if stated < (UInt64(1) << 62) and stated != UInt64(size):
                raise Error("a zstd page did not decompress to its stated size (corrupt file)")
            var out = List[UInt8](unsafe_uninit_length=max(size, 1))
            var got = external_call["ZSTD_decompress", Int](out.unsafe_ptr(), size, src, n)
            if external_call["ZSTD_isError", UInt32](got) != 0 or got != size:
                raise Error("a zstd page did not decompress to its stated size (corrupt file)")
            out.resize(size, 0)
            return out^
        if codec == SNAPPY:
            var stated = 0
            if external_call["snappy_uncompressed_length", Int32](src, n, UnsafePointer(to=stated)) != 0 or stated != size:
                raise Error("a snappy page did not decompress to its stated size (corrupt file)")
            var out = List[UInt8](unsafe_uninit_length=max(size, 1))
            var got = size
            if external_call["snappy_uncompress", Int32](src, n, out.unsafe_ptr(), UnsafePointer(to=got)) != 0 or got != size:
                raise Error("a snappy page did not decompress to its stated size (corrupt file)")
            out.resize(size, 0)
            return out^
        raise Error("compression codec " + String(codec) + " is not supported yet (UNCOMPRESSED, SNAPPY, ZSTD)")


# ── writing ──────────────────────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct _Chunk(Copyable, Movable):
    var offset: Int              # where the column chunk starts (its dictionary page, else its data page)
    var data_offset: Int
    var dict_offset: Int         # -1 without a dictionary
    var compressed: Int
    var uncompressed: Int
    var values: Int


def write_parquet(path: String, t: SharedTable, metadata: String = "", compression: Int = ZSTD,
                  row_group_rows: Int = DEFAULT_ROW_GROUP) raises:
    """Writes the table to `path`; `metadata` (a JSON string) is stored under the key "omx"."""
    _write(path, t._table[], metadata, compression, row_group_rows)


def write_parquet(path: String, t: Table, metadata: String = "", compression: Int = ZSTD,
                  row_group_rows: Int = DEFAULT_ROW_GROUP) raises:
    _write(path, t, metadata, compression, row_group_rows)


def _page_header(mut b: List[UInt8], kind: Int, uncompressed: Int, compressed: Int, values: Int, encoding: Int):
    var w = ThriftWriter()
    w.i32(1, kind)
    w.i32(2, uncompressed)
    w.i32(3, compressed)
    if kind == PAGE_DICTIONARY:
        w.begin_struct(7)
        w.i32(1, values)
        w.i32(2, encoding)
        w.end_struct()
    else:
        w.begin_struct(5)
        w.i32(1, values)
        w.i32(2, encoding)
        w.i32(3, ENC_RLE)          # definition levels
        w.i32(4, ENC_RLE)          # repetition levels (none)
        w.end_struct()
    w.end()
    b.extend(w^.take())


def _check_page(col: Column, uncompressed: Int, compressed: Int) raises:
    """A page header stores both sizes as int32."""
    if uncompressed > INT32_MAX - 4096 or compressed > INT32_MAX - 4096:
        raise Error("column `" + col.name + "`: a page of " + String(uncompressed) + " bytes does not fit a Parquet page" +
                    " (use a smaller row_group_rows)")


def _column_pages(mut file: List[UInt8], col: Column, lo: Int, hi: Int, optional: Bool, codec: Int) raises -> _Chunk:
    """Appends one column chunk (rows lo..hi) to `file`."""
    if hi - lo > INT32_MAX:
        raise Error("column `" + col.name + "`: " + String(hi - lo) + " rows in one row group do not fit a Parquet page (use a smaller row_group_rows)")
    var chunk = _Chunk(len(file), 0, -1, 0, 0, hi - lo)
    var start = len(file)
    if col.kind == TEXT:                                # the dictionary page: every word, PLAIN
        var d = List[UInt8]()
        for word in col.words:
            var bytes = word.as_bytes()
            _put32(d, len(bytes))
            for c in bytes:
                d.append(c)
        var cd = _compress(codec, d)
        _check_page(col, len(d), len(cd))
        chunk.dict_offset = len(file)
        _page_header(file, PAGE_DICTIONARY, len(d), len(cd), len(col.words), ENC_PLAIN)
        chunk.uncompressed += len(file) - chunk.dict_offset + len(d)       # header + page, uncompressed
        file.extend(cd^)
    var body = List[UInt8]()
    if optional:                                        # definition levels: the validity bits, bit-packed runs
        var levels = List[UInt8]()
        var n = hi - lo
        var groups = (n + 7) // 8
        _put_varint(levels, (groups << 1) | 1)
        for g in range(groups):
            var byte = UInt8(0)
            for k in range(8):
                var row = lo + 8 * g + k
                if row < hi and not col.is_null(row):
                    byte |= UInt8(1) << UInt8(k)
            levels.append(byte)
        _put32(body, len(levels))
        body.extend(levels^)
    if col.kind == TEXT:                                # dictionary indices: bit width, then bit-packed runs
        var w = max(_bit_width(len(col.words) - 1), 1)
        body.append(UInt8(w))
        var idx = List[Int64]()
        for row in range(lo, hi):
            if not col.is_null(row):
                idx.append(col.values[row])
        var groups = (len(idx) + 7) // 8
        _put_varint(body, (groups << 1) | 1)
        var acc: UInt64 = 0
        var nbits = 0
        for i in range(groups * 8):
            var v = UInt64(idx[i]) if i < len(idx) else 0
            acc |= v << UInt64(nbits)
            nbits += w
            while nbits >= 8:
                body.append(UInt8(acc & 0xFF))
                acc >>= 8
                nbits -= 8
        if nbits > 0:
            body.append(UInt8(acc & 0xFF))
    elif col.kind == BOOL:                              # PLAIN booleans: one bit each, least significant first
        var acc = 0
        var k = 0
        for row in range(lo, hi):
            if col.is_null(row):
                continue
            if col.values[row] != 0:
                acc |= 1 << k
            k += 1
            if k == 8:
                body.append(UInt8(acc))
                acc = 0
                k = 0
        if k > 0:
            body.append(UInt8(acc))
    elif col.kind == DATE:
        for row in range(lo, hi):
            if not col.is_null(row):
                var v = col.values[row]
                if v < Int64(Int32.MIN) or v > Int64(Int32.MAX):
                    raise Error("column `" + col.name + "`: day " + String(v) + " does not fit a Parquet DATE")
                _put32(body, Int(v))
    else:                                               # INT, DECIMAL: PLAIN int64
        for row in range(lo, hi):
            if not col.is_null(row):
                var v = col.values[row]
                if col.kind == DECIMAL and (v >= Int64(MAX_DECIMAL) or v <= -Int64(MAX_DECIMAL)):
                    raise Error("column `" + col.name + "`: " + String(v) + " does not fit DECIMAL(18, " +
                                String(col.scale) + ") (more than 18 digits)")
                _put64(body, v)
    var cb = _compress(codec, body)
    _check_page(col, len(body), len(cb))
    chunk.data_offset = len(file)
    _page_header(file, PAGE_DATA, len(body), len(cb), hi - lo, ENC_RLE_DICTIONARY if col.kind == TEXT else ENC_PLAIN)
    chunk.uncompressed += len(file) - chunk.data_offset + len(body)
    file.extend(cb^)
    chunk.compressed = len(file) - start
    return chunk^


def _write(path: String, t: Table, metadata: String, requested: Int, row_group_rows: Int) raises:
    var codec = requested
    comptime if NO_CODECS:
        codec = UNCOMPRESSED                       # no zstd / snappy linked: whatever was asked for
    t.check()
    if t.columns() == 0:
        raise Error("a table with no columns cannot be written as Parquet")
    if row_group_rows < 8 or row_group_rows % 8 != 0:
        raise Error("row_group_rows must be a positive multiple of 8, got " + String(row_group_rows))
    var file: List[UInt8] = [0x50, 0x41, 0x52, 0x31]                 # "PAR1"
    var n = t.rows()
    var ncols = t.columns()
    var groups = List[List[_Chunk]]()
    var starts = List[Int]()
    var lo = 0
    while lo < n or (n == 0 and len(groups) == 0):
        var hi = min(lo + row_group_rows, n)
        var chunks = List[_Chunk]()
        for c in range(ncols):
            chunks.append(_column_pages(file, t.cols[c], lo, hi, t.cols[c].nulls > 0, codec))
        groups.append(chunks^)
        starts.append(lo)
        if n == 0:
            break
        lo = hi

    var w = ThriftWriter()
    w.i32(1, 1)                                         # version
    w.list_header(2, T_STRUCT, ncols + 1)               # schema: the root, then one element per column
    w.begin_list_element_struct()
    w.binary(4, "schema")
    w.i32(5, ncols)
    w.end_struct()
    for c in range(ncols):
        ref col = t.cols[c]
        w.begin_list_element_struct()
        var pt = PT_BYTE_ARRAY if col.kind == TEXT else PT_BOOLEAN if col.kind == BOOL else PT_INT32 if col.kind == DATE else PT_INT64
        w.i32(1, pt)
        w.i32(3, OPTIONAL if col.nulls > 0 else REQUIRED)
        w.binary(4, col.name)
        if col.kind == TEXT:
            w.i32(6, CT_UTF8)
            w.begin_struct(10)
            w.begin_struct(1)                           # LogicalType.STRING
            w.end_struct()
            w.end_struct()
        elif col.kind == DATE:
            w.i32(6, CT_DATE)
            w.begin_struct(10)
            w.begin_struct(6)                           # LogicalType.DATE
            w.end_struct()
            w.end_struct()
        elif col.kind == DECIMAL:
            w.i32(6, CT_DECIMAL)
            w.i32(7, col.scale)
            w.i32(8, 18)
            w.begin_struct(10)
            w.begin_struct(5)                           # LogicalType.DECIMAL
            w.i32(1, col.scale)
            w.i32(2, 18)
            w.end_struct()
            w.end_struct()
        w.end_struct()
    w.i64(3, Int64(n))
    w.list_header(4, T_STRUCT, len(groups))             # row groups
    for g in range(len(groups)):
        var rows = min(starts[g] + row_group_rows, n) - starts[g]
        var total = 0
        w.begin_list_element_struct()
        w.list_header(1, T_STRUCT, ncols)
        for c in range(ncols):
            ref ch = groups[g][c]
            ref col = t.cols[c]
            total += ch.uncompressed
            w.begin_list_element_struct()
            w.i64(2, Int64(ch.offset))
            w.begin_struct(3)                           # ColumnMetaData
            var pt = PT_BYTE_ARRAY if col.kind == TEXT else PT_BOOLEAN if col.kind == BOOL else PT_INT32 if col.kind == DATE else PT_INT64
            w.i32(1, pt)
            if col.kind == TEXT:
                w.list_header(2, T_I32, 3)
                w.zigzag(ENC_PLAIN)
                w.zigzag(ENC_RLE)
                w.zigzag(ENC_RLE_DICTIONARY)
            else:
                w.list_header(2, T_I32, 2)
                w.zigzag(ENC_PLAIN)
                w.zigzag(ENC_RLE)
            w.list_header(3, T_BINARY, 1)
            w.raw_binary(col.name)
            w.i32(4, codec)
            w.i64(5, Int64(rows))
            w.i64(6, Int64(ch.uncompressed))
            w.i64(7, Int64(ch.compressed))
            w.i64(9, Int64(ch.data_offset))
            if ch.dict_offset >= 0:
                w.i64(11, Int64(ch.dict_offset))
            w.end_struct()
            w.end_struct()
        w.i64(2, Int64(total))
        w.i64(3, Int64(rows))
        w.end_struct()
    if metadata.byte_length() > 0:
        w.list_header(5, T_STRUCT, 1)
        w.begin_list_element_struct()
        w.binary(1, "omx")
        w.binary(2, metadata)
        w.end_struct()
    w.binary(6, "MX frame version 0.6.0")
    w.end()
    var meta = w^.take()
    var footer = len(meta)
    file.extend(meta^)
    _put32(file, footer)
    file.extend([UInt8(0x50), UInt8(0x41), UInt8(0x52), UInt8(0x31)])
    # An atomic save: a crash never loses the existing file. The temp name carries the process id, so two
    # processes saving the same path never write into one temp file; a failed write removes it.
    var tmp = path + "." + String(external_call["getpid", Int32]()) + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write_bytes(Span(file))
        var dest = path
        if external_call["rename", Int32](tmp.as_c_string_span().ptr(), dest.as_c_string_span().ptr()) != 0:
            raise Error("could not move `" + tmp + "` to `" + path + "`")
    except e:
        _ = external_call["unlink", Int32](tmp.as_c_string_span().ptr())
        raise e^


# ── reading ──────────────────────────────────────────────────────────────────────────────────────────

struct ParquetResult(Movable):
    var table: Table
    var metadata: String             # the "omx" key-value metadata, "" when there is none

    def __init__(out self, var table: Table, var metadata: String):
        self.table = table^
        self.metadata = metadata^

    def take_table(mut self) -> Table:
        """The table, moved out (the result keeps an empty one), so `.share()` works while `metadata` is still read."""
        var t = Table.empty()
        swap(t, self.table)
        return t^


@fieldwise_init
struct _Leaf(Copyable, Movable):
    var name: String
    var ptype: Int
    var type_length: Int
    var repetition: Int
    var converted: Int               # -1 when absent
    var scale: Int
    var precision: Int
    var logical: Int                 # the LogicalType union's field id, -1 when absent
    var ts_unit: Int                 # TIMESTAMP: 1 millis, 2 micros, 3 nanos
    var children: Int
    var unsigned: Bool               # an INTEGER logical type with isSigned false


@fieldwise_init
struct _ChunkMeta(Copyable, Movable):
    var ptype: Int
    var codec: Int
    var values: Int
    var data_off: Int
    var dict_off: Int
    var size: Int                    # total_compressed_size: the bytes of the chunk's pages (headers included)
    var group_rows: Int              # its row group's num_rows, -1 when the footer has none


struct _Bytes:
    """A cursor over `n` bytes at `base`."""
    var base: Int
    var n: Int
    var pos: Int

    def __init__(out self, base: Int, n: Int):
        self.base = base
        self.n = n
        self.pos = 0

    def at(self, i: Int) -> Int:
        return Int(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=self.base)[i])

    def need(self, k: Int) raises:
        if k < 0 or k > self.n - self.pos:
            raise Error("a Parquet page ends early (corrupt file)")

    def byte(mut self) raises -> Int:
        self.need(1)
        self.pos += 1
        return self.at(self.pos - 1)

    def le(mut self, k: Int) raises -> Int64:
        self.need(k)
        var v: UInt64 = 0
        for i in range(k):
            v |= UInt64(self.at(self.pos + i)) << UInt64(8 * i)
        self.pos += k
        if k < 8 and (v >> UInt64(8 * k - 1)) & 1 == 1:          # sign-extend
            v |= ~UInt64(0) << UInt64(8 * k)
        return Int64(v)

    def varint(mut self) raises -> UInt64:
        var v: UInt64 = 0
        var shift = 0
        while True:
            var b = self.byte()
            v |= UInt64(b & 0x7F) << UInt64(shift)
            if b < 0x80:
                return v
            shift += 7
            if shift > 63:
                raise Error("a Parquet page has a bad varint (corrupt file)")

    def zigzag(mut self) raises -> Int64:
        var u = self.varint()
        return Int64(u >> 1) ^ -Int64(u & 1)

    def small(mut self) raises -> Int:
        """A varint count or header: below 2^62, so doubling or multiplying by 8 cannot overflow."""
        var u = self.varint()
        if u >= (UInt64(1) << 62):
            raise Error("a Parquet page has an impossible count (corrupt file)")
        return Int(u)


def _unpack(mut c: _Bytes, width: Int, count: Int, mut out: List[Int64]) raises:
    """`count` values of `width` bits, least significant first (Parquet's bit packing)."""
    if width < 0 or width > 64:
        raise Error("a Parquet page has a bit width of " + String(width) + " (corrupt file)")
    if count < 0:
        raise Error("a Parquet page has a negative count (corrupt file)")
    if width == 0:                                      # the callers cap `count` at the values still needed
        for _ in range(count):
            out.append(0)
        return
    if count > ((c.n - c.pos) * 8) // width:            # checked before count * width, which could overflow
        raise Error("a Parquet page ends early (corrupt file)")
    var nbytes = (count * width + 7) // 8
    var acc: UInt64 = 0
    var have = 0
    var p = c.pos
    var mask = (UInt64(1) << UInt64(width)) - 1 if width < 64 else ~UInt64(0)
    for _ in range(count):
        while have < width:
            acc |= UInt64(c.at(p)) << UInt64(have)
            p += 1
            have += 8
            if have > 56 and have < width:      # widths past 56 bits: take the rest in a second word
                break
        if have >= width:
            out.append(Int64(acc & mask))
            acc = acc >> UInt64(width) if width < 64 else 0
            have -= width
        else:
            var lo = acc
            var got = have
            acc = 0
            have = 0
            while got + have < width:
                acc |= UInt64(c.at(p)) << UInt64(have)
                p += 1
                have += 8
            var need = width - got
            out.append(Int64(lo | ((acc & ((UInt64(1) << UInt64(need)) - 1)) << UInt64(got))))
            acc >>= UInt64(need)
            have -= need
    c.pos += nbytes


def _rle_hybrid(mut c: _Bytes, width: Int, count: Int, mut out: List[Int64]) raises:
    """The RLE / bit-packing hybrid: runs of one repeated value and groups of 8 bit-packed values. Exactly `count`
    values are produced: a run longer than what is still needed is cut, never built."""
    if width < 0 or width > 64:
        raise Error("a Parquet page has a bit width of " + String(width) + " (corrupt file)")
    var start = len(out)
    while len(out) - start < count:
        var h = c.small()
        var left = count - (len(out) - start)
        if h & 1 == 1:
            var groups = h >> 1
            _unpack(c, width, left if groups >= (left + 7) // 8 else groups * 8, out)
        else:
            var v = c.le((width + 7) // 8) if width > 0 else 0
            if width > 0 and width < 64:
                v &= Int64((UInt64(1) << UInt64(width)) - 1)
            for _ in range(min(h >> 1, left)):
                out.append(v)


def _wrap32(v: Int64) -> Int64:
    """An INT32 column's value: the low 32 bits, sign-extended (its deltas wrap modulo 2^32)."""
    var low = v & 0xFFFFFFFF
    return low - (Int64(1) << 32) if low >= (Int64(1) << 31) else low


def _delta_binary_packed(mut c: _Bytes, int32: Bool, limit: Int, mut out: List[Int64]) raises -> Int:
    """DELTA_BINARY_PACKED integers (at most `limit`); returns how many were read. INT32 arithmetic wraps modulo 2^32, as the writer's."""
    var block = c.small()
    var minis = c.small()
    var total = c.small()
    var prev = c.zigzag()
    if total > limit:
        raise Error("a Parquet page holds more delta values than its count says (corrupt file)")
    if total == 0:
        return 0
    if block <= 0 or minis <= 0 or block % minis != 0:
        raise Error("a Parquet delta block has a bad shape (corrupt file)")
    out.append(prev)
    var got = 1
    var per_mini = block // minis
    while got < total:
        var min_delta = c.zigzag()
        var widths = List[Int]()
        for _ in range(minis):
            widths.append(c.byte())
        for m in range(minis):
            if got >= total:
                break                                   # trailing miniblocks of the last block may be absent
            var deltas = List[Int64]()
            _unpack(c, widths[m], min(per_mini, total - got), deltas)
            for d in deltas:
                prev = prev + min_delta + d
                if int32:
                    prev = _wrap32(prev)
                out.append(prev)
                got += 1
    return total


def _ptype_name(p: Int) -> String:
    var names: List[String] = ["BOOLEAN", "INT32", "INT64", "INT96", "FLOAT", "DOUBLE", "BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY"]
    return names[p] if 0 <= p < len(names) else "physical type " + String(p)


def _is_unsigned(leaf: _Leaf) -> Bool:
    return leaf.unsigned or (CT_UINT_8 <= leaf.converted <= CT_UINT_64)


def _is_timestamp(leaf: _Leaf) -> Bool:
    return leaf.ptype == PT_INT64 and (leaf.logical == 8 or leaf.converted == CT_TIMESTAMP_MILLIS or
                                       leaf.converted == CT_TIMESTAMP_MICROS)


def _kind_of(leaf: _Leaf) raises -> Tuple[ColumnKind, Int]:
    """The MX frame kind (and scale) a Parquet leaf column becomes."""
    if leaf.ptype < 0 or leaf.ptype > PT_FIXED:
        raise Error("column `" + leaf.name + "` has an unknown physical type (" + String(leaf.ptype) + ", corrupt file)")
    if leaf.logical == 7 or leaf.converted == CT_TIME_MILLIS or leaf.converted == CT_TIME_MICROS:
        raise Error("column `" + leaf.name + "` is a TIME: MX frame has no kind for it yet")
    var decimal = leaf.converted == CT_DECIMAL or leaf.logical == 5
    if decimal:
        if leaf.precision > 18:
            raise Error("column `" + leaf.name + "`: DECIMAL(" + String(leaf.precision) + ", " + String(leaf.scale) +
                        ") has more digits than an int64 holds (18)")
        if leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64 or leaf.ptype == PT_FIXED:
            return (DECIMAL, leaf.scale)
        if leaf.ptype == PT_BYTE_ARRAY:
            raise Error("column `" + leaf.name + "` is a DECIMAL stored as BYTE_ARRAY: MX frame reads DECIMAL on INT32, INT64 and FIXED_LEN_BYTE_ARRAY only")
    if leaf.ptype == PT_BOOLEAN:
        return (BOOL, 0)
    if leaf.ptype == PT_BYTE_ARRAY:
        if leaf.logical == 1 or leaf.logical == 4 or leaf.logical == 12 or leaf.converted == CT_UTF8 or \
           leaf.converted == CT_ENUM or leaf.converted == CT_JSON:
            return (TEXT, 0)
        raise Error("column `" + leaf.name + "` is BINARY (BYTE_ARRAY without a string type): MX frame has no kind for it yet")
    if leaf.converted == CT_DATE or leaf.logical == 6:
        return (DATE, 0)
    if _is_timestamp(leaf):
        return (DATE, 0)                                # a timestamp becomes its UTC day (the CSV reader's rule)
    if leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64:
        return (INT, 0)
    raise Error("column `" + leaf.name + "` is " + _ptype_name(leaf.ptype) + ": MX frame has no kind for it yet")


def _to_stored(leaf: _Leaf, raw: Int64) raises -> Int64:
    """A plain value of the leaf's physical type as MX frame stores it (timestamps become days, floored; unsigned
    integers keep their value: UINT_32 is zero-extended, UINT_64 past the int64 range is refused)."""
    if _is_timestamp(leaf):
        var per_day = Int64(86_400_000) if (leaf.ts_unit == 1 or leaf.converted == CT_TIMESTAMP_MILLIS) else \
                      Int64(86_400_000_000_000) if leaf.ts_unit == 3 else Int64(86_400_000_000)
        return raw // per_day
    if _is_unsigned(leaf) and leaf.logical != 5 and leaf.converted != CT_DECIMAL:
        if leaf.ptype == PT_INT32:
            return raw & 0xFFFFFFFF
        if raw < 0:
            raise Error("column `" + leaf.name + "` is UINT_64 and holds " + String(UInt64(raw)) +
                        ", past the int64 range MX frame stores")
    return raw


def read_parquet(path: String) raises -> ParquetResult:
    """Reads a Parquet file (written by MX frame, pyarrow, DuckDB, ...) into a Table, plus its "omx" metadata."""
    var file = _open_file(path)
    var n = _file_size(file, path)
    if n < 12:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var map = _Mapped(file._get_raw_fd(), 0, n, path)
    var data = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=map.data)
    if data[0] != 0x50 or data[1] != 0x41 or data[2] != 0x52 or data[3] != 0x31 or \
       data[n - 4] != 0x50 or data[n - 3] != 0x41 or data[n - 2] != 0x52 or data[n - 1] != 0x31:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var flen = Int(data[n - 8]) | (Int(data[n - 7]) << 8) | (Int(data[n - 6]) << 16) | (Int(data[n - 5]) << 24)
    if flen <= 0 or flen > n - 12:
        raise Error("`" + path + "`: the Parquet footer length is wrong (corrupt file)")
    var base = map.data
    var r = ThriftReader(base + n - 8 - flen, flen)

    var footer = _read_footer(r)
    ref leaves = footer.leaves
    ref groups = footer.groups

    var cols = List[_Leaf]()
    var t = _schema_table(path, leaves, cols)
    var ids = List[Int]()
    var bases = List[Int]()
    var ns = List[Int]()
    var widx = List[_WordIndex]()
    var widxs = List[Int]()
    var total = 0
    for c in range(len(cols)):
        ids.append(c)
        bases.append(base)
        ns.append(n)
        widx.append(_WordIndex())
        widx[c].col = Int(t.cols.unsafe_ptr() + c)
    for c in range(len(cols)):
        widxs.append(Int(widx.unsafe_ptr() + c))
    for g in range(len(groups)):
        if len(cols) > 0 and len(groups[g]) == len(cols):
            var n_g = groups[g][0].values
            _check_claim("`" + path + "` row group " + String(g), n_g, len(cols))
            total += n_g                                        # each part checked, so the sum cannot wrap
            _check_claim("`" + path + "`", total, len(cols))
    for c in range(len(cols)):                                  # size every column once; the pages write every row
        t.cols[c].values = List[Int64](unsafe_uninit_length=total)
    var row0 = 0
    for g in range(len(groups)):
        if len(groups[g]) != len(cols):
            raise Error("`" + path + "`: row group " + String(g) + " has " + String(len(groups[g])) +
                        " columns, the schema " + String(len(cols)))
        _decode_group(bases, ns, groups[g], ids, ids, widxs, True, cols, row0, t)
        if len(cols) > 0:
            row0 += groups[g][0].values
    for c in range(len(cols)):                                  # (a presized column holds exactly the rows read)
        if len(t.cols[c].values) > row0:
            t.cols[c].values.shrink(row0)
        if len(t.cols[c].validity) > (row0 + 7) // 8:
            t.cols[c].validity.shrink((row0 + 7) // 8)
    t.check()
    map.release()
    _ = widx^                                                   # the word indexes are read by address above
    return ParquetResult(t^, footer.omx.copy())


def _schema_table(path: String, leaves: List[_Leaf], mut cols: List[_Leaf]) raises -> Table:
    """An empty table with the file's columns (a flat schema: a root, then one element per column); `cols` gets
    their leaves."""
    if len(leaves) == 0:
        raise Error("`" + path + "`: the Parquet schema is empty")
    var t = Table.empty()
    for i in range(1, len(leaves)):
        if leaves[i].children > 0 or leaves[i].repetition == REPEATED:
            raise Error("column `" + leaves[i].name + "` is nested (a list or a struct): not supported yet")
        var k = _kind_of(leaves[i])
        _ = t.add_column(leaves[i].name, k[0], scale=k[1])
        cols.append(leaves[i].copy())
    return t^


@fieldwise_init
struct _Footer(Movable):
    var leaves: List[_Leaf]
    var groups: List[List[_ChunkMeta]]
    var omx: String


def _read_footer(mut r: ThriftReader) raises -> _Footer:
    """The schema, the row groups' chunk metadata and the "omx" metadata from a Parquet footer."""
    var leaves = List[_Leaf]()
    var groups = List[List[_ChunkMeta]]()
    var omx = String("")
    while True:
        var f = r.field()
        if f[1] == T_STOP:
            break
        if f[0] == 2 and f[1] == T_LIST:                # schema
            var h = r.list_header()
            for _ in range(h[1]):
                leaves.append(_read_schema_element(r))
        elif f[0] == 4 and f[1] == T_LIST:              # row groups
            var h = r.list_header()
            for _ in range(h[1]):
                groups.append(_read_row_group(r))
        elif f[0] == 5 and f[1] == T_LIST:              # key-value metadata
            var h = r.list_header()
            for _ in range(h[1]):
                r.enter()
                var key = String("")
                var value = String("")
                while True:
                    var g = r.field()
                    if g[1] == T_STOP:
                        break
                    if g[0] == 1 and g[1] == T_BINARY:
                        key = r.binary()
                    elif g[0] == 2 and g[1] == T_BINARY:
                        value = r.binary()
                    else:
                        r.skip(g[1])
                r.leave()
                if key == "omx":
                    omx = value
        else:
            r.skip(f[1])

    return _Footer(leaves^, groups^, omx^)


def _read_schema_element(mut r: ThriftReader) raises -> _Leaf:
    var leaf = _Leaf("", -1, 0, REQUIRED, -1, 0, 0, -1, 0, 0, False)
    r.enter()
    while True:
        var f = r.field()
        if f[1] == T_STOP:
            break
        if f[0] == 1 and f[1] == T_I32:
            leaf.ptype = Int(r.zigzag())
        elif f[0] == 2 and f[1] == T_I32:
            leaf.type_length = Int(r.zigzag())
        elif f[0] == 3 and f[1] == T_I32:
            leaf.repetition = Int(r.zigzag())
        elif f[0] == 4 and f[1] == T_BINARY:
            leaf.name = r.binary()
        elif f[0] == 5 and f[1] == T_I32:
            leaf.children = Int(r.zigzag())
        elif f[0] == 6 and f[1] == T_I32:
            leaf.converted = Int(r.zigzag())
        elif f[0] == 7 and f[1] == T_I32:
            leaf.scale = Int(r.zigzag())
        elif f[0] == 8 and f[1] == T_I32:
            leaf.precision = Int(r.zigzag())
        elif f[0] == 10 and f[1] == T_STRUCT:            # LogicalType: a union, one field set
            r.enter()
            while True:
                var u = r.field()
                if u[1] == T_STOP:
                    break
                leaf.logical = u[0]
                if u[0] == 5 and u[1] == T_STRUCT:       # DECIMAL(scale, precision)
                    r.enter()
                    while True:
                        var d = r.field()
                        if d[1] == T_STOP:
                            break
                        if d[0] == 1 and d[1] == T_I32:
                            leaf.scale = Int(r.zigzag())
                        elif d[0] == 2 and d[1] == T_I32:
                            leaf.precision = Int(r.zigzag())
                        else:
                            r.skip(d[1])
                    r.leave()
                elif u[0] == 10 and u[1] == T_STRUCT:    # INTEGER(bitWidth, isSigned)
                    r.enter()
                    while True:
                        var d = r.field()
                        if d[1] == T_STOP:
                            break
                        if d[0] == 2 and (d[1] == T_TRUE or d[1] == T_FALSE):
                            leaf.unsigned = d[1] == T_FALSE
                        else:
                            r.skip(d[1])
                    r.leave()
                elif u[0] == 8 and u[1] == T_STRUCT:     # TIMESTAMP(isAdjustedToUTC, unit)
                    r.enter()
                    while True:
                        var d = r.field()
                        if d[1] == T_STOP:
                            break
                        if d[0] == 2 and d[1] == T_STRUCT:   # TimeUnit: MILLIS 1, MICROS 2, NANOS 3
                            r.enter()
                            while True:
                                var tu = r.field()
                                if tu[1] == T_STOP:
                                    break
                                leaf.ts_unit = tu[0]
                                r.skip(tu[1])
                            r.leave()
                        else:
                            r.skip(d[1])
                    r.leave()
                else:
                    r.skip(u[1])
            r.leave()
        else:
            r.skip(f[1])
    r.leave()
    return leaf^


def _read_row_group(mut r: ThriftReader) raises -> List[_ChunkMeta]:
    var chunks = List[_ChunkMeta]()
    var group_rows = -1
    r.enter()
    while True:
        var f = r.field()
        if f[1] == T_STOP:
            break
        if f[0] == 1 and f[1] == T_LIST:
            var h = r.list_header()
            for _ in range(h[1]):
                var meta = _ChunkMeta(-1, 0, 0, -1, -1, 0, -1)
                r.enter()
                while True:
                    var g = r.field()
                    if g[1] == T_STOP:
                        break
                    if g[0] == 1 and g[1] == T_BINARY:
                        raise Error("a Parquet column chunk lives in another file: not supported")
                    if g[0] == 3 and g[1] == T_STRUCT:   # ColumnMetaData
                        r.enter()
                        while True:
                            var m = r.field()
                            if m[1] == T_STOP:
                                break
                            if m[0] == 1 and m[1] == T_I32:
                                meta.ptype = Int(r.zigzag())
                            elif m[0] == 4 and m[1] == T_I32:
                                meta.codec = Int(r.zigzag())
                            elif m[0] == 5 and m[1] == T_I64:
                                meta.values = Int(r.zigzag())
                            elif m[0] == 7 and m[1] == T_I64:
                                meta.size = Int(r.zigzag())
                            elif m[0] == 9 and m[1] == T_I64:
                                meta.data_off = Int(r.zigzag())
                            elif m[0] == 11 and m[1] == T_I64:
                                meta.dict_off = Int(r.zigzag())
                            else:
                                r.skip(m[1])
                        r.leave()
                    else:
                        r.skip(g[1])
                r.leave()
                chunks.append(meta^)
        elif f[0] == 3 and f[1] == T_I64:
            group_rows = Int(r.zigzag())
        else:
            r.skip(f[1])
    for i in range(len(chunks)):
        chunks[i].group_rows = group_rows
    r.leave()
    return chunks^


# ── decoding a row group: pages in bulk, in parallel (F-141) ─────────────────────────────────────────
#
# A row group's chunks (one per column read) are decoded in three steps:
#   1. walk: the page headers of every chunk, serially (cheap). A dictionary page is decoded here (its words become
#      the file-wide codes, or are added to the table's own dictionary for `read_parquet`), a data page becomes a
#      `_Page` job that knows its first row. The columns are then sized once, from the rows the pages add up to.
#   2. TEXT pages that must ADD words to a table (PLAIN text under `read_parquet`) are decoded now, in file order,
#      because adding words mutates the table.
#   3. every other page is decoded in parallel straight into its slice of the column: decompress, unpack the
#      definition levels into the page's validity bits, write the values (a PLAIN int64 is a copy, an int32 one
#      widening loop, dictionary indices are gathered into codes in the same loop that unpacks them), spread the
#      values over the null rows. The pages' validity bits are then merged into the column's bitmap in page order.
# Nothing in the parallel step mutates shared state: a page only writes its own rows and its own `_Page` record.

@fieldwise_init
struct _Page(Copyable, Movable):
    var addr: Int                    # the page body (compressed) in the file or its mapping
    var csize: Int
    var usize: Int
    var nvals: Int
    var encoding: Int
    var v2: Bool
    var def_len: Int
    var def_enc: Int
    var v2_compressed: Bool
    var ctx: Int                     # its chunk's `_Ctx`
    var row: Int                     # its first row in the output column
    var defer: Bool                  # PLAIN text that may add words: misses are recorded, and coded in file order afterwards
    var nulls: Int
    var bits: List[UInt8]            # validity bits of its rows (an OPTIONAL column with nulls on this page)
    var err: String
    var miss_at: List[Int]           # defer: the words the frozen index did not have (address, length, hash) in page order
    var miss_n: List[Int]
    var miss_h: List[UInt64]
    var miss_code: List[Int]         # their codes, given in file order by the serial phase
    var miss_slots: List[Int32]      # a table over the misses (one entry per distinct word of the page)
    var miss_canon: List[Int]        # (big columns) per miss: page << 32 | miss number of the first page that missed the same word
    var keep: List[UInt8]            # the decompressed page, kept while its missed words are read


@fieldwise_init
struct _Ctx(Copyable, Movable):
    var leaf: Int                    # index into the leaves
    var col: Int                     # the output column
    var codec: Int
    var optional: Bool
    var widx: Int                    # address of the TEXT column's `_WordIndex` (0 for other kinds)
    var add: Bool                    # new words are added to the table (else a word that is missing is a corrupt file)
    var dict: List[Int64]            # the chunk's dictionary page as stored values (file-wide codes for TEXT)
    var row0: Int                    # the chunk's first row in the output column
    var dict_seen: Bool              # its dictionary page was read (a second one is a corrupt file)


struct _WordIndex(Movable):
    """A TEXT column's words by their bytes: an open-addressing table over the words of the table's `Column` (found by
    its address), so a value that is already a word is found without making a String. Reads are safe from many
    threads; `insert` runs on the calling thread only."""
    var col: Int                     # address of the table's TEXT Column (its `words` hold the bytes)
    var slots: List[Int32]           # -1 empty, else a code
    var hashes: List[UInt64]         # by code
    var mask: Int

    def __init__(out self):
        self.col = 0
        self.slots = List[Int32](length=1024, fill=-1)
        self.hashes = List[UInt64]()
        self.mask = 1023

    def find(self, p: UnsafePointer[UInt8, MutAnyOrigin], n: Int, h: UInt64) -> Int:
        var words = UnsafePointer[Column, MutAnyOrigin](unsafe_from_address=self.col)[].words.unsafe_ptr()
        var i = Int(h) & self.mask
        while True:
            var code = Int(self.slots[i])
            if code < 0:
                return -1
            if self.hashes[code] == h and words[code].byte_length() == n:
                var q = words[code].unsafe_ptr()
                var same = True
                for k in range(n):
                    if q[k] != p[k]:
                        same = False
                        break
                if same:
                    return code
            i = (i + 1) & self.mask

    def insert(mut self, h: UInt64, code: Int):
        if code >= len(self.hashes):
            self.hashes.append(h)
        if (len(self.hashes) + 1) * 2 > len(self.slots):         # load factor 1/2: double and re-place every code
            var size = len(self.slots) * 2
            self.slots = List[Int32](length=size, fill=-1)
            self.mask = size - 1
            for c in range(len(self.hashes)):
                var j = Int(self.hashes[c]) & self.mask
                while self.slots[j] >= 0:
                    j = (j + 1) & self.mask
                self.slots[j] = Int32(c)
            return
        var j = Int(h) & self.mask
        while self.slots[j] >= 0:
            j = (j + 1) & self.mask
        self.slots[j] = Int32(code)


def _hash_bytes(p: UnsafePointer[UInt8, MutAnyOrigin], n: Int) -> UInt64:
    var h = UInt64(0x9E3779B97F4A7C15) ^ UInt64(n)
    var i = 0
    while i + 8 <= n:
        h = (h ^ (p + i).bitcast[UInt64]().load[alignment=1]()) * UInt64(0xFF51AFD7ED558CCD)
        h ^= h >> 32
        i += 8
    var tail: UInt64 = 0
    var k = 0
    while i < n:
        tail |= UInt64(p[i]) << UInt64(8 * k)
        i += 1
        k += 1
    h = (h ^ tail) * UInt64(0xC4CEB9FE1A85EC53)
    h ^= h >> 29
    return h


def _get_bit(p: UnsafePointer[UInt8, MutAnyOrigin], i: Int) -> Bool:
    return (p[i >> 3] >> UInt8(i & 7)) & 1 == 1


def _assign_bits(dst: UnsafePointer[UInt8, MutAnyOrigin], pos: Int, src: UnsafePointer[UInt8, MutAnyOrigin], nbits: Int):
    """dst bits [pos, pos + nbits) = src bits [0, nbits) (least significant bit first, both)."""
    var i = 0
    while i < nbits and (pos + i) & 7 != 0:
        var m = UInt8(1) << UInt8((pos + i) & 7)
        if _get_bit(src, i):
            dst[(pos + i) >> 3] |= m
        else:
            dst[(pos + i) >> 3] &= ~m
        i += 1
    while i + 8 <= nbits:
        var sb = i >> 3
        var sh = i & 7
        var v = src[sb]
        if sh != 0:
            v = UInt8(((UInt16(src[sb]) | (UInt16(src[sb + 1]) << 8)) >> UInt16(sh)) & 0xFF)
        dst[(pos + i) >> 3] = v
        i += 8
    while i < nbits:
        var m = UInt8(1) << UInt8((pos + i) & 7)
        if _get_bit(src, i):
            dst[(pos + i) >> 3] |= m
        else:
            dst[(pos + i) >> 3] &= ~m
        i += 1


def _set_ones(dst: UnsafePointer[UInt8, MutAnyOrigin], pos: Int, count: Int):
    var i = 0
    while i < count and (pos + i) & 7 != 0:
        dst[(pos + i) >> 3] |= UInt8(1) << UInt8((pos + i) & 7)
        i += 1
    while i + 8 <= count:
        dst[(pos + i) >> 3] = 0xFF
        i += 8
    while i < count:
        dst[(pos + i) >> 3] |= UInt8(1) << UInt8((pos + i) & 7)
        i += 1


def _count_bits(p: UnsafePointer[UInt8, MutAnyOrigin], nbits: Int) -> Int:
    var n = 0
    var full = nbits >> 3
    for i in range(full):
        n += Int(pop_count(p[i]))
    if nbits & 7 != 0:
        n += Int(pop_count(p[full] & ((UInt8(1) << UInt8(nbits & 7)) - 1)))
    return n


def _levels_to_bits(mut c: _Bytes, count: Int, bits: UnsafePointer[UInt8, MutAnyOrigin]) raises:
    """`count` definition levels (RLE / bit-packed hybrid, width 1) into `bits`, already zero: bit i set = value i is
    present. A bit-packed run of 1-bit values is a run of bitmap bits; a repeated 1 sets a range."""
    var pos = 0
    while pos < count:
        var h = c.small()
        var left = count - pos
        if h & 1 == 1:
            var groups = h >> 1
            var take = left if groups >= (left + 7) // 8 else groups * 8
            var nbytes = (take + 7) // 8
            c.need(nbytes)
            _assign_bits(bits, pos, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos), take)
            c.pos += nbytes
            pos += take
        else:
            var v = c.le(1) & 1
            var run = min(h >> 1, left)
            if v == 1:
                _set_ones(bits, pos, run)
            pos += run


def _page_bytes(codec: Int, src: Int, n: Int, size: Int, mut owner: List[UInt8]) raises -> Int:
    """The address of `size` uncompressed bytes: the page itself when it is not compressed, else a decompressed copy
    held in `owner`."""
    if codec == UNCOMPRESSED:
        if n != size or n < 0:
            raise Error("an uncompressed Parquet page has two different sizes (corrupt file)")
        return src
    owner = _decompress(codec, src, n, size)
    return Int(owner.unsafe_ptr())


def _identity(leaf: _Leaf) -> Bool:
    """A plain integer of this leaf is stored as it is (no timestamp-to-day, no unsigned handling)."""
    return not _is_timestamp(leaf) and not (_is_unsigned(leaf) and leaf.logical != 5 and leaf.converted != CT_DECIMAL)


def _note_miss(mut page: _Page, wp: UnsafePointer[UInt8, MutAnyOrigin], n: Int, h: UInt64) -> Int:
    """A word the frozen index lacks, seen by a page decoded in parallel: one entry per distinct word of the page, in
    first-seen order. Returns the placeholder -1 - (its miss number) for the value."""
    if len(page.miss_slots) == 0:
        page.miss_slots = List[Int32](length=1024, fill=-1)
    var mask = len(page.miss_slots) - 1
    var j = Int(h) & mask
    while True:
        var m = Int(page.miss_slots[j])
        if m < 0:
            break
        if page.miss_h[m] == h and page.miss_n[m] == n:
            var q = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=page.miss_at[m])
            var same = True
            for k in range(n):
                if q[k] != wp[k]:
                    same = False
                    break
            if same:
                return -1 - m
        j = (j + 1) & mask
    var m = len(page.miss_at)
    page.miss_at.append(Int(wp))
    page.miss_n.append(n)
    page.miss_h.append(h)
    if (m + 1) * 2 > len(page.miss_slots):               # grow at half full and re-place every miss
        var size = len(page.miss_slots) * 2
        page.miss_slots = List[Int32](length=size, fill=-1)
        mask = size - 1
        for i in range(m + 1):
            var k = Int(page.miss_h[i]) & mask
            while page.miss_slots[k] >= 0:
                k = (k + 1) & mask
            page.miss_slots[k] = Int32(i)
    else:
        page.miss_slots[j] = Int32(m)
    return -1 - m


def _plain_into(mut c: _Bytes, leaf: _Leaf, ctx: _Ctx, count: Int, dout: UnsafePointer[Int64, MutAnyOrigin], tbl: Int,
                store: Bool, pg: Int = 0) raises:
    """`count` PLAIN values to dout[0..count). `store` False (a dictionary build): TEXT words are added, nothing written."""
    if count < 0:
        raise Error("column `" + leaf.name + "`: a page has a negative value count (corrupt file)")
    if leaf.ptype == PT_BOOLEAN:
        c.need((count + 7) // 8)
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
        for i in range(count):
            dout[i] = Int64((p[i >> 3] >> UInt8(i & 7)) & 1)
        c.pos += (count + 7) // 8
    elif leaf.ptype == PT_INT32:
        if count > (c.n - c.pos) // 4:
            raise Error("a Parquet page ends early (corrupt file)")
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos).bitcast[Int32]()
        if _identity(leaf):
            var i = 0
            while i + 4 <= count:
                dout.store[width=4, alignment=1](i, (p + i).load[width=4, alignment=1]().cast[DType.int64]())
                i += 4
            while i < count:
                dout[i] = Int64((p + i).load[alignment=1]())
                i += 1
        else:
            for i in range(count):
                dout[i] = _to_stored(leaf, Int64((p + i).load[alignment=1]()))
        c.pos += count * 4
    elif leaf.ptype == PT_INT64:
        if count > (c.n - c.pos) // 8:
            raise Error("a Parquet page ends early (corrupt file)")
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos).bitcast[Int64]()
        if _identity(leaf):
            var i = 0
            while i + 4 <= count:
                dout.store[width=4, alignment=1](i, (p + i).load[width=4, alignment=1]())
                i += 4
            while i < count:
                dout[i] = (p + i).load[alignment=1]()
                i += 1
        else:
            for i in range(count):
                dout[i] = _to_stored(leaf, (p + i).load[alignment=1]())
        c.pos += count * 8
    elif leaf.ptype == PT_BYTE_ARRAY:
        for i in range(count):
            var n = Int(c.le(4)) & 0xFFFFFFFF
            c.need(n)
            if not _valid_utf8(c.base + c.pos, n):
                raise Error("column `" + leaf.name + "`: a text value is not valid UTF-8")
            var wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
            c.pos += n
            var h = _hash_bytes(wp, n)
            var wi = UnsafePointer[_WordIndex, MutAnyOrigin](unsafe_from_address=ctx.widx)
            var code = wi[].find(wp, n, h)
            if code < 0:
                if not ctx.add:
                    raise Error("column `" + leaf.name + "`: a word is missing from the file's dictionary (corrupt file)")
                if pg != 0:                              # a page decoded in parallel: the index is frozen, note the miss;
                    ref page = UnsafePointer[_Page, MutAnyOrigin](unsafe_from_address=pg)[]
                    code = _note_miss(page, wp, n, h)    # its code comes later, in file order (-1 - miss number now)
                else:
                    var s = String(unsafe_from_utf8=Span[UInt8, MutAnyOrigin](unsafe_ptr=wp, length=n))
                    code = UnsafePointer[Table, MutAnyOrigin](unsafe_from_address=tbl)[].add_word(ctx.col, s)
                    wi[].insert(h, code)
            if store:
                dout[i] = Int64(code)
    elif leaf.ptype == PT_FIXED:                        # a big-endian two's-complement decimal
        var w = leaf.type_length
        if w < 1 or w > 16:
            raise Error("column `" + leaf.name + "` has a fixed length of " + String(w) + " bytes (corrupt file)")
        if count > (c.n - c.pos) // w:
            raise Error("a Parquet page ends early (corrupt file)")
        if w >= 8:                                       # only the last 8 bytes survive an int64: one byte swap each
            var q = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos + (w - 8))
            for i in range(count):
                dout[i] = byte_swap((q + i * w).bitcast[Int64]().load[alignment=1]())
            c.pos += count * w
        else:
            for i in range(count):
                var v: Int64 = -1 if c.at(c.pos) >= 0x80 else 0
                for k in range(w):
                    v = (v << 8) | Int64(c.at(c.pos + k))
                c.pos += w
                dout[i] = v
    else:
        raise Error("column `" + leaf.name + "`: its physical type cannot be read yet")


def _dict_into(mut c: _Bytes, leaf: _Leaf, dict: List[Int64], count: Int, dout: UnsafePointer[Int64, MutAnyOrigin]) raises:
    """`count` dictionary indices (a bit width, then the RLE / bit-packed hybrid) gathered through `dict` into dout."""
    var w = c.byte()
    if w > 64:
        raise Error("a Parquet page has a bit width of " + String(w) + " (corrupt file)")
    var d = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(dict.unsafe_ptr()))
    var dn = len(dict)
    var got = 0
    while got < count:
        var h = c.small()
        var left = count - got
        if h & 1 == 1:
            var groups = h >> 1
            var take = left if groups >= (left + 7) // 8 else groups * 8
            if w == 0:
                if dn < 1:
                    raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
                for i in range(take):
                    dout[got + i] = d[0]
            elif w <= 32:
                if take > ((c.n - c.pos) * 8) // w:
                    raise Error("a Parquet page ends early (corrupt file)")
                var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
                var mask = (UInt64(1) << UInt64(w)) - 1
                var acc: UInt64 = 0
                var have = 0
                var at = 0
                for i in range(take):
                    while have < w:
                        acc |= UInt64(p[at]) << UInt64(have)
                        at += 1
                        have += 8
                    var ix = Int(acc & mask)
                    acc >>= UInt64(w)
                    have -= w
                    if ix >= dn:
                        raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
                    dout[got + i] = d[ix]
                c.pos += (take * w + 7) // 8
            else:
                var idx = List[Int64]()
                _unpack(c, w, take, idx)
                for i in range(take):
                    if idx[i] < 0 or Int(idx[i]) >= dn:
                        raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
                    dout[got + i] = d[Int(idx[i])]
            got += take
        else:
            var v = c.le((w + 7) // 8) if w > 0 else 0
            if w > 0 and w < 64:
                v &= Int64((UInt64(1) << UInt64(w)) - 1)
            if v < 0 or Int(v) >= dn:
                raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
            var run = min(h >> 1, left)
            var x = d[Int(v)]
            for i in range(run):
                dout[got + i] = x
            got += run


def _decode_page(mut p: _Page, ref ctx: _Ctx, leaf: _Leaf, dst: Int, tbl: Int) raises:
    """Decodes one data page into the column slice at `dst` (0: only add the words of a TEXT page, store nothing)."""
    var owner = List[UInt8]()
    var c: _Bytes
    var present = p.nvals
    var nobits = List[UInt8](length=8, fill=0)
    var bits_p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(nobits.unsafe_ptr()))
    if ctx.optional:
        p.bits = List[UInt8](length=(p.nvals + 7) // 8 + 1, fill=0)
        bits_p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr()))
    if p.v2:
        var lv = p.def_len
        if lv < 0 or lv > p.csize or lv > p.usize:
            raise Error("column `" + leaf.name + "`: the definition levels are longer than their page (corrupt file)")
        if ctx.optional:
            var lc = _Bytes(p.addr, lv)
            _levels_to_bits(lc, p.nvals, bits_p)
        var at = _page_bytes(ctx.codec if p.v2_compressed else UNCOMPRESSED, p.addr + lv, p.csize - lv, p.usize - lv, owner)
        c = _Bytes(at, p.usize - lv)
    else:
        var at = _page_bytes(ctx.codec, p.addr, p.csize, p.usize, owner)
        c = _Bytes(at, p.usize)
        if ctx.optional:
            if p.def_enc == ENC_BIT_PACKED:                      # deprecated: most significant bit first, no length
                var nb = (p.nvals + 7) // 8
                c.need(nb)
                for i in range(p.nvals):
                    if (c.at(c.pos + i // 8) >> (7 - i % 8)) & 1 == 1:
                        bits_p[i >> 3] |= UInt8(1) << UInt8(i & 7)
                c.pos += nb
            elif p.def_enc == ENC_RLE:
                var ln = Int(c.le(4)) & 0xFFFFFFFF
                c.need(ln)
                var lc = _Bytes(c.base + c.pos, ln)
                _levels_to_bits(lc, p.nvals, bits_p)
                c.pos += ln
            else:
                raise Error("column `" + leaf.name + "`: definition levels use encoding " + String(p.def_enc) + ", not supported")
    if ctx.optional:
        present = _count_bits(bits_p, p.nvals)
    var store = dst != 0
    if not store and not (p.encoding == ENC_PLAIN and leaf.ptype == PT_BYTE_ARRAY):
        # a words-only build (dst 0) only adds the words of PLAIN text; any other page would write through a null
        # pointer, so it is refused here, before a decoder can reach one
        raise Error("column `" + leaf.name + "`: only PLAIN text pages are read for their words (internal error)")
    var dout = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=dst)
    if p.encoding == ENC_PLAIN:
        _plain_into(c, leaf, ctx, present, dout, tbl, store, Int(UnsafePointer(to=p)) if p.defer else 0)
    elif p.encoding == ENC_RLE_DICTIONARY or p.encoding == ENC_PLAIN_DICTIONARY:
        _dict_into(c, leaf, ctx.dict, present, dout)
    elif p.encoding == ENC_DELTA_BINARY_PACKED and (leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64):
        var raw = List[Int64]()
        _ = _delta_binary_packed(c, leaf.ptype == PT_INT32, present, raw)
        if len(raw) < present:
            raise Error("column `" + leaf.name + "`: a page holds fewer values than its levels say (corrupt file)")
        for i in range(present):
            dout[i] = _to_stored(leaf, raw[i])
    elif p.encoding == ENC_RLE and leaf.ptype == PT_BOOLEAN:
        _ = c.le(4)                                       # a length prefix
        var vals = List[Int64]()
        _rle_hybrid(c, 1, present, vals)
        if len(vals) < present:
            raise Error("column `" + leaf.name + "`: a page holds fewer values than its levels say (corrupt file)")
        for i in range(present):
            dout[i] = vals[i]
    else:
        raise Error("column `" + leaf.name + "`: Parquet encoding " + String(p.encoding) + " is not supported yet")
    p.nulls = p.nvals - present
    if p.nulls > 0 and store:                             # spread the values over the rows, backwards
        var k = present - 1
        for row in range(p.nvals - 1, -1, -1):
            if _get_bit(bits_p, row):
                dout[row] = dout[k]
                k -= 1
            else:
                dout[row] = 0
    elif p.nulls == 0:
        p.bits = List[UInt8]()
    if len(p.miss_at) > 0:
        p.keep = owner^                               # the missed words are read from it by the serial phase
    else:
        _ = owner^
    _ = nobits^


def _walk_chunk(base: Int, n: Int, meta: _ChunkMeta, leaf: _Leaf, mut ctx: _Ctx, ctx_at: Int, tbl: Int,
                mut pages: List[_Page], words_only: Bool) raises:
    """Reads a chunk's page headers (`base` + file offset = the page's address; `n` bounds the offsets). A dictionary
    page is decoded now; each data page is appended to `pages`."""
    var pos = meta.dict_off if meta.dict_off > 0 and meta.dict_off < meta.data_off else meta.data_off
    var done = 0
    comptime if NO_CODECS:
        if meta.codec == ZSTD or meta.codec == SNAPPY:
            raise Error("column `" + leaf.name + "` is " + ("zstd" if meta.codec == ZSTD else "snappy") +
                        "-compressed, and this build has no zstd or snappy (save the file UNCOMPRESSED, or open it on a build with codecs)")
    if meta.values < 0:
        raise Error("column `" + leaf.name + "`: a column chunk has a negative value count (corrupt file)")
    while done < meta.values:
        if pos <= 0 or pos >= n:
            raise Error("column `" + leaf.name + "`: a page offset is outside the file (corrupt file)")
        var h = ThriftReader(base + pos, n - pos)
        var ptype = -1
        var usize = 0
        var csize = 0
        var nvals = 0
        var encoding = 0
        var def_len = 0                  # v2: definition levels' byte length
        var def_enc = ENC_RLE            # v1: how the definition levels are encoded
        var v2_compressed = True
        while True:
            var f = h.field()
            if f[1] == T_STOP:
                break
            if f[0] == 1 and f[1] == T_I32:
                ptype = Int(h.zigzag())
            elif f[0] == 2 and f[1] == T_I32:
                usize = Int(h.zigzag())
            elif f[0] == 3 and f[1] == T_I32:
                csize = Int(h.zigzag())
            elif (f[0] == 5 or f[0] == 7 or f[0] == 8) and f[1] == T_STRUCT:
                h.enter()
                while True:
                    var g = h.field()
                    if g[1] == T_STOP:
                        break
                    if g[0] == 1 and g[1] == T_I32:
                        nvals = Int(h.zigzag())
                    elif f[0] != 8 and g[0] == 2 and g[1] == T_I32:
                        encoding = Int(h.zigzag())
                    elif f[0] == 5 and g[0] == 3 and g[1] == T_I32:
                        def_enc = Int(h.zigzag())
                    elif f[0] == 8 and g[0] == 4 and g[1] == T_I32:
                        encoding = Int(h.zigzag())
                    elif f[0] == 8 and g[0] == 5 and g[1] == T_I32:
                        def_len = Int(h.zigzag())
                    elif f[0] == 8 and g[0] == 6 and g[1] == T_I32:
                        if Int(h.zigzag()) != 0:
                            raise Error("column `" + leaf.name + "` has repetition levels: not supported")
                    elif f[0] == 8 and (g[1] == T_TRUE or g[1] == T_FALSE) and g[0] == 7:
                        v2_compressed = g[1] == T_TRUE
                    else:
                        h.skip(g[1])
                h.leave()
            else:
                h.skip(f[1])
        var body_at = pos + h.pos
        if csize < 0 or usize < 0 or nvals < 0 or nvals > INT32_MAX:
            raise Error("column `" + leaf.name + "`: a page header has a negative size or count (corrupt file)")
        if csize > n - body_at:
            raise Error("column `" + leaf.name + "`: a page runs past the end of the file (corrupt file)")
        pos = body_at + csize
        if ptype == PAGE_DICTIONARY:
            if ctx.dict_seen:
                raise Error("column `" + leaf.name + "`: a chunk has two dictionary pages (corrupt file)")
            ctx.dict_seen = True
            if nvals > 8 * usize + 8:
                raise Error("column `" + leaf.name + "`: a dictionary page holds more values than its bytes can (corrupt file)")
            var page = List[UInt8]()
            var c = _Bytes(_page_bytes(meta.codec, base + body_at, csize, usize, page), usize)
            ctx.dict = List[Int64](unsafe_uninit_length=nvals)
            _plain_into(c, leaf, ctx, nvals, UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(ctx.dict.unsafe_ptr())),
                        tbl, True)
            _ = page^
            continue
        if ptype != PAGE_DATA and ptype != PAGE_DATA_V2:
            continue                                     # an index page: not needed
        if nvals > meta.values - done:
            raise Error("column `" + leaf.name + "`: a page holds more values than its column chunk (corrupt file)")
        var dict_coded = encoding == ENC_RLE_DICTIONARY or encoding == ENC_PLAIN_DICTIONARY
        if words_only and dict_coded:
            done += nvals
            continue
        var defer = leaf.ptype == PT_BYTE_ARRAY and ctx.add and encoding == ENC_PLAIN
        pages.append(_Page(base + body_at, csize, usize, nvals, encoding, ptype == PAGE_DATA_V2, def_len, def_enc,
                           v2_compressed, ctx_at, ctx.row0 + done, defer, 0, List[UInt8](), String(), List[Int](), List[Int](),
                           List[UInt64](), List[Int](), List[Int32](), List[Int](), List[UInt8]()))
        done += nvals


def _physical_memory() -> Int:
    """The machine's physical memory in bytes (sysconf), 0 when it cannot be told."""
    comptime if CompilationTarget.is_linux():
        var pages = external_call["sysconf", Int](Int32(85))         # _SC_PHYS_PAGES
        var size = external_call["sysconf", Int](Int32(30))          # _SC_PAGESIZE
        return pages * size if pages > 0 and size > 0 else 0
    else:
        var pages = external_call["sysconf", Int](Int32(200))        # Darwin's _SC_PHYS_PAGES
        var size = external_call["sysconf", Int](Int32(29))          # Darwin's _SC_PAGESIZE
        return pages * size if pages > 0 and size > 0 else 0


def _check_claim(what: String, rows: Int, ncols: Int) raises:
    """Refuses a row count whose decoded columns (8 bytes a value) could not fit in the machine's memory. The columns
    are sized from the counts a file states, so this is what stops a file of a few KB from asking for terabytes.
    Limits: a claim below the machine's memory (or an address-space cap lower than it) is still sized, untouched until
    its pages decode, and a corrupt page raises when it does."""
    if rows < 0:
        raise Error(what + " claims a negative row count (" + String(rows) + ") (corrupt file)")
    var ram = _physical_memory()
    if ram > 0 and rows > ram // (8 * max(ncols, 1)):
        var need = Float64(rows) * 8.0 * Float64(max(ncols, 1))
        raise Error(what + " claims " + String(rows) + " rows: " + String(need / 1e9) + " GB for " + String(ncols) +
                    (" column" if ncols == 1 else " columns") + ", more than this machine's memory (" + String(Float64(ram) / 1e9) +
                    " GB); a corrupt file, or too big to read whole (stream it with scan_parquet)")


comptime DEDUPE_MIN = 50_000        # misses of a column in a row group from which the cross-page dedupe runs in parallel


def _dedupe_misses(mut pages: List[_Page], ctx: Int, total: Int):
    """For the misses of column chunk `ctx` (all absent from the frozen index), find for each the FIRST miss, in page
    and miss order, of the same word: the words are split into hash partitions and each partition is walked by one
    task, so the serial phase only has to name the first occurrences (in order) and copy their codes."""
    var parts = 64
    for i in range(len(pages)):
        if pages[i].ctx == ctx:
            pages[i].miss_canon = List[Int](length=len(pages[i].miss_at), fill=0)
    var pa = Int(pages.unsafe_ptr())
    var np = len(pages)
    def part(w: Int) {imm}:
        var pg = Pointer[_Page, MutUntrackedOrigin](unsafe_from_address=pa)
        var size = 1024
        while size < 4 * (total // parts + 1):
            size *= 2
        var slots = List[Int](length=size, fill=-1)
        var count = 0
        for i in range(np):
            ref p = (pg + i)[]
            if p.ctx != ctx:
                continue
            for m in range(len(p.miss_at)):
                if Int((p.miss_h[m] >> 32) & UInt64(parts - 1)) != w:
                    continue
                var mask = size - 1
                var j = Int(p.miss_h[m]) & mask
                var found = -1
                while slots[j] >= 0:
                    var r = slots[j]
                    ref q = (pg + (r >> 32))[]
                    var qm = r & 0xFFFFFFFF
                    if q.miss_h[qm] == p.miss_h[m] and q.miss_n[qm] == p.miss_n[m]:
                        var a = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=q.miss_at[qm])
                        var b = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=p.miss_at[m])
                        var same = True
                        for k in range(p.miss_n[m]):
                            if a[k] != b[k]:
                                same = False
                                break
                        if same:
                            found = r
                            break
                    j = (j + 1) & mask
                if found >= 0:
                    p.miss_canon[m] = found
                else:
                    p.miss_canon[m] = (i << 32) | m
                    slots[j] = (i << 32) | m
                    count += 1
                    if count * 2 > size:                 # a skewed partition: double the table and re-place its words
                        var bigger = List[Int](length=size * 2, fill=-1)
                        for e in range(size):
                            if slots[e] >= 0:
                                ref q = (pg + (slots[e] >> 32))[]
                                var k = Int(q.miss_h[slots[e] & 0xFFFFFFFF]) & (size * 2 - 1)
                                while bigger[k] >= 0:
                                    k = (k + 1) & (size * 2 - 1)
                                bigger[k] = slots[e]
                        slots = bigger^
                        size *= 2
    par_n(part, parts, min(parts, core_count()))


def _decode_group(bases: List[Int], ns: List[Int], metas: List[_ChunkMeta], leaf_ids: List[Int], out_cols: List[Int],
                  widxs: List[Int], add_words: Bool, leaves: List[_Leaf], row0: Int, mut t: Table, words_only: Bool = False) raises:
    """Decodes one row group: chunk k (`metas[k]`, at `bases[k]` / `ns[k]`, leaf `leaf_ids[k]`) becomes rows
    row0.. of table column `out_cols[k]`, whose TEXT words are found in `widxs[k]` (a `_WordIndex` over the table's
    column) and, when `add_words`, added to it. `words_only`: only the TEXT words are added to the table (a dictionary build)."""
    if len(metas) == 0:
        return
    var tbl = Int(UnsafePointer(to=t))
    var ctxs = List[_Ctx]()
    var pages = List[_Page]()
    var rows = -1
    for k in range(len(metas)):
        ctxs.append(_Ctx(leaf_ids[k], out_cols[k], metas[k].codec, leaves[leaf_ids[k]].repetition == OPTIONAL, widxs[k],
                         add_words, List[Int64](), row0, False))
        if rows < 0:
            rows = metas[k].values
        elif metas[k].values != rows:
            raise Error("column `" + leaves[leaf_ids[k]].name + "` has " + String(metas[k].values) + " rows in this row group, column `" +
                        leaves[leaf_ids[0]].name + "` has " + String(rows) + " (corrupt file)")
        _walk_chunk(bases[k], ns[k], metas[k], leaves[leaf_ids[k]], ctxs[k], k, tbl, pages, words_only)
    if not words_only:
        for k in range(len(metas)):                     # the pages add up to the chunk (the walk); the chunk to the group
            if metas[k].group_rows >= 0 and metas[k].values != metas[k].group_rows:
                raise Error("column `" + leaves[leaf_ids[k]].name + "` has " + String(metas[k].values) + " values in a row group of " +
                            String(metas[k].group_rows) + " rows (corrupt file)")
        _check_claim("row group", rows, len(out_cols))
        for k in range(len(out_cols)):                  # size the columns once: the pages add up to the rows
            ref col = t.cols[out_cols[k]]
            if len(col.values) < row0:
                raise Error("column `" + col.name + "`: rows do not line up (corrupt file)")
            if len(col.values) < row0 + rows:            # (read_parquet sizes the whole column up front)
                if row0 == 0:                            # every row is written below, values or a zero for a null
                    col.values = List[Int64](unsafe_uninit_length=rows)
                else:
                    col.values.resize(row0 + rows, 0)
            if 0 < len(col.validity) < (len(col.values) + 7) // 8:      # a bitmap that exists covers every row
                col.validity.resize((len(col.values) + 7) // 8, 0xFF)
    var pa = Int(pages.unsafe_ptr())
    var ca = Int(ctxs.unsafe_ptr())
    var la = Int(leaves.unsafe_ptr())
    var cola = Int(t.cols.unsafe_ptr())
    def job(j: Int) {imm}:
        var page = Pointer[_Page, MutUntrackedOrigin](unsafe_from_address=pa)
        ref p = (page + j)[]
        ref ctx = (Pointer[_Ctx, MutUntrackedOrigin](unsafe_from_address=ca) + p.ctx)[]
        ref leaf = (Pointer[_Leaf, MutUntrackedOrigin](unsafe_from_address=la) + ctx.leaf)[]
        ref col = (Pointer[Column, MutUntrackedOrigin](unsafe_from_address=cola) + ctx.col)[]
        try:
            _decode_page(p, ctx, leaf, 0 if words_only else Int(col.values.unsafe_ptr()) + p.row * 8, tbl)
        except e:
            p.err = String(e)
    if len(pages) == 1:
        job(0)
    elif len(pages) > 1:
        par_n(job, len(pages), min(len(pages), core_count()))
    for i in range(len(pages)):
        if pages[i].err.byte_length() > 0:
            raise Error(pages[i].err)
    # the serial phase: the words the frozen index missed get their codes in file order (first seen first, as one
    # serial pass would give them), then the parallel phase writes those codes over the placeholders
    var fixes = List[Int]()
    for k in range(len(ctxs)):
        var total = 0
        for i in range(len(pages)):
            if pages[i].ctx == k:
                total += len(pages[i].miss_at)
        if total >= DEDUPE_MIN:
            _dedupe_misses(pages, k, total)
    for i in range(len(pages)):
        if len(pages[i].miss_at) > 0:
            ref ctx = ctxs[pages[i].ctx]
            var wi = UnsafePointer[_WordIndex, MutAnyOrigin](unsafe_from_address=ctx.widx)
            var deduped = len(pages[i].miss_canon) > 0
            for m in range(len(pages[i].miss_at)):
                var code = -1
                if deduped:
                    var r = pages[i].miss_canon[m]
                    if r != (i << 32) | m:               # the same word was missed first elsewhere: its code is ready
                        pages[i].miss_code.append(pages[r >> 32].miss_code[r & 0xFFFFFFFF])
                        continue
                else:
                    code = wi[].find(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=pages[i].miss_at[m]),
                                     pages[i].miss_n[m], pages[i].miss_h[m])
                if code < 0:
                    var wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=pages[i].miss_at[m])
                    var w = String(unsafe_from_utf8=Span[UInt8, MutAnyOrigin](unsafe_ptr=wp, length=pages[i].miss_n[m]))
                    code = t.add_word(ctx.col, w)
                    wi[].insert(pages[i].miss_h[m], code)
                pages[i].miss_code.append(code)
            fixes.append(i)
    if words_only:
        return
    if len(fixes) > 0:
        var fa = Int(fixes.unsafe_ptr())
        def fill(j: Int) {imm}:
            var page = Pointer[_Page, MutUntrackedOrigin](unsafe_from_address=pa)
            ref p = (page + Pointer[Int, MutUntrackedOrigin](unsafe_from_address=fa)[j])[]
            ref ctx = (Pointer[_Ctx, MutUntrackedOrigin](unsafe_from_address=ca) + p.ctx)[]
            ref col = (Pointer[Column, MutUntrackedOrigin](unsafe_from_address=cola) + ctx.col)[]
            var dst = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(col.values.unsafe_ptr()) + p.row * 8)
            var codes = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(p.miss_code.unsafe_ptr()))
            for r in range(p.nvals):
                var v = dst[r]
                if v < 0:
                    dst[r] = Int64(codes[Int(-1 - v)])
        if len(fixes) == 1:
            fill(0)
        else:
            par_n(fill, len(fixes), min(len(fixes), core_count()))
        for i in fixes:
            pages[i].keep = List[UInt8]()
    _ = fixes^
    for i in range(len(pages)):                         # nulls and validity bitmaps, in page order
        ref p = pages[i]
        ref col = t.cols[ctxs[p.ctx].col]
        col.nulls += p.nulls
        if p.nulls > 0:
            if len(col.validity) == 0:
                col.validity = List[UInt8](length=(len(col.values) + 7) // 8, fill=0xFF)
            _assign_bits(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(col.validity.unsafe_ptr())), p.row,
                         UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr())), p.nvals)
    _ = pages^
    _ = ctxs^


# ── scanning: a file bigger than memory, one row group at a time (F-140) ─────────────────────────────

comptime _MAP_ALIGN = 16384          # mmap offsets must be page multiples: 4 KB (Linux x86), 16 KB (Apple arm64)


struct _Mapped(Movable):
    """A read-only mapping of `length` bytes of a file from `offset`; unmapped when released or dropped, so a
    chunk's pages are in the process only while it is decoded."""
    var addr: Int
    var span: Int
    var data: Int                    # the address of the byte at `offset`

    def __init__(out self, fd: Int, offset: Int, length: Int, path: String) raises:
        var start = offset - offset % _MAP_ALIGN
        self.span = length + (offset - start)
        self.addr = external_call["mmap", Int](Int(0), self.span, Int32(1), Int32(2), fd, start)
        self.data = 0
        if self.addr == -1:
            self.span = 0
            raise Error("couldn't read `" + path + "`: part of the file could not be mapped into memory")
        self.data = self.addr + (offset - start)

    def release(mut self):
        if self.span > 0:
            _ = external_call["munmap", Int32](self.addr, self.span)
            self.span = 0

    def __del__(deinit self):
        if self.span > 0:
            _ = external_call["munmap", Int32](self.addr, self.span)


def _open_file(path: String) raises -> FileHandle:
    var file: FileHandle
    try:
        file = open(path, "r")
    except:
        raise Error("couldn't open `" + path + "`: no such file, or no permission to read it")
    return file^


def _file_size(file: FileHandle, path: String) raises -> Int:
    var size = external_call["lseek", Int](file._get_raw_fd(), Int(0), Int32(2))
    if size < 0:
        raise Error("couldn't read `" + path + "`: not a regular file")
    return size


def _chunk_source(fd: Int, size: Int, path: String, meta: _ChunkMeta, leaf: _Leaf, mut maps: List[_Mapped]) raises -> Tuple[Int, Int]:
    """Maps one column chunk's bytes (never the whole file); returns the (base, bound) `_walk_chunk` takes: base + a
    file offset is that byte's address, bound is the offset just past the chunk. Known limit: if the file is truncated
    while the mapping is decoded, touching a vanished page is a bus error; a change between row groups is caught by
    `_check_unchanged`."""
    if meta.values == 0:
        return (0, 0)
    var lo = meta.dict_off if meta.dict_off > 0 and meta.dict_off < meta.data_off else meta.data_off
    if lo <= 0 or meta.size <= 0 or meta.size > size - lo:
        raise Error("column `" + leaf.name + "`: a column chunk lies outside the file (corrupt file)")
    var m = _Mapped(fd, lo, meta.size, path)
    var base = m.data - lo
    maps.append(m^)
    return (base, lo + meta.size)


struct _ScanInfo(Movable):
    var path: String
    var file: FileHandle             # kept open: a file replaced by rename is still the one that was opened
    var size: Int
    var leaves: List[_Leaf]
    var groups: List[List[_ChunkMeta]]
    var rows_of: List[Int]
    var metadata: String
    var dicts: Table                 # TEXT columns' file-wide words, each built on first use
    var widx: List[_WordIndex]       # and their indexes by bytes
    var loaded: List[Bool]

    def __init__(out self, path: String, var file: FileHandle, size: Int, var leaves: List[_Leaf],
                 var groups: List[List[_ChunkMeta]], var rows_of: List[Int], var metadata: String, var dicts: Table):
        self.path = path
        self.file = file^
        self.size = size
        self.loaded = List[Bool](length=len(leaves), fill=False)
        self.widx = List[_WordIndex]()
        for _ in range(len(leaves)):
            self.widx.append(_WordIndex())
        self.leaves = leaves^
        self.groups = groups^
        self.rows_of = rows_of^
        self.metadata = metadata^
        self.dicts = dicts^

    def check_unchanged(self) raises:
        """A truncated or resized file: refuse before mapping pages that may be gone (a same-size rewrite is not seen)."""
        if external_call["lseek", Int](self.file._get_raw_fd(), Int(0), Int32(2)) != self.size:
            raise Error("the file changed since it was opened (`" + self.path + "`)")


struct ParquetScan(Copyable, Movable):
    """A Parquet file opened for streaming: its footer and columns, none of the data. `Query(scan)` runs a query over it
    one row group at a time (query.mojo); copies share everything, including the open file.

    A TEXT code is the code `read_parquet` gives the word (first seen, in file order), so a filter literal or a
    group count taken here is right for the query. A TEXT column's file-wide dictionary is built the first time it is
    asked for (`word_count`, `code_of`, `text_of`) or a query reads the column, from its dictionary pages (a PLAIN
    text page, as after a writer's dictionary fallback, is decoded for its words); a column nobody reads costs
    nothing. Limit: a high-cardinality text column that IS read keeps all its distinct words in memory.

    The file is held open (a file replaced by rename keeps reading the one opened). If it is truncated or resized, the
    next row group raises "the file changed since it was opened". Known limits: a rewrite in place that keeps the size
    is not detected (only the size is compared), and a truncation during the decoding of one column chunk is a bus
    error. No reader is safe against a concurrent writer; `write_parquet` replaces files by rename."""
    var _info: ArcPointer[_ScanInfo]
    var schema: SharedTable          # the columns (kinds and scales), no rows and no words

    def __init__(out self, var info: _ScanInfo, var schema: SharedTable):
        self._info = ArcPointer(info^)
        self.schema = schema^

    def rows(self) -> Int:
        var n = 0
        for r in self._info[].rows_of:
            n += r
        return n

    def row_groups(self) -> Int:
        return len(self._info[].groups)

    def group_rows(self, g: Int) -> Int:
        return self._info[].rows_of[g]

    def metadata(self) -> String:
        """The file's "omx" key-value metadata, "" when there is none."""
        return self._info[].metadata

    def columns(self) -> Int:
        return self.schema.columns()

    def name(self, c: Int) -> String:
        return self.schema.name(c)

    def kind(self, c: Int) -> ColumnKind:
        return self.schema.kind(c)

    def scale(self, c: Int) -> Int:
        return self.schema.scale(c)

    def index(self, name: String) -> Int:
        return self.schema.index(name)

    def column(self, name: String) raises -> Int:
        return self.schema.column(name)

    def dictionary_loaded(self, c: Int) -> Bool:
        """Whether column `c`'s file-wide dictionary has been built (it is built on first use)."""
        return self._info[].loaded[c]

    def ensure_words(self, c: Int) raises:
        """Builds TEXT column `c`'s file-wide dictionary (once)."""
        if self.schema.kind(c) != TEXT or self._info[].loaded[c]:
            return
        ref info = self._info[]
        info.check_unchanged()
        info.widx[c].col = Int(info.dicts.cols.unsafe_ptr() + c)
        for g in range(len(info.groups)):
            var maps = List[_Mapped]()
            var src = _chunk_source(info.file._get_raw_fd(), info.size, info.path, info.groups[g][c], info.leaves[c], maps)
            _decode_group([src[0]], [src[1]], [info.groups[g][c].copy()], [c], [c], [Int(info.widx.unsafe_ptr() + c)], True,
                          info.leaves, 0, info.dicts, words_only=True)
            _ = maps^
        info.loaded[c] = True

    def word_count(self, c: Int) raises -> Int:
        """A TEXT column's number of words in the whole file: the `groups=` of a query keyed by it."""
        self.ensure_words(c)
        return len(self._info[].dicts.dictionary(c))

    def text_of(self, c: Int, code: Int) raises -> String:
        self.ensure_words(c)
        return self._info[].dicts.text_of(c, code)

    def code_of(self, c: Int, word: String) raises -> Int:
        """The word's code, -1 when the file never holds it."""
        self.ensure_words(c)
        return self._info[].dicts.code_of(c, word)

    def read_group(self, g: Int, cols: List[Int]) raises -> Table:
        """Row group `g` as a table of just the columns `cols` (in that order), decoded: a TEXT value is the code the
        whole-file read gives its word (looked up in the file-wide dictionary while the pages are decoded)."""
        for c in cols:
            self.ensure_words(c)
        ref info = self._info[]
        info.check_unchanged()
        var t = Table.empty()
        var maps = List[_Mapped]()
        var bases = List[Int]()
        var ns = List[Int]()
        var metas = List[_ChunkMeta]()
        var outs = List[Int]()
        var widxs = List[Int]()
        for k in range(len(cols)):
            var c = cols[k]
            _ = t.add_column(self.schema.name(c), self.schema.kind(c), scale=self.schema.scale(c))
            var src = _chunk_source(info.file._get_raw_fd(), info.size, info.path, info.groups[g][c], info.leaves[c], maps)
            bases.append(src[0])
            ns.append(src[1])
            metas.append(info.groups[g][c].copy())
            outs.append(k)
            widxs.append(Int(info.widx.unsafe_ptr() + c) if self.schema.kind(c) == TEXT else 0)
        _decode_group(bases, ns, metas, cols, outs, widxs, False, info.leaves, 0, t)
        _ = maps^
        t.check()
        return t^


def scan_parquet(path: String) raises -> ParquetScan:
    """Opens a Parquet file for streaming (`Query(scan)`): reads the footer, no column data and no dictionary. Peak
    memory of a query over it is one row group's columns (plus the dictionaries of the TEXT columns it reads)."""
    var file = _open_file(path)
    var size = _file_size(file, path)
    var fd = file._get_raw_fd()
    if size < 12:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var head = _Mapped(fd, 0, 4, path)
    var tail = _Mapped(fd, size - 8, 8, path)
    var hp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=head.data)
    var tp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=tail.data)
    if hp[0] != 0x50 or hp[1] != 0x41 or hp[2] != 0x52 or hp[3] != 0x31 or \
       tp[4] != 0x50 or tp[5] != 0x41 or tp[6] != 0x52 or tp[7] != 0x31:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var flen = Int(tp[0]) | (Int(tp[1]) << 8) | (Int(tp[2]) << 16) | (Int(tp[3]) << 24)
    head.release()
    tail.release()
    if flen <= 0 or flen > size - 12:
        raise Error("`" + path + "`: the Parquet footer length is wrong (corrupt file)")
    var fm = _Mapped(fd, size - 8 - flen, flen, path)
    var r = ThriftReader(fm.data, flen)
    var footer = _read_footer(r)
    fm.release()
    var cols = List[_Leaf]()
    var dicts = _schema_table(path, footer.leaves, cols)
    var unused = List[_Leaf]()
    var schema = _schema_table(path, footer.leaves, unused)
    var n_cols = len(cols)
    var groups = List[List[_ChunkMeta]]()
    var rows_of = List[Int]()
    for g in range(len(footer.groups)):
        if len(footer.groups[g]) != n_cols:
            raise Error("`" + path + "`: row group " + String(g) + " has " + String(len(footer.groups[g])) +
                        " columns, the schema " + String(n_cols))
        rows_of.append(footer.groups[g][0].values if n_cols > 0 else 0)
        groups.append(footer.groups[g].copy())
    return ParquetScan(_ScanInfo(path, file^, size, cols^, groups^, rows_of^, footer.omx.copy(), dicts^), schema^.share())
