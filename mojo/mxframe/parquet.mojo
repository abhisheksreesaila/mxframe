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
        var out = List[UInt8](length=max(size, 1), fill=0)
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
            var out = List[UInt8](length=max(size, 1), fill=0)
            var got = external_call["ZSTD_decompress", Int](out.unsafe_ptr(), size, src, n)
            if external_call["ZSTD_isError", UInt32](got) != 0 or got != size:
                raise Error("a zstd page did not decompress to its stated size (corrupt file)")
            out.resize(size, 0)
            return out^
        if codec == SNAPPY:
            var stated = 0
            if external_call["snappy_uncompressed_length", Int32](src, n, UnsafePointer(to=stated)) != 0 or stated != size:
                raise Error("a snappy page did not decompress to its stated size (corrupt file)")
            var out = List[UInt8](length=max(size, 1), fill=0)
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


def _plain(mut c: _Bytes, leaf: _Leaf, count: Int, mut t: Table, col: Int, mut out: List[Int64]) raises:
    """`count` PLAIN values as stored Int64s (TEXT words become the column's codes)."""
    if count < 0:
        raise Error("column `" + leaf.name + "`: a page has a negative value count (corrupt file)")
    if leaf.ptype == PT_BOOLEAN:
        _unpack(c, 1, count, out)
    elif leaf.ptype == PT_INT32:
        for _ in range(count):
            out.append(_to_stored(leaf, c.le(4)))
    elif leaf.ptype == PT_INT64:
        for _ in range(count):
            out.append(_to_stored(leaf, c.le(8)))
    elif leaf.ptype == PT_BYTE_ARRAY:
        for _ in range(count):
            var n = Int(c.le(4)) & 0xFFFFFFFF
            c.need(n)
            if not _valid_utf8(c.base + c.pos, n):
                raise Error("column `" + leaf.name + "`: a text value is not valid UTF-8")
            var s = String(unsafe_from_utf8=Span[UInt8, MutAnyOrigin](
                unsafe_ptr=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos), length=n))
            c.pos += n
            out.append(Int64(t.add_word(col, s)))
    elif leaf.ptype == PT_FIXED:                        # a big-endian two's-complement decimal
        var w = leaf.type_length
        if w < 1 or w > 16:
            raise Error("column `" + leaf.name + "` has a fixed length of " + String(w) + " bytes (corrupt file)")
        if count > (c.n - c.pos) // w:
            raise Error("a Parquet page ends early (corrupt file)")
        for _ in range(count):
            var v: Int64 = -1 if c.at(c.pos) >= 0x80 else 0
            for i in range(w):
                v = (v << 8) | Int64(c.at(c.pos + i))
            c.pos += w
            out.append(v)
    else:
        raise Error("column `" + leaf.name + "`: its physical type cannot be read yet")


def read_parquet(path: String) raises -> ParquetResult:
    """Reads a Parquet file (written by MX frame, pyarrow, DuckDB, ...) into a Table, plus its "omx" metadata."""
    var data: List[UInt8]
    with open(path, "r") as f:
        data = f.read_bytes()
    var n = len(data)
    if n < 12 or data[0] != 0x50 or data[1] != 0x41 or data[2] != 0x52 or data[3] != 0x31 or \
       data[n - 4] != 0x50 or data[n - 3] != 0x41 or data[n - 2] != 0x52 or data[n - 1] != 0x31:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var flen = Int(data[n - 8]) | (Int(data[n - 7]) << 8) | (Int(data[n - 6]) << 16) | (Int(data[n - 5]) << 24)
    if flen <= 0 or flen > n - 12:
        raise Error("`" + path + "`: the Parquet footer length is wrong (corrupt file)")
    var base = Int(data.unsafe_ptr())
    var r = ThriftReader(base + n - 8 - flen, flen)

    var footer = _read_footer(r)
    ref leaves = footer.leaves
    ref groups = footer.groups

    var cols = List[_Leaf]()
    var t = _schema_table(path, leaves, cols)

    for g in range(len(groups)):
        if len(groups[g]) != len(cols):
            raise Error("`" + path + "`: row group " + String(g) + " has " + String(len(groups[g])) +
                        " columns, the schema " + String(len(cols)))
        for c in range(len(cols)):
            _read_chunk(base, n, groups[g][c], cols[c], t, c)
    t.check()
    _ = data^
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
    r.enter()
    while True:
        var f = r.field()
        if f[1] == T_STOP:
            break
        if f[0] == 1 and f[1] == T_LIST:
            var h = r.list_header()
            for _ in range(h[1]):
                var meta = _ChunkMeta(-1, 0, 0, -1, -1, 0)
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
        else:
            r.skip(f[1])
    r.leave()
    return chunks^


def _read_chunk(base: Int, n: Int, meta: _ChunkMeta, leaf: _Leaf, mut t: Table, col: Int, words_only: Bool = False) raises:
    """Appends one column chunk's values to column `col`. `words_only` (a TEXT column's file-wide dictionary): only the
    words are added, in the order a full read adds them; dictionary-indexed data pages are not even decompressed."""
    var pos = meta.dict_off if meta.dict_off > 0 and meta.dict_off < meta.data_off else meta.data_off
    var dictionary = List[Int64]()
    var done = 0
    var optional = leaf.repetition == OPTIONAL
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
            var page = _decompress(meta.codec, base + body_at, csize, usize)
            var c = _Bytes(Int(page.unsafe_ptr()), len(page))
            dictionary.clear()
            _plain(c, leaf, nvals, t, col, dictionary)
            _ = page^
            continue
        if ptype != PAGE_DATA and ptype != PAGE_DATA_V2:
            continue                                     # an index page: not needed
        if nvals > meta.values - done:
            raise Error("column `" + leaf.name + "`: a page holds more values than its column chunk (corrupt file)")
        if words_only and (encoding == ENC_RLE_DICTIONARY or encoding == ENC_PLAIN_DICTIONARY):
            done += nvals
            continue
        var levels = List[Int64]()
        var page: List[UInt8]
        var c: _Bytes
        if ptype == PAGE_DATA_V2:
            var lv = def_len
            if lv < 0 or lv > csize or lv > usize:
                raise Error("column `" + leaf.name + "`: the definition levels are longer than their page (corrupt file)")
            if optional:
                var lc = _Bytes(base + body_at, lv)
                _rle_hybrid(lc, 1, nvals, levels)
            if v2_compressed:
                page = _decompress(meta.codec, base + body_at + lv, csize - lv, usize - lv)
            else:
                page = _decompress(UNCOMPRESSED, base + body_at + lv, csize - lv, usize - lv)
            c = _Bytes(Int(page.unsafe_ptr()), len(page))
        else:
            page = _decompress(meta.codec, base + body_at, csize, usize)
            c = _Bytes(Int(page.unsafe_ptr()), len(page))
            if optional:
                if def_enc == ENC_BIT_PACKED:                   # deprecated: most significant bit first, no length
                    var nb = (nvals + 7) // 8
                    c.need(nb)
                    for i in range(nvals):
                        levels.append(Int64((c.at(c.pos + i // 8) >> (7 - i % 8)) & 1))
                    c.pos += nb
                elif def_enc == ENC_RLE:
                    var ln = Int(c.le(4)) & 0xFFFFFFFF
                    c.need(ln)
                    var lc = _Bytes(c.base + c.pos, ln)
                    _rle_hybrid(lc, 1, nvals, levels)
                    c.pos += ln
                else:
                    raise Error("column `" + leaf.name + "`: definition levels use encoding " + String(def_enc) + ", not supported")
        var present = nvals
        if optional:
            present = 0
            for lvl in levels:
                present += Int(lvl)
        var vals = List[Int64]()
        if encoding == ENC_PLAIN:
            _plain(c, leaf, present, t, col, vals)
        elif encoding == ENC_RLE_DICTIONARY or encoding == ENC_PLAIN_DICTIONARY:
            var w = c.byte()
            var idx = List[Int64]()
            _rle_hybrid(c, w, present, idx)
            for i in idx:
                if i < 0 or Int(i) >= len(dictionary):
                    raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
                vals.append(dictionary[Int(i)])
        elif encoding == ENC_DELTA_BINARY_PACKED and (leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64):
            var raw = List[Int64]()
            _ = _delta_binary_packed(c, leaf.ptype == PT_INT32, present, raw)
            for v in raw:
                vals.append(_to_stored(leaf, v))
        elif encoding == ENC_RLE and leaf.ptype == PT_BOOLEAN:
            _ = c.le(4)                                   # a length prefix
            _rle_hybrid(c, 1, present, vals)
        else:
            raise Error("column `" + leaf.name + "`: Parquet encoding " + String(encoding) + " is not supported yet")
        if len(vals) < present:
            raise Error("column `" + leaf.name + "`: a page holds fewer values than its levels say (corrupt file)")
        if not words_only:
            var k = 0
            for i in range(nvals):
                if optional and levels[i] == 0:
                    t.append_null(col)
                else:
                    t.cols[col].push(vals[k], True)             # already checked: codes from add_word, 0 / 1 bits
                    k += 1
        done += nvals
        _ = page^


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


def _open_file(path: String) raises -> Tuple[FileHandle, Int]:
    """The file and its size."""
    var file: FileHandle
    try:
        file = open(path, "r")
    except:
        raise Error("couldn't open `" + path + "`: no such file, or no permission to read it")
    var size = external_call["lseek", Int](file._get_raw_fd(), Int(0), Int32(2))
    if size < 0:
        file.close()
        raise Error("couldn't read `" + path + "`: not a regular file")
    return (file^, size)


def _read_chunk_mapped(fd: Int, size: Int, path: String, meta: _ChunkMeta, leaf: _Leaf, mut t: Table, col: Int,
                       words_only: Bool = False) raises:
    """One column chunk, read through a mapping of just its bytes (never the whole file)."""
    if meta.values == 0:
        return
    var lo = meta.dict_off if meta.dict_off > 0 and meta.dict_off < meta.data_off else meta.data_off
    if lo <= 0 or meta.size <= 0 or meta.size > size - lo:
        raise Error("column `" + leaf.name + "`: a column chunk lies outside the file (corrupt file)")
    var m = _Mapped(fd, lo, meta.size, path)
    # the page walker takes file offsets: shift the base so that offset `lo` is the mapping's first byte; the
    # chunk's end is the walker's bound
    _read_chunk(m.data - lo, lo + meta.size, meta, leaf, t, col, words_only)
    m.release()


struct _ScanInfo(Movable):
    var path: String
    var size: Int
    var leaves: List[_Leaf]
    var groups: List[List[_ChunkMeta]]
    var rows_of: List[Int]
    var metadata: String

    def __init__(out self, path: String, size: Int, var leaves: List[_Leaf], var groups: List[List[_ChunkMeta]],
                 var rows_of: List[Int], var metadata: String):
        self.path = path
        self.size = size
        self.leaves = leaves^
        self.groups = groups^
        self.rows_of = rows_of^
        self.metadata = metadata^


struct ParquetScan(Copyable, Movable):
    """A Parquet file opened for streaming: its footer, its columns and each TEXT column's file-wide dictionary, none
    of the data. `Query(scan)` runs a query over it one row group at a time (query.mojo); copies share everything.

    A TEXT code is the code `read_parquet` gives the word (first seen, in file order), so a filter literal or a
    group count taken here is right for the query. The dictionaries are found by one pass over the dictionary pages
    (a column whose pages are not all dictionary-encoded is read for those pages) when the scan is opened."""
    var _info: ArcPointer[_ScanInfo]
    var schema: SharedTable          # the columns, no rows; TEXT columns carry the file-wide words

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

    def word_count(self, c: Int) -> Int:
        """A TEXT column's number of words in the whole file: the `groups=` of a query keyed by it."""
        return len(self.schema.dictionary(c))

    def text_of(self, c: Int, code: Int) -> String:
        return self.schema.text_of(c, code)

    def code_of(self, c: Int, word: String) -> Int:
        """The word's code, -1 when the file never holds it."""
        return self.schema.code_of(c, word)

    def read_group(self, g: Int, cols: List[Int]) raises -> Table:
        """Row group `g` as a table of just the columns `cols` (in that order), decoded and mapped onto the file-wide
        dictionaries: a TEXT value is the code the whole-file read gives its word."""
        ref info = self._info[]
        var opened = _open_file(info.path)
        var fd = opened[0]._get_raw_fd()
        var t = Table.empty()
        for k in range(len(cols)):
            var c = cols[k]
            _ = t.add_column(self.schema.name(c), self.schema.kind(c), scale=self.schema.scale(c))
            _read_chunk_mapped(fd, info.size, info.path, info.groups[g][c], info.leaves[c], t, k)
            if self.schema.kind(c) == TEXT:
                self._remap_words(t, k, c)
        opened[0].close()
        t.check()
        return t^

    def _remap_words(self, mut t: Table, k: Int, c: Int) raises:
        """The chunk's own first-seen codes become the file-wide ones (one lookup per distinct word)."""
        ref col = t.cols[k]
        var to = List[Int64]()
        for w in col.words:
            var code = self.schema.code_of(c, w)
            if code < 0:
                raise Error("column `" + col.name + "`: a word is missing from the file's dictionary (corrupt file)")
            to.append(Int64(code))
        if col.nulls == 0:
            for i in range(len(col.values)):
                col.values[i] = to[Int(col.values[i])]
        else:
            for i in range(len(col.values)):
                if not col.is_null(i):
                    col.values[i] = to[Int(col.values[i])]
        col.words = List[String]()
        col.codes = Dict[String, Int]()


def scan_parquet(path: String) raises -> ParquetScan:
    """Opens a Parquet file for streaming (`Query(scan)`): reads the footer and each TEXT column's dictionary, no
    column data. Peak memory of a query over it is one row group's columns, not the file."""
    var opened = _open_file(path)
    var size = opened[1]
    var fd = opened[0]._get_raw_fd()
    if size < 12:
        opened[0].close()
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var head = _Mapped(fd, 0, 4, path)
    var tail = _Mapped(fd, size - 8, 8, path)
    var hp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=head.data)
    var tp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=tail.data)
    if hp[0] != 0x50 or hp[1] != 0x41 or hp[2] != 0x52 or hp[3] != 0x31 or \
       tp[4] != 0x50 or tp[5] != 0x41 or tp[6] != 0x52 or tp[7] != 0x31:
        opened[0].close()
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var flen = Int(tp[0]) | (Int(tp[1]) << 8) | (Int(tp[2]) << 16) | (Int(tp[3]) << 24)
    head.release()
    tail.release()
    if flen <= 0 or flen > size - 12:
        opened[0].close()
        raise Error("`" + path + "`: the Parquet footer length is wrong (corrupt file)")
    var fm = _Mapped(fd, size - 8 - flen, flen, path)
    var r = ThriftReader(fm.data, flen)
    var footer = _read_footer(r)
    fm.release()
    var cols = List[_Leaf]()
    var schema = _schema_table(path, footer.leaves, cols)
    var n_cols = len(cols)
    var groups = List[List[_ChunkMeta]]()
    var rows_of = List[Int]()
    for g in range(len(footer.groups)):
        if len(footer.groups[g]) != n_cols:
            opened[0].close()
            raise Error("`" + path + "`: row group " + String(g) + " has " + String(len(footer.groups[g])) +
                        " columns, the schema " + String(n_cols))
        rows_of.append(footer.groups[g][0].values if n_cols > 0 else 0)
        groups.append(footer.groups[g].copy())
    for c in range(n_cols):                       # each TEXT column's file-wide dictionary
        if schema.kind(c) == TEXT:
            for g in range(len(groups)):
                _read_chunk_mapped(fd, size, path, groups[g][c], cols[c], schema, c, words_only=True)
    opened[0].close()
    return ParquetScan(_ScanInfo(path, size, cols^, groups^, rows_of^, footer.omx.copy()), schema^.share())
