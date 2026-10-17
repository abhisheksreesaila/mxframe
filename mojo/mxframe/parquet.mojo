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
at all; one data page per column per row group.
"""
from std.ffi import external_call
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL
from .thrift import ThriftWriter, ThriftReader, T_I32, T_I64, T_BINARY, T_LIST, T_STRUCT, T_TRUE, T_FALSE, T_STOP

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
comptime CT_DECIMAL = 5
comptime CT_DATE = 6
comptime CT_TIMESTAMP_MILLIS = 9
comptime CT_TIMESTAMP_MICROS = 10
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

comptime DEFAULT_ROW_GROUP = 1 << 20


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
    """`n` compressed bytes at address `src` into `size` bytes."""
    var out = List[UInt8](length=max(size, 1), fill=0)
    if codec == UNCOMPRESSED:
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=src)
        for i in range(size):
            out[i] = p[i]
    elif codec == ZSTD:
        var got = external_call["ZSTD_decompress", Int](out.unsafe_ptr(), size, src, n)
        if external_call["ZSTD_isError", UInt32](got) != 0 or got != size:
            raise Error("a zstd page did not decompress to its stated size (corrupt file)")
    elif codec == SNAPPY:
        var got = size
        if external_call["snappy_uncompress", Int32](src, n, out.unsafe_ptr(), UnsafePointer(to=got)) != 0 or got != size:
            raise Error("a snappy page did not decompress to its stated size (corrupt file)")
    else:
        raise Error("compression codec " + String(codec) + " is not supported yet (UNCOMPRESSED, SNAPPY, ZSTD)")
    out.resize(size, 0)
    return out^


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


def _column_pages(mut file: List[UInt8], col: Column, lo: Int, hi: Int, optional: Bool, codec: Int) raises -> _Chunk:
    """Appends one column chunk (rows lo..hi) to `file`."""
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
                _put64(body, col.values[row])
    var cb = _compress(codec, body)
    chunk.data_offset = len(file)
    _page_header(file, PAGE_DATA, len(body), len(cb), hi - lo, ENC_RLE_DICTIONARY if col.kind == TEXT else ENC_PLAIN)
    chunk.uncompressed += len(file) - chunk.data_offset + len(body)
    file.extend(cb^)
    chunk.compressed = len(file) - start
    return chunk^


def _write(path: String, t: Table, metadata: String, codec: Int, row_group_rows: Int) raises:
    t.check()
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
            total += ch.compressed
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
    w.binary(6, "MX frame")
    w.end()
    var meta = w^.take()
    var footer = len(meta)
    file.extend(meta^)
    _put32(file, footer)
    file.extend([UInt8(0x50), UInt8(0x41), UInt8(0x52), UInt8(0x31)])
    with open(path, "w") as f:
        f.write_bytes(Span(file))


# ── reading ──────────────────────────────────────────────────────────────────────────────────────────

struct ParquetResult(Movable):
    var table: Table
    var metadata: String             # the "omx" key-value metadata, "" when there is none

    def __init__(out self, var table: Table, var metadata: String):
        self.table = table^
        self.metadata = metadata^


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


@fieldwise_init
struct _ChunkMeta(Copyable, Movable):
    var ptype: Int
    var codec: Int
    var values: Int
    var data_off: Int
    var dict_off: Int


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
        if self.pos + k > self.n:
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
                raise Error("a Parquet page has a bad varint")

    def zigzag(mut self) raises -> Int64:
        var u = self.varint()
        return Int64(u >> 1) ^ -Int64(u & 1)


def _unpack(mut c: _Bytes, width: Int, count: Int, mut out: List[Int64]) raises:
    """`count` values of `width` bits, least significant first (Parquet's bit packing)."""
    if width == 0:
        for _ in range(count):
            out.append(0)
        return
    var nbytes = (count * width + 7) // 8
    c.need(nbytes)
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
    """The RLE / bit-packing hybrid: runs of one repeated value and groups of 8 bit-packed values."""
    var start = len(out)
    while len(out) - start < count:
        var h = Int(c.varint())
        if h & 1 == 1:
            var n = (h >> 1) * 8
            _unpack(c, width, n, out)
        else:
            var v = c.le((width + 7) // 8) if width > 0 else 0
            if width > 0 and width < 64:
                v &= Int64((UInt64(1) << UInt64(width)) - 1)
            for _ in range(h >> 1):
                out.append(v)
    out.resize(start + count, 0)


def _wrap32(v: Int64) -> Int64:
    """An INT32 column's value: the low 32 bits, sign-extended (its deltas wrap modulo 2^32)."""
    var low = v & 0xFFFFFFFF
    return low - (Int64(1) << 32) if low >= (Int64(1) << 31) else low


def _delta_binary_packed(mut c: _Bytes, int32: Bool, mut out: List[Int64]) raises -> Int:
    """DELTA_BINARY_PACKED integers; returns how many were read. INT32 arithmetic wraps modulo 2^32, as the writer's."""
    var block = Int(c.varint())
    var minis = Int(c.varint())
    var total = Int(c.varint())
    var prev = c.zigzag()
    if total == 0:
        return 0
    out.append(prev)
    var got = 1
    var per_mini = block // minis if minis > 0 else 0
    while got < total:
        var min_delta = c.zigzag()
        var widths = List[Int]()
        for _ in range(minis):
            widths.append(c.byte())
        for m in range(minis):
            if got >= total:
                break                                   # trailing miniblocks of the last block may be absent
            var deltas = List[Int64]()
            _unpack(c, widths[m], per_mini, deltas)
            for d in deltas:
                if got >= total:
                    break
                prev = prev + min_delta + d
                if int32:
                    prev = _wrap32(prev)
                out.append(prev)
                got += 1
    return total


def _kind_of(leaf: _Leaf) raises -> Tuple[ColumnKind, Int]:
    """The MX frame kind (and scale) a Parquet leaf column becomes."""
    if leaf.ptype == PT_BOOLEAN:
        return (BOOL, 0)
    if leaf.ptype == PT_BYTE_ARRAY:
        return (TEXT, 0)
    var decimal = leaf.converted == CT_DECIMAL or leaf.logical == 5
    if decimal:
        if leaf.precision > 18:
            raise Error("column `" + leaf.name + "`: DECIMAL(" + String(leaf.precision) + ", " + String(leaf.scale) +
                        ") has more digits than an int64 holds (18)")
        if leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64 or leaf.ptype == PT_FIXED:
            return (DECIMAL, leaf.scale)
    if leaf.converted == CT_DATE or leaf.logical == 6:
        return (DATE, 0)
    if leaf.ptype == PT_INT64 and (leaf.logical == 8 or leaf.converted == CT_TIMESTAMP_MILLIS or
                                   leaf.converted == CT_TIMESTAMP_MICROS):
        return (DATE, 0)                                # a timestamp becomes its day (the CSV reader's rule)
    if leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64:
        return (INT, 0)
    var names: List[String] = ["BOOLEAN", "INT32", "INT64", "INT96", "FLOAT", "DOUBLE", "BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY"]
    raise Error("column `" + leaf.name + "` is " + names[leaf.ptype] + ": MX frame has no kind for it yet")


def _to_stored(leaf: _Leaf, raw: Int64) -> Int64:
    """A plain value of the leaf's physical type as MX frame stores it (timestamps become days, floored)."""
    if leaf.ptype == PT_INT64 and (leaf.logical == 8 or leaf.converted == CT_TIMESTAMP_MILLIS or
                                   leaf.converted == CT_TIMESTAMP_MICROS):
        var per_day = Int64(86_400_000) if (leaf.ts_unit == 1 or leaf.converted == CT_TIMESTAMP_MILLIS) else \
                      Int64(86_400_000_000_000) if leaf.ts_unit == 3 else Int64(86_400_000_000)
        return raw // per_day
    return raw


def _plain(mut c: _Bytes, leaf: _Leaf, count: Int, mut t: Table, col: Int, mut out: List[Int64]) raises:
    """`count` PLAIN values as stored Int64s (TEXT words become the column's codes)."""
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
            var s = String(unsafe_from_utf8=Span[UInt8, MutAnyOrigin](
                unsafe_ptr=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos), length=n))
            c.pos += n
            out.append(Int64(t.add_word(col, s)))
    elif leaf.ptype == PT_FIXED:                        # a big-endian two's-complement decimal
        var w = leaf.type_length
        c.need(w)
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

    # the leaves: a flat schema (a root, then one element per column)
    if len(leaves) == 0:
        raise Error("`" + path + "`: the Parquet schema is empty")
    var t = Table.empty()
    var cols = List[_Leaf]()
    for i in range(1, len(leaves)):
        if leaves[i].children > 0 or leaves[i].repetition == REPEATED:
            raise Error("column `" + leaves[i].name + "` is nested (a list or a struct): not supported yet")
        var k = _kind_of(leaves[i])
        _ = t.add_column(leaves[i].name, k[0], scale=k[1])
        cols.append(leaves[i].copy())

    for g in range(len(groups)):
        if len(groups[g]) != len(cols):
            raise Error("`" + path + "`: row group " + String(g) + " has " + String(len(groups[g])) +
                        " columns, the schema " + String(len(cols)))
        for c in range(len(cols)):
            _read_chunk(base, n, groups[g][c], cols[c], t, c)
    t.check()
    _ = data^
    return ParquetResult(t^, omx^)


def _read_schema_element(mut r: ThriftReader) raises -> _Leaf:
    var leaf = _Leaf("", -1, 0, REQUIRED, -1, 0, 0, -1, 0, 0)
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
                var meta = _ChunkMeta(-1, 0, 0, -1, -1)
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


def _read_chunk(base: Int, n: Int, meta: _ChunkMeta, leaf: _Leaf, mut t: Table, col: Int) raises:
    """Appends one column chunk's values to column `col`."""
    var pos = meta.dict_off if meta.dict_off > 0 and meta.dict_off < meta.data_off else meta.data_off
    var dictionary = List[Int64]()
    var done = 0
    var optional = leaf.repetition == OPTIONAL
    while done < meta.values:
        if pos <= 0 or pos >= n:
            raise Error("column `" + leaf.name + "`: a page offset is outside the file (corrupt file)")
        var h = ThriftReader(base + pos, n - pos)
        var ptype = -1
        var usize = 0
        var csize = 0
        var nvals = 0
        var encoding = 0
        var def_len = -1                 # v2: definition levels' byte length
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
        if body_at + csize > n:
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
        var levels = List[Int64]()
        var page: List[UInt8]
        var c: _Bytes
        if ptype == PAGE_DATA_V2:
            var lv = max(def_len, 0)
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
                var ln = Int(c.le(4)) & 0xFFFFFFFF
                var lc = _Bytes(c.base + c.pos, ln)
                _rle_hybrid(lc, 1, nvals, levels)
                c.pos += ln
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
            _ = _delta_binary_packed(c, leaf.ptype == PT_INT32, raw)
            for v in raw:
                vals.append(_to_stored(leaf, v))
        elif encoding == ENC_RLE and leaf.ptype == PT_BOOLEAN:
            _ = c.le(4)                                   # a length prefix
            _rle_hybrid(c, 1, present, vals)
        else:
            raise Error("column `" + leaf.name + "`: Parquet encoding " + String(encoding) + " is not supported yet")
        if len(vals) < present:
            raise Error("column `" + leaf.name + "`: a page holds fewer values than its levels say (corrupt file)")
        var k = 0
        for i in range(nvals):
            if optional and levels[i] == 0:
                t.append_null(col)
            else:
                t.cols[col].push(vals[k], True)             # already checked: codes from add_word, 0 / 1 bits
                k += 1
        done += nvals
        _ = page^
