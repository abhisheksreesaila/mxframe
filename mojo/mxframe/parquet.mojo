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
