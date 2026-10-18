"""F-135 (M4): Parquet written by the Python-free Mojo package reads back in pyarrow and DuckDB with every value,
type, null and the "omx" metadata, for each codec and several row-group sizes."""
import json
import struct
import subprocess
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
WORDS = ["West", "East", "Nörth", "", "South-East"]


@pytest.fixture(scope="module")
def writer(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_write_sample"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_write_sample.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def expected(n):
    rows = []
    for i in range(n):
        rev = None if i % 13 == 4 else (i * 1_000_003 + (1 << 53) if i % 97 == 0 else i * 101 - 50_000)
        rows.append({
            "qty": i * 7 - 3000,
            "revenue": rev,
            "region": None if i % 17 == 2 else WORDS[i % 5],
            "day": 18_000 + i % 400,
            "paid": None if i % 11 == 6 else (i % 3 == 0),
        })
    return rows


@pytest.mark.parametrize("codec", ["zstd", "snappy", "none"])
@pytest.mark.parametrize("n,group", [(0, 1024), (1, 8), (1000, 1024), (5000, 1024)])
def test_pyarrow_and_duckdb_read_what_mojo_wrote(writer, tmp_path, codec, n, group):
    path = tmp_path / "t.parquet"
    subprocess.run([str(writer), str(path), codec, str(n), str(group)], check=True)
    t = pq.read_table(path)
    md = pq.read_metadata(path)
    assert md.num_row_groups == max(1, -(-n // group))
    assert json.loads(t.schema.metadata[b"omx"]) == {"formats": {"revenue": "money"}}
    assert t.schema.field("qty").type == pa.int64()
    assert t.schema.field("revenue").type == pa.decimal128(18, 2)
    assert t.schema.field("region").type == pa.string()
    assert t.schema.field("day").type == pa.date32()
    assert t.schema.field("paid").type == pa.bool_()
    want = expected(n)
    got = t.to_pylist()
    for g, w in zip(got, want):
        assert g["qty"] == w["qty"] and g["region"] == w["region"] and g["paid"] == w["paid"]
        assert (g["revenue"] is None) == (w["revenue"] is None)
        if w["revenue"] is not None:
            assert int(g["revenue"].scaleb(2)) == w["revenue"]          # the exact scaled integer
        assert (g["day"] - __import__("datetime").date(1970, 1, 1)).days == w["day"]
    assert len(got) == n
    con = duckdb.connect()
    s = con.execute(f"SELECT count(*), sum(qty), count(revenue), count(region), sum(CASE WHEN paid THEN 1 ELSE 0 END) "
                    f"FROM read_parquet('{path}')").fetchone()
    assert s[0] == n
    assert (s[1] or 0) == sum(w["qty"] for w in want)
    assert s[2] == sum(w["revenue"] is not None for w in want)
    assert s[3] == sum(w["region"] is not None for w in want)
    assert (s[4] or 0) == sum(bool(w["paid"]) for w in want)


# ── reading files other writers made ─────────────────────────────────────────────────────────────────

import datetime as _dt
import decimal as _dec
import random as _random


@pytest.fixture(scope="module")
def dumper(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_dump"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_dump.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def mojo_read(dumper, path):
    """(columns [(name, kind, scale)], rows of stored values with TEXT as the word)."""
    out = subprocess.run([str(dumper), str(path)], check=True, capture_output=True).stdout.decode("utf-8")
    lines = out.split("\n")
    head = json.loads(lines[0])
    words = {}
    rows = []
    for line in lines[1:]:
        if line.startswith("WORDS "):
            _, c, rest = line.split(" ", 2)
            words[int(c)] = rest.split("\x1f")[:-1]
        elif line:
            rows.append(json.loads(line))
    for r in rows:
        for c, v in enumerate(r):
            if isinstance(v, dict):
                r[c] = words[c][v["word"]]
    return [tuple(c) for c in head["columns"]], rows


def stored(v, t):
    """A pyarrow value as MX frame stores it."""
    if v is None:
        return None
    if pa.types.is_decimal(t):
        return int(v.scaleb(t.scale))
    if pa.types.is_date(t):
        return (v - _dt.date(1970, 1, 1)).days
    if pa.types.is_timestamp(t):
        return (v.replace(tzinfo=None) - _dt.datetime(1970, 1, 1)) // _dt.timedelta(days=1)
    if pa.types.is_boolean(t):
        return int(v)
    return v


def sample(n, seed=3):
    r = _random.Random(seed)
    def maybe(v, k):
        return None if r.random() < k else v
    return pa.table({
        "i64": pa.array([maybe(r.randint(-2**62, 2**62), 0.1) for _ in range(n)], pa.int64()),
        "i32": pa.array([r.randint(-2**31, 2**31 - 1) for _ in range(n)], pa.int32()),
        "dec2": pa.array([maybe(_dec.Decimal(r.randint(-10**9, 10**9)).scaleb(-2), 0.2) for _ in range(n)], pa.decimal128(12, 2)),
        "dec4": pa.array([_dec.Decimal(r.randint(-10**17, 10**17)).scaleb(-4) for _ in range(n)], pa.decimal128(18, 4)),
        "d": pa.array([maybe(_dt.date(1960, 1, 1) + _dt.timedelta(days=r.randint(0, 40000)), 0.05) for _ in range(n)], pa.date32()),
        "ts_us": pa.array([_dt.datetime(1950, 1, 1) + _dt.timedelta(seconds=r.randint(0, 3 * 10**9)) for _ in range(n)], pa.timestamp("us")),
        "ts_ms": pa.array([_dt.datetime(2020, 1, 1) + _dt.timedelta(seconds=r.randint(-10**8, 10**8)) for _ in range(n)], pa.timestamp("ms")),
        "ts_ns": pa.array([_dt.datetime(2001, 1, 1) + _dt.timedelta(seconds=r.randint(0, 10**8)) for _ in range(n)], pa.timestamp("ns")),
        "b": pa.array([maybe(r.random() < 0.5, 0.1) for _ in range(n)], pa.bool_()),
        "s": pa.array([maybe(r.choice(["West", "East", "", "Ünïcode ✓", "x" * 300]), 0.1) for _ in range(n)], pa.string()),
        "s_many": pa.array([f"key-{r.randint(0, n)}" for _ in range(n)], pa.string()),
    })


def check_equal(dumper, path, table):
    cols, rows = mojo_read(dumper, path)
    assert [c[0] for c in cols] == table.column_names
    want = table.to_pylist()
    assert len(rows) == len(want)
    for i, (g, w) in enumerate(zip(rows, want)):
        for c, name in enumerate(table.column_names):
            assert g[c] == stored(w[name], table.schema.field(name).type), (i, name, g[c], w[name])


@pytest.mark.parametrize("opts", [
    dict(compression="zstd"),
    dict(compression="snappy"),
    dict(compression="none"),
    dict(compression="zstd", use_dictionary=False),
    dict(compression="snappy", data_page_version="2.0"),
    dict(compression="zstd", data_page_version="2.0", use_dictionary=False),
    dict(compression="none", use_dictionary=False, column_encoding={"i64": "DELTA_BINARY_PACKED", "i32": "DELTA_BINARY_PACKED"}),
    dict(compression="zstd", row_group_size=777, data_page_size=4096),
])
def test_mojo_reads_what_pyarrow_wrote(dumper, tmp_path, opts):
    t = sample(5000)
    path = tmp_path / "p.parquet"
    pq.write_table(t, path, **opts)
    check_equal(dumper, path, t)


def test_mojo_reads_what_duckdb_wrote(dumper, tmp_path):
    t = sample(3000, seed=9)
    path = tmp_path / "d.parquet"
    con = duckdb.connect()
    con.register("t", t)
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")
    back = pq.read_table(path)            # DuckDB's own types for the same values (its timestamps, decimals)
    check_equal(dumper, path, back)


def test_round_trip_keeps_the_omx_metadata(writer, dumper, tmp_path):
    path = tmp_path / "rt.parquet"
    subprocess.run([str(writer), str(path), "zstd", "2000", "512"], check=True)
    cols, rows = mojo_read(dumper, path)
    assert cols == [("qty", "INT", 0), ("revenue", "DECIMAL", 2), ("region", "TEXT", 0), ("day", "DATE", 0), ("paid", "BOOL", 0)]
    want = expected(2000)
    for g, w in zip(rows, want):
        assert g == [w["qty"], w["revenue"], w["region"], w["day"], None if w["paid"] is None else int(w["paid"])]


def test_unsupported_columns_raise_readably(dumper, tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array([1.5, 2.5])}), path)
    p = subprocess.run([str(dumper), str(path)], capture_output=True, text=True)
    assert p.returncode != 0 and "column `x` is DOUBLE: MX frame has no kind for it yet" in (p.stdout + p.stderr)


# ── readers' own corner cases: what other writers produce ────────────────────────────────────────────

I64 = [-2**63, 2**63 - 1, 0, -2**63, 2**63 - 1, 1, -1]
I32 = [-2**31, 2**31 - 1, 0, -2**31, 5, 2**31 - 1, 0]


def write_and_check(dumper, tmp_path, table, **opts):
    path = tmp_path / "w.parquet"
    pq.write_table(table, path, **opts)
    check_equal(dumper, path, table)


def test_small_and_unsigned_integers_keep_their_values(dumper, tmp_path):
    t = pa.table({
        "i8": pa.array([-128, 127, None], pa.int8()), "i16": pa.array([-32768, 32767, 7], pa.int16()),
        "u8": pa.array([255, 0, 7], pa.uint8()), "u16": pa.array([65535, 0, 7], pa.uint16()),
        "u32": pa.array([3_000_000_000, 2**32 - 1, None], pa.uint32()),
        "u64": pa.array([0, 2**63 - 1, 5], pa.uint64()),
    })
    for opts in (dict(), dict(use_dictionary=False), dict(data_page_version="2.0"),
                 dict(use_dictionary=False, column_encoding={"u32": "DELTA_BINARY_PACKED", "u64": "DELTA_BINARY_PACKED"})):
        write_and_check(dumper, tmp_path, t, **opts)


def test_uint64_past_int64_raises_naming_the_column(dumper, tmp_path):
    for opts in (dict(), dict(use_dictionary=False)):
        path = tmp_path / "u64.parquet"
        pq.write_table(pa.table({"big": pa.array([1, 2**63 + 5], pa.uint64())}), path, **opts)
        refuses(dumper, path, "column `big` is UINT_64 and holds 9223372036854775813, past the int64 range")


def test_text_columns_of_every_shape(dumper, tmp_path):
    for t in (
        pa.table({"s": pa.array([None] * 13, pa.string())}),               # all null: no words at all
        pa.table({"s": pa.array(["w"] * 13)}),                              # one word
        pa.table({"s": pa.array([""] * 11)}),                               # only empty strings
        pa.table({"s": pa.array(["x", None, "", "x"], pa.large_string())}),
        pa.table({"s": pa.array(["a", "b", "a", None]).dictionary_encode()}),
    ):
        for opts in (dict(), dict(use_dictionary=False), dict(data_page_version="2.0")):
            write_and_check(dumper, tmp_path, t, **opts)


def test_all_null_int_column(dumper, tmp_path):
    for opts in (dict(), dict(data_page_version="2.0"), dict(use_dictionary=False)):
        write_and_check(dumper, tmp_path, pa.table({"n": pa.array([None] * 13, pa.int64())}), **opts)


def test_delta_extremes_and_a_single_value(dumper, tmp_path):
    enc = {"i": "DELTA_BINARY_PACKED", "j": "DELTA_BINARY_PACKED"}
    t = pa.table({"i": pa.array(I64, pa.int64()), "j": pa.array(I32, pa.int32())})
    write_and_check(dumper, tmp_path, t, use_dictionary=False, column_encoding=enc)
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array([42], pa.int64())}), use_dictionary=False,
                    column_encoding={"i": "DELTA_BINARY_PACKED"})
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array(range(0, 33_000, 1000), pa.int64())}),
                    use_dictionary=False, column_encoding={"i": "DELTA_BINARY_PACKED"})
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array(I64, pa.int64())}), data_page_version="2.0")


def test_v2_page_of_an_optional_column_without_nulls(dumper, tmp_path):
    t = pa.table({"i": pa.array([1, 2, 3], pa.int64()), "b": pa.array([True, False, True])})
    write_and_check(dumper, tmp_path, t, data_page_version="2.0", compression="zstd")


def test_file_with_a_page_index_and_small_pages(dumper, tmp_path):
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array(range(100), pa.int64())}),
                    write_page_index=True, data_page_size=64, write_batch_size=10)
    t = pa.table({"i": pa.array([None if k % 3 == 0 else k for k in range(13)], pa.int64())})
    write_and_check(dumper, tmp_path, t, data_page_version="1.0", data_page_size=16, write_batch_size=5)


def test_timestamp_in_a_zone_keeps_its_utc_day(dumper, tmp_path):
    # 2020-01-01 23:30 UTC is 18,262 days after 1970; the zone (isAdjustedToUTC) is only a display hint
    t = pa.table({"t": pa.array([_dt.datetime(2020, 1, 1, 23, 30, tzinfo=_dt.timezone.utc)], pa.timestamp("us", tz="Asia/Tokyo"))})
    path = tmp_path / "tz.parquet"
    pq.write_table(t, path)
    assert mojo_read(dumper, path)[1] == [[18262]]


def refuses(dumper, path, *words):
    p = subprocess.run([str(dumper), str(path)], capture_output=True, text=True)
    assert p.returncode == 1, (p.returncode, p.stderr[-300:])       # a normal error exit, never a signal
    for w in words:
        assert w in p.stdout + p.stderr, (w, p.stdout + p.stderr)


def test_columns_without_a_kind_raise_readably(dumper, tmp_path):
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"b": pa.array([b"\xff\xfe", b"ok"], pa.binary())}), path)
    refuses(dumper, path, "column `b` is BINARY (BYTE_ARRAY without a string type)")
    pq.write_table(pa.table({"t": pa.array([_dt.time(1, 2, 3)], pa.time32("ms"))}), path)
    refuses(dumper, path, "column `t` is a TIME")
    pq.write_table(pa.table({"t": pa.array([_dt.time(1, 2, 3)], pa.time64("us"))}), path)
    refuses(dumper, path, "column `t` is a TIME")
    pq.write_table(pa.table({"d": pa.array([_dec.Decimal("1.5")], pa.decimal128(30, 1))}), path)
    refuses(dumper, path, "column `d`: DECIMAL(30, 1) has more digits")


def test_invalid_utf8_in_a_text_column_raises(dumper, tmp_path):
    bad = pa.array([b"ok", b"\xff\xfe"], pa.binary()).view(pa.string())      # a string column holding invalid UTF-8
    for opts in (dict(), dict(use_dictionary=False)):
        path = tmp_path / "u.parquet"
        pq.write_table(pa.table({"s": bad}), path, **opts)
        refuses(dumper, path, "column `s`", "not valid UTF-8")


def test_definition_levels_may_be_bit_packed(dumper, tmp_path):
    # the deprecated BIT_PACKED level encoding: most significant bit first, no length prefix
    body = bytes([0b10100000]) + struct.pack("<qq", 5, 9)                    # levels 1,0,1 (3 rows): values 5, 9
    path = tmp_path / "bp.parquet"
    evil_build(path, body, 3, len(body), len(body), rep=1, def_enc=4)
    assert mojo_read(dumper, path)[1] == [[5], [None], [9]]


# ── corrupt and hostile files: a readable error and a normal exit, never a crash ─────────────────────



def uv(v):
    o = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            o.append(b | 0x80)
        else:
            o.append(b)
            return bytes(o)


def zz(v):
    return uv(((v << 1) ^ (v >> 63)) & ((1 << 64) - 1))


class Compact:
    """A thrift compact-protocol writer, just enough to hand-make Parquet metadata."""
    def __init__(s):
        s.b = bytearray()
        s.last = [0]

    def f(s, fid, t):
        d = fid - s.last[-1]
        if 0 < d <= 15:
            s.b.append((d << 4) | t)
        else:
            s.b.append(t)
            s.b += zz(fid)
        s.last[-1] = fid

    def i32(s, fid, v):
        s.f(fid, 5)
        s.b += zz(v)

    def i64(s, fid, v):
        s.f(fid, 6)
        s.b += zz(v)

    def bin(s, fid, v, rawlen=None):
        s.f(fid, 8)
        s.b += uv(len(v) if rawlen is None else rawlen)
        s.b += v

    def st(s, fid):
        s.f(fid, 12)
        s.last.append(0)

    def lst(s, fid, et, n):
        s.f(fid, 9)
        s.b += bytes([(n << 4) | et]) if n < 15 else bytes([0xF0 | et]) + uv(n)

    def el(s):
        s.last.append(0)

    def end(s):
        s.b.append(0)
        s.last.pop()


def evil_build(path, body, nvals, usize, csize, ptype=2, rep=0, type_length=None, decimal=False, enc=0,
               name=b"x", namelen=None, footer_extra=None, def_enc=3, v2=None, chunk_values=None):
    """One column, one row group, one data page: `body` is the page, the sizes and counts are whatever we say."""
    h = Compact()
    h.i32(1, 3 if v2 else 0)
    h.i32(2, usize)
    h.i32(3, csize)
    if v2:
        h.st(8)
        h.i32(1, nvals)
        h.i32(2, 0)
        h.i32(3, nvals)
        h.i32(4, enc)
        h.i32(5, v2["def_len"])
        h.i32(6, 0)
        h.end()
    else:
        h.st(5)
        h.i32(1, nvals)
        h.i32(2, enc)
        h.i32(3, def_enc)
        h.i32(4, 3)
        h.end()
    h.b.append(0)
    data = b"PAR1"
    off = len(data)
    data += bytes(h.b) + body
    m = Compact()
    m.i32(1, 1)
    m.lst(2, 12, 2)
    m.el(); m.bin(4, b"schema"); m.i32(5, 1); m.end()
    m.el(); m.i32(1, ptype)
    if type_length is not None:
        m.i32(2, type_length)
    m.i32(3, rep)
    m.bin(4, name, namelen)
    if decimal:
        m.i32(6, 5); m.i32(7, 2); m.i32(8, 18)
    m.end()
    m.i64(3, nvals)
    m.lst(4, 12, 1)
    m.el(); m.lst(1, 12, 1)
    m.el(); m.i64(2, off)
    m.st(3); m.i32(1, ptype); m.lst(2, 5, 1); m.b += zz(0); m.lst(3, 8, 1); m.b += uv(1) + b"x"
    m.i32(4, 0); m.i64(5, nvals if chunk_values is None else chunk_values)
    m.i64(6, len(body)); m.i64(7, len(body)); m.i64(9, off); m.end(); m.end()
    m.i64(2, len(body)); m.i64(3, nvals); m.end()
    if footer_extra:
        footer_extra(m)
    m.b.append(0)
    data += bytes(m.b) + struct.pack("<i", len(m.b)) + b"PAR1"
    Path(path).write_bytes(data)


OK3 = struct.pack("<qqq", 1, 2, 3)
RUN_2_40 = uv((1 << 40) << 1) + b"\x01"                                       # an RLE run of 2^40 ones


def unknown_field(kind):
    def add(m):
        if kind == "binary_len_max":                                           # a skipped binary of 2^64-1 bytes
            m.f(99, 8); m.b += uv((1 << 64) - 1)
        elif kind == "list_size_huge":
            m.f(99, 9); m.b += bytes([0xF5]) + uv(1 << 40)
        elif kind == "binary_past_end":
            m.f(99, 8); m.b += uv(1 << 30)
        elif kind == "struct_bomb":                                            # 100,000 nested structs
            m.f(99, 12); m.b += bytes([0x1C]) * 100_000
        elif kind == "double_past_end":
            m.f(99, 7)
    return add


CORRUPT = {
    "uncompressed page claims more bytes than it has": dict(body=OK3, nvals=3, usize=200_000_000, csize=24),
    "uncompressed sizes disagree": dict(body=OK3, nvals=3, usize=16, csize=24),
    "negative compressed size": dict(body=OK3, nvals=3, usize=24, csize=-5),
    "negative uncompressed size": dict(body=OK3, nvals=3, usize=-24, csize=24),
    "column name length 2^64-1": dict(body=OK3, nvals=3, usize=24, csize=24, namelen=(1 << 64) - 1),
    "v1 level length past the page": dict(body=struct.pack("<I", 0x7FFFFFF0) + b"\x03\x01" + OK3, nvals=3,
                                          usize=30, csize=30, rep=1),
    "v2 level length past the page": dict(body=b"\x03\x01" + OK3, nvals=3, usize=26, csize=26, rep=1,
                                          v2=dict(def_len=2_000_000_000)),
    "v2 negative level length": dict(body=OK3, nvals=3, usize=24, csize=24, rep=1, v2=dict(def_len=-4)),
    "dictionary index run longer than the page": dict(body=b"\x08" + uv(((1 << 57) << 1) | 1) + b"\x00" * 8,
                                                      nvals=3, usize=17, csize=17, enc=8),
    "dictionary index width 65": dict(body=b"\x41" + uv(3 << 1) + b"\x00" * 9, nvals=3, usize=11, csize=11, enc=8),
    "fixed-length decimal short of its values": dict(body=struct.pack(">q", 7), nvals=1000, usize=8, csize=8,
                                                     ptype=7, type_length=8, decimal=True),
    "fixed-length decimal with length zero": dict(body=struct.pack(">q", 7), nvals=1, usize=8, csize=8,
                                                  ptype=7, type_length=0, decimal=True),
    "delta total of 2^40": dict(body=uv(128) + uv(4) + uv(1 << 40) + zz(0) + zz(0) + b"\x00" * 4, nvals=3,
                                usize=12, csize=12, enc=5),
    "delta with no miniblocks": dict(body=uv(128) + uv(0) + uv(3) + zz(0) + zz(0), nvals=3, usize=6, csize=6, enc=5),
    "delta miniblock width 65": dict(body=uv(128) + uv(4) + uv(3) + zz(0) + zz(0) + bytes([65, 0, 0, 0]) + b"\xff" * 300,
                                     nvals=3, usize=312, csize=312, enc=5),
    "more values in the page than in the column chunk": dict(body=OK3, nvals=3, usize=24, csize=24, chunk_values=2),
    "column name that is not UTF-8": dict(body=OK3, nvals=3, usize=24, csize=24, name=b"\xff\xfe"),
    "unknown physical type 9": dict(body=OK3, nvals=3, usize=24, csize=24, ptype=9),
    "no physical type": dict(body=OK3, nvals=3, usize=24, csize=24, ptype=-1),
    "skipped binary of 2^64-1 bytes": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("binary_len_max")),
    "skipped list of 2^40 elements": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("list_size_huge")),
    "skipped binary past the footer": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("binary_past_end")),
    "skipped double past the footer": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("double_past_end")),
    "100,000 nested structs": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("struct_bomb")),
}


@pytest.mark.parametrize("what", sorted(CORRUPT))
def test_corrupt_file_raises_instead_of_crashing(dumper, tmp_path, what):
    path = tmp_path / "evil.parquet"
    evil_build(path, **CORRUPT[what])
    refuses(dumper, path, "corrupt")


def test_runs_longer_than_the_values_still_needed_are_cut_not_allocated(dumper, tmp_path):
    # definition levels as one RLE run of 2^40 ones, three rows: reads as three present values
    body = struct.pack("<I", len(RUN_2_40)) + RUN_2_40 + OK3
    path = tmp_path / "run.parquet"
    evil_build(path, body, 3, len(body), len(body), rep=1)
    assert mojo_read(dumper, path)[1] == [[1], [2], [3]]


def test_truncated_files_raise(dumper, tmp_path):
    good = tmp_path / "good.parquet"
    pq.write_table(pa.table({"i": pa.array(range(50), pa.int64())}), good)
    raw = good.read_bytes()
    for cut in (0, 3, 11, 12, len(raw) // 2, len(raw) - 9, len(raw) - 1):
        path = tmp_path / f"cut{cut}.parquet"
        path.write_bytes(raw[:cut])
        refuses(dumper, path)


def test_random_byte_flips_never_crash_the_reader(dumper, tmp_path):
    good = tmp_path / "good.parquet"
    t = sample(60, seed=5).select(["i64", "dec2", "d", "b", "s"])
    pq.write_table(t, good, compression="snappy")
    raw = bytearray(good.read_bytes())
    r = _random.Random(11)
    for _ in range(150):
        b = bytearray(raw)
        for _ in range(r.randint(1, 3)):
            b[r.randrange(len(b))] = r.randrange(256)
        path = tmp_path / "flip.parquet"
        path.write_bytes(bytes(b))
        p = subprocess.run([str(dumper), str(path)], capture_output=True, timeout=60)
        assert p.returncode in (0, 1), (p.returncode, p.stderr[-300:])      # read it or refused it; no signal


# ── writing: refusals, empty dictionaries, row-group sizes ───────────────────────────────────────────

@pytest.fixture(scope="module")
def edge_writer(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_write_edge"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_write_edge.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def test_decimal_past_18_digits_is_refused_naming_the_column(edge_writer, tmp_path):
    path = tmp_path / "d.parquet"
    p = subprocess.run([str(edge_writer), "decimal_overflow", str(path)], capture_output=True, text=True)
    assert p.returncode == 1
    assert "column `money`: 1000000000000000000 does not fit DECIMAL(18, 2)" in p.stdout + p.stderr
    assert not path.exists()


def test_table_without_columns_is_refused(edge_writer, tmp_path):
    path = tmp_path / "none.parquet"
    p = subprocess.run([str(edge_writer), "no_columns", str(path)], capture_output=True, text=True)
    assert p.returncode == 1
    assert "a table with no columns cannot be written as Parquet" in p.stdout + p.stderr
    assert not path.exists()


def test_all_null_text_without_words_in_row_groups_of_8(edge_writer, tmp_path):
    path = tmp_path / "n.parquet"
    subprocess.run([str(edge_writer), "null_text", str(path)], check=True)
    t = pq.read_table(path)
    assert pq.read_metadata(path).num_row_groups == 2
    assert t.column("id").to_pylist() == list(range(13))
    assert t.column("name").to_pylist() == [None] * 13
    assert t.schema.field("name").type == pa.string()
    con = duckdb.connect()
    assert con.execute(f"SELECT count(*), count(name) FROM read_parquet('{path}')").fetchone() == (13, 0)


def test_a_refused_write_leaves_the_existing_file_and_no_temp_file(edge_writer, tmp_path):
    path = tmp_path / "keep.omx"
    p = subprocess.run([str(edge_writer), "atomic", str(path)], capture_output=True, text=True, check=True)
    assert "refused: column `money`" in p.stdout
    assert pq.read_table(path).column("qty").to_pylist() == [41, 42]
    assert sorted(x.name for x in tmp_path.iterdir()) == ["keep.omx"]  # no temp file left, whatever its name


def test_footer_names_the_writer_and_uncompressed_row_group_size(writer, tmp_path):
    path = tmp_path / "m.parquet"
    subprocess.run([str(writer), str(path), "zstd", "3000", "1024"], check=True)
    md = pq.read_metadata(path)
    assert md.created_by == "MX frame version 0.6.0"
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        assert rg.total_byte_size == sum(rg.column(c).total_uncompressed_size for c in range(rg.num_columns))


# ── F-140: a query streamed over row groups equals read_parquet + the same query ─────────────────────

SCAN_WORDS = ["West", "East", "Nörth", "", "South-East", "Far", "Central", "Zed"]
SCAN_ROWS, SCAN_GROUP = 36_000, 4_000


def scan_table(n=SCAN_ROWS, group=SCAN_GROUP, many_words=False, seed=11):
    """region (string; each row group draws from a different window of words, so row-group dictionaries differ from
    each other and from the file's), carrier int64 -1..9 (null for the whole of row group 5), day date32, revenue
    decimal(18, 2) (null for the whole of row group 2), cost int64."""
    r = _random.Random(seed)
    region, carrier, day, revenue, cost = [], [], [], [], []
    for i in range(n):
        rg = i // group
        if many_words:
            region.append(None if r.random() < 0.03 else f"w{r.randint(0, 2999)}")
        else:
            start = (rg * 3) % 8
            region.append(None if r.random() < 0.05 else SCAN_WORDS[(start + r.randint(0, 2)) % 8])
        carrier.append(None if rg == 5 or r.random() < 0.03 else r.randint(-1, 9))
        day.append(_dt.date(1970, 1, 1) + _dt.timedelta(days=r.randint(9000, 15000)))
        revenue.append(None if rg == 2 or r.random() < 0.1 else
                       _dec.Decimal(5_000_000_000_000_000 + r.randint(0, 99) if i % 400 == 0 else r.randint(-10**6, 10**6)).scaleb(-2))
        cost.append(r.randint(-5000, 5000))
    return pa.table({
        "region": pa.array(region, pa.string()), "carrier": pa.array(carrier, pa.int64()),
        "day": pa.array(day, pa.date32()), "revenue": pa.array(revenue, pa.decimal128(18, 2)),
        "cost": pa.array(cost, pa.int64()),
    })


@pytest.fixture(scope="module")
def prober(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_scan_probe"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_scan_probe.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def probe(prober, path, mode, query):
    p = subprocess.run([str(prober), str(path), mode, query], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.strip().split("\n")[-1])


SCAN_QUERIES = ["by_region", "filtered", "pivot", "bins", "global"]


def same_answer(a, b):
    """Every number a caller can read, exactly (not the time or the memory)."""
    for k in ("words", "kinds", "slot_of", "count_slot_of", "counts", "slots", "present"):
        assert a[k] == b[k], k
    assert a["device"] == b["device"] == 0


def agg_values(res, i, g):
    kind, so, cso = res["kinds"][i], res["slot_of"][i], res["count_slot_of"][i]
    if kind == 1:                                       # count()
        return res["counts"][g]
    if kind == 2:                                       # count(e)
        return res["slots"][cso][g] if cso >= 0 else res["counts"][g]
    return res["slots"][so][g]


def check_against_duckdb(path, res, query):
    """The query's numbers per group, from DuckDB on the same file."""
    words = res["words"].split("|")
    cents = "CAST(revenue * 100 AS BIGINT)"
    day = "(CAST(day AS DATE) - DATE '1970-01-01')"
    con = duckdb.connect()
    src = f"read_parquet('{path}')"
    if query == "by_region":
        rows = con.execute(f"SELECT region, sum({cents}), count(*), count(revenue), min(cost), max(cost), min({cents}), max({cents}) "
                           f"FROM {src} WHERE region IS NOT NULL GROUP BY region").fetchall()
        for region, s, n, nv, lo, hi, rlo, rhi in rows:
            g = words.index(region)
            assert res["counts"][g] == n
            assert agg_values(res, 1, g) == n and agg_values(res, 2, g) == nv
            assert agg_values(res, 3, g) == lo and agg_values(res, 4, g) == hi
            if nv:
                assert agg_values(res, 0, g) == int(s)
                assert agg_values(res, 6, g) == rlo and agg_values(res, 7, g) == rhi
                assert agg_values(res, 5, g) == int(s)                       # mean's sum
            assert res["present"][6][g] == (1 if nv else 0)
        assert sum(res["counts"]) == sum(r[2] for r in rows)
    elif query == "filtered":
        rows = con.execute(f"SELECT carrier, sum({cents}), count(*), max(cost) FROM {src} "
                           f"WHERE region = 'East' AND {day} BETWEEN 10000 AND 14000 AND carrier BETWEEN 0 AND 7 GROUP BY carrier").fetchall()
        got = {c: (agg_values(res, 0, c), res["counts"][c], agg_values(res, 2, c)) for c in range(8) if res["counts"][c]}
        assert got == {c: (int(s) if s is not None else 0, n, hi) for c, s, n, hi in rows}
    elif query == "pivot":
        rows = con.execute(f"SELECT region, carrier, sum(cost), count(revenue) FROM {src} WHERE region IS NOT NULL "
                           f"AND carrier BETWEEN 0 AND 7 GROUP BY region, carrier").fetchall()
        got = {g: (agg_values(res, 0, g), agg_values(res, 1, g)) for g in range(len(res["counts"])) if res["counts"][g]}
        assert got == {words.index(r) * 8 + c: (s, nv) for r, c, s, nv in rows}
    elif query == "bins":
        rows = con.execute(f"SELECT least({day} // 30, 99), count(*), sum(cost) FROM {src} WHERE {day} IS NOT NULL GROUP BY 1").fetchall()
        got = {g: (res["counts"][g], agg_values(res, 1, g)) for g in range(100) if res["counts"][g]}
        assert got == {b: (n, s) for b, n, s in rows}
    elif query == "global":
        s, n, rlo, rhi = con.execute(f"SELECT sum(cost), count(*), min({cents}), max({cents}) FROM {src}").fetchone()
        assert (agg_values(res, 0, 0), agg_values(res, 1, 0), agg_values(res, 2, 0), agg_values(res, 3, 0)) == (s, n, rlo, rhi)


PYARROW_FILES = {
    "zstd-dict-v1": dict(compression="zstd", use_dictionary=True, data_page_version="1.0"),
    "snappy-dict-v2": dict(compression="snappy", use_dictionary=True, data_page_version="2.0"),
    "none-plain-v1": dict(compression="none", use_dictionary=False, data_page_version="1.0"),
    "zstd-plain-v2": dict(compression="zstd", use_dictionary=False, data_page_version="2.0"),
    "small-pages": dict(compression="snappy", use_dictionary=True, data_page_size=2_000),
}


def write_with_empty_groups(path, table, **opts):
    """Row groups of SCAN_GROUP rows, with an empty row group before the first, in the middle and after the last."""
    w = pq.ParquetWriter(path, table.schema, **opts)
    w.write_table(table.slice(0, 0))
    for lo in range(0, table.num_rows, SCAN_GROUP):
        w.write_table(table.slice(lo, SCAN_GROUP))
        if lo == 3 * SCAN_GROUP:
            w.write_table(table.slice(0, 0))
    w.write_table(table.slice(0, 0))
    w.close()


@pytest.mark.parametrize("name", sorted(PYARROW_FILES))
def test_streamed_query_equals_read_parquet_on_pyarrow_files(prober, tmp_path, name):
    path = tmp_path / "t.parquet"
    write_with_empty_groups(path, scan_table(), **PYARROW_FILES[name])
    assert pq.read_metadata(path).num_row_groups == 12                   # 9 + 3 empty
    for q in SCAN_QUERIES:
        scanned, read = probe(prober, path, "scan", q), probe(prober, path, "read", q)
        same_answer(scanned, read)
        check_against_duckdb(path, scanned, q)


def test_text_that_falls_back_to_plain_pages_streams_the_same_codes(prober, tmp_path):
    """3,000 distinct words with a 1 KB dictionary limit: each chunk starts dictionary-encoded and finishes PLAIN."""
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(many_words=True), path, row_group_size=SCAN_GROUP, dictionary_pagesize_limit=1024)
    encodings = pq.read_metadata(path).row_group(0).column(0).encodings
    assert "PLAIN" in encodings and "RLE_DICTIONARY" in encodings
    for q in ("by_region", "big"):
        scanned, read = probe(prober, path, "scan", q), probe(prober, path, "read", q)
        same_answer(scanned, read)
    check_against_duckdb(path, probe(prober, path, "scan", "by_region"), "by_region")


@pytest.mark.parametrize("codec", ["zstd", "snappy", "none"])
def test_streamed_query_equals_read_parquet_on_duckdb_files(prober, tmp_path, codec):
    src = tmp_path / "src.parquet"
    pq.write_table(scan_table(), src)
    path = tmp_path / "t.parquet"
    duckdb.connect().execute(f"COPY (SELECT * FROM read_parquet('{src}')) TO '{path}' "
                             f"(FORMAT parquet, ROW_GROUP_SIZE {SCAN_GROUP}, COMPRESSION {'uncompressed' if codec == 'none' else codec})")
    assert pq.read_metadata(path).num_row_groups == 9
    for q in SCAN_QUERIES:
        scanned, read = probe(prober, path, "scan", q), probe(prober, path, "read", q)
        same_answer(scanned, read)
        check_against_duckdb(path, scanned, q)


def test_a_file_with_no_rows_and_a_file_that_is_not_parquet(prober, tmp_path):
    path = tmp_path / "e.parquet"
    pq.write_table(scan_table(0), path)
    same_answer(probe(prober, path, "scan", "global"), probe(prober, path, "read", "global"))
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file at all")
    p = subprocess.run([str(prober), str(bad), "scan", "global"], capture_output=True, text=True)
    assert p.returncode != 0 and "not a Parquet file" in p.stdout + p.stderr


def test_a_truncated_file_raises_instead_of_crashing(prober, tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(), path, row_group_size=SCAN_GROUP)
    data = path.read_bytes()
    for cut in (len(data) // 2, len(data) - 100, 20):
        bad = tmp_path / f"cut{cut}.parquet"
        bad.write_bytes(data[:cut])
        p = subprocess.run([str(prober), str(bad), "scan", "by_region"], capture_output=True, text=True)
        assert p.returncode == 1 and "Unhandled exception" in p.stdout + p.stderr, (cut, p.returncode)


def test_random_byte_flips_never_crash_a_scan(prober, tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(8_000), path, row_group_size=2_000)
    data = bytearray(path.read_bytes())
    r = _random.Random(5)
    for k in range(60):
        flipped = bytearray(data)
        for _ in range(3):
            flipped[r.randrange(len(flipped))] ^= 1 << r.randrange(8)
        bad = tmp_path / "flip.parquet"
        bad.write_bytes(flipped)
        p = subprocess.run([str(prober), str(bad), "scan", "by_region"], capture_output=True, text=True)
        assert p.returncode in (0, 1), (k, p.returncode)                 # an answer or a readable error, never a signal


def test_an_unused_unique_id_text_column_costs_nothing(prober, tmp_path):
    """500k unique 30-byte ids that the writer's dictionary limit pushed to PLAIN pages, never read by the query: the
    stream stays near one row group, where the whole-file read holds the ids."""
    path = tmp_path / "hc.parquet"
    n, g = 2_000_000, 250_000
    w = pq.ParquetWriter(path, pa.schema([("t", pa.string()), ("v", pa.int64())]), compression="none")
    for lo in range(0, n, g):
        ids = pa.array([f"id-{i:012d}-padpadpad" for i in range(lo, lo + g)])
        w.write_table(pa.table([ids, pa.array(range(lo, lo + g), pa.int64())], schema=w.schema))
    w.close()
    scanned, read = probe(prober, path, "scan", "sum_v"), probe(prober, path, "read", "sum_v")
    same_answer(scanned, read)
    assert scanned["slots"][0][0] == n * (n - 1) // 2 and scanned["counts"][0] == n
    assert scanned["peak_kb"] < 90_000 and scanned["peak_kb"] * 4 < read["peak_kb"], (scanned["peak_kb"], read["peak_kb"])


def test_the_serial_build_decodes_the_same_numbers_as_the_parallel_one(prober, tmp_path, monkeypatch):
    """F-141: pages decode in parallel; the serial build (no thread pool) runs them in a loop. Many small pages, nulls,
    dictionary and plain text: every answer is identical, streamed and whole-file."""
    monkeypatch.setenv("MXFRAME_BUILD_FLAGS", "-D MXFRAME_SERIAL")
    serial = tmp_path / "pq_scan_probe_serial"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_scan_probe.mojo"), str(serial)],
                   check=True, capture_output=True, text=True)
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(), path, row_group_size=SCAN_GROUP, data_page_size=1_500, compression="snappy")
    for q in SCAN_QUERIES:
        for mode in ("scan", "read"):
            same_answer(probe(prober, path, mode, q), probe(serial, path, mode, q))
