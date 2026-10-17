"""F-074: a plain string join key runs without Arrow row work and equals DuckDB.

`join_engine.encode_keys` turned every string key into dense int32 codes with Arrow's `dictionary_encode`
over both sides (plus a `fill_null`): a compute call over every row of the join, although the join on
the codes was native.  The codes now come from one native call (`mx_join_key_codes`: a hash table over the
key tuples of both sides, strings compared by their bytes).  A left join gathered both sides with Arrow's
`take` (for any key type); it gathers natively now, the unmatched rows becoming NULL in NumPy.

Every case asserts what a user sees (the rows DuckDB returns) and that no Arrow compute call touched
>= 1,000 rows.  A NULL key never matches (inner, left, semi, anti); keys are non-ASCII, repeat, and
exist on one side only."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe._lazy_pc as _lazy_pc
import mxframe.custom_ops as co
import mxframe.native_exec as nx
from mxframe import LazyFrame, col
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

MIN_ROWS = 1000


def _gpu_usable() -> bool:
    try:
        return co._max_accelerator() is not None
    except Exception:
        return False


DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU"))]
HOWS = ["inner", "left", "semi", "anti"]


LOG = []        # one list for every test: a wrapper another module cached in an earlier test still logs here


@pytest.fixture
def spy(monkeypatch):
    """Arrow compute calls over >= MIN_ROWS rows (what ran outside Mojo)."""
    log = LOG
    log.clear()
    co._JOIN_RESULT_CACHE.clear()            # a cached join result runs nothing: it would hide the Arrow calls
    import pyarrow._compute as _c
    real_call = _c.call_function

    def note(name, args):
        if name == "take" and len(args) > 1:
            args = args[1:2]
        n = 0
        for a in args:
            try:
                n = max(n, len(a))
            except TypeError:
                pass
        if n >= MIN_ROWS:
            log.append(f"{name}[{n}]")

    monkeypatch.setattr(_c, "call_function", lambda name, args, *a, **k: (note(name, args), real_call(name, args, *a, **k))[1])
    rw, rr = _lazy_pc._wrapper, _lazy_pc._real

    def logged(name, f):
        if f is None or name in _lazy_pc._REBUILT:
            return f
        return lambda *args, **kw: (note(name, args), f(*args, **kw))[1]
    monkeypatch.setattr(_lazy_pc, "_wrapper", lambda name: logged(name, rw(name)))
    monkeypatch.setattr(_lazy_pc, "_real", lambda name: logged(name, rr(name)))
    _lazy_pc.pc._cache.clear()
    yield log
    _lazy_pc.pc._cache.clear()


WORDS = ["", "a", "é", "straße", "日本語", "naïve café", "Ω≈ç√", "x" * 40, "😀 emoji", "plain"]


def _keys(n, rng, distinct, tag):
    """n non-ASCII string keys drawn from `distinct` values; the same value set on every call."""
    pool = np.array([f"{WORDS[i % len(WORDS)]}-{i:04d}" for i in range(distinct)], dtype=object)
    return pool[rng.integers(0, distinct, n)]


def _str_col(values, nulls, typ, chunks=1):
    arr = pa.array([None if m else v for v, m in zip(values, nulls)], type=pa.string())
    if typ == "dict":
        arr = arr.dictionary_encode()
    elif typ == "large":
        arr = arr.cast(pa.large_string())
    if chunks == 1:
        return arr
    step = -(-len(arr) // chunks)
    return pa.chunked_array([arr.slice(i, step) for i in range(0, len(arr), step)])


def _tables(ltype="plain", rtype="plain", chunks=1, composite=None, n_left=6000, n_right=2500):
    """l(k, [j], id) and r(rk, [rj], rid, rv): ~40% of the right keys exist on the left, NULL keys on both
    sides, repeated keys on both sides, keys that only one side has."""
    rng = np.random.default_rng(74)
    lk = _keys(n_left, rng, 900, "l")
    rk = _keys(n_right, rng, 1500, "r")                  # keys past 900 exist only on the right side
    ln = rng.random(n_left) < 0.05
    rn = rng.random(n_right) < 0.05
    lt = {"k": _str_col(lk, ln, ltype, chunks), "id": A(np.arange(n_left, dtype=np.int64))}
    rt = {"rk": _str_col(rk, rn, rtype, chunks), "rid": A(np.arange(n_right, dtype=np.int64)),
          "rv": A(rng.integers(0, 50, n_right).astype(np.int64))}
    if composite == "int":
        lj, rj = rng.integers(0, 3, n_left), rng.integers(0, 3, n_right)
        lt["j"], rt["rj"] = A(lj.astype(np.int64)), A(rj.astype(np.int64))
    elif composite == "str":
        lj, rj = _keys(n_left, rng, 4, "j"), _keys(n_right, rng, 4, "j")
        lt["j"], rt["rj"] = _str_col(lj, rng.random(n_left) < 0.03, ltype), _str_col(rj, rng.random(n_right) < 0.03, rtype)
    return pa.table(lt), pa.table(rt)


def _duck_tables(l, r):
    con = duckdb.connect()
    for name, t in (("l", l), ("r", r)):
        t = pa.table({n: (t.column(n).cast(pa.string()) if pa.types.is_dictionary(t.column(n).type) else t.column(n))
                      for n in t.column_names})
        con.register(name + "_arrow", t)
        con.execute(f"create table {name} as select * from {name}_arrow")
    return con


def _on(composite):
    left = ["k"] + (["j"] if composite else [])
    right = ["rk"] + (["rj"] if composite else [])
    return left, right


def _sql_on(composite):
    return " and ".join(f"l.{a} = r.{b}" for a, b in zip(*_on(composite)))


def _check(l, r, how, device, spy, composite=None, with_key=False):
    """The join's rows next to DuckDB's; `with_key` also returns the (string) key column."""
    con = _duck_tables(l, r)
    on = _sql_on(composite)
    cols = ["id", "rid"] if how in ("inner", "left") else ["id"]
    if with_key:
        cols.insert(0, "k")
    sel = ", ".join(("l." if c in ("k", "id") else "r.") + c for c in cols) if how in ("inner", "left") \
        else ", ".join(cols)
    if how == "inner":
        sql = f"select {sel} from l join r on {on}"
    elif how == "left":
        sql = f"select {sel} from l left join r on {on}"
    else:
        neg = "not " if how == "anti" else ""
        sql = f"select {sel} from l where {neg}exists (select 1 from r where {on})"
    want = sorted(con.execute(sql).fetchall(), key=repr)
    left_on, right_on = _on(composite)
    co._JOIN_RESULT_CACHE.clear()
    spy.clear()                  # (building the tables and DuckDB's answer used Arrow compute)
    q = LazyFrame(l).join(LazyFrame(r), left_on=left_on, right_on=right_on, how=how)
    out = q.select(*[col(c) for c in cols]).compute(device=device)
    got = sorted(zip(*[out.column(c).to_pylist() for c in cols]), key=repr)
    assert got == want
    assert spy == [], spy
    return len(want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("how", HOWS)
def test_plain_string_key(how, device, spy):
    l, r = _tables()
    assert _check(l, r, how, device, spy, with_key=True) > 100


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("how", HOWS)
def test_large_string_key(how, device, spy):
    l, r = _tables("large", "large")
    assert _check(l, r, how, device, spy) > 100


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("types", [("dict", "plain"), ("plain", "dict"), ("dict", "dict"), ("large", "plain")],
                         ids=["dict-plain", "plain-dict", "dict-dict", "large-plain"])
def test_dictionary_or_large_against_plain(types, how, device, spy):
    l, r = _tables(*types)
    assert _check(l, r, how, device, spy) > 100


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("how", HOWS)
def test_multi_chunk_key_columns(how, device, spy):
    l, r = _tables(chunks=3)
    assert l.column("k").num_chunks == 3 and r.column("rk").num_chunks == 3
    assert _check(l, r, how, device, spy, with_key=True) > 100


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("composite", ["int", "str"])
def test_composite_keys(composite, how, device, spy):
    l, r = _tables(composite=composite)
    assert _check(l, r, how, device, spy, composite=composite) > 100


@pytest.mark.parametrize("how", HOWS)
def test_a_null_key_never_matches(how, spy):
    """Both sides hold NULL keys (and a key the other side lacks): DuckDB drops them from inner / semi,
    keeps the left rows (with NULL partners) for left and anti."""
    n = 3000
    lk = [None if i % 4 == 0 else f"é{i % 50}" for i in range(n)]
    rk = [None if i % 3 == 0 else f"é{i % 80}" for i in range(n)]
    l = pa.table({"k": pa.array(lk), "id": A(np.arange(n, dtype=np.int64))})
    r = pa.table({"rk": pa.array(rk), "rid": A(np.arange(n, dtype=np.int64)), "rv": A(np.arange(n, dtype=np.int64))})
    assert _check(l, r, how, "cpu", spy) > 100


def test_multi_chunk_dictionaries_with_different_values(spy):
    """Chunks of one dictionary column carry their own dictionaries (a Parquet scan does that)."""
    a = pa.array(["é1", "é2", "é1", "é3"] * 400).dictionary_encode()
    b = pa.array(["é3", "é4", "é2", "é4"] * 400).dictionary_encode()
    l = pa.table({"k": pa.chunked_array([a, b]), "id": A(np.arange(3200, dtype=np.int64))})
    r = pa.table({"rk": pa.array(["é2", "é4", "é9"] * 400), "rid": A(np.arange(1200, dtype=np.int64)),
                  "rv": A(np.arange(1200, dtype=np.int64))})
    assert _check(l, r, "inner", "cpu", spy) > 100


@pytest.mark.parametrize("distinct", [20000, 3], ids=["many-keys", "three-keys-skewed"])
def test_a_join_big_enough_for_every_worker(distinct, spy):
    """300,000 + 60,000 rows span several 64k-row blocks, so the kernel hashes, scatters and builds on all its
    threads; three keys put every row into three partitions (the skewed case).  The count and a checksum of
    the joined pairs are DuckDB's."""
    rng = np.random.default_rng(5)
    pool = np.array([f"é{i}-{'y' * (i % 17)}" for i in range(distinct + 2)], dtype=object)
    lk = pool[rng.integers(0, distinct, 300_000)]
    n_right = 60_000 if distinct > 100 else 4                 # (3 keys x 60,000 rows would join to billions of pairs)
    rk = pool[rng.integers(1, distinct + 2, n_right)]          # key 0 only on the left, the last two only on the right
    l = pa.table({"k": pa.array(lk.tolist()), "id": A(np.arange(300_000, dtype=np.int64))})
    r = pa.table({"rk": pa.array(rk.tolist()), "rid": A(np.arange(n_right, dtype=np.int64))})
    con = _duck_tables(l, r)
    want = con.execute("select count(*), sum(id * 7 + rid) from l join r on l.k = r.rk").fetchone()
    co._JOIN_RESULT_CACHE.clear()
    spy.clear()
    out = LazyFrame(l).join(LazyFrame(r), left_on="k", right_on="rk").select(col("id"), col("rid")).compute(device="cpu")
    ids, rids = out.column("id").to_numpy(), out.column("rid").to_numpy()
    assert spy == [], spy
    assert (out.num_rows, int((ids * 7 + rids).sum())) == (want[0], int(want[1]))


def test_a_date_key_never_meets_an_integer_key_in_a_composite():
    """(str, date32) against (str, int32) must raise like the single-key join and like DuckDB, not match
    the day number (1,500 silent rows before)."""
    a = pa.table({"s": pa.array(["x"] * 1500), "d": pa.array(np.arange(1500, dtype=np.int32)).cast(pa.date32()),
                  "id": A(np.arange(1500, dtype=np.int64))})
    b = pa.table({"s": pa.array(["x"] * 1500), "i": A(np.arange(1500, dtype=np.int32)),
                  "rid": A(np.arange(1500, dtype=np.int64))})
    with pytest.raises(TypeError, match="join key types differ"):
        LazyFrame(a).join(LazyFrame(b), left_on=["s", "d"], right_on=["s", "i"]).compute(device="cpu")
    ok = LazyFrame(a).join(LazyFrame(a.rename_columns(["s", "rd", "rid"])), left_on=["s", "d"],
                           right_on=["s", "rd"]).compute(device="cpu")
    assert ok.num_rows == 1500           # a date against a date still joins


@pytest.mark.parametrize("how", HOWS)
def test_empty_string_keys_next_to_null_keys(how, spy):
    """'' is a key that matches '' and not NULL; NULL matches nothing, on both sides."""
    n = 3000
    lk = ["" if i % 3 == 0 else (None if i % 3 == 1 else f"v{i % 7}") for i in range(n)]
    rk = ["" if i % 5 == 0 else (None if i % 5 == 1 else f"v{i % 9}") for i in range(n)]
    l = pa.table({"k": pa.array(lk), "id": A(np.arange(n, dtype=np.int64))})
    r = pa.table({"rk": pa.array(rk), "rid": A(np.arange(n, dtype=np.int64)), "rv": A(np.arange(n, dtype=np.int64))})
    assert _check(l, r, how, "cpu", spy) > 100


def _bytes_col(values, n):
    """n strings cycling through `values` (bytes, not necessarily UTF-8), built from raw buffers."""
    picked = [values[i % len(values)] for i in range(n)]
    offs = np.zeros(n + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(b) for b in picked])
    return pa.Array.from_buffers(pa.string(), n, [None, pa.py_buffer(offs.tobytes()), pa.py_buffer(b"".join(picked))])


def test_invalid_utf8_keys_compare_by_their_bytes(spy):
    vals = [b"\xff\xfe", b"\xff\xfe\x00", b"a\x80", b"", b"\xc3\xa9", b"e\xcc\x81"]
    rvals = vals[::-1] + [b"zz"]
    l = pa.table({"k": _bytes_col(vals, 2000), "id": A(np.arange(2000, dtype=np.int64))})
    r = pa.table({"rk": _bytes_col(rvals, 1400), "rid": A(np.arange(1400, dtype=np.int64))})
    by_key = {}
    for j in range(1400):
        by_key.setdefault(rvals[j % 7], []).append(j)
    want = sorted((i, j) for i in range(2000) for j in by_key.get(vals[i % 6], []))
    co._JOIN_RESULT_CACHE.clear()
    spy.clear()
    out = LazyFrame(l).join(LazyFrame(r), left_on="k", right_on="rk").select(col("id"), col("rid")).compute(device="cpu")
    assert sorted(zip(out.column("id").to_pylist(), out.column("rid").to_pylist())) == want
    assert len(want) > 1000 and spy == [], spy


@pytest.mark.parametrize("sliced", [False, True], ids=["whole", "sliced"])
@pytest.mark.parametrize("device", DEVICES)
def test_left_join_payload_columns_with_nulls_and_unmatched_rows(sliced, device, spy):
    """The restore after the native gather: unmatched rows are NULL, so are the payload's own NULLs (int,
    utf8, float), also when the inputs are slices with non-zero offsets."""
    rng = np.random.default_rng(1)
    nl, nr = 5000, 3000
    rnull, lnull = rng.random(nr) < 0.2, rng.random(nl) < 0.2
    l = pa.table({"k": pa.array([f"k{i}" for i in rng.integers(0, 2000, nl)]), "id": A(np.arange(nl, dtype=np.int64)),
                  "lv": pa.array([None if m else f"L{i}" for i, m in zip(range(nl), lnull)]),
                  "lf": pa.array([None if m else float(i) for i, m in zip(range(nl), lnull)])})
    r = pa.table({"rk": pa.array([f"k{i}" for i in rng.integers(1000, 3000, nr)]),      # keys 0-999 never match
                  "rid": A(np.arange(nr, dtype=np.int64)),
                  "rv": pa.array([None if m else i for i, m in zip(range(nr), rnull)], type=pa.int64()),
                  "rs": pa.array([None if m else f"s{i}" for i, m in zip(range(nr), rnull)])})
    if sliced:
        l, r = l.slice(7, 4000), r.slice(13, 2500)
    cols = ["id", "lv", "lf", "rid", "rv", "rs"]
    want = sorted(_duck_tables(l, r).execute(
        "select l.id, l.lv, l.lf, r.rid, r.rv, r.rs from l left join r on l.k = r.rk").fetchall(), key=repr)
    co._JOIN_RESULT_CACHE.clear()
    spy.clear()
    out = LazyFrame(l).join(LazyFrame(r), left_on="k", right_on="rk", how="left") \
        .select(*[col(c) for c in cols]).compute(device=device)
    got = sorted(zip(*[out.column(c).to_pylist() for c in cols]), key=repr)
    assert got == want
    assert sum(1 for row in got if row[3] is None) > 500 and sum(1 for row in got if row[3] is not None and row[4] is None) > 100
    assert spy == [], spy
