"""F-156: CAST(text AS number / date / boolean) reads a word the way DuckDB does.

A text cast is evaluated once per word of a dictionary, by `sqlfn.parse_text`. DuckDB's own readers have quirks (digit separators `1_000`, hex and binary
integers, a doubled sign on a double, date words such as `inf`, exponents on decimals that round oddly), so this test feeds thousands of generated words
(spaces, signs, digits, points, exponents, separators, junk, BC and five-digit years, special date words, every case of the boolean words) to
`mojo/tools/cast_probe.mojo` and to DuckDB's TRY_CAST, and requires for each word and each target type:

- the engine reads the word (`ok`): DuckDB gives the same value;
- the engine says it is not one (`bad`): DuckDB gives NULL;
- the engine says it is a DuckDB quirk it does not read (`unsure`): nothing is compared, and a CAST or TRY_CAST over such a word is an error in the engine, so no
  answer is ever wrong. The number of `unsure` words is bounded, so the quirks stay a small, named set.
"""
import fcntl
import itertools
import random
import struct
import subprocess
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

TARGETS = [("INTEGER", "i32"), ("BIGINT", "i64"), ("TINYINT", "i8"), ("DECIMAL(12,3)", ("dec", 3)), ("DECIMAL(18,0)", ("dec", 0)), ("DECIMAL(5,2)", ("dec", 2)),
           ("DOUBLE", "f64"), ("REAL", "f32"), ("DATE", "date"), ("BOOLEAN", "bool")]


@pytest.fixture(scope="session")
def cast_probe():
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "cast_probe"
    sources = list((ROOT / "mojo/mxframe").glob("*.mojo")) + list((ROOT / "kernels_aot").glob("*.mojo")) + [ROOT / "mojo/tools/cast_probe.mojo"]
    with open(BUILD / ".cast_probe.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in sources):
            subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/cast_probe.mojo"), str(out)], check=True, capture_output=True, text=True)
    return out


def numeric_words(rng):
    out = set()
    signs = ["", "+", "-", " ", "  +", "\t-", "+ ", "--", "+-", "-+", "++"]
    ints = ["0", "1", "12", "007", "9", "123456789012", "9223372036854775807", "9223372036854775808", "-9223372036854775808", "99999999999999999999", "00", "000.00", "1_000",
            "1_0", "2147483647", "2147483648", "127", "128", "999999999999999999", "1000000000000000000"]
    fracs = ["", ".", ".5", ".50", ".05", ".005", ".0", ".9999", ".4999", ".5001", ".25", ".125", "._5", ".5_0"]
    exps = ["", "e0", "e1", "E2", "e-1", "e+2", "e10", "e-10", "e", "e+", "e-", "e1.5", "ex", "e1_0"]
    tails = ["", " ", "  ", "\t", "\n", "x", "_", "f", "d", ",", "%", " 1", " ", " "]
    for s, i, f, e, t in itertools.product(signs, ints, fracs, exps, tails):
        if rng.random() < 0.012:
            out.add(s + i + f + e + t)
    out.update(["", " ", ".", "+", "-", "e5", ".e5", "0x1F", "0X1f", "-0x1F", "+0x1F", " 0x1F ", "0x", "0xG", "0x1_F", "0x_1F", "0x1F_", "0x1.5", "0b101", "0B101", "0b2", "0b", "0b1_0", "-0b101",
                "0x7FFFFFFFFFFFFFFF", "0xFFFFFFFFFFFFFFFF", "1_000", "1_0", "_1", "1_", "1__0", "1_.5", "1._5", "1.5_0", "1_0.5", "nan", "NaN", "NAN", "-nan", "+nan", "inf", "-inf", "+inf", "Inf",
                "INF", "infinity", "Infinity", "-Infinity", "+Infinity", "INFINITY", "infin", "1e5", "1E5", "1e400", "-1e400", "1e-400", "0.1e1", "1.e1", ".1e1", "1e1.0", "12abc", "abc12", "1 2",
                "1,2", "1.2.3", "--1", "+-1", "-+1", "++1", "١٢", "１２", "1\u0000", "0.000000000000000000000000000000000000000000001", "123456789012345678901234567890123456789.5",
                "9223372036854775807.4", "9223372036854775807.5", "-9223372036854775808.5", "100000000.00", "99999999.995", "0.9995", "0.9994", "-0.5", "-0.4", "0.5", "1.5", "2.5", "-1.5", "-2.5",
                ".5", "5.", "-.5", "+.5", "+5.", "5e-1", "4e-1", "95e-1", "12e-1", "1e-5", "5000000000000000000000000000000e-30", "0.0000000000000000000000000000005", "1e99999", "1e-99999"])
    return sorted(out)


def date_words(rng):
    out = set()
    for y in ["2024", "0000", "0001", "1", "24", "10000", "99999", "-0044", "+2024", "1900", "2023", "02024", "9999", "5867411", "5867412"]:
        for m in ["1", "01", "12", "13", "0", "00", "2", "02", "001"]:
            for d in ["1", "01", "29", "30", "31", "32", "0", "5", "28", "001"]:
                for sep in ["-", "/", ".", " ", "--", "\\", "_"]:
                    if rng.random() < 0.1:
                        out.add(f"{y}{sep}{m}{sep}{d}")
    out.update(["2024-02-29", "2023-02-29", "1900-02-29", "2000-02-29", "2024-2-9", " 2024-02-09", "2024-02-09 ", "\t2024-02-09\n", "2024-02-09x", "2024-02-09 10:00:00", "2024-02-09T10:00:00",
                "2024-02-09T10:00:00Z", "2024-02-09 10:00", "2024-02-09 25:00:00", "2024-02-09 BC", "2024-02-09 AD", "0044-03-15 BC", "infinity", "-infinity", "+infinity", "inf", "-inf",
                "Infinity", "INFINITY", "epoch", "Epoch", "now", "today", "tomorrow", "yesterday", "20240209", "2024029", "24-02-09", "2024-0209", "2024--02-09", "2024-02-", "2024-", "2024",
                "2024-02", "", " ", "abc", "02-09-2024", "09/02/2024", "Feb 9 2024", "9 Feb 2024", "2024-02-09 +01", "2024-02-30", "٢٠٢٤-02-09", "2024-02-09 ",
                "1969-12-31", "1970-01-01", "5877642-06-25", "-5877641-06-25", "4714-11-24 BC", "2024-02-091", "2024-02-09 (BC)", "2024-02-09(BC)", "1582-10-15", "1582-10-04"])
    return sorted(out)


def bool_words(rng):
    out = set()
    base = ["t", "f", "true", "false", "yes", "no", "y", "n", "on", "off", "1", "0", "tr", "fa", "ye", "tru", "fals", "2", "-1", "10", "01", "", " ", "maybe", "null", "NULL", "unknown",
            "t ", " t", "true ", " true", "\ttrue", "true\n", "T", "F", "True", "False", "TRUE", "FALSE", "tRuE", "Yes", "YES", "No", "NO", "Y", "N", "On", "OFF", "enabled", "ok", "x", "+1",
            "1.0", "0.0", "1e0", "١"]
    out.update(base)
    for w in ["true", "false", "yes", "no", "t", "f", "y", "n"]:
        for _ in range(6):
            out.add("".join(c.upper() if rng.random() < 0.5 else c for c in w))
    return sorted(out)


def read_back(kind, text):
    """The value of one `ok:<n>` field as Python (the engine's stored value)."""
    v = int(text)
    if kind == "f64":
        return struct.unpack("<d", struct.pack("<q", v))[0]
    if kind == "f32":
        return struct.unpack("<d", struct.pack("<q", v))[0]
    if kind == "date":
        import datetime
        return datetime.date(1970, 1, 1) + datetime.timedelta(days=v)
    if kind == "bool":
        return bool(v)
    if isinstance(kind, tuple):
        return Decimal(v).scaleb(-kind[1])
    return v


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return a == b


def compare(cast_probe, words, tmp_path, only=None):
    wf = tmp_path / "words.txt"
    wf.write_text("\n".join("x" + w.encode("utf-8").hex() for w in words) + "\n")
    p = subprocess.run([str(cast_probe), str(wf)], capture_output=True)
    assert p.returncode == 0, p.stderr[-500:]
    lines = p.stdout.decode("utf-8").split("\n")[:len(words)]
    assert len(lines) == len(words)
    con = duckdb.connect()
    con.execute("create table w(i bigint, w varchar)")
    con.executemany("insert into w values (?, ?)", list(enumerate(words)))
    targets = [t for t in TARGETS if only is None or t[0] in only]
    cols = ", ".join(f"TRY_CAST(w AS {name})" for name, _ in targets)
    want = con.execute(f"select i, {cols} from w order by i").fetchall()
    stats = {name: {"ok": 0, "bad": 0, "unsure": 0} for name, _ in targets}
    wrong = []
    idx = {name: k for k, (name, _) in enumerate(TARGETS)}
    for line, row in zip(lines, want):
        fields = line.split("|")
        word = words[row[0]]
        for k, (name, kind) in enumerate(targets):
            got = fields[idx[name]]
            duck = row[k + 1]
            if isinstance(duck, str):                                  # a date DuckDB shows as text (BC, five digits): not NULL
                duck = ("text", duck)
            if got == "unsure":
                stats[name]["unsure"] += 1
            elif got == "bad":
                stats[name]["bad"] += 1
                if duck is not None:
                    wrong.append((name, word, "engine: not a " + name, duck))
            else:
                stats[name]["ok"] += 1
                value = read_back(kind, got.split(":")[1])
                if duck is None or not same(value, duck if not isinstance(duck, tuple) else None):
                    wrong.append((name, word, value, duck))
    return stats, wrong


@pytest.mark.parametrize("group", ["numbers", "dates", "booleans"])
def test_text_casts_read_words_as_duckdb(cast_probe, tmp_path, group):
    rng = random.Random(5)
    if group == "numbers":
        words, only = numeric_words(rng), ["INTEGER", "BIGINT", "TINYINT", "DECIMAL(12,3)", "DECIMAL(18,0)", "DECIMAL(5,2)", "DOUBLE", "REAL"]
    elif group == "dates":
        words, only = date_words(rng), ["DATE"]
    else:
        words, only = bool_words(rng), ["BOOLEAN"]
    stats, wrong = compare(cast_probe, words, tmp_path, only)
    assert not wrong, f"{len(wrong)} words are read differently from DuckDB, e.g. " + "; ".join(f"{n}: {w!r} -> {v!r} (DuckDB {d!r})" for n, w, v, d in wrong[:12])
    for name, st in stats.items():
        decided = st["ok"] + st["bad"]
        assert decided > 0 and st["ok"] > 20, (name, st)
        assert st["unsure"] <= max(60, len(words) // 5), f"{name}: {st['unsure']} of {len(words)} words are left undecided: " + str(st)
