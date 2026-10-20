"""F-151: the oracle harness. A SQL string goes through the real `mx` binary (build/mx) and through DuckDB, and the typed values are
compared exactly: an INT as an int, a DECIMAL as a Decimal (the JSON number text read by a decimal parser, never a float), a DATE as
a date, a FLOAT as a float (NaN equal to NaN, DuckDB's rule), TEXT, BOOL and NULL as themselves.

    mx = mx_binary()                                   # builds build/mx once (a lock keeps xdist workers apart)
    out = run_mx(mx, ["sql", "SELECT count(*) FROM 'x.parquet'"])
    check(mx, sql, con, files={"t": path})             # the same SQL through DuckDB, compared (kinds and scales too)
"""
import datetime
import fcntl
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from engine_sources import engine_sources

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def mx_binary():
    """Builds build/mx with scripts/build_mx.sh when a source is newer (once, even under xdist) and returns its path."""
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "mx"
    sources = engine_sources(ROOT / "mojo/cli/mx.mojo", ROOT / "scripts/build_mx.sh")
    with open(BUILD / ".mx.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in sources):
            subprocess.run(["bash", str(ROOT / "scripts/build_mx.sh")], check=True, capture_output=True, text=True)
    return out


def run_mx(mx, args, files=None, expect_exit=0):
    """Runs `mx <args> [-t name=path ...]`; returns the parsed JSON of stdout (decimals as Decimal) and checks the exit code."""
    cmd = [str(mx)] + list(args)
    for name, path in (files or {}).items():
        cmd += ["-t", f"{name}={path}"]
    p = subprocess.run(cmd, capture_output=True)
    assert p.returncode == expect_exit, f"`mx {' '.join(map(str, args))}` exited {p.returncode}, wanted {expect_exit}: {p.stdout[-400:]!r} {p.stderr[-400:]!r}"
    return json.loads(p.stdout.decode("utf-8"), parse_float=Decimal) if p.stdout.strip() else None


def typed_rows(res):
    """The rows of an `mx sql` JSON result as Python values by column kind."""
    kinds = [c["kind"] for c in res["columns"]]
    rows = []
    for r in res["rows"]:
        out = []
        for v, k in zip(r, kinds):
            if v is None:
                out.append(None)
            elif k == "DECIMAL":
                assert isinstance(v, (Decimal, int)), f"a DECIMAL must be a JSON number, not {v!r}"
                out.append(Decimal(v))
            elif k == "DATE":
                out.append(datetime.date.fromisoformat(v))
            elif k == "FLOAT":
                out.append(float({"Infinity": "inf", "-Infinity": "-inf", "NaN": "nan"}.get(v, v)) if isinstance(v, str) else float(v))
            else:
                out.append(v)
        rows.append(tuple(out))
    return rows


def duck_kind(type_code):
    t = str(type_code)
    if t.startswith("DECIMAL"):
        return "DECIMAL", int(t.split(",")[1].rstrip(")"))
    if t in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"):
        return "INT", 0
    if t in ("DOUBLE", "FLOAT"):
        return "FLOAT", 0
    if t == "VARCHAR":
        return "TEXT", 0
    if t == "DATE":
        return "DATE", 0
    if t == "BOOLEAN":
        return "BOOL", 0
    return t, 0


def sort_key(row):
    return tuple((0, "") if v is None else (1, repr(v)) for v in row)  # (NaN and every other float sort by their text)


def _same_row(a, b):
    """Equal values, NaN equal to NaN (DuckDB's rule) and compared by value, not by type."""
    return len(a) == len(b) and all(x == y or (isinstance(x, float) and isinstance(y, float) and x != x and y != y) for x, y in zip(a, b))


def assert_equal(got, want, ordered, label):
    g = list(got)
    w = [tuple(r) for r in want]
    assert len(g) == len(w), f"{label}: {len(g)} rows, DuckDB has {len(w)}"
    if not ordered:
        g = sorted(g, key=sort_key)
        w = sorted(w, key=sort_key)
    for i, (a, b) in enumerate(zip(g, w)):
        assert _same_row(a, b), f"{label}: row {i}: got {a}, DuckDB has {b}"
        for x, y in zip(a, b):
            assert type(x) is type(y) or x is None or y is None or (isinstance(x, (int, Decimal)) and isinstance(y, (int, Decimal))), \
                f"{label}: row {i}: got {a!r} but DuckDB has {b!r} (a different type)"


def check(mx, sql, con, files=None, duck_sql=None, ordered=False, extra=()):
    """`sql` through mx and through DuckDB (`duck_sql` if the dialects differ): same kinds, scales and values. Returns mx's JSON."""
    res = run_mx(mx, ["sql", "--max-rows", "0", *extra, sql], files)
    assert res and "columns" in res, f"mx gave no result for `{sql}`: {res}"
    cur = con.execute(duck_sql or sql)
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{sql}`: got {c}, DuckDB has {dk}"
    assert not res["truncated"]
    assert_equal(typed_rows(res), want, ordered, sql)
    return res
