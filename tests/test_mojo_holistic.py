"""F-148: count distinct, median / percentile (DuckDB's quantile_disc) and top-k as aggregates of the Mojo Query API,
equal to DuckDB on random data: nulls, negatives, ties, decimals, text, one group to 100,000 groups, over a table read
whole (`read`), a Parquet file streamed one row group at a time (`scan`) and a Frame after a few hundred edits
(`frame`, checked on the table it bakes to). The Mojo side is mojo/tools/holistic_probe.mojo run as a subprocess."""
import decimal
import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
ROW_GROUP = 4096
EDIT_SEED = 77


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "holistic_probe"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/holistic_probe.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def make_file(path, n, groups, seed, ties=False, wide=False, words=300, null_rate=0.1):
    """g int64 group (2% null), x int64 (nulls, negatives, ties when `ties`, a few near +-2^62 when `wide`),
    d decimal(18, 2), s string from a pool of lowercase words that include prefixes of one another."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    x = rng.integers(-5, 6, n) if ties else rng.integers(-1_000_000, 1_000_000, n)
    if wide:
        pick = rng.random(n) < 0.002
        x = np.where(pick, np.where(rng.random(n) < 0.5, (1 << 62) - rng.integers(0, 5, n), -(1 << 62) + rng.integers(0, 5, n)), x)
    cents = rng.integers(-2_000_000, 2_000_000, n) if not ties else rng.integers(-3, 4, n)
    pool = ["".join(rng.choice(list("abcdefgh"), size=rng.integers(1, 7))) for _ in range(words)]
    pool += ["ab", "abc", "abcd", "b", "ba"]
    s = rng.choice(pool, n)
    gm = rng.random(n) < 0.02
    xm = rng.random(n) < null_rate
    dm = rng.random(n) < null_rate
    sm = rng.random(n) < null_rate
    table = pa.table({
        "g": pa.array(g, mask=gm, type=pa.int64()),
        "x": pa.array(x, mask=xm, type=pa.int64()),
        "d": pa.array([None if m else decimal.Decimal(int(c)).scaleb(-2) for c, m in zip(cents, dm)], type=pa.decimal128(18, 2)),
        "s": pa.array(s, mask=sm, type=pa.string()),
    })
    pq.write_table(table, path, row_group_size=ROW_GROUP)


def run_probe(probe, path, mode, which, groups, baked=None):
    cmd = [str(probe), str(path), mode, which, str(groups)]
    if mode == "frame":
        cmd += [str(baked), str(EDIT_SEED)]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout.decode("utf-8")
    return json.loads(out)


AGGS = [  # (index in the probe's list, how DuckDB computes it, kind)
    ("count_distinct(x)", "count(DISTINCT x)", "distinct", "int"),
    ("median(x)", "quantile_disc(x, 0.5)", "value", "int"),
    ("p25(x)", "quantile_disc(x, 0.25)", "value", "int"),
    ("p90(x)", "quantile_disc(x, 0.9)", "value", "int"),
    ("top3(x)", "max(x, 3)", "top", "int"),
    ("count_distinct(s)", "count(DISTINCT s)", "distinct", "int"),
    ("median(s)", "quantile_disc(s, 0.5)", "value", "text"),
    ("top2(s)", "max(s, 2)", "top", "text"),
    ("median(d)", "quantile_disc(d, 0.5)", "value", "dec"),
    ("p75(d)", "quantile_disc(d, 0.75)", "value", "dec"),
    ("count_distinct(d)", "count(DISTINCT d)", "distinct", "int"),
    ("p0(x)", "quantile_disc(x, 0.0)", "value", "int"),
    ("p100(x)", "quantile_disc(x, 1.0)", "value", "int"),
]


def duckdb_rows(path, which, groups):
    cols = ", ".join(sql for _, sql, _, _ in AGGS)
    flt = "WHERE x BETWEEN -5000 AND 5000" if which == "filtered" else ""
    con = duckdb.connect()
    src = f"read_parquet('{path}')"
    if which == "global":
        rows = con.execute(f"SELECT 0 AS g, {cols} FROM {src}").fetchall()
    else:
        cond = f"g >= 0 AND g < {groups}" + (" AND x BETWEEN -5000 AND 5000" if which == "filtered" else "")
        rows = con.execute(f"SELECT g, {cols} FROM {src} WHERE {cond} GROUP BY g").fetchall()
    return {r[0]: r[1:] for r in rows}


def compare(got, want, groups, which, label):
    words = got["words"].split("|")[:-1]
    n_groups = 1 if which == "global" else groups
    assert len(got["aggs"]) == len(AGGS)
    for gi in range(n_groups):
        row = want.get(gi)
        for k, (name, _, kind, ty) in enumerate(AGGS):
            a = got["aggs"][k]
            w = None if row is None else row[k]
            tag = f"{label}: group {gi} {name}"
            if kind == "distinct":
                exp = 0 if w is None else w
                assert a["vals"][gi] == exp, f"{tag}: {a['vals'][gi]} != {exp}"
                assert (a["present"][gi] == 1) == (exp > 0), f"{tag}: present"
            elif kind == "value":
                assert (a["present"][gi] == 1) == (w is not None), f"{tag}: present {a['present'][gi]} vs {w}"
                if w is None:
                    continue
                v = a["vals"][gi]
                if ty == "text":
                    v = words[v]
                elif ty == "dec":
                    w = int(w.scaleb(2))
                assert v == w, f"{tag}: {v!r} != {w!r}"
            else:
                exp = [] if w is None else list(w)
                assert (a["present"][gi] == 1) == (len(exp) > 0), f"{tag}: present"
                v = a["top"][gi]
                if ty == "text":
                    v = [words[c] for c in v]
                assert v == exp, f"{tag}: {v} != {exp}"


CASES = [  # n, groups, options
    pytest.param(60_000, 7, {}, id="7 groups"),
    pytest.param(60_000, 7, {"ties": True}, id="7 groups, many ties"),
    pytest.param(80_000, 7, {"wide": True}, id="7 groups, values near 2^62"),
    pytest.param(120_000, 2_000, {}, id="2,000 groups"),
    pytest.param(250_000, 100_000, {"words": 40}, id="100,000 groups"),
    pytest.param(30_000, 1, {"ties": True, "null_rate": 0.5}, id="one group, half null"),
]


@pytest.mark.parametrize("n,groups,opts", CASES)
@pytest.mark.parametrize("mode", ["read", "scan", "frame"])
def test_equals_duckdb(probe, tmp_path, mode, n, groups, opts):
    path = tmp_path / "t.parquet"
    make_file(path, n, groups, seed=n + groups, **opts)
    shapes = ["all", "filtered"] + (["global"] if groups <= 7 else [])
    for which in shapes:
        baked = tmp_path / f"baked_{which}.parquet"
        got = run_probe(probe, path, mode, which, groups, baked)
        oracle = baked if mode == "frame" else path
        if mode == "frame":
            assert not pq.read_table(baked).equals(pq.read_table(path)), "the edits changed the table"
        want = duckdb_rows(oracle, which, groups)
        assert want and any(any(v is not None and v != 0 for v in row) for row in want.values()), "DuckDB has answers to compare"
        compare(got, want, groups, which, f"{mode} {which}")


def test_a_scan_of_a_multi_row_group_file_is_streamed(probe, tmp_path):
    """The file has many row groups, and the scan's answer equals the whole-file read's (so merging per row group is exact)."""
    path = tmp_path / "t.parquet"
    make_file(path, 100_000, 50, seed=5)
    assert pq.read_metadata(path).num_row_groups >= 20
    scanned = run_probe(probe, path, "scan", "all", 50)
    read = run_probe(probe, path, "read", "all", 50)
    assert scanned["aggs"] == read["aggs"] and scanned["counts"] == read["counts"]
