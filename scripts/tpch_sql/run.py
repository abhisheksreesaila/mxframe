"""F-065: the 22 TPC-H queries as SQL text (`queries.py`) through `mx.sql`, compared with DuckDB running
the same text over the same Parquet files (typed views: DATE and DECIMAL(15,2) restored, see
tests/tpch_duckdb_sql.py).  One fresh process per query; one JSON line per query.

    pixi run python scripts/tpch_sql/run.py [--sf 1] [--queries 1,2,...]

status: equal (same rows as DuckDB), mismatch (runs, differs), raised (mx.sql / compute raised: the
message says why).  Columns are compared by position; strings and integers exactly, floats to 1e-6
relative; rows are matched after sorting, ORDER BY columns of the SELECT list must come back in order,
and rows tied with the last row of a LIMIT are dropped from both sides.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data"))
REL_TOL = 1e-6


def data_dir(sf: str, exact: bool = True) -> str:
    d = DATA / f"tpch_sf{sf}_exact"
    return str(d if exact and (d / "lineitem.parquet").exists() else DATA / f"tpch_sf{sf}")


def load_tables(text: str, directory: str) -> dict:
    """The tables of the query with only the columns its text names (as a user would read them)."""
    import pyarrow.parquet as pq
    import sqlglot
    import sqlglot.expressions as sge
    import mxframe as mx
    tree = sqlglot.parse_one(text, dialect="duckdb")
    names = {c.name.lower() for c in tree.find_all(sge.Column)}
    used = {t.name.lower() for t in tree.find_all(sge.Table)}
    out = {}
    for t in used:
        path = f"{directory}/{t}.parquet"
        have = pq.ParquetFile(path).schema_arrow.names
        cols = [c for c in have if c.lower() in names] or have[:1]
        out[t] = mx.read_parquet_table(path, columns=cols, dictionary="auto")
    return out


def duck_result(text: str, directory: str):
    import pyarrow as pa
    import tpch_duckdb_sql as ref
    t = ref.connect(directory).execute(text).to_arrow_table()
    for i, typ in enumerate(t.schema.types):
        if pa.types.is_decimal(typ):
            t = t.set_column(i, t.schema.field(i).name, t.column(i).cast(pa.float64()))
        elif pa.types.is_date(typ):
            t = t.set_column(i, t.schema.field(i).name, t.column(i).cast(pa.int32()))
    return t


def _is_float(t) -> bool:
    import pyarrow as pa
    return pa.types.is_floating(t) or pa.types.is_decimal(t)


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-9)


def compare(got, want, text: str) -> str:
    """"" when equal, else why not."""
    import sqlglot
    import sqlglot.expressions as sge
    if got.num_columns != want.num_columns:
        return f"columns: got {got.column_names}, DuckDB {want.column_names}"
    if got.num_rows != want.num_rows:
        return f"rows: got {got.num_rows}, DuckDB {want.num_rows}"
    tree = sqlglot.parse_one(text, dialect="duckdb")
    order = tree.args.get("order")
    pos = []
    if order is not None:
        names = [n.lower() for n in want.column_names]
        for o in order.expressions:
            e = o.this
            nm = e.name.lower() if isinstance(e, sge.Column) else None
            if nm in names:
                pos.append(names.index(nm))
    gc = [got.column(i).to_pylist() for i in range(got.num_columns)]
    wc = [want.column(i).to_pylist() for i in range(want.num_columns)]
    for i in pos:
        for r, (a, b) in enumerate(zip(gc[i], wc[i])):
            if not (_close(a, b) if isinstance(b, float) else a == b):
                return f"row {r} out of ORDER BY order in {want.column_names[i]}: got {a!r}, DuckDB {b!r}"
    if tree.args.get("limit") is not None and pos and got.num_rows:
        def keep(cols):
            edge = [cols[i][-1] for i in pos]
            return [r for r in range(got.num_rows)
                    if not all((_close(cols[i][r], e) if isinstance(e, float) else cols[i][r] == e)
                               for i, e in zip(pos, edge))]
        kg, kw = keep(gc), keep(wc)
        gc, wc = [[c[r] for r in kg] for c in gc], [[c[r] for r in kw] for c in wc]
        if len(kg) != len(kw):
            return "rows differ outside the LIMIT boundary tie group"
    key_cols = [i for i, t in enumerate(want.schema.types) if not _is_float(t)]
    def rows(cols):
        rs = list(zip(*cols))
        return sorted(rs, key=lambda r: tuple("" if r[i] is None else str(r[i]) for i in key_cols))
    for n, (g, w) in enumerate(zip(rows(gc), rows(wc))):
        for i, (a, b) in enumerate(zip(g, w)):
            ok = _close(a, b) if _is_float(want.schema.types[i]) else a == b
            if not ok:
                return f"row {n} {want.column_names[i]}: got {a!r}, DuckDB {b!r}"
    return ""


def one(q: int, sf: str) -> dict:
    import mxframe as mx
    from tpch_sql.queries import SQL
    text = SQL[q]
    d = data_dir(sf)
    out = {"q": q}
    try:
        tables = load_tables(text, d)
        got = mx.sql(text, **tables).compute(device="cpu")
    except Exception as e:                                   # noqa: BLE001 (the message is the report)
        out.update(status="raised", error=f"{type(e).__name__}: {str(e).splitlines()[0][:230]}")
        return out
    why = compare(got, duck_result(text, d), text)
    out.update(status="mismatch" if why else "equal", error=why[:230], rows=got.num_rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", default="1")
    ap.add_argument("--queries", default=",".join(map(str, range(1, 23))))
    ap.add_argument("--one", type=int)
    a = ap.parse_args()
    if a.one:
        import conftest  # noqa: F401  (import mxframe from this checkout)
        print("@@SQL " + json.dumps(one(a.one, a.sf)), flush=True)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUNBUFFERED="1")
    for q in map(int, a.queries.split(",")):
        p = subprocess.run([sys.executable, __file__, "--one", str(q), "--sf", a.sf], capture_output=True, text=True,
                           env=env, cwd=ROOT)
        ls = [l for l in p.stdout.splitlines() if l.startswith("@@SQL ")]
        print(ls[-1][6:] if ls else json.dumps({"q": q, "status": "error",
                                               "error": (p.stderr.strip().splitlines() or ["?"])[-1][:230]}), flush=True)


if __name__ == "__main__":
    main()
