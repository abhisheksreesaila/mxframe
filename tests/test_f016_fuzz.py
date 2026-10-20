"""F-016: random plan shapes against DuckDB (cpu and auto).

Each seed builds a random pipeline of 2-4 operators (filter / with_columns / select / aggregate /
join / top-n / distinct) over random data with nulls, runs it through omxcore and through the
equivalent SQL in DuckDB, and compares the rows (order ignored: an order is only checked through
a full-row sort followed by a limit, which decides WHICH rows come back).

Reproduce a failure:  pytest tests/test_f016_fuzz.py -k "cpu and 0123"   (the seed is in the id).
Longer runs:          MXFRAME_FUZZ_SEEDS=2000 pytest tests/test_f016_fuzz.py -m slow
"""
import math
import os

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import conftest  # noqa: F401
from omxcore import LazyFrame, col, lit
from omxcore.lazy_frame import Scan

N_SEEDS = 60
LONG_SEEDS = int(os.environ.get("MXFRAME_FUZZ_SEEDS", "300"))


# --- random data -----------------------------------------------------------------------------

def _key_card(n_rows):
    """Few distinct keys in small tables (many groups collide); more in big ones so joins do not explode."""
    return 6 if n_rows <= 500 else max(6, n_rows // 20)


def _random_table(rng, n, prefix, n_int, n_float, key_card=6, null_rate=0.12):
    cols = {}
    for i in range(n_int):
        v = rng.integers(0, key_card, n).astype(np.int64)
        cols[f"{prefix}i{i}"] = pa.array(v, mask=rng.random(n) < null_rate)
    for i in range(n_float):
        v = np.round(rng.normal(scale=10, size=n), 2)
        cols[f"{prefix}f{i}"] = pa.array(v, mask=rng.random(n) < null_rate)
    return pa.table(cols)


class State:
    """A relation being built: its omxcore plan, the SQL that computes it, and its columns."""

    def __init__(self, lf, sql, kinds):
        self.lf, self.sql, self.kinds = lf, sql, dict(kinds)   # kinds: name -> "int" | "float"

    def ints(self):
        return [c for c, k in self.kinds.items() if k == "int"]

    def nums(self):
        return [c for c, k in self.kinds.items() if k != "str"]

    def strs(self):
        return [c for c, k in self.kinds.items() if k == "str"]

    def all_cols(self):
        return list(self.kinds)

    def keys(self):
        return [c for c, k in self.kinds.items() if k in ("int", "str")]


_CMP = {"gt": lambda a, b: a > b, "lt": lambda a, b: a < b, "ge": lambda a, b: a >= b,
        "le": lambda a, b: a <= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}
_SQL_CMP = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "=", "ne": "<>"}


def _literal(rng, kind):
    if kind == "int":
        return int(rng.integers(-2, 5))
    return float(rng.choice([0.5, 1.5, -2.25, 0.1, 3.7, 0.0]))


def _sql_lit(v):
    return repr(v)


def _predicate(rng, st):
    def one():
        c = str(rng.choice(st.nums()))
        op = str(rng.choice(list(_CMP)))
        v = _literal(rng, st.kinds[c])
        if st.kinds[c] == "int":
            v = int(rng.integers(0, 6))
        return _CMP[op](col(c), lit(v)), f"{c} {_SQL_CMP[op]} {_sql_lit(v)}"
    if st.strs() and rng.random() < 0.3:
        c = str(rng.choice(st.strs()))
        v = str(rng.choice(["a", "b", "c"]))
        if rng.random() < 0.5:
            return col(c) == lit(v), f"{c} = '{v}'"
        return col(c) != lit(v), f"{c} <> '{v}'"
    e, s = one()
    r = rng.random()
    if r < 0.25:
        e2, s2 = one()
        return e & e2, f"({s} and {s2})"
    if r < 0.4:
        e2, s2 = one()
        return e | e2, f"({s} or {s2})"
    return e, s


def _expression(rng, st):
    """A new column: a op b, or a op literal, over numeric columns."""
    a = str(rng.choice(st.nums()))
    op = str(rng.choice(["add", "sub", "mul"]))
    sym = {"add": "+", "sub": "-", "mul": "*"}[op]
    fn = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y}[op]
    if rng.random() < 0.5:
        b = str(rng.choice(st.nums()))
        kind = "int" if st.kinds[a] == st.kinds[b] == "int" else "float"
        return fn(col(a), col(b)), f"{a} {sym} {b}", kind
    v = _literal(rng, st.kinds[a])
    return fn(col(a), lit(v)), f"{a} {sym} {_sql_lit(v)}", st.kinds[a]


# --- operators ---------------------------------------------------------------------------------

class Ctx:
    def __init__(self, rng, tables, n_rows):
        self.rng, self.tables, self.n_rows, self.uid = rng, tables, n_rows, 0
        self.log = []

    def name(self, prefix):
        self.uid += 1
        return f"{prefix}{self.uid}"


def op_filter(ctx, st):
    e, s = _predicate(ctx.rng, st)
    ctx.log.append(f"filter {s}")
    return State(st.lf.filter(e), f"select * from ({st.sql}) where {s}", st.kinds)


def op_with_columns(ctx, st):
    e, s, kind = _expression(ctx.rng, st)
    replace = ctx.rng.random() < 0.3
    name = str(ctx.rng.choice(st.nums())) if replace else ctx.name("w")
    ctx.log.append(f"with_columns {name} = {s}")
    kinds = dict(st.kinds)
    kinds[name] = kind
    excl = f" exclude ({name})" if name in st.kinds else ""
    return State(st.lf.with_columns(e.alias(name)),
                 f"select *{excl}, {s} as {name} from ({st.sql})", kinds)


def op_select(ctx, st):
    names = st.all_cols()
    keep = [c for c in names if ctx.rng.random() < 0.7] or names[:1]
    if not any(st.kinds[c] != "str" for c in keep):
        keep.append(st.nums()[0])
    exprs, sqls, kinds = [col(c) for c in keep], list(keep), {c: st.kinds[c] for c in keep}
    if ctx.rng.random() < 0.6:
        e, s, kind = _expression(ctx.rng, st)
        name = ctx.name("s")
        exprs.append(e.alias(name)); sqls.append(f"{s} as {name}"); kinds[name] = kind
    ctx.log.append(f"select {', '.join(sqls)}")
    return State(st.lf.select(*exprs), f"select {', '.join(sqls)} from ({st.sql})", kinds)


def op_agg(ctx, st):
    rng = ctx.rng
    cand = st.keys()
    n_keys = int(rng.choice([0, 0, 1, 1, 2])) if cand else 0
    keys = list(rng.choice(cand, size=min(n_keys, len(cand)), replace=False)) if n_keys else []
    keys = [str(k) for k in keys]
    aggs, sqls, kinds = [], [], {k: st.kinds[k] for k in keys}
    for _ in range(int(rng.integers(1, 4))):
        fn = str(rng.choice(["sum", "min", "max", "mean", "count"]))
        name = ctx.name("m")
        if st.strs() and rng.random() < 0.15:
            fn = "count"                                # count of a string column
            c = str(rng.choice(st.strs())); e = col(c); kind = "str"
        elif rng.random() < 0.3:
            e, c, kind = _expression(rng, st)          # aggregate of an expression: sum(a * b)
        else:
            c = str(rng.choice(st.nums())); e = col(c); kind = st.kinds[c]
        aggs.append(getattr(e, fn)().alias(name))
        sqls.append(f"{'avg' if fn == 'mean' else fn}({c}) as {name}")
        kinds[name] = ("int" if fn == "count" else "float" if fn == "mean" else kind)
    ctx.log.append(f"agg by {keys}: {', '.join(sqls)}")
    grp = f" group by {', '.join(keys)}" if keys else ""
    sel = ", ".join(keys + sqls)
    return State(st.lf.groupby(*keys).agg(*aggs), f"select {sel} from ({st.sql}){grp}", kinds)


def op_topn(ctx, st):
    rng = ctx.rng
    names = st.all_cols()
    desc = [bool(rng.random() < 0.4) for _ in names]
    n = int(rng.integers(1, 12))
    order = ", ".join(f"{c} {'desc' if d else 'asc'} nulls last" for c, d in zip(names, desc))
    ctx.log.append(f"sort {order} limit {n}")
    return State(st.lf.sort(*names, descending=desc).limit(n),
                 f"select * from ({st.sql}) order by {order} limit {n}", st.kinds)


def op_distinct(ctx, st):
    ctx.log.append("distinct")
    return State(st.lf.distinct(), f"select distinct * from ({st.sql})", st.kinds)


def op_join(ctx, st):
    rng = ctx.rng
    schema = st.lf.schema                       # join keys must be integers in the engine too (int vs float keys are refused by design)
    ints = [c for c in st.ints() if schema is not None and c in schema.names and pa.types.is_integer(schema.field(c).type)]
    if not ints:
        return None
    j = ctx.name("j")
    right = _random_table(rng, max(4, ctx.n_rows // 4), f"{j}_", 2, 1, key_card=_key_card(ctx.n_rows), null_rate=ctx.null_rate)
    rk, ra, rb = f"{j}_i0", f"{j}_i1", f"{j}_f0"
    ctx.tables[j] = right
    rlf = LazyFrame(Scan(right))
    rsql = f"select * from {j}"
    rkinds = {rk: "int", ra: "int", rb: "float"}
    shape = str(rng.choice(["scan", "filter", "agg"]))
    if shape == "filter":
        v = int(rng.integers(0, 4))
        rlf, rsql = rlf.filter(col(ra) > lit(v)), f"select * from {j} where {ra} > {v}"
    elif shape == "agg":
        rlf = rlf.groupby(rk).agg(col(rb).sum().alias(ra), col(rb).count().alias(rb))
        rsql = f"select {rk}, sum({rb}) as {ra}, count({rb}) as {rb} from {j} group by {rk}"
        rkinds = {rk: "int", ra: "float", rb: "int"}
    lk = str(rng.choice(ints))
    keep = [c for c in rkinds if c != rk]
    ctx.log.append(f"join {shape}({j}) on {lk} = {rk}")
    kinds = dict(st.kinds); kinds.update({c: rkinds[c] for c in keep})
    sql = (f"select l.*, {', '.join('r.' + c for c in keep)} from ({st.sql}) l "
           f"join ({rsql}) r on l.{lk} = r.{rk}")
    return State(st.lf.join(rlf, left_on=lk, right_on=rk), sql, kinds)


OPS = [(op_filter, 3), (op_with_columns, 3), (op_select, 2), (op_agg, 4), (op_join, 2),
       (op_topn, 2), (op_distinct, 1)]


def build(seed, n_rows):
    rng = np.random.default_rng(seed)
    null_rate = 0.0 if seed % 2 else 0.12          # the fused kernels only take null-free columns
    base = _random_table(rng, n_rows, "c", int(rng.integers(1, 4)), int(rng.integers(1, 3)), key_card=_key_card(n_rows), null_rate=null_rate)
    if rng.random() < 0.6:
        strings = np.array(["a", "b", "c"], dtype=object)[rng.integers(0, 3, n_rows)]
        base = base.append_column("cs0", pa.array(list(strings), mask=rng.random(n_rows) < null_rate))
    ctx = Ctx(rng, {"base": base}, n_rows)
    ctx.null_rate = null_rate
    kind_of = {"int64": "int", "double": "float", "string": "str"}
    st = State(LazyFrame(Scan(base)), "select * from base", {c: kind_of[str(base.schema.field(c).type)] for c in base.column_names})
    weights = np.array([w for _, w in OPS], dtype=float)
    for _ in range(int(rng.integers(2, 5))):
        for _try in range(5):
            op = OPS[int(rng.choice(len(OPS), p=weights / weights.sum()))][0]
            nxt = op(ctx, st)
            if nxt is not None:
                st = nxt
                break
    return ctx, st


# --- comparison ------------------------------------------------------------------------------

def _close(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) or isinstance(y, float):
        return math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6)
    return x == y


def _rows(t, names):
    return list(zip(*[t.column(n).to_pylist() for n in names])) if names else []


def _key(row):
    return tuple((0, "") if v is None else (1, v) if isinstance(v, str) else (2, round(float(v), 3)) for v in row)


def assert_same(got, ref, what):
    assert sorted(got.column_names) == sorted(ref.column_names), (got.column_names, ref.column_names, what)
    names = sorted(ref.column_names)
    g, r = _rows(got, names), _rows(ref, names)
    assert len(g) == len(r), f"{len(g)} rows, DuckDB {len(r)}: {what}"
    g.sort(key=_key); r.sort(key=_key)
    if all(all(_close(x, y) for x, y in zip(a, b)) for a, b in zip(g, r)):
        return
    unmatched = list(g)                                   # rounding put near-equal rows in different order
    for row in r:
        for i, cand in enumerate(unmatched):
            if all(_close(x, y) for x, y in zip(cand, row)):
                del unmatched[i]
                break
        else:
            raise AssertionError(f"row {row} of DuckDB not in the result; columns {names}: {what}")


def run_seed(seed, device, n_rows):
    ctx, st = build(seed, n_rows)
    what = f"seed {seed} on {device}: " + " -> ".join(ctx.log) + f"\nSQL: {st.sql}"
    con = duckdb.connect()
    for name, t in ctx.tables.items():
        con.register(name, t)
    ref = con.execute(st.sql).to_arrow_table()
    got = st.lf.compute(device=device)
    assert_same(got, ref, what)


@pytest.mark.parametrize("seed", range(N_SEEDS), ids=lambda s: f"{s:04d}")
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_random_plans_match_duckdb(seed, device):
    run_seed(seed, device, n_rows=60)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(LONG_SEEDS), ids=lambda s: f"{s:04d}")
def test_random_plans_match_duckdb_long(seed):
    run_seed(1000 + seed, "cpu", n_rows=int(np.random.default_rng(seed).choice([5, 60, 400])))


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(int(os.environ.get("MXFRAME_FUZZ_LARGE_SEEDS", "40"))), ids=lambda s: f"{s:04d}")
@pytest.mark.parametrize("device", ["auto", "gpu"])
def test_random_plans_match_duckdb_large(seed, device):
    """20k rows: past the auto threshold, so the GPU and the fused kernels take part."""
    from omxcore.aot_kernels import GPU_AOT_AVAILABLE
    if device == "gpu" and not GPU_AOT_AVAILABLE:
        pytest.skip("no GPU AOT kernels")
    run_seed(5000 + seed, device, n_rows=20_000)


@pytest.mark.parametrize("seed", [5061])
@pytest.mark.parametrize("device", ["cpu", "auto", "gpu"])
def test_regression_seeds_from_the_large_runs(seed, device):
    """5061 (F-020): distinct -> select -> grouped sums of a float64 column cancelling to ~0.1. A Project under the
    aggregate declined the fused pipeline and the GPU fallback summed in float32."""
    from omxcore.aot_kernels import GPU_AOT_AVAILABLE
    if device == "gpu" and not GPU_AOT_AVAILABLE:
        pytest.skip("no GPU AOT kernels")
    run_seed(seed, device, n_rows=20_000)
