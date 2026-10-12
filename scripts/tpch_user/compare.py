"""Compare an engine's query result against the DuckDB reference.

Pure function, no GPU. Columns are matched by position (engines name them differently).
Integer / string / bool columns are "key" columns and must match exactly; float and decimal
columns are compared with a relative tolerance and the largest relative error is reported.

verdict:
  ok     every float within `rtol`
  drift  floats off by more than `rtol` but at most `drift_limit` = 1e-2 (float32 accumulation drift)
  wrong  different row count / keys / exact columns, float error above `drift_limit`, or rows
         not in the query's ORDER BY order (`order_by`, checked after the values match)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


# Float ORDER BY keys closer than this are one tie (only accumulation-order noise, ~1e-15, should
# count); real data values such as 9999.99 vs 9999.98 differ by 1e-6 and are NOT ties.
ORDER_RTOL = 1e-9


@dataclass
class Comparison:
    verdict: str            # "ok" | "drift" | "wrong"
    max_rel_err: float
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"


def _is_float(t: pa.DataType) -> bool:
    return pa.types.is_floating(t) or pa.types.is_decimal(t)


def _floats(col) -> np.ndarray:
    if pa.types.is_decimal(col.type):
        col = pc.cast(col, pa.float64())
    return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)


def _keys(tbl: pa.Table, key_idx: list[int]) -> list[tuple]:
    if not key_idx:
        return [()] * tbl.num_rows
    cols = [tbl.column(i).to_pylist() for i in key_idx]
    return list(zip(*cols))


def _sort_key(t):
    # None-safe, type-stable ordering for tuples of mixed keys
    return tuple((v is None, v if v is not None else 0) for v in t)


def _rel_err(a: np.ndarray, e: np.ndarray) -> np.ndarray:
    both_nan = np.isnan(a) & np.isnan(e)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.abs(a - e) / np.maximum(np.abs(e), 1e-30)
    err = np.where(both_nan, 0.0, err)
    err = np.where(a == e, 0.0, err)  # also handles inf == inf
    return np.where(np.isnan(err), np.inf, err)


def compare_tables(actual: pa.Table, expected: pa.Table, *, rtol: float = 1e-6,
                   drift_limit: float = 1e-2, topn_metric: int | None = None,
                   order_by: list[tuple[int, bool]] | None = None) -> Comparison:
    """`order_by` = [(column index, descending)] of the query's ORDER BY (positional, like every
    other column here). After the values match, `actual` must be sorted by those keys."""
    actual, expected, note = _empty_aggregate(actual, expected)
    res = _compare_values(actual, expected, rtol=rtol, drift_limit=drift_limit,
                          topn_metric=topn_metric)
    if note and res.verdict != "wrong":
        res.message = (res.message + "; " if res.message else "") + note
    if res.verdict == "wrong" or not order_by or actual.num_rows < 2:
        return res
    bad = check_order(actual, order_by, ORDER_RTOL)
    if bad is not None:
        return Comparison("wrong", res.max_rel_err, f"ORDER BY violated at result row {bad}")
    return res


def _empty_aggregate(actual: pa.Table, expected: pa.Table):
    """A one-row aggregate over no rows is NULL in SQL (DuckDB, mxframe) and 0 in pandas / Polars.
    In a one-row result a float NULL on one side against 0 on the other is the same answer: both
    become 0, and a note says so. Anything else is left for the normal comparison."""
    if actual.num_rows != 1 or expected.num_rows != 1 or actual.num_columns != expected.num_columns:
        return actual, expected, ""
    hit = False
    for i in range(actual.num_columns):
        a, e = actual.column(i), expected.column(i)
        if not (_is_float(a.type) and _is_float(e.type)):
            continue
        av, ev = a[0].as_py(), e[0].as_py()
        if (av is None and ev == 0) or (ev is None and av == 0):
            zero = pa.array([0.0])
            actual = actual.set_column(i, actual.field(i).with_type(pa.float64()), zero)
            expected = expected.set_column(i, expected.field(i).with_type(pa.float64()), zero)
            hit = True
    return actual, expected, ("empty aggregate: NULL (SQL) vs 0 (pandas/Polars) counted as equal" if hit else "")


def check_order(tbl: pa.Table, order_by: list[tuple[int, bool]], rtol: float = ORDER_RTOL) -> int | None:
    """Index of the first row that breaks the ORDER BY sequence, or None. Float keys that agree
    within `rtol` count as tied (the next key decides), like DuckDB's exact ties after float noise."""
    n = tbl.num_rows
    cols = []
    for i, desc in order_by:
        c = tbl.column(i)
        cols.append((_floats(c) if _is_float(c.type) else c.to_pylist(), _is_float(c.type), desc))
    for r in range(1, n):
        for vals, is_f, desc in cols:
            a, b = vals[r - 1], vals[r]
            if is_f:
                if a == b or abs(a - b) <= rtol * max(abs(a), abs(b), 1e-30) or (a != a and b != b):
                    continue
                cmp = -1 if a < b else 1
            else:
                if a == b:
                    continue
                if a is None or b is None:                # NULLs sort last
                    cmp = 1 if a is None else -1
                else:
                    cmp = -1 if a < b else 1
            if (cmp > 0) != desc:
                return r
            break
    return None


def _compare_values(actual: pa.Table, expected: pa.Table, *, rtol: float, drift_limit: float,
                    topn_metric: int | None) -> Comparison:
    if actual.num_columns != expected.num_columns:
        return Comparison("wrong", float("inf"),
                          f"column count {actual.num_columns} != {expected.num_columns}")
    if actual.num_rows != expected.num_rows:
        return Comparison("wrong", float("inf"),
                          f"rows {actual.num_rows} != {expected.num_rows}")
    n = expected.num_rows
    ncol = expected.num_columns
    if n == 0:
        return Comparison("ok", 0.0)

    float_idx = [i for i in range(ncol) if _is_float(expected.column(i).type)
                 or _is_float(actual.column(i).type)]
    key_idx = [i for i in range(ncol) if i not in float_idx]

    ka = _keys(actual, key_idx)
    ke = _keys(expected, key_idx)

    if topn_metric is not None:
        return _compare_topn(actual, expected, ka, ke, float_idx, key_idx, topn_metric,
                             rtol, drift_limit)

    if sorted(ka, key=_sort_key) != sorted(ke, key=_sort_key):
        missing = len(set(ke) - set(ka))
        return Comparison("wrong", float("inf"),
                          f"key columns differ ({missing} expected keys missing)")
    fa = [_floats(actual.column(i)) for i in float_idx]
    fe = [_floats(expected.column(i)) for i in float_idx]
    oa = sorted(range(n), key=lambda i: (_sort_key(ka[i]), tuple(f[i] for f in fa)))
    oe = sorted(range(n), key=lambda i: (_sort_key(ke[i]), tuple(f[i] for f in fe)))
    return _finish(actual, expected, oa, oe, float_idx, rtol, drift_limit)


def _finish(actual, expected, oa, oe, float_idx, rtol, drift_limit) -> Comparison:
    worst = 0.0
    oa = np.asarray(oa, dtype=np.int64)
    oe = np.asarray(oe, dtype=np.int64)
    for i in float_idx:
        a = _floats(actual.column(i))[oa]
        e = _floats(expected.column(i))[oe]
        if len(a):
            worst = max(worst, float(_rel_err(a, e).max()))
    if worst <= rtol:
        return Comparison("ok", worst)
    if worst <= drift_limit:
        return Comparison("drift", worst, f"max relative error {worst:.2e}")
    return Comparison("wrong", worst, f"max relative error {worst:.2e}")


def _group(keys: list[tuple], metric: np.ndarray) -> dict[tuple, list[int]]:
    """key -> row indices, best metric first (duplicate keys pair up in that order)."""
    g: dict[tuple, list[int]] = defaultdict(list)
    for i in sorted(range(len(keys)), key=lambda i: -metric[i]):
        g[keys[i]].append(i)
    return g


def _compare_topn(actual, expected, ka, ke, float_idx, key_idx, metric, rtol, drift_limit):
    """LIMIT-N queries: rows tied at the cutoff may legitimately differ between engines.

    The tie band is the set of rows whose metric equals the cutoff within the float noise seen
    (`rtol`, or the largest metric error if bigger); every row above it must be present, and
    every row of the result above it must be expected. Inside the band any row is accepted."""
    a_m = _floats(actual.column(metric))
    e_m = _floats(expected.column(metric))
    sa = np.argsort(-a_m, kind="stable")
    se = np.argsort(-e_m, kind="stable")
    worst = float(_rel_err(a_m[sa], e_m[se]).max())
    if worst > drift_limit:
        return Comparison("wrong", worst, f"top-N metric differs, max relative error {worst:.2e}")

    cutoff = float(e_m.min())
    slack = abs(cutoff) * max(rtol, worst)
    e_rows = _group(ke, e_m)
    a_rows = _group(ka, a_m)
    for k, idx in e_rows.items():
        above = sum(1 for i in idx if e_m[i] - cutoff > slack)
        if above > len(a_rows.get(k, ())):
            return Comparison("wrong", worst, f"row {k} missing from result (above the cutoff)")
    for k, idx in a_rows.items():
        above = sum(1 for i in idx if a_m[i] - cutoff > slack)
        if above > len(e_rows.get(k, ())):
            return Comparison("wrong", worst, f"unexpected row {k} (above the cutoff)")

    # other float columns on rows both sides share (duplicate keys pair best-metric first)
    shared = [(a_rows[k][j], e_rows[k][j]) for k in e_rows if k in a_rows
              for j in range(min(len(a_rows[k]), len(e_rows[k])))]
    for c in float_idx:
        if c == metric or not shared:
            continue
        a = _floats(actual.column(c))[[p for p, _ in shared]]
        e = _floats(expected.column(c))[[q for _, q in shared]]
        worst = max(worst, float(_rel_err(a, e).max()))
    if worst <= rtol:
        return Comparison("ok", worst)
    if worst <= drift_limit:
        return Comparison("drift", worst, f"max relative error {worst:.2e}")
    return Comparison("wrong", worst, f"max relative error {worst:.2e}")
