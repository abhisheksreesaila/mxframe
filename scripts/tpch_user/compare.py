"""Compare an engine's query result against the DuckDB reference.

Pure function, no GPU. Columns are matched by position (engines name them differently).
Integer / string / bool columns are "key" columns and must match exactly; float and decimal
columns are compared with a relative tolerance and the largest relative error is reported.

verdict:
  ok     every float within `rtol`
  drift  floats off by more than `rtol` but at most `drift_limit` = 1e-2 (float32 accumulation drift)
  wrong  different row count / keys / exact columns, or float error above `drift_limit`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


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
                   drift_limit: float = 1e-2, topn_metric: int | None = None) -> Comparison:
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
    oa = sorted(range(n), key=lambda i: _sort_key(ka[i]))
    oe = sorted(range(n), key=lambda i: _sort_key(ke[i]))
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


def _compare_topn(actual, expected, ka, ke, float_idx, key_idx, metric, rtol, drift_limit):
    """LIMIT-N queries: rows tied at the cutoff may legitimately differ between engines."""
    a_m = _floats(actual.column(metric))
    e_m = _floats(expected.column(metric))
    sa = np.argsort(-a_m, kind="stable")
    se = np.argsort(-e_m, kind="stable")
    worst = float(_rel_err(a_m[sa], e_m[se]).max())
    if worst > drift_limit:
        return Comparison("wrong", worst, f"top-N metric differs, max relative error {worst:.2e}")

    cutoff = float(e_m.min())
    slack = abs(cutoff) * max(drift_limit, 1e-9)
    e_rows = {ke[i]: i for i in range(len(ke))}
    a_rows = {ka[i]: i for i in range(len(ka))}
    for k, i in e_rows.items():
        if e_m[i] - cutoff > slack and k not in a_rows:
            return Comparison("wrong", worst, f"row {k} missing from result (above the cutoff)")
    for k, i in a_rows.items():
        if a_m[i] - cutoff > slack and k not in e_rows:
            return Comparison("wrong", worst, f"unexpected row {k} (above the cutoff)")

    # other float columns on rows both sides share
    shared = [(a_rows[k], e_rows[k]) for k in e_rows if k in a_rows]
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
