"""Result-comparison helper of the user-felt TPC-H scoreboard (pure function, no GPU)."""
import sys
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tpch_user.compare import compare_tables  # noqa: E402


def T(**cols):
    return pa.table(cols)


def test_same_rows_in_a_different_order_match():
    a = T(k=["b", "a", "c"], v=[2.0, 1.0, 3.0])
    e = T(k=["a", "b", "c"], v=[1.0, 2.0, 3.0])
    r = compare_tables(a, e)
    assert r.ok and r.verdict == "ok" and r.max_rel_err == 0.0


def test_float_within_tolerance_is_ok_and_error_is_reported():
    r = compare_tables(T(k=[1], v=[100.0000001]), T(k=[1], v=[100.0]))
    assert r.ok
    assert 0 < r.max_rel_err <= 1e-6


def test_float_drift_is_flagged_but_not_called_wrong():
    r = compare_tables(T(k=[1], v=[100.001]), T(k=[1], v=[100.0]))
    assert not r.ok and r.verdict == "drift"
    assert abs(r.max_rel_err - 1e-5) < 1e-7


def test_large_float_error_is_wrong():
    r = compare_tables(T(k=[1], v=[150.0]), T(k=[1], v=[100.0]))
    assert not r.ok and r.verdict == "wrong"


def test_row_count_mismatch_is_wrong():
    r = compare_tables(T(k=[1, 2], v=[1.0, 2.0]), T(k=[1], v=[1.0]))
    assert r.verdict == "wrong" and "rows" in r.message


def test_integer_and_string_columns_must_match_exactly():
    assert compare_tables(T(k=["a"], n=[5]), T(k=["a"], n=[6])).verdict == "wrong"
    assert compare_tables(T(k=["a"], v=[1.0]), T(k=["b"], v=[1.0])).verdict == "wrong"


def test_column_names_are_ignored_positionally():
    assert compare_tables(T(x=["a"], y=[1.0]), T(k=["a"], v=[1.0])).ok


def test_column_count_mismatch_is_wrong():
    assert compare_tables(T(k=["a"]), T(k=["a"], v=[1.0])).verdict == "wrong"


def test_single_row_scalar_results():
    assert compare_tables(T(r=[12345.6]), T(r=[12345.6])).ok
    assert compare_tables(T(r=[12345.6]), T(r=[99999.0])).verdict == "wrong"


def test_empty_results_match_and_mixed_empty_does_not():
    e = pa.table({"k": pa.array([], pa.int64()), "v": pa.array([], pa.float64())})
    assert compare_tables(e, e).ok
    assert compare_tables(e, T(k=[1], v=[1.0])).verdict == "wrong"


def test_decimal_and_float32_columns_are_compared_numerically():
    import decimal
    a = pa.table({"k": [1], "v": pa.array([1.5], pa.float32())})
    e = pa.table({"k": [1], "v": pa.array([decimal.Decimal("1.50")], pa.decimal128(15, 2))})
    assert compare_tables(a, e).ok


def test_topn_ties_at_the_cutoff_may_pick_different_rows():
    e = T(k=[1, 2, 3], v=[10.0, 5.0, 5.0])         # LIMIT 3 cut inside a tie group
    a = T(k=[1, 2, 9], v=[10.0, 5.0, 5.0])
    assert compare_tables(a, e).verdict == "wrong"           # strict mode sees a key mismatch
    assert compare_tables(a, e, topn_metric=1).ok             # top-N mode tolerates boundary ties


def test_topn_still_catches_a_wrong_row_above_the_cutoff():
    e = T(k=[1, 2, 3], v=[10.0, 7.0, 5.0])
    a = T(k=[1, 9, 3], v=[10.0, 7.0, 5.0])
    assert compare_tables(a, e, topn_metric=1).verdict == "wrong"


def test_topn_catches_wrong_metric_values():
    e = T(k=[1, 2], v=[10.0, 7.0])
    a = T(k=[1, 2], v=[10.0, 9.0])
    assert compare_tables(a, e, topn_metric=1).verdict == "wrong"


def test_topn_wrong_row_just_above_the_cutoff_is_caught():
    # the old band was 1% of the cutoff (=1.0 here) and let a wrong 100.5 row through
    e = T(k=[1, 2, 3], v=[200.0, 100.5, 100.0])
    a = T(k=[1, 9, 3], v=[200.0, 100.5, 100.0])
    assert compare_tables(a, e, topn_metric=1).verdict == "wrong"


def test_topn_rows_tied_within_rtol_may_swap():
    e = T(k=[1, 2, 3], v=[200.0, 100.0, 100.0])
    a = T(k=[1, 2, 9], v=[200.0, 100.0, 100.0000001])
    assert compare_tables(a, e, topn_metric=1).ok


def test_topn_with_duplicate_keys_neither_crashes_nor_hides_a_missing_row():
    e = T(k=["x", "x", "y"], v=[300.0, 200.0, 100.0])
    assert compare_tables(T(k=["x", "x", "y"], v=[300.0, 200.0, 100.0]), e, topn_metric=1).ok
    a = T(k=["x", "y", "y"], v=[300.0, 200.0, 100.0])          # second "x" replaced by "y"
    assert compare_tables(a, e, topn_metric=1).verdict == "wrong"


def test_order_by_violation_is_wrong_even_when_the_rows_are_right():
    e = T(k=["a", "b", "c"], v=[3.0, 2.0, 1.0])
    a = T(k=["b", "a", "c"], v=[2.0, 3.0, 1.0])
    assert compare_tables(a, e).ok                                       # values only
    r = compare_tables(a, e, order_by=[(1, True)])
    assert r.verdict == "wrong" and "ORDER BY" in r.message
    assert compare_tables(e, e, order_by=[(1, True)]).ok


def test_order_by_uses_later_keys_for_ties_and_tolerates_float_noise():
    e = T(n=[5, 5, 4], k=["a", "b", "z"])
    assert compare_tables(e, e, order_by=[(0, True), (1, False)]).ok
    bad = T(n=[5, 5, 4], k=["b", "a", "z"])
    assert compare_tables(bad, e, order_by=[(0, True), (1, False)]).verdict == "wrong"
    noisy = T(v=[10.0000000001, 10.0, 9.0], k=["a", "b", "c"])       # tied within rtol -> k decides
    assert compare_tables(noisy, noisy, order_by=[(0, True), (1, False)]).ok


# ---- report helpers -------------------------------------------------------------------------
from tpch_user import report  # noqa: E402


def test_unchecked_and_wrong_results_never_win():
    assert report.is_valid({"first_ms": 50.0, "verdict": "ok"})
    assert report.is_valid({"first_ms": 5.0, "verdict": "ref"})
    for v in ("wrong", "unchecked", "error", None):
        assert not report.is_valid({"first_ms": 1.0, "verdict": v})
    assert not report.is_valid({"first_ms": 1.0, "verdict": "ok", "error": "TIMEOUT after 5s"})
    assert report.fmt_cell({"first_ms": 12.0, "verdict": "unchecked"}) == "12 ?"
    assert report.fmt_cell({"first_ms": 12.0, "verdict": "wrong"}) == "12 WRONG"


def test_csv_roundtrip_keeps_types(tmp_path):
    import csv
    p = tmp_path / "r.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=report.CSV_FIELDS)
        w.writeheader()
        w.writerow(report.new_row({"engine": "polars", "sf": "1", "query": 3, "first_ms": 12.5,
                                   "verdict": "ok", "correct": True, "max_rel_err": 0.0}))
        w.writerow(report.new_row({"engine": "mx-gpu", "sf": "10", "query": 21, "error": "CRASH rc=-11"}))
    a, b = report.read_csv(p)
    assert (a["sf"], a["query"], a["first_ms"], a["correct"]) == ("1", 3, 12.5, True)
    assert b["first_ms"] is None and b["error"].startswith("CRASH")


def test_import_plus_first_cell_sums_the_two():
    assert report.fmt_sum_cell({"import_ms": 700.0, "first_ms": 100.0, "verdict": "ok"}) == "800"


def test_order_by_distinct_floats_one_part_in_a_million_apart_are_not_ties():
    t = T(v=[9999.99, 9999.98], k=["z", "a"])            # v DESC then k: correct as given
    assert compare_tables(t, t, order_by=[(0, True), (1, False)]).ok


def test_official_answer_check_flags_a_wrong_reference():
    from tpch_user.official import check_official
    ok = pa.table({"revenue": [123141078.2283]})
    assert check_official(6, ok) == "ok"
    assert check_official(6, pa.table({"revenue": [8.865e7]})).startswith("row")
    assert check_official(9, ok) is None                              # no published answer listed
    assert check_official(14, pa.table({"promo_revenue": [16.3807786]})) == "ok"
    assert check_official(14, pa.table({"promo_revenue": [17.0]})).startswith("row")
