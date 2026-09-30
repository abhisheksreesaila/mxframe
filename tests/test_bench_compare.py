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
    r = compare_tables(T(k=[1], v=[101.0]), T(k=[1], v=[100.0]))
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
