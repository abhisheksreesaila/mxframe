"""F-073: the random suite's result checker (scripts/bench_random/run.py `compare`) on ordered results."""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "bench_random"))
import run  # noqa: E402

SPEC = {"select": None, "group": ["k"], "aggs": [["mean", "x", "a0"]], "distinct": None,
        "order": [["a0", True], ["k", False]], "limit": None}


def verdict(tmp_path, ref, got):
    paths = []
    for name, rows in (("ref", ref), ("got", got)):
        p = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table({"k": [r[0] for r in rows], "a0": [r[1] for r in rows]}), p)
        paths.append(str(p))
    return run.compare(SPEC, *paths)


def test_near_equal_averages_in_value_order_are_not_reported_out_of_order(tmp_path):
    # seed 7129: averages ~1e-6 apart, key order not ascending inside the run. The rows are right
    # (a0 descending); the old checker called them tied and then demanded ascending keys.
    rows = [("DELIVER", 0.046184970130551756), ("COLLECT", 0.046184209592052196),
            ("TAKE", 0.046174495787228514), ("MAIL", 0.04617449562471585)]
    assert verdict(tmp_path, rows, rows) == ("ok", "")


def test_exactly_equal_averages_must_follow_the_key_order(tmp_path):
    ref = [("a", 0.5), ("b", 0.5)]
    assert verdict(tmp_path, ref, [("b", 0.5), ("a", 0.5)]) == ("WRONG", "not in ORDER BY order")


def test_clearly_wrong_direction_is_still_wrong(tmp_path):
    ref = [("a", 0.5), ("b", 0.4)]
    assert verdict(tmp_path, ref, [("b", 0.4), ("a", 0.5)]) == ("WRONG", "not in ORDER BY order")


def test_last_bit_differences_between_engines_may_swap_rows(tmp_path):
    ref = [("a", 0.5000000000000001), ("b", 0.5)]
    got = [("b", 0.5), ("a", 0.5000000000000001)]
    assert verdict(tmp_path, ref, got)[0] == "ok"
