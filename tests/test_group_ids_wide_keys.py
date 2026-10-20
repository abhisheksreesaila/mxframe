"""F-018 (found through TPC-H Q21): group keys whose combined cardinality exceeds int32 / int64."""
import numpy as np
import pyarrow as pa
import pytest

from omxcore import LazyFrame, col


def _table(n, cards, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"k{i}": pa.array(rng.integers(0, c, n).astype(np.int64)) for i, c in enumerate(cards)}
    cols["v"] = pa.array(rng.random(n))
    return pa.table(cols)


def _distinct_count(t, names):
    return len({tuple(r) for r in zip(*(t.column(n).to_pylist() for n in names))})


@pytest.mark.parametrize("cards", [(100_000, 100_000), (60_000, 50_000, 40_000)])
def test_distinct_keeps_every_distinct_combination_beyond_int32(cards):
    t = _table(300_000, cards)
    names = [f"k{i}" for i in range(len(cards))]
    got = LazyFrame(t).select(*names).distinct().compute(device="cpu")
    assert got.num_rows == _distinct_count(t, names)


def test_groupby_with_more_key_combinations_than_int64():
    cards = (100_000, 100_000, 100_000, 100_000)         # 1e20 combinations
    t = _table(200_000, cards, seed=1)
    names = [f"k{i}" for i in range(4)]
    got = LazyFrame(t).groupby(*names).agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu")
    assert got.num_rows == _distinct_count(t, names)
    assert sum(got.column("n").to_pylist()) == t.num_rows
    np.testing.assert_allclose(sum(got.column("s").to_pylist()), sum(t.column("v").to_pylist()), rtol=1e-9)
