"""A group-by on five or more key columns keeps every key (found by the SQL TPC-H Q18, F-065): the
multi-key encoder built for the Mojo group_composite kernel takes four key slots, and a fifth key was
dropped from the composite, so a fifth key's values came back wrong (every group got one value)."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col


def _table(n_keys, groups=57, rows_per_group=7):
    g = np.repeat(np.arange(groups), rows_per_group)
    makers = [lambda g: pa.array([f"name{x:05d}" for x in g]),                     # string
              lambda g: pa.array(g.astype(np.int64) + 1000),
              lambda g: pa.array(g.astype(np.int64) * 7919),
              lambda g: pa.array(g.astype(np.int32) + 8000),
              lambda g: pa.array((g * 1.5 + 100).astype(np.float32)),              # float32
              lambda g: pa.array(g.astype(np.int64) % 5)]
    cols = {f"k{i}": makers[i](g) for i in range(n_keys)}
    cols["v"] = pa.array(np.arange(len(g), dtype=np.int64))
    return g, pa.table(cols)


@pytest.mark.parametrize("n_keys", [4, 5, 6])
def test_every_key_column_comes_back_with_its_own_values(n_keys):
    g, t = _table(n_keys)
    keys = [f"k{i}" for i in range(n_keys)]
    got = LazyFrame(t).groupby(*keys).agg(col("v").sum().alias("s")).compute(device="cpu")
    assert got.num_rows == 57
    rows = sorted(zip(*[got.column(c).to_pylist() for c in keys + ["s"]]), key=lambda r: r[-1])
    first = {}
    for i, key in enumerate(zip(*[t.column(c).to_pylist() for c in keys])):
        first.setdefault(key, []).append(i)
    assert {r[:-1]: r[-1] for r in rows} == {k: sum(v) for k, v in first.items()}
