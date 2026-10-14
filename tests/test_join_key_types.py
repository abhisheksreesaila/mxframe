"""F-018: join key types. Integer keys of different widths join after widening to the wider type, signed
with unsigned included when a wider signed type holds both (uint32 with int32/int64, uint8 with int8...).
uint64 with any signed integer is refused with TypeError on every path (single, composite, dense or not)."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame


def _t(name, vals, typ):
    return pa.table({name: pa.array(vals, typ), name + "v": pa.array([float(v) for v in vals])})


@pytest.mark.parametrize("lt,rt", [(pa.uint32(), pa.int64()), (pa.uint32(), pa.int32()), (pa.uint8(), pa.int8()),
                                   (pa.int16(), pa.int64()), (pa.uint16(), pa.uint32())])
@pytest.mark.parametrize("wide", [False, True])
def test_widening_pairs_join_on_every_path(lt, rt, wide):
    base = 3_000_000 if wide and lt.bit_width >= 32 and rt.bit_width >= 32 else 0
    a = _t("k", [base + i for i in (1, 2, 3, 4)], lt)
    b = _t("k", [base + i for i in (2, 4, 6)], rt)
    got = LazyFrame(a).join(LazyFrame(b), on="k").compute(device="cpu")
    assert sorted(got.column("k").to_pylist()) == [base + 2, base + 4]
    a2 = a.append_column("j", pa.array([1, 1, 2, 2], lt))
    b2 = b.append_column("j", pa.array([1, 2, 2], rt))
    got = LazyFrame(a2).join(LazyFrame(b2), on=["k", "j"]).compute(device="cpu")
    assert sorted(got.column("k").to_pylist()) == [base + 2, base + 4]


@pytest.mark.parametrize("other", [pa.int64(), pa.int32()])
@pytest.mark.parametrize("wide", [False, True])
def test_uint64_with_signed_is_refused_everywhere(other, wide):
    base = 3_000_000 if wide else 0
    a = _t("k", [base + 1, base + 2], pa.uint64())
    b = _t("k", [base + 2, base + 3], other)
    with pytest.raises(TypeError):
        LazyFrame(a).join(LazyFrame(b), on="k").compute(device="cpu")
    a2 = a.append_column("j", pa.array([1, 2], pa.uint64()))
    b2 = b.append_column("j", pa.array([2, 3], other))
    with pytest.raises(TypeError):
        LazyFrame(a2).join(LazyFrame(b2), on=["k", "j"]).compute(device="cpu")


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_composite_uint64_keys_above_2_63(device):
    from mxframe import LazyFrame, col
    top = np.uint64(2**64 - 20)
    a = pa.table({"k1": pa.array(np.array([top, top + np.uint64(1), top, 5], dtype=np.uint64)),
                  "k2": pa.array(np.array([1, 2, 3, 1], dtype=np.int32)), "v": pa.array([1.0, 2.0, 3.0, 4.0])})
    b = pa.table({"k1": pa.array(np.array([top, top + np.uint64(1), 5], dtype=np.uint64)),
                  "k2": pa.array(np.array([1, 2, 1], dtype=np.int32)), "w": pa.array([10.0, 20.0, 30.0])})
    got = LazyFrame(a).join(LazyFrame(b), on=["k1", "k2"]).groupby().agg(col("v").sum().alias("s"), col("w").sum().alias("t")).compute(device=device)
    # DuckDB can't take uint64 values this large; the matches are (top,1), (top+1,2), (5,1)
    assert got["s"][0].as_py() == 1.0 + 2.0 + 4.0 and got["t"][0].as_py() == 10.0 + 20.0 + 30.0
