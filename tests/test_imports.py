import mxframe
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame
import pyarrow as pa

def test_imports():
    assert col is not None
    assert lit is not None
    assert LazyFrame is not None

def test_lazyframe_init():
    table = pa.table({'a': [1, 2, 3]})
    lf = LazyFrame(table)
    assert lf is not None