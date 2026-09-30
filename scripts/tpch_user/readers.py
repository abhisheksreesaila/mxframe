"""Parquet readers. Each engine reads ONLY the columns its query uses (Polars/DuckDB prune
automatically; pandas and MXFrame get an explicit column list, as a knowledgeable user would).

(Named readers.py rather than io.py so it can never shadow the stdlib `io`.)
"""
from __future__ import annotations

import os


def table_path(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, f"{name}.parquet")


def mx_read(path: str, columns: list[str]):
    """MXFrame input = PyArrow table. Swap the body for `mx.scan_parquet` when it exists."""
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns)


def pandas_read(path: str, columns: list[str]):
    import pandas as pd
    return pd.read_parquet(path, columns=columns)
