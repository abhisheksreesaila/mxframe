"""MX frame: a Mojo dataframe engine, usable with no Python (docs/mxframe-mojo-api.md, docs/briefs/mojo-api.md).

Build: `bash scripts/build_mojo_pkg.sh` -> build/mxframe.mojoc. Tests: `pixi run mojo-test`.
"""
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL
from .expr import Expr, col, lit, least, greatest
from .query import Query, QueryResult, Agg, sum, count, mean, min_of, max_of, CPU, GPU, AUTO
from .parquet import write_parquet, UNCOMPRESSED, SNAPPY, ZSTD
