"""MX frame: a Mojo dataframe engine, usable with no Python (docs/mxframe-mojo-api.md, docs/briefs/mojo-api.md).

Build: `bash scripts/build_mojo_pkg.sh` -> build/mxframe.mojoc. Tests: `pixi run mojo-test`.
"""
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .expr import Expr, col, lit, least, greatest
from .query import (Query, QueryResult, Agg, sum, count, mean, min_of, max_of, count_distinct, percentile, median, top_k,
                    CPU, GPU, AUTO, MORSELS_AUTO, MORSELS_ON, MORSELS_OFF)
from .parquet import write_parquet, read_parquet, scan_parquet, ParquetResult, ParquetScan, ParquetWriter, UNCOMPRESSED, SNAPPY, ZSTD
from .csv import read_csv, CsvOptions, CsvResult, CsvColumn, FailedRow
from .frame import Frame, Formula
from .gpu import DeviceTable, gpu_available
from .sql import parse_sql, SqlStatement
from .bind import Catalog, SqlSource, BoundQuery, bind_sql, SRC_TABLE, SRC_FRAME, SRC_SCAN
from .planner import (Planner, RulePlanner, PlannerOptions, PhysicalPlan, KeyLayout, Stats, ColStat, GROUP_AUTO, GROUP_DENSE, GROUP_HASHED,
                      G_NONE, G_DENSE, G_HASHED)
from .plan import LogicalPlan, PlanNode, build_plan
from .sqlexec import sql, sql_with, SqlResult
