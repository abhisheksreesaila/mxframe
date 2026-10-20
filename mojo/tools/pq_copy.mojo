# Reads the Parquet file named first and writes it to the second path (zstd, 1,000-row groups): used by the tests to prove that
# what Mojo reads, it writes back with every value (F-157: FLOAT columns come back as DOUBLE).
from std.sys import argv
from omxcore import read_parquet, write_parquet, ZSTD


def main() raises:
    var r = read_parquet(String(argv()[1]))
    var t = r.take_table()
    write_parquet(String(argv()[2]), t, r.metadata, ZSTD, 1000)
