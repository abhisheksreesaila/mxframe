"""F-058: an in-memory query must not pay for machinery it cannot use. Importing the streamed-execution and spill
modules on every first compute() cost ~1.5-2 ms of a small query's clock (random SF1 seed 2: 5.8 -> 7.9 ms)."""
import subprocess
import sys


def _fresh(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()


def test_an_in_memory_query_does_not_import_streaming():
    out = _fresh(
        "import sys, pyarrow as pa\n"
        "from mxframe import LazyFrame, col\n"
        "LazyFrame(pa.table({'a': [1, 2, 2]})).groupby('a').agg(col('a').count().alias('n')).compute(device='cpu')\n"
        "print('mxframe.streamed' in sys.modules, 'mxframe.spill' in sys.modules)\n")
    assert out == ["False", "False"]


def test_a_parquet_query_still_streams(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"a": list(range(1000))}), tmp_path / "t.parquet", row_group_size=100)
    out = _fresh(
        "import os, sys\n"
        "os.environ['MXFRAME_HOST_MEM_BUDGET'] = '2K'\n"
        "import mxframe as mx\n"
        "from mxframe import col\n"
        f"lf = mx.scan_parquet({str(tmp_path / 't.parquet')!r}).groupby().agg(col('a').sum().alias('s'))\n"
        "t = lf.compute(device='cpu')\n"
        "print(t.column('s')[0].as_py(), lf.last_compile_provenance['path'])\n")
    assert out == ["499500", "streamed"]


def test_a_cold_auto_query_is_not_priced_but_queues_the_gpu_warmup():
    """F-058: with the GPU not started, a query under COLD_GPU_SOURCE_ROWS runs on the CPU without loading the
    calibration on its clock (~3 ms); the warm-up for later queries is still queued, after it returns."""
    out = _fresh(
        "import sys, time, numpy as np, pyarrow as pa\n"
        "from mxframe import LazyFrame, col, auto_mode as am, coproc_fused as cf\n"
        "cf.IDLE_BEFORE_BACKGROUND_S = 0.0\n"
        "seen = []\n"
        "am.start_gpu_warmup_if_any = lambda: seen.append('calibration' in sys.modules)\n"
        "t = pa.table({'k': np.arange(1_000_000) % 7, 'v': np.ones(1_000_000)})\n"
        "lf = LazyFrame(t).groupby('k').agg(col('v').sum().alias('s'))\n"
        "lf.compute(device='auto')\n"
        "during = 'mxframe.calibration' in sys.modules\n"
        "time.sleep(0.3)\n"
        "print(lf.last_compile_provenance['auto_reason'].startswith('gpu not started'), during, len(seen))\n")
    assert out == ["True", "False", "1"]
