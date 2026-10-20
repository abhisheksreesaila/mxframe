"""F-058: an in-memory query must not pay for machinery it cannot use. Importing the streamed-execution and spill
modules on every first compute() cost ~1.5-2 ms of a small query's clock (random SF1 seed 2: 5.8 -> 7.9 ms)."""
import subprocess
import sys


def _fresh(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()


def test_an_in_memory_query_does_not_import_streaming():
    out = _fresh(
        "import sys, pyarrow as pa\n"
        "from omxcore import LazyFrame, col\n"
        "LazyFrame(pa.table({'a': [1, 2, 2]})).groupby('a').agg(col('a').count().alias('n')).compute(device='cpu')\n"
        "print('omxcore.streamed' in sys.modules, 'omxcore.spill' in sys.modules)\n")
    assert out == ["False", "False"]


def test_a_parquet_query_still_streams(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"a": list(range(1000))}), tmp_path / "t.parquet", row_group_size=100)
    out = _fresh(
        "import os, sys\n"
        "os.environ['MXFRAME_HOST_MEM_BUDGET'] = '2K'\n"
        "import omxcore as mx\n"
        "from omxcore import col\n"
        f"lf = mx.scan_parquet({str(tmp_path / 't.parquet')!r}).groupby().agg(col('a').sum().alias('s'))\n"
        "t = lf.compute(device='cpu')\n"
        "print(t.column('s')[0].as_py(), lf.last_compile_provenance['path'])\n")
    assert out == ["499500", "streamed"]


def test_a_cold_auto_query_is_not_priced_but_queues_the_gpu_warmup():
    """F-058: with the GPU not started, a query cheaper on the CPU than the GPU's start-up runs there without loading the
    calibration on its clock (~3 ms); the warm-up for later queries is still queued, after it returns."""
    out = _fresh(
        "import sys, time, numpy as np, pyarrow as pa\n"
        "from omxcore import LazyFrame, col, auto_mode as am, coproc_fused as cf\n"
        "cf.IDLE_BEFORE_BACKGROUND_S = 0.0\n"
        "seen = []\n"
        "am.start_gpu_warmup_if_any = lambda: seen.append('calibration' in sys.modules)\n"
        "t = pa.table({'k': np.arange(1_000_000) % 7, 'v': np.ones(1_000_000)})\n"
        "lf = LazyFrame(t).groupby('k').agg(col('v').sum().alias('s'))\n"
        "lf.compute(device='auto')\n"
        "during = 'omxcore.calibration' in sys.modules\n"
        "time.sleep(0.3)\n"
        "print(lf.last_compile_provenance['auto_reason'].startswith('gpu not started'), during, len(seen))\n")
    assert out == ["True", "False", "1"]


def test_the_cold_rates_bound_every_default_native_price():
    """The cold shortcut's per-operator rates must be at least the dearest default native fit of their kind."""
    from omxcore import auto_mode as am, calibration as cal
    fits = cal._DEFAULT_FITS
    dearest = lambda base: max(v[1] for k, v in fits.items() if k.split(":")[0] == base)       # noqa: E731
    r = am.COLD_NATIVE_MS_PER_MROW
    assert r["join"] >= dearest("native_join@cpu") and r["aggregate"] >= dearest("native_group@cpu")
    assert r["aggregate"] >= dearest("native_global@cpu") and r["sort"] >= fits["sort_topk@cpu"][1]
    assert r["filter"] >= fits["native_filter@cpu"][1] - fits["native_global@cpu"][1]


def test_a_dear_plan_is_priced_even_with_a_cold_gpu():
    """Several dear operators over the same rows can cost more than the start-up: those are priced as before."""
    import numpy as np
    import pyarrow as pa
    from omxcore import LazyFrame, col, auto_mode as am
    n = 6_000_000
    t = pa.table({"k": np.arange(n) % 1_000_000, "v": np.ones(n)})
    cheap = LazyFrame(t).groupby().agg(col("v").sum().alias("s")).plan
    dear = LazyFrame(t).join(LazyFrame(t), on="k").groupby("k").agg(col("v").sum().alias("s")).sort("s").plan
    assert am._cold_cpu_bound_ms(cheap, n) < am.COLD_GPU_START_MS
    assert am._cold_cpu_bound_ms(dear, 2 * n) > am.COLD_GPU_START_MS


def test_no_gpu_stops_queueing_warmups():
    out = _fresh(
        "import numpy as np, pyarrow as pa\n"
        "from omxcore import LazyFrame, col, auto_mode as am, calibration as cal\n"
        "class C: has_gpu = False\n"
        "cal.get_calibration = lambda: C()\n"
        "t = pa.table({'k': np.arange(1_000_000) % 7, 'v': np.ones(1_000_000)})\n"
        "lf = LazyFrame(t).groupby('k').agg(col('v').sum().alias('s'))\n"
        "d1 = am.choose_device(lf.plan)\n"
        "am.start_gpu_warmup_if_any()            # what the queued warm-up runs\n"
        "d2 = am.choose_device(lf.plan)\n"
        "print(d1.warm_later, d2.warm_later, d2.reason.replace(' ', '_'))\n")
    assert out == ["True", "False", "no_gpu"]
