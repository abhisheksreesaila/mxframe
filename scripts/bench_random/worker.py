"""One random query, one engine, in a FRESH process (F-044): load the columns the query uses into the
engine's own in-memory form (untimed), run the query once (timed: query + materialized result),
write the result to Parquet. Prints `@@RESULT {json}`.

    python scripts/bench_random/worker.py ENGINE SPEC_JSON DATA_DIR RESULT_PATH [--shuffle]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _shuffled(path, cols, seed):
    import numpy as np
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=cols)
    return t.take(np.random.default_rng(seed).permutation(t.num_rows))


def main():
    engine, spec_path, data_dir, out = sys.argv[1:5]
    shuffle = "--shuffle" in sys.argv
    import pyarrow  # noqa: F401  (staging: DuckDB would import it inside its first .arrow(), ~70 ms)
    spec = json.load(open(spec_path))
    import gen
    path = lambda n: os.path.join(data_dir, f"{n}.parquet")
    res = {"engine": engine, "seed": spec["seed"]}
    if engine == "mx":
        import omxcore as mx
        from omxcore import LazyFrame
        import omxcore.native_exec as nx
        if shuffle:
            tabs = {}
            for n, cols in spec["tables"].items():
                t = mx.read_parquet_table(path(n), columns=cols, dictionary="auto")
                import numpy as np
                tabs[n] = t.take(np.random.default_rng(len(n)).permutation(t.num_rows))
        else:
            tabs = {n: mx.read_parquet_table(path(n), columns=cols, dictionary="auto")
                    for n, cols in spec["tables"].items()}
        t = {n: LazyFrame(v) for n, v in tabs.items()}
        declines, native = [], []
        for name, log in (("_decline", declines), ("_filter_decline", declines)):
            real = getattr(nx, name)
            setattr(nx, name, lambda r, _real=real, _log=log: (_log.append(r), _real(r))[1])
        real_topk = nx.try_topk
        nx.try_topk = lambda *a, **k: (lambda r: (declines.append("top-k declined") if r is None else None, r)[1])(real_topk(*a, **k))
        for name in ("_call", "_call_filter", "_call_distinct", "_call_count_distinct", "_call_gather"):
            if hasattr(nx, name):
                real = getattr(nx, name)
                setattr(nx, name, lambda *a, _real=real, _n=name: (native.append(_n), _real(*a))[1])
        device = os.environ.get("MXBENCH_DEVICE", "cpu")
        from tpch_user import provenance        # F-055: which path each compute() took
        prov = provenance.install()
        t0 = time.perf_counter()
        result = gen.run_mx(spec, t, device)
        res["paths"] = provenance.summary(prov)
        res["ms"] = (time.perf_counter() - t0) * 1e3
        res["native_calls"] = native
        res["decline_counts"] = nx.decline_counts()          # {reason: count}, unexpected exceptions as "raised: <Type>"
        res["declines"] = sorted(set(declines))
    elif engine == "polars":
        import polars as pl
        if shuffle:
            t = {n: pl.from_arrow(_shuffled(path(n), cols, len(n))).lazy() for n, cols in spec["tables"].items()}
        else:
            t = {n: pl.read_parquet(path(n), columns=cols).lazy() for n, cols in spec["tables"].items()}
        t0 = time.perf_counter()
        result = gen.run_polars(spec, t)
        res["ms"] = (time.perf_counter() - t0) * 1e3
        result = result.to_arrow()
    else:
        import duckdb
        con = duckdb.connect()
        for n, cols in spec["tables"].items():
            if shuffle:
                tmp = _shuffled(path(n), cols, len(n))
                con.register("tmp_" + n, tmp)
                con.execute(f"CREATE TABLE {n} AS SELECT * FROM tmp_{n}")
                con.unregister("tmp_" + n)
            else:
                con.execute(f"CREATE TABLE {n} AS SELECT {', '.join(cols)} FROM read_parquet('{path(n)}')")
        t0 = time.perf_counter()
        result = gen.run_duckdb(spec, con)
        res["ms"] = (time.perf_counter() - t0) * 1e3
    import pyarrow.parquet as pq
    pq.write_table(result, out)
    res["rows"] = result.num_rows
    print("@@RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
