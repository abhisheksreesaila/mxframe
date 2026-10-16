"""F-033: which plans reach the native hooks, and why they decline (coverage before scoping a milestone).

    pixi run python scripts/native_coverage.py [--sf 1] [--queries 1,2,...]      # TPC-H, one fresh process per query

Per query it prints one JSON line: every computed plan's path (`provenance`), every native hook call
(ran, or declined + reason), the PyArrow compute calls on >= 1000 rows that ran (row-level work outside
Mojo), and the decline counts.  "Fully native" = every compute() took a native path and no PyArrow
compute call touched row data (scalar reads between stages are not row work).

The random suite has its own log: `scripts/bench_random/run.py --engines duckdb,mx` writes
`native_calls`, `decline_counts`, `declines` and `paths` per query next to the DuckDB comparison.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

HOOKS = ("try_execute", "try_execute_join", "try_filter", "try_distinct", "try_count_distinct",
         "try_gather", "try_topk", "try_filter_topk", "try_str_slice", "try_str_slice_in")
MIN_ROWS = 1000


def install():
    """Wrap the native hooks and pyarrow's compute dispatch; returns the shared log."""
    import conftest  # noqa: F401  (import mxframe from this checkout)
    import pyarrow._compute as _c
    import mxframe.native_exec as nx
    log = {"hooks": [], "arrow": []}
    for name in HOOKS:
        real = getattr(nx, name, None)
        if real is None:
            continue

        def wrap(*a, _real=real, _name=name, **k):
            r = _real(*a, **k)
            why = str(nx.last_decline() or nx.last_filter())[:80]
            log["hooks"].append(f"{_name}:" + ("ran" if r is not None else f"declined({why})"))
            return r
        setattr(nx, name, wrap)
    real_call = _c.call_function

    def note(name, args):
        n = 0
        if name == "take" and len(args) > 1:        # a take is as big as its indices, not its source
            args = args[1:2]
        for a in args:
            try:
                n = max(n, len(a))
            except TypeError:
                pass
        if n >= MIN_ROWS:
            log["arrow"].append(f"{name}[{n}]")
            if os.environ.get("MXCOV_STACK"):          # who asked for it (stderr)
                import traceback
                where = [f"{s.filename.rsplit('/', 1)[-1]}:{s.lineno}:{s.name}" for s in traceback.extract_stack()[-8:-3]]
                print(f"ARROW {name}[{n}] <- " + " <- ".join(reversed(where)), file=sys.stderr)

    def call(name, args, *rest, **kw):
        note(name, args)
        return real_call(name, args, *rest, **kw)
    _c.call_function = call
    # `pc.<fn>` (mxframe._lazy_pc) calls Function.call directly: wrap what it hands out
    from mxframe import _lazy_pc
    real_wrapper, real_real = _lazy_pc._wrapper, _lazy_pc._real

    def logged(name, f):
        if f is None or name in _lazy_pc._REBUILT:
            return f
        return lambda *args, **kw: (note(name, args), f(*args, **kw))[1]
    _lazy_pc._wrapper = lambda name: logged(name, real_wrapper(name))
    _lazy_pc._real = lambda name: logged(name, real_real(name))
    _lazy_pc.pc._cache.clear()
    return log, nx


def one_tpch(q, sf):
    log, nx = install()
    from tpch_user import provenance, q_mx
    data = str(Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data")) / f"tpch_sf{sf}")
    tables = q_mx.load_mem(q, data)
    prov = provenance.install()
    nx.reset_decline_counts()
    log["hooks"].clear()
    log["arrow"].clear()
    q_mx.run_mem(q, tables, "cpu")
    print("@@COV " + json.dumps({"q": q, "paths": prov, "hooks": log["hooks"], "arrow": log["arrow"],
                                  "declines": nx.decline_counts()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", default="1")
    ap.add_argument("--queries", default=",".join(map(str, range(1, 23))))
    ap.add_argument("--one", type=int, help="(internal) run this query in this process")
    a = ap.parse_args()
    if a.one:
        return one_tpch(a.one, a.sf)
    env = dict(os.environ, MXFRAME_STRICT_NATIVE="1", CUDA_VISIBLE_DEVICES="-1", PYTHONUNBUFFERED="1")
    for q in map(int, a.queries.split(",")):
        p = subprocess.run([sys.executable, __file__, "--one", str(q), "--sf", a.sf], capture_output=True, text=True,
                           env=env, cwd=ROOT)
        ls = [l for l in p.stdout.splitlines() if l.startswith("@@COV ")]
        print(ls[-1][6:] if ls else json.dumps({"q": q, "error": (p.stderr.strip().splitlines() or ["?"])[-1][:300]}),
              flush=True)


if __name__ == "__main__":
    main()
