"""F-063: run TPC-H queries on device="auto" in fresh processes, many times, and count
non-zero exits (a glibc abort is -6, a segfault -11).

    pixi run python scripts/stress_auto_crash.py --runs 100 --queries 17 --load 2

``--load N`` runs N extra worker loops of mixed CPU queries in parallel (keep it small:
the machine is shared).  Exit code 1 if any run failed; failing stderr tails are printed.
"""
import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tests" / "_tpch_runner.py"
DATA = Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data")) / "tpch_sf1_exact"


def one(q, device, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p = subprocess.run([sys.executable, "-X", "faulthandler", str(RUNNER), str(DATA), str(q), device, out],
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--queries", default="17")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load", type=int, default=0)
    ap.add_argument("--scratch", default=str(ROOT / "scratch"))
    a = ap.parse_args()
    Path(a.scratch).mkdir(exist_ok=True)
    qs = [int(x) for x in a.queries.split(",")]
    stop = threading.Event()

    def loader(i):
        mix = [1, 3, 5, 9, 18, 21, 17, 6]
        n = 0
        while not stop.is_set():
            one(mix[(n + i) % len(mix)], "cpu", f"{a.scratch}/load{i}.arrow")
            n += 1
    for i in range(a.load):
        threading.Thread(target=loader, args=(i,), daemon=True).start()
    fails, codes = [], {}
    for r in range(a.runs):
        q = qs[r % len(qs)]
        rc, err = one(q, a.device, f"{a.scratch}/out.arrow")
        codes[rc] = codes.get(rc, 0) + 1
        if rc != 0:
            fails.append((r, q, rc))
            print(f"run {r} Q{q} exit {rc}\n" + "\n".join(err.strip().splitlines()[-12:]), flush=True)
    stop.set()
    print(f"runs={a.runs} load={a.load} exit codes={codes} load1={os.getloadavg()[0]:.1f}")
    sys.exit(1 if fails else 0)


main()
