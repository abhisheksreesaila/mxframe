"""F-107: a residency finalizer that runs on a thread already holding ``coproc_fused._lock`` must not hang.

Seen four times on 2026-10-16: a full-suite process idle at 0% CPU with every thread in
``futex_do_wait``, the last stack in ``coproc_fused`` while it started a thread.  ``weakref.finalize``
callbacks (``_evict``) run on whatever thread drops the last reference or runs the cyclic GC, and a
garbage collection can start at any allocation, including inside ``with _lock:`` (``_start_thread``
holds it across ``Thread.start()``).  ``_evict`` took the same non-reentrant lock: the thread waited for
itself.  Each check runs in a child process with a timeout so a regression fails instead of hanging the suite.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent

PREAMBLE = """
import gc, sys, time
sys.path.insert(0, ".")
import conftest  # noqa: F401
import numpy as np
import pyarrow as pa
from omxcore import coproc_fused as cf

class Dc:
    freed = 0
    def free(self):
        Dc.freed += 1

def make_entry(key):
    owner = pa.array(np.arange(10))
    cf._cache[key] = cf._Entry(Dc(), 80, [], sig=())
    cf._bytes += 80
    cf._finalized.add(key)
    import weakref
    weakref.finalize(owner, cf._evict, key)
    return owner
"""


def _run(body, timeout=30):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(PREAMBLE + body)], capture_output=True,
                          text=True, cwd=str(HERE), timeout=timeout)


def test_owner_dying_while_the_lock_is_held_evicts_without_deadlock():
    r = _run("""
owner = make_entry(("k1",))
with cf._lock:
    del owner                      # the finalizer runs here, on a thread that holds the lock
print("held-ok", len(cf._cache), Dc.freed)
cf._drain_evictions()
print("after", len(cf._cache), Dc.freed, cf._bytes)
""")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[:2] == ["held-ok 1 0", "after 0 1 0"]


def test_cyclic_gc_inside_the_lock_evicts_without_deadlock():
    r = _run("""
owner = make_entry(("k2",))
cycle = [owner]; cycle.append(cycle)       # only the cyclic GC can free it
del owner, cycle
with cf._lock:
    gc.collect()
print("held-ok", len(cf._cache))
cf._drain_evictions()
print("after", len(cf._cache), Dc.freed)
""")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[:2] == ["held-ok 1", "after 0 1"]


def test_owner_dying_without_the_lock_evicts_at_once():
    r = _run("""
owner = make_entry(("k3",))
del owner
print(len(cf._cache), Dc.freed)
""")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "0 1"
