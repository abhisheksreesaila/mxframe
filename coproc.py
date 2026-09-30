"""Router runtime: CPU and GPU workers pull morsels from one queue (co-processing).

After morsel-driven parallelism (Leis et al., SIGMOD 2014) and HetExchange's
Router (Chrysogelos et al., PVLDB 2019): the rows of one pipeline are cut into
morsels; each worker (the multi-core CPU kernels, or the GPU) pulls the next
morsel when it is free, so the faster device naturally takes more rows and
estimate errors correct themselves. Each worker produces a partial result per
morsel; `merge` combines the partials in row order.

A worker may split a morsel into `prepare` (e.g. copy to the GPU) and
`compute`; the runtime prepares morsel k+1 while morsel k computes, so the copy
overlaps the kernel. The CPU kernels are called through ctypes, which releases
the GIL, so Python threads here really run the devices concurrently.

A `fallible` worker (the GPU) that raises hands its rows back to the queue and
stops; the other workers redo them (fragment fallback). Any other error raises.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

Range = Tuple[int, int]


class MorselQueue:
    """Thread-safe cursor over rows [start, end), plus ranges handed back by failed workers."""

    def __init__(self, end: int, start: int = 0, returned: Optional[List[Range]] = None,
                 lock: Optional[threading.Lock] = None):
        self._next = start
        self._end = end
        self._lock = lock or threading.Lock()
        self._returned = returned if returned is not None else []

    def take(self, size: int) -> Optional[Range]:
        with self._lock:
            if self._returned:
                return self._returned.pop()
            if self._next >= self._end:
                return None
            s = self._next
            e = min(s + max(int(size), 1), self._end)
            self._next = e
            return (s, e)

    def give_back(self, rng: Range) -> None:
        with self._lock:
            self._returned.append(rng)


@dataclass
class Worker:
    name: str
    device: str
    morsel_rows: int
    compute: Callable[[Any], Any]
    prepare: Optional[Callable[[Range], Any]] = None
    fallible: bool = False


@dataclass
class WorkerStats:
    device: str
    rows: int = 0
    morsels: int = 0
    busy_ms: float = 0.0
    failed: bool = False
    error: Optional[BaseException] = None


@dataclass
class RouterResult:
    value: Any
    stats: Dict[str, WorkerStats]
    wall_ms: float
    fallbacks: int = 0


def run_router(
    n_rows: int,
    workers: List[Worker],
    merge: Callable[[List[Any]], Any],
    static_fractions: Optional[Dict[str, float]] = None,
) -> RouterResult:
    """Run `workers` over rows [0, n_rows) and merge their partials in row order.

    static_fractions: {worker name: share of rows} gives those workers a fixed
    contiguous slice (no stealing); the rest share a queue over the remainder.
    """
    t0 = time.perf_counter()
    if not workers:
        raise ValueError("run_router needs at least one worker")
    lock = threading.Lock()
    returned: List[Range] = []
    queues: Dict[str, MorselQueue] = {}
    cursor = 0
    for w in workers:
        share = (static_fractions or {}).get(w.name)
        if share is not None:
            end = min(n_rows, cursor + int(round(n_rows * share)))
            queues[w.name] = MorselQueue(end, start=cursor, returned=returned, lock=lock)
            cursor = end
    shared = MorselQueue(n_rows, start=cursor, returned=returned, lock=lock)
    for w in workers:
        queues.setdefault(w.name, shared)

    stats = {w.name: WorkerStats(device=w.device) for w in workers}
    partials: List[Tuple[int, Any]] = []
    partial_lock = threading.Lock()
    fatal: List[BaseException] = []
    stop = threading.Event()

    def record(w: Worker, rng: Range, value: Any, ms: float) -> None:
        with partial_lock:
            partials.append((rng[0], value))
            st = stats[w.name]
            st.rows += rng[1] - rng[0]
            st.morsels += 1
            st.busy_ms += ms

    def fail(w: Worker, err: BaseException, pending: List[Range]) -> None:
        q = queues[w.name]
        for r in pending:
            q.give_back(r)
        stats[w.name].failed = True
        stats[w.name].error = err
        if not w.fallible:
            fatal.append(err)
            stop.set()

    def loop(w: Worker) -> None:
        q = queues[w.name]
        if w.prepare is None:
            while not stop.is_set() and (rng := q.take(w.morsel_rows)) is not None:
                t = time.perf_counter()
                try:
                    value = w.compute(rng)
                except BaseException as e:
                    fail(w, e, [rng])
                    return
                record(w, rng, value, (time.perf_counter() - t) * 1e3)
            return
        # Two-stage worker: prepare morsel k+1 on a helper thread while k computes.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mx-{w.name}-prep") as pool:
            rng = q.take(w.morsel_rows)
            fut = pool.submit(w.prepare, rng) if rng is not None else None
            while fut is not None and not stop.is_set():
                nxt = q.take(w.morsel_rows)
                try:
                    handle = fut.result()
                except BaseException as e:
                    fail(w, e, [rng] + ([nxt] if nxt else []))
                    return
                nfut = pool.submit(w.prepare, nxt) if nxt is not None else None
                t = time.perf_counter()
                try:
                    value = w.compute(handle)
                except BaseException as e:
                    if nfut is not None:
                        try:
                            nfut.result()
                        except BaseException:
                            pass
                    fail(w, e, [rng] + ([nxt] if nxt else []))
                    return
                record(w, rng, value, (time.perf_counter() - t) * 1e3)
                rng, fut = nxt, nfut

    threads = [threading.Thread(target=loop, args=(w,), name=f"mx-{w.name}") for w in workers]
    if len(threads) == 1:
        loop(workers[0])
    else:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if fatal:
        raise fatal[0]

    # Rows handed back after the other workers had already finished: redo them
    # on a healthy worker so nothing is lost.
    fallbacks = sum(1 for s in stats.values() if s.failed)
    if returned or any(queues[w.name] is not shared and stats[w.name].failed for w in workers):
        healthy = [w for w in workers if not stats[w.name].failed]
        if not healthy:
            raise next(s.error for s in stats.values() if s.error is not None)
        rescue = healthy[0]
        leftovers: List[Range] = list(returned)
        returned.clear()
        for w in workers:  # unfinished private slices of failed static workers
            q = queues[w.name]
            if q is not shared and stats[w.name].failed:
                while (r := q.take(1 << 62)) is not None:
                    leftovers.append(r)
        for rng in sorted(leftovers):
            for s in range(rng[0], rng[1], rescue.morsel_rows):
                sub = (s, min(s + rescue.morsel_rows, rng[1]))
                t = time.perf_counter()
                h = rescue.prepare(sub) if rescue.prepare else sub
                record(rescue, sub, rescue.compute(h), (time.perf_counter() - t) * 1e3)

    partials.sort(key=lambda p: p[0])
    value = merge([p[1] for p in partials])
    return RouterResult(value=value, stats=stats, wall_ms=(time.perf_counter() - t0) * 1e3,
                        fallbacks=fallbacks)
