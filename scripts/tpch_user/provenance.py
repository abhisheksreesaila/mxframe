"""Which path every `compute()` took (F-055): `path@device` of the plan, the GPU kernels it called, and
`gpu_sort_fallback` when the GPU sort's result was wrong and redone on the CPU. A "correct" GPU result
that ran on the CPU proves nothing about the GPU, so the sweeps record this for every run."""
from collections import Counter


def install() -> list:
    """Wrap LazyFrame.compute; returns the list the entries are appended to (clear it between runs)."""
    import omxcore.aot_kernels as ak
    import omxcore.sort_check as sc
    from omxcore.lazy_frame import LazyFrame
    entries: list = []
    ak.gpu_call_log = []
    orig = LazyFrame.compute

    def recording(self, *args, **kw):
        if kw.get("validate") is False and kw.get("optimize") is False:
            return orig(self, *args, **kw)       # `LazyFrame.schema`'s zero-row dry run (SQL derived tables): no query ran
        ak.gpu_call_log.clear()
        fallbacks = sc.gpu_sort_fallbacks
        result = orig(self, *args, **kw)
        p = self.last_compile_provenance
        entries.append(f"{p.get('path')}@{p.get('device')}" + ("(gpu_declined)" if p.get("gpu_declined") else ""))
        entries.extend(f"gpu:{n}" for n in sorted(set(ak.gpu_call_log)))
        if sc.gpu_sort_fallbacks > fallbacks:
            entries.append("gpu_sort_fallback")
        return result

    LazyFrame.compute = recording
    return entries


def summary(entries: list) -> str:
    """`native@cpu x1 gpu:sort_topk_i32_gpu x2`: each distinct entry with how many computes had it."""
    return " ".join(f"{k} x{entries.count(k)}" for k in dict.fromkeys(entries))
