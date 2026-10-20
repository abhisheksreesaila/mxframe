"""Charts for docs/optimizations.md and the README (F-039): plain SVG, light and dark, from the
engine-only benchmark CSVs in docs/bench/.

    pixi run python scripts/plot_progress.py docs/bench/m2s-engine.csv F-046

1. docs/img/progress.svg: the 22-query total per milestone, SF10 and SF1 side by side (two panels,
   one axis each), omxcore as a line, Polars and DuckDB (from the final run) as reference lines.
2. docs/img/per-query-sf10.svg / per-query-sf1.svg: how much faster or slower than Polars each
   query is (diverging bars on a log scale centred at 1x).

Colours: the dataviz reference palette (categorical slots 1-3 validated together, light and dark;
the diverging pair blue / red); text never wears a series colour; every line is labelled directly.
"""
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "docs" / "bench"
OUT = ROOT / "docs" / "img"

# (label, file, engine) per milestone; the first row is the engine before the Mojo executor
MILESTONES = [
    ("before", {"10": "engine-sf10", "1": "engine-sf1"}),
    ("M1", {"10": "m1-sf10-native1", "1": "m1-sf1-native1"}),
    ("M2a", {"10": "m2a-sf10", "1": "m2a-sf1"}),
    ("M2b", {"10": "m2b-sf10", "1": "m2b-sf1"}),
    ("M2d", {"10": "m2d-engine", "1": "m2d-engine"}),
    ("M2e", {"10": "m2e-engine", "1": "m2e-engine"}),
    ("M2g", {"10": "m2g-engine", "1": "m2g-engine"}),
    ("M2h", {"10": "m2h-engine", "1": "m2h-engine"}),
    ("M2l", {"10": "m2l-engine", "1": "m2l-engine"}),
    ("M2n", {"10": "m2n-engine", "1": "m2n-engine"}),
    ("F-043", {"10": "m2p-engine", "1": "m2p-engine"}),
]

STYLE = """
  .bg { fill: #fcfcfb } .t1 { fill: #0b0b0b } .t2 { fill: #52514e } .grid { stroke: #e4e3df }
  .axis { stroke: #b8b7b0 } .mx { stroke: #2a78d6; fill: #2a78d6 } .mxl { stroke: #2a78d6; fill: none } .pl { stroke: #eb6834 } .dk { stroke: #1baf7a }
  .fast { fill: #2a78d6 } .slow { fill: #e34948 } .mid { stroke: #8f8e88 }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19 } .t1 { fill: #f0efec } .t2 { fill: #c3c2b7 } .grid { stroke: #383835 }
    .axis { stroke: #5c5b57 } .mx { stroke: #3987e5; fill: #3987e5 } .mxl { stroke: #3987e5; fill: none } .pl { stroke: #d95926 } .dk { stroke: #199e70 }
    .fast { fill: #3987e5 } .slow { fill: #e66767 } .mid { stroke: #8f8e88 }
  }
  text { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 12px }
  .title { font-size: 14px; font-weight: 600 }
"""


def _rows(path):
    return list(csv.DictReader(open(path)))


def total(rows, engine, sf):
    rs = [r for r in rows if r["engine"] == engine and r["sf"] == sf and r.get("first_ms")]
    return sum(float(r["first_ms"]) for r in rs) / 1000 if len(rs) == 22 else None


def per_query(rows, engine, sf):
    return {int(r["query"]): float(r["first_ms"]) for r in rows
            if r["engine"] == engine and r["sf"] == sf and r.get("first_ms")}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def progress_svg(final_rows, final_label):
    series = {}
    for sf in ("10", "1"):
        pts = []
        for label, files in MILESTONES:
            p = BENCH / f"{files[sf]}.csv"
            if p.exists():
                v = total(_rows(p), "mx-cpu", sf)
                if v is not None:
                    pts.append((label, v))
        pts.append((final_label, total(final_rows, "mx-cpu", sf)))
        series[sf] = (pts, total(final_rows, "polars", sf), total(final_rows, "duckdb", sf))

    W, H = 1000, 360
    pw, ph, top, left_pad = 360, 230, 80, 56
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'role="img" aria-label="omxcore engine-only TPC-H total per milestone, against Polars and DuckDB">',
           f"<style>{STYLE}</style>", f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
           '<text class="t1 title" x="24" y="32">omxcore CPU engine: total of the 22 TPC-H queries, seconds (lower is faster)</text>',
           '<text class="t2" x="24" y="52">engine-only first runs: data in memory, fresh process, every result checked against DuckDB</text>']
    for k, sf in enumerate(("10", "1")):
        pts, polars, duck = series[sf]
        x0 = left_pad + k * (pw + 130)
        ymax = max([v for _, v in pts] + [polars or 0, duck or 0]) * 1.1
        step = _nice_step(ymax)
        y = lambda v: top + ph - v / ymax * ph
        x = lambda i: x0 + i * pw / (len(pts) - 1)
        out.append(f'<text class="t1" x="{x0}" y="{top - 8}" font-weight="600">SF{sf}</text>')
        v = 0.0
        while v <= ymax:
            out.append(f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
            out.append(f'<text class="t2" x="{x0 - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
            v += step
        out.append(f'<line class="axis" x1="{x0}" x2="{x0 + pw}" y1="{y(0)}" y2="{y(0)}"/>')
        for ref, cls, name in ((polars, "pl", "Polars"), (duck, "dk", "DuckDB")):
            if ref:
                out.append(f'<line class="{cls}" stroke-width="2" stroke-dasharray="6 4" x1="{x0}" x2="{x0 + pw}" '
                           f'y1="{y(ref):.1f}" y2="{y(ref):.1f}"/>')
                out.append(f'<text class="t2" x="{x0 + pw + 6}" y="{y(ref) + 4:.1f}">{name} {ref:.2f}</text>')
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(val):.1f}" for i, (_, val) in enumerate(pts))
        out.append(f'<path class="mxl" stroke-width="2" d="{path}"/>')
        for i, (label, val) in enumerate(pts):
            out.append(f'<circle class="mx" cx="{x(i):.1f}" cy="{y(val):.1f}" r="4" stroke-width="2" '
                       f'style="stroke: var(--bg, transparent)"><title>{esc(label)}: {val:.2f} s</title></circle>')
            out.append(f'<text class="t2" x="{x(i):.1f}" y="{top + ph + 18}" text-anchor="middle">{esc(label)}</text>')
        first, last = pts[0][1], pts[-1][1]
        out.append(f'<text class="t1" x="{x(0) + 8:.1f}" y="{y(first) + 18:.1f}">omxcore {first:.2f}</text>')
        out.append(f'<text class="t1" x="{x(len(pts) - 1):.1f}" y="{y(last) - 10:.1f}" text-anchor="end">{last:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _nice_step(ymax):
    raw = ymax / 5
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def per_query_svg(final_rows, sf, titles):
    mx, pl = per_query(final_rows, "mx-cpu", sf), per_query(final_rows, "polars", sf)
    qs = sorted(q for q in mx if q in pl)
    ratio = {q: pl[q] / mx[q] for q in qs}             # >1: omxcore faster
    bad = {(r["engine"], int(r["query"])) for r in final_rows if r["sf"] == sf
           and r["engine"] in ("mx-cpu", "polars") and r.get("verdict") not in ("ok", "drift")}
    wrong = {q for _, q in bad}
    verified = [q for q in qs if q not in wrong]       # a wrong result is never counted as a win
    wins = sum(ratio[q] > 1 for q in verified)
    CAP = 16.0                                         # longer bars are cut here, their label says so
    W, row, top = 780, 22, 96
    right = math.log2(min(CAP, max(2.0, max(ratio.values()))))
    left = math.log2(max(2.0, max(1 / r for r in ratio.values())))
    scale = 270 / (left + right)
    mid = 330 + left * scale
    H = top + row * len(qs) + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Per-query speed against Polars at SF{sf}">', f"<style>{STYLE}</style>",
           f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
           f'<text class="t1 title" x="24" y="32">SF{sf}: omxcore against Polars, per TPC-H query ({wins} of {len(verified)} verified faster)</text>',
           '<text class="t2" x="24" y="52">times faster (right) or slower (left) than Polars, log scale; engine-only first runs</text>',
           f'<text class="t2" x="{mid + 8:.1f}" y="{top - 14}">faster \u2192</text>',
           f'<text class="t2" x="{mid - 8:.1f}" y="{top - 14}" text-anchor="end">\u2190 slower</text>']
    for k in range(-int(left), int(right) + 1):
        xx = mid + k * scale
        out.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top - 6}" y2="{top + row * len(qs)}"/>')
        out.append(f'<text class="t2" x="{xx:.1f}" y="{top + row * len(qs) + 16}" text-anchor="middle">{2 ** abs(k):g}x</text>')
    for i, q in enumerate(qs):
        yy = top + i * row
        r = ratio[q]
        w = max(-left, min(right, math.log2(r))) * scale
        x1, ww = (mid, w) if w >= 0 else (mid + w, -w)
        cls = "fast" if r >= 1 else "slow"
        who = [n for e, n in (("mx-cpu", "omxcore"), ("polars", "Polars")) if (e, q) in bad]
        note = f" ({' and '.join(who)} result wrong)" if who else ""
        out.append(f'<text class="t2" x="{24}" y="{yy + 14}">Q{q} {esc(titles.get(q, ""))}{note}</text>')
        out.append(f'<rect class="{cls}" x="{x1:.1f}" y="{yy + 3}" width="{max(ww, 2):.1f}" height="{row - 6}" rx="3">'
                   f'<title>Q{q}: omxcore {mx[q]:.0f} ms, Polars {pl[q]:.0f} ms</title></rect>')
        label = (f"{r:.1f}x" + (" \u2192" if r > CAP else "")) if r >= 1 else f"{1 / r:.1f}x slower"
        lx = (x1 + ww + 6) if w >= 0 else (x1 - 6)
        out.append(f'<text class="t1" x="{lx:.1f}" y="{yy + 14}" text-anchor="{"start" if w >= 0 else "end"}">'
                   f'{label} <tspan class="t2">({mx[q]:.0f} vs {pl[q]:.0f} ms)</tspan></text>')
    out.append(f'<line class="mid" stroke-width="1.5" x1="{mid}" x2="{mid}" y1="{top - 6}" y2="{top + row * len(qs)}"/>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    final = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH / "m2s-engine.csv"
    label = sys.argv[2] if len(sys.argv) > 2 else "F-046"
    rows = _rows(final)
    sys.path.insert(0, str(ROOT / "scripts"))
    from tpch_user.spec import TITLES
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "progress.svg").write_text(progress_svg(rows, label))
    for sf in ("10", "1"):
        (OUT / f"per-query-sf{sf}.svg").write_text(per_query_svg(rows, sf, TITLES))
    print("wrote", ", ".join(str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.svg"))))


if __name__ == "__main__":
    main()
