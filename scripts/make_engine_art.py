"""Draw the Mojo engine's README art from docs/bench/engine-informal.json (the one copy of the numbers).

    pixi run python scripts/make_engine_art.py          # writes docs/img/engine-*.svg
    pixi run python scripts/make_engine_art.py --check  # exit 1 if a committed SVG is stale

The look is editorial paper (picked 2026-10-19 over a dark terminal glow): light ruled paper, one huge serif
number, the docs site's paper tokens -> engine-hero.svg, engine-<chart>.svg

GitHub renders SVG through <img>: no scripts, no web fonts, so every font is a system stack and every
size is static. Text widths are estimated (no font metrics), with slack left for the widest system font.
"""
import json
import math
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "bench" / "engine-informal.json"
OUT = ROOT / "docs" / "img"

MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', Times, serif"

# docs/site/site.css tokens; series colours validated with the dataviz validator (docs/design/docs-site.md)
THEMES = {
    "paper": dict(bg="#F2EFE7", surface="#FFFDF8", surface2="#F8F4EB", border="#D8D0C2", border_hi="#BEB4A4",
                  text="#20221F", mid="#4C5049", dim="#6E716A", accent="#6C3E78", accent_hi="#512B5C",
                  mx="#2a78d6", dk="#1baf7a", pa="#eda100", slow="#e34948", grid="#E4DDD0", green="#18704F"),
}
SERIES = {"mx": "omxcore", "dk": "DuckDB", "pa": "pyarrow"}


def load():
    return json.loads(DATA.read_text())


def lookup(data, ref):
    """A headline or gap points at a row: [section] or [section, row label]."""
    sec = data[ref[0]]
    if len(ref) == 1:
        return sec
    rows = sec.get("rows", []) + sec.get("steps", []) + ([sec["few_groups"]] if "few_groups" in sec else [])
    for r in rows:
        if r["label"] == ref[1]:
            return r
    raise KeyError(ref)


def lo(v):
    return min(v) if isinstance(v, list) else v


def hi(v):
    return max(v) if isinstance(v, list) else v


def fmt(v, unit):
    """0.15 -> '0.15 s', 2208 -> '2,208 ms', [0.5, 0.6] -> '0.5-0.6 s'."""
    if isinstance(v, list):
        d = max(len(f"{x:g}".partition(".")[2]) for x in v)
        return f"{v[0]:.{d}f}–{v[1]:.{d}f} {unit}".strip()
    if unit == "ms":
        s = f"{v:,}" if isinstance(v, int) else f"{v:.1f}" if v.is_integer() else f"{v:g}"  # as written: 2,208 / 4.0 / 20.6
    else:  # seconds under 1 keep two decimals, as docs/mojo-engine.md writes them (0.10 s)
        s = f"{v:.2f}" if v < 1 and len(f"{v:g}".partition(".")[2]) < 2 else f"{v:g}" if v < 10 else f"{v:,.0f}"
    return f"{s} {unit}".strip()


def ratio_text(a, b):
    """How many times b is a (a, b may be ranges)."""
    r1, r2 = lo(a) / hi(b), hi(a) / lo(b)
    if abs(r2 - r1) < 0.05:
        return f"{(r1 + r2) / 2:.1f}x"
    return f"{r1:.1f}–{r2:.1f}x"


def text(x, y, s, size=12, fill="#000", family=SANS, weight=400, anchor="start", ls=0, extra=""):
    l = f' letter-spacing="{ls}"' if ls else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{l}{extra}>{escape(str(s))}</text>')


def wrap(s, chars):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines


def svg_doc(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
            + body + "\n</svg>\n")


# ---------------------------------------------------------------- heroes

def headline_values(data, h):
    row = lookup(data, h["ref"])
    if h["ref"] == ["memory"]:
        return row["peak_mb"], row["file_gb"] * 1000, "file"
    return lo(row["mx"]), hi(row["duckdb"]), "DuckDB"


def split_big(big):
    num, _, unit = big.partition(" ")
    return num, unit


def hero_paper(data):
    t = THEMES["paper"]
    W, H = 1200, 360
    focus = next(h for h in data["headline"] if h.get("focus"))
    rest = [h for h in data["headline"] if h is not focus]
    b = [f'''<defs>
  <pattern id="rule" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M0 23.5H24" stroke="{t['border']}" stroke-opacity=".55"/></pattern>
  <clipPath id="clip"><rect width="{W}" height="{H}" rx="18"/></clipPath>
</defs>
<g clip-path="url(#clip)">
<rect width="{W}" height="{H}" fill="{t['bg']}"/>
<rect y="72" width="{W}" height="{H - 72}" fill="url(#rule)"/>''']
    x = 56
    b.append(f'<text x="{x}" y="46" font-family="{MONO}" font-size="16" font-weight="800" letter-spacing="5" fill="{t["text"]}">'
             f'OM<tspan fill="{t["accent"]}">X</tspan>CORE</text>')
    b.append(text(x + 150, 46, "THE MOJO ENGINE", 12, t["dim"], MONO, 700, ls=3))
    b.append(text(W - 56, 46, "INFORMAL NUMBERS · SHARED MACHINE", 12, t["dim"], MONO, 700, "end", ls=2))
    b.append(f'<path d="M{x} 64H{W - 56}" stroke="{t["text"]}" stroke-width="1.5"/>')
    # the big serif number
    num, unit = split_big(focus["big"])
    b.append(f'<text x="{x - 6}" y="226" font-family="{SERIF}" fill="{t["text"]}"><tspan font-size="168" letter-spacing="-6">{escape(num)}</tspan>'
             f'<tspan font-size="56" font-style="italic" fill="{t["accent"]}" dx="12">{escape(unit)}</tspan></text>')
    lx = x + len(num) * 92 + len(unit) * 30 + 70
    b.append(text(lx, 140, focus["label"], 24, t["text"], SERIF, 400, extra=' font-style="italic"'))
    b.append(text(lx, 176, focus["vs"] + ".", 24, t["mid"], SERIF))
    mx_v, other, _ = headline_values(data, focus)
    b.append(text(lx, 212, f"{ratio_text(other, mx_v)} faster", 15, t["accent"], MONO, 700, ls=1))
    for i, line in enumerate(wrap(data["vision_short"], 64)):
        b.append(text(x, 280 + i * 28, line, 21, t["mid"], SERIF))
    b.append(text(x, 336, data["caveat_short"], 11.5, t["dim"], MONO))
    # ledger column
    cx = 790
    b.append(f'<path d="M{cx - 30} 92V318" stroke="{t["border_hi"]}"/>')
    for i, h in enumerate(rest):
        y = 92 + i * 76
        n, u = split_big(h["big"])
        b.append(f'<text x="{cx}" y="{y + 46}" font-family="{SERIF}" fill="{t["text"]}"><tspan font-size="44">{escape(n)}</tspan>'
                 f'<tspan font-size="20" font-style="italic" fill="{t["accent"]}" dx="6">{escape(u)}</tspan></text>')
        b.append(text(W - 56, y + 26, h["label"], 13, t["text"], SANS, 600, "end"))
        b.append(text(W - 56, y + 46, h["vs"], 13, t["dim"], SANS, 400, "end"))
        if i < len(rest) - 1:
            b.append(f'<path d="M{cx} {y + 70}H{W - 56}" stroke="{t["border"]}"/>')
    b.append("</g>")
    b.append(f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{t["border_hi"]}"/>')
    return svg_doc(W, H, "\n".join(b), "omxcore: the Mojo engine (editorial)",
                   data["vision"] + " " + "; ".join(f'{h["label"]}: {h["big"]} ({h["vs"]})' for h in data["headline"]) + ". " + data["caveat"])


# ---------------------------------------------------------------- charts

def chart(theme, title, subtitle, legend, rows, scale, ticks, footer, value_col=None):
    """Grouped horizontal bars. rows: {label, sub, bars: [(series, value, text)], right}. scale: value -> 0..1."""
    t = THEMES[theme]
    W, LX, PX0, PX1 = 900, 32, 290, 760 if value_col else 840
    head, step = 134, 17
    heights = [max(len(r["bars"]) * step + 16, 52 if r.get("sub") else 0) for r in rows]
    H = head + sum(heights) + 58
    b = [f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="14" fill="{t["surface"]}" stroke="{t["border_hi"]}"/>',
         text(LX, 36, "// OMXCORE · THE MOJO ENGINE", 11, t["accent"], MONO, 700, ls=2),
         text(LX, 64, title, 21, t["text"], SANS, 700),
         text(LX, 86, subtitle, 12, t["dim"], MONO)]
    lx = LX
    for s, name in legend:
        b.append(f'<rect x="{lx}" y="{100}" width="10" height="10" rx="2" fill="{t[s]}"/>')
        b.append(text(lx + 16, 109, name, 12, t["mid"], MONO))
        lx += 16 + len(name) * 7.4 + 22
    # gridlines and tick labels
    for v, lab in ticks:
        gx = PX0 + (PX1 - PX0) * scale(v)
        b.append(f'<path d="M{gx:.1f} {head - 4}V{H - 52}" stroke="{t["grid"]}" stroke-width="{1.5 if v == ticks[0][0] else 1}"/>')
        b.append(text(gx, head - 10, lab, 10.5, t["dim"], MONO, anchor="middle"))
    if value_col:
        b.append(text(W - LX, head - 10, value_col, 10.5, t["dim"], MONO, 700, "end", ls=1))
    y = head + 6
    for r, rh in zip(rows, heights):
        b.append(text(LX, y + 14, r["label"], 13.5, t["text"], SANS, 600))
        if r.get("sub"):
            b.append(text(LX, y + 30, r["sub"], 11, t["dim"], MONO))
        by = y + 2
        for s, v, label in r["bars"]:
            x1 = PX0 + (PX1 - PX0) * scale(lo(v))
            fill = t[s]
            b.append(f'<rect x="{PX0}" y="{by}" width="{max(2, x1 - PX0):.1f}" height="12" rx="3" fill="{fill}"/>')
            end = x1
            if isinstance(v, list):  # a range: solid to the low end, a lighter run to the high end
                x2 = PX0 + (PX1 - PX0) * scale(hi(v))
                b.append(f'<rect x="{x1:.1f}" y="{by}" width="{max(2, x2 - x1):.1f}" height="12" rx="3" fill="{fill}" fill-opacity=".4"/>')
                end = x2
            if isinstance(label, tuple):  # (bold figure, quieter detail)
                b.append(f'<text x="{end + 7:.1f}" y="{by + 10.5:.1f}" font-family="{MONO}" font-size="11.5" font-weight="700" fill="{t["text"]}">'
                         f'{escape(label[0])}<tspan font-weight="400" fill="{t["mid"]}" dx="10">{escape(label[1])}</tspan></text>')
            else:
                b.append(text(end + 7, by + 10.5, label, 11.5, t["mid"], MONO, 600))
            by += step
        if r.get("right"):
            b.append(text(W - LX, y + 17, r["right"], 15, t["text"], MONO, 700, "end"))
        y += rh
        b.append(f'<path d="M{LX} {y - 6}H{W - LX}" stroke="{t["grid"]}" stroke-opacity=".7"/>')
    b.append(text(LX, H - 22, footer, 11, t["dim"], MONO))
    return svg_doc(W, H, "\n".join(b), title, subtitle + ". " + footer), H


def log_scale(a, z):
    return lambda v: (math.log10(v) - math.log10(a)) / (math.log10(z) - math.log10(a))


AREA = {"joins": "join", "sql": "SQL", "streaming": "streamed", "aggregates": "1M groups", "multiway": "join", "exprs": "expr"}
FOOT = "informal: shared 12-core Ryzen, other work running · source: docs/mojo-engine.md"


def named(g, r):
    """A row's chart label: the list's own label, else the row's, prefixed with its area when that helps."""
    if g.get("label"):
        return g["label"]
    prefix = AREA[g["from"][0]]
    return r["label"] if prefix.lower() in r["label"].lower() else f"{prefix} · {r['label']}"


def chart_wins(data, theme):
    """The strongest wins, as listed in the data file (every one is a row docs/mojo-engine.md measured)."""
    rows = []
    for w in data["wins"]:
        r = lookup(data, w["from"])
        rows.append(dict(label=named(w, r), ratio=r["duckdb"] / r["mx"],
                         bars=[("mx", r["mx"], fmt(r["mx"], "ms")), ("dk", r["duckdb"], fmt(r["duckdb"], "ms"))],
                         right=ratio_text(r["duckdb"], r["mx"])))
    rows.sort(key=lambda r: -r["ratio"])
    ticks = [(1, "1 ms"), (10, "10 ms"), (100, "100 ms"), (1000, "1 s")]
    return chart(theme, "Where the Mojo engine leads DuckDB", "SQL, joins, expressions and aggregates · time, log scale · lower is faster",
                 [("mx", "omxcore"), ("dk", "DuckDB 1.5.1")], rows, log_scale(1, 4000), ticks, FOOT, "FASTER")


def chart_read(data, theme):
    p, tx, st = data["parquet"]["rows"], data["text"]["rows"], data["streaming"]
    few, many = st["few_groups"], st["steps"][-1]
    rows = [
        dict(label="Parquet read, numeric", sub=f"20M rows · was {fmt(p[0]['mx'], 's')}",
             bars=[("mx", p[1]["mx"], fmt(p[1]["mx"], "s")), ("pa", p[1]["pyarrow"], fmt(p[1]["pyarrow"], "s"))]),
        dict(label="Text read", sub=f"5M rows · was {fmt(tx[0]['mx'], 's')}",
             bars=[("mx", tx[1]["mx"], fmt(tx[1]["mx"], "s")), ("dk", tx[1]["duckdb"], fmt(tx[1]["duckdb"], "s") + " whole query"),
                   ("pa", tx[1]["pyarrow"], fmt(tx[1]["pyarrow"], "s"))]),
        dict(label="Streamed query, 4 row groups", sub=f"20M rows · {few['memory']}",
             bars=[("mx", few["mx"], fmt(few["mx"], "s")), ("dk", few["duckdb"], fmt(few["duckdb"], "s"))]),
        dict(label="Streamed query, 20 row groups", sub=f"20M rows · {many['memory']}",
             bars=[("mx", many["mx"], fmt(many["mx"], "s")), ("dk", many["duckdb"], fmt(many["duckdb"], "s"))]),
    ]
    ticks = [(0, "0"), (0.2, "0.2 s"), (0.4, "0.4 s"), (0.6, "0.6 s")]
    return chart(theme, "Reading files at memory speed", "Parquet, text and bigger-than-RAM streams · time, linear · lower is faster",
                 [("mx", "omxcore"), ("dk", "DuckDB 1.5.1"), ("pa", "pyarrow")], rows, lambda v: v / 0.75, ticks, FOOT)


def gaps(data):
    out = []
    for g in data["behind"]:
        r = lookup(data, g["from"])
        unit = data[g["from"][0]]["unit"]
        out.append(dict(label=named(g, r), mx=r["mx"], dk=r["duckdb"], unit=unit,
                        lo=lo(r["mx"]) / hi(r["duckdb"]), hi=hi(r["mx"]) / lo(r["duckdb"]), next=g["next"]))
    return sorted(out, key=lambda g: -g["hi"])


def chart_behind(data, theme):
    rows = []
    for g in gaps(data):
        v = [g["lo"], g["hi"]] if g["hi"] - g["lo"] >= 0.05 else g["lo"]
        rows.append(dict(label=g["label"], sub=(f"next: {g['next']}" if g["next"] else ""),
                         bars=[("slow", v, (ratio_text(g['mx'], g['dk']), f"{fmt(g['mx'], g['unit'])} vs {fmt(g['dk'], g['unit'])}"))]))
    top = max(g["hi"] for g in gaps(data))
    ticks = [(v, f"{v}x") for v in range(1, int(top) + 2)]
    # leave room right of the longest bar for its "5.8x  336 ms vs 58 ms" label
    return chart(theme, "Where we're still behind DuckDB", "how many times slower than DuckDB · 1x is a tie · shorter is better",
                 [("slow", "omxcore time / DuckDB time")], rows, lambda v: (v - 1) / ((top - 1) / 0.66), ticks, FOOT)


def paper_frame(t, W, H, kicker, right):
    """The editorial paper sheet every roadmap band shares: ruled paper, wordmark, kicker, a rule."""
    return [f'''<defs>
  <pattern id="rule" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M0 23.5H24" stroke="{t['border']}" stroke-opacity=".45"/></pattern>
  <clipPath id="clip"><rect width="{W}" height="{H}" rx="18"/></clipPath>
</defs>
<g clip-path="url(#clip)">
<rect width="{W}" height="{H}" fill="{t['bg']}"/>
<rect y="72" width="{W}" height="{H}" fill="url(#rule)"/>''',
            f'<text x="56" y="46" font-family="{MONO}" font-size="16" font-weight="800" letter-spacing="5" fill="{t["text"]}">'
            f'OM<tspan fill="{t["accent"]}">X</tspan>CORE</text>',
            text(206, 46, kicker, 12, t["dim"], MONO, 700, ls=3),
            text(W - 56, 46, right, 12, t["dim"], MONO, 700, "end", ls=1),
            f'<path d="M56 64H{W - 56}" stroke="{t["text"]}" stroke-width="1.5"/>']


def paper_close(t, W, H, b):
    b.append("</g>")
    b.append(f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{t["border_hi"]}"/>')
    return "\n".join(b)


def lines_at(b, x, y, s, chars, size, fill, family=SANS, weight=400, lh=None, extra=""):
    """Wrapped text; returns the baseline after the last line."""
    lh = lh or round(size * 1.35)
    for line in wrap(s, chars):
        b.append(text(x, y, line, size, fill, family, weight, extra=extra))
        y += lh
    return y


STATUS = {"today": ("TODAY", "green"), "with 0.7.0": ("WITH 0.7.0", "accent"), "planned": ("PLANNED · M10", "dim")}


def roadmap(data):
    """The vision, what omxcore can be, the path with a marker at where we are, and the agent-native band
    (docs/briefs/positioning.md). Nothing in the band is drawn as done unless its status says today."""
    t, rm = THEMES["paper"], data["roadmap"]
    W, X0, X1 = 1200, 56, 1144
    b = []
    y = 122
    for line in wrap(rm["vision"] + ".", 56):
        b.append(text(X0, y, line, 38, t["text"], SERIF, 400, extra=' font-style="italic"'))
        y += 46
    b.append(text(X0, y + 2, rm["vision_2"] + ".", 24, t["accent"], SERIF, 400))
    # what it can be: numbered, four across
    y += 52
    b.append(text(X0, y, f"WHAT {rm['name'].upper()} CAN BE", 12, t["accent"], MONO, 700, ls=3))
    cw, y0 = (X1 - X0) / 4, y + 18
    for i, item in enumerate(rm["can_be"]):
        cx, cy = X0 + (i % 4) * cw, y0 + (i // 4) * 58
        b.append(text(cx, cy + 30, str(i + 1), 34, t["accent"], SERIF, 400, extra=' font-style="italic"'))
        lines_at(b, cx + 34, cy + 16, item, 24, 15.5, t["text"], SANS, 500, lh=19)
    y = y0 + 2 * 58 + 30
    # the path: solid up to where we are, dashed after
    b.append(text(X0, y, "THE PATH", 12, t["accent"], MONO, 700, ls=3))
    steps, now = rm["path"], next(i for i, s in enumerate(rm["path"]) if s["id"] == rm["now"])
    sw, ly = (X1 - X0) / len(steps), y + 62
    sx = [X0 + 10 + i * sw for i in range(len(steps))]
    b.append(f'<path d="M{sx[0]} {ly}H{X1 - 8}" stroke="{t["border_hi"]}" stroke-width="2" stroke-dasharray="6 6"/>')
    b.append(f'<path d="M{sx[0]} {ly}H{sx[now] + sw * 0.45}" stroke="{t["accent"]}" stroke-width="3"/>')
    b.append(f'<path d="M{X1 - 12} {ly - 6}l10 6l-10 6" fill="none" stroke="{t["border_hi"]}" stroke-width="2"/>')
    bottom = 0
    for i, s in enumerate(steps):
        x = sx[i]
        if i == now:
            b.append(f'<circle cx="{x}" cy="{ly}" r="16" fill="{t["accent"]}" fill-opacity=".15"/>')
            b.append(f'<circle cx="{x}" cy="{ly}" r="8" fill="{t["accent"]}"/>')
            b.append(f'<rect x="{x - 10}" y="{ly - 50}" width="112" height="24" rx="12" fill="{t["accent"]}"/>')
            b.append(text(x + 46, ly - 33.5, "WE ARE HERE", 11, t["surface"], MONO, 700, "middle", ls=1.2))
        else:
            b.append(f'<circle cx="{x}" cy="{ly}" r="7" fill="{t["surface"]}" stroke="{t["border_hi"]}" stroke-width="2"/>')
        tx = x - 10
        b.append(text(tx, ly + 34, f"{i + 1} · {s['tag']}", 10.5, t["accent"], MONO, 700))
        ty = lines_at(b, tx, ly + 56, s["title"], 18, 15, t["text"], SANS, 700, lh=18) + 3
        if "done" in s:
            ty = lines_at(b, tx, ty, f"done: {s['done']}", 26, 12.5, t["mid"], lh=17)
            ty = lines_at(b, tx, ty, f"building: {s['building']}", 26, 12.5, t["dim"], lh=17)
        else:
            ty = lines_at(b, tx, ty, s["detail"], 26, 12.5, t["mid"], lh=17)
        bottom = max(bottom, ty)
    # the agent-native band: a highlight, honest about what is planned
    ag = rm["agents"]
    by = bottom + 20
    band = []
    lx, ly2 = X0, by + 52
    band.append(text(lx, ly2, ag["title"], 34, t["text"], SERIF, 400, extra=' font-style="italic"'))
    band.append(text(lx, ly2 + 26, ag["when"], 10.5, t["accent"], MONO, 700, ls=1))
    yy = lines_at(band, lx, ly2 + 62, ag["lead"], 32, 18, t["text"], SANS, 700, lh=23)
    yy = lines_at(band, lx, yy + 4, ag["lead_line"], 50, 13, t["mid"], lh=18)
    yy = lines_at(band, lx, yy + 8, ag["note"], 54, 11.5, t["dim"], lh=16)
    gx0, gcw = 470, (X1 - 470) / 3
    gy = by + 36
    for i, it in enumerate(ag["items"]):
        cx, cy = gx0 + (i % 3) * gcw, gy + (i // 3) * 112
        label, tone = STATUS[it["status"]]
        col = t[tone] if tone != "accent" else t["accent"]
        w = len(label) * 6.6 + 18
        if it["status"] == "today":
            band.append(f'<rect x="{cx}" y="{cy}" width="{w:.0f}" height="20" rx="10" fill="{col}"/>')
            band.append(text(cx + w / 2, cy + 14, label, 10, t["surface"], MONO, 700, "middle", ls=.6))
        else:
            band.append(f'<rect x="{cx + .5}" y="{cy + .5}" width="{w:.0f}" height="19" rx="9.5" fill="none" stroke="{col}" stroke-dasharray="{"0" if tone == "accent" else "3 2"}"/>')
            band.append(text(cx + w / 2, cy + 14, label, 10, col, MONO, 700, "middle", ls=.6))
        ny = lines_at(band, cx, cy + 42, it["name"], 26, 15, t["text"], SANS, 700, lh=18)
        lines_at(band, cx, ny, it["line"], 34, 12.5, t["mid"], lh=17)
    bh = max(yy, gy + 2 * 112) - by + 8
    b.append(f'<rect x="32" y="{by}" width="{W - 64}" height="{bh}" rx="16" fill="{t["accent"]}" fill-opacity=".07" stroke="{t["accent"]}" stroke-opacity=".35"/>')
    b.append(f'<path d="M{gx0 - 30} {by + 24}V{by + bh - 24}" stroke="{t["accent"]}" stroke-opacity=".3"/>')
    b += band
    H = int(by + bh + 50)
    b.append(text(X0, H - 20, "informal timings: shared machine, see docs/mojo-engine.md · plugins are planned, not built", 11, t["dim"], MONO))
    head = paper_frame(t, W, H, f"THE ROADMAP · {rm['name'].upper()}", rm["source"])
    body = paper_close(t, W, H, head + b)
    desc = (f"Vision: {rm['vision']}. {rm['vision_2']}. What {rm['name']} can be: " + "; ".join(rm["can_be"]) + ". The path: " +
            "; ".join(f"{i + 1}. {s['title']}" + (" (we are here)" if i == now else "") for i, s in enumerate(steps)) +
            f". {ag['title']} ({ag['when'].lower()}): {ag['lead']}. " +
            "; ".join(f"{it['name']} ({it['status']}): {it['line']}" for it in ag["items"]) + ".")
    return svg_doc(W, H, body, f"omxcore roadmap: the path to {rm['name']}", desc)


def hood(data):
    """Modern algorithms (built and in use today) x a modern compiled language."""
    t, h = THEMES["paper"], data["hood"]
    W, X0, X1 = 1200, 56, 1144
    b = []
    left, right = h["title"].split(": ", 1)
    b.append(text(X0, 116, right[0].upper() + right[1:] + ".", 32, t["text"], SERIF, 400, extra=' font-style="italic"'))
    b.append(text(X0, 156, f"ALGORITHMS · {h['algorithms_when']}", 11.5, t["accent"], MONO, 700, ls=2))
    cw = (790 - X0) / 3
    bottom = 0
    for i, a in enumerate(h["algorithms"]):
        cx, cy = X0 + (i % 3) * cw, 186 + (i // 3) * 64
        b.append(f'<rect x="{cx}" y="{cy - 12}" width="3" height="34" rx="1.5" fill="{t["accent"]}" fill-opacity=".55"/>')
        yy = lines_at(b, cx + 12, cy, a["name"], 30, 14, t["text"], SANS, 700, lh=17)
        bottom = max(bottom, lines_at(b, cx + 12, yy, a["line"], 34, 12, t["dim"], lh=16))
    m = h["mojo"]
    px, py, pw = 826, 96, X1 - 826
    ph = bottom - py + 4
    b.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="{t["accent"]}" fill-opacity=".07" stroke="{t["accent"]}" stroke-opacity=".35"/>')
    b.append(text(px + 24, py + 50, m["name"], 40, t["text"], SERIF, 400))
    yy = py + 84
    for p in m["points"]:
        b.append(f'<rect x="{px + 24}" y="{yy - 5}" width="10" height="2" fill="{t["accent"]}"/>')
        yy = lines_at(b, px + 42, yy, p, 36, 13.5, t["text"], SANS, 500, lh=18) + 6
    lines_at(b, px + 24, yy + 12, m["tagline"], 34, 17, t["accent"], SERIF, 400, lh=22, extra=' font-style="italic"')
    H = int(bottom + 22)
    head = paper_frame(t, W, H, left.upper(), h["source"])
    desc = f"{h['title']}. Algorithms, {h['algorithms_when'].lower()}: " + "; ".join(f"{a['name']} ({a['line']})" for a in h["algorithms"]) + \
        f". {m['name']}: " + "; ".join(m["points"]) + f". {m['tagline']}"
    return svg_doc(W, H, paper_close(t, W, H, head + b), h["title"], desc)


def family(data):
    """omxapp over omxcore over your files, with omxplorer as the example app: one native binary."""
    t, f = THEMES["paper"], data["family"]
    W, X0 = 1200, 56
    b = []
    names = [l["name"] for l in f["layers"][:2]]
    b.append(text(X0, 128, f"{names[0]} + {names[1]}", 44, t["text"], SERIF, 400, extra=' font-style="italic"'))
    yy = lines_at(b, X0, 172, f["line"], 52, 17, t["mid"], SANS, 400, lh=24)
    ex = f["example"]
    b.append(text(X0, yy + 14, f"{ex['name']} is {ex['line']}.", 15, t["accent"], SANS, 700))
    sx, sw, y = 600, 470, 92
    # the example app, dashed: it sits on the stack
    b.append(f'<rect x="{sx + .5}" y="{y + .5}" width="{sw}" height="40" rx="10" fill="none" stroke="{t["border_hi"]}" stroke-dasharray="5 4"/>')
    b.append(text(sx + 18, y + 26, ex["name"], 18, t["text"], SERIF, 400, extra=' font-style="italic"'))
    b.append(text(sx + sw - 16, y + 25, "EXAMPLE APP", 10.5, t["dim"], MONO, 700, "end", ls=1.5))
    y += 50
    top = y
    for l in f["layers"]:
        core, files = l.get("core"), l["name"] == "your files"
        hgt = 60
        fill, ink, sub = (t["accent"], t["surface"], t["surface"]) if core else (t["surface"], t["text"], t["mid"])
        b.append(f'<rect x="{sx}" y="{y}" width="{sw}" height="{hgt}" rx="10" fill="{fill}" stroke="{t["accent"] if core else t["border_hi"]}"'
                 f'{" stroke-dasharray=\"2 3\"" if files else ""}/>')
        b.append(text(sx + 18, y + 27, l["name"], 22, ink, SERIF, 400))
        b.append(text(sx + sw - 16, y + 26, l["role"].upper(), 10.5, sub, MONO, 700, "end", ls=1.5))
        b.append(text(sx + 18, y + 47, l["line"], 12.5, sub, SANS, 400, extra=' fill-opacity=".9"' if core else ""))
        if not files:
            mid = y + hgt
        y += hgt + 10
    # one native binary: a bracket over the app and the engine
    bx = sx + sw + 18
    b.append(f'<path d="M{bx} {top - 50}h8V{mid}h-8" fill="none" stroke="{t["accent"]}" stroke-width="1.5"/>')
    b.append(f'<text x="{bx + 26}" y="{(top - 50 + mid) / 2}" font-family="{MONO}" font-size="10.5" font-weight="700" fill="{t["accent"]}" '
             f'letter-spacing="1.5" text-anchor="middle" transform="rotate(90 {bx + 26} {(top - 50 + mid) / 2})">ONE NATIVE BINARY</text>')
    H = int(y + 16)
    head = paper_frame(t, W, H, f["title"].upper(), f["layers"][0]["repo"])
    desc = f"{f['title']}: " + " over ".join(f"{l['name']} ({l['role']}: {l['line']})" for l in f["layers"]) + \
        f". {f['line']} {ex['name']} is {ex['line']}."
    return svg_doc(W, H, paper_close(t, W, H, head + b), f"{f['title']}: {names[0]} + {names[1]}", desc)


def build():
    data = load()
    files = {"engine-hero.svg": hero_paper(data), "engine-roadmap.svg": roadmap(data),
             "engine-hood.svg": hood(data), "engine-family.svg": family(data)}
    for name, fn in (("wins", chart_wins), ("read", chart_read), ("behind", chart_behind)):
        files[f"engine-{name}.svg"] = fn(data, "paper")[0]
    return files


def main():
    files = build()
    if "--check" in sys.argv:
        stale = [n for n, s in files.items() if not (OUT / n).is_file() or (OUT / n).read_text() != s]
        if stale:
            print("stale, run scripts/make_engine_art.py:", *stale)
            sys.exit(1)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for n, s in files.items():
        (OUT / n).write_text(s)
        print("wrote", (OUT / n).relative_to(ROOT))


if __name__ == "__main__":
    main()
