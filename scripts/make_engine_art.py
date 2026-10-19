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
                  mx="#2a78d6", dk="#1baf7a", pa="#eda100", slow="#e34948", grid="#E4DDD0"),
}
SERIES = {"mx": "MX frame", "dk": "DuckDB", "pa": "pyarrow"}


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
        s = f"{v:,.0f}" if v >= 10 or isinstance(v, int) else f"{v:.1f}"
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
             f'M<tspan fill="{t["accent"]}">X</tspan> FRAME</text>')
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
    return svg_doc(W, H, "\n".join(b), "MX frame: the Mojo engine (editorial)",
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
         text(LX, 36, "// MX FRAME · THE MOJO ENGINE", 11, t["accent"], MONO, 700, ls=2),
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


AREA = {"joins": "join", "sql": "SQL", "streaming": "streamed", "aggregates": "1M groups"}
FOOT = "informal: shared 12-core Ryzen, other work running · source: docs/mojo-engine.md"


def chart_wins(data, theme):
    rows = []
    for sec, prefix in (("joins", "join"), ("sql", "SQL"), ("aggregates", "1M groups")):
        for r in data[sec]["rows"]:
            if r["mx"] < r["duckdb"]:
                name = r["label"] if prefix.lower() in r["label"].lower() else f"{prefix} · {r['label']}"
                rows.append(dict(label=name, ratio=r["duckdb"] / r["mx"],
                                 bars=[("mx", r["mx"], fmt(r["mx"], "ms")), ("dk", r["duckdb"], fmt(r["duckdb"], "ms"))],
                                 right=ratio_text(r["duckdb"], r["mx"])))
    rows.sort(key=lambda r: -r["ratio"])
    ticks = [(1, "1 ms"), (10, "10 ms"), (100, "100 ms"), (1000, "1 s")]
    return chart(theme, "Where the Mojo engine leads DuckDB", "SQL, joins and aggregates · time, log scale · lower is faster",
                 [("mx", "MX frame"), ("dk", "DuckDB 1.5.1")], rows, log_scale(1, 4000), ticks, FOOT, "FASTER")


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
                 [("mx", "MX frame"), ("dk", "DuckDB 1.5.1"), ("pa", "pyarrow")], rows, lambda v: v / 0.75, ticks, FOOT)


def gaps(data):
    out = []
    for g in data["behind"]:
        r = lookup(data, g["from"])
        unit = data[g["from"][0]]["unit"]
        out.append(dict(label=r["label"], area=g["from"][0], mx=r["mx"], dk=r["duckdb"], unit=unit,
                        lo=lo(r["mx"]) / hi(r["duckdb"]), hi=hi(r["mx"]) / lo(r["duckdb"]), next=g["next"]))
    return sorted(out, key=lambda g: -g["hi"])


def chart_behind(data, theme):
    rows = []
    for g in gaps(data):
        v = [g["lo"], g["hi"]] if g["hi"] - g["lo"] >= 0.05 else g["lo"]
        rows.append(dict(label=f"{AREA[g['area']]} · {g['label']}", sub=(f"next: {g['next']}" if g["next"] else ""),
                         bars=[("slow", v, (ratio_text(g['mx'], g['dk']), f"{fmt(g['mx'], g['unit'])} vs {fmt(g['dk'], g['unit'])}"))]))
    ticks = [(1, "1x"), (2, "2x"), (3, "3x")]
    return chart(theme, "Where we're still behind DuckDB", "how many times slower than DuckDB · 1x is a tie · shorter is better",
                 [("slow", "MX frame time / DuckDB time")], rows, lambda v: (v - 1) / 2.6, ticks, FOOT)


def build():
    data = load()
    files = {"engine-hero.svg": hero_paper(data)}
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
