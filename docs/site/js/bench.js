// Interactive benchmarks page (F-049). Everything shown is computed here from the files in bench/.
// The first half is pure logic (no DOM), so tests can run it under node and compare with the CSV.

const ENGINES = [["mx-cpu", "omxcore", "--mx"], ["duckdb", "DuckDB", "--dk"], ["polars", "Polars", "--pl"]];
const FILES = {
  stored: "bench/m2s-engine.csv",
  shuffled: "bench/m2s-engine-shuffled.csv",
  randoms: [
    ["SF1, stored rows", "bench/random-sf1-m2s.json"],
    ["SF1, shuffled rows", "bench/random-sf1-shuffled-m2s.json"],
    ["SF10, stored rows", "bench/random-sf10-m2s.json"],
  ],
};

// ---------- logic ----------
const finite = (v) => v !== "" && v != null && Number.isFinite(+v);

// rows -> {queries: [1..22], ms[engine][q], verdict[engine][q], load[]} for one scale factor
function queryMatrix(rows, sf) {
  const m = { queries: [], ms: {}, verdict: {}, load: [] };
  for (const r of rows) {
    if (r.sf !== String(sf)) continue;
    const q = +r.query;
    if (!m.queries.includes(q)) m.queries.push(q);
    (m.ms[r.engine] ||= {})[q] = finite(r.first_ms) ? +r.first_ms : null;
    (m.verdict[r.engine] ||= {})[q] = r.verdict;
    if (finite(r.load1)) m.load.push(+r.load1);
  }
  m.queries.sort((a, b) => a - b);
  return m;
}

// sum of first-run ms over queries, optionally leaving one out; null if any query is missing
function total(m, engine, skip) {
  let s = 0;
  for (const q of m.queries) {
    if (q === skip) continue;
    const v = m.ms[engine] && m.ms[engine][q];
    if (v == null) return null;
    s += v;
  }
  return s;
}

const wrongQueries = (m, engine) => m.queries.filter((q) => m.verdict[engine] && m.verdict[engine][q] === "wrong");
const median = (xs) => { const s = [...xs].sort((a, b) => a - b), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };

// queries where DuckDB is strictly faster than omxcore
const duckdbFaster = (m) => m.queries.filter((q) => m.ms.duckdb[q] < m.ms["mx-cpu"][q]);

// Two totals within 1% of the larger one are a tie, not a win (at SF1 stored, without Q21: 386.9 vs 387.4 ms).
const TIE_TOLERANCE = 0.01;
function compareTotals(a, b, tol = TIE_TOLERANCE) {
  const tie = Math.abs(a - b) <= tol * Math.max(a, b);
  return { tie, aLower: a < b };
}

// random suite: per shape (and "all"), the geometric mean of DuckDB ms / omxcore ms (above 1: omxcore faster)
function randomShapes(results) {
  const by = {};
  for (const r of results) {
    const a = r.mx && r.mx.ms, b = r.duckdb && r.duckdb.ms;
    for (const k of [r.shape, "all"]) {
      const s = (by[k] ||= { shape: k, n: 0, ran: 0, correct: 0, faster: 0, logSum: 0 });
      s.n++;
      if (r.mx && r.mx.verdict === "ok") s.correct++;
      if (finite(a) && finite(b)) { s.ran++; s.logSum += Math.log(b / a); if (a < b) s.faster++; }
    }
  }
  return Object.values(by)
    .map((s) => ({ ...s, ratio: s.ran ? Math.exp(s.logSum / s.ran) : null }))
    .sort((x, y) => (x.shape === "all") - (y.shape === "all") || x.shape.localeCompare(y.shape));
}

if (typeof module !== "undefined") module.exports = { compareTotals, TIE_TOLERANCE, queryMatrix, total, wrongQueries, median, duckdbFaster, randomShapes };

// ---------- page ----------
if (typeof document !== "undefined") (function () {
  const $ = (id) => document.getElementById(id);
  const state = { sf: "10", order: "stored", on: new Set(ENGINES.map(([e]) => e)) };
  const data = { csv: {}, random: [] };
  const sec = (s) => (s / 1000).toFixed(2) + " s";
  const ratioTxt = (r) => (r >= 1 ? r.toFixed(2) + "x" : "1/" + (1 / r).toFixed(2) + "x");
  const engRows = () => ENGINES.filter(([e]) => state.on.has(e));
  const dot = (c) => `<i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:var(${c});margin-right:6px"></i>`;
  const verdictBadge = (v) => v === "wrong" ? '<span class="badge error">wrong result</span>'
    : v === "drift" ? '<span class="badge warn">drift</span>' : v === "ref" ? '<span class="badge brand">reference</span>' : "";
  const matrix = () => queryMatrix(data.csv[state.order], state.sf);
  const chartWidth = (el, min) => Math.max(el.clientWidth || 0, min);

  function button(group, value, label, on) {
    return `<button class="btn${on ? " on" : ""}" type="button" data-g="${group}" data-v="${value}" aria-pressed="${on}">${label}</button>`;
  }
  function controls() {
    $("controls").innerHTML =
      button("sf", "1", "SF1", state.sf === "1") + button("sf", "10", "SF10", state.sf === "10") + '<span class="gap"></span>' +
      button("order", "stored", "Stored rows", state.order === "stored") + button("order", "shuffled", "Shuffled rows", state.order === "shuffled") +
      '<span class="gap"></span>' +
      ENGINES.map(([e, n, c]) => `<label class="eng"><input type="checkbox" data-e="${e}" ${state.on.has(e) ? "checked" : ""}> ${dot(c)}${n}</label>`).join("");
    $("controls").querySelectorAll("button").forEach((b) => (b.onclick = () => { state[b.dataset.g] = b.dataset.v; controls(); render(); }));
    $("controls").querySelectorAll("input").forEach((i) => (i.onchange = () => {
      if (i.checked) state.on.add(i.dataset.e); else if (state.on.size > 1) state.on.delete(i.dataset.e);
      controls(); render();
    }));
  }

  // ----- totals -----
  function tiles() {
    const m = matrix(), list = engRows();
    const t = Object.fromEntries(list.map(([e]) => [e, total(m, e)]));
    const t21 = Object.fromEntries(list.map(([e]) => [e, total(m, e, 21)]));
    const tied = (o, e) => Object.entries(o).some(([k, v]) => k !== e && v != null && compareTotals(v, o[e]).tie);   // a tie is not a win
    const low = (o) => Math.min(...Object.values(o).filter((v) => v != null));
    const best = low(t), best21 = low(t21);
    $("totals").innerHTML = list.map(([e, n, c]) => {
      const w = wrongQueries(m, e);
      return `<div class="card stat"><span class="label">${dot(c)}${n}</span><span class="num">${t[e] == null ? "n/a" : sec(t[e])}</span>` +
        (t[e] === best && list.length > 1 ? '<span class="badge ok">lowest total</span> ' : "") +
        (w.length ? `<span class="badge error">Q${w.join(", Q")} wrong</span>` : "") +
        `<div style="font-size:13px;margin-top:6px;color:var(--text-mid)">without Q21: <span class="num" style="font-size:15px">${t21[e] == null ? "n/a" : sec(t21[e])}</span>` +
        (t21[e] === best21 && list.length > 1 && !tied(t21, e) ? ' <span class="badge ok">lowest</span>' : "") + `</div></div>`;
    }).join("");
    const mx = total(m, "mx-cpu"), dk = total(m, "duckdb"), mx21 = total(m, "mx-cpu", 21), dk21 = total(m, "duckdb", 21);
    const sfTxt = "SF" + state.sf;
    let s = "";
    if (mx != null && dk != null) {
      const cmp = compareTotals(dk21, mx21), lo = cmp.aLower ? ["DuckDB", dk21, "omxcore", mx21] : ["omxcore", mx21, "DuckDB", dk21];
      s = (cmp.tie
        ? `Without Q21, omxcore and DuckDB tie at ${sfTxt} (${sec(mx21)} and ${sec(dk21)}, within 1%), `
        : `Without Q21, ${lo[0]}'s ${sfTxt} total is the lower one (${sec(lo[1])} against ${lo[2]}'s ${sec(lo[3])}), `) +
        `and DuckDB is faster on ${duckdbFaster(m).length} of ${m.queries.length} queries. `;
    }
    $("totals-note").textContent = s + "Query time only: a fresh process per query, data already loaded, no warm-up.";
    $("totals-head").textContent = `TPC-H ${sfTxt} · ${state.order} rows · all ${m.queries.length} queries · engine only`;
  }

  // ----- shared svg bits -----
  function frame(el, W, H) {
    el.innerHTML = "";
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    el.append(s);
    return s;
  }
  const fmtTick = (v) => (v >= 1000 ? v / 1000 + " s" : v + " ms");

  // ----- main chart: dot plot, log x -----
  function dotPlot() {
    const m = matrix(), list = engRows(), el = $("chartB");
    const all = ENGINES.flatMap(([e]) => m.queries.map((q) => m.ms[e] && m.ms[e][q])).filter((v) => v != null && v > 0);
    const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...all)))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...all))));
    const W = chartWidth(el, 640), rh = 22, L = 44, R = 16, T = 6, H = T + m.queries.length * rh + 28;
    const X = (v) => L + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (W - L - R);
    const s = frame(el, W, H);
    for (let g = lo; g <= hi * 1.001; g *= 10) {
      s.append(svg("line", { x1: X(g), x2: X(g), y1: T, y2: H - 22, stroke: cssVar("--border") }));
      s.append(svg("text", { x: X(g), y: H - 8, "text-anchor": "middle" }, fmtTick(Math.round(g))));
    }
    m.queries.forEach((q, i) => {
      const y = T + i * rh + rh / 2;
      const vals = list.map(([e]) => m.ms[e] && m.ms[e][q]).filter((v) => v != null);
      s.append(svg("text", { x: L - 8, y: y + 3, "text-anchor": "end" }, "Q" + q));
      if (vals.length > 1) s.append(svg("line", { x1: X(Math.min(...vals)), x2: X(Math.max(...vals)), y1: y, y2: y, stroke: cssVar("--border-hi"), "stroke-width": 1.5 }));
      [...list].reverse().forEach(([e, , c]) => {          // omxcore drawn last, on top
        const v = m.ms[e] && m.ms[e][q];
        if (v == null) return;
        s.append(svg("circle", { cx: X(v), cy: y, r: 4, fill: cssVar(c), stroke: cssVar("--surface"), "stroke-width": 1.5, "pointer-events": "none" }));
        if (m.verdict[e][q] === "wrong") s.append(svg("circle", { cx: X(v), cy: y, r: 7.5, fill: "none", stroke: cssVar("--red"), "stroke-width": 1.5, "pointer-events": "none" }));
      });
      const hit = svg("rect", { x: 0, y: y - rh / 2, width: W, height: rh, fill: "transparent" });
      hover(hit, () => `<b>Q${q}</b><br>` + list.map(([e, n, c]) => {
        const v = m.ms[e] && m.ms[e][q];
        return `${dot(c)}${n}: <b>${v == null ? "n/a" : fmtMs(v)}</b> ${verdictBadge(m.verdict[e][q])}`;
      }).join("<br>"));
      s.append(hit);
    });
    $("legendB").innerHTML = list.map(([, n, c]) => `<span><i style="background:var(${c})"></i>${n}</span>`).join("") +
      (list.some(([e]) => wrongQueries(m, e).length) ? `<span><i style="background:none;border:1.5px solid var(--red)"></i>wrong result (differs from the reference)</span>` : "");
    table("tblB", ["Query", ...list.map(([, n]) => n + " ms"), "Verdicts"],
      m.queries.map((q) => [`Q${q}`, ...list.map(([e]) => m.ms[e][q] == null ? "n/a" : m.ms[e][q].toFixed(1)),
        list.map(([e, n]) => (m.verdict[e][q] === "wrong" || m.verdict[e][q] === "drift") ? `${n}: ${verdictBadge(m.verdict[e][q])}` : "").filter(Boolean).join(" ")]));
  }

  // ----- second view: speedup vs DuckDB -----
  function speedup() {
    const m = matrix(), el = $("chartC");
    const rs = m.queries.map((q) => [q, m.ms.duckdb[q] / m.ms["mx-cpu"][q]]);
    const mx = Math.max(...rs.map(([, r]) => Math.max(r, 1 / r)));
    const span = [2, 4, 8, 16, 32].find((k) => k >= mx) || 64, lg = Math.log10(span);
    const W = chartWidth(el, 640), rh = 20, L = 44, R = 50, T = 6, H = T + rs.length * rh + 40;
    const half = (W - L - R) / 2, C = L + half;
    const X = (r) => C + (Math.log10(r) / lg) * half;
    const s = frame(el, W, H);
    for (let k = 2; k <= span; k *= 2) for (const r of [k, 1 / k]) {
      s.append(svg("line", { x1: X(r), x2: X(r), y1: T, y2: H - 30, stroke: cssVar("--border") }));
      s.append(svg("text", { x: X(r), y: H - 18, "text-anchor": "middle" }, (r < 1 ? "1/" + 1 / r : r) + "x"));
    }
    s.append(svg("text", { x: C - 6, y: H - 3, "text-anchor": "end" }, "← DuckDB faster"));
    s.append(svg("text", { x: C + 6, y: H - 3 }, "omxcore faster →"));
    rs.forEach(([q, r], i) => {
      const y = T + i * rh + rh / 2, x = X(r);
      s.append(svg("text", { x: L - 8, y: y + 3, "text-anchor": "end" }, "Q" + q));
      s.append(svg("rect", { x: Math.min(C, x), y: y - 4, width: Math.max(2, Math.abs(x - C)), height: 8, rx: 1.5, fill: cssVar(r >= 1 ? "--fast" : "--slow") }));
      s.append(svg("text", { x: r >= 1 ? x + 5 : x - 5, y: y + 3, "text-anchor": r >= 1 ? "start" : "end" }, r.toFixed(2) + "x"));
      const hit = svg("rect", { x: 0, y: y - rh / 2, width: W, height: rh, fill: "transparent" });
      hover(hit, () => `<b>Q${q}</b><br>${dot("--mx")}omxcore ${fmtMs(m.ms["mx-cpu"][q])}<br>${dot("--dk")}DuckDB ${fmtMs(m.ms.duckdb[q])}<br>` +
        (r >= 1 ? `omxcore ${r.toFixed(2)}x faster` : `DuckDB ${(1 / r).toFixed(2)}x faster`));
      s.append(hit);
    });
    s.append(svg("line", { x1: C, x2: C, y1: T, y2: H - 30, stroke: cssVar("--mid"), "stroke-width": 1.5 }));
    s.append(svg("text", { x: C, y: H - 18, "text-anchor": "middle" }, "1x"));
    table("tblC", ["Query", "omxcore ms", "DuckDB ms", "DuckDB / omxcore", "Faster"],
      rs.map(([q, r]) => [`Q${q}`, m.ms["mx-cpu"][q].toFixed(1), m.ms.duckdb[q].toFixed(1), r.toFixed(2) + "x", r >= 1 ? "omxcore" : "DuckDB"]));
  }

  // ----- random shapes -----
  function randomChart() {
    const host = $("randoms");
    host.innerHTML = "";
    const all = data.random.map(([, res]) => randomShapes(res));
    const mx = Math.max(...all.flat().map((s) => Math.max(s.ratio, 1 / s.ratio)));
    const span = [2, 4, 8].find((k) => k >= mx) || 16, lg = Math.log10(span);
    data.random.forEach(([label], k) => {
      const box = document.createElement("div");
      box.innerHTML = `<div class="label" style="margin:0 0 4px">${label} · ${all[k].at(-1).n} queries</div><div class="chart"></div>`;
      host.append(box);
      const el = box.querySelector(".chart"), rows = all[k], W = Math.max(el.clientWidth, 260), rh = 22, L = 112, R = 52, H = rows.length * rh + 34;
      const half = (W - L - R) / 2, C = L + half, X = (r) => C + (Math.log10(r) / lg) * half;
      const s = frame(el, W, H);
      s.style.minWidth = "0";
      for (const r of [1 / span, span]) {
        s.append(svg("line", { x1: X(r), x2: X(r), y1: 0, y2: H - 28, stroke: cssVar("--border") }));
        s.append(svg("text", { x: X(r), y: H - 14, "text-anchor": "middle" }, (r < 1 ? "1/" + 1 / r : r) + "x"));
      }
      rows.forEach((row, i) => {
        const y = i * rh + rh / 2 + 2, x = X(row.ratio), isAll = row.shape === "all";
        s.append(svg("text", { x: L - 8, y: y + 3, "text-anchor": "end", "font-weight": isAll ? 700 : 400 }, row.shape));
        s.append(svg("rect", { x: Math.min(C, x), y: y - 4, width: Math.max(2, Math.abs(x - C)), height: 8, rx: 1.5, fill: cssVar(row.ratio >= 1 ? "--fast" : "--slow") }));
        // the value sits in its own column at the right edge: a bar that points left would run into the shape name
        s.append(svg("text", { x: W - 4, y: y + 3, "text-anchor": "end" }, row.ratio.toFixed(2) + "x"));
        const hit = svg("rect", { x: 0, y: y - rh / 2, width: W, height: rh, fill: "transparent" });
        hover(hit, () => `<b>${row.shape}</b> · ${label}<br>${row.n} queries, omxcore correct ${row.correct}/${row.n}<br>` +
          `faster than DuckDB on ${row.faster}/${row.ran}<br>DuckDB / omxcore, geo. mean: <b>${ratioTxt(row.ratio)}</b>`);
        s.append(hit);
      });
      s.append(svg("line", { x1: C, x2: C, y1: 0, y2: H - 28, stroke: cssVar("--mid"), "stroke-width": 1.5 }));
      s.append(svg("text", { x: C, y: H - 14, "text-anchor": "middle" }, "1x"));
    });
    const shapes = all[0].map((s) => s.shape);
    table("tblR", ["Shape", ...data.random.flatMap(([l]) => [l + ": queries", "correct", "faster than DuckDB", "DuckDB / omxcore"])],
      shapes.map((sh) => [sh, ...all.flatMap((rows) => { const s = rows.find((x) => x.shape === sh);
        return s ? [s.n, `${s.correct}/${s.n}`, `${s.faster}/${s.ran}`, s.ratio.toFixed(2) + "x"] : ["", "", "", ""]; })]));
  }

  // ----- run conditions -----
  function conditions() {
    const rows = ["stored", "shuffled"].map((o) => {
      const l = data.csv[o].filter((r) => finite(r.load1)).map((r) => +r.load1);
      return `<tr><td>TPC-H, ${o} rows</td><td class="r num">${median(l).toFixed(1)}</td><td class="r num">${Math.max(...l).toFixed(1)}</td></tr>`;
    }).join("");
    $("loadtbl").innerHTML = `<tr><th>Run</th><th class="r">median load1</th><th class="r">max load1</th></tr>` + rows;
  }

  function table(id, head, body) {
    $(id).innerHTML = `<tr>${head.map((h, i) => `<th${i ? ' class="r"' : ""}>${h}</th>`).join("")}</tr>` +
      body.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "r num" : "mono"}">${c}</td>`).join("")}</tr>`).join("");
  }

  function render() {
    if (data.csv.stored && data.csv.shuffled) { tiles(); dotPlot(); speedup(); }
    if (data.random.length) randomChart();
  }

  // ----- boot -----
  controls();
  Promise.all(["stored", "shuffled"].map((o) => loadCSV(FILES[o]).then((rows) => (data.csv[o] = rows)).catch((err) => { err.path = FILES[o]; throw err; })))
    .then(() => { conditions(); render(); })
    .catch((err) => ["totals", "chartB", "chartC"].forEach((id) => showError($(id), err.path, err)));
  Promise.all(FILES.randoms.map(([label, path]) => loadJSON(path).then((res) => [label, res]).catch((err) => { err.path = path; throw err; })))
    .then((res) => { data.random = res; randomChart(); })
    .catch((err) => showError($("randoms"), err.path, err));

  document.addEventListener("themechange", render);
  let timer;
  addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(render, 150); });
})();
