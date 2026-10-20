// omxcore docs site: the top bar (the story page's), theme toggle, tooltip, small SVG and CSV helpers.
// Every page includes this file and a <nav class="side"></nav>, which becomes the top bar; pages are listed once, here.
const PAGES = [
  ["story.html", "The story"],
  ["engine.html", "Engine"],
  ["benchmarks.html", "Benchmarks"],
  ["auto-mode.html", "CPU or GPU"],
  ["internals.html", "Internals"],
  ["how-it-got-fast.html", "How it got fast"],
];
const REPO = "https://github.com/abhisheksreesaila/omxcore";

(function theme() {
  const saved = localStorage.getItem("mx-theme");
  if (saved) document.documentElement.dataset.theme = saved;
})();

function toggleTheme() {
  const r = document.documentElement;
  r.dataset.theme = r.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("mx-theme", r.dataset.theme);
  const btn = document.getElementById("theme-btn");
  if (btn) { btn.setAttribute("aria-pressed", r.dataset.theme === "dark"); btn.textContent = r.dataset.theme === "dark" ? "Light" : "Night"; }
  document.dispatchEvent(new Event("themechange"));
}

function renderNav() {
  const nav = document.querySelector("nav.side");
  if (!nav) return;
  const here = location.pathname.split("/").pop() || "index.html";
  const dark = document.documentElement.dataset.theme === "dark";
  nav.innerHTML = `<div class="barw"><a class="brand" href="index.html"><span class="x">omx</span>core</a>` +
    `<span class="badge ok">open source · Apache-2.0</span><span class="sp"></span><div class="links">` +
    PAGES.map(([href, name]) => `<a class="nav${href === here ? " on" : ""}" href="${href}"${href === here ? ' aria-current="page"' : ""}>${name}</a>`).join("") +
    `<a class="nav" href="${REPO}/blob/main/docs/reference.md">API ↗</a>` +
    `<a class="nav" href="${REPO}">GitHub ↗</a></div>` +
    `<button class="btn" id="theme-btn" type="button" aria-pressed="${dark}">${dark ? "Light" : "Night"}</button></div>`;
  document.getElementById("theme-btn").onclick = toggleTheme;
  const main = document.querySelector("main");
  if (main && !main.querySelector(".foot-links"))
    main.insertAdjacentHTML("beforeend", `<p class="foot-links">` +
      `<a href="story.html">The story</a> · <a href="index.html">Overview</a> · ` +
      `<a href="visualize/omxcore_pipeline.html">Kernel visualizer ↗</a> · ` +  // copied in by pages.yml
      `<a href="${REPO}/blob/main/docs/omxcore-mojo-api.md">Mojo API ↗</a> · <a href="${REPO}/blob/main/CHANGELOG.md">Changelog ↗</a> · ` +
      `omxcore is Apache-2.0</p>`);
}

// colour token as a value (charts are SVG and need real colours)
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const tipEl = () => document.querySelector(".tip") || document.body.appendChild(Object.assign(document.createElement("div"), { className: "tip" }));
function hover(el, html) {
  el.addEventListener("mousemove", (e) => {
    const t = tipEl();
    t.innerHTML = typeof html === "function" ? html() : html;
    t.style.display = "block";
    t.style.left = Math.min(e.clientX + 14, innerWidth - t.offsetWidth - 8) + "px";
    t.style.top = e.clientY + 14 + "px";
  });
  el.addEventListener("mouseleave", () => (tipEl().style.display = "none"));
}

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

// minimal CSV parser: header row, comma separated, no quoted fields (tests/test_docs_site.py checks that no CSV a page reads has a quote)
async function loadCSV(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const [head, ...rows] = (await res.text()).trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((line) => Object.fromEntries(line.split(",").map((v, i) => [cols[i], v])));
}

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

// show a load error where the chart would be, with the file that failed
function showError(el, path, err) {
  el.innerHTML = `<div class="status err">Couldn't load <a href="${path}">${path}</a> (${err.message}). ` +
    `The source files are in <a href="${REPO}/tree/main/docs/bench">docs/bench</a>.</div>`;
}

const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms");

document.addEventListener("DOMContentLoaded", renderNav);
