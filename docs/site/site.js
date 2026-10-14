// MXFrame docs site: sidebar nav, theme toggle, tooltip, small SVG and CSV helpers.
// Every page includes this file and a <nav class="side"></nav>; pages are listed once, here.
const PAGES = [
  ["index.html", "Overview"],
  ["benchmarks.html", "Benchmarks"],
  ["auto-mode.html", "How auto decides"],
  ["internals.html", "Behind the scenes"],
  ["how-it-got-fast.html", "How it got fast"],
];
const REPO = "https://github.com/abhisheksreesaila/mxframe";

(function theme() {
  const saved = localStorage.getItem("mx-theme");
  if (saved) document.documentElement.dataset.theme = saved;
})();

function toggleTheme() {
  const r = document.documentElement;
  r.dataset.theme = r.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("mx-theme", r.dataset.theme);
  document.getElementById("theme-btn")?.setAttribute("aria-pressed", r.dataset.theme === "dark");
  document.dispatchEvent(new Event("themechange"));
}

function renderNav() {
  const nav = document.querySelector("nav.side");
  if (!nav) return;
  const here = location.pathname.split("/").pop() || "index.html";
  nav.innerHTML = `<a class="brand" href="index.html">M<span class="x">X</span>FRAME</a>` +
    PAGES.map(([href, name]) => `<a class="nav${href === here ? " on" : ""}" href="${href}"${href === here ? ' aria-current="page"' : ""}>${name}</a>`).join("") +
    `<a class="nav" href="${REPO}#api-reference">API reference ↗</a>` +
    `<div class="foot"><button class="btn" id="theme-btn" type="button" aria-pressed="${document.documentElement.dataset.theme === "dark"}">Light / dark</button>` +
    `<a class="label" href="${REPO}">GitHub ↗</a></div>`;
  document.getElementById("theme-btn").onclick = toggleTheme;
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
