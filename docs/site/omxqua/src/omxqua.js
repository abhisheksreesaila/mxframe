/* omxqua · omxqua.js (no dependencies, ~4 KB)
   1. Press: every tappable thing squashes on pointer-down and springs back on release.
      The amount follows the size, so it feels the same everywhere: small 0.90 · medium 0.96 · large 0.985.
      An element (or its skin) can set --oq-press: ledge to sink 4 px instead (the Kids Sticker skin).
   2. Logo: .mark writes itself on press (its letters or icon are drawn as a stroke, then filled).
   3. Helpers on window.Omxqua for the moving parts: segmented thumbs, gentle menus, toasts,
      the Beacon morph and the Top nav pill (see AGENTS.md).
   Opt out with data-nopress. Respects prefers-reduced-motion. */
(() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PRESS = "button, a.cta, a.more, [data-press], .tile, .blk, .day, .row, .row-demo, .chip, .nav, .mark, .brand-isl, .brand";
  const spring = el => getComputedStyle(el).getPropertyValue("--oq-spring-fast").trim() || "cubic-bezier(.2,.9,.3,1.15)";
  const amount = el => { const w = el.getBoundingClientRect().width; return w <= 48 ? 0.9 : w <= 260 ? 0.96 : 0.985; };

  addEventListener("pointerdown", e => {
    if (reduce || e.button > 0) return;
    const el = e.target.closest(PRESS);
    if (!el || el.disabled || el.closest("[data-nopress]")) return;
    const ledge = getComputedStyle(el).getPropertyValue("--oq-press").trim() === "ledge";
    const down = ledge ? { translate: "0 4px" } : { scale: String(amount(el)) }, rest = ledge ? { translate: "0 0" } : { scale: "1" };
    el.getAnimations().forEach(a => a.id === "oq-press" && a.cancel());
    const a = el.animate([rest, down], { duration: 110, easing: "ease-out", fill: "forwards" }); a.id = "oq-press";
    const up = () => {
      removeEventListener("pointerup", up); removeEventListener("pointercancel", up);
      const b = el.animate([down, rest], { duration: 320, easing: spring(el) }); b.id = "oq-press"; a.cancel();
    };
    addEventListener("pointerup", up); addEventListener("pointercancel", up);
    if (el.matches(".mark, .brand, .brand-isl")) writeLogo(el.matches(".mark") ? el : el.querySelector(".mark"));
  }, true);

  /* Logo: letters → <svg><text>, so the stroke can be drawn; icons keep their paths. */
  function prepare(mark) {
    if (!mark || mark.dataset.oqLogo) return;
    mark.dataset.oqLogo = "1";
    const txt = mark.textContent.trim();
    if (!mark.querySelector("svg") && txt) {
      const big = txt.length === 1;
      mark.innerHTML = `<svg viewBox="0 0 40 40" class="oq-logo" aria-label="${txt}"><text x="20" y="${big ? 27.5 : 25.5}" text-anchor="middle" font-size="${big ? 22 : 17}" font-weight="700" font-family="inherit">${txt}</text></svg>`;
    }
    mark.querySelectorAll("svg path, svg circle, svg text").forEach(p => { if (p.tagName !== "text") p.setAttribute("pathLength", "1"); });
  }
  function writeLogo(mark) {
    if (!mark || reduce) return;
    prepare(mark);
    mark.querySelectorAll("svg text").forEach(t => {
      t.getAnimations().forEach(a => a.cancel());
      t.animate([{ strokeDashoffset: 120, fillOpacity: 0 }, { strokeDashoffset: 0, fillOpacity: 0, offset: .75 }, { strokeDashoffset: 0, fillOpacity: 1 }],
        { duration: 900, easing: "cubic-bezier(.4,0,.2,1)" });
    });
    mark.querySelectorAll("svg path, svg circle").forEach(p => {
      p.getAnimations().forEach(a => a.cancel());
      p.animate([{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }], { duration: 700, easing: "cubic-bezier(.4,0,.2,1)" });
    });
  }
  const style = document.createElement("style");
  style.textContent = `.oq-logo { width: 100%; height: 100%; display: block; overflow: visible; }
    .oq-logo text { fill: currentColor; stroke: currentColor; stroke-width: .8; stroke-dasharray: 120; stroke-dashoffset: 0; paint-order: stroke; }
    .mark svg path, .mark svg circle { stroke-dasharray: 1; stroke-dashoffset: 0; }
    .mark { cursor: pointer; }`;
  document.head.appendChild(style);
  /* write each logo once when it first appears */
  const seen = new WeakSet();
  const scan = () => document.querySelectorAll(".mark").forEach(m => { if (!seen.has(m)) { seen.add(m); prepare(m); if (m.getClientRects().length) setTimeout(() => writeLogo(m), 250); } });
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
  addEventListener("DOMContentLoaded", scan); scan();

  /* ---- helpers (used by docs/showcase.html; same timings as LANGUAGE.md's motion budget) ---- */
  /* Segmented control, tabs, tab bar, top nav: the first child is the thumb; it slides under button.on. */
  function placeThumb(sg) { const on = sg.querySelector("button.on"), t = sg.firstElementChild; if (!on || !t) return; t.style.left = on.offsetLeft + "px"; t.style.width = on.offsetWidth + "px"; }
  /* Settle tier: a panel or view arrives (fade + 6px rise, 280 ms, decelerating). */
  function settle(el) { if (!reduce) el.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 280, easing: "cubic-bezier(.2,0,0,1)" }); }
  /* Gentle tier: menus, popovers, tooltips ease in over 220 ms and fade out in 150 ms (then [hidden]). */
  function gentleIn(el) { el.hidden = false; if (!reduce) el.animate([{ opacity: 0, transform: "translateY(-4px) scale(.97)" }, { opacity: 1, transform: "none" }], { duration: 220, easing: "cubic-bezier(.2,0,0,1)" }); }
  function gentleOut(el) { if (el.hidden) return; if (reduce) { el.hidden = true; return; } el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 150, easing: "cubic-bezier(.3,0,1,1)" }).onfinish = () => el.hidden = true; }
  /* Toast: shows msg in msgEl, rises in if it was hidden, fades out 2.6 s after the last call. */
  const toastTimers = new WeakMap();
  function toast(el, msg, msgEl = el.firstElementChild) {
    msgEl.textContent = msg; clearTimeout(toastTimers.get(el));
    if (el.hidden) { el.hidden = false; if (!reduce) el.animate([{ opacity: 0, transform: "translate(-50%, 8px)" }, { opacity: 1, transform: "translate(-50%, 0)" }], { duration: 220, easing: "cubic-bezier(.2,0,0,1)" }); }
    toastTimers.set(el, setTimeout(() => gentleOut(el), 2600));
  }
  /* Beacon: size the pill to its compact or open content; the CSS transition does the morph. */
  function sizeBeacon(b, animate) {
    const open = b.classList.contains("open"), el = b.querySelector(open ? ".bc-open" : ".bc-compact");
    const w = open ? el.offsetWidth : el.scrollWidth, h = open ? el.offsetHeight : 40;
    if (!animate || reduce) { const t = b.style.transition; b.style.transition = "none"; b.style.width = w + "px"; b.style.height = h + "px"; b.offsetWidth; b.style.transition = t; }
    else { b.style.width = w + "px"; b.style.height = h + "px"; }
  }
  /* Top nav pill: 4 items at rest (the 4th is the active extra if one is picked) · +N · the rest.
     openTopnav(nav, true) widens it to all; foldTopnavLater folds it back 3.5 s after you leave; holdTopnav cancels that. */
  const navTimers = new WeakMap();
  function layoutTopnav(nav, animate) {
    if (!nav) return;
    const btns = [...nav.querySelectorAll("button[data-i]")], more = nav.querySelector("[data-more]"), open = nav.classList.contains("open");
    const active = btns.findIndex(b => b.classList.contains("on"));
    const shown = active >= 4 ? [0, 1, 2, active] : [0, 1, 2, 3];
    btns.forEach((b, i) => { b.classList.toggle("keep", shown.includes(i)); b.style.order = shown.includes(i) ? shown.indexOf(i) : 10 + i; });
    more.style.order = open ? 99 : 5;
    btns.forEach(b => b.classList.toggle("extra", !shown.includes(+b.dataset.i)));
    more.querySelector("span").textContent = btns.length - 4;
    const pad = 10, w = open ? btns.reduce((a, b) => a + b.offsetWidth, 0) + pad : shown.reduce((a, i) => a + btns[i].offsetWidth, 0) + more.offsetWidth + pad;
    if (!animate || reduce) { const t = nav.style.transition; nav.style.transition = "none"; nav.style.width = w + "px"; nav.offsetWidth; nav.style.transition = t; } else nav.style.width = w + "px";
    requestAnimationFrame(() => placeThumb(nav));
  }
  function holdTopnav(nav) { if (nav) clearTimeout(navTimers.get(nav)); }
  function openTopnav(nav, on) { if (!nav) return; holdTopnav(nav); if (nav.classList.contains("open") === on) return; nav.classList.toggle("open", on); layoutTopnav(nav, true); }
  function foldTopnavLater(nav, ms = 3500) { if (!nav) return; holdTopnav(nav); navTimers.set(nav, setTimeout(() => openTopnav(nav, false), ms)); }

  window.Omxqua = Object.assign(window.Omxqua || {}, { writeLogo, placeThumb, settle, gentleIn, gentleOut, toast, sizeBeacon, layoutTopnav, openTopnav, foldTopnavLater, holdTopnav });
})();
