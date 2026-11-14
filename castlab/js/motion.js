/* CastLab interaction layer: page transitions, count-up numbers, cursor glow, quick-search palette,
   install-as-app. All motion is skipped when the visitor prefers reduced motion. */
(function () {
  'use strict';
  const { $, $$, esc } = U, reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- page transition, scroll reveals, tab transitions, count-up ---- */
  CL._viewAt = performance.now();
  let lastView = null, seen = new WeakSet(), queue = [], qTimer = 0;
  const lastAnim = new WeakMap(), SEL = '.card, .kpi, .tour a, .rec, .fishchip, .daychip';
  function countUp(el) {
    const m = el.textContent.trim().match(/^(-?\d+(?:\.\d+)?)(.*)$/); if (!m || el._counted === el.textContent) return;
    const end = parseFloat(m[1]), dec = (m[1].split('.')[1] || '').length, tail = m[2], final = el.textContent, t0 = performance.now(), dur = 650;
    const step = now => { const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3); el.textContent = (end * e).toFixed(dec) + tail; if (k < 1 && el.isConnected) requestAnimationFrame(step); else { el.textContent = final; el._counted = final; } };
    requestAnimationFrame(step);
  }
  function flush() {
    qTimer = 0;
    queue.forEach((el, i) => {
      if (!el.isConnected) return; const d = Math.min(i * 45, 400);
      el.animate([{ opacity: 0, transform: 'translateY(16px) scale(.985)' }, { opacity: 1, transform: 'none' }], { duration: 520, delay: d, easing: 'cubic-bezier(.2,.7,.2,1)', fill: 'backwards' });
      if (el.classList.contains('kpi')) { const v = el.querySelector('.v'); if (v) setTimeout(() => countUp(v), d); }
    });
    queue = [];
  }
  function enqueue(el) { queue.push(el); if (!qTimer) qTimer = requestAnimationFrame(flush); }
  const io = 'IntersectionObserver' in window ? new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.unobserve(e.target); enqueue(e.target); } }), { rootMargin: '0px 0px -6% 0px' }) : null;
  function scan(root, quiet) {
    if (reduce || !root || root.nodeType !== 1) return;
    const list = root.matches && root.matches(SEL) ? [root] : []; root.querySelectorAll(SEL).forEach(x => list.push(x));
    list.forEach(el => { if (seen.has(el)) return; seen.add(el); if (quiet || el.closest('[hidden]')) return; if (io) io.observe(el); else enqueue(el); });
  }
  function entered(view) {
    if (!view || view.id === lastView) return; lastView = view.id; CL._viewAt = performance.now();
    if (reduce) return; seen = new WeakSet();
    view.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: 'ease-out' });
    scan(view);
    if (view.id === 'v-gearsim' && !CL._gsDemo) { CL._gsDemo = true; setTimeout(() => { if (!$('#v-gearsim').hidden) $('#gsAnimate').click(); }, 700); }
  }
  new MutationObserver(muts => {
    const view = $$('.view').find(v => !v.hidden); if (view && view.id !== lastView) entered(view);
    if (reduce) return; const now = performance.now();
    muts.forEach(m => {
      if (m.type === 'attributes') {
        const t = m.target;
        if (/\b(lltab|cctab|gstab)\b/.test(t.className) && !t.hidden) { t.animate([{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { duration: 280, easing: 'cubic-bezier(.2,.7,.2,1)' }); seen = new WeakSet(); scan(t); }
        return;
      }
      const p = m.target; if (!p || !p.closest || !p.closest('.view') || p.closest('.leaflet-container')) return;
      const quiet = now - (lastAnim.get(p) || 0) < 1500; lastAnim.set(p, now);   // repeated redraws (sliders, typing) stay still
      m.addedNodes.forEach(n => scan(n, quiet));
    });
  }).observe($('#main'), { attributes: true, attributeFilter: ['hidden'], subtree: true, childList: true });
  setTimeout(() => entered($$('.view').find(v => !v.hidden)), 0);

  /* ---- charts grow in when a page opens (not on every slider redraw) ---- */
  if (window.Chart && !reduce) Chart.defaults.animation.delay = c => (c.type === 'data' && c.mode === 'default' && performance.now() - CL._viewAt < 1800) ? Math.min(c.dataIndex, 40) * 14 + c.datasetIndex * 90 : 0;

  /* ---- cursor glow on the Today header ---- */
  const hero = $('.hero');
  if (hero && !reduce && matchMedia('(pointer: fine)').matches) hero.addEventListener('pointermove', e => { const r = hero.getBoundingClientRect(); hero.style.setProperty('--mx', (e.clientX - r.left) + 'px'); hero.style.setProperty('--my', (e.clientY - r.top) + 'px'); });

  /* ---- quick search (Ctrl/Cmd + K, or the sidebar button) ---- */
  const pal = document.createElement('div'); pal.className = 'pal'; pal.hidden = true;
  pal.innerHTML = '<div class="pal-box" role="dialog" aria-label="Quick search"><input type="search" id="palIn" placeholder="Jump to a page, a spot or an action…" autocomplete="off"><div id="palList" role="listbox"></div><div class="pal-foot">Enter to open, arrows to move, Esc to close</div></div>';
  document.body.appendChild(pal);
  let items = [], sel = 0;
  function build() {
    const out = $$('.nav a').map(a => ({ label: a.textContent.replace(/\s+(STATS|GEO|MECH)\s*$/, '').trim(), kind: 'Page', run: () => { location.hash = a.getAttribute('href').slice(1); } }));
    (Store.data ? Store.list('locations') : []).filter(l => !l.simulated && isFinite(l.lat)).forEach(l => out.push({ label: l.name, kind: 'Spot', run: () => { CL.pickLoc = l.id; if (location.hash === '#home') CL.modules.home.render(); else location.hash = 'home'; } }));
    [['Log a catch', 'catches'], ['Log a full trip', 'lurelab/sessions'], ['Run prediction accuracy check', 'lurelab/validation'], ['Design a lure experiment', 'lurelab/experiments'], ['Optimise a rod design', 'gearsim/opt'], ['Record a field observation', 'castcheck/obs'], ['Download research report', 'research'], ['Export my data', 'account']].forEach(a => out.push({ label: a[0], kind: 'Action', run: () => { location.hash = a[1]; } }));
    out.push({ label: 'Switch light / dark theme', kind: 'Action', run: () => $('#themeBtn').click() });
    return out;
  }
  function draw() { const q = $('#palIn').value.trim().toLowerCase(), all = build(); items = (q ? all.filter(i => i.label.toLowerCase().includes(q)) : all).slice(0, 9); sel = Math.min(sel, Math.max(0, items.length - 1)); $('#palList').innerHTML = items.length ? items.map((i, k) => '<button type="button" role="option" class="pal-item' + (k === sel ? ' on' : '') + '" data-k="' + k + '"><span>' + esc(i.label) + '</span><small>' + i.kind + '</small></button>').join('') : '<p class="pal-empty">Nothing matches. Try a page name or one of your spots.</p>'; }
  function open() { if (!Store.user) return; pal.hidden = false; $('#palIn').value = ''; sel = 0; draw(); $('#palIn').focus(); }
  function close() { pal.hidden = true; }
  function go(k) { const it = items[k]; if (!it) return; close(); it.run(); }
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); pal.hidden ? open() : close(); return; }
    if (pal.hidden) return;
    if (e.key === 'Escape') close(); else if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(items.length - 1, sel + 1); draw(); } else if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, sel - 1); draw(); } else if (e.key === 'Enter') { e.preventDefault(); go(sel); }
  });
  $('#palIn').addEventListener('input', () => { sel = 0; draw(); });
  pal.addEventListener('click', e => { const b = e.target.closest('.pal-item'); if (b) go(+b.dataset.k); else if (e.target === pal) close(); });
  const sb = $('#searchBtn'); if (sb) sb.onclick = open;

  /* ---- install as an app (offline shell) ---- */
  if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
  let deferred = null; const ib = $('#installBtn');
  window.addEventListener('beforeinstallprompt', e => { e.preventDefault(); deferred = e; if (ib) ib.hidden = false; });
  if (ib) ib.onclick = async () => { if (!deferred) return; deferred.prompt(); await deferred.userChoice; deferred = null; ib.hidden = true; };
})();
