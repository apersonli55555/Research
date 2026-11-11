/* CastLab interaction layer: page transitions, count-up numbers, cursor glow, quick-search palette,
   install-as-app. All motion is skipped when the visitor prefers reduced motion. */
(function () {
  'use strict';
  const { $, $$, esc } = U, reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- page transition + count-up when a section becomes visible ---- */
  let lastView = null;
  function countUp(el) {
    const m = el.textContent.trim().match(/^(-?\d+(?:\.\d+)?)(.*)$/); if (!m || el._counted === el.textContent) return;
    const end = parseFloat(m[1]), dec = (m[1].split('.')[1] || '').length, tail = m[2], final = el.textContent, t0 = performance.now(), dur = 650;
    const step = now => { const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3); el.textContent = (end * e).toFixed(dec) + tail; if (k < 1 && el.isConnected) requestAnimationFrame(step); else { el.textContent = final; el._counted = final; } };
    requestAnimationFrame(step);
  }
  function entered(view) {
    if (reduce || !view || view.id === lastView) return; lastView = view.id;
    view.animate([{ opacity: 0, transform: 'translateY(12px)' }, { opacity: 1, transform: 'none' }], { duration: 300, easing: 'cubic-bezier(.2,.7,.2,1)' });
    setTimeout(() => $$('.kpi .v', view).forEach(countUp), 60);
  }
  new MutationObserver(() => entered($$('.view').find(v => !v.hidden))).observe($('#main'), { attributes: true, attributeFilter: ['hidden'], subtree: true });

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
