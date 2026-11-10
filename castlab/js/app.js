/* CastLab application shell: auth, routing, dashboard, map, weather, BaitRadar, GearMatch recommendations,
   research exports and account management. Science modules (lurelab.js, castcheck.js, gearsim.js) register
   themselves on window.CL.modules and read/write the same Store. */
(function () {
  'use strict';
  const { $, $$, esc, fmt, pct } = U;
  const CL = window.CL = { modules: {}, current: null };

  /* ---------- shared reference tables ---------- */
  /* General angling guidance (not measured data). line: [lo, hi] lb test; lure: [lo, hi] oz; power: rod power range. */
  CL.SPECIES = {
    'Bluegill / sunfish': { line: [2, 6], lure: [0.016, 0.125], power: ['UL', 'L'], braid: false },
    'Crappie': { line: [2, 8], lure: [0.031, 0.25], power: ['UL', 'L'], braid: false },
    'Trout': { line: [2, 8], lure: [0.0625, 0.25], power: ['UL', 'L'], braid: false },
    'Yellow perch': { line: [2, 8], lure: [0.031, 0.25], power: ['UL', 'L'], braid: false },
    'Walleye': { line: [6, 12], lure: [0.125, 0.5], power: ['ML', 'M'], braid: false },
    'Smallmouth bass': { line: [6, 14], lure: [0.125, 0.625], power: ['ML', 'M'], braid: false },
    'Largemouth bass': { line: [8, 25], lure: [0.25, 1], power: ['M', 'H'], braid: true },
    'Northern pike': { line: [12, 30], lure: [0.375, 2], power: ['MH', 'H'], braid: true, leader: 'wire or heavy fluorocarbon bite leader' },
    'Channel catfish': { line: [10, 30], lure: [0.5, 3], power: ['M', 'MH'], braid: false },
    'Striped bass': { line: [12, 30], lure: [0.5, 2], power: ['MH', 'H'], braid: true },
    'Salmon / steelhead': { line: [8, 25], lure: [0.25, 1], power: ['M', 'MH'], braid: false },
    'Redfish': { line: [10, 20], lure: [0.25, 1], power: ['M', 'MH'], braid: true },
    'Muskellunge': { line: [50, 80], lure: [1, 6], power: ['H', 'XH'], braid: true, leader: 'wire or 100+ lb fluorocarbon leader' },
    'Other': { line: [6, 15], lure: [0.125, 1], power: ['ML', 'MH'], braid: false }
  };
  CL.POWERS = ['UL', 'L', 'ML', 'M', 'MH', 'H', 'XH'];
  CL.timeOfDay = h => h == null || h === '' ? null : (h < 5 || h >= 21) ? 'night' : h < 8 ? 'dawn' : h < 11 ? 'morning' : h < 15 ? 'midday' : h < 18 ? 'afternoon' : 'dusk';
  CL.speciesOptions = (sel, withAll) => { sel.innerHTML = (withAll ? '<option value="">All</option>' : '') + Object.keys(CL.SPECIES).map(s => '<option>' + esc(s) + '</option>').join(''); };
  CL.locationOptions = (sel, opts) => { opts = opts || {}; const ls = Store.list('locations').filter(l => opts.withSim || !l.simulated); sel.innerHTML = (opts.blank ? '<option value="">' + esc(opts.blank) + '</option>' : '') + ls.map(l => '<option value="' + l.id + '">' + esc(l.name) + (l.simulated ? ' (simulated)' : '') + '</option>').join(''); if (!ls.length && !opts.blank) sel.innerHTML = '<option value="">Add a location on the Fishing map first</option>'; };
  CL.isMetric = () => Store.data && Store.data.prefs.units === 'metric';

  /* ---------- auth ---------- */
  let authMode = 'login';
  $$('[data-auth]').forEach(b => b.onclick = () => { authMode = b.dataset.auth; $$('[data-auth]').forEach(x => x.classList.toggle('active', x === b)); $('#dnWrap').hidden = authMode !== 'register'; $('#authSubmit').textContent = authMode === 'login' ? 'Sign in' : 'Create account'; $('#authPass').autocomplete = authMode === 'login' ? 'current-password' : 'new-password'; $('#authErr').textContent = ''; });
  $('#authForm').onsubmit = async e => {
    e.preventDefault(); $('#authErr').textContent = '';
    try { if (authMode === 'login') await Store.login($('#authUser').value, $('#authPass').value); else await Store.register($('#authUser').value, $('#authPass').value, $('#authDisplay').value); $('#authPass').value = ''; }
    catch (err) { $('#authErr').textContent = err.message; }
  };
  $('#logoutBtn').onclick = () => Store.logout();
  $('#guestBtn').onclick = () => {
    Store.guest();
    // A real place (public coordinates) and the tagged simulated dataset; nothing here is presented as an observation.
    const lake = Store.add('locations', { name: 'Lake Harriet (Minneapolis)', lat: 44.9221, lon: -93.3053, type: 'Lake / reservoir', notes: 'Example location for the guest tour.' });
    Store.data.prefs.homeLocation = lake.id; CL.generateDemo(240);
    location.hash = 'home'; Store.load();
  };
  function applyTheme(t) { if (t) document.documentElement.setAttribute('data-theme', t); else document.documentElement.removeAttribute('data-theme'); try { t ? localStorage.setItem('castlab.theme', t) : localStorage.removeItem('castlab.theme'); } catch (e) {} if (CL.current) setTimeout(() => route(true), 0); }
  try { const t = localStorage.getItem('castlab.theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch (e) {}
  $('#themeBtn').onclick = () => { const cur = document.documentElement.getAttribute('data-theme'), dark = cur ? cur === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; applyTheme(dark ? 'light' : 'dark'); };

  Store.on(topic => {
    if (topic === 'auth') {
      const signed = !!Store.user; $('#authView').hidden = signed; $('#appView').hidden = !signed;
      if (signed) { $('#whoami').textContent = Store.user.displayName + ' (@' + Store.user.username + ')'; route(true); }
    }
  });

  /* ---------- routing ---------- */
  function route(force) {
    if (!Store.user) return;
    const [view, sub] = (location.hash.slice(1) || 'home').split('/'), name = $('#v-' + view) ? view : 'home';
    $$('.view').forEach(v => v.hidden = v.id !== 'v-' + name);
    $$('.nav a').forEach(a => a.classList.toggle('active', a.dataset.view === name));
    $('#side').classList.remove('open');
    if (CL.current !== name || force) { CL.current = name; const m = CL.modules[name]; if (m && m.render) { try { m.render(sub); } catch (e) { console.error(e); U.toast('Error drawing ' + name + ': ' + e.message, 'bad'); } } }
    else if (sub && CL.modules[name] && CL.modules[name].tab) CL.modules[name].tab(sub);
  }
  window.addEventListener('hashchange', () => route(true));
  document.addEventListener('click', e => {
    const m = e.target.closest('[data-menu]'); if (m) { $('#side').classList.toggle('open'); return; }
    const png = e.target.closest('[data-png]'); if (png) { U.chartPNG(png.dataset.png, 'castlab-' + png.dataset.png + '.png'); return; }
    const csv = e.target.closest('[data-csv]'); if (csv) { U.chartCSV(csv.dataset.csv, 'castlab-' + csv.dataset.csv + '.csv'); return; }
    const tab = e.target.closest('a[data-tab]'); if (tab) { e.preventDefault(); location.hash = tab.getAttribute('href').slice(1) + '/' + tab.dataset.tab; }
  });
  CL.tabs = function (barSel, paneSel, onShow) {
    const bar = $(barSel); const show = t => { $$('button', bar).forEach(b => b.classList.toggle('active', b.dataset.t === t)); $$(paneSel).forEach(p => p.hidden = p.dataset.t !== t); onShow && onShow(t); };
    bar.onclick = e => { const b = e.target.closest('button[data-t]'); if (b) show(b.dataset.t); }; return show;
  };
  CL.baseChartOpts = function (extra) {
    const ink = U.cssVar('--muted'), grid = U.cssVar('--line');
    const o = { responsive: true, maintainAspectRatio: false, animation: { duration: 250 }, plugins: { legend: { labels: { color: ink, boxWidth: 12, font: { size: 11 } } }, tooltip: { mode: 'nearest', intersect: false } },
      scales: { x: { ticks: { color: ink, font: { size: 11 } }, grid: { color: grid } }, y: { ticks: { color: ink, font: { size: 11 } }, grid: { color: grid } } } };
    return deepMerge(o, extra || {});
  };
  function deepMerge(a, b) { for (const k in b) { if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') deepMerge(a[k], b[k]); else a[k] = b[k]; } return a; }
  CL.alpha = (hex, a) => { hex = hex.replace('#', ''); if (hex.length === 3) hex = hex.split('').map(c => c + c).join(''); const n = parseInt(hex, 16); return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')'; };

  /* ---------- dashboard ---------- */
  CL.modules.dashboard = { render() {
    const isGuest = Store.data.prefs.guest, noReal = !Store.list('sessions').some(s => s.source !== 'simulated');
    $('#dashTour').innerHTML = (isGuest || noReal) ? '<div class="card" style="margin-bottom:14px"><div class="card-h"><h2>Start here</h2>' + (isGuest ? '<span class="badge acc">guest profile</span>' : '') + '</div><p class="small muted" style="margin:0">' + (isGuest ? 'This guest profile holds one real lake and 240 simulated fishing sessions (tagged and labelled as simulated). Every tool below works now.' : 'Each tool below works as soon as you open it. Generate the labelled simulated dataset on the Account page to try the statistics before you have your own sessions.') + '</p><div class="tour">' +
      [['#castcheck', 'Shoreline digital twin', 'Real elevation data for the lake in 2D and 3D. Drag the water-level slider and watch the shoreline move.'],
       ['#lurelab/model', 'Catch prediction model', 'Bayesian logistic regression. On simulated data it shows whether the true coefficients are recovered.'],
       ['#lurelab/validation', 'Prediction accuracy', 'Run forward-in-time validation: each session predicted only from earlier ones.'],
       ['#gearsim', 'Rod physics simulator', 'Large-deflection beam model of three rods. Press "Animate hook-set".'],
       ['#gearsim/opt', 'Design optimisation', 'Search 1,280 rod designs for the best balance of bend, weight and cost.'],
       ['#lurelab/experiments', 'Lure experiment', 'Register a randomised A/B lure test with a sample-size plan, then record blocks.'],
       ['#weather', 'Live weather', 'Hourly Open-Meteo forecast for the lake.'],
       ['#research', 'Methods & exports', 'Methodology, verification checks and a downloadable research report.']].map(t => '<a href="' + t[0] + '"><b>' + t[1] + '</b><span>' + t[2] + '</span></a>').join('') + '</div></div>' : '';
    const ses = Store.list('sessions'), userSes = ses.filter(s => s.source !== 'simulated'), simN = ses.length - userSes.length;
    $('#dashProvenance').innerHTML = simN ? '<p class="note warn"><span class="demo-flag">Simulated</span> This profile also holds ' + simN + ' simulated demo sessions. They are excluded from every figure below, which uses only your ' + userSes.length + ' logged sessions.</p>' : (userSes.length ? '' : '<p class="note">No sessions yet. Add a location on the <a href="#map">Fishing map</a>, then log trips in <a href="#lurelab">LureLab</a>. Every number on this page is computed from what you log; nothing is pre-filled.</p>');
    const succ = userSes.filter(s => s.catches > 0).length, hours = userSes.reduce((a, s) => a + (+s.hours || 0), 0), fish = userSes.reduce((a, s) => a + (+s.catches || 0), 0);
    const bp = userSes.length ? CLStats.betaPosterior(succ, userSes.length, 1, 1) : null, gp = hours ? CLStats.gammaPosterior(fish, hours) : null;
    const obs = Store.list('observations'), bench = Store.list('benchTests'), exps = Store.list('experiments').filter(x => !x.simulated);
    $('#dashKpis').innerHTML = [
      ['Sessions logged', userSes.length, succ + ' with a catch · ' + (userSes.length - succ) + ' blank'],
      ['P(≥1 fish) per session', bp ? pct(bp.mean) : '—', bp ? '95% CrI ' + pct(bp.lo) + '–' + pct(bp.hi) + ' · n=' + bp.n : 'no data'],
      ['Catch rate', gp ? fmt(gp.mean, 2) + '/h' : '—', gp ? '95% CrI ' + fmt(gp.lo, 2) + '–' + fmt(gp.hi, 2) + ' · ' + fmt(hours, 1) + ' h' : 'no data'],
      ['Field data', obs.length + bench.length, obs.length + ' GPS observations · ' + bench.length + ' bench tests']
    ].map(k => '<div class="kpi"><div class="l">' + k[0] + '</div><div class="v">' + k[1] + '</div><div class="s">' + k[2] + '</div></div>').join('');
    // monthly chart
    const months = {}; userSes.forEach(s => { const m = (s.date || '').slice(0, 7); if (!m) return; months[m] = months[m] || [0, 0]; months[m][s.catches > 0 ? 0 : 1]++; });
    const keys = Object.keys(months).sort(), col = U.chartColors();
    U.chart('dashMonthly', $('#dashMonthly'), { type: 'bar', data: { labels: keys, datasets: [{ label: 'Caught fish', data: keys.map(k => months[k][0]), backgroundColor: U.cssVar('--ok') }, { label: 'Blank', data: keys.map(k => months[k][1]), backgroundColor: U.cssVar('--bad') }] }, options: CL.baseChartOpts({ scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: 'sessions', color: U.cssVar('--muted') } } } }) });
    // validation
    const v = CL.modules.lurelab && CL.modules.lurelab.quickValidation ? CL.modules.lurelab.quickValidation('user') : null;
    const fw = Store.list('predictions').filter(p => !p.simulated), fwDone = fw.filter(p => p.outcome != null);
    $('#dashValidation').innerHTML = (v && v.n ? '<div class="stat-line"><span>Prequential predictions scored</span><b>' + v.n + '</b></div><div class="stat-line"><span>Brier score, model</span><b>' + fmt(v.brier, 3) + '</b></div><div class="stat-line"><span>Brier score, running base rate</span><b>' + fmt(v.brierBase, 3) + '</b></div><div class="stat-line"><span>Brier skill score</span><b class="' + (v.bss > 0 ? 'status-ok' : 'status-warn') + '">' + fmt(v.bss, 3) + '</b></div>'
      : '<p class="muted small">Prequential validation needs more sessions than the burn-in (default 15) with both outcomes present. ' + (userSes.length ? 'You have ' + userSes.length + '.' : '') + '</p>') +
      '<div class="stat-line"><span>Forward predictions logged / resolved</span><b>' + fw.length + ' / ' + fwDone.length + '</b></div>' + (fwDone.length ? '<div class="stat-line"><span>Brier score on resolved forward predictions</span><b>' + fmt(CLStats.brier(fwDone.map(p => p.p), fwDone.map(p => p.outcome)), 3) + '</b></div>' : '');
    $('#dashExperiments').innerHTML = exps.length ? '<table><tr><th>Experiment</th><th class="num">Blocks</th><th>Status</th></tr>' + exps.map(x => { const t = Store.list('trials').filter(r => r.experimentId === x.id); return '<tr><td><a href="#lurelab/experiments" data-open-exp="' + x.id + '">' + esc(x.title) + '</a></td><td class="num">' + t.length + ' / ' + (2 * x.blocksPerArm) + '</td><td>' + (t.length >= 2 * x.blocksPerArm ? '<span class="badge ok">complete</span>' : '<span class="badge acc">collecting</span>') + '</td></tr>'; }).join('') + '</table>' : '<p class="muted small">No experiments registered. Design one in LureLab → Experiments.</p>';
    const benchTxt = bench.length ? bench.map(b => '<div class="stat-line"><span>' + esc(b.name || 'Bench test') + ' (n=' + b.data.length + ')</span><b>RMSE ' + fmt(b.rmseBefore * 1000, 1) + ' mm → ' + fmt(b.rmseAfter * 1000, 1) + ' mm calibrated</b></div>').join('') : '<p class="muted small">No rod bench tests saved yet (GearMatch physics → Bench test).</p>';
    const cc = CL.modules.castcheck && CL.modules.castcheck.summary ? CL.modules.castcheck.summary() : null;
    $('#dashSimVsReal').innerHTML = '<h4>Rod deflection</h4>' + benchTxt + '<h4 style="margin-top:10px">Shoreline</h4>' + (cc && cc.n ? '<div class="stat-line"><span>Waterline GPS points vs simulated shoreline (n=' + cc.n + ')</span><b>RMSE ' + fmt(cc.rmse, 1) + ' m</b></div>' : '<p class="muted small">' + (obs.length ? 'Load the terrain for a location in CastCheck to compare its ' + obs.filter(o => o.type === 'waterline').length + ' waterline observations.' : 'No GPS waterline observations recorded yet.') + '</p>');
    const recent = userSes.slice().sort((a, b) => (b.date + b.hour).localeCompare(a.date + a.hour)).slice(0, 8);
    $('#dashRecent').innerHTML = recent.length ? '<div class="tablewrap"><table><tr><th>Date</th><th>Location</th><th>Species</th><th>Lure</th><th class="num">Hours</th><th class="num">Fish</th></tr>' + recent.map(s => '<tr><td>' + esc(s.date) + '</td><td>' + esc(Store.locationName(s.locationId)) + '</td><td>' + esc(s.species) + '</td><td>' + esc(s.lure) + '</td><td class="num">' + fmt(s.hours, 2) + '</td><td class="num">' + s.catches + '</td></tr>').join('') + '</table></div>' : '<p class="muted small">Nothing logged yet.</p>';
  } };
  document.addEventListener('click', e => { const a = e.target.closest('[data-open-exp]'); if (a) CL.openExperiment = a.dataset.openExp; });

  /* ---------- fishing map ---------- */
  let map = null, layer = null;
  CL.modules.map = { render() {
    if (!map) {
      map = L.map('map', { worldCopyJump: true }).setView([39.5, -96], 4);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }).addTo(map);
      map.on('click', e => { $('#locLat').value = e.latlng.lat.toFixed(6); $('#locLon').value = e.latlng.lng.toFixed(6); if (CL.mapClick) CL.mapClick(e, map); });
      CL._map = map; map.on('moveend', () => CL.mapMoved && CL.mapMoved(map));
      layer = L.layerGroup().addTo(map);
    }
    setTimeout(() => map.invalidateSize(), 50);
    layer.clearLayers(); const pts = [];
    const ses = Store.list('sessions').filter(s => s.source !== 'simulated');
    Store.list('locations').filter(l => !l.simulated && isFinite(l.lat)).forEach(l => {
      const ls = ses.filter(s => s.locationId === l.id), succ = ls.filter(s => s.catches > 0).length;
      L.circleMarker([l.lat, l.lon], { radius: 9, color: U.cssVar('--accent'), weight: 2, fillOpacity: .35 }).bindPopup('<b>' + esc(l.name) + '</b><br>' + esc(l.type || '') + '<br>' + ls.length + ' sessions, ' + succ + ' with a catch<br><a href="#weather" onclick="CL.pickLoc=\'' + l.id + '\'">Weather</a> · <a href="#castcheck" onclick="CL.pickLoc=\'' + l.id + '\'">Shoreline twin</a>').addTo(layer); pts.push([l.lat, l.lon]);
    });
    ses.forEach(s => { const lat = s.lat != null ? s.lat : (Store.location(s.locationId) || {}).lat, lon = s.lon != null ? s.lon : (Store.location(s.locationId) || {}).lon; if (!isFinite(lat)) return; const j = (h => ((h % 1000) / 1000 - 0.5) * 0.0006)(parseInt(s.id.slice(-6), 36) || 0); L.circleMarker([lat + j, lon - j], { radius: 4, color: s.catches > 0 ? U.cssVar('--ok') : U.cssVar('--bad'), weight: 1, fillOpacity: .8 }).bindPopup(esc(s.date) + ' · ' + esc(s.lure) + '<br>' + s.catches + ' fish in ' + s.hours + ' h').addTo(layer); });
    Store.list('observations').forEach(o => { L.circleMarker([o.lat, o.lon], { radius: 4, color: U.cssVar('--c3'), weight: 1, fillOpacity: .9 }).bindPopup('Field observation: ' + esc(o.type) + (o.value != null ? ' = ' + o.value : '') + '<br>' + esc(o.time) + '<br>GPS ±' + fmt(o.acc, 0) + ' m').addTo(layer); if (o.acc) L.circle([o.lat, o.lon], { radius: o.acc, color: U.cssVar('--c3'), weight: 1, fillOpacity: .05 }).addTo(layer); });
    if (pts.length && !map._fitted) { map.fitBounds(pts, { maxZoom: 13, padding: [30, 30] }); map._fitted = true; }
    const locs = Store.list('locations').filter(l => !l.simulated);
    $('#locTable').innerHTML = locs.length ? '<div class="tablewrap"><table><tr><th>Name</th><th>Type</th><th class="num">Lat</th><th class="num">Lon</th><th class="num">Sessions</th><th class="num">P(≥1 fish), 95% CrI</th><th></th></tr>' + locs.map(l => { const ls = ses.filter(s => s.locationId === l.id), succ = ls.filter(s => s.catches > 0).length, bp = ls.length ? CLStats.betaPosterior(succ, ls.length) : null; return '<tr><td>' + esc(l.name) + '</td><td>' + esc(l.type) + '</td><td class="num">' + fmt(l.lat, 5) + '</td><td class="num">' + fmt(l.lon, 5) + '</td><td class="num">' + ls.length + '</td><td class="num">' + (bp ? pct(bp.mean) + ' (' + pct(bp.lo) + '–' + pct(bp.hi) + ')' : '—') + '</td><td><div class="btns"><button class="btn sm" data-zoom="' + l.id + '">Zoom</button><button class="btn sm danger" data-del-loc="' + l.id + '">Delete</button></div></td></tr>'; }).join('') + '</table></div>' : '<p class="muted small">No locations yet. Click the map or use your GPS.</p>';
  } };
  $('#locForm').onsubmit = e => { e.preventDefault(); const lat = +$('#locLat').value, lon = +$('#locLon').value; if (!(Math.abs(lat) <= 90 && Math.abs(lon) <= 180)) return U.toast('Latitude/longitude out of range.'); Store.add('locations', { name: $('#locName').value.trim(), lat, lon, type: $('#locType').value, notes: $('#locNotes').value }); if (!Store.data.prefs.homeLocation) { Store.data.prefs.homeLocation = Store.list('locations').slice(-1)[0].id; Store.save(); } e.target.reset(); U.toast('Location saved.'); map._fitted = false; CL.modules.map.render(); };
  $('#locGps').onclick = () => { if (!navigator.geolocation) return U.toast('No geolocation in this browser.'); navigator.geolocation.getCurrentPosition(p => { $('#locLat').value = p.coords.latitude.toFixed(6); $('#locLon').value = p.coords.longitude.toFixed(6); map && map.setView([p.coords.latitude, p.coords.longitude], 14); U.toast('GPS fix ±' + Math.round(p.coords.accuracy) + ' m'); }, err => U.toast('GPS: ' + err.message), { enableHighAccuracy: true, timeout: 15000 }); };
  $('#locTable').onclick = e => {
    const z = e.target.closest('[data-zoom]'); if (z) { const l = Store.location(z.dataset.zoom); map.setView([l.lat, l.lon], 14); window.scrollTo({ top: 0, behavior: 'smooth' }); }
    const d = e.target.closest('[data-del-loc]'); if (d) { const n = Store.list('sessions').filter(s => s.locationId === d.dataset.delLoc).length; if (confirm('Delete this location?' + (n ? ' ' + n + ' sessions reference it and will show "—" as location.' : ''))) { Store.remove('locations', d.dataset.delLoc); CL.modules.map.render(); } }
  };

  /* ---------- weather ---------- */
  CL.modules.weather = { render() {
    const sel = $('#wxLoc'), prev = CL.pickLoc || sel.value || Store.data.prefs.homeLocation; CL.locationOptions(sel); if (prev && Store.location(prev)) sel.value = prev; CL.pickLoc = null;
    if (sel.value) loadWeather();
  } };
  async function loadWeather() {
    const l = Store.location($('#wxLoc').value); if (!l) return; $('#wxNow').innerHTML = '<p class="muted small">Loading Open-Meteo…</p>';
    try {
      const w = await U.fetchWeather(l.lat, l.lon), h = w.hourly, now = w.index, lab = h.time.map(t => t.slice(5, 13).replace('T', ' ') + 'h');
      $('#wxNow').innerHTML = '<div class="grid g4" style="margin-top:10px">' + [['Now', U.WMO[w.code] || 'code ' + w.code, w.timezone], ['Air temperature', fmt(w.tempC, 1) + ' °C', fmt(w.tempC * 9 / 5 + 32, 0) + ' °F'], ['Pressure', fmt(w.pressureHpa, 1) + ' hPa', '3-h trend: ' + (w.pressureTrend || '—')], ['Wind', fmt(w.windKph, 0) + ' km/h', 'cloud ' + fmt(w.cloudPct, 0) + '% · precip ' + fmt(w.precipMm, 1) + ' mm']].map(k => '<div class="kpi"><div class="l">' + k[0] + '</div><div class="v" style="font-size:19px">' + k[1] + '</div><div class="s">' + k[2] + '</div></div>').join('') + '</div><p class="small muted" style="margin-top:8px">Source: Open-Meteo forecast API (model output; hours before "now" are the model\'s best analysis, not station observations). Vertical line = current hour.</p>';
      const nowLine = { id: 'now', afterDraw(c) { const x = c.scales.x.getPixelForValue(now), a = c.chartArea; c.ctx.save(); c.ctx.strokeStyle = U.cssVar('--muted'); c.ctx.setLineDash([4, 4]); c.ctx.beginPath(); c.ctx.moveTo(x, a.top); c.ctx.lineTo(x, a.bottom); c.ctx.stroke(); c.ctx.restore(); } };
      const col = U.chartColors();
      U.chart('wxTemp', $('#wxTemp'), { type: 'bar', data: { labels: lab, datasets: [{ type: 'line', label: 'Air temperature (°C)', data: h.temperature_2m, borderColor: col[1], pointRadius: 0, borderWidth: 2, yAxisID: 'y' }, { label: 'Precipitation (mm/h)', data: h.precipitation, backgroundColor: CL.alpha(U.cssVar('--water'), .6), yAxisID: 'y1' }] }, options: CL.baseChartOpts({ scales: { x: { ticks: { maxTicksLimit: 10 } }, y1: { position: 'right', grid: { display: false }, ticks: { color: U.cssVar('--muted') }, beginAtZero: true } } }), plugins: [nowLine] });
      U.chart('wxPress', $('#wxPress'), { type: 'line', data: { labels: lab, datasets: [{ label: 'Surface pressure (hPa)', data: h.surface_pressure, borderColor: col[2], pointRadius: 0, borderWidth: 2, yAxisID: 'y' }, { label: 'Wind (km/h)', data: h.wind_speed_10m, borderColor: col[0], pointRadius: 0, borderWidth: 1.5, yAxisID: 'y1' }] }, options: CL.baseChartOpts({ scales: { x: { ticks: { maxTicksLimit: 10 } }, y1: { position: 'right', grid: { display: false }, ticks: { color: U.cssVar('--muted') }, beginAtZero: true } } }), plugins: [nowLine] });
    } catch (err) { $('#wxNow').innerHTML = '<p class="note bad">Could not load weather: ' + esc(err.message) + '. Open-Meteo needs an internet connection; you can still type conditions manually when logging a session.</p>'; }
  }
  $('#wxGo').onclick = loadWeather; $('#wxLoc').onchange = loadWeather;

  /* ---------- BaitRadar ---------- */
  CL.lureStats = function (name) { const s = Store.list('sessions').filter(x => x.source !== 'simulated' && (x.lure || '').toLowerCase() === (name || '').toLowerCase()); const k = s.filter(x => x.catches > 0).length; return { n: s.length, succ: k, fish: s.reduce((a, x) => a + (+x.catches || 0), 0), hours: s.reduce((a, x) => a + (+x.hours || 0), 0), bp: s.length ? CLStats.betaPosterior(k, s.length) : null }; };
  CL.modules.baitradar = { render() {
    if (CL.baitRec) CL.baitRec();
    const inv = Store.list('inventory'), q = ($('#brSearch').value || '').toLowerCase(), cur = CL.isMetric() ? '' : '$';
    const low = inv.filter(i => (+i.qty) <= (+i.reorder || 0)), value = inv.reduce((a, i) => a + (+i.qty || 0) * (+i.cost || 0), 0);
    $('#brKpis').innerHTML = [['Items', inv.length, inv.reduce((a, i) => a + (+i.qty || 0), 0) + ' units in stock'], ['Low stock', low.length, low.slice(0, 3).map(i => esc(i.name)).join(', ') || 'none'], ['Inventory value', fmt(value, 2), 'sum of quantity × unit cost'], ['Fished in LureLab', inv.filter(i => CL.lureStats(i.name).n).length, 'items with at least one logged session']].map(k => '<div class="kpi"><div class="l">' + k[0] + '</div><div class="v">' + k[1] + '</div><div class="s">' + k[2] + '</div></div>').join('');
    const rows = inv.filter(i => !q || (i.name + ' ' + i.category + ' ' + (i.color || '') + ' ' + (i.where || '')).toLowerCase().includes(q));
    $('#brTable').innerHTML = rows.length ? '<div class="tablewrap"><table><tr><th>Item</th><th>Category</th><th class="num">Weight</th><th class="num">Qty</th><th>Stock</th><th class="num">Sessions</th><th class="num">P(≥1 fish), 95% CrI</th><th></th></tr>' + rows.map(i => { const st = CL.lureStats(i.name), w = +i.weightG; return '<tr><td><b>' + esc(i.name) + '</b>' + (i.color ? '<br><span class="small muted">' + esc(i.color) + '</span>' : '') + (i.where ? '<br><span class="small muted">' + esc(i.where) + '</span>' : '') + '</td><td>' + esc(i.category) + '</td><td class="num">' + (w ? (CL.isMetric() ? fmt(w, 1) + ' g' : fmt(U.gToOz(w), 3) + ' oz') : '—') + '</td><td class="num">' + i.qty + '</td><td>' + ((+i.qty) <= 0 ? '<span class="badge bad">out</span>' : (+i.qty) <= (+i.reorder || 0) ? '<span class="badge warn">reorder</span>' : '<span class="badge ok">ok</span>') + '</td><td class="num">' + st.n + '</td><td class="num">' + (st.bp ? pct(st.bp.mean) + ' (' + pct(st.bp.lo) + '–' + pct(st.bp.hi) + ')' : '—') + '</td><td><div class="btns"><button class="btn sm" data-br-inc="' + i.id + '">+1</button><button class="btn sm" data-br-dec="' + i.id + '">−1</button><button class="btn sm" data-br-edit="' + i.id + '">Edit</button><button class="btn sm danger" data-br-del="' + i.id + '">×</button></div></td></tr>'; }).join('') + '</table></div><p class="small muted" style="margin-top:6px">Catch probability comes from your LureLab sessions where the lure name matches this item exactly. Intervals are wide until an item has been fished many times; compare items properly with a LureLab experiment.</p>' : '<p class="muted small">' + (inv.length ? 'No items match the filter.' : 'Inventory is empty. Add your first item.') + '</p>';
  } };
  function brReset() { $('#brForm').reset(); $('#brId').value = ''; $('#brFormTitle').textContent = 'Add item'; }
  $('#brForm').onsubmit = e => { e.preventDefault(); const o = { name: $('#brName').value.trim(), category: $('#brCat').value, color: $('#brColor').value.trim(), weightG: $('#brWeight').value === '' ? null : +$('#brWeight').value, qty: +$('#brQty').value, reorder: +$('#brReorder').value || 0, cost: $('#brCost').value === '' ? null : +$('#brCost').value, where: $('#brWhere').value.trim() }; if ($('#brId').value) Store.update('inventory', $('#brId').value, o); else Store.add('inventory', o); brReset(); CL.modules.baitradar.render(); U.toast('Saved.'); };
  $('#brCancel').onclick = brReset; $('#brSearch').oninput = () => CL.modules.baitradar.render();
  $('#brTable').onclick = e => {
    const b = e.target.closest('button'); if (!b) return; const id = b.dataset.brInc || b.dataset.brDec || b.dataset.brEdit || b.dataset.brDel, it = Store.get('inventory', id); if (!it) return;
    if (b.dataset.brInc) Store.update('inventory', id, { qty: (+it.qty || 0) + 1 });
    if (b.dataset.brDec) Store.update('inventory', id, { qty: Math.max(0, (+it.qty || 0) - 1) });
    if (b.dataset.brDel && confirm('Delete ' + it.name + '?')) Store.remove('inventory', id);
    if (b.dataset.brEdit) { $('#brId').value = id; $('#brName').value = it.name; $('#brCat').value = it.category; $('#brColor').value = it.color || ''; $('#brWeight').value = it.weightG == null ? '' : it.weightG; $('#brQty').value = it.qty; $('#brReorder').value = it.reorder; $('#brCost').value = it.cost == null ? '' : it.cost; $('#brWhere').value = it.where || ''; $('#brFormTitle').textContent = 'Edit item'; $('#brMore').open = true; window.scrollTo({ top: 0, behavior: 'smooth' }); }
    CL.modules.baitradar.render();
  };
  $('#brCsv').onclick = () => U.download('castlab-inventory.csv', U.toCSV(Store.list('inventory').map(i => Object.assign({}, i, { sessions: CL.lureStats(i.name).n, sessionsWithCatch: CL.lureStats(i.name).succ })), ['id', 'name', 'category', 'color', 'weightG', 'qty', 'reorder', 'cost', 'where', 'sessions', 'sessionsWithCatch']), 'text/csv');

  /* ---------- GearMatch recommendations ---------- */
  CL.modules.gearmatch = { render() { if (!$('#gmSpecies').options.length) CL.speciesOptions($('#gmSpecies')); renderRods(); recommend(); if (CL.gearLocal) CL.gearLocal(); } };
  CL.recommendGear = () => recommend();
  function recommend() {
    const sp = $('#gmSpecies').value, t = $('#gmTech').value, w = $('#gmWater').value, S = CL.SPECIES[sp]; if (!S) return;
    const lerp = (r, f) => r[0] + (r[1] - r[0]) * f, f = t === 'finesse' ? [0, 0.4] : t === 'power' ? [0.6, 1] : [0.25, 0.75];
    let line = [lerp(S.line, f[0]), lerp(S.line, f[1])]; if (w === 'cover') line = line.map(v => v * 1.25);
    const lure = [lerp(S.lure, f[0]), lerp(S.lure, f[1])];
    const pi0 = CL.POWERS.indexOf(S.power[0]), pi1 = CL.POWERS.indexOf(S.power[1]), pAdj = t === 'finesse' ? 0 : t === 'power' || w === 'cover' ? 1 : 0.5;
    const power = CL.POWERS[Math.round(pi0 + (pi1 - pi0) * pAdj)];
    const lineType = (S.braid && (t === 'power' || w === 'cover')) ? 'Braid main line (high strength per diameter, low stretch for hook-sets in cover)' + (S.leader ? ' with ' + S.leader : '') : t === 'finesse' ? 'Fluorocarbon or monofilament (low visibility, some stretch)' : 'Monofilament or fluorocarbon' + (S.leader ? '; use a ' + S.leader : '');
    const lineHi = Math.round(line[1]), drag = [0.25 * lineHi, 0.33 * lineHi], action = t === 'bait' ? 'moderate (forgiving with live bait and treble hooks)' : t === 'power' ? 'fast (quick hook-set)' : 'moderate-fast';
    const inv = Store.list('inventory').filter(i => i.weightG && U.gToOz(i.weightG) >= lure[0] * 0.8 && U.gToOz(i.weightG) <= lure[1] * 1.2 && !/line|terminal/i.test(i.category));
    const rods = Store.list('rods').map(r => { const ok = CL.POWERS.indexOf(r.power) >= pi0 - 1 && CL.POWERS.indexOf(r.power) <= pi1 + 1 && r.lmax >= line[0] && r.lmin <= line[1] && r.wmax >= lure[0] && r.wmin <= lure[1]; return { r, ok }; });
    const ozTxt = v => CL.isMetric() ? fmt(U.ozToG(v), 1) + ' g' : (v < 0.1 ? fmt(v, 3) : fmt(v, 2)) + ' oz', lbTxt = v => CL.isMetric() ? fmt(U.lbToN(v), 0) + ' N' : fmt(v, 0) + ' lb';
    $('#gmOut').innerHTML = '<div class="card-h"><h2>' + esc(sp) + ' · ' + esc($('#gmTech').selectedOptions[0].text) + '</h2><span class="badge">guidance</span></div>' +
      '<div class="grid g2"><div><div class="stat-line"><span>Rod power</span><b>' + power + '</b></div><div class="stat-line"><span>Rod action</span><b>' + action + '</b></div><div class="stat-line"><span>Line strength</span><b>' + lbTxt(line[0]) + ' – ' + lbTxt(line[1]) + '</b></div><div class="stat-line"><span>Lure weight</span><b>' + ozTxt(lure[0]) + ' – ' + ozTxt(lure[1]) + '</b></div><div class="stat-line"><span>Drag setting (25–33% of line)</span><b>' + lbTxt(drag[0]) + ' – ' + lbTxt(drag[1]) + '</b></div></div>' +
      '<div><p class="small"><b>Line:</b> ' + esc(lineType) + '.</p>' + (w === 'current' ? '<p class="small">In strong current go to the heavier end of the lure range to hold bottom; line drag in current grows with line diameter, which favours thinner braid.</p>' : '') + '<p class="small muted">These ranges summarise common angling practice for the species. They are starting points, not tested optima. Use the physics simulator to check stress and deflection, and LureLab to test lures on your own water.</p></div></div>' +
      '<h3 style="margin-top:12px">From your BaitRadar inventory</h3>' + (inv.length ? inv.map(i => { const st = CL.lureStats(i.name); return '<span class="pill">' + esc(i.name) + ' · ' + ozTxt(U.gToOz(i.weightG)) + ' · qty ' + i.qty + (st.n ? ' · caught in ' + st.succ + '/' + st.n + ' sessions' : '') + '</span>'; }).join('') : '<p class="small muted">No inventory items with a recorded weight in this range.</p>') +
      '<h3 style="margin-top:12px">Your rods</h3>' + (rods.length ? rods.map(x => '<span class="pill">' + (x.ok ? '<span class="status-ok">✓</span>' : '<span class="status-warn">≠</span>') + ' ' + esc(x.r.name) + ' (' + x.r.power + ', ' + x.r.lmin + '–' + x.r.lmax + ' lb, ' + x.r.wmin + '–' + x.r.wmax + ' oz)</span>').join('') : '<p class="small muted">No rods saved.</p>') +
      '<div id="gmLocalRec"></div><div class="btns" style="margin-top:12px"><button class="btn primary" id="gmToSim">Check this setup in the physics simulator</button></div>';
    if (CL.gearLocalRec) CL.gearLocalRec(sp);
    $('#gmToSim').onclick = () => { CL.pendingSetup = { lineLb: lineHi, dragLb: +(0.3 * lineHi).toFixed(1), lureOz: +((lure[0] + lure[1]) / 2).toFixed(3), rlmin: Math.round(line[0]), rlmax: Math.round(line[1]), rwmin: +lure[0].toFixed(3), rwmax: +lure[1].toFixed(3), lineType: /Braid/.test(lineType) ? 'braid' : t === 'finesse' ? 'fluoro' : 'mono', power }; location.hash = 'gearsim/compat'; };
  }
  function renderRods() { const rods = Store.list('rods'); $('#rodList').innerHTML = rods.map(r => '<div class="stat-line"><span>' + esc(r.name) + ' <span class="muted small">' + r.lenFt + ' ft ' + r.power + '</span></span><span class="btns"><button class="btn sm danger" data-del-rod="' + r.id + '">×</button></span></div>').join(''); }
  $('#gmGo').onclick = recommend; ['#gmSpecies', '#gmTech', '#gmWater'].forEach(s => $(s).onchange = recommend);
  $('#rodForm').onsubmit = e => { e.preventDefault(); Store.add('rods', { name: $('#rodName').value.trim(), lenFt: +$('#rodLen').value, power: $('#rodPower').value, lmin: +$('#rodLmin').value, lmax: +$('#rodLmax').value, wmin: +$('#rodWmin').value, wmax: +$('#rodWmax').value }); $('#rodName').value = ''; renderRods(); recommend(); };
  $('#rodList').onclick = e => { const b = e.target.closest('[data-del-rod]'); if (b) { Store.remove('rods', b.dataset.delRod); renderRods(); recommend(); } };

  /* ---------- simulated demo data (clearly tagged) ---------- */
  CL.DEMO_TRUTH = { intercept: -0.6, lure: { 'Spinnerbait': 0, 'Soft plastic worm': 0.7, 'Crankbait': 0.25, 'Topwater popper': -0.45 }, timeOfDay: { morning: 0, dawn: 0.55, midday: -0.5, afternoon: -0.15, dusk: 0.45, night: -0.3 }, pressureTrend: { steady: 0, falling: 0.35, rising: -0.3 }, logHours: 0.6 };
  CL.generateDemo = function (n, seed) {
    const r = CLStats.rng(seed || 20261110), T = CL.DEMO_TRUTH, pick = a => a[Math.floor(r() * a.length)];
    let loc = Store.list('locations').find(l => l.simulated); if (!loc) loc = Store.add('locations', { name: 'Simulated demo water', simulated: true, type: 'Simulated', lat: null, lon: null });
    const lures = Object.keys(T.lure), tods = [5, 6, 7, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22], trends = ['steady', 'steady', 'falling', 'rising'], start = Date.parse('2025-04-01'), out = [];
    for (let i = 0; i < n; i++) {
      const hour = pick(tods), lure = pick(lures), trend = pick(trends), hours = Math.round((0.75 + r() * 3.5) * 4) / 4, tod = CL.timeOfDay(hour);
      const eta = T.intercept + T.lure[lure] + T.timeOfDay[tod] + T.pressureTrend[trend] + T.logHours * Math.log(hours / 2), p = CLStats.sigmoid(eta);
      let catches = 0; if (r() < p) { catches = 1; let lam = 0.8 * hours * 0.5, L = Math.exp(-lam), k = 0, q = 1; do { k++; q *= r(); } while (q > L); catches += k - 1; }
      const d = new Date(start + Math.floor(i * (540 / n) + r() * 2) * 864e5).toISOString().slice(0, 10);
      out.push({ id: Store.uid(), created: new Date().toISOString(), source: 'simulated', date: d, hour, timeOfDay: tod, locationId: loc.id, species: pick(['Largemouth bass', 'Smallmouth bass']), lure, hours, catches, weather: { pressureTrend: trend, source: 'simulated' }, trueP: p });
    }
    Store.data.sessions.push(...out); Store.data.prefs.demoTruth = T; Store.save('sessions'); return out.length;
  };

  /* ---------- research exports ---------- */
  CL.modules.research = { render() {
    const sets = [['Sessions', 'sessions', () => Store.list('sessions').map(U.flatSession)], ['Locations', 'locations', () => Store.list('locations')], ['Inventory', 'inventory', () => Store.list('inventory')], ['Experiments', 'experiments', () => Store.list('experiments').map(x => Object.assign({}, x, { order: (x.order || []).join('') }))], ['Experiment trials', 'trials', () => Store.list('trials')], ['Forward predictions', 'predictions', () => Store.list('predictions')], ['Field observations', 'observations', () => Store.list('observations')], ['Bench tests', 'benchTests', () => Store.list('benchTests').flatMap(b => b.data.map(d => ({ test: b.name, testId: b.id, cfg: JSON.stringify(b.cfg), massKg: d.massKg, measuredDropM: d.dropM, predictedDropM: d.pred, Escale: b.scale })))], ['Rods', 'rods', () => Store.list('rods')], ['Catch log', 'catches', () => Store.list('catches').map(c => Object.assign({}, c, { location: Store.locationName(c.locationId) }))]];
    $('#rsDatasets').innerHTML = sets.map((s, i) => '<button class="btn sm" data-set="' + i + '">' + s[0] + ' (' + Store.list(s[1]).length + ') CSV</button>').join('') + '<button class="btn sm primary" id="rsJson">Everything (JSON)</button>';
    $('#rsDatasets').onclick = e => { const b = e.target.closest('[data-set]'); if (b) { const s = sets[+b.dataset.set]; U.download('castlab-' + s[1] + '.csv', U.toCSV(s[2]()), 'text/csv'); } if (e.target.id === 'rsJson') exportAll(); };
    $('#rsMethods').innerHTML = CL.methodsOverview();
    $('#rsVerify').innerHTML = CL.selfTest().map(t => '<div class="stat-line"><span>' + esc(t.name) + '</span><b class="' + (t.ok ? 'status-ok' : 'status-fail') + '">' + (t.ok ? 'pass' : 'FAIL') + ' · ' + esc(t.detail) + '</b></div>').join('') + '<p class="small muted" style="margin-top:6px">The full 56-check suite is in <code>castlab/tests/run.js</code> (run with Node).</p>';
  } };
  CL.selfTest = function () {
    const out = [], chk = (name, got, want, tol) => out.push({ name, ok: Math.abs(got - want) <= tol, detail: (+got).toPrecision(5) + ' vs ' + want });
    chk('Beta(2,3) CDF at 0.5 (exact 11/16)', CLStats.betaCdf(0.5, 2, 3), 0.6875, 1e-9);
    const EI = 100e9 * Math.PI / 64 * 1e-8, r = CLPhysics.solveRod({ E: 100e9, length: 1, buttOD: 0.01, tipOD: 0.01, wallRatio: 1 }, 2 * EI, 90, 0, { N: 200 });
    chk('Elastica tip deflection, α=2 (Bisshopp–Drucker 0.49346)', -r.dy, 0.49346, 3e-4);
    const w = 30, z = new Float32Array(w * w); for (let j = 0; j < w; j++) for (let i = 0; i < w; i++) z[j * w + i] = i * 5 * Math.tan(Math.PI / 18);
    chk('Horn slope on a 10° plane', CLTerrain.slopeDeg(z, w, w, 5)[15 * w + 15], 10, 1e-3);
    return out;
  };
  function exportAll() { U.download('castlab-' + Store.user.username + '-' + U.today() + '.json', JSON.stringify(Store.exportAll(), null, 1), 'application/json'); }
  CL.methodsOverview = function () {
    return '<h2>How CastLab works</h2>' +
      '<p>CastLab keeps all records in one profile so that the field tools and the research tools see the same data: a location saved on the map is the place weather is fetched for, the place sessions are logged at and the place the shoreline twin is built for. Sessions reference BaitRadar items by name, GearMatch recommendations look at your inventory and rods, and the physics simulator receives the setups GearMatch suggests.</p>' +
      '<h3>Data integrity rules</h3><ul><li>Nothing is pre-filled. Every statistic is computed from records you create or import, at the moment you view it.</li><li>Simulated data exist only if you generate them (Account → Simulated demo dataset). They carry <code>source: "simulated"</code>, are labelled wherever they appear and are never pooled with your observations.</li><li>Live data come from named public services: OpenStreetMap (base map), Open-Meteo (weather model output), USGS Water Services (gage height), and AWS Terrain Tiles (elevation). Each is labelled where it is used, along with its known limitations.</li><li>Uncertainty is shown with every estimate as an interval and a sample size. Model accuracy is reported only from out-of-sample predictions.</li></ul>' +
      '<h3>Module methods</h3><p>Full methodology, equations and assumptions are on each module\'s Methodology tab: <a href="#lurelab/methods">LureLab</a>, <a href="#castcheck/methods">CastCheck twin</a>, <a href="#gearsim/methods">GearMatch physics</a>.</p>' +
      '<h3>Reproducibility</h3><p>Monte Carlo steps use a seeded pseudo-random generator (mulberry32) with the seed shown next to the result, so the same data give the same numbers. Experiment lure orders are generated from a seed stored with the experiment at registration. Exports include every raw record, so analyses can be repeated in R or Python.</p>';
  };

  /* ---------- research report ---------- */
  function reportParts() {
    const ses = Store.list('sessions').filter(s => s.source !== 'simulated'), parts = [];
    const succ = ses.filter(s => s.catches > 0).length, hours = ses.reduce((a, s) => a + (+s.hours || 0), 0), fish = ses.reduce((a, s) => a + (+s.catches || 0), 0);
    parts.push(['Dataset', [ses.length + ' fishing sessions logged by the user (' + succ + ' with at least one fish, ' + (ses.length - succ) + ' without), ' + fmt(hours, 1) + ' hours of effort, ' + fish + ' fish, at ' + new Set(ses.map(s => s.locationId)).size + ' locations, ' + (ses.length ? 'from ' + ses.map(s => s.date).sort()[0] + ' to ' + ses.map(s => s.date).sort().slice(-1)[0] : 'no dates') + '.', Store.list('observations').length + ' GPS field observations and ' + Store.list('benchTests').length + ' rod bench tests.', 'Simulated demo records are excluded from this report.']]);
    const L = CL.modules.lurelab && CL.modules.lurelab.reportSection ? CL.modules.lurelab.reportSection() : null; if (L) parts.push(...L);
    const G = CL.modules.gearsim && CL.modules.gearsim.reportSection ? CL.modules.gearsim.reportSection() : null; if (G) parts.push(...G);
    const C = CL.modules.castcheck && CL.modules.castcheck.reportSection ? CL.modules.castcheck.reportSection() : null; if (C) parts.push(...C);
    return parts;
  }
  $('#rsReport').onclick = () => {
    const parts = reportParts(), title = 'CastLab research summary — ' + Store.user.displayName + ' — ' + U.today();
    const html = '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + esc(title) + '</title><style>body{font:15px/1.6 Georgia,serif;max-width:820px;margin:32px auto;padding:0 16px;color:#1b1b1b;background:#fff}h1{font-size:24px}h2{font-size:18px;border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:28px}table{border-collapse:collapse;width:100%;font:13px system-ui,sans-serif;margin:8px 0}th,td{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left}code,pre{font:12.5px ui-monospace,Menlo,monospace;background:#f3f3f3}pre{padding:8px;overflow:auto}.small{font-size:12px;color:#555}</style></head><body><h1>' + esc(title) + '</h1><p class="small">Generated by CastLab from the data stored in this profile at ' + new Date().toISOString() + '. All figures are computed from user-logged records; no values are imputed or invented.</p>' +
      parts.map(p => '<h2>' + esc(p[0]) + '</h2>' + p[1].map(x => x.startsWith('<') ? x : '<p>' + esc(x) + '</p>').join('')).join('') + '</body></html>';
    U.download('castlab-research-summary-' + U.today() + '.html', html, 'text/html');
  };
  $('#rsMd').onclick = () => { const parts = reportParts(); const strip = s => s.replace(/<tr>/g, '\n').replace(/<\/t[dh]>/g, ' | ').replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'); U.download('castlab-research-summary-' + U.today() + '.md', '# CastLab research summary\n\n_Generated ' + new Date().toISOString() + ' from user-logged data._\n\n' + parts.map(p => '## ' + p[0] + '\n\n' + p[1].map(strip).join('\n\n')).join('\n\n') + '\n', 'text/markdown'); };

  /* ---------- account ---------- */
  CL.modules.account = { render() {
    const u = Store.user; $('#acProfile').innerHTML = '<div class="stat-line"><span>Display name</span><b>' + esc(u.displayName) + '</b></div><div class="stat-line"><span>Username</span><b>@' + esc(u.username) + '</b></div><div class="stat-line"><span>Created</span><b>' + esc(u.created.slice(0, 10)) + '</b></div><div class="stat-line"><span>Storage</span><b>this browser only</b></div>';
    $('#acUnits').value = Store.data.prefs.units; CL.locationOptions($('#acHome'), { blank: '— none —' }); $('#acHome').value = Store.data.prefs.homeLocation || '';
    $('#acCounts').innerHTML = '<table><tr><th>Collection</th><th class="num">Records</th><th class="num">of which simulated</th></tr>' + Store.COLLECTIONS.map(c => '<tr><td>' + c + '</td><td class="num">' + Store.list(c).length + '</td><td class="num">' + Store.list(c).filter(x => x.simulated || x.source === 'simulated').length + '</td></tr>').join('') + '</table>';
  } };
  $('#acUnits').onchange = e => { Store.data.prefs.units = e.target.value; Store.save(); };
  $('#acHome').onchange = e => { Store.data.prefs.homeLocation = e.target.value || null; Store.save(); };
  $('#acExport').onclick = exportAll; $('#acImportBtn').onclick = () => $('#acImport').click();
  $('#acImport').onchange = e => { const f = e.target.files[0]; if (!f) return; const rd = new FileReader(); rd.onload = () => { try { Store.importAll(JSON.parse(rd.result), $('#acReplace').checked ? 'replace' : 'merge'); U.toast('Imported.'); CL.modules.account.render(); } catch (err) { U.toast('Import failed: ' + err.message, 'bad'); } }; rd.readAsText(f); e.target.value = ''; };
  $('#acDemo').onclick = () => { if (Store.list('sessions').some(s => s.source === 'simulated')) return U.toast('Simulated data already exist. Remove them first to regenerate.'); const n = CL.generateDemo(240); U.toast(n + ' simulated sessions generated. Choose "Simulated demo data" in LureLab to analyse them.'); CL.modules.account.render(); };
  $('#acDemoDel').onclick = () => { const simLoc = new Set(Store.list('locations').filter(l => l.simulated).map(l => l.id)); const n = Store.removeWhere('sessions', s => s.source === 'simulated'); Store.removeWhere('locations', l => l.simulated); const expIds = new Set(Store.list('experiments').filter(x => x.simulated).map(x => x.id)); Store.removeWhere('experiments', x => x.simulated); Store.removeWhere('trials', t => expIds.has(t.experimentId)); Store.removeWhere('predictions', p => p.simulated); U.toast(n + ' simulated sessions removed.'); CL.modules.account.render(); };
  $('#acDelete').onclick = () => { if (prompt('Type your username to permanently delete this account and its data from this browser:') === Store.user.username) Store.deleteAccount(); };

  if (window.Chart) { Chart.defaults.font.family = "'Instrument Sans', system-ui, sans-serif"; Chart.defaults.font.size = 12; }
  /* ---------- boot ---------- */
  window.addEventListener('DOMContentLoaded', () => { if (!Store.resume()) { $('#authView').hidden = false; } });
  if (document.readyState !== 'loading') { if (!Store.user && !Store.resume()) $('#authView').hidden = false; }
})();
