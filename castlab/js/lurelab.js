/* LureLab: session log, Bayesian estimates, Bayesian logistic regression, controlled experiments and
   prequential validation. Reads and writes the shared Store; nothing here invents data. */
(function () {
  'use strict';
  const { $, $$, esc, fmt, pct } = U, S = CLStats;
  const LL = CL.modules.lurelab = {};
  let showTab = null;

  const ds = () => $('#llData').value;
  const sessions = d => Store.list('sessions').filter(s => (d || ds()) === 'simulated' ? s.source === 'simulated' : s.source !== 'simulated');
  const sortKey = s => (s.date || '') + String(s.hour == null ? 99 : s.hour).padStart(2, '0') + (s.created || '');

  function qnorm(p) { // Acklam's rational approximation, |rel err| < 1.2e-9
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239], b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572], c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783], d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) return -qnorm(1 - p);
    const q = p - 0.5, r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  LL.qnorm = qnorm;

  LL.render = function (sub) {
    if (!showTab) showTab = CL.tabs('#llTabs', '.lltab', t => { try { ({ sessions: renderSessions, estimates: renderEstimates, model: renderModelTab, experiments: renderExperiments, validation: renderValidation, methods: renderMethods })[t](); } catch (e) { console.error(e); U.toast(e.message, 'bad'); } });
    if (!LL._dsInit) { LL._dsInit = true; if (!sessions('user').length && sessions('simulated').length) $('#llData').value = 'simulated'; }
    fillSelects(); dataNote();
    if (CL.openExperiment) sub = 'experiments';
    showTab(sub || ($('#llTabs .active') || {}).dataset.t || 'sessions');
  };
  LL.tab = t => showTab && showTab(t);
  $('#llData').onchange = () => LL.render();

  function dataNote() {
    const sim = ds() === 'simulated', n = sessions().length;
    $('#llDataNote').innerHTML = sim ? '<p class="note warn"><span class="demo-flag">Simulated</span> You are viewing ' + n + ' sessions generated from a known model (Account → Simulated demo dataset). Nothing on these tabs is a real observation; use them to see how the analyses behave and whether the model recovers the true coefficients.' + (n ? '' : ' None exist yet — generate them on the Account page.') + '</p>' : '';
  }
  function fillSelects() {
    const lures = new Set(Store.list('inventory').filter(i => !/line|terminal/i.test(i.category)).map(i => i.name)); Store.list('sessions').forEach(s => s.lure && lures.add(s.lure));
    $('#lureList').innerHTML = Array.from(lures).sort().map(l => '<option value="' + esc(l) + '">').join('');
    ['#sesSpecies', '#expSpecies'].forEach(s => { if (!$(s).options.length) CL.speciesOptions($(s)); });
    if ($('#estSpecies').options.length < 2) { CL.speciesOptions($('#estSpecies'), true); }
    const keep = $('#sesLoc').value; CL.locationOptions($('#sesLoc')); if (keep) $('#sesLoc').value = keep; else if (Store.data.prefs.homeLocation) $('#sesLoc').value = Store.data.prefs.homeLocation;
    CL.locationOptions($('#expLoc'));
    $('#sesExp').innerHTML = '<option value="">— none —</option>' + Store.list('experiments').filter(x => !x.simulated).map(x => '<option value="' + x.id + '">' + esc(x.title) + '</option>').join('');
    if (!$('#sesDate').value) $('#sesDate').value = U.today();
  }

  /* ================= sessions ================= */
  function renderSessions() {
    const list = sessions().slice().sort((a, b) => sortKey(b).localeCompare(sortKey(a))), succ = list.filter(s => s.catches > 0).length;
    $('#sesSummary').innerHTML = list.length + ' sessions · ' + succ + ' successful (≥1 fish) · ' + (list.length - succ) + ' unsuccessful · ' + fmt(list.reduce((a, s) => a + (+s.hours || 0), 0), 1) + ' h · ' + list.reduce((a, s) => a + (+s.catches || 0), 0) + ' fish';
    $('#sesTable').innerHTML = list.length ? '<div class="tablewrap" style="max-height:560px"><table><tr><th>Date</th><th>Location</th><th>Species</th><th>Lure</th><th class="num">h</th><th class="num">Fish</th><th>Outcome</th><th>Conditions</th><th></th></tr>' + list.map(s => { const w = s.weather || {}; return '<tr><td class="mono">' + esc(s.date) + (s.hour != null ? ' ' + String(s.hour).padStart(2, '0') + 'h' : '') + '</td><td>' + esc(Store.locationName(s.locationId)) + '</td><td>' + esc(s.species) + '</td><td>' + esc(s.lure) + '</td><td class="num">' + fmt(s.hours, 2) + '</td><td class="num">' + s.catches + '</td><td>' + (s.catches > 0 ? '<span class="badge ok">success</span>' : '<span class="badge bad">blank</span>') + '</td><td class="small muted">' + [s.timeOfDay, w.tempC != null ? fmt(w.tempC, 0) + '°C' : '', w.pressureTrend ? 'P ' + w.pressureTrend : '', s.clarity].filter(Boolean).join(' · ') + '</td><td>' + (s.source === 'simulated' ? '<span class="demo-flag">sim</span>' : '<div class="btns"><button class="btn sm" data-ses-edit="' + s.id + '">Edit</button><button class="btn sm danger" data-ses-del="' + s.id + '">×</button></div>') + '</td></tr>'; }).join('') + '</table></div>' : '<p class="muted small">No sessions in this dataset yet.</p>';
  }
  function sesForm(s) {
    s = s || {}; const w = s.weather || {};
    $('#sesId').value = s.id || ''; $('#sesDate').value = s.date || U.today(); $('#sesHour').value = s.hour != null ? s.hour : 7; if (s.locationId) $('#sesLoc').value = s.locationId;
    if (s.species) $('#sesSpecies').value = s.species; $('#sesHours').value = s.hours || 2; $('#sesLure').value = s.lure || ''; $('#sesCatch').value = s.catches || 0; $('#sesUsed').value = 0;
    $('#sesWater').value = s.waterTempC != null ? s.waterTempC : ''; $('#sesClarity').value = s.clarity || '';
    $('#sesT').value = w.tempC != null ? w.tempC : ''; $('#sesW').value = w.windKph != null ? w.windKph : ''; $('#sesP').value = w.pressureHpa != null ? w.pressureHpa : ''; $('#sesPT').value = w.pressureTrend || ''; $('#sesC').value = w.cloudPct != null ? w.cloudPct : ''; $('#sesR').value = w.precipMm != null ? w.precipMm : '';
    $('#sesWxSrc').textContent = 'source: ' + (w.source || 'manual'); $('#sesWxSrc').dataset.src = w.source || 'manual';
    $('#sesExp').value = s.experimentId || ''; $('#sesNotes').value = s.notes || ''; $('#sesFormTitle').textContent = s.id ? 'Edit session' : 'Log a session';
  }
  const num = sel => $(sel).value === '' ? null : +$(sel).value;
  $('#sesForm').onsubmit = e => {
    e.preventDefault(); if (!$('#sesLoc').value) return U.toast('Add a location on the Fishing map first.');
    const hour = num('#sesHour'), o = { date: $('#sesDate').value, hour, timeOfDay: CL.timeOfDay(hour), locationId: $('#sesLoc').value, species: $('#sesSpecies').value, hours: +$('#sesHours').value, lure: $('#sesLure').value.trim(), catches: Math.max(0, Math.round(+$('#sesCatch').value)), waterTempC: num('#sesWater'), clarity: $('#sesClarity').value || null,
      weather: { tempC: num('#sesT'), windKph: num('#sesW'), pressureHpa: num('#sesP'), pressureTrend: $('#sesPT').value || null, cloudPct: num('#sesC'), precipMm: num('#sesR'), source: $('#sesWxSrc').dataset.src || 'manual' }, experimentId: $('#sesExp').value || null, notes: $('#sesNotes').value, source: 'user' };
    const used = Math.max(0, Math.round(+$('#sesUsed').value || 0));
    if (used) { const it = Store.list('inventory').find(i => i.name.toLowerCase() === o.lure.toLowerCase()); if (it) { Store.update('inventory', it.id, { qty: Math.max(0, (+it.qty || 0) - used) }); U.toast('BaitRadar: ' + it.name + ' quantity −' + used); } }
    if ($('#sesId').value) Store.update('sessions', $('#sesId').value, o); else Store.add('sessions', o);
    sesForm(); if (ds() === 'simulated') { $('#llData').value = 'user'; dataNote(); } renderSessions(); if (!used) U.toast('Session saved.');
  };
  $('#sesClear').onclick = () => sesForm();
  $('#sesTable').onclick = e => { const ed = e.target.closest('[data-ses-edit]'), dl = e.target.closest('[data-ses-del]'); if (ed) { sesForm(Store.get('sessions', ed.dataset.sesEdit)); window.scrollTo({ top: 0, behavior: 'smooth' }); } if (dl && confirm('Delete this session?')) { Store.remove('sessions', dl.dataset.sesDel); renderSessions(); } };
  $('#sesWxFetch').onclick = async () => { const l = Store.location($('#sesLoc').value); if (!l || !isFinite(l.lat)) return U.toast('Pick a location with coordinates.'); $('#sesWxSrc').textContent = 'fetching…'; try { const w = await U.fetchWeather(l.lat, l.lon, $('#sesDate').value, +$('#sesHour').value || 0); $('#sesT').value = w.tempC; $('#sesW').value = w.windKph; $('#sesP').value = w.pressureHpa; $('#sesPT').value = w.pressureTrend || ''; $('#sesC').value = w.cloudPct; $('#sesR').value = w.precipMm; $('#sesWxSrc').textContent = 'source: ' + w.source; $('#sesWxSrc').dataset.src = w.source; } catch (err) { $('#sesWxSrc').textContent = 'fetch failed: ' + err.message; $('#sesWxSrc').dataset.src = 'manual'; } };
  $('#sesCsv').onclick = () => U.download('castlab-sessions-' + ds() + '.csv', U.toCSV(sessions().map(U.flatSession)), 'text/csv');
  $('#sesImportBtn').onclick = () => $('#sesImport').click();
  $('#sesImport').onchange = e => {
    const f = e.target.files[0]; if (!f) return; const rd = new FileReader();
    rd.onload = () => { try { const rows = parseCSV(rd.result); let n = 0, created = 0; rows.forEach(r => { if (!r.date || r.catches === undefined || !r.lure) return; let loc = Store.list('locations').find(l => l.name === r.location); if (!loc) { loc = Store.add('locations', { name: r.location || 'Imported location', lat: r.lat ? +r.lat : null, lon: r.lon ? +r.lon : null, type: 'Other' }); created++; } const nb = v => v === '' || v == null ? null : +v, hour = nb(r.hour); Store.data.sessions.push({ id: Store.uid(), created: new Date().toISOString(), source: 'user', date: r.date, hour, timeOfDay: r.timeOfDay || CL.timeOfDay(hour), locationId: loc.id, species: r.species || 'Other', lure: r.lure, hours: +r.hours || 1, catches: Math.round(+r.catches) || 0, waterTempC: nb(r.waterTempC), clarity: r.clarity || null, weather: { tempC: nb(r.airTempC), windKph: nb(r.windKph), pressureHpa: nb(r.pressureHpa), pressureTrend: r.pressureTrend || null, cloudPct: nb(r.cloudPct), precipMm: nb(r.precipMm), source: r.weatherSource || 'imported' }, notes: r.notes || '' }); n++; }); Store.save('sessions'); U.toast('Imported ' + n + ' sessions' + (created ? ', created ' + created + ' locations' : '') + '. Required columns: date, lure, catches; optional: location, species, hours, hour, airTempC, windKph, pressureHpa, pressureTrend, cloudPct, precipMm, waterTempC, clarity.'); renderSessions(); } catch (err) { U.toast('CSV import failed: ' + err.message, 'bad'); } };
    rd.readAsText(f); e.target.value = '';
  };
  function parseCSV(t) {
    const rows = []; let row = [], f = '', q = false;
    for (let i = 0; i < t.length; i++) { const c = t[i]; if (q) { if (c === '"' && t[i + 1] === '"') { f += '"'; i++; } else if (c === '"') q = false; else f += c; } else if (c === '"') q = true; else if (c === ',') { row.push(f); f = ''; } else if (c === '\n' || c === '\r') { if (c === '\r' && t[i + 1] === '\n') i++; row.push(f); rows.push(row); row = []; f = ''; } else f += c; }
    if (f || row.length) { row.push(f); rows.push(row); }
    const head = rows.shift().map(h => h.trim()); return rows.filter(r => r.length > 1).map(r => Object.fromEntries(head.map((h, i) => [h, (r[i] || '').trim()])));
  }

  /* ================= estimates ================= */
  const GROUPS = { lure: s => s.lure, species: s => s.species, location: s => Store.locationName(s.locationId), timeOfDay: s => s.timeOfDay, pressureTrend: s => (s.weather || {}).pressureTrend, clarity: s => s.clarity };
  function estimate(list, by, prior, cred) {
    const g = {}; list.forEach(s => { const k = GROUPS[by](s); if (k == null || k === '') return; (g[k] = g[k] || []).push(s); });
    const all = list.length ? list.filter(s => s.catches > 0).length / list.length : 0.5;
    const [a0, b0] = prior === 'pooled' ? [4 * all || 0.5, 4 * (1 - all) || 0.5] : prior.split(',').map(Number);
    return Object.entries(g).map(([k, ss]) => { const succ = ss.filter(s => s.catches > 0).length, h = ss.reduce((a, s) => a + (+s.hours || 0), 0), f = ss.reduce((a, s) => a + (+s.catches || 0), 0); return { key: k, n: ss.length, succ, fail: ss.length - succ, hours: h, fish: f, bp: S.betaPosterior(succ, ss.length, a0, b0, cred), gp: h > 0 ? S.gammaPosterior(f, h, 0.5, 0.01, cred) : null }; }).sort((a, b) => b.n - a.n);
  }
  function renderEstimates() {
    const by = $('#estBy').value, sp = $('#estSpecies').value, cred = +$('#estCred').value, list = sessions().filter(s => !sp || s.species === sp), rows = estimate(list, by, $('#estPrior').value, cred), col = U.chartColors();
    const top = rows.slice(0, 14);
    U.chart('estForest', $('#estForest'), { type: 'bar', data: { labels: top.map(r => r.key + ' (n=' + r.n + ')'), datasets: [{ label: (cred * 100) + '% credible interval', data: top.map(r => [r.bp.lo, r.bp.hi]), backgroundColor: CL.alpha(col[0], .35), borderColor: col[0], borderWidth: 1, barPercentage: .5 }, { type: 'line', label: 'Posterior mean', data: top.map(r => r.bp.mean), showLine: false, pointRadius: 5, pointBackgroundColor: col[1], borderColor: col[1] }, { type: 'line', label: 'Raw proportion', data: top.map(r => r.succ / r.n), showLine: false, pointRadius: 4, pointStyle: 'crossRot', borderColor: U.cssVar('--ink'), pointBorderWidth: 2 }] }, options: CL.baseChartOpts({ indexAxis: 'y', scales: { x: { min: 0, max: 1, title: { display: true, text: 'P(≥1 fish in a session)', color: U.cssVar('--muted') } } } }) });
    const xs = Array.from({ length: 101 }, (_, i) => i / 100);
    U.chart('estDens', $('#estDens'), { type: 'line', data: { labels: xs.map(x => x.toFixed(2)), datasets: top.slice(0, 6).map((r, i) => ({ label: r.key + ' (' + r.succ + '/' + r.n + ')', data: xs.map(x => S.betaPdf(Math.min(.999, Math.max(.001, x)), r.bp.a, r.bp.b)), borderColor: col[i % col.length], pointRadius: 0, borderWidth: 2, fill: false, tension: .2 })) }, options: CL.baseChartOpts({ scales: { x: { title: { display: true, text: 'P(≥1 fish)', color: U.cssVar('--muted') }, ticks: { maxTicksLimit: 11 } }, y: { title: { display: true, text: 'posterior density', color: U.cssVar('--muted') } } } }) });
    $('#estTable').innerHTML = rows.length ? '<div class="tablewrap"><table><tr><th>' + esc(by) + '</th><th class="num">n</th><th class="num">success / blank</th><th class="num">P(≥1 fish)</th><th class="num">' + (cred * 100) + '% CrI</th><th class="num">hours</th><th class="num">fish</th><th class="num">fish/hour</th><th class="num">' + (cred * 100) + '% CrI</th></tr>' + rows.map(r => '<tr><td>' + esc(r.key) + (r.n < 10 ? ' <span class="badge warn">small n</span>' : '') + '</td><td class="num">' + r.n + '</td><td class="num">' + r.succ + ' / ' + r.fail + '</td><td class="num">' + pct(r.bp.mean, 1) + '</td><td class="num">' + pct(r.bp.lo, 1) + ' – ' + pct(r.bp.hi, 1) + '</td><td class="num">' + fmt(r.hours, 1) + '</td><td class="num">' + r.fish + '</td><td class="num">' + (r.gp ? fmt(r.gp.mean, 2) : '—') + '</td><td class="num">' + (r.gp ? fmt(r.gp.lo, 2) + ' – ' + fmt(r.gp.hi, 2) : '—') + '</td></tr>').join('') + '</table></div><p class="small muted" style="margin-top:6px">These are observational comparisons: groups differ in conditions, places and effort as well as in the grouping variable, so a difference here is not evidence that one lure causes more catches. Use the model (which adjusts for recorded conditions) or, better, a controlled experiment.</p>' : '<p class="muted small">No sessions with this field recorded.</p>';
    LL._est = rows;
  }
  ['#estBy', '#estSpecies', '#estPrior', '#estCred'].forEach(s => $(s).onchange = renderEstimates);
  $('#estCsv').onclick = () => U.download('castlab-estimates.csv', U.toCSV((LL._est || []).map(r => ({ group: r.key, n: r.n, successes: r.succ, failures: r.fail, p_mean: r.bp.mean, p_lo: r.bp.lo, p_hi: r.bp.hi, cred: r.bp.cred, prior_a: r.bp.a - r.succ, prior_b: r.bp.b - r.fail, hours: r.hours, fish: r.fish, rate_mean: r.gp && r.gp.mean, rate_lo: r.gp && r.gp.lo, rate_hi: r.gp && r.gp.hi }))), 'text/csv');

  /* ================= logistic model ================= */
  const VARS = [
    { k: 'lure', label: 'Lure', type: 'cat', get: s => s.lure, def: true },
    { k: 'species', label: 'Species', type: 'cat', get: s => s.species },
    { k: 'timeOfDay', label: 'Time of day', type: 'cat', get: s => s.timeOfDay, def: true },
    { k: 'pressureTrend', label: 'Pressure trend', type: 'cat', get: s => (s.weather || {}).pressureTrend, def: true },
    { k: 'clarity', label: 'Water clarity', type: 'cat', get: s => s.clarity },
    { k: 'location', label: 'Location', type: 'cat', get: s => s.locationId },
    { k: 'logHours', label: 'log(hours fished)', type: 'num', get: s => s.hours > 0 ? Math.log(s.hours) : null, def: true, raw: true },
    { k: 'airTemp', label: 'Air temperature (std.)', type: 'num', get: s => (s.weather || {}).tempC },
    { k: 'wind', label: 'Wind speed (std.)', type: 'num', get: s => (s.weather || {}).windKph },
    { k: 'pressure', label: 'Pressure (std.)', type: 'num', get: s => (s.weather || {}).pressureHpa },
    { k: 'waterTemp', label: 'Water temperature (std.)', type: 'num', get: s => s.waterTempC }
  ];
  function specFromUI() {
    if (!$('#mdlVars').children.length) $('#mdlVars').innerHTML = VARS.map(v => '<label class="chk"><input type="checkbox" data-var="' + v.k + '"' + (v.def ? ' checked' : '') + '> ' + esc(v.label) + '</label>').join('');
    return { vars: $$('#mdlVars input:checked').map(i => i.dataset.var), priorSd: +$('#mdlPrior').value || 1, minLevel: +$('#mdlMinLevel').value || 3 };
  }
  const defaultSpec = () => ({ vars: VARS.filter(v => v.def).map(v => v.k), priorSd: 1, minLevel: 3 });
  /* Build an encoder from training sessions only (levels, reference level, standardisation). */
  function buildEncoder(train, spec) {
    const vs = spec.vars.map(k => VARS.find(v => v.k === k)).filter(Boolean), cols = [{ name: '(intercept)', v: null }], enc = [];
    const usable = train.filter(s => vs.every(v => { const x = v.get(s); return x != null && x !== '' && (v.type === 'cat' || isFinite(x)); }));
    for (const v of vs) {
      if (v.type === 'cat') {
        const cnt = {}; usable.forEach(s => { const x = String(v.get(s)); cnt[x] = (cnt[x] || 0) + 1; });
        let levels = Object.keys(cnt).filter(l => cnt[l] >= spec.minLevel); const small = Object.keys(cnt).filter(l => cnt[l] < spec.minLevel);
        const hasOther = small.length > 0 && small.reduce((a, l) => a + cnt[l], 0) >= spec.minLevel;
        levels.sort((a, b) => cnt[b] - cnt[a]); const ref = levels[0]; const dummies = levels.slice(1); if (hasOther) dummies.push('(other)');
        const map = l => cnt[l] >= spec.minLevel ? l : hasOther ? '(other)' : ref;
        dummies.forEach(l => cols.push({ name: v.k + '=' + l, v: v.k, level: l, display: v.label + ': ' + (v.k === 'location' ? Store.locationName(l) : l) + ' vs ' + (v.k === 'location' ? Store.locationName(ref) : ref) }));
        enc.push({ v, kind: 'cat', ref, dummies, map, levels: levels.concat(hasOther ? ['(other)'] : []), counts: cnt });
      } else {
        const xs = usable.map(s => +v.get(s)), m = S.mean(xs), sd = S.sd(xs) || 1;
        cols.push({ name: v.k, v: v.k, display: v.label + (v.raw ? '' : ' (per 1 SD = ' + fmt(sd, 1) + ')') });
        enc.push({ v, kind: 'num', mean: v.raw ? 0 : m, sd: v.raw ? 1 : sd, center: m });
      }
    }
    const encode = s => { const x = [1]; for (const e of enc) { const raw = e.v.get(s); if (raw == null || raw === '' || (e.kind === 'num' && !isFinite(raw))) return null; if (e.kind === 'cat') { const l = e.map(String(raw)); e.dummies.forEach(d => x.push(l === d ? 1 : 0)); } else x.push((+raw - e.mean) / e.sd); } return x; };
    return { cols, enc, encode, usable };
  }
  function fitModel(train, spec) {
    const E = buildEncoder(train, spec), X = [], y = [];
    E.usable.forEach(s => { const x = E.encode(s); if (x) { X.push(x); y.push(s.catches > 0 ? 1 : 0); } });
    if (X.length < 5 || y.every(v => v === y[0])) return { error: X.length < 5 ? 'Need at least 5 complete sessions.' : 'All sessions have the same outcome; at least one success and one blank are needed.', n: X.length };
    const prior = E.cols.map((c, j) => j === 0 ? 2.5 : spec.priorSd);
    const fit = S.logisticFit(X, y, prior); return Object.assign(fit, { E, cols: E.cols, nDropped: train.length - X.length, y, X, spec });
  }
  LL.fitModel = fitModel; LL.buildEncoder = buildEncoder;
  let lastFit = null;
  function renderModelTab() { specFromUI(); doFit(); }
  function doFit() {
    const list = sessions(), spec = specFromUI(), f = fitModel(list, spec); lastFit = f;
    if (f.error) { $('#mdlInfo').innerHTML = '<p class="note warn">' + esc(f.error) + ' (' + f.n + ' usable)</p>'; $('#mdlTable').innerHTML = ''; U.chart('mdlOR', null); $('#predInputs').innerHTML = ''; $('#predOut').innerHTML = ''; return; }
    const p = f.cols.length, truth = ds() === 'simulated' ? Store.data.prefs.demoTruth : null;
    $('#mdlInfo').innerHTML = '<div class="stat-line"><span>Sessions used</span><b>' + f.n + '</b></div><div class="stat-line"><span>Dropped (missing a selected field)</span><b>' + f.nDropped + '</b></div><div class="stat-line"><span>Parameters</span><b>' + p + '</b></div><div class="stat-line"><span>Sessions per parameter</span><b class="' + (f.n / p < 10 ? 'status-warn' : '') + '">' + fmt(f.n / p, 1) + '</b></div><div class="stat-line"><span>Newton iterations</span><b>' + f.iterations + (f.converged ? '' : ' (not converged)') + '</b></div>' + (f.n / p < 10 ? '<p class="small status-warn">Fewer than ~10 sessions per parameter: estimates lean heavily on the prior and intervals are wide. Remove predictors or log more sessions.</p>' : '');
    const rows = f.cols.map((c, j) => { const b = f.beta[j], se = f.se[j]; return { name: c.name, display: c.display || c.name, b, se, or: Math.exp(b), lo: Math.exp(b - 1.96 * se), hi: Math.exp(b + 1.96 * se), pPos: 1 - normCdf(-b / se), truth: truthFor(truth, c) }; });
    const eff = rows.slice(1), col = U.chartColors();
    U.chart('mdlOR', $('#mdlOR'), { type: 'bar', data: { labels: eff.map(r => r.display), datasets: [{ label: '95% interval (odds ratio)', data: eff.map(r => [r.lo, r.hi]), backgroundColor: CL.alpha(col[0], .35), borderColor: col[0], borderWidth: 1, barPercentage: .5 }, { type: 'line', label: 'Posterior mode', data: eff.map(r => r.or), showLine: false, pointRadius: 5, pointBackgroundColor: col[1], borderColor: col[1] }].concat(truth ? [{ type: 'line', label: 'True value (simulation)', data: eff.map(r => r.truth == null ? null : Math.exp(r.truth)), showLine: false, pointRadius: 6, pointStyle: 'crossRot', borderColor: U.cssVar('--ink'), pointBorderWidth: 2 }] : []) }, options: CL.baseChartOpts({ indexAxis: 'y', scales: { x: { type: 'logarithmic', title: { display: true, text: 'odds ratio (log scale; 1 = no effect)', color: U.cssVar('--muted') } } } }) });
    $('#mdlTable').innerHTML = '<div class="tablewrap"><table><tr><th>Term</th><th class="num">β (log-odds)</th><th class="num">SE</th><th class="num">Odds ratio</th><th class="num">95% interval</th><th class="num">P(β&gt;0)</th>' + (truth ? '<th class="num">True β</th><th>Covered</th>' : '') + '</tr>' + rows.map(r => '<tr><td>' + esc(r.display) + '</td><td class="num">' + fmt(r.b, 3) + '</td><td class="num">' + fmt(r.se, 3) + '</td><td class="num">' + fmt(r.or, 2) + '</td><td class="num">' + fmt(r.lo, 2) + ' – ' + fmt(r.hi, 2) + '</td><td class="num">' + fmt(r.pPos, 3) + '</td>' + (truth ? '<td class="num">' + (r.truth == null ? '—' : fmt(r.truth, 3)) + '</td><td>' + (r.truth == null ? '—' : Math.abs(r.b - r.truth) <= 1.96 * r.se ? '<span class="badge ok">yes</span>' : '<span class="badge bad">no</span>') + '</td>' : '') + '</tr>').join('') + '</table></div>' + (truth ? '<p class="small muted" style="margin-top:6px">Simulation check: the data were generated from the "True β" values (relative to each term\'s reference level, re-expressed for the chosen reference). Roughly 95% of terms should be covered; the intercept and log(hours) are on comparable scales only when the same predictors are selected as in the generator (lure, time of day, pressure trend, log hours).</p>' : '');
    LL._modelRows = rows; renderPredictInputs(f);
  }
  function truthFor(T, c) {
    if (!T || !c.v) return null; const e = lastFitEnc(c.v); if (!e) return null;
    if (c.v === 'logHours') return T.logHours; const tbl = T[c.v]; if (!tbl || e.kind !== 'cat' || c.level === '(other)') return null; if (!(c.level in tbl) || !(e.ref in tbl)) return null; return tbl[c.level] - tbl[e.ref];
  }
  function lastFitEnc(k) { return lastFit && lastFit.E && lastFit.E.enc.find(e => e.v.k === k); }
  function normCdf(z) { const t = 1 / (1 + 0.2316419 * Math.abs(z)), d = 0.3989422804014327 * Math.exp(-z * z / 2), p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))); return z > 0 ? 1 - p : p; }
  $('#mdlFit').onclick = doFit;
  function renderPredictInputs(f) {
    $('#predInputs').innerHTML = f.E.enc.map(e => e.kind === 'cat' ? '<label>' + esc(e.v.label) + '<select data-pk="' + e.v.k + '">' + Object.keys(e.counts).sort((a, b) => e.counts[b] - e.counts[a]).map(l => '<option value="' + esc(l) + '">' + esc(e.v.k === 'location' ? Store.locationName(l) : l) + ' (n=' + e.counts[l] + ')</option>').join('') + '</select></label>' : '<label>' + esc(e.v.label.replace(' (std.)', '').replace('log(hours fished)', 'Hours fished')) + '<input type="number" step="any" data-pk="' + e.v.k + '" value="' + (e.v.k === 'logHours' ? 2 : fmt(e.center, 1)) + '"></label>').join('') + '<label>Planned date<input type="date" id="predDate" value="' + U.today() + '"></label>';
    $('#predOut').innerHTML = '';
  }
  function predictFromInputs() {
    const f = lastFit; if (!f || f.error) return null; const fake = { weather: {} };
    $$('#predInputs [data-pk]').forEach(el => { const k = el.dataset.pk, v = el.value; if (k === 'lure') fake.lure = v; else if (k === 'species') fake.species = v; else if (k === 'timeOfDay') fake.timeOfDay = v; else if (k === 'pressureTrend') fake.weather.pressureTrend = v; else if (k === 'clarity') fake.clarity = v; else if (k === 'location') fake.locationId = v; else if (k === 'logHours') fake.hours = +v; else if (k === 'airTemp') fake.weather.tempC = +v; else if (k === 'wind') fake.weather.windKph = +v; else if (k === 'pressure') fake.weather.pressureHpa = +v; else if (k === 'waterTemp') fake.waterTempC = +v; });
    const x = f.E.encode(fake); if (!x) return null; return { inputs: fake, pr: S.logisticPredict(f, x) };
  }
  $('#predGo').onclick = () => {
    const r = predictFromInputs(); if (!r) return U.toast('Fit the model first.');
    const lure = r.inputs.lure, hrs = r.inputs.hours || 2, ls = lure ? sessions().filter(s => s.lure === lure) : sessions(), h = ls.reduce((a, s) => a + (+s.hours || 0), 0), k = ls.reduce((a, s) => a + (+s.catches || 0), 0), gp = h ? S.gammaPosterior(k, h, 0.5, 0.01) : null;
    $('#predOut').innerHTML = '<div class="grid g3"><div class="kpi"><div class="l">P(≥1 fish)</div><div class="v">' + pct(r.pr.p) + '</div><div class="s">95% interval ' + pct(r.pr.lo) + ' – ' + pct(r.pr.hi) + ' (parameter uncertainty)</div></div><div class="kpi"><div class="l">Expected fish</div><div class="v">' + (gp ? fmt(gp.mean * hrs, 1) : '—') + '</div><div class="s">' + (gp ? 'Gamma–Poisson rate for ' + esc(lure || 'all lures') + ' × ' + hrs + ' h; 95% CrI ' + fmt(gp.lo * hrs, 1) + '–' + fmt(gp.hi * hrs, 1) : 'no effort logged') + '</div></div><div class="kpi"><div class="l">Based on</div><div class="v">' + lastFit.n + '</div><div class="s">sessions; this lure ' + ls.length + ' sessions / ' + fmt(h, 1) + ' h</div></div></div><p class="small muted" style="margin-top:6px">The interval reflects uncertainty in the model coefficients, not the trip-to-trip randomness of fishing: even a well-estimated 60% means 4 in 10 such trips are blanks. Check the Validation tab to see how well-calibrated past predictions have been.</p>';
  };
  $('#predLog').onclick = () => {
    const r = predictFromInputs(); if (!r) return U.toast('Fit the model first.');
    Store.add('predictions', { plannedDate: $('#predDate').value, inputs: r.inputs, p: r.pr.p, lo: r.pr.lo, hi: r.pr.hi, nTrain: lastFit.n, spec: lastFit.spec, outcome: null, simulated: ds() === 'simulated' });
    U.toast('Prediction frozen at ' + pct(r.pr.p) + '. Record the outcome on the Validation tab after the trip.');
  };

  /* ================= validation ================= */
  function prequential(list, spec, burn) {
    const sorted = list.slice().sort((a, b) => sortKey(a).localeCompare(sortKey(b))), P = [], Y = [], B = [], D = [];
    for (let i = burn; i < sorted.length; i++) {
      const train = sorted.slice(0, i), f = fitModel(train, spec); if (f.error) continue;
      const x = f.E.encode(sorted[i]); if (!x) continue;
      const k = train.filter(s => s.catches > 0).length;
      P.push(S.logisticPredict(f, x).p); B.push((k + 1) / (train.length + 2)); Y.push(sorted[i].catches > 0 ? 1 : 0); D.push(sorted[i].date);
    }
    if (!P.length) return { n: 0 };
    const brier = S.brier(P, Y), brierBase = S.brier(B, Y);
    return { n: P.length, P, Y, B, D, brier, brierBase, bss: 1 - brier / brierBase, logLoss: S.logLoss(P, Y), logLossBase: S.logLoss(B, Y), acc: S.mean(P.map((p, i) => (p >= .5 ? 1 : 0) === Y[i] ? 1 : 0)), auc: auc(P, Y), base: S.mean(Y) };
  }
  function auc(p, y) { const pos = p.filter((_, i) => y[i]), neg = p.filter((_, i) => !y[i]); if (!pos.length || !neg.length) return NaN; let s = 0; for (const a of pos) for (const b of neg) s += a > b ? 1 : a === b ? 0.5 : 0; return s / (pos.length * neg.length); }
  let qvCache = {};
  LL.quickValidation = function (d) { const list = sessions(d), key = d + list.length + (list.map(s => s.modified || s.created).sort().slice(-1)[0] || ''); if (qvCache.key !== key) qvCache = { key, v: list.length > 16 ? prequential(list, defaultSpec(), 15) : { n: 0 } }; return qvCache.v; };
  function renderValidation() {
    renderForward();
    const list = sessions(); if (list.length <= (+$('#valBurn').value || 15)) { $('#valKpis').innerHTML = '<div class="kpi" style="grid-column:1/-1"><div class="l">Not enough data</div><div class="s">Prequential validation needs more sessions than the burn-in (' + $('#valBurn').value + '). This dataset has ' + list.length + '.</div></div>'; U.chart('valCal', null); U.chart('valCum', null); return; }
  }
  $('#valRun').onclick = () => {
    const list = sessions(), burn = Math.max(5, +$('#valBurn').value || 15), spec = (lastFit && !lastFit.error && lastFit.spec) || specFromUI();
    $('#valKpis').innerHTML = '<p class="muted">Refitting the model ' + Math.max(0, list.length - burn) + ' times…</p>';
    setTimeout(() => {
      const v = prequential(list, spec, burn); LL._val = v;
      if (!v.n) { $('#valKpis').innerHTML = '<div class="kpi" style="grid-column:1/-1"><div class="l">No scorable predictions</div><div class="s">Need more sessions after the burn-in, with both outcomes in the training data and all selected fields recorded.</div></div>'; return; }
      $('#valKpis').innerHTML = [['Predictions scored', v.n, 'out-of-sample, forward in time · base rate ' + pct(v.base)], ['Brier score', fmt(v.brier, 3), 'running base rate: ' + fmt(v.brierBase, 3) + ' (lower is better)'], ['Brier skill score', fmt(v.bss, 3), v.bss > 0 ? 'model beats the base rate' : 'model does NOT beat the base rate'], ['AUC · accuracy', fmt(v.auc, 2) + ' · ' + pct(v.acc), 'log loss ' + fmt(v.logLoss, 3) + ' vs ' + fmt(v.logLossBase, 3)]].map(k => '<div class="kpi"><div class="l">' + k[0] + '</div><div class="v">' + k[1] + '</div><div class="s">' + k[2] + '</div></div>').join('');
      const cal = S.calibration(v.P, v.Y, +$('#valBins').value || 5).filter(b => b.n), col = U.chartColors();
      U.chart('valCal', $('#valCal'), { type: 'scatter', data: { datasets: [{ label: 'Perfect calibration', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, borderColor: U.cssVar('--faint'), borderDash: [5, 5], pointRadius: 0 }, { label: 'Observed frequency (95% CrI)', data: cal.map(b => ({ x: b.meanP, y: b.obs, lo: b.ci.lo, hi: b.ci.hi, n: b.n })), pointRadius: cal.map(b => 3 + Math.sqrt(b.n)), backgroundColor: col[0], borderColor: col[0] }] }, options: CL.baseChartOpts({ scales: { x: { min: 0, max: 1, title: { display: true, text: 'predicted P(≥1 fish)', color: U.cssVar('--muted') } }, y: { min: 0, max: 1, title: { display: true, text: 'observed proportion', color: U.cssVar('--muted') } } }, plugins: { tooltip: { callbacks: { label: c => c.raw.n ? 'pred ' + pct(c.raw.x) + ', obs ' + pct(c.raw.y) + ' (' + pct(c.raw.lo) + '–' + pct(c.raw.hi) + '), n=' + c.raw.n : '' } } } }), plugins: [errorBars] });
      let cm = 0, cb = 0; const cumM = [], cumB = []; v.P.forEach((p, i) => { cm += (p - v.Y[i]) ** 2; cb += (v.B[i] - v.Y[i]) ** 2; cumM.push(cm / (i + 1)); cumB.push(cb / (i + 1)); });
      U.chart('valCum', $('#valCum'), { type: 'line', data: { labels: v.D.map((d, i) => (i + 1) + ' · ' + d), datasets: [{ label: 'Model', data: cumM, borderColor: col[0], pointRadius: 0, borderWidth: 2 }, { label: 'Running base rate', data: cumB, borderColor: col[1], pointRadius: 0, borderWidth: 2, borderDash: [5, 4] }] }, options: CL.baseChartOpts({ scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: 'mean Brier score so far', color: U.cssVar('--muted') } } } }) });
    }, 30);
  };
  const errorBars = { id: 'errorBars', afterDatasetsDraw(c) { const ds = c.data.datasets[1]; if (!ds) return; const m = c.getDatasetMeta(1), ctx = c.ctx; ctx.save(); ctx.strokeStyle = ds.borderColor; ctx.lineWidth = 1.5; ds.data.forEach((d, i) => { if (d.lo == null) return; const x = m.data[i].x, y1 = c.scales.y.getPixelForValue(d.lo), y2 = c.scales.y.getPixelForValue(d.hi); ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.moveTo(x - 4, y1); ctx.lineTo(x + 4, y1); ctx.moveTo(x - 4, y2); ctx.lineTo(x + 4, y2); ctx.stroke(); }); ctx.restore(); } };
  function renderForward() {
    const sim = ds() === 'simulated', list = Store.list('predictions').filter(p => !!p.simulated === sim).sort((a, b) => b.created.localeCompare(a.created)), done = list.filter(p => p.outcome != null);
    $('#fwdTable').innerHTML = (done.length ? '<p class="small"><b>' + done.length + '</b> resolved · Brier ' + fmt(S.brier(done.map(p => p.p), done.map(p => p.outcome)), 3) + ' · mean predicted ' + pct(S.mean(done.map(p => p.p))) + ' vs observed ' + pct(S.mean(done.map(p => p.outcome))) + (done.length < 20 ? ' <span class="badge warn">fewer than 20 — too few to judge calibration</span>' : '') + '</p>' : '') + (list.length ? '<div class="tablewrap"><table><tr><th>Logged</th><th>Trip date</th><th>Conditions</th><th class="num">P(≥1 fish)</th><th class="num">95% interval</th><th>Outcome</th><th></th></tr>' + list.map(p => '<tr><td class="mono small">' + esc(p.created.slice(0, 16).replace('T', ' ')) + '</td><td>' + esc(p.plannedDate) + '</td><td class="small">' + esc([p.inputs.lure, p.inputs.species, p.inputs.timeOfDay, p.inputs.weather && p.inputs.weather.pressureTrend, p.inputs.hours && p.inputs.hours + ' h'].filter(Boolean).join(' · ')) + '</td><td class="num">' + pct(p.p) + '</td><td class="num">' + pct(p.lo) + '–' + pct(p.hi) + '</td><td>' + (p.outcome == null ? '<div class="btns"><button class="btn sm" data-fwd="' + p.id + '" data-o="1">Caught ≥1</button><button class="btn sm" data-fwd="' + p.id + '" data-o="0">Blank</button></div>' : p.outcome ? '<span class="badge ok">caught</span>' : '<span class="badge bad">blank</span>') + '</td><td><button class="btn sm danger" data-fwd-del="' + p.id + '">×</button></td></tr>').join('') + '</table></div>' : '<p class="muted small">No forward predictions yet. On the Model tab, set up a planned trip and press "Log as forward prediction" before you go.</p>');
  }
  $('#fwdTable').onclick = e => { const b = e.target.closest('[data-fwd]'); if (b) { Store.update('predictions', b.dataset.fwd, { outcome: +b.dataset.o, resolved: new Date().toISOString() }); renderForward(); } const d = e.target.closest('[data-fwd-del]'); if (d && confirm('Delete this prediction record? Deleting unfavourable predictions biases the track record.')) { Store.remove('predictions', d.dataset.fwdDel); renderForward(); } };

  /* ================= experiments ================= */
  function sampleSize(la, lb, alpha, power) { if (!(la > 0 && lb > 0) || la === lb) return null; const z = qnorm(1 - alpha / 2) + qnorm(power); return z * z * (1 / la + 1 / lb) / Math.pow(Math.log(lb / la), 2); }
  function updateSS() { const h = sampleSize(+$('#ssA').value, +$('#ssB').value, +$('#ssAlpha').value, +$('#ssPow').value), bm = +$('#expBlock').value || 20; $('#ssOut').innerHTML = h ? 'About <b>' + fmt(h, 1) + ' h per lure</b> (' + Math.ceil(h * 60 / bm) + ' blocks of ' + bm + ' min each) to detect a rate ratio of ' + fmt($('#ssB').value / $('#ssA').value, 2) + ' with ' + pct(+$('#ssPow').value) + ' power at α=' + $('#ssAlpha').value + '. <span class="muted">Normal approximation on the log rate ratio for Poisson counts; real catches are usually over-dispersed (fish come in groups), so treat this as a minimum.</span> <button class="btn sm" type="button" id="ssUse">Use ' + Math.ceil(h * 60 / bm) + ' blocks</button>' : 'Enter two different positive rates.'; const u = $('#ssUse'); if (u) u.onclick = () => $('#expBlocks').value = Math.ceil(h * 60 / bm); }
  ['#ssA', '#ssB', '#ssAlpha', '#ssPow', '#expBlock'].forEach(s => $(s).oninput = updateSS);
  $('#expForm').onsubmit = e => {
    e.preventDefault(); const k = Math.max(2, +$('#expBlocks').value || 12), seed = (Date.now() % 2147483647) >>> 0, r = S.rng(seed), order = [];
    for (let i = 0; i < k; i++) order.push(...(r() < .5 ? ['A', 'B'] : ['B', 'A']));
    const x = Store.add('experiments', { title: $('#expTitle').value.trim(), lureA: $('#expA').value.trim(), lureB: $('#expB').value.trim(), species: $('#expSpecies').value, locationId: $('#expLoc').value, hypothesis: $('#expHyp').value.trim() || 'The catch rate (fish per hour) differs between lure A and lure B under matched conditions.', blockMin: +$('#expBlock').value || 20, blocksPerArm: k, seed, order, registered: new Date().toISOString(), plannedSS: { la: +$('#ssA').value, lb: +$('#ssB').value, alpha: +$('#ssAlpha').value, power: +$('#ssPow').value } });
    e.target.reset(); updateSS(); CL.openExperiment = x.id; renderExperiments(); U.toast('Experiment registered. Fish the blocks in the listed order.');
  };
  function renderExperiments() {
    updateSS(); const sim = ds() === 'simulated', list = Store.list('experiments').filter(x => !!x.simulated === sim);
    $('#expList').innerHTML = list.length ? '<table><tr><th>Experiment</th><th class="num">Blocks done</th><th></th></tr>' + list.map(x => { const n = Store.list('trials').filter(t => t.experimentId === x.id).length; return '<tr><td><b>' + esc(x.title) + '</b><br><span class="small muted">' + esc(x.lureA) + ' vs ' + esc(x.lureB) + ' · ' + esc(x.species) + '</span></td><td class="num">' + n + ' / ' + x.order.length + '</td><td><div class="btns"><button class="btn sm" data-exp-open="' + x.id + '">Open</button><button class="btn sm danger" data-exp-del="' + x.id + '">×</button></div></td></tr>'; }).join('') + '</table>' : '<p class="muted small">No experiments in this dataset.</p>';
    if (CL.openExperiment && Store.get('experiments', CL.openExperiment)) renderExpDetail(CL.openExperiment); else $('#expDetail').hidden = true;
  }
  $('#expList').onclick = e => { const o = e.target.closest('[data-exp-open]'), d = e.target.closest('[data-exp-del]'); if (o) { CL.openExperiment = o.dataset.expOpen; renderExpDetail(o.dataset.expOpen); $('#expDetail').scrollIntoView({ behavior: 'smooth' }); } if (d && confirm('Delete experiment and all its trial data?')) { Store.removeWhere('trials', t => t.experimentId === d.dataset.expDel); Store.remove('experiments', d.dataset.expDel); CL.openExperiment = null; renderExperiments(); } };
  function binomPmf(k, n, p) { return Math.exp(S.lgamma(n + 1) - S.lgamma(k + 1) - S.lgamma(n - k + 1) + (k ? k * Math.log(p) : 0) + (n - k ? (n - k) * Math.log(1 - p) : 0)); }
  LL.analyseExperiment = function (x) {
    const t = Store.list('trials').filter(r => r.experimentId === x.id), arm = a => { const r = t.filter(q => q.arm === a); return { blocks: r.length, hours: r.reduce((s, q) => s + q.minutes / 60, 0), fish: r.reduce((s, q) => s + q.fish, 0), zero: r.filter(q => !q.fish).length, per: r.map(q => q.fish / (q.minutes / 60)) }; };
    const A = arm('A'), B = arm('B'), out = { A, B, trials: t, complete: t.length >= x.order.length };
    if (!A.hours || !B.hours) return out;
    out.pA = S.gammaPosterior(A.fish, A.hours, 0.5, 0.01); out.pB = S.gammaPosterior(B.fish, B.hours, 0.5, 0.01); out.rr = S.rateRatio(out.pB, out.pA, x.seed, 40000);
    const n = A.fish + B.fish, p0 = A.hours / (A.hours + B.hours);
    if (n > 0) { const obs = binomPmf(A.fish, n, p0); let pv = 0; for (let k = 0; k <= n; k++) { const pk = binomPmf(k, n, p0); if (pk <= obs * (1 + 1e-9)) pv += pk; } out.pValue = Math.min(1, pv);
      const lo = A.fish === 0 ? 0 : S.betaQuantile(0.025, A.fish, n - A.fish + 1), hi = A.fish === n ? 1 : S.betaQuantile(0.975, A.fish + 1, n - A.fish), toRR = q => (1 - q) / q * A.hours / B.hours; // RR = rateB / rateA
      out.rrExact = { est: (B.fish / B.hours) / (A.fish / A.hours || NaN), lo: hi < 1 ? toRR(hi) : 0, hi: lo > 0 ? toRR(lo) : Infinity }; }
    const all = A.per.concat(B.per), mu = S.mean(all), v = S.sd(all) ** 2; out.dispersion = mu > 0 ? v / mu * (x.blockMin / 60) : NaN; // variance-to-mean of counts per block
    return out;
  };
  function renderExpDetail(id) {
    const x = Store.get('experiments', id), box = $('#expDetail'); if (!x) { box.hidden = true; return; } box.hidden = false;
    const res = LL.analyseExperiment(x), done = res.trials.length, next = x.order[done], col = U.chartColors();
    box.innerHTML = '<div class="card-h"><h2>' + esc(x.title) + '</h2><div class="btns">' + (x.simulated ? '<span class="demo-flag">Simulated</span>' : '') + '<button class="btn sm" id="expMd">Export write-up</button><button class="btn sm" id="expCsv">Trials CSV</button></div></div>' +
      '<div class="grid g2"><div><h3>Protocol (registered ' + esc(x.registered.slice(0, 10)) + ')</h3><div class="stat-line"><span>Lure A</span><b>' + esc(x.lureA) + '</b></div><div class="stat-line"><span>Lure B</span><b>' + esc(x.lureB) + '</b></div><div class="stat-line"><span>Species · location</span><b>' + esc(x.species) + ' · ' + esc(Store.locationName(x.locationId)) + '</b></div><div class="stat-line"><span>Design</span><b>' + x.blocksPerArm + ' pairs of ' + x.blockMin + '-min blocks, order randomised within pairs (seed ' + x.seed + ')</b></div><p class="small" style="margin-top:6px"><b>Hypothesis.</b> ' + esc(x.hypothesis) + '</p>' +
      '<p class="small">Block order: <span class="mono">' + x.order.map((a, i) => i < done ? '<span class="muted">' + a + '</span>' : i === done ? '<b style="color:var(--accent)">[' + a + ']</b>' : a).join(' ') + '</span></p>' +
      (next ? '<form id="trialForm" class="card" style="background:var(--panel-2)"><h3>Block ' + (done + 1) + ' of ' + x.order.length + ': fish lure ' + next + ' — ' + esc(next === 'A' ? x.lureA : x.lureB) + '</h3><div class="row"><label>Minutes actually fished<input type="number" id="trMin" value="' + x.blockMin + '" min="1" step="1"></label><label>Fish caught<input type="number" id="trFish" value="0" min="0" step="1"></label></div><label>Notes<input type="text" id="trNotes"></label><div class="btns"><button class="btn primary" type="submit">Record block</button><button class="btn" type="button" id="trUndo"' + (done ? '' : ' disabled') + '>Undo last</button></div></form>' : '<p class="note">All planned blocks recorded. <button class="btn sm" id="trUndo" type="button">Undo last</button></p>') + '</div>' +
      '<div><h3>Results' + (res.complete ? '' : ' <span class="badge warn">interim — ' + done + '/' + x.order.length + ' blocks</span>') + '</h3>' + (res.rr ? '<div class="stat-line"><span>A: ' + esc(x.lureA) + '</span><b>' + res.A.fish + ' fish / ' + fmt(res.A.hours, 2) + ' h = ' + fmt(res.pA.mean, 2) + '/h (95% CrI ' + fmt(res.pA.lo, 2) + '–' + fmt(res.pA.hi, 2) + ')</b></div><div class="stat-line"><span>B: ' + esc(x.lureB) + '</span><b>' + res.B.fish + ' fish / ' + fmt(res.B.hours, 2) + ' h = ' + fmt(res.pB.mean, 2) + '/h (95% CrI ' + fmt(res.pB.lo, 2) + '–' + fmt(res.pB.hi, 2) + ')</b></div><div class="stat-line"><span>Rate ratio B/A (Bayesian)</span><b>' + fmt(res.rr.median, 2) + ' (95% CrI ' + fmt(res.rr.lo, 2) + '–' + fmt(res.rr.hi, 2) + ')</b></div><div class="stat-line"><span>P(B catches faster than A)</span><b>' + fmt(res.rr.pGreater, 3) + '</b></div>' + (res.rrExact ? '<div class="stat-line"><span>Exact conditional test (H₀: equal rates)</span><b>p = ' + fmt(res.pValue, 4) + '; RR 95% CI ' + fmt(res.rrExact.lo, 2) + '–' + (isFinite(res.rrExact.hi) ? fmt(res.rrExact.hi, 2) : '∞') + '</b></div>' : '') + '<div class="stat-line"><span>Over-dispersion (variance/mean of block counts)</span><b class="' + (res.dispersion > 1.5 ? 'status-warn' : '') + '">' + fmt(res.dispersion, 2) + '</b></div>' + (res.dispersion > 1.5 ? '<p class="small status-warn">Counts are more variable than a Poisson model assumes, so the intervals above are too narrow. Treat borderline results with caution.</p>' : '') : '<p class="muted small">Results appear once both lures have at least one recorded block.</p>') + '<div class="chartbox short" style="margin-top:8px"><canvas id="expCum"></canvas></div></div></div>';
    if (res.trials.length) { let a = 0, b = 0, ha = 0, hb = 0; const ca = [], cb = []; res.trials.forEach((t, i) => { if (t.arm === 'A') { a += t.fish; ha += t.minutes / 60; } else { b += t.fish; hb += t.minutes / 60; } ca.push({ x: i + 1, y: ha ? a / ha : null }); cb.push({ x: i + 1, y: hb ? b / hb : null }); });
      U.chart('expCum', $('#expCum'), { type: 'line', data: { datasets: [{ label: 'A cumulative fish/h', data: ca, borderColor: col[0], pointRadius: 2, spanGaps: true }, { label: 'B cumulative fish/h', data: cb, borderColor: col[1], pointRadius: 2, spanGaps: true }] }, options: CL.baseChartOpts({ scales: { x: { type: 'linear', title: { display: true, text: 'block', color: U.cssVar('--muted') } }, y: { beginAtZero: true } } }) }); }
    const tf = $('#trialForm'); if (tf) tf.onsubmit = e => { e.preventDefault(); Store.add('trials', { experimentId: x.id, idx: done, arm: next, minutes: Math.max(1, +$('#trMin').value), fish: Math.max(0, Math.round(+$('#trFish').value)), date: new Date().toISOString(), notes: $('#trNotes').value }); renderExpDetail(id); renderExperiments(); };
    const ud = $('#trUndo'); if (ud) ud.onclick = () => { const last = res.trials.slice().sort((a, b) => b.idx - a.idx)[0]; if (last && confirm('Remove the last recorded block?')) { Store.remove('trials', last.id); renderExpDetail(id); } };
    $('#expMd').onclick = () => U.download('castlab-experiment-' + x.id + '.md', experimentWriteup(x, res).map(p => '## ' + p[0] + '\n\n' + p[1].join('\n\n')).join('\n\n'), 'text/markdown');
    $('#expCsv').onclick = () => U.download('castlab-experiment-' + x.id + '-trials.csv', U.toCSV(res.trials.map(t => ({ experiment: x.title, block: t.idx + 1, arm: t.arm, lure: t.arm === 'A' ? x.lureA : x.lureB, minutes: t.minutes, fish: t.fish, recorded: t.date, notes: t.notes }))), 'text/csv');
  }
  function experimentWriteup(x, res) {
    const m = ['Design: two-arm comparison of ' + x.lureA + ' (A) and ' + x.lureB + ' (B) for ' + x.species + ' at ' + Store.locationName(x.locationId) + ', registered ' + x.registered.slice(0, 10) + ' before data collection. Fishing time was divided into ' + x.blocksPerArm + ' pairs of ' + x.blockMin + '-minute blocks; within each pair the lure order was randomised (seeded PRNG, seed ' + x.seed + ') so that time-of-day drift and fish activity changes affect both lures equally.', 'Hypothesis: ' + x.hypothesis, 'Outcome: fish caught per block and minutes actually fished. Analysis: catch counts modelled as Poisson with rate λ per hour; Gamma(0.5, 0.01 h) prior on each λ; posterior Gamma(0.5 + fish, 0.01 + hours). The rate ratio λB/λA and P(λB > λA) were computed from 40,000 seeded Monte Carlo draws. A frequentist check uses the exact conditional binomial test (given total fish n, fish on A ~ Binomial(n, hoursA / total hours) under H₀) with a Clopper–Pearson interval for the rate ratio. Over-dispersion is reported as the variance-to-mean ratio of block counts.'];
    if (x.plannedSS && x.plannedSS.la) { const h = sampleSize(x.plannedSS.la, x.plannedSS.lb, x.plannedSS.alpha, x.plannedSS.power); if (h) m.push('Planned sample size: about ' + h.toFixed(1) + ' h per lure for rates ' + x.plannedSS.la + ' vs ' + x.plannedSS.lb + ' fish/h at α=' + x.plannedSS.alpha + ', power ' + x.plannedSS.power + '.'); }
    const r = [];
    if (!res.rr) r.push('No analysable data yet (' + res.trials.length + ' blocks).');
    else { r.push((res.complete ? 'All ' : 'INTERIM: ') + res.trials.length + ' of ' + x.order.length + ' planned blocks recorded.'); r.push('A (' + x.lureA + '): ' + res.A.fish + ' fish in ' + res.A.hours.toFixed(2) + ' h, ' + res.pA.mean.toFixed(2) + ' fish/h (95% CrI ' + res.pA.lo.toFixed(2) + '–' + res.pA.hi.toFixed(2) + '). B (' + x.lureB + '): ' + res.B.fish + ' fish in ' + res.B.hours.toFixed(2) + ' h, ' + res.pB.mean.toFixed(2) + ' fish/h (95% CrI ' + res.pB.lo.toFixed(2) + '–' + res.pB.hi.toFixed(2) + ').'); r.push('Rate ratio B/A: ' + res.rr.median.toFixed(2) + ' (95% CrI ' + res.rr.lo.toFixed(2) + '–' + res.rr.hi.toFixed(2) + '); P(B > A) = ' + res.rr.pGreater.toFixed(3) + '.' + (res.rrExact ? ' Exact conditional test p = ' + res.pValue.toFixed(4) + '; 95% CI for the rate ratio ' + res.rrExact.lo.toFixed(2) + '–' + (isFinite(res.rrExact.hi) ? res.rrExact.hi.toFixed(2) : '∞') + '.' : '')); r.push('Variance-to-mean ratio of block counts: ' + (isFinite(res.dispersion) ? res.dispersion.toFixed(2) : 'n/a') + (res.dispersion > 1.5 ? ' (over-dispersed; Poisson intervals are optimistic).' : '.'));
      r.push(res.rr.lo > 1 ? 'Interpretation: the data favour B; the whole 95% interval is above 1.' : res.rr.hi < 1 ? 'Interpretation: the data favour A; the whole 95% interval is below 1.' : 'Interpretation: the interval includes 1, so these data do not establish a difference between the lures.'); }
    return [['Experiment: ' + x.title + ' — Methodology', m], ['Experiment: ' + x.title + ' — Results', r], ['Limitations', ['Single angler, single water body: results may not generalise. Blocks are not fully independent (fish schools, changing light). Interim looks at the data inflate false-positive rates if used to stop early; judge the result at the registered sample size.']]];
  }

  /* ================= methodology ================= */
  function renderMethods() {
    $('#llMethods').innerHTML = '<h2>LureLab methodology</h2>' +
      '<h3>1. Data</h3><p>Each session records date, start hour, location, target species, lure, hours fished and number of fish, plus optional conditions (air temperature, wind, surface pressure and its 3-hour trend, cloud, precipitation, water temperature and clarity). Weather is either measured by the angler or taken from the Open-Meteo hourly model for that hour and location; the source is stored with each record. A session is a <i>success</i> if at least one fish was caught. Blank sessions are data: leaving them out would bias every estimate upward.</p>' +
      '<h3>2. Catch probability by group (Beta–Binomial)</h3><p>For a group with <i>k</i> successes in <i>n</i> sessions and a Beta(a₀, b₀) prior:</p><div class="eq">θ | data ~ Beta(a₀ + k, b₀ + n − k)\nposterior mean = (a₀ + k) / (a₀ + b₀ + n)</div><p>Priors offered: uniform Beta(1,1), Jeffreys Beta(½,½), and a weakly pooled prior centred on the overall success rate with weight equal to 4 sessions (partial pooling toward the mean, which tempers extreme estimates from small groups). Credible intervals are equal-tailed quantiles of the posterior, computed by inverting the regularised incomplete beta function.</p>' +
      '<h3>3. Catch rate (Gamma–Poisson)</h3><p>Fish counts are treated as Poisson with rate λ fish per hour. With <i>F</i> fish over <i>H</i> hours and a Gamma(0.5, 0.01) prior (shape, rate), the posterior is Gamma(0.5 + F, 0.01 + H). This prior is nearly non-informative at fishing scales. Real catches are often over-dispersed (fish arrive in groups); in that case these intervals are too narrow.</p>' +
      '<h3>4. Bayesian logistic regression</h3><div class="eq">logit P(success) = β₀ + Σ βⱼ xⱼ\nβ₀ ~ Normal(0, 2.5²),  βⱼ ~ Normal(0, σ²)  (σ set on the Model tab, default 1)</div><p>Categorical predictors are dummy-coded against their most frequent level; levels with fewer sessions than the minimum are merged into "(other)" (or into the reference if "other" would itself be too small). Continuous weather predictors are standardised (mean 0, SD 1 in the training data); hours fished enters as log(hours) so its coefficient is an elasticity-like effect of effort. The posterior mode is found by Newton–Raphson on the log posterior; the posterior covariance is approximated by the inverse Hessian at the mode (Laplace approximation). Reported intervals are mode ± 1.96 SE, exponentiated to odds ratios. Prediction intervals propagate this covariance through the linear predictor.</p>' +
      '<h3>5. Controlled lure comparison</h3><p>The registered design is a randomised paired-block design: equal-length blocks, lure order randomised within each pair from a stored seed. Analysis: independent Gamma–Poisson posteriors per lure, Monte Carlo rate ratio and P(B > A), and the exact conditional binomial test as a frequentist check. Sample size uses the normal approximation for the log rate ratio:</p><div class="eq">hours per lure ≈ (z₁₋α/₂ + z₁₋β)² (1/λA + 1/λB) / (ln λB/λA)²</div>' +
      '<h3>6. Validation</h3><p><b>Prequential (forward-chaining) validation.</b> Sessions are ordered by date and hour. For each session after a burn-in, the model is refitted using only earlier sessions, a probability is predicted, and the prediction is scored against the observed outcome. This mimics using the model in real time and prevents information from the future leaking into a prediction. Reported metrics: Brier score (mean squared error of probabilities), log loss, AUC, accuracy at 0.5, and the Brier skill score against a running base-rate forecast ((k+1)/(n+2) from earlier sessions). A skill score at or below zero means the model is no better than quoting your overall success rate. The calibration plot bins predictions and shows the observed proportion in each bin with a Jeffreys 95% interval.</p><p><b>Forward predictions.</b> A prediction can be frozen with a timestamp before a trip and resolved afterwards. These are the cleanest test of the model because the outcome could not have influenced the model or the angler\'s choices.</p>' +
      '<h3>7. Simulation check</h3><p>The optional demo dataset is drawn from a logistic model with known coefficients (shown on the Model tab when the simulated dataset is selected). Recovering those coefficients within their intervals checks the estimation code; it says nothing about real fish.</p>' +
      '<h3>8. Limitations</h3><ul><li>Observational sessions are not experiments: anglers choose lures in response to conditions, so lure coefficients can be confounded. Use the experiment tool for causal comparisons.</li><li>Model output is only as good as what is logged; weather from a forecast model can differ from the water\'s microclimate.</li><li>The Laplace approximation can be poor for small samples or separated data (a level that always or never catches); the prior keeps estimates finite, but intervals are then prior-driven.</li></ul>';
  }

  /* ================= report ================= */
  LL.reportSection = function () {
    const list = sessions('user'); if (!list.length) return [['LureLab', ['No sessions logged.']]];
    const tbl = rows => '<table><tr><th>Lure</th><th>n</th><th>success</th><th>P(≥1 fish) [95% CrI]</th><th>fish/h [95% CrI]</th></tr>' + rows.map(r => '<tr><td>' + esc(r.key) + '</td><td>' + r.n + '</td><td>' + r.succ + '</td><td>' + pct(r.bp.mean) + ' [' + pct(r.bp.lo) + '–' + pct(r.bp.hi) + ']</td><td>' + (r.gp ? fmt(r.gp.mean, 2) + ' [' + fmt(r.gp.lo, 2) + '–' + fmt(r.gp.hi, 2) + ']' : '—') + '</td></tr>').join('') + '</table>';
    const est = estimate(list, 'lure', '1,1', 0.95), f = fitModel(list, defaultSpec()), v = LL.quickValidation('user');
    const parts = [['LureLab — Methodology', ['Per-lure catch probability: Beta–Binomial with uniform Beta(1,1) prior; catch rate: Gamma–Poisson with Gamma(0.5, 0.01) prior. Adjusted analysis: Bayesian logistic regression of P(≥1 fish) on lure, time of day, pressure trend and log(hours), Normal(0, 1) priors on effects, Normal(0, 2.5) on the intercept, Laplace approximation. Validation: prequential forward-chaining with a 15-session burn-in, scored by Brier score against a running base rate.']],
      ['LureLab — Results', ['<h3>Catch outcomes by lure (unadjusted)</h3>' + tbl(est)]]];
    if (!f.error) parts[1][1].push('<h3>Logistic regression (n=' + f.n + ', ' + f.nDropped + ' dropped for missing fields)</h3><table><tr><th>Term</th><th>Odds ratio</th><th>95% interval</th></tr>' + f.cols.map((c, j) => '<tr><td>' + esc(c.display || c.name) + '</td><td>' + fmt(Math.exp(f.beta[j]), 2) + '</td><td>' + fmt(Math.exp(f.beta[j] - 1.96 * f.se[j]), 2) + '–' + fmt(Math.exp(f.beta[j] + 1.96 * f.se[j]), 2) + '</td></tr>').join('') + '</table>');
    else parts[1][1].push('Logistic regression not fitted: ' + f.error);
    parts[1][1].push(v && v.n ? 'Prequential validation over ' + v.n + ' out-of-sample predictions: Brier ' + v.brier.toFixed(3) + ' vs ' + v.brierBase.toFixed(3) + ' for the running base rate (skill score ' + v.bss.toFixed(3) + '); AUC ' + (isFinite(v.auc) ? v.auc.toFixed(2) : 'n/a') + '.' : 'Prequential validation not yet possible (needs more than 15 sessions with both outcomes).');
    const fw = Store.list('predictions').filter(p => !p.simulated && p.outcome != null); parts[1][1].push(fw.length ? fw.length + ' forward predictions resolved; Brier ' + S.brier(fw.map(p => p.p), fw.map(p => p.outcome)).toFixed(3) + '.' : 'No resolved forward predictions.');
    Store.list('experiments').filter(x => !x.simulated).forEach(x => parts.push(...experimentWriteup(x, LL.analyseExperiment(x))));
    return parts;
  };
})();
