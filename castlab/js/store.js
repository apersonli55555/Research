/* CastLab shared data layer: local user accounts and one profile-scoped data store used by every module
   (map, weather, BaitRadar, GearMatch, LureLab, CastCheck twin, physics bench tests, dashboard).
   Everything lives in this browser's localStorage. There is no server: accounts are local profiles,
   and data moves between devices by export/import. */
(function (root) {
  'use strict';
  const K_USERS = 'castlab.users.v1', K_SESSION = 'castlab.session.v1', SCHEMA = 1;
  const COLLECTIONS = ['locations', 'sessions', 'inventory', 'experiments', 'trials', 'predictions', 'observations', 'benchTests', 'rods', 'catches'];
  const listeners = new Set();
  let mem = {}; // fallback when localStorage is unavailable (private mode etc.)

  function lsGet(k) { try { const v = localStorage.getItem(k); return v == null ? mem[k] : v; } catch (e) { return mem[k]; } }
  function lsSet(k, v) { mem[k] = v; try { localStorage.setItem(k, v); } catch (e) { /* quota or disabled: kept in memory for this tab */ } }
  function lsDel(k) { delete mem[k]; try { localStorage.removeItem(k); } catch (e) {} }
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const users = () => JSON.parse(lsGet(K_USERS) || '[]');

  async function hashPw(pw, saltHex) {
    const enc = new TextEncoder(), salt = saltHex ? Uint8Array.from(saltHex.match(/../g).map(h => parseInt(h, 16))) : crypto.getRandomValues(new Uint8Array(16));
    const key = await crypto.subtle.importKey('raw', enc.encode(pw), 'PBKDF2', false, ['deriveBits']);
    const bits = await crypto.subtle.deriveBits({ name: 'PBKDF2', salt, iterations: 150000, hash: 'SHA-256' }, key, 256);
    const hex = a => Array.from(new Uint8Array(a)).map(b => b.toString(16).padStart(2, '0')).join('');
    return { salt: hex(salt), hash: hex(bits) };
  }

  const Store = {
    user: null, data: null,
    emptyData() { const d = { schema: SCHEMA, prefs: { units: 'imperial', homeLocation: null } }; COLLECTIONS.forEach(c => d[c] = []); return d; },
    async register(username, password, displayName) {
      username = (username || '').trim().toLowerCase();
      if (!/^[a-z0-9_.-]{3,32}$/.test(username)) throw new Error('Username: 3–32 letters, digits, dot, dash or underscore.');
      if ((password || '').length < 8) throw new Error('Password must be at least 8 characters.');
      const u = users(); if (u.some(x => x.username === username)) throw new Error('That username already exists on this device.');
      const h = window.crypto && crypto.subtle ? await hashPw(password) : { salt: '', hash: 'nocrypto:' + password.length };
      u.push({ id: uid(), username, displayName: displayName || username, created: new Date().toISOString(), salt: h.salt, hash: h.hash });
      lsSet(K_USERS, JSON.stringify(u));
      return this.login(username, password);
    },
    async login(username, password) {
      username = (username || '').trim().toLowerCase();
      const u = users().find(x => x.username === username); if (!u) throw new Error('No account with that username on this device.');
      if (u.guest) throw new Error('Guest profiles have no password. Use "Explore with example data" instead.');
      if (u.salt) { const h = await hashPw(password, u.salt); if (h.hash !== u.hash) throw new Error('Incorrect password.'); }
      this.user = { id: u.id, username: u.username, displayName: u.displayName, created: u.created };
      lsSet(K_SESSION, u.id); this.load(); return this.user;
    },
    guest() {
      const u = users(), id = uid(), username = 'guest-' + id.slice(-5);
      u.push({ id, username, displayName: 'Guest', created: new Date().toISOString(), salt: '', hash: '', guest: true });
      lsSet(K_USERS, JSON.stringify(u)); this.user = { id, username, displayName: 'Guest', created: new Date().toISOString(), guest: true };
      lsSet(K_SESSION, id); this.data = this.emptyData(); this.data.prefs.guest = true; return this.user;
    },
    resume() { const id = lsGet(K_SESSION); if (!id) return null; const u = users().find(x => x.id === id); if (!u) return null; this.user = { id: u.id, username: u.username, displayName: u.displayName, created: u.created }; this.load(); return this.user; },
    logout() { lsDel(K_SESSION); this.user = null; this.data = null; this.emit('auth'); },
    deleteAccount() { if (!this.user) return; lsSet(K_USERS, JSON.stringify(users().filter(x => x.id !== this.user.id))); lsDel('castlab.data.' + this.user.id); this.logout(); },
    load() {
      let d = null; try { d = JSON.parse(lsGet('castlab.data.' + this.user.id) || 'null'); } catch (e) {}
      this.data = Object.assign(this.emptyData(), d || {}); COLLECTIONS.forEach(c => { if (!Array.isArray(this.data[c])) this.data[c] = []; });
      this.emit('auth');
    },
    save(topic) { if (!this.user) return; lsSet('castlab.data.' + this.user.id, JSON.stringify(this.data)); this.emit(topic || 'data'); },
    list(c) { return this.data ? this.data[c] : []; },
    get(c, id) { return this.list(c).find(x => x.id === id); },
    add(c, obj) { obj = Object.assign({ id: uid(), created: new Date().toISOString() }, obj); this.data[c].push(obj); this.save(c); return obj; },
    update(c, id, patch) { const o = this.get(c, id); if (o) { Object.assign(o, patch, { modified: new Date().toISOString() }); this.save(c); } return o; },
    remove(c, id) { this.data[c] = this.data[c].filter(x => x.id !== id); this.save(c); },
    removeWhere(c, pred) { const n = this.data[c].length; this.data[c] = this.data[c].filter(x => !pred(x)); this.save(c); return n - this.data[c].length; },
    on(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    emit(topic) { listeners.forEach(fn => { try { fn(topic); } catch (e) { console.error(e); } }); },
    exportAll() { return { app: 'CastLab', schema: SCHEMA, exported: new Date().toISOString(), user: this.user && { username: this.user.username, displayName: this.user.displayName }, data: this.data }; },
    importAll(obj, mode) {
      if (!obj || !obj.data) throw new Error('Not a CastLab export file.');
      if (mode === 'replace') this.data = Object.assign(this.emptyData(), obj.data);
      else COLLECTIONS.forEach(c => { const have = new Set(this.data[c].map(x => x.id)); (obj.data[c] || []).forEach(x => { if (!have.has(x.id)) this.data[c].push(x); }); });
      this.save('import');
    },
    location(id) { return this.get('locations', id); },
    locationName(id) { const l = this.location(id); return l ? l.name : '—'; },
    uid, COLLECTIONS
  };

  /* ---------- shared helpers used by all modules ---------- */
  const U = {
    esc: s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])),
    $: (sel, el) => (el || document).querySelector(sel),
    $$: (sel, el) => Array.from((el || document).querySelectorAll(sel)),
    fmt: (v, d) => (v == null || !isFinite(v)) ? '—' : (+v).toFixed(d == null ? 2 : d),
    pct: (v, d) => (v == null || !isFinite(v)) ? '—' : (100 * v).toFixed(d == null ? 0 : d) + '%',
    download(name, text, type) { const b = new Blob([text], { type: type || 'text/plain' }), a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); },
    downloadDataURL(name, url) { const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); },
    toCSV(rows, cols) {
      if (!rows.length) return (cols || []).join(',') + '\n';
      cols = cols || Array.from(rows.reduce((s, r) => { Object.keys(r).forEach(k => s.add(k)); return s; }, new Set()));
      const q = v => { if (v == null) return ''; if (typeof v === 'object') v = JSON.stringify(v); v = String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
      return cols.join(',') + '\n' + rows.map(r => cols.map(c => q(r[c])).join(',')).join('\n') + '\n';
    },
    flatSession(s) { const w = s.weather || {}; return { id: s.id, date: s.date, location: Store.locationName(s.locationId), lat: s.lat, lon: s.lon, species: s.species, lure: s.lure, lureColor: s.lureColor, hours: s.hours, catches: s.catches, success: s.catches > 0 ? 1 : 0, timeOfDay: s.timeOfDay, airTempC: w.tempC, windKph: w.windKph, pressureHpa: w.pressureHpa, pressureTrend: w.pressureTrend, cloudPct: w.cloudPct, precipMm: w.precipMm, waterTempC: s.waterTempC, clarity: s.clarity, weatherSource: w.source, dataSource: s.source || 'user', experimentId: s.experimentId || '', notes: s.notes }; },
    lbToN: lb => lb * 4.4482216, nToLb: n => n / 4.4482216, ozToG: oz => oz * 28.349523, gToOz: g => g / 28.349523,
    today: () => new Date().toISOString().slice(0, 10),
    toast(msg, kind) { let t = document.getElementById('toast'); if (!t) return; t.textContent = msg; t.className = 'toast show ' + (kind || ''); clearTimeout(t._h); t._h = setTimeout(() => t.className = 'toast', 3500); },
    haversine(lat1, lon1, lat2, lon2) { const R = 6371000, r = Math.PI / 180, dl = (lat2 - lat1) * r, dn = (lon2 - lon1) * r, a = Math.sin(dl / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dn / 2) ** 2; return 2 * R * Math.asin(Math.sqrt(a)); },
    cssVar: n => getComputedStyle(document.documentElement).getPropertyValue(n).trim(),
    chartColors() { return ['--c1', '--c2', '--c3', '--c4', '--c5', '--c6'].map(U.cssVar); },
    /* Register a Chart.js chart in a slot so it is destroyed before being re-drawn. */
    chart(slot, canvas, cfg) { U._charts = U._charts || {}; if (U._charts[slot]) U._charts[slot].destroy(); if (!canvas || !window.Chart) return null; const c = new Chart(canvas, cfg); U._charts[slot] = c; return c; },
    chartPNG(slot, name) { const c = U._charts && U._charts[slot]; if (!c) return U.toast('Draw the graph first.'); U.downloadDataURL(name, c.toBase64Image('image/png', 1)); },
    chartCSV(slot, name) { const c = U._charts && U._charts[slot]; if (!c) return U.toast('Draw the graph first.'); const rows = []; c.data.datasets.forEach(ds => (ds.data || []).forEach((p, i) => rows.push({ series: ds.label, x: typeof p === 'object' ? p.x : c.data.labels[i], y: typeof p === 'object' ? p.y : p }))); U.download(name, U.toCSV(rows, ['series', 'x', 'y']), 'text/csv'); }
  };
  /* Weather from Open-Meteo (no key). Returns conditions at a given local date/hour, or current. Also the 3-hour pressure trend. */
  U.fetchWeather = async function (lat, lon, dateStr, hour) {
    const today = U.today(), past = dateStr && dateStr < today ? Math.ceil((Date.parse(today) - Date.parse(dateStr)) / 864e5) + 1 : 1;
    if (past > 92) throw new Error('Open-Meteo forecast API only covers the last 92 days; enter weather manually.');
    const url = 'https://api.open-meteo.com/v1/forecast?latitude=' + lat.toFixed(4) + '&longitude=' + lon.toFixed(4) + '&hourly=temperature_2m,wind_speed_10m,surface_pressure,cloud_cover,precipitation,weather_code&past_days=' + past + '&forecast_days=7&timezone=auto';
    const ck = 'castlab.wx.' + lat.toFixed(3) + ',' + lon.toFixed(3) + ',' + past, fresh = !dateStr || dateStr >= today; let j = null;
    try { const c = JSON.parse(sessionStorage.getItem(ck) || 'null'); if (c && Date.now() - c.t < 15 * 60000) j = c.j; } catch (e) {}
    if (!j) {
    let r; for (let a = 0; a < 3; a++) { try { r = await fetch(url); if (r.ok || r.status === 429) break; } catch (e) { if (a === 2) throw new Error('the weather service did not answer'); } await new Promise(z => setTimeout(z, 700 * (a + 1))); }
    if (!r || !r.ok) throw new Error(r && r.status === 429 ? 'the weather service is busy, try again in a minute' : 'Open-Meteo HTTP ' + (r ? r.status : 'error')); j = await r.json();
      try { sessionStorage.setItem(ck, JSON.stringify({ t: Date.now(), j })); } catch (e) {}
    }
    const h = j.hourly;
    const target = (dateStr || today) + 'T' + String(hour == null ? new Date().getHours() : hour).padStart(2, '0') + ':00';
    let i = h.time.indexOf(target); if (i < 0) throw new Error('No weather for that hour.');
    const trend = i >= 3 ? h.surface_pressure[i] - h.surface_pressure[i - 3] : null;
    return { tempC: h.temperature_2m[i], windKph: h.wind_speed_10m[i], pressureHpa: h.surface_pressure[i], pressureTrend: trend == null ? null : trend > 1 ? 'rising' : trend < -1 ? 'falling' : 'steady', cloudPct: h.cloud_cover[i], precipMm: h.precipitation[i], code: h.weather_code[i], utcOffset: j.utc_offset_seconds || 0, source: 'Open-Meteo (' + target + ' ' + j.timezone + ')', hourly: h, index: i, timezone: j.timezone };
  };
  U.WMO = { 0: 'Clear', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast', 45: 'Fog', 48: 'Rime fog', 51: 'Light drizzle', 53: 'Drizzle', 55: 'Dense drizzle', 61: 'Light rain', 63: 'Rain', 65: 'Heavy rain', 71: 'Light snow', 73: 'Snow', 75: 'Heavy snow', 80: 'Rain showers', 81: 'Rain showers', 82: 'Violent showers', 95: 'Thunderstorm', 96: 'Thunderstorm, hail', 99: 'Thunderstorm, hail' };

  /* USGS Water Services: nearest active gage-height sites, latest value and daily history. */
  U.usgsSites = async function (lat, lon, deg) {
    deg = deg || 0.25; const bb = [lon - deg, lat - deg, lon + deg, lat + deg].map(v => v.toFixed(4)).join(',');
    const r = await fetch('https://waterservices.usgs.gov/nwis/site/?format=rdb&bBox=' + bb + '&parameterCd=00065&siteStatus=active&hasDataTypeCd=iv&siteType=ST,LK,ES');
    if (r.status === 404) return []; if (!r.ok) throw new Error('USGS site service HTTP ' + r.status);
    const lines = (await r.text()).split('\n').filter(l => l && l[0] !== '#'); if (lines.length < 3) return [];
    const head = lines[0].split('\t'), ix = n => head.indexOf(n);
    return lines.slice(2).map(l => { const c = l.split('\t'); return { site: c[ix('site_no')], name: c[ix('station_nm')], type: c[ix('site_tp_cd')], lat: +c[ix('dec_lat_va')], lon: +c[ix('dec_long_va')] }; })
      .filter(s => s.site && isFinite(s.lat)).map(s => Object.assign(s, { distKm: U.haversine(lat, lon, s.lat, s.lon) / 1000 })).sort((a, b) => a.distKm - b.distKm);
  };
  U.usgsSeries = async function (site, service, period) {
    const r = await fetch('https://waterservices.usgs.gov/nwis/' + service + '/?format=json&sites=' + site + '&parameterCd=00065&' + period); if (!r.ok) throw new Error('USGS ' + service + ' HTTP ' + r.status);
    const j = await r.json(), ts = j.value.timeSeries[0]; if (!ts) return { values: [] };
    const nd = ts.variable.noDataValue, unit = ts.variable.unit.unitCode;
    const values = ts.values[0].value.map(v => ({ t: v.dateTime, v: +v.value, q: v.qualifiers })).filter(v => isFinite(v.v) && v.v !== nd);
    return { values, unit, name: ts.sourceInfo.siteName };
  };

  root.Store = Store; root.U = U;
})(window);
