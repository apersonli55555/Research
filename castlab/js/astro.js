/* CastLab astronomy + fishing-time index — pure functions (browser: window.CLAstro, Node: module.exports).
   Sun and moon positions use the standard low-precision formulas from "Astronomical Algorithms" (Meeus) as
   popularised by the SunCalc library (accuracy: about a minute for the sun, a few minutes for the moon). */
(function (root) {
  'use strict';
  const rad = Math.PI / 180, DAY = 864e5, J1970 = 2440588, J2000 = 2451545, e = rad * 23.4397;
  const toDays = ms => ms / DAY - 0.5 + J1970 - J2000;
  const ra = (l, b) => Math.atan2(Math.sin(l) * Math.cos(e) - Math.tan(b) * Math.sin(e), Math.cos(l));
  const dec = (l, b) => Math.asin(Math.sin(b) * Math.cos(e) + Math.cos(b) * Math.sin(e) * Math.sin(l));
  const alt = (H, phi, d) => Math.asin(Math.sin(phi) * Math.sin(d) + Math.cos(phi) * Math.cos(d) * Math.cos(H));
  const sidereal = (d, lw) => rad * (280.16 + 360.9856235 * d) - lw;
  function sunCoords(d) { const M = rad * (357.5291 + 0.98560028 * d), C = rad * (1.9148 * Math.sin(M) + 0.02 * Math.sin(2 * M) + 0.0003 * Math.sin(3 * M)), L = M + C + rad * 102.9372 + Math.PI; return { dec: dec(L, 0), ra: ra(L, 0) }; }
  function moonCoords(d) { const L = rad * (218.316 + 13.176396 * d), M = rad * (134.963 + 13.064993 * d), F = rad * (93.272 + 13.22935 * d), l = L + rad * 6.289 * Math.sin(M), b = rad * 5.128 * Math.sin(F); return { ra: ra(l, b), dec: dec(l, b), dist: 385001 - 20905 * Math.cos(M) }; }
  /* Altitudes in degrees at a UTC timestamp (ms). */
  function sunAltitude(ms, lat, lon) { const d = toDays(ms), c = sunCoords(d); return alt(sidereal(d, -lon * rad) - c.ra, lat * rad, c.dec) / rad; }
  function moonAltitude(ms, lat, lon) { const d = toDays(ms), c = moonCoords(d); return alt(sidereal(d, -lon * rad) - c.ra, lat * rad, c.dec) / rad; }
  const PHASES = ['New moon', 'Waxing crescent', 'First quarter', 'Waxing gibbous', 'Full moon', 'Waning gibbous', 'Last quarter', 'Waning crescent'];
  function moonPhase(ms) {
    const d = toDays(ms), s = sunCoords(d), m = moonCoords(d), sd = 149598000;
    const phi = Math.acos(Math.sin(s.dec) * Math.sin(m.dec) + Math.cos(s.dec) * Math.cos(m.dec) * Math.cos(s.ra - m.ra)), inc = Math.atan2(sd * Math.sin(phi), m.dist - sd * Math.cos(phi));
    const ang = Math.atan2(Math.cos(s.dec) * Math.sin(s.ra - m.ra), Math.sin(s.dec) * Math.cos(m.dec) - Math.cos(s.dec) * Math.sin(m.dec) * Math.cos(s.ra - m.ra));
    const phase = 0.5 + 0.5 * inc * (ang < 0 ? -1 : 1) / Math.PI;
    return { fraction: (1 + Math.cos(inc)) / 2, phase, name: PHASES[Math.round(phase * 8) % 8] };
  }
  /* Sun and moon events in [startMs, startMs + hours h): threshold crossings by 5-minute scan with linear
     interpolation; moon transit (overhead) and underfoot are the local altitude maximum and minimum. */
  function events(startMs, lat, lon, hours) {
    const step = 5 * 60000, n = Math.round((hours || 24) * 12), out = { sunrise: [], sunset: [], moonrise: [], moonset: [], transit: [], underfoot: [] };
    const S0 = -0.833, M0 = 0.133; let ps = sunAltitude(startMs - step, lat, lon) - S0, pm = moonAltitude(startMs - step, lat, lon), pm2 = moonAltitude(startMs - 2 * step, lat, lon);
    for (let k = 0; k <= n; k++) {
      const t = startMs + k * step, s = sunAltitude(t, lat, lon) - S0, m = moonAltitude(t, lat, lon), cross = (a, b) => t - step + step * (a / (a - b));
      if (ps < 0 && s >= 0) out.sunrise.push(cross(ps, s)); if (ps >= 0 && s < 0) out.sunset.push(cross(ps, s));
      if (pm - M0 < 0 && m - M0 >= 0) out.moonrise.push(cross(pm - M0, m - M0)); if (pm - M0 >= 0 && m - M0 < 0) out.moonset.push(cross(pm - M0, m - M0));
      if (pm > pm2 && pm >= m) out.transit.push(t - step); if (pm < pm2 && pm <= m) out.underfoot.push(t - step);
      ps = s; pm2 = pm; pm = m;
    }
    return out;
  }

  /* Fishing-time index for one hour. A transparent rule of thumb, NOT a validated prediction.
     h: { ms (hour start, UTC), dPressure3h (hPa), windKph, precipMm, code (WMO), ev (events covering the hour) }.
     Returns each component in [0,1], the weighted index 0-100 and plain-language reasons. */
  const W = { light: 0.35, pressure: 0.25, wind: 0.20, solunar: 0.20 };
  function near(ms, list, full, zero) { let best = 0; for (const t of list) { const d = Math.abs(ms - t) / 60000; best = Math.max(best, d <= full ? 1 : d >= zero ? 0 : 1 - (d - full) / (zero - full)); } return best; }
  function scoreHour(h) {
    const mid = h.ms + 30 * 60000, ev = h.ev, why = [], c = {};
    const dawn = near(mid, ev.sunrise, 60, 150), dusk = near(mid, ev.sunset, 60, 150);
    c.light = Math.max(dawn, dusk, h.sunAlt != null && h.sunAlt < -6 ? 0.25 : 0.1); if (dawn > 0.6) why.push('around sunrise'); else if (dusk > 0.6) why.push('around sunset');
    const major = Math.max(near(mid, ev.transit, 60, 120), near(mid, ev.underfoot, 60, 120)), minor = Math.max(near(mid, ev.moonrise, 30, 75), near(mid, ev.moonset, 30, 75));
    c.solunar = Math.max(major, 0.6 * minor); if (major > 0.6) why.push('moon overhead/underfoot'); else if (minor > 0.6) why.push('moonrise/moonset');
    const dp = h.dPressure3h;
    if (dp == null || !isFinite(dp)) c.pressure = 0.5; else if (dp <= -4) { c.pressure = 0.3; why.push('pressure dropping fast (front arriving)'); } else if (dp <= -0.5) { c.pressure = 1; why.push('falling pressure'); } else if (dp < 0.5) c.pressure = 0.65; else if (dp < 2.5) c.pressure = 0.45; else { c.pressure = 0.3; why.push('pressure rising quickly'); }
    const w = h.windKph;
    if (w == null || !isFinite(w)) c.wind = 0.5; else if (w < 4) c.wind = 0.6; else if (w <= 20) { c.wind = 1; if (w >= 8) why.push('light chop'); } else if (w <= 35) { c.wind = 0.4; why.push('windy'); } else { c.wind = 0; why.push('very windy'); }
    let index = 100 * (W.light * c.light + W.pressure * c.pressure + W.wind * c.wind + W.solunar * c.solunar), unsafe = false;
    if (h.code >= 95) { index = 0; unsafe = true; why.length = 0; why.push('thunderstorm: stay off the water'); } else if (h.precipMm >= 4) { index *= 0.6; why.push('heavy rain'); }
    return { index: Math.round(index), components: c, why, unsafe };
  }
  /* Group consecutive hours scoring >= threshold into windows, best first. hours: [{ms, index, why}] */
  function windows(hours, threshold) {
    const out = []; let cur = null;
    hours.forEach(h => { if (h.index >= threshold) { if (cur && h.ms - cur.end === 0) { cur.end = h.ms + 36e5; cur.peak = Math.max(cur.peak, h.index); h.why.forEach(w => cur.why.add(w)); } else { cur = { start: h.ms, end: h.ms + 36e5, peak: h.index, why: new Set(h.why) }; out.push(cur); } } else cur = null; });
    return out.map(w => Object.assign(w, { why: Array.from(w.why) })).sort((a, b) => b.peak - a.peak);
  }

  root.CLAstro = { sunAltitude, moonAltitude, moonPhase, events, scoreHour, windows, WEIGHTS: W };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.CLAstro;
})(typeof window !== 'undefined' ? window : globalThis);
