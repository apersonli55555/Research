/* CastLab terrain engine — pure functions on elevation grids (Float32Array, row-major, row 0 = north).
   Used by the Shoreline Digital Twin. No DOM, no network. */
(function (root) {
  'use strict';
  const R_EARTH = 6378137;

  // ---------- Web-Mercator tile maths ----------
  function lonLatToPixel(lon, lat, z) { const n = 256 * Math.pow(2, z), s = Math.sin(lat * Math.PI / 180); return { x: (lon + 180) / 360 * n, y: (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * n }; }
  function pixelToLonLat(x, y, z) { const n = 256 * Math.pow(2, z); return { lon: x / n * 360 - 180, lat: Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI }; }
  function metersPerPixel(lat, z) { return 2 * Math.PI * R_EARTH * Math.cos(lat * Math.PI / 180) / (256 * Math.pow(2, z)); }
  /* Terrarium encoding (Mapzen / AWS Terrain Tiles): elevation_m = R*256 + G + B/256 - 32768. */
  function decodeTerrarium(rgba, n) { const z = new Float32Array(n); for (let i = 0; i < n; i++) z[i] = rgba[4 * i] * 256 + rgba[4 * i + 1] + rgba[4 * i + 2] / 256 - 32768; return z; }

  // ---------- synthetic demonstration terrain (clearly labelled in the UI; includes "true" bathymetry) ----------
  function syntheticTerrain(w, h, cell, seed) {
    let a = (seed >>> 0) || 7; const rnd = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const waves = []; for (let k = 0; k < 14; k++) waves.push({ fx: (rnd() - 0.5) * 0.09, fy: (rnd() - 0.5) * 0.09, ph: rnd() * 6.28, amp: 1.5 + rnd() * 3 });
    const z = new Float32Array(w * h), cx = w * 0.5, cy = h * 0.52;
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const dx = (i - cx) / (w * 0.30), dy = (j - cy) / (h * 0.22), r = Math.sqrt(dx * dx + dy * dy);
      let v = 100 + 14 * (r - 1) + 4 * Math.max(0, r - 1.6) * Math.max(0, r - 1.6); // bowl: rim at r=1 -> z=100
      for (const q of waves) v += q.amp * 0.35 * Math.sin(i * q.fx * 2 + j * q.fy * 2 + q.ph);
      z[j * w + i] = v;
    }
    return { z, w, h, cell, synthetic: true };
  }

  // ---------- analysis ----------
  /* Slope in degrees by Horn's (1981) 3x3 finite-difference method. */
  function slopeDeg(z, w, h, cell) {
    const out = new Float32Array(w * h), g = (i, j) => z[Math.min(h - 1, Math.max(0, j)) * w + Math.min(w - 1, Math.max(0, i))];
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const dzdx = ((g(i + 1, j - 1) + 2 * g(i + 1, j) + g(i + 1, j + 1)) - (g(i - 1, j - 1) + 2 * g(i - 1, j) + g(i - 1, j + 1))) / (8 * cell);
      const dzdy = ((g(i - 1, j + 1) + 2 * g(i, j + 1) + g(i + 1, j + 1)) - (g(i - 1, j - 1) + 2 * g(i, j - 1) + g(i + 1, j - 1))) / (8 * cell);
      out[j * w + i] = Math.atan(Math.sqrt(dzdx * dzdx + dzdy * dzdy)) * 180 / Math.PI;
    }
    return out;
  }
  /* Detect the water surface a DEM recorded: the largest 4-connected patch of cells that are level with all
     neighbours (|dz| < tol). Returns { mask, level (median elevation), cells } or null when no flat patch >= minCells. */
  function detectWater(z, w, h, tol, minCells) {
    tol = tol == null ? 0.15 : tol; minCells = minCells || 40;
    const flat = new Uint8Array(w * h);
    for (let j = 1; j < h - 1; j++) for (let i = 1; i < w - 1; i++) { const c = z[j * w + i]; let ok = 1; for (let dj = -1; dj <= 1 && ok; dj++) for (let di = -1; di <= 1; di++) if (Math.abs(z[(j + dj) * w + i + di] - c) > tol) { ok = 0; break; } flat[j * w + i] = ok; }
    const lab = new Int32Array(w * h); let best = 0, bestN = 0, id = 0; const stack = [];
    for (let s = 0; s < w * h; s++) {
      if (!flat[s] || lab[s]) continue; id++; let n = 0; stack.push(s); lab[s] = id;
      while (stack.length) { const p = stack.pop(); n++; const i = p % w, j = (p - i) / w; const nb = [i > 0 ? p - 1 : -1, i < w - 1 ? p + 1 : -1, j > 0 ? p - w : -1, j < h - 1 ? p + w : -1]; for (const q of nb) if (q >= 0 && flat[q] && !lab[q] && Math.abs(z[q] - z[s]) < 4 * tol + 0.5) { lab[q] = id; stack.push(q); } }
      if (n > bestN) { bestN = n; best = id; }
    }
    if (bestN < minCells) return null;
    const vals = []; for (let s = 0; s < w * h; s++) if (lab[s] === best) vals.push(z[s]);
    vals.sort((x, y) => x - y); const level = vals[vals.length >> 1], mask = new Uint8Array(w * h);
    // include the 1-cell fringe that is level with the patch (the erosion above removes it)
    for (let s = 0; s < w * h; s++) if (lab[s] === best) mask[s] = 1;
    for (let pass = 0; pass < 2; pass++) { const add = [];
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) { const s = j * w + i; if (!mask[s] && Math.abs(z[s] - level) <= tol && ((i > 0 && mask[s - 1]) || (i < w - 1 && mask[s + 1]) || (j > 0 && mask[s - w]) || (j < h - 1 && mask[s + w]))) add.push(s); }
      for (const s of add) mask[s] = 1; }
    let cells = 0; for (let s = 0; s < w * h; s++) cells += mask[s];
    return { mask, level, cells };
  }
  /* Water mask from a chosen level on a surface that already has bathymetry (synthetic demo): connected region below level containing the lowest cell. */
  function basinMask(z, w, h, level) { let lo = 0; for (let s = 1; s < w * h; s++) if (z[s] < z[lo]) lo = s; return flood(z, w, h, level, [lo]); }
  /* 4-connected flood: cells reachable from seeds through cells with z <= level ("bathtub with connectivity"). */
  function flood(z, w, h, level, seeds) {
    const wet = new Uint8Array(w * h), stack = [];
    for (const s of seeds) if (z[s] <= level && !wet[s]) { wet[s] = 1; stack.push(s); }
    while (stack.length) { const p = stack.pop(), i = p % w, j = (p - i) / w;
      if (i > 0 && !wet[p - 1] && z[p - 1] <= level) { wet[p - 1] = 1; stack.push(p - 1); }
      if (i < w - 1 && !wet[p + 1] && z[p + 1] <= level) { wet[p + 1] = 1; stack.push(p + 1); }
      if (j > 0 && !wet[p - w] && z[p - w] <= level) { wet[p - w] = 1; stack.push(p - w); }
      if (j < h - 1 && !wet[p + w] && z[p + w] <= level) { wet[p + w] = 1; stack.push(p + w); } }
    return wet;
  }
  /* Median slope (deg) of dry cells within `ring` cells of the water mask = representative bank slope. */
  function bankSlope(slope, mask, w, h, ring) {
    ring = ring || 2; const v = [];
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) { const s = j * w + i; if (mask[s]) continue; let near = false; for (let dj = -ring; dj <= ring && !near; dj++) for (let di = -ring; di <= ring; di++) { const ii = i + di, jj = j + dj; if (ii >= 0 && jj >= 0 && ii < w && jj < h && mask[jj * w + ii]) { near = true; break; } } if (near) v.push(slope[s]); }
    if (!v.length) return { median: NaN, n: 0 };
    v.sort((a, b) => a - b); return { median: v[v.length >> 1], p25: v[Math.floor(v.length * 0.25)], p75: v[Math.floor(v.length * 0.75)], n: v.length };
  }
  /* DEMs store the water surface, not the bed. This builds an EXTRAPOLATED bed for drawdown scenarios by
     continuing the measured bank slope under water: bed = level - min(maxDepth, distanceToShore * tan(slope)).
     It is an assumption, not a measurement; the UI labels it as such. */
  function extrapolateBed(z, mask, w, h, cell, level, slopeDegVal, maxDepth) {
    const dist = new Float32Array(w * h).fill(-1), q = [];
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) { const s = j * w + i; if (!mask[s]) continue; if ((i > 0 && !mask[s - 1]) || (i < w - 1 && !mask[s + 1]) || (j > 0 && !mask[s - w]) || (j < h - 1 && !mask[s + w])) { dist[s] = 0.5; q.push(s); } }
    for (let k = 0; k < q.length; k++) { const p = q[k], i = p % w, j = (p - i) / w; const nb = [i > 0 ? p - 1 : -1, i < w - 1 ? p + 1 : -1, j > 0 ? p - w : -1, j < h - 1 ? p + w : -1]; for (const n of nb) if (n >= 0 && mask[n] && dist[n] < 0) { dist[n] = dist[p] + 1; q.push(n); } }
    const out = new Float32Array(z), t = Math.tan(slopeDegVal * Math.PI / 180);
    for (let s = 0; s < w * h; s++) if (mask[s]) out[s] = level - Math.min(maxDepth, Math.max(0, dist[s]) * cell * t);
    return out;
  }
  /* Wet mask for a water level on surface z: flood from every reference-water cell that is still below `level`. */
  function wetAt(z, w, h, level, refMask) { const seeds = []; for (let s = 0; s < w * h; s++) if (refMask[s] && z[s] <= level) seeds.push(s); return flood(z, w, h, level, seeds); }
  /* Shoreline as line segments (marching squares on the wet mask, grid units) and its length in metres. */
  function shoreline(wet, w, h, cell) {
    const seg = []; let len = 0;
    for (let j = 0; j < h - 1; j++) for (let i = 0; i < w - 1; i++) {
      const a = wet[j * w + i], b = wet[j * w + i + 1], c = wet[(j + 1) * w + i + 1], d = wet[(j + 1) * w + i], code = a * 8 + b * 4 + c * 2 + d;
      if (code === 0 || code === 15) continue;
      const T = [i + 0.5, j], Rr = [i + 1, j + 0.5], B = [i + 0.5, j + 1], Lf = [i, j + 0.5], push = (p, q) => { seg.push(p[0], p[1], q[0], q[1]); len += Math.hypot(p[0] - q[0], p[1] - q[1]); };
      switch (code) { case 1: case 14: push(Lf, B); break; case 2: case 13: push(B, Rr); break; case 3: case 12: push(Lf, Rr); break; case 4: case 11: push(T, Rr); break; case 6: case 9: push(T, B); break; case 7: case 8: push(Lf, T); break; case 5: push(Lf, T); push(B, Rr); break; case 10: push(T, Rr); push(Lf, B); break; }
    }
    return { segments: seg, length: len * cell };
  }
  function area(mask, cell) { let n = 0; for (let s = 0; s < mask.length; s++) n += mask[s]; return n * cell * cell; }
  function bilinear(z, w, h, x, y) { x = Math.min(w - 1.001, Math.max(0, x)); y = Math.min(h - 1.001, Math.max(0, y)); const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j, s = j * w + i; return z[s] * (1 - fx) * (1 - fy) + z[s + 1] * fx * (1 - fy) + z[s + w] * (1 - fx) * fy + z[s + w + 1] * fx * fy; }
  /* Elevation profile along a transect from A to B (grid coords). */
  function transect(z, w, h, cell, ax, ay, bx, by, n) {
    n = n || 200; const len = Math.hypot(bx - ax, by - ay) * cell, d = [], e = [];
    for (let k = 0; k <= n; k++) { const t = k / n; d.push(t * len); e.push(bilinear(z, w, h, ax + (bx - ax) * t, ay + (by - ay) * t)); }
    return { dist: d, elev: e, length: len };
  }
  /* Distance along a profile (from its start) to the first place the ground drops to the water level; null if never. */
  function waterlineOnProfile(p, level) {
    for (let k = 1; k < p.elev.length; k++) { const a = p.elev[k - 1] - level, b = p.elev[k] - level; if ((a > 0 && b <= 0) || (a <= 0 && b > 0)) return p.dist[k - 1] + (p.dist[k] - p.dist[k - 1]) * (a / (a - b)); }
    return null;
  }
  /* Shortest distance (m) from grid point (x,y) to a set of shoreline segments. */
  function distToShoreline(segments, x, y, cell) {
    let best = Infinity;
    for (let k = 0; k < segments.length; k += 4) { const x1 = segments[k], y1 = segments[k + 1], x2 = segments[k + 2], y2 = segments[k + 3], dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy; let t = l2 ? ((x - x1) * dx + (y - y1) * dy) / l2 : 0; t = Math.max(0, Math.min(1, t)); const d = Math.hypot(x - x1 - t * dx, y - y1 - t * dy); if (d < best) best = d; }
    return best * cell;
  }
  function hillshade(z, w, h, cell, azDeg, altDeg, zf) {
    const az = (azDeg == null ? 315 : azDeg) * Math.PI / 180, alt = (altDeg == null ? 45 : altDeg) * Math.PI / 180, out = new Float32Array(w * h); zf = zf || 1;
    const g = (i, j) => z[Math.min(h - 1, Math.max(0, j)) * w + Math.min(w - 1, Math.max(0, i))];
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) { const dzdx = (g(i + 1, j) - g(i - 1, j)) / (2 * cell) * zf, dzdy = (g(i, j + 1) - g(i, j - 1)) / (2 * cell) * zf; const sl = Math.atan(Math.hypot(dzdx, dzdy)), asp = Math.atan2(dzdy, -dzdx); out[j * w + i] = Math.max(0, Math.cos(Math.PI / 2 - alt) * Math.cos(sl) + Math.sin(Math.PI / 2 - alt) * Math.sin(sl) * Math.cos(az - Math.PI / 2 - asp)); }
    return out;
  }

  root.CLTerrain = { lonLatToPixel, pixelToLonLat, metersPerPixel, decodeTerrarium, syntheticTerrain, slopeDeg, detectWater, basinMask, flood, bankSlope, extrapolateBed, wetAt, shoreline, area, bilinear, transect, waterlineOnProfile, distToShoreline, hillshade };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.CLTerrain;
})(typeof window !== 'undefined' ? window : globalThis);
