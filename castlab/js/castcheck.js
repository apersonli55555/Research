/* CastCheck shoreline digital twin. ES module (needs three.js). Terrain maths lives in terrain.js (CLTerrain). */
import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';

const { $, $$, esc, fmt } = U, T = CLTerrain, S = CLStats;
const CC = CL.modules.castcheck = {};
const ZOOM = 15, TARGET = 320;
let st = null;            // current terrain state
let showTab = null, three = null, stage = null, animH = null;

/* ================= loading ================= */
CC.render = function (sub) {
  if (!showTab) showTab = CL.tabs('#ccTabs', '.cctab', t => ({ twin: redraw, water: renderWater, obs: renderObs, methods: renderMethods })[t]());
  const sel = $('#ccLoc'), prev = CL.pickLoc || sel.value || Store.data.prefs.homeLocation; CL.locationOptions(sel); if (prev && Store.location(prev)) sel.value = prev; const picked = CL.pickLoc; CL.pickLoc = null;
  if (!$('#obsTime').value) $('#obsTime').value = new Date().toISOString().slice(0, 16).replace('T', ' ');
  showTab(sub || ($('#ccTabs .active') || {}).dataset.t || 'twin');
  if (picked && (!st || st.locId !== picked)) loadReal();
  else if (!st && Store.location($('#ccLoc').value) && isFinite(Store.location($('#ccLoc').value).lat)) loadReal();
};
CC.tab = t => showTab && showTab(t);

function loadImage(url) { return new Promise((res, rej) => { const im = new Image(); im.crossOrigin = 'anonymous'; im.onload = () => res(im); im.onerror = () => rej(new Error('tile failed: ' + url)); im.src = url; }); }
async function loadReal() {
  const loc = Store.location($('#ccLoc').value); if (!loc || !isFinite(loc.lat)) return U.toast('Choose a saved location with coordinates.');
  const n = +$('#ccSize').value, p = T.lonLatToPixel(loc.lon, loc.lat, ZOOM), tx0 = Math.floor(p.x / 256 - n / 2 + 0.5), ty0 = Math.floor(p.y / 256 - n / 2 + 0.5);
  $('#ccSource').innerHTML = 'Downloading ' + n * n + ' elevation tiles…';
  try {
    const cv = document.createElement('canvas'); cv.width = cv.height = 256 * n; const ctx = cv.getContext('2d', { willReadFrequently: true });
    await Promise.all(Array.from({ length: n * n }, (_, k) => { const i = k % n, j = Math.floor(k / n); return loadImage('https://s3.amazonaws.com/elevation-tiles-prod/terrarium/' + ZOOM + '/' + (tx0 + i) + '/' + (ty0 + j) + '.png').then(im => ctx.drawImage(im, i * 256, j * 256)); }));
    const N = 256 * n, raw = T.decodeTerrarium(ctx.getImageData(0, 0, N, N).data, N * N), f = Math.max(1, Math.round(N / TARGET)), w = Math.floor(N / f), z = new Float32Array(w * w);
    for (let j = 0; j < w; j++) for (let i = 0; i < w; i++) { let s = 0; for (let b = 0; b < f; b++) for (let a = 0; a < f; a++) s += raw[(j * f + b) * N + i * f + a]; z[j * w + i] = s / (f * f); }
    const cell = T.metersPerPixel(loc.lat, ZOOM) * f;
    setTerrain({ z, w, h: w, cell, synthetic: false, locId: loc.id, locName: loc.name, origin: { x: tx0 * 256, y: ty0 * 256, f }, center: { lat: loc.lat, lon: loc.lon } });
    findGages(true);
  } catch (err) { $('#ccSource').innerHTML = '<span class="status-fail">Could not load elevation tiles (' + esc(err.message) + '). An internet connection is required; try the synthetic demo offline.</span>'; }
}
$('#ccLoad').onclick = loadReal;
$('#ccDemo').onclick = () => { const g = T.syntheticTerrain(200, 200, 5, 42); setTerrain(Object.assign(g, { locId: null, locName: 'Synthetic demonstration basin', trueLevel: 100 })); };

function setTerrain(t) {
  st = t; const { z, w, h, cell } = t;
  let water;
  if (t.synthetic) { water = { mask: T.basinMask(z, w, h, 100), level: 100 }; water.cells = water.mask.reduce((a, v) => a + v, 0); }
  else water = T.detectWater(z, w, h, 0.12, 30);
  if (!water) { let lo = 0; for (let s = 1; s < w * h; s++) if (z[s] < z[lo]) lo = s; const mask = new Uint8Array(w * h); mask[lo] = 1; water = { mask, level: z[lo], cells: 1, none: true }; }
  st.ref = water; st.slope = T.slopeDeg(z, w, h, cell); st.bank = T.bankSlope(st.slope, water.mask, w, h, 2);
  st.hill = T.hillshade(z, w, h, cell, 315, 45, 2);
  let zmin = Infinity, zmax = -Infinity; for (const v of z) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; } st.zmin = zmin; st.zmax = zmax;
  buildBed(); st.refShore = T.shoreline(water.mask, w, h, cell); st.transect = null; st.stageBand = null;
  const src = t.synthetic ? '<span class="demo-flag">Demonstration</span> Synthetic basin generated from a seeded formula (seed 42), 200×200 cells at 5 m, reference water level 100 m. It is not a real place; its lake bed is known exactly, so no extrapolation is involved.'
    : 'Elevation: <a href="https://registry.opendata.aws/terrain-tiles/" target="_blank" rel="noopener">AWS Terrain Tiles</a> (Terrarium encoding; sources include USGS 3DEP, SRTM and others, varying by location), zoom ' + ZOOM + ', resampled to ' + w + '×' + h + ' cells of ' + fmt(cell, 1) + ' m around ' + esc(t.locName) + '. ' + (water.none ? '<span class="status-warn">No flat water surface was found in this area; the reference level is the lowest point, which is probably not a water surface. Results are not meaningful until the area includes a water body.</span>' : 'Detected water surface: ' + fmt(water.level, 2) + ' m, ' + fmt(T.area(water.mask, cell) / 1e4, 2) + ' ha in view.');
  $('#ccSource').innerHTML = src;
  $('#ccLvl').value = 0; $('#ccMaxDepth').disabled = !!t.synthetic;
  computeHyp(); redraw(); init3D(); renderObs();
}
function buildBed() { st.bed = st.synthetic ? st.z : T.extrapolateBed(st.z, st.ref.mask, st.w, st.h, st.cell, st.ref.level, isFinite(st.bank.median) ? st.bank.median : 5, +$('#ccMaxDepth').value || 8); }
$('#ccMaxDepth').onchange = () => { if (!st || st.synthetic) return; buildBed(); computeHyp(); redraw(); update3D(); };

/* ================= simulation ================= */
function simulate(delta) {
  const level = st.ref.level + delta, wet = T.wetAt(st.bed, st.w, st.h, level, st.ref.mask), shore = T.shoreline(wet, st.w, st.h, st.cell);
  const a0 = T.area(st.ref.mask, st.cell), a1 = T.area(wet, st.cell);
  return { delta, level, wet, shore, area: a1, dArea: a1 - a0, shift: st.refShore.length ? (a1 - a0) / st.refShore.length : NaN, newBank: T.bankSlope(st.slope, wet, st.w, st.h, 1) };
}
function computeHyp() { st.hyp = []; for (let d = -6; d <= 6.001; d += 0.25) { const lvl = st.ref.level + d, wet = T.wetAt(st.bed, st.w, st.h, lvl, st.ref.mask); st.hyp.push({ d: +d.toFixed(2), area: T.area(wet, st.cell) }); } }

/* ================= 2D ================= */
function elevColor(t) { const stops = [[0, [38, 115, 77]], [0.35, [120, 160, 80]], [0.6, [190, 170, 110]], [0.85, [150, 110, 80]], [1, [240, 240, 240]]]; t = Math.max(0, Math.min(1, t)); for (let k = 1; k < stops.length; k++) if (t <= stops[k][0]) { const [t0, c0] = stops[k - 1], [t1, c1] = stops[k], f = (t - t0) / (t1 - t0); return c0.map((v, i) => v + (c1[i] - v) * f); } return stops[4][1]; }
function redraw() {
  if (!st) { const cv = $('#cc2d'), ctx = cv.getContext('2d'); ctx.fillStyle = U.cssVar('--panel-2'); ctx.fillRect(0, 0, cv.width, cv.height); ctx.fillStyle = U.cssVar('--muted'); ctx.font = '15px sans-serif'; ctx.fillText('Load terrain for a location, or the synthetic demo.', 30, 320); $('#ccStats').innerHTML = ''; return; }
  const delta = +$('#ccLvl').value, sim = simulate(delta); st.sim = sim;
  $('#ccLvlTxt').textContent = (delta >= 0 ? '+' : '') + delta.toFixed(2) + ' m';
  const { w, h } = st, img = new ImageData(w, h), layer = $('#ccLayer').value, wc = hexRGB(U.cssVar('--water'));
  for (let s = 0; s < w * h; s++) {
    let c; const hs = 0.55 + 0.45 * st.hill[s];
    if (layer === 'slope') { const f = Math.min(1, st.slope[s] / 35); c = [60 + 190 * f, 170 - 120 * f, 80 - 40 * f]; }
    else c = elevColor((st.bed[s] - st.zmin) / Math.max(1, st.zmax - st.zmin)).map(v => v * hs);
    if (sim.wet[s]) { const dpth = Math.min(1, (sim.level - st.bed[s]) / 6); c = layer === 'change' && !st.ref.mask[s] ? [30, 80, 200] : wc.map((v, i) => v * (1 - 0.35 * dpth) * 0.9 + c[i] * 0.1); }
    else if (layer === 'change' && st.ref.mask[s]) c = [214, 180, 120];
    img.data[4 * s] = c[0]; img.data[4 * s + 1] = c[1]; img.data[4 * s + 2] = c[2]; img.data[4 * s + 3] = 255;
  }
  const off = document.createElement('canvas'); off.width = w; off.height = h; off.getContext('2d').putImageData(img, 0, 0);
  const cv = $('#cc2d'), ctx = cv.getContext('2d'), K = cv.width / w; ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, cv.width, cv.height);
  const seg = (sh, color, wdt, dash) => { ctx.strokeStyle = color; ctx.lineWidth = wdt; ctx.setLineDash(dash || []); ctx.beginPath(); for (let k = 0; k < sh.segments.length; k += 4) { ctx.moveTo(sh.segments[k] * K + K / 2, sh.segments[k + 1] * K + K / 2); ctx.lineTo(sh.segments[k + 2] * K + K / 2, sh.segments[k + 3] * K + K / 2); } ctx.stroke(); ctx.setLineDash([]); };
  if (st.stageBand) { ['lo', 'hi'].forEach(k => { const s2 = T.shoreline(T.wetAt(st.bed, w, h, st.ref.level + st.stageBand[k], st.ref.mask), w, h, st.cell); seg(s2, 'rgba(160,120,255,.9)', 1.2, [4, 3]); }); }
  if ($('#ccShowRef').checked) seg(st.refShore, '#ffffff', 1.4);
  if (delta !== 0) seg(sim.shore, U.cssVar('--sim'), 2.2);
  if (st.transect) { const t = st.transect; ctx.strokeStyle = '#e11d48'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(t[0] * K, t[1] * K); ctx.lineTo(t[2] * K, t[3] * K); ctx.stroke(); ctx.fillStyle = '#e11d48'; ctx.font = 'bold 13px sans-serif'; ctx.fillText('A', t[0] * K + 4, t[1] * K - 4); ctx.fillText('B', t[2] * K + 4, t[3] * K - 4); }
  if ($('#ccShowObs').checked && !st.synthetic) obsInGrid().forEach(o => { ctx.fillStyle = o.o.type === 'waterline' ? '#ffd400' : '#c084fc'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(o.gx * K, o.gy * K, 5, 0, 7); ctx.fill(); ctx.stroke(); if (o.o.acc) { ctx.strokeStyle = 'rgba(255,212,0,.6)'; ctx.beginPath(); ctx.arc(o.gx * K, o.gy * K, o.o.acc / st.cell * K, 0, 7); ctx.stroke(); } });
  ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(8, cv.height - 26, 150, 18); ctx.fillStyle = '#fff'; ctx.font = '11px monospace'; const bar = Math.pow(10, Math.floor(Math.log10(w * st.cell / 4))); ctx.fillRect(14, cv.height - 14, bar / st.cell * K, 3); ctx.fillText(bar + ' m', 20 + bar / st.cell * K, cv.height - 11);
  const extrap = delta < 0 && !st.synthetic;
  $('#ccStats').innerHTML = '<div class="stat-line"><span>Water level</span><b>' + fmt(sim.level, 2) + ' m</b></div><div class="stat-line"><span>Water area in view</span><b>' + fmt(sim.area / 1e4, 2) + ' ha</b></div><div class="stat-line"><span>' + (sim.dArea >= 0 ? 'Flooded' : 'Exposed') + ' vs reference</span><b>' + fmt(Math.abs(sim.dArea) / 1e4, 3) + ' ha</b></div><div class="stat-line"><span>Mean shoreline shift</span><b>' + fmt(sim.shift, 1) + ' m ' + (sim.shift >= 0 ? 'inland' : 'lakeward') + '</b></div><div class="stat-line"><span>Shoreline length in view</span><b>' + fmt(sim.shore.length / 1000, 2) + ' km</b></div><div class="stat-line"><span>Reference bank slope (median, IQR)</span><b>' + fmt(st.bank.median, 1) + '° (' + fmt(st.bank.p25, 1) + '–' + fmt(st.bank.p75, 1) + '°)</b></div><div class="stat-line"><span>Bank slope at simulated shoreline</span><b>' + fmt(sim.newBank.median, 1) + '°</b></div>' + (extrap ? '<p class="small status-warn" style="margin-top:6px">Below the reference level the bed is extrapolated from the bank slope (assumption, not measured bathymetry).</p>' : '') + (isFinite(st.bank.median) && st.bank.median > 0.05 ? '<p class="small muted">Planar-bank check: a ' + fmt(Math.abs(delta), 2) + ' m change on a ' + fmt(st.bank.median, 1) + '° bank moves the edge ≈ ' + fmt(Math.abs(delta) / Math.tan(st.bank.median * Math.PI / 180), 1) + ' m (Δh / tan slope).</p>' : '');
  drawProfile(); drawHyp(); update3D();
}
function hexRGB(c) { c = c.trim(); if (c[0] === '#') { let h = c.slice(1); if (h.length === 3) h = h.split('').map(x => x + x).join(''); const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; } const m = c.match(/\d+/g); return m ? m.slice(0, 3).map(Number) : [60, 140, 210]; }
$('#ccLvl').oninput = () => { stopAnim(); redraw(); };
['#ccLayer', '#ccShowObs', '#ccShowRef'].forEach(s => $(s).onchange = redraw);
$('#ccZero').onclick = () => { stopAnim(); $('#ccLvl').value = 0; redraw(); };
function stopAnim() { if (animH) { cancelAnimationFrame(animH); animH = null; } }
$('#ccAnim').onclick = () => { if (!st) return; stopAnim(); const t0 = performance.now(); const step = now => { const t = (now - t0) / 6000; if (t > 1) { animH = null; return; } $('#ccLvl').value = (-3 + 6 * (0.5 - 0.5 * Math.cos(t * Math.PI))).toFixed(2); redraw(); animH = requestAnimationFrame(step); }; animH = requestAnimationFrame(step); };
(function transectDrag() {
  const cv = $('#cc2d'); let down = null; const gp = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * st.w, (e.clientY - r.top) / r.height * st.h]; };
  cv.addEventListener('pointerdown', e => { if (!st) return; down = gp(e); cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (!down) return; const p = gp(e); st.transect = [down[0], down[1], p[0], p[1]]; redraw(); });
  cv.addEventListener('pointerup', () => { down = null; });
})();
function drawProfile() {
  if (!st) return; let t = st.transect; if (!t) { t = [st.w * 0.1, st.h / 2, st.w * 0.9, st.h / 2]; }
  const pz = T.transect(st.z, st.w, st.h, st.cell, t[0], t[1], t[2], t[3], 240), pb = T.transect(st.bed, st.w, st.h, st.cell, t[0], t[1], t[2], t[3], 240), col = U.chartColors();
  const wetLine = pb.elev.map(e => e <= st.sim.level ? st.sim.level : null);
  U.chart('ccProfile', $('#ccProfile'), { type: 'line', data: { labels: pz.dist.map(d => d.toFixed(0)), datasets: [{ label: 'Surface (DEM)', data: pz.elev, borderColor: col[5], pointRadius: 0, borderWidth: 2 }, { label: st.synthetic ? 'Bed (synthetic, exact)' : 'Bed (extrapolated under water)', data: pb.elev, borderColor: col[1], borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5 }, { label: 'Water at chosen level', data: wetLine, borderColor: U.cssVar('--water'), backgroundColor: CL.alpha(U.cssVar('--water'), .25), fill: { target: 1 }, pointRadius: 0, borderWidth: 2, spanGaps: false }] }, options: CL.baseChartOpts({ scales: { x: { title: { display: true, text: 'distance along transect A→B (m)' + (st.transect ? '' : ' — default west–east line; drag on the map'), color: U.cssVar('--muted') }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: 'elevation (m)', color: U.cssVar('--muted') } } } }) });
}
function drawHyp() {
  if (!st || !st.hyp) return; const col = U.chartColors(), d = +$('#ccLvl').value;
  U.chart('ccHyp', $('#ccHyp'), { type: 'line', data: { datasets: [{ label: 'Water area (ha)', data: st.hyp.map(p => ({ x: p.d, y: p.area / 1e4 })), borderColor: col[0], pointRadius: 0, borderWidth: 2, segment: { borderDash: c => !st.synthetic && c.p0.parsed.x < 0 ? [5, 4] : undefined } }, { label: 'Current slider', data: [{ x: d, y: st.sim.area / 1e4 }], pointRadius: 6, backgroundColor: col[1], borderColor: col[1] }] }, options: CL.baseChartOpts({ animation: false, scales: { x: { type: 'linear', title: { display: true, text: 'water level change (m)' + (st.synthetic ? '' : '; dashed = extrapolated bed'), color: U.cssVar('--muted') } }, y: { title: { display: true, text: 'ha', color: U.cssVar('--muted') } } } }) });
}
$('#cc2dPng').onclick = () => U.downloadDataURL('castlab-shoreline-2d.png', $('#cc2d').toDataURL('image/png'));

/* ================= 3D ================= */
function init3D() {
  const wrap = $('#cc3dWrap');
  if (!three) {
    try {
      const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true }); renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      wrap.innerHTML = ''; wrap.appendChild(renderer.domElement);
      const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(45, 1, 1, 100000), controls = new OrbitControls(camera, renderer.domElement);
      scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.2)); const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-1, 2, -1); scene.add(sun);
      three = { renderer, scene, camera, controls }; controls.addEventListener('change', render3D);
      new ResizeObserver(() => { const s = wrap.clientWidth; if (!s) return; renderer.setSize(s, s); camera.aspect = 1; camera.updateProjectionMatrix(); render3D(); }).observe(wrap);
    } catch (e) { wrap.innerHTML = '<p class="small muted" style="padding:16px">3D view unavailable: WebGL could not start in this browser (' + esc(e.message) + '). The 2D map and all calculations still work.</p>'; return; }
  }
  const { scene, camera, controls } = three; ['terrain', 'water'].forEach(k => { if (three[k]) { scene.remove(three[k]); three[k].geometry.dispose(); three[k].material.dispose(); } });
  const step = Math.max(1, Math.ceil(st.w / 180)), nx = Math.floor((st.w - 1) / step) + 1, ny = Math.floor((st.h - 1) / step) + 1, W = (nx - 1) * step * st.cell, H = (ny - 1) * step * st.cell;
  const geo = new THREE.PlaneGeometry(W, H, nx - 1, ny - 1); geo.rotateX(-Math.PI / 2); geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(nx * ny * 3), 3));
  const wgeo = geo.clone(); three.terrain = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));
  three.water = new THREE.Mesh(wgeo, new THREE.MeshPhongMaterial({ color: new THREE.Color(U.cssVar('--water')), transparent: true, opacity: 0.78, shininess: 80 }));
  scene.add(three.terrain, three.water); three.grid = { step, nx, ny };
  const span = Math.max(W, H); camera.near = span / 1000; camera.far = span * 20; camera.position.set(span * 0.55, span * 0.55, span * 0.75); controls.target.set(0, 0, 0); camera.updateProjectionMatrix(); controls.update();
  update3D();
}
function update3D() {
  if (!three || !three.terrain || !st || !st.sim) return; const { step, nx, ny } = three.grid, ve = +$('#ccVe').value, base = st.ref.level;
  $('#ccVeTxt').textContent = ve + '×';
  const pos = three.terrain.geometry.attributes.position, colA = three.terrain.geometry.attributes.color, wpos = three.water.geometry.attributes.position;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = j * nx + i, s = (j * step) * st.w + i * step, e = st.bed[s], y = (e - base) * ve;
    pos.setY(k, y); const c = st.ref.mask[s] && !st.synthetic ? [0.55, 0.5, 0.42] : elevColor((e - st.zmin) / Math.max(1, st.zmax - st.zmin)).map(v => v / 255); colA.setXYZ(k, c[0], c[1], c[2]);
    wpos.setY(k, st.sim.wet[s] ? (st.sim.level - base) * ve : y - 2 * ve);
  }
  pos.needsUpdate = colA.needsUpdate = wpos.needsUpdate = true; three.terrain.geometry.computeVertexNormals(); three.water.geometry.computeVertexNormals();
  three.scene.background = new THREE.Color(U.cssVar('--panel-2')); render3D();
}
function render3D() { if (three && three.terrain) three.renderer.render(three.scene, three.camera); }
$('#ccVe').oninput = update3D;
let spinH = null;
$('#ccSpin').onclick = () => { if (!three || !three.terrain) return U.toast('Load terrain first.'); const on = !three.controls.autoRotate; three.controls.autoRotate = on; three.controls.autoRotateSpeed = 1.1; $('#ccSpin').classList.toggle('primary', on); cancelAnimationFrame(spinH); const loop = () => { if (!three.controls.autoRotate || $('#v-castcheck').hidden) { three.controls.autoRotate = false; $('#ccSpin').classList.remove('primary'); return; } three.controls.update(); spinH = requestAnimationFrame(loop); }; if (on) loop(); };
$('#cc3dPng').onclick = () => { if (!three) return; render3D(); U.downloadDataURL('castlab-shoreline-3d.png', three.renderer.domElement.toDataURL('image/png')); };

/* ================= USGS water history ================= */
async function findGages(silent) {
  const loc = st && !st.synthetic ? st.center : (Store.location($('#ccLoc').value) || {}); if (!isFinite(loc.lat)) return silent || U.toast('Pick a location with coordinates.');
  $('#ccGage').innerHTML = '<option>Searching USGS…</option>';
  try { const sites = (await U.usgsSites(loc.lat, loc.lon, 0.3)).slice(0, 15); $('#ccGage').innerHTML = sites.length ? sites.map(s => '<option value="' + s.site + '" data-name="' + esc(s.name) + '" data-dist="' + s.distKm + '">' + esc(s.name) + ' (' + s.site + ', ' + fmt(s.distKm, 1) + ' km)</option>').join('') : '<option value="">No active USGS stage gages within ~30 km</option>'; }
  catch (err) { $('#ccGage').innerHTML = '<option value="">USGS search failed: ' + esc(err.message) + '</option>'; }
}
$('#ccFindGage').onclick = () => findGages(false);
function quantile(sorted, q) { const p = (sorted.length - 1) * q, i = Math.floor(p), f = p - i; return sorted[i] + (sorted[Math.min(sorted.length - 1, i + 1)] - sorted[i]) * f; }
$('#ccGageGo').onclick = async () => {
  const site = $('#ccGage').value; if (!site) return U.toast('Choose a gage.'); const opt = $('#ccGage').selectedOptions[0];
  $('#ccWaterKpis').innerHTML = '<p class="muted">Loading USGS record…</p>';
  try {
    let [dv, iv] = await Promise.all([U.usgsSeries(site, 'dv', 'period=' + $('#ccPeriod').value), U.usgsSeries(site, 'iv', 'period=P1D')]), derived = false;
    if (!dv.values.length) { // some gages publish only instantaneous stage: average it to daily means (IV service serves at most 120 days)
      const ivh = await U.usgsSeries(site, 'iv', 'period=P120D'), days = {};
      ivh.values.forEach(v => { const d = v.t.slice(0, 10); (days[d] = days[d] || []).push(v.v); });
      dv = { unit: ivh.unit, values: Object.keys(days).sort().map(d => ({ t: d, v: S.mean(days[d]), q: [] })) }; derived = true;
    }
    if (!dv.values.length) throw new Error('no stage data for this gage and period');
    const vals = dv.values.map(v => v.v).sort((a, b) => a - b), toM = (dv.unit || 'ft').toLowerCase().startsWith('ft') ? 0.3048 : 1, cur = iv.values.length ? iv.values[iv.values.length - 1] : dv.values[dv.values.length - 1];
    stage = { site, name: opt.dataset.name, dist: +opt.dataset.dist, unit: dv.unit, toM, values: dv.values, p10: quantile(vals, .1), p50: quantile(vals, .5), p90: quantile(vals, .9), min: vals[0], max: vals[vals.length - 1], cur, curIsIv: !!iv.values.length, n: vals.length };
    const pr = vals.filter(v => v <= cur.v).length / vals.length;
    $('#ccWaterKpis').innerHTML = [['Current stage', fmt(cur.v, 2) + ' ' + esc(stage.unit), (stage.curIsIv ? 'instantaneous, ' : 'daily mean, ') + esc(cur.t.slice(0, 16).replace('T', ' ')) + (cur.q && cur.q.includes('P') ? ' · provisional' : '')], ['Median over period', fmt(stage.p50, 2) + ' ' + esc(stage.unit), 'n = ' + stage.n + ' daily values' + (derived ? ' (means of instantaneous readings; this gage publishes no daily values, so history is limited to 120 days)' : '')], ['Current − median', (cur.v - stage.p50 >= 0 ? '+' : '') + fmt((cur.v - stage.p50) * toM, 2) + ' m', 'percentile of record: ' + Math.round(pr * 100)], ['10th–90th percentile', fmt(stage.p10, 2) + '–' + fmt(stage.p90, 2) + ' ' + esc(stage.unit), 'range ' + fmt(stage.min, 2) + '–' + fmt(stage.max, 2)]].map(k => '<div class="kpi"><div class="l">' + k[0] + '</div><div class="v" style="font-size:19px">' + k[1] + '</div><div class="s">' + k[2] + '</div></div>').join('') + (stage.dist > 5 ? '<p class="note warn" style="grid-column:1/-1">This gage is ' + fmt(stage.dist, 1) + ' km from the location. Check it is on the same water body before applying its changes.</p>' : '');
    const col = U.chartColors();
    U.chart('ccStage', $('#ccStage'), { type: 'line', data: { labels: dv.values.map(v => v.t.slice(0, 10)), datasets: [{ label: 'Daily mean gage height (' + stage.unit + ')', data: dv.values.map(v => v.v), borderColor: col[0], pointRadius: 0, borderWidth: 1.5 }, { label: 'Median', data: dv.values.map(() => stage.p50), borderColor: col[5], borderDash: [6, 4], pointRadius: 0, borderWidth: 1 }, { label: '10th percentile', data: dv.values.map(() => stage.p10), borderColor: CL.alpha(col[2], .7), borderDash: [2, 3], pointRadius: 0, borderWidth: 1 }, { label: '90th percentile', data: dv.values.map(() => stage.p90), borderColor: CL.alpha(col[2], .7), borderDash: [2, 3], pointRadius: 0, borderWidth: 1 }] }, options: CL.baseChartOpts({ scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: 'gage height (' + stage.unit + ', local datum)', color: U.cssVar('--muted') } } } }) });
    $('#ccApplyStage').disabled = !st || st.synthetic;
  } catch (err) { $('#ccWaterKpis').innerHTML = '<p class="note bad" style="grid-column:1/-1">USGS request failed: ' + esc(err.message) + '</p>'; }
};
$('#ccApplyStage').onclick = () => { if (!stage || !st) return; const d = (stage.cur.v - stage.p50) * stage.toM; $('#ccLvl').value = Math.max(-6, Math.min(6, d)).toFixed(2); st.stageBand = { lo: (stage.p10 - stage.p50) * stage.toM, hi: (stage.p90 - stage.p50) * stage.toM }; showTab('twin'); U.toast('Applied: reference shoreline treated as median stage; purple dashed lines = 10th/90th percentile stage.'); };
function renderWater() { if (!$('#ccGage').value && st && !st.synthetic) findGages(true); }

/* ================= field observations ================= */
function obsInGrid() {
  if (!st || st.synthetic || !st.origin) return [];
  return Store.list('observations').map(o => { const p = T.lonLatToPixel(o.lon, o.lat, ZOOM), gx = (p.x - st.origin.x) / st.origin.f, gy = (p.y - st.origin.y) / st.origin.f; return { o, gx, gy, inside: gx >= 0 && gy >= 0 && gx < st.w - 1 && gy < st.h - 1 }; }).filter(x => x.inside);
}
function compareObs() {
  const rows = []; if (!st || st.synthetic) return rows; const cache = {};
  obsInGrid().forEach(({ o, gx, gy }) => {
    let delta = +$('#ccLvl').value, basis = 'slider level';
    if (o.gage != null && stage && stage.p50 != null) { delta = (o.gage - stage.p50) * stage.toM; basis = 'gage reading at the time'; }
    const key = delta.toFixed(2); if (!cache[key]) cache[key] = simulate(delta); const sim = cache[key];
    const r = { id: o.id, type: o.type, time: o.time, acc: o.acc, basis, delta };
    if (o.type === 'waterline') { const d = T.distToShoreline(sim.shore.segments, gx - 0.5, gy - 0.5, st.cell), inWater = sim.wet[Math.round(gy) * st.w + Math.round(gx)]; r.measured = 0; r.simulated = inWater ? d : -d; r.residual = r.simulated; r.unit = 'm'; r.what = 'distance from GPS point to simulated shoreline (+ = point lies inside simulated water)'; }
    else if (o.type === 'slope' && o.value != null) { r.measured = o.value; r.simulated = T.bilinear(st.slope, st.w, st.h, gx, gy); r.residual = r.simulated - r.measured; r.unit = '°'; r.what = 'DEM slope − clinometer slope'; }
    else if (o.type === 'depth' && o.value != null) { r.measured = o.value; r.simulated = Math.max(0, sim.level - T.bilinear(st.bed, st.w, st.h, gx, gy)); r.residual = r.simulated - r.measured; r.unit = 'm'; r.what = 'simulated depth (extrapolated bed) − measured depth'; }
    else return; rows.push(r);
  });
  return rows;
}
CC.summary = () => { const r = (CC._cmp || []).filter(x => x.type === 'waterline'); return r.length ? { n: r.length, rmse: S.rmse(r.map(x => x.residual)) } : null; };
function renderObs() {
  const all = Store.list('observations'), rows = compareObs(); CC._cmp = rows;
  const by = t => rows.filter(r => r.type === t), sum = (t, label) => { const r = by(t); if (!r.length) return ''; const res = r.map(x => x.residual); return '<div class="stat-line"><span>' + label + ' (n=' + r.length + ')</span><b>RMSE ' + fmt(S.rmse(res), 2) + ' ' + r[0].unit + ' · bias ' + fmt(S.mean(res), 2) + '</b></div>'; };
  const accMed = S.median(by('waterline').map(r => r.acc).filter(v => v != null));
  $('#obsSummary').innerHTML = !st ? '<p class="note">Load the terrain for the location where the observations were taken to compare them with the simulation.</p>' : st.synthetic ? '<p class="note warn">The synthetic demo has no geographic position, so field observations cannot be compared with it.</p>' : (rows.length ? sum('waterline', 'Waterline position') + sum('slope', 'Bank slope') + sum('depth', 'Water depth') + (by('waterline').length ? (function () { const tol = Math.hypot(st.cell, isFinite(accMed) ? accMed : 5), rm = S.rmse(by('waterline').map(r => r.residual)); return '<p class="small ' + (rm > 2 * tol ? 'status-warn' : 'muted') + '">Input resolution: grid cell ' + fmt(st.cell, 1) + ' m' + (isFinite(accMed) ? ', median reported GPS accuracy ' + fmt(accMed, 1) + ' m' : '') + ' (combined ≈ ' + fmt(tol, 1) + ' m). ' + (rm <= 2 * tol ? 'The waterline RMSE is within about twice that, so the simulation agrees with the field points as well as these inputs allow.' : 'The waterline RMSE is well above that, so the disagreement is real: check that the points were taken at the water\'s edge, that the stage used for each point is right, or the DEM may not represent this shoreline.') + '</p>'; })() : '') : '<p class="muted small">' + all.length + ' observations stored; none fall inside the loaded terrain.</p>');
  $('#obsTable').innerHTML = all.length ? '<div class="tablewrap" style="max-height:300px"><table><tr><th>Time</th><th>Type</th><th class="num">GPS ±m</th><th class="num">Measured</th><th class="num">Simulated</th><th class="num">Residual</th><th>Level basis</th><th></th></tr>' + all.map(o => { const r = rows.find(x => x.id === o.id); return '<tr><td class="small mono">' + esc(o.time) + '</td><td>' + esc(o.type) + '</td><td class="num">' + fmt(o.acc, 1) + '</td><td class="num">' + (r ? fmt(r.measured, 2) : o.value != null ? fmt(o.value, 2) : '—') + '</td><td class="num">' + (r ? fmt(r.simulated, 2) : '—') + '</td><td class="num">' + (r ? fmt(r.residual, 2) + ' ' + r.unit : '<span class="muted small">not in view</span>') + '</td><td class="small">' + (r ? esc(r.basis) + ' (Δ ' + fmt(r.delta, 2) + ' m)' : '') + '</td><td><button class="btn sm danger" data-obs-del="' + o.id + '">×</button></td></tr>'; }).join('') + '</table></div>' : '<p class="muted small">No field observations yet. Stand at the water\'s edge and take a GPS fix to record a waterline point.</p>';
  const col = U.chartColors();
  U.chart('obsChart', rows.length ? $('#obsChart') : null, { type: 'bar', data: { labels: rows.map((r, i) => (i + 1) + ' ' + r.type), datasets: [{ label: 'Residual (simulated − measured)', data: rows.map(r => r.residual), backgroundColor: rows.map(r => r.type === 'waterline' ? col[0] : r.type === 'slope' ? col[2] : col[1]) }] }, options: CL.baseChartOpts({ plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'residual (m or °)', color: U.cssVar('--muted') } } } }) });
}
$('#obsGps').onclick = () => { if (!navigator.geolocation) return U.toast('No geolocation in this browser.'); $('#obsGpsMsg').textContent = 'Waiting for GPS…'; navigator.geolocation.getCurrentPosition(p => { $('#obsLat').value = p.coords.latitude.toFixed(7); $('#obsLon').value = p.coords.longitude.toFixed(7); $('#obsAcc').value = p.coords.accuracy.toFixed(1); $('#obsTime').value = new Date(p.timestamp).toISOString().slice(0, 16).replace('T', ' '); $('#obsGpsMsg').textContent = 'Fix ±' + p.coords.accuracy.toFixed(1) + ' m (as reported by the device). For waterline points, wait for accuracy under ~5 m.'; }, err => $('#obsGpsMsg').textContent = 'GPS error: ' + err.message, { enableHighAccuracy: true, timeout: 20000, maximumAge: 0 }); };
$('#obsForm').onsubmit = e => { e.preventDefault(); const nb = s => $(s).value === '' ? null : +$(s).value; const type = $('#obsType').value; if (type !== 'waterline' && nb('#obsVal') == null) return U.toast('Enter the measured value.'); Store.add('observations', { type, lat: +$('#obsLat').value, lon: +$('#obsLon').value, acc: nb('#obsAcc'), value: nb('#obsVal'), time: $('#obsTime').value, gage: nb('#obsGage'), gageSite: stage ? stage.site : null, locationId: $('#ccLoc').value || null, notes: $('#obsNotes').value }); ['#obsLat', '#obsLon', '#obsAcc', '#obsVal', '#obsGage', '#obsNotes'].forEach(s => $(s).value = ''); renderObs(); redraw(); U.toast('Observation saved.'); };
$('#obsTable').onclick = e => { const b = e.target.closest('[data-obs-del]'); if (b && confirm('Delete this observation?')) { Store.remove('observations', b.dataset.obsDel); renderObs(); redraw(); } };
$('#obsCsv').onclick = () => U.download('castlab-field-observations.csv', U.toCSV(Store.list('observations').map(o => { const r = (CC._cmp || []).find(x => x.id === o.id) || {}; return Object.assign({}, o, { location: Store.locationName(o.locationId), simulated: r.simulated, residual: r.residual, levelBasis: r.basis, levelDelta_m: r.delta }); })), 'text/csv');

/* ================= methodology ================= */
function renderMethods() {
  $('#ccMethods').innerHTML = '<h2>Shoreline digital twin methodology</h2>' +
    '<h3>1. Terrain data</h3><p>Elevation comes from the public <a href="https://registry.opendata.aws/terrain-tiles/" target="_blank" rel="noopener">AWS Terrain Tiles</a> dataset (Mapzen "Terrarium" PNG encoding, elevation = R×256 + G + B/256 − 32768 m). It merges several sources (USGS 3DEP in the United States, SRTM and others elsewhere), so native resolution and vertical accuracy vary by place: roughly 10 m or finer where 3DEP is used and about 30 m elsewhere, with metre-scale vertical error. Tiles at zoom 15 are mosaicked around the location and block-averaged to about ' + TARGET + ' cells per side.</p>' +
    '<h3>2. Reference water surface</h3><p>Elevation models record the water surface, not the bed, and most sources flatten water bodies. The twin identifies the largest connected patch of cells whose 3×3 neighbourhood is level to within 0.12 m and takes its median elevation as the reference level. That surface was captured at an unknown date and stage, which is why gage changes are applied relative to a typical (median) stage and why field observations are needed to check the anchor.</p>' +
    '<h3>3. Bank slope</h3><p>Slope is computed with Horn\'s (1981) third-order finite-difference method. The reported bank slope is the median (and interquartile range) of dry cells within two cells of the reference shoreline.</p>' +
    '<h3>4. Water-level scenarios</h3><p>For a level h = h_ref + Δ, the wetted area is every cell reachable from the reference water through cells with ground elevation ≤ h (four-connected flood fill). This "connected bathtub" model ignores flow, wind set-up, waves and levees that are below the grid resolution. The shoreline is traced from the wet mask with marching squares. Mean horizontal shoreline shift = change in wetted area ÷ reference shoreline length. On a planar bank of slope β, a level change Δh moves the edge Δh / tan β horizontally; the readout shows this check alongside the full 2D result.</p>' +
    '<h3>5. Below-waterline assumption</h3><p>For drawdowns there is no measured bathymetry, so the bed is extrapolated: depth below the reference surface = min(max depth, distance to shore × tan(bank slope)), with distance computed by breadth-first search through water cells. This is an assumption, shown dashed in graphs and flagged in the readout. The synthetic demo has an exact bed, so its drawdowns involve no extrapolation.</p>' +
    '<h3>6. Historical versus current water</h3><p>Daily mean gage height (USGS parameter 00065) for the chosen period gives the median and 10th/90th percentile stages; the latest instantaneous value gives the current stage. Gage datums are local, so only differences are used: Δ = (current − median) × 0.3048 m when reported in feet. The 10th/90th percentile shorelines show the typical range. A gage on another water body, or far up- or downstream on a river with a steep water-surface slope, will not represent the location.</p>' +
    '<h3>7. Comparison with field measurements</h3><ul><li><b>Waterline points</b>: the GPS position of the water\'s edge is compared with the simulated shoreline at the stage recorded with the observation (or the slider level if none). The residual is the distance to the simulated shoreline, positive when the point lies inside simulated water.</li><li><b>Bank slope</b>: clinometer reading vs DEM slope at that point (bilinear interpolation).</li><li><b>Depth</b>: measured depth vs simulated depth over the extrapolated bed.</li></ul><p>Summary statistics are RMSE and mean bias. Expected agreement is limited by GPS accuracy (reported by the device), the grid cell size and the DEM\'s vertical error; residuals of that size do not indicate model failure.</p>' +
    '<h3>8. Verification</h3><p>Automated tests check the Terrarium decoding, Web-Mercator round trips, Horn slope on a planar surface, water-surface detection, the shoreline advance on a 10° planar bank (Δh / tan 10°), transect waterline interpolation, shoreline length and point-to-shoreline distance (<code>castlab/tests/run.js</code>).</p>';
}

CC.reportSection = function () {
  const all = Store.list('observations'); if (!all.length) return [['Shoreline twin', ['No field observations recorded.']]];
  const rows = CC._cmp || []; const by = t => rows.filter(r => r.type === t);
  const res = ['Observations recorded: ' + all.length + ' (' + ['waterline', 'slope', 'depth'].map(t => all.filter(o => o.type === t).length + ' ' + t).join(', ') + ').'];
  if (rows.length) ['waterline', 'slope', 'depth'].forEach(t => { const r = by(t); if (r.length) res.push(t + ': n = ' + r.length + ', RMSE ' + S.rmse(r.map(x => x.residual)).toFixed(2) + ' ' + r[0].unit + ', mean bias ' + S.mean(r.map(x => x.residual)).toFixed(2) + ' ' + r[0].unit + ' (terrain: ' + (st ? st.locName + ', cell ' + st.cell.toFixed(1) + ' m' : '') + ').'); });
  else res.push('No comparison computed in this session: load the terrain for the observation site in CastCheck before generating the report.');
  return [['Shoreline twin — Methodology', ['Terrain from AWS Terrain Tiles (zoom 15, block-averaged); reference water surface detected as the largest level patch; connected flood fill for water-level scenarios; bed below the reference surface extrapolated from the median bank slope; level changes from USGS daily gage height relative to the period median. GPS waterline points compared with the simulated shoreline at the recorded stage.']], ['Shoreline twin — Results', res]];
};

// The module loads after app.js; if the user is already on this view, draw it.
if (CL.current === 'castcheck') CC.render();
