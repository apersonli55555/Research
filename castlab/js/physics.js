/* CastLab rod mechanics — pure functions, SI units throughout (m, N, Pa, kg).
   Model: the rod blank is an inextensible, tapered, hollow circular cantilever (Euler-Bernoulli beam, large
   rotations = "elastica") clamped at the grip and loaded by the line tension at the tip.

     EI(s) * dtheta/ds = M(s)                      (moment-curvature, linear elastic)
     dM/ds = Fx*sin(theta) - Fy*cos(theta)         (moment equilibrium for a dead tip load F)
     dx/ds = cos(theta), dy/ds = sin(theta)        (inextensible centreline)
     theta(0) = theta0 (clamped),  M(L) = 0 (free tip)

   Solved by shooting from the tip: guess the tip angle, integrate back to the grip with RK4 and root-find on
   theta(0) - theta0. The tip angle is bracketed between the clamp angle and the load direction.
   Assumptions and limits are documented in README.md and in the Methodology panel of the simulator. */
(function (root) {
  'use strict';
  const G = 9.80665, D2R = Math.PI / 180;

  /* Nominal, textbook-order laminate properties. They are NOT manufacturer data for any specific blank;
     calibrate E against a bench deflection test (see fitModulus). costIndex is a user-editable relative
     material cost (currency per kg of blank) used only by the optimiser. */
  const MATERIALS = {
    eglass: { name: 'E-glass / epoxy', E: 40e9, rho: 1900, strength: 600e6, costIndex: 60 },
    carbon_std: { name: 'Carbon / epoxy (standard modulus)', E: 110e9, rho: 1550, strength: 900e6, costIndex: 220 },
    carbon_hm: { name: 'Carbon / epoxy (high modulus)', E: 160e9, rho: 1600, strength: 800e6, costIndex: 480 },
    composite: { name: 'Glass / carbon hybrid', E: 70e9, rho: 1750, strength: 700e6, costIndex: 120 }
  };
  /* Approximate elongation at break by line type (typical published ranges; editable in the UI). */
  const LINES = {
    mono: { name: 'Monofilament (nylon)', stretchAtBreak: 0.25 },
    fluoro: { name: 'Fluorocarbon', stretchAtBreak: 0.20 },
    braid: { name: 'Braid (UHMWPE)', stretchAtBreak: 0.04 }
  };

  function matOf(cfg) { const m = MATERIALS[cfg.material] || MATERIALS.carbon_std; return { E: cfg.E || m.E, rho: cfg.rho || m.rho, strength: cfg.strength || m.strength, costIndex: cfg.costIndex != null ? cfg.costIndex : m.costIndex, name: m.name }; }
  /* Cross-section at arc length s. Outer diameter tapers linearly; wallRatio = wall thickness / outer radius (1 = solid). */
  function section(cfg, s) {
    const D = cfg.buttOD + (cfg.tipOD - cfg.buttOD) * (s / cfg.length), q = 1 - cfg.wallRatio;
    return { D, I: Math.PI / 64 * D * D * D * D * (1 - q * q * q * q), A: Math.PI / 4 * D * D * (1 - q * q) };
  }
  function mass(cfg) { const N = 200, h = cfg.length / N, rho = matOf(cfg).rho; let m = 0; for (let i = 0; i < N; i++) m += section(cfg, (i + 0.5) * h).A * h; return m * rho; }

  /* Integrate (theta, M) from the tip (s=L) back to the grip (s=0). Returns theta(0); optionally records profiles. */
  function backInt(cfg, E, Fx, Fy, thetaTip, N, rec) {
    const L = cfg.length, h = L / N; let th = thetaTip, M = 0;
    const EI = s => E * section(cfg, s).I;
    const dM = t => Fx * Math.sin(t) - Fy * Math.cos(t);
    if (rec) { rec.theta[N] = th; rec.M[N] = 0; }
    for (let i = N; i > 0; i--) {
      const s = i * h, e1 = EI(s), e2 = EI(s - h / 2), e3 = EI(s - h);
      const k1t = M / e1, k1m = dM(th);
      const k2t = (M - h / 2 * k1m) / e2, k2m = dM(th - h / 2 * k1t);
      const k3t = (M - h / 2 * k2m) / e2, k3m = dM(th - h / 2 * k2t);
      const k4t = (M - h * k3m) / e3, k4m = dM(th - h * k3t);
      th -= h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t); M -= h / 6 * (k1m + 2 * k2m + 2 * k3m + k4m);
      if (rec) { rec.theta[i - 1] = th; rec.M[i - 1] = M; }
    }
    return th;
  }

  /* Solve the loaded shape.
     T: line tension at the tip (N). betaDeg: line direction below horizontal (0 = horizontal away from the angler,
     90 = straight down). theta0Deg: rod (clamp) angle above horizontal. */
  function solveRod(cfg, T, betaDeg, theta0Deg, opts) {
    opts = opts || {}; const N = opts.N || 160, mat = matOf(cfg), E = mat.E * (opts.Escale || 1), L = cfg.length;
    const th0 = (theta0Deg || 0) * D2R, phiRaw = -(betaDeg == null ? 90 : betaDeg) * D2R;
    let dphi = phiRaw - th0; while (dphi > Math.PI) dphi -= 2 * Math.PI; while (dphi < -Math.PI) dphi += 2 * Math.PI;
    const phi = th0 + dphi, Fx = T * Math.cos(phi), Fy = T * Math.sin(phi);
    let tip = th0;
    if (T > 0 && Math.abs(dphi) > 1e-9) {
      // residual g(thetaTip) = theta(0) - theta0; g(theta0) and g(phi) have opposite signs -> bracketed (Illinois regula falsi)
      let a = th0, b = phi, ga = backInt(cfg, E, Fx, Fy, a, N) - th0, gb = phi - th0, side = 0;
      for (let it = 0; it < 80; it++) {
        const c = (a * gb - b * ga) / (gb - ga), gc = backInt(cfg, E, Fx, Fy, c, N) - th0; tip = c;
        if (Math.abs(gc) < 1e-10 || Math.abs(b - a) < 1e-12) break;
        if (gc * gb > 0) { b = c; gb = gc; if (side === -1) ga /= 2; side = -1; } else { a = c; ga = gc; if (side === 1) gb /= 2; side = 1; }
      }
    }
    const rec = { theta: new Float64Array(N + 1), M: new Float64Array(N + 1) };
    if (T > 0) backInt(cfg, E, Fx, Fy, tip, N, rec); else rec.theta.fill(th0);
    const h = L / N, x = new Float64Array(N + 1), y = new Float64Array(N + 1), stress = new Float64Array(N + 1), s = new Float64Array(N + 1);
    let U = 0, sMax = 0, sMaxAt = 0;
    for (let i = 0; i <= N; i++) {
      s[i] = i * h; const sec = section(cfg, s[i]);
      stress[i] = Math.abs(rec.M[i]) * (sec.D / 2) / sec.I; if (stress[i] > sMax) { sMax = stress[i]; sMaxAt = s[i]; }
      if (i > 0) { // Simpson-like midpoint for position, trapezoid for energy
        const tm = 0.5 * (rec.theta[i] + rec.theta[i - 1]);
        x[i] = x[i - 1] + h * Math.cos(tm); y[i] = y[i - 1] + h * Math.sin(tm);
        const secP = section(cfg, s[i - 1]);
        U += 0.5 * h * (rec.M[i] * rec.M[i] / (2 * E * sec.I) + rec.M[i - 1] * rec.M[i - 1] / (2 * E * secP.I));
      }
    }
    const x0 = L * Math.cos(th0), y0 = L * Math.sin(th0), dx = x[N] - x0, dy = y[N] - y0;
    const defl = Math.hypot(dx, dy), M0 = Math.abs(rec.M[0]);
    return { s, x, y, theta: rec.theta, M: rec.M, stress, tipX: x[N], tipY: y[N], dx, dy, deflection: defl, deflRatio: defl / L,
      tipAngleDeg: rec.theta[N] / D2R, buttMoment: M0, leverArm: T > 0 ? M0 / T : 0, maxStress: sMax, maxStressAt: sMaxAt,
      safetyFactor: sMax > 0 ? mat.strength / sMax : Infinity, strainEnergy: U, stiffness: defl > 0 ? T / defl : Infinity, T, E };
  }

  /* Force-deflection curve: tensions (N) -> tip deflection (m). */
  function loadCurve(cfg, Tmax, betaDeg, theta0Deg, n, opts) {
    const out = []; n = n || 24;
    for (let i = 0; i <= n; i++) { const T = Tmax * i / n, r = solveRod(cfg, T, betaDeg, theta0Deg, Object.assign({ N: 100 }, opts)); out.push({ T, deflection: r.deflection, drop: -r.dy, sf: r.safetyFactor, energy: r.strainEnergy }); }
    return out;
  }
  /* Small-deflection closed form for a UNIFORM cantilever with transverse tip load: delta = F L^3 / (3 E I). Used for verification. */
  function linearTipDeflection(F, L, EI) { return F * L * L * L / (3 * EI); }

  /* Bench test (rod clamped horizontal, mass hung from the tip): predicted vertical tip drop in metres. */
  function benchDrop(cfg, massKg, Escale) { return -solveRod(cfg, massKg * G, 90, 0, { N: 100, Escale: Escale || 1 }).dy; }
  /* Compare simulation with a physical bench test. data: [{massKg, dropM}]. Returns per-point residuals and RMSE. */
  function compareBench(cfg, data, Escale) {
    const rows = data.map(d => { const pred = benchDrop(cfg, d.massKg, Escale); return { massKg: d.massKg, measured: d.dropM, predicted: pred, residual: pred - d.dropM }; });
    return { rows, n: rows.length, rmse: rows.length ? Math.sqrt(rows.reduce((a, r) => a + r.residual * r.residual, 0) / rows.length) : NaN };
  }
  /* One-parameter calibration: scale factor k on E that minimises the squared error of the bench test (golden section on log k). */
  function fitModulus(cfg, data) {
    if (!data || data.length < 2) return null;
    const sse = lk => { const c = compareBench(cfg, data, Math.exp(lk)); return c.rmse * c.rmse * c.n; };
    const gr = (Math.sqrt(5) - 1) / 2; let a = Math.log(0.05), b = Math.log(20), c = b - gr * (b - a), d = a + gr * (b - a);
    for (let i = 0; i < 60; i++) { if (sse(c) < sse(d)) b = d; else a = c; c = b - gr * (b - a); d = a + gr * (b - a); }
    const k = Math.exp(0.5 * (a + b)), after = compareBench(cfg, data, k), before = compareBench(cfg, data, 1);
    return { scale: k, E: matOf(cfg).E * k, rmseBefore: before.rmse, rmseAfter: after.rmse, n: data.length };
  }

  /* Line elongation under tension, linear-elastic to break: dL = L * strainAtBreak * T / breakingStrength. */
  function lineStretch(lineType, lengthM, T, breakN, stretchOverride) {
    const e = stretchOverride != null ? stretchOverride : (LINES[lineType] || LINES.mono).stretchAtBreak;
    return breakN > 0 ? lengthM * e * Math.min(1, T / breakN) : 0;
  }

  /* Equipment compatibility. All force inputs in newtons, lure weights in grams.
     setup: { lineBreakN, knotEfficiency, dragN, reelMaxDragN, lureG, rodLineMinN, rodLineMaxN, rodLureMinG, rodLureMaxG }.
     result: solveRod output at the drag tension. Thresholds are stated in each message so they can be audited. */
  function compatibility(setup, result) {
    const c = [], add = (name, status, detail) => c.push({ name, status, detail });
    const kn = setup.knotEfficiency || 0.8, eff = setup.lineBreakN * kn;
    if (setup.lineBreakN > 0 && setup.dragN > 0) {
      const r = setup.dragN / setup.lineBreakN;
      add('Drag vs line strength', r <= 0.34 ? 'ok' : r <= 0.5 ? 'warn' : 'fail', 'Drag is ' + (r * 100).toFixed(0) + '% of rated line strength (common guidance: 25–33%).');
      add('Knot-limited margin', eff / setup.dragN >= 2 ? 'ok' : eff / setup.dragN >= 1.2 ? 'warn' : 'fail', 'Knot-limited strength ' + eff.toFixed(1) + ' N (efficiency ' + (kn * 100).toFixed(0) + '%, assumed) is ' + (eff / setup.dragN).toFixed(2) + '× the drag setting.');
    }
    if (setup.reelMaxDragN > 0 && setup.dragN > 0) add('Reel drag capacity', setup.dragN <= 0.8 * setup.reelMaxDragN ? 'ok' : setup.dragN <= setup.reelMaxDragN ? 'warn' : 'fail', 'Drag setting uses ' + (setup.dragN / setup.reelMaxDragN * 100).toFixed(0) + '% of the reel\'s rated maximum.');
    if (setup.rodLineMaxN > 0 && setup.lineBreakN > 0) { const inR = setup.lineBreakN >= (setup.rodLineMinN || 0) && setup.lineBreakN <= setup.rodLineMaxN; add('Line vs rod line rating', inR ? 'ok' : setup.lineBreakN > setup.rodLineMaxN * 1.3 ? 'fail' : 'warn', inR ? 'Line strength is inside the rod\'s labelled line range.' : 'Line strength is outside the rod\'s labelled line range.'); }
    if (setup.rodLureMaxG > 0 && setup.lureG > 0) { const inR = setup.lureG >= (setup.rodLureMinG || 0) && setup.lureG <= setup.rodLureMaxG; add('Lure vs rod lure rating', inR ? 'ok' : setup.lureG > setup.rodLureMaxG * 1.25 ? 'fail' : 'warn', inR ? 'Lure weight is inside the rod\'s labelled casting range.' : 'Lure weight is outside the rod\'s labelled casting range.'); }
    if (result && isFinite(result.safetyFactor)) add('Blank bending stress at drag', result.safetyFactor >= 2 ? 'ok' : result.safetyFactor >= 1.2 ? 'warn' : 'fail', 'Peak outer-fibre stress ' + (result.maxStress / 1e6).toFixed(0) + ' MPa at ' + (result.maxStressAt / result.s[result.s.length - 1] * 100).toFixed(0) + '% of length; safety factor ' + result.safetyFactor.toFixed(2) + ' against the nominal material strength.');
    return c;
  }

  /* Multi-objective design search. Enumerates the design grid, solves each rod at the design load and returns
     every design with: feasibility (safety factor >= minSF), performance score, mass, cost index, Pareto flag
     (non-dominated in {maximise performance, minimise mass, minimise cost}) and a weighted score.
     performance = max(0, 1 - |deflRatio - target| / target): how closely the rod loads to the target curve depth. */
  function optimize(space, load, prefs) {
    const out = [];
    for (const material of space.materials) for (const length of space.lengths) for (const buttOD of space.buttODs) for (const tipOD of space.tipODs) for (const wallRatio of space.wallRatios) {
      if (tipOD >= buttOD) continue;
      const cfg = { material, length, buttOD, tipOD, wallRatio }; if (space.costIndex && space.costIndex[material] != null) cfg.costIndex = space.costIndex[material];
      const r = solveRod(cfg, load.T, load.beta, load.theta0, { N: 60 }), m = mass(cfg);
      out.push({ cfg, mass: m, cost: m * matOf(cfg).costIndex, deflRatio: r.deflRatio, sf: r.safetyFactor, energy: r.strainEnergy,
        performance: Math.max(0, 1 - Math.abs(r.deflRatio - prefs.targetDefl) / prefs.targetDefl), feasible: r.safetyFactor >= prefs.minSF });
    }
    const feas = out.filter(d => d.feasible);
    for (const d of feas) d.pareto = !feas.some(o => o !== d && o.performance >= d.performance && o.mass <= d.mass && o.cost <= d.cost && (o.performance > d.performance || o.mass < d.mass || o.cost < d.cost));
    const rng = k => { const v = feas.map(d => d[k]); return [Math.min(...v), Math.max(...v)]; };
    if (feas.length) {
      const [m0, m1] = rng('mass'), [c0, c1] = rng('cost'), wS = (prefs.wP + prefs.wW + prefs.wC) || 1;
      for (const d of feas) d.score = (prefs.wP * d.performance + prefs.wW * (m1 > m0 ? 1 - (d.mass - m0) / (m1 - m0) : 1) + prefs.wC * (c1 > c0 ? 1 - (d.cost - c0) / (c1 - c0) : 1)) / wS;
      feas.sort((a, b) => b.score - a.score);
    }
    return { all: out, feasible: feas, best: feas[0] || null, evaluated: out.length };
  }

  root.CLPhysics = { G, MATERIALS, LINES, matOf, section, mass, solveRod, loadCurve, linearTipDeflection, benchDrop, compareBench, fitModulus, lineStretch, compatibility, optimize };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.CLPhysics;
})(typeof window !== 'undefined' ? window : globalThis);
