/* CastLab statistics engine — pure functions, no DOM. Works in the browser (window.CLStats) and Node (module.exports).
   Methods: conjugate Beta-Binomial and Gamma-Poisson models, Bayesian logistic regression (MAP + Laplace
   approximation), proper scoring rules and prequential (forward-chaining) validation. */
(function (root) {
  'use strict';

  // ---------- seeded random numbers (reproducible Monte Carlo) ----------
  function rng(seed) { // mulberry32
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function randGamma(r, k) { // Marsaglia & Tsang (2000), unit scale
    if (k < 1) return randGamma(r, k + 1) * Math.pow(r() || 1e-12, 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v; do { x = randn(r); v = 1 + c * x; } while (v <= 0);
      v = v * v * v; const u = r();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  function randBeta(r, a, b) { const x = randGamma(r, a), y = randGamma(r, b); return x / (x + y); }

  // ---------- special functions ----------
  function lgamma(x) { // Lanczos, g=7
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1; let a = c[0]; const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  function betacf(a, b, x) {
    const FP = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FP) d = FP; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d;
      const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-14) break;
    }
    return h;
  }
  function betaCdf(x, a, b) { // regularized incomplete beta I_x(a,b)
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function betaPdf(x, a, b) {
    if (x <= 0 || x >= 1) return 0;
    return Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x));
  }
  function bisect(f, lo, hi, target) { for (let i = 0; i < 80; i++) { const m = 0.5 * (lo + hi); if (f(m) < target) lo = m; else hi = m; } return 0.5 * (lo + hi); }
  function betaQuantile(p, a, b) { return bisect(x => betaCdf(x, a, b), 0, 1, p); }
  function gammaP(a, x) { // regularized lower incomplete gamma
    if (x <= 0) return 0;
    if (x < a + 1) { let ap = a, sum = 1 / a, del = sum; for (let n = 0; n < 500; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-14) break; } return sum * Math.exp(-x + a * Math.log(x) - lgamma(a)); }
    const FP = 1e-300; let b = x + 1 - a, c = 1 / FP, d = 1 / b, h = d;
    for (let i = 1; i < 500; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < FP) d = FP; c = b + an / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-14) break; }
    return 1 - Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
  }
  function gammaQuantile(p, shape, rate) {
    let hi = (shape + 10 * Math.sqrt(shape) + 10) / rate;
    while (gammaP(shape, hi * rate) < p) hi *= 2;
    return bisect(x => gammaP(shape, x * rate), 0, hi, p);
  }

  // ---------- conjugate models ----------
  /* Beta-Binomial: successes s out of n sessions, prior Beta(a0,b0). Returns posterior summary with an
     equal-tailed credible interval at level `cred`. */
  function betaPosterior(s, n, a0, b0, cred) {
    a0 = a0 == null ? 1 : a0; b0 = b0 == null ? 1 : b0; cred = cred || 0.95;
    const a = a0 + s, b = b0 + (n - s), t = (1 - cred) / 2;
    return { a, b, n, s, mean: a / (a + b), lo: betaQuantile(t, a, b), hi: betaQuantile(1 - t, a, b), cred };
  }
  /* P(theta_A > theta_B) for independent Beta posteriors, by numerical integration of pdf_A(x) * cdf_B(x). */
  function probGreater(A, B) {
    const N = 2000; let sum = 0;
    for (let i = 0; i < N; i++) { const x = (i + 0.5) / N; sum += betaPdf(x, A.a, A.b) * betaCdf(x, B.a, B.b); }
    return Math.min(1, Math.max(0, sum / N));
  }
  /* Credible interval for theta_A - theta_B by seeded Monte Carlo. */
  function betaDiff(A, B, seed, draws) {
    const r = rng(seed || 12345), d = []; draws = draws || 20000;
    for (let i = 0; i < draws; i++) d.push(randBeta(r, A.a, A.b) - randBeta(r, B.a, B.b));
    d.sort((x, y) => x - y);
    return { mean: mean(d), lo: d[Math.floor(0.025 * draws)], hi: d[Math.floor(0.975 * draws)], draws };
  }
  /* Gamma-Poisson: `count` fish in `exposure` hours, prior Gamma(shape a0, rate b0). Posterior over fish/hour. */
  function gammaPosterior(count, exposure, a0, b0, cred) {
    a0 = a0 == null ? 0.5 : a0; b0 = b0 == null ? 0.5 : b0; cred = cred || 0.95;
    const shape = a0 + count, rate = b0 + exposure, t = (1 - cred) / 2;
    return { shape, rate, count, exposure, mean: shape / rate, lo: gammaQuantile(t, shape, rate), hi: gammaQuantile(1 - t, shape, rate), cred };
  }
  function gammaPdf(x, shape, rate) { return x <= 0 ? 0 : Math.exp(shape * Math.log(rate) - lgamma(shape) + (shape - 1) * Math.log(x) - rate * x); }
  /* Posterior of the rate ratio lambda_A / lambda_B by seeded Monte Carlo. */
  function rateRatio(A, B, seed, draws) {
    const r = rng(seed || 2468), q = []; draws = draws || 20000; let gt = 0;
    for (let i = 0; i < draws; i++) { const a = randGamma(r, A.shape) / A.rate, b = randGamma(r, B.shape) / B.rate; q.push(a / b); if (a > b) gt++; }
    q.sort((x, y) => x - y);
    return { median: q[Math.floor(0.5 * draws)], lo: q[Math.floor(0.025 * draws)], hi: q[Math.floor(0.975 * draws)], pGreater: gt / draws, draws };
  }

  // ---------- basic descriptive ----------
  function mean(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN; }
  function sd(a) { if (a.length < 2) return NaN; const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / (a.length - 1)); }
  function median(a) { if (!a.length) return NaN; const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]); }
  function rmse(a) { return a.length ? Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length) : NaN; }

  // ---------- linear algebra (small dense) ----------
  function invert(Ain) {
    const n = Ain.length, A = Ain.map((r, i) => r.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      if (Math.abs(A[p][c]) < 1e-14) throw new Error('singular matrix');
      [A[c], A[p]] = [A[p], A[c]];
      const d = A[c][c]; for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
      for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; if (f) for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
    }
    return A.map(r => r.slice(n));
  }
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  /* Bayesian logistic regression. X: rows of features (first column should be 1 for the intercept),
     y: 0/1, priorSd: per-coefficient Normal(0, sd) prior. MAP by Newton-Raphson; posterior covariance from the
     inverse Hessian at the mode (Laplace approximation). */
  function logisticFit(X, y, priorSd) {
    const n = X.length, p = X[0].length, beta = new Array(p).fill(0), prec = priorSd.map(s => 1 / (s * s));
    let H, converged = false, iter = 0;
    for (; iter < 60; iter++) {
      const g = beta.map((b, j) => b * prec[j]); H = Array.from({ length: p }, (_, j) => { const r = new Array(p).fill(0); r[j] = prec[j]; return r; });
      for (let i = 0; i < n; i++) {
        let eta = 0; const xi = X[i]; for (let j = 0; j < p; j++) eta += xi[j] * beta[j];
        const mu = sigmoid(eta), w = mu * (1 - mu), e = mu - y[i];
        for (let j = 0; j < p; j++) { if (!xi[j]) continue; g[j] += xi[j] * e; const hj = H[j], xw = xi[j] * w; for (let k = 0; k < p; k++) hj[k] += xw * xi[k]; }
      }
      const Hinv = invert(H); let maxStep = 0;
      for (let j = 0; j < p; j++) { let st = 0; for (let k = 0; k < p; k++) st += Hinv[j][k] * g[k]; beta[j] -= st; maxStep = Math.max(maxStep, Math.abs(st)); }
      if (maxStep < 1e-8) { converged = true; break; }
    }
    const cov = invert(H);
    return { beta, cov, se: cov.map((r, j) => Math.sqrt(Math.max(0, r[j]))), converged, iterations: iter + 1, n };
  }
  function logisticPredict(fit, x, z) {
    z = z || 1.96; let eta = 0; const p = x.length;
    for (let j = 0; j < p; j++) eta += x[j] * fit.beta[j];
    let v = 0; for (let j = 0; j < p; j++) { if (!x[j]) continue; for (let k = 0; k < p; k++) v += x[j] * fit.cov[j][k] * x[k]; }
    const s = Math.sqrt(Math.max(0, v));
    return { p: sigmoid(eta), lo: sigmoid(eta - z * s), hi: sigmoid(eta + z * s), eta, se: s };
  }

  // ---------- scoring / validation ----------
  function brier(p, y) { return mean(p.map((v, i) => (v - y[i]) * (v - y[i]))); }
  function logLoss(p, y) { return mean(p.map((v, i) => { const q = Math.min(1 - 1e-9, Math.max(1e-9, v)); return -(y[i] ? Math.log(q) : Math.log(1 - q)); })); }
  function calibration(p, y, bins) {
    bins = bins || 5; const out = [];
    for (let b = 0; b < bins; b++) out.push({ lo: b / bins, hi: (b + 1) / bins, n: 0, sumP: 0, sumY: 0 });
    p.forEach((v, i) => { const b = out[Math.min(bins - 1, Math.floor(v * bins))]; b.n++; b.sumP += v; b.sumY += y[i]; });
    return out.map(b => Object.assign(b, { meanP: b.n ? b.sumP / b.n : NaN, obs: b.n ? b.sumY / b.n : NaN, ci: b.n ? betaPosterior(b.sumY, b.n, 0.5, 0.5) : null }));
  }

  root.CLStats = { rng, randn, randGamma, randBeta, lgamma, betaCdf, betaPdf, betaQuantile, gammaP, gammaQuantile, gammaPdf,
    betaPosterior, probGreater, betaDiff, gammaPosterior, rateRatio, mean, sd, median, rmse, invert, sigmoid,
    logisticFit, logisticPredict, brier, logLoss, calibration };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.CLStats;
})(typeof window !== 'undefined' ? window : globalThis);
