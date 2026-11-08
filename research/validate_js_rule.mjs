#!/usr/bin/env node
// Replays SisFall through the fall rule exactly as it ships in ../index.html.
// The code between "BEGIN fall-rule-v2" and "END fall-rule-v2" is extracted from the page
// and evaluated as-is, so this tests the deployed JavaScript, not a re-implementation.
//
// Usage: node research/validate_js_rule.mjs /path/to/SisFall_dataset
// If research/results/rule_v2_trials.json exists (written by rule_v2.py), also reports
// per-trial agreement with the Python reference implementation.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, '..', 'index.html'), 'utf8');
const m = html.match(/\/\/ BEGIN fall-rule-v2[^\n]*\n([\s\S]*?)\/\/ END fall-rule-v2/);
if (!m) { console.error('fall-rule-v2 block not found in index.html'); process.exit(1); }
const { createFallDetectorV2, FALL_V2 } = new Function(m[1] + '\nreturn {createFallDetectorV2, FALL_V2};')();

const root = process.argv[2];
if (!root) { console.error('usage: node validate_js_rule.mjs /path/to/SisFall_dataset'); process.exit(1); }
const G = 9.80665, FS = 50, K = 200 / FS, CAL = 2 * FS;
const DEV = new Set([...Array(12)].map((_, i) => 'SA' + String(i + 1).padStart(2, '0'))
  .concat(['SE01', 'SE02', 'SE03', 'SE04', 'SE05', 'SE07']));

function load(file) {
  const raw = [];
  for (const line of fs.readFileSync(file, 'latin1').split('\n')) {
    const p = line.trim().replace(/;$/, '').split(',');
    if (p.length >= 3 && p[0].trim() !== '') raw.push([+p[0], +p[1], +p[2]]);
  }
  const out = [];
  for (let i = 0; i + K <= raw.length; i += K) {
    const v = [0, 0, 0];
    for (let j = 0; j < K; j++) for (let a = 0; a < 3; a++) v[a] += raw[i + j][a];
    out.push(v.map(c => (c / K) * (32 / 8192) * G));
  }
  return out;
}

function runTrial(acc) {
  const mags = acc.map(v => Math.hypot(v[0], v[1], v[2]));
  const baseline = mags.slice(0, CAL).reduce((a, b) => a + b, 0) / CAL;
  const det = createFallDetectorV2();
  for (let i = CAL; i < acc.length; i++) {
    const ev = det.step({ t: i * 1000 / FS, mag: mags[i], x: acc[i][0], y: acc[i][1], z: acc[i][2] }, baseline, 1);
    if (ev && ev.type === 'fall') return true;
  }
  return false;
}

const results = {};
const per = { test: { f: {}, a: {} }, all: { f: {}, a: {} } };
for (const subj of fs.readdirSync(root).sort()) {
  const dir = path.join(root, subj);
  if (!fs.statSync(dir).isDirectory()) continue;
  for (const fn of fs.readdirSync(dir).sort()) {
    const mm = fn.match(/^([DF]\d\d)_(S[AE]\d\d)_R\d\d\.txt$/);
    if (!mm) continue;
    const acc = load(path.join(dir, fn));
    if (acc.length <= 3 * FS) continue;
    const hit = runTrial(acc);
    const id = fn.slice(0, -4), fall = mm[1][0] === 'F';
    results[id] = hit;
    for (const split of DEV.has(mm[2]) ? ['all'] : ['all', 'test']) {
      const d = per[split][fall ? 'f' : 'a'];
      d[mm[2]] = d[mm[2]] || [0, 0];
      d[mm[2]][0] += fall ? hit : !hit;
      d[mm[2]][1] += 1;
    }
  }
}
const pooled = d => { let k = 0, n = 0; for (const v of Object.values(d)) { k += v[0]; n += v[1]; } return [k / n, n]; };
const fmt = ([p, n]) => `${(100 * p).toFixed(1)}% (n=${n})`;
console.log('params', JSON.stringify(FALL_V2));
for (const split of ['test', 'all'])
  console.log(`${split.padEnd(4)} sensitivity ${fmt(pooled(per[split].f))}  specificity ${fmt(pooled(per[split].a))}`);

const ref = path.join(here, 'results', 'rule_v2_trials.json');
if (fs.existsSync(ref)) {
  const py = JSON.parse(fs.readFileSync(ref, 'utf8'));
  const ids = Object.keys(results).filter(k => k in py);
  const diff = ids.filter(k => py[k] !== results[k]);
  console.log(`agreement with Python reference: ${ids.length - diff.length}/${ids.length} trials` +
    (diff.length ? `; differing: ${diff.slice(0, 15).join(', ')}${diff.length > 15 ? ' …' : ''}` : ''));
}
fs.writeFileSync(path.join(here, 'results', 'js_rule_trials.json'), JSON.stringify(results));
