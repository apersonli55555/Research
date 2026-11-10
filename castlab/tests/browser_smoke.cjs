/* Headless browser smoke test: serves castlab/ and drives every section.
   Usage: node castlab/tests/browser_smoke.cjs [outDir]   (needs Playwright + Chromium) */
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..'), out = process.argv[2] || path.join(root, 'tests', 'out');
fs.mkdirSync(out, { recursive: true });
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.json': 'application/json' };
const srv = http.createServer((q, r) => { const p = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/^\/$/, '/index.html')); fs.readFile(p, (e, d) => { if (e) { r.writeHead(404); r.end(); return; } r.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream' }); r.end(d); }); });

(async () => {
  await new Promise(r => srv.listen(8765, r));
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  const browser = await chromium.launch(proxy ? { proxy: { server: proxy, bypass: '127.0.0.1,localhost' }, args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] } : { args: ['--use-gl=swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = []; page.on('pageerror', e => errors.push('pageerror: ' + e.message)); page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  const shot = n => page.screenshot({ path: path.join(out, n + '.png'), fullPage: false });
  const step = async (name, fn) => { try { await fn(); console.log('ok   ' + name); } catch (e) { console.log('FAIL ' + name + ': ' + e.message); errors.push(name + ': ' + e.message); } };

  await page.goto('http://127.0.0.1:8765/index.html');
  await step('register', async () => { await page.click('[data-auth=register]'); await page.fill('#authDisplay', 'Test Angler'); await page.fill('#authUser', 'tester'); await page.fill('#authPass', 'correct horse'); await page.click('#authSubmit'); await page.waitForSelector('#appView:not([hidden])'); });
  await step('add location', async () => { await page.goto('http://127.0.0.1:8765/index.html#map'); await page.fill('#locName', 'Lake Harriet'); await page.fill('#locLat', '44.9221'); await page.fill('#locLon', '-93.3053'); await page.click('#locForm button[type=submit]'); await page.waitForSelector('#locTable td'); await shot('map'); });
  await step('inventory', async () => { await page.goto('http://127.0.0.1:8765/index.html#baitradar'); await page.fill('#brName', 'Spinnerbait'); await page.click('#brMore summary'); await page.fill('#brWeight', '10.6'); await page.fill('#brQty', '3'); await page.click('#brForm button[type=submit]'); await page.waitForSelector('#brTable td'); });
  await step('log sessions', async () => {
    await page.goto('http://127.0.0.1:8765/index.html#lurelab');
    for (const [c, h] of [[2, 2], [0, 1.5], [1, 3]]) { await page.fill('#sesLure', 'Spinnerbait'); await page.fill('#sesCatch', String(c)); await page.fill('#sesHours', String(h)); await page.fill('#sesUsed', '1'); await page.click('#sesForm button[type=submit]'); }
    const n = await page.$$eval('#sesTable tr', r => r.length); if (n !== 4) throw new Error('expected 3 rows, got ' + (n - 1));
    const qty = await page.evaluate(() => Store.list('inventory')[0].qty); if (qty !== 0) throw new Error('inventory not decremented: ' + qty);
  });
  await step('generate demo', async () => { await page.goto('http://127.0.0.1:8765/index.html#account'); await page.click('#acDemo'); await page.waitForTimeout(200); const n = await page.evaluate(() => Store.list('sessions').filter(s => s.source === 'simulated').length); if (n !== 240) throw new Error('demo n=' + n); });
  await step('lurelab estimates (sim)', async () => { await page.goto('http://127.0.0.1:8765/index.html#lurelab'); await page.selectOption('#llData', 'simulated'); await page.click('#llTabs [data-t=estimates]'); await page.waitForSelector('#estTable table'); await shot('lurelab-estimates'); });
  await step('lurelab model (sim)', async () => { await page.click('#llTabs [data-t=model]'); await page.waitForSelector('#mdlTable table'); const cov = await page.$$eval('#mdlTable .badge', b => b.map(x => x.textContent)); console.log('     coverage badges:', cov.join(' ')); await page.click('#predGo'); await page.waitForSelector('#predOut .kpi'); await page.click('#predLog'); await shot('lurelab-model'); });
  await step('lurelab validation (sim)', async () => { await page.click('#llTabs [data-t=validation]'); await page.click('#valRun'); await page.waitForSelector('#valKpis .kpi .v', { timeout: 30000 }); console.log('     ' + (await page.textContent('#valKpis')).replace(/\s+/g, ' ').slice(0, 220)); await shot('lurelab-validation'); });
  await step('experiment', async () => {
    await page.selectOption('#llData', 'user'); await page.click('#llTabs [data-t=experiments]'); await page.fill('#expTitle', 'Colour test'); await page.fill('#expA', 'White spinnerbait'); await page.fill('#expB', 'Chartreuse spinnerbait'); await page.click('#expForm button[type=submit]'); await page.waitForSelector('#trialForm');
    for (const f of [1, 0, 2, 1, 0, 3]) { await page.fill('#trFish', String(f)); await page.click('#trialForm button[type=submit]'); await page.waitForTimeout(50); }
    await page.waitForSelector('#expDetail .stat-line'); console.log('     ' + (await page.textContent('#expDetail')).replace(/\s+/g, ' ').match(/Rate ratio[^)]*\)/)[0]); await shot('lurelab-experiment');
  });
  await step('gearsim', async () => { await page.goto('http://127.0.0.1:8765/index.html#gearsim'); await page.waitForSelector('#gsTable table'); await page.click('#gsAnimate'); await page.waitForTimeout(800); await shot('gearsim'); await page.waitForTimeout(1600); });
  await step('compat via gearmatch', async () => { await page.goto('http://127.0.0.1:8765/index.html#gearmatch'); await page.selectOption('#gmSpecies', 'Largemouth bass'); await page.click('#gmGo'); await page.click('#gmToSim'); await page.waitForSelector('#cpOut2 .stat-line'); await shot('gearsim-compat'); });
  await step('optimisation', async () => { await page.click('#gsTabs [data-t=opt]'); await page.click('#opGo'); await page.waitForSelector('#opTable table', { timeout: 60000 }); console.log('     ' + await page.textContent('#opInfo')); await shot('gearsim-opt'); });
  await step('bench test', async () => { await page.click('#gsTabs [data-t=bench]'); const ins = await page.$$('#bnRows input[data-f=mm]'); for (const [i, v] of [[0, 140], [1, 255], [2, 430], [3, 560]]) await ins[i].fill(String(v)); await page.click('#bnRun'); await page.waitForSelector('#bnOut .kpi'); await page.fill('#bnName', 'smoke'); await page.click('#bnSave'); console.log('     ' + (await page.textContent('#bnOut .grid')).replace(/\s+/g, ' ')); await shot('gearsim-bench'); });
  await step('castcheck synthetic', async () => { await page.goto('http://127.0.0.1:8765/index.html#castcheck'); await page.click('#ccDemo'); await page.waitForSelector('#ccStats .stat-line'); await page.fill('#ccLvl', '2'); await page.dispatchEvent('#ccLvl', 'input'); await page.waitForTimeout(400); console.log('     ' + (await page.textContent('#ccStats')).replace(/\s+/g, ' ').slice(0, 200)); await shot('castcheck-demo'); });
  await step('castcheck real terrain', async () => { await page.selectOption('#ccLoc', { label: 'Lake Harriet' }); await page.click('#ccLoad'); await page.waitForFunction(() => /Detected water|No flat water|Could not load/.test(document.querySelector('#ccSource').textContent), null, { timeout: 60000 }); console.log('     ' + (await page.textContent('#ccSource')).slice(0, 260)); await page.fill('#ccLvl', '1.5'); await page.dispatchEvent('#ccLvl', 'input'); await page.waitForTimeout(800); await shot('castcheck-real'); });
  await step('usgs history', async () => { await page.click('#ccTabs [data-t=water]'); await page.waitForFunction(() => document.querySelector('#ccGage').options.length && !/Searching/.test(document.querySelector('#ccGage').textContent), null, { timeout: 40000 }); console.log('     gage: ' + (await page.textContent('#ccGage option'))); if (await page.$eval('#ccGage', s => !!s.value)) { await page.click('#ccGageGo'); await page.waitForSelector('#ccWaterKpis .kpi, #ccWaterKpis .note', { timeout: 40000 }); console.log('     ' + (await page.textContent('#ccWaterKpis')).replace(/\s+/g, ' ').slice(0, 260)); await shot('castcheck-water'); } });
  await step('field observation', async () => { await page.click('#ccTabs [data-t=obs]'); await page.fill('#obsLat', '44.9190'); await page.fill('#obsLon', '-93.3100'); await page.fill('#obsAcc', '4'); await page.click('#obsForm button[type=submit]'); await page.waitForTimeout(300); console.log('     ' + (await page.textContent('#obsSummary')).replace(/\s+/g, ' ').slice(0, 200)); });
  await step('weather', async () => { await page.goto('http://127.0.0.1:8765/index.html#weather'); await page.waitForFunction(() => /Source|Could not/.test(document.querySelector('#wxNow').textContent), null, { timeout: 30000 }); console.log('     ' + (await page.textContent('#wxNow')).replace(/\s+/g, ' ').slice(0, 160)); });
  await step('dashboard', async () => { await page.goto('http://127.0.0.1:8765/index.html#dashboard'); await page.waitForSelector('#dashKpis .kpi'); await shot('dashboard'); });
  await step('research + report', async () => { await page.goto('http://127.0.0.1:8765/index.html#research'); console.log('     ' + (await page.textContent('#rsVerify')).replace(/\s+/g, ' ').slice(0, 200)); const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#rsReport')]); await dl.saveAs(path.join(out, 'report.html')); });
  await step('mobile layout', async () => { await page.setViewportSize({ width: 390, height: 844 }); await page.goto('http://127.0.0.1:8765/index.html#lurelab'); await page.waitForTimeout(300); const sw = await page.evaluate(() => document.documentElement.scrollWidth); if (sw > 392) throw new Error('horizontal overflow ' + sw); await shot('mobile'); });
  await step('dark theme', async () => { await page.setViewportSize({ width: 1400, height: 1000 }); await page.emulateMedia({ colorScheme: 'dark' }); await page.goto('http://127.0.0.1:8765/index.html#gearsim'); await page.waitForTimeout(500); await shot('dark'); });

  console.log('\nerrors (' + errors.length + '):\n' + errors.join('\n'));
  await browser.close(); srv.close(); process.exit(errors.length ? 1 : 0);
})();
