// Drives index.html in headless Chromium with synthetic DeviceMotion events and a fake clock.
// Usage: node research/browser_smoke.cjs   (needs the playwright package and a Chromium build)
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  let failed = 0;
  async function scenario(name, afterVec, impact=true, ai=false){
    const expectAlert = name.includes('lying');
    const page = await browser.newPage(); if(ai) await page.addInitScript(()=>{window.__ai=true;});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.clock.install({ time: 1_700_000_000_000 });
    await page.goto('file://' + require('path').resolve(__dirname, '..', 'index.html'));
    await page.evaluate(() => { window.speechSynthesis && (window.speechSynthesis.speak = ()=>{}); grantAccess(); if(window.__ai){ const g=generateSyntheticSamples(); model=trainMLP(g.samples,g.labels,6,300); } });
    const send = async (v, ms) => {
      for (let t = 0; t < ms; t += 20) {
        await page.evaluate(([x,y,z]) => window.dispatchEvent(new DeviceMotionEvent('devicemotion',
          { accelerationIncludingGravity:{x,y,z}, rotationRate:{alpha:0,beta:0,gamma:0} })), v);
        await page.clock.runFor(20);
      }
    };
    await send([0, 9.81, 0], 3000);            // calibration (2 s) + 1 s upright
    if (impact) await send([5, 28, 8], 100);   // ~30 m/s² impact
    await send([0, 2, 0], 200);                // brief rebound
    await send(afterVec, 6000);                // rest for 6 s
    const shown = await page.evaluate(() => document.getElementById('checkinOverlay').classList.contains('show'));
    const chip = await page.evaluate(() => document.getElementById('ruleChip').textContent);
    const ok = shown === expectAlert && errors.length === 0; if(!ok) failed++;
    console.log(`${ok?'PASS':'FAIL'} ${name.padEnd(36)} alert=${shown} chip="${chip}" errors=${JSON.stringify(errors)}`);
    await page.close();
  }
  await scenario('fall: impact, then lying still', [9.81, 0, 0.3]);
  await scenario('no fall: impact, then upright still', [0.3, 9.81, 0]);
  await scenario('no impact: tilted still', [9.81, 0, 0], false);
  await scenario('AI on: impact, then lying still', [9.81,0,0.3], true, true);
  await scenario('AI on: walking-ish, upright', [0.5,10.5,0.5], false, true);
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
