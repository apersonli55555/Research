/* CastLab field tools: Today page, best fishing times, fish reported nearby (iNaturalist), catch log,
   bait/gear suggestions from local forage, and water-body lookup on the map (OpenStreetMap Nominatim).
   Guidance here is labelled as guidance; counts and records come from named public sources or the user's own log. */
(function () {
  'use strict';
  const { $, $$, esc, fmt, pct } = U, A = CLAstro, S = CLStats;
  const HOUR = 36e5;

  /* ================= shared helpers ================= */
  CL.ensureLocation = function (name, lat, lon, type) {
    const near = Store.list('locations').find(l => isFinite(l.lat) && U.haversine(l.lat, l.lon, lat, lon) < 200);
    if (near) return near;
    const l = Store.add('locations', { name: name || ('Spot ' + lat.toFixed(4) + ', ' + lon.toFixed(4)), lat, lon, type: type || 'Other', notes: '' });
    if (!Store.data.prefs.homeLocation) { Store.data.prefs.homeLocation = l.id; Store.save(); }
    return l;
  };
  const spot = () => Store.data.prefs.currentSpot && Store.location(Store.data.prefs.currentSpot) ? Store.data.prefs.currentSpot : Store.data.prefs.homeLocation;
  const setSpot = id => { Store.data.prefs.currentSpot = id; Store.save('prefs'); };
  const clock = (ms, off) => { const d = new Date(ms + off * 1000), h = d.getUTCHours(), m = d.getUTCMinutes(); return ((h + 11) % 12 + 1) + (m ? ':' + String(m).padStart(2, '0') : '') + (h < 12 ? ' am' : ' pm'); };
  const dayName = (ms, off, nowMs) => { const day = t => Math.floor((t + off * 1000) / 864e5), d = day(ms) - day(nowMs); return d === 0 ? 'Today' : d === 1 ? 'Tomorrow' : new Date(ms + off * 1000).toLocaleDateString(undefined, { weekday: 'long', timeZone: 'UTC' }); };
  const rating = i => i >= 75 ? ['Excellent', 'ok'] : i >= 60 ? ['Good', 'ok'] : i >= 45 ? ['Fair', 'warn'] : ['Slow', ''];
  const typeFromOSM = t => /river|stream|canal/.test(t) ? 'River / stream' : /pond/.test(t) ? 'Pond' : /bay|coast|sea|strait/.test(t) ? 'Coastal / estuary' : /lake|reservoir|water/.test(t) ? 'Lake / reservoir' : 'Other';

  /* ================= best fishing times ================= */
  function ownFactor() {
    const ses = Store.list('sessions').filter(s => s.source !== 'simulated' && s.timeOfDay); if (ses.length < 20) return null;
    const all = ses.filter(s => s.catches > 0).length / ses.length, bins = {};
    ses.forEach(s => { const b = bins[s.timeOfDay] = bins[s.timeOfDay] || { n: 0, k: 0 }; b.n++; if (s.catches > 0) b.k++; });
    let mx = 0; Object.values(bins).forEach(b => { b.mean = S.betaPosterior(b.k, b.n, 4 * all || 0.5, 4 * (1 - all) || 0.5).mean; if (b.n >= 3) mx = Math.max(mx, b.mean); });
    if (!mx) return null; Object.values(bins).forEach(b => b.rel = b.n >= 3 ? b.mean / mx : null); return { n: ses.length, bins };
  }
  const btCache = {};
  CL.bestTimes = async function (loc) {
    const key = loc.id + ':' + Math.floor(Date.now() / (20 * 60000)); if (btCache[key]) return btCache[key];
    const w = await U.fetchWeather(loc.lat, loc.lon), h = w.hourly, off = w.utcOffset, toMs = i => Date.parse(h.time[i] + ':00Z') - off * 1000;
    const start = w.index, end = h.time.length, ev = A.events(toMs(start) - 6 * HOUR, loc.lat, loc.lon, Math.ceil((end - start) + 12)), own = ownFactor(), all = [];
    for (let i = start; i < end; i++) {
      const ms = toMs(i), sc = A.scoreHour({ ms, dPressure3h: i >= 3 ? h.surface_pressure[i] - h.surface_pressure[i - 3] : null, windKph: h.wind_speed_10m[i], precipMm: h.precipitation[i], code: h.weather_code[i], ev, sunAlt: A.sunAltitude(ms + HOUR / 2, loc.lat, loc.lon) });
      let index = sc.index; const localHour = +h.time[i].slice(11, 13), ob = own && own.bins[CL.timeOfDay(localHour)];
      if (ob && ob.rel != null && !sc.unsafe) index = Math.round(0.75 * index + 25 * ob.rel);
      all.push({ ms, i, label: h.time[i], localHour, index, rule: sc.index, why: sc.why, unsafe: sc.unsafe, c: sc.components, tempC: h.temperature_2m[i], windKph: h.wind_speed_10m[i] });
    }
    const hours = all.slice(0, 48), top = Math.max(...hours.map(x => x.index)), wins = A.windows(hours, Math.max(50, top - 12)).slice(0, 4), now = toMs(start);
    const first = k => (ev[k].find(t => t >= now - 12 * HOUR && t <= now + 24 * HOUR));
    return btCache[key] = { w, off, hours, all, loc, wins, ev, own, now, phase: A.moonPhase(now), sunrise: ev.sunrise.find(t => t > now - 14 * HOUR), sunset: ev.sunset.find(t => t > now - 6 * HOUR), first };
  };
  /* The week ahead: one chip per day (best hour, score, 24-hour sparkline) and the hourly grid for the chosen day. */
  let selDay = 0;
  function daysOf(bt) {
    const o = bt.off, dayOf = ms => Math.floor((ms + o * 1000) / 864e5), d0 = dayOf(bt.now), m = {};
    bt.all.forEach(x => { const k = dayOf(x.ms) - d0; (m[k] = m[k] || []).push(x); });
    return Object.keys(m).map(Number).sort((a, b) => a - b).filter(k => k === 0 || m[k].length >= 12).slice(0, 7).map(k => {
      const hs = m[k], safe = hs.filter(x => !x.unsafe), pool = safe.length ? safe : hs, best = pool.reduce((a, x) => x.index > a.index ? x : a, pool[0]), d = new Date(hs[0].ms + o * 1000);
      return { k, hs, best, storm: hs.some(x => x.unsafe), name: k === 0 ? 'Today' : k === 1 ? 'Tomorrow' : d.toLocaleDateString(undefined, { weekday: 'short', timeZone: 'UTC' }), date: d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' }) };
    });
  }
  CL.daysOf = daysOf;
  const hcls = x => x.unsafe ? 'h-bad' : x.index >= 75 ? 'h-top' : x.index >= 60 ? 'h-good' : x.index >= 45 ? 'h-fair' : 'h-slow';
  function timesHTML(bt) {
    const hr = h => ((h + 11) % 12 + 1) + (h < 12 ? ' am' : ' pm'), days = daysOf(bt);
    if (!days.some(d => d.k === selDay)) selDay = 0;
    const day = days.find(d => d.k === selDay), best = day.best;
    const spark = d => '<svg class="spark" viewBox="0 0 96 26" preserveAspectRatio="none" aria-hidden="true">' + d.hs.map(x => { const hgt = Math.max(2, x.index / 100 * 26); return '<rect x="' + (x.localHour * 4) + '" y="' + (26 - hgt).toFixed(1) + '" width="3" height="' + hgt.toFixed(1) + '" rx="1" class="' + hcls(x) + '"/>'; }).join('') + '</svg>';
    return '<div class="week">' + days.map(d => '<button type="button" class="daychip' + (d.k === selDay ? ' on' : '') + '" data-day="' + d.k + '" aria-pressed="' + (d.k === selDay) + '"><span class="dn">' + d.name + '</span><span class="dd">' + d.date + '</span><b class="' + hcls(d.best) + '">' + d.best.index + '</b>' + spark(d) + '<span class="dh">' + (d.storm && d.best.unsafe ? 'storms' : hr(d.best.localHour)) + '</span></button>').join('') + '</div>' +
      '<h4 style="margin:14px 0 6px">' + day.name + (day.k === 0 && day.hs[0].localHour > 0 ? ' (from now)' : ', ' + day.date) + '</h4><div class="hours">' + day.hs.map((x, i) => '<div class="hcell ' + hcls(x) + (x === best && !x.unsafe ? ' h-best' : '') + '" style="--i:' + i + '" title="' + esc(x.why.join(', ') || 'steady conditions') + '"><span>' + hr(x.localHour) + '</span><b>' + x.index + '</b></div>').join('') + '</div>' +
      '<p class="small" style="margin:8px 0 0">Best hour: <b>' + hr(best.localHour) + '</b> (' + best.index + '/100, ' + rating(best.index)[0].toLowerCase() + ')' + (best.why.length ? ', for ' + esc(best.why.join(', ')) : '') + '</p>' +
      '<div class="legend" style="margin-top:8px"><span><i class="hk h-top"></i>Excellent 75+</span><span><i class="hk h-good"></i>Good 60–74</span><span><i class="hk h-fair"></i>Fair 45–59</span><span><i class="hk h-slow"></i>Slow</span><span><i class="hk h-bad"></i>Thunderstorm</span></div>' +
      '<p class="small muted" style="margin:8px 0 0">' + (bt.own ? 'Includes your own record by time of day (' + bt.own.n + ' logged trips, 25% weight). ' : '') + 'Each hour is scored 0–100 from light, pressure trend, wind and moon position in this spot\'s forecast. Forecasts get less reliable further out, so treat days 4 to 7 as a rough guide. It is guidance, not a tested prediction' + (bt.own ? '.' : '; after 20 logged trips it starts using your own results.') + '</p>';
  }
  function bindWeek(box, bt) { box._bt = bt; if (box._bound) return; box._bound = true; box.addEventListener('click', e => { const b = e.target.closest('[data-day]'); if (!b || !box._bt) return; selDay = +b.dataset.day; box.innerHTML = timesHTML(box._bt); }); }
  async function drawWeatherBest() {
    const loc = Store.location($('#wxLoc').value); if (!loc || !isFinite(loc.lat)) return;
    $('#wxBestMethod').innerHTML = '<p>Each hour gets four scores between 0 and 1, combined with fixed weights into a 0–100 index:</p><ul><li><b>Light (' + A.WEIGHTS.light * 100 + '%)</b>: 1 within an hour of sunrise or sunset, fading to 0 by 2.5 hours away; low at midday and at night. Sun times are computed for the spot\'s coordinates.</li><li><b>Pressure trend (' + A.WEIGHTS.pressure * 100 + '%)</b>: 3-hour change in forecast surface pressure. Slowly falling scores highest, steady next, rising or crashing lowest.</li><li><b>Wind (' + A.WEIGHTS.wind * 100 + '%)</b>: 4–20 km/h scores highest; calm a little lower; above 35 km/h scores 0.</li><li><b>Moon position (' + A.WEIGHTS.solunar * 100 + '%)</b>: the "solunar" periods, an hour either side of the moon being overhead or underfoot and half an hour around moonrise and moonset.</li></ul><p>Heavy rain cuts the index by 40%. Any forecast thunderstorm sets it to 0.</p><p><b>How much to trust it.</b> These are widely used angling rules of thumb. Low light and weather changes have reasonable support; evidence for solunar periods is weak, which is why they carry the smallest weight. The index has not been validated against catch data. Once you have logged 20 trips, your own success rate by time of day (a Beta-Binomial estimate, shown in LureLab) is blended in at 25% weight, and LureLab\'s Validation tab tells you whether time of day actually predicts your catches.</p>';
    try {
      const bt = await CL.bestTimes(loc), col = i => { const v = bt.hours[i].index; return bt.hours[i].unsafe ? U.cssVar('--bad') : v >= 60 ? U.cssVar('--ok') : v >= 45 ? U.cssVar('--warn') : U.cssVar('--faint'); };
      U.chart('wxBest', $('#wxBest'), { type: 'bar', data: { labels: bt.hours.map(x => x.label.slice(5, 13).replace('T', ' ') + 'h'), datasets: [{ label: 'Fishing-time index (0–100)', data: bt.hours.map(x => x.index), backgroundColor: bt.hours.map((_, i) => col(i)) }] }, options: CL.baseChartOpts({ plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: c => { const x = bt.hours[c.dataIndex]; return (x.why.join(', ') || 'steady conditions') + '\nlight ' + x.c.light.toFixed(2) + ' · pressure ' + x.c.pressure.toFixed(2) + ' · wind ' + x.c.wind.toFixed(2) + ' · moon ' + x.c.solunar.toFixed(2); } } } }, scales: { y: { min: 0, max: 100 }, x: { ticks: { maxTicksLimit: 12 } } } }) });
      $('#wxBestList').innerHTML = timesHTML(bt); bindWeek($('#wxBestList'), bt);
    } catch (e) { $('#wxBestList').innerHTML = '<p class="small muted">Best times need the weather forecast, which could not be loaded.</p>'; }
  }
  const wxOrig = CL.modules.weather.render; CL.modules.weather.render = function () { wxOrig(); drawWeatherBest(); };
  $('#wxGo').addEventListener('click', drawWeatherBest); $('#wxLoc').addEventListener('change', drawWeatherBest);

  /* ================= fish reported nearby (iNaturalist) ================= */
  const FAM = { 1303618: 'minnows', 49609: 'minnows', 49236: 'shad', 58647: 'topminnows', 59116: 'topminnows', 85612: 'silversides', 85949: 'silversides', 51911: 'silversides', 47645: 'bottom', 47308: 'bottom' };
  const SPORT_FAM = [49588, 48158, 51369, 47520, 51393, 58655];
  const KEYMAP = [[/largemouth|florida bass/i, 'Largemouth bass'], [/smallmouth|spotted bass/i, 'Smallmouth bass'], [/striped bass|white bass|hybrid/i, 'Striped bass'], [/bluegill|sunfish|pumpkinseed|warmouth|rock bass/i, 'Bluegill / sunfish'], [/crappie/i, 'Crappie'], [/trout|char/i, 'Trout'], [/walleye|sauger/i, 'Walleye'], [/muskellunge|muskie/i, 'Muskellunge'], [/pike|pickerel/i, 'Northern pike'], [/catfish|bullhead/i, 'Channel catfish'], [/salmon|steelhead/i, 'Salmon / steelhead'], [/red drum|redfish/i, 'Redfish'], [/yellow perch/i, 'Yellow perch']];
  function classify(t) {
    const name = t.preferred_common_name || t.name, anc = t.ancestor_ids || [], o = { id: t.id, name, sci: t.name, photo: t.default_photo && t.default_photo.square_url, photoM: t.default_photo && t.default_photo.medium_url, url: 'https://www.inaturalist.org/taxa/' + t.id };
    o.key = (KEYMAP.find(k => k[0].test(name)) || [])[1] || null;
    if (/darter|sculpin|goby|madtom/i.test(name)) { o.role = 'forage'; o.group = 'bottom'; }
    else if (/carp|goldfish|koi|buffalo/i.test(name)) o.role = 'other';
    else { const g = anc.map(a => FAM[a]).find(Boolean); if (g) { o.role = 'forage'; o.group = g; } else if (o.key || anc.some(a => SPORT_FAM.includes(a))) o.role = 'sport'; else o.role = 'other'; }
    if (o.key === 'Bluegill / sunfish') o.alsoForage = 'sunfish'; if (o.key === 'Yellow perch') o.alsoForage = 'perch';
    return o;
  }
  const fishCache = {};
  CL.localFish = async function (lat, lon, km) {
    km = km || 20; const key = lat.toFixed(2) + ',' + lon.toFixed(2) + ',' + km; if (fishCache[key]) return fishCache[key];
    const q = 'taxon_id=47178&lat=' + lat.toFixed(4) + '&lng=' + lon.toFixed(4) + '&radius=' + km + '&quality_grade=research';
    const [sc, ob] = await Promise.all([fetch('https://api.inaturalist.org/v1/observations/species_counts?' + q + '&per_page=60').then(r => { if (!r.ok) throw new Error('iNaturalist HTTP ' + r.status); return r.json(); }), fetch('https://api.inaturalist.org/v1/observations?' + q + '&per_page=14&order_by=observed_on&photos=true').then(r => r.ok ? r.json() : { results: [] })]);
    const species = sc.results.map(r => Object.assign(classify(r.taxon), { count: r.count }));
    const recent = (ob.results || []).filter(o => o.taxon).map(o => ({ name: o.taxon.preferred_common_name || o.taxon.name, date: o.observed_on, user: o.user && o.user.login, place: o.place_guess, url: o.uri, photo: o.photos && o.photos[0] && o.photos[0].url, lat: o.geojson && o.geojson.coordinates[1], lon: o.geojson && o.geojson.coordinates[0], obscured: o.obscured }));
    return fishCache[key] = { species, recent, total: ob.total_results || 0, km, nSpecies: sc.total_results };
  };
  const INAT_NOTE = 'Source: <a href="https://www.inaturalist.org" target="_blank" rel="noopener">iNaturalist</a> research-grade records of ray-finned fishes. These are fish that people photographed and had identified: catches, sightings and fish found dead. Counts show how often a species is reported, not how many live there, and popular or easy-to-photograph species are over-represented. Some record positions are deliberately blurred.';
  function chips(list, max) { return list.slice(0, max || 12).map(f => '<a class="fishchip" href="' + f.url + '" target="_blank" rel="noopener" title="' + esc(f.sci) + '">' + (f.photo ? '<img src="' + esc(f.photo) + '" alt="" loading="lazy" width="34" height="34">' : '<span class="ph"></span>') + '<span><b>' + esc(f.name) + '</b><small>' + f.count + ' report' + (f.count === 1 ? '' : 's') + '</small></span></a>').join(''); }
  function recentHTML(d, n) { return d.recent.length ? '<div class="tablewrap" style="max-height:330px"><table><tr><th></th><th>Fish</th><th>Date</th><th>Reported by</th><th>Place</th></tr>' + d.recent.slice(0, n || 10).map(o => '<tr><td>' + (o.photo ? '<img src="' + esc(o.photo) + '" alt="" loading="lazy" width="40" height="40" style="border-radius:6px;object-fit:cover">' : '') + '</td><td><a href="' + esc(o.url) + '" target="_blank" rel="noopener">' + esc(o.name) + '</a></td><td class="mono small">' + esc(o.date || '') + '</td><td>' + esc(o.user || '') + '</td><td class="small muted">' + esc(o.place || '') + (o.obscured ? ' (position blurred)' : '') + '</td></tr>').join('') + '</table></div>' : '<p class="muted small">No recent reports with photos.</p>'; }
  function cards(list, max) { return '<div class="reel">' + list.slice(0, max || 12).map(f => '<a class="fishcard" href="' + f.url + '" target="_blank" rel="noopener" title="' + esc(f.sci) + '">' + (f.photoM || f.photo ? '<img src="' + esc(f.photoM || f.photo) + '" alt="' + esc(f.name) + '" loading="lazy">' : '<span class="ph"></span>') + '<span class="fc"><b>' + esc(f.name) + '</b><small>' + f.count + ' report' + (f.count === 1 ? '' : 's') + '</small></span></a>').join('') + '</div>'; }
  function fishHTML(d, withRecent, big) {
    const sp = d.species.filter(f => f.role === 'sport'), fo = d.species.filter(f => f.role === 'forage'), ot = d.species.filter(f => f.role === 'other');
    if (!d.species.length) return '<p class="muted small">No research-grade fish records within ' + d.km + ' km on iNaturalist. That means nobody has reported any here yet, not that there are no fish.</p>';
    return '<p class="small muted" style="margin:0 0 8px">' + d.nSpecies + ' fish species and ' + d.total + ' reports within ' + d.km + ' km.</p>' +
      (sp.length ? '<h4>Sport fish</h4>' + (big ? cards(sp) : '<div class="chips">' + chips(sp) + '</div>') : '') + (fo.length ? '<h4 style="margin-top:10px">Minnows and baitfish swimming around</h4><div class="chips">' + chips(fo) + '</div>' : '<h4 style="margin-top:10px">Minnows and baitfish</h4><p class="small muted">None reported nearby. Small fish are under-reported, so this is not proof they are absent.</p>') + (ot.length ? '<h4 style="margin-top:10px">Other fish</h4><div class="chips">' + chips(ot, 8) + '</div>' : '') +
      (withRecent ? '<h4 style="margin-top:12px">Latest reports</h4>' + recentHTML(d, 8) : '') + '<p class="small muted" style="margin-top:8px">' + INAT_NOTE + '</p>';
  }

  /* ================= bait and combo suggestions ================= */
  const BASICS = { 'Bluegill / sunfish': 'Small jigs (1/64–1/32 oz), worms or waxworms under a float, tiny spinners.', 'Crappie': 'Small tube or marabou jigs, live minnows under a float.', 'Yellow perch': 'Small jigs tipped with worm or minnow, small spoons.', 'Trout': 'Inline spinners, small spoons, flies; worms or dough bait where legal.', 'Walleye': 'Jig and minnow, crawler harness, minnow-profile crankbaits, best in low light.', 'Smallmouth bass': 'Tubes, ned rigs, jerkbaits, small swimbaits.', 'Largemouth bass': 'Soft plastic worms, spinnerbaits, squarebill crankbaits, jigs; topwater at dawn and dusk.', 'Northern pike': 'Spoons, inline spinners, large swimbaits and jerkbaits, with a bite leader.', 'Channel catfish': 'Cut bait, nightcrawlers or prepared bait on the bottom.', 'Striped bass': 'Swimbaits, topwater plugs, live or cut shad or herring.', 'Salmon / steelhead': 'Spoons, spinners and plugs; roe or beads where legal.', 'Redfish': 'Gold spoons, soft plastic paddletails, shrimp or cut bait.', 'Muskellunge': 'Large bucktails, glide baits and big rubber baits on a heavy leader.', 'Other': 'Match the size and colour of the local baitfish.' };
  const PATTERNS = {
    shad: ['Shad and herring', 'Silver or white swimbaits, lipless crankbaits, blade baits and spoons.', ['shad', 'silver', 'white', 'spoon', 'lipless', 'swimbait', 'chrome', 'blade']],
    minnows: ['Minnows and shiners', 'Slim minnow shapes: jerkbaits, inline spinners and small swimbaits in silver or gold.', ['minnow', 'jerk', 'spinner', 'shiner', 'silver', 'gold']],
    silversides: ['Silversides and smelt', 'Slender silver or see-through soft jerkbaits and small spoons.', ['fluke', 'jerk', 'silver', 'spoon', 'smelt', 'clear']],
    topminnows: ['Topminnows', 'Small topwater baits and unweighted soft plastics fished near the surface.', ['popper', 'topwater', 'fly', 'frog']],
    bottom: ['Sculpins, darters and gobies', 'Bottom baits: tube jigs, ned rigs and small jigs in brown or olive.', ['tube', 'jig', 'ned', 'brown', 'olive', 'goby', 'craw']],
    sunfish: ['Young sunfish', 'Bluegill-pattern swimbaits, squarebill crankbaits and swim jigs in green pumpkin.', ['bluegill', 'squarebill', 'crank', 'green pumpkin', 'chatter', 'swim jig']],
    perch: ['Young perch', 'Perch or firetiger crankbaits and jerkbaits.', ['perch', 'firetiger', 'crank']]
  };
  const PRED = { 'Walleye': 1, 'Smallmouth bass': 1, 'Largemouth bass': 1, 'Northern pike': 1, 'Striped bass': 1, 'Salmon / steelhead': 1, 'Muskellunge': 1, 'Trout': 1, 'Redfish': 1, 'Crappie': ['minnows', 'silversides', 'shad'], 'Yellow perch': ['minnows'] };
  CL.baitAdvice = function (target, fish) {
    const inv = Store.list('inventory').filter(i => !/line|terminal/i.test(i.category)), S0 = CL.SPECIES[target] || CL.SPECIES.Other, out = { target, basics: BASICS[target] || BASICS.Other, own: [], patterns: [] };
    const ses = Store.list('sessions').filter(s => s.source !== 'simulated' && s.species === target), by = {};
    ses.forEach(s => { const b = by[s.lure] = by[s.lure] || { n: 0, k: 0 }; b.n++; if (s.catches > 0) b.k++; });
    out.own = Object.entries(by).filter(e => e[1].n >= 3).map(([lure, b]) => Object.assign({ lure }, b, { bp: S.betaPosterior(b.k, b.n) })).sort((a, b) => b.bp.mean - a.bp.mean).slice(0, 3); out.ownN = ses.length;
    if (fish && PRED[target]) {
      const groups = {}; fish.species.forEach(f => { const g = f.group || f.alsoForage; if (g && (PRED[target] === 1 || PRED[target].includes(g))) (groups[g] = groups[g] || []).push(f); });
      out.patterns = Object.entries(groups).map(([g, fs]) => ({ g, label: PATTERNS[g][0], text: PATTERNS[g][1], reports: fs.reduce((a, f) => a + f.count, 0), present: fs.slice(0, 3).map(f => f.name), owned: inv.filter(i => PATTERNS[g][2].some(k => (i.name + ' ' + (i.color || '')).toLowerCase().includes(k))) })).sort((a, b) => b.reports - a.reports);
    }
    const fits = i => i.weightG && U.gToOz(i.weightG) >= S0.lure[0] * 0.8 && U.gToOz(i.weightG) <= S0.lure[1] * 1.2;
    out.sizeOwned = inv.filter(fits); const lb = v => CL.isMetric() ? fmt(U.lbToN(v), 0) + ' N' : fmt(v, v < 3 && v % 1 ? 1 : 0) + ' lb';
    const pIdx = [CL.POWERS.indexOf(S0.power[0]), CL.POWERS.indexOf(S0.power[1])], rod = Store.list('rods').find(r => { const p = CL.POWERS.indexOf(r.power); return p >= pIdx[0] && p <= pIdx[1] && r.lmax >= S0.line[0] && r.lmin <= S0.line[1]; });
    const lurePick = (out.own[0] && out.own[0].lure) || (out.patterns[0] && out.patterns[0].owned[0] && out.patterns[0].owned[0].name) || (out.sizeOwned[0] && out.sizeOwned[0].name);
    out.combo = { rod: S0.power.join('–') + ' power' + (rod ? ' (your ' + rod.name + ' fits)' : ''), line: lb(S0.line[0]) + '–' + lb(S0.line[1]) + (S0.braid ? ' mono, fluorocarbon or braid in cover' : ' mono or fluorocarbon') + (S0.leader ? ', plus ' + S0.leader : ''), lure: lurePick || (out.patterns[0] ? out.patterns[0].text : out.basics), drag: lb(0.25 * S0.line[1]) + '–' + lb(0.33 * S0.line[1]), fromOwn: !!out.own[0] };
    return out;
  };
  function adviceHTML(a, fish, compact) {
    const own = a.own.length ? '<h4>What has worked for you</h4>' + a.own.map(o => '<div class="stat-line"><span>' + esc(o.lure) + '</span><b>' + o.k + ' of ' + o.n + ' trips with a fish · ' + pct(o.bp.mean) + ' (95% CrI ' + pct(o.bp.lo) + '–' + pct(o.bp.hi) + ')</b></div>').join('') : '<p class="small muted">' + (a.ownN ? 'You have ' + a.ownN + ' logged trips for ' + esc(a.target) + ', but no lure with 3 or more yet.' : 'No logged trips for ' + esc(a.target) + ' yet, so there is nothing from your own results to rank.') + '</p>';
    const pat = a.patterns.length ? '<h4 style="margin-top:10px">Match the local baitfish</h4>' + a.patterns.map(p => '<div class="rec"><b>' + esc(p.label) + '</b> <span class="small muted">reported here: ' + esc(p.present.join(', ')) + ' (' + p.reports + ' reports)</span><br>' + esc(p.text) + (p.owned.length ? '<br><span class="small">In your BaitRadar: ' + p.owned.map(i => '<span class="pill">' + esc(i.name) + ' · qty ' + i.qty + '</span>').join('') + '</span>' : '<br><span class="small muted">Nothing like this in your BaitRadar inventory yet.</span>') + '</div>').join('') : (fish && PRED[a.target] ? '<p class="small muted">No baitfish reported nearby to match.</p>' : '');
    const combo = '<h4 style="margin-top:10px">Combo to start with</h4><div class="stat-line"><span>Rod</span><b>' + esc(a.combo.rod) + '</b></div><div class="stat-line"><span>Line</span><b>' + esc(a.combo.line) + '</b></div><div class="stat-line"><span>Lure or bait</span><b style="text-align:right;max-width:65%">' + esc(a.combo.lure) + (a.combo.fromOwn ? ' (your best so far)' : '') + '</b></div><div class="stat-line"><span>Drag</span><b>' + esc(a.combo.drag) + '</b></div>';
    return '<p style="margin:0 0 8px"><b>' + esc(a.target) + ':</b> ' + esc(a.basics) + '</p>' + (compact ? pat + combo : '<div class="grid g2"><div>' + own + pat + '</div><div>' + combo + '</div></div>') + '<p class="small muted" style="margin-top:8px">Bait and gear ranges are common angling guidance. The baitfish come from public reports near this spot. Only the "worked for you" lines are measured results. Check local rules on live bait, seasons and size limits.</p>';
  }
  function targetOptions(sel, fish) {
    const keep = sel.value, here = fish ? Array.from(new Set(fish.species.filter(f => f.key).map(f => f.key))) : [], rest = Object.keys(CL.SPECIES).filter(k => !here.includes(k));
    sel.innerHTML = (here.length ? '<optgroup label="Reported at this spot">' + here.map(k => '<option>' + esc(k) + '</option>').join('') + '</optgroup><optgroup label="Other species">' : '') + rest.map(k => '<option>' + esc(k) + '</option>').join('') + (here.length ? '</optgroup>' : '');
    if (keep && Array.from(sel.options).some(o => o.value === keep)) sel.value = keep;
  }
  async function fishFor(id) { const l = Store.location(id); if (!l || !isFinite(l.lat)) return null; try { return await CL.localFish(l.lat, l.lon); } catch (e) { return null; } }

  // BaitRadar panel
  CL.baitRec = async function () {
    const sel = $('#brLoc'), keep = sel.value || spot(); CL.locationOptions(sel, { blank: '— no spot —' }); if (keep && Store.location(keep)) sel.value = keep;
    $('#brRecOut').innerHTML = '<p class="muted small">Loading…</p>'; const fish = sel.value ? await fishFor(sel.value) : null; targetOptions($('#brTarget'), fish);
    $('#brRecOut').innerHTML = adviceHTML(CL.baitAdvice($('#brTarget').value, fish), fish) + (sel.value && !fish ? '<p class="small status-warn">Could not load local fish reports, so baitfish matching is skipped.</p>' : '');
  };
  $('#brLoc').addEventListener('change', () => CL.baitRec()); $('#brTarget').addEventListener('change', async () => { const fish = $('#brLoc').value ? await fishFor($('#brLoc').value) : null; $('#brRecOut').innerHTML = adviceHTML(CL.baitAdvice($('#brTarget').value, fish), fish); });

  // GearMatch: species reported at a spot, plus local bait suggestions under the recommendation
  let gmFish = null;
  CL.gearLocal = async function () {
    const sel = $('#gmLoc'), keep = sel.value || spot(); CL.locationOptions(sel, { blank: '— any water —' }); if (keep && Store.location(keep)) sel.value = keep;
    gmFish = sel.value ? await fishFor(sel.value) : null; const keys = gmFish ? Array.from(new Set(gmFish.species.filter(f => f.key).map(f => f.key))) : [];
    $('#gmLocal').innerHTML = !sel.value ? '' : keys.length ? 'Reported here (tap to choose): ' + keys.map(k => '<button class="btn sm" type="button" data-gm-sp="' + esc(k) + '">' + esc(k) + '</button>').join(' ') : '<span class="muted">No sport fish reports loaded for this spot.</span>';
    CL.gearLocalRec($('#gmSpecies').value);
  };
  CL.gearLocalRec = function (sp) { const box = $('#gmLocalRec'); if (!box) return; box.innerHTML = gmFish ? '<h3 style="margin-top:12px">For this spot</h3>' + adviceHTML(CL.baitAdvice(sp, gmFish), gmFish, true) : ''; };
  $('#gmLoc').addEventListener('change', () => CL.gearLocal());
  $('#gmLocal').addEventListener('click', e => { const b = e.target.closest('[data-gm-sp]'); if (b) { $('#gmSpecies').value = b.dataset.gmSp; CL.recommendGear(); } });

  /* ================= water-body lookup (Nominatim) ================= */
  async function reverseWater(lat, lon) {
    const base = 'https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=' + lat.toFixed(6) + '&lon=' + lon.toFixed(6);
    let j = await fetch(base + '&layer=natural&zoom=14').then(r => r.json()).catch(() => null);
    if (j && j.name) return { name: j.name, type: typeFromOSM((j.type || '') + ' ' + (j.category || '')), water: /water|lake|river|reservoir|pond|bay|stream|wetland/.test((j.category || '') + (j.type || '')) };
    j = await fetch(base + '&zoom=12').then(r => r.json()).catch(() => null);
    return j && j.display_name ? { name: 'Near ' + j.display_name.split(',').slice(0, 2).join(','), type: 'Other', water: false } : null;
  }
  async function searchWater(q) {
    const r = await fetch('https://nominatim.openstreetmap.org/search?format=jsonv2&limit=10&q=' + encodeURIComponent(q)); if (!r.ok) throw new Error('search HTTP ' + r.status);
    const all = await r.json(), wet = all.filter(x => /water|waterway|natural/.test(x.category) || /lake|river|reservoir|pond|bay|stream/.test(x.type));
    return (wet.length ? wet : all).slice(0, 6).map(x => ({ name: x.name || x.display_name.split(',')[0], full: x.display_name, lat: +x.lat, lon: +x.lon, type: typeFromOSM(x.type + ' ' + x.category) }));
  }
  function searchUI(input, out, onPick) {
    const go = async () => { const q = $(input).value.trim(); if (!q) return; $(out).innerHTML = '<p class="small muted">Searching OpenStreetMap…</p>'; try { const res = await searchWater(q); $(out)._res = res; $(out).innerHTML = res.length ? res.map((x, i) => '<div class="stat-line"><span><b style="font-family:var(--sans)">' + esc(x.name) + '</b><br><span class="small muted">' + esc(x.full) + '</span></span><button class="btn sm primary" type="button" data-pick="' + i + '">Choose</button></div>').join('') : '<p class="small muted">Nothing found. Try adding the state or country.</p>'; } catch (e) { $(out).innerHTML = '<p class="small status-fail">Search failed: ' + esc(e.message) + '</p>'; } };
    $(input + 'Go').onclick = go; $(input).addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); go(); } });
    $(out).onclick = e => { const b = e.target.closest('[data-pick]'); if (b) { onPick($(out)._res[+b.dataset.pick]); $(out).innerHTML = ''; $(input).value = ''; } };
  }

  /* ================= map: click a water body, search, community layer ================= */
  let pending = null, inatLayer = null, inatTimer = null;
  CL.mapClick = async function (e, map) {
    const lat = e.latlng.lat, lon = e.latlng.lng, pop = L.popup({ maxWidth: 280 }).setLatLng(e.latlng).setContent('Looking up this spot…').openOn(map);
    const w = await reverseWater(lat, lon); pending = { name: w ? w.name : 'Spot ' + lat.toFixed(4) + ', ' + lon.toFixed(4), lat, lon, type: w ? w.type : 'Other' };
    $('#locName').value = pending.name; if (w) $('#locType').value = w.type;
    pop.setContent('<b>' + esc(pending.name) + '</b><br><span style="font-size:12px">' + (w && w.water ? esc(w.type) : 'No named water body found exactly here') + '</span><div style="display:flex;flex-direction:column;gap:5px;margin-top:8px"><button class="btn sm primary" data-spot="home">Best times and fish here</button><button class="btn sm" data-spot="castcheck">Shoreline twin (CastCheck)</button><button class="btn sm" data-spot="save">Just save this spot</button></div>');
  };
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-spot]'); if (!b || !pending) return; const l = CL.ensureLocation(pending.name, pending.lat, pending.lon, pending.type); setSpot(l.id); CL._map && CL._map.closePopup();
    if (b.dataset.spot === 'save') { U.toast('Saved "' + l.name + '".'); CL.modules.map.render(); } else { CL.pickLoc = l.id; location.hash = b.dataset.spot; }
  });
  searchUI('#mapSearch', '#mapSearchOut', x => { if (CL._map) { CL._map.setView([x.lat, x.lon], 13); CL.mapClick({ latlng: L.latLng(x.lat, x.lon) }, CL._map); } });
  async function drawInat(map) {
    if (!$('#mapInat').checked) { if (inatLayer) inatLayer.clearLayers(); return; }
    if (!inatLayer) inatLayer = L.layerGroup().addTo(map); const c = map.getCenter(), km = Math.max(2, Math.min(30, Math.round(map.getBounds().getNorthEast().distanceTo(c) / 1000)));
    try { const d = await CL.localFish(c.lat, c.lng, km); inatLayer.clearLayers(); d.recent.filter(o => isFinite(o.lat)).forEach(o => L.circleMarker([o.lat, o.lon], { radius: 6, color: '#b45309', weight: 2, fillColor: '#f59e0b', fillOpacity: .85 }).bindPopup((o.photo ? '<img src="' + esc(o.photo) + '" width="75" height="75" style="float:left;margin-right:8px;border-radius:6px;object-fit:cover" alt="">' : '') + '<b>' + esc(o.name) + '</b><br>' + esc(o.date || '') + '<br>by ' + esc(o.user || '') + (o.obscured ? '<br><i>position blurred</i>' : '') + '<br><a href="' + esc(o.url) + '" target="_blank" rel="noopener">iNaturalist record</a>').addTo(inatLayer)); U.toast(d.recent.length ? 'Showing the ' + d.recent.length + ' latest fish reports within ' + km + ' km (of ' + d.total + ').' : 'No fish reports within ' + km + ' km.'); }
    catch (err) { U.toast('Could not load iNaturalist reports: ' + err.message); }
  }
  $('#mapInat').addEventListener('change', () => CL._map && drawInat(CL._map));
  CL.mapMoved = map => { if (!$('#mapInat').checked) return; clearTimeout(inatTimer); inatTimer = setTimeout(() => drawInat(map), 900); };


  /* ---- Today visuals ---- */
  const hr12 = h => ((h + 11) % 12 + 1) + (h < 12 ? ' am' : ' pm');
  function smooth(pts) { let d = 'M' + pts[0][0].toFixed(1) + ' ' + pts[0][1].toFixed(1); for (let i = 0; i < pts.length - 1; i++) { const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)], k = 1 / 6; d += ' C' + (p1[0] + (p2[0] - p0[0]) * k).toFixed(1) + ' ' + (p1[1] + (p2[1] - p0[1]) * k).toFixed(1) + ' ' + (p2[0] - (p3[0] - p1[0]) * k).toFixed(1) + ' ' + (p2[1] - (p3[1] - p1[1]) * k).toFixed(1) + ' ' + p2[0].toFixed(1) + ' ' + p2[1].toFixed(1); } return d; }
  let lastBt = null;
  function heroCurve(bt) {
    const box = $('#hmCurve'); lastBt = bt; if (!bt) { box.innerHTML = ''; return; }
    const W = Math.max(280, box.clientWidth || 800), H = W < 520 ? 150 : 180, top = 34, base = H - 26, n = bt.hours.length, t0 = bt.hours[0].ms, t1 = bt.hours[n - 1].ms, X = ms => (ms - t0) / (t1 - t0) * W, Y = v => base - v / 100 * (base - top), o = bt.off;
    const pts = bt.hours.map(h => [X(h.ms), Y(h.index)]), line = smooth(pts), dayOf = ms => Math.floor((ms + o * 1000) / 864e5), d0 = dayOf(bt.now);
    let nights = ''; const sets = bt.ev.sunset.concat([t0 - 864e5]).sort((a, b) => a - b); sets.forEach(s => { const rise = bt.ev.sunrise.find(r => r > s) || t1, a = Math.max(t0, s), b = Math.min(t1, rise); if (b > a) nights += '<rect x="' + X(a).toFixed(1) + '" y="0" width="' + (X(b) - X(a)).toFixed(1) + '" height="' + base + '" fill="#000" opacity=".2"/>'; });
    let ticks = ''; bt.hours.forEach(h => { if (h.localHour % (W < 520 ? 12 : 6) === 0 && X(h.ms) > 24 && X(h.ms) < W - 24) ticks += '<text x="' + X(h.ms).toFixed(1) + '" y="' + (H - 7) + '" text-anchor="middle" class="tk">' + (h.localHour === 0 ? (dayOf(h.ms) === d0 + 1 ? 'Tomorrow' : 'Midnight') : hr12(h.localHour)) + '</text>' + (h.localHour === 0 ? '<line x1="' + X(h.ms).toFixed(1) + '" x2="' + X(h.ms).toFixed(1) + '" y1="6" y2="' + base + '" class="mid"/>' : ''); });
    const best = k => { const hs = bt.hours.filter(x => dayOf(x.ms) === d0 + k && !x.unsafe); return hs.length ? hs.reduce((a, x) => x.index > a.index ? x : a, hs[0]) : null; };
    let marks = ''; [best(0), best(1)].forEach(b => { if (!b || b.index < 45) return; const x = X(b.ms), y = Y(b.index), anchor = x < 60 ? 'start' : x > W - 60 ? 'end' : 'middle'; marks += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="6.5" class="pk"/><text x="' + x.toFixed(1) + '" y="' + (y - 13).toFixed(1) + '" text-anchor="' + anchor + '" class="pl">' + hr12(b.localHour) + ', ' + b.index + '</text>'; });
    const hit = bt.hours.map(h => '<rect x="' + (X(h.ms) - W / n / 2).toFixed(1) + '" y="0" width="' + (W / n).toFixed(1) + '" height="' + base + '" fill="transparent"><title>' + (dayOf(h.ms) === d0 ? 'Today ' : 'Tomorrow ') + hr12(h.localHour) + ': ' + h.index + '/100' + (h.why.length ? ' (' + esc(h.why.join(', ')) + ')' : '') + '</title></rect>').join('');
    box.innerHTML = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '"><defs><linearGradient id="hcg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5db0e6" stop-opacity=".6"/><stop offset="1" stop-color="#5db0e6" stop-opacity="0"/></linearGradient></defs>' + nights + '<line x1="0" x2="' + W + '" y1="' + Y(60).toFixed(1) + '" y2="' + Y(60).toFixed(1) + '" class="ref"/><text x="' + (W - 8) + '" y="' + (Y(60) - 5).toFixed(1) + '" text-anchor="end" class="tk">good, 60</text><path d="' + line + ' L' + W + ' ' + base + ' L0 ' + base + 'Z" fill="url(#hcg)"/><path d="' + line + '" class="ln" pathLength="1"/><line x1="0" x2="' + W + '" y1="' + base + '" y2="' + base + '" class="ax"/>' + ticks + marks + hit + '</svg>';
  }
  window.addEventListener('resize', () => { clearTimeout(heroCurve._t); heroCurve._t = setTimeout(() => { if (lastBt && CL.current === 'home') heroCurve(lastBt); }, 200); });
  function gauge(v, cls) { const r = 38, c = 2 * Math.PI * r; return '<svg class="gauge" width="104" height="104" viewBox="0 0 104 104" role="img" aria-label="Fishing score ' + v + ' out of 100"><circle cx="52" cy="52" r="' + r + '" class="gt"/><circle cx="52" cy="52" r="' + r + '" class="gv ' + cls + '" stroke-dasharray="' + (c * v / 100).toFixed(1) + ' ' + c.toFixed(1) + '" transform="rotate(-90 52 52)"/><text x="52" y="59" text-anchor="middle" class="gn">' + v + '</text></svg>'; }
  function wxIcon(code, night) {
    const cloud = '<path d="M14 34h20a7.5 7.500 0 0 0 1-14.900A10.500 10.500 0 0 0 14.500 17 8.500 8.500 0 0 0 14 34z" class="wc"/>', sun = night ? '<path d="M30 6a11 11 0 1 0 10 15A9 9 0 0 1 30 6z" class="wm"/>' : '<circle cx="35" cy="12" r="7.5" class="ws"/>';
    let g; if (code >= 95) g = cloud + '<path d="M24 36l-4 7h5l-3 6" class="wb"/>'; else if (code >= 71 && code <= 77 || code === 85 || code === 86) g = cloud + '<circle cx="18" cy="40" r="1.600" class="wd"/><circle cx="25" cy="43" r="1.600" class="wd"/><circle cx="32" cy="40" r="1.600" class="wd"/>'; else if (code >= 51) g = cloud + '<path d="M18 38l-2 5M25 38l-2 5M32 38l-2 5" class="wr"/>'; else if (code === 3 || code === 45 || code === 48) g = cloud; else if (code >= 1) g = sun + cloud; else g = night ? '<path d="M27 8a15 15 0 1 0 13 21A12 12 0 0 1 27 8z" class="wm"/>' : '<circle cx="24" cy="24" r="9" class="ws"/><path d="M24 6v5M24 37v5M6 24h5M37 24h5M11.300 11.300l3.500 3.500M33.200 33.200l3.500 3.500M11.300 36.700l3.500-3.500M33.200 14.800l3.500-3.500" class="wy"/>';
    return '<svg class="wxi" width="56" height="56" viewBox="0 0 48 48" aria-hidden="true">' + g + '</svg>';
  }

  /* ---- share image, calendar, spot ranking ---- */
  function pickBest(bt) { const d = daysOf(bt), t0 = d[0] && d[0].best, t1 = d[1] && d[1].best; if (t0 && t1 && !t1.unsafe && (t0.unsafe || (t0.index < 60 && t1.index > t0.index))) return { day: d[1], h: t1 }; return t0 ? { day: d[0], h: t0 } : null; }
  $('#hmShare').onclick = async () => {
    const bt = lastBt; if (!bt) return U.toast('Pick a spot and wait for the forecast first.');
    try { await Promise.all([document.fonts.load('700 84px "Bricolage Grotesque"'), document.fonts.load('500 28px "Instrument Sans"')]); } catch (e) {}
    const W = 1200, H = 630, cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const c = cv.getContext('2d'), disp = '"Bricolage Grotesque", sans-serif', sans = '"Instrument Sans", sans-serif';
    c.fillStyle = '#0f2a3d'; c.fillRect(0, 0, W, H); c.strokeStyle = 'rgba(255,255,255,.07)'; c.lineWidth = 2;
    for (let k = 1; k <= 9; k++) { c.beginPath(); for (let i = 0; i <= 90; i++) { const a = i / 90 * Math.PI * 2, rr = k * 62 * (1 + 0.13 * Math.sin(2 * a + k) + 0.07 * Math.sin(3 * a + 1.7 * k)), x = 880 + rr * Math.cos(a) * 1.25, y = 150 + rr * Math.sin(a) * 0.85; if (i) c.lineTo(x, y); else c.moveTo(x, y); } c.closePath(); c.stroke(); }
    c.fillStyle = '#f2f7fa'; c.font = '700 34px ' + disp; c.fillText('CastLab', 60, 78);
    c.fillStyle = 'rgba(242,247,250,.7)'; c.font = '600 28px ' + sans; c.fillText(bt.loc.name, 60, 128);
    c.fillStyle = '#fff'; c.font = '700 82px ' + disp; c.fillText($('#hmHeadline').textContent, 60, 232);
    c.fillStyle = 'rgba(242,247,250,.75)'; c.font = '500 27px ' + sans;
    let line = '', y = 282; $('#hmSub').textContent.split(' ').forEach(wd => { if (c.measureText(line + wd).width > 1060) { c.fillText(line, 60, y); y += 36; line = ''; } line += wd + ' '; }); c.fillText(line, 60, y);
    const x0 = 60, x1 = W - 60, top = 390, base = 548, n = bt.hours.length, X = i => x0 + i / (n - 1) * (x1 - x0), Y = v => base - v / 100 * (base - top);
    const g = c.createLinearGradient(0, top, 0, base); g.addColorStop(0, 'rgba(93,176,230,.6)'); g.addColorStop(1, 'rgba(93,176,230,0)');
    c.beginPath(); c.moveTo(X(0), base); bt.hours.forEach((hh, i) => c.lineTo(X(i), Y(hh.index))); c.lineTo(X(n - 1), base); c.closePath(); c.fillStyle = g; c.fill();
    c.beginPath(); bt.hours.forEach((hh, i) => { if (i) c.lineTo(X(i), Y(hh.index)); else c.moveTo(X(i), Y(hh.index)); }); c.strokeStyle = '#9ad4fb'; c.lineWidth = 4; c.lineJoin = 'round'; c.stroke();
    c.strokeStyle = 'rgba(255,255,255,.25)'; c.lineWidth = 1; c.beginPath(); c.moveTo(x0, base); c.lineTo(x1, base); c.stroke();
    c.font = '500 18px ' + sans; c.fillStyle = 'rgba(242,247,250,.6)'; c.textAlign = 'center';
    bt.hours.forEach((hh, i) => { if (hh.localHour % 6 === 0 && i > 1 && i < n - 2) c.fillText(hh.localHour === 0 ? 'midnight' : hr12(hh.localHour), X(i), base + 26); });
    const pk = pickBest(bt);
    if (pk) { const i = bt.hours.indexOf(pk.h); if (i >= 0) { c.beginPath(); c.arc(X(i), Y(pk.h.index), 10, 0, 7); c.fillStyle = '#e4572e'; c.fill(); c.lineWidth = 4; c.strokeStyle = '#fff'; c.stroke(); c.fillStyle = '#fff'; c.font = '700 22px ' + disp; c.fillText(hr12(pk.h.localHour) + ', ' + pk.h.index, Math.min(x1 - 60, Math.max(x0 + 60, X(i))), Y(pk.h.index) - 20); } }
    c.textAlign = 'left'; c.font = '500 17px ' + sans; c.fillStyle = 'rgba(242,247,250,.55)'; c.fillText('Fishing score, next 48 hours. A rule of thumb from light, pressure, wind and moon; not a guarantee.', 60, H - 22);
    U.downloadDataURL('castlab-' + bt.loc.name.replace(/[^a-z0-9]+/gi, '-').toLowerCase() + '.png', cv.toDataURL('image/png')); U.toast('Forecast image saved.');
  };
  $('#hmCal').onclick = () => {
    const bt = lastBt, pk = bt && pickBest(bt); if (!pk || pk.h.unsafe) return U.toast('No good hour to add yet. Pick a spot and wait for the forecast.');
    const z = ms => new Date(ms).toISOString().replace(/[-:]/g, '').slice(0, 15) + 'Z', clean = s => s.replace(/[\\,;]/g, ' ').replace(/\s+/g, ' ').trim(), why = pk.h.why.join(' / ') || 'steady conditions';
    const ics = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//CastLab//Fishing//EN', 'BEGIN:VEVENT', 'UID:' + pk.h.ms + '-' + bt.loc.id + '@castlab', 'DTSTAMP:' + z(Date.now()), 'DTSTART:' + z(pk.h.ms - HOUR), 'DTEND:' + z(pk.h.ms + 2 * HOUR), 'SUMMARY:Fishing at ' + clean(bt.loc.name), 'DESCRIPTION:CastLab rates ' + hr12(pk.h.localHour) + ' at ' + pk.h.index + '/100 (' + clean(why) + '). Rule-of-thumb score. Check the forecast before you go.', 'GEO:' + bt.loc.lat.toFixed(5) + ';' + bt.loc.lon.toFixed(5), 'END:VEVENT', 'END:VCALENDAR'].join('\r\n');
    U.download('castlab-fishing.ics', ics, 'text/calendar'); U.toast('Calendar event saved for ' + pk.day.name.toLowerCase() + ' around ' + hr12(pk.h.localHour) + '.');
  };
  $('#hmCompareGo').onclick = async () => {
    const spots = Store.list('locations').filter(l => !l.simulated && isFinite(l.lat)).slice(0, 6), rows = [], box = $('#hmCompare');
    for (const l of spots) { box.innerHTML = '<p class="small muted" style="margin:0">Scoring ' + esc(l.name) + ' (' + (rows.length + 1) + ' of ' + spots.length + ')…</p>'; try { const d = daysOf(await CL.bestTimes(l)); rows.push({ l, t: d[0] && d[0].best, m: d[1] && d[1].best }); } catch (e) { rows.push({ l, err: e.message }); } }
    rows.sort((a, b) => (b.t ? b.t.index : -1) - (a.t ? a.t.index : -1));
    const cell = x => x ? '<b class="pillscore ' + hcls(x) + '">' + x.index + '</b> at ' + hr12(x.localHour) : '—';
    box.innerHTML = '<div class="tablewrap" style="max-height:none"><table><tr><th>Spot</th><th>Best today</th><th>Best tomorrow</th><th></th></tr>' + rows.map((x, i) => '<tr><td>' + (i === 0 && x.t ? '<span class="badge ok">best today</span> ' : '') + '<b>' + esc(x.l.name) + '</b></td><td>' + (x.err ? '<span class="small status-fail">' + esc(x.err) + '</span>' : cell(x.t)) + '</td><td>' + cell(x.m) + '</td><td><button class="btn sm" type="button" data-go-spot="' + x.l.id + '">Open</button></td></tr>').join('') + '</table></div>' + (Store.list('locations').filter(l => !l.simulated && isFinite(l.lat)).length > 6 ? '<p class="small muted">Showing your first 6 spots.</p>' : '');
  };
  $('#hmCompare').addEventListener('click', e => { const b = e.target.closest('[data-go-spot]'); if (b) { CL.pickLoc = b.dataset.goSpot; CL.modules.home.render(); window.scrollTo({ top: 0, behavior: 'smooth' }); } });

  /* ================= Today page ================= */
  let homeFish = null, homeSeq = 0;
  CL.modules.home = { render() {
    const sel = $('#hmLoc'), want = CL.pickLoc || spot(); CL.pickLoc = null; CL.locationOptions(sel, { blank: Store.list('locations').some(l => !l.simulated) ? null : 'No spots yet — search below or use your location' });
    if (want && Store.location(want) && !Store.location(want).simulated) sel.value = want; if (sel.value) setSpot(sel.value); $('#hmCompareCard').hidden = Store.list('locations').filter(l => !l.simulated && isFinite(l.lat)).length < 2; refreshHome();
  } };
  async function refreshHome() {
    const loc = Store.location($('#hmLoc').value), seq = ++homeSeq; homeFish = null;
    $('#hmSpotName').textContent = loc ? loc.name : 'Today';
    if (!loc || !isFinite(loc.lat)) { heroCurve(null); $('#hmHeadline').textContent = 'Where are you fishing?'; $('#hmSub').textContent = 'Pick a spot and CastLab rates every hour of today and tomorrow, shows which fish people have reported there, and suggests what to tie on.'; $('#hmTimes').innerHTML = '<p class="muted small">Pick a spot first: search for a lake or river, tap "Use my location", or choose one on the map.</p>'; $('#hmNow').innerHTML = $('#hmFish').innerHTML = ''; targetOptions($('#hmTarget'), null); $('#hmSetup').innerHTML = adviceHTML(CL.baitAdvice($('#hmTarget').value, null), null); return; }
    $('#hmTwin').onclick = () => { CL.pickLoc = loc.id; };
    heroCurve(null); $('#hmHeadline').textContent = 'Checking the next 48 hours…'; $('#hmSub').textContent = 'Reading the forecast, sun and moon for ' + loc.name + '.';
    $('#hmTimes').innerHTML = $('#hmNow').innerHTML = $('#hmFish').innerHTML = '<p class="muted small">Loading…</p>';
    CL.bestTimes(loc).then(bt => { if (seq !== homeSeq) return; const w = bt.w, cur = bt.hours[0], r = rating(cur.index);
      $('#hmTimes').innerHTML = timesHTML(bt); bindWeek($('#hmTimes'), bt); heroCurve(bt);
      { const dayOf = ms => Math.floor((ms + bt.off * 1000) / 864e5), d0 = dayOf(bt.now), hr = h => ((h + 11) % 12 + 1) + (h < 12 ? ' am' : ' pm'), best = k => { const hs = bt.hours.filter(x => dayOf(x.ms) === d0 + k); return hs.length ? hs.reduce((a, x) => x.index > a.index ? x : a, hs[0]) : null; }, t = best(0), tm = best(1);
        if (t && t.unsafe && (!tm || tm.unsafe)) { $('#hmHeadline').textContent = 'Stay off the water today'; $('#hmSub').textContent = 'Thunderstorms are in the forecast.'; }
        else if (t && tm && !tm.unsafe && (t.unsafe || (t.index < 60 && tm.index > t.index))) { $('#hmHeadline').textContent = 'Best hour tomorrow: ' + hr(tm.localHour); $('#hmSub').textContent = 'Rated ' + tm.index + ' out of 100' + (tm.why.length ? ' for ' + tm.why.slice(0, 3).join(', ') : '') + '. The rest of today tops out at ' + t.index + ' (' + hr(t.localHour) + ').'; }
        else if (t) { $('#hmHeadline').textContent = 'Best hour today: ' + hr(t.localHour); $('#hmSub').textContent = 'Rated ' + t.index + ' out of 100' + (t.why.length ? ' for ' + t.why.slice(0, 3).join(', ') : '') + '.' + (tm ? ' Tomorrow peaks at ' + hr(tm.localHour) + ' (' + tm.index + ').' : ''); } }
      const night = A.sunAltitude(bt.now + HOUR / 2, loc.lat, loc.lon) < -0.8, gcls = cur.unsafe ? 'g-bad' : cur.index >= 60 ? 'g-ok' : cur.index >= 45 ? 'g-warn' : 'g-low';
      $('#hmNow').innerHTML = '<div class="nowtop"><div class="nowscore">' + gauge(cur.index, gcls) + '<div><div class="nowlabel">Fishing right now</div><div class="nowword">' + (cur.unsafe ? 'Unsafe' : r[0]) + '</div><div class="small muted">' + esc(cur.why.slice(0, 2).join(', ') || 'steady conditions') + '</div></div></div><div class="nowwx">' + wxIcon(w.code, night) + '<div><div class="nowtemp">' + fmt(w.tempC * 9 / 5 + 32, 0) + '°<span>F</span></div><div class="small muted">' + fmt(w.tempC, 0) + ' °C, ' + esc((U.WMO[w.code] || '').toLowerCase()) + '</div></div></div></div><div class="stat-line"><span>Wind</span><b>' + fmt(w.windKph / 1.609, 0) + ' mph / ' + fmt(w.windKph, 0) + ' km/h</b></div><div class="stat-line"><span>Pressure</span><b>' + fmt(w.pressureHpa, 0) + ' hPa, ' + esc(w.pressureTrend || '—') + '</b></div><div class="stat-line"><span>Cloud cover</span><b>' + fmt(w.cloudPct, 0) + '%</b></div><div class="stat-line"><span>Sunrise and sunset</span><b>' + (bt.sunrise ? clock(bt.sunrise, bt.off) : '—') + ' / ' + (bt.sunset ? clock(bt.sunset, bt.off) : '—') + '</b></div><div class="stat-line"><span>Moon</span><b>' + esc(bt.phase.name) + ', ' + Math.round(bt.phase.fraction * 100) + '% lit</b></div>' + (bt.hours.slice(0, 6).some(x => x.unsafe) ? '<p class="note bad" style="margin-top:8px">Thunderstorms are forecast in the next few hours. Stay off the water.</p>' : '') + '<p class="small muted" style="margin-top:6px">Forecast for ' + esc(loc.name) + ' from Open-Meteo.</p>';
    }).catch(err => { if (seq !== homeSeq) return; heroCurve(null); $('#hmHeadline').textContent = 'No forecast right now'; $('#hmSub').textContent = 'The weather service did not answer. Fish reports and gear suggestions below still work.'; $('#hmTimes').innerHTML = '<p class="note bad">Could not load the forecast (' + esc(err.message) + '). Check your connection and try again.</p>'; $('#hmNow').innerHTML = ''; });
    CL.localFish(loc.lat, loc.lon).then(d => { if (seq !== homeSeq) return; homeFish = d; $('#hmFish').innerHTML = fishHTML(d, true, true); targetOptions($('#hmTarget'), d); $('#hmSetup').innerHTML = adviceHTML(CL.baitAdvice($('#hmTarget').value, d), d); })
      .catch(err => { if (seq !== homeSeq) return; $('#hmFish').innerHTML = '<p class="note bad">Could not load fish reports (' + esc(err.message) + ').</p>'; targetOptions($('#hmTarget'), null); $('#hmSetup').innerHTML = adviceHTML(CL.baitAdvice($('#hmTarget').value, null), null); });
  }
  $('#hmLoc').onchange = () => { setSpot($('#hmLoc').value); refreshHome(); };
  $('#hmTarget').onchange = () => { $('#hmSetup').innerHTML = adviceHTML(CL.baitAdvice($('#hmTarget').value, homeFish), homeFish); };
  const pickSpot = x => { const l = CL.ensureLocation(x.name, x.lat, x.lon, x.type); CL.pickLoc = l.id; CL.modules.home.render(); U.toast('Spot set to ' + l.name + '.'); };
  searchUI('#hmSearch', '#hmSearchOut', pickSpot);
  $('#hmGps').onclick = () => { if (!navigator.geolocation) return U.toast('This browser cannot share your location. Search by name instead.'); U.toast('Getting your location…'); navigator.geolocation.getCurrentPosition(async p => { const w = await reverseWater(p.coords.latitude, p.coords.longitude); pickSpot({ name: w ? w.name : 'My location', lat: p.coords.latitude, lon: p.coords.longitude, type: w ? w.type : 'Other' }); }, err => U.toast('Location not available (' + err.message + '). Search by name instead.'), { enableHighAccuracy: true, timeout: 15000 }); };

  /* ================= catch log ================= */
  CL.modules.catches = { render() {
    const m = CL.isMetric(); $('#ctLenLbl').firstChild.textContent = 'Length (' + (m ? 'cm' : 'in') + ')'; $('#ctWtLbl').firstChild.textContent = 'Weight (' + (m ? 'kg' : 'lb') + ')';
    const keep = $('#ctLoc').value || spot(); CL.locationOptions($('#ctLoc')); if (keep && Store.location(keep)) $('#ctLoc').value = keep;
    const kn = $('#ctNear').value || spot(); CL.locationOptions($('#ctNear')); if (kn && Store.location(kn)) $('#ctNear').value = kn;
    if (!$('#ctDate').value) { $('#ctDate').value = U.today(); $('#ctHour').value = new Date().getHours(); }
    const lures = new Set(Store.list('inventory').filter(i => !/line|terminal/i.test(i.category)).map(i => i.name)); Store.list('sessions').forEach(s => s.lure && lures.add(s.lure)); Store.list('catches').forEach(c => c.lure && lures.add(c.lure)); $('#lureList').innerHTML = Array.from(lures).sort().map(l => '<option value="' + esc(l) + '">').join('');
    speciesList(null); drawCatches(); drawOthers();
  } };
  function speciesList(fish) { const names = new Set(Object.keys(CL.SPECIES).filter(k => k !== 'Other')); Store.list('catches').forEach(c => names.add(c.species)); if (fish) fish.species.forEach(f => names.add(f.name)); $('#ctSpeciesList').innerHTML = Array.from(names).sort().map(n => '<option value="' + esc(n) + '">').join(''); }
  function drawCatches() {
    const m = CL.isMetric(), list = Store.list('catches').slice().sort((a, b) => (b.date + String(b.hour).padStart(2, '0')).localeCompare(a.date + String(a.hour).padStart(2, '0'))), len = c => c.lengthCm == null ? '—' : m ? fmt(c.lengthCm, 1) + ' cm' : fmt(c.lengthCm / 2.54, 1) + ' in', wt = c => c.weightKg == null ? '—' : m ? fmt(c.weightKg, 2) + ' kg' : fmt(c.weightKg / 0.45359237, 2) + ' lb';
    const bySp = {}; list.forEach(c => bySp[c.species] = (bySp[c.species] || 0) + 1); const topSp = Object.entries(bySp).sort((a, b) => b[1] - a[1])[0], big = list.filter(c => c.lengthCm != null).sort((a, b) => b.lengthCm - a.lengthCm)[0], heavy = list.filter(c => c.weightKg != null).sort((a, b) => b.weightKg - a.weightKg)[0];
    $('#ctKpis').innerHTML = [['Fish logged', list.length, list.filter(c => c.released).length + ' released'], ['Species', Object.keys(bySp).length, topSp ? 'most: ' + esc(topSp[0]) + ' (' + topSp[1] + ')' : 'none yet'], ['Longest', big ? len(big) : '—', big ? esc(big.species) + ', ' + esc(big.date) : 'add a length'], ['Heaviest', heavy ? wt(heavy) : '—', heavy ? esc(heavy.species) + ', ' + esc(heavy.date) : 'add a weight']].map(k => '<div class="kpi"><div class="l">' + k[0] + '</div><div class="v">' + k[1] + '</div><div class="s">' + k[2] + '</div></div>').join('');
    $('#ctCharts').hidden = !list.length;
    if (list.length) { const col = U.chartColors(), sp = Object.entries(bySp).sort((a, b) => b[1] - a[1]).slice(0, 8), byH = new Array(24).fill(0); list.forEach(c => { if (c.hour != null) byH[c.hour]++; });
      U.chart('ctSpecies', $('#ctSpeciesChart'), { type: 'bar', data: { labels: sp.map(x => x[0]), datasets: [{ label: 'Fish', data: sp.map(x => x[1]), backgroundColor: col[0], borderRadius: 6 }] }, options: CL.baseChartOpts({ indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { ticks: { precision: 0 } } } }) });
      U.chart('ctHour', $('#ctHourChart'), { type: 'bar', data: { labels: byH.map((_, h) => hr12(h)), datasets: [{ label: 'Fish', data: byH, backgroundColor: col[1], borderRadius: 4 }] }, options: CL.baseChartOpts({ plugins: { legend: { display: false } }, scales: { y: { ticks: { precision: 0 } }, x: { ticks: { maxTicksLimit: 8 } } } }) }); }
    $('#ctTable').innerHTML = list.length ? '<div class="tablewrap"><table><tr><th>Date</th><th>Fish</th><th class="num">Length</th><th class="num">Weight</th><th>Lure or bait</th><th>Where</th><th></th><th></th></tr>' + list.map(c => '<tr><td class="mono small">' + esc(c.date) + (c.hour != null ? ' ' + String(c.hour).padStart(2, '0') + 'h' : '') + '</td><td><b>' + esc(c.species) + '</b>' + (c.notes ? '<br><span class="small muted">' + esc(c.notes) + '</span>' : '') + '</td><td class="num">' + len(c) + '</td><td class="num">' + wt(c) + '</td><td>' + esc(c.lure || '—') + '</td><td>' + esc(Store.locationName(c.locationId)) + '</td><td>' + (c.released ? '<span class="badge ok">released</span>' : '<span class="badge">kept</span>') + '</td><td><button class="btn sm danger" data-ct-del="' + c.id + '" type="button">×</button></td></tr>').join('') + '</table></div>' : '<p class="muted small">No fish logged yet. Add your first catch with the form.</p>';
  }
  async function drawOthers() {
    const loc = Store.location($('#ctNear').value); if (!loc || !isFinite(loc.lat)) { $('#ctOthers').innerHTML = '<p class="muted small">Choose a spot to see reports from the area.</p>'; return; }
    $('#ctOthers').innerHTML = '<p class="muted small">Loading reports near ' + esc(loc.name) + '…</p>';
    try { const d = await CL.localFish(loc.lat, loc.lon); speciesList(d); $('#ctOthers').innerHTML = '<p class="small muted" style="margin:0 0 8px">' + d.total + ' reports of ' + d.nSpecies + ' fish species within ' + d.km + ' km of ' + esc(loc.name) + '. Latest first:</p>' + recentHTML(d, 14) + '<p class="small muted" style="margin-top:8px">' + INAT_NOTE + ' CastLab has no server, so it cannot share catches between CastLab users; to add your own fish to this public record, post it on iNaturalist.</p>'; }
    catch (e) { $('#ctOthers').innerHTML = '<p class="note bad">Could not load reports (' + esc(e.message) + ').</p>'; }
  }
  $('#ctNear').onchange = drawOthers;
  $('#ctForm').onsubmit = e => { e.preventDefault(); const m = CL.isMetric(), v = s => $(s).value === '' ? null : +$(s).value, len = v('#ctLen'), w = v('#ctWt');
    Store.add('catches', { species: $('#ctSpecies').value.trim(), lengthCm: len == null ? null : m ? len : len * 2.54, weightKg: w == null ? null : m ? w : w * 0.45359237, locationId: $('#ctLoc').value || null, lure: $('#ctLure').value.trim(), date: $('#ctDate').value, hour: v('#ctHour'), released: $('#ctReleased').checked, notes: $('#ctNotes').value.trim() });
    ['#ctSpecies', '#ctLen', '#ctWt', '#ctNotes'].forEach(s => $(s).value = ''); drawCatches(); U.toast('Catch saved.'); };
  $('#ctTable').onclick = e => { const b = e.target.closest('[data-ct-del]'); if (b && confirm('Delete this catch?')) { Store.remove('catches', b.dataset.ctDel); drawCatches(); } };
  $('#ctCsv').onclick = () => U.download('castlab-catches.csv', U.toCSV(Store.list('catches').map(c => Object.assign({}, c, { location: Store.locationName(c.locationId) }))), 'text/csv');
})();
