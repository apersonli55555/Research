# CastLab

CastLab combines everyday fishing tools with research-grade statistics and engineering simulation in one static website. All sections share one user profile, one set of fishing locations and one data store.

Open `castlab/index.html` through any static web server (for example `python3 -m http.server` in this folder, then visit `http://localhost:8000`). Geolocation and some browser features need HTTPS or `localhost`.

## Sections

| Section | What it does |
|---|---|
| **Today** | Home page: pick a spot (saved, searched by name, or your location) and see the best fishing times, current conditions, fish reported nearby and what to use |
| **Catch log** | Diary of fish landed (species, size, lure, place) with personal bests, plus what other people have reported in the area |
| **Research dashboard** | Session counts, catch probability and catch rate with credible intervals, prediction track record, experiments and simulation-vs-measurement summaries, all computed live from your records |
| **Fishing map** | Leaflet/OpenStreetMap map of saved locations, sessions and GPS observations. Click any lake or river to name it and open its best times or shoreline twin; search water bodies by name; optional layer of fish reported by others |
| **Weather & best times** | Open-Meteo hourly forecast plus a 48-hour fishing-time index built from sunrise/sunset, pressure trend, wind and moon position, with the formula shown |
| **BaitRadar inventory** | Lure, bait and tackle inventory with reorder alerts and value. Stock drops automatically when sessions log lures lost; each item shows its observed catch rate from LureLab |
| **GearMatch** | Species and technique setup recommendations (rod power, line, lure weight, drag) matched against your inventory and saved rods, with a one-click hand-off to the physics simulator |
| **LureLab** | Session log (including blanks), Beta-Binomial and Gamma-Poisson estimates, Bayesian logistic regression, trip predictions, registered randomised lure experiments, prequential validation and frozen forward predictions |
| **CastCheck shoreline twin** | 2D and 3D terrain from public elevation tiles, a water-level slider with connected flood simulation, bank-slope analysis, transects, USGS gage history, and comparison with GPS field observations |
| **GearMatch physics** | Large-deflection (elastica) model of a tapered hollow rod blank: animated deflection, stress along the blank, three-configuration comparison, setup compatibility and weakest link, Pareto optimisation of performance, mass and cost, and bench-test calibration against physical measurements |
| **Methods & exports** | CSV for every dataset, full JSON export, an auto-generated research report (HTML or Markdown) with methodology and results, and in-browser verification checks |
| **Account & data** | Local accounts (PBKDF2-hashed passwords), units, import/export, and an optional simulated demo dataset |

Every graph has PNG and CSV export buttons.

## Data integrity

* Nothing is pre-filled. Every number is computed from records the user creates or imports.
* The optional demo dataset is generated from a stated logistic model with known coefficients. Its records are tagged `source: "simulated"`, labelled wherever they appear, excluded from the dashboard and report, and never pooled with real observations. Its purpose is to show that the estimators recover known values.
* The synthetic terrain demo is labelled as a demonstration and cannot be compared with field observations.
* External data sources are named where they are used, along with their limitations:
  * OpenStreetMap: base map
  * Open-Meteo: weather model output, not station observations
  * USGS Water Services: gage height relative to a local datum; only changes are used
  * AWS Terrain Tiles: elevation; source and accuracy vary by location
  * iNaturalist: research-grade fish records near a spot. These are reports (catches, sightings, found fish), not population counts
  * OpenStreetMap Nominatim: names of lakes and rivers
* The fishing-time index is a transparent rule of thumb, labelled as guidance and not validated against catch data. After 20 logged trips it blends in the user's own success rate by time of day.
* Bait and gear suggestions are labelled as common guidance; only the "worked for you" lines are measured results.
* Model accuracy is reported only from out-of-sample predictions: prequential forward-chaining, and forward predictions frozen before the trip.

## Storage

There is no server. Accounts are local profiles in the browser's `localStorage`; use **Account → Export** to back up or move data. Clearing site data deletes it.

## Code layout

```
castlab/
  index.html          page shell and all section markup
  css/castlab.css     light/dark theme, responsive layout
  js/stats.js         Beta/Gamma special functions, conjugate posteriors, Bayesian logistic regression (Newton + Laplace), scoring rules
  js/physics.js       tapered-beam elastica solver (RK4 shooting), bench calibration, compatibility, Pareto optimiser
  js/terrain.js       Terrarium decoding, Horn slope, water detection, flood fill, marching-squares shoreline, transects
  js/store.js         accounts, shared data store, CSV/download helpers, Open-Meteo and USGS clients
  js/astro.js         sun and moon position, rise/set/transit times, fishing-time index
  js/field.js         Today page, best times, nearby fish, catch log, bait/gear suggestions, map lookups
  js/app.js           routing, dashboard, map, weather, BaitRadar, GearMatch, exports, account
  js/lurelab.js       LureLab
  js/gearsim.js       physics simulator UI
  js/castcheck.js     shoreline twin (ES module, three.js)
  vendor/             Leaflet 1.9.4, Chart.js 4.4.7, three.js r170 (+ OrbitControls), served locally
  tests/run.js        56 numerical verification checks (node castlab/tests/run.js)
  tests/browser_smoke.cjs  end-to-end headless-browser test of every section (Playwright)
```

## Verification

`node castlab/tests/run.js` checks the engines against closed-form and published values:

* Beta CDF and quantiles against exact values and R's `qbeta`; Gamma quantiles against R's `qgamma`; P(A>B) exact cases.
* Logistic regression recovers known coefficients from 4,000 simulated observations, each within 3 standard errors.
* Rod solver vs the large-deflection cantilever tables (Bisshopp & Drucker 1945; Mattiasson 1981) at four load levels, to within 2×10⁻⁴ L. Also checked: the small-load limit FL³/3EI, an energy balance, axial loading and the butt moment.
* Sun and moon: sunrise, sunset, moonrise, moonset, phase and illuminated fraction against the SunCalc reference values.
* Terrain: Horn slope on a plane; water detection; shoreline advance Δh/tan β on a 10° bank; transect waterline; shoreline length; point-to-line distance; Web-Mercator round trip.

## Known limitations

* LureLab sessions are observational: lure effects can be confounded by when and where each lure was chosen. Causal comparisons need the experiment tool.
* The shoreline twin is a connected "bathtub" model. Below the DEM water surface the bed is extrapolated from the bank slope (labelled as an assumption), and DEM resolution limits shoreline accuracy to roughly the grid cell size.
* The rod model is static, ignores guides, ferrules and self-weight, and uses nominal material constants until calibrated with a bench test. Its safety factors are not failure predictions.
