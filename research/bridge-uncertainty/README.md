# Physics-guided, conformally calibrated temperature compensation of bridge natural frequencies (KW51)

**Question.** When a model trained in winter must predict a bridge's natural frequencies in spring (temperatures it has never seen), do its 90 % prediction intervals still contain 90 % of the measurements? Does a small physics model of temperature-dependent stiffness help them do so?

**Data.** We use the public KW51 railway-bridge monitoring dataset (Maes & Lombaert, Zenodo [10.5281/zenodo.3745914](https://zenodo.org/record/3745914), CC-BY-NC-SA-4.0). We use only the 13 MB `trackedmodes.mat` file: hourly natural frequencies of 14 modes identified by the dataset authors, plus steel temperature, humidity, and weather. Oct 2018 to Jan 2020, with a retrofit from 15 May to 27 Sep 2019. **No new measurements were made.**

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make all            # or: bash scripts/run_all_parallel.sh   (~30-60 min on 4 cores)
```
Or step by step:
```bash
python -m pytest -q                  # unit + smoke tests
python scripts/run_experiments.py    # main split, random split, detection (downloads data, MD5-checked)
python scripts/run_robustness.py     # 5 hand-designed folds (incl. cold-side and post-retrofit)
python scripts/run_rolling.py        # 22 systematic rolling windows (slow; see Makefile for the parallel version)
python scripts/paired_tests.py       # paired day-block bootstrap tests of coverage differences
python scripts/run_sensitivity.py    # hyperparameter sensitivity + physics-parameter CIs (~15 min)
python scripts/run_kink_check.py     # physics form vs a plain two-regime line (~1 min)
python scripts/run_shared_threshold.py  # shared freezing threshold + slope by mode type (~3 min)
python scripts/make_figures.py       # figures/ and results/summary.md
```
Everything is seeded and deterministic. `results/summary.md` is regenerated from the CSVs, so every number in the paper should be traceable to a CSV row. Tests run automatically on GitHub (`.github/workflows/bridge-uncertainty-tests.yml`).

## Layout
```
src/kw51uq/data.py          download + load + features (lagged temperature etc.)
src/kw51uq/models.py        9 models: baselines, ablations, proposed physics-guided model
src/kw51uq/conformal.py     split conformal and adaptive conformal inference (ACI)
src/kw51uq/metrics.py       coverage, width, interval score, day-block bootstrap CI
src/kw51uq/detection.py     daily alarm rule and PCA (output-only) baseline
scripts/run_experiments.py  main + random split + detection
scripts/run_robustness.py   5 designed folds
scripts/run_rolling.py      systematic rolling-origin windows
scripts/paired_tests.py     paired statistical comparisons
scripts/make_figures.py     all figures + results/summary.md
docs/                       one-page summary, manuscript draft, ISEF board outline, LUMO email draft
```

## Methods, explained

### 1. What is being measured
A natural frequency is a rate at which the bridge "prefers" to vibrate. For a simple oscillator, *f = (1/2π)·√(k/m)*: stiffer means higher frequency. Damage or repairs change *k*, so *f* is a health indicator. But temperature also changes *k*, often by as much as moderate damage does. "Temperature compensation" means predicting what *f* should be at the current temperature, then looking at the **residual** (measured minus predicted).

### 2. Models (all predict one mode's frequency from covariates)
| Model | Idea | Why it's included |
|---|---|---|
| Linear(T) | *f = a + bT* | simplest standard compensation |
| Cubic(T) | cubic polynomial in *T* | polynomial regression is common in SHM; tests extrapolation |
| GP | Gaussian process on (T, 24 h mean T, RH) | flexible Bayesian model with its own uncertainty |
| GBM (no physics) | gradient-boosted trees on the same features as the hybrid | isolates what the physics adds |
| Physics only | *f = f₀·√(1 + a(T − 10)) · (1 + b·softplus(T_c − T))* | 4 interpretable parameters |
| Linear + GBM / Physics(no cold) + GBM | ablations: same learner on a linear trend / on physics without the cold term | is it the physics form, or any trend? |
| GP + linear kernel | GP that can extrapolate a linear trend | a fairer GP |
| **Physics + GBM (proposed)** | physics curve plus a small tree model fitted to the physics residuals | physics carries the trend; data fixes the rest |

**Where the physics term comes from.** For steel, Young's modulus *E* decreases slightly as temperature rises. Since *k ∝ E* and *f ∝ √k*, *f ∝ √(1 + a·ΔT)*, with *a* the relative change of *E* per °C. Below a critical temperature *T_c* the bridge stiffens a lot more (see Fig. 2). One possible cause is frozen ballast or bearings, but **this dataset alone does not prove the cause.** We model that stiffening with a smooth hinge (softplus). The fitted *a* can be checked against handbook values; that check is a task for you (see below).

### 3. Conformal prediction (how the error bars are made)
Split conformal (Angelopoulos & Bates 2021, arXiv:2107.07511):
1. Train the model on the **training** months.
2. On separate **calibration** months, compute the absolute errors |y − ŷ|.
3. Take the ⌈(n+1)(1−α)⌉-th smallest error as *q*.
4. The interval for any new hour is ŷ ± q.

**Guarantee:** if calibration and test points are *exchangeable* (any reordering is equally likely), the interval contains the truth with probability ≥ 1 − α. Seasonal bridge data are **not** exchangeable, so the guarantee is not promised. This project **measures how badly it breaks** and whether physics helps.

**Adaptive conformal inference (ACI)** (Gibbs & Candès, NeurIPS 2021, arXiv:2106.00170) nudges α after every observation: it widens after a miss and narrows after a hit. It restores long-run coverage under drift. The catch for monitoring: it also adapts to *real* structural change, which would hide damage. So we use it only to study coverage, never for detection.

### 4. Experimental design
- **Random split** (Experiment A): hours from Nov 2018 to 14 May 2019 are randomly assigned to train (50 %), calibration (20 %) and test (30 %), repeated with 5 seeds. This is the *optimistic* setting where the exchangeability guarantee roughly applies. Note that it leaks information, because neighbouring hours are nearly identical.
- **Chronological split** (Experiment B, the main one): train Nov 2018–Jan 2019, calibrate Feb 2019, test 1 Mar–14 May 2019. The highest temperature in training is 13.6 °C, in calibration 18.1 °C, and in test 26.3 °C. **Steel temperature is missing for October 2018**, so October is not used.
- **Retrofit detection** (Experiment C): intervals stay frozen at the Feb calibration. For each day, compute the median of residual/q. An alarm is raised if |median| > 1. The false-alarm rate is measured on pre-retrofit test days and the detection rate on post-retrofit days (28 Sep 2019–15 Jan 2020).
- **Robustness** (Experiment D, `scripts/run_robustness.py`): five folds. Three run forward in time (winter→spring). One runs warm→cold before the retrofit and is the only out-of-sample test of sub-zero hours. One runs warm→cold after the retrofit, a different structural state. Coverage is reported separately for test hours inside and outside the training temperature range.
- **Modes:** 3, 5, 6, 9 and 13, the ones with less than 15 % missing frequencies before the retrofit.

### 5. Metrics
- **Coverage**: the fraction of test hours inside the interval. Target: 0.90.
- **Width**: interval width in Hz. Narrower is better, *if* coverage holds.
- **Interval (Winkler) score**: width + (2/α)·(miss distance). Lower is better, and it rewards being narrow *and* right.
- **Conditional coverage by temperature bin**: is coverage OK at *every* temperature, or only on average?
- **Day-block bootstrap 95 % CI**: resamples whole days, because hours within a day are correlated. Treating them as independent would make CIs far too narrow.

## Results (from `results/summary.md` and `results/robustness_folds.csv`; all numbers are real outputs of this code)

**Version note.** v2 added stronger baselines (GP + linear kernel, Linear + GBM), an ablation (physics without the cold term), and a 5-fold robustness study. It also switched humidity from the deck sensor (missing all of Dec 2018) to the weather station. That fix raised the GP's main-split coverage from 0.79 to 0.86. **The proposed model was not changed after seeing the robustness results.**

1. **Exchangeable (random) split:** all 9 models reach 0.90 ± 0.01 coverage. Conformal prediction works when its assumption holds.
2. **Main seasonal split, mean coverage over 5 modes:** Physics+GBM 0.904, Physics only 0.903, GBM 0.890, Linear 0.871, Physics(no cold)+GBM 0.871, Linear+GBM 0.858, GP 0.855, GP+linear kernel 0.847, Cubic 0.826.
3. **Warm-side extrapolation** (folds F2 and F3, test hours warmer than anything in training, 5 modes): coverage of the nominal 90 % interval.

   | Model | Median | Worst |
   |---|---|---|
   | **Physics+GBM** | **0.90** | **0.84** |
   | Physics only | 0.88 | 0.76 |
   | GBM | 0.77 | 0.59 |
   | Physics(no cold)+GBM | 0.75 | 0.34 |
   | GP+linear kernel | 0.69 | 0.04 |
   | Linear+GBM | 0.66 | 0.15 |
   | Linear | 0.56 | 0.01 |
   | GP | 0.36 | 0.01 |
   | Cubic | 0.26 | 0.00 |

4. **What actually helps is the cold term, and not for the reason you'd guess.** Fitted on winter data *without* a cold term, the warm-weather temperature slope comes out about 2× too steep for modes 6 and 13 (e.g. mode 13: −1.1e-3 /°C vs −4.7e-4 /°C fitted on warm hours only). The freeze-related stiffening drags the line. Modelling the cold regime separately keeps that stiffening out of the slope, so the slope extrapolates sensibly into spring. (With the cold term the slope is somewhat *too shallow*, about −2.9e-4 for mode 13, but it errs in the less harmful direction.)
5. **Cold-side extrapolation fails for every model.** Fold B1 trains on spring and tests on winter. Median outside-range coverage is 0.42–0.64 for all 9 models. On hours below 0 °C (about 130 h over 11 days), mode 6 coverage is 0.02–0.13 and mode 13 is 0.00–0.01 for every model. No model, including this one, can predict a stiffening mechanism it never saw.
6. **The proposed model is not always best.** In the post-retrofit warm→cold fold (Q1) it reaches 0.86 on out-of-range hours, while Linear, GBM, GP and Physics(no cold)+GBM reach 0.90–0.92. The likely reason is that the training data never got below 4.7 °C, so the cold-term parameters are not identifiable. A principled fix is a prior on *T_c* and *b* from an earlier winter; that is future work, deliberately not tuned on these test folds.
7. **The GP's own Bayesian intervals** cover 0.83–0.99 on the main split, but by being wide: 2.1× the proposed model's width on mode 13. Their interval score is worse than the proposed model's on modes 6, 9 and 13 (e.g. 0.097 vs 0.052 Hz on mode 13) and about equal on modes 3 and 5.
8. **ACI** brings every model to 0.889–0.907 coverage, but it adapts to whatever it sees, including real damage.
9. **Retrofit detection:** about +0.06 Hz on mode 6, −0.03 Hz on mode 9 and +0.10 Hz on mode 13. All models detect it on 100 % of post-retrofit days for those modes. False-alarm days before the retrofit: Physics+GBM 0–1.3 %, GBM 0–2.7 %, Linear/Cubic/GP/GP+linear kernel up to 7–19 %. The retrofit is too large to rank methods by detection power, so the minimum-detectable-shift study (task 4) is still needed.
10. **Fitted physics:** *a* < 0 for modes 5, 6, 9 and 13 and positive (unexplained) for mode 3; *T_c* ≈ 1.1–1.4 °C for all modes.

### v3: systematic rolling windows and paired statistical tests
- **Rolling-origin study** (`scripts/run_rolling.py`): 22 windows (8-week train, 3-week calibration, 4-week test, stepped weekly), 11 forward and 11 backward, × 5 modes × 9 models. No split is chosen by hand.
  - *Forward windows, test hours warmer than training* (55 window×mode cells): the proposed model's median coverage is **0.907**, its worst cell 0.67, and 80 % of cells reach ≥ 0.85. Best alternatives: Physics only 0.903 (65 % of cells ≥ 0.85), GBM 0.884 (55 %), GP 0.79, Linear + GBM 0.77, Cubic 0.32.
  - *Backward windows, test hours colder than training* (35 cells): **every model fails** (medians 0.50–0.62; worst 0.05).
  - *Inside the training range*: all models have medians of 0.90–0.92.
- **Paired tests** (`scripts/paired_tests.py`): the coverage difference (proposed − other) is computed on the same hours. CIs come from resampling whole calendar days, and a Holm correction covers the 8 comparisons.
  - Warm side, rolling windows (3,259 hours): +0.031 vs Physics only [0.009, 0.059], +0.084 vs GBM [0.051, 0.115], +0.16 vs Physics(no cold)+GBM, +0.20 vs GP+linear kernel, +0.24 vs Linear+GBM, +0.36 vs GP, +0.38 vs Linear, +0.61 vs Cubic. **All Holm-adjusted p ≤ 0.002.** The designed folds F2+F3 give the same picture.
  - Cold side, rolling windows: no significant differences (all CIs include 0, except a −0.019 [−0.032, −0.004] *disadvantage* vs Physics only).
  - Post-retrofit fold Q1: the proposed model is **significantly worse** than five comparators (−0.11 to −0.13; pooled coverage 0.78 vs 0.89–0.91).
- **Bottom line, now statistically backed:** the physics-guided model is the only model that keeps nominal coverage when extrapolating to warmer temperatures, and the gain is significant against every alternative. It does not help, and can hurt, when the freezing regime is unseen or unidentifiable.

### v4: reviewer checks (all on the 11 forward rolling windows unless noted)
- **Not buying coverage with width.** On warm-side out-of-range hours the proposed model has the *narrowest* median band; every other model's is 0.5–26 % wider. Its coverage is still the highest.
- **Not tuned.** Each of 9 residual-learner settings (depth 2/3/5 × learning rate 0.02/0.05/0.1) gives warm-side coverage of 0.883–0.899. The physics-free GBM given the *same* sweep tops out at 0.826, so every setting of the proposed model beats every setting of the competitor. Freezing-hinge width *s*: 0.5 °C → 0.907, 1 °C (default) → 0.895, 2 °C → 0.838. A softer hinge lets the cold effect leak into the warm slope, consistent with the ablation. The default was fixed in advance and is *not* the best setting. Inside the training range, all settings of both models give 0.908–0.918. (`results/sensitivity.csv`)
- **Physics parameters with uncertainty** (day-block bootstrap, `results/physics_parameter_ci.csv`):
  - (df/f) per °C is −1.3e-4 [−1.9e-4, −5.0e-5] for mode 6, −1.3e-4 [−1.5e-4, −1.0e-4] for mode 9 and −2.9e-4 [−4.0e-4, −1.4e-4] for mode 13. These are negative, as thermal softening predicts.
  - Mode 5's interval includes 0. Mode 3 is positive, +2.1e-4 [4.7e-5, 3.5e-4]: a real but unexplained effect.
  - *T_c* intervals span roughly −1.4 to +2.4 °C for every mode, which includes 0 °C. That is consistent with a freezing mechanism, but not proof of one.
- **Code self-review:** no leakage of future data into features, conformal and bootstrap arithmetic checked, and default settings reproduce the saved predictions exactly.

### v5: the "physics" is really a two-regime shape (important correction, 10 Nov 2026)
`√(1 + a·ΔT)` is almost a straight line when *a* is as small as it is here (about 5e-4 per °C). So we tested a **physics-free model of the same shape**: a straight line plus a smooth kink, `f = c0 + c1·T + c2·softplus(T_c − T)`, with the same boosted-tree correction (`models.KinkGBM`, `scripts/run_kink_check.py`).

| Setting (22 rolling windows, paired on the same hours) | Proposed | Kink line + GBM | Difference [95 % CI] |
|---|---|---|---|
| Warm side, out of range (3,259 h) | 0.895 | 0.896 | −0.002 [−0.006, +0.002] |
| Cold side, out of range (5,244 h) | 0.475 | 0.472 | +0.003 [+0.001, +0.006] |
| Inside the training range (63,778 h) | 0.918 | 0.918 | +0.000 [−0.000, +0.001] |

The two models are **indistinguishable**, and they fit the same slopes and thresholds to three digits.

**What this changes.**
- The earlier statement that the proposed model "beats every physics-free alternative" is **wrong**. It beat the eight alternatives first tested. A physics-free kinked line matches it.
- The benefit does **not** come from the thermo-elastic formula. It comes from **separating the freezing regime from the warm-weather trend**, which is what the ablation in v2 already pointed to.
- The physics still earns its place in two ways: it is *why* you would look for a softening slope and a freezing threshold in the first place, and it makes the fitted numbers interpretable. But "physics-guided" should be read as "physics-motivated two-regime model", and the claim to make is about regime separation.

### v6: mode types, shared freezing threshold, retrofit dates (10 Nov 2026)
From `scripts/run_shared_threshold.py`:
- **Mode types** (from the dataset authors' plotting code): lateral deck modes 3 and 11; global vertical 5, 6, 10, 12, 13; torsional 9; arch 1, 2, 4, 7, 8, 14. Of the five modes studied here, three are vertical, one is lateral and one is torsional.
- **Temperature trend by type** (pre-retrofit hours above 5 °C): all 12 non-lateral modes soften as they warm, each with a 95 % range below zero. Lateral mode 3 stiffens (+1.2e-4 per °C [+0.6e-4, +1.7e-4]). Lateral mode 11 shows no clear trend (−0.6e-4 [−2.0e-4, +0.6e-4]). So the "unexplained" positive sign of mode 3 goes with its being a lateral mode. See `docs/mode3_lateral_note.md` for hypotheses.
- **Shared freezing threshold:** one common threshold for all five modes is 1.22 °C [−0.40, 2.25]. Its uncertainty (width 2.65 °C) is no smaller than the separate fits' (2.36–3.01 °C), because all modes see the same few freezing days.
- **Retrofit dates** (15 May–27 Sep 2019) are confirmed in the authors' code.

### v7: what the dataset authors' 2022 paper already says (read 10 Nov 2026)
Reading the full text of Maes, Van Meerbeeck, Reynders & Lombaert (2022, MSSP 165, 108380) showed that two things this project presented as its own findings are **already in that paper**:
- **The two-regime shape.** They report "a bilinear relation between the identified natural frequencies and the bridge temperature ... with a knee point around 0 °C", and note the same was seen on the Z24 bridge.
- **The cause of the cold stiffening.** They state that finite element model updating indicates it "is likely due to frost of the porphyry ballast layer supporting the tracks". Earlier versions of this project called the cause "not established"; the authors do give one.
- **The slope bias.** They restrict their linear regression to temperatures above 2 °C because the cold data "leads to a bias in the linear regression". That is the same effect as this project's ablation.

So the regime structure and its explanation are prior work, to be cited. What this project adds is narrower: it keeps the cold data in one model instead of discarding it, and it measures what the regime structure does to **conformal coverage** inside and outside the training range.

Other facts from the paper: the bridge is 115 m long and 12.4 m wide, carries two ballasted tracks, and rests on four pot bearings. Their mode table calls mode 3 the "1st lateral mode of the bridge deck", mode 11 the "2nd lateral mode of the bridge deck (+ torsion)" and **mode 14 the "3rd lateral mode of the bridge deck (+ torsion)"**. Their plotting code colours mode 14 as an arch mode, so v6 above counted it as one. Mode 14 clearly softens when warm, so only the *first* lateral deck mode stiffens. The paper says frequencies "tend to decrease with increasing temperature" and does not discuss mode 3's opposite trend.

## Honest limitations
1. **Cold regime:** no method handles sub-zero temperatures when trained without them. This is the biggest open problem.
2. **Small extrapolation samples:** outside-range samples are small (median 96–379 hours per fold; the >20 °C bin is 8 days).
3. **One bridge.** Generalization to other structures is unknown.
4. **The retrofit is a stiffening, not damage.** Say "state change".
5. **Inherited mode tracking:** frequencies come from the dataset authors' mode identification and tracking.
6. **Window design:** the rolling windows cover only one pre-retrofit season (Nov 2018–May 2019), so they overlap heavily and are not independent replications. The paired tests account for shared days, but not for the fact that it is one winter and one spring.

## Your tasks (this is the part that makes the project yours)
1. ~~Verify the retrofit dates.~~ Done: confirmed in the dataset authors' code. Still read Maes & Lombaert (2021) for the bridge description.
2. **Find a citable value** for steel's dE/dT/E near 0–30 °C (try the NIST technical note on temperature-dependent properties of structural steel, or Eurocode 3 tables). Compare it with the fitted *a* values in `results/physics_parameters.csv`. Do they agree in sign and order of magnitude?
3. **Literature check:** search Google Scholar for "conformal" + "structural health monitoring", "conformal" + "natural frequency", and "KW51". Read Jones/Pitchforth/Cross (EWSHM 2024) on physics-informed GP kernels. Write down how your work differs.
4. **Minimum detectable shift** (your main original experiment). Add a known synthetic shift of δ = 0.1 %, 0.25 %, 0.5 % or 1 % to the pre-retrofit *test* frequencies, labelled clearly as synthetic. Measure the detection rate vs δ for each model at a fixed false-alarm rate. This is what can actually separate the methods.
5. **Cold-regime prior (the biggest open problem):** fit *T_c* and *b* on one winter, carry them as fixed or prior values into a model trained on warm data, and test on fold B1. Can physics transfer across seasons?
6. **PINN variant (connects to your earlier paper):** replace the GBM residual with a small neural net whose loss includes a penalty that keeps ∂f/∂T near the physics prediction. Compare it honestly with Physics+GBM.
7. **Ablations:** drop the cold term, drop humidity, drop lagged temperature. Which matters?

Write each result into a lab notebook *in your own words* before it goes into the paper.

## AI-assistance disclosure
The code scaffold, baseline experiments, figures and draft text were produced with an AI coding assistant and reviewed by the student. Science fairs (including ISEF) and journals have rules on disclosing AI assistance and adult or mentor help. **Check the current rules and disclose accurately.**

## References (verified to exist; read before citing)
- Maes, K. & Lombaert, G. (2021). Monitoring railway bridge KW51 before, during, and after retrofitting. *J. Bridge Eng.* 26(3). doi:10.1061/(ASCE)BE.1943-5592.0001668
- Maes, K., Van Meerbeeck, L., Reynders, E., Lombaert, G. (2022). Validation of vibration-based structural health monitoring on retrofitted railway bridge KW51. *MSSP* 165, 108380.
- Peeters, B. & De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. *EESD* 30(2).
- Angelopoulos, A.N. & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511
- Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 34*. arXiv:2106.00170
- Wittenberg, Mendler, Knoth, Gertheiss (2025). Multivariate long-term profile monitoring with application to the KW51 railway bridge. arXiv:2506.20295
- Wittenberg, Neumann, Mendler, Gertheiss (2025). Covariate-adjusted functional data analysis for structural health monitoring. *Data-Centric Engineering* 6:e27. arXiv:2408.02106
- Neumann, Wittenberg, Gertheiss (2025). Confidence intervals for conditional covariances of natural frequencies. arXiv:2509.26348 (IOMAC 2025)
- Neumann, Wittenberg, Gertheiss (2026). Higher-order multivariate environmental influences in SHM. arXiv:2603.22950; and Removal of multivariate environmental influences in SHM through conditional covariances and supervised learning. arXiv:2606.05026
- Hinchliffe, Jones, Cross (2025). Parameter estimation by optimization of interpretable hyperparameters with physics-informed Gaussian processes. IWSHM 2025.
- Gneiting & Raftery (2007), *Strictly proper scoring rules, prediction, and estimation*, JASA. Source of the interval score. **Not re-verified in this session; check it.**
