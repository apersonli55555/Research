# Physics-guided, conformally calibrated temperature compensation of bridge natural frequencies (KW51)

**Question.** When a model trained in winter must predict a bridge's natural frequencies in spring (temperatures it has never seen), do its 90 % prediction intervals still contain 90 % of the measurements? Does a small physics model of temperature-dependent stiffness help them do so?

**Data.** We use the public KW51 railway-bridge monitoring dataset (Maes & Lombaert, Zenodo [10.5281/zenodo.3745914](https://zenodo.org/record/3745914), CC-BY-NC-SA-4.0). We use only the 13 MB `trackedmodes.mat` file: hourly natural frequencies of 14 modes identified by the dataset authors, plus steel temperature, humidity, and weather. Oct 2018 to Jan 2020, with a retrofit from 15 May to 27 Sep 2019. **No new measurements were made.**

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                  # 5 sanity tests (~1 s)
python scripts/run_experiments.py    # downloads data (MD5-checked), ~5 min on 4 cores
python scripts/make_figures.py       # figures/ and results/summary.md
```

Everything is seeded and deterministic. `results/summary.md` is regenerated from the CSVs, so every number in the paper should be traceable to a CSV row.

## Layout
```
src/kw51uq/data.py        download + load + features (lagged temperature etc.)
src/kw51uq/models.py      baselines and the proposed physics-guided model
src/kw51uq/conformal.py   split conformal and adaptive conformal inference (ACI)
src/kw51uq/metrics.py     coverage, width, interval score, day-block bootstrap CI
src/kw51uq/detection.py   daily alarm rule and PCA (output-only) baseline
scripts/run_experiments.py, scripts/make_figures.py
results/*.csv, results/summary.md, figures/*.pdf|png
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
- **Modes:** 3, 5, 6, 9 and 13, the ones with less than 15 % missing frequencies before the retrofit.

### 5. Metrics
- **Coverage**: the fraction of test hours inside the interval. Target: 0.90.
- **Width**: interval width in Hz. Narrower is better, *if* coverage holds.
- **Interval (Winkler) score**: width + (2/α)·(miss distance). Lower is better, and it rewards being narrow *and* right.
- **Conditional coverage by temperature bin**: is coverage OK at *every* temperature, or only on average?
- **Day-block bootstrap 95 % CI**: resamples whole days, because hours within a day are correlated. Treating them as independent would make CIs far too narrow.

## Results so far (from `results/summary.md`; all numbers are real outputs of this code)
- **Random split:** every model reaches 0.90 ± 0.01 coverage. Conformal prediction works when its assumption holds.
- **Chronological split, mean over 5 modes:** coverage was **0.907 for Physics+GBM**, 0.903 Physics only, 0.901 GBM, 0.870 Linear, 0.826 Cubic and **0.788 GP**.
- **At test temperatures above 20 °C** (68 hours from only **8 days**; small sample), Physics+GBM covered **0.87–0.96** across the 5 modes. GBM without physics covered 0.51–0.90. GP covered 0.01–0.96 (it collapses on modes 6, 9 and 13), and the cubic covered 0.00–0.02. Mechanism: trees and RBF-kernel GPs cannot extrapolate a trend, and polynomials extrapolate the wrong one.
- **Interval score:** Physics+GBM was best or within 0.001 Hz of best on all 5 modes. The gain over GBM-no-physics is **small on average** and concentrated at unseen temperatures.
- **ACI** brings every model to about 0.89–0.91 coverage. So "physics is needed for coverage" is only true for *fixed* (non-adaptive) intervals, which are the ones you need for detection.
- **Retrofit detection:** the retrofit shifted mode 6 by about +0.06 Hz (+2 %), mode 9 by about −0.03 Hz, and mode 13 by about +0.10 Hz. Almost every method detects this on 100 % of post-retrofit days for modes 6, 9 and 13. Modes 3 and 5 do not respond. The models differ mainly in **false alarms** before the retrofit: 0–1.3 % for the physics and GBM models, versus 7–23 % for Linear, Cubic and GP on modes 6 and 13. **The retrofit is too large a change to rank the methods by detection power**, so a minimum-detectable-shift study is needed (task 4 below).
- **Fitted physics:** *a* < 0 (frequency falls as T rises) for modes 5, 6, 9 and 13. For mode 3 it came out **positive**, which is not physical for *E(T)*. Other effects (boundary conditions, ballast, the rail) must be involved. *T_c* ≈ 1.1–1.4 °C for all five modes. Another KW51 study (arXiv:2506.20295) reports a kink at 2–3 °C. Fig. 2 also shows that **no test hour is below 0 °C**, so the cold-stiffening part is fitted but **not tested out-of-sample**.

## Honest limitations
1. There is one chronological split. Results could depend on where the split boundaries fall, so a rolling-origin evaluation is needed.
2. Only 8 days are above 20 °C, so the headline extrapolation result rests on a small sample. Note the CI widths.
3. The physics model is a hypothesis, not a validated structural model. Mode 3's sign is unexplained.
4. The frequencies are the dataset authors' identified modes, so mode-tracking errors are inherited.
5. The retrofit is a *stiffening*, not damage. Say "state change", never "damage detection".
6. It is one bridge. Generalization is unknown.

## Your tasks (this is the part that makes the project yours)
1. **Verify** the retrofit dates (15 May–27 Sep 2019) in Maes & Lombaert (2021) or Maes et al. (2022). Fix `data.py` if they differ.
2. **Find a citable value** for steel's dE/dT/E near 0–30 °C (try the NIST technical note on temperature-dependent properties of structural steel, or Eurocode 3 tables). Compare it with the fitted *a* values in `results/physics_parameters.csv`. Do they agree in sign and order of magnitude?
3. **Literature check:** search Google Scholar for "conformal" + "structural health monitoring", "conformal" + "natural frequency", and "KW51". Read Jones/Pitchforth/Cross (EWSHM 2024) on physics-informed GP kernels. Write down how your work differs.
4. **Minimum detectable shift** (your main original experiment). Add a known synthetic shift of δ = 0.1 %, 0.25 %, 0.5 % or 1 % to the pre-retrofit *test* frequencies, labelled clearly as synthetic. Measure the detection rate vs δ for each model at a fixed false-alarm rate. This is what can actually separate the methods.
5. **Rolling-origin evaluation:** slide the train/cal/test windows month by month and report the coverage spread.
6. **PINN variant (connects to your earlier paper):** replace the GBM residual with a small neural net whose loss includes a penalty that keeps ∂f/∂T near the physics prediction. Compare it honestly with Physics+GBM.
7. **Ablations:** drop the cold term, drop humidity, drop lagged temperature. Which matters?

Write each result into a lab notebook *in your own words* before it goes into the paper.

## AI-assistance disclosure
The code scaffold, baseline experiments, figures and draft text were produced with an AI coding assistant (Claude) and reviewed by the student. Science fairs (including ISEF) and journals have rules on disclosing AI assistance and adult or mentor help. **Check the current rules and disclose accurately.**

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
