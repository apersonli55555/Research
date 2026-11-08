# Do conformal error bars survive an unseen season? Physics-guided temperature compensation of bridge natural frequencies on the KW51 railway bridge

*[Your name]¹ — ¹[High school], Austin, Texas, USA*
*DRAFT v0.4, 2026-11-08. Every number below comes from `results/` in this repository at the commit that contains this file. Items marked **[TODO]** need your own work or verification before submission.*

---

## Abstract
Vibration-based structural health monitoring infers changes in stiffness from shifts in natural frequencies, but temperature shifts those frequencies by a comparable amount. Alarm thresholds must therefore come from prediction intervals that stay trustworthy when conditions change. Split conformal prediction guarantees coverage only for exchangeable data, an assumption that seasonal monitoring data violate.

Using public hourly natural frequencies of the KW51 steel railway bridge, we evaluate nine temperature-compensation models, including ablations, all wrapped in the same split-conformal procedure. We test them under a random split, five designed seasonal folds, and 22 systematic rolling-origin windows, and report coverage separately for test hours inside and outside the training temperature range. Model differences are tested with paired day-block bootstrap comparisons.

- **Random split:** every model is calibrated (0.90 ± 0.01).
- **Warm-side extrapolation** (test hours warmer than any training hour): a model combining a thermo-elastic stiffness curve with a cold-stiffening term and a gradient-boosted residual learner reaches a median coverage of 0.907 across 55 rolling window×mode cells, with 80 % of cells at or above 0.85. Its pooled coverage is 0.895, versus 0.810 for the best physics-free model and 0.534 for a Gaussian process. It is significantly better than all eight alternatives: paired differences +0.03 to +0.61, Holm-adjusted p ≤ 0.002.
- **Ablations:** the benefit comes mainly from the cold-stiffening term. Without it, freeze-related stiffening inflates the fitted warm-side temperature slope (by up to about two-fold), and median coverage drops to 0.75.
- **Cold-side extrapolation:** when models trained on warmer data are tested on colder hours, every model fails (rolling medians 0.50–0.62, with no significant differences). In a post-retrofit warm→cold fold, the proposed model is significantly *worse* than five baselines (pooled 0.78 vs 0.89–0.91).
- **Detection:** with intervals frozen after calibration, the proposed model flags the 2019 retrofit on every post-retrofit day for three of five modes, with 0–1.3 % false-alarm days.

Physics priors make conformal intervals robust when extrapolating along a mechanism the model has seen. They do not help with an unseen one. **[TODO: add the minimum-detectable-shift result.]**

**Keywords:** structural health monitoring; conformal prediction; temperature compensation; physics-guided machine learning; uncertainty quantification; distribution shift

## 1. Introduction
Natural frequencies are attractive health indicators: they are global, can be identified from ambient vibration, and fall when stiffness is lost. Their major weakness is sensitivity to environmental and operational variability. In the Z24-bridge study, seasonal temperature changes produced frequency changes that had to be removed before damage could be distinguished [Peeters & De Roeck 2001]. The usual remedy is to regress frequency on temperature (or to remove environmental variation with output-only methods such as PCA) and to monitor the residuals [Maes et al. 2022].

An alarm threshold on residuals is an uncertainty statement: "this frequency is inconsistent with the bridge's healthy state at this temperature." If the threshold is too tight, there are false alarms; if it is too loose, damage is missed. Conformal prediction [Angelopoulos & Bates 2021] gives a principled threshold with a coverage guarantee under exchangeability. Monitoring data are serially correlated and seasonal, and a model calibrated in one season is used in another. Whether conformal coverage survives this shift, and what kind of model helps it survive, are empirical questions. This paper asks them directly.

**Contributions.**
1. A controlled comparison of nine temperature-compensation models on KW51, including strong baselines (a GP with a linear-trend kernel) and ablations, all using the same split-conformal wrapper.
2. An evaluation protocol that reports coverage separately for test conditions **inside vs outside the training range**, across designed folds (warm-side, cold-side, and a second structural state) and 22 systematic rolling-origin windows. Model differences are compared with a paired bootstrap that resamples calendar days. Average coverage hides the failures; this protocol exposes them and tests them.
3. Evidence that a four-parameter physics prior keeps conformal coverage near nominal under warm-side extrapolation, plus an ablation identifying *why*: the cold-stiffening term removes a confound from the warm-side slope.
4. A clear negative result: no model, physics-guided or not, extrapolates into an unseen sub-zero stiffening regime.
5. Fully reproducible code (MD5-checked download, seeded experiments, auto-generated tables).

**[TODO: novelty statement. Run the Google Scholar searches from README task 3, then write one sentence on what is new relative to Wittenberg et al. (2025) and Neumann et al. (2025–2026), and the Cross group's physics-informed GP work. Do not write "first" unless you are sure.]**

## 2. Related work
**Environmental compensation on KW51.** Maes & Lombaert (2021) describe the monitoring campaign and dataset. Maes et al. (2022) remove environmental variation with robust PCA and two black-box models, then show that the retrofit can be detected. Wittenberg et al. (2025a) model the nonlinear temperature effect with functional-data regression and multivariate EWMA control charts, and report a kink in the frequency–temperature relation between 2 and 3 °C. Wittenberg et al. (2025b) propose covariate-adjusted functional data analysis. Neumann et al. (2025) construct bootstrap confidence intervals for *conditional covariances* of natural frequencies and evaluate their coverage in simulation, finding severe under-coverage below 0 °C. Neumann et al. (2026a, 2026b) study higher-order multivariate environmental influences and their removal. To our reading, none of these evaluates the coverage of prediction intervals for the frequencies themselves under seasonal shift **[TODO: confirm by reading the full texts]**.

**Physics-informed uncertainty in SHM.** Physics-informed Gaussian-process kernels derived from oscillator dynamics have been used for interpretable parameter estimation [Hinchliffe et al. 2025]. Related kernels encode partial physical insight and change points [Jones et al. 2024; Pitchforth et al. 2024, **TODO: get exact citations**]. A change-point kernel is a natural alternative to our cold-stiffening term, and a comparison is future work.

**Conformal prediction in SHM.** Conformal methods have been used for anomaly detection on fibre-optic strain data from a railway bridge [**TODO: full citation, Ann. Appl. Stat. 15(3), "Unsupervised streaming anomaly detection for instrumented infrastructure"**] and for damage identification with graph neural networks [**TODO: full citation, Adv. Eng. Informatics 2025, "A hybrid data-physics framework with conformal GNN…"**]. Adaptive conformal inference [Gibbs & Candès 2021] targets distribution shift in general.

## 3. Data
The KW51 bridge is a steel bowstring railway bridge in Leuven, Belgium **[TODO: add span and structural description from Maes & Lombaert 2021]**. It was monitored from October 2018 to January 2020 and retrofitted between 15 May and 27 September 2019 **[TODO: verify dates]** to correct a construction error in the connections of the diagonals. We use the dataset authors' tracked modal parameters (`trackedmodes.mat`, Zenodo 10.5281/zenodo.3745914). These comprise 11,328 hourly records of 14 natural frequencies and damping ratios identified with reference-based covariance-driven stochastic subspace identification, plus steel surface temperature, relative humidity and weather-station variables.

Missing data:
- Steel temperature is missing throughout October 2018 and for 72 % of November 2018.
- Deck relative humidity is missing for all of December 2018, so the weather-station humidity is used throughout.

We analyse modes 3, 5, 6, 9 and 13, which have less than 15 % missing frequencies before the retrofit. Median pre-retrofit frequencies are 1.893, 2.580, 2.923, 4.109 and 6.320 Hz. Fig. 1 shows mode 6 and steel temperature over the whole campaign. Note the abrupt stiffening during the sub-zero spell of late January 2019.

> **Figure 1** (`figures/fig1_overview.pdf`). Mode-6 frequency (top) and steel temperature (bottom), with the training, calibration, test, retrofit and post-retrofit periods of the main split shaded.

## 4. Methods
### 4.1 Models
All models predict one mode's frequency *f* from covariates: steel temperature *T*, its trailing 24-hour mean, its 3-hour change, air relative humidity, air temperature, and the hour of day encoded as sine and cosine.

| Model | Description |
|---|---|
| Linear(T), Cubic(T) | least-squares polynomials in *T* |
| GP | Gaussian process, ARD-RBF + white-noise kernel on (*T*, 24 h mean *T*, RH), 1,500-hour random subsample |
| GP + linear kernel | as GP, plus a DotProduct kernel so that it can extrapolate a linear trend |
| GBM | histogram gradient-boosted trees (depth 3, 300 iterations, learning rate 0.05, L2 = 1) on all covariates |
| Linear + GBM | linear trend in *T*, plus a GBM on its residuals (ablation: trend without physics) |
| Physics only | f(T) = f₀ √(1 + a(T − 10 °C)) · (1 + b · s·ln(1 + e^{(T_c − T)/s})), s = 1 °C |
| Physics(no cold) + GBM | physics curve with b = 0, plus a GBM (ablation: no cold term) |
| **Physics + GBM (proposed)** | physics curve plus a GBM on its residuals |

The square-root factor follows from f ∝ √k and k ∝ E(T), with *a* the fractional change of the governing stiffness per °C. The softplus factor represents additional stiffening below *T_c*; its physical cause is not established by this data. Physics parameters are fitted by bounded nonlinear least squares with a soft-L1 loss. The model specification was fixed before the robustness folds (Sec. 4.3) were run, and was not changed afterwards.

### 4.2 Prediction intervals
*Split conformal.* With calibration residuals r_i = |y_i − ŷ_i|, i = 1…n, the half-width is q = r_(⌈(n+1)(1−α)⌉), and the interval is ŷ ± q. Under exchangeability, coverage is at least 1 − α. The GPs are also evaluated with their own Gaussian predictive intervals.

*Adaptive conformal inference (ACI)* updates the working level online as α_{t+1} = α_t + γ(α − err_t), with γ = 0.005 [Gibbs & Candès 2021]. All results use α = 0.1.

### 4.3 Splits
- **Random:** pre-retrofit hours (1 Nov 2018–14 May 2019) are assigned at random to train/calibration/test in proportions 50/20/30, over 5 seeds.
- **Main seasonal split (F2):** train Nov 2018–Jan 2019, calibrate Feb, test 1 Mar–14 May 2019. Maximum steel temperature is 13.6 °C in training and 26.3 °C in test.
- **Robustness folds:**

| Fold | Train | Calibrate | Test | Training T range | Out-of-range test hours (median per mode) |
|---|---|---|---|---|---|
| F1 fwd | Nov–Dec 2018 | Jan 2019 | Feb–Mar 2019 | −0.4 to 13.6 °C | 90 |
| F2 fwd (main) | Nov 2018–Jan 2019 | Feb | Mar–14 May | −3.0 to 13.6 °C | 348 |
| F3 fwd | Nov 2018–Feb 2019 | Mar | Apr–14 May | −3.0 to 18.1 °C | 96 |
| B1 warm→cold | Apr–14 May 2019 | Mar | Nov 2018–Feb 2019 | 1.9 to 26.3 °C | 379 |
| Q1 post-retrofit, warm→cold | 28 Sep–Oct 2019 | Nov | Dec 2019–15 Jan 2020 | 4.7 to 22.7 °C | 191 |

B1 runs backwards in time. It is not a forecast; it tests extrapolation toward cold temperatures, and it is the only fold with sub-zero test hours. Q1 uses the post-retrofit structure, a different structural state.

### 4.4 Metrics
We report coverage, mean interval width, the interval score [Gneiting & Raftery 2007, **TODO verify**] and RMSE. Coverage is also reported by temperature bin and for test hours inside and outside the training temperature range, with 95 % day-block bootstrap intervals (2,000 resamples of whole days).

### 4.5 Change detection
Using the F2 models and their frozen α = 0.1 half-width *q*, the daily statistic is D_d = median_{t∈d}(r_t / q), computed on days with at least 12 valid hours. An alarm is raised when |D_d| > 1. False-alarm rate is computed on the 75 pre-retrofit test days and detection rate on 98–108 post-retrofit days. An output-only PCA baseline uses the five-mode frequency vector with 1 or 2 retained components and a conformal threshold on the residual norm.

## 5. Results
### 5.1 Exchangeable vs seasonal (main split)
**Table 1.** Coverage of nominal 90 % split-conformal intervals. The per-mode columns are for the main seasonal split (F2); the last two columns give the means.

| Model | M3 | M5 | M6 | M9 | M13 | Mean (F2) | Mean (random) |
|---|---|---|---|---|---|---|---|
| Linear(T) | 0.859 | 0.890 | 0.895 | 0.867 | 0.841 | 0.871 | 0.900 |
| Cubic(T) | 0.824 | 0.851 | 0.826 | 0.791 | 0.836 | 0.826 | 0.903 |
| GP | 0.932 | 0.897 | 0.819 | 0.816 | 0.813 | 0.855 | 0.901 |
| GP + linear kernel | 0.861 | 0.893 | 0.875 | 0.835 | 0.772 | 0.847 | 0.900 |
| GBM | 0.906 | 0.896 | 0.874 | 0.886 | 0.888 | 0.890 | 0.903 |
| Linear + GBM | 0.844 | 0.883 | 0.878 | 0.853 | 0.831 | 0.858 | 0.902 |
| Physics only | 0.929 | 0.902 | 0.879 | 0.881 | 0.925 | 0.903 | 0.901 |
| Physics(no cold) + GBM | 0.863 | 0.887 | 0.891 | 0.864 | 0.852 | 0.871 | 0.901 |
| **Physics + GBM** | 0.907 | 0.895 | 0.894 | 0.888 | 0.937 | **0.904** | 0.901 |

Under the random split every model is calibrated to within 0.01 (Fig. 3a). Under the seasonal split, most models under-cover (Fig. 3b).

> **Figure 3** (`fig3_calibration_curves.pdf`). Empirical vs nominal coverage, mean of five modes. (a) Random split. (b) Main seasonal split.

### 5.2 Where coverage fails: outside the training range
Coverage inside the training temperature range is close to nominal for all models, in all folds (`results/summary.md`). Failures occur on out-of-range hours (Fig. 6).

**Table 2.** Coverage on out-of-range test hours for the warm-side forward folds F2 and F3 (10 fold-mode cells).

| Model | Median | Worst |
|---|---|---|
| Cubic(T) | 0.26 | 0.00 |
| GP | 0.36 | 0.01 |
| Linear(T) | 0.56 | 0.01 |
| Linear + GBM | 0.66 | 0.15 |
| GP + linear kernel | 0.69 | 0.04 |
| Physics(no cold) + GBM | 0.75 | 0.34 |
| GBM | 0.77 | 0.59 |
| Physics only | 0.88 | 0.76 |
| **Physics + GBM** | **0.90** | **0.84** |

The finer view for F2 (Fig. 4; Table 3) shows the pattern by temperature. Above 20 °C, which is 8 days of data, the proposed model covers 0.82–0.96 across modes (mode 6: 0.82, 95 % CI [0.74, 0.90]). The GP covers 0.01–0.06 on modes 6, 9 and 13, and the physics-free GBM 0.42–0.59 on those modes.

**Table 3.** Main split, coverage for test hours above 20 °C (68 hours over 8 days per mode).

| Model | M3 | M5 | M6 | M9 | M13 |
|---|---|---|---|---|---|
| GP | 0.96 | 0.90 | 0.04 | 0.06 | 0.01 |
| GP + linear kernel | 0.25 | 0.89 | 0.29 | 0.28 | 0.00 |
| GBM | 0.88 | 0.92 | 0.49 | 0.59 | 0.42 |
| Linear + GBM | 0.12 | 0.92 | 0.25 | 0.63 | 0.00 |
| Physics(no cold) + GBM | 0.38 | 0.92 | 0.57 | 0.71 | 0.01 |
| **Physics + GBM** | 0.93 | 0.92 | 0.82 | 0.91 | 0.96 |

> **Figure 4** (`fig4_coverage_by_temperature_all_modes.pdf`). Main split: coverage by temperature bin, per mode, with day-block bootstrap CIs.
> **Figure 6** (`fig6_robustness_outside_range.pdf`). Coverage on out-of-range test hours for all folds, modes and models.

### 5.2b Systematic rolling-origin windows and paired tests
To remove the possibility that the designed folds were favourable by chance, we evaluated 22 rolling windows: an 8-week training block, a 3-week calibration block and a 4-week test block, stepped weekly across the pre-retrofit season. Eleven windows run forward in time and eleven backward. Every window position is used.

**Table 2b.** Rolling windows, coverage on out-of-range test hours, cells with at least 24 hours (`results/summary.md`).

| Model | Warm side: median | Warm side: worst | Warm side: share of cells ≥ 0.85 | Cold side: median |
|---|---|---|---|---|
| Cubic(T) | 0.323 | 0.000 | 0.16 | 0.611 |
| Linear(T) | 0.674 | 0.000 | 0.31 | 0.529 |
| Linear + GBM | 0.771 | 0.026 | 0.36 | 0.573 |
| GP | 0.791 | 0.010 | 0.44 | 0.622 |
| GP + linear kernel | 0.824 | 0.011 | 0.38 | 0.517 |
| Physics(no cold) + GBM | 0.833 | 0.141 | 0.42 | 0.578 |
| GBM | 0.884 | 0.578 | 0.55 | 0.502 |
| Physics only | 0.903 | 0.512 | 0.65 | 0.578 |
| **Physics + GBM** | **0.907** | **0.674** | **0.80** | 0.562 |

The warm side has 55 cells per model and the cold side 35.

Paired comparisons use exactly the same test hours for both models. The 95 % CI resamples calendar days, keeping together all modes and overlapping windows that share a day, and p-values are Holm-adjusted over the 8 comparators (`results/paired_tests.csv`, Fig. 8).
- **Warm side** (3,259 hours, 28 days): the proposed model's pooled coverage is 0.895.
  - It exceeds Physics only by +0.031 [0.009, 0.059] and GBM by +0.084 [0.051, 0.115].
  - It exceeds Physics(no cold) + GBM by +0.161, GP + linear kernel by +0.202, Linear + GBM by +0.239, GP by +0.361, Linear by +0.378 and Cubic by +0.614.
  - All adjusted p ≤ 0.002.
- **Cold side** (5,244 hours): no comparator differs significantly, except Physics only, which is 0.019 *better* [0.004, 0.032].
- **Inside the training range** (63,778 hours): all differences are below 0.01.

The designed folds F2 + F3 give the same ordering and similar effect sizes. In fold Q1 the proposed model is significantly worse than Linear, GBM, Linear + GBM, GP + linear kernel and Physics(no cold) + GBM, by 0.11–0.13.

> **Figure 7** (`fig7_rolling_origin.pdf`). Rolling-origin coverage on out-of-range hours: (a) warm side, (b) cold side.
> **Figure 8** (`fig8_paired_differences.pdf`). Paired coverage differences with 95 % day-block bootstrap CIs.

### 5.2c Reviewer checks: width and tuning
**Width.** On warm-side out-of-range hours of the forward rolling windows, the proposed model has the narrowest median interval of all nine models. The others are 0.5 % (GBM) to 26 % (Linear) wider. Its higher coverage therefore does not come from wider intervals.

**Tuning.** We re-ran the forward rolling windows for 9 residual-learner settings (maximum depth 2, 3 or 5 × learning rate 0.02, 0.05 or 0.1), applied identically to the proposed model and to the physics-free GBM.

| | Warm-side pooled coverage (range over 9 settings) |
|---|---|
| Proposed model | 0.883–0.899 |
| GBM | 0.770–0.826 |

Every setting of the proposed model therefore exceeds every setting of the GBM. Varying the hinge width *s* gives 0.907 (0.5 °C), 0.895 (1 °C, default) and 0.838 (2 °C). The default was fixed before these runs and is not the best-performing value. The degradation at *s* = 2 °C is consistent with the mechanism of Sec. 5.3: a wider hinge spreads the cold-regime stiffening into the warm-side slope. Inside the training range, coverage is 0.908–0.918 for all settings of both models (`results/sensitivity.csv`).

### 5.3 Why: the cold term de-confounds the warm-side slope
The two ablations isolate the source of the gain. Adding a linear trend to the GBM (Linear + GBM) does *not* help: median 0.66 on warm-side out-of-range hours, versus 0.77 for the GBM alone. Removing the cold term from the physics model drops it from 0.90 to 0.75.

The mechanism is visible in the fitted slopes (Table 4). Training data include a sub-zero spell during which frequencies jump upward (Fig. 1, Fig. 2). A single trend, linear or square-root, absorbs this jump as a steeper slope, which then over-predicts the decrease of frequency in spring. The cold term absorbs the jump separately. Its slope is shallower than the one fitted on warm hours alone, but its error is smaller in the direction that matters for extrapolating to warmer temperatures.

**Table 4.** Fractional temperature slope (df/f per °C) fitted on F2 training data. The spring-test slope is shown for reference only; no model uses it.

| Mode | Physics (full) | Physics (no cold) | Linear (all hours) | Linear (T > 5 °C hours) | Spring test (reference) |
|---|---|---|---|---|---|
| 3 | +2.1e-4 | −3.9e-4 | −6.4e-4 | +9.7e-5 | +1.4e-4 |
| 5 | −9.9e-5 | −3.8e-4 | −3.8e-4 | −2.8e-4 | −5.5e-5 |
| 6 | −1.3e-4 | −4.7e-4 | −6.0e-4 | −2.5e-4 | −2.2e-4 |
| 9 | −1.3e-4 | −2.5e-4 | −2.7e-4 | −1.7e-4 | −1.6e-4 |
| 13 | −2.9e-4 | −1.1e-3 | −1.6e-3 | −4.7e-4 | −4.7e-4 |

> **Figure 2** (`fig2_physics_fit.pdf`). Mode-6 frequency vs steel temperature with the physics curve fitted on training data only.

**[TODO: compare these slopes with a handbook value of (dE/dT)/E for structural steel. Note that (df/f)/°C = ½ (dE/dT)/E only if E is the sole temperature-dependent stiffness. Discuss what else could contribute: bearings, ballast, rail continuity, thermal expansion and the resulting axial forces.]**

### 5.4 Limits: cold-side extrapolation and an unidentifiable cold term
In fold B1 (train spring, test winter), median out-of-range coverage is 0.42–0.64 for every model, the proposed one included (0.62). On the hours below 0 °C (101–131 per mode, over 11 days), coverage is 0.02–0.13 on mode 6 and 0.00–0.01 on mode 13 for all models. A cold-stiffening term whose threshold and magnitude were never observed cannot be fitted, so physics structure alone does not rescue extrapolation into an unseen mechanism.

In fold Q1 (post-retrofit, training minimum 4.7 °C), the proposed model reaches 0.86 median out-of-range coverage, below Linear, GBM, GP and Physics(no cold) + GBM (0.90–0.92). We attribute this to the cold-term parameters being unidentifiable without sub-freezing training data. A natural remedy is to carry *T_c* and *b* across seasons as a prior. We have not tuned the model on these folds.

### 5.5 Sharpness and the GP's own uncertainty
On the main split, the proposed model has the best or near-best interval score in every mode: 0.0274, 0.0648, 0.0200, 0.0184 and 0.0515 Hz for modes 3–13. The GP's own Bayesian intervals cover 0.83–0.99, better than its conformal intervals, because the GP widens its predictive variance when extrapolating. They are much wider, though (mode 13: 0.094 vs 0.044 Hz for the proposed model), and their interval score is worse on modes 6, 9 and 13 (0.026, 0.024 and 0.097 Hz) and about equal on modes 3 and 5.

### 5.6 Adaptive conformal inference
ACI gives coverage between 0.889 and 0.907 for every model and mode on the main split, at the cost of wider intervals for weaker models. ACI uses the true frequency at every hour to adjust its level. During a real structural change it would widen until the change is absorbed, which defeats detection. ACI provides calibrated *forecasts*; fixed, physics-guided intervals provide *alarms*.

### 5.7 Physical parameters
Fitted on main-split training data only, with 95 % day-block bootstrap CIs (300 resamples; `results/physics_parameter_ci.csv`):

| Mode | (df/f)/°C = ½a | 95 % CI | T_c (°C) | 95 % CI |
|---|---|---|---|---|
| 3 | +2.1e-4 | [+4.7e-5, +3.5e-4] | 1.11 | [−0.5, 2.2] |
| 5 | −9.9e-5 | [−2.3e-4, +3.1e-5] | 1.17 | [−0.9, 2.4] |
| 6 | −1.3e-4 | [−1.9e-4, −5.0e-5] | 1.36 | [−0.6, 2.4] |
| 9 | −1.3e-4 | [−1.5e-4, −1.0e-4] | 1.25 | [−0.4, 2.0] |
| 13 | −2.9e-4 | [−4.0e-4, −1.4e-4] | 1.18 | [−1.4, 2.4] |

- **Sign:** the temperature coefficient is negative, as thermal softening of steel predicts, with CIs excluding zero for modes 6, 9 and 13. It is indistinguishable from zero for mode 5.
- **Mode 3:** the coefficient is positive with a CI excluding zero, which steel softening alone cannot explain **[TODO: discuss candidate mechanisms]**.
- **Freezing threshold:** the *T_c* CIs overlap across modes and all include 0 °C. That is consistent with a freezing-related mechanism, but does not establish it. The same regime appears as a 2–3 °C kink in Wittenberg et al. (2025a).
- **Not validated out of sample:** no test hour of the main split fell below 0 °C, so the cold term is not validated out-of-sample there (see Sec. 5.4).

**[TODO: compare ½a with a handbook value of (dE/dT)/E for structural steel.]**

### 5.8 Retrofit detection
Median post-retrofit residuals (Physics + GBM) are +0.061 Hz on mode 6 (+2.1 %), −0.031 Hz on mode 9 and +0.097 Hz on mode 13, and below 0.01 Hz on modes 3 and 5. For modes 6, 9 and 13, every model alarms on 100 % of post-retrofit days.

False-alarm rates on the 75 pre-retrofit test days:

| Model | Modes 3, 5, 6, 9, 13 |
|---|---|
| **Physics + GBM** | 0, 0, 0, 1.3 %, 0 |
| GBM | 0, 0, 0, 1.3 %, 2.7 % |
| GP | up to 13.3 % |
| GP + linear kernel | up to 18.7 % |
| Cubic | up to 10.7 % |

The output-only PCA baseline (k = 1) detects every post-retrofit day with 1.3 % false alarms. The retrofit is large relative to the uncertainty bands, so it does not rank the methods by detection power (Fig. 5). **[TODO: minimum-detectable-shift study.]**

> **Figure 5** (`fig5_detection_mode6.pdf`). Daily median of residual/q for mode 6, before and after the retrofit.

## 6. Discussion
- **What physics buys.** Conformal coverage under seasonal shift is governed by how a model extrapolates. A small physics prior helps when the extrapolation follows a mechanism the training data constrain: here, the smooth thermo-elastic trend, once separated from the freezing regime. It does not help when the test conditions activate a mechanism the training data never showed: here, sub-zero stiffening after a warm-only training period. The ablations show that a *structurally correct* decomposition matters more than having "a trend": Linear + GBM was worse than GBM alone.
- **Practical recommendation.** Report coverage conditional on the environmental covariate, and specifically on out-of-range conditions. Average coverage of 0.89 for the GBM hid out-of-range coverage as low as 0.59 on the warm side.
- **ACI versus fixed intervals.** ACI's guarantee is attractive, but in monitoring the point is *not* to adapt to the thing being detected. A combination (ACI on a slow time scale, physics-guided fixed intervals for alarms) is a direction for future work.

## 7. Limitations
1. **No sub-zero training → test transfer:** the cold regime is the open problem. A cross-season prior on *T_c*, *b* is untested.
2. **Small extrapolation samples:** median 90–379 out-of-range hours per fold and mode; the > 20 °C bin of the main split is 8 days.
3. **One season of rolling windows:** the 22 rolling windows span one pre-retrofit season and overlap heavily. They show that the result is not an artefact of fold choice, but they are not independent replications.
4. **One bridge;** the post-retrofit fold gives a second structural state, not a second structure. LUMO is a candidate second structure.
5. **The retrofit is a stiffening intervention, not damage.**
6. **Inherited mode tracking:** the analysis relies on the dataset authors' mode identification and tracking.
7. **Autocorrelation:** hourly residuals are autocorrelated. Day-level statistics and block bootstrap mitigate but do not remove this.

## 8. Conclusion
On KW51, split-conformal temperature compensation is well calibrated when data are exchangeable, but it can lose most of its coverage at temperatures outside the training range. A four-parameter thermo-elastic prior with a separate cold-stiffening term kept warm-side out-of-range coverage near nominal (median 0.907 over 55 rolling window×mode cells; pooled 0.895). It was significantly better than every alternative, including the best physics-free model (pooled 0.810; difference +0.084 [0.051, 0.115]). Ablations show that the gain comes from separating the freezing regime from the thermo-elastic trend. No model, including ours, extrapolated into an unseen freezing regime. Encoding physics helps conformal intervals extrapolate along known mechanisms; it is not a substitute for data on unknown ones.

## Data and code availability
Data: Zenodo 10.5281/zenodo.3745914 (CC-BY-NC-SA-4.0). Code: [GitHub URL, TODO]. `scripts/run_experiments.py`, `scripts/run_robustness.py` and `scripts/make_figures.py` regenerate every table and figure.

## Acknowledgments and AI-use disclosure
**[TODO: write honestly.]** Example: "Code scaffolding, initial experiments and an initial draft of this manuscript were produced with the assistance of an AI system (Claude, Anthropic). The author verified the data, reran all experiments, performed [list your tasks], and rewrote the text." Check each venue's policy. The KW51 data were collected and published by K. Maes and G. Lombaert (KU Leuven).

## References (all verified to exist in a literature search on 2026-11-07 unless marked)
1. Angelopoulos, A.N., Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
2. Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems 34*. arXiv:2106.00170.
3. Gneiting, T., Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *J. Am. Stat. Assoc.* 102(477). **[not re-verified]**
4. Hinchliffe, Jones, Cross (2025). Parameter estimation by optimization of interpretable hyperparameters with physics-informed Gaussian processes. *Proc. IWSHM 2025*, 1752–1759. **[add author initials from the paper]**
5. Maes, K., Lombaert, G. (2021). Monitoring railway bridge KW51 before, during, and after retrofitting. *J. Bridge Eng.* 26(3). doi:10.1061/(ASCE)BE.1943-5592.0001668.
6. Maes, K., Lombaert, G. (2020). Monitoring data for railway bridge KW51 in Leuven, Belgium, before, during, and after retrofitting (v1.0). Zenodo. doi:10.5281/zenodo.3745914.
7. Maes, K., Van Meerbeeck, L., Reynders, E., Lombaert, G. (2022). Validation of vibration-based structural health monitoring on retrofitted railway bridge KW51. *Mech. Syst. Signal Process.* 165, 108380. doi:10.1016/j.ymssp.2021.108380.
8. Neumann, L., Wittenberg, P., Gertheiss, J. (2025). Confidence intervals for conditional covariances of natural frequencies. arXiv:2509.26348.
9. Neumann, L., Wittenberg, P., Gertheiss, J. (2026a). Higher-order multivariate environmental influences in structural health monitoring. arXiv:2603.22950.
10. Neumann, L., Wittenberg, P., Gertheiss, J. (2026b). Removal of multivariate environmental influences in structural health monitoring through conditional covariances and supervised learning. arXiv:2606.05026.
11. Peeters, B., De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. *Earthq. Eng. Struct. Dyn.* 30(2).
12. Wittenberg, P., Mendler, A., Knoth, S., Gertheiss, J. (2025a). Multivariate long-term profile monitoring with application to the KW51 railway bridge. arXiv:2506.20295.
13. Wittenberg, P., Neumann, L., Mendler, A., Gertheiss, J. (2025b). Covariate-adjusted functional data analysis for structural health monitoring. *Data-Centric Engineering* 6, e27. arXiv:2408.02106.

---

## Where to submit (honest assessment of the *current* results)
| Venue | Fit now | What would change it |
|---|---|---|
| GARSEF → Texas state → ISEF (Engineering Mechanics, or Systems Software / Computational) | **Good.** A clear question, a real bridge, an ablation-backed mechanism, and an honest negative result on the cold side | Add the minimum-detectable-shift study and do tasks 1–3 yourself, so you can defend every number |
| NHSJS | **Good** after the TODOs are resolved | — |
| IEEE Trans. Instrumentation & Measurement | **Still not yet**, but closer: 5 folds and ablations now, though still one bridge and small out-of-range samples | A second structure (LUMO), minimum-detectable-shift curves, a cross-season prior that fixes the cold side, and a stronger novelty argument against the Gertheiss group's KW51 work. Realistically you'd need a faculty co-author |
| Review of Scientific Instruments | **Poor fit** (no new instrument or technique) | — |
| More realistic stretch venues | *Data-Centric Engineering*, *J. Civil Structural Health Monitoring*, or an IWSHM/EWSHM conference paper | Same as TIM, with a lower bar |
