# Do conformal error bars survive an unseen season? Physics-guided temperature compensation of bridge natural frequencies on the KW51 railway bridge

*[Your name]¹ — ¹[High school], Austin, Texas, USA*
*DRAFT v0.1, 2026-11-07. Every number below comes from `results/` in this repository at the commit that contains this file. Items marked **[TODO]** need your own work or verification before submission.*

---

## Abstract
Vibration-based structural health monitoring infers changes in stiffness from shifts in natural frequencies, but temperature shifts those frequencies by a comparable amount. Compensating for temperature therefore requires prediction intervals that remain trustworthy when operating conditions change. Split conformal prediction offers finite-sample coverage guarantees, but only for exchangeable data, an assumption that seasonal monitoring data violate. Using publicly available hourly natural frequencies of the KW51 steel railway bridge (Leuven, Belgium), we evaluate how six temperature-compensation models, all wrapped in the same split-conformal procedure, keep their nominal 90 % coverage when trained in winter and tested in spring at temperatures up to 12.7 °C above the training range. Under a random (exchangeable) split, all models achieve 0.900–0.904 mean coverage across five modes. Under the chronological split, mean coverage falls to 0.79 for a Gaussian process and 0.83 for a cubic polynomial. A hybrid model combining a four-parameter thermo-elastic stiffness curve with a gradient-boosted residual learner retains 0.91, and 0.87–0.96 in every mode for test temperatures above 20 °C. That last result rests on 8 days of data. A physics-free boosted-tree model reaches similar average coverage (0.90) but falls to 0.51–0.79 above 20 °C in four of five modes. Adaptive conformal inference restores ≈0.90 coverage for every model, but would also adapt to genuine structural change. With intervals frozen after calibration, the physics-guided model flags the 2019 retrofit on all post-retrofit days for three of five modes with 0–1.3 % false-alarm days. **[TODO: update with the minimum-detectable-shift and rolling-origin results.]**

**Keywords:** structural health monitoring; conformal prediction; temperature compensation; physics-guided machine learning; uncertainty quantification; distribution shift

## 1. Introduction
Natural frequencies are attractive health indicators: they are global, can be identified from ambient vibration, and fall when stiffness is lost. Their major weakness is sensitivity to environmental and operational variability. In the Z24-bridge study, seasonal temperature changes produced frequency changes that had to be removed before damage could be distinguished [Peeters & De Roeck 2001]. The usual remedy is to regress frequency on temperature (or to remove environmental variation with output-only methods such as PCA) and to monitor the residuals [Maes et al. 2022].

An alarm threshold on residuals is an uncertainty statement: "this frequency is inconsistent with the bridge's healthy state at this temperature." If the threshold is too tight, there are false alarms; if it is too loose, damage is missed. Conformal prediction [Angelopoulos & Bates 2021] gives a principled threshold with a coverage guarantee under exchangeability. Monitoring data are serially correlated and seasonal, so a model calibrated in one season is used in another. Whether conformal coverage survives this shift is an empirical question. This paper asks it directly.

**Contributions.**
1. A controlled comparison of six temperature-compensation models on KW51, all using the same split-conformal wrapper, under exchangeable (random) and seasonal (chronological) splits.
2. Coverage *conditional on temperature*, with day-block bootstrap confidence intervals. It shows where coverage fails: at temperatures outside the training range.
3. A four-parameter physics-guided model (thermo-elastic stiffness plus cold stiffening), combined with a residual learner, that keeps near-nominal coverage under seasonal extrapolation.
4. An honest comparison with adaptive conformal inference, and the trade-off between restoring coverage and masking structural change.
5. Fully reproducible code (MD5-checked download, seeded experiments).

**[TODO: novelty statement. Run the Google Scholar searches from README task 3, then write one sentence on what is new relative to Wittenberg et al. (2025) and Neumann et al. (2025–2026), and the Cross group's physics-informed GP work. Do not write "first" unless you are sure.]**

## 2. Related work
**Environmental compensation on KW51.** Maes & Lombaert (2021) describe the monitoring campaign and dataset. Maes et al. (2022) remove environmental variation with robust PCA and two black-box models, then show that the retrofit can be detected. Wittenberg et al. (2025a) model the nonlinear temperature effect with functional-data regression and multivariate EWMA control charts, and report a kink in the frequency–temperature relation between 2 and 3 °C. Wittenberg et al. (2025b) propose covariate-adjusted functional data analysis. Neumann et al. (2025) construct bootstrap confidence intervals for *conditional covariances* of natural frequencies and evaluate their coverage in simulation. Neumann et al. (2026a, 2026b) study higher-order multivariate environmental influences and their removal. To our reading, none of these evaluates the coverage of prediction intervals for the frequencies themselves under seasonal shift **[TODO: confirm by reading the full texts]**.

**Physics-informed uncertainty in SHM.** Physics-informed Gaussian-process kernels derived from oscillator dynamics have been used for interpretable parameter estimation [Hinchliffe et al. 2025]. Related kernels encode partial physical insight and change points [Jones et al. 2024; Pitchforth et al. 2024, **TODO: get exact citations**].

**Conformal prediction in SHM.** Conformal methods have been used for anomaly detection on fibre-optic strain data from a railway bridge [**TODO: full citation of the Annals of Applied Statistics 15(3) paper "Unsupervised streaming anomaly detection for instrumented infrastructure"**] and for damage identification with graph neural networks [**TODO: full citation, Adv. Eng. Informatics 2025, "A hybrid data-physics framework with conformal GNN…"**]. Adaptive conformal inference [Gibbs & Candès 2021] targets distribution shift in general.

## 3. Data
The KW51 bridge is a steel bowstring railway bridge in Leuven, Belgium **[TODO: add span and structural description from Maes & Lombaert 2021]**. It was monitored from October 2018 to January 2020 and retrofitted between 15 May and 27 September 2019 **[TODO: verify dates]** to correct a construction error in the connections of the diagonals. We use the dataset authors' tracked modal parameters (`trackedmodes.mat`, Zenodo 10.5281/zenodo.3745914). These comprise 11,328 hourly records of 14 natural frequencies and damping ratios identified with reference-based covariance-driven stochastic subspace identification, together with steel surface temperature, relative humidity and weather-station variables. Steel temperature is missing throughout October 2018 and most of November 2018. We analyse modes 3, 5, 6, 9 and 13, which have less than 15 % missing frequencies before the retrofit. Median pre-retrofit frequencies are 1.893, 2.580, 2.923, 4.109 and 6.320 Hz. Fig. 1 shows mode 6 and steel temperature over the whole campaign.

> **Figure 1** (`figures/fig1_overview.pdf`). Mode-6 frequency (top) and steel temperature (bottom), with the training, calibration, test, retrofit and post-retrofit periods shaded.

## 4. Methods
### 4.1 Models
All models predict one mode's frequency *f* from covariates *x*. Covariates are steel temperature *T*, its trailing 24-hour mean, its 3-hour change, deck relative humidity, air temperature, and the hour of day encoded as sine and cosine.

- **Linear(T)** and **Cubic(T)**: least-squares polynomials in *T*.
- **GP**: Gaussian-process regression with an ARD-RBF plus white-noise kernel on (*T*, 24 h mean *T*, RH), trained on a random subsample of 1,500 hours.
- **GBM (no physics)**: histogram gradient-boosted trees (depth 3, 300 iterations, learning rate 0.05, L2 = 1) on all covariates.
- **Physics only**:
  f(T) = f₀ √(1 + a(T − 10 °C)) · (1 + b · s·ln(1 + e^{(T_c − T)/s})), with s = 1 °C.
  The square-root factor follows from f ∝ √k and k ∝ E(T), with *a* the fractional change of the governing stiffness per °C. The softplus factor represents additional stiffening below *T_c*. Parameters are fitted by bounded nonlinear least squares with a soft-L1 loss.
- **Physics + GBM (proposed)**: the physics curve plus a GBM, with the same settings, fitted to the physics residuals.

### 4.2 Prediction intervals
*Split conformal.* With calibration residuals r_i = |y_i − ŷ_i|, i = 1…n, the half-width is q = r_(⌈(n+1)(1−α)⌉), and the interval is ŷ ± q. Under exchangeability, coverage is at least 1 − α [Angelopoulos & Bates 2021]. The GP is also evaluated with its own Gaussian predictive intervals μ ± z_{1−α/2} σ.

*Adaptive conformal inference (ACI)* [Gibbs & Candès 2021] updates the working level online as α_{t+1} = α_t + γ(α − err_t), with γ = 0.005, using the fixed calibration scores.

### 4.3 Splits
- **Random:** pre-retrofit hours (1 Nov 2018–14 May 2019) are assigned at random to train/calibration/test in proportions 50/20/30, over 5 seeds.
- **Chronological:** train 1 Nov 2018–31 Jan 2019, calibrate 1–28 Feb 2019, test 1 Mar–14 May 2019. Maximum steel temperature is 13.6 °C in training, 18.1 °C in calibration and 26.3 °C in test. About 1,600–1,700 training hours, 650–670 calibration hours and 1,700–1,800 test hours are available per mode.

### 4.4 Metrics
We report coverage, mean interval width, the interval score [Gneiting & Raftery 2007, **TODO verify**] and RMSE. Coverage conditional on temperature bin (0–5, 5–10, 10–15, 15–20, > 20 °C) is reported with 95 % day-block bootstrap intervals (2,000 resamples of whole days).

### 4.5 Change detection
With the chronological-split models and their α = 0.1 split-conformal half-width *q* frozen, the daily statistic is D_d = median_{t∈d}(r_t / q), computed on days with at least 12 valid hours. An alarm is raised when |D_d| > 1. False-alarm rate is computed on pre-retrofit test days (75) and detection rate on post-retrofit days (28 Sep 2019–15 Jan 2020; 98–108 days depending on mode). An output-only PCA baseline uses the five-mode frequency vector: standardization, 1 or 2 retained components, and the residual norm with a conformal threshold.

## 5. Results
### 5.1 Exchangeable vs seasonal splits
**Table 1.** Coverage of nominal 90 % split-conformal intervals (chronological split; random-split means in the last column).

| Model | M3 | M5 | M6 | M9 | M13 | Mean (chrono) | Mean (random) |
|---|---|---|---|---|---|---|---|
| Linear(T) | 0.859 | 0.890 | 0.895 | 0.867 | 0.841 | 0.870 | 0.900 |
| Cubic(T) | 0.824 | 0.851 | 0.826 | 0.791 | 0.836 | 0.826 | 0.903 |
| GP (conformal) | 0.867 | 0.899 | 0.724 | 0.712 | 0.738 | 0.788 | 0.901 |
| GBM (no physics) | 0.897 | 0.888 | 0.897 | 0.908 | 0.916 | 0.901 | 0.901 |
| Physics only | 0.929 | 0.902 | 0.879 | 0.881 | 0.925 | 0.903 | 0.901 |
| **Physics + GBM** | 0.912 | 0.884 | 0.917 | 0.901 | 0.922 | **0.907** | 0.904 |

Under the random split every model is calibrated to within 0.01 (Fig. 3a). Under the chronological split, coverage curves for Cubic and GP lie well below the diagonal at all levels (Fig. 3b). At 90 % nominal, the GP's own Bayesian intervals cover only 0.72–0.75 on modes 6 and 9 (`results/coverage_all.csv`).

> **Figure 3** (`fig3_calibration_curves.pdf`). Empirical vs nominal coverage, mean of five modes. (a) Random split. (b) Chronological split.

### 5.2 Where coverage fails: conditional on temperature
**Table 2.** Coverage for test hours with T > 20 °C (68 hours, 8 days per mode). The physics-guided model's 95 % block-bootstrap CIs are given in brackets.

| Model | M3 | M5 | M6 | M9 | M13 |
|---|---|---|---|---|---|
| Linear(T) | 0.09 | 0.82 | 0.07 | 0.59 | 0.00 |
| Cubic(T) | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 |
| GP | 0.96 | 0.92 | 0.07 | 0.06 | 0.01 |
| GBM (no physics) | 0.79 | 0.90 | 0.66 | 0.75 | 0.51 |
| **Physics + GBM** | 0.93 [0.87, 0.97] | 0.92 [0.86, 0.97] | 0.87 [0.79, 0.94] | 0.91 [0.84, 0.97] | 0.96 [0.86, 1.00] |

Coverage in the 0–15 °C bins, which training covers, is close to nominal for all models (Fig. 4). The failures appear only at temperatures above the training range, and each has a clear mechanism:
- Tree ensembles predict a constant beyond their training range.
- RBF-kernel GPs revert to the training mean, which is pulled up by the cold-weather stiffening in late January.
- The cubic polynomial extrapolates its curvature.
- The physics curve keeps extrapolating a gentle monotone decline (Fig. 2), which matches the spring data.

> **Figure 2** (`fig2_physics_fit.pdf`). Mode-6 frequency vs steel temperature with the physics curve fitted on training data only.
> **Figure 4** (`fig4_coverage_by_temperature_all_modes.pdf`). Coverage by temperature bin for each mode, with 95 % day-block bootstrap CIs.

### 5.3 Sharpness
Physics + GBM has the lowest or near-lowest interval score in every mode: 0.0272, 0.0645, 0.0190, 0.0187 and 0.0525 Hz for modes 3–13. Physics-only gives 0.0275, 0.0640, 0.0215, 0.0183 and 0.0578, and GBM 0.0285, 0.0653, 0.0192, 0.0189 and 0.0570. Interval widths are similar (e.g. mode 6: 0.0154 vs 0.0147 Hz for GBM). **The average advantage over a strong physics-free learner is therefore small. The benefit is concentrated in the extrapolation regime.**

### 5.4 Adaptive conformal inference
With γ = 0.005, ACI gives coverage between 0.889 and 0.907 for every model and mode (`results/aci.csv`), including the GP and the cubic, at the cost of wider intervals for the weaker models (e.g. GP mode 9: 0.020 → 0.026 Hz). ACI uses the true frequency at every hour to adjust its level. During a real structural change it would widen until the change is absorbed, which defeats detection. ACI and physics-guided compensation therefore address different needs: ACI provides calibrated *forecasts*, and fixed physics-guided intervals provide *alarms*.

### 5.5 Physical parameters
Fitted on training data only (`results/physics_parameters.csv`):

| Mode | a (1/°C) | ½a = (df/f)/°C | T_c (°C) | b |
|---|---|---|---|---|
| 3 | +4.2e-4 | +2.1e-4 | 1.11 | 6.3e-3 |
| 5 | −2.0e-4 | −1.0e-4 | 1.17 | 2.2e-3 |
| 6 | −2.6e-4 | −1.3e-4 | 1.36 | 3.2e-3 |
| 9 | −2.6e-4 | −1.3e-4 | 1.25 | 1.0e-3 |
| 13 | −5.8e-4 | −2.9e-4 | 1.18 | 9.7e-3 |

The sign is negative, as expected for thermal softening of steel, in four of five modes. Mode 3 has an unphysical positive coefficient, which suggests that another temperature-dependent mechanism dominates for that mode (e.g. boundary conditions or the track–ballast system) **[TODO: compare a with handbook dE/dT/E; discuss]**. The consistent T_c ≈ 1.1–1.4 °C across modes agrees in character with the kink reported by Wittenberg et al. (2025a), who place it at 2–3 °C, and differs in value. No test hour fell below 0 °C, so the cold-stiffening term is **not validated out-of-sample**.

### 5.6 Retrofit detection
The retrofit changed median post-retrofit residuals by +0.061 Hz on mode 6 (≈ +2.1 %), −0.031 Hz on mode 9 and +0.098 Hz on mode 13, and by less than 0.01 Hz on modes 3 and 5 (Physics + GBM). For modes 6, 9 and 13, all models except the GP on mode 13 (0.45) alarm on 100 % of post-retrofit days. False-alarm rates on pre-retrofit test days:
- **Physics + GBM:** 0, 0, 0, 1.3 % and 0 % (modes 3, 5, 6, 9, 13).
- **Linear:** 6.7 % on mode 6 and 9.3 % on mode 13.
- **Cubic:** 8–11 %.
- **GP:** 21–23 % on modes 6 and 13.

The output-only PCA baseline (k = 1) also detects every post-retrofit day, with a 1.3 % false-alarm rate. Mode 6's post-retrofit daily statistic is about 8 for Physics + GBM against about 3 for the GP (Fig. 5). The retrofit is a large change relative to the uncertainty bands, so it does not discriminate between methods by detection power. **[TODO: minimum-detectable-shift study (README task 4).]**

> **Figure 5** (`fig5_detection_mode6.pdf`). Daily median of residual/q for mode 6, before and after the retrofit.

## 6. Discussion
- **Why the hybrid works.** The headline result is not that physics makes intervals narrower in general; it barely does. The result is that **conformal coverage under seasonal shift is governed by how a model extrapolates**. A four-parameter physics prior makes that extrapolation explicit, interpretable and checkable against materials data. A purely data-driven learner can match average coverage but fails exactly where monitoring matters: under conditions not yet seen.
- **Practical recommendation.** Report coverage *conditional on the environmental covariate*, not just on average. Average coverage of 0.90 hid failures of 0.51–0.79 above 20 °C for the physics-free GBM.
- **ACI versus fixed intervals.** ACI's guarantee is attractive, but in monitoring the point is *not* to adapt to the thing being detected. A combination (ACI on a slow time scale, physics-guided fixed intervals for alarms) is a direction for future work.

## 7. Limitations
1. There is one chronological split. Rolling-origin evaluation is needed **[TODO]**.
2. The extrapolation regime above 20 °C contains 8 days, and the CIs reflect this.
3. The test period contains no sub-zero temperatures.
4. This is a single bridge. Validation on another structure (e.g. the LUMO lattice tower, whose data are public) would strengthen the claims.
5. The retrofit is a stiffening intervention, not damage.
6. The analysis relies on the dataset authors' mode identification and tracking.
7. Hourly residuals are autocorrelated. Day-level statistics and block bootstrap mitigate but do not remove this.

## 8. Conclusion
On KW51, split-conformal temperature compensation is well calibrated when data are exchangeable but can lose most of its coverage in a season with unseen temperatures. Encoding a simple thermo-elastic stiffness law, with a cold-stiffening term, before learning the residual kept coverage near nominal across all observed temperature bins and five modes. It also gave low false-alarm rates when the intervals were frozen for change detection. The gains are concentrated in the extrapolation regime, and their size should be confirmed with more data, more structures, and a minimum-detectable-shift analysis.

## Data and code availability
Data: Zenodo 10.5281/zenodo.3745914 (CC-BY-NC-SA-4.0). Code: [GitHub URL, TODO]. All results are regenerated by `scripts/run_experiments.py` and `scripts/make_figures.py`.

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
| GARSEF → Texas state → ISEF (Engineering Mechanics, or Systems Software / Computational) | **Good.** A clear question, a real bridge, a negative result for standard methods, a physical explanation | Add the minimum-detectable-shift study and do tasks 1–3 yourself, so you can defend every number |
| NHSJS | **Good** after the TODOs are resolved | — |
| IEEE Trans. Instrumentation & Measurement | **Not yet.** One dataset, one split, small extrapolation sample, modest average gain | Rolling-origin evaluation, a second structure, minimum-detectable-shift curves, and a stronger novelty argument against the Gertheiss group's KW51 work. Realistically you'd need a faculty co-author |
| Review of Scientific Instruments | **Poor fit** (no new instrument or technique) | — |
| More realistic stretch venues | *Data-Centric Engineering*, *J. Civil Structural Health Monitoring*, or an IWSHM/EWSHM conference paper | Same as TIM, with a lower bar |
