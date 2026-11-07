# Step 1–2: Literature survey, research questions, and dataset check

*Prepared 2026-11-07. Everything below was checked during this session. A paper is marked **[verified]** only if its title, venue, and authors (or a stable ID such as a DOI or arXiv number) turned up in an independent index or repository. Anything I could not confirm is labeled as unverified, so please read the abstracts yourself before you cite.*

---

## 1. Literature survey (what exists, by theme)

### A. Temperature effects on bridge natural frequencies (structural health monitoring, SHM)
| # | Paper | Status | Why it matters to you |
|---|---|---|---|
| A1 | Peeters, B. & De Roeck, G. (2001). *One-year monitoring of the Z24-Bridge: environmental effects versus damage events.* Earthquake Eng. & Structural Dynamics 30(2). | [verified] | Classic paper: temperature changes natural frequencies by about as much as damage does, so you must "compensate" before you can detect damage. |
| A2 | Maes, K. & Lombaert, G. (2021). *Monitoring railway bridge KW51 before, during, and after retrofitting.* ASCE J. Bridge Eng. 26(3). doi:10.1061/(ASCE)BE.1943-5592.0001668 | [verified] | The paper that describes the KW51 dataset. **Cite it if you use the data.** |
| A3 | Maes, K., Van Meerbeeck, L., Reynders, E., Lombaert, G. (2022). *Validation of vibration-based structural health monitoring on retrofitted railway bridge KW51.* MSSP 165, 108380. doi:10.1016/j.ymssp.2021.108380 | [verified] | **The baseline to beat on KW51.** Uses robust PCA and two black-box models to remove environmental effects, then detects the retrofit. Snippets don't say whether it reports calibrated intervals; read the full text. |
| A4 | *Multivariate Long-term Profile Monitoring with Application to the KW51 Railway Bridge.* arXiv:2506.20295 | [verified exists; authors not recorded here] | Functional-data approach on KW51. Reports a **kink in the frequency–temperature curve between 2 °C and 3 °C for all modes**, so the relation is nonlinear near freezing. |
| A5 | Dalmasso, Civera, De Biagi, Chiaia (2025). *GPR-based missing data imputation… for bridge SHM.* Advances in Bridge Engineering. doi:10.1186/s43251-025-00169-1 | [verified] | Applies a Gaussian process to KW51, but for filling gaps, not for compensation. |
| A6 | Hinchliffe, Jones, Cross (2025). *Parameter Estimation by Optimization of Interpretable Hyperparameters with Physics-Informed Gaussian Processes.* IWSHM 2025, pp. 1752–1759. | [verified] | A physics-informed GP kernel built from an oscillator. This physics enters in a different way from the thermal model proposed here. |
| A7 | Jones, Pitchforth, Cross. *Gaussian process kernels for partial physical insight.* EWSHM 2024. Also Pitchforth, Jones, Cross on *physically-informed change-point kernels* (EWSHM 2024). | [verified exist; full text not read] | **Closest prior work to RQ1.** Change-point kernels may already handle the freezing kink. Read these before claiming novelty. |
| A8 | Worden & Cross (2018), Bayesian treed GP for nonlinear environmental effects; Moser & Moaveni (2011), polynomial temperature regression on a bridge. | **unverified** (seen only as citations inside other papers) | Likely real. Verify before citing. |

### B. Bearing fault diagnosis: benchmarks, physics, and uncertainty
| # | Paper | Status | Note |
|---|---|---|---|
| B1 | Smith, W.A. & Randall, R.B. (2015). *Rolling element bearing diagnostics using the CWRU data: A benchmark study.* MSSP 64–65, 100–131. doi:10.1016/j.ymssp.2015.04.021 | [verified] | Physics-based envelope analysis on all of CWRU. Lists problems in the data. |
| B2 | Hendriks, Dumond, Knox (2022). *Towards better benchmarking using the CWRU bearing fault dataset.* MSSP 169, 108732. | [verified] | Shows that **common CWRU train/test splits leak data**, which inflates reported accuracy. |
| B3 | Lessmeier, Kimotho, Zimmer, Sextro (2016). *Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals… A benchmark data set.* PHM Europe. | [verified] | The Paderborn dataset paper. It has **artificial *and* real (accelerated-lifetime) damage**, which gives a genuine distribution shift to test on. |
| B4 | Xu, Zhao, Wang, Bashir (2024). *Physics-informed probabilistic deep network with interpretable mechanism for trustworthy mechanical fault diagnosis.* Adv. Eng. Informatics 62, 102806. doi:10.1016/j.aei.2024.102806 | [verified] | Physics plus uncertainty for fault diagnosis. Its abstract does not mention domain shift. |
| B5 | *Calibrated Prediction Set in Fault Detection with Risk Guarantees via Significance Tests.* arXiv:2508.01208 | [verified exists] | Conformal prediction for fault detection. Direct prior work for RQ2. |
| B6 | Alam et al. *Physics-Informed Multimodal Bearing Fault Classification under Variable Operating Conditions using Transfer Learning.* arXiv:2508.07536 | [verified exists] | Physics plus domain shift, without calibrated uncertainty. |

### C. Bearing prognostics (remaining useful life, RUL) with uncertainty
| # | Paper | Status | Note |
|---|---|---|---|
| C1 | *Uncertainty-Aware Bearing Remaining Useful Life Prediction Based on Conformal Prediction.* PHM Society Europe (papers.phmsociety.org/index.php/phme/article/view/4902) | [verified exists] | **Already does physics-flavored features plus conformal RUL on XJTU-SY.** This fills much of the RQ3 gap. |
| C2 | *RULSurv…* arXiv:2405.01614 | [verified exists] | Survival-analysis RUL with credible intervals on XJTU-SY. |

### D. Methods background (for you to learn from)
| # | Paper | Status |
|---|---|---|
| D1 | Angelopoulos, A.N. & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511 | [verified]. **Read this first.** It has code notebooks. |

---

## 2. Three candidate research questions

### RQ1 (recommended): Physics-guided, conformally calibrated temperature compensation for bridge modal frequencies
> *Does a thermal-stiffness physics prior combined with a learned residual and split-conformal intervals keep valid prediction-interval coverage when the model must extrapolate to unseen seasons, and does it flag the KW51 retrofit with fewer false alarms than PCA, polynomial, and plain-GP baselines?*

- **Physics idea:** Steel's Young's modulus E drops slightly as temperature rises. Since f ∝ √(k/m) and k ∝ E, the natural frequencies should fall a little with temperature. Your model would be *f(T) = f₀·√(1 + α·(T − T₀)) + learned residual*, and α is a fitted parameter you can **check against materials data**. That check makes the model interpretable. (Near 0 °C there is a kink, see A4, which may come from frozen ballast or bearings; your model needs a term for it.)
- **Uncertainty idea:** *Conformal prediction* wraps any model and turns its errors on a held-out "calibration" set into intervals. Under exchangeability, 90 % intervals will contain 90 % of new data. Time series violate exchangeability, so **testing whether coverage holds when training on autumn/winter and testing on spring is the scientific question**.
- **Gap (stated carefully):** KW51 compensation so far has used PCA and black-box models (A3) and functional data analysis (A4). Physics-informed GPs exist (A6, A7). **In my searches I found no paper that applies conformal prediction to frequency-based bridge monitoring**, or one that tests interval coverage under seasonal extrapolation on KW51. "I didn't find it" is not proof it doesn't exist. Before writing "first", search Google Scholar for "conformal" + "structural health monitoring" yourself and read A7.
- **Feasibility:** High. You need only the 13 MB `trackedmodes.mat` file, a laptop runs it, and the retrofit gives a known ground-truth state change.
- **Risk:** The retrofit made the bridge *stiffer*, not damaged. You detect a "state change", and you must word it that way. The exact retrofit dates must come from A2 or A3, not be guessed from the data.

### RQ2: Do conformal fault-diagnosis sets stay valid from artificial to real bearing damage?
> *If you calibrate conformal prediction sets on Paderborn bearings with artificial damage, do they still reach their target coverage on bearings with real accelerated-lifetime damage? Do physics features (envelope-spectrum energy at the kinematic fault frequencies BPFO and BPFI) hold coverage better than a CNN?*

- **Gap:** Bearing diagnosis papers mostly report accuracy, often on leaky CWRU splits (B2). Conformal methods for fault detection are new (B5), and B4 has uncertainty but no shift test. An honest **bearing-wise, artificial→real** coverage study is a narrower gap than RQ1 but a real one.
- **Feasibility:** Medium. The full Paderborn set is about 20 GB of RAR files, though a subset is workable. You need the bearing geometry to compute fault frequencies (it is on the Paderborn site).
- **Risk:** Very crowded field, so reviewers have seen hundreds of bearing papers.

### RQ3: Physics-guided degradation model plus conformal RUL intervals (XJTU-SY / NASA IMS)
- **Honest assessment:** C1 already does nearly this. The remaining gap would be a small variation, such as a Paris-law-style crack-growth prior. **I don't recommend it.**

---

## 3. Dataset check (download tested in this session)

| Dataset | What it is | Download check | License | Fit |
|---|---|---|---|---|
| **KW51 bridge** (Zenodo 3745914) | Steel railway bridge, Leuven. Oct 2018–Jan 2020, before/during/after retrofit | **Downloaded and opened** `trackedmodes.mat`: 11,328 hourly records, 14 modes (frequency f and damping ξ, plus mode shapes), 11 environmental channels (steel temperature, humidity, weather). Raw acceleration is also available (~2–4 GB per month). | CC-BY-NC-SA-4.0 | **Best for RQ1** |
| CWRU bearings | Classic bench, seeded faults | HTTP 200 on sample file `97.mat` (3.9 MB) | Public, no explicit license | Leaky benchmark (B2). Use only as a secondary set. |
| Paderborn (KAt) | Bearings with artificial and real damage, vibration plus motor current | HTTP 200 on `K001.rar` (174 MB) | CC-BY-NC-4.0 | Best for RQ2 |
| NASA IMS bearings | Run-to-failure | HTTP 200 on `4. Bearings.zip` (1.08 GB) | Public | RQ3 |
| LUMO (Hannover lattice tower) | Reversible damage, Aug–Dec 2020 | Metadata/DOI 10.25835/0027803 found, **not downloaded** | CC-BY (check page) | Possible second structure for RQ1 |
| Z24 bridge | Classic SHM benchmark | **Not verified.** I believe it is available on request from KU Leuven. | ? | — |
| XJTU-SY | Run-to-failure bearings | **Not verified this session** | ? | — |

**Data sanity check on KW51 (not a result, only a check that the expected signal exists):** using data before May 2019, the correlation between steel temperature and frequency is about −0.6 for modes 6, 9 and 13, and −0.19 for mode 3. The negative sign is what the physics predicts.

## 4. Journal fit (honest, before any results exist)
- **NHSJS:** Realistic for RQ1 or RQ2 if the work is careful and reproducible.
- **ISEF (Engineering Mechanics or Systems Software):** RQ1 has a clear physics story and a real-world bridge, which judges like.
- **IEEE Trans. Instrumentation & Measurement:** Possible only if the results clearly beat A3-style baselines *and* the coverage analysis is rigorous. This is a high bar with mostly professional authors, so a mentor co-author would help a lot.
- **Review of Scientific Instruments:** Poor fit. RSI wants new instruments or measurement techniques, not data analysis of an existing dataset.
- **Better-fit alternatives:** *Journal of Civil Structural Health Monitoring*, *Data-Centric Engineering*, *Sensors* (MDPI), or the IWSHM/EWSHM conference proceedings.

## Sources
- https://zenodo.org/record/3745914
- https://ascelibrary.com/doi/10.1061/%28ASCE%29BE.1943-5592.0001668
- https://www.x-mol.com/paperRedirect/1433865258959433728 (Maes et al. 2022 MSSP)
- https://arxiv.org/pdf/2506.20295
- https://library.kab.ac.ug/Record/doaj-art-c4974630b86648609cb390bd38fce5dd/Description (Dalmasso et al.)
- https://dpi-proceedings.com/index.php/shm2025/article/view/37481 (Hinchliffe et al.)
- https://eprints.whiterose.ac.uk/id/eprint/228404 , https://eprints.whiterose.ac.uk/id/eprint/228403 (EWSHM 2024, Cross group)
- https://structurae.net/en/literature/journal-article/one-year-monitoring-of-the-z24-bridge-environmental-effects-versus-damage-events
- https://eld.bl.uk/catalog/100151692692_0x000001 (Hendriks et al. 2022)
- https://www.scilit.net/publications/ba15437eec97e158f16f1e5596e27b37 (Smith & Randall 2015)
- https://papers.phmsociety.org/index.php/phme/article/view/1577 (Lessmeier et al. 2016)
- https://dblp.uni-trier.de/rec/journals/aei/XuZWB24.html (Xu et al. 2024)
- https://arxiv.org/abs/2508.01208 , https://arxiv.org/html/2508.07536v1
- https://papers.phmsociety.org/index.php/phme/article/view/4902 , https://arxiv.org/pdf/2405.01614
- https://arxiv.org/abs/2107.07511v6
- https://service.tib.eu/ldmservice/dataset/luh-lumo
- https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/
