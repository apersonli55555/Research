# Post-impact settling, not peak acceleration, limits a browser-based smartphone fall detector: a reproducible benchmark on SisFall

*Draft manuscript. Every number below comes from `research/benchmark_sisfall.py` and `research/rule_v2.py`, run on the public SisFall dataset. Nothing is estimated or invented. Re-run them to reproduce the numbers (see `research/README.md`).*

---

## Abstract

**Background.** Phone-based fall detectors usually combine an impact threshold with a check that the person lies still afterwards. Few published detectors are evaluated with their exact deployed logic, and fewer report results on subjects held out from tuning.

**Methods.** We ported the detection pipeline of an open-source, browser-only fall detector (Motion Sense, `index.html`) to Python without changing its logic. That pipeline is a 24 m/s² peak rule, a post-peak variance gate and a 5 s stillness confirmation, plus a small neural network trained on synthetic feature vectors. We replayed all 4,505 trials of SisFall through it: 38 subjects (23 aged 19–30, 15 aged 60–75), 1,798 simulated falls and 2,707 activities of daily living (ADLs), resampled to 50 Hz to match phone motion-sensor rates. Confidence intervals come from a subject-level cluster bootstrap. We then proposed a revised rule: impact, then settling, then a 5 s hold, plus a posture-change gate. We tuned it on 18 subjects and reported it once on the 20 held-out subjects.

**Results.** The deployed rule detected **18.0 %** of falls (95 % CI 15.3–20.8) at **97.9 %** specificity, with 3.85 false alarms per hour of ADL recording. 97 % of falls exceeded the peak threshold, so the threshold was not the cause. The post-peak variance gate rejected most falls because, inside the 1.5 s window, the body is still bouncing (median lowest post-peak variance 3.7 against a 2.5 gate). The synthetic-data neural network fired on 98.6 % of ADL trials (endFlat = 0). With the 5 s confirmation added, the deployed rule-plus-network pipeline reached 81.7 % sensitivity and 55.8 % specificity. On the held-out subjects, the revised rule reached **93.9 %** sensitivity (88.6–97.9) at **95.5 %** specificity (94.6–96.4). On the elderly participants' ADLs alone, specificity was 95.5 % (93.8–97.2). Without the posture gate, no setting in the grid reached 95 % specificity on the tuning subjects. A gradient-boosted model trained on real data with leave-one-subject-out cross-validation reached 96.3 % sensitivity and 97.3 % specificity (AUC 0.994). That model is a trial-level upper reference, not a streaming detector.

**Conclusions.** For this class of detector, the main error was the timing of the stillness check, not the impact threshold. Replacing synthetic training vectors with real data, or with an orientation-change gate, matters more than model capacity. These are simulated falls by mostly young adults with a waist-mounted sensor. They do not establish real-world performance in older adults.

---

## 1. Introduction

Falls are a leading cause of injury in older adults. Lying on the floor for a long time after a fall is linked to worse outcomes, which motivates automatic detection. Smartphones are already owned and carried, so a detector that runs in a web page needs no app install and no extra hardware. Two gaps recur in the smartphone fall-detection literature:

1. **Evaluated logic ≠ deployed logic.** Detectors are often evaluated offline in a form that differs from what runs on the device: different windowing, no confirmation state machine, different sampling rate.
2. **Tuning leakage.** Thresholds are often chosen on the same subjects they are reported on.

We address both for one open-source detector. We replay a public dataset through a line-by-line port of the deployed state machine, and we tune any changes on disjoint subjects.

## 2. Methods

### 2.1 Dataset
SisFall (Sucerquia, López & Vargas-Bonilla, *Sensors* 2017; CC-BY-4.0; Zenodo record 22212281). Waist-mounted device, 200 Hz. We used the ADXL345 accelerometer (±16 g, 13-bit) and the ITG3200 gyroscope (±2000 °/s, 16-bit). There are 19 ADL types (D01–D19) and 15 fall types (F01–F15). Elderly subjects performed ADLs only, except SE06. We used all 4,505 parseable trials.

Raw counts were converted to m/s² and °/s with the dataset's stated sensor ranges. Signals were block-averaged by a factor of 4 (anti-aliasing) to 50 Hz, close to the 50–60 Hz at which browsers deliver `DeviceMotionEvent`.

### 2.2 Deployed pipeline (faithful port)
Constants and logic were copied from `index.html`:

* **Calibration:** resting baseline = mean |a| over the first 2 s (`finishCalibration`).
* **Window:** 1.5 s rolling buffer, evaluated at every sample (`onMotion` → `runLiveDetection`).
* **Features** (`computeFeatures`): peak |a|, variance of |a|, post-peak variance ("stillness"), |peak − baseline|, peak and variance of |ω|. The two microphone features are set to 0 because SisFall has no audio. The `endFlat` orientation flag is evaluated at both 0 and 1 as bounds.
* **Rule** (`thresholdDetect`): peak > 24 m/s² **and** stillness < 2.5. The height scale factor is 1.
* **Confirmation** (`checkFallConfirmation`): alert if every sample over the next 5 s satisfies |a − baseline| ≤ 1.5 m/s². Any excursion cancels.
* **Network** (`generateSyntheticSamples`, `trainMLP`): a 9-6-1 tanh/sigmoid MLP trained by full-batch gradient descent (2,000 epochs, lr 0.3, L2 0.002) on the app's 70 synthetic vectors. Results are averaged over 5 seeds. p > 0.5 starts the same confirmation as the rule.

A trial counts as detected if at least one alert is raised. For ADL trials, any alert is a false positive. False alarms per hour re-arm the detector after each alert.

### 2.3 Proposed rule (v2)
1. **Impact:** |a| > T.
2. **Settle:** within S seconds of the impact, a still period starts (|a − baseline| ≤ τ).
3. **Hold:** the still period lasts 5 s without interruption, the same duration as the deployed confirmation.
4. **Posture gate** (optional): the angle between the mean gravity vector in the second before impact (−1.0 to −0.2 s) and the mean during the hold exceeds θ. The gate compares the device with itself, so it does not depend on how the phone sits in a pocket.

All four steps need only `accelerationIncludingGravity`, which the app already reads.

**Protocol.** Subjects were split before any tuning. *Dev:* SA01–SA12, SE01–SE05, SE07 (18 subjects, 2,191 trials). *Test:* SA13–SA23, SE06, SE08–SE15 (20 subjects, 2,314 trials). The grid was T ∈ {16, 20, 24, 28, 32} m/s², S ∈ {1, 2, 3} s, τ ∈ {1, 1.5, 2, 3} m/s², θ ∈ {off, 30°, 45°, 60°} (240 settings). Selection rule: highest dev sensitivity with dev specificity ≥ 95 %. The test set was evaluated once, with the selected setting.

### 2.4 Learned reference
Logistic regression and histogram gradient boosting were trained on eight features. Six are window features from a window placed at the trial's global peak. The other two are the mean |a − baseline| and the variance of |a| 2–4 s after the peak. Evaluation is leave-one-subject-out, one decision per trial. This is an optimistic reference because the peak is located with hindsight, not a deployable detector.

### 2.5 Statistics
Sensitivity and specificity are pooled over trials. 95 % CIs come from 2,000 subject-level bootstrap resamples, because trials from the same subject are not independent.

## 3. Results

### 3.1 The deployed pipeline (all 38 subjects)

| Pipeline | Sensitivity [95 % CI] | Specificity [95 % CI] |
|---|---|---|
| Rule + 5 s confirmation (as shipped) | **0.180** [0.153, 0.208] | **0.979** [0.971, 0.987] |
| Rule trigger only (no confirmation) | 0.286 [0.258, 0.314] | 0.881 [0.865, 0.899] |
| Rule OR network + confirmation, endFlat = 0 | 0.817 [0.742, 0.883] | 0.558 [0.529, 0.586] |
| Rule OR network + confirmation, endFlat = 1 | 0.820 [0.746, 0.886] | 0.539 [0.509, 0.569] |
| Network alone fires, endFlat = 0 | 1.000 | 0.014 [0.007, 0.021] |
| Learned logistic regression (LOSO, trial-level) | 0.937 [0.905, 0.963] | 0.946 [0.930, 0.962] (AUC 0.983) |
| Learned gradient boosting (LOSO, trial-level) | 0.963 [0.943, 0.982] | 0.973 [0.960, 0.983] (AUC 0.994) |

The shipped rule produced 323 true positives, 1,475 false negatives, 56 false positives and 2,651 true negatives. That is 3.85 false alarms per hour over 14.5 h of ADL recordings. Specificity on the elderly ADLs alone was 98.6 %.

### 3.2 Why falls were missed
* **The impact threshold was not the problem.** 97.0 % of fall trials exceeded 24 m/s². The median fall peak was 52.5 m/s².
* **The variance gate was mistimed.** Only 28.6 % of falls ever met *peak > 24 and stillness < 2.5* in the same 1.5 s window. In the first repetition of every fall type per subject (n = 360 trials), the lowest post-peak variance reached in any window containing a peak above 24 had a median of 3.7 (IQR 2.3–5.4). Within 1.5 s of impact the body is still rebounding and rolling.
* **The confirmation was too strict at the start.** In the same subset, |a| first returns within 1.5 m/s² of baseline a median 0.14 s after the peak. The deployed confirmation starts at the trigger and cancels on the first excursion, so post-impact rebound cancels it. By 2–4 s after the peak, 86.7 % of falls are still within tolerance.
* **The synthetic-data network does not transfer.** It fired on nearly every ADL trial, including 95 % of car entries/exits (D17), 84 % of bending without knee flexion (D16) and 77 % of slow lying (D12) after confirmation. Its training set separates falls from non-falls almost entirely by the `endFlat` flag and by sound, and neither exists in this data. This is a distribution shift, not a capacity problem.

### 3.3 Proposed rule v2 (held-out test subjects only)

| Detector (test subjects, n = 20) | Sensitivity [95 % CI] | Specificity [95 % CI] | Elderly-ADL specificity |
|---|---|---|---|
| Shipped rule + confirmation | 0.183 [0.143, 0.221] | 0.987 [0.979, 0.994] | — |
| Shipped rule OR network | 0.900 [0.852, 0.942] | 0.549 [0.513, 0.583] | — |
| **v2 with posture gate** (T = 20, S = 3 s, τ = 3, θ = 30°) | **0.939** [0.886, 0.979] | **0.955** [0.946, 0.964] | 0.955 [0.938, 0.972] |
| v2 without posture gate (T = 24, S = 2 s, τ = 3) | 0.924 [0.866, 0.967] | 0.887 [0.870, 0.907] | 0.926 [0.898, 0.952] |
| Learned GBM, LOSO (reference) | 0.959 [0.923, 0.988] | 0.972 [0.951, 0.988] | — |

On the dev subjects, the selected v2 setting scored 0.903 sensitivity and 0.954 specificity, so test performance did not degrade. No magnitude-only setting reached 95 % dev specificity; the best was 94.7 %. The posture gate is therefore what makes the specificity target reachable.

Remaining v2 false positives were concentrated in D11, *"trying to get up and collapse into a chair"* (48 % positive). This activity is arguably a near-fall. All other ADL types were ≤ 9 % positive. Per-fall-type sensitivity ranged from 85 % (F02, backward slip) to 100 % (F05, F11, F12, F14).

## 4. Discussion

**Principal finding.** A widely used pattern, impact threshold plus immediate stillness, can look reasonable when tested on one's own phone and still miss most falls. The stillness check sits inside the same short window as the impact. Moving the check to after a settling period, and requiring a change in body orientation, recovered most of the lost sensitivity without giving up specificity. Every change is implementable in the existing browser code.

**Synthetic bootstrapping.** Training a classifier on hand-written "plausible" feature ranges is attractive when no data is available. Here it produced a detector that fires on almost everything. Public datasets with permissive licences, SisFall among them, make that shortcut unnecessary.

**Limitations.**
1. Falls were *simulated* by young adults (plus one judo-trained older adult) onto mats. Real falls in older adults are slower and more varied, and published work shows sizeable performance drops on real-world falls (e.g. the FARSEEING repository).
2. SisFall's sensor is waist-fixed. A phone in a trouser pocket or a bag has different dynamics, so the thresholds need re-validation on smartphone data (e.g. MobiAct, UMAFall).
3. Audio and the phone-flat flag could not be evaluated.
4. Trials are short (12–100 s), so false alarms per hour of real life are not estimable here.
5. The grid's selected S and τ lie on the edge of the grid searched.

## 5. Data and code availability
Code: `research/` in this repository. The data are public: SisFall, Zenodo record 22212281, CC-BY-4.0. Results: `research/results/summary.json`, `research/results/rule_v2.json`, `research/results/trials.json` (per-trial outputs).

## References
1. Sucerquia A, López JD, Vargas-Bonilla JF. SisFall: A Fall and Movement Dataset. *Sensors* 2017;17(1):198.
2. Bagalà F, et al. Evaluation of accelerometer-based fall detection algorithms on real-world falls. *PLoS ONE* 2012;7(5):e37062.
3. Igual R, Medrano C, Plaza I. Challenges, issues and trends in fall detection systems. *BioMed Eng OnLine* 2013;12:66.
4. Kangas M, et al. Comparison of low-complexity fall detection algorithms for body attached accelerometers. *Gait & Posture* 2008;28(2):285–291.

*(The reference list is a starting point. Verify each citation before submission.)*
