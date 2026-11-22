# Preliminary results: staged → real fall detection

**Run:** Colab, NVIDIA L4. Encoders: CLIP ViT-L-14 (datacomp_xl_s13b_b90k) and VideoMAE-B (Kinetics-400).
**Labels:** one labeller, the first 150 videos of the random test sample (presented in random order, so this is a random subset of the 422-video test sample). No training-pool clips labelled yet, so real-trained probes were not run.
**Counts:** staged (UR Fall) 316 clips / 60 falls. Real (OOPS!) test set 231 clips from 149 videos / 96 falls.

AUROC, average precision (AP) and recall at 5% false-positive rate, each with a 95% video-level bootstrap CI.

| Evaluated on | Method | AUROC | AP | Recall @ 5% FPR |
|---|---|---|---|---|
| staged (5-fold CV) | zero-shot CLIP | 0.912 [0.865, 0.951] | 0.689 [0.529, 0.854] | 0.550 [0.197, 0.734] |
| staged (5-fold CV) | probe: CLIP appearance | 1.000 [0.999, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] |
| staged (5-fold CV) | probe: CLIP temporal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| staged (5-fold CV) | probe: VideoMAE | 0.998 [0.995, 1.000] | 0.992 [0.977, 1.000] | 1.000 [1.000, 1.000] |
| staged (5-fold CV) | probe: CLIP + VideoMAE | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| **real** | zero-shot CLIP | 0.718 [0.652, 0.780] | 0.592 [0.506, 0.696] | 0.062 [0.022, 0.242] |
| **real** | staged-trained: CLIP appearance | 0.733 [0.667, 0.795] | 0.584 [0.494, 0.698] | 0.062 [0.010, 0.257] |
| **real** | staged-trained: CLIP temporal | 0.751 [0.683, 0.810] | 0.671 [0.575, 0.764] | 0.177 [0.068, 0.476] |
| **real** | staged-trained: VideoMAE | 0.773 [0.712, 0.829] | 0.720 [0.639, 0.798] | 0.302 [0.147, 0.489] |
| **real** | staged-trained: CLIP + VideoMAE | 0.783 [0.717, 0.837] | 0.718 [0.625, 0.802] | 0.219 [0.101, 0.500] |

## What this shows
- **Large staged-to-real gap.** Every probe is at or near perfect on staged data (AUROC ≥ 0.998, recall 100% at 5% FPR). On real falls the best reaches AUROC 0.78 and catches at most ~30% of falls at 5% FPR, and that method's 95% interval tops out at 0.49.
- **Motion-aware representations trend better** (CLIP appearance 0.733 → CLIP temporal 0.751 → VideoMAE 0.773). The intervals overlap, so the ordering is not established at n = 149 videos.
- **UR Fall is saturated.** A perfect in-domain score means it cannot rank methods; harder staged datasets are needed.

## Not yet measured
- Real-trained and staged+real probes (need labelled training-pool clips). This decides whether the gap comes from training on staged data or from real falls being intrinsically hard.
- Falls vs. other accidents only (`real_hard` rows, added after this run).
- Second-labeller agreement (Cohen's κ).

---

# Run 2: with real training labels

**Labels:** 165 test videos (255 clips, 102 falls); 34 training-pool videos (56 clips, 26 falls). Same encoders and settings as run 1. Values transcribed from the Colab output (AUROC [95% CI]); see `staged_to_real.json` on Drive for the full table.

Real test set, AUROC by training data:

| Representation | Staged-trained (30 lab falls) | Real-trained (26 real falls) | Staged + real |
|---|---|---|---|
| CLIP appearance | 0.723 [0.661, 0.788] | 0.718 [0.653, 0.778] | 0.737 [0.671, 0.800] |
| CLIP temporal | 0.728 [0.667, 0.790] | 0.809 [0.752, 0.862] | 0.793 [0.736, 0.849] |
| VideoMAE | 0.765 [0.711, 0.820] | 0.726 [0.666, 0.783] | 0.792 [0.738, 0.842] |
| CLIP + VideoMAE | 0.759 [0.696, 0.820] | 0.832 [0.780, 0.881] | 0.826 [0.771, 0.878] |

Zero-shot CLIP: 0.717. Best recall at 5% FPR: about 0.37 (staged + real VideoMAE).

Falls vs. other accidents only (`real_hard`, 165 clips, 102 falls): AUROC 0.69–0.80 across methods; best staged + real VideoMAE 0.800 [0.725, 0.859].

## Reading
- With a similar number of falls, real training data beats staged data for the motion-aware representations (CLIP temporal 0.73 → 0.81; CLIP + VideoMAE 0.76 → 0.83). Training on staged data is a large part of the gap. Intervals overlap and n is small, so this needs more training labels to confirm.
- Even the best real-trained detector stays far below staged performance (0.83 vs. 1.00), so real falls are also harder.
- Separating falls from other accidents is the hardest sub-problem and the natural target for a method.

---

# Run 3: 268 training clips (final labelling)

**Labels:** 165 test videos (255 clips, 102 falls); training pool 268 clips with 131 falls. Same settings. Transcribed from the Colab output; the head-to-head and learning-curve tables were not produced in this run (it used the code from before they were added).

Real test set, AUROC [95% CI]; recall at 5% FPR in the last column for the best rows:

| Representation | Staged-trained | Real-trained | Staged + real |
|---|---|---|---|
| CLIP appearance | 0.723 [0.661, 0.788] | 0.732 [0.667, 0.793] | 0.741 [0.677, 0.802] |
| CLIP temporal | 0.728 [0.667, 0.790] | 0.866 [0.818, 0.907] | 0.857 [0.807, 0.899] |
| VideoMAE | 0.765 [0.711, 0.820] | 0.787 [0.732, 0.841] | 0.803 [0.747, 0.854] |
| CLIP + VideoMAE | 0.759 [0.696, 0.820] | **0.899 [0.859, 0.934]** | 0.892 [0.850, 0.929] |

Recall at 5% FPR: CLIP + VideoMAE staged-trained 0.167 [0.098, 0.375] vs. real-trained 0.431 [0.325, 0.670].

Falls vs. other accidents (`real_hard`, 165 clips, 102 falls), AUROC: CLIP + VideoMAE staged-trained 0.715 [0.633, 0.795] vs. real-trained 0.860 [0.794, 0.914]; CLIP temporal real-trained 0.835 [0.765, 0.891].

## Reading
- **Training data, not model capacity, drives the gap.** With the same frozen features and a linear probe, 131 real falls lift CLIP + VideoMAE from 0.76 to 0.90 AUROC; the intervals do not overlap.
- **Staged data adds nothing once real data exists** (0.892 with both vs. 0.899 real only).
- **Temporal information matters, and the two encoders are complementary.** CLIP appearance barely benefits from real data (0.72 → 0.73), CLIP temporal gains most (0.73 → 0.87), and CLIP temporal + VideoMAE is best.
- **Caveat: "real-trained" is in-distribution.** Training and test clips both come from OOPS!, so part of the gain is ordinary in-domain advantage. A second real-world test source is needed to separate "real vs. staged" from "same vs. different dataset". The training pool was also pre-selected by CLIP zero-shot score (the test set was not).

## Run 3b: head-to-head tests and learning curve (same labels, no new labelling)

Paired video-level bootstrap on the same 255 test clips (2,000 resamples); "p < 0.001" = no resample crossed zero.

| Comparison (real test set) | AUROC diff | 95% CI | p |
|---|---|---|---|
| CLIP appearance: real- vs staged-trained | +0.009 | [−0.055, +0.068] | 0.770 |
| CLIP appearance: staged+real vs staged | +0.018 | [−0.037, +0.070] | 0.508 |
| CLIP temporal: real- vs staged-trained | **+0.138** | [+0.078, +0.201] | < 0.001 |
| CLIP temporal: staged+real vs staged | +0.129 | [+0.075, +0.186] | < 0.001 |
| VideoMAE: real- vs staged-trained | +0.022 | [−0.036, +0.083] | 0.493 |
| VideoMAE: staged+real vs staged | +0.038 | [−0.012, +0.091] | 0.135 |
| CLIP + VideoMAE: real- vs staged-trained | **+0.139** | [+0.087, +0.193] | < 0.001 |
| CLIP + VideoMAE: staged+real vs staged | +0.133 | [+0.085, +0.183] | < 0.001 |
| Staged-trained: CLIP temporal vs CLIP appearance | +0.005 | [−0.061, +0.073] | 0.922 |
| Staged-trained: VideoMAE vs CLIP appearance | +0.042 | [−0.028, +0.116] | 0.263 |
| Staged-trained: CLIP + VideoMAE vs CLIP appearance | +0.036 | [−0.027, +0.104] | 0.302 |

Learning curve, CLIP + VideoMAE (staged-only AUROC 0.759); mean ± s.d. over 10 random draws of real training videos:

| Real training videos | Real falls (avg) | Real-trained AUROC | Staged + real AUROC |
|---|---|---|---|
| 5 | 4 | 0.725 ± 0.094 | 0.784 ± 0.014 |
| 10 | 8 | 0.764 ± 0.035 | 0.788 ± 0.011 |
| 20 | 15 | 0.815 ± 0.039 | 0.815 ± 0.032 |
| 40 | 30 | 0.826 ± 0.019 | 0.827 ± 0.016 |
| 80 | 60 | 0.871 ± 0.019 | 0.866 ± 0.015 |
| 160 | 116 | 0.897 ± 0.006 | 0.890 ± 0.005 |
| 180 | 131 | 0.899 | 0.892 |

## Reading
- **The real-data gain is significant for motion-aware representations** (+0.14 AUROC, p < 0.001 for CLIP temporal and CLIP + VideoMAE) and **not significant for static appearance** (+0.01, p = 0.77). VideoMAE alone also gains little (+0.02, p = 0.49); the gain comes from the CLIP temporal features, alone or combined.
- **Under staged training, the choice of encoder makes no significant difference** (all p ≥ 0.26).
- **About 8 real falls match the 30 staged falls (60 clips)**, about 15 beat them, and AUROC is still rising at 131, so more real data would likely help further.
- **Staged data helps only when real data is tiny** (4 real falls: 0.784 with staged vs. 0.725 without); from about 15 real falls on, it adds nothing.

---

# Run 4: real vs. synthetic vs. staged (OmniFall OF-Syn)

**Synthetic data:** OmniFall OF-Syn, one 2 s clip per video: all 2,136 videos with a fall segment (clip centred on the fall) and 3,000 hash-sampled videos without one (5,136 clips). Same encoders (CLIP ViT-L-14 DataComp, VideoMAE-B K400), same probes, same real test set (255 clips, 165 videos, 102 falls). Copied from the Colab text output (exact).

Real test set, AUROC [95% CI] (full table: `staged_to_real.md` on Drive):

| Representation | Staged | Synthetic (2,136 falls) | Real (131 falls) | Synthetic + real |
|---|---|---|---|---|
| CLIP appearance | 0.723 [0.661, 0.788] | 0.728 [0.670, 0.783] | 0.732 [0.667, 0.793] | 0.800 [0.749, 0.855] |
| CLIP temporal | 0.728 [0.667, 0.790] | 0.795 [0.738, 0.849] | 0.866 [0.818, 0.907] | 0.823 [0.767, 0.871] |
| VideoMAE | 0.765 [0.711, 0.820] | 0.789 [0.734, 0.838] | 0.787 [0.732, 0.841] | 0.842 [0.796, 0.883] |
| CLIP + VideoMAE | 0.759 [0.696, 0.820] | 0.814 [0.760, 0.864] | **0.899 [0.859, 0.934]** | 0.872 [0.827, 0.911] |

Staged + synthetic: CLIP + VideoMAE 0.809 [0.758, 0.861] (no gain over synthetic alone).
Falls vs. other accidents (`real_hard`), CLIP + VideoMAE: synthetic 0.772 [0.696, 0.844], real 0.860 [0.794, 0.914], synthetic + real 0.848 [0.786, 0.906].

Head-to-head (paired video-level bootstrap):

| Comparison | ΔAUROC | 95% CI | p |
|---|---|---|---|
| CLIP + VideoMAE: real (131) vs. synthetic (2,136) | **+0.085** | [+0.037, +0.137] | < 0.001 |
| CLIP + VideoMAE: synthetic vs. staged | +0.054 | [+0.001, +0.108] | 0.045 |
| CLIP + VideoMAE: synthetic + real vs. real | −0.027 | [−0.066, +0.010] | 0.166 |
| CLIP temporal: real vs. synthetic | +0.071 | [+0.018, +0.129] | 0.008 |
| CLIP temporal: synthetic vs. staged | +0.067 | [+0.011, +0.124] | 0.018 |
| CLIP temporal: synthetic + real vs. real | −0.043 | [−0.092, +0.005] | 0.081 |
| CLIP appearance: real vs. synthetic | +0.004 | [−0.069, +0.073] | 0.920 |
| CLIP appearance: synthetic + real vs. real | +0.068 | [+0.008, +0.134] | 0.024 |
| VideoMAE: real vs. synthetic | −0.002 | [−0.063, +0.056] | 0.959 |
| VideoMAE: synthetic + real vs. real | +0.055 | [+0.008, +0.103] | 0.024 |

Equal budget, CLIP + VideoMAE: k training falls + k non-falls from one source; AUROC on the real test set, mean ± s.d. over 10 random draws (3 draws above 60 falls; 1 at the full set):

| Training falls | 4 | 8 | 15 | 30 | 60 | 125 | 131 | 250 | 500 | 1000 | 2000 | 2136 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Staged | 0.751 ± 0.020 | 0.745 ± 0.014 | 0.744 ± 0.009 | 0.752 ± 0.008 | 0.754 | – | – | – | – | – | – | – |
| Synthetic | 0.723 ± 0.025 | 0.724 ± 0.020 | 0.742 ± 0.011 | 0.768 ± 0.024 | 0.786 ± 0.024 | 0.825 ± 0.018 | – | 0.818 ± 0.017 | 0.819 ± 0.004 | 0.818 ± 0.024 | 0.817 ± 0.006 | 0.812 |
| Real | 0.670 ± 0.130 | 0.772 ± 0.028 | 0.789 ± 0.027 | 0.836 ± 0.018 | 0.862 ± 0.013 | 0.896 ± 0.003 | 0.895 | – | – | – | – | – |

"Synthetic falls needed to match 131 real falls: not reached with the 2,136 available."

## Reading
- **Synthetic data plateaus.** AUROC rises to about 0.82 by 125 synthetic falls and stays there up to 2,136 (0.812–0.825). Real data keeps rising: 30 real falls (0.836 ± 0.018) already exceed every synthetic budget, and 131 real falls reach 0.899.
- **Real beats synthetic at the full budget** (+0.085, p < 0.001) with 16× fewer falls, and on falls vs. other accidents (0.860 vs. 0.772).
- **We replicate OmniFall's finding that synthetic beats staged** (+0.054, p = 0.045; staged stays flat at about 0.75 at every budget).
- **Mixing synthetic into real data does not help the best representation** (−0.027, n.s.), but does help the weaker single representations (CLIP appearance +0.068, VideoMAE +0.055; both p = 0.024).
- **With only a handful of falls, real data is unreliable** (4 falls: 0.670 ± 0.130, below staged and synthetic). Real overtakes both from about 8 falls.
- **Main caveat: real training is in-domain.** Real training and test clips both come from OOPS!; synthetic and staged are out-of-domain. Part of the real-data advantage may be in-domain advantage. A second real-world test source (remaining work item 1) is needed before claiming real data is intrinsically worth more. Probes are linear on frozen features; OmniFall fine-tunes, which may use synthetic data better.

---

# Run 5: external real-world test on HMDB51 (nothing trained on it)

**External test:** HMDB51 (CC BY 4.0), split-1 parquet files: all 100 `fall_floor` clips vs. 20 hash-sampled clips from each other class (1,000), 2 s windows every 1 s (2,740 windows), clip score = max window. Hard subset (fixed in advance): falls vs. cartwheel, flic_flac, somersault, handstand, dive, jump, sit, situp, pushup, stand, climb (320 clips). Training sets unchanged (staged UR Fall; synthetic OF-Syn 2,136 falls; real OOPS! training pool 131 falls). Copied from the Colab text output.

AUROC [95% CI] on HMDB51:

| Representation | Zero-shot | Staged | Synthetic | Real (OOPS!) | Synthetic + real |
|---|---|---|---|---|---|
| CLIP appearance | 0.834 [0.802, 0.866] | 0.783 [0.746, 0.818] | 0.813 [0.773, 0.849] | 0.626 [0.575, 0.678] | 0.803 [0.761, 0.842] |
| CLIP temporal | – | 0.887 [0.854, 0.915] | 0.866 [0.826, 0.905] | 0.825 [0.782, 0.870] | 0.842 [0.798, 0.887] |
| VideoMAE | – | 0.664 [0.609, 0.724] | 0.877 [0.844, 0.908] | 0.779 [0.732, 0.828] | 0.901 [0.873, 0.928] |
| CLIP + VideoMAE | – | 0.899 [0.870, 0.926] | **0.917 [0.892, 0.943]** | 0.888 [0.850, 0.925] | 0.901 [0.869, 0.932] |

Hard subset (falls vs. acrobatic / ground-contact actions), AUROC: CLIP + VideoMAE staged 0.833 [0.783, 0.877], synthetic 0.880 [0.843, 0.915], real 0.777 [0.718, 0.833], synthetic + real 0.855 [0.811, 0.896]; VideoMAE synthetic + real 0.878 [0.838, 0.911]; CLIP appearance real 0.365 [0.305, 0.425] (below chance).

Head-to-head on HMDB51 (paired bootstrap by clip):

| Comparison | ΔAUROC | 95% CI | p |
|---|---|---|---|
| CLIP + VideoMAE: real vs. synthetic | −0.029 | [−0.057, −0.003] | 0.029 |
| CLIP + VideoMAE: real vs. staged | −0.011 | [−0.048, +0.027] | 0.566 |
| CLIP + VideoMAE: synthetic vs. staged | +0.018 | [−0.010, +0.048] | 0.211 |
| CLIP + VideoMAE: synthetic + real vs. real | +0.013 | [−0.010, +0.036] | 0.264 |
| CLIP temporal: real vs. synthetic | −0.040 | [−0.080, −0.002] | 0.041 |
| CLIP temporal: real vs. staged | −0.062 | [−0.106, −0.016] | 0.006 |
| VideoMAE: real vs. synthetic | −0.098 | [−0.153, −0.046] | < 0.001 |
| VideoMAE: synthetic vs. staged | +0.214 | [+0.153, +0.277] | < 0.001 |
| VideoMAE: synthetic + real vs. real | +0.122 | [+0.074, +0.172] | < 0.001 |
| CLIP appearance: real vs. synthetic | −0.187 | [−0.236, −0.137] | < 0.001 |
| CLIP appearance: synthetic + real vs. real | +0.176 | [+0.133, +0.220] | < 0.001 |

## Reading
- **The real-data advantage does not transfer.** On FallOrNot (in-domain for the real training data) real beats synthetic by +0.085 (p < 0.001); on HMDB51 real is slightly *worse* than synthetic (−0.029, p = 0.029) and no better than staged. The run-4 headline ("30 real falls beat 2,136 synthetic falls") holds only in-domain and must not be stated as a general claim.
- **Rankings flip between real-world test sets.** Synthetic is best on HMDB51, real is best on FallOrNot. Which training source looks best depends on which "real-world" benchmark is used.
- **HMDB51 falls look more like staged falls than OOPS! falls do.** Staged probes score 0.899 on HMDB51 but 0.759 on FallOrNot, and zero-shot CLIP 0.834 vs. 0.717. HMDB51 `fall_floor` contains many movie and trimmed, clearly framed falls, so it is a different, likely easier, kind of "real" than OOPS! accidents.
- **Combining synthetic and real is the most robust choice**: it is within the confidence interval of the best source on both test sets (FallOrNot 0.872 vs. 0.899, n.s.; HMDB51 0.901 vs. 0.917).
- **Synthetic data rescues the weaker representations out of domain** (VideoMAE +0.214 over staged; CLIP appearance trained on OOPS! real data falls to 0.626).

---

# Run 6: fine-tuned VideoMAE (pre-registered test)

**Setup** (`finetune.py`, fixed before running): VideoMAE-B (Kinetics-400) fine-tuned end-to-end, 5 epochs, batch 8, AdamW lr 5e-5, weight decay 0.05, one-cycle schedule, class-balanced sampling, horizontal-flip augmentation, seed 0, last checkpoint. Training data: synthetic (1,000 + 1,000 hash-sampled OF-Syn clips), real (the 268 labelled OOPS! training clips) or both. Clips are the same 16-frame windows used everywhere else (523 OOPS! clips extracted from 345 labelled videos). Scored on FallOrNot (255 clips) and HMDB51 (1,100 clips, max over windows). **One seed only.** Copied from the Colab text output.

**Pre-registered rule:** fine-tuned (synthetic + real) beats the best frozen probe on both test sets, paired 95% CI above zero.
- FallOrNot: vs. best probe (real, CLIP + VideoMAE): −0.002 [−0.040, +0.037], p = 0.933
- HMDB51: vs. best probe (synthetic, CLIP + VideoMAE): +0.028 [−0.003, +0.058], p = 0.074
- **RESULT: NOT MET.**

AUROC [95% CI]:

| Model | FallOrNot | FallOrNot hard | HMDB51 | HMDB51 hard |
|---|---|---|---|---|
| Fine-tuned, synthetic | 0.829 [0.779, 0.875] | 0.746 [0.665, 0.816] | **0.967 [0.954, 0.979]** | **0.938 [0.913, 0.962]** |
| Fine-tuned, real | **0.906 [0.868, 0.940]** | **0.876 [0.819, 0.930]** | 0.902 [0.869, 0.931] | 0.762 [0.704, 0.815] |
| Fine-tuned, synthetic + real | 0.897 [0.860, 0.931] | 0.854 [0.792, 0.910] | 0.945 [0.925, 0.964] | 0.859 [0.820, 0.900] |
| Best probe on that set (for reference) | 0.899 (real, CLIP + VideoMAE) | 0.860 (real, CLIP + VideoMAE) | 0.917 (synthetic, CLIP + VideoMAE) | 0.880 (synthetic, CLIP + VideoMAE) |

Recall at 5% FPR: fine-tuned synthetic + real 0.529 [0.283, 0.667] on FallOrNot and 0.770 [0.612, 0.859] on HMDB51; fine-tuned synthetic 0.840 [0.714, 0.916] on HMDB51.

Fine-tuned vs. probe with the same training data (paired):

| Training data | FallOrNot ΔAUROC | HMDB51 ΔAUROC |
|---|---|---|
| Synthetic | +0.015 [−0.036, +0.068], p = 0.555 | **+0.050 [+0.024, +0.077], p < 0.001** |
| Real | +0.007 [−0.029, +0.047], p = 0.723 | +0.014 [−0.024, +0.053], p = 0.466 |
| Synthetic + real | +0.025 [−0.016, +0.069], p = 0.262 | **+0.044 [+0.009, +0.081], p = 0.019** |

## Reading (exploratory beyond the pre-registered rule)
- **Pre-registered test not met.** Fine-tuning on synthetic + real is not significantly better than the best probe on either test set.
- **The ranking flip holds for fine-tuned models too**, a second model family: real-trained is best on FallOrNot (0.906), synthetic-trained is best on HMDB51 (0.967), and each is clearly worse on the other set (synthetic 0.829 on FallOrNot; real 0.902 on HMDB51, 0.762 on its hard subset).
- **Synthetic + real is the robust choice again**: within the CI of the best model on FallOrNot (0.897 vs. 0.906) and second-best on HMDB51 (0.945 vs. 0.967). It is the only model in the top group on both sets, though not significantly best on either.
- **Fine-tuning helps synthetic data out of domain** (+0.050 and +0.044 on HMDB51, both significant) but brings no significant gain on FallOrNot. This is consistent with OmniFall's use of fine-tuning with synthetic data.
- **Caveats:** one seed (fine-tuning results vary between runs); training loss approached zero, so models may overfit; hyper-parameters were not tuned.

---

# Run 7: fine-tuning repeated with seeds 1 and 2

Same settings as run 6; only the random seed changes (seed 0 = run 6). Copied from the Colab text output (full training logs included; seed-0 rows reproduce run 6 exactly).

**Pre-registered rule applied to every seed** (fine-tuned synthetic + real vs. the best frozen probe on each test set):

| Seed | FallOrNot ΔAUROC | HMDB51 ΔAUROC | Rule |
|---|---|---|---|
| 0 (primary) | −0.002 [−0.040, +0.037] | +0.028 [−0.003, +0.058] | not met |
| 1 | −0.003 [−0.037, +0.034] | +0.049 [+0.020, +0.075] | not met |
| 2 | −0.011 [−0.050, +0.027] | +0.035 [+0.004, +0.065] | not met |

**AUROC across 3 seeds, mean ± s.d.:**

| Fine-tuned on | FallOrNot | FallOrNot hard | HMDB51 | HMDB51 hard |
|---|---|---|---|---|
| Real | **0.902 ± 0.012** | **0.863 ± 0.018** | 0.920 ± 0.026 | 0.795 ± 0.067 |
| Synthetic | 0.838 ± 0.017 | 0.762 ± 0.024 | **0.960 ± 0.006** | **0.922 ± 0.015** |
| Synthetic + real | 0.894 ± 0.005 | 0.844 ± 0.011 | 0.954 ± 0.011 | 0.890 ± 0.027 |

Per seed (FallOrNot / HMDB51): real 0.906, 0.888, 0.912 / 0.902, 0.909, 0.950; synthetic 0.829, 0.859, 0.828 / 0.967, 0.959, 0.955; synthetic + real 0.897, 0.896, 0.888 / 0.945, 0.966, 0.952.

## Reading
- **The pre-registered rule is not met for any seed.** On FallOrNot, fine-tuning on synthetic + real never beats the best probe (differences −0.011 to −0.002). On HMDB51 it does for seeds 1 and 2 (CI above zero), so fine-tuning helps out of domain fairly consistently, but not in domain.
- **The ranking flip holds on average across seeds**: real-trained is best on FallOrNot (0.902 vs. synthetic 0.838), and synthetic-trained is best on HMDB51 (0.960 vs. real 0.920).
- **Synthetic + real is near the top on both test sets and is the most stable**: lowest seed-to-seed spread (s.d. 0.005 on FallOrNot, 0.011 on HMDB51), within 0.008 of the best mean on FallOrNot and 0.006 on HMDB51.
- **Real-only fine-tuning is unstable out of domain** (HMDB51 0.902–0.950; hard subset 0.751–0.872), consistent with only 268 real training clips.

---

# Run 8: second annotator (inter-annotator agreement)

**Setup:** a second person (a volunteer, not an author) labelled test clips with `Labeler(..., second_annotator=True)`: only the test clips the first annotator had labelled, in random order, without seeing the first annotator's labels, into a separate file (`labels_friend.csv`). Compared with `vision/agreement.py`. Copied from the Colab text output (counts and statistics verified to be mutually consistent).

- Test clips labelled by both: 100; compared (fall / not fall by both): 98; involving "unsure": 2.
- First annotator fall & second fall: 61 | first fall & second not: 4 | first not & second fall: 16 | both not: 17.
- Raw agreement: 0.796. **Cohen's κ = 0.498 [95% CI 0.297, 0.678]** (moderate agreement).
- Disagreements are asymmetric: in 16 of 20 the second annotator called a fall where the first did not (first annotator: 65 falls of 98; second: 77 of 98).

## Reading
- Agreement on "did a person fall?" is only moderate. Most disagreements are clips the first annotator judged not a fall and the second judged a fall, which suggests the boundary cases (near-falls, partial falls, sitting or being knocked down) are interpreted differently. The fall definition needs sharper written guidance, and test labels carry real uncertainty.
- **[TODO]** Robustness check: re-run the main comparisons with the second annotator's labels on these 100 test videos (`experiment.py --test-labels labels_friend.csv`) to see whether the conclusions change.

---

# Run 9: robustness — test set graded with the second annotator's labels

**Setup:** `experiment.py --test-labels labels_friend.csv`: the FallOrNot test set becomes the 99 videos the second annotator labelled fall / not fall (152 clips, 78 falls), graded with their labels; training labels unchanged. HMDB51 rows are unaffected (identical to run 5). Fine-tuned FallOrNot rows were skipped because one clip in this subset (a clip the first annotator marked "unsure") has no fine-tuned score. Copied from the Colab text output.

FallOrNot (second annotator's labels), AUROC [95% CI]:

| Representation | Staged | Synthetic | Real | Synthetic + real |
|---|---|---|---|---|
| CLIP appearance | 0.708 [0.637, 0.776] | 0.697 [0.619, 0.771] | 0.694 [0.622, 0.771] | 0.743 [0.663, 0.817] |
| CLIP temporal | 0.688 [0.609, 0.769] | 0.733 [0.659, 0.805] | 0.788 [0.717, 0.854] | 0.736 [0.659, 0.805] |
| VideoMAE | 0.698 [0.618, 0.766] | 0.710 [0.631, 0.785] | 0.678 [0.602, 0.756] | 0.739 [0.657, 0.811] |
| CLIP + VideoMAE | 0.714 [0.638, 0.791] | 0.731 [0.653, 0.800] | **0.797 [0.732, 0.863]** | 0.746 [0.668, 0.812] |

Zero-shot CLIP 0.664 [0.591, 0.734]. Staged in-domain (cross-validation) unchanged at 1.000.

Head-to-head, CLIP + VideoMAE (second annotator's labels): real vs. staged **+0.083 [+0.010, +0.157], p = 0.024**; real vs. synthetic +0.065 [−0.006, +0.137], p = 0.079; synthetic vs. staged +0.017 [−0.053, +0.091], p = 0.656; synthetic + real vs. real −0.051 [−0.107, +0.007], p = 0.083. CLIP temporal: real vs. staged **+0.100 [+0.020, +0.182], p = 0.010**.

Learning curve (CLIP + VideoMAE; staged-only 0.714): 4 falls 0.673 ± 0.082; 8: 0.709 ± 0.045; 15: 0.756 ± 0.036; 30: 0.759 ± 0.017; 60: 0.785 ± 0.023; 116: 0.800 ± 0.008; 131: 0.797.

Equal budget (CLIP + VideoMAE): synthetic peaks at 0.766 ± 0.022 (125 falls) and ends at 0.732 (2,136); real reaches 0.795 ± 0.007 (125) and 0.795 (131). "Synthetic falls needed to match 131 real falls: not reached."

## Reading
- **Holds under the second annotator:** the staged-to-real gap (1.00 in-domain vs. ≤ 0.80 on real falls); real-trained beating staged-trained (+0.083, p = 0.024; CLIP temporal +0.100, p = 0.010); the learning-curve shape (about 8 real falls match the staged set, 15 exceed it); synthetic data plateauing below real at equal budget.
- **Weaker under the second annotator:** the in-domain real-over-synthetic advantage shrinks to +0.065 and is no longer significant (p = 0.079), and synthetic + real is not significantly different from real.
- **Absolute numbers depend on the annotator:** the best AUROC drops from 0.899 to 0.797. The second annotator labels more borderline events as falls, which are harder to detect. Reported numbers should be read as specific to the labelling definition.

---

# Derived: shortfall from the best training source (no new run)

Arithmetic on runs 4, 5 and 7: for each test, best AUROC among training sources minus each source's AUROC. Probes = CLIP + VideoMAE; fine-tuned = mean of 3 seeds. "hard" = accident negatives only (FallOrNot `real_hard`) or the HMDB51 hard subset. Differences of point estimates; no confidence intervals.

| | Trained on | FallOrNot | FallOrNot hard | HMDB51 | HMDB51 hard | Max |
|---|---|---|---|---|---|---|
| Probe | Staged | 0.140 | 0.145 | 0.018 | 0.047 | 0.145 |
| Probe | Synthetic | 0.085 | 0.088 | 0 | 0 | 0.088 |
| Probe | Real | 0 | 0 | 0.029 | 0.103 | 0.103 |
| Probe | Synthetic + real | 0.027 | 0.012 | 0.016 | 0.025 | **0.027** |
| Fine-tuned | Synthetic | 0.064 | 0.101 | 0 | 0 | 0.101 |
| Fine-tuned | Real | 0 | 0 | 0.040 | 0.127 | 0.127 |
| Fine-tuned | Synthetic + real | 0.008 | 0.019 | 0.006 | 0.032 | **0.032** |

Note: by plain worst-case AUROC over the two full test sets, real-only (probe 0.888, fine-tuned 0.902) is as good as synthetic + real (0.872, 0.894). The robustness claim for synthetic + real rests on worst-case shortfall, driven by the hard subsets. Correction to run 7's reading: synthetic + real has the smallest seed spread on FallOrNot only; on HMDB51 synthetic-only varies least (s.d. 0.006 vs. 0.011).

---

# Run 10: compilation leakage check, compilation-level intervals, source-balanced method

`experiment.py --group-by compilation` (FallOrNot confidence intervals and paired tests resampled by source compilation instead of by video). Point estimates are unchanged from runs 3–7. Copied from the Colab text output.

## Leakage check
OOPS! clips are cut from YouTube compilations. 111 of the 268 real training clips come from a compilation that also appears in the test set; 157 clips (77 falls) do not. Real probes retrained on the compilation-disjoint subset, compared with 20 random subsets of the same size and class counts (AUROC on FallOrNot):

| Representation | Real, all 268 clips | Compilation-disjoint (157) | Random same size (mean ± s.d.) | Staged |
|---|---|---|---|---|
| CLIP appearance | 0.732 | 0.746 | 0.731 ± 0.019 | 0.723 |
| CLIP temporal | 0.866 | 0.838 | 0.853 ± 0.013 | 0.728 |
| VideoMAE | 0.787 | 0.788 | 0.777 ± 0.022 | 0.765 |
| CLIP + VideoMAE | 0.899 | 0.866 | 0.874 ± 0.014 | 0.759 |

**Reading:** no clear evidence of compilation leakage. The disjoint subset scores within about one standard deviation of random subsets of the same size (CLIP + VideoMAE 0.866 vs. 0.874 ± 0.014; CLIP temporal 0.838 vs. 0.853 ± 0.013, the largest gap). Most of the drop from 0.899 is the smaller training set. Trained only on compilations absent from the test set, real data (0.866) still beats synthetic (0.814) and staged (0.759).

## Compilation-level intervals (CLIP + VideoMAE unless noted)
- AUROC CIs: staged 0.759 [0.698, 0.814]; synthetic 0.814 [0.753, 0.869]; real 0.899 [0.858, 0.935]; synthetic + real 0.872 [0.825, 0.914].
- real vs. staged +0.139 [+0.085, +0.193], p < 0.001; real vs. synthetic +0.085 [+0.030, +0.137], p = 0.001; synthetic + real vs. real −0.027 [−0.068, +0.012], p = 0.190.
- **synthetic vs. staged +0.054 [−0.000, +0.110], p = 0.053: no longer significant** (video-level: p = 0.045). CLIP temporal: synthetic vs. staged +0.067 [+0.007, +0.126], p = 0.030; real vs. staged +0.138 [+0.073, +0.198], p < 0.001; real vs. synthetic +0.071 [+0.019, +0.122], p = 0.006.
- CLIP appearance real vs. staged +0.009, p = 0.759; VideoMAE real vs. staged +0.022, p = 0.492. Staged-trained encoder comparisons: all p ≥ 0.296.
- Fine-tuning rule, FallOrNot side: seed 0 −0.002 [−0.042, +0.040]; seed 1 −0.003 [−0.038, +0.033]; seed 2 −0.011 [−0.046, +0.027]. Still not met for any seed.

## Source-balanced synthetic + real probe (rule fixed in advance on 10 Nov 2026)
Rule: must beat the plain synthetic + real mixture on both test sets, paired 95% CI above zero (CLIP + VideoMAE).
- FallOrNot: +0.016 [+0.005, +0.028], p = 0.004. HMDB51: +0.002 [−0.004, +0.008], p = 0.491. **RESULT: NOT MET.**
- Balanced vs. real-only on FallOrNot: −0.011 [−0.044, +0.022], p = 0.520. Balanced vs. synthetic-only on HMDB51: −0.014 [−0.032, +0.002], p = 0.091.

| Balanced probe | FallOrNot | FallOrNot hard | HMDB51 | HMDB51 hard |
|---|---|---|---|---|
| CLIP + VideoMAE | 0.888 [0.842, 0.928] | 0.862 [0.794, 0.918] | 0.903 [0.872, 0.936] | 0.851 [0.805, 0.893] |
| CLIP temporal | 0.841 | 0.827 | 0.853 | 0.784 |
| VideoMAE | 0.859 | 0.815 | 0.900 | 0.863 |
| CLIP appearance | 0.769 | 0.765 | 0.758 | 0.658 |

Recall at 5% FPR, CLIP + VideoMAE: 0.588 [0.361, 0.722] on FallOrNot (plain mixture 0.480; real 0.431); 0.610 on HMDB51 (same as the plain mixture).

**Reading:** balancing gives a small, significant gain on FallOrNot and no change on HMDB51, so the rule is not met. The balanced probe is statistically indistinguishable from the best single source on each test set (real on FallOrNot, synthetic on HMDB51). Its largest shortfall from the best of the four original sources is 0.029 (HMDB51 hard), about the same as the plain mixture (0.027).

---

# Run 11: third test set, OmniFall OF-ItW (independent annotation team)

`ofitw_prep.py` + `experiment.py --ofitw --group-by compilation`. OF-ItW = 818 OOPS! videos annotated by OmniFall's team with segment-level classes. Clips: one 2 s clip per "fall" segment (positive); one per other segment of at least 1 s that is not "fallen" and does not overlap a fall (negative). All 818 videos were found (2,405 clips, 914 falls). **Test split: 571 videos, 1,671 clips, 636 falls. Train + val split: 734 clips, 278 falls.** 20 of our own real training clips belong to OF-ItW test videos and were removed before training "real (ours)". Intervals resampled by compilation. Copied from the Colab text output.

AUROC [95% CI] on the OF-ItW test split:

| Representation | Staged | Synthetic | Real (ours, 248 clips) | Real (OF-ItW train + val) | Synthetic + real (OF-ItW) |
|---|---|---|---|---|---|
| CLIP appearance | 0.746 [0.725, 0.769] | 0.727 [0.706, 0.750] | 0.781 [0.762, 0.803] | 0.829 [0.808, 0.852] | 0.825 [0.805, 0.847] |
| CLIP temporal | 0.785 [0.765, 0.806] | 0.846 [0.826, 0.865] | 0.900 [0.885, 0.914] | 0.925 [0.910, 0.938] | 0.883 [0.866, 0.900] |
| VideoMAE | 0.771 [0.750, 0.793] | 0.817 [0.796, 0.836] | 0.844 [0.827, 0.862] | 0.909 [0.896, 0.922] | 0.899 [0.884, 0.914] |
| CLIP + VideoMAE | 0.815 [0.795, 0.833] | 0.872 [0.855, 0.890] | 0.927 [0.915, 0.939] | **0.951 [0.941, 0.960]** | 0.929 [0.917, 0.941] |

Zero-shot CLIP 0.731 [0.707, 0.753]. Recall at 5% FPR, CLIP + VideoMAE: staged 0.362, synthetic 0.546, real (ours) 0.687, real (OF-ItW) 0.796, synthetic + real 0.714.

Head-to-head, CLIP + VideoMAE (paired, by compilation; all p < 0.001):

| Comparison | ΔAUROC | 95% CI |
|---|---|---|
| Real (OF-ItW train) vs. synthetic | +0.078 | [+0.062, +0.095] |
| Real (OF-ItW train) vs. staged | +0.136 | [+0.117, +0.155] |
| Real (ours) vs. synthetic | +0.055 | [+0.039, +0.071] |
| Real (ours) vs. staged | +0.113 | [+0.094, +0.131] |
| Synthetic vs. staged | +0.058 | [+0.040, +0.077] |
| Synthetic + real (OF-ItW) vs. real (OF-ItW) | −0.022 | [−0.031, −0.013] |
| Synthetic + real (OF-ItW) vs. synthetic | +0.056 | [+0.044, +0.067] |

**Our labels vs. OmniFall's labels** on the 28 videos both labelled (ours: fall / not; theirs: a fall segment inside our event window [t−1, t+1] or not): fall & fall 25, fall & not 1, not & fall 0, not & not 2. Raw agreement 0.964; Cohen's κ = 0.781 (computed from these counts; n is small, and OF-ItW contains only fall-containing videos, so the sample is skewed towards falls).

## Reading
- **The FallOrNot ordering is confirmed on a 3.5× larger test set with independent labels:** staged 0.815 < synthetic 0.872 < real 0.927–0.951, every step significant (p < 0.001).
- **Synthetic beats staged here with p < 0.001** (+0.058), so OmniFall's finding is reproduced on its own benchmark; the borderline result on FallOrNot (p = 0.053) was a matter of sample size.
- **Our hand-labelled real clips transfer to OmniFall's labels:** probes trained on our 248 clips reach 0.927, close to probes trained on OmniFall's own 734 training clips (0.951). The label definitions are compatible, and where both teams labelled the same video they agree on 27 of 28.
- **Plain mixing hurts here:** adding 5,136 synthetic clips to the real ones lowers AUROC significantly (−0.022). Motivates a better way to combine the two sources.
- **Caveats:** OF-ItW negatives are other moments of the same fall videos (walking, standing, getting up), not other accidents, so it complements FallOrNot's accident negatives. Both are OOPS! footage, so the real-data advantage is again in-domain; HMDB51 remains the only out-of-domain test. OF-ItW's own train and test splits share compilations (117).

# Run 12: anchored probe (method test, rule fixed in advance: MET)

`adapt.py` (no GPU; commit `7d539e5`). The success rule was committed to `AGENTS.md` (commit `c666999`) before the script produced any result on real data. Copied from the Colab text output.

**Method.** Logistic probe on CLIP + VideoMAE features, trained on k real falls + k real non-falls, with its weights penalised for moving away from the synthetic-trained probe's weights (instead of from zero). Penalty strength C chosen from {0.001, 0.01, 0.1, 1, 10} by video-grouped cross-validation on the training clips only. Baselines: real-only probe on the same clips; synthetic-only probe. Ten random draws per budget (seed 0); mean ± s.d. over draws.

**FallOrNot** (real clips drawn from our 268-clip training pool), AUROC:

| k | Anchored | Real-only | Synthetic-only | Median C |
|---|---|---|---|---|
| 4 | 0.813 ± 0.005 | 0.687 ± 0.124 | 0.814 | 0.001 |
| 8 | 0.819 ± 0.012 | 0.756 ± 0.063 | 0.814 | 0.1 |
| 15 | 0.834 ± 0.015 | 0.791 ± 0.022 | 0.814 | 1 |
| 30 | 0.853 ± 0.014 | 0.826 ± 0.030 | 0.814 | 10 |
| 60 | 0.875 ± 0.010 | 0.871 ± 0.010 | 0.814 | 10 |
| 125 | 0.893 ± 0.005 | 0.892 ± 0.005 | 0.814 | 10 |

Mean over budgets: anchored 0.848, real-only 0.804, synthetic-only 0.814. Every budget within 0.005 of the better baseline: yes. Rule met on FallOrNot.

**OF-ItW test split** (real clips drawn from OF-ItW train + val), AUROC:

| k | Anchored | Real-only | Synthetic-only | Median C |
|---|---|---|---|---|
| 4 | 0.876 ± 0.005 | 0.794 ± 0.046 | 0.873 | 0.001 |
| 8 | 0.877 ± 0.006 | 0.825 ± 0.024 | 0.873 | 0.001 |
| 15 | 0.891 ± 0.012 | 0.847 ± 0.024 | 0.873 | 10 |
| 30 | 0.907 ± 0.019 | 0.891 ± 0.014 | 0.873 | 10 |
| 60 | 0.921 ± 0.008 | 0.908 ± 0.016 | 0.873 | 10 |
| 125 | 0.935 ± 0.007 | 0.937 ± 0.007 | 0.873 | 10 |

Mean over budgets: anchored 0.901, real-only 0.867, synthetic-only 0.873. Every budget within 0.005 of the better baseline: yes. Rule met on OF-ItW.

**METHOD RESULT: SUCCESS** (rule met on both test sets).

**Full real pool** (secondary, not part of the rule; paired bootstrap, by compilation on the OOPS!-based sets, by clip on HMDB51):

| Test set | Anchored | Real-only | Synthetic-only | Anchored vs. real | Anchored vs. synthetic |
|---|---|---|---|---|---|
| FallOrNot | 0.899 | 0.899 | 0.814 | −0.000 [−0.021, +0.019], p = 0.957 | +0.084 [+0.042, +0.125], p < 0.001 |
| OF-ItW | 0.943 | 0.951 | 0.873 | −0.008 [−0.012, −0.003], p = 0.002 | +0.071 [+0.055, +0.085], p < 0.001 |
| HMDB51 | 0.902 | 0.888 | 0.917 | +0.014 [−0.000, +0.028], p = 0.055 | −0.015 [−0.039, +0.006], p = 0.139 |

## Reading
- **First test specified in advance that was met.** With 4 to 30 real falls the anchored probe is far better and far more stable than a real-only probe (k = 4: 0.813 ± 0.005 vs. 0.687 ± 0.124 on FallOrNot; 0.876 vs. 0.794 on OF-ItW), and it never drops meaningfully below the synthetic-only probe.
- **The gain is a small-data gain.** From k = 60 upwards the anchored and real-only probes are level (within 0.004 on FallOrNot; at k = 125 on OF-ItW the anchored probe is 0.002 lower).
- **With the full real pool it does not beat real-only:** equal on FallOrNot, and significantly, though slightly, worse on OF-ItW (−0.008, p = 0.002). Do not claim it is the best method at every budget.
- **HMDB51 (out of domain):** 0.902, between real-only (0.888) and synthetic-only (0.917); neither difference is significant. It recovers about half of the gap that real-only training loses there.
- **Caveats:** the technique (regularising towards a source model, "biased regularisation") is known in transfer learning; what is new is the test on this problem, not the idea. Plain pooling of synthetic and real clips was not run at these small budgets, so the table does not compare against it. The s.d. is over random training draws with the test set fixed. The OF-ItW synthetic-only value here (0.873) differs from run 11 (0.872) in the third decimal because this script standardises features with synthetic statistics only.

# Run 13: anchored fine-tuning (method test for end-to-end models, rule fixed in advance: MET)

`adapt_ft.py` (Colab L4; commit `3ef3562`, which also committed the rule to `AGENTS.md` before any result). Copied from the Colab text output.

**Method.** Start from the synthetic-fine-tuned VideoMAE-B (`model_synthetic`, seed 0, run 6) and fine-tune end to end on k real falls + k real non-falls with loss = cross-entropy + (0.01 / 2) × squared distance of all weights from the synthetic model (L2-SP); no other weight decay. Baselines: real-only fine-tuning from the Kinetics checkpoint on the same clips (run-6 recipe); the synthetic model itself. Learning rate 5e-5, batch 8, one-cycle schedule, steps = max(60, 5 × clips / 8) for both. Three random draws per budget (seed 0); mean ± s.d. over draws.

FallOrNot AUROC (synthetic-only: 0.829, identical to run 6 seed 0), and HMDB51 AUROC for the first draw of each budget (synthetic-only: 0.967):

| k | Anchored | Real-only | HMDB51 anchored (draw 0) | HMDB51 real-only (draw 0) |
|---|---|---|---|---|
| 8 | 0.826 ± 0.003 | 0.652 ± 0.042 | 0.960 | 0.880 |
| 30 | 0.855 ± 0.007 | 0.813 ± 0.013 | 0.941 | 0.888 |
| 125 | 0.898 ± 0.010 | 0.894 ± 0.008 | 0.942 | 0.815 |

Mean over budgets: anchored 0.860, real-only 0.786, synthetic-only 0.829. Every budget within 0.01 of the better baseline: yes. **FINE-TUNING METHOD RESULT: SUCCESS.**

## Reading
- **The anchoring result carries over from frozen probes to end-to-end fine-tuning.** With 8 real falls, anchored fine-tuning keeps the synthetic model's level (0.826 vs. 0.829) while real-only fine-tuning collapses (0.652); with 30 it is ahead of both (0.855 vs. 0.813 and 0.829); with 125 it matches real-only (0.898 vs. 0.894, within the spread over draws).
- **Out of domain (HMDB51, secondary, one draw per budget, no intervals):** anchored models stay at 0.94 to 0.96, far above real-only fine-tuning on the same clips (0.82 to 0.89), but below the unadapted synthetic model (0.967): adapting to OOPS! clips costs about 0.01 to 0.03 on HMDB51.
- **Caveats:** one test set in the rule (FallOrNot; pixel clips for OF-ItW were not stored); three draws; one starting model (seed 0); the penalty strength 0.01 was fixed, not tuned, and no ablation separates the effect of the synthetic starting point from the effect of the penalty; the tolerance was 0.01 (0.005 for the probe test in run 12). L2-SP is an existing technique (Li et al., 2018).

# Run 14: ablation of run 13 (same synthetic starting model, no penalty)

`adapt_ft.py` at commit `94e53ca` in the `Research` repository (Colab L4). How to read the result was written in `AGENTS.md` before the run. "Init-only" = the same synthetic starting model, clips, steps and learning rate as anchored fine-tuning, but ordinary weight decay (0.05) and no pull towards the synthetic weights. Anchored and real-only columns are the saved run 13 results (same draws). Copied from the Colab text output.

| k | Anchored | Real-only | Init-only (no penalty) | HMDB51 anchored (draw 0) | HMDB51 real-only (draw 0) | HMDB51 init-only (draw 0) |
|---|---|---|---|---|---|---|
| 8 | 0.826 ± 0.003 | 0.652 ± 0.042 | 0.826 ± 0.003 | 0.960 | 0.880 | 0.959 |
| 30 | 0.855 ± 0.007 | 0.813 ± 0.013 | 0.855 ± 0.007 | 0.941 | 0.888 | 0.940 |
| 125 | 0.898 ± 0.010 | 0.894 ± 0.008 | 0.900 ± 0.010 | 0.942 | 0.815 | 0.940 |

Synthetic-only: FallOrNot 0.829, HMDB51 0.967. Mean over budgets: anchored 0.860, init-only 0.860, real-only 0.786.

## Reading (as fixed in advance)
- **Init-only is within 0.01 of anchored at every budget** (differences 0.000, 0.000, +0.002 on FallOrNot; at most 0.002 on HMDB51). So, for fine-tuning, **the gain comes from starting at the synthetic-fine-tuned model; the L2-SP penalty at strength 0.01 adds nothing measurable.** The paper must say this and must not credit the penalty.
- The practical recipe for fine-tuned models is therefore simply sequential fine-tuning: synthetic first, then the few real clips, with a short schedule (60 to 156 steps at learning rate 5e-5).
- This does not carry over automatically to the probe result of run 12, where the penalty is the only thing that links the probe to the synthetic solution (a convex problem has no "starting point" effect) and its strength was cross-validated.
- **Caveats:** one penalty strength (0.01) was tried, so a stronger penalty might behave differently; short schedules may be what keeps the no-penalty model close to its start; three draws, one starting model, one in-domain test set.

# Run 15: zero-label adaptation from unlabelled accident video (rule fixed in advance: MET)

`nolabel.py` at commit `4d18779` in the `Research` repository (Colab, no GPU). The rule was committed to `AGENTS.md` in the same commit, before any result on real data. Copied from the Colab text output.

**Method (no human label is read).** Unlabelled pool: every embedded OOPS! video except all FallOrNot test videos and all OF-ItW videos: **7,063 accident clips + 3,473 pre-accident clips from 7,063 videos**. Round 0: score the accident clips with the synthetic-trained probe. Each round: top 20% of accident clips = pseudo-falls (1,413); bottom 20% of accident clips + all pre-accident clips = pseudo-non-falls (4,886); train a fresh CLIP + VideoMAE probe on these real clips only; re-score. Three rounds. Deterministic, one run.

AUROC:

| Method | FallOrNot | OF-ItW test | HMDB51 |
|---|---|---|---|
| Synthetic-only | 0.814 | 0.872 | 0.917 |
| **Zero-label (ours)** | **0.894** | **0.937** | **0.953** |
| Ablation: 1 round | 0.877 | 0.924 | 0.945 |
| Ablation: no pre-accident clips | 0.837 | 0.875 | 0.934 |
| Ablation: pre-accident clips as the only negatives | 0.896 | 0.939 | 0.951 |
| Ablation: synthetic + pre-accident negatives, no pseudo-labels | 0.782 | 0.843 | 0.902 |
| Robustness: pool without test-set compilations (1,110 accident clips) | 0.879 | 0.927 | 0.938 |
| Reference: real probe trained on human labels (full pool) | 0.899 | 0.951 | 0.888 |

Paired tests (by compilation on FallOrNot and OF-ItW, by clip on HMDB51):

| Test set | Zero-label vs. synthetic-only | Zero-label vs. human-labelled real probe | Compilation-disjoint pool vs. synthetic-only |
|---|---|---|---|
| FallOrNot | +0.081 [+0.033, +0.129], p < 0.001 | −0.004 [−0.039, +0.031], p = 0.832 | +0.065 [+0.014, +0.114], p = 0.015 |
| OF-ItW | +0.065 [+0.052, +0.077], p < 0.001 | −0.014 [−0.024, −0.003], p = 0.009 | +0.054 [+0.040, +0.068], p < 0.001 |
| HMDB51 (secondary) | +0.036 [+0.016, +0.056], p = 0.001 | not tested | not tested |

Share of the synthetic-to-labelled-real gap closed without labels: FallOrNot 0.95, OF-ItW 0.82.

Real-only learning curves recomputed in this run (k labelled falls + k non-falls, mean of 10 draws): FallOrNot 4: 0.699, 8: 0.742, 15: 0.770, 30: 0.827, 60: 0.861, 125: 0.894; OF-ItW 4: 0.794, 8: 0.825, 15: 0.847, 30: 0.891, 60: 0.908, 125: 0.937. **Label equivalent: 125 labelled falls (plus 125 non-falls) on both test sets**, the largest budget on the curve.

**ZERO-LABEL RESULT: SUCCESS** (rule met on both test sets).

## Reading
- **Unlabelled real accident video closes most of the synthetic-to-real gap with zero human labels.** On FallOrNot the zero-label probe (0.894) is statistically indistinguishable from the probe trained on our 268 hand-labelled clips (0.899); on OF-ItW it is slightly but significantly lower than the probe trained on OF-ItW's own labels (0.937 vs. 0.951).
- **It is the first training recipe in this project that is best or near-best on all three test sets.** On HMDB51 it scores 0.953, above synthetic-only (0.917, p = 0.001) and above the human-labelled real probe (0.888; difference not tested). The earlier "ranking flip" (real wins in domain, synthetic wins out of domain) does not apply to it.
- **What makes it work (ablations).** The free pre-accident negatives are the key ingredient: without them the gain almost vanishes (0.837 / 0.875), while using them as the only negatives is as good as the full method (0.896 / 0.939), so the bottom-20% accident clips are unnecessary. Free negatives alone, added to synthetic data without pseudo-falls, hurt (0.782 / 0.843): real pseudo-falls and real free negatives are needed together. Three rounds help over one (+0.017 / +0.013).
- **Leakage.** With a pool restricted to compilations absent from both test sets (only 1,110 accident clips, 16% of the pool) the gain is smaller but still significant on both (+0.065, +0.054); the smaller pool and the removal of shared compilations are confounded.
- **Caveats.** One deterministic run with one fixed setting (20%, 3 rounds); frozen-feature probes only, not fine-tuned models; FallOrNot and OF-ItW are both OOPS! footage like the pool, so those two tests are in-domain for the unlabelled data (HMDB51 is not); only 3,473 of 7,063 videos have a pre-accident clip; the pool includes the 268 clips we labelled (their labels are not read); pseudo-labelling is a known technique family, and "no prior work does this for fall detection" rests on a short search and is not yet verified. The learning curve here uses a different random stream from run 12, so its real-only values differ slightly (e.g. k = 8: 0.742 here, 0.756 in run 12).

# Run 16: exploratory sensitivity analysis of run 15 (declared in advance as exploratory)

`nolabel.py --explore` at commit `ec95a2b` (`Research` repository; Colab, no GPU). Not a pass/fail test. One thing varied at a time around the fixed setting (20%, 3 rounds, synthetic start). Copied from the Colab text output. Pseudo-fall precision = share of real falls among the pseudo-falls that happen to carry one of our human labels (labels used only for this measurement).

| What is varied | Setting | FallOrNot | OF-ItW | HMDB51 | Pseudo-fall precision |
|---|---|---|---|---|---|
| Rounds (20%) | 1 | 0.877 | 0.924 | 0.945 | 0.80 (n = 50) |
| | 2 | 0.891 | 0.934 | 0.954 | 0.89 (n = 47) |
| | **3 (fixed setting)** | **0.894** | **0.937** | **0.953** | 0.96 (n = 46) |
| | 4 | 0.896 | 0.938 | 0.950 | 0.96 (n = 45) |
| | 5 | 0.898 | 0.941 | 0.950 | 0.98 (n = 45) |
| Fraction pseudo-labelled (3 rounds) | 5% | 0.877 | 0.928 | 0.943 | |
| | 10% | 0.888 | 0.931 | 0.941 | |
| | 30% | 0.893 | 0.940 | 0.957 | |
| | 40% | 0.878 | 0.930 | 0.957 | |
| Pool size (mean of 3 random subsets) | 2% (about 141 videos) | 0.816 | 0.866 | 0.924 | |
| | 5% (about 353 videos) | 0.864 | 0.912 | 0.945 | |
| | 10% (about 706 videos) | 0.874 | 0.918 | 0.944 | |
| | 25% (about 1,765 videos) | 0.890 | 0.930 | 0.949 | |
| | 50% (about 3,531 videos) | 0.894 | 0.937 | 0.950 | |
| Starting scorer | CLIP zero-shot text prompts (no synthetic data at all) | 0.899 | 0.936 | 0.941 | |
| | **Random scores (control, one seed)** | **0.882** | **0.931** | **0.920** | |

## Reading
- **The control changes the interpretation of run 15.** Starting from RANDOM scores still gives 0.882 / 0.931 / 0.920, i.e. most of the gain over synthetic-only (0.814 / 0.872 / 0.917) on the two OOPS!-based tests. The fall-specific starting point adds only +0.012 (FallOrNot), +0.006 (OF-ItW) and +0.033 (HMDB51) over the random start (single seed, no intervals). As fixed in advance in AGENTS.md: **the paper must say that most of the in-domain gain comes from the accident-versus-ordinary-activity structure (accident clips vs. pre-accident clips), not from fall-specific pseudo-labels.** A plausible reason: most OOPS! accidents are falls (102 of 165 FallOrNot test videos), so "is this the accident moment?" is already a good fall detector on these test sets.
- **Synthetic data is not needed for this recipe:** starting from CLIP's zero-shot text prompts gives 0.899 / 0.936 / 0.941.
- **Robust to its settings:** 2 to 5 rounds and 10 to 30% all lie within 0.01 of the fixed setting on the in-domain tests; more rounds raise pseudo-fall precision from 0.80 to 0.98 (small n). The fixed setting stays the headline; 5 rounds and 30% look slightly better in places and must not replace it.
- **Pool size:** about 350 unlabelled videos already give 0.864 / 0.912; gains level off around 1,800 to 3,500 videos; 141 videos give nothing.
- **Open question this run cannot answer (run 17):** whether the zero-label probe separates falls from OTHER ACCIDENTS (FallOrNot hard subset) or only accidents from ordinary activity.

# Run 17: hard-subset analysis of the zero-label probe (reading fixed in advance)

`nolabel.py --hard` at commit `152c81a` (`Research` repository; Colab, no GPU). How to read the outcome was written in `AGENTS.md` before the run. FallOrNot hard = accident-moment clips only (102 falls vs. 63 other accidents); HMDB51 hard = falls vs. fall-like classes (320 clips). Copied from the Colab text output. Random start: mean ± s.d. over 5 seeds. "Accident detector" = one probe trained on all unlabelled accident clips (positive) vs. all pre-accident clips (negative), no pseudo-labels.

| Method | FallOrNot | OF-ItW | HMDB51 | FallOrNot hard | HMDB51 hard |
|---|---|---|---|---|---|
| Synthetic-only | 0.814 | 0.872 | 0.917 | 0.772 | 0.880 |
| **Zero-label (ours)** | 0.894 | 0.937 | 0.953 | 0.835 | 0.920 |
| Start from CLIP zero-shot prompts (no synthetic data) | 0.899 | 0.936 | 0.941 | 0.840 | 0.885 |
| Start from random scores (control, 5 seeds) | 0.865 ± 0.011 | 0.931 ± 0.001 | 0.914 ± 0.013 | 0.767 ± 0.021 | 0.886 ± 0.029 |
| Accident detector (no pseudo-labels) | 0.849 | 0.926 | 0.882 | 0.724 | 0.884 |
| Real probe trained on our human labels (268 clips) | 0.899 | 0.931* | 0.888 | 0.860 | 0.777 |

\* This script scores our 268-clip probe on OF-ItW without removing the 20 clips that belong to OF-ItW test videos (run 11 removes them and reports 0.927); use run 11's number in the paper.

Paired tests, AUROC difference [95% CI], p:

| Comparison | FallOrNot hard | HMDB51 hard |
|---|---|---|
| Zero-label vs. synthetic-only | +0.064 [−0.005, +0.131], p = 0.065 | +0.039 [+0.014, +0.066], p = 0.005 |
| Zero-label vs. random-start control (seed 1) | +0.043 [−0.003, +0.092], p = 0.068 | +0.031 [+0.008, +0.057], p = 0.010 |
| Zero-label vs. accident detector | +0.112 [+0.044, +0.186], p = 0.001 | +0.036 [+0.003, +0.070], p = 0.031 |
| Zero-label vs. human-labelled real | −0.025 [−0.073, +0.022], p = 0.290 | +0.143 [+0.101, +0.185], p < 0.001 |
| Accident detector vs. synthetic-only | −0.048 [−0.150, +0.042], p = 0.319 | +0.003 [−0.034, +0.039], p = 0.859 |

## Reading (following the rules fixed in advance)
- **(a) On FallOrNot hard the zero-label probe is NOT significantly above synthetic-only** (+0.064, interval [−0.005, +0.131], p = 0.065). As fixed in advance, the paper must say that in-domain discrimination of falls from other accidents by the zero-label probe is not established, and that the significant full-test gain is mainly accident-vs-ordinary-activity. Labels are not shown to be dispensable for the hard cases (human-labelled real: 0.860; difference −0.025, n.s.; n = 165, so the test is weak in both directions).
- **(b) Pseudo-labelling adds something beyond accident timing.** The plain accident detector is clearly worse than the zero-label probe on FallOrNot (0.849 vs. 0.894), HMDB51 (0.882 vs. 0.953) and both hard subsets (+0.112, p = 0.001; +0.036, p = 0.031), and on the hard subsets it is no better than synthetic-only. So the accident-timing signal alone is not the method.
- **(c) Not met as stated** (zero-label does not beat both synthetic-only and the random start on FallOrNot hard with intervals above zero; both are p ≈ 0.07). On HMDB51 hard it does beat both (p = 0.005, p = 0.010) and beats human-labelled real training by +0.143.
- **Random-start control over 5 seeds:** 0.865 ± 0.011 on FallOrNot (the single seed in run 16 gave 0.882, at the upper end), 0.767 ± 0.021 on FallOrNot hard (no better than synthetic-only, 0.772). The random start therefore learns accident detection, and the fall-specific starting scorer is what lifts the hard subsets (0.835 / 0.840 with the synthetic or zero-shot start).
- **Summary:** zero-label = accident-timing structure (most of the easy gain) + a fall-specific starting scorer (the hard-subset gain: significant out of domain, borderline in domain).

# Addendum: values quoted in the paper that come from the full Colab output of runs 10–11

The paper quotes two numbers that appear in the complete `experiment.py` output (pasted by the owner for runs 10 and 11) but were not copied into the run tables above: staged-trained VideoMAE probe on FallOrNot, recall at 5% false positives **0.304** [0.216, 0.510]; zero-shot CLIP on the FallOrNot hard subset, AUROC **0.686** [0.588, 0.782]. Rounded p-values in the paper (0.57, 0.52, 0.09, 0.29) correspond to 0.566, 0.520, 0.091 and 0.290 above; κ = 0.50 is 0.498. "62% of accidents are falls" = 102 falls among the 165 labelled videos of FallOrNot's random sample.

# Run 20: near-duplicate leakage check for the zero-label probe (reading fixed in advance)

`nolabel.py --dupes` at commit `62c2bb7` (`Research` repository; Colab, no GPU). How to read it was written in `AGENTS.md` before the run. Runs 18 and 19 (extended hard test, zero-label fine-tuning) were declared earlier and have not been reported yet; the numbering follows the order of declaration. Copied from the Colab text output.

**Method.** The same web clip can appear in several OOPS! compilations, which the compilation-disjoint pool of run 15 does not catch. For every unlabelled pool clip, similarity = highest cosine between its mean CLIP frame embedding and that of any FallOrNot-test or OF-ItW-test clip. Pool videos with any clip above a threshold are removed and the fixed zero-label pipeline (20%, 3 rounds, synthetic start) is rerun.

Similarity of pool clips to their closest test clip, quantiles: median 0.828, 90% 0.904, 99% 0.992, 99.9% 0.997, max 0.998. (Mean CLIP embeddings of unrelated clips are already highly similar, so only the top of this distribution indicates possible duplicates: about 1% of pool clips exceed 0.99.)

| Removed: pool videos with similarity above | Accident clips left | FallOrNot | OF-ItW | HMDB51 |
|---|---|---|---|---|
| nothing removed (run 15) | 7,063 | 0.894 | 0.937 | 0.953 |
| 0.98 | 6,841 (222 removed) | 0.893 | 0.937 | 0.956 |
| 0.95 | 6,709 (354 removed) | 0.892 | 0.936 | 0.954 |
| 0.92 | 6,524 (539 removed) | 0.892 | 0.937 | 0.954 |
| 0.90 | 6,112 (951 removed) | 0.888 | 0.938 | 0.955 |

Zero-label vs. synthetic-only (0.814 / 0.872), paired, by compilation:

| Threshold | FallOrNot | OF-ItW |
|---|---|---|
| none | +0.081 [+0.033, +0.129], p < 0.001 | +0.065 [+0.052, +0.077], p < 0.001 |
| 0.98 | +0.079 [+0.030, +0.128], p < 0.001 | +0.065 [+0.052, +0.077], p < 0.001 |
| 0.95 | +0.078 [+0.030, +0.125], p < 0.001 | +0.064 [+0.051, +0.077], p < 0.001 |
| 0.92 | +0.078 [+0.031, +0.128], p < 0.001 | +0.065 [+0.052, +0.078], p < 0.001 |
| 0.90 | +0.074 [+0.027, +0.121], p < 0.001 | +0.065 [+0.052, +0.078], p < 0.001 |

## Reading (as fixed in advance)
- **The gain persists.** At the pre-specified 0.95 threshold the zero-label probe is still significantly above synthetic-only on both in-domain tests (+0.078 and +0.064, both p < 0.001); removing up to 951 of the most test-like videos changes in-domain AUROC by at most 0.006 and HMDB51 not at all. Near-duplicate leakage does not explain the in-domain result.
- Possible near-duplicates do exist (about 1% of pool clips have a test clip with similarity above 0.99), so the check was worth running.
- **Caveats:** similarity uses CLIP appearance only (mean of frame embeddings), so re-encoded or cropped copies with different appearance could be missed; one deterministic run per threshold. This check addresses the unlabelled pool only; the labelled real training pool was checked at the compilation level in run 10.

# Run 18: extended hard test, falls vs. other accidents without labels (rule fixed in advance: MET)

`nolabel.py --hard2` at commit `e10ddec` (`Research` repository; Colab). Rule committed to `AGENTS.md` before the run (commit of 16 Nov 2026). Copied from the Colab text output.

**Setup.** Every human-labelled accident-moment clip is used as test data; ALL of those videos are removed from the unlabelled pool first (6,900 accident + 3,396 pre-accident clips remain), and the fixed zero-label pipeline (20%, 3 rounds, synthetic start) is retrained. A = FallOrNot hard (random sample, 165 clips: 102 falls, 63 other accidents); B = accident clips of our labelled training pool (selected by a CLIP zero-shot score, so not a random sample; 180 clips: 131 falls, 49 other accidents); C = A + B (primary; 345 clips: 233 falls, 112 other accidents). Random start: mean ± s.d. over 5 seeds.

AUROC, falls vs. other accidents:

| Method | A | B | C (primary) |
|---|---|---|---|
| Synthetic-only | 0.772 | 0.707 | 0.746 |
| **Zero-label (labelled videos removed from pool)** | **0.840** | **0.754** | **0.804** |
| Start from CLIP zero-shot prompts | 0.842 | 0.740 | 0.800 |
| Start from random scores (5 seeds) | 0.768 ± 0.013 | 0.685 ± 0.029 | 0.732 ± 0.019 |
| Accident detector (no pseudo-labels) | 0.729 | 0.678 | 0.707 |
| Human-labelled real probe | 0.860 | – | – |

Paired tests, AUROC difference [95% CI], p (by compilation):

| Comparison | A | B | C (primary) |
|---|---|---|---|
| Zero-label vs. synthetic-only | +0.069 [+0.002, +0.134], p = 0.046 | +0.047 [−0.018, +0.110], p = 0.145 | **+0.058 [+0.015, +0.103], p = 0.013** |
| Zero-label vs. random start (seed 1) | +0.066 [+0.017, +0.117], p = 0.010 | +0.026 [−0.028, +0.080], p = 0.359 | +0.049 [+0.013, +0.087], p = 0.005 |
| Zero-label vs. accident detector | +0.111 [+0.044, +0.185], p = 0.001 | +0.076 [−0.004, +0.154], p = 0.062 | +0.098 [+0.052, +0.144], p < 0.001 |
| Zero-label vs. human-labelled real | −0.020 [−0.068, +0.028], p = 0.419 | – | – |

**EXTENDED HARD TEST RESULT: SUCCESS** (rule met on C).

## Reading
- **Without any human label, the zero-label probe separates falls from other accidents better than synthetic training, in domain** (C: +0.058, p = 0.013), and better than both controls that carry no fall-specific signal (random start +0.049, p = 0.005; accident detector +0.098, p < 0.001). This replaces run 17's borderline result (A: +0.064, p = 0.065), which used a pool that still contained the labelled videos (unread).
- On A alone the gain is now just significant (+0.069, p = 0.046, a secondary comparison); on B alone it is not (+0.047, p = 0.145). B is model-selected and its other accidents were chosen to look like falls to CLIP, so it is the harder subset; the primary claim rests on C.
- The deficit to the human-labelled probe on A is not significant (−0.020, p = 0.419), with 165 clips.
- **Caveats:** one deterministic run; C mixes a random sample (A) with a model-selected one (B); the human-labelled reference cannot be evaluated on B or C because it is trained on B.

# Run 19: zero-label fine-tuning of VideoMAE (rule fixed in advance: MET)

`nolabel_ft.py` (select, prepare, train, score, report); the models were trained during the owner's run of the cell at commit `5ce5db5` and reported again at commit `e10ddec` (`Research` repository; Colab L4). Rule committed to `AGENTS.md` before any result. Copied from the Colab text output.

**Setup.** No human label is read for training. `select` repeats the fixed run-15 pipeline and takes the 1,000 highest-scoring accident clips (pseudo-falls) and 1,000 pre-accident clips chosen by name hash (free non-falls); 1,863 distinct videos. VideoMAE-B is fine-tuned end to end from the Kinetics checkpoint with the run-6 recipe, seeds 0, 1, 2. Compared with the synthetic and human-labelled real fine-tuned models of runs 6–7 (same seeds).

AUROC, mean ± s.d. over 3 seeds (population s.d.; run 7 reports the same means with sample s.d.):

| Fine-tuned on | FallOrNot | FallOrNot hard | HMDB51 | HMDB51 hard |
|---|---|---|---|---|
| **Zero-label (pseudo-falls + free negatives, no human label)** | **0.910 ± 0.004** | 0.852 ± 0.006 | 0.959 ± 0.003 | 0.893 ± 0.017 |
| Synthetic (runs 6–7) | 0.838 ± 0.014 | 0.762 ± 0.020 | 0.960 ± 0.005 | 0.922 ± 0.012 |
| Real, human labels (runs 6–7) | 0.902 ± 0.010 | 0.863 ± 0.015 | 0.920 ± 0.021 | 0.795 ± 0.055 |

Paired differences per seed (seed 0; seed 1; seed 2), AUROC difference [95% CI], p (by compilation on FallOrNot, by clip on HMDB51):

| Comparison | Test | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| Zero-label vs. synthetic | FallOrNot | +0.081 [+0.045, +0.118], p < 0.001 | +0.047 [+0.010, +0.083], p = 0.020 | +0.086 [+0.045, +0.128], p < 0.001 |
| | FallOrNot hard | +0.110 [+0.050, +0.169], p = 0.001 | +0.054 [−0.010, +0.114], p = 0.111 | +0.107 [+0.048, +0.167], p < 0.001 |
| | HMDB51 | −0.004, p = 0.79 | −0.001, p = 0.98 | −0.000, p = 0.98 |
| | HMDB51 hard | −0.028 [−0.064, +0.006], p = 0.130 | −0.010, p = 0.60 | −0.047 [−0.085, −0.010], p = 0.008 |
| Zero-label vs. real (human labels) | FallOrNot | +0.004, p = 0.82 | +0.017, p = 0.39 | +0.003, p = 0.87 |
| | FallOrNot hard | −0.021, p = 0.42 | +0.001, p = 0.99 | −0.013, p = 0.67 |
| | HMDB51 | +0.061 [+0.039, +0.086], p < 0.001 | +0.050 [+0.024, +0.078], p < 0.001 | +0.005, p = 0.51 |
| | HMDB51 hard | +0.149 [+0.105, +0.196], p < 0.001 | +0.149 [+0.103, +0.196], p < 0.001 | −0.003, p = 0.82 |

**ZERO-LABEL FINE-TUNING RESULT: SUCCESS** (mean above synthetic, and CI above zero for all three seeds on FallOrNot).

## Reading
- **The zero-label result holds for end-to-end fine-tuned video models.** With no human label, fine-tuned VideoMAE reaches 0.910 on FallOrNot, above synthetic fine-tuning (0.838; significant for every seed) and level with fine-tuning on our hand labels (0.902; no seed differs significantly). On FallOrNot's hard subset it is far above synthetic (0.852 vs. 0.762; significant for 2 of 3 seeds) and level with hand labels (0.863; n.s.).
- **Out of domain it keeps the synthetic model's level overall** (HMDB51 0.959 vs. 0.960) and is far above hand-labelled real fine-tuning for two seeds. On HMDB51's hard subset it is somewhat below synthetic fine-tuning (0.893 vs. 0.922; significant for one seed).
- **This is the first training recipe in the project that is best or level with the best on all four fine-tuned columns except HMDB51 hard**, where it is 0.029 below synthetic but 0.098 above hand-labelled real.
- **Caveats:** three seeds; one fixed selection of 2,000 clips (no ablation of 1,000 per class); the unlabelled pool includes the videos behind our 268 labelled training clips (labels not read); FallOrNot is OOPS! footage like the pool.

# Run 21: consequence-guided zero-label adaptation (rule fixed in advance: NOT MET)

`post_prep.py` + `nolabel.py --post` at commit `e10ddec` (`Research` repository; Colab L4). Rule committed to `AGENTS.md` before any result. Copied from the Colab text output.

**Setup.** `post_prep.py` embedded the clip [t+1, t+3] s after the accident for all 8,266 processed OOPS! videos (about 26 minutes at 5.4 videos/s). Consequence score = CLIP "person on the ground" minus "person upright" (prompts fixed in `post_prep.py`). Pool = run-18 pool (every human-labelled video removed): 6,900 accident clips, 5,891 with an after-clip, 3,396 pre-accident clips. In every self-training round accident clips are ranked by the mean of (rank of probe score, rank of consequence score), neutral 0.5 without an after-clip.

Diagnostic (labels used only to measure): the consequence score alone separates falls from other accidents on C with AUROC 0.681 (301 clips with an after-clip).

| Method | A: FallOrNot hard | B: training-pool accident clips | C (primary) | FallOrNot | OF-ItW | HMDB51 | HMDB51 hard |
|---|---|---|---|---|---|---|---|
| Synthetic-only | 0.772 | 0.707 | 0.746 | 0.814 | 0.872 | 0.917 | 0.880 |
| Zero-label (run-15 setting, same pool) | 0.840 | 0.754 | 0.804 | 0.897 | 0.937 | 0.952 | 0.915 |
| **Consequence-guided zero-label** | 0.861 | 0.745 | 0.812 | 0.897 | 0.930 | 0.921 | 0.846 |
| Consequence only, then standard rounds (secondary) | 0.861 | 0.761 | 0.819 | 0.905 | 0.941 | 0.922 | 0.849 |

Paired tests (by compilation): consequence-guided vs. zero-label: A +0.021 [−0.017, +0.060], p = 0.266; B −0.009, p = 0.730; **C +0.008 [−0.020, +0.036], p = 0.604**; FallOrNot +0.000, p = 0.922; OF-ItW −0.007 [−0.014, −0.000], p = 0.047. Consequence-guided vs. synthetic-only: A +0.090, p = 0.010; C +0.066, p = 0.004.

**CONSEQUENCE RESULT: NOT MET** (condition (i) failed: no significant gain over plain zero-label on C; condition (ii) held: at most 0.007 below on the full tests).

## Reading
- **The after-accident signal does not improve the zero-label method in domain** (+0.008 on C, p = 0.60), although on its own it carries some information about falls (AUROC 0.681 on C).
- **It hurts out of domain:** HMDB51 0.921 vs. 0.952 and HMDB51 hard 0.846 vs. 0.915 (not tested for significance; HMDB51 was not part of the rule). A plausible reason: it ties the pseudo-falls to OOPS!-specific after-states (people on the ground in fail videos), but this was not tested.
- The consequence-only start looks slightly better in domain (C 0.819) but has the same out-of-domain drop; it was a secondary variant and must not replace the rule's method.
- Report in the paper as a negative result; it supports keeping the simpler method.

# Run 22: fine-tuned models on OF-ItW, the third test set (rule fixed in advance: MET)

`nolabel_ft.py ofitw` at commit `974ef02` (`Research` repository; Colab L4). Rule committed to `AGENTS.md` before any result (commit `1c019d2`; a first attempt failed with a Google Drive read error before any score was computed, and the fix only moved the clip store to local disk). Copied from the Colab text output.

**Setup.** OF-ItW test split, clips as in `ofitw_prep.py`, excluding the 20 videos behind our labelled training clips: 552 videos, 1,613 clips, 614 falls. Every fine-tuned VideoMAE-B model already on Drive (runs 6–7 and 19, seeds 0–2) is scored. Paired tests resample by compilation.

| Fine-tuned on | AUROC mean ± s.d. (population, 3 seeds) | Seeds 0 / 1 / 2 |
|---|---|---|
| Synthetic | 0.936 ± 0.005 | 0.941 / 0.939 / 0.928 |
| Real, our hand labels (268 clips) | 0.957 ± 0.005 | 0.963 / 0.951 / 0.959 |
| Synthetic + real | 0.964 ± 0.001 | 0.964 / 0.963 / 0.966 |
| **Zero-label (no human label)** | **0.967 ± 0.003** | 0.963 / 0.967 / 0.971 |

Zero-label vs. synthetic, per seed: +0.023 [+0.014, +0.031], p < 0.001; +0.027 [+0.018, +0.037], p < 0.001; +0.042 [+0.032, +0.052], p < 0.001.
Zero-label vs. real (hand labels), per seed: +0.001 [−0.007, +0.008], p = 0.862; +0.016 [+0.005, +0.027], p = 0.004; +0.012 [+0.002, +0.020], p = 0.014.

**OF-ItW FINE-TUNING RESULT: SUCCESS.**

## Reading
- **On the third test set, labelled by an independent team, the zero-label fine-tuned model is the best of the four**: above synthetic for every seed, and above fine-tuning on our hand labels for two of three seeds (level for the third). Synthetic + real (0.964) is within 0.003; it was not tested against zero-label.
- Fine-tuned models score much higher on OF-ItW than frozen probes (best probe 0.951, trained on OF-ItW's own labels, run 11); the two are not directly compared here.
- **Combined with run 19:** fine-tuned on zero-label data, VideoMAE is best or level with the best on FallOrNot (0.910), OF-ItW (0.967) and HMDB51 (0.959), and level with hand labels on FallOrNot's hard subset; its one shortfall is HMDB51's hard subset (0.893 vs. synthetic 0.922).
- **Caveats:** three seeds; OF-ItW is OOPS! footage like the unlabelled pool (its videos were excluded from the pool); OF-ItW negatives are other moments of fall videos, not other accidents.

# Run 23: timestamp-free self-training (rule fixed in advance: NOT MET)

`window_prep.py` + `nolabel.py --windows` at commit `0f7b315` (`Research` repository; Colab L4). Rule committed to `AGENTS.md` before any result (commit `c8200fa`). Copied from the Colab text output.

**Setup.** Every 2 s window (1 s stride, at most 30) of all 7,063 pool videos: 60,512 windows (about 3.4 hours at 0.58 videos/s; the stream reconnected four times). The annotated accident time is never read: each round, each video's highest-scoring window is its event clip and the window starting 3 s earlier its non-fall; otherwise the fixed run-15 setting (synthetic start, 20%, 3 rounds).

| Method | FallOrNot | OF-ItW | HMDB51 |
|---|---|---|---|
| Synthetic-only | 0.814 | 0.872 | 0.917 |
| Self-training WITH annotated accident times (run 15, same videos) | 0.894 | 0.937 | 0.953 |
| **Timestamp-free self-training** | 0.836 | 0.912 | 0.948 |

Paired tests (by compilation): timestamp-free vs. synthetic-only: FallOrNot +0.022 [−0.022, +0.065], p = 0.337; OF-ItW +0.040 [+0.029, +0.051], p < 0.001. Timestamp-free vs. with timestamps: FallOrNot −0.058 [−0.093, −0.026], p < 0.001; OF-ItW −0.025 [−0.032, −0.017], p < 0.001.

Falls vs. other accidents (345 labelled accident clips, labelled videos removed from the pool): synthetic-only 0.746, timestamp-free 0.768; +0.022 [−0.015, +0.061], p = 0.271.

Diagnostic: the chosen event window lies within 1 s of the (unused) annotated time for 0.49 of videos after round 1 and 0.52 at the end, against 0.30 for a random window.

**TIMESTAMP-FREE RESULT: NOT MET** (significant on OF-ItW, not on FallOrNot).

## Reading
- **The annotated accident time matters.** Without it, the gain over synthetic training shrinks from +0.081 to +0.022 (n.s.) on FallOrNot and from +0.065 to +0.040 (still significant) on OF-ItW; with timestamps is significantly better on both. On HMDB51 the timestamp-free probe (0.948) stays close to the timestamped one (0.953).
- The detector finds the accident moment only about half the time (0.52 vs. 0.30 by chance), so many "free non-falls" are taken from the wrong place.
- This quantifies the method's main limitation: it needs human-annotated accident timestamps. The paper must keep "needs timestamps" as a limitation and report this run as a follow-up test that was not met.
- **Caveats:** one setting (3 s offset, argmax window), one run; a better localiser was not tried and must not be tuned on these test results.

# Run 24: zero-shot vision-language baselines (comparison; prompt and scoring fixed in advance)

`vlm_baseline.py` at commit `7aad716` (`Research` repository; Colab L4; transformers upgraded with `pip -U` in the same cell). Prompt, input frames and score fixed in `AGENTS.md` before any result. Copied from the Colab text output.

**Setup.** Prompt: "These are 8 frames, in time order, from a 2-second video clip. Does a person fall down in this clip? Answer with Yes or No." Input: frames 1, 3, …, 15 of the same stored 224 × 224 clips the fine-tuned models were scored on. Score: logit("Yes") − logit("No") for the first answer token. FallOrNot test (255 clips, 102 falls) and hard subset (165, 102), OF-ItW test without our labelled videos (1,613, 614), HMDB51 (1,100, 100; max over 2,740 windows) and hard subset (320, 100). Fine-tuned columns: mean ± s.d. over 3 seeds (runs 6, 7, 19, 22).

| Test set | Qwen3-VL-4B-Instruct (zero-shot) | InternVL3.5-8B-HF (zero-shot) | Fine-tuned synthetic | Fine-tuned hand labels | Fine-tuned self-trained (ours) |
|---|---|---|---|---|---|
| FallOrNot | 0.920 | **0.936** | 0.838 ± 0.014 | 0.902 ± 0.010 | 0.910 ± 0.004 |
| FallOrNot hard | 0.902 | **0.930** | 0.762 ± 0.020 | 0.863 ± 0.015 | 0.852 ± 0.006 |
| OF-ItW | 0.946 | 0.964 | 0.936 ± 0.005 | 0.957 ± 0.005 | **0.967 ± 0.003** |
| HMDB51 | 0.953 | 0.950 | **0.960 ± 0.005** | 0.920 ± 0.021 | 0.959 ± 0.003 |
| HMDB51 hard | 0.873 | 0.848 | **0.922 ± 0.012** | 0.795 ± 0.055 | 0.893 ± 0.017 |

Cost on one L4: Qwen3-VL-4B 0.22 s per clip (0.88 s on the first, smaller set, which includes warm-up), peak GPU memory 9.2 GB; InternVL3.5-8B 0.99–1.01 s per clip, 18.1 GB.

Paired differences VLM minus fine-tuned self-trained model, per seed (0; 1; 2), by compilation on OOPS!-based sets, by clip on HMDB51:

| Test set | Qwen3-VL-4B | InternVL3.5-8B |
|---|---|---|
| FallOrNot | +0.010 (p = 0.56); +0.015 (p = 0.44); +0.006 (p = 0.76) | +0.026 (p = 0.14); +0.031 (p = 0.12); +0.022 (p = 0.20) |
| FallOrNot hard | +0.047 (p = 0.069); +0.059 [+0.002, +0.119] (p = 0.042); +0.044 (p = 0.14) | **+0.075 [+0.024, +0.135] (p = 0.005); +0.087 [+0.032, +0.148] (p < 0.001); +0.072 [+0.021, +0.131] (p = 0.002)** |
| OF-ItW | **−0.017 (p = 0.003); −0.020 (p < 0.001); −0.024 (p < 0.001)** | +0.001 (p = 0.90); −0.003 (p = 0.57); −0.007 (p = 0.16) |
| HMDB51 | −0.010 (p = 0.32); −0.005 (p = 0.60); −0.001 (p = 0.85) | −0.013 (p = 0.19); −0.009 (p = 0.42); −0.005 (p = 0.53) |
| HMDB51 hard | −0.037 (p = 0.036); −0.027 (p = 0.16); +0.003 (p = 0.88) | **−0.062 (p < 0.001); −0.051 (p = 0.002)**; −0.021 (p = 0.20) |

Against the other fine-tuned models (selected): both VLMs beat fine-tuned synthetic on FallOrNot and its hard subset for every seed (p ≤ 0.010); InternVL beats fine-tuned hand labels on FallOrNot hard for 2 of 3 seeds (p = 0.007, 0.030; third p = 0.055); both VLMs are below fine-tuned synthetic on HMDB51 hard (InternVL for every seed, p ≤ 0.003).

## Reading (as fixed in advance: wherever a VLM is higher than our model, say so prominently)
- **A zero-shot 8B vision-language model, with no fall training at all, is the strongest detector on FallOrNot (0.936) and especially on falls vs. other accidents (0.930), significantly above our self-trained fine-tuned model on the hard subset for every seed.** Qwen3-VL-4B is also higher in point estimate on both (0.920, 0.902), significant on the hard subset for one seed. The abstract and introduction must state this.
- Our self-trained model is better on OF-ItW than Qwen3-VL-4B (significant for every seed) and level with InternVL; on HMDB51's hard subset it is better than InternVL (significant for 2 of 3 seeds); synthetic fine-tuning is best there.
- The VLMs cost 9–18 GB of GPU memory and 0.2–1 s per 2 s clip on an L4; our models' inference cost was not measured.
- **Implication:** the falls-vs-other-accidents problem that our method only partly solves is largely solved by a generic VLM zero-shot. A natural follow-up is to use the VLM as the teacher in place of the synthetic probe (not yet run).
- **Caveats:** one prompt each, fixed in advance; 8 frames at 224 px; the VLMs' training data are unknown and may include OOPS!-like web video.

# Run 25: both annotators on the same clips (no new labels)

`nolabel.py --annotators labels_friend.csv --ft-work …` at commit `7aad716`. Copied from the Colab text output. 98 FallOrNot test videos labelled fall / not fall by both annotators; 151 clips; falls: annotator 1 65, annotator 2 77; 20 event clips disagree (consensus subset = the other 131 clips).

| Model | Annotator 1 (ours) | Annotator 2 (independent) | Consensus |
|---|---|---|---|
| Synthetic-only probe | 0.802 | 0.728 | 0.811 |
| Hand-labelled real probe | 0.891 | 0.795 | 0.895 |
| Self-trained probe (ours) | 0.890 | **0.839** | **0.906** |
| Fine-tuned synthetic, seeds 0/1/2 | 0.833 / 0.849 / 0.808 | 0.754 / 0.794 / 0.738 | 0.837 / 0.869 / 0.814 |
| Fine-tuned hand labels, seeds 0/1/2 | 0.901 / 0.888 / 0.902 | 0.811 / 0.804 / 0.805 | 0.904 / 0.892 / 0.906 |
| Fine-tuned self-trained, seeds 0/1/2 | 0.914 / 0.907 / 0.921 | 0.812 / 0.816 / 0.830 | 0.916 / 0.909 / 0.928 |

Paired tests (by compilation):

| Comparison | Annotator 1 | Annotator 2 | Consensus |
|---|---|---|---|
| Self-trained probe vs. synthetic probe | +0.088 [+0.023, +0.153], p = 0.009 | +0.111 [+0.045, +0.172], p < 0.001 | +0.095 [+0.028, +0.161], p = 0.002 |
| Self-trained probe vs. hand-labelled probe | −0.002, p = 0.910 | +0.044 [−0.009, +0.096], p = 0.104 | +0.011, p = 0.596 |
| Self-trained FT vs. synthetic FT (seeds 0/1/2) | p = 0.002 / 0.037 / < 0.001 (all > 0) | p = 0.019 / 0.423 / < 0.001 | p = 0.004 / 0.156 / 0.001 |
| Self-trained FT vs. hand-labelled FT (seeds 0/1/2) | +0.013 / +0.019 / +0.019, all n.s. | +0.001 / +0.011 / +0.024, all n.s. | +0.012 / +0.017 / +0.022, all n.s. |

## Reading
- **The main comparisons hold under the independent annotator's labels:** the self-trained probe beats the synthetic probe under both annotators and the consensus (p ≤ 0.009); fine-tuned self-training beats fine-tuned synthetic for 3 of 3 seeds under annotator 1 and 2 of 3 under annotator 2.
- Against hand-labelled training nothing is significant under either annotator. Under annotator 2, whose labels the hand-labelled models were not trained on, the self-trained probe is 0.044 higher (p = 0.10); the hand-labelled models may benefit from sharing annotator 1's labelling style, which this subset cannot settle.
- **Caveats:** 98 videos; the hand-labelled models were trained on annotator 1's labels of other videos.

# Run 26: shortcut audit (rules fixed in advance) and a fourth, independent test set (Kinetics-700)

Commit `87ae06f` (`Research` repository; Colab L4). Rules committed to `AGENTS.md` before any result. Copied from the Colab text output.

## (a) Kinetics-700 validation as a fourth test set
`kinetics_prep.py`: 147 fall videos (faceplanting, falling off bike, falling off chair), 782 hard negatives (16 fixed look-alike classes), 681 easy negatives (one per other class); 1,610 clips, 7,510 windows (2 s every 2 s, at most 5; max over windows). Not from fail compilations. Kinetics hard = falls + the 782 look-alike negatives (929 clips).

## (b) Accident prompt vs. fall prompt (VLMs), same clips
Accident prompt (never mentions falls): "Does something go wrong in this clip, such as an accident or a mishap?" Both scored as fall detectors. Paired, by compilation.

| Test set | Qwen3-VL-4B fall | Qwen accident | accident − fall | InternVL3.5-8B fall | InternVL accident | accident − fall |
|---|---|---|---|---|---|---|
| FallOrNot | 0.920 | 0.846 | −0.074 [−0.118, −0.033], p = 0.001 | 0.936 | 0.891 | −0.045 [−0.074, −0.018], p = 0.003 |
| FallOrNot hard | 0.902 | 0.738 | −0.164 [−0.234, −0.099], p < 0.001 | 0.930 | 0.825 | −0.105 [−0.156, −0.057], p < 0.001 |
| C (345 labelled accident clips) | 0.881 | 0.711 | −0.170 [−0.222, −0.118], p < 0.001 | 0.891 | 0.799 | −0.092 [−0.130, −0.057], p < 0.001 |
| OF-ItW | 0.946 | 0.898 | −0.049 [−0.063, −0.035], p < 0.001 | 0.964 | 0.946 | −0.018 [−0.025, −0.011], p < 0.001 |

Rule: H1 (accident prompt ≥ 0.90 on OF-ItW): Qwen **False (0.898)**, InternVL True (0.946). H2 (accident prompt below fall prompt on C, CI below zero): both True. **Rule required H1 and H2 for both models: NOT MET** (Qwen misses H1 by 0.002).

## (c) Every detector on every test set (descriptive)
AUROC. Probes trained on the run-18 pool (no labelled video) unless stated; random start: mean ± s.d. over 5 seeds. VLM rows: zero-shot, fall prompt.

| Detector | FallOrNot | FallOrNot hard | C | OF-ItW | HMDB51 | HMDB51 hard | Kinetics | Kinetics hard |
|---|---|---|---|---|---|---|---|---|
| Synthetic-only probe | 0.814 | 0.772 | 0.746 | 0.872 | 0.917 | 0.880 | 0.774 | 0.716 |
| Self-trained, synthetic teacher | 0.897 | 0.840 | 0.804 | 0.937 | 0.952 | 0.915 | **0.808** | **0.742** |
| Self-trained, VLM teacher (run 27) | 0.903 | 0.858 | 0.834 | 0.948 | 0.926 | 0.856 | 0.787 | 0.698 |
| VLM teacher, 1 round (secondary) | 0.916 | 0.886 | 0.858 | 0.951 | 0.919 | 0.833 | 0.755 | 0.658 |
| Accident detector (no fall information) | 0.851 | 0.729 | 0.707 | 0.923 | 0.885 | 0.887 | 0.737 | 0.725 |
| Random start (no fall information) | 0.870 ± 0.006 | 0.768 ± 0.013 | 0.732 ± 0.019 | 0.930 ± 0.004 | 0.903 ± 0.015 | 0.876 ± 0.014 | 0.759 ± 0.021 | 0.719 ± 0.022 |
| Hand-labelled real probe (268 clips) | 0.899 | 0.860 | – | 0.931* | 0.888 | 0.777 | 0.720 | 0.609 |
| Qwen3-VL-4B, zero-shot | 0.920 | 0.902 | 0.881 | 0.946† | 0.953 | 0.873 | 0.822 | 0.724 |
| InternVL3.5-8B, zero-shot | **0.936** | **0.930** | **0.891** | 0.964† | 0.950 | 0.848 | 0.805 | 0.704 |

\* includes 20 clips from OF-ItW test videos that run 11 removes. † on the 1,613 OF-ItW clips without our labelled videos (probes: 1,671 clips; the teacher comparison below matched 1,531).

## Reading
- **The shortcut signature appears in both VLMs, but the rule fixed in advance was not met** (Qwen's accident prompt reached 0.898 on OF-ItW, 0.002 below the 0.90 threshold). In both models, replacing "did someone fall?" by "did something go wrong?" costs little on OF-ItW (−0.018, −0.049) and a lot on falls vs. other accidents (C: −0.092, −0.170); for InternVL the accident prompt alone scores 0.946 on OF-ItW. The difference between the two drops was not tested formally.
- **Detectors with no fall information reach 0.923–0.930 on OF-ItW** (accident detector, random start), against 0.937–0.964 for fall-aware detectors, but only 0.707–0.732 on C.
- **Kinetics-700 (independent of OOPS!) is hard for every detector:** falls vs. look-alike motions 0.609–0.742. The cheap self-trained probe (synthetic teacher) is the best detector there (0.742; 0.808 on all of Kinetics), above both VLMs (0.704–0.724); the hand-labelled probe is the worst (0.609). Paired tests between these were not run.
- **Caveats:** one prompt per question; VLM training data unknown; Kinetics falls are trimmed action clips of three classes; fine-tuned models on Kinetics were scored (`scores_kinetics_*.npz` on Drive) but not yet reported.

# Run 27: VLM-taught self-training (rule fixed in advance: NOT MET)

`vlm_teacher.py` (InternVL3.5-8B scored 6,900 pool event clips at 1.03 s/clip) + `nolabel.py --teacher`. Copied from the Colab text output.

VLM-taught vs. synthetic-taught probe (same pool), paired: FallOrNot +0.006 (p = 0.40); FallOrNot hard +0.017 (p = 0.14); **C +0.029 [+0.012, +0.048], p < 0.001**; OF-ItW +0.011 [+0.005, +0.016], p < 0.001; HMDB51 −0.026 (p < 0.001); HMDB51 hard −0.059 (p < 0.001); Kinetics −0.021 (p = 0.006); Kinetics hard −0.043 (p < 0.001).

Taught probe minus its teacher, same clips: FallOrNot −0.033 [−0.074, +0.006] (0.903 vs. 0.936); FallOrNot hard −0.072 (p = 0.007); **C −0.057 [−0.094, −0.023]** (0.834 vs. 0.891); OF-ItW −0.017 [−0.028, −0.007] (0.946 vs. 0.963, n = 1,531); HMDB51 −0.024 (p = 0.084); HMDB51 hard +0.007 (p = 0.69); Kinetics −0.019 (p = 0.21); Kinetics hard −0.006 (p = 0.79).

**VLM-TEACHER RESULT: NOT MET** — (i) met (beats the synthetic teacher on C); (ii) not met on FallOrNot (−0.033) and C (−0.057), met on OF-ItW (−0.017).

## Reading
- A VLM teacher helps in domain (C +0.029, OF-ItW +0.011) but hurts out of domain (HMDB51, Kinetics), and the small student does not reach its teacher on the hard OOPS!-based tests. One round from the VLM scores looked better in domain (secondary, must not replace the fixed setting).

# Run 28: inference cost (`speed.py`, NVIDIA L4, batch 1, from decoded frames)

| Detector | Seconds per 2 s clip | Peak GPU memory |
|---|---|---|
| Frozen CLIP ViT-L/14 (8 frames) + VideoMAE-B (16 frames) + linear probe | 0.063 | 2.43 GB |
| Fine-tuned VideoMAE-B (16 frames) | 0.023 | 0.59 GB |
| Qwen3-VL-4B (run 24) | 0.22 | 9.2 GB |
| InternVL3.5-8B (run 24) | 1.0 | 18.1 GB |

The fine-tuned self-trained VideoMAE is about 43 times faster than InternVL3.5-8B and uses about 30 times less memory; the probe is about 16 times faster.
