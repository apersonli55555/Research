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
