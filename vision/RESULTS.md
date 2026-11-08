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
