# Staged Falls Don't Transfer: Measuring the Lab-to-Real Gap in Video Fall Detection

*Working draft. Every number comes from `RESULTS.md` (run 3) unless marked **[TODO]**. Items marked **[TODO]** or **[VERIFY]** must be done before submission.*

---

## Abstract

Video fall detectors are trained and evaluated almost entirely on falls staged by actors in laboratories, where reported accuracy approaches 100%. We ask whether that accuracy survives contact with real falls. We build **OOPS-Fall**, a real-world fall benchmark derived from the in-the-wild OOPS! dataset: a model-independent random sample of accident videos, labelled by a human for whether a person falls, with hard negatives (other accidents) and ordinary-activity negatives. Using frozen image (CLIP) and video (VideoMAE) encoders with linear probes, detectors trained on the staged UR Fall dataset reach 1.00 AUROC in-domain but only 0.72–0.77 on real falls, and catch 6–30% of falls at a 5% false-positive rate. The same probes trained on 131 real falls reach 0.90 AUROC (recall 43% at 5% FPR), and adding staged data on top brings no further gain. Motion-aware representations account for most of the improvement: a temporal CLIP representation rises from 0.73 to 0.87 with real data, while a static appearance representation stays near 0.73. Distinguishing falls from other accidents is the hardest sub-problem. Our results suggest that staged benchmarks substantially overstate readiness for deployment, and that small amounts of real data matter more than model choice. **[TODO: update with full test set, second real source, and the proposed method once available.]**

## 1. Introduction

Falls are a leading cause of injury among older adults, and long periods lying on the floor after a fall are associated with worse outcomes **[VERIFY: cite WHO fall fact sheet and a clinical source on "long lie"]**. Camera-based fall detection promises automatic alerts without wearables, and the literature reports near-perfect accuracy on benchmark datasets. Those benchmarks, however, consist of young actors falling deliberately onto mattresses in a lab, filmed from fixed cameras.

Real falls differ: they are sudden, partly occluded, filmed from arbitrary viewpoints and lighting, and embedded among other abrupt events (collisions, dropped objects, people catching themselves). Whether staged training data prepares a detector for these conditions has rarely been measured, because real fall video with labels is scarce.

**Contributions.**
1. **OOPS-Fall**, a real-world fall benchmark built from OOPS! with a model-independent test sample, human fall labels, and both hard (other accidents) and easy (ordinary activity) negatives. We release labels and timestamps, not videos.
2. **A controlled staged-to-real study.** Identical frozen encoders and linear probes, evaluated on staged and real falls with video-level bootstrap confidence intervals.
3. **Findings.** (i) A large gap: in-domain AUROC 1.00 vs. 0.72–0.77 on real falls. (ii) 131 real falls close most of it (0.76 → 0.90), and staged data adds nothing on top. (iii) Temporal information carries the gain; static appearance does not. (iv) Falls vs. other accidents is the hardest part.
4. **[TODO]** A method targeted at (iv), and a second real-world test source.

## 2. Related work **[TODO: expand to ~1 column; verify every citation]**

- **Staged fall datasets:** UR Fall Detection (Kwolek & Kepski, *Computer Methods and Programs in Biomedicine* 2014); Le2i (Charfi et al., *Journal of Electronic Imaging* 2013) **[VERIFY]**; Multiple Cameras Fall Dataset (Auvinet et al.) **[VERIFY]**.
- **Real-world falls with wearables:** the gap between simulated and real falls has been documented for accelerometers, e.g. Bagalà et al., *PLoS ONE* 2012, and the FARSEEING real-world fall repository **[VERIFY]**. We show the analogous gap for video.
- **Unintentional action:** OOPS! (Epstein, Chen & Vondrick, CVPR 2020) localises the moment an action becomes unintentional; we reuse its videos and transition times and add fall labels.
- **Foundation encoders:** CLIP (Radford et al., ICML 2021) and OpenCLIP/DataComp weights (Gadre et al., NeurIPS 2023 **[VERIFY]**); VideoMAE (Tong et al., NeurIPS 2022), fine-tuned on Kinetics-400 (Kay et al. 2017).
- **Domain shift / benchmark realism:** **[TODO]**.

## 3. OOPS-Fall benchmark

**Source.** OOPS! contains ~20k in-the-wild "fail" videos; 10,961 have an annotated transition time *t* (the moment the action becomes unintentional) agreed by at least two annotators. We use the median annotator time.

**Clips.** For every annotated video we cut an **event clip** [t−1 s, t+1 s] and, when t ≥ 3.5 s, a **pre-event clip** [t−3.5 s, t−1.5 s] showing ordinary activity before the accident. Every clip is 2 s, sampled at 16 frames.

**Test sample (model-independent).** A clip enters the test set if the MD5 hash of its video name falls below a fixed threshold (~600 of 10,961 videos; 422 after decoding). Selection depends on no model, which prevents the benchmark from favouring any detector. Clips were labelled in a fixed random order, so any labelled prefix is itself a random sample.

**Training pool.** Separately, the 3,000 event clips with the highest zero-shot CLIP "fall" score form a pool for training real-data probes. It never overlaps the test sample. **Note:** this pool is model-selected; it is used only for training.

**Labels.** A fall is "an event which results in a person coming to rest inadvertently on the ground or floor or other lower level" (WHO definition). Annotators choose *fall*, *not a fall* (including other accidents, near-falls, animals) or *unsure* (excluded). Positives are event clips labelled *fall*; negatives are event clips labelled *not a fall* (**hard negatives**) and all pre-event clips of labelled test videos (**easy negatives**).

**Current size.** Test: 165 videos, 255 clips, 102 falls. Training: 268 clips, 131 falls. **[TODO: label the remaining 257 test videos; add a second annotator on the test set and report Cohen's κ.]**

## 4. Experimental setup

**Staged data.** UR Fall: 30 falls filmed by a side and a ceiling camera, and 40 activities of daily living (ADLs). Fall clips are centred on the frames labelled "falling"; negatives are pre-fall clips and 2 s windows (1 s stride) over each ADL video. 316 clips, 60 falls. Training uses all of them; in-domain numbers use 5-fold cross-validation grouped by sequence.

**Representations (all frozen).**
- *CLIP appearance*: mean of 8 L2-normalised CLIP ViT-L/14 frame embeddings.
- *CLIP temporal*: [mean, mean of last third − mean of first third, mean absolute frame-to-frame change].
- *VideoMAE*: pooled VideoMAE-B (Kinetics-400) embedding of 16 frames.
- *CLIP + VideoMAE*: concatenation of CLIP temporal and VideoMAE.

**Detectors.** Zero-shot CLIP (fall vs. non-fall prompt margin, prompts fixed in advance) and L2-regularised logistic regression on standardised features (C = 0.1, class-balanced), trained on staged data, real data, or both.

**Metrics.** AUROC, average precision, and recall at 5% false-positive rate, with 95% CIs from a video-level bootstrap (1,000 resamples).

## 5. Results

**Table 1. AUROC on real falls (OOPS-Fall test set) by training data.**

| Representation | Staged-trained | Real-trained | Staged + real |
|---|---|---|---|
| Zero-shot CLIP | 0.717 [0.658, 0.778] | – | – |
| CLIP appearance | 0.723 [0.661, 0.788] | 0.732 [0.667, 0.793] | 0.741 [0.677, 0.802] |
| CLIP temporal | 0.728 [0.667, 0.790] | 0.866 [0.818, 0.907] | 0.857 [0.807, 0.899] |
| VideoMAE | 0.765 [0.711, 0.820] | 0.787 [0.732, 0.841] | 0.803 [0.747, 0.854] |
| CLIP + VideoMAE | 0.759 [0.696, 0.820] | **0.899 [0.859, 0.934]** | 0.892 [0.850, 0.929] |

**Table 2. The staged-to-real gap (CLIP + VideoMAE).**

| Evaluated on | Trained on | AUROC | Recall @ 5% FPR |
|---|---|---|---|
| Staged (UR Fall, CV) | staged | 1.000 [1.000, 1.000] | 1.000 |
| Real (OOPS-Fall) | staged | 0.759 [0.696, 0.820] | 0.167 [0.098, 0.375] |
| Real (OOPS-Fall) | real | 0.899 [0.859, 0.934] | 0.431 [0.325, 0.670] |

**Table 3. Falls vs. other accidents only (hard negatives), AUROC.**

| Representation | Staged-trained | Real-trained |
|---|---|---|
| CLIP + VideoMAE | 0.715 [0.633, 0.795] | 0.860 [0.794, 0.914] |
| CLIP temporal | 0.671 [0.582, 0.753] | 0.835 [0.765, 0.891] |

**[TODO]** Figure 1: ROC curves (`roc_real.png`). Figure 2: learning curve, AUROC vs. number of real training falls (`learning_curve.png`). Table 4: paired bootstrap differences. All three are produced by the current `experiment.py` with one Colab run (Setup → Step 4), no further labelling.

## 6. Analysis

**The gap is large, and in-domain scores hide it.** Every probe is perfect on UR Fall, so the staged benchmark cannot rank methods at all, yet the same probes catch at most 30% of real falls at 5% FPR.

**Data matters more than the encoder.** Changing the training data from staged to real raises CLIP + VideoMAE by 0.14 AUROC; changing the encoder under staged training moves it by at most 0.05.

**Staged data does not help once real data is present** (0.892 with both vs. 0.899 real only), which suggests it encodes a different, narrower notion of "fall".

**Motion is what transfers.** Appearance alone barely benefits from real data, while the temporal representation gains 0.14. Real falls are recognisable from how the scene changes, not from a single frame of a person on the ground.

**Falls vs. other accidents is the bottleneck.** Hard-negative AUROC is lower than overall AUROC throughout. **[TODO: qualitative examples from `worst_errors.csv`.]**

## 7. Limitations

- **In-distribution training.** Real training and test clips both come from OOPS!, so part of the real-data gain is ordinary in-domain advantage. **[TODO: evaluate on a second real-world source.]**
- **Population.** OOPS! shows mostly younger people in sports and everyday mishaps, not older adults at home, the main use case.
- **Model-selected training pool** (top CLIP zero-shot scores) may favour CLIP-friendly falls; the test set is not affected.
- **Single annotator** so far; **[TODO: κ]**.
- **One staged dataset**, which is saturated. **[TODO: add Le2i or another, if obtainable.]**
- **Frozen encoders and linear probes only.** Fine-tuned models may close more of the gap; the comparison isolates the effect of data.

## 8. Ethics and data

OOPS! videos are public YouTube compilations. We release only labels, timestamps and code. **[TODO: confirm OOPS! terms of use with its authors.]** UR Fall is CC BY-NC-SA 4.0 for non-commercial academic use.

## 9. Conclusion

Detectors that look perfect on staged falls miss most real ones. A small amount of real data, used with motion-aware representations, closes most of the gap, while more staged data does not. We release OOPS-Fall so that fall detection can be measured where it matters.

---

## Remaining work before submission

| Item | Who | Notes |
|---|---|---|
| Head-to-head tests and learning curve | one Colab run (Setup → Step 4) | code ready; no labelling |
| Remaining 257 test videos | annotator | about 30 min |
| Second annotator on the test set (κ) | second person | about 30 min |
| Second real-world test source | Claude | look for public real-fall video |
| Second staged dataset | Claude | Le2i page is offline; find a mirror or alternative |
| Proposed method for falls vs. other accidents | Claude | e.g. person-centred motion features or fine-tuning |
| OOPS! permission email | owner | draft on request |
| Related work and citations | Claude | verify every reference |
| CVPR LaTeX template, figures | Claude | 8 pages plus references |
