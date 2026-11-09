# How Much Is a Real Fall Worth? Data Efficiency and False Alarms in Real-World Video Fall Detection

> **Positioning note (November 2026). Read before editing.** OmniFall (Schneider et al., arXiv 2505.19889; v1 May 2025, v3 July 2026) already shows the staged-to-wild gap on fall-containing OOPS videos ("OF-ItW", 818 videos, frame-level labels), shows that synthetic training beats staged training on them, and reports one upper-bound model trained on staged + synthetic + in-the-wild data. **We must not claim to be first to show the gap or to build a real-world test set from OOPS.** What OmniFall does not report (per v2/v3 as read; re-check the final appendix): (1) a model-independent random sample of *all* OOPS accidents, so non-fall accidents act as false-alarm negatives (OF-ItW is built only from fall-containing clips); (2) how much real data is needed (no learning curve; no real-only vs. staged-only comparison); (3) appearance vs. motion representations under real training; (4) an equal-budget comparison of real vs. synthetic vs. staged data. The paper is reframed around these. Our benchmark is renamed **FallOrNot** to avoid confusion with OmniFall's "OOPS-Fall". Sections below still use the old framing in places **[TODO: rewrite]**.

*Working draft. Every number comes from `RESULTS.md` (run 3) unless marked **[TODO]**. Items marked **[TODO]** or **[VERIFY]** must be done before submission.*

---

## Abstract

Video fall detectors are commonly trained and evaluated on falls staged by actors in laboratories, where reported accuracy approaches 100%. We ask whether that accuracy survives contact with real falls. We build **FallOrNot**, a real-world fall benchmark derived from the in-the-wild OOPS! dataset: a model-independent random sample of accident videos, labelled by a human for whether a person falls, with hard negatives (other accidents) and ordinary-activity negatives. Using frozen image (CLIP) and video (VideoMAE) encoders with linear probes, detectors trained on the staged UR Fall dataset reach 1.00 AUROC in-domain but only 0.72–0.77 on real falls, and catch 6–30% of falls at a 5% false-positive rate. The same probes trained on 131 real falls reach 0.90 AUROC (recall 43% at 5% FPR), and adding staged data on top brings no further gain. Motion-aware representations account for most of the improvement: a temporal CLIP representation rises from 0.73 to 0.87 with real data, while a static appearance representation stays near 0.73. Distinguishing falls from other accidents is the hardest sub-problem. Our results suggest that staged benchmarks substantially overstate readiness for deployment, and that small amounts of real data matter more than model choice. With OmniFall's 12k synthetic videos, synthetic training plateaus near 0.82 AUROC on our in-domain test set, where 30 real falls outperform all 2,136 synthetic falls; but on an independent real-world test set (HMDB51) the ranking flips, synthetic training is best, and the real-data advantage disappears. Training on synthetic and real data together is the only choice that stays within the confidence interval of the best source on both. **[TODO: update with the second real-world source and the proposed method once available.]**

## 1. Introduction

Falls are a leading cause of injury among older adults, and long periods lying on the floor after a fall are associated with worse outcomes [WHO Falls fact sheet; Fleming & Brayne 2008]. Camera-based fall detection promises automatic alerts without wearables, and the literature reports near-perfect accuracy on benchmark datasets. Those benchmarks, however, consist of young actors falling deliberately onto mattresses in a lab, filmed from fixed cameras.

Real falls differ: they are sudden, partly occluded, filmed from arbitrary viewpoints and lighting, and embedded among other abrupt events (collisions, dropped objects, people catching themselves). Whether staged training data prepares a detector for these conditions has received little direct measurement for video, because real fall video with labels is scarce **[TODO: support with a wider literature review]**.

**Contributions.**
1. **FallOrNot**, a real-world fall benchmark built from OOPS! with a model-independent test sample, human fall labels, and both hard (other accidents) and easy (ordinary activity) negatives. We release labels and timestamps, not videos.
2. **A controlled staged-to-real study.** Identical frozen encoders and linear probes, evaluated on staged and real falls with video-level bootstrap confidence intervals.
3. **Findings.** (i) A large gap: in-domain AUROC 1.00 vs. 0.72–0.77 on real falls. (ii) 131 real falls close most of it (0.76 → 0.90), and staged data adds nothing on top. (iii) Temporal information carries the gain; static appearance does not. (iv) Falls vs. other accidents is the hardest part.
4. **[TODO]** A method targeted at (iv), and a second real-world test source.

## 2. Related work **[TODO: expand to ~1 column]**

- **Staged fall datasets:** UR Fall Detection (Kwolek & Kepski, *Computer Methods and Programs in Biomedicine* 2014); Le2i (Charfi et al., *Journal of Electronic Imaging* 22(4), 2013); Multiple Cameras Fall Dataset (Auvinet et al.) **[VERIFY: identify the dataset technical report; the related journal article, Auvinet et al., 2011, is a different item and must not be substituted silently]**.
- **Real-world falls with wearables:** the gap between simulated and real falls has been documented for accelerometers, e.g. Bagalà et al., *PLoS ONE* 2012, and the FARSEEING real-world fall repository of body-worn sensor signals (Klenk et al., *European Review of Aging and Physical Activity* 2016). We show the analogous gap for video.
- **Closest work: OmniFall** (Schneider et al., arXiv 2505.19889, 2025; v3 2026). Unifies eight staged datasets, adds 12k synthetic videos (OF-Syn) and an in-the-wild test set of 818 fall-containing OOPS videos with frame-level labels (OF-ItW). Shows a large staged-to-wild drop and that synthetic beats staged training; reports one upper-bound model trained with in-the-wild data. Our work differs in its negatives (all accidents, sampled independently of any model), its focus on how much real data is needed, and its real vs. synthetic vs. staged equal-budget comparison. **[VERIFY: venue, if accepted by submission time; licence of OF-ItW labels: the HF card states CC BY-NC-SA 4.0 in text but cc-by-nc-4.0 in metadata.]**
- **Unintentional action:** OOPS! (Epstein, Chen & Vondrick, CVPR 2020) localises the moment an action becomes unintentional; we reuse its videos and transition times and add fall labels.
- **Foundation encoders:** CLIP (Radford et al., ICML 2021), used through the OpenCLIP implementation (https://github.com/mlfoundations/open_clip) with DataComp pretrained weights (Gadre et al., NeurIPS 2023 Datasets and Benchmarks); VideoMAE (Tong et al., NeurIPS 2022), fine-tuned on Kinetics-400 (Kay et al. 2017).
- **Domain shift / benchmark realism:** **[TODO]**.

## 3. FallOrNot benchmark

**Source.** OOPS! contains ~20k in-the-wild "fail" videos; 10,961 have an annotated transition time *t* (the moment the action becomes unintentional) agreed by at least two annotators. We use the median annotator time.

**Clips.** For every annotated video we cut an **event clip** [t−1 s, t+1 s] and, when t ≥ 3.5 s, a **pre-event clip** [t−3.5 s, t−1.5 s] showing ordinary activity before the accident. Every clip is 2 s, sampled at 16 frames.

**Test sample (model-independent).** A clip enters the test set if the MD5 hash of its video name falls below a fixed threshold (~600 of 10,961 videos; 422 after decoding). Selection depends on no model, which prevents the benchmark from favouring any detector. Clips were labelled in a fixed random order, so any labelled prefix is itself a random sample.

**Training pool.** Separately, the 3,000 event clips with the highest zero-shot CLIP "fall" score form a pool for training real-data probes. It never overlaps the test sample. **Note:** this pool is model-selected; it is used only for training.

**Labels.** Following the WHO definition, a fall is an event in which a person inadvertently comes to rest on the ground, floor or another lower level [WHO Falls fact sheet]. Annotators choose *fall*, *not a fall* (including other accidents, near-falls, animals) or *unsure* (excluded). Positives are event clips labelled *fall*; negatives are event clips labelled *not a fall* (**hard negatives**) and all pre-event clips of labelled test videos (**easy negatives**).

**Size.** Test: 165 videos (255 clips, 102 falls), the first 165 of the 422 sampled test videos in a fixed random order, and therefore itself a random sample. Training: 268 clips, 131 falls. Labelling is complete; the test set, labels, prompts and clip definition are frozen.

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

**Table 1. AUROC on real falls (FallOrNot test set) by training data.**

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
| Real (FallOrNot) | staged | 0.759 [0.696, 0.820] | 0.167 [0.098, 0.375] |
| Real (FallOrNot) | real | 0.899 [0.859, 0.934] | 0.431 [0.325, 0.670] |

**Table 3. Falls vs. other accidents only (hard negatives), AUROC.**

| Representation | Staged-trained | Real-trained |
|---|---|---|
| CLIP + VideoMAE | 0.715 [0.633, 0.795] | 0.860 [0.794, 0.914] |
| CLIP temporal | 0.671 [0.582, 0.753] | 0.835 [0.765, 0.891] |

**Table 4. Head-to-head differences on the same test clips** (paired video-level bootstrap, 2,000 resamples).

| Comparison | ΔAUROC | 95% CI | p |
|---|---|---|---|
| CLIP appearance: real- vs. staged-trained | +0.009 | [−0.055, +0.068] | 0.77 |
| CLIP temporal: real- vs. staged-trained | +0.138 | [+0.078, +0.201] | < 0.001 |
| VideoMAE: real- vs. staged-trained | +0.022 | [−0.036, +0.083] | 0.49 |
| CLIP + VideoMAE: real- vs. staged-trained | +0.139 | [+0.087, +0.193] | < 0.001 |
| Staged-trained: VideoMAE vs. CLIP appearance | +0.042 | [−0.028, +0.116] | 0.26 |
| Staged-trained: CLIP + VideoMAE vs. CLIP appearance | +0.036 | [−0.027, +0.104] | 0.30 |

**Figure 2 / Table 5. How much real data?** CLIP + VideoMAE, mean ± s.d. over 10 random draws of real training videos; staged-only AUROC 0.759.

| Real falls | 4 | 8 | 15 | 30 | 60 | 116 | 131 |
|---|---|---|---|---|---|---|---|
| Real only | 0.725 ± 0.094 | 0.764 ± 0.035 | 0.815 ± 0.039 | 0.826 ± 0.019 | 0.871 ± 0.019 | 0.897 ± 0.006 | 0.899 |
| Staged + real | 0.784 ± 0.014 | 0.788 ± 0.011 | 0.815 ± 0.032 | 0.827 ± 0.016 | 0.866 ± 0.015 | 0.890 ± 0.005 | 0.892 |

**[TODO]** Render Figure 1 (`roc_real.png`) and Figure 2 (`learning_curve.png`) for the paper.

### 5.2 Real vs. synthetic vs. staged training data

Synthetic falls are OmniFall's OF-Syn (2,136 fall clips, 3,000 non-fall clips; one 2 s clip per video, same encoders and probes).

**Table 6. AUROC on the real test set, CLIP + VideoMAE, by training source.**

| Training data | AUROC | Recall @ 5% FPR |
|---|---|---|
| Staged (UR Fall, 60 fall clips) | 0.759 [0.696, 0.820] | 0.167 [0.098, 0.375] |
| Synthetic (OF-Syn, 2,136 falls) | 0.814 [0.760, 0.864] | 0.275 [0.149, 0.479] |
| Staged + synthetic | 0.809 [0.758, 0.861] | 0.314 [0.128, 0.510] |
| Real (131 falls) | **0.899 [0.859, 0.934]** | 0.431 [0.325, 0.670] |
| Synthetic + real | 0.872 [0.827, 0.911] | 0.480 [0.323, 0.667] |

Real vs. synthetic: +0.085 [+0.037, +0.137], p < 0.001. Synthetic vs. staged: +0.054 [+0.001, +0.108], p = 0.045. Synthetic + real vs. real: −0.027 [−0.066, +0.010], p = 0.17.

**Table 7. Equal budget** (k falls + k non-falls from one source; mean ± s.d. over random draws).

| Training falls | 4 | 8 | 15 | 30 | 60 | 125 | 500 | 2,136 |
|---|---|---|---|---|---|---|---|---|
| Staged | 0.751 ± 0.020 | 0.745 ± 0.014 | 0.744 ± 0.009 | 0.752 ± 0.008 | 0.754 | – | – | – |
| Synthetic | 0.723 ± 0.025 | 0.724 ± 0.020 | 0.742 ± 0.011 | 0.768 ± 0.024 | 0.786 ± 0.024 | 0.825 ± 0.018 | 0.819 ± 0.004 | 0.812 |
| Real | 0.670 ± 0.130 | 0.772 ± 0.028 | 0.789 ± 0.027 | 0.836 ± 0.018 | 0.862 ± 0.013 | 0.896 ± 0.003 | – | – |

**Findings.** (i) Synthetic data plateaus near 0.82 AUROC from about 125 falls up to 2,136, while real data keeps improving: 30 real falls (0.836) already exceed every synthetic budget. (ii) We replicate OmniFall's result that synthetic beats staged training. (iii) Adding synthetic data to real data does not help the best representation, though it helps weaker single representations (CLIP appearance +0.068, VideoMAE +0.055, both p = 0.024). (iv) With very few falls (4) real data is unreliable; real overtakes both other sources from about 8 falls. **Caveat:** real training is in-domain (same OOPS! source as the test set) while synthetic and staged are out-of-domain; a second real-world source is needed to separate "real" from "in-domain" **[TODO]**. Our probes are linear on frozen features; OmniFall fine-tunes end-to-end, which may exploit synthetic data better.

### 5.3 External real-world test (HMDB51)

Nothing is trained on HMDB51 (CC BY 4.0): 100 human-labelled `fall_floor` clips vs. 20 hash-sampled clips from each other class; clip score = maximum over 2 s windows.

**Table 8. AUROC on HMDB51, CLIP + VideoMAE.**

| Training data | All negatives (1,100 clips) | Hard subset (320 clips) |
|---|---|---|
| Staged | 0.899 [0.870, 0.926] | 0.833 [0.783, 0.877] |
| Synthetic | **0.917 [0.892, 0.943]** | **0.880 [0.843, 0.915]** |
| Real (OOPS!) | 0.888 [0.850, 0.925] | 0.777 [0.718, 0.833] |
| Synthetic + real | 0.901 [0.869, 0.932] | 0.855 [0.811, 0.896] |

Real vs. synthetic on HMDB51: −0.029 [−0.057, −0.003], p = 0.029 (compare +0.085, p < 0.001 on FallOrNot).

**Finding.** The value of real data measured on FallOrNot is largely in-domain: OOPS!-trained probes do not transfer better than synthetic or staged ones to HMDB51, and which training source looks best flips between the two real-world test sets. HMDB51 falls appear closer to staged falls (staged probes 0.899 there vs. 0.759 on FallOrNot), so the two benchmarks measure different kinds of "real". **Implication:** conclusions about training data drawn from a single in-the-wild benchmark, ours included, do not generalise; fall detectors should be evaluated on several real-world sources, and mixing synthetic with real data is the most robust choice we found.

### 5.4 Fine-tuning (pre-registered)

We fine-tuned VideoMAE-B end-to-end with hyper-parameters fixed in advance (5 epochs, batch 8, AdamW 5e-5) and pre-registered one test: fine-tuning on synthetic + real data must beat the best frozen probe on both test sets (paired 95% CI above zero). **It did not, for any of three seeds.** On FallOrNot the difference is −0.011 to −0.002; on HMDB51 it is +0.028 to +0.049, significant for two of three seeds.

**Table 9. AUROC of fine-tuned models, mean ± s.d. over 3 seeds.**

| Training data | FallOrNot | HMDB51 | HMDB51 hard |
|---|---|---|---|
| Synthetic | 0.838 ± 0.017 | **0.960 ± 0.006** | **0.922 ± 0.015** |
| Real | **0.902 ± 0.012** | 0.920 ± 0.026 | 0.795 ± 0.067 |
| Synthetic + real | 0.894 ± 0.005 | 0.954 ± 0.011 | 0.890 ± 0.027 |

The ranking flip of §5.3 reappears with fine-tuned models: on average, real data wins on FallOrNot and synthetic data wins on HMDB51. Synthetic + real is within 0.008 of the best mean on both test sets and has the smallest seed-to-seed spread, which makes it the most reliable recipe we found. Real-only fine-tuning varies strongly between seeds out of domain (HMDB51 hard subset 0.751–0.872).

## 6. Analysis

**The gap is large, and in-domain scores hide it.** Every probe is perfect on UR Fall, so the staged benchmark cannot rank methods at all, yet the same probes catch at most 30% of real falls at 5% FPR.

**Data matters more than the encoder.** Changing the training data from staged to real raises CLIP + VideoMAE by 0.14 AUROC (p < 0.001). Changing the encoder under staged training makes no significant difference (all p ≥ 0.26).

**A handful of real falls is worth the whole staged dataset.** About 8 real falls match the 30 staged falls (60 clips) of UR Fall, about 15 exceed them, and performance is still rising at 131 (Table 5), so the benchmark is far from saturated. Staged data helps only when real data is tiny (4 real falls: 0.784 with staged data vs. 0.725 without).

**Staged data does not help once real data is present** (0.892 with both vs. 0.899 real only), which suggests it encodes a different, narrower notion of "fall".

**Motion is what transfers.** Appearance alone does not benefit from real data (+0.009, p = 0.77), while the temporal CLIP representation gains 0.138 (p < 0.001). Interestingly, VideoMAE alone gains little (+0.022, p = 0.49) despite being a video model; its Kinetics-trained features may emphasise action categories over the abrupt change a fall produces, but it adds to CLIP temporal when combined. **[TODO: test this explanation.]**

**Falls vs. other accidents is the bottleneck.** Hard-negative AUROC is lower than overall AUROC throughout. **[TODO: qualitative examples from `worst_errors.csv`.]**

## 7. Limitations

- **In-distribution training.** Real training and test clips both come from OOPS!, so part of the real-data gain is ordinary in-domain advantage. **[TODO: evaluate on a second real-world source.]**
- **Population.** OOPS! shows mostly younger people in sports and everyday mishaps, not older adults at home, the main use case.
- **Model-selected training pool** (top CLIP zero-shot scores) may favour CLIP-friendly falls; the test set is not affected.
- **Single annotator.** Inter-annotator agreement has not been measured.
- **One staged dataset**, which is saturated. **[TODO: add Le2i or another, if obtainable.]**
- **Frozen encoders and linear probes only.** Fine-tuned models may close more of the gap; the comparison isolates the effect of data.

## 8. Ethics and data

OOPS! videos are public YouTube compilations. We release only labels, timestamps and code. UR Fall is CC BY-NC-SA 4.0 for non-commercial academic use.

## 9. Conclusion

Detectors that look perfect on staged falls miss most real ones. A small amount of real data, used with motion-aware representations, closes most of the gap, while more staged data does not. We release FallOrNot so that fall detection can be measured where it matters.

---

## Remaining work before submission

| Item | Who | Notes |
|---|---|---|
| Optional: second annotator on the existing test clips (κ) | second person, if available | about 30 min |
| Agreement with OmniFall's independent OF-ItW annotations on overlapping videos (37 of our 422 test-sample videos are in OF-ItW) | Claude, one Colab run | no labelling |
| Real vs. synthetic (OF-Syn, 9.7 GB, public) vs. staged at equal training budget | Claude, Colab runs | no labelling |
| Rewrite abstract, introduction and contributions around the new positioning | ChatGPT | after the new experiments |
| Second real-world test source | Claude | look for public real-fall video |
| Second staged dataset | Claude | Le2i page is offline; find a mirror or alternative |
| Proposed method for falls vs. other accidents | Claude | e.g. person-centred motion features or fine-tuning |
| Related work and citations | Claude | verify every reference |
| CVPR LaTeX template, figures | Claude | 8 pages plus references |

---

## References (checked; see notes)

- Bagalà, F., et al. Evaluation of accelerometer-based fall detection algorithms on real-world falls. *PLoS ONE* 7(5): e37062, 2012.
- Charfi, I., Miteran, J., Dubois, J., Atri, M., Tourki, R. Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and Adaboost-based classification. *Journal of Electronic Imaging* 22(4): 041106, 2013. doi:10.1117/1.JEI.22.4.041106
- Epstein, D., Chen, B., Vondrick, C. Oops! Predicting unintentional action in video. *CVPR* 2020.
- Fleming, J., Brayne, C. (Cambridge City over-75s Cohort study collaboration). Inability to get up after falling, subsequent time on floor, and summoning help: prospective cohort study in people over 90. *BMJ* 337: a2227, 2008. (Supports an association with adverse outcomes, not causation.)
- Schneider, D., et al. OmniFall: From staged through synthetic to wild, a unified multi-domain dataset for robust fall detection. arXiv:2505.19889, 2025 (v3 2026). **[VERIFY: final title and venue]**
- Gadre, S. Y., et al. DataComp: In search of the next generation of multimodal datasets. *NeurIPS* 2023, Datasets and Benchmarks Track.
- Kay, W., et al. The Kinetics human action video dataset. arXiv:1705.06950, 2017.
- Klenk, J., et al. The FARSEEING real-world fall repository: a large-scale collaborative database to collect and share sensor signals from real-world falls. *European Review of Aging and Physical Activity* 13: 8, 2016.
- Kwolek, B., Kepski, M. Human fall detection on embedded platform using depth maps and wireless accelerometer. *Computer Methods and Programs in Biomedicine*, 2014.
- OpenCLIP (software). https://github.com/mlfoundations/open_clip **[TODO: use the repository's recommended citation]**
- Radford, A., et al. Learning transferable visual models from natural language supervision. *ICML* 2021.
- Tong, Z., Song, Y., Wang, J., Wang, L. VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. *NeurIPS* 2022.
- World Health Organization. Falls (fact sheet). https://www.who.int/news-room/fact-sheets/detail/falls **[TODO: record the access date]**
- Auvinet, E., et al. Multiple Cameras Fall Dataset technical report **[VERIFY]**.

