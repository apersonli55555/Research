# Results log — vocabulary-set interference

Every run is recorded here, including failed ones, with its date and the exact command. Paper numbers come only
from this file.

> **Date correction (24 Nov 2026).** Headings in this log, `PLAN.md` and `NOVELTY.md` written on 23–24 Nov 2026 had been labelled with
> later dates (25–28 Nov). Each was corrected to the date its line was committed (from `git blame`); the order of
> entries, and so every "before any result" statement, is unchanged and can be checked in the git history.

## Smoke tests (CPU, 3 COCO images — code checks only, not results)
- 22 Nov 2026: `data --n-images 3`, then `run` for each model on a few pairs. OWLv2 fast path vs full forward:
  max |difference| 3.6e-7 over 36 runs (so `--owl-fast` is exact up to float error). The analysis runs end to end.
  These numbers are too few to mean anything and are not used anywhere.

## OVD run 1: pilot (500 COCO val2017 images, rule in PLAN.md)

### Run 1a — FAILED (bug), 22 Nov 2026, CPU in the development container, OWLv2 `--owl-fast`
All 44,478 runs finished, but mAP50 was 0.039 and the detection rate 5%, far below OWLv2's known COCO accuracy.
Recomputing 6 images directly showed the stored scores belonged to other images from the third image on (e.g.
fire hydrant stored 0.111 vs. recomputed 0.799). Cause: the fast-path cache was keyed by `id(img)`, which Python
reuses after an image is freed. Fixed (keyed by file name, commit after a82541a) with a regression test
(`test_owl_fast_cache_is_keyed_by_file_not_object_id`). The buggy file is kept as `results_owlv2.BUGGY_idcache.jsonl`
and is not used anywhere. Any Colab OWLv2 pilot run made with code before the fix must be deleted and re-run.
Grounding DINO and YOLO-World pilot runs do not use this cache and are unaffected.

## OVD runs 2–4 (full study, corrections, latency) — pending; code `ovd/vocab_full.py`, analysis `ovd/full_analysis.py`

### Run 1b — OWLv2, 22 Nov 2026, CPU (development container), fixed code, `--owl-fast`, 500 dev images
Command: `vocab_pilot.py run --model owlv2 --owl-fast` then `analyze --boot 1000` (OWLv2 file only).
2,471 pairs, 3,722 present-target instances, 44,478 runs.
- mAP50 (deployed): k1 0.746, rand5 0.744, near5 0.729, rand20 0.742, near20 0.727, all80 0.726;
  k1 − all80 = +0.020, 95% CI [+0.013, +0.027].
- Mean |raw score change| vs k1: 0.000 in every group (raw scores invariant, as expected by construction).
- Detection flips vs k1: rand5 0.002, near5 0.022, rand20 0.007, near20 0.025, all80 0.028; localisation flips 0.000.
- False alarms on absent targets: k1 0.045 → all80 0.033 (label competition also suppresses false alarms).
- Detected at k1 but missed with all 80: 0.035; order-only flips 0.000.
- GO criteria for OWLv2: a False (2.0 < 3 points), b False, c False. **OWLv2 alone: STOP.** The whole effect is top-1
  label competition; similar distractors (near) cause ~10× more flips than random ones of the same size.

### Run 1c — YOLO-World v2-S, 22 Nov 2026, CPU (development container), 500 dev images
Command: `vocab_pilot.py run --model yoloworld` (text embeddings cached per name; identical outputs, max diff 1e-6),
then `analyze --boot 1000`. 2,471 pairs, 3,717 present instances, 44,478 runs (resumed once after a time-limit stop;
0 corrupt lines).
- mAP50 (deployed): k1 0.598, rand5 0.609, near5 0.596, rand20 0.608, near20 0.600, all80 0.600;
  k1 − all80 = −0.003, 95% CI [−0.012, +0.010] (no net AP change).
- Mean |raw score change| vs k1: rand5 0.032, near5 0.034, rand20 0.042, near20 0.048, all80 0.052 (grows with size).
- Detection flips vs k1: rand5 0.040, near5 0.041, rand20 0.051, near20 0.059, all80 0.064.
- Best-box localisation flips: rand5 0.032 … all80 0.068.
- Absent-target false alarms: k1 0.024 → all80 0.011. Detected at k1, missed at all80: 0.035.
- Order-only: flips 0.000, score range 0.0000 (exactly order-invariant, as the max-over-classes guide predicts).
- GO criteria: a False, b False (6.4% < 10%), c False. **YOLO-World alone: STOP.** Per-instance outputs do change
  (≈6% flips) but gains and losses cancel in AP.

### Combined decision so far (OWLv2 + YOLO-World): STOP for both. Grounding DINO is pending: a CPU subsample (first
120 pairs, not the fixed rule) is running here; the full 500-image Grounding DINO pilot needs the owner's GPU.

### Run 1d — Grounding DINO base, 22 Nov 2026, CPU, SUBSAMPLE: the first 120 (image, target) pairs in plan order
Command: `vocab_pilot.py run --model gdino --limit 120` (backbone output cached per image; identical outputs, max diff
0 vs. uncached runs), then `analyze --boot 1000` on the same 120 pairs for all three models (2,160 runs each).
This is NOT the fixed GO test (which is defined on all 500 images); it is a CPU-feasible preview.

| model (120 pairs, 156 instances) | mAP50 k1 | mAP50 all80 | k1−all80 [95% CI] | mean abs score change at 80 | flips vs k1 at 80 | loc. flips at 80 | absent false alarms k1→80 |
|---|---|---|---|---|---|---|---|
| Grounding DINO | 0.836 | 0.835 | +0.001 [−0.029, +0.025] | 0.142 | **0.154** | 0.071 | 0.146 → 0.000 |
| YOLO-World     | 0.746 | 0.702 | +0.044 [−0.011, +0.095] | 0.068 | 0.071 | 0.071 | 0.000 → 0.000 |
| OWLv2          | 0.870 | 0.852 | +0.018 [+0.005, +0.058] | 0.000 | 0.026 | 0.000 | 0.062 → 0.042 |

Grounding DINO details: detected rate k1 0.641 → rand5 0.740, near5 0.756, all80 0.718; mean raw target score k1
0.432 → all80 0.524 (adding names *raises* the target score on average); flips rand5 0.137, near5 0.128, rand20 0.165,
near20 0.173; order-only flips 0.000 (score range 0.0001). Criterion (b) is met on this subsample (0.154 ≥ 0.10).
**Provisional decision: GO, driven by Grounding DINO**; to be confirmed by the full 500-image Grounding DINO pilot on GPU.
Note the subsample makes YOLO-World's AP gap look larger (+0.044) than on all 500 images (−0.003): small samples vary.

## OVD run 2a — OWLv2 on held-out COCO, 22 Nov 2026, CPU (development container)
Command: `vocab_full.py data --dataset coco` (1,000 images, ranks 501–1,500; 5,013 pairs) then `run --model owlv2`
(115,299 runs incl. corrections) and `full_analysis.py --boot 1000`. Full table: `ovd/results/run2_coco_owlv2_cpu.md`.
(Analysis fixes before this table: the "far" rows were missing from the output, and the FIX false-alarm rate now uses
the independent read-out it is defined with.)
- Raw target scores: mean |change| 0.0000 in every condition (exact invariance confirmed on held-out data).
- Deployed mAP50: k1 0.744; far5 0.743, far20 0.743; rand5 0.743, rand20 0.738; vis5 0.730, vis20 0.724;
  near5 0.724, near20 0.724; all 80 (5 orders) 0.721 (Δ vs k1 95% CI [−0.032, −0.017]).
- Detection flips vs k1: far 0.001–0.002, rand5 0.002, rand20 0.008, vis5 0.019, vis20 0.025, near5 0.025,
  near20 0.027, all 0.031. Order-only flips 0.000.
- **Correction (independent read-out, FIX):** mAP50 0.744 = k1 exactly; FIX − all = +0.0227, 95% CI [+0.0173, +0.0318];
  interference 0 (flips 0.000). Absent-target false alarms: FIX 0.060 (= k1) vs. all 0.040.
- **Baseline: CLIP vocabulary pruning (VocAda-style, K=10):** mAP50 0.637; FIX − prune = +0.106 [+0.098, +0.125];
  pruning removes the target in many images (19% detection flips, 26% localisation flips).

## OVD run 2b — OWLv2 on held-out LVIS v1 val, 22 Nov 2026, CPU (development container)
Command: `vocab_full.py data --dataset lvis` (600 images, 2,689 pairs; ≤3 present + 2 verified-absent targets per
image) then `run --model owlv2` (83,359 runs) and `full_analysis.py --boot 1000`. Table: `ovd/results/run2_lvis_owlv2_cpu.md`.
- Raw scores: mean |change| 0.0000 everywhere (exact invariance holds at 1,203 classes).
- Deployed (top-1 label) mAP50: k1 0.765; rand5 0.763, rand20 0.752, rand80 0.720, rand200 0.689, rand500 0.626;
  far5/20/80 0.761/0.761/0.752; vis5/20/80 0.684/0.656/0.613; near5/20/80 0.644/0.602/0.572; all 1,203 (5 orders) 0.558.
- Detection flips vs k1: rand500 0.110, near80 0.173, vis80 0.134, far80 0.005, all 0.190. Order-only flips 0.000.
- **FIX (independent read-out):** mAP50 0.765 (= k1); FIX − all = +0.207, 95% CI [+0.179, +0.216]; interference 0.
  Trade-off: absent-target (verified-negative) false alarms FIX 0.063 vs. all 0.014.
- Pruning baseline (K=50): mAP50 0.313; FIX − prune = +0.452 [+0.431, +0.478].
- Caveat for the paper: "deployed" = the common single-label post-processing (Hugging Face
  `post_process_object_detection` keeps each box's top class). Benchmark LVIS evaluation scores (box, class) pairs
  without top-1 assignment and therefore does not suffer this loss; the finding concerns deployment-style output.

### Run 1e — FULL PILOT on the owner's Colab L4 GPU, 23 Nov 2026 — **PILOT RESULT: GO**
Code `4703704` (research repo). Commands: `vocab_pilot.py run` for gdino, owlv2 `--owl-fast` (re-run after deleting the
file made with the buggy pre-fix cache), yoloworld; `analyze --boot 1000`. 2,471 pairs, 3,722 instances (YOLO-World 3,717).
| model | mAP50 k1 | mAP50 all80 | k1−all80 [95% CI] | abs score change @80 | flips @80 | loc flips @80 | absent FA k1→80 | GO a/b/c |
|---|---|---|---|---|---|---|---|---|
| Grounding DINO | 0.693 | 0.776 | **−0.083 [−0.095, −0.061]** | 0.135 | **0.167** | 0.085 | 0.164 → 0.004 | **T/T/F** |
| OWLv2 | 0.746 | 0.726 | +0.020 [+0.013, +0.027] | 0.000 | 0.028 | 0.000 | 0.045 → 0.033 | F/F/F |
| YOLO-World | 0.598 | 0.600 | −0.002 [−0.013, +0.010] | 0.052 | 0.063 | 0.068 | 0.024 → 0.011 | F/F/F |
Grounding DINO by group: mAP50 rand5 0.783, near5 0.776, rand20 0.779, near20 0.776; detected rate k1 0.581 → 0.706;
mean raw target score 0.404 → 0.509; order-only flips 0.000 (score range 0.0002).
**Independent check:** OWLv2 GPU numbers equal the CPU run 1b to every printed decimal; YOLO-World GPU vs CPU run 1c
differ by ≤ 0.001 (e.g. near5 mAP50 0.595 vs 0.596, flips@80 0.063 vs 0.064). The CPU Grounding DINO subsample (1d)
is consistent in direction and size (flips 0.154 vs 0.167).
**Decision (fixed rule): GO.** Interpretation: for Grounding DINO, single-class prompting is the *worst* context —
the extra names raise target scores and remove false alarms; YOLO-World changes per instance without net AP change;
OWLv2 changes only through label competition.

## OVD run 2c — YOLO-World on held-out COCO, 23 Nov 2026, CPU (fused conv+BN, cached text-free layers; max diff 2e-5)
Command: `vocab_full.py run --dataset coco --model yoloworld` (115,299 runs incl. FIX), `full_analysis.py --boot 1000`.
Table: `ovd/results/run2_coco_yoloworld_cpu.md`.
- Deployed mAP50: k1 0.604; far5 0.618, far20 0.623; rand5 0.618, rand20 0.613; near5 0.589, near20 0.590;
  vis5 0.590, vis20 0.591; all 80 0.588 (Δ vs k1 CI [−0.031, −0.003]). Independent read-out ("indep") 0.617–0.624.
- Mean |score change| vs k1: 0.032 (rand5) … 0.053 (all); flips 0.039 … 0.066; localisation flips 0.085 … 0.140.
  Order-only flips 0.000. Instances flipping under some vocabulary: 0.083.
- **FIX (fixed guide = all 80 names; head = deployment vocabulary; independent read-out):** mAP50 0.624, interference 0;
  FIX − k1 = +0.020 [+0.015, +0.026]; FIX − all = +0.037 [+0.025, +0.051]; FIX − prune = +0.097 [+0.086, +0.116].
  Absent false alarms: FIX 0.009, k1 0.022, all 0.006.
- Reading: unrelated names in the *guide* help (more guidance), similar/visual names hurt only through top-1 label
  competition; fixing a rich guide and removing competition gives the best of both.

## OVD run 2d — YOLO-World on held-out LVIS, 23 Nov 2026, CPU
Command: `vocab_full.py run --dataset lvis --model yoloworld` (83,359 runs), `full_analysis.py --boot 1000`.
Table: `ovd/results/run2_lvis_yoloworld_cpu.md`.
- Deployed mAP50: k1 0.364; rand5/20/80/200/500 0.397/0.410/0.405/0.369/0.309; far5/20/80 0.397/0.420/0.429;
  near5/20/80 0.283/0.257/0.243; vis5/20/80 0.325/0.302/0.295; all 1,203 (5 orders) 0.249; prune (K=50) 0.172.
- Independent read-out mAP50 per group: 0.385–0.455 (max rand200 0.455, rand80 0.452; all 0.442).
- Mean |score change| vs k1 grows with size: 0.022 (rand5) … 0.087 (all); flips 0.030 … 0.119; localisation flips
  0.18 … 0.44. Order-only flips 0.000. Instances flipping under some vocabulary: 0.181.
- **FIX (fixed guide = all 1,203 names, independent read-out):** mAP50 0.442, interference 0;
  FIX − k1 = +0.078 [+0.062, +0.096]; FIX − all = +0.193 [+0.176, +0.217]; FIX − prune = +0.270 [+0.259, +0.304].
- Caveat (reported, not tuned): random guides of 80–200 names give slightly higher independent-read-out AP (0.452–0.455)
  than the full-list guide (0.442), so the full list is not the best possible fixed guide. Choosing a guide must be done
  on the development split, not on these held-out numbers.

## OVD run 3a — YOLO-World fixed-guide ablation, DEVELOPMENT split only (500 pilot images), 23 Nov 2026, CPU
Command: `guide_ablation.py --pilot <pilot> --lvis-plan <plan_lvis.json>`; head = all 80 COCO names, independent
read-out, mAP50 over the 2,471 pilot pairs. File: `ovd/results/ablation_yoloworld_guide_dev.json`.
| guide | size | mAP50 |
|---|---|---|
| COCO 80 (the dataset's own label space) | 80 | **0.6033** |
| single class (guide = target) | 1 | 0.5803 |
| LVIS all | 1,203 | 0.5706 |
| LVIS random, seeds 0 / 1 | 20 | 0.5462 / 0.5954 |
| LVIS random, seeds 0 / 1 | 80 | 0.5660 / 0.5792 |
| LVIS random, seeds 0 / 1 | 200 | 0.5682 / 0.5681 |
| LVIS random, seeds 0 / 1 | 500 | 0.5697 / 0.5731 |
Conclusion (development data): the best fixed guide is the deployment domain's full label space, the choice fixed in
PLAN amendment A; out-of-domain guides are worse than single-class inference and small random guides are unstable.
No held-out number was used for this choice.

## OVD run 3b — error analysis on held-out runs 2a–2d, 23 Nov 2026, CPU
Command: `error_analysis.py --dataset {coco,lvis} --model {owlv2,yoloworld}`; files `ovd/results/error_*.json`.
Per class: AP50 drop k1 → full list (single-label output) vs. s_max = highest CLIP text similarity to another name.
| set / model | classes | Spearman(drop, s_max) [95% CI] | classes losing | mean drop by s_max quartile (low → high) |
|---|---|---|---|---|
| LVIS / OWLv2 | 390 | +0.178 [+0.088, +0.285] | 0.63 | +0.177, +0.171, +0.227, +0.253 |
| LVIS / YOLO-World | 390 | +0.252 [+0.164, +0.345] | 0.51 | +0.043, +0.084, +0.135, +0.196 |
| COCO / OWLv2 | 80 | +0.350 [+0.138, +0.537] | 0.70 | +0.007, +0.013, +0.046, +0.028 |
| COCO / YOLO-World | 80 | +0.158 [−0.077, +0.410] | 0.53 | +0.018, +0.003, +0.024, +0.019 |
Largest losers are near-synonym / part-whole pairs: calf–cow, teapot–teakettle, billboard–signboard, tiara–crown,
bullet train–train (OWLv2 LVIS); carton–box, stool–chair, toothpaste–toothbrush (YOLO-World LVIS); hot dog–sandwich,
snowboard–skateboard, truck–bus (OWLv2 COCO); bear–teddy bear, dog–cat (YOLO-World COCO). The correction restores
most of them (e.g. all eight top OWLv2 LVIS losers back to their k1 AP); exceptions for YOLO-World: dining table
(fix 0.00), Dixie cup (0.33), crape (0.50). Caveat: many LVIS classes have 1–2 instances in this sample, so per-class
APs are coarse (often 0 or 1).

## OVD run 3c — class-wise normalisation / reference calibration, DEVELOPMENT split, 23 Nov 2026, CPU
Command: `norm_ablation.py --model {owlv2,yoloworld}`; images split by hash into calibration / evaluation halves
(per-class score statistics from calibration images; numbers on evaluation images). YOLO-World uses the fixed COCO
guide, so only label competition varies. Files: `ovd/results/norm_ablation_*.json`.
| model | vocabulary | raw top-1 | z-norm top-1 | quantile top-1 | independent |
|---|---|---|---|---|---|
| OWLv2 mAP50 (flips vs k1) | near20 | 0.7305 (0.029) | 0.7259 (0.036) | 0.7233 (0.045) | 0.7548 (0.000) |
| OWLv2 mAP50 (flips vs k1) | all 80 | 0.7287 (0.032) | 0.7174 (0.052) | 0.7188 (0.073) | 0.7548 (0.000) |
| YOLO-World mAP50 (flips) | near20 | 0.6036 (0.001) | 0.5926 (0.005) | 0.5583 (0.029) | 0.6151 (0.000) |
| YOLO-World mAP50 (flips) | all 80 | 0.6023 (0.001) | 0.5923 (0.006) | 0.5854 (0.026) | 0.6151 (0.000) |
(k1: 0.7548 OWLv2, 0.6151 YOLO-World, identical for all variants.) **Negative result:** normalising or calibrating
scores before the top-1 decision makes interference and accuracy worse for both models; only removing label
competition removes it.

## Traceability check (23 Nov 2026)
`python ovd/check_numbers.py` → "107 distinct decimal numbers in the manuscript; 107 traced; 0 not traced".

## OVD run 2e — Grounding DINO on held-out COCO, CPU SUBSAMPLE (first 60 pairs, 82 instances), 24 Nov 2026
Commands: `vocab_full.py run --dataset coco --model gdino_qs --limit 60` and `--model gdino --limit 60` (1,380 runs incl.
FIX), `full_analysis.py --boot 1000`. Table: `ovd/results/run2e_coco_gdino_sub60_cpu.md`. Provisional (small n);
the full held-out run is on the owner's GPU.
- Deployed mAP50: k1 0.739; rand5 0.777, rand20 0.746; far5 0.776, far20 0.777; near5 0.717, near20 0.721;
  vis5 0.757, vis20 0.722; all 80 0.727; prune 0.558. Mean |score change| vs k1 0.076–0.118; flips 0.032–0.146.
  Instances flipping under some vocabulary 0.195. Order-only flips 0.000 (score range 0.0001).
  Absent-target false alarms: k1 0.292, every other vocabulary 0.000.
- **FIX (fixed context = all 80 names in canonical order, independent read-out):** 0.765; FIX − k1 +0.026
  [−0.059, +0.115]; FIX − all +0.039 [−0.004, +0.086]; FIX − prune +0.207 [+0.032, +0.345]. CIs wide at n = 60 pairs.
- **Query-selection probe** (82 instances): mean |score change| k1 → rand20_0: full model 0.102, query selection on
  target tokens only 0.105; k1 → all_0: 0.118 vs. 0.118 (signed +0.040 vs. +0.042). Restricting query selection does not
  reduce the effect → the vocabulary dependence arises in image–text fusion (feature enhancer / decoder cross-attention),
  not in language-guided query selection.

## OVD run 2f — Grounding DINO on held-out LVIS, CPU SUBSAMPLE (first 20 pairs), 24 Nov 2026 — chunking check
Command: `vocab_full.py run --dataset lvis --model gdino --limit 20` (620 runs), `full_analysis.py --boot 1000`.
Table: `ovd/results/run2f_lvis_gdino_sub20_cpu.md`. Very small sample: only the chunking result is used.
- The full 1,203-name list needs 14 chunks of 86–92 names; random 200/500-name vocabularies are chunked too
  (mean target chunk 85.1 / 86.8 names).
- **Order-only effect appears with chunking:** among the 5 orders of the full list, detection flips 0.012 and mean
  target-score range 0.039 (mean of per-pair max-score range 0.0377, median 0.0171), vs. 0.000 / 0.0001–0.0002 at
  80 names in one prompt (runs 1e, 2e). Order matters only through which names share the target's chunk.
- FIX (fixed canonical chunks) has zero order effect by construction. Other numbers (mAP50 k1 0.633, all 0.700,
  FIX 0.766; CIs touching zero) are too noisy at 20 pairs to report.

## OVD run 4a — latency, CPU (4-core development container; RELATIVE overhead only), held-out COCO, 24 Nov 2026
Command: `vocab_full.py bench --dataset coco --model {yoloworld,owlv2,gdino} --bench-images 3` (3 warm-up + 3 timed
images per setting; s/image). File: `ovd/results/bench_coco_cpu.json`. Absolute GPU latency comes from the owner's L4.
| model | method | full 80 | random 5 | random 20 |
|---|---|---|---|---|
| YOLO-World | original | 0.114 | 0.126 | 0.114 |
| YOLO-World | fixed guide (ours) | 0.108 | 0.105 | 0.116 |
| YOLO-World | single-class | 5.299 | 0.330 | 1.262 |
| OWLv2 | original (= independent read-out, one pass) | 3.241 | 3.060 | 3.149 |
| OWLv2 | single-class | 251.788 | 15.415 | 60.809 |
| Grounding DINO | original | 7.563 | 6.433 | 6.651 |
| Grounding DINO | fixed context (ours) | 7.256 | 7.381 | 7.682 |
| Grounding DINO | single-class | 344.863 | 24.025 | 90.213 |
Relative to original: fixed guide 0.83–1.02×, fixed context 0.96–1.16×, OWLv2 independent read-out 1.0× (same
pass); single-class 2.6–78× (46× for YOLO-World and Grounding DINO, 78× for OWLv2 at 80 names). Success criterion
(iii) (< 2× original at 80 classes) is met on CPU for all three corrections; GPU confirmation pending.

## Independent verification (24 Nov 2026)
`independent_check.py` recomputes headline mAP50 values (k1, full list, FIX) from the raw result files with separately
written IoU, matching and AP code (no code shared with `full_analysis.py`). All 15 values (COCO and LVIS for OWLv2 and
YOLO-World, COCO subsample for Grounding DINO) agree with the reported numbers; largest absolute difference 1.1e-16.
Output: `ovd/results/independent_check.txt`. Also: GPU vs. CPU reproduction of the pilot (run 1e) and the traceability
check (`check_numbers.py`, all manuscript numbers traced).

## OVD run 2g — Grounding DINO on held-out COCO, CPU SUBSAMPLE extended to the first 200 pairs (314 instances), 24 Nov 2026
Same commands as run 2e with `--limit 200` (resumed; 4,600 runs incl. FIX; probe 600 runs). Supersedes run 2e's
numbers. Table: `ovd/results/run2g_coco_gdino_sub200_cpu.md`. Still provisional relative to the full GPU run.
- Deployed mAP50: k1 0.769; rand5 0.791, rand20 0.771; far5 0.792, far20 0.787; near5 0.749, near20 0.745;
  vis5 0.792, vis20 0.764; all 80 0.753; prune 0.673. Mean |score change| vs k1 0.068–0.116; flips 0.086–0.134;
  localisation flips 0.068–0.169. Instances flipping under some vocabulary 0.191. Order-only flips 0.000 (range 0.0002).
  Absent-target false alarms: k1 0.159, all other vocabularies 0.000.
- **FIX (fixed context):** 0.782; FIX − all +0.028 [+0.0002, +0.060]; FIX − k1 +0.013 [−0.029, +0.054];
  FIX − prune +0.109 [+0.069, +0.195]; interference 0.
- **Query-selection probe** (314 instances): mean |score change| k1 → all_0: 0.116 full vs. 0.117 target-only query
  selection; k1 → rand20_0: 0.093 vs. 0.094. Confirms run 2e: the effect is not query-slot competition.

## OVD run 2h — Grounding DINO on held-out LVIS, CPU SUBSAMPLE extended to the first 100 pairs, 23 Nov 2026
Same command as run 2f with `--limit 100` (resumed; 3,100 runs). Supersedes run 2f. Table:
`ovd/results/run2h_lvis_gdino_sub100_cpu.md`. Provisional relative to the full GPU run.
- **Chunking order effect (confirms 2f):** among the 5 orders of all 1,203 names (≈ 14 chunks of ~87 names),
  detection flips 0.026, mean score range 0.0389 (per-pair max-score range mean 0.0329, median 0.0163).
- Interference: mean |score change| vs k1 0.089–0.126; flips 0.175–0.233; instances flipping under some vocabulary
  0.291. Absent (verified-negative) false alarms: k1 0.532; other vocabularies 0.000–0.064.
- Deployed mAP50: k1 0.700; rand5/20/80/200/500 0.781/0.783/0.751/0.771/0.727; far5/20/80 0.818/0.792/0.758;
  near5/20/80 0.735/0.680/0.695; vis5/20/80 0.763/0.758/0.760; all 0.751; prune 0.388.
- **FIX (fixed canonical chunks):** 0.756, interference 0; FIX − k1 +0.056 [−0.013, +0.113]; FIX − all +0.005
  [−0.022, +0.029]; FIX − prune +0.368 [+0.233, +0.467]. Not significant vs. k1 or all at n = 100 pairs.

## OVD run 4b — latency on the owner's Colab L4 GPU, held-out COCO, 23 Nov 2026
Command: `vocab_full.py bench --dataset coco --model {gdino,yoloworld,owlv2}` (30 images after 3 warm-up, CUDA-synchronised,
s/image). Code version 668d666 (research repo).
| model | method | full 80 | random 5 | random 20 |
|---|---|---|---|---|
| Grounding DINO | original | 0.406 | 0.356 | 0.371 |
| Grounding DINO | fixed context (ours) | 0.409 | 0.407 | 0.406 |
| Grounding DINO | single-class | 19.146 | 1.307 | 4.873 |
| YOLO-World | original | 0.016 | 0.015 | 0.015 |
| YOLO-World | fixed guide (ours) | 0.015 | 0.015 | 0.015 |
| YOLO-World | single-class | 0.621 | 0.045 | 0.160 |
| OWLv2 | original (= independent read-out) | 0.190 | 0.180 | 0.182 |
| OWLv2 | single-class | 14.190 | 0.888 | 3.553 |
Overhead of the corrections vs. original: Grounding DINO 1.01× (80), 1.14× (5), 1.09× (20); YOLO-World 0.94–1.00×;
OWLv2 1.0× (same pass). Single-class at 80 names: 47× (Grounding DINO), 39× (YOLO-World), 75× (OWLv2).
**Success criterion (iii) (< 2× at 80 classes): met for all three corrections on the L4.** The CPU ratios (run 4a) agree.

### Reproducibility of the held-out plans (owner's Colab vs. development container)
`md5sum plan_coco.json`: f92528394033c5b07545914e745113fd on both machines (identical COCO plan).
`md5sum plan_lvis.json`: 22114501f7a6a38314b3a9033cac7e02 (Colab) vs. 1b56ab8013aab9dd1b0f6c2e948f1992 (container) —
**the LVIS plans differ**. Rebuilding the LVIS plan a second time in the container gives the identical file
(1b56ab…; the code is deterministic on one machine), so the difference is between machines — most likely tiny
floating-point differences in CLIP embeddings reordering near-tied neighbours among 1,203 names (COCO, 80 names,
matched). `ovd/plan_fingerprint.py` hashes each part (images, pairs, each distractor family) to locate it. Consequence: Grounding DINO LVIS (Colab) and
OWLv2 / YOLO-World LVIS (container) are each internally consistent but may not use identical distractor lists.

## OVD runs 2i/2j/4c — FULL held-out Grounding DINO on the owner's L4 (COCO 1,000 images; LVIS 600 images) + LVIS latency, 23 Nov 2026
Commands: `vocab_full.py run --model {gdino_qs,gdino}` for coco and lvis (LVIS resumed after a disconnect and a GPU
usage limit; plan file unchanged, md5 22114501…), `bench --dataset lvis`, `full_analysis.py --boot 1000`,
`independent_check.py` (6/6 headline values agree, max diff 1.1e-16). Verbatim output: `ovd/results/run2_gdino_full_gpu.md`.
These supersede the CPU subsamples 2e–2h (which agree in direction).
- **COCO:** mAP50 k1 0.707, all 0.792, FIX 0.805; FIX − k1 +0.0975 [+0.085, +0.106]; FIX − all +0.013 [+0.009, +0.017];
  FIX − prune +0.116 [+0.106, +0.133]. Instances flipping under some vocabulary 0.278; flips vs k1 0.160–0.205; order-only
  flips 0.000. Absent false alarms k1 0.179 vs. ≤ 0.028 with distractors, FIX 0.004.
  Probe (7,342 instances): k1 → all mean |Δs| 0.140 full vs. 0.142 target-only query selection; k1 → rand20 0.110 vs 0.111.
- **LVIS:** mAP50 k1 0.496, all 0.508, FIX 0.558; FIX − k1 +0.062 [+0.040, +0.076]; FIX − all +0.050 [+0.043, +0.067];
  FIX − prune +0.302 [+0.292, +0.340]. near80 0.424, vis80 0.434, far80 0.519. Order-only flips 0.036 (score range
  0.029) — chunking. Instances flipping under some vocabulary 0.218. Absent false alarms k1 0.559. Probe (4,798):
  0.091 vs 0.091; 0.098 vs 0.097.
- **LVIS latency (L4, s/image):** Grounding DINO original full 4.284 / r5 0.363 / r20 0.369 / r80 0.410 / r200 0.962;
  fixed context 4.278 / 1.613 / 3.380 / 4.279 / 4.275 → overhead 1.00× (full), 4.4× (r5), 9.2× (r20), 10.4× (r80),
  4.4× (r200); single-class r5 1.286 / r20 4.801 / r80 18.831. YOLO-World fixed guide 0.016–0.017 vs original
  0.014–0.016 (1.0–1.14×; full 1.0×). OWLv2 independent read-out = original pass.
  **Criterion (iii) not met for Grounding DINO's fixed context on LVIS with small vocabularies**: it must run every
  canonical chunk that contains a requested class (up to 14 chunks). It is met for the full list and on COCO.

## Final verification (23 Nov 2026)
Paper: 0 TODOs; `check_numbers.py` → 214/214 traced. Figures redrawn from `ovd/results/figure_data_heldout.json`
(analysis summaries; Grounding DINO rows parsed from `run2_gdino_full_gpu.md`). Success criteria: see NOVELTY.md.
`independent_check.py --all-groups` (every table row, not only k1/all/FIX) on all raw files held in the container:
COCO OWLv2+YOLO-World 24 values, LVIS OWLv2+YOLO-World 36, Grounding DINO COCO subsample 12, LVIS subsample 18 —
90/90 agree, largest absolute difference 2.2e-16. Output: `ovd/results/independent_check_allgroups.txt`. (The full
Grounding DINO GPU files are on the owner's Drive; their headline values were checked on Colab, 6/6.)

## OVD run 4d — batched fixed context for Grounding DINO (engineering for criterion iii), 23 Nov 2026
`GDino.batch(img, chunks)`: all canonical chunks touched by a vocabulary in one forward pass (prompts padded to the
longest; image backbone computed once and expanded). Equivalence vs. per-chunk passes (CPU, 3 LVIS chunks, one image):
max |score difference| 3.3e-05, max box difference 0.9 px, identical top-20 boxes in every chunk. CPU time 28.7 s vs.
28.4 s (no CPU gain, as expected for a compute-bound CPU; a 12-chunk batch exceeded the container's 15 GB RAM).
GPU latency pending (`bench --dataset lvis --model gdino`, new method `fixctx_batch`, batches of ≤ 8 chunks).

## LVIS plan identity — RESOLVED (owner's Colab, code c00cb88, 23 Nov 2026)
`plan_fingerprint.py` on the Colab plan vs. the container plan: names 5df393fa6e, images 6c0fb871bf, gt 1255111891,
pairs e8e7bb9f47, all ce7c7c30da, far cc79b599f5, k 6657b21726, near 3576012995, rand d9f380a5c1 — **identical**.
Different: prune (Colab d921807c72 vs. 712e3f5d61) and vis (cfd5a3bf25 vs. 6065fe809a) — the only two lists built from
CLIP *image* embeddings, which differ slightly between machines. So all three detectors use the same LVIS images,
targets and random/unrelated/similar/full-list vocabularies; only the vis* and prune rows of Grounding DINO (Colab plan)
vs. OWLv2/YOLO-World (container plan) are built from slightly different lists.

## OVD run 4e — batched fixed context on the L4 (LVIS, s/image), 23 Nov 2026 — NEGATIVE
| vocabulary | original | fixed context | fixed context, batched (≤ 8 chunks) | single-class |
|---|---|---|---|---|
| full 1,203 | 4.210 | 4.301 | 4.178 | — |
| random 5 | 0.355 | 1.615 | 1.557 | 1.245 |
| random 20 | 0.361 | 3.398 | 3.300 | 4.627 |
| random 80 | 0.405 | 4.289 | 4.156 | 18.170 |
| random 200 | 0.968 | 4.292 | 4.162 | — |
(One CUDA out-of-memory warning during the batched runs; the allocator recovered and all timings completed.) Batching
saves only 2–4%: Grounding DINO's cost is per prompt, not per launch. **Criterion (iii) remains unmet for Grounding
DINO's fixed context with 5–200-name vocabularies drawn from the 1,203 LVIS classes (4.3–10.6×)**; met for the full list
and for every COCO vocabulary. Repeated original/fixed-context timings agree with run 4c within 2%.
`independent_check.py --all-groups` on the owner's full GPU Grounding DINO files (Colab, CPU runtime): 30 values
(COCO 12 + LVIS 18 groups) agree, largest absolute difference 1.1e-16. Independent verification is now complete for every
reported mAP: 6/6 + 30/30 (GPU) and 90/90 (CPU).

## OVD run 3d — anchored context for Grounding DINO (PLAN amendment C), DEVELOPMENT split, 23 Nov 2026 — NEGATIVE
Command: `anchor_dev.py --pilot <pilot> --lvis-plan <plan_lvis.json> --pairs 40` (720 runs, CPU; resumed once after a
container restart). Anchors: 40 fixed non-COCO LVIS names (first by MD5). Compared with original inference on the same
40 pairs (run 1d file).
| | mean abs score change vs own k1 | flips vs own k1 | mAP50 k1 / rand5 / near5 / rand20 / near20 / all80 |
|---|---|---|---|
| original | 0.1100 | 0.2204 | 0.748 / 0.820 / 0.812 / 0.783 / 0.805 / 0.765 |
| anchored | 0.0376 | 0.0920 | 0.728 / 0.731 / 0.734 / 0.758 / 0.780 / 0.767 |
Interference reduced by 66% (score) and 58% (flips), below the pre-set 80%; best mAP50 0.780 vs. 0.820 (−4 points), below
the pre-set accuracy criterion. **Not adopted; not run on held-out data.** The cost problem for small vocabularies from
large label spaces (run 4e) remains open.

## OVD run 3e — expert idea 1: image-selected shortlist (PLAN amendment D), held-out LVIS, CPU, 24 Nov 2026 — NEGATIVE
`shortlist.py rank` (CLIP ranking of all 1,203 names for each of the 600 images; mean shortlist 93.3 names) then
`replay` on the saved fixed-chunk outputs of the first 100 pairs (run 2h): instance survival 0.455 (small 0.310, medium
0.429, large 0.850); mAP50 fixed chunks 0.7558 → replay 0.5114. Real one-pass run on the same 100 pairs (24 images):
mAP50 k1 0.7002, all (order 0) 0.7688, fixed chunks 0.7558, **shortlist 0.5159**; target in shortlist for 0.380 of pairs.
Not adopted: whole-image CLIP ranking misses most small objects.

## OVD run 3f — expert idea 2, variant A (PLAN amendment E), same 100 LVIS pairs, CPU — NEGATIVE (scoring), boxes OK
`shared_readout.py --variant A`: (a) class-agnostic box recall of ground truth (n = 189) among the fixed prompt's 900
boxes vs. the target's canonical chunk: IoU 0.5 0.799 vs 0.852; IoU 0.75 0.683 vs 0.672. (b) mAP50 with raw projected
text tokens 0.3819 vs fixed chunks 0.7558. The shared boxes are nearly sufficient; raw text tokens are not.
Variant B (`read_only_text.py`): replay of the fixed prompt's own text through the encoder text side reproduces the
model's encoder text output exactly (max abs difference 0.0). Result pending.
- Variant B (read-only class tokens on the fixed pass's visual states; exact-replay validated): **mAP50 0.6643** vs fixed
  chunks 0.7558 (k1 0.7002, all 0.7688 on the same pairs). Invariant, one pass + cheap text replay. Not adopted (−9.2).
- Variant C (B + per-class decoder re-run; decoder replay validated exact, 0.0): **mAP50 0.6302**; CPU timing fixed
  pass 7.81 s/image, per-class decoder 1.406 s (≈ 18% of a pass). Not adopted; worse than B, so the residual gap is not
  in the decoder but most likely in the encoder's image features, which were fused with the fixed prompt's words.
Summary of the cost search (all pre-registered, all on held-out-free or fixed-rule evaluations): batching (−2–4% time),
anchored context (−4 AP, 58–66% interference cut), image shortlist (0.516), shared boxes + raw tokens (0.382), + read-only
tokens (0.664), + decoder re-run (0.630) vs fixed chunks 0.756. No one-pass invariant variant reaches the fixed chunks.

## OVD run 3g — training-free FIXED CLASS BANK for Grounding DINO (implemented and run by the owner's expert), reported 24 Nov 2026
**Status: VERIFIED (24 Nov 2026).** Source code (notebook cells, verbatim): `ovd/bank/fixed_bank_pilot_cells.py`,
`ovd/bank/validate_fixed_bank_cells.py`. Result files: `ovd/results/bank/`. Method: the 1,203 LVIS names encoded once by Grounding DINO's own BERT backbone in canonical
256-token chunks; per class, the mean of its name's contextualised wordpiece states → one cached vector per class;
at inference the 1,203-vector bank is injected at the text-backbone output (before text projection and the multimodal
encoder; text-length limit extended; identity text self-attention mask; zero text position ids). One encoder + query
selection + decoder pass per image with all classes in fusion; the request selects output columns only → list-invariant
by construction. Controls (expert): re-injecting a five-class prompt's own cached text states reproduces native boxes and
logits (atol = rtol = 1e-5); one class alone vs. inside a reversed 200-class list → identical outputs (1e-5).
- **Development pilot** (the 24 images / 100 pairs of runs 3d–3f, Tesla T4): mAP50 bank 0.76497 vs fixed chunks 0.75574
  (+0.92); latency bank 1.027 s vs standard five-class 0.609 s (1.685×).
- **Fresh validation** (method frozen; 100 LVIS images not in the pilot, fixed hash order; 467 image–target pairs;
  five-class requests = the image's planned targets, filled deterministically; one NVIDIA L4):
  | method | mAP50 | mean s/image |
  |---|---|---|
  | standard five-class Grounding DINO | 0.712894 | 0.355950 |
  | fixed bank (1,203 classes) | 0.660563 | 0.640633 |
  | canonical fixed chunks needed for the request | 0.643725 | 1.398597 |
  Bank − fixed chunks +1.68 AP (paired image-bootstrap 95% CI −1.65 to +3.97: observed pass of the −1-point margin;
  non-inferiority NOT established); 2.18× faster than fixed chunks. Bank vs standard five-class: 1.80× latency, −5.23 AP.
  Timing excludes disk I/O, export and one-time bank construction; latency measured for five-class requests only.
  Evaluator: `full_analysis.APTable`, independent per-class mode, project-sampled LVIS pairs (not official LVIS AP).
- Expert's additional negatives on the pilot: 2B shortlist with fixed-chunk substitution 0.6707 (43/53 present and 2/47
  absent targets selected; 62.30 s/image serial all-class scoring); hard / soft NMS on 2B detections 0.6699 / 0.6693.
- Files to import: `fixed_bank_l4.ipynb`, `validate_fixed_bank.ipynb`, `MyDrive/ovd/fixed_bank_1795484911264138629.json`,
  `MyDrive/ovd/bank_validation_1795488385837158854/{manifest.json, per_image.jsonl, summary.json}`.
- Reading against the pre-registered criteria: interference 0 (by construction); accuracy vs. the better baseline — on
  this validation the standard (uncorrected) five-class request is the better baseline and the bank is 5.2 AP below it,
  so criterion (ii) is NOT met there, while it is +1.7 vs. the previous invariant method; cost 1.80× standard (< 2×, met).

### Independent verification of run 3g (24 Nov 2026)
Separately written IoU/matching/AP code (`independent_check.py` functions) on `ovd/results/bank/…/per_image.jsonl` with the
container's LVIS plan (identical images/targets/ground truth, fingerprint-checked): bank 0.660563, stock5 0.712894, fixed
0.643725 — identical to `summary.json` (difference 0.0). Paired image bootstrap (300 resamples, own code) for bank − fixed:
[−0.0163, +0.0387] vs. reported [−0.0165, +0.0397]. The 100 validation images do not overlap the 24 pilot images; all
467 pairs are in the plan. Pilot (`fixed_bank_1795484911264138629.json`, 100 records): bank 0.764969 recomputed = reported.

## OVD run 3h — fixed class bank + classwise Gaussian Soft-NMS, 600-image "confirmation" (owner's expert), reported 24 Nov 2026
**Status: VERIFIED 24 Nov 2026 (see below).** Source to import:
`MyDrive/ovd/fixed_bank_soft_confirm_v1/{summary.json, manifest.json, per_image/}` (+ the code that produced it).
Reported (2,669 LVIS image–class pairs, five-class requests, `full_analysis.APTable` independent mode, NVIDIA L4):
| method | mAP50 | s/image |
|---|---|---|
| fixed bank + classwise Gaussian Soft-NMS | 0.65011 | 0.650 |
| standard five-class Grounding DINO (unmodified) | 0.63122 | 0.360 |
| standard five-class + Soft-NMS | 0.65008 | — |
| fixed chunks + Soft-NMS | 0.61749 | 1.399 |
| raw bank / raw chunks | 0.59424 / 0.59424 | — |
Reported: bank+Soft-NMS − standard +1.89 [−0.07, +2.66] (non-inferiority at −1 vs the unmodified standard prompt);
vs standard+Soft-NMS CI [−1.78, +0.91] (non-inferiority NOT established); vs chunks+Soft-NMS +3.26 [+1.67, +4.50]; 1.807×.
**Open checks before use:** (1) the LVIS plan has exactly 600 images / 2,689 pairs, so a 600-image set very likely
includes the 24 pilot + 100 validation images used to develop the bank and choose Soft-NMS — the confirmation must be
re-scored excluding them (and the data used to choose Soft-NMS and its parameters must be documented);
(2) raw bank and raw chunks identical to five decimals (0.59424) is implausible for different methods (they differed on
the 100-image validation: 0.661 vs 0.644) — check for duplicated detections; (3) standard five-class on this set is 0.631
vs 0.713 on the 100-image validation — explain (different images/requests?); (4) independent recomputation.

### Independent verification of run 3h (24 Nov 2026)
Files imported: `ovd/results/bank/fixed_bank_soft_confirm_v1/` (summary, manifest, overlap_audit, 600 per-image files;
the notebook that produced them was not included). Checks with separately written code and ground truth rebuilt from the
official `lvis_v1_val.json` (log: `independent_verification.log`):
- The 600 images are all LVIS v1 val images, disjoint from the 24 pilot images, the 100 validation images and the
  original 600-image plan (0 overlap each) — fresh test images. Soft-NMS settings in the manifest: σ 0.5, score
  threshold 0.02, top 50.
- All 2,669 targets: "present" flags agree with the LVIS annotations and every absent target is a verified negative (0
  mismatches).
- Open check (2) resolved: raw bank and raw chunks have identical detection lists for 933 pairs — all of them empty; all
  1,448 pairs where both have detections differ in top score. The 0.5942437 vs 0.5942388 near-tie is a coincidence.
- All six mAP50 values recomputed exactly (difference 0.0): bank 0.594244, bank_soft 0.650106, stock5 0.631216,
  stock5_soft 0.650080, fixed 0.594239, fixed_soft 0.617486.
- Paired image bootstrap (200 resamples, own code): bank_soft − stock5 [−0.0014, +0.0250] (reported [−0.0007, +0.0266]);
  − fixed_soft [+0.0178, +0.0434] (reported [+0.0167, +0.0450]); − stock5_soft [−0.0174, +0.0095] (reported
  [−0.0178, +0.0091]).
- Open check (3): the standard prompt's lower score here (0.631 vs 0.713 on the 100-image validation) reflects a
  different image sample; on both, the comparison is paired on identical images and requests.
**Conclusion:** bank + classwise Soft-NMS is non-inferior (−1-point margin) to the unmodified standard five-class prompt
(+1.89, CI lower bound −0.07 / −0.14), significantly better than fixed chunks + Soft-NMS (+3.26), list-invariant by
construction, at 1.81× standard latency. Against standard + Soft-NMS it is equal in point estimate (+0.003) but
non-inferiority is not established (CI lower −1.78). Open: provenance of the Soft-NMS settings (chosen on development
data?) and the producing notebook — to be added from the expert.
Derived: fixed_soft / bank_soft latency = 1.3991 / 0.6499 = 2.15× (bank faster).

## Expert review of the draft (24 Nov 2026) — actions
- Soft-NMS provenance (stated by the owner's expert): class-wise Gaussian Soft-NMS (σ 0.5, threshold 0.02, top 50) was
  selected on the 24-image pilot before the 600-image confirmation. The producing notebook is still to be added.
- Confirmation-set coverage (recomputed from `manifest.json`): 374 classes with a positive target; 126 of them on one
  image only.
- Related work: Ruis, Burghouts & Kuijf, "Fantastic Tractor-Dogs and How Not to Find Them with Open-Vocabulary
  Detectors", ICLR 2026 (title/authors read from the proceedings PDF) — early-fusion false positives on images without
  prompted objects, attention-sink tokens; now cited and contrasted. SL-Ground (Li, Yang, Jui, Lu; arXiv 2610.24814) and
  MM-Grounding-DINO (Zhao, Chen, Xu, Li, Wang, et al.; arXiv 2401.02361) verified from arXiv pages.

## Run 3h follow-up analyses from saved outputs (expert improvements #10, #8, #2), 24 Nov 2026
`python ovd/bank_analysis.py --lvis lvis_v1_val.json` (no inference; ground truth rebuilt from the official LVIS v1 val
file). Full output: `results/bank/bank_analysis_v1.txt`.
- **#10 Table 1 reproduction.** Point estimates reproduce exactly (stock5 0.6312, stock5_soft 0.6501, fixed 0.5942,
  fixed_soft 0.6175, bank 0.5942, bank_soft 0.6501). Our bootstrap (1,000 resamples, seed 0) gives CIs within 0.003 of
  the saved ones (e.g. bank_soft − stock5 [−0.0006, +0.0264] vs saved [−0.0007, +0.0266]); the difference is resampling noise.
- **#8 What drives the +3.26 over fixed_soft.** Not single-image classes: without the 126 classes that occur in one image
  the gain is +2.78 (248 classes). By object size (median box area): small +6.03, medium +2.39, large +0.51. By name
  length: one word +3.62, two or more words +2.20. By positive-image count: 1 → +4.22, 2–3 → +3.08, 4+ → +2.58.
- **#2 Duplicate boxes** (detection on an object that already has a higher-scoring match at IoU ≥ 0.5):
  raw bank 12,103 duplicates (24.9% of its false positives; 4,807 outrank a true positive of their class) vs raw fixed
  4,754 (16.4%; 1,646) and raw stock5 5,551 (8.6%; 1,696). Soft-NMS removes 91% of the bank's duplicates (to 1,145; 334
  outranking a TP). This explains why Soft-NMS adds +5.59 to the bank but +2.32 to fixed chunks: the bank produces many
  more duplicate boxes, and Soft-NMS removes them.

## Run 3h scored with the official LVIS evaluator (expert improvement #1), 24 Nov 2026
`python ovd/bank/official_lvis_eval.py --lvis lvis_v1_val.json --boot 100` (lvis-api 0.5.3, `LVISEval`, bbox, 300
detections per image; the only change to the package is restoring the `np.float` alias removed in NumPy 1.24). The
official ground truth is restricted to the evaluated pairs with LVIS's own fields (600 images, 5,292 boxes of the
present targets, 1,200 verified-negative pairs as `neg_category_ids`, `not_exhaustive_category_ids` restricted to the
targets). Output: `results/bank/official_lvis_eval.{json,log}`.

| method | official AP50 | study mAP50 | official AP | APr | APc | APf |
|---|---|---|---|---|---|---|
| standard five-class | 0.6314 | 0.6312 | 0.4933 | 0.3307 | 0.5289 | 0.4893 |
| standard + Soft-NMS | 0.6502 | 0.6501 | 0.4982 | 0.3376 | 0.5295 | 0.4950 |
| fixed chunks | 0.5946 | 0.5942 | 0.4760 | 0.2833 | 0.5221 | 0.4704 |
| fixed chunks + Soft-NMS | 0.6176 | 0.6175 | 0.4831 | 0.2833 | 0.5202 | 0.4795 |
| fixed bank | 0.5947 | 0.5942 | 0.4712 | 0.3750 | 0.5101 | 0.4652 |
| fixed bank + Soft-NMS | 0.6502 | 0.6501 | 0.5012 | 0.3750 | 0.5145 | 0.5009 |

Official AP50 differences (paired image bootstrap, 100 resamples, duplicated images re-indexed): bank + Soft-NMS −
standard +0.0188 [−0.0005, +0.0279]; − standard + Soft-NMS +0.0000 [−0.0173, +0.0102]; − chunks + Soft-NMS +0.0326
[+0.0160, +0.0469]. All conclusions of run 3h are unchanged under the official evaluator (values within 0.0005 of the
study's own; the official one uses 101-point interpolation). Over IoU 0.5:0.95 (official AP) bank + Soft-NMS is also
the highest (0.501 vs 0.493 standard, 0.498 standard + Soft-NMS).

## Run 3i — OWLv2 / YOLO-World corrections on the 600-image confirmation set (PLAN Amendment F), CPU, 24 Nov 2026
`python ovd/bank/replicate_owl_yolo.py run|analyze` — the run-3h images, 2,669 pairs and five-name requests (1,469 present
targets). Outputs and log: `results/bank/replication_owl_yolo/`. mAP50 (run-3h definition), paired image bootstrap
(1,000 resamples) of fix − condition; detection flips against the single-class output on present targets at each
model's deployment threshold (OWLv2 0.10, YOLO-World 0.25).

| model | condition | mAP50 | fix − condition [95% CI] | flips vs single class |
|---|---|---|---|---|
| OWLv2 | independent read-out (fix; = single class) | 0.7887 | — | 0 (by construction) |
| OWLv2 | five-name request, single-label output | 0.7733 | +0.0153 [+0.0063, +0.0201] | 4/1,469 (0.3%) |
| YOLO-World | fixed guide + independent read-out (fix) | 0.4904 | — | (list-invariant) |
| YOLO-World | five-name request | 0.4243 | +0.0661 [+0.0448, +0.0794] | 88/1,469 (6.0%) |
| YOLO-World | five-name request, single-label output | 0.4214 | +0.0690 [+0.0485, +0.0844] | 92/1,469 (6.3%) |
| YOLO-World | single class | 0.3793 | +0.1111 [+0.0831, +0.1223] | — |

Independent check: the official LVIS evaluator on the same outputs gives 0.7884 / 0.7731 / 0.4914 / 0.4251 / 0.4222 /
0.3801 (all within 0.001). Pre-registered criterion (ii) — the fix within 1 AP50 of the better baseline — is met for both
models on this third held-out sample (the fix is the best condition for both). With five-name requests the OWLv2
label-competition loss is small (1.5 points; it was 20.7 at 1,203 names), and YOLO-World's request is 6.0% flips from the
single-class output; the fixed guide differs from the single-class output on 20.8% of present targets, which is the
accuracy gain of the full guide (+11.1), not interference (its output does not depend on the request).
Run 3i with the same class-wise Gaussian Soft-NMS on every condition (`results/bank/replication_owl_yolo/analysis_softnms.txt`,
300 resamples): Soft-NMS changes nothing material for these detectors — OWLv2 fix 0.7905 vs request 0.7751 (+1.54
[+0.61, +2.04]); YOLO-World fix 0.4925 vs request 0.4224 (+7.01 [+4.61, +8.25]) and single class 0.3766. Their gains are
not a post-processing effect. COCO note: the 80 COCO names fit in one 256-token prompt, so for Grounding DINO on COCO the
fixed context is already one full-list pass (1.01–1.14× cost, run 4b) and the bank — introduced for LVIS's 1,203 names —
is not needed there.

## Run 3j — fixed-bank follow-ups on the owner's Colab T4 (expert improvements #3–#7, #9), 24 Nov 2026
Notebook `bank/followup_bank.ipynb` → `bank/followup.py` at commit e074876 (Tesla T4, torch 2.14.1, transformers 5.22.1);
same 600 images / 2,669 pairs as run 3h. Raw outputs: `results/bank/followup_t4_v1/followup_v1_data.zip`; analysis
(`bank/followup_analysis.py`, 1,000 paired image-bootstrap resamples, seed 0): `results/bank/followup_t4_v1/analysis.txt`.
**Checks.** (A) Reproduction of the saved L4 run 3h on 20 images (166 lists): top-score |diff| median 0.0004, max 0.015;
no empty/non-empty mismatch. Independent CPU runs of the same code: 15 images, parts B/C/F, 697 lists, max top-score
diff 0.0053; 295 images of part F, 3,951 lists, max 0.0123, median 0.00003, no empty/non-empty mismatch (large top-box
differences are swaps between near-tied boxes). `crosscheck_cpu15.txt`, `crosscheck_cpu295_tiny.txt`.

**#3 Image-independent requests** (each pair: target + 4 names drawn from the 1,203 without the image, two draws A/B):
standard prompt 0.6142 (A) / 0.6136 (B) vs 0.6312 with the run-3h requests (which were built from the image's own
targets); with Soft-NMS 0.6342 / 0.6342. Bank + Soft-NMS 0.6501 (list-invariant, so unchanged): − request A +0.0359
[+0.0162, +0.0432]; − request A + Soft-NMS +0.0159 [−0.0007, +0.0254] (B: +0.0159 [−0.0001, +0.0284]) — non-inferior at
the 1-point margin and close to superior. Single class 0.5260 (+Soft-NMS 0.5491). Interference on fresh images: the two
image-independent requests flip the standard prompt's detection (top score ≥ 0.35) for 132/1,469 present targets
(9.0%) and move the top box for 148 (10.1%); single class vs request A: 578 (39.3%) flips.
**#4 Dictionary.** Class vectors do not depend on the other names: rebuilt from two random orderings and from each name
alone they equal the canonical bank to ≤ 1.8e-6 (cosine 1.0000). Which classes are in the bank does matter a little:
banks with a fixed random half of the classes, on the pairs they contain (+Soft-NMS): half A 0.6597 vs full 0.6503
(full − half −0.0094 [−0.0196, +0.0064]; 1,301 pairs), half B 0.6475 vs 0.6611 (+0.0137 [−0.0021, +0.0293]; 1,265
pairs) — about ±1 point, neither significant. The bank removes dependence on the request, not on the chosen dictionary.
**#6 Construction ablations** (+Soft-NMS; canonical = mean wordpiece, identity mask: 0.6494): first wordpiece 0.6400
(canonical − variant +0.0094 [+0.0012, +0.0148]); last wordpiece 0.6430 (+0.0064 [−0.0018, +0.0133]); full text mask
0.6409 (+0.0085 [−0.0009, +0.0204]); raw (no Soft-NMS): full mask −0.0134 [+0.0017, +0.0239] worse. The expert's
choices are the best of those tested.
**#5 Cost vs vocabulary size** (30 images, T4, median s/image; peak extra GPU memory): standard prompt 0.659 s (K=1),
0.674 (5), 0.737 (50), 1.187 (100, 2 chunks), 1.713 (200), 2.745 (400), 5.206 (800), 7.306 s (1,203, 14 chunks), memory
1.2–1.5 GB; fixed chunks 0.777 (1) to 7.30 s (≥ 50 names: all 14 chunks touched); bank 1.153–1.159 s for every K,
2.7 GB. Crossover with the standard prompt at about K = 100 (1.156 vs 1.187 s); at 1,203 names the bank is 6.3× faster;
at K = 5 it costs 1.71× (T4; 1.81× on the L4). One-time set-up: 0.29 s, 3.7 MB of class vectors.
**#7 Second checkpoint (grounding-dino-tiny) — NEGATIVE for the bank.** Parity check passed. mAP50: standard five-class
0.6848 (+Soft-NMS 0.6936), fixed chunks 0.6596 (+Soft-NMS 0.6783), bank 0.5651, bank + Soft-NMS 0.6273. Bank +
Soft-NMS − standard −0.0575 [−0.0735, −0.0451]; − fixed chunks + Soft-NMS −0.0510 [−0.0622, −0.0370]. Time: bank 0.917 s,
standard 0.458 s (2.00×), chunks 2.121 s. The bank does not transfer to the tiny checkpoint: there it is worse than both
the standard prompt and the other invariant method (fixed chunks), and its cost reaches the 2× limit. Fixed chunks
themselves are 2.52 points below the standard prompt on tiny (0.6596 vs 0.6848; with Soft-NMS 0.6783 vs 0.6936).
**#9 Qualitative figure** (`paper/fig_qual.pdf`, `bank/make_qual_figure.py`): fixed selection rule (present pairs whose
standard-prompt detection flips between the two image-independent requests, ordered by MD5): 100 of 1,469 pairs
(6.8%) qualify; the bank + Soft-NMS detects 45 of them at ≥ 0.35 and has a correct top box on 91.

## Expert revision package (24 Nov 2026) — imported after independent checks
- **Producing notebook of run 3h** (`bank/fixed_bank_soft_confirm.ipynb`, previously missing). Its first cell states the
  primary test before results (lower paired-bootstrap bound of bank + Soft-NMS − unmodified five-class prompt above −0.01
  and time under 2×). Checked here against the saved run: its selection rules reproduce the manifest exactly — 600/600
  images in its SHA-256 order, the targets of 600/600 images (3 present by MD5 order, 2 verified absent), and the
  five-name requests of 600/600 images. Its Soft-NMS, export and image-bootstrap code match those verified earlier.
- **Request sensitivity** (`results/bank/request_sensitivity/`: `request_audit.py`, `.json`, `.pdf`; exploratory, on the
  saved run-3j outputs, 3,000 paired image resamples). Re-run here on our supplement: byte-identical output. At the 0.35
  threshold the two image-independent requests trigger only for A on 67 present pairs and only for B on 65: a net
  difference of two detections, but 132/1,469 decisions changed (9.0%, CI 7.5–10.6%). On verified-absent targets 14/1,200
  change (1.2%, CI 0.6–1.8%). Our own recomputation (seed 0): 67 / 65 / 132, CI 7.5–10.5%; absent 6 / 8 / 14, CI 0.6–1.8%.
  Across thresholds 0.20–0.50 the present-pair change rate is 5.6–10.1%. Interference largely cancels in aggregate
  counts while changing individual decisions.
- The revised PDF in the package was built from an earlier source and omits two later statements (on grounding-dino-tiny
  no invariant method meets the 1-point criterion); those statements are kept. Only the page-3 text edits were adopted.

## Run 3k (G2) — what a change of request does to individual objects, saved run-3j outputs, CPU, 24 Nov 2026
`python ovd/bank/instance_effects.py --data results/bank/followup_t4_v1/followup_v1_data.zip` → `results/bank/instance_effects.{json,txt}`.
Grounding DINO base, standard five-class prompt, the two image-independent requests A/B, 1,469 present pairs with 5,292
ground-truth instances; detections ≥ 0.35 greedily matched at IoU ≥ 0.5. Found under both requests 857, only A 96, only B
85, neither 4,254. 181 instances (3.4% of all, image-bootstrap CI 2.7–4.2%; 17.4% of the 1,038 found under either) are
found under just one request. Where the top box passes 0.35 and matches under both requests (515 pairs), it switches to a
different instance of the class in 23 (4.5%; 23 of 206 multi-instance pairs). A change of request therefore mostly decides
*whether* an object is found, rarely *which* correct box ranks first.

## Run 3l — Amendment G1/G3: full-image LVIS test and tiny diagnosis, owner's Colab L4, 24 Nov 2026
`bank/full_image_bank.ipynb` → `bank/full_image.py` (commit 6520d59; NVIDIA L4, torch 2.14.1, transformers 5.22.1).
Outputs: `results/bank/full_image_v1/full_image_v1_outputs.zip`; analysis (`bank/full_image_analysis.py`, lvis-api,
200 paired image resamples on 4 workers): `results/bank/full_image_v1/analysis.{txt,json}`.
**Checks.** Part H reproduces the saved run-3h top scores exactly on the same GPU type (5,177 stock5 and 5,066 bank lists,
max |diff| 0.0000). Part I vs the CPU smoke run on the same 2 images: sorted top-300 scores within 0.007; the
differences are rank swaps between near-tied classes.
**G1 — standard full-image LVIS protocol (1,000 fresh images, 13,073 boxes, all 1,203 classes, 300 dets/image):**

| method | AP | AP50 | AP75 | APr | APc | APf | APs | APm | APl |
|---|---|---|---|---|---|---|---|---|---|
| chunks + Soft-NMS | 0.3982 | 0.5066 | 0.4186 | 0.3376 | 0.4048 | 0.3985 | 0.2251 | 0.5308 | 0.6901 |
| bank + Soft-NMS | 0.3779 | 0.4941 | 0.3953 | 0.2478 | 0.3379 | 0.3975 | 0.2341 | 0.4957 | 0.6382 |
| chunks (raw) | 0.3848 | 0.4770 | 0.4073 | 0.3538 | 0.4042 | 0.3794 | 0.2093 | 0.5230 | 0.6717 |
| bank (raw) | 0.3437 | 0.4344 | 0.3626 | 0.2503 | 0.3125 | 0.3587 | 0.2242 | 0.4613 | 0.5929 |

Bank + Soft-NMS − chunks + Soft-NMS: AP −0.0203 [−0.0256, −0.0053]; AP50 −0.0126 [−0.0163, +0.0052]. **NEGATIVE for the
bank under the standard protocol**: significantly lower AP; on AP50 non-inferiority at −1 point is not established
(lower bound −1.63). The loss is in common and rare classes (APc −6.7, APr −9.0 points); frequent classes are equal (APf
−0.1). Time: bank 0.632 s/image vs chunks 4.215 s (6.67× faster); median peak extra GPU memory 2,452 vs 1,376 MB.
The run-3h sampled-pair advantage (+3.26 mAP50 over chunks) therefore does not carry over to the full-image protocol.
Budget check (`budget_use.txt`): the bank does not spend more of the 300 detections on verified-absent classes (0.6% for
both) and covers more present frequent classes (85% vs 78% with ≥ 1 detection) but fewer common (60% vs 67%) and rare
(38% vs 44%; only 16 instances) ones — so the loss is not budget crowding by false alarms (our first hypothesis, rejected).
**G3 — tiny diagnosis** (600 confirmation images, 5,292 present instances; candidate recall = any of 900 boxes at IoU ≥ 0.5):
base: standard 0.845 / bank 0.842, median rank of the best match 5 / 5, its median score 0.208 / 0.143;
tiny: standard 0.845 / bank 0.797, median rank 5 / 6, median score 0.290 / 0.157. On base the bank keeps the boxes and
their ranking but lowers the score scale; on tiny it also loses candidate boxes (−4.8 points recall), i.e. the bank's
text input degrades the tiny model's box proposals, not only its scores.

## Run 3m (Amendment H) — bank-proposed context ("hybrid"): development stage, CPU, 24 Nov 2026
`bank/hybrid.py --mode dev` (124 development images = 24 pilot + 100 validation; CPU, ≈ 38 s/image for all three K);
analysis `bank/hybrid_analysis.py` (official LVIS evaluator, 300 dets/image). Outputs `results/bank/hybrid/dev_outputs.zip`,
`dev_analysis.json`.

| method (Soft-NMS) | AP | AP50 | APr | APc | APf |
|---|---|---|---|---|---|
| bank | 0.4720 | 0.5932 | 0.1334 | 0.3004 | 0.4916 |
| hybrid K=10 | 0.4663 | 0.5894 | 0.1334 | 0.3012 | 0.4852 |
| hybrid K=25 | 0.4544 | 0.5890 | 0.0852 | 0.3121 | 0.4718 |
| hybrid K=50 | 0.4375 | 0.5752 | 0.1411 | 0.3146 | 0.4522 |

Pre-registered rule (best development AP with Soft-NMS; ties → smaller K) selects **K = 10**, frozen for the test. Note:
no K beats the bank alone on the development images; AP falls as K grows (1 extra prompt pass in all cases: mean 10.0 /
25.0 / 47.7 candidates). The plan has no stop rule, so the frozen 1,000-image test runs as registered (CPU, started
24 Nov 2026 20:4x, ≈ 20 s/image).

**Exploratory diagnosis (written after the development result, before any test result).**
(a) `bank/hybrid_diag.py` on the development images (`dev_diag_k10.json`, `dev_diag_k50.json`): the short candidate prompt
raises the scores of candidate rows, for present and absent classes alike (median top score, K=10: present 0.405 → 0.553,
absent 0.308 → 0.435), so candidates take more of the 300 slots (23.0% → 28.8%; K=50: 72.1% → 83.5%). Classes that were
never candidates lose AP (K=10 summed change −1.06 over 121 classes; K=50 −3.33 over 55), and the candidate classes
themselves do not gain (K=10 mean per-class AP 0.6264 → 0.6236, 43 up / 59 down; K=50 0.5831 → 0.5559).
(b) `bank/hybrid_ceiling.py` on the run-3l outputs only (`ceiling_run3l.txt`; 1,000 images, 520 classes with ground
truth): the chunks − bank gap of +0.0203 mean per-class AP comes from common (+0.0169) and rare (+0.0028) classes, and
almost all of it from classes that are among the bank's own top-K classes in fewer than half of the images where they are
present (K=10: +0.0202 of +0.0203; K=50: +0.0201). Present classes are bank top-10 candidates in 47% of (image, class)
pairs (rare 25%, common 21%, frequent 37% per class on average). So the bank's full-image loss is a failure to
*propose* the right common/rare classes, which a bank-proposed shortlist cannot repair by construction; the hybrid is
expected to stay near the bank on the test. Both analyses are exploratory and not used to change the frozen test.
