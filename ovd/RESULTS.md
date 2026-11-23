# Results log — vocabulary-set interference

Every run is recorded here, including failed ones, with its date and the exact command. Paper numbers come only
from this file.

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
Relative to original: fixed guide 0.83–1.02×, fixed context 0.96–1.15×, OWLv2 independent read-out 1.0× (same
pass); single-class 2.6–78× (46× for YOLO-World and Grounding DINO, 78× for OWLv2 at 80 names). Success criterion
(iii) (< 2× original at 80 classes) is met on CPU for all three corrections; GPU confirmation pending.
