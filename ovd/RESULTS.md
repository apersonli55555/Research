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
