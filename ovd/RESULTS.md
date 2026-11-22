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
