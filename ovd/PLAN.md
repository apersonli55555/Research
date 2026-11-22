# Vocabulary-Set Interference in Open-Vocabulary Object Detection — study plan

Status: **fixed before any result on real data** (22 Nov 2026). Changes after a result are allowed only as clearly
labelled follow-ups, recorded in `ovd/RESULTS.md` with the date and the reason.

## Question
For a fixed image and a fixed target class whose prompt text is unchanged, do the *other* class names in an
open-vocabulary detector's query vocabulary change the detector's output for that target (scores, boxes, AP,
detections at an operating threshold)? Can a training-free inference change remove most of that dependence without
losing accuracy?

## Hypotheses (directional, stated before data)
- **H1 (set interference):** for early-fusion or text-conditioned detectors (Grounding DINO, YOLO-World) the target's
  score and detections depend on which other classes are in the vocabulary.
- **H2 (order):** reordering a fixed vocabulary has much smaller effects than changing its membership.
- **H3 (similarity):** semantically similar distractors cause more interference than random ones of equal number.
- **H4 (mechanism):** OWLv2 scores each query independently, so its raw class scores cannot change; any change in its
  deployed output comes only from label competition (each box keeps its top class). Smoke tests on 2 COCO pairs
  already showed identical raw scores across vocabularies (expected by construction, not a result).

## Data splits (no new labels)
- **Development (method design, thresholds, all choices):** the 500 COCO val2017 images with the lowest MD5 hash of
  the file name among images with at least one non-crowd annotation.
- **Held-out COCO:** the next 2,000 images by the same hash order (disjoint).
- **Held-out LVIS v1.0 val:** images chosen by the same hash rule, excluding every image in the development set;
  used for vocabulary sizes 200, 500 and 1,203. Nothing is tuned on held-out data.

## Models (fixed checkpoints)
Grounding DINO base (`IDEA-Research/grounding-dino-base`), OWLv2 base patch16 ensemble
(`google/owlv2-base-patch16-ensemble`), YOLO-World v2 S (`yolov8s-worldv2.pt`, ultralytics). Deployment thresholds:
0.35, 0.10, 0.25. Prompts: Grounding DINO `"c1 . c2 . … ."`; OWLv2 `"a photo of a {c}"`; YOLO-World class names.

## Stage 1: pilot (OVD run 1), development split only — `ovd/vocab_pilot.py`
Targets per image: every present COCO class + 2 absent classes chosen by hash. Conditions per target: k=1;
random distractors k=5 and k=20 (5 draws each, random order); nearest-by-CLIP-text distractors k=5 and k=20;
all 80 classes in 5 random orders. Seeds derive from the MD5 of image and target.

**GO rule (fixed):** continue to Stage 2 if, for at least one model, (a) mAP50(k=1) − mAP50(all 80) ≥ 0.03 with a
95% bootstrap CI (resampled by image) above zero; or (b) ≥ 10% of target instances detected at k=1 are missed with all
80 classes (mean over the 5 orders); or (c) ≥ 10% of target instances flip detected/missed between orders of the
same 80 classes. Otherwise STOP and write up the negative result (and recommend the runner-up topic, monotone
camera-tone invariance in monocular depth).

## Stage 2: full characterisation (if GO)
Vocabulary sizes 1, 5, 20, 80 (COCO) and 200, 500, 1,203 (LVIS), with ≥ 5 seeded draws or orders per condition.
Distractor families: random; semantically similar (nearest CLIP text embeddings); visually confusable (nearest class
prototypes = mean CLIP image embedding of up to 50 COCO/LVIS *train* crops per class, never test crops); order-only.
Measurements: mAP50 and COCO mAP, target score shift, best-box IoU shift, detection flips at the deployment
threshold, absent-class false alarms, per-class robustness, latency. Grounding DINO's 256-token limit: vocabularies
that do not fit are split into consecutive chunks that fit (the standard practice), and chunking is analysed as its
own factor.

**Mechanism separation:** for each model, compare raw class scores vs. deployed outputs (top-1 label assignment,
threshold, NMS where the model applies it). For Grounding DINO, additionally ablate the text–image fusion (per-class
independent passes) to separate fusion interference from label competition.

## Stage 3: training-free correction (designed on the development split only)
Candidates: independent class scoring (one pass per class or per fixed small group); fixed-reference anchor
vocabulary (score every target inside the same fixed reference set, independent of the deployment vocabulary);
class-wise score normalisation against a reference distribution; cached text embeddings for speed. Interference
metric: mean |target-score change| vs. k=1 and detection-flip rate, averaged over sizes and distractor families.
**Success criteria (fixed now, evaluated once on held-out data):** (i) interference reduced by ≥ 80% relative to the
uncorrected model; (ii) mAP within 1 point of the better of uncorrected and single-class inference; (iii) latency
< 2× uncorrected for an 80-class vocabulary where feasible. Each criterion is reported separately whatever the outcome.

## Statistics
Bootstrap 95% CIs resampled by image (1,000 resamples in final runs); paired comparisons on identical images; no
tuning on held-out data; every number in the paper copied from `ovd/RESULTS.md`, which records every run, including
failed ones.
