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

## Amendment A (22 Nov 2026, before any Stage 2/3 result and before the pilot result) — implementation details
Code: `ovd/vocab_full.py` (runs, corrections, latency) and `ovd/full_analysis.py` (all statistics).
- **Sizes and counts:** COCO held-out = 1,000 images (ranks 501–1,500 by hash; compute budget), conditions k=1,
  random 5/20 (5 draws), near/vis/far 5/20, all 80 (5 orders) = 22 per pair. LVIS = 600 images, ≤ 3 present targets
  per image (hash-chosen) + 2 verified-absent targets (`neg_category_ids`); k=1, random 5/20/80/200/500 (3 draws),
  near/vis/far 5/20/80, all 1,203 (5 orders) = 30 per pair.
- **Visually confusable distractors (replaces train-crop prototypes, which would need the 1 GB LVIS train file):**
  the classes CLIP ViT-B/32 ranks highest for the target's own largest ground-truth crop (whole image for absent
  targets). It uses no label beyond the target box every condition already uses. **Unrelated** = random classes from
  the half of the list least similar to the target (CLIP text).
- **Chunking:** a Grounding DINO vocabulary over 256 tokens is packed in its order into consecutive chunks; only the
  target's chunk is run. Because chunk membership depends on order, an order effect at 1,203 classes is expected to
  come from chunking; the analysis reports chunk size per condition so this is visible.
- **Corrections, defined before results:** Grounding DINO *fixed context* (each class is scored inside a fixed
  canonical chunk of the dataset's full list; the deployment vocabulary only selects which classes are output);
  YOLO-World *fixed guide* (the C2fAttn guide text is the fixed full list; the deployment vocabulary is used only in the
  classification head); all models *independent read-out* (no top-1 label competition). These are vocabulary-invariant
  by construction, so their interference is 0 by definition; what is tested is (ii) accuracy and (iii) latency.
- **Interference of a method** = change of its own output between vocabulary V and its own single-class output.
- **Mechanism probe:** Grounding DINO with query selection restricted to the target's tokens (`gdino_qs`) on k1,
  rand20_0 and all_0; if most of the k1→V change remains, query-slot competition is not the main cause.
- **Accuracy reference ("strongest relevant baseline"):** the better of single-class inference (k1) and original
  inference with the full list (`all`), by deployed mAP50 on the same pairs.
- **Latency:** `bench` stage, 30 images after 3 warm-up images, CUDA-synchronised, per image, for original, single-class
  and the correction, at full list and random 5/20 (/80/200 for LVIS) vocabularies.

## Amendment B (22 Nov 2026, before any pilot result was seen) — GO rule made two-sided
The original rules (a) and (b) only counted *losses* from k=1 to all 80 (AP drop; instances lost). The CPU smoke
tests showed that Grounding DINO's target score can *rise* with more classes, and the goal asks for "differences",
so an effect in either direction must count. New rule: (a) |mAP50(k=1) − mAP50(all 80)| ≥ 0.03 with the 95% CI
excluding zero; (b) ≥ 10% of target instances change detected/missed status between k=1 and all 80 (either direction,
mean over the 5 orders); (c) unchanged. `lost_at_80` is still reported. No pilot output had been produced when this
was changed (the pilot cell was handed to the owner but its results were not yet reported back).

## Amendment C (26 Nov 2026, before any result of this method) — anchored context for Grounding DINO (cost)
Motivation: exact invariance (fixed canonical chunks) costs 4–10× for small vocabularies from the 1,203 LVIS names.
Method "anchored context": one prompt = A ∪ V, where A is a FIXED anchor list (the 40 LVIS names that are not COCO
names, first by MD5 of the name), A before V; independent read-out. One forward pass when it fits 256 tokens.
Not invariant by construction; its interference is measured. Design/selection on the DEVELOPMENT split only
(first 40 pilot pairs, CPU, compared with original inference on the same pairs from run 1d); success = interference
(mean |score change| vs. the method's own single-class output, and detection flips) reduced ≥ 80% vs. original and
mAP50 not lower than original's best condition minus 1 point. Only if it passes is it run on held-out LVIS.
Clarification (before any result; the first attempt crashed after 13 runs on an over-long prompt): if A ∪ V exceeds
256 tokens, V is split in order into consecutive pieces that each fit together with A, and the piece holding the target
is run (the same rule as standard chunking, with A always included).
