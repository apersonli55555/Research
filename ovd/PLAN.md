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

## Amendment C (23 Nov 2026, before any result of this method) — anchored context for Grounding DINO (cost)
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

## Amendment D (24 Nov 2026, before any result) — image-selected shortlist for Grounding DINO (suggested by the owner's expert)
Method "shortlist": for each image, rank ALL names of the label space by CLIP ViT-B/32 (laion2b_s34b_b79k) similarity
between the whole image and "a photo of a {name}" (the same CLIP model as the distractor families); pack the
highest-ranked names, in rank order, into ONE Grounding DINO prompt within the 256-token limit; run once; independent
read-out; a requested class outside the shortlist returns no detection. The prompt depends only on the image, so the
output is invariant to the requested vocabulary by construction. No free parameters (shortlist length = what fits).
Evaluation: (1) replay diagnostic on saved fixed-chunk outputs (drop omitted classes; instance survival rate by
object size; AP); (2) the real one-pass run on the held-out LVIS images, compared with k1 / full list / fixed chunks on
the same pairs; (3) latency = one pass + CLIP. Success = the run-3 criteria: interference 0, mAP within 1 point of the
better baseline, cost < 2× original.

## Amendment E (24 Nov 2026, before any result) — shared fixed-prompt boxes + independent class readout (expert idea 2)
One Grounding DINO pass with a FIXED prompt F (the 80 COCO names, canonical order) gives the 900 decoder boxes and
last-layer query embeddings. Each requested class c is scored independently: s_c(q) = sigmoid(max over c's tokens of
q · t), where t are c's projected text tokens from the text backbone run on "c ." alone (cacheable, image-independent).
Invariant by construction; one pass + cheap text. Tests on the same 100 held-out LVIS pairs as run 2h: (a) class-agnostic
box recall of ground truth at IoU 0.5 / 0.75 among F's 900 boxes vs. the target's canonical-chunk 900 boxes; (b) mAP50
vs. fixed chunks. Adopt only if mAP50 is within 1 point of the fixed chunks (no tuning on these pairs).
Amendment E2 (24 Nov 2026, after variant A's result, before any variant-B result): variant B scores class c with its
text tokens obtained by replaying ONLY the text side of each encoder layer ("c ." alone) against the fixed pass's saved
per-layer visual states (`read_only_text.py`; validated: replaying the fixed prompt's own text reproduces the model's
encoder text output with max abs difference 0.0). Visual states, proposals and decoder queries come only from F.
Same 100 pairs, same adoption rule (within 1 mAP50 point of the fixed chunks).
Amendment E3 (24 Nov 2026, after variant B's result 0.664, before any variant-C result): variant C = variant B's
read-only class tokens, plus a re-run of ONLY the decoder per class (same queries, proposals and visual memory from the
fixed pass; the decoder cross-attends only to the class's tokens); boxes and scores from the model's own heads.
Validated: replaying the decoder with the fixed prompt's own text reproduces the model's boxes and logits exactly (0.0).
Invariant by construction. Same 100 pairs, same adoption rule; cost reported as fixed pass + per-class decoder time.

## Amendment F (24 Nov 2026, before any result): OWLv2 / YOLO-World corrections on the 600-image confirmation set
Replication of the two non-Grounding-DINO corrections on a third held-out sample (the run-3h images, targets and
five-name requests; `bank/replicate_owl_yolo.py`; CPU). Conditions: OWLv2 five-name request with single-label output vs
independent read-out; YOLO-World five-name request (raw and single-label), single class, fixed guide + independent
read-out. Reported: mAP50 (run-3h definition) with paired image-bootstrap CIs of fix − each condition, and detection
flips against the single-class output at each model's deployment threshold. All conditions and all results are reported,
whatever their direction. The criterion is the pre-registered (ii): the fix within 1 AP50 point of the better baseline.

## Amendment G (24 Nov 2026, before any result): full-image LVIS evaluation, instance-level request effects, tiny diagnosis
Prompted by an external review asking whether the bank's sampled-pair result holds under the standard benchmark protocol.
**G1 Full-image LVIS (GPU, `bank/full_image.py` part I).** 1,000 LVIS v1 val images never used before (manifest
`bank/full_image_manifest.json`: images with ≥ 1 annotation, excluding the 2,700 images of every earlier set, ordered by
sha256('full-image-v1/'+file name)). Grounding DINO base scores all 1,203 classes per image with (a) the canonical
256-token chunks — this is also the standard full-vocabulary inference, since the full list in canonical order *is* the
chunks — and (b) the fixed class bank (frozen, as in run 3h). Per class the top 50 boxes with score ≥ 0.02 are kept;
raw = the top 300 (box, class, score) per image; Soft-NMS = class-wise Gaussian Soft-NMS (σ 0.5, threshold 0.02, top 50
per class, as frozen in run 3h) then the top 300 per image. Scored with the official LVIS evaluator (lvis-api; federated
pos/neg sets, not-exhaustive ignore, 300 detections per image) on these images. Primary: AP and AP50 of bank + Soft-NMS −
chunks + Soft-NMS, paired image bootstrap (95% CI). Also reported: raw variants, APr/c/f, APs/m/l, seconds per image
(model passes to score tensors) and peak GPU memory. Non-inferiority margin for the bank: −1 AP50 point (as before).
All results reported whatever their direction; the bank, Soft-NMS and thresholds are not changed after seeing results.
**G2 Instance-level effect of the request (CPU, saved run-3j outputs).** For every present ground-truth instance and the
two image-independent requests A/B: is it matched (IoU ≥ 0.5, greedy by score, detections with score ≥ 0.35) under both,
only one, or neither; and how often the top-scoring box moves to a different instance of the same class.
**G3 Why the bank fails on tiny (GPU, part H; diagnostic only, nothing is selected with it).** On the 600 confirmation
images, for each present target instance, base and tiny, standard five-class prompt and bank: candidate recall (does
any of the 900 boxes reach IoU ≥ 0.5) and the score rank of the best matching box. Lower recall points to the boxes;
equal recall with worse ranks points to the scores.

## Amendment H (24 Nov 2026, before any result): bank-proposed context ("hybrid") for Grounding DINO
Motivation: run 3l — the bank is 6.7× faster than exact chunks but 2.0 AP worse on the full-image protocol (common/rare
classes), while on base it keeps the right boxes and their ranking (G3). Method (training-free, request-invariant by
construction because every step depends only on the image):
 1. one bank pass (frozen, as in run 3h) → its raw top-300 (class, score, box) list (as in run 3l);
 2. candidate classes = the first K distinct classes in that list (= the K classes with the highest bank score);
 3. one standard Grounding DINO pass whose prompt is the candidates in canonical LVIS order (split into ≤ 256-token
    chunks if needed); candidates are scored from this pass, all other classes keep their bank scores;
 4. per class top 50 boxes ≥ 0.02, class-wise Gaussian Soft-NMS (σ 0.5), top 300 per image — as in run 3l.
K ∈ {10, 25, 50} is chosen ONLY on the development images (the 24 pilot + 100 validation LVIS images; official full-image
LVIS AP with Soft-NMS; ties → smaller K). Then ONE test on the 1,000 run-3l images (not used for developing this method)
against the run-3l chunks + Soft-NMS (primary: AP and AP50 difference, paired image bootstrap) and bank + Soft-NMS.
Cost reported as forward passes (1 bank + ⌈candidate tokens/256⌉ prompt passes vs 14 for chunks) and, if a GPU run is
available, seconds. Inference for this amendment runs on CPU (numerical differences to the L4 outputs ≤ 0.007 in score,
run 3l check). All results reported whatever their direction.

## Amendment I (25 Nov 2026, before any result): class-wise score normalization of the bank (fixed reference statistics)
Motivation: run 3m — the bank's full-image deficit vs chunks lies in classes the bank does not rank highly. Per-class AP
depends only on the order of a class's own detections, so a per-class monotone score map can change AP only through
the 300-detections-per-image cap; it can move that budget from over-scored to under-scored (common/rare) classes. The goal
lists class-wise normalization and fixed-reference calibration; this tests them for the bank.
Method (training-free, no labels, request-invariant because the constants are fixed and per class):
 1. Reference statistics, label-free: bank pass on 300 reference images (the first 300 of the 600 run-3h confirmation
    images in their SHA-256 order; disjoint from the development and test images); per class c, the logit of the
    image's highest score for c; μ_c, σ_c = mean and std over the 300 images; μ_g, σ_g = means of μ_c and σ_c over classes.
 2. Per test image: bank pass → per class top 50 ≥ 0.02 → class-wise Gaussian Soft-NMS on the raw scores (identical to the
    run-3l bank lists) → per-class monotone map of the kept scores → top 300 per image. Maps:
    none: s; shift: σ(logit s − μ_c + μ_g); z: σ((logit s − μ_c)·σ_g/σ_c + μ_g).
Development (124 images, as Amendment H): (a) diagnostic ceiling — official AP of the bank's Soft-NMS lists with no
per-image cap (max 10,000) vs the 300 cap; (b) choose among none/shift/z by official AP (300 cap); ties → earlier in
that list. Test: the chosen map on the 1,000 run-3l images against run-3l chunks + Soft-NMS (primary: AP and AP50
difference, 200 paired image resamples; "within 1 AP" = CI lower bound above −0.01); the other maps reported as
secondary. CPU inference. All results reported whatever their direction. If the ceiling (a) shows that removing the cap
recovers < 0.5 AP on development, the test is still run but the expected outcome is recorded as negative beforehand.
