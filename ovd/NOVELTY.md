# Novelty and reviewer-risk assessment (written 22 Nov 2026, before any GPU result)

Sources were read on 22 Nov 2026 (abstracts and the available HTML; some HTML was truncated). Anything not read in
full is marked [VERIFY]. This file will be updated after the pilot and the full runs; claims depend on those results.

## Closest prior work (2024–2026)
| Work | What it shows | What it does not do |
|---|---|---|
| Tsourveloudis, *Do Open-Vocabulary Detectors Transfer to Aerial Imagery?* arXiv 2601.22164 (Jan 2026) | On LAE-80C, Grounding DINO F1 0.5% with all 80 names vs 7.4% with only the classes present in the image (avg. 3.2) and 5.9% batched; attributes it to "attention competition"; recommends vocabulary filtering | No fixed-target protocol (the oracle also leaks which classes are present); no mechanism measurement; no fix tested; aerial domain only |
| Liu et al., *VocAda: Test-time Vocabulary Adaptation*, ICIP 2025, arXiv 2506.00333 | Pruning the vocabulary per image (captioner + selector) improves AP50 (Detic OVE-COCO all 45.0 → 47.1; oracle 50.9) | Uses Detic / VLDet / CoDet, not Grounding DINO / YOLO-World / OWLv2; no fixed-target measurement; no mechanism; needs a 7B captioner (cost) |
| Soon & Hsieh, *Confidence Scores in OVD Are a Biased Mixture of Scale and Semantics*, arXiv 2607.10993 (ICPR-W 2026) | Scores depend on object scale and on the wording of the target's own query; temperature fix | Never varies the *other* classes |
| DETR-ViP, ICLR 2026, arXiv 2604.14684 | States Grounding DINO is sensitive to the number of prompts and needs chunking [VERIFY: the exact claim and any numbers] | Visual prompts; no systematic study of text-vocabulary interference |
| SL-Ground, arXiv 2610.24814 (2026) | Grounding DINO on miniLVIS needs 31 chunks; cross-chunk outputs overlap with conflicting labels | Proposes a new single-pass model (training), not a measurement of interference |
| MM-Grounding-DINO docs (mmdetection) | Different `--chunked-size` values give different predictions | Documentation, not a study |
| ICLR 2026 "Fantastic tractor-dogs…" [VERIFY authors/title] | Confident false positives on background-only images with the 1,203 LVIS classes | Not a counterfactual on vocabulary composition |
| Grounding DINO (Liu et al., ECCV 2024) | Sub-sentence prompts with per-phrase attention masks, introduced to avoid "cross-talk" between class names | Does not measure the remaining cross-talk through fusion / query selection |

## What would be new (if the experiments support it)
1. **A counterfactual, fixed-target protocol.** Same image, same target string, same model and preprocessing; only
   the other names change (size 1–1,203; random, semantically similar, visually confusable, unrelated; 5 orders). This
   isolates interference from the confounds in "all classes vs. present classes" comparisons.
2. **Mechanism decomposition per architecture, partly exact:** OWLv2 raw scores are invariant by construction
   (interference only via top-1 label competition); YOLO-World's text guidance is `sigmoid(max over classes)`, so it is
   exactly order-invariant and only set-dependent through the guide (verified: fixing the guide makes scores invariant to
   6e-7); Grounding DINO's text encoder is block-masked per phrase, so the remaining dependence comes from fusion,
   query selection (probed by a hook), label competition and chunking (where *order* changes chunk membership).
3. **Invariant-by-construction, training-free corrections** (fixed context / fixed guide / independent read-out),
   evaluated for accuracy and latency against original, single-class and (where feasible) vocabulary-pruning baselines.

## Evidence so far (23 Nov 2026; numbers from ovd/RESULTS.md)
- Pre-registered pilot: GO (Grounding DINO mAP50 k=1 0.693 vs. 80 names 0.776, CI of difference [6.1, 9.5] points;
  16.7% flips). Reproduced across machines (GPU vs. CPU) for OWLv2 and YOLO-World.
- Held-out COCO: OWLv2 interference is purely label competition (−2.3 AP50 at 80 names, 3× larger for similar/visual
  distractors than random); YOLO-World guide helps / competition hurts; fixed guide + independent read-out is best
  (+2.0 over single-class, +3.7 over the full list).
- Held-out LVIS: OWLv2 single-label output loses 20.7 AP50 at 1,203 classes; independent read-out recovers it.
- Exact order invariance for all three models at ≤ 256 tokens (predicted from the architectures).

## Reviewer risks (honest)
- **"Obvious / known" (high).** Practitioners know vocabulary choice matters (aerial paper, VocAda, chunk-size docs).
  Mitigation: the paper's value must be the controlled measurement + mechanism + a fix with a clear accuracy/cost
  trade-off, not the existence of the effect alone.
- **"The fix is trivial" (high).** Fixing the context or guide is simple. So far it matches or beats the best baseline
  (YOLO-World +2.0 over single-class; OWLv2 equal to single-class at 1/K the passes); Grounding DINO and latency pending.
- **"Independent read-out is what benchmarks already do" (high, OWLv2).** True: LVIS/COCO evaluation scores
  (box, class) pairs. The OWLv2 loss concerns deployment-style single-label outputs (the default HF post-processing);
  the paper states this explicitly. The YOLO-World guide effect and the Grounding DINO fusion effect are not of this kind.
- **Absent-class false alarms (medium).** Independent read-out raises false alarms on verified-absent LVIS classes
  (6.3% vs. 1.4%); reported as a trade-off.
- **Effect size (unknown until the pilot).** If the pilot shows < 3 AP points and < 10% flips, the GO rule says STOP.
- **Scope.** Base / small checkpoints only (one L4). Reviewers may ask for Grounding DINO 1.5 / LLMDet / YOLOE /
  Florence-2; at least one extra recent model would strengthen replication [TODO after Stage 2].
- **Missing published baseline.** VocAda needs LLaVA-Next-7B + Llama3-8B; a CLIP-only VocAda variant fits on the L4 and
  should be added as the published baseline [TODO].
- **ICCV 2027 dates/policies** must be checked on the official site before submission; this file states none.

## Submission readiness: NOT READY
Missing: Grounding DINO held-out runs + query-selection probe + GPU latency (running on the owner's L4), YOLO-World LVIS
(running on CPU), final figures, full manuscript pass, independent re-check of every number in the paper. Every claim above in "What would
be new" items 2–3 beyond the CPU smoke checks is still a hypothesis.
