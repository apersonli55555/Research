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
- **Published baseline only approximated (medium).** VocAda needs LLaVA-Next-7B + Llama3-8B; we evaluate a CLIP-only
  VocAda-style pruning baseline (top-K classes per image, K fixed before results). It is far below every other variant
  (e.g. COCO OWLv2 0.637 vs. 0.744; LVIS YOLO-World 0.172 vs. 0.442) because it often prunes the target. Reviewers may
  object that the full captioner-based VocAda would prune better; running it is future work and is stated as a limitation.
- **ICCV 2027 dates/policies** must be checked on the official site before submission; this file states none.

## Pre-registered success criteria (PLAN.md), evaluated once on held-out data (26 Nov 2026)
| criterion | Grounding DINO | YOLO-World | OWLv2 |
|---|---|---|---|
| goal 1: interference ≥ 3 AP or > 10% flips | **met** (COCO 9.8 AP k1 vs ours; 27.8% instances flip) | **met** (LVIS 18.1% flip; 11.5 AP k1 vs full) | **met** on LVIS (20.7 AP, 19.0% flips); COCO 2.3 AP / 3.1% (below) |
| (i) interference reduced ≥ 80% | **met** (100%, by construction) | **met** (100%) | **met** (100%) |
| (ii) mAP within 1 point of the better baseline | **met** (+1.3 COCO, +5.0 LVIS) | **met** (+2.0, +7.8) | **met** (= k1 on both) |
| (iii) cost < 2× original | **met** on COCO (1.01–1.14×) and the full LVIS list (1.00×); **not met** for 5–200-name LVIS vocabularies (4.4–10.4×) | **met** (0.94–1.14×) | **met** (1.0×) |
| goal 5: replication | 3 architectures × 2 held-out datasets (COCO 1,000 images, LVIS 600) | | |
Independent verification: `independent_check.py` 21/21 headline mAPs (6 GPU + 15 CPU), max diff 1e-16; `check_numbers.py`
214/214 manuscript numbers traced; pilot reproduced GPU vs. CPU to ≤ 0.001.

## Submission readiness (26 Nov 2026): COMPLETE AS A RESEARCH STUDY; NOT YET SUBMISSION-READY
All planned experiments, corrections, ablations, statistics, figures and a 6-page draft with no open TODOs are done and
traceable. Before an ICCV submission:
1. Resolve the LVIS plan difference between machines (run `ovd/plan_fingerprint.py` on the Colab plan; if images/pairs
   differ, re-run OWLv2/YOLO-World LVIS on the Colab plan — about 1 GPU hour).
2. A human (teacher / mentor) read of the paper; check the official ICCV 2027 template, page limit and policies.
3. Recommended for reviewers: one more recent detector (e.g. a newer Grounding DINO variant) and the full
   captioner-based VocAda baseline; a cheaper invariant method for small vocabularies in early-fusion models.
Main reviewer risks remain "known/obvious" (mitigated by the controlled protocol and mechanism) and "the fix is simple"
(mitigated by it matching or beating every baseline at ~1× cost, except the stated LVIS small-vocabulary case).
