# Submission readiness, novelty and reviewer risks — consolidated (27 Nov 2026)

This file supersedes the dated status lines in `NOVELTY.md` (kept there as history). Every number is from `RESULTS.md`
and the files it names; `paper/check_numbers.py` traces the manuscript's numbers to those files.

## Verdict
**Complete as a reproducible research study. Not ready for a "strong accept" submission.** The measurement, mechanism
and per-architecture invariant corrections are done, checked and reported, including negatives. The headline target of
the goal — a training-free correction that is invariant, within 1 AP of the *strongest* baseline and < 2× its cost on
every architecture and dataset — is **not met for Grounding DINO** (the bank with the shift is 0.69 AP below equally
corrected chunks, CI −1.50 to +0.59) and does **not transfer** to grounding-dino-tiny. Realistic outcome as written:
borderline / weak accept at best at ICCV; a workshop or a findings-style venue is the safer target unless the open items
below are closed.

## Goal targets, per architecture (held-out data only)
| target | Grounding DINO (base) | YOLO-World v2-S | OWLv2 (base) |
|---|---|---|---|
| interference ≥ 3 AP or > 10% flips | met (COCO 9.8 AP; 27.8% flips) | met (LVIS 18.1% flips) | met on LVIS (20.7 AP); COCO below (2.3 AP) |
| ≥ 80% interference reduction | met: 100%, measured (run 3o: 0 of 171 lists change) | met: 100% by construction (fixed guide) | met: 100% by construction (independent read-out) |
| within 1 AP of the strongest baseline | **sampled pairs: met; full-image LVIS: not met** (bank+shift − chunks+shift −0.69 [−1.50, +0.59]) | met (fix is best on all three samples; +7.0 mAP50 over the request) | met (= single class) |
| cost < 2× | met (bank 1 pass, 6.67× faster than 14 chunks; 1.81× the five-class standard prompt) | met (0.94–1.14×) | met (1.0×) |
| replication | COCO + LVIS (sampled pairs); LVIS full-image; **tiny checkpoint: fails** (−5.75 points) | COCO + LVIS (sampled pairs, three samples) | COCO + LVIS (sampled pairs, three samples) |

Class-wise shift (Amendments I, I2, K), frozen map, label-free reference statistics, pre-registered comparisons:
| detector, dataset (full-image protocol) | none → shift (AP) | shift − none [95% CI] |
|---|---|---|
| Grounding DINO bank, LVIS (run 3n) | 0.3774 → 0.4052 | +2.8 (significant) |
| Grounding DINO chunks, LVIS (run 3p) | 0.3980 → 0.4121 | +1.41 [+0.97, +1.88] |
| YOLO-World v2-S, LVIS, Soft-NMS pipeline (Amendment K1, primary) | 0.2948 → 0.2967 | +0.20 [−0.22, +0.50] (null) |
| YOLO-World v2-S, LVIS, standard NMS pipeline | 0.2975 → 0.2950 | −0.25 [−0.73, +0.25] (null) |
| YOLO-World v2-S, COCO, Soft-NMS pipeline (Amendment K2, primary) | 0.3570 → 0.3567 | bootstrap running |
| YOLO-World v2-S, COCO, standard NMS pipeline | 0.3632 → 0.3615 | bootstrap running |

The shift does **not** generalize to YOLO-World: it is a Grounding DINO-specific correction. Exploratory mechanism
check: the shift changes AP only through the per-image cap, and with YOLO-World on COCO the cap binds on only 33 of 1,000
images (pipeline P; LVIS 641 of 1,000).

## Novelty (2024–2026 literature; sources and dates in `NOVELTY.md`)
What is new and supported:
1. A counterfactual, fixed-target protocol (same image, same target string, only the other names change) that isolates
   vocabulary interference from the "all classes vs present classes" confound of Tsourveloudis (2026) and VocAda (2025).
2. A per-architecture mechanism decomposition, partly exact: OWLv2 only through top-1 label competition; YOLO-World only
   through its guide (fixing the guide makes scores invariant to 6e-7); Grounding DINO through fusion, query selection,
   label competition and chunk membership.
3. Per-object evidence that averages hide interference (9.0% of decisions flip at near-equal counts; 17.4% of found
   objects depend on the request).
4. Invariant inference per family with a measured accuracy/cost trade-off, including documented negatives (runs 3d, 3e,
   3f, 3l, 3m, tiny).
What is **not** new: the class-wise logit shift. Dave et al. (arXiv:2102.01066, 2021) showed that the per-image cap makes
LVIS AP depend on cross-class calibration. Our contribution there is only the diagnosis that the bank's full-image deficit
is a score-scale problem (runs 3l/3m) and that a label-free per-class constant fixes most of it; run 3p shows the same fix
helps standard chunks too, so it is a general post-hoc correction, not a bank-specific one.

## Reviewer risks, ranked
1. **"The fix is known / the shift is prior art" (high).** True for the shift (Dave et al.). Mitigation: the paper credits
   it and claims only the diagnosis and the measurement.
2. **"No accurate and fast invariant method for early-fusion detectors" (high).** Against corrected chunks the bank is
   −0.69 AP (not shown within 1 AP). Stated in the abstract and Limitations.
3. **"Does not transfer across checkpoints" (high).** The bank fails on grounding-dino-tiny (−5.75; its text input also
   loses candidate boxes, −4.8 points recall). Only one base checkpoint per family.
4. **"Gain is a protocol artifact of the 300-detection cap" (medium).** Partly true: under a per-class budget the shift
   changes nothing. Stated.
5. **"Obvious that vocabulary matters" (medium).** Mitigated by the controlled protocol and mechanism, not by the effect.
6. **Approximate VocAda baseline (medium).** CLIP-only pruning instead of the 7B captioner pipeline.
7. **Scope (medium).** No newer detectors (Grounding DINO 1.5, LLMDet, YOLOE, Florence-2).
8. **Effect size on COCO for OWLv2 (low).** 2.3 AP / 3.1% flips, below the interference threshold.

## Readiness checklist
- [x] Pre-registered plan with dated amendments, each committed before its result (`PLAN.md`, git history).
- [x] Every manuscript number traced (`paper/check_numbers.py`).
- [x] Independent re-evaluation (no lvis-api) of the full-image LVIS results at four decimals.
- [x] Alternative explanations tested: duplicate boxes (Soft-NMS), single-image classes, budget crowding (rejected),
      missing boxes vs score scale (run 3m), reference-image choice (ref sensitivity), the cap (stated), fairness of the
      shift (run 3p), generality of the shift (Amendment K).
- [x] Format: CVPR 2027 author kit (same `cvpr.sty` family as ICCV); 8 pages including references, letter size, fonts
      embedded, anonymous. ICCV 2027 checked 27 Nov 2026: the official
      site (iccv.thecvf.com) lists only the venue (Hong Kong) and dates (3–7 Oct 2027); no call for papers, deadline,
      page limit or author kit yet.
- [x] Supplement: anonymized code, results and raw outputs, each zip under 30 MB.
- [ ] **Human read-through** by the owner / a mentor (cannot be done by the assistant).
- [ ] **Re-check format and policies** against the official ICCV 2027 kit once released.
- [ ] Optional, to raise the score: a second Grounding DINO checkpoint where the bank works, or a cheaper invariant method
      that matches corrected chunks; the full captioner-based VocAda baseline; one newer detector.
