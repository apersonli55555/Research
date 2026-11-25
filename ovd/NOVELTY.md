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

## Pre-registered success criteria (PLAN.md), evaluated once on held-out data (23 Nov 2026)
| criterion | Grounding DINO | YOLO-World | OWLv2 |
|---|---|---|---|
| goal 1: interference ≥ 3 AP or > 10% flips | **met** (COCO 9.8 AP k1 vs ours; 27.8% instances flip) | **met** (LVIS 18.1% flip; 11.5 AP k1 vs full) | **met** on LVIS (20.7 AP, 19.0% flips); COCO 2.3 AP / 3.1% (below) |
| (i) interference reduced ≥ 80% | **met** (100%, by construction) | **met** (100%) | **met** (100%) |
| (ii) mAP within 1 point of the better baseline | **met** (+1.3 COCO, +5.0 LVIS) | **met** (+2.0, +7.8) | **met** (= k1 on both) |
| (iii) cost < 2× original | **met** on COCO (1.01–1.14×) and the full LVIS list (1.00×); fixed chunks **not met** for 5–200-name LVIS vocabularies (4.3–10.6×); **fixed class bank (expert, run 3g, independently verified): 1.80× standard five-class inference, met** — but 5.2 AP below standard five-class on that sample | **met** (0.94–1.14×) | **met** (1.0×) |
| goal 5: replication | 3 architectures × 2 held-out datasets (COCO 1,000 images, LVIS 600) | | |
Independent verification: `independent_check.py` 6/6 GPU headline mAPs + 90/90 table values (all groups) on the CPU-run files, max diff 2.2e-16; `check_numbers.py`
214/214 manuscript numbers traced; pilot reproduced GPU vs. CPU to ≤ 0.001.

## Submission readiness (23 Nov 2026): COMPLETE AS A RESEARCH STUDY; NOT YET SUBMISSION-READY
All planned experiments, corrections, ablations, statistics, figures and a 6-page draft with no open TODOs are done and
traceable. Before an ICCV submission:
1. ~~Resolve the LVIS plan difference~~ — resolved: images, targets and all text-ranked vocabularies identical; only the
   CLIP-image-based vis/prune lists differ (disclosed in the paper).
2. A human (teacher / mentor) read of the paper; check the official ICCV 2027 template, page limit and policies.
3. Recommended for reviewers: one more recent detector (e.g. a newer Grounding DINO variant) and the full
   captioner-based VocAda baseline; a cheaper invariant method for small vocabularies in early-fusion models.
Main reviewer risks remain "known/obvious" (mitigated by the controlled protocol and mechanism) and "the fix is simple"
(mitigated by it matching or beating every baseline at ~1× cost, except the stated LVIS small-vocabulary case).

### Update 24 Nov 2026 — fixed class bank (owner's expert, run 3g)
A training-free fixed bank of 1,203 class embeddings gives list-invariant Grounding DINO detections in one pass:
0.661 vs 0.644 mAP50 for fixed chunks on 100 fresh LVIS images (CI −1.65 to +3.97; non-inferiority not established),
0.641 s vs 1.399 s (1.80× standard five-class inference, 0.356 s / 0.713 mAP50). Imported and independently recomputed (exact match; see RESULTS.md).

### Update 24 Nov 2026 — 600-image confirmation (run 3h, independently verified)
Fixed class bank + class-wise Soft-NMS on 600 fresh LVIS images: 0.650 vs 0.631 mAP50 for unmodified standard five-class
Grounding DINO (+1.89, CI −0.07 to +2.66 → non-inferior at −1), +3.26 over fixed chunks + Soft-NMS, 1.81× standard latency,
list-invariant. Pre-registered criteria for Grounding DINO on LVIS with small vocabularies: (i) interference 0 — met;
(ii) within 1 AP of the unmodified standard prompt — met (non-inferiority); vs. standard + Soft-NMS not established;
(iii) cost < 2× — met (1.81×). Remaining before submission: Soft-NMS provenance + producing notebook from the expert;
human read; ICCV 2027 template/rules.

### Update 24 Nov 2026 — expert improvements
Done from saved outputs: #1 official LVIS evaluator (same AP50 to within 0.0005; all CIs and conclusions unchanged);
#2 duplicate boxes explain the Soft-NMS gain; #8 gain not driven by single-image classes (+2.78 without them), largest
on small objects; #10 one-command Table 1 reproduction. Shown by construction and checked: class vectors do not depend on
the other names (identical to 3e-6 for other orders / names alone). Running on the owner's GPU (notebook
`bank/followup_bank.ipynb`): #3 image-independent requests, #4 half-dictionary banks, #5 cost 1–1,203 names, #6 pooling and
mask ablations, #7 grounding-dino-tiny, #9 qualitative figure.
Run 3i (24 Nov 2026, Amendment F): on the 600 confirmation images the OWLv2 and YOLO-World corrections replicate on a
third held-out sample (OWLv2 +1.5 [+0.6, +2.0] over single-label five-name output; YOLO-World fixed guide 0.490 vs
0.424 request, +6.6 [+4.5, +7.9], and 0.379 single-class); official LVIS evaluator agrees within 0.001.

### Update 24 Nov 2026 — run 3j (expert follow-ups on the owner's T4; cross-checked on CPU)
- #3: with requests drawn without the image, the standard prompt drops to 0.614 and the bank + Soft-NMS leads by 3.6
  [1.6, 4.3] (by 1.6 [−0.07, 2.5] when both use Soft-NMS); two random requests flip 9.0% of the standard prompt's
  detections on fresh images. Strengthens the main claim.
- #4/#6: class vectors are context-free; dictionary composition shifts mAP by about ±1 (n.s.); the expert's mean-pool +
  identity-mask design is the best tested.
- #5: bank cost is constant (1.16 s on T4, +1.2 GB); break-even with the standard prompt at ≈100 names, 6.3× faster at
  1,203.
- #7: **negative** — on grounding-dino-tiny the bank + Soft-NMS is 5.75 points below the standard prompt and 5.1 below
  fixed chunks + Soft-NMS, at 2.00× cost. The paper now says the bank is checkpoint-specific; fixed chunks remain the
  invariant method that holds on both checkpoints.
- #9: qualitative figure with a pre-fixed selection rule and the totals in the caption.
**Reviewer risk added:** "the bank only works for one checkpoint" (high; stated in abstract and limitations).
**Readiness:** all ten expert items addressed with evidence. Remaining before submission: a human read, ICCV 2027
template and rules, and the anonymous supplement rebuilt with the new scripts/results.
- Criterion (ii) on grounding-dino-tiny: no invariant method tested is within 1 AP50 of the standard prompt (fixed chunks
  −2.52, bank + Soft-NMS −5.75); stated in the paper's results and limitations.
- ICCV 2027 rules (checked 24 Nov 2026): no official call for papers or author kit is published yet; third-party trackers
  only estimate an early-to-mid March 2027 deadline. The draft follows the previous ICCV review format (8 pages
  excluding references; this draft's body ends on page 8). It must be re-checked against the official kit at iccv.thecvf.com when published.

### Update 24 Nov 2026 — run 3l (Amendment G): full-image LVIS protocol — NEGATIVE for the bank
Bank + Soft-NMS vs fixed chunks + Soft-NMS on 1,000 fresh images, official evaluator: AP −2.0 [−2.56, −0.53], AP50 −1.26
[−1.63, +0.52] (non-inferiority not established), 6.67× faster. Loss on common/rare classes; frequent classes equal.
**Revised claim:** the paper's contribution is the controlled measurement + mechanism + per-object evidence that
averages hide interference (9.0% decisions flip at near-equal counts; 17.4% of found objects depend on the request),
with invariant inference for each family: YOLO-World's fixed guide is the clear accuracy win; OWLv2's independent
read-out fixes deployment-style outputs; for Grounding DINO exact chunking is the accurate invariant option and the bank
a 6.7× faster approximation that costs 2 AP. **Reviewer risk raised:** "no accurate *and* fast invariant method for
early-fusion detectors" (high; stated plainly). A better Grounding DINO method is open work (prompt prepared for the
owner's expert).

### Update 25 Nov 2026 — run 3m (Amendment H): bank-proposed context — NEGATIVE
Pre-registered training-free attempt to close the gap (bank proposes its top-10 classes, one standard pass re-scores
them; 2 passes): −2.24 AP [−2.71, −0.74] vs chunks on the same 1,000 images, at the bank's level. Exploratory
mechanism (confirmed on the test images): the bank's whole full-image deficit is in classes it does not propose
(per-class AP 0.219 vs chunks 0.269; on proposed classes 0.528 vs 0.520), so any bank-proposed shortlist is capped;
short focused prompts additionally inflate scores of plausible-but-absent classes. Novelty assessment unchanged: no
accurate *and* fast invariant method for Grounding DINO is shown; the negative result and its mechanism are reported.
Paper now 8 pages including references (body ends on page 7).

### Update 25 Nov 2026 — run 3n (Amendment I): class-wise score shift — PRIMARY CRITERION MET (with a pending fairness check)
Pre-registered (map chosen on 124 development images, frozen before the test): a fixed per-class logit shift of the bank's
Soft-NMS scores, estimated without labels on 300 disjoint reference images. On the same 1,000 full-image test images:
0.4052 AP vs 0.3982 for standard 14-pass inference, **+0.70 AP [−0.23, +2.05]** (non-inferior at 1 AP),
**+2.18 AP50 [+1.09, +3.88]** (superior); one pass, 6.67× faster on the L4, list-invariant by construction; the map adds
8.6 ms/image. Reproduced at four decimals by an independent LVIS evaluator (no lvis-api). This meets the goal's Grounding
DINO criterion (invariant, within 1 AP of the strongest baseline, < 2× cost) under the standard LVIS protocol.
**Novelty, honestly scoped.** The shift itself is not new as an idea: Dave et al. (arXiv:2102.01066, 2021) showed that the
per-image detection cap makes LVIS AP sensitive to cross-class calibration and that calibration raises pooled AP. What is
new here is (a) the diagnosis that the fixed bank's full-image deficit is a cross-class *score-scale* problem, not missing
boxes or false-alarm crowding (runs 3l/3m), and (b) that a label-free, request-independent per-class constant restores
parity with exact chunking in one pass while keeping list invariance.
**Reviewer risks (new).** (1) "The gain is a protocol artifact of the 300 cap": partly true — under a per-class budget
(AP-fixed) the map changes nothing; the paper says so. The bank's uncapped AP on development (0.5397) shows the cap binds.
(2) "Apply the same shift to chunks": pre-registered as Amendment I2; needs the owner's GPU (`bank/calib_chunks.ipynb`,
≈ 1.5 h on an L4) or a ≈ 40 h CPU fallback run (started 25 Nov 09:30). If chunks + shift gains as much, the paper must
present the shift as a general post-hoc fix and the bank's claim reverts to "6.7× faster at equal calibration".
(3) Single checkpoint (base); the bank still fails on tiny.
Paper: 8 pages; the body now ends on page 8 (references start on page 8); 312 of 312 numbers traced.
