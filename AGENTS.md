# Instructions for AI assistants working on this repository

This file is for any AI assistant (ChatGPT, Codex, Claude or others) helping with this project. Read it fully before changing anything.

## The project in one paragraph

We are writing a computer-vision research paper, titled **"How Much Is a Real Fall Worth? Real, Synthetic and Staged Training Data for Video Fall Detection, and Adapting with a Handful of Real Clips"** (target: ICCV 2027 main conference, a long shot that needs the plan under Remaining work; fallback: a CVPR 2027 workshop). Fall detectors are usually trained and tested on falls staged by actors in a lab, where they score near 100%. We built a real-world test set (**FallOrNot**) from the OOPS! dataset of in-the-wild accident videos, labelled by a human, and showed that lab-trained detectors fail on real falls (AUROC about 0.76). Real training data fixes most of that on FallOrNot (about 0.90), but on a second real-world source (HMDB51) synthetic data is best; training on both is the safest recipe.

## Where things are

| Path | What it is |
|---|---|
| `vision/paper/paper.tex`, `paper.pdf` | **The paper.** Every number comes from `RESULTS.md`. Figures: `vision/paper/make_figures.py`. |
| `vision/PAPER_DRAFT.md` | Older outline, superseded by `vision/paper/`. Do not edit. |
| `vision/RESULTS.md` | All measured results, run by run. **The single source of truth for numbers.** |
| `vision/README.md` | Pipeline overview, data sources and licences. |
| `vision/fall_cvpr.ipynb` | Google Colab notebook that runs the whole pipeline (the owner runs it on an L4 GPU). |
| `vision/common.py` | Frame sampling (16 frames per 2 s clip), CLIP and VideoMAE encoders, zero-shot prompts. |
| `vision/oops_mine.py` | Streams the 48 GB OOPS! archive, embeds clips, selects the test sample and the training pool. |
| `vision/urfall_prep.py` | Prepares the staged UR Fall dataset. |
| `vision/labeler.py` | In-Colab labelling tool. |
| `vision/ofsyn_prep.py`, `vision/hmdb_prep.py`, `vision/ofitw_prep.py` | Prepare OmniFall synthetic clips, the external HMDB51 test set, and OmniFall's in-the-wild (OF-ItW) test set. |
| `vision/finetune.py` | Fine-tunes VideoMAE (fixed hyper-parameters, seeds) and scores both test sets. |
| `vision/agreement.py` | Inter-annotator agreement (Cohen's kappa). |
| `vision/experiment.py` | All evaluation: probes, bootstrap CIs, paired tests, learning curve, equal budget, external test, fine-tuned scores, `--test-labels`. |
| `vision/label_page/`, `vision/build_label_page.py`, `vision/import_labels.py` | Optional web labelling page (not used for the current results). |
| `index.html`, `research/` | An earlier, separate project: a phone motion-sensor fall detector and its SisFall benchmark. Not part of the CVPR paper; do not mix them up. |

Data (clips, embeddings, labels, result files) lives on the owner's Google Drive in `fall-cvpr/`, not in this repository.

## IMPORTANT: closest prior work (found November 2026)

**OmniFall** (Schneider et al., arXiv 2505.19889; v1 May 2025, v3 July 2026) already shows the staged-to-wild gap on fall-containing OOPS videos, builds an in-the-wild OOPS test set it calls "OOPS-Fall"/"OF-ItW", shows synthetic training beats staged training, and trains one upper-bound model with in-the-wild data. **Never claim we are the first to show the gap or to build a real-world fall test set from OOPS.** Our benchmark is renamed **FallOrNot**. Our distinct contributions are: non-fall accidents as false-alarm negatives (random, model-independent sample of all accidents); how much real data is needed (learning curve, real-only vs. staged-only); appearance vs. motion; and real vs. synthetic (OF-Syn) vs. staged at equal budget, tested on two real-world sources. **Run 4 (in-domain, FallOrNot): 30 real falls beat all 2,136 synthetic falls. Run 5 (external, HMDB51): the advantage disappears and synthetic is best (real vs. synthetic −0.029, p = 0.029). Never state "real beats synthetic" as a general claim;** the supported message is that rankings flip between real-world test sets and that synthetic + real has the smallest worst-case shortfall from the best source (at most 0.03 AUROC; see the derived table at the end of RESULTS.md). Do not say it has the best worst-case AUROC: real-only matches it on that measure. Say the fine-tuning test was "specified in advance" (committed to the repo before the run), not "pre-registered". Cite OmniFall prominently in Related Work.

## Current state (keep this section updated)

- **Done:** pipeline; 165 labelled test videos (255 clips, 102 falls); 268 labelled training clips (131 falls); results runs 1–12 (run 12 = anchored-probe method test, rule fixed in advance MET; run 11 = OmniFall OF-ItW third test set; run 10 = leakage check, compilation-level intervals, source-balanced method; run 4 = real vs. synthetic vs. staged; run 5 = external HMDB51 test; runs 6–7 = fine-tuning over 3 seeds, test specified in advance not met; run 8 = second annotator; run 9 = re-grading with their labels) plus a derived shortfall table; full paper; OOPS! data-use question resolved by the owner (do not mention it in the paper or elsewhere).
- **Citations checked** (by ChatGPT, applied by Claude): all fully specified references verified; all references in vision/paper/paper.tex now verified (MCFD tech report 1350, HMDB51 ICCV 2011 pp. 2556-2563, OpenCLIP Zenodo DOI, WHO fact sheet 21 Oct 2026). A broader literature review for related work is still welcome.
- **Paper:** full LaTeX draft in `vision/paper/paper.tex` (compiles with pdflatex; figures from `vision/paper/make_figures.py`, numbers from RESULTS.md); `paper.pdf` is the compiled copy. All citations verified (November 2026); no open [VERIFY]/[TODO] marks.
- **Run 11 is in `vision/paper/paper.tex`** (new subsection and table, 10 Nov 2026). `vision/paper/cvpr/main.tex` has NOT been re-synced (it lacks runs 11 and 12); rebuild it from `paper.tex` in the ICCV 2027 template once that template is published.
- **Run 12 (11 Nov 2026): anchored probe, rule fixed in advance MET on both test sets.** Mean over budgets: FallOrNot anchored 0.848 vs. real-only 0.804 vs. synthetic-only 0.814; OF-ItW 0.901 vs. 0.867 vs. 0.873. It is in `paper.tex` (section "anchored probe", table, `fig_adapt.pdf`). **Do not overclaim:** the gain is at small budgets only (k ≤ 30); with the full real pool it equals real-only on FallOrNot and is slightly worse on OF-ItW (−0.008, p = 0.002); on HMDB51 it sits between real-only and synthetic-only (n.s.); the technique is known biased regularisation (cited), not a new idea; plain pooling was not run at small budgets.
- **Key results:** see `vision/RESULTS.md`, runs 3–12 and the derived shortfall table.
- **Outreach (9 Nov 2026):** the owner emailed the paper link to David Schneider (OmniFall), Carl Vondrick (OOPS!) and Michal Kępski (UR Fall) asking for feedback. No replies yet as of 10 Nov. If feedback arrives, apply it to `vision/paper/paper.tex`. The emailed link is the raw `paper.pdf` on this branch, so keep that path and branch alive.
- **Annotators:** a second, independent annotator labelled 100 of the test videos (run 8): raw agreement 0.796, Cohen's κ = 0.498 [0.297, 0.678], moderate; disagreements mostly second annotator = fall, first = not fall. Report this honestly as a limitation. Run 9 re-grades the test set with the second annotator's labels: main findings hold except that the in-domain real-over-synthetic advantage is no longer significant.
- **Labelling is finished.** The owner will not label more clips. Do not plan work that needs more labels from them.

## Remaining work, in priority order

0. **Third test set: done (run 11).** Staged 0.815 < synthetic 0.872 < real 0.927-0.951 on 571 OF-ItW test videos, all p < 0.001; our labels agree with OmniFall's on 27 of 28 shared videos. Original task description: `vision/ofitw_prep.py` embeds OmniFall's OF-ItW clips (818 OOPS! videos labelled by OmniFall's annotation team; test split 572 videos, 1,671 clips, 636 falls). `experiment.py --ofitw` scores every probe on that test split (our real training clips that fall in it are removed), trains probes on OF-ItW's own train+val split, and compares our labels with OmniFall's on shared videos. Record the output as run 11, then add it to the paper.
1. **Second real-world test source: done (HMDB51, run 5).** A third source of real accident falls (closer to OOPS! than HMDB51's movie falls) would further help separate "real vs. synthetic" from "which benchmark".
2. **Second staged dataset.** UR Fall is saturated (AUROC 1.00 in-domain). The original Le2i page is offline; find a legitimate mirror or another staged dataset.
3. **Fine-tuning experiment: done (runs 6 and 7, 3 seeds). The rule specified in advance was NOT MET for any seed.**
4. **Source-balanced mixing: done (run 10). Rule fixed in advance NOT MET** (FallOrNot +0.016, p = 0.004; HMDB51 +0.002, n.s.). Same run: no clear evidence of compilation leakage (disjoint 0.866 vs. random same size 0.874 ± 0.014), and FallOrNot intervals in the paper are now resampled by compilation. With those intervals, synthetic vs. staged on FallOrNot is NOT significant (p = 0.053): do not write that we replicate OmniFall's synthetic-beats-staged result, only that the direction agrees.
5. **Label adjudication (optional):** `vision/LABELING_GUIDE.md` sharpens the fall definition. `Labeler(..., adjudicate=True)` lets the two annotators settle only their 20 disagreements into `labels_adjudicated.csv`; evaluate with `experiment.py --test-overrides labels_adjudicated.csv`.
6. **Further method work** aimed at the hardest sub-problem: telling falls apart from other accidents (`real_hard` results). Explore why VideoMAE alone gains little from real data while CLIP temporal features gain a lot.
7. **Target venue: ICCV 2027 main conference (owner's decision, 10 Nov 2026).** The CVPR 2027 attempt was dropped (the owner could not complete OpenReview registration before the Nov 10 enrollment deadline). ICCV 2027's call for papers is not out yet; trackers project a deadline around 5 March 2027 (unconfirmed; check iccv.thecvf.com). Fallback: a CVPR 2027 workshop. `vision/paper/cvpr/main.tex` is the CVPR-template version with an LLM Usage Statement; reuse it for the ICCV template. Plan, in order:
   a. Run 11: OmniFall OF-ItW as a third, larger test set (item 0).
   b. Reframe the paper around few-shot adaptation: how to adapt a fall detector to a new domain with a handful of real clips.
   c. **DONE, rule MET (run 12).** Method test: anchored probe (rule fixed 10 Nov 2026, before any result; code in `vision/adapt.py`).** A logistic probe on CLIP + VideoMAE features is trained on k real falls + k real non-falls, with its weights penalised for moving away from the synthetic-trained probe's weights (not from zero). The penalty strength C is chosen from {0.001, 0.01, 0.1, 1, 10} by video-grouped cross-validation on those training clips only (C = 0.01 when there are too few videos to cross-validate). Baselines: real-only probe on the same clips; synthetic-only probe. Budgets k = 4, 8, 15, 30, 60, 125, ten random draws each (seed 0). Test sets: FallOrNot (real clips from our training pool) and the OF-ItW test split (real clips from OF-ItW train + val). **Success = on both test sets, (i) at every budget the anchored probe's mean AUROC is at least max(real-only mean, synthetic-only) − 0.005, and (ii) averaged over the six budgets it is higher than both baselines.** Report the outcome whatever it is; do not change the rule, grid, budgets or features after seeing results. Secondary, not part of the rule: HMDB51 with the full real pool.
   c2. **Method test for fine-tuned models: anchored fine-tuning (rule fixed 11 Nov 2026, before any result; code in `vision/adapt_ft.py`; pending one Colab GPU run, to be recorded as run 13).** Start from the synthetic-fine-tuned VideoMAE (`model_synthetic`, seed 0, run 6) and fine-tune end to end on k real falls + k real non-falls with loss = cross-entropy + (0.01 / 2) x squared distance of all weights from the synthetic model (L2-SP), no other weight decay. Baselines: real-only fine-tuning from the Kinetics checkpoint on the same clips with the run-6 recipe; the synthetic model itself. Same learning rate (5e-5), batch (8) and schedule for both; steps = max(60, 5 x clips / 8). Budgets k = 8, 30, 125; three random draws each (seed 0). Test set: FallOrNot (the only in-domain test for which pixel clips are stored). **Success = (i) at every budget the anchored mean AUROC is at least max(real-only mean, synthetic-only) - 0.01, and (ii) averaged over the three budgets it is higher than both baselines.** The tolerance is 0.01 rather than 0.005 because fine-tuning varies more between runs (seed s.d. 0.012-0.017 in run 7) and only three draws are affordable. Report the outcome whatever it is; do not change the rule, the 0.01 strength, steps, budgets or draws after seeing results. Secondary, not part of the rule: HMDB51 for the first draw of each budget.
   d. The owner still needs an active OpenReview profile (institution domain roundrockisd.org; high-school student 2025-2029).
8. **Mentor feedback.** The owner is looking for a teacher or professor to read the paper.
9. Optional: the second annotator could label the remaining 65 test videos (only if they volunteer). Never ask the owner for more labelling.

## Rules (these protect the paper's credibility)

1. **Never invent, estimate or "fill in" numbers.** Every number in the paper must come from a real run recorded in `vision/RESULTS.md`. If a number is missing, write **[TODO]**.
2. **Never fabricate citations.** If you cannot verify a reference, mark it **[VERIFY]** and do not guess details.
3. **Do not change the test set.** It is the hash-based random sample in `oops_mine.py` (`in_random_sample`, `--random-n 600`). Changing its selection, the prompts in `common.py`, or the clip definition invalidates existing labels and results.
4. **Do not tune anything on the test set.** Choose settings on training data only (for example with cross-validation).
5. **No AI labels as ground truth.** Labels must come from humans. AI may suggest which clips to label, never the label itself.
6. **Release labels and timestamps, not videos.** Cite OOPS! (Epstein, Chen & Vondrick, CVPR 2020) and UR Fall (Kwolek & Kepski, 2014). UR Fall is CC BY-NC-SA 4.0, non-commercial.
7. **Report honestly.** Include confidence intervals, say when differences are not significant, and keep the Limitations section accurate.
8. **When results change, update `vision/RESULTS.md` first**, then the paper.

## How to run things

- **Full pipeline:** open `vision/fall_cvpr.ipynb` in Google Colab with a GPU runtime and run the cells in order. Outputs go to Google Drive `fall-cvpr/`.
- **Re-run only the analysis** (no GPU, no labelling): mount Drive, clone this branch, then run
  `python vision/experiment.py --staged <drive>/fall-cvpr/urfall.npz --oops <drive>/fall-cvpr/oops`.
- **Code changes:** keep them small, test them on a tiny run first (`oops_mine.py --limit 20 --device cpu --model ViT-B-32 --pretrained laion2b_s34b_b79k`), and commit with a clear message.

## Working with the owner

The owner is not a machine-learning specialist. Explain things in plain language, give exact copy-paste steps for Colab, keep requests short, and say clearly when something is uncertain or unproven.

## Coordinating between assistants

More than one AI assistant may work on this repository. Before starting, read the latest commits (`git log`) and `vision/RESULTS.md`. Work on one item from "Remaining work" at a time, commit when done, and update "Current state" above, so the next assistant knows where things stand.
