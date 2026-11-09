# Instructions for AI assistants working on this repository

This file is for any AI assistant (ChatGPT, Codex, Claude or others) helping with this project. Read it fully before changing anything.

## The project in one paragraph

We are writing a computer-vision research paper, working title **"How Much Is a Real Fall Worth? Data Efficiency and False Alarms in Real-World Video Fall Detection"** (target venue under review: CVPR is now a long shot, see below). Fall detectors are usually trained and tested on falls staged by actors in a lab, where they score near 100%. We built a real-world test set (**FallOrNot**) from the OOPS! dataset of in-the-wild accident videos, labelled by a human, and showed that lab-trained detectors fail on real falls (AUROC about 0.76), while a few real training falls fix most of it (about 0.90, p < 0.001).

## Where things are

| Path | What it is |
|---|---|
| `vision/PAPER_DRAFT.md` | The paper draft. Every number comes from `RESULTS.md`. |
| `vision/RESULTS.md` | All measured results, run by run. **The single source of truth for numbers.** |
| `vision/README.md` | Pipeline overview, data sources and licences. |
| `vision/fall_cvpr.ipynb` | Google Colab notebook that runs the whole pipeline (the owner runs it on an L4 GPU). |
| `vision/common.py` | Frame sampling (16 frames per 2 s clip), CLIP and VideoMAE encoders, zero-shot prompts. |
| `vision/oops_mine.py` | Streams the 48 GB OOPS! archive, embeds clips, selects the test sample and the training pool. |
| `vision/urfall_prep.py` | Prepares the staged UR Fall dataset. |
| `vision/labeler.py` | In-Colab labelling tool. |
| `vision/experiment.py` | All evaluation: probes, bootstrap CIs, paired tests, learning curve, figures. |
| `vision/label_page/`, `vision/build_label_page.py`, `vision/import_labels.py` | Optional web labelling page (not used for the current results). |
| `index.html`, `research/` | An earlier, separate project: a phone motion-sensor fall detector and its SisFall benchmark. Not part of the CVPR paper; do not mix them up. |

Data (clips, embeddings, labels, result files) lives on the owner's Google Drive in `fall-cvpr/`, not in this repository.

## IMPORTANT: closest prior work (found November 2026)

**OmniFall** (Schneider et al., arXiv 2505.19889; v1 May 2025, v3 July 2026) already shows the staged-to-wild gap on fall-containing OOPS videos, builds an in-the-wild OOPS test set it calls "OOPS-Fall"/"OF-ItW", shows synthetic training beats staged training, and trains one upper-bound model with in-the-wild data. **Never claim we are the first to show the gap or to build a real-world fall test set from OOPS.** Our benchmark is renamed **FallOrNot**. Our distinct contributions are: non-fall accidents as false-alarm negatives (random, model-independent sample of all accidents); how much real data is needed (learning curve, real-only vs. staged-only); appearance vs. motion; and real vs. synthetic (OF-Syn) vs. staged at equal budget, tested on two real-world sources. **Run 4 (in-domain, FallOrNot): 30 real falls beat all 2,136 synthetic falls. Run 5 (external, HMDB51): the advantage disappears and synthetic is best (real vs. synthetic −0.029, p = 0.029). Never state "real beats synthetic" as a general claim;** the supported message is that rankings flip between real-world test sets and synthetic + real is the most robust choice. Cite OmniFall prominently in Related Work. See the positioning note at the top of `vision/PAPER_DRAFT.md`.

## Current state (keep this section updated)

- **Done:** pipeline; 165 labelled test videos (255 clips, 102 falls); 268 labelled training clips (131 falls); results runs 1–7 (run 4 = real vs. synthetic vs. staged; run 5 = external HMDB51 test; runs 6–7 = fine-tuning over 3 seeds, pre-registered test not met); paper draft; OOPS! data-use question resolved by the owner (do not mention it in the paper or elsewhere).
- **Citations checked** (by ChatGPT, applied by Claude): all fully specified references verified; still open: the Multiple Cameras Fall Dataset report [VERIFY], the OpenCLIP software citation, and a literature review for the domain-shift paragraph and for claims about how rarely real falls are evaluated.
- **Key results:** see `vision/RESULTS.md`, runs 3, 3b, 4, 5, 6 and 7.
- **Annotators:** the test set was labelled by a single annotator. No second-annotator labels have been provided; do not report inter-annotator agreement unless a second label file is analysed with `vision/agreement.py`.
- **Labelling is finished.** The owner will not label more clips. Do not plan work that needs more labels from them.

## Remaining work, in priority order

1. **Second real-world test source: done (HMDB51, run 5).** A third source of real accident falls (closer to OOPS! than HMDB51's movie falls) would further help separate "real vs. synthetic" from "which benchmark".
2. **Second staged dataset.** UR Fall is saturated (AUROC 1.00 in-domain). The original Le2i page is offline; find a legitimate mirror or another staged dataset.
3. **Fine-tuning experiment: done (runs 6 and 7, 3 seeds). Pre-registered rule NOT MET for any seed.** Exploratory: the ranking flip holds on average across seeds; synthetic + real is near the top on both test sets with the smallest seed spread.
4. **Proposed method** aimed at the hardest sub-problem: telling falls apart from other accidents (`real_hard` results). Explore why VideoMAE alone gains little from real data while CLIP temporal features gain a lot.
5. **Related work and citations.** Every citation marked [VERIFY] must be checked against the real paper (authors, venue, year).
6. **CVPR LaTeX version** of the paper with figures (`roc_real.png`, `learning_curve.png` on the owner's Drive).
7. Optional, only if a second person volunteers: they label the existing test clips so we can report agreement (Cohen's κ). Never ask the owner for more labelling.

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
