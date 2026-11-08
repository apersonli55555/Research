# Instructions for AI assistants working on this repository

This file is for any AI assistant (ChatGPT, Codex, Claude or others) helping with this project. Read it fully before changing anything.

## The project in one paragraph

We are writing a computer-vision research paper, targeting CVPR, titled **"Staged Falls Don't Transfer: Measuring the Lab-to-Real Gap in Video Fall Detection"**. Fall detectors are usually trained and tested on falls staged by actors in a lab, where they score near 100%. We built a real-world test set (**OOPS-Fall**) from the OOPS! dataset of in-the-wild accident videos, labelled by a human, and showed that lab-trained detectors fail on real falls (AUROC about 0.76), while a few real training falls fix most of it (about 0.90, p < 0.001).

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

## Current state (keep this section updated)

- **Done:** pipeline; 165 labelled test videos (255 clips, 102 falls); 268 labelled training clips (131 falls); results runs 1–3b; paper draft; permission from the OOPS! authors (email received November 2026, kept by the owner).
- **Key results:** see `vision/RESULTS.md`, run 3 and run 3b.
- **Labelling is finished.** The owner will not label more clips. Do not plan work that needs more labels from them.

## Remaining work, in priority order

1. **Second real-world test source.** The real training and test clips both come from OOPS!, so part of the gain may be in-domain advantage. Find another public source of real (not staged) fall videos with a licence that allows research use, and evaluate the existing models on it.
2. **Second staged dataset.** UR Fall is saturated (AUROC 1.00 in-domain). The original Le2i page is offline; find a legitimate mirror or another staged dataset.
3. **Proposed method** aimed at the hardest sub-problem: telling falls apart from other accidents (`real_hard` results). Explore why VideoMAE alone gains little from real data while CLIP temporal features gain a lot.
4. **Related work and citations.** Every citation marked [VERIFY] must be checked against the real paper (authors, venue, year).
5. **CVPR LaTeX version** of the paper with figures (`roc_real.png`, `learning_curve.png` on the owner's Drive).
6. Optional: a second person labels the test set, so we can report agreement (Cohen's κ).

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
