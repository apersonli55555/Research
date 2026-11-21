# How Much Is a Real Fall Worth?

Research code and paper on video fall detection: which training data (staged lab falls, synthetic videos, hand-labelled real accident videos, or real accident videos without fall annotations) prepares a detector for real falls, tested on three test sets from two real-world sources.

**Paper:** [`vision/paper/paper.pdf`](vision/paper/paper.pdf) (LaTeX source in `vision/paper/`; supplement in `vision/paper/supp.pdf`)

## Main findings

- **Much of the benefit of real accident video does not need fall annotations.** Combining a synthetic-trained teacher, pseudo-labels and the "before the accident" clip that OOPS!'s human-annotated accident times provide, a probe trained without additional real-video fall annotations reaches 0.894 / 0.937 / 0.953 AUROC on FallOrNot / OmniFall OF-ItW / HMDB51 (synthetic-only training: 0.814 / 0.872 / 0.917), under a success rule fixed before the run. None of the three ingredients is new; the contribution is combining them for fall detection and measuring what each part contributes.
- **Controls show what that gain is.** Most of the in-domain gain is "accident versus ordinary activity", which the accident time gives away; fall-specific information adds a smaller gain on falls versus other accidents (+0.058 AUROC on 345 hand-labelled accident clips, p = 0.013, rule fixed in advance).
- **Fine-tuned video models.** VideoMAE fine-tuned the same way has the highest observed mean AUROC on FallOrNot (0.910) and OF-ItW (0.967), but trails synthetic training on HMDB51's hard subset (0.893 vs. 0.922).
- **Caveats:** FallOrNot and OF-ItW both come from OOPS!; the largest falls-vs-accidents test mixes a random and a model-selected sample; "not significantly different" from hand labels does not mean equivalent.
- Staged-trained detectors score 1.00 AUROC in the lab but about 0.76 on FallOrNot; among labelled sources, the ranking flips between test sets.
- Every test fixed in advance is reported, including the three that were not met (among them a consequence-based labelling idea).

## Repository layout

| Path | Contents |
|---|---|
| [`vision/`](vision/) | Pipeline, experiments, results ([`RESULTS.md`](vision/RESULTS.md)) and the paper |
| [`vision/fall_cvpr.ipynb`](vision/fall_cvpr.ipynb) | Google Colab notebook that runs the pipeline |
| [`research/`](research/) | Earlier, separate study: a phone motion-sensor fall detector benchmarked on SisFall |
| [`index.html`](index.html) | The phone fall-detector web app from that earlier study |
| [`AGENTS.md`](AGENTS.md) | Notes and rules for AI assistants working on this repository |

## Data

Videos are not included. The pipeline streams them from their public sources: OOPS! (Epstein, Chen & Vondrick, CVPR 2020), UR Fall (Kwolek & Kepski, 2014; CC BY-NC-SA 4.0), OmniFall OF-Syn (Schneider et al., 2025) and HMDB51 (Kuehne et al., ICCV 2011; CC BY 4.0).
