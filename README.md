# How Much Is a Real Fall Worth?

Research code and paper on video fall detection: which training data (staged lab falls, synthetic videos, hand-labelled real accident videos, or unlabelled real accident videos) prepares a detector for real falls, tested on three test sets from two real-world sources.

**Paper:** [`vision/paper/paper.pdf`](vision/paper/paper.pdf) (LaTeX source in `vision/paper/`)

## Main findings

- **Real accident video is useful without labels.** A probe trained on 7,063 unlabelled accident clips, using pseudo-labels plus the free "before the accident" clips as non-falls, reaches 0.894 / 0.937 / 0.953 AUROC on FallOrNot / OmniFall OF-ItW / HMDB51 (synthetic-only training: 0.814 / 0.872 / 0.917), under a success rule fixed before the run. It reads no human label.
- **Controls show what that gain is.** Most of the in-domain gain is "accident versus ordinary activity"; pseudo-labelling adds fall-specific information beyond it. Telling falls from *other accidents* without labels is significant out of domain (p = 0.005) but not in domain (p = 0.065).
- Detectors trained on staged lab falls score 1.00 AUROC in the lab but about 0.76 on FallOrNot, our test set of real accident videos drawn from OOPS!, with non-fall accidents as false-alarm negatives.
- Among labelled sources the ranking flips between test sets: hand-labelled real falls win on FallOrNot, synthetic falls win on HMDB51.
- With only a handful of labelled real falls, a probe anchored to the synthetic solution is never meaningfully worse than the better single source (rule fixed in advance, met).
- Tests fixed in advance that were not met, and a second annotator who agreed only moderately (Cohen's kappa 0.50), are reported in the paper.

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
