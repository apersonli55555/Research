# How Much Is a Real Fall Worth?

Research code and paper on video fall detection: which training data (staged lab falls, synthetic videos, or real accident videos) prepares a detector for real falls, tested on two real-world sources.

**Paper:** [`vision/paper/paper.pdf`](vision/paper/paper.pdf) (LaTeX source in `vision/paper/`)

## Main findings

- Detectors trained on staged lab falls score 1.00 AUROC in the lab but about 0.76 on FallOrNot, our test set of real accident videos (drawn from OOPS!, including non-fall accidents as false-alarm negatives).
- On FallOrNot, about 30 real training falls outperform 2,136 synthetic ones.
- On an independent source (HMDB51) the ranking flips: synthetic training is best.
- Training on synthetic and real data together never falls more than 0.03 AUROC below the best source on any test.
- A fine-tuning test specified in advance was not met, and a second annotator agreed only moderately (Cohen's kappa 0.50); both are reported in the paper.

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
