# Research brief for an outside expert (deep research)

Paper (PDF): https://github.com/apersonli55555/Research/blob/claude/beautiful-brown-gdjzcq/vision/paper/paper.pdf
All results, run by run: https://github.com/apersonli55555/Research/blob/claude/beautiful-brown-gdjzcq/vision/RESULTS.md

## The paper in five sentences
We study video fall detection on real accident videos (OOPS!), with three test sets: FallOrNot (our own, model-independent sample where other accidents are negatives), OmniFall's OF-ItW, and HMDB51. Main result: a detector trained on 7,063 *unlabelled* accident videos, using pseudo-labels from a synthetic-trained detector plus free negatives (the clip just before each annotated accident time), with *no human label*, matches or beats training on hand labels. Frozen CLIP + VideoMAE probes: 0.894 / 0.937 / 0.953 AUROC (synthetic-only 0.814 / 0.872 / 0.917); fine-tuned VideoMAE: 0.910 on FallOrNot, 0.967 on OF-ItW (best of all fine-tuned models), 0.959 on HMDB51. Controls show most of the in-domain gain is accident-vs-ordinary-activity structure, and pseudo-labels add a smaller fall-specific gain (falls vs. other accidents +0.058, p = 0.013). Every main claim was tested under a success rule committed to the repository before the run (nine such tests, six met, three not met and reported).

## What we need from you

### 1. Novelty check (most important)
Find the closest prior work, with full citations and links, to each of these, and say plainly whether any of them already does what we do:
- training a **fall detector** (or any human-action/accident detector) from **unlabelled real video** with **pseudo-labels or self-training**, especially with a teacher trained on **synthetic** video;
- using **event timestamps** or **"the moment before the event"** as free negative examples (in fall detection, accident/anomaly detection, unintentional-action work on OOPS!, or dashcam accident anticipation);
- any work that uses **OOPS!** (Epstein et al., CVPR 2020) **for fall detection**, other than OmniFall (Schneider et al., ECCV 2026, arXiv 2505.19889), which we already cite;
- **zero-label / annotation-free fall detection** with vision-language models.
We already cite: OmniFall; Wu et al. 2023 (Neural Networks, weakly supervised MIL fall detection); Madsen et al. 2024 (WACV workshops); Xiao et al. 2023 (unsupervised RGB-to-depth adaptation for falls); Sultani et al. 2018; Noisy Student (Xie et al. 2020); UPL (Huang et al. 2022); Chan et al. 2016 (dashcam accident anticipation). Please search Google Scholar, Semantic Scholar, arXiv and CVF open access, including 2025–2026 papers. For each close paper: what it does, what data, and how it differs from ours.

### 2. Strongest fair baseline
OmniFall reports that large multimodal / video-language models do well zero-shot on in-the-wild falls. Please find:
- which open video-language models (exact Hugging Face model IDs, sizes, licences) can run zero-shot "did a person fall in this 2-second clip?" on **one 24 GB GPU (Colab L4)**;
- what OmniFall reports for such models on OF-ItW (metric, numbers, model names);
- any other published fall detector with code and weights that could be evaluated on our clips.

### 3. Venue facts (ICCV 2027 and fallbacks)
- ICCV 2027: official dates (abstract/paper deadline), page limit, template, supplementary rules, LLM-usage policy, dual-submission and arXiv rules, OpenReview requirements.
- Fallbacks: CVPR 2027 workshops or other venues where this paper would fit (healthcare, assistive technology, elderly care, video understanding, data-centric or label-efficient learning), with deadlines.

### 4. Reviewer-style critique
Read the paper as a strict ICCV reviewer would. List the five most likely reasons for rejection and, for each, the cheapest experiment or rewrite that would answer it. We can run more experiments on a Colab L4 GPU with the existing data (OOPS!, OmniFall OF-Syn and OF-ItW, HMDB51, UR Fall), but we cannot label more data.

## Please do not
- invent citations: give links for every paper you mention, and say "not found" when you find nothing;
- assume our numbers: they are in RESULTS.md.
