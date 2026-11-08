# Staged-to-real fall detection (CVPR-track project)

**Question.** Video fall detectors report ~99% accuracy on datasets of actors falling onto mats in a lab. Do they detect *real* falls?

**Plan.**
1. **Build a real-world fall test set** from OOPS! (Epstein et al., CVPR 2020), which has about 20k in-the-wild "fail" videos, about 11k of them annotated with when the accident happens. Falls are not labelled there, so we label them. The test set is a model-independent random sample, so no detector is favoured by how clips were picked.
2. **Measure the gap.** Train on staged data (UR Fall), test on staged data and on real falls, using the same clip definition and frozen encoder for both.
3. **Close the gap** with a method that generalises. To be designed once the gap and its failure modes are measured. Candidates: video foundation-model features, pose-and-motion priors, synthetic or real hard negatives.
4. **Release the benchmark** (labels, splits, code).

**Status.** The pipeline is built and smoke-tested on CPU with a small model and fake labels. That test only checks that the code runs; the numbers are meaningless. No real results yet.

## Run it
Open [`fall_cvpr.ipynb`](fall_cvpr.ipynb) in Google Colab: *File → Open notebook → GitHub*, then paste this repository's URL and pick the branch. Then run the steps in order (GPU runtime).

| File | Purpose |
|---|---|
| `common.py` | frame sampling, clip export, frozen CLIP encoder and zero-shot prompts |
| `oops_mine.py` | streams the 48 GB OOPS! archive once (decoding in background threads); saves 8 ordered frame embeddings per clip; keeps candidate clips; resumable |
| `labeler.py` | in-Colab labelling tool: test-set clips first, keyboard shortcuts F/N/U/Z, saves every label |
| `urfall_prep.py` | UR Fall clips: falls (side and ceiling cameras), pre-fall and everyday activity |
| `experiment.py` | staged → real evaluation with *appearance* (mean-frame) and *temporal* (motion-aware) probes: AUROC, AP and recall at 5% FPR with video-level bootstrap CIs, an ROC chart, and the worst errors |

## Data and licences
- **UR Fall Detection Dataset**: Kwolek & Kepski, *CMPB* 2014. CC BY-NC-SA 4.0, non-commercial academic use.
- **OOPS!**: Epstein, Chen & Vondrick, *CVPR* 2020. The original host is offline, so the notebook streams a third-party Hugging Face mirror (`pengxiang/oops`) whose archive has the original structure. Before publishing, confirm the dataset's terms of use with its authors and cite them. Videos come from public YouTube compilations; release labels and timestamps, not videos.

## Known gaps before a submission
- Only one staged dataset so far. Le2i's original page is offline; add Le2i, the Multiple Cameras Fall dataset, or others if obtainable.
- Labels from one person. Have a second person label the test set to report agreement (Cohen's κ).
- No novel method yet. This pipeline only measures the problem.
