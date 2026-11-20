# Video fall detection from staged, synthetic, labelled and unlabelled real video

**Question.** Video fall detectors report ~99% accuracy on datasets of actors falling onto mats in a lab. Which training data prepares them for *real* falls, and does real video have to be labelled to be useful?

**What is here.**
1. **FallOrNot**, a real-world test set from OOPS! (Epstein et al., CVPR 2020): a model-independent random sample of accident videos, hand-labelled as fall or not fall, so that other accidents act as false-alarm negatives.
2. **A comparison of training sources** (staged UR Fall, synthetic OmniFall OF-Syn, hand-labelled real OOPS! clips) on three test sets: FallOrNot, OmniFall's OF-ItW test split and HMDB51.
3. **Zero-label adaptation** (`nolabel.py`): training on thousands of unlabelled accident clips, using pseudo-labels and the free "before the accident" clips as non-falls, with controls that show where the gain comes from.
4. **Few-label adaptation** (`adapt.py`, `adapt_ft.py`): anchoring a probe to the synthetic solution, and sequential fine-tuning.

**Status.** All results are recorded run by run in [`RESULTS.md`](RESULTS.md) (runs 1 to 17), and the paper is in [`paper/`](paper/). Every success rule was committed to `AGENTS.md` before the corresponding run; rules that were not met are reported too.

## Run it
Open [`fall_cvpr.ipynb`](fall_cvpr.ipynb) in Google Colab: *File → Open notebook → GitHub*, then paste this repository's URL and pick the branch. Then run the steps in order (GPU runtime).

| File | Purpose |
|---|---|
| `common.py` | frame sampling (16 frames per clip), clip export, frozen CLIP image encoder with zero-shot prompts, frozen VideoMAE video encoder |
| `oops_mine.py` | streams the 48 GB OOPS! archive once (decoding in background threads); saves 8 ordered frame embeddings per clip; keeps candidate clips; resumable |
| `labeler.py` | in-Colab labelling tool: test-set clips first, keyboard shortcuts F/N/U/Z, saves every label |
| `urfall_prep.py` | UR Fall clips: falls (side and ceiling cameras), pre-fall and everyday activity |
| `ofsyn_prep.py` | OmniFall synthetic videos (OF-Syn): streams the 9.7 GB archive, one 2 s clip per video (fall-centred, or a hash-sampled non-fall), same encoders |
| `hmdb_prep.py` | External real-world test: HMDB51 fall_floor clips vs. 20 hash-sampled clips per other class (and a pre-defined hard subset), 2 s windows every 1 s, clip score = max window |
| `finetune.py` | Fine-tunes VideoMAE-B on synthetic, real or both (fixed hyper-parameters) and scores FallOrNot and HMDB51; stages are resumable. Pre-registered test reported by `experiment.py --finetuned` |
| `ofitw_prep.py` | OmniFall's in-the-wild test set (OF-ItW): 818 OOPS! videos labelled by OmniFall's team, same clip definition and encoders |
| `adapt.py` | Anchored probe: a few real clips adapt the synthetic-trained probe (rule fixed in advance) |
| `adapt_ft.py` | The same question for end-to-end fine-tuning, with a no-penalty ablation |
| `nolabel.py` | Zero-label adaptation from unlabelled accident clips; `--explore`, `--hard`, `--hard2` add sensitivity, hard-subset and extended hard tests |
| `nolabel_ft.py` | Zero-label training set used to fine-tune VideoMAE end to end |
| `agreement.py` | Agreement between two annotators (Cohen's kappa) |
| `experiment.py` | staged → real evaluation of four representations (CLIP appearance, CLIP temporal, VideoMAE, CLIP+VideoMAE): AUROC, AP and recall at 5% FPR with video-level bootstrap CIs, an ROC chart, and the worst errors |

## Data and licences
- **UR Fall Detection Dataset**: Kwolek & Kepski, *CMPB* 2014. CC BY-NC-SA 4.0, non-commercial academic use.
- **OOPS!**: Epstein, Chen & Vondrick, *CVPR* 2020. The original host is offline, so the notebook streams a third-party Hugging Face mirror (`pengxiang/oops`) whose archive has the original structure. Cite them. Videos come from public YouTube compilations; release labels and timestamps, not videos.

## Known gaps before a submission
- Only one staged dataset (UR Fall); Le2i's original page is offline.
- A second annotator agreed only moderately with the first (Cohen's kappa 0.50) on 100 test videos; our labels agree with OmniFall's on 27 of 28 shared videos.
- Without labels, the gain on falls vs. other accidents in domain is modest (+0.058, p = 0.013, run 18) and not significant on the model-selected training-pool clips alone. Zero-label fine-tuning (`nolabel_ft.py`, run 19) and consequence-guided adaptation (`post_prep.py` + `nolabel.py --post`, run 21) have their rules fixed and are running.
- The zero-label method relies on accident timestamps, which OOPS! provides and arbitrary video does not.
