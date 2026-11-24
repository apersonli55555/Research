# Vocabulary-set interference in open-vocabulary detection

Does an open-vocabulary detector's answer for one class change when *other* class names are added, removed or
reordered? The study plan, fixed before any result, is in `PLAN.md`. Every run is recorded in `RESULTS.md`.

## Setup
Python 3.10+, a CUDA GPU (we use one NVIDIA L4, 24 GB). The code was checked on CPU on 3 images.

    pip install torch transformers ultralytics open_clip_torch pillow numpy pytest scikit-learn
    python -m pytest ovd/test_vocab_pilot.py -q        # unit tests (matching, AP, seeding)

Public data only: the COCO val2017 annotations and images are downloaded automatically by the `data` stage.

## Pilot (OVD run 1)
    python ovd/vocab_pilot.py data    --out OUT            # sample 500 images by hash, build the fixed plan
    python ovd/vocab_pilot.py run     --out OUT --model gdino
    python ovd/vocab_pilot.py run     --out OUT --model owlv2 --owl-fast
    python ovd/vocab_pilot.py run     --out OUT --model yoloworld
    python ovd/vocab_pilot.py analyze --out OUT --boot 1000

`run` is resumable (it appends to `OUT/results_<model>.jsonl` and skips finished runs). `analyze` writes
`OUT/pilot_summary.{json,md}` and prints the fixed GO / STOP decision.

Model checkpoints: `IDEA-Research/grounding-dino-base`, `google/owlv2-base-patch16-ensemble` (Hugging Face) and
`yolov8s-worldv2.pt` (ultralytics, AGPL-3.0). Seeds derive from MD5 hashes of the image and class names, so every
vocabulary draw is identical on every machine.

## Full study (OVD runs 2–4, only after a GO from the pilot)
    for DS in coco lvis; do
      python ovd/vocab_full.py data --dataset $DS --out OUT
      for M in owlv2 yoloworld gdino gdino_qs; do python ovd/vocab_full.py run --dataset $DS --out OUT --model $M; done
      for M in owlv2 yoloworld gdino; do python ovd/vocab_full.py bench --dataset $DS --out OUT --model $M; done
    done
    python ovd/full_analysis.py --out OUT --boot 1000      # writes OUT/full_summary.{md,json}

## Fixed class bank (OVD runs 3g–3h and follow-ups)
The bank method and its 600-image confirmation run: `bank/*_cells.py` (the expert's notebook cells),
`results/bank/fixed_bank_soft_confirm_v1/` (saved per-image outputs). All of the following run on CPU from saved files,
with the official `lvis_v1_val.json`:

    python ovd/bank_analysis.py --lvis lvis_v1_val.json                 # Table 1 + CIs, gain breakdown, duplicate boxes
    pip install lvis pycocotools
    python ovd/bank/official_lvis_eval.py --lvis lvis_v1_val.json       # the same outputs scored by the official LVIS evaluator

Follow-up GPU experiments (requests chosen without the image, bank dictionary and construction ablations, cost from 1 to
1,203 names, grounding-dino-tiny, figure thumbnails): open `bank/followup_bank.ipynb` in Colab on an L4 and run all
(resumable; ≈1.5–2 h), or `python ovd/bank/followup.py --out OUT`. Then
`python ovd/bank/followup_analysis.py --dir OUT --lvis lvis_v1_val.json`.
