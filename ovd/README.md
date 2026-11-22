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
