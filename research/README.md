# Research: benchmarking Motion Sense on SisFall

This folder evaluates the detector in `../index.html` on the public SisFall dataset (Zenodo record 22212281, CC-BY-4.0). It also proposes and tests a revised rule. The write-up is in [`MANUSCRIPT.md`](MANUSCRIPT.md).

## Reproduce

```bash
pip install numpy scikit-learn
curl -L -o SisFall.zip https://zenodo.org/api/records/22212281/files/SisFall.zip/content
unzip SisFall.zip            # -> SisFall_dataset/
python3 research/benchmark_sisfall.py SisFall_dataset   # shipped pipeline + learned reference (~1 min)
python3 research/rule_v2.py           SisFall_dataset   # proposed rule, dev/test split (~1 min)
python3 research/diagnose_misses.py   SisFall_dataset   # why the shipped rule misses falls
```

Outputs are written to `research/results/`.

## Headline numbers

| | Sensitivity | Specificity |
|---|---|---|
| Shipped rule + 5 s confirmation (all 38 subjects) | 18.0 % | 97.9 % |
| Proposed v2 rule (held-out 20 subjects) | 93.9 % | 95.5 % |
