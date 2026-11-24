# Vocabulary-Set Interference in Open-Vocabulary Object Detection — package for review

Repo: github.com/apersonli55555/Research, branch `claude/beautiful-brown-gdjzcq` (this zip = its `ovd/` folder).

Start here:
- `paper/paper.pdf` — current 6-page draft (LaTeX: `paper/paper.tex`). Every decimal number in it is traced to
  `RESULTS.md` or `results/` by `check_numbers.py` (242/242).
- `RESULTS.md` — complete lab log of every run (including failed and negative ones), with commands and dates.
- `PLAN.md` — pre-registered plan and dated amendments (rules fixed before results).
- `NOVELTY.md` — novelty vs. 2024–2026 work, pre-registered success criteria, reviewer risks, readiness.
- `results/` — analysis summaries and raw per-image outputs (incl. `results/bank/` for the fixed class bank:
  pilot, 100-image validation, 600-image confirmation + `independent_verification.log`).
- Code: `vocab_pilot.py` (pilot), `vocab_full.py` (held-out runs), `full_analysis.py` (evaluator),
  `independent_check.py` (separately written re-check), `bank/` (the fixed-bank notebook cells), ablations
  (`guide_ablation.py`, `norm_ablation.py`, `anchor_dev.py`, `shortlist.py`, `shared_readout.py`, `read_only_text.py`),
  tests `test_*.py`.

Open questions for you:
1. The notebook that produced `results/bank/fixed_bank_soft_confirm_v1/` is not in the repo — please share it.
2. Which data were used to choose class-wise Gaussian Soft-NMS and σ = 0.5 (should be pilot/validation only)?
3. Anything in the paper you would change before submission (claims, framing, missing baselines, related work)?
