#!/usr/bin/env bash
# Runs the expensive studies efficiently on a multi-core machine.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-4}
python scripts/run_experiments.py
python scripts/run_robustness.py
for k in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_rolling.py "$k" "$N" &
done
wait
python scripts/run_rolling.py merge "$N"
python scripts/paired_tests.py
python scripts/run_sensitivity.py
python scripts/run_kink_check.py
python scripts/make_figures.py
