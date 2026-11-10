# One-command reproduction: `make all` (about 30-60 min on a 4-core laptop).
# Alternative without make: bash scripts/run_all_parallel.sh
PY ?= python

.PHONY: all test main robustness rolling stats sensitivity figures clean
all: test main robustness rolling stats sensitivity figures

test:
	$(PY) -m pytest -q
main:
	$(PY) scripts/run_experiments.py
robustness:
	$(PY) scripts/run_robustness.py
rolling:   # 4 single-threaded shards in parallel (much faster than one multi-threaded run)
	for k in 0 1 2 3; do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PY) scripts/run_rolling.py $$k 4 & done; wait
	$(PY) scripts/run_rolling.py merge 4
sensitivity:
	$(PY) scripts/run_sensitivity.py
	$(PY) scripts/run_kink_check.py
	$(PY) scripts/run_shared_threshold.py
stats:
	$(PY) scripts/paired_tests.py
figures:
	$(PY) scripts/make_figures.py
clean:
	rm -rf data results/*.pkl results/*.log
