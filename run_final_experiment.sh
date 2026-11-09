#!/usr/bin/env bash
set -euo pipefail

# Final three-method evaluation: Conventional ILP, QoS-aware ILP, LinUCB.
# Run from the Python project root in Git Bash.
PYTHONPATH=src python src/qos_contextual_bandit.py \
  --train-scenarios 300 \
  --test-scenarios 100 \
  --seed 20261110 \
  --topology-seed 2 \
  --sa-maxiter 10\
  --sa-maxfun 1000 \
  --alpha 0.7 \
  --ridge 1.0 \
  --output-dir results/qos_final_1000x100


source result_plotter.sh --results-dir results/qos_final_1000x100