#!/usr/bin/env bash
# Reproduce the principal results (about 4 minutes on one core).
# Requires data/raw/qmof.csv (see data/README.md); checksum is verified by the runner.
set -euo pipefail
for s in 1 2 3; do python -m eps_audit.experiment --config configs/qmof_main.json --seed "$s"; done
python -m eps_audit.experiment --config configs/qmof_delta05.json --seed 1
python -m eps_audit.experiment --config configs/qmof_delta15.json --seed 1
python -m eps_audit.analysis results/qmof_main results/qmof_delta05 results/qmof_delta15
mkdir -p results/figures
python scripts/fig_graph_example.py configs/qmof_main.json results/figures/fig_graph_example
