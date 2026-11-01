#!/usr/bin/env bash
# Lightweight smoke test on a tiny synthetic table (seconds; no external data).
set -euo pipefail
rm -rf results/smoke
python -m eps_audit.experiment --config configs/smoke.json --seed 1
python -m eps_audit.analysis results/smoke
python -m pytest -q
