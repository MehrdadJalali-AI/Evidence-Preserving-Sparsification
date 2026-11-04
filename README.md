# Evidence-Preserving Sparsification (EPS)
<p align="center">
  <img src="EPS.jpg" alt="Evidence-Preserving Sparsification (EPS)" width="30%">
</p>
Reference implementation and reproducibility package for

> M. Jalali and M. H. Nadimi-Shahraki, *Evidence-Preserving Knowledge Graph Sparsification for Auditable AI in Materials Discovery* (manuscript in preparation for *Discover Informatics*).

EPS reduces a scientific evidence graph under a node budget while keeping whole **audit units**:
support units for comparable claims (claim, evidence record, condition, qualification, provenance,
derivation), contradiction units, and closed-world **exclusion records** derived from a versioned
dataset snapshot. A certificate reports, per unit, whether it is complete, partial or omitted, and
which requirements (R1-R5, R2x) remain unmet. An executable checker replays the answer from the
retained subgraph alone.

## Install
```bash
pip install -e ".[test]"          # Python >= 3.10
```

## Smoke test (seconds, no external data)
```bash
bash scripts/smoke.sh              # tiny synthetic table + unit tests
python examples/toy_example.py     # one query, certificate, readers, replay
```

## Data
QMOF is not redistributed. See [`data/README.md`](data/README.md): download `qmof_database.zip`
(figshare, doi:10.6084/m9.figshare.13147324), place `qmof.csv` at `data/raw/qmof.csv`.
The runner verifies SHA-256 `9991b2d51d71a6cc4c97affd8ca36ac18d5b75554b4f3f1ac9ebb35bc9995f26`
(the snapshot used for the reported results; a different file aborts the run).

## Reproduce the principal results (~4 min, one core)
```bash
python3.11 -m pip install -r requirements-results.txt -e ".[test]"
bash scripts/reproduce.sh
```
This runs 3 seeds x 300 queries (main setting, contradiction threshold 1.0 eV) and one seed at
0.5 and 1.5 eV, then writes `results/<run>/analysis/` (summary with bootstrap CIs, McNemar tests,
budget-to-95%, sensitivity, LaTeX tables, trade-off figure) and the real-query example figure.
The runner is resumable: rerunning continues after the last completed query. The result environment
is pinned because the cross-fitted prediction model is sensitive to library-version changes.

## Layout
| Path | Content |
|---|---|
| `src/eps_audit/graph.py` | evidence graph, closed-world exclusion records, cross-fitted predictions, prediction reproduction |
| `src/eps_audit/units.py` | queries, query neighbourhood, canonical audit units, certificate |
| `src/eps_audit/select.py` | EPS budgeted selection and all baselines (distinct-node budget accounting) |
| `src/eps_audit/readers.py` | permissive (stress-test) and conservative readers |
| `src/eps_audit/replay.py` | executable answer replay |
| `src/eps_audit/metrics.py` | per-query metrics and their denominators |
| `src/eps_audit/experiment.py` | resumable runner (per-query CSV output) |
| `src/eps_audit/analysis.py` | statistics, tables, figure |
| `configs/` | versioned experiment configurations |
| `tests/` | unit tests: exclusion rules, budget compliance, shared-node accounting, formal properties, replay, readers, cross-fitting |
| `results/` | compact released outputs (per-query results gzipped, analysis) |

## Protocol (summary)
* Queries: 20 random QMOF materials; argmax or threshold (> 2.5 eV) on the band gap at HLE17, HSE06* or HSE06; 300 per seed, seeds 1-3.
* Predictions: HistGradientBoostingRegressor (max_iter = 300, random_state = seed) on PBE gap + density, PLD, LCD, number of atoms, cell volume; missing descriptors -> column median; 2-fold cross-fitting, no hyper-parameter tuning; predictions are reproducible exactly (checked per run).
* Contradiction: |prediction - DFT HSE06| > delta (1.0 eV; sensitivity 0.5 and 1.5 eV).
* Budget: k = max(|candidates|, floor(fraction x |query neighbourhood|)); every method pays for distinct retained nodes.
* Statistics: bootstrap 95% CIs over queries (2,000 resamples, seed 0); exact McNemar tests paired by (seed, query).

## Citation
See `CITATION.cff`.

## Licence
Code: MIT (`LICENSE`) — **pending author confirmation**. QMOF data remain under their own licence.
