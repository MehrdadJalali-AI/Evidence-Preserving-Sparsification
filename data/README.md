# Data

Raw data are not part of this repository.

1. Download `qmof_database.zip` from the QMOF figshare record: https://doi.org/10.6084/m9.figshare.13147324
   (calculation records on NOMAD: https://doi.org/10.17172/NOMAD/2021.10.10-1).
2. Unzip and copy `qmof.csv` to `data/raw/qmof.csv`.
3. Expected SHA-256 of the snapshot used in the paper:
   `9991b2d51d71a6cc4c97affd8ca36ac18d5b75554b4f3f1ac9ebb35bc9995f26`
   (20,372 MOFs; band gaps: PBE 20,372, HLE17 10,756, HSE06* 10,811, HSE06 10,810).

Exclusion records are defined relative to this snapshot under a closed-world assumption:
"no DFT band gap at level f" means the column `outputs.<f>.bandgap` is missing in this file.
They say nothing about calculations outside this snapshot.
