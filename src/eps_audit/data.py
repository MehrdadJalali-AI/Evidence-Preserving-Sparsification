"""Data loading: QMOF table, band-gap columns, checksums, and a small synthetic table for tests."""
import hashlib
import re
import numpy as np
import pandas as pd

FEATURES = ["info.density", "info.pld", "info.lcd", "info.natoms", "info.volume"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_table(path):
    df = pd.read_csv(path, low_memory=False)
    if "qmof_id" not in df.columns:
        raise ValueError("expected a 'qmof_id' column (QMOF qmof.csv)")
    return df.reset_index(drop=True)


def functionals(df):
    """Map level-of-theory name -> band-gap column (pattern outputs.<f>.bandgap)."""
    out = {}
    for c in df.columns:
        m = re.fullmatch(r"outputs\.(\w+)\.bandgap", c)
        if m:
            out[m.group(1)] = c
    if not out:
        raise ValueError("no outputs.<functional>.bandgap columns found")
    return out


def synthetic_table(n=60, seed=0):
    """Tiny QMOF-schema table for tests and the smoke example. NOT real data."""
    r = np.random.default_rng(seed)
    pbe = np.clip(r.gamma(2.0, 0.9, n), 0.0, 6.0)
    df = pd.DataFrame({
        "qmof_id": [f"syn-{i:04d}" for i in range(n)],
        "info.density": r.normal(1.2, 0.3, n).clip(0.2),
        "info.pld": r.gamma(2.5, 1.6, n),
        "info.lcd": r.gamma(3.0, 1.8, n),
        "info.natoms": r.integers(20, 400, n),
        "info.volume": r.gamma(4.0, 800.0, n),
        "outputs.pbe.bandgap": pbe,
        "outputs.hle17.bandgap": pbe * 1.25 + 0.3 + r.normal(0, 0.15, n),
        "outputs.hse06.bandgap": pbe * 1.35 + 0.55 + r.normal(0, 0.6, n),
    })
    for col, keep in [("outputs.hle17.bandgap", .7), ("outputs.hse06.bandgap", .5)]:
        df.loc[r.random(n) > keep, col] = np.nan
    return df
