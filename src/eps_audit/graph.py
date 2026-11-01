"""Auditable evidence graph.

Node types: M material, C claim, E evidence record, K condition (level of theory),
Q qualification (dft | predicted), P provenance (dataset snapshot; prediction model),
D derivation (column extraction; model prediction; snapshot lookup), X exclusion record.

Exclusion records follow a closed-world assumption relative to one versioned snapshot:
X:<m>:<f> exists iff the snapshot contains no DFT band gap at level <f> for material <m>.
It records the candidate, condition, eligibility rule and lookup outcome, and links to the
snapshot provenance node (with checksum) and the lookup derivation node.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import FEATURES

HUB_TYPES = ("P", "D")
ELIGIBILITY_RULE = ("comparable iff the snapshot holds a DFT band gap at the requested "
                    "level of theory (column outputs.<f>.bandgap is non-missing)")


class EvidenceGraph:
    def __init__(self):
        self.ntype, self.attr = {}, {}
        self.adj = defaultdict(set)
        self.erel = {}
        self.claims_of = defaultdict(list)
        self.contra = defaultdict(set)
        self.meta = {}

    def add(self, nid, t, **a):
        self.ntype[nid] = t
        self.attr[nid] = a

    def link(self, u, v, rel):
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.erel[(u, v)] = rel

    def has_dft(self, m, f):
        return f"C:{m}:{f}" in self.ntype

    def induced_edges(self, S):
        """Number of edges of the subgraph induced by S (hub nodes are iterated from the non-hub side)."""
        n = 0
        for u in S:
            if self.ntype[u] in HUB_TYPES:
                continue
            for v in self.adj[u]:
                if v in S and (self.ntype[v] in HUB_TYPES or u < v):
                    n += 1
        return n


def build_graph(df, funcs, target="hse06", snapshot_id="QMOF", snapshot_sha256=None,
                seed=1, contra_ev=1.0, model_params=None):
    """Build the evidence graph. Predictions use 2-fold cross-fitting: every material that has a
    target value is predicted by the model trained on the other fold; materials without a target
    value receive the mean of both fold models. No hyper-parameters are tuned."""
    model_params = dict(max_iter=300, random_state=seed, **(model_params or {}))
    g = EvidenceGraph()
    g.add("P:snapshot", "P", source=snapshot_id, sha256=snapshot_sha256)
    g.add("P:model", "P", source="HistGradientBoostingRegressor", params=model_params)
    for f in funcs:
        g.add(f"D:extract:{f}", "D", op=f"read outputs.{f}.bandgap (eV)")
        g.add(f"D:lookup:{f}", "D", op=f"closed-world lookup of outputs.{f}.bandgap", rule=ELIGIBILITY_RULE)
    g.add("D:model", "D", op=f"cross-fitted prediction of {target} band gap from PBE gap + geometry")

    def add_claim(m, tag, f, val, kind):
        c = f"C:{m}:{tag}"
        g.add(c, "C", material=m, prop="bandgap", value=float(val), unit="eV")
        for t, a in (("E", dict(record=f"{m}/{tag}")), ("K", dict(level_of_theory=f)), ("Q", dict(kind=kind))):
            g.add(f"{t}:{m}:{tag}", t, **a)
        g.link(c, f"M:{m}", "about")
        g.link(c, f"E:{m}:{tag}", "supportedBy")
        g.link(c, f"K:{m}:{tag}", "measuredUnder")
        g.link(c, f"Q:{m}:{tag}", "qualifiedBy")
        if kind == "dft":
            g.link(f"E:{m}:{tag}", "P:snapshot", "prov:wasAttributedTo")
            g.link(f"E:{m}:{tag}", f"D:extract:{f}", "prov:wasGeneratedBy")
        else:
            g.link(f"E:{m}:{tag}", "P:model", "prov:wasAttributedTo")
            g.link(f"E:{m}:{tag}", "D:model", "prov:wasGeneratedBy")
        g.claims_of[m].append(c)
        return c

    ids = df["qmof_id"].astype(str).tolist()
    for i, m in enumerate(ids):
        g.add(f"M:{m}", "M")
        for f, col in funcs.items():
            v = df.at[i, col]
            if pd.notna(v):
                add_claim(m, f, f, v, "dft")
            else:
                x = f"X:{m}:{f}"
                g.add(x, "X", candidate=m, condition=f, rule=ELIGIBILITY_RULE, outcome="absent",
                      snapshot=snapshot_id, sha256=snapshot_sha256)
                g.link(x, f"M:{m}", "exclusionOf")
                g.link(x, "P:snapshot", "prov:wasDerivedFrom")
                g.link(x, f"D:lookup:{f}", "prov:wasGeneratedBy")

    # cross-fitted delta-learning predictions
    feats = [c for c in FEATURES if c in df.columns]
    X = df[feats].astype(float)
    X = X.fillna(X.median()).to_numpy()            # missing descriptors: column median
    if "pbe" in funcs and funcs["pbe"] != funcs[target]:
        X = np.column_stack([df[funcs["pbe"]].astype(float).to_numpy(), X])
    y = df[funcs[target]].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    idx = np.where(~np.isnan(y))[0]
    rng.shuffle(idx)
    folds = [np.sort(idx[: len(idx) // 2]), np.sort(idx[len(idx) // 2:])]
    models = [HistGradientBoostingRegressor(**model_params).fit(X[fo], y[fo]) for fo in folds]
    pred = 0.5 * (models[0].predict(X) + models[1].predict(X))
    pred[folds[0]] = models[1].predict(X[folds[0]])
    pred[folds[1]] = models[0].predict(X[folds[1]])
    g.meta.update(folds=folds, X=X, y=y, model_params=model_params, seed=seed, target=target,
                  pred=pred, contra_ev=contra_ev,
                  pred_mae=float(np.mean(np.abs(pred[idx] - y[idx]))),
                  pred_r2=float(1 - np.sum((pred[idx] - y[idx]) ** 2) / np.sum((y[idx] - y[idx].mean()) ** 2)))
    for i, m in enumerate(ids):
        pc = add_claim(m, f"pred_{target}", target, pred[i], "predicted")
        dc = f"C:{m}:{target}"
        if dc in g.ntype and abs(g.attr[dc]["value"] - pred[i]) > contra_ev:
            g.link(pc, dc, "contradicts")
            g.contra[pc].add(dc)
            g.contra[dc].add(pc)
    g.meta["n_contradictions"] = sum(len(v) for v in g.contra.values()) // 2
    return g


def reproduce_predictions(g, sample=200):
    """Executable prediction reproduction: refit both fold models from the recorded
    configuration and fold indices and compare with stored predictions."""
    X, y, folds = g.meta["X"], g.meta["y"], g.meta["folds"]
    models = [HistGradientBoostingRegressor(**g.meta["model_params"]).fit(X[fo], y[fo]) for fo in folds]
    test = np.concatenate([folds[0][:sample // 2], folds[1][:sample // 2]])
    rep = np.where(np.isin(test, folds[0]), models[1].predict(X[test]), models[0].predict(X[test]))
    return float(np.max(np.abs(rep - g.meta["pred"][test])))
