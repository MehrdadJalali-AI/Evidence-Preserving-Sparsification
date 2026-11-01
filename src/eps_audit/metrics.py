"""Per-query, per-method metrics. Conventions:
* correct_<reader>: reader answered AND result and contested flag(s) equal the reference
  (abstentions count as incorrect; coverage is reported separately as answered_<reader>).
* unsupported: among claims USED by the reader, those lacking their evidence node or its
  provenance node in S; denominator n_used (queries with n_used = 0 contribute no ratio).
* contradiction retention: among contradiction pairs on the reference's used claims, pairs
  with both claims in S; denominator n_contra_pairs (queries without pairs are excluded).
* audit_sufficient: every canonical unit is complete (all R1-R5 met); replay_ok: executable
  replay reproduces the reference answer from S alone.
"""
from .units import certificate
from .readers import READERS
from .replay import replay


def evidence_ok(g, c, S):
    m, t = c.split(":", 2)[1:]
    e = f"E:{m}:{t}"
    return e in S and any(p in S for p in g.adj[e] if g.ntype[p] == "P")


def same(a, ref):
    return a["result"] == ref["result"] and a["contested"] == ref["contested"]


def evaluate(g, q, S, units, ref):
    cert = certificate(units, S)
    row = dict(size=len(S), edges=g.induced_edges(S), audit_sufficient=int(cert["audit_sufficient"]))
    for r in ("R1", "R2", "R3", "R4", "R5", "R2x"):
        row[f"unmet_{r}"] = int(r in cert["unmet"])
    for kind, col in (("support", "support_cov"), ("exclusion", "exclusion_cov"), ("contradiction", "contra_unit_cov")):
        us = [u for u in cert["units"] if u["kind"] == kind]
        row[col] = (sum(u["status"] == "complete" for u in us) / len(us)) if us else float("nan")
        row[f"n_{kind}"] = len(us)
    for name, reader in READERS.items():
        a = reader(g, q, S)
        row[f"answered_{name}"] = int(a["answered"])
        row[f"correct_{name}"] = int(a["answered"] and same(a, ref))
        row[f"n_used_{name}"] = len(a["used"])
        row[f"n_unsupported_{name}"] = sum(not evidence_ok(g, c, S) for c in a["used"])
    pairs = [(c, cc) for c in ref["used"] for cc in g.contra[c]]
    row["n_contra_pairs"] = len(pairs)
    row["n_contra_kept"] = sum(c in S and cc in S for c, cc in pairs)
    rp = replay(g, q, S, units)
    row["replay_ok"] = int(rp is not None and rp["result"] == ref["result"] and rp["contested"] == ref["contested"])
    return row
