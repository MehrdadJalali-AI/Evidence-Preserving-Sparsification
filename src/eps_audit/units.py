"""Queries, query neighbourhoods and the canonical family of audit units.

Canonical selection sigma(q): for every candidate m and requested level f
  * if the snapshot holds a DFT claim C:m:f  -> one SUPPORT unit
        {M:m, C:m:f, E:m:f, K:m:f, Q:m:f, P:snapshot, D:extract:f}       (R1, R2, R3, R5)
    and, for every claim contradicting C:m:f, one CONTRADICTION unit
        {M:m, C:m:f, c', E', K', Q', P:model, D:model}                     (R4)
    Each claim has exactly one evidence record in this graph; if several existed, sigma
    would select the lexicographically smallest record id (deterministic).
  * otherwise -> one EXCLUSION unit {M:m, X:m:f, P:snapshot, D:lookup:f}    (R2 closed-world)
Units are node sets; retained subgraphs are the subgraphs induced by the retained nodes.
"""
from collections import deque
import numpy as np

REQS = {"support": ("R1", "R2", "R3", "R5"), "contradiction": ("R4",), "exclusion": ("R2x",)}


def make_queries(mids, conds, n, ncand, seed, threshold=2.5):
    rng = np.random.default_rng(seed)
    qs = []
    for i in range(n):
        f = str(rng.choice(conds))
        cands = [str(c) for c in rng.choice(mids, size=ncand, replace=False)]
        qs.append(dict(qid=i, op="argmax" if i % 2 == 0 else "threshold", cond=f, cands=cands, thr=threshold))
    return qs


def neighbourhood(g, q, hops=3):
    """Nodes within `hops` of the candidate materials. Expansion never passes through other
    material nodes or through shared provenance/derivation hubs (P, D), which are reached but
    not expanded."""
    seeds = {f"M:{m}" for m in q["cands"]}
    seen = set(seeds)
    dq = deque((s, 0) for s in seeds)
    while dq:
        u, d = dq.popleft()
        if d == hops or (g.ntype[u] in ("P", "D")):
            continue
        for v in g.adj[u]:
            if v in seen or g.ntype[v] == "M":
                continue
            seen.add(v)
            dq.append((v, d + 1))
    return seen


def audit_units(g, q, contra=True):
    """Return list of units: dict(kind, cand, value, nodes, parent)."""
    f = q["cond"]
    units = []
    for m in q["cands"]:
        c = f"C:{m}:{f}"
        if c in g.ntype:
            sup = dict(kind="support", cand=m, value=g.attr[c]["value"], parent=None,
                       nodes=frozenset({f"M:{m}", c, f"E:{m}:{f}", f"K:{m}:{f}", f"Q:{m}:{f}",
                                        "P:snapshot", f"D:extract:{f}"}))
            units.append(sup)
            if contra:
                for cc in sorted(g.contra[c]):
                    t = cc.split(":", 2)[2]
                    units.append(dict(kind="contradiction", cand=m, value=sup["value"], parent=c,
                                      nodes=frozenset({f"M:{m}", c, cc, f"E:{m}:{t}", f"K:{m}:{t}", f"Q:{m}:{t}",
                                                       "P:model", "D:model"})))
        else:
            units.append(dict(kind="exclusion", cand=m, value=None, parent=None,
                              nodes=frozenset({f"M:{m}", f"X:{m}:{f}", "P:snapshot", f"D:lookup:{f}"})))
    return units


def justification(units):
    return frozenset().union(*(u["nodes"] for u in units)) if units else frozenset()


def certificate(units, S):
    """Per-unit status and unmet requirements for a retained node set S.
    status: complete | omitted (no own node retained) | partial.

    An own node is a node in the unit that is neither an anchor material nor in any
    other unit. Thus, shared nodes alone cannot make a unit partial.
    """
    out, unmet = [], set()
    anchors = {n for u in units for n in u["nodes"] if n.startswith("M:")}
    for i, u in enumerate(units):
        missing = sorted(u["nodes"] - S)
        other_nodes = set().union(*(v["nodes"] for j, v in enumerate(units) if j != i))
        own = set(u["nodes"]) - (anchors | other_nodes)
        status = "complete" if not missing else ("omitted" if not (own & S) else "partial")
        if missing:
            unmet.update(REQS[u["kind"]])
        out.append(dict(kind=u["kind"], cand=u["cand"], status=status, missing=missing))
    return dict(units=out, unmet=sorted(unmet), audit_sufficient=not unmet)
