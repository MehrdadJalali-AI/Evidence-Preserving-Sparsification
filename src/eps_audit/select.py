"""EPS budgeted selection and baselines. Every method pays for DISTINCT retained nodes
(set union), so shared provenance/derivation nodes are counted once; all reduced methods
use the same query neighbourhood N and budget k = max(|C|, floor(frac * |N|))."""
import random
import numpy as np

from .units import audit_units, justification, certificate


def budget(q, N, frac):
    return max(len(q["cands"]), int(frac * len(N)))


def eps(g, q, k, drop=(), units=None):
    """Greedy unit selection. Bundles: each support unit with its contradiction units, in
    decreasing claim value; if a bundle does not fit, its support unit alone is tried (its
    R4 requirement is then reported unmet). Exclusion units follow, in candidate order.
    Units are added whole or not at all. `drop` removes node types (ablations) or 'contra'."""
    units = units if units is not None else audit_units(g, q)
    if "contra" in drop:
        units = [u for u in units if u["kind"] != "contradiction"]
    keep = lambda ns: {n for n in ns if n[0] not in drop}
    S = {f"M:{m}" for m in q["cands"]}
    sup = sorted([u for u in units if u["kind"] == "support"], key=lambda u: (-u["value"], u["cand"]))
    for s in sup:
        bundle = keep(s["nodes"]).union(*[keep(u["nodes"]) for u in units
                                          if u["kind"] == "contradiction" and u["cand"] == s["cand"]])
        if len(S | bundle) <= k:
            S |= bundle
        elif len(S | keep(s["nodes"])) <= k:
            S |= keep(s["nodes"])
    for u in units:
        if u["kind"] == "exclusion" and len(S | keep(u["nodes"])) <= k:
            S |= keep(u["nodes"])
    return S


def _ppr_scores(g, q, N, alpha=0.15, iters=30):
    nodes = sorted(N)
    ix = {n: i for i, n in enumerate(nodes)}
    s = np.zeros(len(nodes))
    for m in q["cands"]:
        s[ix[f"M:{m}"]] = 1.0
    s /= s.sum()
    p = s.copy()
    nbr = [[ix[v] for v in g.adj[n] if v in ix] for n in nodes]
    for _ in range(iters):
        new = alpha * s
        for i, ns in enumerate(nbr):
            if ns:
                new[ns] += (1 - alpha) * p[i] / len(ns)
        p = new
    return {n: p[ix[n]] for n in nodes}


def ppr(g, q, N, k, scores=None):
    """Top-k nodes by personalized PageRank (ties broken by node id)."""
    scores = scores or _ppr_scores(g, q, N)
    return set(sorted(N, key=lambda n: (-scores[n], n))[:k])


def random_nodes(g, q, N, k, seed):
    keep = {f"M:{m}" for m in q["cands"]}
    rest = sorted(n for n in N if n not in keep)
    random.Random(seed).shuffle(rest)
    return keep | set(rest[: max(0, k - len(keep))])


def degree(g, q, N, k):
    """Highest-degree nodes (global degree; ties by id)."""
    return set(sorted(N, key=lambda n: (-len(g.adj[n]), n))[:k])


def _closure(g, N, c):
    u = {c} | {v for v in g.adj[c] if v in N}
    for v in list(u):
        if g.ntype.get(v) == "E":
            u |= {w for w in g.adj[v] if w in N}
    return u


def type_aware(g, q, N, k, scores=None, condition_aware=False):
    """Claims ranked by PPR (condition-aware: claims at the query level first); each claim is
    added with its 1-hop neighbourhood plus its evidence's provenance/derivation, if the union
    stays within k; otherwise skipped (greedy, no backtracking)."""
    scores = scores or _ppr_scores(g, q, N)
    claims = [n for n in N if g.ntype[n] == "C"]
    def key(c):
        m, t = c.split(":", 2)[1:]
        at = g.attr[f"K:{m}:{t}"]["level_of_theory"] == q["cond"]
        return ((not at) if condition_aware else False, -scores[c], c)
    S = {f"M:{m}" for m in q["cands"]}
    for c in sorted(claims, key=key):
        u = _closure(g, N, c)
        if len(S | u) <= k:
            S |= u
    return S


def conflict_free(g, q, k, units=None):
    """CONSIST-inspired heuristic (our own construction, not the published CONSIST algorithm):
    minimal support units for comparable claims, in decreasing value; contradiction and
    exclusion units are never retained."""
    units = units if units is not None else audit_units(g, q)
    S = {f"M:{m}" for m in q["cands"]}
    for u in sorted([u for u in units if u["kind"] == "support"], key=lambda u: (-u["value"], u["cand"])):
        if len(S | u["nodes"]) <= k:
            S |= u["nodes"]
    return S
