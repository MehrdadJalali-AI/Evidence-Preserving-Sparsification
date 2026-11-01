"""Deterministic readers that answer a query from a retained node set S.

permissive (stress test): a claim whose condition node is missing is assumed comparable,
  and one whose qualification node is missing is assumed to be DFT.
conservative: abstains if any retained claim of a candidate lacks its condition or
  qualification node (it cannot tell whether that claim is comparable); otherwise it uses only
  claims at the requested level that are qualified as DFT. Whether every candidate is accounted
  for (support or exclusion) is NOT required here; it is measured separately by audit
  sufficiency and executable replay.
Both report a winner (argmax) or a set (threshold) and the contested flag(s): a selected claim
is contested if a claim contradicting it is present in S.
"""


def _claim_info(g, c, S):
    m, t = c.split(":", 2)[1:]
    k, q = f"K:{m}:{t}", f"Q:{m}:{t}"
    return (m, g.attr[k]["level_of_theory"] if k in S else None, g.attr[q]["kind"] if q in S else None)


def _finish(g, q, S, pool, abstain=False):
    if abstain:
        return dict(result=None, contested=None, used=[], answered=False)
    if q["op"] == "argmax":
        if not pool:
            return dict(result=None, contested=False, used=[], answered=True)
        best = max(pool, key=lambda c: (g.attr[c]["value"], c))
        return dict(result=g.attr[best]["material"], contested=any(x in S for x in g.contra[best]),
                    used=[best], answered=True)
    sel = sorted(c for c in pool if g.attr[c]["value"] > q["thr"])
    return dict(result=frozenset(g.attr[c]["material"] for c in sel),
                contested=frozenset(g.attr[c]["material"] for c in sel if any(x in S for x in g.contra[c])),
                used=sel, answered=True)


def permissive(g, q, S):
    pool = []
    for m in q["cands"]:
        for c in g.claims_of[m]:
            if c in S:
                _, f, kind = _claim_info(g, c, S)
                if (f is None or f == q["cond"]) and kind != "predicted":
                    pool.append(c)
    return _finish(g, q, S, pool)


def conservative(g, q, S):
    pool = []
    for m in q["cands"]:
        for c in g.claims_of[m]:
            if c not in S:
                continue
            _, f, kind = _claim_info(g, c, S)
            if f is None or kind is None:
                return _finish(g, q, S, [], abstain=True)
            if f == q["cond"] and kind == "dft":
                pool.append(c)
    return _finish(g, q, S, pool)


READERS = {"permissive": permissive, "conservative": conservative}
