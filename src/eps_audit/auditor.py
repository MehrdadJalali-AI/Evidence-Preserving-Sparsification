"""Method-independent auditor. It does not use the canonical audit units of EPS. Given the full
evidence graph (to know which claims and contradictions exist) and a retained node set S, it
accepts ANY justification that lets an auditor re-derive the answer:
  * every candidate is accounted for, either by a documented claim at the requested level
    qualified as DFT (claim, condition, qualification, evidence linked to some provenance and
    some derivation node), or by a documented exclusion record (linked to provenance and
    derivation);
  * every claim that determines the answer (the winner for argmax; every selected claim for a
    threshold query) has each of its contradicting claims documented in S;
  * the answer recomputed from the documented material equals the reference answer.
Unlike canonical audit sufficiency, it requires contradictions only for answer-determining
claims and accepts any provenance/derivation path."""
from .readers import documented_claim, documented_exclusion


def _level(g, c):
    m, t = c.split(":", 2)[1:]
    return g.attr[f"K:{m}:{t}"]["level_of_theory"], g.attr[f"Q:{m}:{t}"]["kind"]


def independent_audit(g, q, S, ref):
    f = q["cond"]
    vals = {}
    for m in q["cands"]:
        doc = [c for c in g.claims_of[m] if _level(g, c) == (f, "dft") and documented_claim(g, c, S)]
        if doc:
            best = max(doc, key=lambda c: (g.attr[c]["value"], c))
            vals[m] = best
        elif not documented_exclusion(g, m, f, S):
            return False
    if q["op"] == "argmax":
        det = [max(vals.values(), key=lambda c: (g.attr[c]["value"], c))] if vals else []
    else:
        det = sorted(c for c in vals.values() if g.attr[c]["value"] > q["thr"])
    for c in det:
        if not all(documented_claim(g, cc, S) for cc in g.contra[c]):
            return False
    contested = lambda c: any(cc in S for cc in g.contra[c])
    if q["op"] == "argmax":
        res = dict(result=g.attr[det[0]]["material"] if det else None, contested=contested(det[0]) if det else False)
    else:
        res = dict(result=frozenset(g.attr[c]["material"] for c in det),
                   contested=frozenset(g.attr[c]["material"] for c in det if contested(c)))
    return res["result"] == ref["result"] and res["contested"] == ref["contested"]
