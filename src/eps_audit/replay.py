"""Executable answer replay. From S alone, the checker re-executes the recorded derivations:
for each candidate it requires either a complete support unit (claim, evidence, condition,
qualification, snapshot provenance, extraction derivation) or a complete exclusion unit;
it then applies the query operation to the replayed values and contested flags and returns
the recomputed answer. Replay succeeds iff every candidate is accounted for and the
recomputed answer equals the reference answer. This replays the ANSWER from recorded
values; it does not re-run DFT calculations (prediction reproduction is checked separately
in graph.reproduce_predictions)."""
from .units import audit_units


def replay(g, q, S, units=None):
    units = units if units is not None else audit_units(g, q)
    f = q["cond"]
    vals = {}
    for m in q["cands"]:
        sup = [u for u in units if u["kind"] == "support" and u["cand"] == m]
        exc = [u for u in units if u["kind"] == "exclusion" and u["cand"] == m]
        if sup and sup[0]["nodes"] <= S:
            vals[m] = sup[0]["value"]
        elif exc and exc[0]["nodes"] <= S:
            continue
        else:
            return None
    contested = lambda m: any(u["nodes"] <= S for u in units if u["kind"] == "contradiction" and u["cand"] == m)
    if q["op"] == "argmax":
        if not vals:
            return dict(result=None, contested=False)
        best = max(vals, key=lambda m: (vals[m], f"C:{m}:{f}"))
        return dict(result=best, contested=contested(best))
    sel = {m for m, v in vals.items() if v > q["thr"]}
    return dict(result=frozenset(sel), contested=frozenset(m for m in sel if contested(m)))
