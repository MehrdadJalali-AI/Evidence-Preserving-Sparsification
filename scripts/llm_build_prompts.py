"""Build LLM-reader prompts from retained subgraphs (requires data/raw/qmof.csv).

Usage:
    python scripts/llm_build_prompts.py configs/qmof_main.json results/llm/prompts.jsonl

Sample: every HSE06 query whose reference answer is contested, plus a fixed random sample of
other queries (seeded), across seeds 1-3. Conditions: EPS at 30% and 20% (with its audit
certificate), and PPR, exclusion-aware closure and conflict-free at 30% (no certificate; these
methods do not produce one). Node identifiers are anonymised (n1, n2, ...) so that the level
of theory or the DFT/ML qualification cannot be read from an identifier; materials keep their
QMOF ids. The reference answer is stored with each prompt for scoring but is never sent.
"""
import json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eps_audit.data import load_table, functionals, sha256
from eps_audit.graph import build_graph
from eps_audit.units import make_queries, neighbourhood, audit_units, certificate
from eps_audit.select import budget, eps, ppr, exclusion_aware_closure, conflict_free, _ppr_scores
from eps_audit.readers import conservative, qualified, documented_claim, documented_exclusion

LEVEL = {"hle17": "HLE17", "hse06_10hf": "HSE06* (10% HF)", "hse06": "HSE06", "pbe": "PBE"}
TYPE = {"M": "Material", "C": "Claim", "E": "Evidence", "K": "Condition", "Q": "Qualification",
        "P": "Provenance", "D": "Derivation", "X": "ExclusionRecord"}
N_OTHER = 16            # uncontested queries sampled per seed (in addition to all contested HSE06 queries)


def describe(g, n, alias):
    t, a = g.ntype[n], g.attr[n]
    if t == "M":
        return f"material {n[2:]}"
    if t == "C":
        return f"band gap = {a['value']:.2f} eV"
    if t == "E":
        return "evidence record"
    if t == "K":
        return f"level of theory = {LEVEL.get(a['level_of_theory'], a['level_of_theory'])}"
    if t == "Q":
        return "DFT-computed value" if a["kind"] == "dft" else "ML-predicted value"
    if t == "P":
        return (f"dataset snapshot ({a.get('source', '')}), SHA-256 {str(a.get('sha256'))[:12]}..."
                if n == "P:snapshot" else f"prediction model ({a.get('source', '')})")
    if t == "D":
        return a.get("op", "derivation")
    if t == "X":
        return (f"closed-world lookup: no DFT band gap at level {LEVEL.get(a['condition'], a['condition'])} "
                f"for material {a['candidate']} in the snapshot")
    return t


def serialise(g, S):
    order = sorted(S, key=lambda n: ("MCEKQXPD".index(g.ntype[n]), n))
    alias, k = {}, 0
    for n in order:
        if g.ntype[n] == "M":
            alias[n] = n[2:]
        else:
            k += 1
            alias[n] = f"n{k}"
    lines = ["NODES (id | type | content):"]
    lines += [f"{alias[n]} | {TYPE[g.ntype[n]]} | {describe(g, n, alias)}" for n in order]
    lines.append("EDGES (source --relation--> target):")
    for (u, v), rel in sorted(g.erel.items()):
        if u in S and v in S:
            lines.append(f"{alias[u]} --{rel}--> {alias[v]}")
    return "\n".join(lines), alias


def cert_text(units, S, alias):
    c = certificate(units, S)
    out = ["AUDIT CERTIFICATE produced by the reduction method:",
           f"audit_sufficient = {c['audit_sufficient']}; unmet requirements = {c['unmet'] or 'none'}"]
    for u in c["units"]:
        out.append(f"- {u['kind']} unit for material {u['cand']}: {u['status']}")
    return "\n".join(out)


def question(q):
    lv = LEVEL.get(q["cond"], q["cond"])
    cands = ", ".join(q["cands"])
    if q["op"] == "argmax":
        task = f"Which of the candidate materials has the largest band gap at level of theory {lv}?"
    else:
        task = f"Which of the candidate materials have a band gap above {q['thr']} eV at level of theory {lv}?"
    return task, cands


INSTRUCTIONS = """You are auditing an answer that must be derived ONLY from the evidence graph below (no outside knowledge).
Rules:
1. A band-gap claim may be used only if the graph shows its level of theory (a Condition node) equal to the requested level AND it is marked as a DFT-computed value (a Qualification node). Never compare values from different levels of theory, and never use ML-predicted values as answers.
2. If a usable claim has a contradicting claim (relation 'contradicts') in the graph, report it as contested.
3. A candidate is accounted for if it has a usable claim or an exclusion record for the requested level. List every candidate that is NOT accounted for.
4. If the graph does not let you answer reliably, set abstain to true.
Return ONLY a JSON object with keys:
  "abstain": true|false,
  "answer": material id (largest-band-gap question) or list of material ids (threshold question), or null,
  "contested": list of material ids in your answer whose value is contested,
  "unaccounted": list of candidate material ids not accounted for,
  "evidence_nodes": list of node ids you relied on."""


def main(cfg_path, out_path):
    cfg = json.load(open(cfg_path))
    if sha256(cfg["data"]) != cfg["sha256"]:
        raise SystemExit("checksum mismatch")
    df = load_table(cfg["data"])
    funcs = functionals(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_written = 0
    with open(out_path, "w") as fh:
        for seed in (1, 2, 3):
            g = build_graph(df, funcs, cfg["target"], cfg.get("snapshot_id", "QMOF"), cfg["sha256"], seed, cfg["contra_ev"])
            conds = [f for f in cfg["conditions"] if f in funcs]
            qs = make_queries(df["qmof_id"].astype(str).tolist(), conds, cfg["n_queries"], cfg["n_candidates"], seed, cfg["threshold"])
            refs = {q["qid"]: conservative(g, q, set(g.ntype)) for q in qs}
            contested = [q for q in qs if q["cond"] == "hse06" and refs[q["qid"]]["contested"]]
            others = [q for q in qs if q not in contested]
            random.Random(seed).shuffle(others)
            for q in contested + others[:N_OTHER]:
                ref, units, N = refs[q["qid"]], audit_units(g, q), neighbourhood(g, q)
                scores = _ppr_scores(g, q, N)
                conds_ = [("EPS", 0.3, True), ("EPS", 0.2, True), ("PPR relevance", 0.3, False),
                          ("Exclusion-aware closure", 0.3, False), ("Conflict-free (CONSIST-inspired)", 0.3, False)]
                for meth, frac, with_cert in conds_:
                    k = budget(q, N, frac)
                    S = {"EPS": lambda: eps(g, q, k, units=units),
                         "PPR relevance": lambda: ppr(g, q, N, k, scores),
                         "Exclusion-aware closure": lambda: exclusion_aware_closure(g, q, N, k, scores),
                         "Conflict-free (CONSIST-inspired)": lambda: conflict_free(g, q, k, units=units)}[meth]()
                    graph_txt, alias = serialise(g, S)
                    task, cands = question(q)
                    parts = [INSTRUCTIONS, f"QUESTION: {task}", f"CANDIDATES: {cands}", graph_txt]
                    if with_cert:
                        parts.append(cert_text(units, S, alias))
                    qa = qualified(g, q, S)
                    unacc = [m for m in q["cands"] if not (
                        (f"C:{m}:{q['cond']}" in g.ntype and documented_claim(g, f"C:{m}:{q['cond']}", S))
                        or documented_exclusion(g, m, q["cond"], S))]
                    res = ref["result"]
                    rec = dict(id=f"s{seed}-q{q['qid']}-{meth.split()[0]}-{int(frac*100)}", seed=seed, qid=q["qid"],
                               op=q["op"], cond=q["cond"], method=meth, budget=frac, certificate=with_cert,
                               n_nodes=len(S), prompt="\n\n".join(parts),
                               ref_result=(sorted(res) if isinstance(res, frozenset) else res),
                               ref_contested=(sorted(ref["contested"]) if isinstance(ref["contested"], frozenset) else
                                              ([res] if ref["contested"] else [])),
                               ref_contested_any=bool(ref["contested"]),
                               unaccounted_true=sorted(unacc), qualified_unaccounted=qa["unaccounted"],
                               aliases=sorted(alias.values()))
                    fh.write(json.dumps(rec) + "\n")
                    n_written += 1
    print(f"wrote {n_written} prompts -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
