"""Small runnable example: build an evidence graph from a synthetic table, run EPS on one
query, print the certificate, both readers' answers and the executable replay result."""
from eps_audit.data import synthetic_table, functionals
from eps_audit.graph import build_graph
from eps_audit.units import make_queries, neighbourhood, audit_units, justification, certificate
from eps_audit.select import budget, eps, ppr
from eps_audit.readers import permissive, conservative
from eps_audit.replay import replay

df = synthetic_table(40, seed=1)
g = build_graph(df, functionals(df), "hse06", "synthetic", "n/a", seed=1, contra_ev=0.8)
q = make_queries(df["qmof_id"].tolist(), ["hse06"], 1, 6, seed=4)[0]
units = audit_units(g, q); N = neighbourhood(g, q)
ref = conservative(g, q, set(g.ntype))
print("query:", q["op"], q["cond"], q["cands"])
print("neighbourhood:", len(N), "nodes; minimum audit-sufficient set:", len(justification(units)), "nodes")
for frac in (0.15, 0.5):
    k = budget(q, N, frac)
    for name, S in (("EPS", eps(g, q, k, units=units)), ("PPR", ppr(g, q, N, k))):
        c = certificate(units, S)
        print(f"k={k:3d} {name}: |S|={len(S):3d} audit_sufficient={c['audit_sufficient']} unmet={c['unmet']} "
              f"conservative={conservative(g, q, S)['result']} permissive={permissive(g, q, S)['result']} "
              f"replay_ok={replay(g, q, S, units) == {k2: ref[k2] for k2 in ('result', 'contested')}}")
print("reference answer:", ref["result"], "contested:", ref["contested"])
