"""Figure: a real QMOF query shown on its query neighbourhood -- full vs EPS vs PPR at the
same budget. Usage: python scripts/fig_graph_example.py configs/qmof_main.json results/figures/fig_graph_example"""
import json, sys
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from eps_audit.data import load_table, functionals, sha256
from eps_audit.graph import build_graph
from eps_audit.units import neighbourhood, audit_units, justification
from eps_audit.select import budget, eps, ppr
from eps_audit.readers import conservative, permissive

cfg = json.load(open(sys.argv[1])); out = sys.argv[2]
df = load_table(cfg["data"]); funcs = functionals(df)
g = build_graph(df, funcs, "hse06", cfg["snapshot_id"], sha256(cfg["data"]), seed=1, contra_ev=cfg["contra_ev"])
mids = df["qmof_id"].astype(str).tolist(); rng = np.random.default_rng(11)
for _ in range(50000):
    q = dict(qid=0, op="argmax", cond="hse06", cands=[str(c) for c in rng.choice(mids, 6, replace=False)], thr=2.5)
    units = audit_units(g, q); ref = conservative(g, q, set(g.ntype))
    nexc = sum(u["kind"] == "exclusion" for u in units)
    if not ref["contested"] or nexc < 2:
        continue
    N = neighbourhood(g, q); k = max(budget(q, N, 0.3), len(justification(units) | {f"M:{m}" for m in q["cands"]}))
    Se = eps(g, q, k, units=units); Sp = ppr(g, q, N, len(Se))
    if permissive(g, q, Sp)["result"] != ref["result"] and len(N) < 120:
        break
print("query", q["cands"], "|N|", len(N), "|S|", len(Se))
H = nx.Graph(); H.add_nodes_from(N)
H.add_edges_from([(u, v) for (u, v) in g.erel if u in N and v in N])
pos = nx.spring_layout(H, pos=nx.kamada_kawai_layout(H), k=0.25, iterations=80, seed=3)
COL = dict(M="#4C566A", C="#1B6CA8", E="#2A9D8F", K="#E9A23B", Q="#8E5AC8", P="#6C757D", D="#B5651D", X="#C0392B")
SIZE = dict(M=320, C=130, E=70, K=70, Q=70, P=380, D=240, X=150)
NAME = dict(M="Material", C="Claim", E="Evidence", K="Condition", Q="Qualification", P="Provenance", D="Derivation", X="Exclusion record")

def verdict(a):
    if not a["answered"]: return "abstains"
    ok = a["result"] == ref["result"] and a["contested"] == ref["contested"]
    return ("correct" if ok else "wrong") + (", contested" if a["contested"] else ", uncontested")

def draw(ax, S, title, sub):
    con = [(u, v) for u, v in H.edges if g.erel.get((u, v), g.erel.get((v, u))) == "contradicts"]
    nx.draw_networkx_edges(H, pos, ax=ax, edgelist=[e for e in H.edges if not (e[0] in S and e[1] in S)], edge_color="#e3e3e3", width=0.5)
    nx.draw_networkx_edges(H, pos, ax=ax, edgelist=[e for e in H.edges if e[0] in S and e[1] in S and e not in con], edge_color="#777", width=0.9)
    for e in con:
        on = e[0] in S and e[1] in S
        nx.draw_networkx_edges(H, pos, ax=ax, edgelist=[e], edge_color="#C0392B" if on else "#f3c4bd", width=2.4 if on else 1.0, style="dashed")
    for t in COL:
        ng = [n for n in H if n not in S and g.ntype[n] == t]; nk = [n for n in H if n in S and g.ntype[n] == t]
        nx.draw_networkx_nodes(H, pos, nodelist=ng, ax=ax, node_color="#efefef", node_size=SIZE[t] * .65, edgecolors="#d0d0d0", linewidths=.4)
        nx.draw_networkx_nodes(H, pos, nodelist=nk, ax=ax, node_color=COL[t], node_size=SIZE[t], edgecolors="white", linewidths=.7,
                               node_shape="s" if t == "X" else "o")
    for i, m in enumerate(q["cands"]):
        x, y = pos[f"M:{m}"]
        ax.text(x, y + .065, f"MOF {chr(65+i)}", ha="center", fontsize=9, fontweight="bold",
                color="#222" if f"M:{m}" in S else "#aaa", bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.85))
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.text(0, -.03, sub, transform=ax.transAxes, fontsize=10, va="top"); ax.set_axis_off()

fig, ax = plt.subplots(1, 3, figsize=(17, 6.4))
draw(ax[0], N, "(a) Query neighbourhood", f"{len(N)} nodes")
draw(ax[1], Se, "(b) EPS", f"{len(Se)} nodes ({100*len(Se)/len(N):.0f}%) — audit-sufficient\nconservative reader: {verdict(conservative(g, q, Se))}")
draw(ax[2], Sp, "(c) PPR relevance, same size", f"{len(Sp)} nodes — not audit-sufficient\nconservative reader: {verdict(conservative(g, q, Sp))}; permissive: {verdict(permissive(g, q, Sp))}")
h = [Line2D([0], [0], marker="s" if t == "X" else "o", ls="", markerfacecolor=COL[t], markeredgecolor="white", markersize=10, label=NAME[t]) for t in COL]
h += [Line2D([0], [0], color="#C0392B", ls="--", lw=2.4, label="Contradiction"),
      Line2D([0], [0], marker="o", ls="", markerfacecolor="#efefef", markeredgecolor="#ccc", markersize=10, label="Not retained")]
fig.legend(handles=h, loc="lower center", ncol=10, fontsize=10, frameon=False)
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=170)
json.dump(dict(cands=q["cands"], N=len(N), S_eps=len(Se)), open(out + ".json", "w"))
