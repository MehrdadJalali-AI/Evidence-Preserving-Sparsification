"""Resumable experiment runner. Usage:
    python -m eps_audit.experiment --config configs/qmof_main.json --seed 1
Writes <out>/seed_<s>.csv (appended per query; rerunning resumes after the last completed
query), <out>/queries_seed_<s>.json and <out>/meta_seed_<s>.json."""
import argparse, csv, json, os, time
from .data import load_table, functionals, sha256, synthetic_table
from .graph import build_graph, reproduce_predictions
from .units import make_queries, neighbourhood, audit_units, justification
from .select import budget, eps, ppr, random_nodes, degree, type_aware, conflict_free, exclusion_aware_closure, eps_cost_first, _ppr_scores
from .oracle import ilp_max_units
from .readers import conservative
from .metrics import evaluate

METHODS = ["Full neighbourhood", "EPS", "EPS (cost-first order)", "Conflict-free (CONSIST-inspired)", "Condition-aware closure",
           "Exclusion-aware closure",
           "Type-aware closure", "PPR relevance", "Degree", "Random",
           "EPS-noCond", "EPS-noQual", "EPS-noProv", "EPS-noContra", "EPS-noExcl"]


def run_methods(g, q, N, frac, units, scores, seed):
    k = budget(q, N, frac)
    return k, {
        "Full neighbourhood": N,
        "EPS": eps(g, q, k, units=units),
        "EPS (cost-first order)": eps_cost_first(g, q, k, units=units),
        "Conflict-free (CONSIST-inspired)": conflict_free(g, q, k, units=units),
        "Condition-aware closure": type_aware(g, q, N, k, scores, condition_aware=True),
        "Exclusion-aware closure": exclusion_aware_closure(g, q, N, k, scores),
        "Type-aware closure": type_aware(g, q, N, k, scores),
        "PPR relevance": ppr(g, q, N, k, scores),
        "Degree": degree(g, q, N, k),
        "Random": random_nodes(g, q, N, k, seed * 100003 + q["qid"]),
        "EPS-noCond": eps(g, q, k, drop=("K",), units=units),
        "EPS-noQual": eps(g, q, k, drop=("Q",), units=units),
        "EPS-noProv": eps(g, q, k, drop=("E", "P", "D"), units=units),
        "EPS-noContra": eps(g, q, k, drop=("contra",), units=units),
        "EPS-noExcl": eps(g, q, k, drop=("X",), units=units),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_queries", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    os.makedirs(cfg["out"], exist_ok=True)
    if cfg.get("data") == "synthetic":
        df, sha = synthetic_table(cfg.get("n_synthetic", 60), seed=0), "synthetic"
    else:
        sha = sha256(cfg["data"])
        if cfg.get("sha256") and sha != cfg["sha256"]:
            raise SystemExit(f"checksum mismatch: {sha} != {cfg['sha256']}")
        df = load_table(cfg["data"])
    funcs = functionals(df)
    g = build_graph(df, funcs, cfg["target"], cfg.get("snapshot_id", "QMOF"), sha, a.seed, cfg["contra_ev"])
    conds = [f for f in cfg["conditions"] if f in funcs]
    nq = a.max_queries or cfg["n_queries"]
    qs = make_queries(df["qmof_id"].astype(str).tolist(), conds, nq, cfg["n_candidates"], a.seed, cfg["threshold"])
    json.dump(qs, open(os.path.join(cfg["out"], f"queries_seed_{a.seed}.json"), "w"))
    meta = dict(seed=a.seed, sha256=sha, n_materials=len(df), n_nodes=len(g.ntype), n_edges=len(g.erel),
                n_contradictions=g.meta["n_contradictions"], pred_mae_eV=g.meta["pred_mae"], pred_r2=g.meta["pred_r2"],
                fold_sizes=[len(f) for f in g.meta["folds"]], model_params=g.meta["model_params"],
                prediction_reproduction_max_abs_diff=reproduce_predictions(g), config=cfg)
    json.dump(meta, open(os.path.join(cfg["out"], f"meta_seed_{a.seed}.json"), "w"), indent=2)
    path = os.path.join(cfg["out"], f"seed_{a.seed}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {int(r["qid"]) for r in csv.DictReader(fh)}
    writer, fh, t0 = None, open(path, "a", newline=""), time.time()
    for q in qs:
        if q["qid"] in done:
            continue
        units = audit_units(g, q)
        J = justification(units)
        ref = conservative(g, q, set(g.ntype))
        N = neighbourhood(g, q)
        assert J <= N, "audit units must lie inside the query neighbourhood"
        scores = _ppr_scores(g, q, N)
        rows = []
        for frac in cfg["budgets"]:
            k, res = run_methods(g, q, N, frac, units, scores, a.seed)
            for name in METHODS:
                S = res[name]
                if name != "Full neighbourhood":
                    assert len(S) <= k, (name, len(S), k)
                row = dict(seed=a.seed, qid=q["qid"], op=q["op"], cond=q["cond"], budget=frac, k=k,
                           N=len(N), J=len(J), method=name, **evaluate(g, q, S, units, ref))
                row["ilp_units"] = ilp_max_units(q, units, k) if (name in ("EPS", "EPS (cost-first order)") and frac <= cfg.get("ilp_max_budget", 0.2)) else ""
                row["contested_ref"] = int(bool(ref["contested"]))
                rows.append(row)
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if not done:
                writer.writeheader()
        writer.writerows(rows)
        fh.flush()
    fh.close()
    print(f"seed {a.seed}: done in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
