"""Aggregate per-query results: bootstrap 95% CIs (2,000 resamples of queries, seed 0),
exact McNemar tests (paired by seed and query), budget-to-95%, figure and LaTeX tables.
Usage: python -m eps_audit.analysis results/qmof_main [results/qmof_delta05 results/qmof_delta15]"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest

RNG = np.random.default_rng(0)
SHOW = ["EPS", "Conflict-free (CONSIST-inspired)", "Condition-aware closure", "Exclusion-aware closure",
        "Type-aware closure", "PPR relevance", "Degree", "Random"]
TIGHT = ["EPS", "EPS (cost-first order)", "Conflict-free (CONSIST-inspired)", "Exclusion-aware closure", "Condition-aware closure"]
HSE = ["EPS", "EPS-noContra", "Conflict-free (CONSIST-inspired)", "Exclusion-aware closure", "Condition-aware closure", "PPR relevance"]
ABL = ["EPS", "EPS-noCond", "EPS-noQual", "EPS-noProv", "EPS-noContra", "EPS-noExcl"]


def load(d):
    files = {}
    for f in sorted(glob.glob(os.path.join(d, "seed_*.csv*"))):   # prefer plain CSV over .gz copy
        files.setdefault(os.path.basename(f).split(".")[0], f)
    df = pd.concat([pd.read_csv(f) for f in files.values()])
    assert not df.duplicated(["seed", "qid", "budget", "method"]).any(), "duplicate rows"
    df["unsupported_rate"] = np.where(df.n_used_conservative > 0, df.n_unsupported_conservative / df.n_used_conservative.where(df.n_used_conservative > 0), np.nan)
    df["unsupported_rate_perm"] = np.where(df.n_used_permissive > 0, df.n_unsupported_permissive / df.n_used_permissive.where(df.n_used_permissive > 0), np.nan)
    df["contra_ret"] = np.where(df.n_contra_pairs > 0, df.n_contra_kept / df.n_contra_pairs.where(df.n_contra_pairs > 0), np.nan)
    return df


def ci(x, B=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    bs = RNG.choice(x, size=(B, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(x)


METRICS = ["correct_conservative", "answered_conservative", "correct_permissive", "audit_sufficient",
           "support_cov", "exclusion_cov", "unsupported_rate", "unsupported_rate_perm", "contra_ret",
           "replay_ok", "size", "edges", "unmet_R1", "unmet_R2", "unmet_R3", "unmet_R4", "unmet_R5", "unmet_R2x",
           "correct_claimfilter", "answered_claimfilter", "indep_audit", "qual_correct", "qual_flagged", "qual_unaccounted",
           "n_units_complete", "ilp_units", "n_units"]


def summarize(df):
    rows = []
    for (m, b), t in df.groupby(["method", "budget"]):
        r = dict(method=m, budget=b, n_queries=len(t))
        for k in METRICS:
            mu, lo, hi, n = ci(t[k]); r.update({k: mu, k + "_lo": lo, k + "_hi": hi, k + "_n": n})
        ans = t[t.answered_conservative == 1]
        r["selective_acc_conservative"] = ans.correct_conservative.mean() if len(ans) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def mcnemar(df, col, b=0.3):
    piv = df[df.budget == b].pivot_table(index=["seed", "qid"], columns="method", values=col)
    out = []
    for m in piv.columns:
        if m == "EPS":
            continue
        a, o = piv["EPS"].values, piv[m].values
        n01, n10 = int(((a == 1) & (o == 0)).sum()), int(((a == 0) & (o == 1)).sum())
        out.append(dict(metric=col, method=m, eps_only=n01, other_only=n10,
                        p=binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0))
    return pd.DataFrame(out)


CI_COLS = {"correct_conservative", "correct_permissive", "correct_claimfilter"}


def fmt(r, k, d=2):
    if pd.isna(r[k]): return "--"
    if d == 0: return f"{r[k]:.0f}"
    if k in CI_COLS: return f"{r[k]:.2f} [{r[k+'_lo']:.2f}, {r[k+'_hi']:.2f}]"
    return f"{r[k]:.2f}"


def tex_table(S, names, b, cols, heads, cap, lab):
    t = S[(S.budget == b) & S.method.isin(names)].set_index("method").loc[names].reset_index()
    body = "\n".join(r.method + " & " + " & ".join(fmt(r, c, 0 if c in ("size", "edges") else 2) for c in cols) + r" \\" for _, r in t.iterrows())
    return ("\\begin{table*}[t]\\centering\\footnotesize\n\\caption{" + cap + "}\\label{" + lab + "}\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{l" + "c" * len(cols) + "}\\toprule\nMethod & " + " & ".join(heads) + "\\\\\\midrule\n" + body +
            "\n\\bottomrule\\end{tabular}}\n\\end{table*}\n")


def figure(S, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = ["#1b6ca8", "#7b2cbf", "#264653", "#e76f51", "#2a9d8f", "#d1495b", "#edae49", "#66a182"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.7))
    panels = [("correct_conservative", "Correct (conservative reader)"), ("answered_conservative", "Answer coverage (conservative)"),
              ("audit_sufficient", "Audit-sufficient (R1–R5, R2x)"), ("correct_permissive", "Correct (permissive reader)")]
    for m, c in zip(SHOW, col):
        t = S[S.method == m].sort_values("budget")
        for a, (k, lab) in zip(ax, panels):
            a.plot(t.budget * 100, t[k], marker="o", ms=4, lw=1.7, color=c, label=m)
            a.fill_between(t.budget * 100, t[k + "_lo"], t[k + "_hi"], color=c, alpha=.15, lw=0)
            a.set_title(lab, fontsize=11.5); a.set_ylim(-0.03, 1.03); a.grid(alpha=.3)
            a.set_xlabel("Budget k (% of query neighbourhood)", fontsize=9.7); a.tick_params(labelsize=9.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=9.2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=170)


def main(argv=None):
    argv = argv or sys.argv[1:]
    d, sweeps = argv[0], argv[1:]
    df = load(d); out = os.path.join(d, "analysis"); os.makedirs(out, exist_ok=True)
    S = summarize(df); S.to_csv(os.path.join(out, "summary_ci.csv"), index=False)
    pd.concat([mcnemar(df, "correct_conservative"), mcnemar(df, "audit_sufficient")]).to_csv(os.path.join(out, "mcnemar_b030.csv"), index=False)
    S[S.correct_conservative >= .95].groupby("method").budget.min().to_csv(os.path.join(out, "budget_to_95_conservative.csv"))
    figure(S, os.path.join(out, "fig_tradeoff"))
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "meta_seed_*.json")))]
    nq = df.groupby("seed").qid.nunique().sum()
    cap = (f"Full QMOF ({metas[0]['n_materials']:,} MOFs), {len(metas)} seeds, {nq} queries, budget $k$ = 30\\% of the query neighbourhood "
           "(Full neighbourhood: unreduced). Means; bracketed: bootstrap 95\\% CI. Correct and coverage use the conservative reader; "
           "Correct (filter) uses the claim-filtering reader; Indep.\\ audit is the method-independent auditor; nodes and edges are rounded "
           "to the nearest integer. Unsupp.\\ and Claim-pair contra.\\ use the denominators defined in Section~\\ref{sec:case}; "
           "Claim-pair contra.\\ counts only retained claim-node pairs, not complete R4 units.")
    cols = ["correct_conservative", "answered_conservative", "correct_claimfilter", "audit_sufficient", "indep_audit", "replay_ok", "correct_permissive", "unsupported_rate_perm", "contra_ret", "size", "edges"]
    heads = ["Correct", "Coverage", "Correct (filter)", "Audit-suff.", "Indep.\\ audit", "Replay", "Correct (perm.)", "Unsupp.\\ (perm.)", "Claim-pair contra.", "Nodes", "Edges"]
    open(os.path.join(out, "table_main.tex"), "w").write(tex_table(S, ["Full neighbourhood"] + SHOW, 0.3, cols, heads, cap, "tab:main"))
    cols2 = ["correct_conservative", "answered_conservative", "audit_sufficient", "replay_ok", "correct_permissive",
             "unmet_R1", "unmet_R2", "unmet_R3", "unmet_R4", "unmet_R5", "unmet_R2x", "size"]
    heads2 = ["Correct", "Coverage", "Audit-suff.", "Replay", "Correct (perm.)",
              "R1 unmet", "R2 unmet", "R3 unmet", "R4 unmet", "R5 unmet", "R2x unmet", "Nodes"]
    open(os.path.join(out, "table_ablation.tex"), "w").write(tex_table(S, ABL, 1.0, cols2, heads2,
        "Ablations without budget pressure ($k$ = 100\\% of the neighbourhood). Means; bracketed: bootstrap 95\\% CI. Requirement scoring is per unit: an incomplete support unit marks all of R1, R2, R3, and R5 unmet.", "tab:abl"))
    # HSE06-stratified table (beta = 0.3): all HSE06 queries and the contested subset
    h = df[(df.cond == "hse06") & (df.budget == 0.3)]
    nh, nc = h.drop_duplicates(["seed", "qid"]).shape[0], h[h.contested_ref == 1].drop_duplicates(["seed", "qid"]).shape[0]
    lines = []
    for m in HSE:
        a_, c_ = h[h.method == m], h[(h.method == m) & (h.contested_ref == 1)]
        lines.append(f"{m} & {a_.correct_conservative.mean():.2f} & {a_.audit_sufficient.mean():.2f} & {a_.indep_audit.mean():.2f} & "
                     f"{c_.correct_conservative.mean():.2f} & {c_.audit_sufficient.mean():.2f} & {a_['size'].mean():.0f} \\\\")
    open(os.path.join(out, "table_hse06.tex"), "w").write(
        "\\begin{table}[t]\\centering\\footnotesize\n\\caption{HSE06 queries only, budget $k$ = 30\\%: the only level at which "
        f"contradictions exist. All HSE06 queries ($n={nh}$) and the subset whose reference answer is contested ($n={nc}$). "
        "Correct uses the conservative reader.}\\label{tab:hse06}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccccc}\\toprule\n"
        f" & \\multicolumn{{3}}{{c}}{{All HSE06 ($n={nh}$)}} & \\multicolumn{{2}}{{c}}{{Contested ($n={nc}$)}} & \\\\\n"
        "Method & Correct & Audit-suff. & Indep.\\ audit & Correct & Audit-suff. & Nodes\\\\\\midrule\n" + "\n".join(lines) +
        "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")
    # tight-budget table: greedy order vs exact optimum, qualified answers
    lines = []
    for b_ in (0.1, 0.2):
        for m in TIGHT:
            t = df[(df.budget == b_) & (df.method == m)]
            il = pd.to_numeric(t.ilp_units, errors="coerce")
            ratio = (t.n_units_complete / il).mean() if il.notna().any() else np.nan
            opt = (t.n_units_complete == il).mean() if il.notna().any() else np.nan
            lines.append(f"{int(b_*100)}\\% & {m} & {t.correct_conservative.mean():.2f} & {t.qual_correct.mean():.2f} & "
                         f"{t.qual_unaccounted.mean():.1f} & {t.n_units_complete.mean():.1f} / {t.n_units.mean():.1f} & "
                         + (f"{il.mean():.1f} & {ratio:.2f} & {opt:.2f}" if il.notna().any() else "-- & -- & --") + " \\\\")
        if b_ == 0.1:
            lines.append("\\midrule")
    open(os.path.join(out, "table_tight.tex"), "w").write(
        "\\begin{table*}[t]\\centering\\footnotesize\n\\caption{Tight budgets. Correct: conservative reader. Qualified: certificate-aware "
        "reader that answers from documented candidates and reports how many are unaccounted for (mean shown). Units: complete canonical units / "
        "all units. ILP: exact maximum number of complete units under the same budget (HiGHS); Ratio: greedy / ILP; Opt.: share of instances "
        "where the greedy order attains the optimum.}\\label{tab:tight}\n\\begin{tabular}{llccccccc}\\toprule\n"
        "Budget & Method & Correct & Qualified & Unaccounted & Units & ILP & Ratio & Opt.\\\\\\midrule\n" + "\n".join(lines) +
        "\n\\bottomrule\\end{tabular}\n\\end{table*}\n")
    if sweeps:
        rows = []
        for sd in [d] + sweeps:
            x = load(sd); x = x[(x.budget == 0.3) & x.method.isin(["EPS", "Conflict-free (CONSIST-inspired)", "Condition-aware closure"])]
            ms = [json.load(open(f)) for f in glob.glob(os.path.join(sd, "meta_seed_*.json"))]
            for m, t in x.groupby("method"):
                mu, lo, hi, n = ci(t.correct_conservative)
                rows.append(dict(contra_ev=ms[0]["config"]["contra_ev"], method=m, correct=mu, lo=lo, hi=hi, n=n,
                                 n_contradictions=np.mean([z["n_contradictions"] for z in ms]),
                                 audit_sufficient=t.audit_sufficient.mean()))
        pd.DataFrame(rows).sort_values(["contra_ev", "method"]).to_csv(os.path.join(out, "sensitivity.csv"), index=False)
    json.dump(dict(n_queries=int(nq), seeds=[m["seed"] for m in metas],
                   pred_mae=[m["pred_mae_eV"] for m in metas], pred_r2=[m["pred_r2"] for m in metas],
                   n_contradictions=[m["n_contradictions"] for m in metas],
                   prediction_reproduction_max_abs_diff=[m["prediction_reproduction_max_abs_diff"] for m in metas],
                   sha256=metas[0]["sha256"]), open(os.path.join(out, "run_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
