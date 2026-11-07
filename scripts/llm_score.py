"""Score LLM-reader responses against the reference answers (standard library only).

    python3 scripts/llm_score.py            # reads results/llm/prompts.jsonl and responses_*.jsonl

Writes results/llm/llm_scores.csv and results/llm/table_llm.tex. Per model and condition:
  correct      answer set AND contested set equal the reference (abstentions count as wrong)
  abstain      share of abstentions
  contested    on queries with a contested reference: share where the contest was reported
  unacc_recall / unacc_prec   detection of candidates truly unaccounted for in the retained graph
  invalid_ids  share of responses citing a node id that is not in the prompt
  tokens       mean prompt tokens reported by the API
"""
import csv, glob, json, os, re
from collections import defaultdict

D = "results/llm"


def parse(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None


def as_set(x):
    if x is None:
        return set()
    if isinstance(x, str):
        return {x.strip()} if x.strip() else set()
    return {str(v).strip() for v in x}


def main():
    P = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, "prompts.jsonl"))}
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "responses_*.jsonl"))):
        model = os.path.basename(f)[len("responses_"):-len(".jsonl")]
        last = {}
        for l in open(f):
            r = json.loads(l)
            if r.get("ok") or r["id"] not in last:
                last[r["id"]] = r
        acc = defaultdict(lambda: defaultdict(list))
        per = []
        for pid, r in last.items():
            p = P[pid]
            cond = f"{p['method']} @ {int(p['budget']*100)}%" + (" + certificate" if p["certificate"] else "")
            a = parse(r.get("text")) if r.get("ok") else None
            s = acc[cond]
            s["n"].append(1)
            if a is None:
                s["parse_fail"].append(1); s["correct"].append(0); s["abstain"].append(0)
                continue
            s["parse_fail"].append(0)
            abst = bool(a.get("abstain"))
            ans, ref = as_set(a.get("answer")), as_set(p["ref_result"])
            con, refc = as_set(a.get("contested")), as_set(p["ref_contested"])
            s["abstain"].append(int(abst))
            s["correct"].append(int((not abst) and ans == ref and con == refc))
            if p["ref_contested_any"] and not abst:
                s["contested"].append(int(bool(con & refc)))
            ua, ut = as_set(a.get("unaccounted")), set(p["unaccounted_true"])
            if ut:
                s["unacc_recall"].append(len(ua & ut) / len(ut))
            if ua:
                s["unacc_prec"].append(len(ua & ut) / len(ua))
            ids = as_set(a.get("evidence_nodes"))
            s["invalid_ids"].append(int(bool(ids - set(p["aliases"]))))
            s["tokens"].append(r.get("usage", {}).get("promptTokenCount", float("nan")))
            per.append(dict(id=pid, seed=p["seed"], qid=p["qid"], level=p["cond"], op=p["op"], method=p["method"],
                            budget=p["budget"], contested_ref=int(p["ref_contested_any"]), correct=s["correct"][-1],
                            answer_ok=int((not abst) and ans == ref), abstain=int(abst),
                            unacc_true=len(ut), unacc_listed=len(ua & ut)))
        if per:
            with open(os.path.join(D, f"llm_per_prompt_{model}.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
        order = ["EPS @ 30", "EPS @ 20", "Exclusion", "Conflict", "PPR"]
        rank = lambda c: next((i for i, o in enumerate(order) if c.startswith(o)), len(order))
        for cond, s in sorted(acc.items(), key=lambda kv: (rank(kv[0]), kv[0])):
            mean = lambda k: (sum(s[k]) / len(s[k])) if s[k] else float("nan")
            rows.append(dict(model=model, condition=cond, n=len(s["n"]), correct=mean("correct"), abstain=mean("abstain"),
                             contested=mean("contested") if len(s["contested"]) >= 10 else float("nan"), unacc_recall=mean("unacc_recall"), unacc_prec=mean("unacc_prec"),
                             invalid_ids=mean("invalid_ids"), parse_fail=mean("parse_fail"), tokens=mean("tokens")))
    if not rows:
        raise SystemExit("no responses found")
    with open(os.path.join(D, "llm_scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tex = lambda t: t.replace("%", "\\%")
    f2 = lambda v: "--" if v != v else f"{v:.2f}"
    body = "\n".join(f"{r['model']} & {tex(r['condition'])} & {r['n']} & {f2(r['correct'])} & {f2(r['abstain'])} & {f2(r['contested'])} & "
                     f"{f2(r['unacc_recall'])} & {f2(r['invalid_ids'])} & {r['tokens']:.0f} \\\\" for r in rows)
    open(os.path.join(D, "table_llm.tex"), "w").write(
        "\\begin{table*}[t]\\centering\\footnotesize\n\\caption{LLM readers on retained subgraphs (100 queries, including all 52 queries "
        "with a contested reference). Correct: answer and contested flags equal the reference. Contested: share of contested-reference "
        "queries answered by the model where the contest is reported (-- if fewer than 10 such answers). Unacc.\\ recall: share of truly unaccounted candidates that the model lists. "
        "Invalid ids: responses citing a node not in the prompt. Tokens: mean prompt tokens.}\\label{tab:llm}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llccccccc}\\toprule\nModel & Condition & $n$ & Correct & Abstain & Contested & "
        "Unacc.\\ recall & Invalid ids & Tokens\\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}}\n\\end{table*}\n")
    for r in rows:
        print(f"{r['model']:28s} {r['condition']:52s} n={r['n']:3d} correct={f2(r['correct'])} abstain={f2(r['abstain'])} "
              f"contested={f2(r['contested'])} unacc_recall={f2(r['unacc_recall'])} tokens={r['tokens']:.0f}")


if __name__ == "__main__":
    main()
