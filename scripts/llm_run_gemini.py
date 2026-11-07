"""Run the LLM-reader prompts against the Gemini API (Python standard library only).

The API key is read from the environment variable GEMINI_API_KEY (never stored on disk).

    export GEMINI_API_KEY=...                      # your key
    python3 scripts/llm_run_gemini.py --list       # models your key can use
    python3 scripts/llm_run_gemini.py --models gemini-flash-latest --limit 5      # quick test
    python3 scripts/llm_run_gemini.py --models gemini-flash-latest,gemini-3.1-flash-lite

Writes results/llm/responses_<model>.jsonl (one line per prompt). The run is resumable:
rerunning skips prompts that already have a successful response. Temperature 0, JSON output.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

API = "https://generativelanguage.googleapis.com/v1beta"


def call(url, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def list_models(key):
    out = call(f"{API}/models?key={key}&pageSize=200")
    for m in out.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(m["name"].replace("models/", ""), "-", m.get("displayName", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="results/llm/prompts.jsonl")
    ap.add_argument("--out_dir", default="results/llm")
    ap.add_argument("--models", default="gemini-flash-latest")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (for a test)")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between calls (free tier: keep >= 4)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set the environment variable GEMINI_API_KEY first.")
    if a.list:
        return list_models(key)
    prompts = [json.loads(l) for l in open(a.prompts)]
    if a.limit:
        prompts = prompts[: a.limit]
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        path = os.path.join(a.out_dir, f"responses_{model}.jsonl")
        done = set()
        if os.path.exists(path):
            for l in open(path):
                r = json.loads(l)
                if r.get("ok"):
                    done.add(r["id"])
        todo = [p for p in prompts if p["id"] not in done]
        print(f"{model}: {len(done)} done, {len(todo)} to go")
        with open(path, "a") as fh:
            for i, p in enumerate(todo, 1):
                body = {"contents": [{"role": "user", "parts": [{"text": p["prompt"]}]}],
                        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
                rec, wait = dict(id=p["id"], model=model, ok=False), 30
                for attempt in range(6):
                    t0 = time.time()
                    try:
                        out = call(f"{API}/models/{model}:generateContent?key={key}", body)
                        cand = out.get("candidates", [{}])[0]
                        text = "".join(x.get("text", "") for x in cand.get("content", {}).get("parts", []))
                        rec.update(ok=True, text=text, finish=cand.get("finishReason"),
                                   usage=out.get("usageMetadata", {}), latency_s=round(time.time() - t0, 2))
                        break
                    except urllib.error.HTTPError as e:
                        msg = e.read().decode()[:300]
                        rec.update(error=f"HTTP {e.code}: {msg}")
                        if e.code in (429, 500, 503):
                            time.sleep(wait); wait *= 2; continue
                        break
                    except Exception as e:                     # network hiccup
                        rec.update(error=str(e)[:300]); time.sleep(wait); wait *= 2
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                status = "ok" if rec["ok"] else rec.get("error", "error")[:80]
                print(f"  [{i}/{len(todo)}] {p['id']}: {status}")
                if not rec["ok"] and "HTTP 4" in rec.get("error", "") and "429" not in rec.get("error", ""):
                    sys.exit(f"Stopping: {rec['error']}")
                time.sleep(a.sleep)


if __name__ == "__main__":
    main()
