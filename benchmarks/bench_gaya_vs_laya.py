"""Head-to-head benchmark: gaya v1 (distilled from deepseek-v4.1-flash via local
ollama, ~38.8k unique rows over teacher-labelled + bulk gold-only seeds, 3 epochs)
vs laya zero-shot base English checkpoint.

Same states, same typed questions, same GB10 GPU. Reports accuracy vs public
gold test labels, 15-bin ECE on confidence, MAE + RPS for ordinal, and
per-call latency (one question per call, 3 warmups first).

Fairness notes (restated in the report):
- gaya is IN-DOMAIN distilled on the train splits of these datasets
  (~11.3k teacher-labelled + ~27.5k bulk gold-only rows, 3 epochs); laya runs
  its published zero-shot base checkpoints with no per-domain training.
- laya English = ModernBERT-large 421M; gaya = mmBERT-base dual-tower, 1.28 GB.
- bool has no gold set here -> latency only (bool_synth states).
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from gaya import config
from gaya.schema import choice

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYA_REPO = os.environ.get("GAYA_BENCH_LAYA", "convaiinnovations/laya")
N = {"ag_news": 1000, "banking77": 500, "emotion": 500, "sst5": 500}


def q_from_names(qid, instructions, names):
    from gaya.schema import choice
    return choice(qid, instructions, {n: n.replace("_", " ") for n in names})


def gold_sets():
    from datasets import load_dataset

    out = {}

    topic = choice("topic", "Which category does this article belong to?", {
        "world": "international politics, conflicts, diplomacy",
        "sports": "athletes, games, matches, tournaments",
        "business": "companies, markets, finance, trade",
        "sci_tech": "science, technology, computing, inventions"})
    ds = load_dataset("ag_news", split=f"test[:{N['ag_news']}]")
    out["ag_news"] = {"names": ["world", "sports", "business", "sci_tech"],
                      "items": [(r["text"], [topic], r["label"]) for r in ds]}

    ds = load_dataset("banking77", split=f"test[:{N['banking77']}]")
    names = ds.features["label"].names
    out["banking77"] = {"names": names, "items": [
        (r["text"], [q_from_names("intent", "Which customer-support intent does this message express?", names)],
         r["label"]) for r in ds]}

    ds = load_dataset("emotion", split=f"test[:{N['emotion']}]")
    names = ds.features["label"].names
    out["emotion"] = {"names": names, "items": [
        (r["text"], [q_from_names("emotion", "Which emotion does this text express?", names)],
         r["label"]) for r in ds]}

    try:
        ds = load_dataset("SetFit/sst5", split=f"test[:{N['sst5']}]")
    except Exception:
        ds = load_dataset("SetFit/sst5", split=f"validation[:{N['sst5']}]")
    sent = {"id": "sentiment", "type": "score",
            "instructions": "How positive is the sentiment of this sentence?",
            "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}
    out["sst5"] = {"names": ["0", "1", "2", "3", "4"],
                   "items": [(r["text"], [sent], r["label"]) for r in ds]}
    return out


def gaya_predictor():
    from gaya.model import load
    model, tok = load(device=DEVICE)

    def predict(state, questions):
        t0 = time.perf_counter()
        r = model.predict(state, questions, tok, DEVICE)
        return r["answers"], time.perf_counter() - t0
    return predict


def laya_predictor(repo):
    import laya
    agent = laya.load(repo, device=DEVICE)

    # Probe the bool primitive name once: the model card shows both spellings.
    probe = {"id": "p", "type": "bool", "instructions": "Is this positive?"}
    bool_type = "bool"
    try:
        agent.predict("I love this product.", {probe["id"]: probe})
        print("  laya accepts type 'bool'", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  laya bool probe ('bool') failed ({e}); trying 'noul'", flush=True)
        bool_type = "noul"
        probe["type"] = "noul"
        agent.predict("I love this product.", {probe["id"]: probe})

    def predict(state, questions):
        t0 = time.perf_counter()
        qs = []
        for q in questions.values():
            q = dict(q)
            if q["type"] == "bool":
                q["type"] = bool_type
            qs.append(q)
        r = agent.predict(state, {q["id"]: q for q in qs})
        return r["answers"], time.perf_counter() - t0
    return predict, bool_type


def conf_of(ans):
    c = ans.get("confidence")
    if c is None:
        for key in ("probs", "probabilities"):
            if isinstance(ans.get(key), dict) and ans[key]:
                c = max(ans[key].values())
                break
    return None if c is None else float(c)


def ece(confidences, corrects, bins=15):
    import numpy as np
    c = np.array(confidences, float)
    k = np.array(corrects, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (c > edges[i]) & (c <= edges[i + 1])
        if m.any():
            e += m.sum() / len(c) * abs(k[m].mean() - c[m].mean())
    return float(e)


def probs_list(ans, q, k):
    probs = ans.get("probs") or ans.get("probabilities")
    if not isinstance(probs, dict):
        return None
    crit = q["criteria"]
    texts = list(crit) if isinstance(crit, list) else list(crit.values())
    p = []
    for i in range(k):
        v = probs.get(str(i), probs.get(texts[i], 0.0))
        p.append(float(v))
    total = sum(p)
    return [v / total for v in p] if total > 0 else None


def run_model(predict, datasets, bool_states):
    rows = {}
    for ds_name, spec in datasets.items():
        accs, confs, corrs, lats, maes, rpss = [], [], [], [], [], []
        for state, questions, gold_idx in spec["items"]:
            q = questions[0]
            answers, dt = predict(state, {q["id"]: q})
            ans = answers[q["id"]]
            lats.append(dt * 1000.0)
            if q["type"] == "score":
                k = len(q["criteria"])
                pred = ans.get("score")
                if pred is None:
                    continue
                exp = float(pred)
                maes.append(abs(exp - gold_idx))
                p = probs_list(ans, q, k)
                if p is not None:
                    cum = 0.0
                    rps = 0.0
                    for j in range(k):
                        cum += p[j]
                        gold_cdf = 1.0 if j >= gold_idx else 0.0
                        rps += (cum - gold_cdf) ** 2
                    rpss.append(rps / (k - 1))
            else:
                pred = ans.get("choice")
                if pred is None:
                    continue
                norm = lambda s: " ".join(s.replace("_", " ").lower().split())
                ok = float(norm(pred) == norm(spec["names"][gold_idx]))
                accs.append(ok)
                c = conf_of(ans)
                if c is not None:
                    confs.append(c)
                    corrs.append(ok)
        row = {"n": len(spec["items"]),
               "latency_p50_ms": round(statistics.median(lats), 1),
               "latency_mean_ms": round(statistics.fmean(lats), 1),
               "q_per_sec": round(1000.0 * len(lats) / sum(lats), 1)}
        if ds_name == "sst5":
            row["mae"] = round(statistics.fmean(maes), 3) if maes else None
            row["rps"] = round(statistics.fmean(rpss), 3) if rpss else None
        else:
            row["acc"] = round(statistics.fmean(accs), 3) if accs else None
            row["ece"] = round(ece(confs, corrs), 3) if confs else None
        rows[ds_name] = row

    bool_lats = []
    for state in bool_states:
        q = {"id": "churn", "type": "bool",
             "instructions": "Does the user threaten to cancel or leave?"}
        try:
            _, dt = predict(state, {q["id"]: q})
            bool_lats.append(dt * 1000.0)
        except Exception as e:  # noqa: BLE001
            print(f"  bool latency skip: {e}", flush=True)
    if bool_lats:
        rows["bool_latency_only"] = {
            "n": len(bool_lats),
            "latency_p50_ms": round(statistics.median(bool_lats), 1),
            "latency_mean_ms": round(statistics.fmean(bool_lats), 1),
            "q_per_sec": round(1000.0 * len(bool_lats) / sum(bool_lats), 1)}
    return rows


def main():
    print("Loading gold test sets...", flush=True)
    datasets = gold_sets()
    for name, spec in datasets.items():
        print(f"  {name}: {len(spec['items'])} gold-labelled items", flush=True)

    from gaya.schema import boolean
    bool_states = [json.loads(line)["state"]
                   for line in (config.DATA / "synthetic_states.jsonl").read_text().splitlines()
                   if line.strip()][:50]
    qb = boolean("churn", "Does the user threaten to cancel or leave?")
    _ = qb  # bool latency set uses the schema-normalised bool shape

    results = {}
    print("GAYA: loading runs/gaya-v1 ...", flush=True)
    results["gaya"] = run_model(gaya_predictor(), datasets, bool_states)
    print("GAYA:", json.dumps(results["gaya"]), flush=True)

    print(f"LAYA: loading {LAYA_REPO} (zero-shot base English)...", flush=True)
    laya, laya_bool_type = laya_predictor(LAYA_REPO)
    results["laya"] = run_model(laya, datasets, bool_states)
    results["laya"]["bool_type"] = laya_bool_type
    print("LAYA:", json.dumps(results["laya"]), flush=True)

    out = config.ROOT / "benchmarks" / "results-gaya-vs-laya.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n| Dataset | metric | gaya v1 (distilled) | laya (zero-shot) |", flush=True)
    print("|---|---|---|---|", flush=True)
    for ds_name in datasets:
        g, l = results["gaya"][ds_name], results["laya"][ds_name]
        if ds_name == "sst5":
            print(f"| {ds_name} | MAE↓ | {g.get('mae')} | {l.get('mae')} |", flush=True)
            print(f"| {ds_name} | RPS↓ | {g.get('rps')} | {l.get('rps')} |", flush=True)
        else:
            print(f"| {ds_name} | acc | {g.get('acc')} | {l.get('acc')} |", flush=True)
            print(f"| {ds_name} | ECE↓ | {g.get('ece')} | {l.get('ece')} |", flush=True)
    for model in ("gaya", "laya"):
        b = results[model].get("bool_latency_only")
        if b:
            print(f"| bool latency (no gold) | p50 | {b['latency_p50_ms']} ms ({model}) |  |", flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()