"""Seed tasks for distillation: public datasets + schemas covering every primitive
and the high-cardinality stress case laya fails (Banking77).

Hang-proofing for unattended runs: child processes materialise the HF caches
online under a hard subprocess timeout; the parent loads strictly offline, so a
stalled CDN connection can freeze a child (skipped) but never the pipeline."""
import json
import os
import random
import subprocess
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # parent never touches the network

from . import config
from .schema import boolean, choice, score

MATERIALISE = {
    "ag_news": "from datasets import load_dataset; load_dataset('ag_news')",
    "banking77": "from datasets import load_dataset; load_dataset('banking77')",
    "emotion": "from datasets import load_dataset; load_dataset('emotion')",
    "sst5": "from datasets import load_dataset; load_dataset('SetFit/sst5')",
}


def _materialise(name: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-c", MATERIALISE[name]],
        env={**os.environ, "HF_HUB_OFFLINE": "0", "HF_HUB_DOWNLOAD_TIMEOUT": "30"},
        timeout=int(os.environ.get("GAYA_SEED_TIMEOUT", "240")),
        capture_output=True)
    return r.returncode == 0


def load_seeds():
    from datasets import load_dataset

    needs = {name for name, n in config.SEEDS.items() if n > 0}
    if "bool_synth" in needs:
        needs.discard("bool_synth")
        needs.add("ag_news")
    for name in sorted(needs):
        if not _materialise(name):
            print(f"SEED WARN: {name} not materialised; skipping that seed", flush=True)

    items = []

    def take(name, build):
        n = config.SEEDS.get(name, 0)
        if n <= 0:
            return
        try:
            items.extend(build(n)[:n])
        except Exception as e:  # noqa: BLE001
            print(f"SEED WARN: {name} unavailable ({e}); skipping", flush=True)

    take("ag_news", lambda n: [
        {"state": r["text"], "questions": [choice(
            "topic", "Which category does this article belong to?", {
                "world": "international politics, conflicts, diplomacy",
                "sports": "athletes, games, matches, tournaments",
                "business": "companies, markets, finance, trade",
                "sci_tech": "science, technology, computing, inventions"})],
         "source": "ag_news",
         "gold": ["world", "sports", "business", "sci_tech"][r["label"]]}
        for r in load_dataset("ag_news", split=f"train[:{n}]")])

    take("banking77", lambda n: (lambda ds, names: [
        {"state": r["text"], "questions": [choice(
            "intent", "Which customer-support intent does this message express?",
                {nm: nm.replace("_", " ") for nm in names})],
         "source": "banking77",
         "gold": names[r["label"]]}
        for r in ds])(load_dataset("banking77", split=f"train[:{n}]"),
                      load_dataset("banking77", split="train[:1]").features["label"].names))

    take("emotion", lambda n: (lambda ds, names: [
        {"state": r["text"], "questions": [choice(
            "emotion", "Which emotion does this text express?", {nm: nm for nm in names})],
         "source": "emotion", "gold": names[r["label"]]}
        for r in ds])(load_dataset("emotion", split=f"train[:{n}]"),
                      load_dataset("emotion", split="train[:1]").features["label"].names))

    take("sst5", lambda n: [
        {"state": r["text"], "questions": [score(
            "sentiment", "How positive is the sentiment of this sentence?",
            ["very negative", "negative", "neutral", "positive", "very positive"])],
         "source": "sst5", "gold": str(r["label"])}
        for r in load_dataset("SetFit/sst5", split=f"train[:{n}]")])

    bool_templates = [
        (("is_sports", "Is this article primarily about sports?"), 1),
        (("is_tech", "Is this article primarily about science or technology?"), 3),
        (("is_business", "Is this article primarily about business or finance?"), 2),
    ]
    take("bool_synth", lambda n: [
        {"state": r["text"], "questions": [boolean(*bool_templates[i % 3][0])],
         "source": "bool_synth",
         "gold": "A" if r["label"] == bool_templates[i % 3][1] else "B"}
        for i, r in enumerate(load_dataset("ag_news", split=f"train[-{n}:]") )])

    synth = config.DATA / "synthetic_states.jsonl"
    if synth.exists():
        triage = [
            choice("department", "Which department should handle this?",
                   {"billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, system errors",
                    "other": "everything else"}),
            score("urgency", "How urgent is this?", ["not urgent", "soon", "blocking"]),
            boolean("churn", "Does the user threaten to cancel or leave?"),
        ]
        for line in synth.read_text().splitlines():
            if line.strip():
                items.append({"state": json.loads(line)["state"],
                              "questions": triage, "source": "synthetic"})
    random.seed(7)
    random.shuffle(items)
    return items