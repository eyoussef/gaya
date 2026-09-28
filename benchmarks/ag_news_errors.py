"""Throwaway: ag_news error analysis for the current gaya checkpoint (first N test items)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from collections import Counter

import torch

from gaya.model import load
from gaya.schema import choice

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

model, tok = load(device=DEVICE)
q = choice("topic", "Which category does this article belong to?", {
    "world": "international politics, conflicts, diplomacy",
    "sports": "athletes, games, matches, tournaments",
    "business": "companies, markets, finance, trade",
    "sci_tech": "science, technology, computing, inventions"})
labels = ["world", "sports", "business", "sci_tech"]

from datasets import load_dataset
ds = load_dataset("ag_news", split=f"test[:{N}]")

confusion = Counter()
errors = []
for i, r in enumerate(ds):
    ans = model.predict(r["text"], {"topic": q}, tok, DEVICE)["answers"]["topic"]
    pred = labels.index(ans["choice"])
    gold = r["label"]
    confusion[(labels[gold], labels[pred])] += 1
    if pred != gold:
        errors.append({"i": i, "gold": labels[gold], "pred": labels[pred],
                       "conf": round(ans["confidence"], 3),
                       "text": r["text"][:120]})

acc = 1 - len(errors) / N
print(f"AG_NEWS n={N} acc={acc:.3f}")
for gold in labels:
    row = {p: confusion[(gold, p)] for p in labels}
    print(f"  {gold:10s} -> {row}")
print("\nTop errors (gold -> pred, model confidence):")
for e in errors[:25]:
    print(f"  #{e['i']} {e['gold']}->{e['pred']} conf={e['conf']} | {e['text']}")