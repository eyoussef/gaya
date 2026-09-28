"""Bulk gold-only training rows for public seed datasets.

Gold labels come from the dataset itself, so no teacher calls are needed: targets are
one-hot over the same question schema the teacher-distilled rows use. This scales
supervision to dataset size without scaling teacher cost. Rows are appended to a
separate shard (train-001.jsonl); load_shards() dedups by key, so reruns are idempotent.

Note: if SEEDS ranges are later extended to overlap bulk ranges, teacher rows and bulk
rows share keys; first occurrence in file order wins (bulk file sorts after train-000,
so teacher rows win and get re-distilled — the safer choice).
"""
import hashlib
import json

from . import config
from .data import _materialise
from .schema import choice, score

# (dataset, hub id, seeded range end, bulk range end, builder of one row)
TOPIC = lambda: choice("topic", "Which category does this article belong to?", {
    "world": "international politics, conflicts, diplomacy",
    "sports": "athletes, games, matches, tournaments",
    "business": "companies, markets, finance, trade",
    "sci_tech": "science, technology, computing, inventions"})


def _key(state: str, qid: str) -> str:
    h = hashlib.sha1()
    h.update(state.encode())
    h.update(qid.encode())
    return h.hexdigest()[:16]


def _onehot(labels, gold):
    if gold not in labels:
        raise ValueError(f"gold {gold!r} not in question labels {list(labels)[:8]}")
    return {lab: (1.0 if lab == gold else 0.0) for lab in labels}


def _existing_keys() -> set:
    existing = set()
    for f in sorted(config.SHARDS.glob("train-*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line)["key"])
    return existing


_TEMPLATE_TARGETS = {"Is this article primarily about sports?": 1,
                     "Is this article primarily about science or technology?": 3,
                     "Is this article primarily about business or finance?": 2}


def _labmaps():
    """text -> (label index, feature names) per dataset for gold backfill."""
    from datasets import load_dataset

    def reg(maps, key, ds):
        try:
            feat = ds.features["label"]
            names = feat.names if hasattr(feat, "names") else None
        except Exception:
            names = None
        maps[key] = ({r["text"]: r["label"] for r in ds}, names)

    maps = {}
    specs = [("ag", "ag_news", "train[:21500]"), ("ag_tail", "ag_news", "train[-800:]"),
             ("b77", "banking77", "train[:7000]"), ("emo", "emotion", "train[:4200]"),
             ("sst5", "SetFit/sst5", "train[:4800]")]
    for key, name, split in specs:
        try:
            reg(maps, key, load_dataset(name, split=split))
        except Exception:
            maps[key] = ({}, None)
    return maps


def _repair_gold():
    """Fix legacy rows: spaced banking77 gold, and missing gold on public-dataset rows."""
    f = config.SHARDS / "train-000.jsonl"
    if not f.exists():
        return
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    changed = 0
    maps = None

    def getmaps():
        nonlocal maps
        if maps is None:
            maps = _labmaps()
        return maps

    for r in rows:
        src = r.get("source")
        labs = list(r["question"]["criteria"])
        g = r.get("gold")
        fixed = None
        if src == "banking77":
            if g is not None and g not in labs and isinstance(g, str) \
                    and g.replace(" ", "_") in labs:
                fixed = g.replace(" ", "_")
            elif g is None:
                idx, names = getmaps().get("b77", ({}, None))
                if idx.get(r["state"]) is not None and names:
                    fixed = names[idx[r["state"]]]
        if g is None and src in ("ag_news", "bool_synth", "emotion", "sst5"):
            m = getmaps().get("ag" if src == "ag_news" else
                         "ag_tail" if src == "bool_synth" else
                         "emo" if src == "emotion" else "sst5", ({}, None))
            idx = m[0].get(r["state"])
            if idx is not None:
                if src == "ag_news":
                    fixed = ["world", "sports", "business", "sci_tech"][idx]
                elif src == "bool_synth":
                    tgt = _TEMPLATE_TARGETS.get(r["question"]["instructions"])
                    if tgt is not None:
                        fixed = "A" if idx == tgt else "B"
                elif src == "emotion":
                    fixed = (m[1] or [])[idx] if m[1] else None
                elif src == "sst5":
                    fixed = str(idx)
        if fixed is not None and fixed in labs:
            r["gold"] = fixed
            changed += 1
    if changed:
        tmp = f.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(f)
    print(f"BULK REPAIR: fixed/backfilled gold for {changed} teacher rows", flush=True)


def bulk():
    from datasets import load_dataset

    _repair_gold()
    plan = [
        ("ag_news", int(config.env("GAYA_BULK_AGNEWS", "21500")), lambda ds, names: [
            {"state": r["text"], "question": TOPIC(), "gold":
             ["world", "sports", "business", "sci_tech"][r["label"]]} for r in ds]),
        ("banking77", int(config.env("GAYA_BULK_BANK77", "7000")), lambda ds, names: [
            {"state": r["text"], "question": choice(
                "intent", "Which customer-support intent does this message express?",
                {nm: nm.replace("_", " ") for nm in names}), "gold":
             names[r["label"]]} for r in ds]),
        ("emotion", int(config.env("GAYA_BULK_EMOTION", "4200")), lambda ds, names: [
            {"state": r["text"], "question": choice(
                "emotion", "Which emotion does this text express?", {nm: nm for nm in names}),
             "gold": names[r["label"]]} for r in ds]),
        ("sst5", int(config.env("GAYA_BULK_SST5", "4800")), lambda ds, names: [
            {"state": r["text"], "question": score(
                "sentiment", "How positive is the sentiment of this sentence?",
                ["very negative", "negative", "neutral", "positive", "very positive"]),
             "gold": str(r["label"])} for r in ds]),
    ]

    needs = [name for name, end, _ in plan if end > 0]
    for name in sorted(set(needs)):
        if not _materialise(name):
            print(f"BULK WARN: {name} not materialised; skipping that bulk source",
                  flush=True)
            needs.remove(name)

    existing = _existing_keys()
    qids = {"ag_news": "topic", "banking77": "intent", "emotion": "emotion",
            "sst5": "sentiment"}
    out_path = config.SHARDS / "train-001.jsonl"
    out = out_path.open("a")
    added = skipped = 0
    for name, end, build in plan:
        if end <= 0 or name not in needs:
            continue
        try:
            if name == "sst5":
                ds = load_dataset("SetFit/sst5", split=f"train[:{end}]")
                rows = build(ds, None)  # sst5 label feature is a Value, no .names
            else:
                ds = load_dataset(name, split=f"train[:{end}]")
                names = ds.features["label"].names
                rows = build(ds, names)
        except Exception as e:  # noqa: BLE001
            print(f"BULK WARN: {name} unavailable ({e}); skipping", flush=True)
            continue
        for r in rows:
            k = _key(r["state"], qids[name])
            if k in existing:
                skipped += 1
                continue
            labels = list(r["question"]["criteria"]) if isinstance(
                r["question"]["criteria"], dict) else [str(i) for i in
                                                       range(len(r["question"]["criteria"]))]
            out.write(json.dumps({
                "key": k, "source": f"{name}_bulk", "state": r["state"],
                "question": r["question"], "probs": _onehot(labels, r["gold"]),
                "teachers": ["gold"], "gold": r["gold"]}) + "\n")
            existing.add(k)
            added += 1
        print(f"BULK {name}: up to {end} rows scanned", flush=True)
    out.close()
    print(f"BULK COMPLETE: added {added}, skipped(existing) {skipped} -> {out_path}",
          flush=True)


if __name__ == "__main__":
    bulk()