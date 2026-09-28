"""GAYA trainer: distillation (soft-label proper-scoring losses) + calibration fit.

Losses:
- choice/bool: cross-entropy against teacher soft targets (log rule — strictly proper).
- score: ranked probability score (strictly proper, order-aware) + small log term.
After training, one temperature is fitted per (question type, option-count bucket)
on the held-out shard; without it the mean ECE stays ~0.4 (laya: 0.466 -> 0.081)."""
import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config
from .model import GayaModel
from .schema import BOOL, CHOICE, QTYPES, SCORE, option_texts

RUN_DIR = config.RUNS / "gaya-v1"


def load_shards():
    files = sorted(config.SHARDS.glob("train-*.jsonl"))
    rows = []
    for f in files:
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    seen, deduped = set(), []
    for r in rows:
        if r["key"] not in seen:  # killed workers can append before done.json saves
            deduped.append(r)
            seen.add(r["key"])
    rows = deduped
    if len(rows) < 50:
        raise SystemExit(f"only {len(rows)} distilled rows in {config.SHARDS}; run the teacher first")
    random.seed(13)
    random.shuffle(rows)
    nval = max(1, int(0.05 * len(rows)))
    return rows[:-nval], rows[-nval:]


def collate(batch, tok):
    states = [r["state"] for r in batch]
    opts, qtypes, groups, targets = [], [], [], []
    for r in batch:
        q = r["question"]
        texts = option_texts(q)
        opts += texts
        groups.append(len(texts))
        qtypes.append(QTYPES.index(q["type"]))
        t = torch.tensor([float(r["probs"].get(lab, 0.0)) for lab in q["criteria"]])
        t = t / t.sum().clamp(min=1e-6)
        gold = r.get("gold")
        if gold is not None and config.GOLD_MIX > 0:
            labs = list(q["criteria"])
            if gold in labs:
                onehot = torch.zeros(len(labs))
                onehot[labs.index(gold)] = 1.0
                t = config.GOLD_MIX * onehot + (1.0 - config.GOLD_MIX) * t
                t = t / t.sum().clamp(min=1e-6)
        targets.append(t)
    s = tok(states, padding=True, truncation=True, max_length=config.STATE_MAX, return_tensors="pt")
    o = tok(opts, padding=True, truncation=True, max_length=config.OPT_MAX, return_tensors="pt")
    return (s["input_ids"], s["attention_mask"], o["input_ids"], o["attention_mask"],
            qtypes, groups, targets)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    nll = rps = acc = n = 0.0
    for s_ids, s_mask, o_ids, o_mask, qtypes, groups, targets in loader:
        s_ids, s_mask, o_ids, o_mask = (s_ids.to(device), s_mask.to(device),
                                        o_ids.to(device), o_mask.to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(s_ids, s_mask, o_ids, o_mask, qtypes, groups, list(range(len(qtypes))))
        for lg, t, qi in zip(logits, targets, qtypes):
            logp = F.log_softmax(lg.float(), -1)
            nll += -(t.to(logp.device) * logp).sum().item()
            p = logp.exp()
            cs, ct = p.cumsum(0), t.to(p.device).cumsum(0)
            rps += (((cs - ct) ** 2).sum() / max(1, len(t) - 1)).item()
            acc += float(p.argmax() == t.argmax())
            n += 1
    return {"nll": nll / n, "rps": rps / n, "acc_vs_teacher": acc / n, "n": int(n)}


def fit_temperatures(model, rows, tok, device):
    """Grid-fit one temperature per (question type, option-count bucket) on held-out data."""
    model.eval()
    per_call = 16
    buckets: dict[tuple, list] = {}
    for i in range(0, len(rows), per_call):
        chunk = rows[i:i + per_call]
        s_ids, s_mask, o_ids, o_mask, qtypes, groups, targets = collate(chunk, tok)
        s_ids, s_mask, o_ids, o_mask = (s_ids.to(device), s_mask.to(device),
                                        o_ids.to(device), o_mask.to(device))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(s_ids, s_mask, o_ids, o_mask, qtypes, groups, list(range(len(qtypes))))
        for lg, t, qi in zip(logits, targets, qtypes):
            k = lg.shape[0]
            key = (QTYPES[qi], config.bucket(k))
            probs = F.softmax(lg.float(), -1).cpu()
            buckets.setdefault(key, []).append((probs, t))
    calib = {}
    for (qt, bk), pairs in buckets.items():
        best_t, best_loss = 1.0, float("inf")
        for T in [round(0.5 + 0.05 * i, 2) for i in range(41)]:
            loss = 0.0
            for p, t in pairs:
                lp = (p.clamp_min(1e-9).log() / T).log_softmax(-1)
                loss += -(t * lp).sum().item()
            if loss < best_loss:
                best_loss, best_t = loss, T
        calib[f"{qt}:{bk}"] = best_t
    return calib


def main(max_steps: int | None, epochs: int, batch_size: int, lr_head: float, lr_backbone: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, val_rows = load_shards()
    print(f"TRAIN START: {len(train_rows)} train / {len(val_rows)} val on {device}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.BACKBONE)
    model = GayaModel().to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr_backbone},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
         "lr": lr_head}], weight_decay=0.01)
    train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, tok), num_workers=2)
    val_loader = DataLoader(val_rows, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, tok))
    steps_total = max(1, epochs * len(train_loader))
    warm = max(1, config.WARMUP)

    def _lr_scale(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, steps_total - warm))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_scale)
    step = 0
    stale = 0
    best = float("inf")
    t0 = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_file = (RUN_DIR / "train_log.jsonl").open("a")
    for epoch in range(epochs):
        for s_ids, s_mask, o_ids, o_mask, qtypes, groups, targets in train_loader:
            s_ids, s_mask, o_ids, o_mask = (s_ids.to(device), s_mask.to(device),
                                            o_ids.to(device), o_mask.to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, _ = model(s_ids, s_mask, o_ids, o_mask, qtypes, groups, list(range(len(qtypes))))
            loss = 0.0
            for lg, t, qi in zip(logits, targets, qtypes):
                logp = F.log_softmax(lg.float(), -1)
                if QTYPES[qi] == SCORE:
                    p = logp.exp()
                    cs, ct = p.cumsum(0), t.to(p.device).cumsum(0)
                    loss = loss + (((cs - ct) ** 2).sum() / max(1, len(t) - 1)
                                   + 0.2 * -(t.to(p.device) * logp).sum() / len(t))
                else:
                    loss = loss + -(t.to(logp.device) * logp).sum() / len(t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                print(f"STEP {step} epoch {epoch} loss {loss.item():.4f} "
                      f"({step / max(1e-9, time.time() - t0):.2f} it/s)", flush=True)
            if max_steps and step >= max_steps:
                break
        m = evaluate(model, val_loader, device)
        log_file.write(json.dumps({"epoch": epoch, **m}) + "\n")
        log_file.flush()
        print(f"EPOCH {epoch} val {json.dumps(m)}", flush=True)
        improved = m["nll"] < best
        if improved:
            best = m["nll"]
            torch.save({"state_dict": model.state_dict(), "backbone": config.BACKBONE,
                        "calib": {}}, RUN_DIR / "model.pt")
            print("CHECKPOINT saved (best val nll)", flush=True)
        if max_steps and step >= max_steps:
            break
        stale = stale + 1 if not improved else 0
        if stale >= 2:
            print("EARLY STOP: val nll plateaued", flush=True)
            break
    calib = fit_temperatures(model, val_rows, tok, device)
    (RUN_DIR / "calib.json").write_text(json.dumps(calib, indent=2))
    print("CALIBRATION", json.dumps(calib), flush=True)
    print("TRAIN COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH)
    ap.add_argument("--lr-head", type=float, default=config.LR_HEAD)
    ap.add_argument("--lr-backbone", type=float, default=config.LR_BACKBONE)
    a = ap.parse_args()
    main(a.max_steps, a.epochs, a.batch_size, a.lr_head, a.lr_backbone)