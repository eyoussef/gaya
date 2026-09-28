"""GAYA architecture: bidirectional encoder + dual-tower cross-attention decision head.

Fixes laya's weak spots by construction:
- Options are encoded as independent short sequences (per-option budget is OPT_MAX
  tokens, never divided by option count) and scored by cross-attention against the
  state representation -> no label dilution at 77+ options.
- Ordinal `score` is trained with the ranked probability score (strictly proper,
  order-aware) instead of an unordered softmax.
- `bool` uses neutral anchors so labels cannot dominate the answer.
- Calibration: per-(question-type, option-count-bucket) temperatures fitted after
  training; probabilities come from proper-scoring training, not just argmax.
"""
import hashlib
import json
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .schema import BOOL, CHOICE, QTYPES, SCORE, option_labels, option_texts


def qtype_index(t: str) -> int:
    return QTYPES.index(t)


def bucket_of(k: int) -> str:
    from . import config as c
    return c.bucket(k)


class GayaModel(nn.Module):
    def __init__(self, backbone: str | None = None, n_head_layers: int = 2,
                 n_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.backbone_name = backbone or config.BACKBONE
        enc_cfg = AutoConfig.from_pretrained(self.backbone_name)
        self.encoder = AutoModel.from_pretrained(self.backbone_name)
        h = enc_cfg.hidden_size
        self.hidden = h
        self.type_emb = nn.Embedding(len(QTYPES), h)
        self.head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h, nhead=n_heads if h % n_heads == 0 else 8,
                dim_feedforward=4 * h, dropout=dropout, batch_first=True,
                norm_first=True, activation="gelu"),
            num_layers=n_head_layers)
        self.scorer = nn.Linear(h, 1)
        self.act_head = nn.Linear(h, 2)
        self.calib: Dict[str, float] = {}
        self._opt_cache: Dict[Any, Dict[str, torch.Tensor]] = {}
        self._opt_order: List[Any] = []

    def _pool(self, out, mask):
        m = mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, state_ids, state_mask, opt_ids, opt_mask, qtypes, group_sizes, state_idx):
        """One forward pass over the state tower and the (flattened) option tower.
        `state_idx[j]` = state row that question j is decided against (batched decode:
        one state, many questions). Returns per-question option logits + pooled states."""
        s = self._pool(self.encoder(input_ids=state_ids, attention_mask=state_mask).last_hidden_state,
                       state_mask)
        o = self._pool(self.encoder(input_ids=opt_ids, attention_mask=opt_mask).last_hidden_state,
                       opt_mask)
        logits, ptr = [], 0
        for i, k in enumerate(group_sizes):
            te = self.type_emb.weight[qtypes[i]]
            v = o[ptr:ptr + k] + s[state_idx[i]:state_idx[i] + 1] + te
            ptr += k
            v = self.head(v.unsqueeze(0)).squeeze(0)
            logits.append(self.scorer(v).squeeze(-1))
        return logits, s

    def _temp(self, qtype: str, k: int) -> float:
        return self.calib.get(f"{qtype}:{bucket_of(k)}", 1.0)

    @torch.no_grad()
    def predict(self, state, questions, tokenizer, device: str = "cuda") -> Dict[str, Any]:
        self.eval()
        device = device if torch.cuda.is_available() else "cpu"
        if not self.calib:
            calib_file = config.RUNS / "gaya-v1" / "calib.json"
            if calib_file.exists():
                self.calib = json.loads(calib_file.read_text())
        st = self._encode([json.dumps(state) if not isinstance(state, str) else state],
                          tokenizer, config.STATE_MAX, device)
        opts: List[str] = []
        groups, qidx = [], []
        for q in questions.values():
            texts = option_texts(q)
            opts += texts
            groups.append(len(texts))
            qidx.append(qtype_index(q["type"]))
        ck = (hashlib.sha1("\x00".join(opts).encode()).hexdigest()[:16],
              config.OPT_MAX, device)
        ot = self._opt_cache.get(ck)
        if ot is None:
            ot = self._encode(opts, tokenizer, config.OPT_MAX, device)
            self._opt_cache[ck] = ot
            self._opt_order.append(ck)
            if len(self._opt_order) > 32:
                self._opt_cache.pop(self._opt_order.pop(0), None)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
        with amp:
            logits, s = self.forward(st["input_ids"], st["attention_mask"],
                                     ot["input_ids"], ot["attention_mask"], qidx, groups,
                                     [0] * len(groups))
        answers = {}
        for (qid, q), lg in zip(questions.items(), logits):
            k = lg.shape[0]
            p = F.softmax(lg / self._temp(q["type"], k), dim=-1).float().cpu()
            labels = option_labels(q)
            if q["type"] == BOOL:
                probs = {"yes": p[0].item(), "no": p[1].item()}
                answers[qid] = {"bool": probs["yes"], "probs": probs,
                                "confidence": max(probs.values())}
            else:
                choice_label = max(range(k), key=lambda j: p[j].item())
                probs = {labels[j]: p[j].item() for j in range(k)}
                ans: Dict[str, Any] = {"choice": labels[choice_label], "probs": probs,
                                       "confidence": p[choice_label].item()}
                if q["type"] == SCORE:
                    ans["score"] = sum(j * p[j].item() for j in range(k))
                answers[qid] = ans
        act = self.act_head(s).softmax(-1)[0]
        return {"answers": answers,
                "routing": {"model": self.backbone_name, "act_probability": act[1].item()}}

    def _encode(self, texts, tokenizer, maxlen, device):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=maxlen,
                        return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}


def load(path: str | None = None, device: str | None = None):
    """Load a trained gaya checkpoint (default runs/gaya-v1) ready for predict()."""
    from pathlib import Path
    from transformers import AutoTokenizer

    path = Path(path) if path else config.RUNS / "gaya-v1"
    ckpt = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model = GayaModel(ckpt.get("backbone") or config.BACKBONE)
    model.load_state_dict(ckpt["state_dict"])
    calib_file = path / "calib.json"
    if calib_file.exists():
        model.calib = json.loads(calib_file.read_text())
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev).eval(), AutoTokenizer.from_pretrained(model.backbone_name)


def smoke(backbone: str | None = None):
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(backbone or config.BACKBONE)
    model = GayaModel(backbone).to(device).eval()
    from .schema import boolean, choice, score
    questions = {
        "department": choice("department", "Which department should handle this?",
                             {"billing": "invoices, payments, refunds",
                              "technical": "bugs, outages, system errors",
                              "other": "everything else"}),
        "urgency": score("urgency", "How urgent is this?",
                         ["not urgent", "soon", "blocking"]),
        "churn": boolean("churn", "Does the user threaten to cancel?"),
    }
    state = {"from": "user@acme.com", "body": "We were billed twice for March. Refund today or we cancel."}
    res = model.predict(state, questions, tok, device)
    print("TEACHER-FREE FORWARD PASS OK on", device)
    print(json.dumps(res["answers"], indent=2)[:600])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        smoke(a.backbone)
    else:
        ap.error("pass --selftest")