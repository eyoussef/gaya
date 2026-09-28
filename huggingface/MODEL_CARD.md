---
license: mit
base_model: jhu-clsp/mmBERT-base
tags:
- distillation
- decision-model
- dual-tower
- text-classification
- calibration
datasets:
- ag_news
- banking77
- emotion
- SetFit/sst5
metrics:
- accuracy
- ece
---

# GAYA v1

**GAYA** is a non-autoregressive decision model distilled from a System 2 teacher
(`deepseek-v4.1-flash` via local Ollama). A bidirectional `mmBERT-base` encoder feeds a
dual-tower cross-attention decision head that answers **typed questions** (`choice`,
`score`, `bool`) against any input state, with calibrated probabilities from
proper-scoring training. Code: [`github.com/eyoussef/gaya`](https://github.com/eyoussef/gaya).

## Results (GB10 GPU, public gold test splits, 2026-09-28)

| Task | n | GAYA acc | LAYA (zero-shot) acc | GAYA ECE | LAYA ECE |
|---|---|---|---|---|---|
| ag_news | 1000 | 0.946 | 0.947 | 0.035 | 0.187 |
| banking77 | 500 | 0.812 | 0.380 | 0.193 | 0.510 |
| emotion | 500 | 0.890 | 0.590 | 0.045 | 0.265 |
| sst5 (ordinal) | 500 | MAE 0.608 / RPS 0.099 | MAE 0.983 / RPS 0.183 | — | — |

Latency p50: 14.7–17.1 ms on 3–6-way tasks, 53.4 ms on the 77-way banking77 task
(per-option tokens are never divided by option count — the deliberate accuracy trade).
LAYA baseline: `convaiinnovations/laya`, zero-shot, no per-domain training; GAYA is
in-domain distilled on the train splits of the same datasets.

## Usage

```python
from huggingface_hub import snapshot_download
from gaya.model import load
from gaya.schema import choice, boolean

model, tok = load(snapshot_download("eyoussef/gaya"), device="cuda")
```

Then `pip install git+https://github.com/eyoussef/gaya` for the `gaya` package and see
the repo README for the full quickstart (typed questions are defined at request time).

## Training

- 38,776 unique rows: ~11.3k teacher-labelled (elicited distribution ⊕ sampled answer,
  0.5/0.5, K=5 questions/state) + ~27.5k bulk gold-only rows across ag_news, banking77,
  emotion, sst5, bool_synth.
- Targets blend `0.7 × onehot(gold) + 0.3 × teacher` probs; cross-entropy for
  choice/bool, ranked probability score for score.
- 3 epochs (mmBERT-base, lr 1e-5 backbone / 3e-4 head, cosine + 30-step warmup), then
  per-(type × option-bucket) temperature calibration on the 1,938-row held-out split:
  choice 1–5 → 1.25, 6–20 → 1.05, >20 → 1.15; score → 1.75; bool → 1.15.

## Limitations

- In-domain: seeds and evaluation cover English public datasets plus mixed-language
  synthetic support states; unseen domains/languages are untested.
- 77+ option questions cost one option tower pass per option (banking77 p50 53 ms).
- `bool`/ordinal probabilities depend on the fitted temperatures; retrain to refit.

## License

MIT. Backbone [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT).
Distilled from `deepseek-v4.1-flash`.