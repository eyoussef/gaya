# GAYA

**GAYA** is a non-autoregressive decision model distilled from a System 2 teacher
(`deepseek-v4.1-flash` via local Ollama). A bidirectional `mmBERT-base` encoder feeds a
dual-tower cross-attention decision head that answers *typed questions* — `choice`,
`score`, `bool` — against any input state, with calibrated probabilities from
proper-scoring losses.

Weights: [`eyoussef/gaya`](https://huggingface.co/eyoussef/gaya) · Code: this repo · License: MIT

## Headline results (GB10 GPU, public gold test splits)

| Task | n | GAYA acc | LAYA acc | GAYA ECE | LAYA ECE | GAYA p50 | LAYA p50 |
|---|---|---|---|---|---|---|---|
| ag_news | 1000 | 0.946 | 0.947 | **0.035** | 0.187 | **17.1 ms** | 22.3 ms |
| banking77 | 500 | **0.812** | 0.380 | **0.193** | 0.510 | 53.4 ms | **29.0 ms** |
| emotion | 500 | **0.890** | 0.590 | **0.045** | 0.265 | **14.7 ms** | 20.1 ms |
| sst5 | 500 | MAE **0.608** / RPS **0.099** | MAE 0.983 / RPS 0.183 | | | **15.8 ms** | 20.7 ms |
| bool (latency) | 40 | — | — | | | **16.3 ms** | 22.9 ms |

Benchmark: `benchmarks/bench_gaya_vs_laya.py` (2026-09-28). LAYA =
`convaiinnovations/laya` zero-shot base English; GAYA is in-domain distilled on the
train splits of the same datasets — see the fairness note below.

## Why GAYA exists

LAYA's weak spots, fixed by construction:

- **No label dilution at high option counts.** Each option is encoded as an independent
  short sequence (per-option budget `OPT_MAX` tokens, never divided by option count) and
  scored by cross-attention against the state. LAYA's marker sequence degrades at
  Banking77's 77 intents; GAYA holds 0.812 there.
- **Ordinal `score`** is trained with the ranked probability score (strictly proper,
  order-aware) instead of an unordered softmax.
- **`bool` uses neutral anchors** (`the answer is yes: <instructions>`), so option labels
  cannot leak semantics the model should read from the state.
- **Calibration:** one temperature per (question type × option-count bucket), fitted on
  held-out data after training; without it mean ECE sits ~0.4 (LAYA: 0.510 → GAYA: 0.193
  at best on the hardest task).

## Install

```bash
git clone https://github.com/eyoussef/gaya
cd gaya
pip install -e .            # or: pip install git+https://github.com/eyoussef/gaya
```

Requires Python ≥ 3.10 and a `torch` with CUDA (CPU works too, slower).

## Quickstart

```python
from huggingface_hub import snapshot_download
from gaya.model import load
from gaya.schema import choice, boolean

model, tok = load(snapshot_download("eyoussef/gaya"), device="cuda")

state = "User: my card was charged twice for the same order. If this isn't fixed today, I'm closing my account."

answers = model.predict(state, {
    "intent": choice("intent", "Which customer-support intent does this message express?", {
        "refund": "user wants money back for a charge",
        "billing": "questions about charges or invoices",
        "technical": "product not working",
    }),
    "churn": boolean("churn", "Does the user threaten to cancel or leave?"),
}, tok, device="cuda")["answers"]

print(answers["intent"]["choice"], answers["intent"]["confidence"])  # refund, calibrated prob
print(answers["churn"]["bool"])                                       # {"yes": p, "no": p}
```

`choice`/`score`/`bool` questions are defined at request time (`gaya/schema.py`); the
answer space is part of the request, not the checkpoint. Multiple checkpoints can be
dispatched by Unicode script via `gaya.router.Router` (LRU-resident, auto-discovers
`runs/`).

## Training

End-to-end pipeline (teacher selftest → distillation → bulk gold-only seeds → train +
calibrate → benchmark):

```bash
bash scripts/distill_pipeline.sh
```

Runs detached/survives sessions when started through the job runner. All knobs are
environment variables (defaults in `gaya/config.py`): `GAYA_TEACHER`, `GAYA_TEACHERS`
(ensemble), `OLLAMA_HOST`, `GAYA_N_AGNEWS`/`GAYA_N_BANK77`/… (seed sizes),
`GAYA_K_SAMPLES` (questions per state), `GAYA_GOLD_MIX` (0.7: soft-label/gold blend),
`GAYA_EPOCHS`, `GAYA_BATCH`, `GAYA_LR_HEAD`, `GAYA_LR_BACKBONE`.

Training data (v1, 38,776 unique rows after dedupe):

| Source | Rows | Provenance |
|---|---|---|
| ag_news | 21,511 | 6,011 teacher-labelled (K=5/state) + 15,500 bulk gold-only |
| banking77 | 7,007 | 2,007 teacher + 5,000 bulk gold (77-way stress case) |
| emotion | 4,208 | 1,208 teacher + 3,000 bulk gold |
| sst5 | 4,803 | 806 teacher + 3,997 bulk gold (ordinal `score` head) |
| bool_synth | 1,193 | teacher-labelled synthetic churn-style booleans |
| synthetic triage | 120 | teacher-generated multi-question states |

Teacher labels mix an elicited full distribution with an empirical single-sample answer
(0.5/0.5); targets blend `GOLD_MIX × onehot(gold) + (1 − GOLD_MIX) × teacher` (0.7 by
default). Split: 95/5 → 1,938 held-out rows; final run: 3 epochs, val NLL 0.542,
acc-vs-teacher 0.869.

## Benchmark reproduction

```bash
pip install laya                     # laya is only needed for the comparison arm
python benchmarks/bench_gaya_vs_laya.py
python benchmarks/ag_news_errors.py 1000   # confusion matrix + top errors
```

**Fairness note:** GAYA is in-domain distilled on the train splits of the benchmarked
datasets; LAYA runs its published zero-shot checkpoints with no per-domain training.
GAYA is mmBERT-base dual-tower (1.28 GB); LAYA English is ModernBERT-large (421 M).

## Repo layout

```
gaya/            model (dual-tower head), teacher (ollama distillation), router,
                 train + calibration, seeds (data.py, bulk_seeds.py), typed questions (schema.py)
benchmarks/      head-to-head vs laya, ag_news error analysis
data/            synthetic_states.jsonl (committed); shards/ is gitignored, regenerated by the teacher
runs/            checkpoints (gitignored); gaya-v1 is on the HF Hub
scripts/         distill_pipeline.sh, publish_hf.py
```

## License & credits

MIT (see `LICENSE`). Built on [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT).
Distilled from `deepseek-v4.1-flash` (local Ollama); comparison baseline
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).