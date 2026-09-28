#!/usr/bin/env bash
# GAYA distillation + training pipeline — run detached (survives session close):
#   hub op start name=gaya-pipeline application=bash args=[scripts/distill_pipeline.sh] persist detached
set -euo pipefail
cd "$(dirname "$0")/.."

export GAYA_TEACHER="${GAYA_TEACHER:-deepseek-v4.1-flash:cloud}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -z "${HF_TOKEN:-}" ] && [ -f .hf_token ]; then
  export HF_TOKEN="$(cat .hf_token)"
fi

# 1. Environment: venv reusing the system torch (2.12.1+cu130), idempotent.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt

# 2. Wait for the local ollama server (up to 4 min).
for i in $(seq 1 120); do
  curl -sf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "$OLLAMA_HOST/api/tags" >/dev/null || { echo "FATAL: ollama server unreachable at $OLLAMA_HOST"; exit 1; }

# 3. Teacher selftest (proves the deepseek checkpoint answers structured JSON).
.venv/bin/python -m gaya.teacher --selftest || { echo "FATAL: teacher selftest failed"; exit 1; }

# 4. Optional synthetic states via the teacher (multi-question triage schema).
if [ "${GAYA_SYNTH:-1}" = "1" ]; then
  .venv/bin/python -m gaya.teacher --collect --n 40 || echo "WARN: collect failed, continuing with public seeds"
fi

# 5. Distillation: teacher soft labels -> resumable jsonl shards.
.venv/bin/python -m gaya.teacher --generate

# 5.5 Bulk gold-only rows (dataset labels are ground truth; no teacher calls).
.venv/bin/python -m gaya.bulk_seeds

# 6. Train + calibrate.
.venv/bin/python -m gaya.train

# 7. Head-to-head vs laya on gold test splits.
.venv/bin/python benchmarks/bench_gaya_vs_laya.py || echo "WARN: bench skipped"

echo "GAYA PIPELINE COMPLETE"