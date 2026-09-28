import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SHARDS = DATA / "shards"
RUNS = ROOT / "runs"
for _p in (DATA, SHARDS, RUNS):
    _p.mkdir(parents=True, exist_ok=True)


def env(key: str, default: str) -> str:
    return os.environ.get(key, default)


TEACHER = env("GAYA_TEACHER", "deepseek-v4.1-flash:cloud")
TEACHERS = [t.strip() for t in env("GAYA_TEACHERS", TEACHER).split(",") if t.strip()]
OLLAMA = env("OLLAMA_HOST", "http://127.0.0.1:11434")

# Dual-tower budgets: unlike laya, per-option tokens are NOT divided by option count.
BACKBONE = env("GAYA_BACKBONE", "jhu-clsp/mmBERT-base")
STATE_MAX = int(env("GAYA_STATE_MAX", "1024"))
OPT_MAX = int(env("GAYA_OPT_MAX", "128"))

K_SAMPLES = int(env("GAYA_K_SAMPLES", "5"))
TEACHER_TEMP = float(env("GAYA_TEACHER_TEMP", "0.8"))
WORKERS = int(env("GAYA_WORKERS", "8"))

SEEDS = {
    "ag_news": int(env("GAYA_N_AGNEWS", "6000")),
    "banking77": int(env("GAYA_N_BANK77", "2000")),
    "emotion": int(env("GAYA_N_EMOTION", "1200")),
    "sst5": int(env("GAYA_N_SST5", "800")),
    "bool_synth": int(env("GAYA_N_BOOL", "800")),
}

# Target blending: t = GOLD_MIX * onehot(gold) + (1 - GOLD_MIX) * teacher probs.
GOLD_MIX = float(env("GAYA_GOLD_MIX", "0.8"))
EPOCHS = int(env("GAYA_EPOCHS", "3"))
WARMUP = int(env("GAYA_WARMUP", "30"))
LR_HEAD = float(env("GAYA_LR_HEAD", "3e-4"))
LR_BACKBONE = float(env("GAYA_LR_BACKBONE", "1e-5"))
BATCH = int(env("GAYA_BATCH", "16"))

TEMP_BUCKETS = ((5, "1-5"), (20, "6-20"), (10**9, ">20"))


def bucket(k: int) -> str:
    for cap, name in TEMP_BUCKETS:
        if k <= cap:
            return name
    return ">20"