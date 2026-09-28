"""GAYA router: Unicode script detection + checkpoint dispatch with LRU residency.

Parity with laya's Router (its headline feature), plus the structural advantage:
each option is encoded as an independent short sequence (per-option budget is
OPT_MAX tokens, never divided by option count), so high-cardinality choice
questions (77+ options) do not dilute the way laya's marker sequence does.
"""
import json
import os
import time
from collections import OrderedDict

import torch

from . import config

_SCRIPT_RANGES = (
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x30FF, "kana"),
    (0xAC00, 0xD7AF, "hangul"),
    (0x4E00, 0x9FFF, "han"),
)


def detect_script(text: str) -> str:
    """Dominant Unicode script by alpha-char count. Pure Python, sub-millisecond."""
    counts: dict = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        script = "latin"
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                script = name
                break
        counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else "latin"


class Router:
    """Script-aware dispatch over registered gaya checkpoints, LRU-resident."""

    def __init__(self, checkpoints=None, device=None, max_loaded=2):
        self._checkpoints = OrderedDict(checkpoints or {})
        self._loaded: OrderedDict = OrderedDict()
        self.max_loaded = max_loaded
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for name, path in _auto_discovered().items():
            self._checkpoints.setdefault(name, path)

    def register(self, name: str, path: str):
        self._checkpoints[name] = path
        self._checkpoints.move_to_end(name)

    def list_checkpoints(self):
        return list(self._checkpoints)

    def resident(self):
        return list(self._loaded)

    def unload(self, name: str | None = None):
        if name is None:
            self._loaded.clear()
        else:
            self._loaded.pop(name, None)

    def attach(self, name: str, model):
        self._loaded[name] = model
        self._loaded.move_to_end(name)

    def _load(self, name: str):
        from pathlib import Path
        from transformers import AutoTokenizer
        from .model import GayaModel

        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        path = self._checkpoints[name]
        ckpt_file = Path(path) / "model.pt" if os.path.isdir(path) else None
        if ckpt_file is not None and ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            model = GayaModel(ckpt.get("backbone") or config.BACKBONE)
            model.load_state_dict(ckpt["state_dict"])
        else:
            model = GayaModel(config.BACKBONE)
        model = model.to(self.device).eval()
        self._loaded[name] = model
        self._loaded.move_to_end(name)
        while len(self._loaded) > self.max_loaded:
            oldest = next(iter(self._loaded))
            if oldest == name:
                break
            del self._loaded[oldest]
            print(f"ROUTER evicted {oldest} (LRU)", flush=True)
        return model

    def predict(self, state, questions):
        t0 = time.perf_counter()
        s_text = json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state
        script = detect_script(s_text)
        names = list(self._checkpoints)
        if len(names) > 1 and script != "latin":
            # Non-Latin script -> prefer a multilingual-backed checkpoint.
            multilingual = [n for n in names if "multi" in n or "multilingual" in n]
            if multilingual:
                name = multilingual[-1]
            else:
                name = names[-1]
        else:
            name = names[0]
        model = self._load(name)
        answers = model.predict(state, questions)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "answers": answers,
            "routing": {
                "model": name,
                "script": script,
                "latency_ms": latency_ms,
                "reason": f"script={script}; checkpoint={name}",
            },
        }


def _auto_discovered():
    """Auto-register every trained checkpoint under runs/."""
    out = {}
    runs = config.RUNS
    if not runs.is_dir():
        return out
    for d in sorted(runs.iterdir()):
        if d.is_dir() and (d / "model.pt").exists():
            out[d.name] = str(d)
    return out