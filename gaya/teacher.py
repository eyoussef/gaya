"""Teacher distillation via local Ollama: elicit full answer distributions and write
resumable jsonl shards. Distillation signal = mixture of the teacher's elicited
probability vector and its self-consistency sample distribution (K samples),
averaged across every configured teacher (GAYA_TEACHERS enables ensembles)."""
import argparse
import concurrent.futures as cf
import json
import threading
import time
import urllib.request
from pathlib import Path

from . import config
from .data import load_seeds
from .schema import option_labels, question_prompt

ELICITED_PROMPT = """You label training data for a calibrated decision model. Read the STATE and answer the QUESTION.

STATE:
{state}

{qp}

Return exactly one JSON object: {{"probabilities": {{{{...}}}}}} mapping EVERY option label to a probability in [0,1]. Probabilities must cover all options exactly once and sum to 1.0. No other keys, no other text.

Return exactly: {{"probabilities": {{"<label>": <p>, ...}}}}"""

SAMPLED_PROMPT = """You label training data for a decision model. Read the STATE and answer the QUESTION.

STATE:
{state}

{qp}

Return exactly one JSON object: {{"answer": "<label>"}} with your single best answer."""

COLLECT_PROMPT = """Generate {n} realistic, highly varied customer-support states (emails, tickets, chat messages) across billing, technical, security and account domains, with mixed tones and mixed languages (mostly English, some Spanish/Hindi/French/German). Vary urgency from calm to angry, include some ambiguous cases.

Return exactly one JSON object: {{"states": ["<state 1>", "<state 2>", ...]}} with exactly {n} strings."""


def chat(prompt: str, model: str | None = None, temperature: float = 0.0,
         json_mode: bool = True, num_predict: int = 1024, tries: int = 4):
    body = {"model": model or config.TEACHER,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict}}
    if json_mode:
        body["format"] = "json"
    last = None
    for t in range(tries):
        req = urllib.request.Request(
            config.OLLAMA.rstrip("/") + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(json.load(r)["message"]["content"])
        except Exception as e:  # noqa: BLE001
            last = e
            if "think" in str(e).lower():
                body.pop("think", None)  # teacher cannot disable thinking; retry plain
                continue
            time.sleep(2 ** t)
    raise RuntimeError(f"ollama chat failed after {tries} tries: {last}")


def _normalise(probs, labels):
    out = {lab: float(probs.get(lab, 0.0)) for lab in labels}
    s = sum(out.values())
    if s <= 0:
        out = {lab: 1.0 / len(labels) for lab in labels}
    else:
        out = {lab: p / s for lab, p in out.items()}
    return out


def distill_question(state, q, teacher):
    labels = option_labels(q)
    el = None
    try:
        r = chat(ELICITED_PROMPT.format(state=state, qp=question_prompt(q)),
                 model=teacher, temperature=0.0,
                 num_predict=max(512, 48 * len(labels) + 128))
        el = _normalise(r.get("probabilities", {}), labels)
    except Exception:
        el = None
    counts = {lab: 1.0 for lab in labels}  # Laplace smoothing
    got = el is not None
    for _ in range(config.K_SAMPLES):
        try:
            r = chat(SAMPLED_PROMPT.format(state=state, qp=question_prompt(q)),
                     model=teacher, temperature=config.TEACHER_TEMP, num_predict=256)
            a = r.get("answer")
            if a in counts:
                counts[a] += 1.0
                got = True
        except Exception:
            continue
    if not got:
        return None  # teacher unreachable; leave row unlabeled so a rerun redoes it
    tot = sum(counts.values())
    emp = {lab: c / tot for lab, c in counts.items()}
    if el is None:
        return emp
    return {lab: 0.5 * el[lab] + 0.5 * emp[lab] for lab in labels}


def item_key(item, qi):
    import hashlib
    h = hashlib.sha1()
    h.update(item["state"].encode())
    h.update(item["questions"][qi]["id"].encode())
    return h.hexdigest()[:16]


class _Writer:
    def __init__(self):
        self.lock = threading.Lock()
        self.fh = open(config.SHARDS / "train-000.jsonl", "a")

    def write(self, row):
        with self.lock:
            self.fh.write(json.dumps(row) + "\n")
            self.fh.flush()


_DONE_LOCK = threading.Lock()


def _load_done():
    f = config.SHARDS / "done.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_done(done):
    tmp = config.SHARDS / "done.json.tmp"
    tmp.write_text(json.dumps(done))
    tmp.replace(config.SHARDS / "done.json")


def generate(limit: int | None = None):
    items = load_seeds()
    if limit:
        items = items[:limit]
    done = _load_done()
    todo = [(it, qi) for it in items for qi in range(len(it["questions"]))
            if item_key(it, qi) not in done]
    print(f"DISTILL START: {len(todo)} questions to label "
          f"(teachers={config.TEACHERS}, K={config.K_SAMPLES}, workers={config.WORKERS})",
          flush=True)
    writer = _Writer()
    counter = {"n": 0}
    start = time.time()

    def work(pair):
        it, qi = pair
        q = it["questions"][qi]
        probs = {}
        for teacher in config.TEACHERS:
            p = distill_question(it["state"], q, teacher)
            if p is None:
                print(f"TEACHER FAIL: {item_key(it, qi)}; row skipped for rerun", flush=True)
                return
            probs[teacher] = p
        merged = {}
        for lab in option_labels(q):
            merged[lab] = sum(p[lab] for p in probs.values()) / len(probs)
        s = sum(merged.values())
        merged = {lab: p / s for lab, p in merged.items()}
        row = {"key": item_key(it, qi), "source": it["source"], "state": it["state"],
               "question": q, "probs": merged, "teachers": list(probs),
               "gold": it.get("gold")}
        writer.write(row)
        with _DONE_LOCK:
            done[row["key"]] = True
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                _save_done(done)
                rate = counter["n"] / max(1e-9, time.time() - start)
                print(f"DISTILLED {counter['n']}/{len(todo)} ({rate:.1f} q/s)", flush=True)

    with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        list(ex.map(work, todo))
    _save_done(done)
    print(f"DISTILL COMPLETE: {counter['n']} questions in {time.time()-start:.0f}s", flush=True)


def collect(n: int = 5):
    out = config.DATA / "synthetic_states.jsonl"
    have = sum(1 for line in open(out)) if out.exists() else 0
    if have >= n:
        print(f"COLLECT SKIP: {have} states already on disk", flush=True)
        return
    states = []
    while len(states) < n:
        r = chat(COLLECT_PROMPT.format(n=min(5, n - len(states))), num_predict=3072)
        for s in r.get("states", []):
            if isinstance(s, str) and len(s) > 30:
                states.append(s)
    with open(config.DATA / "synthetic_states.jsonl", "a") as f:
        for s in states:
            f.write(json.dumps({"state": s}) + "\n")
    print(f"COLLECTED {len(states)} synthetic states", flush=True)


def selftest():
    r = chat('Return exactly {"ok": true}', temperature=0.0, num_predict=32)
    assert r.get("ok") is True, f"structured output broken: {r}"
    webfetch_prompt = ('Use webfetch to open https://example.com and reply with exactly '
                       'one JSON object {"title": "<page title>"}')
    try:
        w = chat(webfetch_prompt, temperature=0.0, num_predict=256)
        print("WEBFETCH:", json.dumps(w)[:200])
    except Exception as e:  # noqa: BLE001
        print("WEBFETCH unavailable (distillation does not need it):", e)
    print("TEACHER SELFTEST OK", config.TEACHER)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.collect:
        collect(a.n)
    elif a.generate:
        generate(a.limit)
    else:
        ap.error("choose --selftest | --generate | --collect")