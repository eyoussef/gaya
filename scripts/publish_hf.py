"""Publish the gaya-v1 checkpoint + model card to the Hugging Face Hub.

Uploads huggingface/MODEL_CARD.md (as README.md), runs/gaya-v1/calib.json and
runs/gaya-v1/model.pt to the model repo. Token: HF_TOKEN env or .hf_token file.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi

REPO_ID = os.environ.get("HF_REPO", "eyoussef/gaya")


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = ROOT / ".hf_token"
    return f.read_text().strip() if f.exists() else None


def main() -> None:
    api = HfApi(token=token())
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=ROOT / "huggingface" / "MODEL_CARD.md",
                    path_in_repo="README.md", repo_id=REPO_ID,
                    commit_message="Model card: gaya v1")
    api.upload_file(path_or_fileobj=ROOT / "runs" / "gaya-v1" / "calib.json",
                    path_in_repo="calib.json", repo_id=REPO_ID,
                    commit_message="Per-(type, bucket) calibration temperatures")
    api.upload_file(path_or_fileobj=ROOT / "runs" / "gaya-v1" / "model.pt",
                    path_in_repo="model.pt", repo_id=REPO_ID,
                    commit_message="gaya v1 weights (mmBERT-base dual-tower)")
    print(f"UPLOADED https://huggingface.co/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()