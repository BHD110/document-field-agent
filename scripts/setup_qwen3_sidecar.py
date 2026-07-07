"""Download the optional Qwen3-0.6B GGUF asset for local A/B testing.

The product default remains Qwen2.5-0.5B. This script only prepares an
experimental model that can be selected by setting:

    DOC_WORKBENCH_LOCAL_LLM_MODEL_PATH=models/qwen3-0.6b-q8_0.gguf
    DOC_WORKBENCH_LOCAL_LLM_DISPLAY_NAME=Qwen3-0.6B-GGUF-Q8_0

Downloaded artifacts are ignored by git and are meant for local experiments.
"""
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_URL = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
MODEL_PATH = MODEL_DIR / "qwen3-0.6b-q8_0.gguf"


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Exists: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(path)
    print(f"Wrote {path}")


def main():
    download(MODEL_URL, MODEL_PATH)
    print("Optional Qwen3-0.6B test model is ready.")


if __name__ == "__main__":
    main()
