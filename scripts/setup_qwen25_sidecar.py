"""Download the fixed local field-extraction sidecar assets.

This fetches:
- Qwen2.5-0.5B-Instruct GGUF Q5_K_M
- llama.cpp Windows x64 release assets containing llama-server.exe

The downloaded artifacts are ignored by git and are meant for local packaging.
"""
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LLAMA_DIR = ROOT / "tools" / "llama"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q5_k_m.gguf"
MODEL_PATH = MODEL_DIR / "qwen2.5-0.5b-instruct-q5_k_m.gguf"
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


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


def find_llama_asset() -> tuple[str, str]:
    with urllib.request.urlopen(GITHUB_API) as response:
        release = json.loads(response.read().decode("utf-8"))
    assets = release.get("assets", [])
    candidates = []
    for asset in assets:
        name = asset.get("name", "").lower()
        url = asset.get("browser_download_url")
        if url and name.endswith(".zip") and "win" in name and ("x64" in name or "x86_64" in name):
            score = 0
            if "avx2" in name:
                score += 3
            if "cpu" in name:
                score += 2
            if "cublas" in name or "cuda" in name:
                score -= 5
            candidates.append((score, asset.get("name", ""), url))
    if not candidates:
        raise RuntimeError("No Windows x64 llama.cpp release zip found.")
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def install_llama_server():
    exe = LLAMA_DIR / "llama-server.exe"
    if exe.exists():
        print(f"Exists: {exe}")
        return
    name, url = find_llama_asset()
    zip_path = LLAMA_DIR / name
    download(url, zip_path)
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            lower = member.lower()
            if lower.endswith(".exe") or lower.endswith(".dll"):
                target = LLAMA_DIR / Path(member).name
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    if not exe.exists():
        raise RuntimeError("llama-server.exe was not found in the downloaded release.")
    print(f"Installed {exe}")


def main():
    download(MODEL_URL, MODEL_PATH)
    install_llama_server()
    print("Local Qwen2.5 sidecar assets are ready.")


if __name__ == "__main__":
    main()
