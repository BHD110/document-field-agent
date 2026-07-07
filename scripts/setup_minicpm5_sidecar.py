"""Download the fixed local field-extraction sidecar assets.

This fetches:
- OpenBMB MiniCPM5-1B GGUF Q4_K_M
- llama.cpp release assets containing llama-server

The downloaded artifacts are ignored by git and are meant for local packaging.
"""
import json
import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LLAMA_DIR = ROOT / "tools" / "llama"
MODEL_URL = "https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / "minicpm5-1b-q4_k_m.gguf"
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
    want_windows = os.name == "nt"
    for asset in assets:
        name = asset.get("name", "")
        lower = name.lower()
        url = asset.get("browser_download_url")
        if not url:
            continue
        if want_windows:
            if not (lower.endswith(".zip") and "win" in lower and ("x64" in lower or "x86_64" in lower)):
                continue
        else:
            if not (lower.endswith((".tar.gz", ".tgz", ".zip")) and ("linux" in lower or "ubuntu" in lower) and ("x64" in lower or "x86_64" in lower)):
                continue
        score = 0
        if "avx2" in lower:
            score += 3
        if "cpu" in lower:
            score += 2
        if "cublas" in lower or "cuda" in lower:
            score -= 5
        candidates.append((score, name, url))
    if not candidates:
        raise RuntimeError("No suitable llama.cpp release archive found.")
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def install_llama_server():
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    exe = LLAMA_DIR / exe_name
    if exe.exists():
        print(f"Exists: {exe}")
        return
    name, url = find_llama_asset()
    archive_path = LLAMA_DIR / name
    download(url, archive_path)
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                lower = member.lower()
                if lower.endswith((".exe", ".dll", "/llama-server", "/llama-server.exe")):
                    target = LLAMA_DIR / Path(member).name
                    with zf.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        import tarfile

        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                lower = member.name.lower()
                if lower.endswith((".so", "/llama-server")) and member.isfile():
                    target = LLAMA_DIR / Path(member.name).name
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    if not exe.exists():
        raise RuntimeError(f"{exe_name} was not found in the downloaded release.")
    if os.name != "nt":
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed {exe}")


def main():
    download(MODEL_URL, MODEL_PATH)
    install_llama_server()
    print("Local MiniCPM5-1B sidecar assets are ready.")


if __name__ == "__main__":
    main()
