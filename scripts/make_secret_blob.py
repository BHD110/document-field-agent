"""Create the runtime secret blob for Document Workbench.

The blob is only obfuscation for first-version distribution speed. It is not a
security boundary. Keep keys low-quota, revocable, and rotated.
"""
import base64
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "webapp" / "data" / "secrets.blob"
KEY = b"doc-workbench-v1"
NAMES = ("DASHSCOPE_API_KEY", "PADDLEOCR_VL_TOKEN")


def main():
    data = {name: os.environ.get(name, "") for name in NAMES if os.environ.get(name)}
    if not data:
        raise SystemExit("No supported secret env vars found.")
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    obfuscated = bytes(b ^ KEY[i % len(KEY)] for i, b in enumerate(raw))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(base64.b64encode(obfuscated).decode("ascii"), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
