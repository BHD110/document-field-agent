# Repository Guidelines

## Project Structure & Module Organization

`webapp/server.py` contains the FastAPI application, PP-OCRv6 ONNX inference pipeline, SQLite history, task/page/field extraction APIs, PDF rendering, local/cloud model orchestration, and export endpoints. The single-page document workbench UI and tutorial live in `webapp/static/`; runtime uploads, rendered task pages, thumbnails, annotations, settings, `history.db`, logs, and `secrets.blob` belong under `webapp/data/` and must not be committed. Top-level utilities include `bench_local_v2.py` for OmniDocBench evaluation, `gen_result_vis.py` for result panels, and `run_apple_vision.py` for the macOS comparison. Keep sample inputs, screenshots, and published result images in `assets/`. Model and runtime directories such as `ppocrv6_onnx/`, `models/`, and `tools/llama/` are downloaded artifacts and are Git-ignored.

The v1 product is a batch document field extraction workbench:

- A `task` represents one upload job; each PDF page or image becomes a `page` under that task.
- Field results are stored per page and exported at task level.
- The legacy `/ocr` route remains for compatibility, but the current UI uses `/tasks`.
- Table reconstruction UI is intentionally not part of v1.

## Build, Test, and Development Commands

Create and activate Python 3.10+ environment, then install the web application dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-webapp.txt
bash scripts/download_models.sh all
python webapp/server.py
```

The studio runs at `http://localhost:8766` by default; override with `DOC_WORKBENCH_PORT` or `PORT` when needed. `GET /health` is the quickest service check. Use `python bench_local_v2.py small` to generate and evaluate predictions for one model. Benchmarking also requires the separately populated `OmniDocBench/` tree and packages from `requirements.txt`.

For local field extraction assets:

```bash
python scripts/setup_minicpm5_sidecar.py
```

This downloads the fixed local text model (`MiniCPM5-1B-GGUF Q4_K_M`) and llama.cpp CPU sidecar files into ignored directories. Do not expose Qwen2.5, Qwen3, or other MiniCPM quantizations as user-facing local model choices unless product requirements change.

For cloud keys, generate the ignored runtime blob from environment variables:

```bash
python scripts/make_secret_blob.py
```

Supported variables are `DASHSCOPE_API_KEY` and `PADDLEOCR_VL_TOKEN`. The blob is obfuscation only, not real key security; keep keys low-quota, revocable, and rotated.

Before submitting Python changes, run:

```bash
python -m compileall webapp bench_local_v2.py gen_result_vis.py run_apple_vision.py scripts/make_secret_blob.py scripts/setup_minicpm5_sidecar.py pilotdeck/mcp_server.py
```

## Coding Style & Naming Conventions

Use four-space indentation in Python, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for module constants. Add type hints to new or changed interfaces where practical. In embedded JavaScript, follow the existing two-space indentation, `camelCase` functions, and `const` by default. No formatter or linter is configured, so preserve nearby style, keep imports understandable, and save source as UTF-8.

## Testing Guidelines

There is currently no committed automated test suite or coverage threshold. Smoke-test affected API routes and the corresponding browser flow with at least one image from `assets/` or a representative local sample. For OCR changes, compare recognized lines, box placement, field extraction, confidence status, and exports; for UI changes, check upload, history, settings, rendered HTML, export success modal, and narrow-screen behavior. Put future Python tests in `tests/` using names such as `test_server.py` and document any new test dependency.

Useful v1 smoke checks:

- `GET /info`, `/health`, `/usage`, `/tasks`
- `POST /tasks` with one image and fields
- `POST /tasks` with a PDF and verify one task with multiple pages
- `GET /tasks/{id}/export?fmt=xlsx` when fields exist
- `GET /tasks/{id}/export?fmt=xlsx` should fail when no fields exist
- `POST /templates/fields` should read the first row of the first Excel sheet
- Cloud mode should render returned HTML and save PaddleOCR-VL `layout_det_res` output images when available

## Commit & Pull Request Guidelines

Recent history uses short, imperative English subjects, sometimes with a prefix such as `fix:` or `article:`. Keep each commit focused. Pull requests should explain the user-visible effect, list validation performed, link relevant issues, and include before/after screenshots for UI changes or benchmark evidence for inference changes. Do not commit models, uploads, local history, generated predictions, runtime logs, `webapp/data/secrets.blob`, or credentials.

## Product and Security Notes

- Local mode uses PP-OCRv6 plus the fixed MiniCPM5-1B local text model through llama.cpp on CPU.
- Cloud mode uses PaddleOCR-VL-1.6 for document parsing and DashScope `qwen-plus` for field extraction.
- PilotDeck integration lives under `pilotdeck/`; the `local-document-entry` Skill must be maintained according to the system `skill-creator` workflow and validated with `quick_validate.py`.
- Cloud result HTML must be sanitized/rendered for display, while field extraction should use plain text / key-value text derived from the HTML.
- Field results are read-only in v1. Confidence below `0.7` is shown as low confidence; values below the minimum value confidence should be treated as missing.
- API keys must never appear in source, config, logs, docs, or untracked notes. Use environment variables or the ignored obfuscated blob only.
