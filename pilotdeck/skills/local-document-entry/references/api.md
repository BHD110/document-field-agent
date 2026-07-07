# Local Document Entry API Reference

## Service

Start the document workbench first:

```bash
python webapp/server.py
```

Use `DOC_WORKBENCH_URL` to point the MCP adapter at a non-default service, for example:

```bash
DOC_WORKBENCH_URL=http://127.0.0.1:8767 python pilotdeck/mcp_server.py
```

## MCP tools

- `create_document_task`
  - Input: `paths: string[]`, `fields?: string[]`, `mode?: "local" | "cloud"`, `title?: string`
  - Output: task id, status, page counts.
- `get_task_status`
  - Input: `task_id`
  - Output: status, page counts, missing fields, low-confidence fields, export links.
- `list_task_pages`
  - Input: `task_id`
  - Output: page OCR text, field results, confidence, evidence, image URLs.
- `export_task_excel`
  - Input: `task_id`, optional `output_path`
  - Output: saved Excel file path.
- `export_task_report`
  - Input: `task_id`, optional `fmt: "md" | "json"`, optional `output_path`
  - Output: Markdown/JSON review report and optional saved path.

## HTTP endpoints wrapped by MCP

- `POST /agent/tasks/from-path`
- `GET /agent/tasks/{task_id}/summary`
- `GET /agent/tasks/{task_id}/report?fmt=md|json`
- `GET /tasks/{task_id}/pages`
- `GET /tasks/{task_id}/export?fmt=xlsx`

## Expected field-result semantics

- `status="ok"` means the value can be used directly.
- `status="low_confidence"` or `confidence < 0.7` means the value should be reviewed.
- Empty values or `status="missing"` belong in the missing-field report.
