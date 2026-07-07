---
name: local-document-entry
description: Use when PilotDeck needs to batch process local PDFs, images, scanned documents, or folders through the document workbench; extract user-specified fields; export Excel; and produce missing-field or low-confidence review reports for office document entry workflows.
---

# Local Document Entry

## Workflow

1. Confirm the user gave explicit local file or folder paths and the target fields.
2. Start or verify the document workbench service before calling tools.
3. Use the MCP tools in this order:
   - `create_document_task`
   - `get_task_status`
   - `list_task_pages` when page-level evidence is needed
   - `export_task_excel`
   - `export_task_report`
4. Report the Excel path, review report path, missing fields, and low-confidence fields.

## Operating rules

- Prefer `mode="local"` for privacy and contest demos; this uses PP-OCRv6 plus MiniCPM5-1B.
- Use cloud mode only when the user explicitly asks for cloud processing.
- Treat confidence below `0.7` as a review item, not a final failure.
- Do not edit extracted field results manually; rerun the task or ask for corrected fields when needed.
- If a source path cannot be read, stop and tell the user which path failed.

## References

- Read `references/api.md` when wiring or troubleshooting MCP/HTTP calls.
- Read `references/examples.md` when preparing a demo prompt, field template, or contest write-up.
