"""MCP/CLI adapter for the local document workbench.

Default mode runs a small stdio JSON-RPC MCP server. For smoke tests, use:

    python pilotdeck/mcp_server.py --call create_document_task --args-json "{\"paths\":[\"sample.png\"],\"fields\":[\"姓名\"]}"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = os.environ.get("DOC_WORKBENCH_URL", "http://127.0.0.1:8766").rstrip("/")


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^\w.\-()]+", "_", value or "document", flags=re.UNICODE).strip("_")
    return value or "document"


def http_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except Exception:
            detail = raw
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc


def http_bytes(path: str) -> bytes:
    req = urllib.request.Request(BASE_URL + path, headers={"Accept": "*/*"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {raw}") from exc


def create_document_task(arguments: dict) -> dict:
    paths = arguments.get("paths") or arguments.get("path")
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("paths is required")
    payload = {
        "paths": paths,
        "fields": arguments.get("fields") or [],
        "mode": arguments.get("mode") or "local",
        "title": arguments.get("title") or "",
    }
    task = http_json("POST", "/agent/tasks/from-path", payload)
    return {
        "task_id": task.get("id"),
        "status": task.get("status"),
        "title": task.get("title"),
        "total_pages": task.get("total_pages"),
        "processed_pages": task.get("processed_pages"),
        "failed_pages": task.get("failed_pages"),
        "has_fields": task.get("has_fields"),
    }


def get_task_status(arguments: dict) -> dict:
    task_id = arguments.get("task_id") or arguments.get("id")
    if not task_id:
        raise ValueError("task_id is required")
    return http_json("GET", f"/agent/tasks/{urllib.parse.quote(str(task_id))}/summary")


def list_task_pages(arguments: dict) -> dict:
    task_id = arguments.get("task_id") or arguments.get("id")
    if not task_id:
        raise ValueError("task_id is required")
    return http_json("GET", f"/tasks/{urllib.parse.quote(str(task_id))}/pages")


def export_task_excel(arguments: dict) -> dict:
    task_id = arguments.get("task_id") or arguments.get("id")
    if not task_id:
        raise ValueError("task_id is required")
    output_path = arguments.get("output_path")
    if not output_path:
        output_path = str(Path.cwd() / "pilotdeck" / "exports" / f"{_safe_name(str(task_id))}.xlsx")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = http_bytes(f"/tasks/{urllib.parse.quote(str(task_id))}/export?fmt=xlsx")
    path.write_bytes(body)
    return {"task_id": task_id, "output_path": str(path), "bytes": len(body)}


def export_task_report(arguments: dict) -> dict:
    task_id = arguments.get("task_id") or arguments.get("id")
    fmt = arguments.get("fmt") or "md"
    if fmt not in ("md", "json"):
        raise ValueError("fmt must be md or json")
    if not task_id:
        raise ValueError("task_id is required")
    if fmt == "json":
        data = http_json("GET", f"/agent/tasks/{urllib.parse.quote(str(task_id))}/report?fmt=json")
        output = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        output = http_bytes(f"/agent/tasks/{urllib.parse.quote(str(task_id))}/report?fmt=md").decode("utf-8")
    output_path = arguments.get("output_path")
    result = {"task_id": task_id, "fmt": fmt, "content": output}
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        result["output_path"] = str(path)
    return result


TOOLS = {
    "create_document_task": {
        "description": "Create a local document workbench task from explicit local file or directory paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}},
                "fields": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "enum": ["local", "cloud"], "default": "local"},
                "title": {"type": "string"},
            },
            "required": ["paths"],
        },
        "handler": create_document_task,
    },
    "get_task_status": {
        "description": "Return task status, page counts, missing fields, low-confidence fields, and export links.",
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        "handler": get_task_status,
    },
    "list_task_pages": {
        "description": "List task pages with OCR text, field results, image URLs, and confidence data.",
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        "handler": list_task_pages,
    },
    "export_task_excel": {
        "description": "Export task field results to an Excel file and save it locally.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["task_id"],
        },
        "handler": export_task_excel,
    },
    "export_task_report": {
        "description": "Export a Markdown or JSON review report for a task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "fmt": {"type": "string", "enum": ["md", "json"], "default": "md"},
                "output_path": {"type": "string"},
            },
            "required": ["task_id"],
        },
        "handler": export_task_report,
    },
}


def tool_list() -> list[dict]:
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
        }
        for name, spec in TOOLS.items()
    ]


def mcp_success(result: Any) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
            }
        ],
        "isError": False,
    }


def mcp_error(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def handle_request(message: dict) -> dict | None:
    msg_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if msg_id is None:
        return None
    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "document-entry-workbench", "version": "0.1.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tool_list()}
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name not in TOOLS:
                result = mcp_error(f"Unknown tool: {name}")
            else:
                result = mcp_success(TOOLS[name]["handler"](arguments))
        else:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": msg_id, "result": mcp_error(str(exc))}


def run_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            response = handle_request(message)
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--call", choices=sorted(TOOLS))
    parser.add_argument("--args-json", default="{}")
    parser.add_argument("--args-file")
    args = parser.parse_args()
    if args.list_tools:
        print(json.dumps(tool_list(), ensure_ascii=False, indent=2))
        return
    if args.call:
        raw_args = Path(args.args_file).read_text(encoding="utf-8-sig") if args.args_file else args.args_json
        result = TOOLS[args.call]["handler"](json.loads(raw_args))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
        return
    run_stdio()


if __name__ == "__main__":
    main()
