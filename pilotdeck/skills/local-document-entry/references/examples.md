# Local Document Entry Examples

## Contest demo prompt

```text
用 $local-document-entry 处理这个目录里的教案扫描件：
D:\samples\lesson-plans

字段：
授课日期、授课班级、授课题目、所用教材、任课教师

要求：
1. 使用本地模式。
2. 导出 Excel。
3. 生成低置信度和缺失字段复核清单。
4. 最后告诉我 Excel 和报告保存在哪里。
```

## Single image prompt

```text
用 $local-document-entry 处理这张图片：
C:\Users\a1866\.codex\attachments\6da70f50-38b0-4aa8-abaf-ab305f7b3fa0\image-1.png

字段：
授课日期、授课班级、授课题目、所用教材、任课教师

请导出 Excel，并生成 Markdown 复核报告。
```

## Contest framing

This skill demonstrates PilotDeck as the orchestration layer, while the document workbench remains the local execution engine:

```text
PilotDeck WorkSpace
  → Local Document Entry Skill
  → MCP tool calls
  → PP-OCRv6 / PaddleOCR-VL parsing
  → candidate recall + table structure judgment
  → MiniCPM5-1B field decision
  → confidence review
  → Excel export
```
