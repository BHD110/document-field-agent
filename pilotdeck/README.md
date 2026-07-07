# PilotDeck 本地文档录入 Agent

这是文档工作台的 Agent 适配层：PilotDeck 负责调度，文档工作台负责本地 OCR、字段抽取、置信度校验和 Excel 导出。

## 运行

1. 准备 MiniCPM5-1B 本地 sidecar：

   ```bash
   python scripts/setup_minicpm5_sidecar.py
   ```

2. 启动文档工作台：

   ```bash
   python webapp/server.py
   ```

3. 在 PilotDeck 中接入 MCP 服务：

   ```bash
   python pilotdeck/mcp_server.py
   ```

   如果文档工作台不是默认端口：

   ```bash
   DOC_WORKBENCH_URL=http://127.0.0.1:8767 python pilotdeck/mcp_server.py
   ```

## 核心链路

```text
PilotDeck WorkSpace
  → local-document-entry Skill
  → MCP tool calls
  → 文档工作台 FastAPI
  → OCR/VL 解析
  → 字段候选召回 + 表格结构判断
  → MiniCPM5-1B 裁决
  → 置信度校验
  → Excel + Markdown 复核报告
```

## 设计思路

不是依赖超大模型硬抽取，而是用工程化流程把 1B 本地小模型放在“裁决”位置：OCR/VL 先解析，规则和版面结构先召回候选，小模型只做字段选择与缺失判断，最后输出可复核的 Excel。
