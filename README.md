# LangChain Office Agent

基于你 `offciatool` 的思路，做的 **LangChain Agent 版本**：

- Web 聊天 + 文件上传（图片/文档）
- Agent 可调用本地工具（读写文件、白名单命令、网页抓取）
- 会话自动摘要压缩（长对话降 token）
- 前端展示执行轨迹与工具调用明细
- token 统计（本轮/会话/全局）

核心差异：本项目使用 **LangChain v1 `create_agent`** 作为 Agent 主循环，而不是手写 tool loop。

## 快速启动

建议 Python 3.11/3.12（Python 3.14 下 LangChain 会有兼容性 warning）。

```bash
cd /Users/dalizhou/Desktop/langchain_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，至少填 OPENAI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

打开：

- [http://127.0.0.1:8080](http://127.0.0.1:8080)

## 主要能力

### 1) Agent + 本地工具

默认内置 7 个工具：

- `run_shell`
- `list_directory`
- `read_text_file`
- `copy_file`
- `write_text_file`
- `replace_in_file`
- `fetch_web`

安全约束：

- 命令白名单：`OFFICETOOL_ALLOWED_COMMANDS`
- 路径限制：默认仅允许 `workspace root + OFFICETOOL_EXTRA_ALLOWED_ROOTS`
- 可选完全放开：`OFFICETOOL_ALLOW_ANY_PATH=true`（仅可信环境）

### 2) 会话压缩

当历史轮次超过阈值后自动摘要，减少上下文占用：

- `OFFICETOOL_SUMMARY_TRIGGER_TURNS`
- `OFFICETOOL_MAX_CONTEXT_TURNS`

### 3) 文件上传

- 文档：txt/md/csv/json/pdf/docx/代码文本
- 图片：png/jpg/jpeg/webp/gif/heic/heif

说明：当前 LangChain Agent 流程里，文档会抽取文本注入；图片会以元数据提示注入（不是像素级视觉解析）。

## 目录结构

```text
app/
  agent.py
  attachments.py
  config.py
  local_tools.py
  main.py
  models.py
  storage.py
  static/
    index.html
    styles.css
    app.js
  data/
    sessions/
    uploads/
```

## 常用 API

- `GET /api/health`
- `POST /api/session/new`
- `POST /api/upload`
- `POST /api/chat`
- `GET /api/stats`
- `POST /api/stats/clear`

## 环境变量

直接看 `.env.example`。常用项：

- `OPENAI_API_KEY`
- `OFFICETOOL_OPENAI_BASE_URL`
- `OFFICETOOL_DEFAULT_MODEL`
- `OFFICETOOL_ALLOWED_COMMANDS`
- `OFFICETOOL_EXTRA_ALLOWED_ROOTS`
- `OFFICETOOL_ALLOW_ANY_PATH`

