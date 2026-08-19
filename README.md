# 深度研究智能体

一个意在提升个人研究效率的个人项目，也是一个为研究课题提供初始方向并生成初始报告的 Agent 研究助手。

用户输入研究主题并可选提供本地资料后，工具型 Agent 会搜索、抓取和分析网页内容，再生成中文 Markdown 研究报告。项目重点展示 Agent 工具调用、Qwen／DeepSeek 模型路由、网页研究、FAISS 本地 RAG，以及稳定 Web 交互的完整闭环。

## 项目定位

- 单个用户在本地或个人演示环境运行
- 输入一个研究主题
- 可选择 Qwen 或 DeepSeek
- 可选上传本地文档作为补充证据
- Agent 调用搜索、抓取、分析和报告工具
- 在浏览器中查看任务状态并下载最终报告
- 页面刷新后可恢复最近任务和已完成报告

项目不是通用 AI 平台。首版不实现用户系统、多租户、Redis、Celery、WebSocket、微服务或高并发部署。

## 当前状态

已完成：

- 动态研究主题输入
- Qwen／DeepSeek 聊天模型切换
- 独立 Embedding 配置和备用模型降级
- 工具型 Agent、网页研究和报告生成
- 本地文档解析、切分、内容哈希去重和不可信证据隔离
- Qwen Embedding、FAISS Top-K 4 检索与索引落盘
- 命令行通过多个 `--file` 使用本地文档 RAG
- 命令行与后续 FastAPI 共用的异步研究运行入口
- 结构化研究结果、真实阶段回调、协作式取消和稳定错误处理
- 优先返回 Reporter 实际生成的 Markdown 报告文件
- P1-M1 全仓 Ruff 规范清理
- P1-M2a 至 P1-M2d 高价值类型修复

稳定 Web 界面正在按阶段实现。W1 研究运行入口已经完成，下一步是 W2 FastAPI、SQLite 和单进程任务管理；前端代码尚未完成，当前可用入口仍是命令行。

## 目标技术栈

| 层次 | 技术 | 用途 |
| --- | --- | --- |
| 研究核心 | Python 3.10、AsyncIO | Agent 主程序、模型与异步工具调用 |
| 后端 | FastAPI、Uvicorn、Pydantic v2 | 文件上传、研究任务、状态、取消与报告接口 |
| 元数据 | SQLite、SQLAlchemy 2 | 保存任务、文件和报告元数据，支持刷新恢复 |
| 前端 | React、TypeScript、Vite | 稳定的单页用户界面 |
| 组件与报告 | Ant Design、react-markdown | 表单、上传、状态反馈和安全 Markdown 展示 |
| 配置 | MMEngine Config、python-dotenv | 配置合并和本地密钥读取 |
| 大模型 | Qwen、DeepSeek、OpenAI 兼容接口 | 工具调用、分析和报告生成 |
| 网页研究 | Crawl4AI、DDGS、HTTPX | 网页搜索、抓取和解析 |
| 本地 RAG | FAISS、Qwen Embedding | 本地文档向量索引与 Top-K 检索 |
| 质量检查 | pytest、Ruff、mypy、前端类型检查与组件测试 | 离线验证后端与关键交互 |

FastAPI、React、SQLite 等 Web 技术属于目标路线，尚未全部落地；现有 Agent、命令行和本地 RAG 已可运行。

## 目标架构

```text
React + TypeScript 单页界面
  ├─ 研究主题与模型选择
  ├─ 可选文档上传
  ├─ 状态轮询与任务取消
  └─ Markdown 报告查看与下载
                ↓ HTTP
FastAPI + 单进程 AsyncIO 任务管理
  ├─ Pydantic 输入校验
  ├─ SQLite 任务元数据
  └─ 研究运行入口
                ↓
工具型 Agent
  ├─ 网页搜索、抓取与分析
  ├─ 本地文档 → 切分 → Embedding → FAISS Top-K
  └─ 中文 Markdown 报告
```

首版使用 1 至 2 秒 HTTP 轮询，不使用 SSE 或 WebSocket；同一时间只执行一个研究任务。SQLite 负责界面状态恢复，FAISS 负责文档向量检索。

## 当前项目结构

```text
configs/       研究场景与模型配置
docs/          项目范围、技术需求和开发计划
examples/      当前命令行入口
src/
  application/ 研究输入、阶段和结果模型
  research_runner.py 命令行与后续 API 共用的异步研究入口
  document_retriever.py 本地文档切分与 FAISS 检索
  agent/       工具型 Agent 执行循环
  model/       Qwen／DeepSeek 等模型适配与降级
  tool/        搜索、文档分析和报告工具
  environment/ 文件系统与 FAISS 检索
  memory/      研究过程会话记忆
  prompt/      工具调用提示模板
tests/         离线测试
```

后续按计划增加 FastAPI、SQLite 任务管理与 `frontend/`，不会重写现有研究核心或另建一套 RAG。

## 配置模型

复制 `.env.template` 为 `.env`，密钥只保存在本地，不得提交到 GitHub。

默认使用 Qwen：

```env
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-plus
EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的密钥
```

切换到 DeepSeek 时，聊天模型使用 DeepSeek，FAISS 仍可使用 Qwen Embedding：

```env
MODEL_PROVIDER=deepseek
MODEL_NAME=deepseek-v4-flash
DEEPSEEK_API_KEY=你的密钥

EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的密钥
```

模型标识可能随提供方更新，请以实际账号当前可用模型为准。

## 当前运行方式

在项目解释器环境中执行：

```bash
python examples/run_tool_calling_agent.py --task "调研 RAG 系统中的幻觉问题，并生成中文报告。"
```

临时切换聊天模型：

```bash
python examples/run_tool_calling_agent.py --task "研究主题" --provider deepseek --model deepseek-v4-flash
```

省略 `--task` 时，终端会提示输入研究主题。

使用本地文档 RAG 时，可重复传入 `--file`：

```bash
python examples/run_tool_calling_agent.py \
  --task "根据本地资料比较两种 RAG 方案" \
  --file "资料/方案一.pdf" \
  --file "资料/方案二.docx"
```

程序会把文档转换为 Markdown，按 1000 字和 150 字重叠切分，使用当前 Embedding 模型写入会话级 FAISS，并把最相关的 4 个片段交给 Agent。

Web 启动方式会在对应开发阶段完成后补充，当前不提供尚不可用的命令。

## 设计文档

- [项目范围](docs/PROJECT_SCOPE.md)
- [稳定 Web 前端开发计划](docs/DEVELOPMENT_PLAN.md)
- [技术需求](docs/TECHNICAL_REQUIREMENTS.md)
- [许可证中文说明](LICENSE_zh.md)

## 范围原则

后续功能必须直接服务于“输入主题、结合可选资料并生成研究报告”，或者解决该流程中的输入校验、状态恢复、取消和错误反馈。项目保留 FastAPI、React 单页前端和 SQLite 元数据，明确不扩张到多用户、分布式任务队列或微服务平台。
